import json
import math
import tempfile
import typing
from pathlib import Path
from types import MappingProxyType

import gymnasium as gym

from building_gen.benchmark.specs import WarmupPreconditioningSpec
from building_gen.benchmark.warmup_preconditioning import (
    WARMUP_PRECONDITIONING_FILENAME,
    WarmupPreconditioningRuntimeContext,
)
from building_gen.control.action_space import (
    ExpandedControlAction,
    ManifestActionTransform,
)
from building_gen.control.capabilities import (
    ActuatorRole,
    ControlCapabilityManifest,
    ControlChannel,
)
from building_gen.model_geometry import (
    controlled_zone_names as model_controlled_zone_names,
)
from building_gen.model_geometry import resolve_conditioned_floor_area_m2
from building_gen.model_geometry import zone_floor_areas_m2 as model_zone_floor_areas_m2
from building_gen.scheduled_reward import HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3
from building_gen.reward_foundation import HOT_OCCUPIED_ENERGY_COMFORT_V2
from building_gen.preprocessing.component_meters import component_meter_sources
from building_gen.wrapper.config import EnergyPlusConfig
from building_gen.wrapper.energyplus_env import EnergyPlusEnvironment
from building_gen.wrapper.eplus_interface import EnergyPlusInterface
from building_gen.wrapper.manifest_control import verified_manifest_prepared_model_bytes
from building_gen.wrapper.model_metadata import load_epjson
from building_gen.wrapper.model_metadata import (
    timesteps_per_hour as model_timesteps_per_hour,
)
from building_gen.wrapper.rewards import get_reward_function
from building_gen.wrapper.simulation_clock import (
    CalendarYearSemantics,
    validate_calendar_year_semantics,
)
from building_gen.wrapper.spaces import ActionSpace, ObservationSpace

_THERMOSTAT_CHANNELS = (
    ("heating_setpoint_c", "degC", 18.0, 23.0),
    ("cooling_offset_c", "delta_degC", 2.0, 4.0),
)
_B2B_UNITARY_PROFILE_ID = "b2b_unitarysystem_v1"
_B2B_VAV_PROFILE_ID = "b2b_vavsystem_v1"
_B2B_HEATING_PROFILE_ID = "b2b_heating_only_v1"
_SUPPORTED_MANIFEST_PROFILE_IDS = frozenset(
    {
        _B2B_UNITARY_PROFILE_ID,
        _B2B_VAV_PROFILE_ID,
        _B2B_HEATING_PROFILE_ID,
    }
)
_B2B_COMPOSITE_DIRECT_PROFILE_ORDER = (
    _B2B_UNITARY_PROFILE_ID,
    _B2B_VAV_PROFILE_ID,
    _B2B_HEATING_PROFILE_ID,
)
_MANIFEST_ACTION_TRANSFORM_ID = "hot_manifest_direct_normalized_v1"


def _validate_vav_runtime_surface(manifest: ControlCapabilityManifest) -> None:
    """Defend the policy/physical dimensions promised by the VAV profile.

    The capability type already validates every profile channel and direct
    binding.  This runtime-boundary check keeps the VAV contract explicit:
    two loop commands plus minimum-flow and heating commands per served zone,
    with one additional fixed 40 C cooling endpoint per served zone.
    """

    loop_count = len(manifest.equipment)
    served_zone_count = sum(len(item.served_entity_ids) for item in manifest.equipment)
    expected_command_dim = 2 * loop_count + 2 * served_zone_count
    expected_physical_dim = expected_command_dim + served_zone_count
    fixed = manifest.fixed_actuators
    if manifest.agent_command_dim != expected_command_dim:
        raise ValueError(
            "VAV manifest policy dimension must equal 2*loops + 2*served_zones: "
            f"expected {expected_command_dim}, found {manifest.agent_command_dim}"
        )
    if len(manifest.full_actuators) != expected_physical_dim:
        raise ValueError(
            "VAV manifest physical endpoint dimension must equal policy_dim + served_zones: "
            f"expected {expected_physical_dim}, found {len(manifest.full_actuators)}"
        )
    if len(fixed) != served_zone_count or any(
        item.role is not ActuatorRole.FIXED
        or item.channel is not ControlChannel.COOLING_SETPOINT_C
        or item.fixed_value != 40.0
        for item in fixed
    ):
        raise ValueError("VAV manifest requires exactly one fixed 40 C cooling endpoint per served zone")


def _validate_heating_runtime_surface(manifest: ControlCapabilityManifest) -> None:
    """Defend the one-command/one-endpoint heating-only contract."""

    zone_count = len(manifest.equipment)
    if any(len(item.served_entity_ids) != 1 for item in manifest.equipment):
        raise ValueError("heating-only manifest requires exactly one served zone per equipment group")
    if manifest.agent_command_dim != zone_count or len(manifest.full_actuators) != zone_count:
        raise ValueError(
            "heating-only manifest requires H policy commands = H physical endpoints: "
            f"zones={zone_count}, commands={manifest.agent_command_dim}, "
            f"endpoints={len(manifest.full_actuators)}"
        )
    if manifest.fixed_actuators or any(
        item.role is not ActuatorRole.AGENT or item.channel is not ControlChannel.HEATING_SETPOINT_C
        for item in manifest.full_actuators
    ):
        raise ValueError("heating-only manifest requires one agent heating-setpoint endpoint per served zone")
    if len(manifest.effect_variables) != zone_count or any(
        len(item.actuator_endpoint_ids) != 1 or item.semantic.value != "zone_air_temperature"
        for item in manifest.effect_variables
    ):
        raise ValueError("heating-only manifest requires one zone-temperature effect descriptor per endpoint")


def _validate_composite_direct_runtime_surface(manifest: ControlCapabilityManifest) -> None:
    """Defend D=2U+2L+2Z+H and disjoint family ownership at admission."""

    unitary = [item for item in manifest.equipment if item.profile_id == _B2B_UNITARY_PROFILE_ID]
    vav = [item for item in manifest.equipment if item.profile_id == _B2B_VAV_PROFILE_ID]
    heating = [item for item in manifest.equipment if item.profile_id == _B2B_HEATING_PROFILE_ID]
    if any(len(item.served_entity_ids) != 1 for item in (*unitary, *heating)):
        raise ValueError("composite unitary/heating equipment must each serve exactly one zone")
    served_ids = [entity.casefold() for item in manifest.equipment for entity in item.served_entity_ids]
    if len(served_ids) != len(set(served_ids)):
        raise ValueError("composite direct served zones cannot overlap across equipment or families")
    vav_zone_count = sum(len(item.served_entity_ids) for item in vav)
    expected_command_dim = 2 * len(unitary) + 2 * len(vav) + 2 * vav_zone_count + len(heating)
    expected_physical_dim = expected_command_dim + vav_zone_count
    if manifest.agent_command_dim != expected_command_dim:
        raise ValueError(
            "composite direct policy dimension must equal 2U+2L+2Z+H: "
            f"expected {expected_command_dim}, found {manifest.agent_command_dim}"
        )
    if len(manifest.full_actuators) != expected_physical_dim:
        raise ValueError(
            "composite direct physical dimension must equal D+Z: "
            f"expected {expected_physical_dim}, found {len(manifest.full_actuators)}"
        )
    fixed = manifest.fixed_actuators
    if len(fixed) != vav_zone_count or any(
        item.profile_id != _B2B_VAV_PROFILE_ID
        or item.role is not ActuatorRole.FIXED
        or item.channel is not ControlChannel.COOLING_SETPOINT_C
        or item.fixed_value != 40.0
        for item in fixed
    ):
        raise ValueError("composite direct requires exactly the VAV fixed 40 C cooling endpoints")


def _thermostat_action_channels(
    action_channels: typing.Sequence[typing.Mapping[str, typing.Any]] | None,
) -> tuple[tuple[str, str, float, float], ...]:
    """Validate the physical channels consumed by the thermostat transform."""

    if action_channels is None:
        return _THERMOSTAT_CHANNELS
    if isinstance(action_channels, (str, bytes)):
        raise TypeError("action_channels must be a sequence of mappings")
    resolved: list[tuple[str, str, float, float]] = []
    for index, raw_channel in enumerate(action_channels):
        if not isinstance(raw_channel, typing.Mapping):
            raise TypeError(f"action_channels[{index}] must be a mapping")
        required = {"name", "unit", "physical_min", "physical_max"}
        missing = sorted(required - set(raw_channel))
        unknown = sorted(set(raw_channel) - required)
        if missing or unknown:
            raise ValueError(f"action_channels[{index}] has missing keys {missing!r} " f"and unknown keys {unknown!r}")
        minimum = raw_channel["physical_min"]
        maximum = raw_channel["physical_max"]
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
        ):
            raise TypeError(f"action_channels[{index}] physical bounds must be numeric")
        minimum = float(minimum)
        maximum = float(maximum)
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum >= maximum:
            raise ValueError(f"action_channels[{index}] has invalid physical bounds")
        resolved.append(
            (
                str(raw_channel["name"]),
                str(raw_channel["unit"]),
                minimum,
                maximum,
            )
        )

    expected_semantics = tuple((name, unit) for name, unit, _, _ in _THERMOSTAT_CHANNELS)
    actual_semantics = tuple((name, unit) for name, unit, _, _ in resolved)
    if actual_semantics != expected_semantics:
        raise ValueError(
            "thermostat actions must be ordered as " f"{expected_semantics!r}; received {actual_semantics!r}"
        )
    return tuple(resolved)


def _resolve_warmup_contract(
    contract: WarmupPreconditioningSpec | None,
    context: WarmupPreconditioningRuntimeContext | None,
) -> WarmupPreconditioningSpec:
    if contract is None:
        contract = WarmupPreconditioningSpec.disabled()
    if not isinstance(contract, WarmupPreconditioningSpec):
        raise TypeError("warmup_preconditioning must be a WarmupPreconditioningSpec")
    if contract.enabled:
        if not isinstance(context, WarmupPreconditioningRuntimeContext):
            raise ValueError("enabled warmup preconditioning requires a typed runtime context")
        if context.contract != contract:
            raise ValueError("warmup preconditioning runtime context contract mismatch")
    elif context is not None:
        raise ValueError("disabled warmup preconditioning cannot carry a runtime context")
    return contract


def _resolve_control_interface(
    *,
    action_channels: typing.Sequence[typing.Mapping[str, typing.Any]] | None,
    action_transform_id: str,
    control_manifest: ControlCapabilityManifest | None,
    warmup_preconditioning: WarmupPreconditioningSpec,
    stability_weight: float,
    reward_type: str,
) -> tuple[
    ControlCapabilityManifest | None,
    ManifestActionTransform | None,
    tuple[tuple[str, str, float, float], ...],
    str,
]:
    if control_manifest is None:
        resolved_channels = _thermostat_action_channels(action_channels)
        supported_transforms = {
            "linear_quantized_0_1c_legacy",
            "linear_unquantized_v1",
            "bounded_thermostat_15_30_v1",
            "bounded_thermostat_12_32_v1",
        }
        if action_transform_id not in supported_transforms:
            raise ValueError(
                f"unsupported action_transform_id {action_transform_id!r}; "
                f"expected one of {sorted(supported_transforms)!r}"
            )
        return None, None, resolved_channels, action_transform_id

    if type(control_manifest) is not ControlCapabilityManifest:
        raise TypeError("control_manifest must be an exact ControlCapabilityManifest")
    canonical = ControlCapabilityManifest.from_json(control_manifest.to_json())
    singleton_supported = (
        len(canonical.profile_ids) == 1 and canonical.profile_ids[0] in _SUPPORTED_MANIFEST_PROFILE_IDS
    )
    composite_supported = len(canonical.profile_ids) >= 2 and canonical.profile_ids == tuple(
        item for item in _B2B_COMPOSITE_DIRECT_PROFILE_ORDER if item in canonical.profile_ids
    )
    if not singleton_supported and not composite_supported:
        raise ValueError(
            "manifest-driven simulator control requires one native family profile or a registry-ordered "
            "composite subset of at least two direct families; "
            f"received {canonical.profile_ids!r}"
        )
    if canonical.profile_ids == (_B2B_VAV_PROFILE_ID,):
        _validate_vav_runtime_surface(canonical)
    elif canonical.profile_ids == (_B2B_HEATING_PROFILE_ID,):
        _validate_heating_runtime_surface(canonical)
    elif composite_supported:
        _validate_composite_direct_runtime_surface(canonical)
    if action_channels is not None:
        raise ValueError("action_channels cannot be combined with a manifest-driven action interface")
    if warmup_preconditioning.enabled:
        raise ValueError("manifest control cannot be combined with thermostat warmup preconditioning")
    if action_transform_id not in {
        "linear_quantized_0_1c_legacy",
        _MANIFEST_ACTION_TRANSFORM_ID,
    }:
        raise ValueError("action_transform_id cannot override the manifest-driven action transform")
    if stability_weight != 0.0:
        raise ValueError("manifest control does not yet define a cross-channel stability reward")
    if reward_type == "custom":
        raise ValueError("the thermostat-specific custom reward is unavailable for manifest control")
    return canonical, ManifestActionTransform(canonical), _THERMOSTAT_CHANNELS, _MANIFEST_ACTION_TRANSFORM_ID


def _load_manifest_epjson(
    path: str,
    manifest: ControlCapabilityManifest,
) -> dict[str, typing.Any]:
    contents = verified_manifest_prepared_model_bytes(manifest, path)
    try:
        model = json.loads(contents.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"EnergyPlus model is not valid UTF-8 epJSON: {path}") from error
    if not isinstance(model, dict):
        raise ValueError(f"EnergyPlus model must contain a JSON object: {path}")
    return model


def _resolve_model_timesteps(building_data: typing.Mapping[str, typing.Any], requested: int | None) -> int:
    declared = model_timesteps_per_hour(building_data)
    if requested is None:
        return declared
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise TypeError("timesteps_per_hour must be an integer")
    if requested != declared:
        raise ValueError(
            "timesteps_per_hour does not match the epJSON Timestep object: "
            f"requested {requested}, model declares {declared}"
        )
    return requested


def _validate_manifest_controlled_zones(
    controlled_zones: list[str],
    manifest: ControlCapabilityManifest,
) -> None:
    manifest_zones = tuple(
        variable.key_value
        for variable in manifest.observation_variables
        if variable.semantic.value == "zone_air_temperature"
    )
    controlled_by_casefold = {name.casefold(): name for name in controlled_zones}
    manifest_by_casefold = {name.casefold(): name for name in manifest_zones}
    if (
        len(controlled_by_casefold) != len(controlled_zones)
        or len(manifest_by_casefold) != len(manifest_zones)
        or set(controlled_by_casefold) != set(manifest_by_casefold)
    ):
        raise ValueError(
            "controlled_zones must exactly match the manifest capability's required zone observations: "
            f"controlled={sorted(controlled_zones)!r}, manifest={sorted(manifest_zones)!r}"
        )


def _resolve_controlled_zones(
    building_data: typing.Mapping[str, typing.Any],
    building_characteristics: typing.Mapping[str, typing.Any],
    control_manifest: ControlCapabilityManifest | None,
) -> list[str]:
    configured_zones = building_characteristics.get("controlled_zones")
    if configured_zones:
        if isinstance(configured_zones, (str, bytes)):
            raise TypeError("controlled_zones must be a collection of zone names")
        controlled_zones = list(configured_zones)
    elif control_manifest is not None:
        # A separate control profile can intentionally cover only one HVAC
        # family in a mixed building.  In manifest mode the authenticated
        # required zone observations are therefore the authoritative default;
        # inferring every HVAC-connected model zone would silently demand a
        # composite profile.  Explicit caller configuration is still checked
        # for exact equality below.
        controlled_zones = [
            variable.key_value
            for variable in control_manifest.observation_variables
            if variable.semantic.value == "zone_air_temperature"
        ]
    else:
        controlled_zones = list(model_controlled_zone_names(building_data))
    if not controlled_zones:
        raise ValueError("No HVAC-controlled zones could be identified in the epJSON model")
    if control_manifest is not None:
        _validate_manifest_controlled_zones(controlled_zones, control_manifest)
    return controlled_zones


def _config_action_channel_names(
    resolved_action_channels: tuple[tuple[str, str, float, float], ...],
    control_manifest: ControlCapabilityManifest | None,
) -> tuple[str, ...]:
    if control_manifest is None:
        return tuple(channel[0] for channel in resolved_action_channels)
    return tuple(command.name for command in control_manifest.agent_commands)


def _build_action_space_contract(
    config: EnergyPlusConfig,
    controlled_zones: list[str],
    resolved_action_channels: tuple[tuple[str, str, float, float], ...],
    manifest_action_transform: ManifestActionTransform | None,
    control_manifest: ControlCapabilityManifest | None,
) -> tuple[typing.Any, typing.Any, typing.Callable[[typing.Any], typing.Any], tuple[str, ...], tuple[typing.Any, ...]]:
    if manifest_action_transform is not None:
        assert control_manifest is not None
        action_names = tuple(command.name for command in control_manifest.agent_commands)
        endpoint_mapping = tuple(
            (
                command.name,
                ((0.0, command.lower_bound), (1.0, command.upper_bound)),
            )
            for command in control_manifest.agent_commands
        )
        return (
            manifest_action_transform,
            manifest_action_transform.normalized_space,
            manifest_action_transform.expand,
            action_names,
            endpoint_mapping,
        )

    action_space = ActionSpace(config, controlled_zones)
    action_names = tuple(
        f"{zone_name}:{channel[0]}" for zone_name in controlled_zones for channel in resolved_action_channels
    )
    endpoint_mapping = tuple(
        (
            f"{zone_name}:{channel[0]}",
            ((0.0, channel[2]), (1.0, channel[3])),
        )
        for zone_name in controlled_zones
        for channel in resolved_action_channels
    )
    return action_space, action_space.gym_space, action_space.parse_action, action_names, endpoint_mapping


def create_simulator(
    path_to_building: str,
    path_to_weather: str,
    building_characteristics: dict,
    reward_type: str,
    energy_weight: float = 1.0,
    eplus_output_dir: str | None = None,
    episode_length_hours: int = 24,
    timesteps_per_hour: int | None = None,
    *,
    calendar_year_semantics: CalendarYearSemantics | None = None,
    comfort_weight: float = 1.0,
    stability_weight: float = 0.0,
    comfort_temp_range: tuple[float, float] = (20.0, 24.0),
    energy_scale_kwh_per_m2: float | None = None,
    comfort_scale_person_celsius_squared_hours_per_m2: float | None = None,
    comfort_schedule: typing.Mapping[str, typing.Any] | None = None,
    action_channels: typing.Sequence[typing.Mapping[str, typing.Any]] | None = None,
    action_transform_id: str = "linear_quantized_0_1c_legacy",
    warmup_preconditioning: WarmupPreconditioningSpec | None = None,
    warmup_preconditioning_context: WarmupPreconditioningRuntimeContext | None = None,
    control_manifest: ControlCapabilityManifest | None = None,
    additional_temperature_zones: tuple[str, ...] = (),
) -> gym.Env:
    """
    Create a simulator for a given building and weather file.

    Args:
        path_to_building: Path to the building file (epJSON)
        path_to_weather: Path to the weather file (epw)
        building_characteristics: Dictionary with building characteristics.
            ``floor_area_m2`` is optional: it is computed from the floor
            polygons of the controlled zones and, when supplied, validated
            against that computed value.
        reward_type: Type of reward function to use
        energy_weight: Weight for energy penalty in reward function
        comfort_weight: Weight for comfort penalty in reward function
        stability_weight: Weight for action-change penalty in reward function
        comfort_temp_range: Lower and upper comfort bounds in degrees Celsius
        energy_scale_kwh_per_m2: Fixed energy normalization scale for reward v2
        comfort_scale_person_celsius_squared_hours_per_m2: Fixed comfort
            normalization scale for reward v2
        eplus_output_dir: Directory for EnergyPlus outputs
        episode_length_hours: Number of simulated hours before termination
        timesteps_per_hour: Expected model timestep setting. If omitted, read it
            from the epJSON model; if provided, it must match the model.
        calendar_year_semantics: Optional explicit interpretation of simulator
            calendar years. Direct legacy callers may omit it to retain
            weather-filename inference.
        action_channels: Ordered physical action channels. Benchmark callers
            pass these directly from their identity-bearing TaskSpec.
        action_transform_id: Version of the normalized-to-physical transform.
        warmup_preconditioning: Explicit disabled or enabled runtime contract.
        warmup_preconditioning_context: Task/model identity for an enabled
            contract. It is required when the contract is enabled and rejected
            when the contract is disabled.
        control_manifest: Optional canonical B2B unitary, VAV, heating-only, or
            multi-family composite-direct capability manifest.
            When supplied, its normalized manifest action space replaces the
            legacy thermostat action adapter and its prepared-model hash is
            verified before the environment is returned.

    Returns:
        gym.Env: EnergyPlus environment
    """

    uses_scheduled_reward = reward_type == HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3
    if uses_scheduled_reward:
        if stability_weight != 0:
            raise ValueError("scheduled occupied reward does not include a stability penalty")
        from building_gen.scheduled_reward import ScheduledOccupiedRewardConfig
        ScheduledOccupiedRewardConfig.from_mapping({
            "energy_weight": energy_weight, "comfort_weight": comfort_weight,
            "energy_scale_kwh_per_m2": energy_scale_kwh_per_m2,
            "comfort_scale_person_celsius_squared_hours_per_m2": comfort_scale_person_celsius_squared_hours_per_m2,
            "comfort_schedule": comfort_schedule})
    elif comfort_schedule is not None:
        raise ValueError("comfort_schedule requires the scheduled occupied reward")
    if not isinstance(building_characteristics, typing.Mapping):
        raise TypeError("building_characteristics must be a mapping")
    building_characteristics = dict(building_characteristics)
    if calendar_year_semantics is not None:
        calendar_year_semantics = validate_calendar_year_semantics(calendar_year_semantics)
    warmup_preconditioning = _resolve_warmup_contract(
        warmup_preconditioning,
        warmup_preconditioning_context,
    )
    (
        control_manifest,
        manifest_action_transform,
        resolved_action_channels,
        action_transform_id,
    ) = _resolve_control_interface(
        action_channels=action_channels,
        action_transform_id=action_transform_id,
        control_manifest=control_manifest,
        warmup_preconditioning=warmup_preconditioning,
        stability_weight=stability_weight,
        reward_type=reward_type,
    )
    if isinstance(episode_length_hours, bool) or not isinstance(episode_length_hours, int):
        raise TypeError("episode_length_hours must be an integer")
    if episode_length_hours <= 0:
        raise ValueError("episode_length_hours must be positive")

    if control_manifest is None:
        building_data = load_epjson(path_to_building)
    else:
        building_data = _load_manifest_epjson(path_to_building, control_manifest)
    explicit_meter_sources = component_meter_sources(building_data)
    timesteps_per_hour = _resolve_model_timesteps(building_data, timesteps_per_hour)

    # Resolve the zones before the reward configuration because the energy
    # denominator is the sum of these zones' true floor polygons.
    controlled_zones = _resolve_controlled_zones(
        building_data,
        building_characteristics,
        control_manifest,
    )

    floor_area_m2 = resolve_conditioned_floor_area_m2(
        building_data,
        zone_names=controlled_zones,
        supplied_floor_area_m2=building_characteristics.get("floor_area_m2"),
    )
    building_characteristics["controlled_zones"] = controlled_zones
    building_characteristics["floor_area_m2"] = floor_area_m2

    output_dir = eplus_output_dir or tempfile.mkdtemp(prefix="hot-energyplus-")
    heating_range = (
        resolved_action_channels[0][2],
        resolved_action_channels[0][3],
    )
    offset_range = (
        resolved_action_channels[1][2],
        resolved_action_channels[1][3],
    )
    config_action_channel_names = _config_action_channel_names(resolved_action_channels, control_manifest)
    config = EnergyPlusConfig(
        episode_length_hours=episode_length_hours,
        timesteps_per_hour=timesteps_per_hour,
        calendar_year_semantics=calendar_year_semantics,
        building_floor_area=floor_area_m2,
        output_dir=output_dir,
        energy_penalty_weight=energy_weight,
        comfort_penalty_weight=comfort_weight,
        stability_penalty_weight=stability_weight,
        comfort_temp_range=comfort_temp_range,
        heating_setpoint_range=heating_range,
        offset_range=offset_range,
        cooling_setpoint_range=(
            heating_range[0] + offset_range[0],
            (32.0 if action_transform_id == "bounded_thermostat_12_32_v1" else 30.0) if action_transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1") else heating_range[1] + offset_range[1],
        ),
        action_channel_names=config_action_channel_names,
        action_transform_id=action_transform_id,
    )

    print(f"Using controlled zones: {controlled_zones}; " f"conditioned floor area: {floor_area_m2:.6g} m2")

    # Create spaces - only use controlled zones for observations
    action_space, gym_action_space, action_transform, action_names, endpoint_mapping = _build_action_space_contract(
        config,
        controlled_zones,
        resolved_action_channels,
        manifest_action_transform,
        control_manifest,
    )
    observation_space = ObservationSpace(config, controlled_zones)

    # Create reward function
    reward_arguments = {
        "energy_weight": energy_weight,
        "comfort_weight": comfort_weight,
        "stability_weight": stability_weight,
    }
    uses_occupied_reward = reward_type in (HOT_OCCUPIED_ENERGY_COMFORT_V2, HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3)
    if uses_occupied_reward:
        model_areas = model_zone_floor_areas_m2(building_data, zone_names=controlled_zones)
        areas_by_casefold = {zone_name.casefold(): area for zone_name, area in model_areas.items()}
        zone_floor_areas_m2 = {zone_name: areas_by_casefold[zone_name.casefold()] for zone_name in controlled_zones}
        reward_arguments.update(
            zone_floor_areas_m2=zone_floor_areas_m2,
            energy_scale_kwh_per_m2=energy_scale_kwh_per_m2,
            comfort_scale_person_celsius_squared_hours_per_m2=(comfort_scale_person_celsius_squared_hours_per_m2),
        )
    if uses_scheduled_reward:
        reward_arguments["comfort_schedule"] = comfort_schedule
    reward_function = get_reward_function(
        reward_type,
        config,
        controlled_zones,
        **reward_arguments,
    )

    # Create EnergyPlus interface factory
    def make_energyplus():
        reward_function.reset()
        return EnergyPlusInterface(
            building_path=path_to_building,
            weather_path=path_to_weather,
            config=config,
            controlled_zones=controlled_zones,
            include_zone_occupancy=uses_occupied_reward,
            **({"additional_temperature_zones": tuple(additional_temperature_zones)} if additional_temperature_zones else {}),
            warmup_preconditioning=warmup_preconditioning,
            warmup_preconditioning_context=warmup_preconditioning_context,
            control_manifest=control_manifest,
            **({"component_meter_sources": explicit_meter_sources} if explicit_meter_sources is not None else {}),
        )

    # Create reward function wrapper that adapts to the environment's expected signature
    def reward_wrapper(observations, actions=None, energy_ledger=None):
        # If no actions provided (like during reset), create empty actions
        if actions is None:
            actions = {zone: (0.0, 0.0) for zone in controlled_zones}
        elif isinstance(actions, ExpandedControlAction):
            # Manifest control supports rewards whose action term is absent.
            # Nonzero stability and the thermostat-specific custom reward are
            # rejected above rather than silently reinterpreted.
            actions = {}
        if uses_occupied_reward:
            return reward_function.calculate_reward_with_components(observations, actions, energy_ledger)
        # Preserve the scalar-only legacy reward interface.
        return reward_function.calculate_reward(observations, actions, energy_ledger)

    # Create environment
    env = EnergyPlusEnvironment(
        make_energyplus=make_energyplus,
        reward_fn=reward_wrapper,
        interval_reward_fn=reward_function.calculate_interval_reward if uses_scheduled_reward else None,
        observation_space=observation_space.gym_space,
        observation_transform=observation_space.pack_observation,
        action_space=gym_action_space,
        action_transform=action_transform,
        building_characteristics=building_characteristics,
        controlled_zones=controlled_zones,
        observation_names=observation_space.observation_keys,
        observation_encoding_id=observation_space.encoding_id,
        action_names=getattr(action_space, "action_names", action_names),
        action_endpoint_mapping=getattr(
            action_space,
            "endpoint_mapping",
            endpoint_mapping,
        ),
        action_transform_id=getattr(
            action_space,
            "transform_id",
            action_transform_id,
        ),
        control_manifest=control_manifest,
    )
    env.supports_native_schedule = uses_occupied_reward and control_manifest is None
    # Evaluation must obtain geometry independently of reward component claims.
    env.controlled_zone_floor_areas_m2 = MappingProxyType(
        dict(zone_floor_areas_m2) if uses_occupied_reward else {}
    )
    env.warmup_preconditioning_contract_payload = warmup_preconditioning.identity_payload()
    env.warmup_preconditioning_contract_sha256 = warmup_preconditioning.sha256
    env.warmup_preconditioning_evidence_path = (
        str(Path(output_dir).resolve() / WARMUP_PRECONDITIONING_FILENAME) if warmup_preconditioning.enabled else None
    )

    return env
