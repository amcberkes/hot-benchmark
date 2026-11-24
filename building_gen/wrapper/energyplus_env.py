"""The simulation module provides a .step api, but without a reward function and
without a well shaped observation and action space. This module seeks to wrap
the more "primitive" api into a gymnasium environment that we can pass without
trouble to any gymnasium consumer.

"""

import os
import sys
import typing

import gymnasium

import building_gen.wrapper.eplus_interface as simulation
from building_gen.accounting import CarrierEnergyLedger
from building_gen.control.capabilities import ControlCapabilityManifest
from building_gen.wrapper.manifest_control import (
    MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION,
)
from building_gen.wrapper.simulation_clock import clock_info, runtime_context_info

ObsType = typing.TypeVar("ObsType")
ActType = typing.TypeVar("ActType")
RewardOutput = float | tuple[float, typing.Mapping[str, typing.Any]]
SUBMITTED_ACTION_SCHEMA_VERSION = "hot-zone-setpoint-submission/1.0.0"
ENERGYPLUS_CLOSE_JOIN_TIMEOUT_SECONDS = 60.0


def _submitted_action_records(action: typing.Any) -> list[dict[str, typing.Any]]:
    """Serialize the exact zone setpoints passed to ``EnergyPlusInterface``.

    The interface contract is an insertion-ordered mapping from zone name to
    ``(heating_setpoint_c, cooling_setpoint_c)``.  These are submitted values,
    not actuator read-backs.
    """

    if not isinstance(action, typing.Mapping) or not action:
        raise TypeError("transformed action must be a non-empty zone mapping")
    records: list[dict[str, typing.Any]] = []
    for zone_name, setpoints in action.items():
        if not isinstance(zone_name, str) or not zone_name:
            raise TypeError("transformed action zone names must be non-empty strings")
        if not isinstance(setpoints, (tuple, list)) or len(setpoints) != 2:
            raise TypeError("each transformed zone action must contain heating and cooling setpoints")
        records.append(
            {
                "zone": zone_name,
                "heating_setpoint_c": float(setpoints[0]),
                "cooling_setpoint_c": float(setpoints[1]),
            }
        )
    return records


class EnergyPlusEnvironment(gymnasium.Env, typing.Generic[ObsType, ActType]):
    """What does environment.EnergyPlusEnvironment have that
    simulation.EnergyPlusInterface doesn't?

    1. The ability to be reset()
    2. A well defined observation and action space.
    3. A well defined reward function.

    While this class wraps all of those components together, the user will still
    have to provide their own reward function, action & observation reshaper and
    a way to initialize an EnergyPlusInterface object.

    """

    metadata = {"render_modes": []}

    """This function will be called each time we need to .reset() the
    environment."""
    make_energyplus: typing.Callable[[], simulation.EnergyPlusInterface]

    reward_function: typing.Callable[
        [typing.Any],  # Raw observation
        RewardOutput,
    ]

    observation_space: gymnasium.Space[ObsType]

    """This function will be called to transform the output of the raw
    EnergyPlus controller into a point of the observation_space."""
    observation_transform: typing.Callable[
        [typing.Any],  # Raw observation
        ObsType,  # Observation space
    ]

    action_space: gymnasium.Space[ActType]
    """This function will be called to transform a point in the action_space a
    raw EnergyPlus action."""
    action_transform: typing.Callable[
        [ActType],  # Action space
        typing.Any,  # Raw action
    ]

    ENERGYPLUS_PATH = "/usr/local/EnergyPlus-24.1.0"

    @classmethod
    def setup(cls):
        # Use container path when in container, otherwise use env variable
        in_container = os.getenv("SINGULARITY_CONTAINER", "") != ""
        if in_container:
            path = "/usr/local/EnergyPlus-24.1.0"
        else:
            path = os.getenv("ENERGYPLUS_PATH")

        if path and os.path.exists(path):
            sys.path.append(path)

    def __init__(
        self,
        make_energyplus: typing.Callable[[], simulation.EnergyPlusInterface],
        reward_fn: typing.Callable[[typing.Any, typing.Any, CarrierEnergyLedger], RewardOutput],
        observation_space: gymnasium.Space[ObsType],
        observation_transform: typing.Callable[[typing.Any], ObsType],
        action_space: gymnasium.Space[ActType],
        action_transform: typing.Callable[[ActType], typing.Any],
        building_characteristics: typing.Dict[str, typing.Any],
        controlled_zones: typing.List[str],
        observation_names: typing.List[str],
        observation_encoding_id: str = "unspecified_legacy_encoding",
        action_names: typing.Sequence[str] | None = None,
        action_endpoint_mapping: typing.Sequence[typing.Any] | None = None,
        action_transform_id: str = "unspecified_legacy_transform",
        control_manifest: ControlCapabilityManifest | None = None,
        interval_reward_fn: typing.Callable[..., RewardOutput] | None = None,
    ):
        super(EnergyPlusEnvironment, self).__init__()
        self.make_energyplus = make_energyplus
        self.reward_fn = reward_fn
        if interval_reward_fn is not None and not callable(interval_reward_fn):
            raise TypeError("interval_reward_fn must be callable")
        self.interval_reward_fn = interval_reward_fn
        self.observation_space = observation_space
        self.observation_transform = observation_transform
        self.action_space = action_space
        self.action_transform = action_transform

        self.all_zones = building_characteristics.get("zone_lists", [])
        self.controlled_zones = controlled_zones
        self.uncontrolled_zones = [zone for zone in self.all_zones if zone not in self.controlled_zones]
        self.observation_names = observation_names
        self.observation_encoding_id = observation_encoding_id
        self.action_names = tuple(action_names or ())
        self.action_endpoint_mapping = tuple(action_endpoint_mapping or ())
        self.action_transform_id = action_transform_id
        if control_manifest is None:
            self.control_manifest = None
        else:
            if type(control_manifest) is not ControlCapabilityManifest:
                raise TypeError("control_manifest must be an exact ControlCapabilityManifest")
            self.control_manifest = ControlCapabilityManifest.from_json(control_manifest.to_json())

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, typing.Any] | None = None,
    ) -> typing.Tuple[typing.Any, dict[str, typing.Any]]:
        super().reset(seed=seed)
        if hasattr(self, "ep"):
            self.close()
        self.ep = self.make_energyplus()
        obs, over, energy_ledger = self.ep.start()
        if over or energy_ledger is None:
            raise simulation.SimulationCrashed("EnergyPlus ended before producing the initial observation")
        transformed_obs = self.observation_transform(obs)
        self.last_obs = transformed_obs
        if self.ep.last_clock is None:
            raise RuntimeError("Initial EnergyPlus observation did not include a simulation clock")
        if self.ep.last_runtime_context is None:
            raise RuntimeError("Initial EnergyPlus observation did not include runtime context")
        if getattr(self.ep, "last_energy_interval_clock", None) is not None:
            raise RuntimeError("Initial EnergyPlus observation unexpectedly includes a completed energy interval")
        if getattr(self.ep, "last_energy_interval_runtime_context", None) is not None:
            raise RuntimeError(
                "Initial EnergyPlus observation unexpectedly includes completed energy-interval runtime context"
            )
        if getattr(self.ep, "last_energy_interval_observation", None) is not None:
            raise RuntimeError("Initial EnergyPlus observation unexpectedly includes energy-interval observation")
        # Reset reports an initial state, not a policy transition, so there are
        # deliberately no reward components until the first call to step().
        info = {
            "raw_observation": obs,
            "energy_ledger_kwh": energy_ledger.as_dict(),
            "energy_meter_sources": energy_ledger.meter_source_mapping,
            "simulation_clock": dict(clock_info(self.ep.last_clock)),
            "energyplus_runtime_context": dict(runtime_context_info(self.ep.last_runtime_context)),
            "energy_interval_clock": None,
            "energy_interval_runtime_context": None,
            "energy_interval_observation": None,
            "energyplus_exit_code": getattr(self.ep, "last_exit_code", None),
        }
        control_manifest = getattr(self, "control_manifest", None)
        if control_manifest is not None:
            if getattr(self.ep, "last_manifest_control_evidence", None) is not None:
                raise RuntimeError("initial EnergyPlus observation unexpectedly includes manifest action evidence")
            runtime = getattr(self.ep, "manifest_control_runtime", None)
            if runtime is None:
                raise RuntimeError("manifest Gym environment created an interface without manifest control")
            if runtime.capability_sha256 != control_manifest.capability_sha256:
                raise RuntimeError("Gym and EnergyPlus control capability identities disagree")
            info.update(
                {
                    "manifest_control_schema_version": MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION,
                    "control_capability_sha256": runtime.capability_sha256,
                    "task_action_binding_sha256": runtime.action_binding_sha256,
                    "manifest_action_evidence": None,
                }
            )
        return transformed_obs, info

    def step(self, action) -> typing.Tuple[ObsType, float, bool, bool, typing.Any]:
        return self._step(action, native_schedule=False)

    def step_native_schedule(self) -> typing.Tuple[ObsType, float, bool, bool, typing.Any]:
        """Advance without policy setpoints using an action-independent reward."""
        if not getattr(self, "supports_native_schedule", False):
            raise ValueError("native schedule stepping requires the occupied energy/comfort reward and thermostat interface")
        return self._step(None, native_schedule=True)

    def _step(self, action, *, native_schedule: bool) -> typing.Tuple[ObsType, float, bool, bool, typing.Any]:
        import logging

        logger = logging.getLogger("energyplus_env")

        logger.debug("Environment step - Input action: %r", action)

        # Decode policy actions or release overrides for native schedule control.
        a = simulation.NativeScheduleAction() if native_schedule else self.action_transform(action)
        logger.debug("Environment step - Transformed action: %r", a)

        obs, finished, energy_ledger = self.ep.step(a)
        logger.debug("Environment step - %d observations, finished=%s", len(obs), finished)

        transformed_obs = self.observation_transform(obs)
        self.last_obs = transformed_obs
        self.last_action = a
        interval_reward_fn = getattr(self, "interval_reward_fn", None)
        if interval_reward_fn is None:
            reward_result = self.reward_fn(obs, {} if native_schedule else a, energy_ledger)
        else:
            interval_clock = getattr(self.ep, "last_energy_interval_clock", None)
            interval_observation = getattr(self.ep, "last_energy_interval_observation", None)
            if interval_clock is None or interval_observation is None:
                raise RuntimeError("interval reward requires paired interval observation and clock")
            reward_result = interval_reward_fn(
                dict(interval_observation), {} if native_schedule else a, energy_ledger,
                dict(clock_info(interval_clock)))
        reward_components: typing.Mapping[str, typing.Any] = {}
        if isinstance(reward_result, tuple):
            if len(reward_result) != 2:
                raise TypeError("reward_fn tuple result must contain reward and components")
            reward, reward_components = reward_result
            if not isinstance(reward_components, typing.Mapping):
                raise TypeError("reward components must be a mapping")
        else:
            reward = reward_result
        reward = float(reward)
        logger.debug("Environment step - Calculated reward: %.6g", reward)

        if self.ep.last_clock is None:
            raise RuntimeError("EnergyPlus observation did not include a simulation clock")
        if self.ep.last_runtime_context is None:
            raise RuntimeError("EnergyPlus observation did not include runtime context")
        if getattr(self.ep, "last_energy_interval_clock", None) is None:
            raise RuntimeError("EnergyPlus transition did not include its completed energy-interval clock")
        if getattr(self.ep, "last_energy_interval_runtime_context", None) is None:
            raise RuntimeError("EnergyPlus transition did not include its completed energy-interval runtime context")
        if getattr(self.ep, "last_energy_interval_observation", None) is None:
            raise RuntimeError("EnergyPlus transition did not include its completed energy-interval observation")

        info = {
            "raw_observation": obs,
            "energy_ledger_kwh": energy_ledger.as_dict(),
            "energy_meter_sources": energy_ledger.meter_source_mapping,
            "simulation_clock": dict(clock_info(self.ep.last_clock)),
            # Legacy key: exact next-begin context for nonterminal transitions;
            # at the synthesized terminal boundary it remains the completed
            # interval context. Scientific reconciliation must always use the
            # explicitly paired energy-interval keys below.
            "energyplus_runtime_context": dict(runtime_context_info(self.ep.last_runtime_context)),
            "energy_interval_clock": dict(clock_info(self.ep.last_energy_interval_clock)),
            "energy_interval_runtime_context": dict(runtime_context_info(self.ep.last_energy_interval_runtime_context)),
            "energy_interval_observation": dict(self.ep.last_energy_interval_observation),
            "energyplus_exit_code": getattr(self.ep, "last_exit_code", None),
        }
        control_manifest = getattr(self, "control_manifest", None)
        if native_schedule:
            info["native_control"] = {
                "schema_version": "hot-native-schedule-control/1.0.0",
                "thermostat_overrides": "released",
                "submitted_policy_setpoints": False,
                "actuator_readback_available": False,
            }
        elif control_manifest is None:
            info.update(
                {
                    "submitted_action_schema_version": SUBMITTED_ACTION_SCHEMA_VERSION,
                    "submitted_action_physical": _submitted_action_records(a),
                    "actuator_readback": getattr(self.ep, "last_thermostat_readback", None) or {
                        "available": False,
                        "reason": (
                            "The wrapper records setpoints submitted to EnergyPlus but does "
                            "not read actuator values back after submission."
                        ),
                    },
                }
            )
        else:
            evidence = getattr(self.ep, "last_manifest_control_evidence", None)
            if not isinstance(evidence, typing.Mapping):
                raise RuntimeError("EnergyPlus transition omitted manifest action evidence")
            if evidence.get("schema_version") != MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION:
                raise RuntimeError("EnergyPlus transition has an unknown manifest action evidence schema")
            if evidence.get("control_capability_sha256") != control_manifest.capability_sha256:
                raise RuntimeError("EnergyPlus transition evidence has the wrong control capability identity")
            runtime = getattr(self.ep, "manifest_control_runtime", None)
            if runtime is None or evidence.get("task_action_binding_sha256") != runtime.action_binding_sha256:
                raise RuntimeError("EnergyPlus transition evidence has the wrong task action binding identity")
            evidence = runtime.verify_evidence(evidence)
            submitted = evidence.get("submitted_endpoints")
            readbacks = evidence.get("actuator_readbacks")
            effects = evidence.get("effect_readbacks")
            if not isinstance(submitted, list) or len(submitted) != len(control_manifest.full_actuators):
                raise RuntimeError("EnergyPlus transition evidence has incomplete endpoint submissions")
            if not isinstance(readbacks, list) or len(readbacks) != len(control_manifest.full_actuators):
                raise RuntimeError("EnergyPlus transition evidence has incomplete actuator read-backs")
            if not isinstance(effects, list) or len(effects) != len(control_manifest.effect_variables):
                raise RuntimeError("EnergyPlus transition evidence has incomplete effect read-backs")
            info.update(
                {
                    "manifest_control_schema_version": MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION,
                    "control_capability_sha256": control_manifest.capability_sha256,
                    "task_action_binding_sha256": runtime.action_binding_sha256,
                    "manifest_action_evidence": dict(evidence),
                    "submitted_action_schema_version": MANIFEST_CONTROL_RUNTIME_SCHEMA_VERSION,
                    "submitted_action_physical": submitted,
                    "actuator_readback": {
                        "available": True,
                        "records": readbacks,
                    },
                    "actuator_effect_readback": {
                        "available": True,
                        "records": effects,
                    },
                }
            )
        if reward_components:
            info["reward_components"] = dict(reward_components)

        return (
            transformed_obs,
            reward,
            finished,
            False,
            info,
        )

    def close(self) -> None:
        """Stop and join the active EnergyPlus worker or fail closed."""

        ep = getattr(self, "ep", None)
        if ep is None:
            return
        ep.close()
        thread = getattr(ep, "simulation_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=ENERGYPLUS_CLOSE_JOIN_TIMEOUT_SECONDS)
        if thread is not None and thread.is_alive():
            raise RuntimeError(
                "EnergyPlus simulation thread did not stop within "
                f"{ENERGYPLUS_CLOSE_JOIN_TIMEOUT_SECONDS:g} seconds during close; "
                "preserving the interface reference for inspection"
            )
        self.ep = None
