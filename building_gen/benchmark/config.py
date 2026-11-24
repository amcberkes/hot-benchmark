"""Strict YAML loader for control benchmark suites.

PyYAML is imported only when a file is loaded; importing the benchmark package
itself remains standard-library-only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from building_gen.benchmark._canonical import refreeze_mapping, thaw_mapping
from building_gen.benchmark.specs import (
    LEGACY_TASK_SCHEMA_VERSION,
    TASK_SCHEMA_VERSION,
    ActionChannelSpec,
    ActionProfileSpec,
    AssetSpec,
    BenchmarkSuite,
    CalendarWindowSpec,
    EpisodeSpec,
    EvaluationProtocolSpec,
    ExperimentSpec,
    RewardSpec,
    SpaceSpec,
    TaskSpec,
    WarmupPreconditioningSpec,
)

LEGACY_SUITE_CONFIG_SCHEMA_VERSION = "hot-control-suite/1.0.0"
PRECONDITIONING_LEGACY_SUITE_CONFIG_SCHEMA_VERSION = "hot-control-suite/1.1.0"
SUITE_CONFIG_SCHEMA_VERSION = "hot-control-suite/1.2.0"
TRAINER_CONFIG_SCHEMA_VERSION = "hot-external-trainer-config/1.0.0"


def _null_paths(value: Any, prefix: str = "parameters") -> tuple[str, ...]:
    if value is None:
        return (prefix,)
    if isinstance(value, Mapping):
        paths = []
        for key, item in value.items():
            paths.extend(_null_paths(item, f"{prefix}.{key}"))
        return tuple(paths)
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_null_paths(item, f"{prefix}[{index}]"))
        return tuple(paths)
    return ()


@dataclass(frozen=True)
class ExternalTrainerConfig:
    """A separately versioned trainer configuration, possibly a template."""

    algorithm_id: str
    seeds: tuple[int, ...]
    parameters: tuple[tuple[str, Any], ...]
    notes: str = ""
    schema_version: str = TRAINER_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.algorithm_id.strip():
            raise ValueError("algorithm_id cannot be empty")
        seeds = tuple(self.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("trainer seeds must be non-empty and unique")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("trainer seeds must be non-negative integers")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "parameters", refreeze_mapping(self.parameters, field="trainer parameters"))

    @property
    def parameter_mapping(self) -> dict[str, Any]:
        return thaw_mapping(self.parameters)

    @property
    def unresolved_parameters(self) -> tuple[str, ...]:
        return _null_paths(self.parameter_mapping)

    def assert_runnable(self) -> None:
        unresolved = self.unresolved_parameters
        if unresolved:
            raise ValueError("trainer configuration still contains unresolved values: " + ", ".join(unresolved))

    def apply_to(self, experiment: ExperimentSpec) -> ExperimentSpec:
        """Embed resolved trainer values in an experiment's deterministic ID."""

        if experiment.algorithm_id != self.algorithm_id:
            raise ValueError(
                f"trainer config for {self.algorithm_id!r} cannot be applied to {experiment.algorithm_id!r}"
            )
        if experiment.seeds != self.seeds:
            raise ValueError(f"experiment seeds {experiment.seeds!r} do not match trainer config seeds {self.seeds!r}")
        self.assert_runnable()
        existing = experiment.parameter_mapping
        existing.pop("trainer_config", None)
        existing["trainer"] = self.parameter_mapping
        return replace(experiment, parameters=existing, enabled=True)


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    return value


def _sequence(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field} must be a YAML list")
    return value


def _only(mapping: Mapping[str, Any], allowed: set[str], *, field: str) -> None:
    unknown = sorted(set(mapping) - allowed)
    if unknown:
        raise ValueError(f"{field} contains unknown field(s): {', '.join(unknown)}")


def _asset(value: Any, *, field: str) -> AssetSpec:
    item = _mapping(value, field=field)
    _only(item, {"asset_id", "relative_path", "sha256"}, field=field)
    return AssetSpec(
        asset_id=item["asset_id"],
        relative_path=item["relative_path"],
        sha256=item.get("sha256"),
    )


def _space(value: Any, *, field: str) -> SpaceSpec:
    item = _mapping(value, field=field)
    _only(
        item,
        {"kind", "shape", "dtype", "encoding", "lower_bound", "upper_bound"},
        field=field,
    )
    return SpaceSpec(
        kind=item["kind"],
        shape=tuple(_sequence(item["shape"], field=f"{field}.shape")),
        dtype=item.get("dtype", "float32"),
        encoding=item.get("encoding", "raw"),
        lower_bound=item.get("lower_bound"),
        upper_bound=item.get("upper_bound"),
    )


def _action_profile(
    value: Any,
    *,
    field: str,
    suite_schema_version: str,
) -> ActionProfileSpec:
    item = _mapping(value, field=field)
    allowed_fields = {"profile_id", "controlled_entities", "channels"}
    if suite_schema_version != LEGACY_SUITE_CONFIG_SCHEMA_VERSION:
        allowed_fields.add("transform_id")
    _only(
        item,
        allowed_fields,
        field=field,
    )
    if suite_schema_version != LEGACY_SUITE_CONFIG_SCHEMA_VERSION:
        if "transform_id" not in item:
            raise ValueError(f"{field}.transform_id is required by suite schema 1.1 or newer")
        transform_id = item["transform_id"]
    else:
        transform_id = "linear_quantized_0_1c_legacy"
    channels = []
    for index, raw_channel in enumerate(_sequence(item["channels"], field=f"{field}.channels")):
        channel = _mapping(raw_channel, field=f"{field}.channels[{index}]")
        _only(
            channel,
            {"name", "unit", "physical_min", "physical_max"},
            field=f"{field}.channels[{index}]",
        )
        channels.append(
            ActionChannelSpec(
                name=channel["name"],
                unit=channel["unit"],
                physical_min=channel["physical_min"],
                physical_max=channel["physical_max"],
            )
        )
    return ActionProfileSpec(
        profile_id=item["profile_id"],
        controlled_entities=tuple(_sequence(item["controlled_entities"], field=f"{field}.controlled_entities")),
        channels=tuple(channels),
        transform_id=transform_id,
    )


def _warmup_preconditioning(value: Any, *, field: str) -> WarmupPreconditioningSpec:
    item = _mapping(value, field=field)
    allowed = {
        "schema_version",
        "strategy_id",
        "maximum_warmup_days",
        "expected_source_maximum_warmup_days",
        "epw_data_period_start_weekday",
        "epw_data_period_start_month",
        "epw_data_period_start_day",
        "callback",
        "readiness_gate",
        "required_kind_of_sim",
        "required_kind_name",
        "required_warmup_flag",
        "actuator_component_type",
        "actuator_control_type",
        "actuator_keys",
        "actuator_value_c",
        "underlying_schedule_names",
        "underlying_schedule_constant_value_c",
        "cooling_coil_types",
        "expected_cooling_coil_sizing_eio_line_count",
        "expected_cooling_coil_sizing_eio_sha256",
        "post_warmup_operation",
        "treat_weather_as_actual",
    }
    _only(item, allowed, field=field)
    if "strategy_id" not in item:
        raise ValueError(f"{field}.strategy_id is required")
    values = dict(item)
    if "cooling_coil_types" in values:
        values["cooling_coil_types"] = tuple(_sequence(values["cooling_coil_types"], field=f"{field}.cooling_coil_types"))
    if "actuator_keys" in values:
        values["actuator_keys"] = tuple(_sequence(values["actuator_keys"], field=f"{field}.actuator_keys"))
    if "underlying_schedule_names" in values:
        values["underlying_schedule_names"] = tuple(
            _sequence(
                values["underlying_schedule_names"],
                field=f"{field}.underlying_schedule_names",
            )
        )
    missing = object()
    configured_treat_weather_as_actual = values.pop("treat_weather_as_actual", missing)
    contract = WarmupPreconditioningSpec(**values)
    expected_treat_weather_as_actual = contract.identity_payload().get(
        "treat_weather_as_actual",
        missing,
    )
    if configured_treat_weather_as_actual is missing:
        if expected_treat_weather_as_actual is not missing:
            raise ValueError(f"{field}.treat_weather_as_actual is required by warmup preconditioning schema v2")
    elif (
        expected_treat_weather_as_actual is missing
        or configured_treat_weather_as_actual != expected_treat_weather_as_actual
    ):
        raise ValueError(f"{field}.treat_weather_as_actual must exactly match the versioned warmup contract")
    return contract


def _task(
    value: Any,
    *,
    index: int,
    suite_schema_version: str,
) -> TaskSpec:
    field = f"tasks[{index}]"
    item = _mapping(value, field=field)
    allowed_fields = {
        "name",
        "building",
        "weather",
        "observation_space",
        "action_space",
        "action_profile",
        "episode",
        "reward",
        "metadata",
    }
    if suite_schema_version != LEGACY_SUITE_CONFIG_SCHEMA_VERSION:
        allowed_fields.add("calendar_year")
    if suite_schema_version == SUITE_CONFIG_SCHEMA_VERSION:
        allowed_fields.add("warmup_preconditioning")
        allowed_fields.add("component_energy_meters")
        allowed_fields.add("weekly_training_reward")
    _only(item, allowed_fields, field=field)
    if suite_schema_version == SUITE_CONFIG_SCHEMA_VERSION:
        if "warmup_preconditioning" not in item:
            raise ValueError(f"{field}.warmup_preconditioning is required by suite schema 1.2")
        warmup_preconditioning = _warmup_preconditioning(
            item["warmup_preconditioning"],
            field=f"{field}.warmup_preconditioning",
        )
    else:
        # Suite schemas 1.0 and 1.1 predate the intervention.  Their meaning
        # is frozen as explicitly disabled; they never inherit a new value.
        warmup_preconditioning = WarmupPreconditioningSpec.disabled()
    episode = _mapping(item["episode"], field=f"{field}.episode")
    episode_fields = {"timesteps_per_hour", "episode_length_hours"}
    if suite_schema_version == SUITE_CONFIG_SCHEMA_VERSION:
        episode_fields.add("training_end_behavior")
    _only(episode, episode_fields, field=f"{field}.episode")
    reward = _mapping(item["reward"], field=f"{field}.reward")
    _only(reward, {"reward_id", "parameters"}, field=f"{field}.reward")
    observation_space = _space(
        item["observation_space"],
        field=f"{field}.observation_space",
    )
    if (
        suite_schema_version == LEGACY_SUITE_CONFIG_SCHEMA_VERSION
        and observation_space.encoding == "hot_observation_v2"
        and observation_space.lower_bound is None
    ):
        # In schema 1.0, v2 implied normalized bounds but did not require them
        # to be repeated in YAML. Make that legacy implication explicit in the
        # migrated, identity-bearing TaskSpec.
        observation_space = replace(
            observation_space,
            lower_bound=0.0,
            upper_bound=1.0,
        )
    return TaskSpec(
        name=item["name"],
        building=_asset(item["building"], field=f"{field}.building"),
        weather=_asset(item["weather"], field=f"{field}.weather"),
        observation_space=observation_space,
        action_space=_space(item["action_space"], field=f"{field}.action_space"),
        action_profile=_action_profile(
            item["action_profile"],
            field=f"{field}.action_profile",
            suite_schema_version=suite_schema_version,
        ),
        episode=EpisodeSpec(
            timesteps_per_hour=episode["timesteps_per_hour"],
            episode_length_hours=episode["episode_length_hours"],
            training_end_behavior=episode.get("training_end_behavior", "terminal"),
        ),
        reward=RewardSpec(
            reward_id=reward["reward_id"],
            parameters=reward.get("parameters", {}),
        ),
        warmup_preconditioning=warmup_preconditioning,
        calendar_year=item.get("calendar_year"),
        component_energy_meters=item.get("component_energy_meters"),
        weekly_training_reward=(RewardSpec(**_mapping(item['weekly_training_reward'], field=f'{field}.weekly_training_reward'))
                                if item.get('weekly_training_reward') is not None else None),
        metadata=item.get("metadata", {}),
        schema_version=(
            TASK_SCHEMA_VERSION if suite_schema_version == SUITE_CONFIG_SCHEMA_VERSION else LEGACY_TASK_SCHEMA_VERSION
        ),
    )


def _experiment(value: Any, *, index: int) -> ExperimentSpec:
    field = f"experiments[{index}]"
    item = _mapping(value, field=field)
    _only(
        item,
        {
            "name",
            "algorithm_id",
            "train_tasks",
            "evaluation_tasks",
            "seeds",
            "parameters",
            "enabled",
            "notes",
            "evaluation_protocol",
        },
        field=field,
    )
    protocol = None
    if item.get("evaluation_protocol") is not None:
        raw_protocol = _mapping(item["evaluation_protocol"], field=f"{field}.evaluation_protocol")
        _only(
            raw_protocol,
            {"protocol_id", "windows", "episodes_per_window", "deterministic_actions"},
            field=f"{field}.evaluation_protocol",
        )
        windows = []
        for window_index, raw_window in enumerate(
            _sequence(raw_protocol["windows"], field=f"{field}.evaluation_protocol.windows")
        ):
            window = _mapping(
                raw_window,
                field=f"{field}.evaluation_protocol.windows[{window_index}]",
            )
            _only(
                window,
                {"name", "start_month", "start_day", "end_month", "end_day"},
                field=f"{field}.evaluation_protocol.windows[{window_index}]",
            )
            windows.append(CalendarWindowSpec(**window))
        protocol = EvaluationProtocolSpec(
            protocol_id=raw_protocol["protocol_id"],
            windows=tuple(windows),
            episodes_per_window=raw_protocol.get("episodes_per_window", 1),
            deterministic_actions=raw_protocol.get("deterministic_actions", True),
        )
    return ExperimentSpec(
        name=item["name"],
        algorithm_id=item["algorithm_id"],
        train_tasks=tuple(_sequence(item.get("train_tasks", []), field=f"{field}.train_tasks")),
        evaluation_tasks=tuple(_sequence(item["evaluation_tasks"], field=f"{field}.evaluation_tasks")),
        seeds=tuple(_sequence(item["seeds"], field=f"{field}.seeds")),
        parameters=item.get("parameters", {}),
        enabled=item.get("enabled", True),
        notes=item.get("notes", ""),
        evaluation_protocol=protocol,
    )


def benchmark_suite_from_mapping(value: Mapping[str, Any]) -> BenchmarkSuite:
    """Parse and validate a suite mapping without touching the filesystem."""

    root = _mapping(value, field="suite")
    _only(root, {"schema_version", "name", "tasks", "experiments"}, field="suite")
    suite_schema_version = root.get("schema_version")
    supported_versions = {
        LEGACY_SUITE_CONFIG_SCHEMA_VERSION,
        PRECONDITIONING_LEGACY_SUITE_CONFIG_SCHEMA_VERSION,
        SUITE_CONFIG_SCHEMA_VERSION,
    }
    if suite_schema_version not in supported_versions:
        raise ValueError(
            f"unsupported suite schema_version {suite_schema_version!r}; "
            f"expected one of {sorted(supported_versions)!r}"
        )
    tasks = tuple(
        _task(
            item,
            index=index,
            suite_schema_version=suite_schema_version,
        )
        for index, item in enumerate(_sequence(root["tasks"], field="tasks"))
    )
    experiments = tuple(
        _experiment(item, index=index)
        for index, item in enumerate(_sequence(root.get("experiments", []), field="experiments"))
    )
    return BenchmarkSuite(name=root["name"], tasks=tasks, experiments=experiments)


def load_benchmark_suite(path: str | Path) -> BenchmarkSuite:
    """Load a strict benchmark suite from YAML."""

    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as input_file:
        value = yaml.safe_load(input_file)
    return benchmark_suite_from_mapping(value)


def external_trainer_config_from_mapping(value: Mapping[str, Any]) -> ExternalTrainerConfig:
    root = _mapping(value, field="trainer config")
    _only(
        root,
        {"schema_version", "algorithm_id", "seeds", "parameters", "notes"},
        field="trainer config",
    )
    if root.get("schema_version") != TRAINER_CONFIG_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported trainer schema_version {root.get('schema_version')!r}; "
            f"expected {TRAINER_CONFIG_SCHEMA_VERSION!r}"
        )
    return ExternalTrainerConfig(
        algorithm_id=root["algorithm_id"],
        seeds=tuple(_sequence(root["seeds"], field="trainer config.seeds")),
        parameters=root["parameters"],
        notes=root.get("notes", ""),
    )


def load_external_trainer_config(path: str | Path) -> ExternalTrainerConfig:
    import yaml

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as input_file:
        value = yaml.safe_load(input_file)
    return external_trainer_config_from_mapping(value)


__all__ = [
    "ExternalTrainerConfig",
    "LEGACY_SUITE_CONFIG_SCHEMA_VERSION",
    "PRECONDITIONING_LEGACY_SUITE_CONFIG_SCHEMA_VERSION",
    "SUITE_CONFIG_SCHEMA_VERSION",
    "TRAINER_CONFIG_SCHEMA_VERSION",
    "benchmark_suite_from_mapping",
    "external_trainer_config_from_mapping",
    "load_benchmark_suite",
    "load_external_trainer_config",
]
