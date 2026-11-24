"""Immutable task and experiment specifications for HOT control benchmarks."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import PurePosixPath
from typing import Any, Literal

from building_gen.benchmark._canonical import (
    canonical_json,
    content_id,
    refreeze_mapping,
    thaw_mapping,
)

LEGACY_TASK_SCHEMA_VERSION = "hot-control-task/1.1.0"
TASK_SCHEMA_VERSION = "hot-control-task/1.2.0"
EXPERIMENT_SCHEMA_VERSION = "hot-control-experiment/1.0.0"
WARMUP_PRECONDITIONING_SCHEMA_VERSION = "hot-runperiod-warmup-api-contract/1.0.0"
WARMUP_PRECONDITIONING_V2_SCHEMA_VERSION = "hot-runperiod-warmup-api-contract/2.0.0"
WARMUP_PRECONDITIONING_V3_SCHEMA_VERSION = "hot-runperiod-warmup-api-contract/3.0.0"
RUNPERIOD_WARMUP_PRECONDITIONING_V3_STRATEGY = "runperiod-cooling-setpoint-preconditioning-v3"
WARMUP_PRECONDITIONING_DISABLED_STRATEGY = "disabled"
RUNPERIOD_WARMUP_PRECONDITIONING_STRATEGY = "runperiod-cooling-setpoint-preconditioning-v1"
RUNPERIOD_WARMUP_PRECONDITIONING_V2_STRATEGY = "runperiod-cooling-setpoint-preconditioning-v2"
RUNPERIOD_WARMUP_TREAT_WEATHER_AS_ACTUAL = "Yes"
RUNPERIOD_WARMUP_CALLBACK = "callback_begin_system_timestep_before_predictor"
RUNPERIOD_WARMUP_READINESS_GATE = "api_data_fully_ready"
RUNPERIOD_WARMUP_KIND = 3
RUNPERIOD_WARMUP_KIND_NAME = "Weather File Run Period"
RUNPERIOD_WARMUP_COMPONENT_TYPE = "Zone Temperature Control"
RUNPERIOD_WARMUP_CONTROL_TYPE = "Cooling Setpoint"
RUNPERIOD_WARMUP_POST_OPERATION = "reset_actuator_before_policy_action"
RUNPERIOD_WEEKDAYS = (
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
)
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REPRESENTATIVE_CALENDAR_YEAR = 2001
_LEAP_CALENDAR_VALIDATION_YEAR = 2000


def _identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise ValueError(
            f"{field_name} must start with an alphanumeric character and contain only "
            "letters, numbers, '.', '_' or '-'"
        )
    return value


def _nonempty(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} cannot be empty")
    return value.strip()


@dataclass(frozen=True)
class AssetSpec:
    """Portable reference to one building or weather asset.

    Paths are data-root-relative and are sufficient for declaring a benchmark
    before the files are materialized.  A SHA256 should be supplied for a
    publication run so the exact bytes can be verified.
    """

    asset_id: str
    relative_path: str
    sha256: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_id", _identifier(self.asset_id, field_name="asset_id"))
        path = PurePosixPath(_nonempty(self.relative_path, field_name="relative_path"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative_path must be data-root-relative and cannot contain '..'")
        object.__setattr__(self, "relative_path", path.as_posix())
        if self.sha256 is not None:
            normalized = self.sha256.lower().removeprefix("sha256:")
            if not _SHA256_RE.fullmatch(normalized):
                raise ValueError("sha256 must contain exactly 64 hexadecimal characters")
            object.__setattr__(self, "sha256", normalized)

    def identity_payload(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "relative_path": self.relative_path,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SpaceSpec:
    """Framework-neutral signature of an observation or action space."""

    kind: str
    shape: tuple[int, ...]
    dtype: str = "float32"
    encoding: str = "raw"
    lower_bound: float | None = None
    upper_bound: float | None = None

    def __post_init__(self) -> None:
        kind = self.kind.lower().strip()
        if kind not in {"box", "discrete", "multidiscrete"}:
            raise ValueError(f"unsupported space kind {self.kind!r}")
        shape = tuple(self.shape)
        if not shape or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in shape):
            raise ValueError("shape must contain positive integers")
        if (self.lower_bound is None) != (self.upper_bound is None):
            raise ValueError("lower_bound and upper_bound must be provided together")
        if self.lower_bound is not None:
            bounds = (self.lower_bound, self.upper_bound)
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                for value in bounds
            ):
                raise ValueError("space bounds must be finite numeric values")
            lower_bound, upper_bound = (float(value) for value in bounds)
            if lower_bound >= upper_bound:
                raise ValueError("lower_bound must be smaller than upper_bound")
            object.__setattr__(self, "lower_bound", lower_bound)
            object.__setattr__(self, "upper_bound", upper_bound)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", _nonempty(self.dtype, field_name="dtype"))
        object.__setattr__(self, "encoding", _identifier(self.encoding, field_name="encoding"))

    @property
    def flat_size(self) -> int:
        size = 1
        for dimension in self.shape:
            size *= dimension
        return size

    def signature(self) -> tuple[Any, ...]:
        return (
            self.kind,
            self.shape,
            self.dtype,
            self.encoding,
            self.lower_bound,
            self.upper_bound,
        )


@dataclass(frozen=True)
class ActionChannelSpec:
    """One physical control channel repeated for every controlled entity."""

    name: str
    unit: str
    physical_min: float
    physical_max: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field_name="action channel name"))
        object.__setattr__(self, "unit", _nonempty(self.unit, field_name="action channel unit"))
        bounds = (self.physical_min, self.physical_max)
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
            for value in bounds
        ):
            raise ValueError(f"action channel {self.name!r} requires finite numeric bounds")
        minimum, maximum = (float(value) for value in bounds)
        if minimum >= maximum:
            raise ValueError(f"invalid physical bounds for action channel {self.name!r}")
        object.__setattr__(self, "physical_min", minimum)
        object.__setattr__(self, "physical_max", maximum)


@dataclass(frozen=True)
class ActionProfileSpec:
    """Semantic layout of an action vector."""

    profile_id: str
    controlled_entities: tuple[str, ...]
    channels: tuple[ActionChannelSpec, ...]
    transform_id: str = "linear_quantized_0_1c_legacy"

    def __post_init__(self) -> None:
        object.__setattr__(self, "profile_id", _identifier(self.profile_id, field_name="action profile_id"))
        object.__setattr__(self, "transform_id", _identifier(self.transform_id, field_name="action transform_id"))
        entities = tuple(self.controlled_entities)
        channels = tuple(self.channels)
        if not entities:
            raise ValueError("an action profile requires at least one controlled entity")
        if len(set(entities)) != len(entities) or any(not str(item).strip() for item in entities):
            raise ValueError("controlled_entities must be unique, non-empty names")
        if not channels:
            raise ValueError("an action profile requires at least one channel")
        if len({channel.name for channel in channels}) != len(channels):
            raise ValueError("action channel names must be unique")
        if self.transform_id == "linear_quantized_0_1c_legacy":
            non_aligned = [
                value
                for channel in channels
                for value in (channel.physical_min, channel.physical_max)
                if not math.isclose(value * 10.0, round(value * 10.0), abs_tol=1e-9)
            ]
            if non_aligned:
                raise ValueError("legacy tenth-degree action bounds must be aligned to 0.1 C")
        object.__setattr__(self, "controlled_entities", entities)
        object.__setattr__(self, "channels", channels)

    @property
    def expected_flat_size(self) -> int:
        return len(self.controlled_entities) * len(self.channels)

    def transfer_signature(self) -> tuple[Any, ...]:
        """Semantics a fixed-size checkpoint assumes across tasks.

        Entity names may differ between buildings; their number, channel order,
        units, and physical scaling may not.
        """

        return (
            self.profile_id,
            self.transform_id,
            len(self.controlled_entities),
            tuple(
                (channel.name, channel.unit, channel.physical_min, channel.physical_max) for channel in self.channels
            ),
        )


@dataclass(frozen=True)
class EpisodeSpec:
    """The simulation horizon and temporal resolution used by a task."""

    timesteps_per_hour: int
    episode_length_hours: int
    training_end_behavior: str = "terminal"

    def __post_init__(self) -> None:
        if self.training_end_behavior not in ("terminal", "truncate"):
            raise ValueError("training_end_behavior must be terminal or truncate")
        for field_name, value in (
            ("timesteps_per_hour", self.timesteps_per_hour),
            ("episode_length_hours", self.episode_length_hours),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

    @property
    def max_steps(self) -> int:
        return self.timesteps_per_hour * self.episode_length_hours


@dataclass(frozen=True)
class RewardSpec:
    """Named reward implementation and all parameters that affect it."""

    reward_id: str
    parameters: tuple[tuple[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reward_id", _identifier(self.reward_id, field_name="reward_id"))
        object.__setattr__(
            self,
            "parameters",
            refreeze_mapping(self.parameters, field="reward parameters"),
        )

    @property
    def parameter_mapping(self) -> dict[str, Any]:
        return thaw_mapping(self.parameters)


@dataclass(frozen=True)
class CalendarWindowSpec:
    """Inclusive month/day RunPeriod used for one evaluation window."""

    name: str
    start_month: int
    start_day: int
    end_month: int
    end_day: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field_name="calendar window name"))
        endpoint_parts = (
            self.start_month,
            self.start_day,
            self.end_month,
            self.end_day,
        )
        if any(type(part) is not int for part in endpoint_parts):
            raise ValueError(f"calendar window {self.name!r} has an invalid month/day")
        try:
            start = date(
                _LEAP_CALENDAR_VALIDATION_YEAR,
                self.start_month,
                self.start_day,
            )
            end = date(
                _LEAP_CALENDAR_VALIDATION_YEAR,
                self.end_month,
                self.end_day,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"calendar window {self.name!r} has an invalid month/day") from error
        if end < start:
            raise ValueError("calendar windows cannot cross the end of the year")

    def date_bounds_for_calendar_year(
        self,
        calendar_year: int | None = None,
    ) -> tuple[date, date]:
        """Resolve inclusive endpoints in one representative or physical year.

        ``None`` retains the historical non-leap 2001 interpretation.  An
        explicit year must be a native :class:`int` supported by
        :class:`datetime.date`; notably, ``bool`` and integer-like objects are
        rejected rather than silently reinterpreted.
        """

        if calendar_year is None:
            resolved_year = _REPRESENTATIVE_CALENDAR_YEAR
        else:
            if type(calendar_year) is not int:
                raise ValueError("calendar_year must be None or a native integer")
            if not 1 <= calendar_year <= 9999:
                raise ValueError("calendar_year must be between 1 and 9999")
            resolved_year = calendar_year

        try:
            start = date(resolved_year, self.start_month, self.start_day)
            end = date(resolved_year, self.end_month, self.end_day)
        except ValueError as error:
            raise ValueError(
                f"calendar window {self.name!r} is invalid for calendar year " f"{resolved_year}"
            ) from error
        if end < start:
            raise ValueError("calendar windows cannot cross the end of the year")
        return start, end

    def dates_for_calendar_year(
        self,
        calendar_year: int | None = None,
    ) -> tuple[date, ...]:
        """Return every date in the inclusive year-aware window."""

        start, end = self.date_bounds_for_calendar_year(calendar_year)
        return tuple(start + timedelta(days=offset) for offset in range((end - start).days + 1))

    def duration_hours_for_calendar_year(
        self,
        calendar_year: int | None = None,
    ) -> int:
        """Return inclusive whole-day duration for a representative or actual year."""

        start, end = self.date_bounds_for_calendar_year(calendar_year)
        return ((end - start).days + 1) * 24

    @property
    def duration_hours(self) -> int:
        """Historical duration under the representative non-leap 2001 calendar."""

        return self.duration_hours_for_calendar_year()


@dataclass(frozen=True)
class EvaluationProtocolSpec:
    """Calendar windows and repeat count for comparable policy evaluation."""

    protocol_id: str
    windows: tuple[CalendarWindowSpec, ...]
    episodes_per_window: int = 1
    deterministic_actions: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "protocol_id", _identifier(self.protocol_id, field_name="protocol_id"))
        windows = tuple(self.windows)
        if not windows or len({window.name for window in windows}) != len(windows):
            raise ValueError("evaluation windows must be non-empty and have unique names")
        if (
            isinstance(self.episodes_per_window, bool)
            or not isinstance(self.episodes_per_window, int)
            or self.episodes_per_window <= 0
        ):
            raise ValueError("episodes_per_window must be a positive integer")
        object.__setattr__(self, "windows", windows)


@dataclass(frozen=True, slots=True)
class WarmupPreconditioningSpec:
    """Identity-bearing EnergyPlus RunPeriod warmup intervention.

    ``disabled()`` is a real contract, not the absence of a contract.  The
    enabled strategy is deliberately narrow: it may only override cooling
    setpoint actuators while EnergyPlus reports a Weather File Run Period
    warmup, and it must release those actuators before every ordinary policy
    action.  Values that describe the callback gate are retained in the task
    identity so a future implementation cannot silently reinterpret them.
    """

    strategy_id: str = WARMUP_PRECONDITIONING_DISABLED_STRATEGY
    maximum_warmup_days: int | None = None
    expected_source_maximum_warmup_days: int | None = None
    epw_data_period_start_weekday: str | None = None
    epw_data_period_start_month: int | None = None
    epw_data_period_start_day: int | None = None
    callback: str | None = None
    readiness_gate: str | None = None
    required_kind_of_sim: int | None = None
    required_kind_name: str | None = None
    required_warmup_flag: bool | None = None
    actuator_component_type: str | None = None
    actuator_control_type: str | None = None
    actuator_keys: tuple[str, ...] = ()
    actuator_value_c: float | None = None
    underlying_schedule_names: tuple[str, ...] = ()
    underlying_schedule_constant_value_c: float | None = None
    cooling_coil_types: tuple[str, ...] = ()
    expected_cooling_coil_sizing_eio_line_count: int | None = None
    expected_cooling_coil_sizing_eio_sha256: str | None = None
    post_warmup_operation: str | None = None
    schema_version: str = WARMUP_PRECONDITIONING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version not in {
            WARMUP_PRECONDITIONING_SCHEMA_VERSION,
            WARMUP_PRECONDITIONING_V2_SCHEMA_VERSION,
            WARMUP_PRECONDITIONING_V3_SCHEMA_VERSION,
        }:
            raise ValueError("unsupported warmup preconditioning schema_version " f"{self.schema_version!r}")
        object.__setattr__(
            self,
            "strategy_id",
            _identifier(self.strategy_id, field_name="warmup preconditioning strategy_id"),
        )
        object.__setattr__(self, "actuator_keys", tuple(self.actuator_keys))
        object.__setattr__(
            self,
            "underlying_schedule_names",
            tuple(self.underlying_schedule_names),
        )
        object.__setattr__(self, "cooling_coil_types", tuple(self.cooling_coil_types))
        if self.schema_version == WARMUP_PRECONDITIONING_V3_SCHEMA_VERSION:
            if self.strategy_id != RUNPERIOD_WARMUP_PRECONDITIONING_V3_STRATEGY:
                raise ValueError("v3 requires its explicit enabled strategy")
            if (not self.cooling_coil_types or any(not isinstance(t, str) or not t.startswith("Coil:Cooling:") or t != t.strip() for t in self.cooling_coil_types)
                    or len({t.casefold() for t in self.cooling_coil_types}) != len(self.cooling_coil_types)):
                raise ValueError("v3 requires unique explicit cooling-coil types")
            object.__setattr__(self, "cooling_coil_types", tuple(sorted(self.cooling_coil_types)))
        elif self.cooling_coil_types:
            raise ValueError("explicit cooling-coil types require warmup contract v3")
        if self.strategy_id == WARMUP_PRECONDITIONING_DISABLED_STRATEGY:
            if self.schema_version != WARMUP_PRECONDITIONING_SCHEMA_VERSION:
                raise ValueError("disabled warmup preconditioning is frozen at schema version 1.0.0")
            disabled_defaults = {
                "maximum_warmup_days": (self.maximum_warmup_days, None),
                "expected_source_maximum_warmup_days": (
                    self.expected_source_maximum_warmup_days,
                    None,
                ),
                "epw_data_period_start_weekday": (
                    self.epw_data_period_start_weekday,
                    None,
                ),
                "epw_data_period_start_month": (
                    self.epw_data_period_start_month,
                    None,
                ),
                "epw_data_period_start_day": (
                    self.epw_data_period_start_day,
                    None,
                ),
                "callback": (self.callback, None),
                "readiness_gate": (self.readiness_gate, None),
                "required_kind_of_sim": (self.required_kind_of_sim, None),
                "required_kind_name": (self.required_kind_name, None),
                "required_warmup_flag": (self.required_warmup_flag, None),
                "actuator_component_type": (self.actuator_component_type, None),
                "actuator_control_type": (self.actuator_control_type, None),
                "actuator_keys": (self.actuator_keys, ()),
                "actuator_value_c": (self.actuator_value_c, None),
                "underlying_schedule_names": (
                    self.underlying_schedule_names,
                    (),
                ),
                "underlying_schedule_constant_value_c": (
                    self.underlying_schedule_constant_value_c,
                    None,
                ),
                "expected_cooling_coil_sizing_eio_line_count": (
                    self.expected_cooling_coil_sizing_eio_line_count,
                    None,
                ),
                "expected_cooling_coil_sizing_eio_sha256": (
                    self.expected_cooling_coil_sizing_eio_sha256,
                    None,
                ),
                "post_warmup_operation": (self.post_warmup_operation, None),
            }
            populated = sorted(
                name
                for name, (value, default) in disabled_defaults.items()
                if value != default or type(value) is not type(default)
            )
            if populated:
                raise ValueError(
                    "disabled warmup preconditioning cannot declare intervention fields: " f"{populated!r}"
                )
            return
        supported_enabled_contracts = {
            (WARMUP_PRECONDITIONING_V3_SCHEMA_VERSION, RUNPERIOD_WARMUP_PRECONDITIONING_V3_STRATEGY),
            (
                WARMUP_PRECONDITIONING_SCHEMA_VERSION,
                RUNPERIOD_WARMUP_PRECONDITIONING_STRATEGY,
            ),
            (
                WARMUP_PRECONDITIONING_V2_SCHEMA_VERSION,
                RUNPERIOD_WARMUP_PRECONDITIONING_V2_STRATEGY,
            ),
        }
        if (self.schema_version, self.strategy_id) not in supported_enabled_contracts:
            raise ValueError(f"unsupported warmup preconditioning strategy {self.strategy_id!r}")

        exact_fields = {
            "callback": (self.callback, RUNPERIOD_WARMUP_CALLBACK),
            "readiness_gate": (
                self.readiness_gate,
                RUNPERIOD_WARMUP_READINESS_GATE,
            ),
            "required_kind_of_sim": (
                self.required_kind_of_sim,
                RUNPERIOD_WARMUP_KIND,
            ),
            "required_kind_name": (
                self.required_kind_name,
                RUNPERIOD_WARMUP_KIND_NAME,
            ),
            "required_warmup_flag": (self.required_warmup_flag, True),
            "actuator_component_type": (
                self.actuator_component_type,
                RUNPERIOD_WARMUP_COMPONENT_TYPE,
            ),
            "actuator_control_type": (
                self.actuator_control_type,
                RUNPERIOD_WARMUP_CONTROL_TYPE,
            ),
            "post_warmup_operation": (
                self.post_warmup_operation,
                RUNPERIOD_WARMUP_POST_OPERATION,
            ),
        }
        mismatches = [
            f"{name}={actual!r} (expected {expected!r})"
            for name, (actual, expected) in exact_fields.items()
            if actual != expected
        ]
        if mismatches:
            raise ValueError(
                "enabled RunPeriod warmup preconditioning has unsupported semantics: " + "; ".join(mismatches)
            )
        for field_name in (
            "maximum_warmup_days",
            "expected_source_maximum_warmup_days",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.maximum_warmup_days <= self.expected_source_maximum_warmup_days:
            raise ValueError("maximum_warmup_days must exceed expected_source_maximum_warmup_days")
        if self.epw_data_period_start_weekday not in RUNPERIOD_WEEKDAYS:
            raise ValueError("epw_data_period_start_weekday must be one of " + ", ".join(RUNPERIOD_WEEKDAYS))
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (
                self.epw_data_period_start_month,
                self.epw_data_period_start_day,
            )
        ):
            raise ValueError("EPW data-period start month/day must be integers")
        try:
            date(
                2001,
                self.epw_data_period_start_month,
                self.epw_data_period_start_day,
            )
        except ValueError as error:
            raise ValueError("EPW data-period start month/day is invalid") from error
        for field_name in (
            "actuator_value_c",
            "underlying_schedule_constant_value_c",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field_name} must be a finite numeric value")
            object.__setattr__(self, field_name, float(value))
        for field_name, names in (
            ("actuator_keys", self.actuator_keys),
            ("underlying_schedule_names", self.underlying_schedule_names),
        ):
            if not names or any(not isinstance(name, str) or not name.strip() for name in names):
                raise ValueError(f"{field_name} must contain non-empty strings")
            if len({name.casefold() for name in names}) != len(names):
                raise ValueError(f"{field_name} must be unique ignoring case")
        if len(self.actuator_keys) != len(self.underlying_schedule_names):
            raise ValueError("underlying_schedule_names must identify one schedule per actuator key")
        if (
            isinstance(self.expected_cooling_coil_sizing_eio_line_count, bool)
            or not isinstance(
                self.expected_cooling_coil_sizing_eio_line_count,
                int,
            )
            or self.expected_cooling_coil_sizing_eio_line_count <= 0
        ):
            raise ValueError("expected_cooling_coil_sizing_eio_line_count must be a positive integer")
        if not isinstance(
            self.expected_cooling_coil_sizing_eio_sha256,
            str,
        ) or not _SHA256_RE.fullmatch(self.expected_cooling_coil_sizing_eio_sha256):
            raise ValueError("expected_cooling_coil_sizing_eio_sha256 must be an exact lowercase SHA256")

    @classmethod
    def disabled(cls) -> "WarmupPreconditioningSpec":
        return cls(strategy_id=WARMUP_PRECONDITIONING_DISABLED_STRATEGY)

    @property
    def enabled(self) -> bool:
        return self.strategy_id != WARMUP_PRECONDITIONING_DISABLED_STRATEGY

    @property
    def requires_actual_weather_year(self) -> bool:
        """Whether the contract requires a physical year and actual-weather RunPeriod."""

        return (
            self.schema_version == WARMUP_PRECONDITIONING_V2_SCHEMA_VERSION
            and self.strategy_id == RUNPERIOD_WARMUP_PRECONDITIONING_V2_STRATEGY
        )

    def identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "strategy_id": self.strategy_id,
        }
        if not self.enabled:
            return payload
        payload.update(
            {
                "maximum_warmup_days": self.maximum_warmup_days,
                "expected_source_maximum_warmup_days": (self.expected_source_maximum_warmup_days),
                "epw_data_period_start_weekday": self.epw_data_period_start_weekday,
                "epw_data_period_start_month": self.epw_data_period_start_month,
                "epw_data_period_start_day": self.epw_data_period_start_day,
                "callback": self.callback,
                "readiness_gate": self.readiness_gate,
                "required_kind_of_sim": self.required_kind_of_sim,
                "required_kind_name": self.required_kind_name,
                "required_warmup_flag": self.required_warmup_flag,
                "actuator_component_type": self.actuator_component_type,
                "actuator_control_type": self.actuator_control_type,
                "actuator_keys": list(self.actuator_keys),
                "actuator_value_c": self.actuator_value_c,
                "underlying_schedule_names": list(self.underlying_schedule_names),
                "underlying_schedule_constant_value_c": (self.underlying_schedule_constant_value_c),
                "expected_cooling_coil_sizing_eio_line_count": (self.expected_cooling_coil_sizing_eio_line_count),
                "expected_cooling_coil_sizing_eio_sha256": (self.expected_cooling_coil_sizing_eio_sha256),
                "post_warmup_operation": self.post_warmup_operation,
            }
        )
        if self.cooling_coil_types:
            payload["cooling_coil_types"] = list(self.cooling_coil_types)
        if self.requires_actual_weather_year:
            payload["treat_weather_as_actual"] = RUNPERIOD_WARMUP_TREAT_WEATHER_AS_ACTUAL
        return payload

    @property
    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.identity_payload()).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TaskSpec:
    """All environment-facing information needed by one benchmark task."""

    name: str
    building: AssetSpec
    weather: AssetSpec
    observation_space: SpaceSpec
    action_space: SpaceSpec
    action_profile: ActionProfileSpec
    episode: EpisodeSpec
    reward: RewardSpec
    warmup_preconditioning: WarmupPreconditioningSpec = field(default_factory=WarmupPreconditioningSpec.disabled)
    metadata: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    schema_version: str = TASK_SCHEMA_VERSION
    calendar_year: int | None = None
    component_energy_meters: tuple[tuple[str, str], ...] | None = None
    weekly_training_reward: RewardSpec | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field_name="task name"))
        if self.schema_version not in {
            LEGACY_TASK_SCHEMA_VERSION,
            TASK_SCHEMA_VERSION,
        }:
            raise ValueError(f"unsupported task schema_version {self.schema_version!r}")
        if self.weekly_training_reward is not None:
            from building_gen.weekly_reward import validate_training_reward_spec
            if self.schema_version != TASK_SCHEMA_VERSION or self.episode.episode_length_hours != 168:
                raise ValueError('weekly training reward requires current schema and seven-day episodes')
            validate_training_reward_spec(self.weekly_training_reward)
            if self.reward.reward_id != 'hot_scheduled_occupied_energy_comfort_v3':
                raise ValueError('weekly training reward requires scheduled completed-interval base scoring')
        if not isinstance(self.warmup_preconditioning, WarmupPreconditioningSpec):
            raise TypeError("warmup_preconditioning must be a WarmupPreconditioningSpec")
        if self.schema_version == LEGACY_TASK_SCHEMA_VERSION and self.warmup_preconditioning.enabled:
            raise ValueError("legacy TaskSpec schemas cannot enable warmup preconditioning")
        if self.action_space.flat_size != self.action_profile.expected_flat_size:
            raise ValueError(
                f"task {self.name!r} declares action shape {self.action_space.shape} "
                f"({self.action_space.flat_size} values), but action profile {self.action_profile.profile_id!r} "
                f"requires {self.action_profile.expected_flat_size} values"
            )
        if self.warmup_preconditioning.enabled:
            expected_zones = tuple(zone.casefold() for zone in self.action_profile.controlled_entities)
            actual_zones = tuple(zone.casefold() for zone in self.warmup_preconditioning.actuator_keys)
            if actual_zones != expected_zones:
                raise ValueError(
                    "warmup preconditioning actuator-key order must match the task's "
                    "controlled-entity order, ignoring EnergyPlus name case"
                )
        if self.component_energy_meters is not None:
            if self.schema_version != TASK_SCHEMA_VERSION:
                raise ValueError('component energy meters require the current task schema')
            meters = dict(self.component_energy_meters)
            if len(meters) != len(self.component_energy_meters):
                raise ValueError("duplicate component energy carrier")
            if 'electricity' not in meters or set(meters) - {'electricity', 'natural_gas'}:
                raise ValueError('component energy meters require electricity and optionally natural gas')
            boundaries = set()
            for carrier, name in meters.items():
                if not isinstance(name, str):
                    raise ValueError("component energy meter name must be a string")
                match = re.fullmatch(r'HOT:COMPONENTHVAC:' + carrier.upper() + r':([0-9A-F]{16})', name)
                if not match:
                    raise ValueError('invalid component energy meter name')
                boundaries.add(match[1])
            if len(boundaries) != 1:
                raise ValueError('component energy meters must share one boundary identity')
            object.__setattr__(self, 'component_energy_meters', tuple(sorted(meters.items())))
        metadata = refreeze_mapping(self.metadata, field="task metadata")
        metadata_mapping = thaw_mapping(metadata)

        # ``weather_year`` used to live in descriptive metadata, which meant
        # that it changed EnergyPlus' calendar without changing ``task_id``.
        # Promote that legacy spelling into the identity-bearing field so old
        # programmatic TaskSpecs remain reproducible while new manifests can
        # use ``calendar_year`` directly.
        legacy_year = metadata_mapping.pop("weather_year", None)
        calendar_year = self.calendar_year
        if legacy_year is not None:
            if calendar_year is not None and calendar_year != legacy_year:
                raise ValueError("calendar_year conflicts with metadata.weather_year")
            calendar_year = legacy_year
        if calendar_year is not None:
            if type(calendar_year) is not int or not 1 <= calendar_year <= 9999:
                raise ValueError("calendar_year must be a native integer between 1 and 9999")
        if self.warmup_preconditioning.requires_actual_weather_year:
            if calendar_year is None:
                raise ValueError("warmup preconditioning v2 requires an actual-weather calendar_year")
            assert self.warmup_preconditioning.epw_data_period_start_month is not None
            assert self.warmup_preconditioning.epw_data_period_start_day is not None
            anchor_weekday = date(
                calendar_year,
                self.warmup_preconditioning.epw_data_period_start_month,
                self.warmup_preconditioning.epw_data_period_start_day,
            ).strftime("%A")
            if anchor_weekday != self.warmup_preconditioning.epw_data_period_start_weekday:
                raise ValueError("warmup preconditioning v2 EPW weekday anchor disagrees with calendar_year")
        object.__setattr__(self, "calendar_year", calendar_year)
        object.__setattr__(
            self,
            "metadata",
            refreeze_mapping(metadata_mapping, field="task metadata"),
        )

    @property
    def publication_ready(self) -> bool:
        """Whether both input assets are pinned by content checksum."""

        return self.building.sha256 is not None and self.weather.sha256 is not None

    @property
    def calendar_year_semantics(
        self,
    ) -> Literal["actual_weather_year", "typical_weather_year"]:
        """Return the task-authoritative interpretation of simulator years.

        This is derived entirely from the existing identity-bearing
        ``calendar_year`` field and is deliberately absent from
        :meth:`identity_payload`, so valid historical task IDs do not change.
        """

        return "actual_weather_year" if self.calendar_year is not None else "typical_weather_year"

    def identity_payload(self) -> dict[str, Any]:
        """Return the semantic payload used to derive ``task_id``."""

        payload = {
            "schema_version": self.schema_version,
            "building": self.building.identity_payload(),
            "weather": self.weather.identity_payload(),
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "action_profile": self.action_profile,
            "episode": {
                "timesteps_per_hour": self.episode.timesteps_per_hour,
                "episode_length_hours": self.episode.episode_length_hours,
                **({"training_end_behavior": self.episode.training_end_behavior}
                   if self.episode.training_end_behavior != "terminal" else {}),
            },
            "calendar_year": self.calendar_year,
            "reward": {
                "reward_id": self.reward.reward_id,
                "parameters": self.reward.parameter_mapping,
            },
        }
        if self.weekly_training_reward is not None:
            payload['weekly_training_reward'] = {
                'reward_id': self.weekly_training_reward.reward_id,
                'parameters': self.weekly_training_reward.parameter_mapping,
            }
        if self.component_energy_meters is not None:
            payload["component_energy_meters"] = dict(self.component_energy_meters)
        if self.schema_version == TASK_SCHEMA_VERSION:
            payload["warmup_preconditioning"] = self.warmup_preconditioning.identity_payload()
        elif self.warmup_preconditioning.enabled:
            raise ValueError("legacy TaskSpec schemas cannot enable warmup preconditioning")
        return payload

    @property
    def task_id(self) -> str:
        return content_id("HOT control task", self.identity_payload(), prefix="task")


@dataclass(frozen=True)
class ExperimentSpec:
    """A model/policy, training tasks, evaluation tasks, and repeat seeds."""

    name: str
    algorithm_id: str
    train_tasks: tuple[str, ...]
    evaluation_tasks: tuple[str, ...]
    seeds: tuple[int, ...]
    parameters: tuple[tuple[str, Any], ...] = field(default_factory=tuple)
    enabled: bool = True
    notes: str = ""
    evaluation_protocol: EvaluationProtocolSpec | None = None
    schema_version: str = EXPERIMENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field_name="experiment name"))
        object.__setattr__(self, "algorithm_id", _identifier(self.algorithm_id, field_name="algorithm_id"))
        train_tasks = tuple(self.train_tasks)
        evaluation_tasks = tuple(self.evaluation_tasks)
        seeds = tuple(self.seeds)
        if len(set(train_tasks)) != len(train_tasks):
            raise ValueError("train_tasks cannot contain duplicates")
        if not evaluation_tasks or len(set(evaluation_tasks)) != len(evaluation_tasks):
            raise ValueError("evaluation_tasks must be non-empty and cannot contain duplicates")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be non-empty and cannot contain duplicates")
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise ValueError("seeds must be non-negative integers")
        object.__setattr__(
            self,
            "parameters",
            refreeze_mapping(self.parameters, field="experiment parameters"),
        )
        object.__setattr__(self, "train_tasks", train_tasks)
        object.__setattr__(self, "evaluation_tasks", evaluation_tasks)
        object.__setattr__(self, "seeds", seeds)

    @property
    def parameter_mapping(self) -> dict[str, Any]:
        return thaw_mapping(self.parameters)

    def identity_payload(
        self,
        *,
        task_ids: Mapping[str, str],
        algorithm_version: str,
    ) -> dict[str, Any]:
        missing = sorted((set(self.train_tasks) | set(self.evaluation_tasks)) - set(task_ids))
        if missing:
            raise KeyError(f"experiment {self.name!r} references unknown task(s): {', '.join(missing)}")
        return {
            "schema_version": self.schema_version,
            "algorithm_id": self.algorithm_id,
            "algorithm_version": algorithm_version,
            "train_task_ids": [task_ids[name] for name in self.train_tasks],
            "evaluation_task_ids": [task_ids[name] for name in self.evaluation_tasks],
            "seeds": self.seeds,
            "parameters": self.parameter_mapping,
            "evaluation_protocol": self.evaluation_protocol,
        }

    def experiment_id(self, *, task_ids: Mapping[str, str], algorithm_version: str) -> str:
        return content_id(
            "HOT control experiment",
            self.identity_payload(task_ids=task_ids, algorithm_version=algorithm_version),
            prefix="experiment",
        )


@dataclass(frozen=True)
class BenchmarkSuite:
    """An immutable named collection of tasks and experiment plans."""

    name: str
    tasks: tuple[TaskSpec, ...]
    experiments: tuple[ExperimentSpec, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _identifier(self.name, field_name="suite name"))
        tasks = tuple(self.tasks)
        experiments = tuple(self.experiments)
        if not tasks:
            raise ValueError("a benchmark suite requires at least one task")
        task_names = [task.name for task in tasks]
        if len(set(task_names)) != len(task_names):
            raise ValueError("task names must be unique within a benchmark suite")
        experiment_names = [experiment.name for experiment in experiments]
        if len(set(experiment_names)) != len(experiment_names):
            raise ValueError("experiment names must be unique within a benchmark suite")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "experiments", experiments)

    @property
    def task_map(self) -> dict[str, TaskSpec]:
        return {task.name: task for task in self.tasks}

    @property
    def experiment_map(self) -> dict[str, ExperimentSpec]:
        return {experiment.name: experiment for experiment in self.experiments}


__all__ = [
    "ActionChannelSpec",
    "ActionProfileSpec",
    "AssetSpec",
    "BenchmarkSuite",
    "CalendarWindowSpec",
    "EpisodeSpec",
    "ExperimentSpec",
    "EvaluationProtocolSpec",
    "RewardSpec",
    "SpaceSpec",
    "TaskSpec",
    "TASK_SCHEMA_VERSION",
    "LEGACY_TASK_SCHEMA_VERSION",
    "WarmupPreconditioningSpec",
    "WARMUP_PRECONDITIONING_SCHEMA_VERSION",
    "WARMUP_PRECONDITIONING_V2_SCHEMA_VERSION",
    "WARMUP_PRECONDITIONING_V3_SCHEMA_VERSION",
    "WARMUP_PRECONDITIONING_DISABLED_STRATEGY",
    "RUNPERIOD_WARMUP_PRECONDITIONING_STRATEGY",
    "RUNPERIOD_WARMUP_PRECONDITIONING_V2_STRATEGY",
    "RUNPERIOD_WARMUP_PRECONDITIONING_V3_STRATEGY",
    "RUNPERIOD_WARMUP_TREAT_WEATHER_AS_ACTUAL",
    "RUNPERIOD_WARMUP_CALLBACK",
    "RUNPERIOD_WARMUP_READINESS_GATE",
    "RUNPERIOD_WARMUP_KIND",
    "RUNPERIOD_WARMUP_KIND_NAME",
    "RUNPERIOD_WARMUP_COMPONENT_TYPE",
    "RUNPERIOD_WARMUP_CONTROL_TYPE",
    "RUNPERIOD_WARMUP_POST_OPERATION",
    "RUNPERIOD_WEEKDAYS",
]
