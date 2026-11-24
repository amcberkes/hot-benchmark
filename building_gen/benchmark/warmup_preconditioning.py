"""Runtime provenance for the identity-bearing RunPeriod warmup contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from building_gen.benchmark._canonical import (
    canonical_json,
    refreeze_mapping,
    thaw_mapping,
)
from building_gen.benchmark.specs import (
    RUNPERIOD_WARMUP_KIND,
    RUNPERIOD_WARMUP_TREAT_WEATHER_AS_ACTUAL,
    RUNPERIOD_WEEKDAYS,
    CalendarWindowSpec,
    TaskSpec,
    WarmupPreconditioningSpec,
)

WARMUP_PRECONDITIONING_FILENAME = "warmup_preconditioning_evidence.json"
WARMUP_PRECONDITIONING_EVIDENCE_SCHEMA_VERSION = "hot-warmup-preconditioning-evidence/1.0.0"
WARMUP_PRECONDITIONING_V2_EVIDENCE_SCHEMA_VERSION = "hot-warmup-preconditioning-evidence/2.0.0"
EVALUATION_WARMUP_PRECONDITIONING_SCHEMA_VERSION = "hot-evaluation-warmup-preconditioning/1.0.0"
_SHA256_LENGTH = 64
_RUNPERIOD_FIXED_FIELDS = {
    "use_weather_file_holidays_and_special_days": "Yes",
    "use_weather_file_daylight_saving_period": "Yes",
    "apply_weekend_holiday_rule": "No",
    "use_weather_file_rain_indicators": "Yes",
    "use_weather_file_snow_indicators": "Yes",
}


class WarmupPreconditioningEvidenceError(RuntimeError):
    """Raised when runtime evidence does not prove the declared contract."""


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _strict_integer(value: Any, *, field: str, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WarmupPreconditioningEvidenceError(f"{field} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise WarmupPreconditioningEvidenceError(f"{field} must be {qualifier}")
    return value


def _strict_count_mapping(value: Any, *, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise WarmupPreconditioningEvidenceError(f"{field} must be an object")
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key or key != str(int(key)):
            raise WarmupPreconditioningEvidenceError(f"{field} keys must be canonical integer strings")
        result[key] = _strict_integer(count, field=f"{field}.{key}")
    return result


@dataclass(frozen=True, slots=True)
class WarmupPreconditioningRuntimeContext:
    """Immutable task/model context forwarded to one EnergyPlus worker."""

    task_id: str
    source_building_sha256: str
    weather_sha256: str
    materialized_model_sha256: str
    run_period: tuple[tuple[str, Any], ...]
    contract: WarmupPreconditioningSpec

    def __post_init__(self) -> None:
        if not isinstance(self.task_id, str) or not self.task_id:
            raise ValueError("warmup runtime task_id cannot be empty")
        for field_name in (
            "source_building_sha256",
            "weather_sha256",
            "materialized_model_sha256",
        ):
            value = getattr(self, field_name)
            if not _is_sha256(value):
                raise ValueError(f"warmup runtime {field_name} must be a lowercase SHA256")
        if not isinstance(self.contract, WarmupPreconditioningSpec):
            raise TypeError("warmup runtime contract must be a WarmupPreconditioningSpec")
        object.__setattr__(
            self,
            "run_period",
            refreeze_mapping(self.run_period, field="warmup runtime RunPeriod"),
        )
        if len(self.run_period) != 1:
            raise ValueError("warmup runtime context requires exactly one RunPeriod")

    @property
    def run_period_mapping(self) -> dict[str, Any]:
        return thaw_mapping(self.run_period)

    @property
    def run_period_sha256(self) -> str:
        return canonical_mapping_sha256(self.run_period_mapping)


def canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    """Hash one JSON mapping with HOT's canonical serialization."""

    return _sha256_bytes(canonical_json(value).encode("utf-8"))


def warmup_preconditioning_evidence_schema_version(
    contract: WarmupPreconditioningSpec,
) -> str:
    """Return the leaf evidence schema required by one contract version."""

    if not isinstance(contract, WarmupPreconditioningSpec):
        raise TypeError("contract must be a WarmupPreconditioningSpec")
    if contract.cooling_coil_types:
        return "hot-warmup-preconditioning-evidence/3.0.0"
    return (
        WARMUP_PRECONDITIONING_V2_EVIDENCE_SCHEMA_VERSION
        if contract.requires_actual_weather_year
        else WARMUP_PRECONDITIONING_EVIDENCE_SCHEMA_VERSION
    )


def expected_warmup_run_period(
    *,
    contract: WarmupPreconditioningSpec,
    window: CalendarWindowSpec,
    calendar_year: int | None,
) -> dict[str, Any]:
    """Derive the only supported preconditioned RunPeriod from task identity."""

    if not contract.enabled:
        raise ValueError("cannot derive a preconditioned RunPeriod from a disabled contract")
    if contract.requires_actual_weather_year and (type(calendar_year) is not int or not 1 <= calendar_year <= 9999):
        raise ValueError("warmup preconditioning v2 requires an actual-weather calendar_year between 1 and 9999")
    reference_year = calendar_year if calendar_year is not None else 2001
    anchor = date(
        reference_year,
        contract.epw_data_period_start_month,
        contract.epw_data_period_start_day,
    )
    if calendar_year is not None and anchor.strftime("%A") != contract.epw_data_period_start_weekday:
        raise ValueError("identity-bearing EPW weekday anchor disagrees with the declared calendar year")
    window_start = date(reference_year, window.start_month, window.start_day)
    anchor_index = RUNPERIOD_WEEKDAYS.index(contract.epw_data_period_start_weekday)
    weekday = RUNPERIOD_WEEKDAYS[(anchor_index + (window_start - anchor).days) % len(RUNPERIOD_WEEKDAYS)]
    period: dict[str, Any] = {
        "begin_month": window.start_month,
        "begin_day_of_month": window.start_day,
        "end_month": window.end_month,
        "end_day_of_month": window.end_day,
        "day_of_week_for_start_day": weekday,
        **_RUNPERIOD_FIXED_FIELDS,
    }
    if calendar_year is not None:
        period["begin_year"] = calendar_year
        period["end_year"] = calendar_year
    if contract.requires_actual_weather_year:
        period["treat_weather_as_actual"] = RUNPERIOD_WARMUP_TREAT_WEATHER_AS_ACTUAL
    return {f"HOT_Benchmark_{window.name}": period}


def expected_warmup_clock_anchor(
    *,
    contract: WarmupPreconditioningSpec,
    window: CalendarWindowSpec,
    calendar_year: int | None,
) -> tuple[int, int]:
    """Return EnergyPlus day-of-week/day-of-year for the first decision."""

    run_period = expected_warmup_run_period(
        contract=contract,
        window=window,
        calendar_year=calendar_year,
    )
    period = run_period[f"HOT_Benchmark_{window.name}"]
    day_of_week = RUNPERIOD_WEEKDAYS.index(period["day_of_week_for_start_day"]) + 1
    reference_year = calendar_year if calendar_year is not None else 2001
    day_of_year = (
        date(
            reference_year,
            window.start_month,
            window.start_day,
        )
        .timetuple()
        .tm_yday
    )
    return day_of_week, day_of_year


def _read_evidence(path: Path) -> tuple[Mapping[str, Any], bytes]:
    if path.is_symlink():
        raise WarmupPreconditioningEvidenceError("warmup preconditioning evidence must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise WarmupPreconditioningEvidenceError(
            f"cannot resolve warmup preconditioning evidence {path}: {error}"
        ) from error
    if not resolved.is_file():
        raise WarmupPreconditioningEvidenceError(f"warmup preconditioning evidence is not a regular file: {resolved}")
    try:
        raw_bytes = resolved.read_bytes()
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise WarmupPreconditioningEvidenceError(
            f"cannot read warmup preconditioning evidence {resolved}: {error}"
        ) from error
    if not isinstance(payload, Mapping):
        raise WarmupPreconditioningEvidenceError("warmup preconditioning evidence must contain one JSON object")
    return payload, raw_bytes


def _validate_run_period(
    value: Any,
    *,
    contract: WarmupPreconditioningSpec,
    window: CalendarWindowSpec,
    calendar_year: int | None,
) -> None:
    if not isinstance(value, Mapping) or list(value) != [f"HOT_Benchmark_{window.name}"]:
        raise WarmupPreconditioningEvidenceError("warmup evidence must identify the exact materialized RunPeriod")
    period = value[f"HOT_Benchmark_{window.name}"]
    if not isinstance(period, Mapping):
        raise WarmupPreconditioningEvidenceError("warmup evidence RunPeriod definition must be an object")
    try:
        expected_mapping = expected_warmup_run_period(
            contract=contract,
            window=window,
            calendar_year=calendar_year,
        )
    except (TypeError, ValueError) as error:
        raise WarmupPreconditioningEvidenceError(f"cannot derive the identity-bearing RunPeriod: {error}") from error
    expected_period = expected_mapping[f"HOT_Benchmark_{window.name}"]
    # The frozen v1 AMY task/evidence predates the contract-v2 split.  Its
    # task identity carries a physical calendar year while its leaf evidence
    # contains EnergyPlus' historical actual-weather flag.  Require that
    # exact archived representation for v1 validation only; v1 execution
    # remains blocked and the v1 serializer above remains byte-for-byte fixed.
    if calendar_year is not None and not contract.requires_actual_weather_year:
        expected_period = {
            **expected_period,
            "treat_weather_as_actual": RUNPERIOD_WARMUP_TREAT_WEATHER_AS_ACTUAL,
        }
    if set(period) != set(expected_period):
        raise WarmupPreconditioningEvidenceError(
            "warmup evidence RunPeriod fields do not match the complete supported set: "
            f"missing={sorted(set(expected_period) - set(period))!r}, "
            f"unexpected={sorted(set(period) - set(expected_period))!r}"
        )
    mismatches = {
        name: (period[name], expected_value)
        for name, expected_value in expected_period.items()
        if type(period[name]) is not type(expected_value) or period[name] != expected_value
    }
    if mismatches:
        raise WarmupPreconditioningEvidenceError(
            f"warmup evidence RunPeriod does not match task-derived calendar/flags: {mismatches!r}"
        )


def validate_warmup_preconditioning_payload(
    payload: Mapping[str, Any],
    *,
    task: TaskSpec,
    window: CalendarWindowSpec,
    executed_model_sha256: str,
) -> Mapping[str, Any]:
    """Validate an already decoded enabled intervention evidence payload."""

    if not task.warmup_preconditioning.enabled:
        raise WarmupPreconditioningEvidenceError("disabled warmup preconditioning has no enabled evidence contract")
    if not _is_sha256(executed_model_sha256):
        raise WarmupPreconditioningEvidenceError("expected executed-model SHA256 is malformed")
    if not isinstance(payload, Mapping):
        raise WarmupPreconditioningEvidenceError("warmup preconditioning evidence payload must be an object")
    expected_top_level = {
        "schema_version",
        "task_id",
        "contract",
        "assets",
        "materialized_model_sha256",
        "materialized_model_contract",
        "run_period",
        "run_period_sha256",
        "callback_evidence",
        "observed_warmup_days",
        "cooling_coil_sizing_eio",
    }
    if set(payload) != expected_top_level:
        raise WarmupPreconditioningEvidenceError(
            "warmup evidence fields do not match the supported schema: "
            f"missing={sorted(expected_top_level - set(payload))!r}, "
            f"unexpected={sorted(set(payload) - expected_top_level)!r}"
        )
    expected_evidence_schema = warmup_preconditioning_evidence_schema_version(task.warmup_preconditioning)
    if payload["schema_version"] != expected_evidence_schema:
        raise WarmupPreconditioningEvidenceError("unsupported warmup preconditioning evidence schema")
    if payload["task_id"] != task.task_id:
        raise WarmupPreconditioningEvidenceError("warmup evidence task_id does not match TaskSpec")

    contract = payload["contract"]
    if not isinstance(contract, Mapping) or set(contract) != {
        "payload",
        "sha256",
    }:
        raise WarmupPreconditioningEvidenceError("warmup evidence contract must contain exactly payload and sha256")
    expected_contract = task.warmup_preconditioning.identity_payload()
    if contract["payload"] != expected_contract:
        raise WarmupPreconditioningEvidenceError("warmup evidence contract payload does not match TaskSpec")
    actual_contract_sha256 = _sha256_bytes(canonical_json(contract["payload"]).encode("utf-8"))
    if contract["sha256"] != actual_contract_sha256 or actual_contract_sha256 != task.warmup_preconditioning.sha256:
        raise WarmupPreconditioningEvidenceError("warmup evidence contract SHA256 is inconsistent")

    assets = payload["assets"]
    if not isinstance(assets, Mapping) or set(assets) != {
        "source_building_sha256",
        "weather_sha256",
    }:
        raise WarmupPreconditioningEvidenceError("warmup evidence assets have an invalid schema")
    if assets["source_building_sha256"] != task.building.sha256:
        raise WarmupPreconditioningEvidenceError("warmup evidence source-building SHA256 does not match TaskSpec")
    if assets["weather_sha256"] != task.weather.sha256:
        raise WarmupPreconditioningEvidenceError("warmup evidence weather SHA256 does not match TaskSpec")
    if payload["materialized_model_sha256"] != executed_model_sha256:
        raise WarmupPreconditioningEvidenceError("warmup evidence materialized-model SHA256 does not match the episode")
    model_contract = payload["materialized_model_contract"]
    if not isinstance(model_contract, Mapping) or set(model_contract) != {
        "maximum_number_of_warmup_days",
        "underlying_schedule_constant_values_c",
    }:
        raise WarmupPreconditioningEvidenceError("warmup evidence materialized-model contract has an invalid schema")
    materialized_cap = _strict_integer(
        model_contract["maximum_number_of_warmup_days"],
        field="materialized_model_contract.maximum_number_of_warmup_days",
        positive=True,
    )
    if materialized_cap != task.warmup_preconditioning.maximum_warmup_days:
        raise WarmupPreconditioningEvidenceError("materialized maximum warmup days do not match TaskSpec")
    schedule_records = model_contract["underlying_schedule_constant_values_c"]
    expected_schedule_names = task.warmup_preconditioning.underlying_schedule_names
    if not isinstance(schedule_records, list) or len(schedule_records) != len(expected_schedule_names):
        raise WarmupPreconditioningEvidenceError(
            "materialized cooling schedules do not prove the unchanged TaskSpec values"
        )
    expected_schedule_value = task.warmup_preconditioning.underlying_schedule_constant_value_c
    for record, expected_name in zip(
        schedule_records,
        expected_schedule_names,
        strict=True,
    ):
        if not isinstance(record, Mapping) or set(record) != {
            "name",
            "hourly_value_c",
        }:
            raise WarmupPreconditioningEvidenceError(
                "materialized cooling schedules do not prove the unchanged TaskSpec values"
            )
        value = record["hourly_value_c"]
        if (
            record["name"] != expected_name
            or not isinstance(value, float)
            or not math.isfinite(value)
            or value != expected_schedule_value
        ):
            raise WarmupPreconditioningEvidenceError(
                "materialized cooling schedules do not prove the unchanged TaskSpec values"
            )
    _validate_run_period(
        payload["run_period"],
        contract=task.warmup_preconditioning,
        window=window,
        calendar_year=task.calendar_year,
    )
    run_period_sha256 = payload["run_period_sha256"]
    if not _is_sha256(run_period_sha256):
        raise WarmupPreconditioningEvidenceError("warmup evidence RunPeriod SHA256 is malformed")
    expected_run_period_sha256 = canonical_mapping_sha256(payload["run_period"])
    if run_period_sha256 != expected_run_period_sha256:
        raise WarmupPreconditioningEvidenceError(
            "warmup evidence RunPeriod SHA256 does not authenticate the exact mapping"
        )

    callback = payload["callback_evidence"]
    callback_fields = {
        "callback_not_ready_count",
        "ready_callback_counts_by_kind",
        "ready_warmup_callback_counts_by_kind",
        "precondition_set_calls_by_kind",
        "precondition_reset_calls_by_kind",
        "precondition_set_readback_count",
        "precondition_set_readback_min_c",
        "precondition_set_readback_max_c",
        "handle_resolution_count",
        "resolved_actuator_keys",
        "policy_action_flow_count",
        "set_outside_gate_count",
        "reset_before_policy_action_violation_count",
        "callback_errors",
    }
    if not isinstance(callback, Mapping) or set(callback) != callback_fields:
        raise WarmupPreconditioningEvidenceError("warmup callback evidence fields do not match the supported schema")
    _strict_integer(
        callback["callback_not_ready_count"],
        field="callback_not_ready_count",
    )
    ready_by_kind = _strict_count_mapping(
        callback["ready_callback_counts_by_kind"],
        field="ready_callback_counts_by_kind",
    )
    warmup_by_kind = _strict_count_mapping(
        callback["ready_warmup_callback_counts_by_kind"],
        field="ready_warmup_callback_counts_by_kind",
    )
    set_by_kind = _strict_count_mapping(
        callback["precondition_set_calls_by_kind"],
        field="precondition_set_calls_by_kind",
    )
    reset_by_kind = _strict_count_mapping(
        callback["precondition_reset_calls_by_kind"],
        field="precondition_reset_calls_by_kind",
    )
    kind_key = str(RUNPERIOD_WARMUP_KIND)
    if set(set_by_kind) != {kind_key} or set(reset_by_kind) != {kind_key}:
        raise WarmupPreconditioningEvidenceError("preconditioning set/reset calls must occur only in kind_of_sim == 3")
    if set_by_kind[kind_key] <= 0 or reset_by_kind[kind_key] <= 0:
        raise WarmupPreconditioningEvidenceError(
            "enabled preconditioning requires both warmup sets and post-warmup resets"
        )
    warmup_callbacks = warmup_by_kind.get(kind_key, 0)
    ready_callbacks = ready_by_kind.get(kind_key, 0)
    if warmup_callbacks <= 0 or ready_callbacks <= warmup_callbacks:
        raise WarmupPreconditioningEvidenceError("kind-3 callback counts do not prove warmup and runtime phases")
    callbacks_per_warmup_day = 24 * task.episode.timesteps_per_hour
    if warmup_callbacks % callbacks_per_warmup_day != 0:
        raise WarmupPreconditioningEvidenceError(
            "kind-3 warmup callback count is not divisible by one declared simulation day"
        )
    derived_warmup_days = warmup_callbacks // callbacks_per_warmup_day
    actuator_count = len(task.warmup_preconditioning.actuator_keys)
    if set_by_kind[kind_key] != warmup_callbacks * actuator_count:
        raise WarmupPreconditioningEvidenceError(
            "warmup set count does not cover every actuator on every kind-3 warmup callback"
        )
    if reset_by_kind[kind_key] != (ready_callbacks - warmup_callbacks) * actuator_count:
        raise WarmupPreconditioningEvidenceError(
            "post-warmup reset count does not cover every actuator on every kind-3 callback"
        )
    readback_count = _strict_integer(
        callback["precondition_set_readback_count"],
        field="precondition_set_readback_count",
    )
    if readback_count != set_by_kind[kind_key]:
        raise WarmupPreconditioningEvidenceError("warmup actuator readback count does not match set count")
    expected_value = task.warmup_preconditioning.actuator_value_c
    for field_name in (
        "precondition_set_readback_min_c",
        "precondition_set_readback_max_c",
    ):
        value = callback[field_name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or float(value) != expected_value
        ):
            raise WarmupPreconditioningEvidenceError(f"{field_name} does not prove exact {expected_value} C actuation")
    if (
        _strict_integer(
            callback["handle_resolution_count"],
            field="handle_resolution_count",
        )
        != 1
    ):
        raise WarmupPreconditioningEvidenceError("preconditioning actuator handles must be resolved exactly once")
    if callback["resolved_actuator_keys"] != list(task.warmup_preconditioning.actuator_keys):
        raise WarmupPreconditioningEvidenceError("resolved preconditioning actuator keys do not match TaskSpec order")
    expected_policy_hours = (
        window.duration_hours_for_calendar_year(task.calendar_year)
        if task.warmup_preconditioning.requires_actual_weather_year
        else window.duration_hours
    )
    expected_policy_actions = expected_policy_hours * task.episode.timesteps_per_hour
    if (
        _strict_integer(
            callback["policy_action_flow_count"],
            field="policy_action_flow_count",
        )
        != expected_policy_actions
    ):
        raise WarmupPreconditioningEvidenceError("warmup evidence policy-action count does not cover the episode")
    for field_name in (
        "set_outside_gate_count",
        "reset_before_policy_action_violation_count",
    ):
        if _strict_integer(callback[field_name], field=field_name) != 0:
            raise WarmupPreconditioningEvidenceError(f"warmup evidence reports {field_name}")
    errors = callback["callback_errors"]
    if not isinstance(errors, list) or errors:
        raise WarmupPreconditioningEvidenceError("warmup callback evidence contains errors")

    warmup_days = _strict_integer(
        payload["observed_warmup_days"],
        field="observed_warmup_days",
        positive=True,
    )
    maximum_days = task.warmup_preconditioning.maximum_warmup_days
    assert maximum_days is not None
    if warmup_days > maximum_days:
        raise WarmupPreconditioningEvidenceError("observed warmup days exceed the identity-bearing cap")
    if derived_warmup_days != warmup_days:
        raise WarmupPreconditioningEvidenceError(
            "EIO warmup days do not equal the days derived from kind-3 warmup callbacks"
        )
    sizing = payload["cooling_coil_sizing_eio"]
    if not isinstance(sizing, Mapping) or set(sizing) != {
        "line_count",
        "sha256",
    }:
        raise WarmupPreconditioningEvidenceError("cooling-coil sizing EIO evidence has an invalid schema")
    line_count = _strict_integer(
        sizing["line_count"],
        field="cooling_coil_sizing_eio.line_count",
    )
    if not _is_sha256(sizing["sha256"]):
        raise WarmupPreconditioningEvidenceError("cooling-coil sizing EIO SHA256 is malformed")
    if (
        line_count != task.warmup_preconditioning.expected_cooling_coil_sizing_eio_line_count
        or sizing["sha256"] != task.warmup_preconditioning.expected_cooling_coil_sizing_eio_sha256
    ):
        raise WarmupPreconditioningEvidenceError(
            "cooling-coil sizing EIO slice does not match the identity-bearing baseline"
        )
    return payload


def validate_warmup_preconditioning_evidence(
    path: str | Path,
    *,
    task: TaskSpec,
    window: CalendarWindowSpec,
    executed_model_sha256: str,
) -> tuple[Mapping[str, Any], bytes]:
    """Read and validate one episode's exact enabled intervention evidence."""

    payload, raw_bytes = _read_evidence(Path(path))
    validated = validate_warmup_preconditioning_payload(
        payload,
        task=task,
        window=window,
        executed_model_sha256=executed_model_sha256,
    )
    return validated, raw_bytes


__all__ = [
    "EVALUATION_WARMUP_PRECONDITIONING_SCHEMA_VERSION",
    "WARMUP_PRECONDITIONING_EVIDENCE_SCHEMA_VERSION",
    "WARMUP_PRECONDITIONING_V2_EVIDENCE_SCHEMA_VERSION",
    "WARMUP_PRECONDITIONING_FILENAME",
    "WarmupPreconditioningEvidenceError",
    "WarmupPreconditioningRuntimeContext",
    "canonical_mapping_sha256",
    "expected_warmup_run_period",
    "expected_warmup_clock_anchor",
    "warmup_preconditioning_evidence_schema_version",
    "validate_warmup_preconditioning_evidence",
    "validate_warmup_preconditioning_payload",
]
