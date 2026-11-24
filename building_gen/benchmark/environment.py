"""Materialize reproducible, calendar-aware HOT benchmark environments.

This module deliberately imports neither Gymnasium nor EnergyPlus at import
time.  Asset and RunPeriod handling can therefore be tested on a workstation;
the EnergyPlus-backed simulator factory is imported only when an environment
is actually created.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

from building_gen.benchmark.energyplus_diagnostics import DIAGNOSTICS_FILENAME
from building_gen.benchmark.specs import (
    LEGACY_TASK_SCHEMA_VERSION,
    TASK_SCHEMA_VERSION,
    AssetSpec,
    CalendarWindowSpec,
    RewardSpec,
    TaskSpec,
    WarmupPreconditioningSpec,
)
from building_gen.benchmark.warmup_preconditioning import (
    WARMUP_PRECONDITIONING_FILENAME,
    WarmupPreconditioningRuntimeContext,
    expected_warmup_run_period,
)
from building_gen.reward_foundation import (
    HOT_OCCUPIED_ENERGY_COMFORT_V2,
    OccupiedEnergyComfortV2Config,
)


class AssetIntegrityError(ValueError):
    """Raised when a benchmark asset is absent or differs from its manifest."""


class EnvironmentCompatibilityError(ValueError):
    """Raised when the constructed environment disagrees with its task spec."""


class WeatherCalendarError(ValueError):
    """Raised when an EPW cannot define an unambiguous simulation calendar."""


@dataclass(frozen=True, slots=True)
class ResolvedTaskAssets:
    """Checksum-verified absolute paths for one task's immutable inputs."""

    building: Path
    weather: Path


@dataclass(frozen=True, slots=True)
class EpwDataPeriod:
    """The single calendar period declared by an EPW ``DATA PERIODS`` header."""

    name: str
    records_per_hour: int
    start_weekday: str
    start_month: int
    start_day: int
    end_month: int
    end_day: int
    start_year: int | None = None
    end_year: int | None = None


_WEEKDAYS = (
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
)
_WEEKDAY_BY_CASEFOLD = {weekday.casefold(): weekday for weekday in _WEEKDAYS}
_EPW_DATE_RE = re.compile(r"^\s*(\d{1,2})\s*/\s*(\d{1,2})(?:\s*/\s*(\d{4}))?\s*$")


def _epw_month_day(value: str, *, field_name: str, path: Path) -> tuple[int | None, int, int]:
    match = _EPW_DATE_RE.fullmatch(value)
    if match is None:
        raise WeatherCalendarError(
            f"EPW {path} has invalid {field_name} {value!r}; expected month/day or month/day/year"
        )
    month = int(match.group(1))
    day = int(match.group(2))
    year = int(match.group(3)) if match.group(3) is not None else None
    try:
        date(year if year is not None else 2001, month, day)
    except ValueError as error:
        raise WeatherCalendarError(f"EPW {path} has invalid {field_name} {value!r}") from error
    return year, month, day


@lru_cache(maxsize=128)
def _read_epw_data_period_cached(
    resolved_path: str,
    size_bytes: int,
    modified_time_ns: int,
) -> EpwDataPeriod:
    """Parse a data period; stat fields make the path cache safe if it changes."""

    del size_bytes, modified_time_ns  # Values deliberately participate in the cache key.
    path = Path(resolved_path)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as input_file:
            matches: list[tuple[int, list[str]]] = []
            for line_number, row in enumerate(csv.reader(input_file), start=1):
                if not row:
                    continue
                first_field = row[0].strip()
                # An EPW data record begins with its numeric year. DATA PERIODS
                # belongs to the fixed header, so the 8,760 records need not be
                # scanned on every environment construction.
                if first_field.lstrip("+-").isdigit():
                    break
                if first_field.casefold() == "data periods":
                    matches.append((line_number, row))
                # The standard header is eight records. This generous cap
                # prevents a malformed non-EPW text file from being scanned in
                # full while still detecting duplicate header rows.
                if line_number >= 64:
                    break
    except UnicodeDecodeError as error:
        raise WeatherCalendarError(f"EPW {path} is not valid UTF-8 text") from error

    if len(matches) != 1:
        raise WeatherCalendarError(f"EPW {path} must contain exactly one DATA PERIODS header; found {len(matches)}")
    line_number, row = matches[0]
    if len(row) < 7:
        raise WeatherCalendarError(
            f"EPW {path} DATA PERIODS header on line {line_number} has {len(row)} fields; " "expected at least 7"
        )
    if any(field.strip() for field in row[7:]):
        raise WeatherCalendarError(f"EPW {path} DATA PERIODS header on line {line_number} has unexpected fields")
    try:
        number_of_periods = int(row[1].strip())
        records_per_hour = int(row[2].strip())
    except ValueError as error:
        raise WeatherCalendarError(f"EPW {path} DATA PERIODS counts must be integers") from error
    if number_of_periods != 1:
        raise WeatherCalendarError(f"EPW {path} declares {number_of_periods} data periods; HOT requires exactly one")
    if records_per_hour <= 0:
        raise WeatherCalendarError(f"EPW {path} records per hour must be a positive integer")

    period_name = row[3].strip()
    if not period_name:
        raise WeatherCalendarError(f"EPW {path} data-period name cannot be empty")
    weekday_token = row[4].strip()
    weekday = _WEEKDAY_BY_CASEFOLD.get(weekday_token.casefold())
    if weekday is None:
        raise WeatherCalendarError(
            f"EPW {path} has invalid start weekday {weekday_token!r}; " f"expected one of {', '.join(_WEEKDAYS)}"
        )
    start_year, start_month, start_day = _epw_month_day(row[5], field_name="data-period start date", path=path)
    end_year, end_month, end_day = _epw_month_day(row[6], field_name="data-period end date", path=path)
    if (start_year is None) != (end_year is None):
        raise WeatherCalendarError(f"EPW {path} DATA PERIODS dates must either both include a year or both omit it")
    if start_year is not None and start_year != end_year:
        raise WeatherCalendarError(f"EPW {path} has a cross-year data period, which HOT does not support")
    reference_year = start_year if start_year is not None else 2001
    if date(reference_year, end_month, end_day) < date(reference_year, start_month, start_day):
        raise WeatherCalendarError(f"EPW {path} has a cross-year data period, which HOT does not support")
    return EpwDataPeriod(
        name=period_name,
        records_per_hour=records_per_hour,
        start_weekday=weekday,
        start_year=start_year,
        start_month=start_month,
        start_day=start_day,
        end_year=end_year,
        end_month=end_month,
        end_day=end_day,
    )


def read_epw_data_period(weather_file: str | Path) -> EpwDataPeriod:
    """Parse the one continuous data period required by HOT weather assets.

    EPW permits multiple data periods, but a benchmark window would then need
    an explicit rule for choosing among them.  HOT therefore rejects that
    ambiguity, as well as duplicated ``DATA PERIODS`` header rows.
    """

    path = Path(weather_file).resolve()
    stat = path.stat()
    return _read_epw_data_period_cached(
        str(path),
        stat.st_size,
        stat.st_mtime_ns,
    )


@lru_cache(maxsize=128)
def _read_epw_record_years_cached(
    resolved_path: str,
    size_bytes: int,
    modified_time_ns: int,
) -> tuple[int, ...]:
    """Return all numeric years present in the EPW weather records."""

    del size_bytes, modified_time_ns
    years: set[int] = set()
    with Path(resolved_path).open("r", encoding="utf-8-sig", newline="") as input_file:
        for row in csv.reader(input_file):
            if row and row[0].strip().lstrip("+-").isdigit():
                years.add(int(row[0].strip()))
    return tuple(sorted(years))


def _epw_record_years(weather_file: str | Path) -> tuple[int, ...]:
    path = Path(weather_file).resolve()
    stat = path.stat()
    return _read_epw_record_years_cached(
        str(path),
        stat.st_size,
        stat.st_mtime_ns,
    )


def run_period_start_weekday(
    weather_file: str | Path,
    window: CalendarWindowSpec,
    *,
    calendar_year: int | None = None,
) -> str:
    """Return EnergyPlus' weekday enum for ``window`` from the pinned EPW.

    Typical-year files use the EPW header as the calendar anchor.  For an
    actual-weather-year task, ``calendar_year`` supplies leap-year semantics
    and the EPW anchor is also checked against that physical calendar.
    """

    period = read_epw_data_period(weather_file)
    if calendar_year is not None:
        if isinstance(calendar_year, bool) or not isinstance(calendar_year, int) or calendar_year <= 0:
            raise ValueError("calendar_year must be a positive integer")
        if (period.start_year, period.end_year) != (calendar_year, calendar_year):
            raise WeatherCalendarError(
                f"calendar_year {calendar_year} does not match year-qualified EPW DATA PERIODS "
                f"{(period.start_year, period.end_year)!r}"
            )
        record_years = _epw_record_years(weather_file)
        if record_years != (calendar_year,):
            raise WeatherCalendarError(
                f"calendar_year {calendar_year} does not match EPW record years " f"{record_years!r}"
            )
        reference_year = calendar_year
    else:
        if period.start_year is not None or period.end_year is not None:
            raise WeatherCalendarError("representative-weather tasks require yearless EPW DATA PERIODS dates")
        # 2001 is a non-leap year. Only date differences are used for TMY
        # files; the weekday anchor still comes from the EPW header.
        reference_year = 2001

    try:
        period_start = date(reference_year, period.start_month, period.start_day)
        period_end = date(reference_year, period.end_month, period.end_day)
        window_start = date(reference_year, window.start_month, window.start_day)
        window_end = date(reference_year, window.end_month, window.end_day)
    except ValueError as error:
        raise WeatherCalendarError(
            f"weather calendar cannot represent window {window.name!r} in {reference_year}"
        ) from error
    if window_start < period_start or window_end > period_end:
        raise WeatherCalendarError(
            f"window {window.name!r} lies outside EPW data period "
            f"{period.start_month}/{period.start_day}-{period.end_month}/{period.end_day}"
        )

    if calendar_year is not None:
        actual_anchor = period_start.strftime("%A")
        if actual_anchor != period.start_weekday:
            raise WeatherCalendarError(
                f"EPW start weekday {period.start_weekday} disagrees with "
                f"{calendar_year}-{period.start_month:02d}-{period.start_day:02d} "
                f"({actual_anchor})"
            )

    anchor_index = _WEEKDAYS.index(period.start_weekday)
    day_offset = (window_start - period_start).days
    return _WEEKDAYS[(anchor_index + day_offset) % len(_WEEKDAYS)]


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA256 digest of a file without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_asset(asset: AssetSpec, data_root: str | Path) -> Path:
    """Resolve and verify a data-root-relative benchmark asset.

    Checksums are mandatory here even though :class:`AssetSpec` permits an
    unpinned asset while a suite is being drafted.  A runnable benchmark must
    never silently consume different bytes.
    """

    if asset.sha256 is None:
        raise AssetIntegrityError(f"asset {asset.asset_id!r} has no SHA256 checksum and cannot be executed")
    root = Path(data_root).expanduser().resolve()
    path = root.joinpath(*Path(asset.relative_path).parts)
    if not path.is_file():
        raise AssetIntegrityError(f"asset {asset.asset_id!r} does not exist at {path}")
    actual = sha256_file(path)
    if actual != asset.sha256:
        raise AssetIntegrityError(
            f"SHA256 mismatch for asset {asset.asset_id!r} at {path}: " f"expected {asset.sha256}, found {actual}"
        )
    return path.resolve()


def resolve_task_assets(task: TaskSpec, data_root: str | Path) -> ResolvedTaskAssets:
    """Resolve and checksum both immutable inputs of ``task``."""

    return ResolvedTaskAssets(
        building=resolve_asset(task.building, data_root),
        weather=resolve_asset(task.weather, data_root),
    )


def _run_period(
    window: CalendarWindowSpec,
    template: Mapping[str, Any] | None,
    calendar_year: int | None,
    start_weekday: str,
) -> dict[str, Any]:
    period = dict(template or {})
    # Do not inherit a multi-year range from the source: the benchmark horizon
    # is defined solely by this inclusive window.  An actual weather year can
    # be reintroduced explicitly from pinned task metadata.
    period.pop("begin_year", None)
    period.pop("end_year", None)
    period.update(
        {
            "begin_month": window.start_month,
            "begin_day_of_month": window.start_day,
            "end_month": window.end_month,
            "end_day_of_month": window.end_day,
            # EnergyPlus 24.1 requires a literal weekday here. It is derived
            # from the EPW calendar for this exact month/day window.
            "day_of_week_for_start_day": start_weekday,
            "use_weather_file_holidays_and_special_days": "Yes",
            "use_weather_file_daylight_saving_period": "Yes",
            "use_weather_file_rain_indicators": "Yes",
            "use_weather_file_snow_indicators": "Yes",
        }
    )
    if calendar_year is not None:
        if isinstance(calendar_year, bool) or not isinstance(calendar_year, int) or calendar_year <= 0:
            raise ValueError("calendar_year must be a positive integer")
        period["begin_year"] = calendar_year
        period["end_year"] = calendar_year
        period["treat_weather_as_actual"] = "Yes"
    else:
        period["treat_weather_as_actual"] = "No"
    return period


def materialize_run_period(
    source_building: str | Path,
    window: CalendarWindowSpec,
    work_dir: str | Path,
    *,
    weather_file: str | Path,
    output_name: str | None = None,
    calendar_year: int | None = None,
    warmup_preconditioning: WarmupPreconditioningSpec | None = None,
    expected_source_building_sha256: str | None = None,
    expected_weather_sha256: str | None = None,
) -> Path:
    """Copy an epJSON model and replace its RunPeriod with ``window``.

    The source model is read-only.  Exactly one RunPeriod is written so an
    unnoticed second period cannot extend the benchmark episode.
    """

    source = Path(source_building)
    try:
        source_bytes = source.read_bytes()
        source_text = source_bytes.decode("utf-8")
        model = json.loads(source_text)
    except (OSError, UnicodeError) as error:
        raise ValueError(f"cannot read epJSON building asset: {source}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"building asset is not valid epJSON: {source}") from error
    if (
        expected_source_building_sha256 is not None
        and hashlib.sha256(source_bytes).hexdigest() != expected_source_building_sha256
    ):
        raise AssetIntegrityError(
            "exact building bytes parsed for RunPeriod materialization do not match TaskSpec SHA256"
        )
    weather_path = Path(weather_file)
    if expected_weather_sha256 is not None and sha256_file(weather_path) != expected_weather_sha256:
        raise AssetIntegrityError(
            "exact weather bytes used for RunPeriod calendar materialization do not match TaskSpec SHA256"
        )
    if not isinstance(model, dict):
        raise ValueError(f"epJSON root must be an object: {source}")
    source_model = deepcopy(model)

    if warmup_preconditioning is None:
        warmup_preconditioning = WarmupPreconditioningSpec.disabled()
    if not isinstance(warmup_preconditioning, WarmupPreconditioningSpec):
        raise TypeError("warmup_preconditioning must be a WarmupPreconditioningSpec")
    if warmup_preconditioning.enabled:
        if warmup_preconditioning.requires_actual_weather_year:
            if calendar_year is None:
                raise WeatherCalendarError("warmup preconditioning v2 requires an actual-weather calendar_year")
        elif calendar_year is not None:
            raise WeatherCalendarError(
                "the frozen v1 warmup preconditioning contract does not support actual-weather calendar_year"
            )

    template = None
    existing = model.get("RunPeriod")
    if existing is not None:
        if not isinstance(existing, dict):
            raise ValueError("epJSON RunPeriod must be an object")
        if existing:
            first_period = next(iter(existing.values()))
            if not isinstance(first_period, dict):
                raise ValueError("each epJSON RunPeriod entry must be an object")
            template = first_period

    data_period = read_epw_data_period(weather_file)
    if warmup_preconditioning.enabled:
        expected_anchor = (
            warmup_preconditioning.epw_data_period_start_weekday,
            warmup_preconditioning.epw_data_period_start_month,
            warmup_preconditioning.epw_data_period_start_day,
        )
        actual_anchor = (
            data_period.start_weekday,
            data_period.start_month,
            data_period.start_day,
        )
        if actual_anchor != expected_anchor:
            raise WeatherCalendarError(
                "pinned EPW DATA PERIODS calendar anchor does not match the "
                f"warmup contract: expected {expected_anchor!r}, found {actual_anchor!r}"
            )

    start_weekday = run_period_start_weekday(
        weather_file,
        window,
        calendar_year=calendar_year,
    )
    if warmup_preconditioning.enabled:
        model["RunPeriod"] = expected_warmup_run_period(
            contract=warmup_preconditioning,
            window=window,
            calendar_year=calendar_year,
        )
        materialized_weekday = next(iter(model["RunPeriod"].values()))["day_of_week_for_start_day"]
        if materialized_weekday != start_weekday:
            raise WeatherCalendarError("task-derived warmup RunPeriod weekday disagrees with the pinned EPW")
    else:
        model["RunPeriod"] = {
            f"HOT_Benchmark_{window.name}": _run_period(
                window,
                template,
                calendar_year,
                start_weekday,
            )
        }
    _materialize_warmup_preconditioning(model, warmup_preconditioning)
    _assert_materialized_model_delta(
        source_model,
        model,
        warmup_preconditioning,
    )

    destination_dir = Path(work_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / (output_name or f"{source.stem}__{window.name}.epJSON")
    if destination.resolve() == source.resolve():
        raise ValueError("materialized RunPeriod must not overwrite the source model")

    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            json.dump(model, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination.resolve()


def _snapshot_weather_asset(
    source: Path,
    destination: Path,
    *,
    expected_sha256: str,
) -> Path:
    """Copy exactly verified weather bytes into one episode work directory."""

    source_bytes = source.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != expected_sha256:
        raise AssetIntegrityError("weather asset changed after checksum resolution and before episode snapshot")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.resolve() == source.resolve():
        raise ValueError("episode weather snapshot must not overwrite the source asset")
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(source_bytes)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    if sha256_file(destination) != expected_sha256:
        raise AssetIntegrityError("episode weather snapshot does not match TaskSpec SHA256")
    return destination.resolve()


def _casefold_member(
    values: Mapping[str, Any],
    name: str,
    *,
    collection_name: str,
) -> tuple[str, Any]:
    matches = [
        (key, value) for key, value in values.items() if isinstance(key, str) and key.casefold() == name.casefold()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"epJSON {collection_name} must contain exactly one case-insensitive "
            f"match for {name!r}; found {len(matches)}"
        )
    return matches[0]


def _materialize_warmup_preconditioning(
    model: dict[str, Any],
    contract: WarmupPreconditioningSpec,
) -> None:
    """Apply only the separately declared global warmup cap.

    The RunPeriod-only actuator override is implemented by the runtime API.
    This function changes no thermostat schedule, sizing input, convergence
    tolerance, or minimum-warmup setting.
    """

    if not contract.enabled:
        return
    buildings = model.get("Building")
    if not isinstance(buildings, Mapping) or len(buildings) != 1:
        raise ValueError("enabled warmup preconditioning requires exactly one epJSON Building object")
    building_name, raw_building = next(iter(buildings.items()))
    if not isinstance(raw_building, Mapping):
        raise ValueError(f"epJSON Building {building_name!r} must be an object")
    building = dict(raw_building)
    source_cap = building.get("maximum_number_of_warmup_days")
    if (
        isinstance(source_cap, bool)
        or not isinstance(source_cap, int)
        or source_cap != contract.expected_source_maximum_warmup_days
    ):
        raise ValueError(
            "source maximum_number_of_warmup_days drifted: "
            f"expected {contract.expected_source_maximum_warmup_days}, found {source_cap!r}"
        )

    schedules = model.get("Schedule:Constant")
    if not isinstance(schedules, Mapping):
        raise ValueError("enabled warmup preconditioning requires epJSON Schedule:Constant objects")
    expected_schedule_value = contract.underlying_schedule_constant_value_c
    for schedule_name in contract.underlying_schedule_names:
        actual_name, raw_schedule = _casefold_member(
            schedules,
            schedule_name,
            collection_name="Schedule:Constant",
        )
        if not isinstance(raw_schedule, Mapping):
            raise ValueError(f"epJSON Schedule:Constant {actual_name!r} must be an object")
        actual_value = raw_schedule.get("hourly_value")
        if (
            isinstance(actual_value, bool)
            or not isinstance(actual_value, (int, float))
            or float(actual_value) != expected_schedule_value
        ):
            raise ValueError(
                f"source cooling schedule {actual_name!r} must remain exactly "
                f"{expected_schedule_value} C; found {actual_value!r}"
            )

    building["maximum_number_of_warmup_days"] = contract.maximum_warmup_days
    model["Building"] = {building_name: building}


def _assert_materialized_model_delta(
    source_model: Mapping[str, Any],
    materialized_model: Mapping[str, Any],
    contract: WarmupPreconditioningSpec,
) -> None:
    """Reject any materialization change beyond RunPeriod and the exact cap."""

    expected = deepcopy(dict(source_model))
    expected["RunPeriod"] = deepcopy(materialized_model["RunPeriod"])
    if contract.enabled:
        buildings = expected.get("Building")
        if not isinstance(buildings, dict) or len(buildings) != 1:
            raise ValueError("cannot assert the warmup materialization delta without one Building object")
        building_name, building = next(iter(buildings.items()))
        if not isinstance(building, dict):
            raise ValueError("cannot assert a malformed Building object")
        building["maximum_number_of_warmup_days"] = contract.maximum_warmup_days
        expected["Building"] = {building_name: building}
    if materialized_model != expected:
        raise RuntimeError(
            "RunPeriod materialization changed fields outside the declared RunPeriod "
            "and maximum-warmup-days contract"
        )


def _finite_number(parameters: Mapping[str, Any], name: str) -> float:
    if name not in parameters:
        raise ValueError(f"reward parameter {name!r} is required")
    value = parameters[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"reward parameter {name!r} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"reward parameter {name!r} must be finite")
    return value


def simulator_reward_arguments(reward: RewardSpec, *, controlled_zones=None) -> dict[str, Any]:
    """Translate a pinned RewardSpec to explicit ``create_simulator`` kwargs."""

    parameters = reward.parameter_mapping
    from building_gen.scheduled_reward import (
        HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3, ScheduledOccupiedRewardConfig)
    if reward.reward_id == HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3:
        config = ScheduledOccupiedRewardConfig.from_mapping(parameters)
        if controlled_zones is not None:
            config.comfort_schedule.validate_zones(tuple(controlled_zones))
        return {"reward_type": reward.reward_id, **config.parameter_mapping,
                "stability_weight": 0.0,
                "comfort_temp_range": (config.comfort_schedule.default.minimum_c,
                                       config.comfort_schedule.default.maximum_c)}
    if reward.reward_id == HOT_OCCUPIED_ENERGY_COMFORT_V2:
        config = OccupiedEnergyComfortV2Config.from_mapping(parameters)
        return {
            "reward_type": reward.reward_id,
            "energy_weight": config.energy_weight,
            "comfort_weight": config.comfort_weight,
            "stability_weight": 0.0,
            "comfort_temp_range": (
                config.comfort_min_c,
                config.comfort_max_c,
            ),
            "energy_scale_kwh_per_m2": config.energy_scale_kwh_per_m2,
            "comfort_scale_person_celsius_squared_hours_per_m2": (
                config.comfort_scale_person_celsius_squared_hours_per_m2
            ),
        }
    supported = {
        "energy_weight",
        "comfort_weight",
        "stability_weight",
        "comfort_min_c",
        "comfort_max_c",
    }
    unknown = sorted(set(parameters) - supported)
    if unknown:
        raise ValueError(f"reward {reward.reward_id!r} has unsupported parameter(s): {', '.join(unknown)}")

    energy_weight = _finite_number(parameters, "energy_weight")
    comfort_weight = _finite_number(parameters, "comfort_weight")
    stability_weight = _finite_number(parameters, "stability_weight")
    comfort_min = _finite_number(parameters, "comfort_min_c")
    comfort_max = _finite_number(parameters, "comfort_max_c")
    if min(energy_weight, comfort_weight, stability_weight) < 0:
        raise ValueError("reward weights must be non-negative")
    if comfort_min >= comfort_max:
        raise ValueError("comfort_min_c must be smaller than comfort_max_c")

    return {
        "reward_type": reward.reward_id,
        "energy_weight": energy_weight,
        "comfort_weight": comfort_weight,
        "stability_weight": stability_weight,
        "comfort_temp_range": (comfort_min, comfort_max),
    }


def _expected_observation_names(task: TaskSpec) -> tuple[str, ...]:
    """Return the feature order defined by HOT observation encodings."""

    from building_gen.wrapper.observation_variants import VARIANTS, extra_names
    encoding = task.observation_space.encoding
    if encoding != "hot_observation_v2" and encoding not in VARIANTS:
        raise EnvironmentCompatibilityError(
            "cannot validate observation order for unsupported encoding " f"{task.observation_space.encoding!r}"
        )
    return tuple(
        [f"{zone_name}_temperature" for zone_name in task.action_profile.controlled_entities]
        + [
            "outdoor_temperature",
            "outdoor_humidity",
            "hvac_electricity",
            "hvac_gas",
            "hour_of_day",
            "day_of_week",
            "day_of_year",
        ]
        + (extra_names(encoding, task.action_profile.controlled_entities) if encoding in VARIANTS else [])
    )


def _expected_action_names(task: TaskSpec) -> tuple[str, ...]:
    return tuple(
        f"{entity_name}:{channel.name}"
        for entity_name in task.action_profile.controlled_entities
        for channel in task.action_profile.channels
    )


def _expected_action_endpoint_mapping(task: TaskSpec) -> tuple[Any, ...]:
    if task.action_profile.transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1"):
        return ()  # Cooling depends on the simultaneously chosen heating target.
    return tuple(
        (
            f"{entity_name}:{channel.name}",
            (
                (0.0, float(channel.physical_min)),
                (1.0, float(channel.physical_max)),
            ),
        )
        for entity_name in task.action_profile.controlled_entities
        for channel in task.action_profile.channels
    )


def _box_space_problems(
    environment: Any,
    *,
    attribute: str,
    expected_shape: tuple[int, ...],
    expected_dtype: str,
    expected_lower: float,
    expected_upper: float,
) -> list[str]:
    """Validate the exact normalized Gymnasium Box contract lazily."""

    import gymnasium as gym
    import numpy as np

    space = getattr(environment, attribute, None)
    if not isinstance(space, gym.spaces.Box):
        return [f"{attribute} type {type(space).__name__} != gymnasium.spaces.Box"]
    problems: list[str] = []
    actual_shape = tuple(space.shape)
    if actual_shape != expected_shape:
        label = attribute.removesuffix("_space")
        problems.append(f"{label} shape {actual_shape} != task shape {expected_shape}")
    try:
        declared_dtype = np.dtype(expected_dtype)
    except TypeError:
        problems.append(f"task declares invalid {attribute} dtype {expected_dtype!r}")
    else:
        if np.dtype(space.dtype) != declared_dtype:
            problems.append(f"{attribute} dtype {np.dtype(space.dtype)} != task dtype {declared_dtype}")
    if not np.all(np.isfinite(space.low)) or not np.all(np.isfinite(space.high)):
        problems.append(f"{attribute} bounds must be finite")
    if not np.allclose(space.low, expected_lower, rtol=0.0, atol=0.0):
        problems.append(f"{attribute} lower bounds are not all {expected_lower}")
    if not np.allclose(space.high, expected_upper, rtol=0.0, atol=0.0):
        problems.append(f"{attribute} upper bounds are not all {expected_upper}")
    return problems


def validate_environment(environment: Any, task: TaskSpec) -> None:
    """Fail if runtime spaces, feature order, or action scaling drift."""

    actual_zones = tuple(getattr(environment, "controlled_zones", ()))
    expected_zones = task.action_profile.controlled_entities
    problems: list[str] = []
    if task.observation_space.kind != "box":
        problems.append("HOT benchmark observation_space.kind must be 'box'")
    if task.action_space.kind != "box":
        problems.append("HOT benchmark action_space.kind must be 'box'")
    if task.observation_space.lower_bound != 0.0 or task.observation_space.upper_bound != 1.0:
        problems.append("task observation space must declare bounds [0, 1]")
    if (
        task.action_space.encoding != "normalized_0_1"
        or task.action_space.lower_bound != 0.0
        or task.action_space.upper_bound != 1.0
    ):
        problems.append("task action space must declare encoding='normalized_0_1' and bounds [0, 1]")
    problems.extend(
        _box_space_problems(
            environment,
            attribute="observation_space",
            expected_shape=task.observation_space.shape,
            expected_dtype=task.observation_space.dtype,
            expected_lower=0.0,
            expected_upper=1.0,
        )
    )
    problems.extend(
        _box_space_problems(
            environment,
            attribute="action_space",
            expected_shape=task.action_space.shape,
            expected_dtype=task.action_space.dtype,
            expected_lower=0.0,
            expected_upper=1.0,
        )
    )
    if actual_zones != expected_zones:
        problems.append(f"controlled-zone order {actual_zones!r} != task order {expected_zones!r}")
    actual_observation_encoding = getattr(
        environment,
        "observation_encoding_id",
        None,
    )
    if actual_observation_encoding != task.observation_space.encoding:
        problems.append(
            f"observation encoding {actual_observation_encoding!r} != task encoding "
            f"{task.observation_space.encoding!r}"
        )
    try:
        expected_observation_names = _expected_observation_names(task)
    except EnvironmentCompatibilityError as error:
        problems.append(str(error))
    else:
        actual_observation_names = tuple(getattr(environment, "observation_names", ()))
        if actual_observation_names != expected_observation_names:
            problems.append(
                "observation order "
                f"{actual_observation_names!r} != task encoding order "
                f"{expected_observation_names!r}"
            )
    expected_action_names = _expected_action_names(task)
    actual_action_names = tuple(getattr(environment, "action_names", ()))
    if actual_action_names != expected_action_names:
        problems.append(f"action order {actual_action_names!r} != task order {expected_action_names!r}")
    expected_mapping = _expected_action_endpoint_mapping(task)
    actual_mapping = tuple(getattr(environment, "action_endpoint_mapping", ()))
    if actual_mapping != expected_mapping:
        problems.append("normalized-to-physical action endpoint mapping does not match TaskSpec")
    actual_transform_id = getattr(environment, "action_transform_id", None)
    if actual_transform_id != task.action_profile.transform_id:
        problems.append(
            f"action transform {actual_transform_id!r} != task transform " f"{task.action_profile.transform_id!r}"
        )
    expected_contract_payload = task.warmup_preconditioning.identity_payload()
    missing = object()
    actual_contract_payload = getattr(
        environment,
        "warmup_preconditioning_contract_payload",
        missing,
    )
    actual_contract_sha256 = getattr(
        environment,
        "warmup_preconditioning_contract_sha256",
        missing,
    )
    runtime_contract_required = task.schema_version == TASK_SCHEMA_VERSION or task.warmup_preconditioning.enabled
    if runtime_contract_required:
        if actual_contract_payload is missing or actual_contract_sha256 is missing:
            problems.append("current runtime must advertise a complete warmup preconditioning contract")
        else:
            if actual_contract_payload != expected_contract_payload:
                problems.append("runtime warmup preconditioning contract payload does not match TaskSpec")
            if actual_contract_sha256 != task.warmup_preconditioning.sha256:
                problems.append("runtime warmup preconditioning contract SHA256 does not match TaskSpec")
    elif task.schema_version == LEGACY_TASK_SCHEMA_VERSION:
        if (actual_contract_payload is missing) != (actual_contract_sha256 is missing):
            problems.append("legacy runtime advertises an incomplete warmup preconditioning contract")
        elif actual_contract_payload is not missing:
            if actual_contract_payload != expected_contract_payload:
                problems.append("runtime warmup preconditioning contract payload does not match TaskSpec")
            if actual_contract_sha256 != task.warmup_preconditioning.sha256:
                problems.append("runtime warmup preconditioning contract SHA256 does not match TaskSpec")
    if problems:
        close = getattr(environment, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass
        raise EnvironmentCompatibilityError("; ".join(problems))


def _prepare_energyplus_output_directory(run_dir: Path) -> Path:
    """Create the run output directory and remove only stale completion evidence."""

    output_dir = run_dir / "energyplus"
    output_dir.mkdir(parents=True, exist_ok=True)
    stale_paths = tuple(
        output_dir / name
        for name in (
            "eplusout.err",
            "eplusout.end",
            DIAGNOSTICS_FILENAME,
            WARMUP_PRECONDITIONING_FILENAME,
        )
    )
    directories = [path for path in stale_paths if path.is_dir() and not path.is_symlink()]
    if directories:
        raise IsADirectoryError(
            "refusing to replace EnergyPlus output path(s) that are directories: " + ", ".join(map(str, directories))
        )
    for path in stale_paths:
        # Unlinking an exact symlink removes the link, not its target.  Other
        # files in a caller-provided work directory are intentionally retained.
        path.unlink(missing_ok=True)
    return output_dir


def create_benchmark_environment(
    task: TaskSpec,
    window: CalendarWindowSpec,
    *,
    data_root: str | Path,
    work_dir: str | Path,
    simulator_factory: Callable[..., Any] | None = None,
) -> Any:
    """Create one checksum-pinned, month-specific EnergyPlus environment."""

    try:
        window.date_bounds_for_calendar_year(task.calendar_year)
    except ValueError as error:
        resolved_year = task.calendar_year if task.calendar_year is not None else 2001
        raise WeatherCalendarError(
            f"calendar window {window.name!r} is invalid for task calendar year {resolved_year}"
        ) from error
    assets = resolve_task_assets(task, data_root)
    reward_arguments = simulator_reward_arguments(task.reward, controlled_zones=task.action_profile.controlled_entities)
    run_dir = Path(work_dir).resolve()
    if task.building.sha256 is None or task.weather.sha256 is None:
        raise AssetIntegrityError("runnable benchmark environments require checksum-pinned assets")
    weather_snapshot = _snapshot_weather_asset(
        assets.weather,
        run_dir / assets.weather.name,
        expected_sha256=task.weather.sha256,
    )
    model_path = materialize_run_period(
        assets.building,
        window,
        run_dir,
        weather_file=weather_snapshot,
        output_name=f"{task.name}__{window.name}.epJSON",
        calendar_year=task.calendar_year,
        warmup_preconditioning=task.warmup_preconditioning,
        expected_source_building_sha256=task.building.sha256,
        expected_weather_sha256=task.weather.sha256,
    )
    executed_model_sha256 = sha256_file(model_path)
    runtime_context = None
    if task.warmup_preconditioning.enabled:
        if task.building.sha256 is None or task.weather.sha256 is None:
            raise AssetIntegrityError(
                "enabled warmup preconditioning requires checksum-pinned building and weather assets"
            )
        try:
            materialized_model = json.loads(model_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read materialized warmup-preconditioned model {model_path}") from error
        run_period = materialized_model.get("RunPeriod")
        if not isinstance(run_period, Mapping):
            raise ValueError("materialized model has no valid RunPeriod object")
        runtime_context = WarmupPreconditioningRuntimeContext(
            task_id=task.task_id,
            source_building_sha256=task.building.sha256,
            weather_sha256=task.weather.sha256,
            materialized_model_sha256=executed_model_sha256,
            run_period=run_period,
            contract=task.warmup_preconditioning,
        )
    from building_gen.preprocessing.component_meters import component_meter_sources
    model_meters = component_meter_sources(json.loads(model_path.read_text(encoding='utf-8')))
    declared_meters = dict(task.component_energy_meters) if task.component_energy_meters is not None else None
    if model_meters != declared_meters:
        raise ValueError('prepared model component meters do not match task-declared energy boundary')
    if simulator_factory is None:
        from building_gen.wrapper.create_simulator import create_simulator

        simulator_factory = create_simulator

    energyplus_output_dir = _prepare_energyplus_output_directory(run_dir)
    diagnostics_path = energyplus_output_dir / DIAGNOSTICS_FILENAME
    environment = simulator_factory(
        path_to_building=str(model_path),
        path_to_weather=str(weather_snapshot),
        building_characteristics={"controlled_zones": list(task.action_profile.controlled_entities)},
        eplus_output_dir=str(energyplus_output_dir),
        episode_length_hours=window.duration_hours_for_calendar_year(task.calendar_year),
        timesteps_per_hour=task.episode.timesteps_per_hour,
        calendar_year_semantics=task.calendar_year_semantics,
        action_channels=tuple(
            {
                "name": channel.name,
                "unit": channel.unit,
                "physical_min": channel.physical_min,
                "physical_max": channel.physical_max,
            }
            for channel in task.action_profile.channels
        ),
        action_transform_id=task.action_profile.transform_id,
        warmup_preconditioning=task.warmup_preconditioning,
        warmup_preconditioning_context=runtime_context,
        **reward_arguments,
    )
    from building_gen.wrapper.observation_variants import VARIANTS, ObservationVariant
    if task.observation_space.encoding in VARIANTS:
        environment = ObservationVariant(environment, task)
    validate_environment(environment, task)
    # The evaluator reads this only after the episode has terminated and the
    # environment has closed. The EnergyPlus worker writes the referenced file
    # before it emits its final completion result.
    environment.energyplus_diagnostics_path = diagnostics_path
    environment.energyplus_executed_model_sha256 = executed_model_sha256
    environment.benchmark_episode_work_dir = run_dir
    return environment


__all__ = [
    "AssetIntegrityError",
    "EpwDataPeriod",
    "EnvironmentCompatibilityError",
    "ResolvedTaskAssets",
    "WeatherCalendarError",
    "create_benchmark_environment",
    "materialize_run_period",
    "read_epw_data_period",
    "resolve_asset",
    "resolve_task_assets",
    "run_period_start_weekday",
    "sha256_file",
    "simulator_reward_arguments",
    "validate_environment",
]
