"""Unit-safe energy-carrier and operational-carbon accounting.

EnergyPlus meter callbacks report energy over the current reporting interval in
joules. This module converts those values exactly once and deliberately keeps a
missing meter (``None``) distinct from a meter that reported zero consumption.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

JOULES_PER_KWH = 3_600_000.0


class EnergyCarrier(str, Enum):
    """Energy carriers represented by the benchmark."""

    ELECTRICITY = "electricity"
    NATURAL_GAS = "natural_gas"
    DISTRICT_HEATING_WATER = "district_heating_water"
    DISTRICT_HEATING_STEAM = "district_heating_steam"
    DISTRICT_COOLING = "district_cooling"


CONTROLLED_HVAC_METER_BY_CARRIER: Mapping[EnergyCarrier, str] = MappingProxyType(
    {
        EnergyCarrier.ELECTRICITY: "Electricity:HVAC",
        EnergyCarrier.NATURAL_GAS: "NaturalGas:HVAC",
    }
)


def _validate_nonnegative_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative, got {value!r}")
    return result


def joules_to_kwh(energy_joules: float) -> float:
    """Convert interval energy from joules to kilowatt-hours."""

    return _validate_nonnegative_finite(energy_joules, name="energy_joules") / JOULES_PER_KWH


@dataclass(frozen=True, slots=True)
class EnergyPlusMtrEnvironment:
    """One exact code-1 environment header from an EnergyPlus MTR file.

    ``sequence_index`` is one-based and intentionally remains part of the
    identity.  EnergyPlus may emit multiple environments with the same title
    and geographic fields (for example, repeated sizing environments), so the
    header fields alone are not a unique occurrence identifier.
    """

    sequence_index: int
    title: str
    latitude: Decimal
    longitude: Decimal
    time_zone: Decimal
    elevation_m: Decimal

    def __post_init__(self) -> None:
        if type(self.sequence_index) is not int or self.sequence_index <= 0:
            raise ValueError("MTR environment sequence_index must be a positive native int")
        if type(self.title) is not str or not self.title or self.title != self.title.strip():
            raise ValueError("MTR environment title must be a non-empty stripped string")
        if any(character in self.title for character in "\r\n"):
            raise ValueError("MTR environment title must occupy one line")
        for field_name in ("latitude", "longitude", "time_zone", "elevation_m"):
            value = getattr(self, field_name)
            if type(value) is not Decimal or not value.is_finite():
                raise ValueError(f"MTR environment {field_name} must be a finite Decimal")
        if not Decimal("-90") <= self.latitude <= Decimal("90"):
            raise ValueError("MTR environment latitude must lie in [-90, 90]")
        if not Decimal("-180") <= self.longitude <= Decimal("180"):
            raise ValueError("MTR environment longitude must lie in [-180, 180]")
        if not Decimal("-24") <= self.time_zone <= Decimal("24"):
            raise ValueError("MTR environment time_zone must lie in [-24, 24]")


_ENERGYPLUS_WEEKDAY_NUMBER = MappingProxyType(
    {
        "sunday": 1,
        "monday": 2,
        "tuesday": 3,
        "wednesday": 4,
        "thursday": 5,
        "friday": 6,
        "saturday": 7,
    }
)
_ENERGYPLUS_DESIGN_DAY_TYPES = frozenset({"summerdesignday", "winterdesignday"})
_ENERGYPLUS_DAY_TYPES = frozenset(
    {
        "Sunday",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Holiday",
        "SummerDesignDay",
        "WinterDesignDay",
        "CustomDay1",
        "CustomDay2",
    }
)


@dataclass(frozen=True, slots=True)
class EnergyPlusMtrIntervalLabel:
    """The exact interval label carried by one official code-2 record.

    EnergyPlus' code-2 hour is one-based: hour 1 labels intervals in API hour
    0, and hour 24 labels intervals in API hour 23.  Start and end minutes are
    retained as :class:`~decimal.Decimal` values so parsing never rounds the
    simulator's labels; in particular, ``EndMinute=60`` is not rolled into the
    next hour or calendar day.
    """

    day_of_simulation: int
    month: int
    day_of_month: int
    dst_indicator: int
    hour_1_based: int
    start_minute: Decimal
    end_minute: Decimal
    day_type: str

    def __post_init__(self) -> None:
        for field_name in (
            "day_of_simulation",
            "month",
            "day_of_month",
            "dst_indicator",
            "hour_1_based",
        ):
            if type(getattr(self, field_name)) is not int:
                raise ValueError(f"MTR interval {field_name} must be a native int")
        if self.day_of_simulation <= 0:
            raise ValueError("MTR interval day_of_simulation must be positive")
        if not 1 <= self.month <= 12:
            raise ValueError("MTR interval month must lie in [1, 12]")
        try:
            # Leap year 2000 permits every month/day pair representable by an
            # EnergyPlus simulation, including February 29.
            date(2000, self.month, self.day_of_month)
        except ValueError as error:
            raise ValueError("MTR interval month/day is invalid") from error
        if self.dst_indicator not in (0, 1):
            raise ValueError("MTR interval DST indicator must be 0 or 1")
        if not 1 <= self.hour_1_based <= 24:
            raise ValueError("MTR interval one-based hour must lie in [1, 24]")
        for field_name in ("start_minute", "end_minute"):
            value = getattr(self, field_name)
            if type(value) is not Decimal or not value.is_finite():
                raise ValueError(f"MTR interval {field_name} must be a finite Decimal")
        if not Decimal(0) <= self.start_minute < self.end_minute <= Decimal(60):
            raise ValueError("MTR interval minutes must satisfy 0 <= start < end <= 60")
        if type(self.day_type) is not str or not self.day_type or self.day_type != self.day_type.strip():
            raise ValueError("MTR interval day_type must be a non-empty stripped string")
        if any(character in self.day_type for character in "\r\n"):
            raise ValueError("MTR interval day_type must occupy one line")
        if self.day_type not in _ENERGYPLUS_DAY_TYPES:
            raise ValueError(f"MTR interval day_type is not an official EnergyPlus value: {self.day_type!r}")

    @property
    def duration_minutes(self) -> Decimal:
        return self.end_minute - self.start_minute

    @property
    def api_hour(self) -> int:
        """Return the corresponding zero-based EnergyPlus API hour."""

        return self.hour_1_based - 1

    @property
    def weekday_number(self) -> int | None:
        """Return EnergyPlus' Sunday=1 weekday number, if this is a weekday."""

        return _ENERGYPLUS_WEEKDAY_NUMBER.get(self.day_type.casefold())

    @property
    def is_design_day(self) -> bool:
        """Whether the code-2 day type explicitly identifies a design day."""

        return self.day_type.casefold() in _ENERGYPLUS_DESIGN_DAY_TYPES


@dataclass(frozen=True, slots=True)
class EnergyPlusTimestepMeterRecord:
    """One exact ``[J] !TimeStep`` meter value and its MTR provenance."""

    environment: EnergyPlusMtrEnvironment
    interval: EnergyPlusMtrIntervalLabel
    joules: float

    def __post_init__(self) -> None:
        if type(self.environment) is not EnergyPlusMtrEnvironment:
            raise TypeError("MTR meter record environment has the wrong type")
        if type(self.interval) is not EnergyPlusMtrIntervalLabel:
            raise TypeError("MTR meter record interval has the wrong type")
        if type(self.joules) not in (int, float):
            raise TypeError("MTR meter value must be a native int or float")
        object.__setattr__(
            self,
            "joules",
            _validate_nonnegative_finite(self.joules, name="MTR meter value"),
        )


def _canonical_mtr_environment(value: object) -> EnergyPlusMtrEnvironment:
    """Rebuild one exact base environment without consulting virtual equality."""

    if type(value) is not EnergyPlusMtrEnvironment:
        raise TypeError("MTR meter record environment must be an exact EnergyPlusMtrEnvironment")
    return EnergyPlusMtrEnvironment(
        sequence_index=value.sequence_index,
        title=value.title,
        latitude=value.latitude,
        longitude=value.longitude,
        time_zone=value.time_zone,
        elevation_m=value.elevation_m,
    )


def _canonical_mtr_interval(value: object) -> EnergyPlusMtrIntervalLabel:
    """Rebuild one exact base interval without consulting virtual equality."""

    if type(value) is not EnergyPlusMtrIntervalLabel:
        raise TypeError("MTR meter record interval must be an exact EnergyPlusMtrIntervalLabel")
    return EnergyPlusMtrIntervalLabel(
        day_of_simulation=value.day_of_simulation,
        month=value.month,
        day_of_month=value.day_of_month,
        dst_indicator=value.dst_indicator,
        hour_1_based=value.hour_1_based,
        start_minute=value.start_minute,
        end_minute=value.end_minute,
        day_type=value.day_type,
    )


def _canonical_mtr_record(value: object) -> EnergyPlusTimestepMeterRecord:
    """Rebuild one exact base record and reject mutated stored scalars."""

    if type(value) is not EnergyPlusTimestepMeterRecord:
        raise TypeError("MTR snapshot stream contains a non-base meter record")
    if type(value.joules) is not float:
        raise TypeError("MTR snapshot stored meter value must be a canonical native float")
    return EnergyPlusTimestepMeterRecord(
        environment=_canonical_mtr_environment(value.environment),
        interval=_canonical_mtr_interval(value.interval),
        joules=value.joules,
    )


@dataclass(frozen=True, slots=True)
class EnergyPlusTimestepMeterSnapshot:
    """Requested meter streams parsed from one exact immutable MTR payload."""

    source_sha256: str
    source_size_bytes: int
    streams: tuple[tuple[str, tuple[EnergyPlusTimestepMeterRecord, ...]], ...]

    def __post_init__(self) -> None:
        if type(self.source_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None:
            raise ValueError("MTR snapshot source_sha256 must be exactly 64 lowercase hexadecimal characters")
        if type(self.source_size_bytes) is not int or self.source_size_bytes <= 0:
            raise ValueError("MTR snapshot source_size_bytes must be a positive native int")
        if type(self.streams) is not tuple or not self.streams:
            raise ValueError("MTR snapshot streams must be a nonempty tuple")

        normalized: list[tuple[str, tuple[EnergyPlusTimestepMeterRecord, ...]]] = []
        seen: set[str] = set()
        for index, entry in enumerate(self.streams):
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError(f"MTR snapshot streams[{index}] must be a meter/records tuple")
            meter_name, records = entry
            if type(meter_name) is not str:
                raise TypeError("MTR snapshot meter names must be exact base strings")
            if not meter_name or meter_name != meter_name.strip():
                raise ValueError("MTR snapshot meter names must be canonical nonempty strings")
            if meter_name in seen:
                raise ValueError(f"MTR snapshot contains duplicate meter stream {meter_name!r}")
            seen.add(meter_name)
            if type(records) is not tuple or not records:
                raise ValueError(f"MTR snapshot meter stream {meter_name!r} must be a nonempty tuple")
            canonical_records = tuple(_canonical_mtr_record(record) for record in records)
            normalized.append((meter_name, canonical_records))

        normalized.sort()
        reference_labels: tuple[tuple[EnergyPlusMtrEnvironment, EnergyPlusMtrIntervalLabel], ...] | None = None
        for _, canonical_records in normalized:
            labels = tuple((record.environment, record.interval) for record in canonical_records)
            if reference_labels is None:
                reference_labels = labels
            elif labels != reference_labels:
                raise ValueError("MTR snapshot meter streams do not share identical environment/interval labels")
        for _, canonical_records in normalized:
            _validate_snapshot_stream_grid(canonical_records)
        object.__setattr__(self, "streams", tuple(normalized))

    @property
    def meter_names(self) -> tuple[str, ...]:
        return tuple(meter_name for meter_name, _ in self.streams)

    def records_for(self, meter_name: str) -> tuple[EnergyPlusTimestepMeterRecord, ...]:
        if type(meter_name) is not str:
            raise TypeError("meter_name must be an exact base string")
        if not meter_name or meter_name != meter_name.strip():
            raise ValueError("meter_name must be a canonical nonempty string")
        for candidate, records in self.streams:
            if candidate == meter_name:
                return records
        raise KeyError(meter_name)

    @property
    def environments(self) -> tuple[EnergyPlusMtrEnvironment, ...]:
        first_stream = self.streams[0][1]
        environments: list[EnergyPlusMtrEnvironment] = []
        for record in first_stream:
            if not environments or record.environment != environments[-1]:
                environments.append(record.environment)
        return tuple(environments)


def _mtr_decimal(token: str, *, field: str, line_number: int) -> Decimal:
    try:
        value = Decimal(token.strip())
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Malformed {field} on MTR line {line_number}") from error
    if not value.is_finite():
        raise ValueError(f"Non-finite {field} on MTR line {line_number}")
    return value


def _mtr_integer(token: str, *, field: str, line_number: int) -> int:
    value = _mtr_decimal(token, field=field, line_number=line_number)
    if value != value.to_integral_value():
        raise ValueError(f"Non-integral {field} on MTR line {line_number}")
    return int(value)


def _calendar_day_follows(previous: EnergyPlusMtrIntervalLabel, current: EnergyPlusMtrIntervalLabel) -> bool:
    for year in (2000, 2001):
        try:
            previous_date = date(year, previous.month, previous.day_of_month)
        except ValueError:
            continue
        following = previous_date + timedelta(days=1)
        if (following.month, following.day) == (current.month, current.day_of_month):
            return True
    return False


def _validate_mtr_interval_successor(
    previous: EnergyPlusMtrIntervalLabel,
    current: EnergyPlusMtrIntervalLabel,
    *,
    line_number: int,
) -> None:
    if current.duration_minutes != previous.duration_minutes:
        raise ValueError(f"MTR code-2 interval duration changed on line {line_number}")

    if current.day_of_simulation == previous.day_of_simulation:
        if (current.month, current.day_of_month, current.day_type) != (
            previous.month,
            previous.day_of_month,
            previous.day_type,
        ):
            raise ValueError(f"MTR code-2 calendar label changed within a simulation day on line {line_number}")
        if previous.end_minute < Decimal(60):
            expected_hour = previous.hour_1_based
            expected_start = previous.end_minute
        else:
            expected_hour = previous.hour_1_based + 1
            expected_start = Decimal(0)
        if expected_hour > 24 or (current.hour_1_based, current.start_minute) != (
            expected_hour,
            expected_start,
        ):
            raise ValueError(f"MTR code-2 intervals are missing, duplicated, or out of order on line {line_number}")
        return

    if current.day_of_simulation != previous.day_of_simulation + 1:
        raise ValueError(f"MTR code-2 day-of-simulation sequence is not contiguous on line {line_number}")
    if previous.hour_1_based != 24 or previous.end_minute != Decimal(60):
        raise ValueError(f"MTR code-2 simulation day advanced before minute 60 on line {line_number}")
    if current.hour_1_based != 1 or current.start_minute != Decimal(0):
        raise ValueError(f"MTR code-2 simulation day did not restart at hour 1, minute 0 on line {line_number}")
    if not _calendar_day_follows(previous, current):
        raise ValueError(f"MTR code-2 month/day did not advance canonically on line {line_number}")
    previous_weekday = previous.weekday_number
    current_weekday = current.weekday_number
    if previous_weekday is not None and current_weekday is not None:
        expected_weekday = previous_weekday % 7 + 1
        if current_weekday != expected_weekday:
            raise ValueError(f"MTR code-2 weekday did not advance canonically on line {line_number}")


def _parse_mtr_environment(
    parts: list[str],
    *,
    sequence_index: int,
    line_number: int,
) -> EnergyPlusMtrEnvironment:
    if len(parts) != 6:
        raise ValueError(f"Malformed MTR code-1 environment record on line {line_number}")
    title = parts[1].strip()
    return EnergyPlusMtrEnvironment(
        sequence_index=sequence_index,
        title=title,
        latitude=_mtr_decimal(parts[2], field="environment latitude", line_number=line_number),
        longitude=_mtr_decimal(parts[3], field="environment longitude", line_number=line_number),
        time_zone=_mtr_decimal(parts[4], field="environment time zone", line_number=line_number),
        elevation_m=_mtr_decimal(parts[5], field="environment elevation", line_number=line_number),
    )


def _parse_mtr_interval(parts: list[str], *, line_number: int) -> EnergyPlusMtrIntervalLabel:
    if len(parts) != 9:
        raise ValueError(f"Malformed MTR code-2 interval record on line {line_number}")
    return EnergyPlusMtrIntervalLabel(
        day_of_simulation=_mtr_integer(parts[1], field="day of simulation", line_number=line_number),
        month=_mtr_integer(parts[2], field="month", line_number=line_number),
        day_of_month=_mtr_integer(parts[3], field="day of month", line_number=line_number),
        dst_indicator=_mtr_integer(parts[4], field="DST indicator", line_number=line_number),
        hour_1_based=_mtr_integer(parts[5], field="one-based hour", line_number=line_number),
        start_minute=_mtr_decimal(parts[6], field="start minute", line_number=line_number),
        end_minute=_mtr_decimal(parts[7], field="end minute", line_number=line_number),
        day_type=parts[8].strip(),
    )


@dataclass(frozen=True, slots=True)
class _MtrDictionaryEntry:
    report_id: str
    item_count: int
    description: str
    frequency: str | None


_MTR_STANDARD_ITEM_COUNTS = MappingProxyType({"1": 5, "2": 8, "3": 5, "4": 2, "5": 1, "6": 1})
_MTR_STANDARD_DESCRIPTIONS = MappingProxyType(
    {
        "1": "Environment Title[],Latitude[deg],Longitude[deg],Time Zone[],Elevation[m]",
        "2": (
            "Day of Simulation[],Month[],Day of Month[],DST Indicator[1=yes 0=no],"
            "Hour[],StartMinute[],EndMinute[],DayType"
        ),
        "3": ("Cumulative Day of Simulation[],Month[],Day of Month[]," "DST Indicator[1=yes 0=no],DayType"),
        "4": "Cumulative Days of Simulation[],Month[]",
        "5": "Cumulative Days of Simulation[]",
        "6": "Calendar Year of Simulation[]",
    }
)
_MTR_STANDARD_DESCRIPTION_SUFFIXES = MappingProxyType(
    {
        "3": "! When Daily Meters Requested",
        "4": "! When Monthly Meters Requested",
        "5": "! When Run Period Meters Requested",
        "6": "! When Annual Meters Requested",
    }
)
_MTR_FREQUENCY_CONTEXT = MappingProxyType(
    {
        "Each Call": "2",
        "TimeStep": "2",
        "Hourly": "2",
        "Daily": "3",
        "Monthly": "4",
        "RunPeriod": "5",
        "Annual": "6",
    }
)
_MTR_FREQUENCY_PATTERN = re.compile(r"!\s*(Each Call|TimeStep|Hourly|Daily|Monthly|RunPeriod|Annual)(?:\s.*)?$")
_MTR_RECORDS_WRITTEN_PATTERN = re.compile(r"Number of Records Written=\s*([0-9]+)")


def _parse_mtr_dictionary_entry(line: str, *, line_number: int) -> _MtrDictionaryEntry:
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 3:
        raise ValueError(f"Malformed MTR data-dictionary record on line {line_number}")
    report_id = parts[0]
    if not re.fullmatch(r"[1-9][0-9]*", report_id):
        raise ValueError(f"Malformed MTR data-dictionary report ID on line {line_number}")
    item_count = _mtr_integer(parts[1], field="data-dictionary item count", line_number=line_number)
    if item_count < 1 or len(parts) != item_count + 2:
        raise ValueError(f"MTR data-dictionary item count does not match line {line_number}")
    description = ",".join(parts[2:])
    frequency_match = _MTR_FREQUENCY_PATTERN.search(description)
    return _MtrDictionaryEntry(
        report_id=report_id,
        item_count=item_count,
        description=description,
        frequency=None if frequency_match is None else frequency_match.group(1),
    )


def _validate_mtr_standard_dictionary_entry(
    entry: _MtrDictionaryEntry,
    *,
    line_number: int,
) -> None:
    """Validate the semantic binding of one reserved timestamp report ID.

    EnergyPlus 24.1 appends an informational ``! When ... Meters Requested``
    suffix to the daily and coarser meter timestamps.  Older files omit that
    suffix, so it is accepted only when absent or when it is the exact official
    meter-file suffix.  Merely matching the item count is not sufficient: a
    malformed dictionary must never let an arbitrary record masquerade as an
    environment or interval label.
    """

    expected_description = _MTR_STANDARD_DESCRIPTIONS[entry.report_id]
    description = " ".join(entry.description.split())
    expected = " ".join(expected_description.split())
    allowed = {expected}
    suffix = _MTR_STANDARD_DESCRIPTION_SUFFIXES.get(entry.report_id)
    if suffix is not None:
        allowed.add(f"{expected} {suffix}")
    if description not in allowed:
        raise ValueError(
            f"EnergyPlus MTR standard dictionary ID {entry.report_id} has an invalid description "
            f"on line {line_number}"
        )


def _validate_selected_environment_grid(
    environment: EnergyPlusMtrEnvironment,
    selected: list[EnergyPlusTimestepMeterRecord],
    *,
    line_number: int,
) -> None:
    if not selected:
        raise ValueError(
            f"MTR environment #{environment.sequence_index}:{environment.title} has no selected Timestep records"
        )
    first = selected[0].interval
    last = selected[-1].interval
    if (first.day_of_simulation, first.hour_1_based, first.start_minute) != (1, 1, Decimal(0)):
        raise ValueError(
            f"MTR environment #{environment.sequence_index}:{environment.title} selected Timestep grid "
            "does not begin at simulation day 1, hour 1, minute 0"
        )
    if (last.hour_1_based, last.end_minute) != (24, Decimal(60)):
        raise ValueError(
            f"MTR environment #{environment.sequence_index}:{environment.title} selected Timestep grid "
            "does not end at hour 24, minute 60"
        )
    duration = first.duration_minutes
    if Decimal(60) % duration != 0:
        raise ValueError(
            f"MTR environment #{environment.sequence_index}:{environment.title} selected Timestep duration "
            "does not divide 60 minutes"
        )
    expected_count = last.day_of_simulation * 24 * int(Decimal(60) / duration)
    if len(selected) != expected_count:
        raise ValueError(
            f"MTR environment #{environment.sequence_index}:{environment.title} selected Timestep grid has "
            f"{len(selected)} records; expected {expected_count}"
        )


def _validate_snapshot_stream_grid(records: tuple[EnergyPlusTimestepMeterRecord, ...]) -> None:
    """Validate a reconstructed stream without trusting its original objects."""

    current_environment = records[0].environment
    environment_records: list[EnergyPlusTimestepMeterRecord] = []
    completed_environments = 0

    def close_environment() -> None:
        nonlocal completed_environments, environment_records
        completed_environments += 1
        if current_environment.sequence_index != completed_environments:
            raise ValueError("MTR snapshot environment sequence indexes must be contiguous from one")
        _validate_selected_environment_grid(
            current_environment,
            environment_records,
            line_number=len(records) + 1,
        )
        environment_records = []

    for record_index, record in enumerate(records, start=1):
        if record.environment != current_environment:
            close_environment()
            current_environment = record.environment
        if environment_records:
            _validate_mtr_interval_successor(
                environment_records[-1].interval,
                record.interval,
                line_number=record_index,
            )
        environment_records.append(record)
    close_environment()


def _canonical_mtr_meter_names(meter_names: tuple[str, ...]) -> tuple[str, ...]:
    if type(meter_names) is not tuple or not meter_names:
        raise TypeError("meter_names must be a nonempty tuple of canonical strings")
    canonical: list[str] = []
    seen: set[str] = set()
    for meter_name in meter_names:
        if type(meter_name) is not str:
            raise TypeError("meter_names must contain only exact base strings")
        if not meter_name or meter_name != meter_name.strip():
            raise ValueError("meter_names must contain only canonical nonempty strings")
        if meter_name in seen:
            raise ValueError(f"meter_names contains duplicate meter {meter_name!r}")
        seen.add(meter_name)
        canonical.append(meter_name)
    return tuple(sorted(canonical))


def parse_energyplus_timestep_meter_bytes(
    meter_bytes: bytes,
    meter_names: tuple[str, ...],
) -> EnergyPlusTimestepMeterSnapshot:
    """Parse requested ``[J] !TimeStep`` streams from one exact MTR payload.

    The data dictionary, every environment and timestamp context, all
    requested meter values, and the records-written postamble are consumed in
    one pass over ``meter_bytes``.  A returned snapshot therefore cannot mix
    carriers from different filesystem opens or MTR revisions.
    """

    if type(meter_bytes) is not bytes or not meter_bytes:
        raise TypeError("meter_bytes must be nonempty immutable bytes")
    canonical_meter_names = _canonical_mtr_meter_names(meter_names)
    # Deliberately retain the legacy reader's raw UnicodeDecodeError behavior.
    text = meter_bytes.decode("utf-8", errors="strict")

    target_by_description = {f"{meter_name} [J] !TimeStep": meter_name for meter_name in canonical_meter_names}
    dictionary: dict[str, _MtrDictionaryEntry] = {}
    target_id_by_meter: dict[str, str] = {}
    meter_by_target_id: dict[str, str] = {}
    records_by_meter: dict[str, list[EnergyPlusTimestepMeterRecord]] = {
        meter_name: [] for meter_name in canonical_meter_names
    }
    phase = "dictionary"
    saw_program_version = False
    saw_end_of_data = False
    saw_records_written = False
    declared_records_written: int | None = None
    actual_records_written = 0
    environment: EnergyPlusMtrEnvironment | None = None
    environment_records_by_meter: dict[str, list[EnergyPlusTimestepMeterRecord]] = {
        meter_name: [] for meter_name in canonical_meter_names
    }
    pending_interval: EnergyPlusMtrIntervalLabel | None = None
    pending_selected_seen: set[str] = set()
    current_context_code: str | None = None
    current_context_frequencies: set[str] = set()
    current_context_data_records = 0

    def close_time_context(*, line_number: int) -> None:
        nonlocal pending_interval, current_context_code
        nonlocal current_context_frequencies, current_context_data_records
        if current_context_code in {"2", "3", "4", "5", "6"} and current_context_data_records == 0:
            raise ValueError(f"MTR code-{current_context_code} timestamp has no data record before line {line_number}")
        if current_context_code == "2":
            if pending_interval is None:
                raise AssertionError("internal MTR parser state lost its code-2 interval")
            if "TimeStep" in current_context_frequencies:
                missing = sorted(set(canonical_meter_names) - pending_selected_seen)
                if len(missing) == 1:
                    raise ValueError(
                        f"MTR code-2 Timestep interval has no {missing[0]!r} value before line {line_number}"
                    )
                if missing:
                    raise ValueError(
                        f"MTR code-2 Timestep interval is missing requested meter values {missing!r} "
                        f"before line {line_number}"
                    )
        pending_interval = None
        pending_selected_seen.clear()
        current_context_code = None
        current_context_frequencies = set()
        current_context_data_records = 0

    def close_environment(*, line_number: int) -> None:
        close_time_context(line_number=line_number)
        if environment is None:
            return
        for meter_name in canonical_meter_names:
            _validate_selected_environment_grid(
                environment,
                environment_records_by_meter[meter_name],
                line_number=line_number,
            )
            environment_records_by_meter[meter_name] = []

    with io.StringIO(text, newline="") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if phase == "dictionary":
                if line == "End of Data Dictionary":
                    if not saw_program_version:
                        raise ValueError("EnergyPlus MTR file has no Program Version record")
                    for report_id, expected_count in _MTR_STANDARD_ITEM_COUNTS.items():
                        entry = dictionary.get(report_id)
                        if entry is None or entry.item_count != expected_count:
                            raise ValueError(
                                f"EnergyPlus MTR standard dictionary ID {report_id} must declare {expected_count} items"
                            )
                    missing = sorted(set(canonical_meter_names) - set(target_id_by_meter))
                    if len(missing) == 1:
                        raise ValueError(f"Meter {missing[0]!r} with unit J and Timestep frequency is absent")
                    if missing:
                        raise ValueError(f"Meters {missing!r} with unit J and Timestep frequency are absent")
                    phase = "data"
                    continue
                if line.startswith("Program Version,"):
                    if saw_program_version or dictionary:
                        raise ValueError(f"Duplicate or misplaced Program Version record on MTR line {line_number}")
                    version_parts = [part.strip() for part in line.split(",")]
                    if (
                        len(version_parts) < 2
                        or version_parts[0] != "Program Version"
                        or version_parts[1] != "EnergyPlus"
                    ):
                        raise ValueError(f"Malformed Program Version record on MTR line {line_number}")
                    saw_program_version = True
                    continue
                entry = _parse_mtr_dictionary_entry(line, line_number=line_number)
                if entry.report_id in dictionary:
                    raise ValueError(f"Duplicate MTR dictionary report ID {entry.report_id} on line {line_number}")
                if entry.report_id in _MTR_STANDARD_ITEM_COUNTS:
                    if entry.item_count != _MTR_STANDARD_ITEM_COUNTS[entry.report_id]:
                        raise ValueError(
                            f"EnergyPlus MTR standard dictionary ID {entry.report_id} must declare "
                            f"{_MTR_STANDARD_ITEM_COUNTS[entry.report_id]} items"
                        )
                    _validate_mtr_standard_dictionary_entry(entry, line_number=line_number)
                dictionary[entry.report_id] = entry
                target_meter = target_by_description.get(entry.description)
                if target_meter is not None:
                    if entry.item_count != 1 or int(entry.report_id) <= 6:
                        raise ValueError(f"Meter {target_meter!r} has malformed dictionary binding")
                    if target_meter in target_id_by_meter:
                        raise ValueError(
                            f"Meter {target_meter!r} has multiple Timestep dictionary IDs: "
                            f"{[target_id_by_meter[target_meter], entry.report_id]}"
                        )
                    target_id_by_meter[target_meter] = entry.report_id
                    meter_by_target_id[entry.report_id] = target_meter
                continue

            if phase == "postamble":
                match = _MTR_RECORDS_WRITTEN_PATTERN.fullmatch(line)
                if match is None or saw_records_written:
                    raise ValueError(f"Unexpected content after End of Data on MTR line {line_number}")
                declared_records_written = int(match.group(1))
                saw_records_written = True
                continue

            if line in {"End of Data Dictionary", "Program Version"} or line.startswith("Program Version,"):
                raise ValueError(f"Duplicate or misplaced MTR structural record on line {line_number}")
            if line == "End of Data":
                close_environment(line_number=line_number)
                saw_end_of_data = True
                phase = "postamble"
                continue
            parts = [part.strip() for part in line.split(",")]
            record_code = parts[0]
            entry = dictionary.get(record_code)
            if entry is None:
                raise ValueError(f"Unknown MTR report code {record_code!r} on line {line_number}")
            if len(parts) != entry.item_count + 1:
                raise ValueError(f"MTR data item count does not match report code {record_code} on line {line_number}")
            if record_code == "1":
                close_environment(line_number=line_number)
                environment = _parse_mtr_environment(
                    parts,
                    sequence_index=1 if environment is None else environment.sequence_index + 1,
                    line_number=line_number,
                )
                continue
            if record_code == "2":
                if environment is None:
                    raise ValueError(f"MTR code-2 interval appears before a code-1 environment on line {line_number}")
                close_time_context(line_number=line_number)
                pending_interval = _parse_mtr_interval(parts, line_number=line_number)
                current_context_code = "2"
                continue
            if record_code in _MTR_STANDARD_ITEM_COUNTS:
                if environment is None:
                    raise ValueError(
                        f"MTR code-{record_code} timestamp appears before a code-1 environment on line {line_number}"
                    )
                close_time_context(line_number=line_number)
                current_context_code = record_code
                continue

            actual_records_written += 1
            expected_context = _MTR_FREQUENCY_CONTEXT.get(entry.frequency or "")
            if expected_context is None:
                raise ValueError(f"MTR report code {record_code} has no recognized reporting frequency")
            if current_context_code != expected_context:
                raise ValueError(
                    f"MTR {entry.frequency} report code {record_code} appears outside its code-{expected_context} "
                    f"context on line {line_number}"
                )
            current_context_data_records += 1
            current_context_frequencies.add(entry.frequency or "")
            if expected_context != "2":
                continue
            if pending_interval is None:
                raise ValueError(
                    f"MTR {entry.frequency} report code {record_code} appears before a code-2 interval on line {line_number}"
                )
            if entry.frequency == "Hourly" and (
                pending_interval.start_minute != Decimal(0) or pending_interval.end_minute != Decimal(60)
            ):
                raise ValueError(
                    f"MTR Hourly report code {record_code} has a non-hourly code-2 label on line {line_number}"
                )

            target_meter = meter_by_target_id.get(record_code)
            if target_meter is not None:
                if environment is None:
                    raise ValueError(
                        f"Meter {target_meter!r} appears before a code-1 environment on MTR line {line_number}"
                    )
                if target_meter in pending_selected_seen:
                    raise ValueError(
                        f"Duplicate meter {target_meter!r} value for one code-2 interval on MTR line {line_number}"
                    )
                try:
                    numeric_value = float(parts[1])
                except ValueError as error:
                    raise ValueError(f"Malformed meter {target_meter!r} value on MTR line {line_number}") from error
                numeric_value = _validate_nonnegative_finite(
                    numeric_value,
                    name=f"{target_meter} meter value on MTR line {line_number}",
                )
                environment_records = environment_records_by_meter[target_meter]
                if environment_records:
                    _validate_mtr_interval_successor(
                        environment_records[-1].interval,
                        pending_interval,
                        line_number=line_number,
                    )
                record = EnergyPlusTimestepMeterRecord(environment, pending_interval, numeric_value)
                environment_records.append(record)
                records_by_meter[target_meter].append(record)
                pending_selected_seen.add(target_meter)

    if phase == "dictionary":
        raise ValueError("EnergyPlus MTR file has no End of Data Dictionary marker")
    if not saw_end_of_data:
        raise ValueError("EnergyPlus MTR file has no End of Data marker")
    if not saw_records_written:
        raise ValueError("EnergyPlus MTR file has no Number of Records Written postamble")
    if declared_records_written != actual_records_written:
        raise ValueError(
            "EnergyPlus MTR Number of Records Written does not match data: "
            f"declared {declared_records_written}, counted {actual_records_written}"
        )
    for meter_name in canonical_meter_names:
        if not records_by_meter[meter_name]:
            raise ValueError(f"Meter {meter_name!r} has no Timestep records")
    return EnergyPlusTimestepMeterSnapshot(
        source_sha256=hashlib.sha256(meter_bytes).hexdigest(),
        source_size_bytes=len(meter_bytes),
        streams=tuple((meter_name, tuple(records_by_meter[meter_name])) for meter_name in canonical_meter_names),
    )


def _read_regular_mtr_bytes(meter_file: str | Path) -> bytes:
    """Read one no-follow, single-link MTR descriptor snapshot."""

    path = Path(meter_file)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("exact MTR path snapshots require O_NOFOLLOW support")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"EnergyPlus meter file not found: {path}") from error
    except OSError as error:
        raise ValueError(f"EnergyPlus meter file cannot be opened without following links: {path}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"EnergyPlus meter file must be a single-link regular file: {path}")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1024 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
        before_identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        after_identity = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        if before_identity != after_identity:
            raise ValueError(f"EnergyPlus meter file changed while it was read: {path}")
        payload = b"".join(chunks)
        if len(payload) != before.st_size:
            raise ValueError(f"EnergyPlus meter file size changed while it was read: {path}")
        return payload
    finally:
        os.close(descriptor)


def read_energyplus_timestep_meter_snapshot(
    meter_file: str | Path,
    meter_names: tuple[str, ...],
) -> EnergyPlusTimestepMeterSnapshot:
    """Read and parse requested meters from one descriptor-stable MTR file."""

    canonical_meter_names = _canonical_mtr_meter_names(meter_names)
    return parse_energyplus_timestep_meter_bytes(
        _read_regular_mtr_bytes(meter_file),
        canonical_meter_names,
    )


def _read_energyplus_timestep_meter_records(
    meter_file: str | Path,
    meter_name: str,
) -> tuple[EnergyPlusTimestepMeterRecord, ...]:
    """Compatibility implementation for the existing single-meter reader."""

    path = Path(meter_file)
    if not path.is_file():
        raise FileNotFoundError(f"EnergyPlus meter file not found: {path}")
    if type(meter_name) is not str:
        raise TypeError("meter_name must be an exact base string")
    if not meter_name.strip():
        raise ValueError("meter_name must not be empty")
    canonical_meter_name = meter_name.strip()
    snapshot = parse_energyplus_timestep_meter_bytes(
        path.read_bytes(),
        (canonical_meter_name,),
    )
    return snapshot.records_for(canonical_meter_name)


def read_energyplus_timestep_meter_records(
    meter_file: str | Path,
    meter_name: str,
) -> tuple[EnergyPlusTimestepMeterRecord, ...]:
    """Read one strict, environment-bound ``[J] !TimeStep`` meter series."""

    return _read_energyplus_timestep_meter_records(
        meter_file,
        meter_name,
    )


def read_energyplus_timestep_meter_joules(meter_file: str | Path, meter_name: str) -> tuple[float, ...]:
    """Project joules from the same strict environment-bound MTR reader."""

    return tuple(
        record.joules
        for record in _read_energyplus_timestep_meter_records(
            meter_file,
            meter_name,
        )
    )


def energy_intensity_kwh_per_m2(ledger: "CarrierEnergyLedger", floor_area_m2: float) -> float:
    """Return reported site energy per unit floor area for one interval."""

    area = _validate_nonnegative_finite(floor_area_m2, name="floor_area_m2")
    if area == 0.0:
        raise ValueError("floor_area_m2 must be positive")
    if not ledger.reported_carriers:
        raise ValueError("Cannot calculate energy intensity without a reported carrier")
    return ledger.total_reported_kwh / area


@dataclass(frozen=True)
class CarrierEnergyLedger:
    """Per-interval energy by carrier, in kWh.

    ``None`` means that the corresponding meter was not available. ``0.0``
    means that the meter was available and reported no consumption.
    """

    electricity_kwh: float | None = None
    natural_gas_kwh: float | None = None
    district_heating_water_kwh: float | None = None
    district_heating_steam_kwh: float | None = None
    district_cooling_kwh: float | None = None
    # Exact simulator source attached by the integration that read each
    # reported carrier.  The tuple representation keeps the frozen ledger
    # immutable and JSON-independent while preserving carrier identity.
    meter_sources: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        for field_name, value in self.as_dict().items():
            if value is not None:
                _validate_nonnegative_finite(value, name=field_name)
        normalized_sources: list[tuple[str, str]] = []
        seen: set[str] = set()
        allowed = {carrier.value for carrier in EnergyCarrier}
        for carrier, meter_name in self.meter_sources:
            if carrier not in allowed:
                raise ValueError(f"unknown energy carrier in meter_sources: {carrier!r}")
            if carrier in seen:
                raise ValueError(f"duplicate energy carrier in meter_sources: {carrier!r}")
            if not isinstance(meter_name, str) or not meter_name.strip():
                raise ValueError(f"meter source for {carrier!r} must be a non-empty string")
            seen.add(carrier)
            normalized_sources.append((carrier, meter_name.strip()))
        object.__setattr__(self, "meter_sources", tuple(sorted(normalized_sources)))

    @classmethod
    def from_joules(
        cls,
        *,
        electricity_joules: float | None = None,
        natural_gas_joules: float | None = None,
        district_heating_water_joules: float | None = None,
        district_heating_steam_joules: float | None = None,
        district_cooling_joules: float | None = None,
        meter_sources: Mapping[EnergyCarrier | str, str] | None = None,
    ) -> "CarrierEnergyLedger":
        """Build a ledger from EnergyPlus interval-meter readings."""

        def convert(value: float | None) -> float | None:
            return None if value is None else joules_to_kwh(value)

        canonical_sources = tuple(
            (
                carrier.value if isinstance(carrier, EnergyCarrier) else str(carrier),
                meter_name,
            )
            for carrier, meter_name in (meter_sources or {}).items()
        )
        return cls(
            electricity_kwh=convert(electricity_joules),
            natural_gas_kwh=convert(natural_gas_joules),
            district_heating_water_kwh=convert(district_heating_water_joules),
            district_heating_steam_kwh=convert(district_heating_steam_joules),
            district_cooling_kwh=convert(district_cooling_joules),
            meter_sources=canonical_sources,
        )

    def as_dict(self) -> dict[str, float | None]:
        return {
            EnergyCarrier.ELECTRICITY.value: self.electricity_kwh,
            EnergyCarrier.NATURAL_GAS.value: self.natural_gas_kwh,
            EnergyCarrier.DISTRICT_HEATING_WATER.value: self.district_heating_water_kwh,
            EnergyCarrier.DISTRICT_HEATING_STEAM.value: self.district_heating_steam_kwh,
            EnergyCarrier.DISTRICT_COOLING.value: self.district_cooling_kwh,
        }

    def value(self, carrier: EnergyCarrier) -> float | None:
        return self.as_dict()[carrier.value]

    @property
    def meter_source_mapping(self) -> dict[str, str]:
        """Return the exact simulator meter name recorded for each carrier."""

        return dict(self.meter_sources)

    @property
    def reported_carriers(self) -> tuple[EnergyCarrier, ...]:
        return tuple(carrier for carrier in EnergyCarrier if self.value(carrier) is not None)

    @property
    def total_reported_kwh(self) -> float:
        return sum(value for value in self.as_dict().values() if value is not None)


@dataclass(frozen=True)
class EmissionFactor:
    """Traceable operational-emission factor for one carrier and interval."""

    carrier: EnergyCarrier
    kg_co2e_per_kwh: float
    source: str
    version: str
    region: str
    effective_at: datetime | None = None

    def __post_init__(self) -> None:
        _validate_nonnegative_finite(self.kg_co2e_per_kwh, name="kg_co2e_per_kwh")
        for field_name in ("source", "version", "region"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")


@dataclass(frozen=True)
class OperationalCarbonResult:
    """Operational emissions and the factors used to calculate them."""

    kg_co2e_by_carrier: Mapping[EnergyCarrier, float]
    factors: Mapping[EnergyCarrier, EmissionFactor]

    @property
    def total_kg_co2e(self) -> float:
        return sum(self.kg_co2e_by_carrier.values())

    def as_dict(self) -> dict[str, object]:
        return {
            "kg_co2e_by_carrier": {carrier.value: value for carrier, value in self.kg_co2e_by_carrier.items()},
            "total_kg_co2e": self.total_kg_co2e,
            "factors": {
                carrier.value: {
                    "kg_co2e_per_kwh": factor.kg_co2e_per_kwh,
                    "source": factor.source,
                    "version": factor.version,
                    "region": factor.region,
                    "effective_at": factor.effective_at.isoformat() if factor.effective_at else None,
                }
                for carrier, factor in self.factors.items()
            },
        }


def compute_operational_carbon(
    ledger: CarrierEnergyLedger,
    factors: Mapping[EnergyCarrier, EmissionFactor],
) -> OperationalCarbonResult:
    """Calculate interval emissions using one explicit factor per reported carrier."""

    emissions: dict[EnergyCarrier, float] = {}
    used_factors: dict[EnergyCarrier, EmissionFactor] = {}
    for carrier in ledger.reported_carriers:
        factor = factors.get(carrier)
        if factor is None:
            raise ValueError(f"Missing emission factor for reported carrier {carrier.value!r}")
        if factor.carrier is not carrier:
            raise ValueError(f"Factor keyed as {carrier.value!r} declares carrier {factor.carrier.value!r}")
        energy_kwh = ledger.value(carrier)
        assert energy_kwh is not None
        emissions[carrier] = energy_kwh * factor.kg_co2e_per_kwh
        used_factors[carrier] = factor

    return OperationalCarbonResult(emissions, used_factors)
