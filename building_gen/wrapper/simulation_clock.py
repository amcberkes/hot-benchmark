"""EnergyPlus simulation-calendar records and normalized time features.

This module deliberately has no :mod:`pyenergyplus` import so its conversion
logic can be unit tested outside the EnergyPlus container.  The calendar year
reported by EnergyPlus is a *simulation* year.  In particular, a TMY weather
file represents a constructed typical year, so that number must not be used as
the year of a historical or marginal-emissions profile.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Mapping, TypedDict

CalendarYearSemantics = Literal[
    "actual_weather_year",
    "typical_weather_year",
    "unknown",
]


class SimulationClockRecord(TypedDict):
    """Primitive-only form stored in step results and Gym ``info``."""

    calendar_year: int
    month: int
    day_of_month: int
    hour: int
    minutes: int
    day_of_week: int
    day_of_year: int
    timestep_index: int
    timesteps_per_hour: int
    episode_step_index: int
    calendar_year_semantics: CalendarYearSemantics


class EnergyPlusRuntimeContextRecord(TypedDict):
    """Primitive-only EnergyPlus environment context stored beside the clock."""

    environment_num: int
    kind_of_sim: int
    dst_indicator: int
    holiday_index: int


_TYPICAL_YEAR_MARKER = re.compile(r"(?:^|[_.-])(?:TMY(?:[23x])?|IWEC|CWEC|WYEC)(?:[_.-]|$)", re.IGNORECASE)
_ACTUAL_YEAR_MARKER = re.compile(r"(?:^|[_.-])AMY(?:[_.-]|$)", re.IGNORECASE)


def infer_calendar_year_semantics(weather_path: str | Path | None) -> CalendarYearSemantics:
    """Infer HOT's weather-year convention from a weather filename.

    HOT names actual-weather files with ``AMY`` and typical-weather files with
    markers such as ``TMY3``.  Unmarked files remain ``unknown`` rather than
    being assumed to describe an actual year.
    """

    if weather_path is None:
        return "unknown"
    name = Path(weather_path).name
    if _ACTUAL_YEAR_MARKER.search(name):
        return "actual_weather_year"
    if _TYPICAL_YEAR_MARKER.search(name):
        return "typical_weather_year"
    return "unknown"


def validate_calendar_year_semantics(value: str) -> CalendarYearSemantics:
    """Validate an explicit calendar-year interpretation."""

    valid = {"actual_weather_year", "typical_weather_year", "unknown"}
    if value not in valid:
        raise ValueError(f"calendar-year semantics must be one of {sorted(valid)}, got {value!r}")
    return value  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class SimulationClock:
    """Calendar fields read from EnergyPlus for one emitted observation.

    ``timestep_index`` is EnergyPlus' one-based zone-timestep number within the
    current hour. ``episode_step_index`` is HOT's zero-based observation index.
    ``calendar_year_semantics`` prevents a plausible-looking TMY simulation
    year from being mistaken for an actual weather or emissions year.
    """

    calendar_year: int
    month: int
    day_of_month: int
    hour: int
    minutes: int
    day_of_week: int
    day_of_year: int
    timestep_index: int
    timesteps_per_hour: int
    episode_step_index: int
    calendar_year_semantics: CalendarYearSemantics = "unknown"

    def __post_init__(self) -> None:
        validate_calendar_year_semantics(self.calendar_year_semantics)
        _require_native_int("calendar_year", self.calendar_year)
        if self.calendar_year < 0:
            raise ValueError("calendar_year must be non-negative")
        _require_range("month", self.month, 1, 12)
        _require_range("day_of_month", self.day_of_month, 1, 31)
        _require_range("hour", self.hour, 0, 23)
        _require_range("minutes", self.minutes, 1, 60)
        _require_range("day_of_week", self.day_of_week, 1, 7)
        _require_range("day_of_year", self.day_of_year, 1, 366)
        _require_native_int("timesteps_per_hour", self.timesteps_per_hour)
        if self.timesteps_per_hour <= 0 or 60 % self.timesteps_per_hour != 0:
            raise ValueError("timesteps_per_hour must be a positive divisor of 60")
        _require_range("timestep_index", self.timestep_index, 1, self.timesteps_per_hour)
        expected_minutes = self.timestep_index * (60 // self.timesteps_per_hour)
        if self.minutes != expected_minutes:
            raise ValueError(
                "minutes must equal timestep_index * (60 / timesteps_per_hour); "
                f"expected {expected_minutes}, got {self.minutes}"
            )
        _require_native_int("episode_step_index", self.episode_step_index)
        if self.episode_step_index < 0:
            raise ValueError("episode_step_index must be non-negative")
        # EnergyPlus can use calendar year zero for an unbounded legacy
        # RunPeriod. A leap representative validates its month/day without
        # inventing year semantics; positive simulator years are checked
        # exactly, including their day-of-year label.
        validation_year = self.calendar_year if self.calendar_year > 0 else 2000
        try:
            calendar_date = date(validation_year, self.month, self.day_of_month)
        except ValueError as error:
            raise ValueError("month/day is not valid for the simulation calendar year") from error
        if self.calendar_year > 0 and self.day_of_year != calendar_date.timetuple().tm_yday:
            raise ValueError(
                f"day_of_year does not match calendar date; expected "
                f"{calendar_date.timetuple().tm_yday}, got {self.day_of_year}"
            )

    def as_dict(self) -> SimulationClockRecord:
        """Return a JSON-safe record (all fields are primitive values)."""

        return {
            "calendar_year": self.calendar_year,
            "month": self.month,
            "day_of_month": self.day_of_month,
            "hour": self.hour,
            "minutes": self.minutes,
            "day_of_week": self.day_of_week,
            "day_of_year": self.day_of_year,
            "timestep_index": self.timestep_index,
            "timesteps_per_hour": self.timesteps_per_hour,
            "episode_step_index": self.episode_step_index,
            "calendar_year_semantics": self.calendar_year_semantics,
        }

    def normalized_observation(self) -> dict[str, float]:
        """Return the wrapper's three bounded calendar features.

        The encodings retain the existing observation names and 0--1 bounds:

        * ``hour_of_day`` is the interval-end local simulation time divided by
          24, so 00:15 is 0.010416... and 23:60 is 1.0;
        * ``day_of_week`` is ``(EnergyPlus day - 1) / 7`` (Sunday is 0 and
          Saturday is 6/7), retaining the previous seven-day fraction;
        * ``day_of_year`` is ``(EnergyPlus day - 1) / 365`` (January 1 is 0;
          December 31 of a leap year is 1.0).

        These are policy inputs, not timestamps for emissions-factor lookup.
        """

        result = {
            "hour_of_day": (self.hour + self.minutes / 60.0) / 24.0,
            "day_of_week": (self.day_of_week - 1) / 7.0,
            "day_of_year": (self.day_of_year - 1) / 365.0,
        }
        if not all(0.0 <= value <= 1.0 for value in result.values()):
            raise ValueError(f"normalized clock observation is outside [0, 1]: {result}")
        return result


@dataclass(frozen=True, slots=True)
class EnergyPlusRuntimeContext:
    """Exact non-calendar API context for one emitted EnergyPlus state.

    This is deliberately separate from :class:`SimulationClock` so the stable
    Gym clock schema remains frozen while MTR reconciliation can bind the
    simulator environment occurrence, type, DST state, and special-day type.
    EnergyPlus ``kind_of_sim == 3`` identifies a weather-file run period.
    """

    environment_num: int
    kind_of_sim: int
    dst_indicator: int
    holiday_index: int

    def __post_init__(self) -> None:
        _require_native_int("environment_num", self.environment_num)
        if self.environment_num <= 0:
            raise ValueError("environment_num must be positive")
        _require_range("kind_of_sim", self.kind_of_sim, 1, 6)
        _require_range("dst_indicator", self.dst_indicator, 0, 1)
        _require_range("holiday_index", self.holiday_index, 0, 5)

    def as_dict(self) -> EnergyPlusRuntimeContextRecord:
        return {
            "environment_num": self.environment_num,
            "kind_of_sim": self.kind_of_sim,
            "dst_indicator": self.dst_indicator,
            "holiday_index": self.holiday_index,
        }


def simulation_clock_from_exchange(
    exchange: Any,
    state: int,
    *,
    episode_step_index: int,
    calendar_year_semantics: CalendarYearSemantics = "unknown",
) -> SimulationClock:
    """Read one canonical zone-timestep clock from an EnergyPlus exchange.

    EnergyPlus ``minutes()`` follows the active system substep at some runtime
    callbacks, so it can be 17 or 19 minutes while the enclosing first
    15-minute zone timestep is still being reported.  HOT labels policy and
    meter intervals on the zone-timestep grid instead: the one-based zone
    timestep number and the fixed number of zone timesteps per hour determine
    the canonical interval-end minute.
    """

    timestep_index = exchange.zone_time_step_number(state)
    timesteps_per_hour = exchange.num_time_steps_in_hour(state)
    _require_native_int("timesteps_per_hour", timesteps_per_hour)
    if timesteps_per_hour <= 0 or 60 % timesteps_per_hour != 0:
        raise ValueError("timesteps_per_hour must be a positive divisor of 60")
    _require_range("timestep_index", timestep_index, 1, timesteps_per_hour)
    minutes = timestep_index * (60 // timesteps_per_hour)

    return SimulationClock(
        calendar_year=exchange.calendar_year(state),
        month=exchange.month(state),
        day_of_month=exchange.day_of_month(state),
        hour=exchange.hour(state),
        minutes=minutes,
        day_of_week=exchange.day_of_week(state),
        day_of_year=exchange.day_of_year(state),
        timestep_index=timestep_index,
        timesteps_per_hour=timesteps_per_hour,
        episode_step_index=episode_step_index,
        calendar_year_semantics=calendar_year_semantics,
    )


def energyplus_runtime_context_from_exchange(
    exchange: Any,
    state: int,
    *,
    kind_of_sim: int | None = None,
) -> EnergyPlusRuntimeContext:
    """Capture exact environment metadata in the same callback as its clock."""

    observed_kind = exchange.kind_of_sim(state) if kind_of_sim is None else kind_of_sim
    dst_indicator = exchange.daylight_savings_time_indicator(state)
    if type(dst_indicator) is not bool:
        raise ValueError(
            "EnergyPlus daylight_savings_time_indicator must return a native bool, "
            f"got {type(dst_indicator).__name__}"
        )
    return EnergyPlusRuntimeContext(
        environment_num=exchange.current_environment_num(state),
        kind_of_sim=observed_kind,
        dst_indicator=1 if dst_indicator else 0,
        holiday_index=exchange.holiday_index(state),
    )


def advance_zone_timestep(clock: SimulationClock) -> SimulationClock:
    """Advance a validated clock by one canonical EnergyPlus zone timestep.

    EnergyPlus labels interval ends as ``minute = 60`` rather than carrying
    that instant into the next hour.  Advancing the one-based zone-timestep
    index therefore preserves that convention and advances the calendar only
    after the final timestep of hour 23.
    """

    if 60 % clock.timesteps_per_hour != 0:
        raise ValueError("timesteps_per_hour must divide 60 to construct a canonical clock")

    minutes_per_timestep = 60 // clock.timesteps_per_hour
    next_timestep_index = clock.timestep_index + 1
    if next_timestep_index <= clock.timesteps_per_hour:
        return SimulationClock(
            calendar_year=clock.calendar_year,
            month=clock.month,
            day_of_month=clock.day_of_month,
            hour=clock.hour,
            minutes=next_timestep_index * minutes_per_timestep,
            day_of_week=clock.day_of_week,
            day_of_year=clock.day_of_year,
            timestep_index=next_timestep_index,
            timesteps_per_hour=clock.timesteps_per_hour,
            episode_step_index=clock.episode_step_index + 1,
            calendar_year_semantics=clock.calendar_year_semantics,
        )

    next_hour = clock.hour + 1
    calendar_year = clock.calendar_year
    month = clock.month
    day_of_month = clock.day_of_month
    day_of_week = clock.day_of_week
    day_of_year = clock.day_of_year

    if next_hour == 24:
        next_hour = 0
        day_of_week = day_of_week % 7 + 1
        days_in_month = calendar.monthrange(calendar_year, month)[1]
        if day_of_month < days_in_month:
            day_of_month += 1
            day_of_year += 1
        elif month < 12:
            month += 1
            day_of_month = 1
            day_of_year += 1
        else:
            calendar_year += 1
            month = 1
            day_of_month = 1
            day_of_year = 1

    return SimulationClock(
        calendar_year=calendar_year,
        month=month,
        day_of_month=day_of_month,
        hour=next_hour,
        minutes=minutes_per_timestep,
        day_of_week=day_of_week,
        day_of_year=day_of_year,
        timestep_index=1,
        timesteps_per_hour=clock.timesteps_per_hour,
        episode_step_index=clock.episode_step_index + 1,
        calendar_year_semantics=clock.calendar_year_semantics,
    )


def clock_info(
    clock: SimulationClock | SimulationClockRecord,
) -> Mapping[str, int | str]:
    """Build the stable Gym-info payload for a clock record."""

    if isinstance(clock, SimulationClock):
        return clock.as_dict()
    return clock


def runtime_context_info(
    context: EnergyPlusRuntimeContext | EnergyPlusRuntimeContextRecord,
) -> Mapping[str, int]:
    """Build the stable Gym-info payload for EnergyPlus runtime context."""

    if isinstance(context, EnergyPlusRuntimeContext):
        return context.as_dict()
    return context


def _require_range(name: str, value: int, minimum: int, maximum: int) -> None:
    _require_native_int(name, value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}], got {value}")


def _require_native_int(name: str, value: object) -> None:
    if type(value) is not int:
        raise ValueError(f"{name} must be a native int, got {type(value).__name__}")


__all__ = [
    "CalendarYearSemantics",
    "EnergyPlusRuntimeContext",
    "EnergyPlusRuntimeContextRecord",
    "SimulationClock",
    "SimulationClockRecord",
    "advance_zone_timestep",
    "clock_info",
    "energyplus_runtime_context_from_exchange",
    "infer_calendar_year_semantics",
    "simulation_clock_from_exchange",
    "runtime_context_info",
    "validate_calendar_year_semantics",
]
