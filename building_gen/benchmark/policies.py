"""Pure reference policies used to validate and benchmark control tasks."""

from __future__ import annotations

from building_gen.benchmark.specs import TaskSpec


def _require_normalized_thermostat(task: TaskSpec) -> None:
    profile = task.action_profile
    if profile.profile_id != "thermostat_v1":
        raise ValueError(f"task {task.name!r} uses action profile {profile.profile_id!r}, not 'thermostat_v1'")
    if task.action_space.kind != "box" or task.action_space.shape != (profile.expected_flat_size,):
        raise ValueError("thermostat reference policies require a flat Box action space")
    if task.action_space.lower_bound != 0.0 or task.action_space.upper_bound != 1.0:
        raise ValueError("thermostat reference policies require normalized [0, 1] actions")


def static_midpoint_action(task: TaskSpec) -> tuple[float, ...]:
    """Return the midpoint of every normalized action channel."""

    _require_normalized_thermostat(task)
    return (0.5,) * task.action_space.flat_size


def _normalize(value: float, minimum: float, maximum: float, *, label: str) -> float:
    if not minimum <= value <= maximum:
        raise ValueError(f"{label} target {value} is outside declared physical bounds [{minimum}, {maximum}]")
    return (value - minimum) / (maximum - minimum)


def ashrae_seasonal_action(task: TaskSpec, *, month: int) -> tuple[float, ...]:
    """Return the legacy HOT seasonal thermostat baseline in normalized units.

    November--March uses 20/23 degC heating/cooling setpoints.  April--October
    uses 22/26 degC.  The ``thermostat_v1`` action encodes heating temperature
    followed by cooling-minus-heating offset for every zone.
    """

    _require_normalized_thermostat(task)
    if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
        raise ValueError("month must be an integer from 1 through 12")
    channels = task.action_profile.channels
    expected_names = ("heating_setpoint_c", "cooling_offset_c")
    if tuple(channel.name for channel in channels) != expected_names:
        raise ValueError(f"thermostat_v1 channels must be ordered as {expected_names!r}")

    if month in {11, 12, 1, 2, 3}:
        heating, cooling = 20.0, 23.0
    else:
        heating, cooling = 22.0, 26.0
    if task.action_profile.transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1"):
        return fixed_thermostat_action(task,heating_c=heating,cooling_c=cooling)
    physical_values = (heating, cooling - heating)
    normalized_pair = tuple(
        _normalize(value, channel.physical_min, channel.physical_max, label=channel.name)
        for value, channel in zip(physical_values, channels)
    )
    return normalized_pair * len(task.action_profile.controlled_entities)


__all__ = ["ashrae_seasonal_action", "static_midpoint_action"]


def comfort_band_action(task: TaskSpec) -> tuple[float, ...]:
    """Fixed 20/24 C thermostat pair, aligned with this campaign's comfort band."""
    _require_normalized_thermostat(task)
    if task.action_profile.transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1"):
        return fixed_thermostat_action(task,heating_c=20.,cooling_c=24.)
    channels = task.action_profile.channels
    if tuple(c.name for c in channels) != ("heating_setpoint_c", "cooling_offset_c"):
        raise ValueError("Unexpected thermostat channel order")
    pair = tuple(_normalize(v, c.physical_min, c.physical_max, label=c.name)
                 for v, c in zip((20.0, 4.0), channels))
    return pair * len(task.action_profile.controlled_entities)


def fixed_thermostat_action(task: TaskSpec, *, heating_c: float, cooling_c: float) -> tuple[float, ...]:
    """Encode an explicit thermostat pair without clipping or changing its deadband."""
    import math
    _require_normalized_thermostat(task)
    if task.action_profile.transform_id in ("bounded_thermostat_15_30_v1", "bounded_thermostat_12_32_v1"):
        from building_gen.bounded_thermostat import encode,validate_ranges
        h,o=task.action_profile.channels
        if (h.name,o.name)!=("heating_setpoint_c","cooling_offset_c"):
            raise ValueError("Unexpected thermostat channel order")
        validate_ranges((h.physical_min,h.physical_max),(o.physical_min,o.physical_max),task.action_profile.transform_id)
        return encode(heating_c,cooling_c,task.action_profile.transform_id)*len(task.action_profile.controlled_entities)
    if task.action_profile.transform_id != 'linear_unquantized_v1':
        raise ValueError('fixed_thermostat requires the linear_unquantized_v1 transform')
    for name, value in [('heating_c', heating_c), ('cooling_c', cooling_c)]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{name} must be a finite number')
    if cooling_c <= heating_c:
        raise ValueError('cooling_c must exceed heating_c')
    channels = task.action_profile.channels
    if tuple(c.name for c in channels) != ('heating_setpoint_c', 'cooling_offset_c'):
        raise ValueError('Unexpected thermostat channel order')
    pair = tuple(_normalize(v, c.physical_min, c.physical_max, label=c.name)
                 for v, c in zip((heating_c, cooling_c-heating_c), channels))
    return pair * len(task.action_profile.controlled_entities)


def seasonal_thermostat_action(task: TaskSpec, *, month: int, winter_months,
                               winter_heating_c: float, winter_cooling_c: float,
                               summer_heating_c: float, summer_cooling_c: float) -> tuple[float, ...]:
    """Select an explicitly configured calendar-season thermostat pair."""
    if isinstance(month, bool) or not isinstance(month, int) or not 1 <= month <= 12:
        raise ValueError('month must be an integer from 1 through 12')
    if (not isinstance(winter_months, (list, tuple))
            or not winter_months or len(winter_months) == 12
            or any(isinstance(m, bool) or not isinstance(m, int) or not 1 <= m <= 12
                   for m in winter_months)
            or len(set(winter_months)) != len(winter_months)):
        raise ValueError('winter_months must contain unique month integers and leave a summer season')
    winter = fixed_thermostat_action(task, heating_c=winter_heating_c, cooling_c=winter_cooling_c)
    summer = fixed_thermostat_action(task, heating_c=summer_heating_c, cooling_c=summer_cooling_c)
    return winter if month in winter_months else summer


def scheduled_thermostat_action(task: TaskSpec, *, day_of_week: int, minute_of_day: float,
        occupied_weekdays, occupied_start_minute: int, occupied_end_minute: int,
        occupied_heating_c: float, occupied_cooling_c: float,
        unoccupied_heating_c: float, unoccupied_cooling_c: float) -> tuple[float, ...]:
    """Calendar occupancy schedule: Sunday=1; start inclusive, end exclusive."""
    import math
    if (not isinstance(occupied_weekdays, (list, tuple)) or not occupied_weekdays
            or any(type(d) is not int or not 1 <= d <= 7 for d in occupied_weekdays)
            or len(set(occupied_weekdays)) != len(occupied_weekdays)):
        raise ValueError('occupied_weekdays must contain unique integers 1 (Sunday) through 7')
    if (type(occupied_start_minute) is not int or type(occupied_end_minute) is not int
            or not 0 <= occupied_start_minute < occupied_end_minute <= 1440):
        raise ValueError('occupied interval must be integer minutes with 0 <= start < end <= 1440')
    if (type(day_of_week) is not int or not 1 <= day_of_week <= 7
            or isinstance(minute_of_day, bool) or not isinstance(minute_of_day, (int, float))
            or not math.isfinite(minute_of_day) or not 0 <= minute_of_day < 1440):
        raise ValueError('invalid schedule decision time')
    occupied = fixed_thermostat_action(task, heating_c=occupied_heating_c, cooling_c=occupied_cooling_c)
    unoccupied = fixed_thermostat_action(task, heating_c=unoccupied_heating_c, cooling_c=unoccupied_cooling_c)
    active = day_of_week in occupied_weekdays and occupied_start_minute <= minute_of_day < occupied_end_minute
    return occupied if active else unoccupied


def occupancy_reactive_action(task: TaskSpec, *, occupied,
        occupied_heating_c: float, occupied_cooling_c: float,
        unoccupied_heating_c: float, unoccupied_cooling_c: float) -> tuple[float, ...]:
    """Per-zone measured-occupancy thermostat, without forecast or preconditioning."""
    zones = task.action_profile.controlled_entities
    if len(occupied) != len(zones) or any(type(value) is not bool for value in occupied):
        raise ValueError('occupied must contain one boolean per controlled zone')
    active = fixed_thermostat_action(task, heating_c=occupied_heating_c, cooling_c=occupied_cooling_c)
    empty = fixed_thermostat_action(task, heating_c=unoccupied_heating_c, cooling_c=unoccupied_cooling_c)
    return tuple(value for i, flag in enumerate(occupied)
                 for value in (active if flag else empty)[2*i:2*i+2])


def holiday_aware_thermostat_action(task: TaskSpec, *, special_day_type: int,
                                   occupied, **parameters) -> tuple[float, ...]:
    """Keep weekly preconditioning; on EnergyPlus holidays react to actual occupancy.

    EnergyPlus special-day category 1 is Holiday. Other categories retain the
    declared weekly schedule. No future occupancy is used.
    """
    if type(special_day_type) is not int or not 0 <= special_day_type <= 5:
        raise ValueError('invalid special-day category')
    scheduled = scheduled_thermostat_action(task, **parameters)
    reactive = occupancy_reactive_action(task, occupied=occupied,
        **{key: parameters[key] for key in ('occupied_heating_c','occupied_cooling_c',
                                           'unoccupied_heating_c','unoccupied_cooling_c')})
    return reactive if special_day_type == 1 else scheduled
