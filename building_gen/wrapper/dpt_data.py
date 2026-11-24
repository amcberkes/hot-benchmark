"""Shared-zone DPT transition packing with explicit observed reward semantics."""
import math
import numpy as np

ZONE_FEATURES = ('temperature', 'occupant_count', 'comfort_minimum_c',
                 'comfort_maximum_c', 'occupied')
GLOBAL_FEATURES = ('outdoor_temperature', 'outdoor_humidity', 'hvac_electricity',
                   'hvac_gas', 'hour_of_day', 'day_of_week', 'day_of_year',
                   'hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos')
ENCODING = 'hot_observation_occupancy_bounds_calendar_v1'


def pack_states(observations, names, zones):
    """Preserve HOT normalizers and action-zone order, without fitting new scales."""
    x = np.asarray(observations, dtype=np.float32)
    expected = set(GLOBAL_FEATURES) | {z+'_'+f for z in zones for f in ZONE_FEATURES}
    if (not zones or len(set(zones)) != len(zones) or len(set(names)) != len(names)
            or set(names) != expected or x.ndim != 2 or x.shape[1] != len(names)
            or not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1)):
        raise ValueError('invalid DPT named observation layout')
    indices = [[names.index(z+'_'+f) for f in ZONE_FEATURES]
               + [names.index(f) for f in GLOBAL_FEATURES] for z in zones]
    return x[:, indices]


def interval_reward(*, energy_kwh, area_m2, interval_hours, squared_person_degree_hours,
                    person_hours, reference_kw_per_m2, comfort_weight=1.):
    """A deployable shared building reward; no target-fitted energy reference.

    Source-only reference power is fixed before target deployment. The interval
    comfort mean uses actual occupied person-hours; it is zero in empty periods.
    Evaluation still pools exposures/person-hours across the entire calendar.
    """
    if not math.isfinite(comfort_weight) or comfort_weight < 0:
        raise ValueError('invalid physical comfort weight')
    energy, comfort, _ = interval_feedback(energy_kwh=energy_kwh, area_m2=area_m2,
        interval_hours=interval_hours, squared_person_degree_hours=squared_person_degree_hours,
        person_hours=person_hours, reference_kw_per_m2=reference_kw_per_m2)
    return energy + comfort_weight*comfort


def interval_feedback(*, energy_kwh, area_m2, interval_hours, squared_person_degree_hours,
                      person_hours, reference_kw_per_m2):
    """Separate negative energy/comfort costs and elapsed hours, with fixed units.

    The comfort channel is unweighted. It is shared building-level feedback,
    not a measured zone energy or individual-zone comfort label.
    """
    values = (energy_kwh, area_m2, interval_hours, squared_person_degree_hours,
              person_hours, reference_kw_per_m2)
    if (any(not math.isfinite(x) or x < 0 for x in values)
            or min(area_m2, interval_hours, reference_kw_per_m2) <= 0
            or (person_hours == 0 and squared_person_degree_hours != 0)):
        raise ValueError('invalid interval physical quantities or source scale')
    comfort = squared_person_degree_hours/person_hours if person_hours else 0.
    energy = energy_kwh/(area_m2*reference_kw_per_m2)
    return (-energy, -interval_hours*comfort, interval_hours)


def pack_episode(observations, actions, teacher_actions, rewards, names, zones):
    """Create completed tuples within one weekly episode.

    Recorded next-decision observations supply normalized next states. The last
    action has no following decision row and is explicitly omitted, never joined
    to a simulator reset. All simulator transitions still count in the budget.
    """
    states = pack_states(observations, names, zones)
    n, z, _ = states.shape
    behavior, labels = np.asarray(actions, dtype=np.float32), np.asarray(teacher_actions, dtype=np.float32)
    rewards = np.asarray(rewards, dtype=np.float32)
    if (n < 2 or behavior.shape != (n, 2*z) or labels.shape != behavior.shape
            or rewards.shape != (n,) or not np.isfinite(rewards).all()
            or any(not np.isfinite(x).all() or np.any(x < 0) or np.any(x > 1)
                   for x in (behavior, labels))):
        raise ValueError('invalid episode actions, labels or rewards')
    return dict(states=states[:-1], next_states=states[1:],
        actions=behavior[:-1].reshape(n-1, z, 2),
        teacher_actions=labels[:-1].reshape(n-1, z, 2), rewards=rewards[:-1, None])
