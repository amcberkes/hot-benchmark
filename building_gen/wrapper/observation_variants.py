"""Versioned observation ablations using only decision-time information."""
import gymnasium as gym
import numpy as np
import math
from building_gen.comfort import ComfortSchedule
from building_gen.wrapper.observation_encoding import clipped_affine, saturating_nonnegative

CALENDAR_ENCODING = 'hot_observation_occupancy_bounds_calendar_v1'
SPECIAL_DAY_ENCODING = 'hot_observation_occupancy_bounds_calendar_special_day_v1'

VARIANTS = {
    CALENDAR_ENCODING: (True, True, False),
    SPECIAL_DAY_ENCODING: (True, True, False),
    'hot_observation_occupancy_v1': (True, False, False),
    'hot_observation_bounds_v1': (False, True, False),
    'hot_observation_occupancy_bounds_v1': (True, True, False),
    'hot_observation_occupancy_bounds_actions_v1': (True, True, True),
}


def extra_names(encoding, zones):
    occupancy, bounds, actions = VARIANTS[encoding]
    names = []
    if occupancy:
        names += [f'{z}_occupant_count' for z in zones]
    if bounds:
        names += [f'{z}_comfort_{b}_c' for z in zones for b in ('minimum', 'maximum')]
    if actions:
        names += [f'{z}_previous_{b}' for z in zones for b in ('heating_setpoint', 'cooling_offset')]
        names += ['previous_action_available']
    if encoding in (CALENDAR_ENCODING, SPECIAL_DAY_ENCODING):
        names += [f'{z}_occupied' for z in zones]
        names += ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos']
    if encoding == SPECIAL_DAY_ENCODING:
        names += [f'special_day_type_{i}' for i in range(6)]
    return names


class ObservationVariant(gym.Wrapper):
    """Append declared features; retain raw evidence and the underlying reward.

    Occupancy uses n/(n+10), where 10 persons is a fixed numerical scale,
    independent of policy outcomes. Bounds use (T-10)/30, without silently
    clipping out-of-range bounds. Previous commands use end-interval actuator
    readbacks, normalized with the task's action channel ranges. Missing history
    on reset/native control is zeros plus an explicit availability indicator.
    """
    def __init__(self, env, task):
        super().__init__(env)
        if env.observation_encoding_id != 'hot_observation_v2':
            raise ValueError('observation variants require the base v2 encoding')
        self.observation_encoding_id = task.observation_space.encoding
        self.features = VARIANTS[self.observation_encoding_id]
        self.zones = tuple(task.action_profile.controlled_entities)
        self.channels = task.action_profile.channels
        if self.features[2] and tuple(c.name for c in self.channels) != (
                'heating_setpoint_c', 'cooling_offset_c'):
            raise ValueError('previous-action encoding requires thermostat channels')
        self.schedule = None
        if self.features[1]:
            self.schedule = ComfortSchedule.from_mapping(task.reward.parameter_mapping['comfort_schedule'])
            self.schedule.validate_zones(self.zones)
        self.observation_names = list(env.observation_names) + extra_names(self.observation_encoding_id, self.zones)
        self.observation_space = gym.spaces.Box(0, 1, shape=(len(self.observation_names),), dtype=np.float32)

    def __getattr__(self, name):
        # Benchmark provenance and actuator contracts remain those of the child.
        if name == 'env':
            raise AttributeError(name)
        return getattr(self.env, name)

    def _encode(self, observation, info, *, reset=False, native=False):
        occupancy, bounds, actions = self.features
        values = list(observation)
        if occupancy:
            raw = info['raw_observation']
            values += [saturating_nonnegative(raw[f'{z}_occupant_count'], scale=10.0) for z in self.zones]
        if bounds:
            clock = info['simulation_clock']
            minute = 60 * (clock['hour'] + (clock['timestep_index'] - 1) / clock['timesteps_per_hour'])
            for zone in self.zones:
                band = self.schedule.band_at(zone, weekday=clock['day_of_week'], minute=minute, month=clock.get('month'))
                for bound in (band.minimum_c, band.maximum_c):
                    if not 10 <= bound <= 40:
                        raise ValueError('comfort bounds outside observation encoding range')
                    values.append(clipped_affine(bound, 10, 40))
        if actions:
            if reset or native:
                values += [0.0] * (2 * len(self.zones)) + [0.0]
            else:
                evidence = info.get('actuator_readback', {})
                if evidence.get('available') is not True or evidence.get('phase') != 'end_zone_timestep':
                    raise ValueError('previous-action observation requires end-interval actuator readbacks')
                records = evidence['records']
                if tuple(r['zone'] for r in records) != self.zones:
                    raise ValueError('previous-action readback zone order differs from task')
                for record in records:
                    heat, cool = record['heating_setpoint_c'], record['cooling_setpoint_c']
                    for value, channel in zip((heat, cool - heat), self.channels):
                        if not channel.physical_min <= value <= channel.physical_max:
                            raise ValueError('previous actuator value outside declared action range')
                        values.append(clipped_affine(value, channel.physical_min, channel.physical_max))
                values.append(1.0)
        if self.observation_encoding_id in (CALENDAR_ENCODING, SPECIAL_DAY_ENCODING):
            raw=info['raw_observation'];clock=info['simulation_clock']
            values += [float(raw[f'{z}_occupant_count'] > 0) for z in self.zones]
            hour=clock['hour']+(clock['timestep_index']-1)/clock['timesteps_per_hour']
            for phase in (hour/24., (clock['day_of_week']-1)/7.):
                values += [(math.sin(2*math.pi*phase)+1)/2, (math.cos(2*math.pi*phase)+1)/2]
        if self.observation_encoding_id == SPECIAL_DAY_ENCODING:
            context = info.get('energyplus_runtime_context', {})
            day_type = context.get('holiday_index')
            if type(day_type) is not int or not 0 <= day_type <= 5:
                raise ValueError('special-day observation requires valid decision-time holiday_index')
            values += [float(day_type == i) for i in range(6)]
        encoded = np.asarray(values, dtype=np.float32)
        if not self.observation_space.contains(encoded):
            raise ValueError('invalid observation variant values')
        return encoded

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._encode(obs, info, reset=True), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._encode(obs, info), reward, terminated, truncated, info

    def step_native_schedule(self):
        obs, reward, terminated, truncated, info = self.env.step_native_schedule()
        return self._encode(obs, info, native=True), reward, terminated, truncated, info
