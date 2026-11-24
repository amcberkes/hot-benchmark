"""Versioned occupied reward with explicit zone and weekly comfort schedules."""
from dataclasses import dataclass, asdict
from typing import Mapping, Any, Iterable
import math

from building_gen.comfort import ComfortSchedule
from building_gen.reward_foundation import (
    OccupiedEnergyComfortV2Config, ZoneComfortSample,
    calculate_hot_occupied_energy_comfort_v2,
)

HOT_SCHEDULED_OCCUPIED_ENERGY_COMFORT_V3 = 'hot_scheduled_occupied_energy_comfort_v3'


def verify_scheduled_interval(
    info: Mapping[str, Any], *, scalar_reward: float,
    interval_controlled_hvac_site_energy_kwh: float,
    zone_floor_areas_m2: Mapping[str, float], controlled_zones: tuple[str, ...],
    timesteps_per_hour: int, config: 'ScheduledOccupiedRewardConfig',
) -> dict:
    """Verify one Gym transition using its completed-interval physical evidence.

    ``zone_floor_areas_m2`` must be supplied from model geometry. Next-state
    observations and the decision clock are deliberately not used here.
    """
    from building_gen.wrapper.simulation_clock import SimulationClock

    if not isinstance(info, Mapping):
        raise ValueError('scheduled interval info must be a mapping')
    if (not controlled_zones or len(set(controlled_zones)) != len(controlled_zones)
            or set(zone_floor_areas_m2) != set(controlled_zones)):
        raise ValueError('model zone areas must match unique controlled zones exactly')
    clock_mapping = info.get('energy_interval_clock')
    observations = info.get('energy_interval_observation')
    if not isinstance(clock_mapping, Mapping) or not isinstance(observations, Mapping):
        raise ValueError('scheduled verification requires completed interval clock and observations')
    clock = SimulationClock(**clock_mapping)
    if type(timesteps_per_hour) is not int or timesteps_per_hour != clock.timesteps_per_hour:
        raise ValueError('completed interval timestep differs from task timestep')
    zones = []
    for zone in controlled_zones:
        try:
            temperature = observations[f'{zone}_temperature']
            occupants = observations[f'{zone}_occupant_count']
        except KeyError as error:
            raise ValueError(f'completed interval is missing physical samples for {zone}') from error
        zones.append(ZoneComfortSample(zone, zone_floor_areas_m2[zone], temperature, occupants))
    return verify_scheduled_reward_components(
        info.get('reward_components'), scalar_reward=scalar_reward,
        interval_controlled_hvac_site_energy_kwh=interval_controlled_hvac_site_energy_kwh,
        timestep_hours=1 / timesteps_per_hour, zones=zones, config=config,
        weekday=clock.day_of_week, month=clock.month,
        start_minute=60 * (clock.hour + (clock.timestep_index - 1) / timesteps_per_hour),
    )


def verify_scheduled_reward_components(
    components: Mapping[str, Any], *, scalar_reward: float,
    interval_controlled_hvac_site_energy_kwh: float, timestep_hours: float,
    zones: Iterable[ZoneComfortSample], config: 'ScheduledOccupiedRewardConfig',
    weekday: int, start_minute: float, month: int | None = None,
) -> dict:
    """Recompute saved evidence from physical samples and the pinned schedule.

    Inputs must come from the completed interval and authenticated model areas,
    not from the component record being checked. Returns canonical recomputed
    evidence; rejects missing/extra fields, changed bands and nonfinite numbers.
    """
    expected = calculate_scheduled_occupied_reward(
        interval_controlled_hvac_site_energy_kwh=interval_controlled_hvac_site_energy_kwh,
        timestep_hours=timestep_hours, zones=zones, config=config,
        weekday=weekday, start_minute=start_minute, month=month,
    )

    def compare(actual, reference, path):
        if isinstance(reference, dict):
            if not isinstance(actual, Mapping) or set(actual) != set(reference):
                raise ValueError(f'{path} fields differ from recomputed evidence')
            for key, value in reference.items():
                compare(actual[key], value, f'{path}.{key}')
        elif isinstance(reference, list):
            if not isinstance(actual, (list, tuple)) or len(actual) != len(reference):
                raise ValueError(f'{path} segments differ from recomputed evidence')
            for index, value in enumerate(reference):
                compare(actual[index], value, f'{path}[{index}]')
        elif isinstance(reference, (int, float)):
            if (isinstance(actual, bool) or not isinstance(actual, (int, float))
                    or not math.isfinite(actual)
                    or not math.isclose(actual, reference, rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError(f'{path} differs from recomputed evidence')
        elif actual != reference:
            raise ValueError(f'{path} differs from recomputed evidence')

    compare(components, expected, 'reward_components')
    compare(scalar_reward, expected['reward'], 'scalar_reward')
    return expected


@dataclass(frozen=True)
class ScheduledOccupiedRewardConfig:
    energy_weight: float
    comfort_weight: float
    energy_scale_kwh_per_m2: float
    comfort_scale_person_celsius_squared_hours_per_m2: float
    comfort_schedule: ComfortSchedule

    def __post_init__(self):
        if not isinstance(self.comfort_schedule, ComfortSchedule):
            raise TypeError('comfort_schedule must be a ComfortSchedule')
        validated = self.fixed_config()
        for field in ('energy_weight', 'comfort_weight', 'energy_scale_kwh_per_m2',
                      'comfort_scale_person_celsius_squared_hours_per_m2'):
            object.__setattr__(self, field, getattr(validated, field))

    def fixed_config(self) -> OccupiedEnergyComfortV2Config:
        """Reuse unchanged physical/scale validation; v2 itself remains unchanged."""
        return OccupiedEnergyComfortV2Config(
            self.energy_weight, self.comfort_weight, self.energy_scale_kwh_per_m2,
            self.comfort_scale_person_celsius_squared_hours_per_m2,
            self.comfort_schedule.default.minimum_c, self.comfort_schedule.default.maximum_c)

    @property
    def parameter_mapping(self) -> dict:
        return {name: getattr(self, name) for name in (
            'energy_weight', 'comfort_weight', 'energy_scale_kwh_per_m2',
            'comfort_scale_person_celsius_squared_hours_per_m2')} | {
                'comfort_schedule': self.comfort_schedule.to_mapping()}

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> 'ScheduledOccupiedRewardConfig':
        fields={'energy_weight','comfort_weight','energy_scale_kwh_per_m2',
                'comfort_scale_person_celsius_squared_hours_per_m2','comfort_schedule'}
        if not isinstance(value, Mapping) or set(value) != fields:
            raise ValueError('scheduled reward requires exactly its weights, scales and comfort_schedule')
        return cls(**{**value, 'comfort_schedule': ComfortSchedule.from_mapping(value['comfort_schedule'])})


def calculate_scheduled_occupied_reward(*, interval_controlled_hvac_site_energy_kwh: float,
        timestep_hours: float, zones: Iterable[ZoneComfortSample],
        config: ScheduledOccupiedRewardConfig, weekday: int, start_minute: float, month: int | None = None) -> dict:
    """Score physical interval samples and retain every band segment used.

    Temperature and occupant samples are held constant within an interval, as in
    v2. Schedule boundaries are integrated exactly, including midnight rollover.
    The caller must supply the interval-start local standard-time clock.
    """
    if not isinstance(config, ScheduledOccupiedRewardConfig):
        raise TypeError('config must be ScheduledOccupiedRewardConfig')
    samples=tuple(zones)
    base=calculate_hot_occupied_energy_comfort_v2(
        interval_controlled_hvac_site_energy_kwh=interval_controlled_hvac_site_energy_kwh,
        timestep_hours=timestep_hours,zones=samples,config=config.fixed_config())
    schedule=config.comfort_schedule
    schedule.validate_zones(tuple(sample.zone_id for sample in samples))
    segments=[]
    exposures=[]
    for sample in samples:
        for day,left,right,band in schedule.interval_segments(sample.zone_id,weekday=weekday,
                start_minute=start_minute,duration_minutes=60*timestep_hours,month=month):
            exposure=sample.occupant_count*band.violation_c(sample.temperature_c)**2*(right-left)/60
            exposures.append(exposure)
            segments.append({'zone_id':sample.zone_id,'weekday':day,'start_minute':left,
                'end_minute':right,'minimum_c':band.minimum_c,'maximum_c':band.maximum_c,
                'temperature_c':sample.temperature_c,'occupant_count':sample.occupant_count,
                'exposure_person_celsius_squared_hours':exposure})
    exposure=math.fsum(exposures)/base.controlled_floor_area_m2
    normalized=exposure/config.comfort_scale_person_celsius_squared_hours_per_m2
    weighted=config.comfort_weight*normalized
    return {**asdict(base),'schema_version':'hot-scheduled-reward-components/1.0.0',
        'comfort_exposure_person_celsius_squared_hours_per_m2':exposure,
        'normalized_comfort_penalty':normalized,'weighted_comfort_penalty':weighted,
        'reward':-(base.weighted_energy_penalty+weighted),'comfort_segments':segments}
