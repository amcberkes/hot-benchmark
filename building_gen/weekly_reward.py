"""Compile weekly, person-hour-normalized objectives into the interval scorer.

Calibration values are fixed before a policy runs. They must come from the
pinned exogenous occupancy schedule and training-only reference energy data.
This module does not estimate them from policy outcomes.
"""
from dataclasses import dataclass
import math

from building_gen.comfort import ComfortSchedule
from building_gen.scheduled_reward import ScheduledOccupiedRewardConfig
from building_gen.reward_foundation import _positive_finite


@dataclass(frozen=True)
class WeeklyRewardCalibration:
    occupied_person_hours: float
    reference_energy_kwh: float
    controlled_floor_area_m2: float
    timesteps_per_hour: int = 4

    def __post_init__(self):
        for name in ('occupied_person_hours', 'reference_energy_kwh', 'controlled_floor_area_m2'):
            object.__setattr__(self, name, _positive_finite(getattr(self, name), name=name))
        if type(self.timesteps_per_hour) is not int or self.timesteps_per_hour <= 0:
            raise ValueError('timesteps_per_hour must be a positive integer')

    @property
    def steps(self):
        return 168 * self.timesteps_per_hour

    def interval_config(self, *, comfort_schedule: ComfortSchedule,
                        energy_weight: float, comfort_scale_celsius_squared: float = 1.0):
        """Return an exactly equivalent, auditable scheduled-reward config.

        With H intervals, mean reward equals minus weekly occupied comfort MSE
        divided by its scale, minus energy_weight times E_week/E_reference.
        The legacy scorer's floor-area factors cancel algebraically; area is
        retained only to compile its existing explicit denominator convention.
        This preserves completed-interval scoring and schedule segmentation.
        """
        comfort_scale = _positive_finite(comfort_scale_celsius_squared,
                                        name='comfort_scale_celsius_squared')
        area_steps = self.controlled_floor_area_m2 * self.steps
        return ScheduledOccupiedRewardConfig(
            energy_weight=energy_weight, comfort_weight=1.0,
            energy_scale_kwh_per_m2=self.reference_energy_kwh / area_steps,
            comfort_scale_person_celsius_squared_hours_per_m2=(
                self.occupied_person_hours * comfort_scale / area_steps),
            comfort_schedule=comfort_schedule,
        )

    def verify_completed_week(self, *, steps: int, occupied_person_hours: float,
                              controlled_floor_area_m2: float):
        """Reject short episodes or calibration mismatches before accepting a run.

        This end-of-episode check cannot substitute for preflight calibration;
        callers must persist its outcome and invalidate mismatching training.
        """
        if type(steps) is not int or steps != self.steps:
            raise ValueError('completed week has the wrong number of intervals')
        for name, value, expected in (
            ('occupied_person_hours', occupied_person_hours, self.occupied_person_hours),
            ('controlled_floor_area_m2', controlled_floor_area_m2, self.controlled_floor_area_m2),
        ):
            value = _positive_finite(value, name=name)
            if not math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f'completed week differs from calibration: {name}')


WEEKLY_TRAINING_REWARD_ID = 'hot_weekly_person_hour_reference_v1'
LINEAR_WEEKLY_TRAINING_REWARD_ID = 'hot_weekly_person_hour_linear_reference_v1'


def validate_training_reward_spec(spec):
    from building_gen.benchmark.specs import AssetSpec, RewardSpec
    if not isinstance(spec, RewardSpec) or spec.reward_id not in (WEEKLY_TRAINING_REWARD_ID, LINEAR_WEEKLY_TRAINING_REWARD_ID):
        raise ValueError('unsupported weekly training reward spec')
    parameters = spec.parameter_mapping
    scale_key = 'comfort_scale_celsius' if spec.reward_id == LINEAR_WEEKLY_TRAINING_REWARD_ID else 'comfort_scale_celsius_squared'
    if set(parameters) != {'calibration', 'energy_weight', scale_key}:
        raise ValueError('weekly training reward requires calibration, energy_weight and comfort scale')
    asset = AssetSpec(**parameters['calibration'])
    if asset.sha256 is None:
        raise ValueError('weekly calibration must be checksum-pinned')
    weight = parameters['energy_weight']
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
        raise ValueError('energy_weight must be finite and nonnegative')
    _positive_finite(parameters[scale_key], name=scale_key)
    return asset


def calibration_environment_binding(task):
    """Physics and calendar identities; intentionally excludes observation ablations."""
    from building_gen.benchmark._canonical import canonical_value
    return canonical_value(dict(building=task.building.identity_payload(), weather=task.weather.identity_payload(),
        calendar_year=task.calendar_year, timesteps_per_hour=task.episode.timesteps_per_hour,
        action_profile=task.action_profile, component_energy_meters=task.component_energy_meters,
        warmup_preconditioning=task.warmup_preconditioning.identity_payload()))


class WeeklyTrainingReward:
    """Pinned training-only normalization with completed-week occupancy checks.

    The base environment's scalar reward remains available as diagnostic
    evidence. Evaluation and checkpoint selection use physical weekly metrics.
    This transform is part of the source task identity, including its catalog
    hash and weights; it is not a policy-outcome-dependent running normalizer.
    """
    def __init__(self, task, data_root):
        import hashlib
        import json
        from pathlib import Path
        self.linear = task.weekly_training_reward.reward_id == LINEAR_WEEKLY_TRAINING_REWARD_ID
        self.parameters = task.weekly_training_reward.parameter_mapping
        asset = validate_training_reward_spec(task.weekly_training_reward)
        root = Path(data_root).resolve()
        path = (root / asset.relative_path).resolve()
        path.relative_to(root)
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != asset.sha256:
            raise ValueError('weekly calibration checksum mismatch')
        catalog = json.loads(payload)
        if catalog.get('schema_version') != 'hot-weekly-reference-calibration/1.0.0':
            raise ValueError('unsupported weekly calibration schema')
        if catalog.get('environment_binding') != calibration_environment_binding(task):
            raise ValueError('weekly calibration physics/calendar binding mismatch')
        self.frequency = task.episode.timesteps_per_hour
        self.weeks = {}
        for row in catalog['weeks']:
            key = tuple(row[k] for k in ('start_month', 'start_day', 'end_month', 'end_day'))
            from building_gen.benchmark.specs import CalendarWindowSpec
            window = CalendarWindowSpec('calibration', *key)
            if window.duration_hours_for_calendar_year(task.calendar_year) != 168 or key in self.weeks:
                raise ValueError('invalid or duplicate weekly calibration window')
            if row['role'] not in ('training', 'validation', 'test'):
                raise ValueError('unknown weekly calibration role')
            calibration = WeeklyRewardCalibration(row['occupied_person_hours'], row['reference_energy_kwh'],
                row['controlled_floor_area_m2'], self.frequency)
            self.weeks[key] = (row['role'], calibration)
        if not self.weeks:
            raise ValueError('weekly calibration catalog is empty')
        self.calibration = None

    def begin(self, window):
        key = tuple(getattr(window, k) for k in ('start_month', 'start_day', 'end_month', 'end_day'))
        if key not in self.weeks or self.weeks[key][0] != 'training':
            raise ValueError('training week absent from training-only calibration coverage')
        self.calibration = self.weeks[key][1]
        self.people = []
        self.steps = 0

    def score(self, info, *, terminated, truncated):
        if self.calibration is None:
            raise RuntimeError('weekly reward must begin before scoring')
        components = info['reward_components']
        if components.get('schema_version') != 'hot-scheduled-reward-components/1.0.0':
            raise ValueError('weekly normalization requires completed-interval scheduled reward evidence')
        def finite(name, *, positive=False):
            value = components[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid physical reward component: {name}')
            if positive and value == 0:
                raise ValueError(f'zero physical reward component: {name}')
            return value
        area = finite('controlled_floor_area_m2', positive=True)
        duration = finite('timestep_hours', positive=True)
        if not math.isclose(area, self.calibration.controlled_floor_area_m2, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError('weekly calibration floor-area mismatch')
        if not math.isclose(duration, 1/self.frequency, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('weekly calibration interval-duration mismatch')
        people = finite('total_occupant_count') * duration
        exposure = finite('comfort_exposure_person_celsius_squared_hours_per_m2') * area
        energy = finite('interval_controlled_hvac_site_energy_kwh')
        self.steps += 1
        self.people.append(people)
        if self.steps > self.calibration.steps or (self.steps == self.calibration.steps and not (terminated or truncated)):
            raise ValueError('weekly reward episode length mismatch')
        if terminated or truncated:
            if type(info.get('energyplus_exit_code')) is not int or info['energyplus_exit_code'] != 0:
                raise ValueError('weekly reward requires successful simulator completion')
            self.calibration.verify_completed_week(steps=self.steps, occupied_person_hours=math.fsum(self.people),
                                                   controlled_floor_area_m2=area)
        if self.linear:
            exposure = occupied_linear_exposure(components['comfort_segments'])
        scale_key = 'comfort_scale_celsius' if self.linear else 'comfort_scale_celsius_squared'
        comfort = self.calibration.steps * exposure / (self.calibration.occupied_person_hours * self.parameters[scale_key])
        energy_term = self.calibration.steps * energy / self.calibration.reference_energy_kwh
        reward = -(comfort + self.parameters['energy_weight'] * energy_term)
        metrics = {}
        if 'comfort_segments' in components:
            physical = occupied_comfort_exposures(components['comfort_segments'])
            metrics = {k: self.calibration.steps*v/self.calibration.occupied_person_hours
                       for k,v in physical.items()}
        return reward, dict(schema_version='hot-weekly-training-reward/1.0.0', reward=reward,
            normalized_physical_comfort=metrics,
            normalized_comfort=comfort, normalized_energy=energy_term,
            comfort_measure='linear_degree_violation' if self.linear else 'squared_degree_violation',
            energy_weight=self.parameters['energy_weight'], completed_week_verified=bool(terminated or truncated))


def build_weekly_calibration(task, measurements, *, month_to_season, provenance):
    """Fit a model/weather/season reference from training weeks only.

    Measurements must come from authenticated fixed-controller trajectories.
    This builder handles within-month weekly windows, as used by the training
    calendar. It rejects crossing-month windows rather than guessing how their
    energy should be split across seasons. Validation/test energy never enters
    the fitted scales; their exogenous occupancy totals may be retained.
    """
    from building_gen.benchmark.specs import CalendarWindowSpec
    if set(month_to_season) != set(range(1, 13)) or any(
            not isinstance(v, str) or not v.strip() for v in month_to_season.values()):
        raise ValueError('declare a nonempty season for every month')
    measurements = tuple(measurements)
    if not measurements or not provenance:
        raise ValueError('calibration requires measurements and source provenance')
    seen, area, fit = set(), None, {}
    day_roles = {}
    checked = []
    for measurement in measurements:
        required = {'start_month','start_day','end_month','end_day','role',
                    'occupied_person_hours','controlled_floor_area_m2','energy_kwh'}
        if set(measurement) != required:
            raise ValueError('weekly measurement fields differ from the calibration schema')
        key = tuple(measurement[k] for k in ('start_month','start_day','end_month','end_day'))
        window = CalendarWindowSpec('calibration', *key)
        if key in seen or window.duration_hours_for_calendar_year(task.calendar_year) != 168:
            raise ValueError('duplicate or nonweekly calibration measurement')
        if window.start_month != window.end_month:
            raise ValueError('calibration currently requires within-month weeks')
        seen.add(key)
        role = measurement['role']
        if role not in ('training','validation','test'):
            raise ValueError('unknown calibration measurement role')
        for day in range(window.start_day, window.end_day + 1):
            calendar_day = (window.start_month, day)
            previous_role = day_roles.setdefault(calendar_day, role)
            if previous_role != role:
                raise ValueError('calibration calendar days overlap across split roles')
        for name in ('occupied_person_hours','controlled_floor_area_m2'):
            _positive_finite(measurement[name], name=name)
        energy = measurement['energy_kwh']
        if isinstance(energy, bool) or not isinstance(energy,(int,float)) or not math.isfinite(energy) or energy < 0:
            raise ValueError('reference energy must be finite and nonnegative')
        if area is None:
            area = measurement['controlled_floor_area_m2']
        if area != measurement['controlled_floor_area_m2']:
            raise ValueError('calibration measurements change controlled floor area')
        season = month_to_season[window.start_month]
        if role == 'training':
            fit.setdefault(season, []).append(energy)
        checked.append((dict(measurement),season))
    rates = {season: math.fsum(energies)/(len(energies)*area*168) for season,energies in fit.items()}
    for season, rate in rates.items():
        _positive_finite(rate, name=f'{season} reference power density')
    weeks=[]
    for measurement,season in checked:
        if season not in rates:
            raise ValueError('no training-only energy calibration for a requested season')
        weeks.append({k:v for k,v in measurement.items() if k != 'energy_kwh'} | {
            'reference_energy_kwh': rates[season]*area*168})
    return dict(schema_version='hot-weekly-reference-calibration/1.0.0',
        environment_binding=calibration_environment_binding(task), weeks=weeks,
        energy_fit=dict(method='total training reference energy / training floor-area-hours',
                        month_to_season={str(k):v for k,v in month_to_season.items()},
                        power_density_kw_per_m2=rates,
                        training_week_counts={k:len(v) for k,v in fit.items()}),
        provenance=provenance)


def occupied_comfort_exposures(segments):
    """Integrate occupied severity and duration, including exact band boundaries."""
    squared, linear, above_half = [], [], []
    for segment in segments:
        keys=('occupant_count','start_minute','end_minute','temperature_c','minimum_c','maximum_c')
        n,start,end,t,lo,hi=(segment[k] for k in keys)
        if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (n,start,end,t,lo,hi)) or n<0 or end<start or lo>hi:
            raise ValueError('invalid comfort segment')
        person_hours = n*(end-start)/60
        deviation = max(lo-t,0.,t-hi)
        squared.append(person_hours*deviation**2)
        linear.append(person_hours*deviation)
        above_half.append(person_hours*(deviation > .5))
    return dict(mse=math.fsum(squared), mean_degree=math.fsum(linear),
                exposure_above_half=math.fsum(above_half))


def occupied_linear_exposure(segments):
    """Integrate person-degree-hours over validated comfort schedule segments."""
    return occupied_comfort_exposures(segments)['mean_degree']
