"""Training-only energy objective with a rollout-wise projected comfort penalty."""
from dataclasses import asdict, dataclass
import math


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return float(value)


@dataclass(frozen=True)
class ComfortPenaltyConfig:
    target_mse: float
    initial_penalty: float
    dual_learning_rate: float
    reference_sha256: str
    steps_per_week: int = 1008
    weeks_per_rollout: int = 12
    target_mean_degree: float | None = None
    target_exposure_above_half: float | None = None

    def __post_init__(self):
        _number(self.target_mse, 'target_mse', positive=True)
        extra = (self.target_mean_degree, self.target_exposure_above_half)
        if any(v is not None for v in extra):
            for value, name in zip(extra, ('target_mean_degree', 'target_exposure_above_half')):
                _number(value, name, positive=True)
            if self.target_exposure_above_half > 1:
                raise ValueError('exposure target must be a fraction')
        _number(self.initial_penalty, 'initial_penalty')
        _number(self.dual_learning_rate, 'dual_learning_rate')
        if (not isinstance(self.reference_sha256, str) or len(self.reference_sha256) != 64
                or any(c not in '0123456789abcdef' for c in self.reference_sha256)):
            raise ValueError('reference_sha256 must be a lowercase SHA256 digest')
        if type(self.steps_per_week) is not int or self.steps_per_week <= 0 or self.steps_per_week % 168:
            raise ValueError('steps_per_week must describe a complete weekly episode')
        if type(self.weeks_per_rollout) is not int or self.weeks_per_rollout != 12:
            raise ValueError('comfort update requires twelve monthly training weeks')


class ComfortPenalty:
    """Consumes authenticated weekly normalization evidence, not validation scores.

    The caller supplies the training month and invokes finish_rollout only after
    collection. This state alone does not certify simulator or split validity.
    A zero dual learning rate implements the matched reward-scale control.
    """
    def __init__(self, config):
        if not isinstance(config, ComfortPenaltyConfig):
            raise TypeError('ComfortPenaltyConfig required')
        self.config = config
        self.penalty = float(config.initial_penalty)
        self.targets = dict(mse=config.target_mse)
        if config.target_mean_degree is not None:
            self.targets.update(mean_degree=config.target_mean_degree,
                                exposure_above_half=config.target_exposure_above_half)
        self.penalties = {name: float(config.initial_penalty) for name in self.targets}
        self._interval_costs = []
        self._weeks = []
        self.history = []

    def score(self, evidence, *, training_month, episode_complete):
        if type(training_month) is not int or not 1 <= training_month <= 12:
            raise ValueError('training month must be 1 through 12')
        if type(episode_complete) is not bool:
            raise ValueError('episode_complete must be boolean')
        if evidence.get('schema_version') != 'hot-weekly-training-reward/1.0.0':
            raise ValueError('weekly training reward evidence required')
        if evidence.get('completed_week_verified') is not episode_complete:
            raise ValueError('completed week must be verified by the environment')
        if len(self._weeks) >= self.config.weeks_per_rollout:
            raise ValueError('finish the previous rollout before collecting more data')
        if evidence.get('comfort_measure', 'squared_degree_violation') != 'squared_degree_violation':
            raise ValueError('MSE constraint requires squared comfort evidence')
        comfort = _number(evidence['normalized_comfort'], 'normalized_comfort')
        energy = _number(evidence['normalized_energy'], 'normalized_energy')
        steps = len(self._interval_costs) + 1
        if episode_complete != (steps == self.config.steps_per_week):
            raise ValueError('partial or overlong weekly episode')
        if self._interval_costs and self._interval_costs[0][0] != training_month:
            raise ValueError('training month changed within a week')
        if episode_complete and training_month in {x['month'] for x in self._weeks}:
            raise ValueError('rollout must visit each training month once')
        costs = dict(mse=comfort)
        if len(self.targets) > 1:
            physical = evidence.get('normalized_physical_comfort', {})
            costs = {name: _number(physical[name], name) for name in self.targets}
            if not math.isclose(costs['mse'], comfort, rel_tol=1e-9, abs_tol=1e-12):
                raise ValueError('physical MSE differs from unit-scaled weekly evidence')
        reward = -(energy + math.fsum(self.penalties[name]*costs[name]/target
                                      for name,target in self.targets.items()))
        if not math.isfinite(reward):
            raise ValueError('nonfinite constrained reward')
        self._interval_costs.append((training_month, costs))
        if episode_complete:
            self._weeks.append(dict(month=training_month,
                comfort_mse=math.fsum(x[1]['mse'] for x in self._interval_costs)/self.config.steps_per_week,
                metrics={name: math.fsum(x[1][name] for x in self._interval_costs)/self.config.steps_per_week
                         for name in self.targets}))
            self._interval_costs = []
        return reward

    def finish_rollout(self):
        if self._interval_costs or len(self._weeks) != self.config.weeks_per_rollout:
            raise ValueError('penalty update requires twelve complete weeks')
        if {x['month'] for x in self._weeks} != set(range(1, 13)):
            raise ValueError('penalty update requires all twelve months')
        means = {name: math.fsum(x['metrics'][name] for x in self._weeks)/12
                 for name in self.targets}
        residuals = {name: means[name]/target - 1 for name,target in self.targets.items()}
        next_penalties = {name: max(0., self.penalties[name] + self.config.dual_learning_rate*residuals[name])
                          for name in self.targets}
        if not all(math.isfinite(v) for v in next_penalties.values()):
            raise ValueError('nonfinite penalty update')
        record = dict(update=len(self.history)+1, penalty_before=self.penalty,
            penalty_after=next_penalties['mse'], mean_training_mse=means['mse'],
            normalized_constraint_residual=residuals['mse'], weeks=self._weeks,
            penalties_before=dict(self.penalties), penalties_after=next_penalties,
            mean_training_metrics=means, normalized_constraint_residuals=residuals)
        self.penalties = next_penalties
        self.penalty = next_penalties['mse']
        self.history.append(record)
        self._weeks = []
        return record

    def evidence(self):
        from copy import deepcopy
        return dict(schema_version='hot-comfort-penalty/1.0', config=asdict(self.config),
            penalty=self.penalty, penalties=dict(self.penalties), targets=dict(self.targets), completed_updates=len(self.history),
            pending_week_steps=len(self._interval_costs), pending_weeks=deepcopy(self._weeks),
            history=deepcopy(self.history))


# Imported here so the arithmetic contract above stays independent of SB3.
import gymnasium as gym


class ComfortPenaltyEnvironment(gym.Wrapper):
    """Apply a fixed-within-rollout objective to a weekly training environment."""
    def __init__(self, env, config):
        super().__init__(env)
        self.comfort_penalty = ComfortPenalty(config)

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        calendar = info.get('training_calendar', {})
        window = calendar.get('window', {})
        if (window.get('duration_hours') != 168
                or window.get('start_month') != window.get('end_month')):
            raise ValueError('comfort penalty requires within-month training weeks')
        evidence = info.get('training_reward_components', {})
        reward = self.comfort_penalty.score(evidence,
            training_month=window['start_month'], episode_complete=bool(terminated or truncated))
        info = dict(info, unconstrained_training_reward=base_reward,
            comfort_penalty_interval=dict(penalty=self.comfort_penalty.penalty,
                penalties=dict(self.comfort_penalty.penalties), targets=dict(self.comfort_penalty.targets),
                target_mse=self.comfort_penalty.config.target_mse, reward=reward))
        return observation, reward, terminated, truncated, info

    def finish_comfort_rollout(self):
        return self.comfort_penalty.finish_rollout()

    def continuation_state(self):
        raise ValueError('comfort-penalty training continuation is not supported')


def reference_target(reference, *, catalog_sha256, training_windows,
                     metric='mse'):
    """Recompute the equal-month target and require complete training coverage."""
    fields = dict(mse=('comfort_mse_celsius_squared', 'equal_month_mean_mse'),
                  mean_degree=('comfort_mean_degree_celsius', 'equal_month_mean_degree'),
                  exposure_above_half=('comfort_exposure_above_half_fraction', 'equal_month_exposure_above_half'))
    if metric not in fields:
        raise ValueError('unknown comfort reference metric')
    row_field, aggregate_field = fields[metric]
    if (reference.get('schema_version') not in {'hot-training-comfort-reference/1.0', 'hot-training-comfort-reference/2.0'}
            or reference.get('catalog_sha256') != catalog_sha256):
        raise ValueError('comfort reference does not match the training calibration')
    by_month = {}
    seen = set()
    for row in reference['weeks']:
        key = tuple(row[k] for k in ('start_month','start_day','end_month','end_day'))
        if key in seen or key not in training_windows:
            raise ValueError('comfort reference contains duplicate or nontraining windows')
        seen.add(key)
        by_month.setdefault(key[0], []).append(_number(row[row_field], 'reference comfort'))
    if seen != set(training_windows) or set(by_month) != set(range(1,13)):
        raise ValueError('comfort reference must cover every training week and month')
    target = math.fsum(math.fsum(v)/len(v) for v in by_month.values())/12
    _number(target, 'reference target', positive=True)
    recorded = _number(reference[aggregate_field], 'recorded reference target', positive=True)
    if not math.isclose(target, recorded, rel_tol=1e-12, abs_tol=0):
        raise ValueError('recorded comfort target differs from training measurements')
    return target


def load_comfort_penalty_config(path, task, data_root):
    """Resolve a pinned training reference before creating the simulator."""
    import hashlib,json
    from pathlib import Path
    from building_gen.weekly_reward import WeeklyTrainingReward
    spec = json.loads(Path(path).read_text())
    if set(spec) != {'schema_version','reference','initial_penalty','dual_learning_rate'} or spec['schema_version'] not in {'hot-comfort-constraint/1.0', 'hot-comfort-constraint/2.0'}:
        raise ValueError('invalid comfort constraint configuration')
    if task.weekly_training_reward is None or task.episode.episode_length_hours != 168:
        raise ValueError('comfort constraint requires weekly calibrated rewards')
    params = task.weekly_training_reward.parameter_mapping
    if params['comfort_scale_celsius_squared'] != 1.0:
        raise ValueError('comfort constraint requires unit physical comfort scale')
    calibrated = WeeklyTrainingReward(task,data_root)
    asset = spec['reference']
    if set(asset) != {'relative_path','sha256'}:
        raise ValueError('comfort reference requires relative_path and sha256')
    root = Path(data_root).resolve()
    reference_path = (root/asset['relative_path']).resolve()
    reference_path.relative_to(root)
    raw = reference_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != asset['sha256']:
        raise ValueError('comfort reference checksum mismatch')
    reference = json.loads(raw)
    training_windows = {key for key,(role,_) in calibrated.weeks.items() if role=='training'}
    target = reference_target(reference,catalog_sha256=params['calibration']['sha256'],
                              training_windows=training_windows)
    extra = {}
    if spec['schema_version'] == 'hot-comfort-constraint/2.0':
        if reference.get('schema_version') != 'hot-training-comfort-reference/2.0':
            raise ValueError('multi-metric constraint requires a multi-metric reference')
        extra = {f'target_{metric}': reference_target(reference,
                    catalog_sha256=params['calibration']['sha256'],
                    training_windows=training_windows, metric=metric)
                 for metric in ('mean_degree', 'exposure_above_half')}
    return ComfortPenaltyConfig(target,spec['initial_penalty'],spec['dual_learning_rate'],
        asset['sha256'],steps_per_week=task.episode.max_steps, **extra)
