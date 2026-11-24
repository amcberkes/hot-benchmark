"""Thin, versioned Stable-Baselines3 adapters for HOT Gym environments.

These adapters are separate from the authenticated scientific PPO pipeline.
Callers own environment cleanup. No additional observation/reward normalization is
applied by default: the HOT environment defines that contract. PPO optionally
uses an explicit fixed training-only reward scale, recorded in its receipt.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any, Mapping

SB3_VERSION = '2.7.1'
ALGORITHMS = ('ppo', 'a2c', 'sac', 'td3', 'ddpg', 'dqn')
OFF_POLICY = frozenset(('sac', 'td3', 'ddpg', 'dqn'))


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _implementation(algorithm: str):
    import stable_baselines3 as sb3
    if sb3.__version__ != SB3_VERSION:
        raise ValueError(f'expected stable-baselines3 {SB3_VERSION}, got {sb3.__version__}')
    if algorithm not in ALGORITHMS:
        raise ValueError(f'unknown RL algorithm {algorithm!r}; choose {ALGORITHMS}')
    return getattr(sb3, algorithm.upper())


def _parameters(algorithm: str, parameters: Mapping[str, Any] | None) -> dict[str, Any]:
    values = json.loads(_json(dict(parameters or {})))
    if algorithm in {'td3', 'ddpg'}:
        values.setdefault('action_noise_std', 0.1)
        std = values['action_noise_std']
        if isinstance(std, bool) or not isinstance(std, (int, float)) or not math.isfinite(std) or std < 0:
            raise ValueError('action_noise_std must be finite and nonnegative')
    elif 'action_noise_std' in values:
        raise ValueError('action_noise_std is supported only for TD3 and DDPG')
    if 'energy_only_training' in values and (algorithm != 'ppo' or type(values['energy_only_training']) is not bool):
        raise ValueError('energy_only_training requires PPO and a boolean')
    if 'training_energy_weight' in values:
        from building_gen.wrapper.training_energy_weight import validate_energy_weight
        validate_energy_weight(values['training_energy_weight'])
        if algorithm != 'ppo' or values.get('energy_only_training', False):
            raise ValueError('training_energy_weight requires PPO without energy-only training')
    if 'save_initial_checkpoint' in values and (algorithm != 'ppo' or type(values['save_initial_checkpoint']) is not bool):
        raise ValueError('save_initial_checkpoint requires PPO and a boolean')
    if 'record_rollout_diagnostics' in values and (algorithm != 'ppo' or type(values['record_rollout_diagnostics']) is not bool):
        raise ValueError('record_rollout_diagnostics requires PPO and a boolean')
    if 'initial_action_mean' in values:
        value = values['initial_action_mean']
        if algorithm != 'ppo':
            raise ValueError('initial_action_mean is supported only for PPO')
        entries = value if isinstance(value, list) else [value]
        if not entries or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in entries):
            raise ValueError('initial_action_mean must be a finite scalar or nonempty flat list')
    if 'anchor_occupied_indices' in values and (algorithm != 'ppo' or 'anchor_coefficient' not in values):
        raise ValueError('occupancy-conditioned anchor requires anchored PPO')
    if 'anchor_coefficient' in values:
        coefficient = values['anchor_coefficient']
        if algorithm != 'ppo' or isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)) or not math.isfinite(coefficient) or coefficient <= 0:
            raise ValueError('anchor_coefficient requires PPO and a finite positive value')
    if 'training_reward_scale' in values:
        scale = values['training_reward_scale']
        if algorithm != 'ppo' or isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('training_reward_scale requires PPO and a finite positive value')
    if 'critic_learning_rate_multiplier' in values:
        ratio=values['critic_learning_rate_multiplier']
        if (algorithm != 'ppo' or values.get('policy_architecture') != 'shared_zone_v1'
                or type(ratio) not in (int,float) or not math.isfinite(ratio) or ratio <= 0):
            raise ValueError('critic_learning_rate_multiplier requires shared-zone PPO and a finite positive ratio')
    return values


def create_agent(algorithm: str, environment: Any, *, seed: int, parameters: Mapping[str, Any] | None = None):
    """Create one continuous-control baseline, or DQN on an explicit Discrete task."""
    import gymnasium as gym
    import numpy as np
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not isinstance(environment, gym.Env):
        raise TypeError('adapter requires one Gymnasium environment, not a vector wrapper')
    observation = environment.observation_space
    action = environment.action_space
    if not isinstance(observation, gym.spaces.Box) or len(observation.shape) != 1:
        raise ValueError('HOT baseline observations must be a flat Box')
    if algorithm == 'dqn':
        if not isinstance(action, gym.spaces.Discrete):
            raise ValueError('DQN requires an explicitly defined Discrete action task')
    elif not isinstance(action, gym.spaces.Box) or len(action.shape) != 1:
        raise ValueError('continuous HOT baselines require a flat Box action space')
    elif not (np.all(np.isfinite(action.low)) and np.all(np.isfinite(action.high)) and np.all(action.high > action.low)):
        raise ValueError('continuous actions require finite, increasing bounds')
    kwargs = _parameters(algorithm, parameters)
    kwargs.pop('record_rollout_diagnostics', None)
    kwargs.pop('save_initial_checkpoint', None)
    architecture = kwargs.pop('policy_architecture', None)
    critic_multiplier = kwargs.pop('critic_learning_rate_multiplier', None)
    if 'critic_learning_rate_multiplier' in kwargs.get('policy_kwargs', {}):
        raise ValueError('Set critic_learning_rate_multiplier at the top parameter level')
    policy_class = 'MlpPolicy'
    if architecture in {'scheduled_residual_v1', 'scheduled_residual_sde_v1', 'scheduled_coordinated_sde_v1'}:
        if algorithm != 'ppo' or any(k in kwargs for k in ('initial_action_mean','anchor_coefficient')):
            raise ValueError('scheduled residual requires fresh plain PPO')
        if environment.get_wrapper_attr('action_transform_id') != 'bounded_thermostat_12_32_v1':
            raise ValueError('scheduled residual requires the declared wide thermostat transform')
        from building_gen.wrapper.scheduled_residual_policy import ScheduledResidualPolicy, ScheduledResidualSDEPolicy, layout
        from building_gen.wrapper.coordinated_sde_policy import CoordinatedScheduledPolicy
        bounds, clock = layout(list(environment.get_wrapper_attr('observation_names')))
        policy_class = ScheduledResidualSDEPolicy if architecture == 'scheduled_residual_sde_v1' else ScheduledResidualPolicy
        if architecture == 'scheduled_coordinated_sde_v1': policy_class = CoordinatedScheduledPolicy
        kwargs['policy_kwargs'] = dict(kwargs.get('policy_kwargs',{}), bounds_indices=bounds, clock_indices=clock)
    elif architecture is not None:
        if algorithm != 'ppo' or architecture != 'shared_zone_v1' or any(k in kwargs for k in ('initial_action_mean','anchor_coefficient')):
            raise ValueError('shared zone architecture requires plain fresh PPO without action-mean overrides')
        from building_gen.wrapper.shared_zone_policy import SharedZonePolicy, observation_layout
        local, global_ = observation_layout(list(environment.get_wrapper_attr('observation_names')))
        policy_class = SharedZonePolicy
        kwargs['policy_kwargs'] = dict(kwargs.get('policy_kwargs', {}),local_indices=local,global_indices=global_)
        if critic_multiplier is not None:
            kwargs['policy_kwargs']['critic_learning_rate_multiplier'] = critic_multiplier
    reserved = {'policy', 'env', 'seed', 'device', 'tensorboard_log', '_init_setup_model', 'action_noise'} & kwargs.keys()
    if reserved:
        raise ValueError(f'reserved adapter parameters: {sorted(reserved)}')
    if algorithm in {"td3", "ddpg"}:
        from stable_baselines3.common.noise import NormalActionNoise
        std = kwargs.pop("action_noise_std")
        kwargs["action_noise"] = NormalActionNoise(np.zeros(action.shape), np.full(action.shape, std))
    initial_mean = kwargs.pop('initial_action_mean', None)
    if initial_mean is not None:
        initial_mean = np.asarray(initial_mean, dtype=float)
        if initial_mean.ndim != 0 and initial_mean.shape != action.shape:
            raise ValueError('initial_action_mean vector must match the action shape')
        initial_mean = np.broadcast_to(initial_mean, action.shape).copy()
        if not np.all((action.low < initial_mean) & (initial_mean < action.high)):
            raise ValueError('initial_action_mean must lie strictly inside every action bound')
        if kwargs.get('use_sde', False):
            raise ValueError('initial_action_mean requires ordinary Gaussian PPO')
    if kwargs.pop('energy_only_training', False):
        from building_gen.wrapper.energy_only_reward import EnergyOnlyReward
        environment = EnergyOnlyReward(environment)
    energy_weight = kwargs.pop('training_energy_weight', None)
    if energy_weight is not None:
        from building_gen.wrapper.training_energy_weight import TrainingEnergyWeight
        environment = TrainingEnergyWeight(environment, energy_weight)
    scale = kwargs.pop('training_reward_scale', None)
    if scale is not None:
        from building_gen.wrapper.training_reward_scale import TrainingRewardScale
        environment = TrainingRewardScale(environment, scale)
    implementation = _implementation(algorithm)
    if critic_multiplier is not None and critic_multiplier != 1:
        from building_gen.wrapper.split_rate_ppo import SplitRatePPO
        implementation = SplitRatePPO
    if 'anchor_coefficient' in kwargs:
        from building_gen.wrapper.anchored_ppo import AnchoredPPO
        implementation = AnchoredPPO
    agent = implementation(policy_class, environment, seed=seed, device='cpu', **kwargs)
    if initial_mean is not None:
        # Change only initial Gaussian location, preserving state-dependent weights,
        # physical action bounds and SB3 likelihood/clipping behavior.
        import torch
        with torch.no_grad():
            agent.policy.action_net.bias.copy_(torch.as_tensor(initial_mean, dtype=agent.policy.action_net.bias.dtype, device=agent.policy.action_net.bias.device))
    return agent


def train_agent(algorithm: str, environment: Any, *, seed: int, total_timesteps: int,
                task_signature: Mapping[str, Any], output_dir: Path,
                parameters: Mapping[str, Any] | None = None, resume_from: Path | None = None,
                checkpoint_interval: int | None = None, comfort_penalty_config=None,
                actor_initialization: Path | None = None, actor_reward_tasks=None,
                transfer_initialization=None) -> Path:
    """Train and save model plus replay state, recording actual transitions.

    On-policy algorithms can round the requested budget up to a complete rollout.
    The receipt records both counts. This is not a scientific budget certificate.
    """
    if type(total_timesteps) is not int or total_timesteps <= 0:
        raise ValueError('total_timesteps must be a positive integer')
    if checkpoint_interval is not None and (type(checkpoint_interval) is not int or checkpoint_interval <= 0):
        raise ValueError('checkpoint_interval must be a positive integer')
    if not task_signature:
        raise ValueError('task_signature must identify observation, action and reward semantics')
    signature = json.loads(_json(dict(task_signature)))
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f'refusing to overwrite run: {destination}')
    if (parameters or {}).get('energy_only_training') and (comfort_penalty_config is not None or resume_from is not None):
        raise ValueError('energy-only training requires fresh training without another comfort wrapper')
    if 'training_energy_weight' in (parameters or {}) and (comfort_penalty_config is not None or resume_from is not None):
        raise ValueError('training energy weighting requires fresh PPO without another comfort wrapper')
    penalty_environment = None
    if comfort_penalty_config is not None:
        from building_gen.wrapper.comfort_penalty import ComfortPenaltyConfig, ComfortPenaltyEnvironment
        if not isinstance(comfort_penalty_config, ComfortPenaltyConfig) or algorithm != 'ppo':
            raise ValueError('comfort penalty requires PPO and a ComfortPenaltyConfig')
        if resume_from is not None:
            raise ValueError('comfort-penalty resume is not supported')
        rollout_steps = comfort_penalty_config.steps_per_week * 12
        if (parameters or {}).get('n_steps', 2048) != rollout_steps or total_timesteps % rollout_steps:
            raise ValueError('comfort penalty requires complete twelve-week PPO rollouts')
        if checkpoint_interval is not None and checkpoint_interval % rollout_steps:
            raise ValueError('comfort checkpoints must align with complete rollouts')
        penalty_environment = ComfortPenaltyEnvironment(environment, comfort_penalty_config)
        environment = penalty_environment
    if (parameters or {}).get('save_initial_checkpoint') and (checkpoint_interval is None or resume_from is not None):
        raise ValueError('initial checkpoint requires fresh training with checkpoint_interval')
    if actor_reward_tasks is not None and actor_initialization is None:
        raise ValueError('reward compatibility requires actor initialization')
    if actor_initialization is not None and (algorithm != 'ppo' or resume_from is not None):
        raise ValueError('actor initialization requires fresh PPO, not resume')
    if transfer_initialization is not None and (algorithm != 'ppo' or resume_from is not None or actor_initialization is not None):
        raise ValueError('transfer initialization requires fresh PPO without another initialization')
    initialization_receipt = None
    parent = None
    parent_bytes = None
    resolved_parameters = _parameters(algorithm, parameters)
    if resume_from is not None and 'critic_learning_rate_multiplier' in resolved_parameters:
        raise ValueError('Split-rate PPO requires fresh training; resume is not supported by this adapter')
    if 'training_reward_scale' in resolved_parameters and resume_from is not None:
        raise ValueError('training reward scaling requires fresh PPO')
    # create_agent applies the fixed scale outside ComfortPenaltyEnvironment.
    # Dual updates therefore retain physical comfort units, and SB3 adds the
    # terminal value in scaled return units after scaling the interval reward.
    if 'anchor_coefficient' in resolved_parameters and (actor_initialization is None or resume_from is not None):
        raise ValueError('anchored PPO requires fresh actor initialization; resume is unsupported')
    if resume_from is None:
        agent = create_agent(algorithm, environment, seed=seed, parameters=parameters)
    else:
        parent_bytes = (Path(resume_from)/'training.json').read_bytes()
        parent = json.loads(parent_bytes)
        if 'training_energy_weight' in parent.get('parameters', {}):
            raise ValueError('training energy weighting resume is unsupported')
        if 'critic_learning_rate_multiplier' in parent.get('parameters', {}):
            raise ValueError('Split-rate PPO resume is unsupported by this adapter')
        if 'training_reward_scale' in parent.get('parameters', {}):
            raise ValueError('training reward scaling resume is unsupported')
        if 'anchor_coefficient' in parent.get('parameters', {}):
            raise ValueError('anchored PPO resume is unsupported')
        if parent.get('comfort_constraint') is not None:
            raise ValueError('comfort-penalty resume is not supported')
        if parent.get('algorithm') != algorithm or parent.get('seed') != seed:
            raise ValueError('resume algorithm and seed must match the parent checkpoint')
        if parameters is not None and resolved_parameters != parent['parameters']:
            raise ValueError('resume cannot change algorithm parameters')
        resolved_parameters = parent['parameters']
        expected_dependencies = {name: importlib.metadata.version(name) for name in ('torch', 'gymnasium', 'numpy')}
        if parent.get('dependencies') != expected_dependencies:
            raise ValueError('resume dependency versions must match the parent checkpoint')
        continuation = parent.get('continuation', {})
        if continuation.get('available') is not True or continuation.get('mode') != 'fresh_episode_boundary':
            raise ValueError('parent checkpoint has no supported episode-boundary continuation')
        current = getattr(environment, 'continuation_state', None)
        if not callable(current) or _json(current()) != _json(continuation['calendar']):
            raise ValueError('resume environment calendar does not match the saved boundary')
        agent = load_agent(resume_from, environment, task_signature=signature, restore_random=True)
        if agent.num_timesteps != parent['completed_timesteps'] or agent._n_updates != parent['optimizer_updates']:
            raise ValueError('resume counters disagree with parent checkpoint')
        if (Path(resume_from)/'training.json').read_bytes() != parent_bytes:
            raise ValueError('parent receipt changed during resume preflight')
    if actor_initialization is not None:
        from building_gen.wrapper.actor_initialization import initialize_actor
        initialization_receipt = initialize_actor(agent, actor_initialization, task_id=signature['task_id'], seed=seed, reward_tasks=actor_reward_tasks)
    transfer_receipt = None
    if transfer_initialization is not None:
        from building_gen.wrapper.transfer_initialization import initialize_transferred_actor
        directory, source_task, target_task = transfer_initialization
        if signature['task_id'] != target_task.task_id:
            raise ValueError('transfer target signature differs')
        transfer_receipt = initialize_transferred_actor(agent, directory, source_task, target_task, seed=seed)
    if 'anchor_coefficient' in resolved_parameters:
        agent.freeze_anchor()
    initial_timesteps = agent.num_timesteps
    destination.mkdir(parents=True, exist_ok=False)
    status = {'schema_version': 'hot-sb3-training/1.1.0', 'status': 'running',
              'implementation': 'stable-baselines3', 'implementation_version': SB3_VERSION,
              'algorithm': algorithm, 'seed': seed, 'device': 'cpu',
              'dependencies': {name: importlib.metadata.version(name) for name in ('torch', 'gymnasium', 'numpy')},
              'requested_timesteps': total_timesteps, 'parameters': resolved_parameters,
              'initial_timesteps': initial_timesteps,
              'task_signature': signature, 'artifacts': {}}
    if transfer_receipt is not None:
        status['transfer_initialization'] = transfer_receipt
    if initialization_receipt is not None:
        status['actor_initialization'] = initialization_receipt
    if parent is not None:
        status['parent_checkpoint'] = {'training_receipt_sha256': hashlib.sha256(parent_bytes).hexdigest(),
                                       'model_sha256': parent['artifacts']['model.zip'],
                                       'continuation_mode': 'fresh_episode_boundary',
                                       'exact_simulator_continuation': False}
    if penalty_environment is not None:
        from dataclasses import asdict
        status['comfort_constraint'] = asdict(comfort_penalty_config)

    def save_penalty(directory):
        (directory/'comfort_penalty.json').write_text(
            _json(penalty_environment.comfort_penalty.evidence())+'\n')

    receipt = destination / 'training.json'

    def record():
        temporary = destination / '.training.tmp'
        temporary.write_text(_json(status) + '\n', encoding='utf-8')
        temporary.replace(receipt)

    record()
    from stable_baselines3.common.logger import configure
    logger = configure(str(destination), ["csv"])
    agent.set_logger(logger)
    callback = None
    if checkpoint_interval is not None:
        from stable_baselines3.common.callbacks import BaseCallback
        from building_gen.wrapper.random_state import capture_random_state

        status['checkpoint_interval'] = checkpoint_interval
        status['validation_checkpoints'] = []

        class SnapshotCallback(BaseCallback):
            """Snapshot between rollout collection cycles, after optimizer updates."""
            def __init__(self):
                super().__init__()
                self.next_step = initial_timesteps + checkpoint_interval
                self.last_saved = None

            def _on_step(self):
                return True

            def _save(self, initial=False):
                step = int(self.model.num_timesteps)
                if (step <= initial_timesteps and not initial) or step == self.last_saved:
                    return
                directory = destination / 'checkpoints' / f'step-{step}'
                directory.mkdir(parents=True, exist_ok=False)
                self.model.save(directory / 'model.zip')
                names = ['model.zip', 'random_state.json']
                if penalty_environment is not None:
                    save_penalty(directory)
                    names.append('comfort_penalty.json')
                (directory / 'random_state.json').write_text(_json(capture_random_state()) + '\n')
                if algorithm in OFF_POLICY:
                    self.model.save_replay_buffer(directory / 'replay_buffer.pkl')
                    names.append('replay_buffer.pkl')
                # Snapshot diagnostics describe this point, not a copied training curve.
                (directory / 'progress.csv').write_text(f'time/total_timesteps,train/n_updates\n{step},{self.model._n_updates}\n')
                snapshot = {key: value for key, value in status.items()
                            if key not in {'validation_checkpoints', 'checkpoint_interval'}}
                snapshot.update(status='complete', checkpoint_kind='initialization_snapshot' if initial else 'periodic_evaluation_snapshot',
                    completed_timesteps=step, optimizer_updates=int(self.model._n_updates),
                    requested_timesteps=step-initial_timesteps,
                    continuation={'available': False, 'reason': 'evaluation snapshot; simulator continuation not captured'},
                    artifacts={name: hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names},
                    diagnostics={'progress.csv': hashlib.sha256((directory/'progress.csv').read_bytes()).hexdigest()})
                (directory/'training.json').write_text(_json(snapshot) + '\n')
                status['validation_checkpoints'].append({
                    'training_step': step, 'directory': str(directory.relative_to(destination)),
                    'training_receipt_sha256': hashlib.sha256((directory/'training.json').read_bytes()).hexdigest()})
                record()
                self.last_saved = step

            def _on_training_start(self):
                if resolved_parameters.get('save_initial_checkpoint', False):
                    self._save(initial=True)

            def _on_rollout_start(self):
                if self.model.num_timesteps >= self.next_step and self.model._n_updates > 0:
                    self._save()
                    while self.next_step <= self.model.num_timesteps:
                        self.next_step += checkpoint_interval

            def _on_training_end(self):
                self._save()

        callback = SnapshotCallback()
    if penalty_environment is not None:
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList
        class ComfortCallback(BaseCallback):
            def _on_step(self):
                return True
            def _on_rollout_end(self):
                update = penalty_environment.finish_comfort_rollout()
                self.logger.record('constraint/penalty', update['penalty_after'])
                self.logger.record('constraint/training_mse', update['mean_training_mse'])
                self.logger.record('constraint/residual', update['normalized_constraint_residual'])
                for name, value in update['normalized_constraint_residuals'].items():
                    self.logger.record(f'constraint/{name}_residual', value)
                    self.logger.record(f'constraint/{name}_penalty', update['penalties_after'][name])
                save_penalty(destination)
        callback = CallbackList([callback, ComfortCallback()]) if callback else ComfortCallback()
    if resolved_parameters.get('record_rollout_diagnostics', False):
        from stable_baselines3.common.callbacks import CallbackList
        from building_gen.wrapper.rollout_diagnostics import RolloutDiagnosticsCallback
        diagnostic_callback = RolloutDiagnosticsCallback(destination/'rollout_diagnostics.json')
        callback = CallbackList([callback, diagnostic_callback]) if callback else diagnostic_callback
    try:
        agent.learn(total_timesteps=total_timesteps, log_interval=1, reset_num_timesteps=resume_from is None,
                    callback=callback)
        logger.record("time/total_timesteps", agent.num_timesteps)
        logger.dump(step=agent.num_timesteps)
        logger.close()
        agent.save(destination / 'model.zip')
        from building_gen.wrapper.random_state import capture_random_state
        (destination/'random_state.json').write_text(_json(capture_random_state())+'\n', encoding='utf-8')
        artifacts = ['model.zip', 'random_state.json']
        if penalty_environment is not None:
            save_penalty(destination)
            artifacts.append('comfort_penalty.json')
        if algorithm in OFF_POLICY:
            agent.save_replay_buffer(destination / 'replay_buffer.pkl')
            artifacts.append('replay_buffer.pkl')
        continuation = getattr(environment, 'continuation_state', None)
        if callable(continuation):
            try:
                calendar_state = continuation()
            except ValueError as error:
                status['continuation'] = {'available': False, 'reason': str(error)}
            else:
                status['continuation'] = {'available': True, 'mode': 'fresh_episode_boundary',
                                          'calendar': calendar_state,
                                          'exact_simulator_continuation': False}
        else:
            status['continuation'] = {'available': False, 'reason': 'environment has no calendar continuation contract'}
        status.update(status='complete', completed_timesteps=agent.num_timesteps,
                      optimizer_updates=agent._n_updates,
                      diagnostics={'progress.csv': hashlib.sha256((destination/'progress.csv').read_bytes()).hexdigest()},
                      artifacts={name: hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in artifacts})
        if resolved_parameters.get('record_rollout_diagnostics', False):
            status['diagnostics']['rollout_diagnostics.json'] = hashlib.sha256((destination/'rollout_diagnostics.json').read_bytes()).hexdigest()
        record()
        return receipt
    except BaseException as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        record()
        raise
    finally:
        logger.close()


def load_agent(run_dir: Path, environment: Any, *, task_signature: Mapping[str, Any], restore_random: bool = False):
    """Load a trusted local bundle and reject semantic mismatches or corruption.

    Replay state is restored for off-policy agents. New bundles optionally restore
    Python/NumPy/CPU Torch generators. Simulator and environment RNG continuation
    is not implemented; this does not claim bitwise training continuation.
    """
    directory = Path(run_dir)
    status = json.loads((directory/'training.json').read_text(encoding='utf-8'))
    if status.get('schema_version') not in {'hot-sb3-training/1.0.0', 'hot-sb3-training/1.1.0'} or status.get('status') != 'complete':
        raise ValueError('baseline training bundle is not complete')
    if status.get('implementation_version') != SB3_VERSION or status.get('implementation') != 'stable-baselines3':
        raise ValueError('baseline implementation does not match')
    if _json(status.get('task_signature')) != _json(dict(task_signature)):
        raise ValueError('task observation/action/reward semantics do not match checkpoint')
    algorithm = status['algorithm']
    required = {'model.zip', 'replay_buffer.pkl'} if algorithm in OFF_POLICY else {'model.zip'}
    if status['schema_version'] == 'hot-sb3-training/1.1.0':
        required.add('random_state.json')
    elif restore_random:
        raise ValueError('legacy checkpoint does not contain random generator state')
    if status.get('comfort_constraint') is not None:
        if algorithm != 'ppo' or restore_random:
            raise ValueError('comfort-penalty checkpoints support PPO evaluation only')
        required.add('comfort_penalty.json')
    if set(status['artifacts']) != required:
        raise ValueError('checkpoint artifact set is incomplete or unexpected')
    for name, digest in status['artifacts'].items():
        if hashlib.sha256((directory/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'checkpoint artifact checksum mismatch: {name}')
    if status.get('comfort_constraint') is not None:
        state = json.loads((directory/'comfort_penalty.json').read_text())
        if state.get('config') != status['comfort_constraint']:
            raise ValueError('comfort penalty configuration differs from checkpoint')
    agent = _implementation(algorithm).load(directory/'model.zip', env=environment, device='cpu')
    if algorithm in OFF_POLICY:
        agent.load_replay_buffer(directory/'replay_buffer.pkl')
    if restore_random:
        from building_gen.wrapper.random_state import restore_random_state
        restore_random_state(json.loads((directory/'random_state.json').read_text(encoding='utf-8')))
    return agent
