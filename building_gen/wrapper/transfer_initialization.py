"""Authenticate a source PPO bundle and initialize target shared actor weights."""
import hashlib
import json
from pathlib import Path
import torch
from building_gen.wrapper.random_state import capture_random_state, restore_random_state


def initialize_transferred_actor(agent, directory, source_task, target_task, *, seed):
    from stable_baselines3 import PPO
    from building_gen.benchmark._canonical import canonical_value
    from building_gen.benchmark.environment import _expected_observation_names
    from building_gen.wrapper.rl_baselines import SB3_VERSION
    from building_gen.wrapper.shared_zone_policy import SharedZonePolicy, observation_layout
    from building_gen.wrapper.shared_zone_transfer import validate_shared_targets
    if not isinstance(agent, PPO) or agent.num_timesteps or agent._n_updates:
        raise ValueError('transfer initialization requires fresh target PPO')
    validate_shared_targets(source_task, [target_task])
    directory = Path(directory)
    raw = (directory / 'training.json').read_bytes()
    receipt = json.loads(raw)
    signature = {'task_id': source_task.task_id, 'definition': canonical_value(source_task.identity_payload())}
    if (receipt.get('status') != 'complete' or receipt.get('algorithm') != 'ppo'
            or receipt.get('seed') != seed or receipt.get('task_signature') != signature
            or receipt.get('implementation') != 'stable-baselines3'
            or receipt.get('implementation_version') != SB3_VERSION):
        raise ValueError('source transfer checkpoint identity differs')
    artifacts = receipt.get('artifacts', {})
    if set(artifacts) != {'model.zip', 'random_state.json'}:
        raise ValueError('transfer requires a standard complete PPO bundle')
    for name, expected in artifacts.items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'source transfer checksum differs: {name}')
    random_state = capture_random_state()
    try:
        source = PPO.load(directory / 'model.zip', device='cpu')
    finally:
        restore_random_state(random_state)
    if (source.seed != seed or source.num_timesteps != receipt['completed_timesteps']
            or source._n_updates != receipt['optimizer_updates']):
        raise ValueError('source transfer model counters differ')
    for policy, task in [(source.policy, source_task), (agent.policy, target_task)]:
        if not isinstance(policy, SharedZonePolicy):
            raise ValueError('transfer initialization requires shared-zone policies')
        local, global_ = observation_layout(list(_expected_observation_names(task)))
        if policy.local_indices != local or policy.global_indices != global_:
            raise ValueError('transfer observation layout differs')
        if policy.observation_space.shape != task.observation_space.shape or policy.action_space.shape != task.action_space.shape:
            raise ValueError('transfer space dimensions differ')
    source_state = source.policy.state_dict()
    target_state = agent.policy.state_dict()
    keys = {key for key in target_state if key.startswith('mlp_extractor.zone_actor.') or key == 'log_std'}
    if keys != {key for key in source_state if key.startswith('mlp_extractor.zone_actor.') or key == 'log_std'}:
        raise ValueError('transfer actor architecture differs')
    if any(target_state[k].shape != source_state[k].shape or not torch.isfinite(source_state[k]).all() for k in keys):
        raise ValueError('transfer actor tensor shape/value differs')
    if agent.policy.optimizer.state:
        raise ValueError('target optimizer must be fresh')
    if (directory / 'training.json').read_bytes() != raw:
        raise ValueError('source receipt changed during transfer preflight')
    for key in keys:
        target_state[key] = source_state[key].clone()
    agent.policy.load_state_dict(target_state, strict=True)
    return {'kind': 'source_ppo_shared_actor_and_distribution',
            'receipt_sha256': hashlib.sha256(raw).hexdigest(),
            'model_sha256': artifacts['model.zip'], 'source_task_id': source_task.task_id,
            'target_task_id': target_task.task_id, 'source_steps': source.num_timesteps,
            'seed': seed, 'copied_parameters': sorted(keys), 'critic': 'fresh',
            'optimizer': 'fresh', 'policy_variance': 'copied', 'target_transitions': 0}
