"""Initialize a fresh PPO actor from an authenticated supervised schedule fit."""
import hashlib
import json
from pathlib import Path
from building_gen.wrapper.random_state import capture_random_state, restore_random_state


def reward_only_compatibility(source_task, target_task):
    """Allow only a declared energy-weight change in the weekly training reward."""
    import copy
    source = copy.deepcopy(source_task.identity_payload())
    target = copy.deepcopy(target_task.identity_payload())
    from building_gen.weekly_reward import WEEKLY_TRAINING_REWARD_ID, LINEAR_WEEKLY_TRAINING_REWARD_ID
    sr=source.get('weekly_training_reward',{});tr=target.get('weekly_training_reward',{})
    if sr.get('reward_id')==WEEKLY_TRAINING_REWARD_ID and tr.get('reward_id')==LINEAR_WEEKLY_TRAINING_REWARD_ID:
        original_target=copy.deepcopy(tr)
        if sr['parameters'].get('comfort_scale_celsius_squared')!=1. or tr['parameters'].get('comfort_scale_celsius')!=1.:
            raise ValueError('linear initialization comparison requires declared unit comfort scales')
        tr['reward_id']=WEEKLY_TRAINING_REWARD_ID
        tr['parameters']['comfort_scale_celsius_squared']=tr['parameters'].pop('comfort_scale_celsius')
        if source!=target:
            raise ValueError('linear initialization permits only the declared comfort-measure change')
        return dict(kind='weekly_squared_to_linear_comfort',source_task_id=source_task.task_id,target_task_id=target_task.task_id,source_reward=sr,target_reward=original_target)
    try:
        source_weight = source['weekly_training_reward']['parameters'].pop('energy_weight')
        target_weight = target['weekly_training_reward']['parameters'].pop('energy_weight')
    except KeyError as error:
        raise ValueError('weekly energy-weight specifications required') from error
    import math
    for weight in (source_weight, target_weight):
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError('finite nonnegative energy weights required')
    if source != target:
        raise ValueError('actor initialization permits only an energy-weight change')
    return dict(kind='weekly_energy_weight_only', source_task_id=source_task.task_id,
                target_task_id=target_task.task_id, source_energy_weight=source_weight,
                target_energy_weight=target_weight)


def initialize_actor(agent, directory, *, task_id, seed, reward_tasks=None):
    from stable_baselines3 import PPO
    if not isinstance(agent, PPO) or agent.num_timesteps != 0 or agent._n_updates != 0:
        raise ValueError('actor initialization requires fresh PPO')
    directory = Path(directory)
    raw = (directory/'imitation.json').read_bytes()
    receipt = json.loads(raw)
    model = directory/'actor_initialization.zip'
    compatibility = None
    source_task_id = task_id
    if reward_tasks is not None:
        source_task, target_task = reward_tasks
        if target_task.task_id != task_id:
            raise ValueError('target initialization task mismatch')
        compatibility = reward_only_compatibility(source_task, target_task)
        source_task_id = source_task.task_id
    if receipt['status'] != 'complete' or receipt['seed'] != seed or receipt['data']['task_id'] != source_task_id:
        raise ValueError('actor initialization identity mismatch')
    digest = hashlib.sha256(model.read_bytes()).hexdigest()
    if digest != receipt['model_sha256']:
        raise ValueError('actor initialization checksum mismatch')
    state = capture_random_state()
    try:
        source = PPO.load(model, device='cpu')
    finally:
        restore_random_state(state)
    if source.num_timesteps != 0 or source._n_updates != 0 or source.seed != seed:
        raise ValueError('source is not a zero-transition supervised initialization')
    if source.observation_space != agent.observation_space or source.action_space != agent.action_space:
        raise ValueError('initialization spaces mismatch')
    from building_gen.wrapper.shared_zone_policy import SharedZonePolicy
    shared = isinstance(agent.policy, SharedZonePolicy)
    if shared != isinstance(source.policy, SharedZonePolicy):
        raise ValueError('initialization actor architecture mismatch')
    if shared:
        import torch
        for name in ('local_indices', 'global_indices'):
            if not torch.equal(getattr(source.policy.mlp_extractor, name), getattr(agent.policy.mlp_extractor, name)):
                raise ValueError('shared initialization observation layout mismatch')
    actor_prefixes = ('mlp_extractor.zone_actor.',) if shared else ('mlp_extractor.policy_net.', 'action_net.')
    target = agent.policy.state_dict()
    source_state = source.policy.state_dict()
    keys = {k for k in target if k.startswith(actor_prefixes)}
    if not keys or keys != {k for k in source_state if k.startswith(actor_prefixes)}:
        raise ValueError('initialization actor architecture mismatch')
    if any(target[k].shape != source_state[k].shape for k in keys):
        raise ValueError('initialization actor tensor shape mismatch')
    import torch
    if any(not torch.isfinite(source_state[k]).all() for k in keys):
        raise ValueError('nonfinite initialization weights')
    for key in keys:
        target[key] = source_state[key].clone()
    agent.policy.load_state_dict(target, strict=True)
    return dict(kind='supervised_actor_only', receipt_sha256=hashlib.sha256(raw).hexdigest(),
                model_sha256=digest, source_task_id=source_task_id, seed=seed,
                **({"reward_compatibility": compatibility} if compatibility is not None else {}),
                copied_parameters=sorted(keys), critic='fresh', optimizer='fresh',
                policy_variance='fresh', environment_transitions=0)
