"""Fit the PPO actor to audited schedule examples; no RL or simulator updates."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import gymnasium as gym
import torch
from building_gen.wrapper.rl_baselines import create_agent


class ShapeEnvironment(gym.Env):
    def __init__(self, dimension, names=None, action_dimension=10):
        self.observation_names = names
        self.observation_space = gym.spaces.Box(0, 1, shape=(dimension,), dtype=np.float32)
        self.action_space = gym.spaces.Box(0, 1, shape=(action_dimension,), dtype=np.float32)

    def reset(self, **kwargs):
        raise RuntimeError('supervised-only environment must never be stepped')

    def step(self, action):
        raise RuntimeError('supervised-only environment must never be stepped')


def fit(data_root, parameters_path, output, observation, seed, epochs=200, suite_path=None, task_name=None):
    torch.set_num_threads(1)
    if epochs <= 0 or seed < 0 or observation not in (0, 1):
        raise ValueError('invalid fit settings')
    audit_bytes = (data_root/'audit.json').read_bytes()
    audit = json.loads(audit_bytes)
    if audit['status'] != 'passed':
        raise ValueError('demonstrations did not pass audit')
    task = None
    if task_name is not None:
        from building_gen.benchmark.config import load_benchmark_suite
        from building_gen.wrapper.dpt_data import ENCODING
        if suite_path is None or observation != 1:
            raise ValueError('named shared fitting requires a suite and O1 observations')
        task = load_benchmark_suite(suite_path).task_map[task_name]
        if (task.observation_space.encoding != ENCODING or task.episode.episode_length_hours != 168
                or task.action_profile.transform_id != 'bounded_thermostat_12_32_v1'):
            raise ValueError('named fit requires weekly calendar observations and bounded 12–32 thermostat actions')
    filename = (task_name if task_name is not None else f'wide_O{observation}_W0')+'.npz'
    records = [r for r in audit['outputs'] if r['file'] == filename]
    if len(records) != 1:
        raise ValueError('one authenticated demonstration record required')
    record = records[0]
    week_steps = task.episode.max_steps if task is not None else 1008
    action_dimension = task.action_space.shape[0] if task is not None else 10
    path = data_root/record['file']
    if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
        raise ValueError('demonstration checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        x = torch.tensor(data['observations'])
        y = torch.tensor(data['actions'])
        # Entire final day per training week is excluded from supervised updates.
        steps = data['step']
        if steps.shape != (4*week_steps,) or not np.array_equal(steps, np.tile(np.arange(week_steps), 4)):
            raise ValueError('four complete ordered demonstration weeks required')
        train = torch.tensor(steps < 6*week_steps//7)
    if x.shape != (4*week_steps, record['observation_dimension']) or y.shape != (4*week_steps, action_dimension) or int(train.sum()) != 4*6*week_steps//7:
        raise ValueError('unexpected demonstration dimensions/split')
    if any(not torch.isfinite(v).all() or torch.any(v < 0) or torch.any(v > 1) for v in (x, y)):
        raise ValueError('finite normalized demonstrations required')
    parameters = json.loads(parameters_path.read_text())
    names = None
    shared = parameters.get('policy_architecture') == 'shared_zone_v1'
    if task_name is not None and not shared:
        raise ValueError('named task fitting requires the shared actor')
    if shared:
        from building_gen.benchmark.config import load_benchmark_suite
        from building_gen.benchmark.environment import _expected_observation_names
        if suite_path is None or observation != 1:
            raise ValueError('shared fitting requires named O1 task semantics')
        task = task or load_benchmark_suite(suite_path).task_map['wide_O1_W0']
        if task.task_id != record['task_id']:
            raise ValueError('demonstration task differs from named shared layout')
        names = _expected_observation_names(task)
        if len(names) != x.shape[1]:raise ValueError('shared observation dimension mismatch')
    agent = create_agent('ppo', ShapeEnvironment(x.shape[1], names, action_dimension), seed=seed, parameters=parameters)
    policy = agent.policy
    before_value = {k: v.clone() for k, v in policy.state_dict().items() if 'value' in k or k == 'log_std'}
    actor_parameters = (list(policy.mlp_extractor.zone_actor.parameters()) if shared else
                        list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters()))
    optimizer = torch.optim.Adam(actor_parameters, lr=.001)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.where(train)[0]

    def means(inputs):
        features = policy.extract_features(inputs)
        return policy.action_net(policy.mlp_extractor.forward_actor(features))

    def metrics(mask):
        with torch.no_grad():
            prediction = means(x[mask])
            target = y[mask]
            clipped = prediction.clamp(0, 1)
            def physical(a):
                heat = 12+12*a[:, ::2]
                cool = heat+2+(30-heat)*a[:, 1::2]
                return torch.stack((heat, cool), dim=-1)
            error = (physical(clipped)-physical(target)).abs()
            return dict(normalized_mse=float((prediction-target).square().mean()), physical_mae_c=float(error.mean()), physical_max_error_c=float(error.max()), clipped_fraction=float(((prediction<0)|(prediction>1)).float().mean()))

    output.mkdir(parents=True, exist_ok=False)
    history = [dict(epoch=0, fit=metrics(train), withheld_training_days=metrics(~train))]
    for epoch in range(1, epochs+1):
        order = indices[torch.randperm(len(indices), generator=generator)]
        for batch in order.split(256):
            loss = (means(x[batch])-y[batch]).square().mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite supervised loss')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0 or epoch == epochs:
            history.append(dict(epoch=epoch, fit=metrics(train), withheld_training_days=metrics(~train)))
    for key, value in before_value.items():
        if not torch.equal(value, policy.state_dict()[key]):
            raise ValueError('supervised fit changed critic or policy variance')
    agent.save(output/'actor_initialization.zip')
    receipt = dict(status='complete', seed=seed, observation=observation, epochs=epochs, learning_rate=.001, batch_size=256, fit_examples=int(train.sum()), withheld_examples=int((~train).sum()), demonstration_audit_sha256=hashlib.sha256(audit_bytes).hexdigest(), data=record, parameter_sha256=hashlib.sha256(parameters_path.read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), model_sha256=hashlib.sha256((output/'actor_initialization.zip').read_bytes()).hexdigest(), history=history, scope='Supervised actor initialization only; fixed final epoch, no validation/test selection. Withheld last days are from training weeks, not independent deployment tests. Critic and log_std unchanged; zero environment transitions. Requires separate closed-loop evaluation before use as control evidence.')
    (output/'imitation.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_root', type=Path); p.add_argument('parameters', type=Path); p.add_argument('output', type=Path)
    p.add_argument('--observation', type=int, required=True); p.add_argument('--seed', type=int, required=True); p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--suite', type=Path)
    p.add_argument('--task', help='Explicit shared task; requires four complete audited weeks for that task.')
    a = p.parse_args()
    fit(a.data_root, a.parameters, a.output, a.observation, a.seed, a.epochs, a.suite, a.task)
