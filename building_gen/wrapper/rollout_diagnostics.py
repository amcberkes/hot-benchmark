"""Read-only summaries of a completed PPO rollout before optimizer updates."""
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


def describe(values):
    x=np.asarray(values,dtype=np.float64)
    if not x.size or not np.isfinite(x).all():raise ValueError('invalid diagnostic values')
    return dict(mean=float(x.mean()),std=float(x.std()),minimum=float(x.min()),maximum=float(x.max()))


def correlation(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    if len(a)<2:return None
    a=a-a.mean();b=b-b.mean();denominator=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/denominator) if denominator>0 else None


def summarize(model):
    buffer=model.rollout_buffer
    if not buffer.full or buffer.n_envs!=1:raise ValueError('requires one complete single-environment rollout')
    observations=buffer.observations[:,0]
    means=[];scales=[]
    with torch.no_grad():
        for start in range(0,len(observations),256):
            tensor=torch.as_tensor(observations[start:start+256],device=model.device)
            distribution=model.policy.get_distribution(tensor).distribution
            means.append(distribution.mean.cpu().numpy());scales.append(distribution.stddev.cpu().numpy())
    mean=np.concatenate(means);scale=np.concatenate(scales);actions=buffer.actions[:,0]
    if mean.shape!=actions.shape or scale.shape!=actions.shape or np.any(scale<=0):raise ValueError('requires diagonal continuous Gaussian policy')
    noise=(actions-mean)/scale
    advantage=buffer.advantages[:,0]
    adjacent=~buffer.episode_starts[1:,0].astype(bool)
    return dict(training_step=int(model.num_timesteps),transitions=len(observations),
        rewards=describe(buffer.rewards),values=describe(buffer.values),returns=describe(buffer.returns),advantages=describe(advantage),
        positive_advantage_fraction=float((advantage>0).mean()),
        raw_action_clipped_fraction=float(((actions<model.action_space.low)|(actions>model.action_space.high)).mean()),
        policy_std=describe(scale),standardized_action_noise=describe(noise),
        noise_lag1_correlation_by_action=[correlation(noise[:-1,i][adjacent],noise[1:,i][adjacent]) for i in range(actions.shape[1])],
        noise_advantage_correlation_by_action=[correlation(noise[:,i],advantage) for i in range(actions.shape[1])])


class RolloutDiagnosticsCallback(BaseCallback):
    def __init__(self,destination):
        super().__init__();self.destination=Path(destination);self.records=[]
    def _on_step(self):return True
    def _on_rollout_end(self):
        self.records.append(summarize(self.model))
        payload=dict(schema_version='hot-rollout-diagnostics/1.0.0',
            scope='Pre-update stochastic PPO buffers. Rewards include SB3 timeout bootstraps; they are not physical interval costs. Noise correlations describe the collected rollout and do not prove causal learning quality or convergence.',records=self.records)
        temporary=self.destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n');temporary.replace(self.destination)
