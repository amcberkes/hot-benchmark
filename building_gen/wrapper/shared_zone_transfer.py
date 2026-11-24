"""Inference with shared actor weights on a new zone layout.

Default inference is deterministic. An explicit sampling seed enables a local
Gaussian diagnostic stream without changing the global Torch RNG.

This adapter does not transfer a critic, optimizer, or training state. Callers
must separately validate task normalization and physical action semantics.
"""
import numpy as np
import torch
from building_gen.wrapper.shared_zone_policy import SharedZonePolicy, observation_layout


class SharedZoneActorAdapter:
    def __init__(self, policy, source_names, target_names, *, sampling_seed=None):
        if not isinstance(policy, SharedZonePolicy):
            raise TypeError('shared SB3 zone policy required')
        source_names=list(source_names);target_names=list(target_names)
        if len(set(source_names))!=len(source_names) or len(set(target_names))!=len(target_names):
            raise ValueError('observation names must be unique')
        source_local,source_global=observation_layout(source_names)
        target_local,target_global=observation_layout(target_names)
        if source_local!=policy.local_indices or source_global!=policy.global_indices:
            raise ValueError('source layout differs from saved policy')
        if [source_names[i] for i in source_global]!=[target_names[i] for i in target_global]:
            raise ValueError('global feature semantics/order differ')
        self.policy=policy;self.local=target_local;self.global_=target_global
        self.width=len(target_names)
        if sampling_seed is not None and (type(sampling_seed) is not int or not 0<=sampling_seed<2**63):
            raise ValueError('sampling_seed must be a nonnegative signed-64-bit integer')
        self.generator=None
        if sampling_seed is not None:
            self.generator=torch.Generator(device=policy.device).manual_seed(sampling_seed)

    def __call__(self, observation):
        x=np.asarray(observation,dtype=np.float32)
        if x.shape!=(self.width,) or not np.isfinite(x).all() or np.any(x<0) or np.any(x>1):
            raise ValueError('expected finite normalized target observation')
        self.policy.set_training_mode(False)
        with torch.no_grad():
            tensor=torch.as_tensor(x,device=self.policy.device)
            local=tensor[torch.as_tensor(self.local,device=tensor.device)]
            global_=tensor[self.global_].unsqueeze(0).expand(len(self.local),-1)
            actions=self.policy.mlp_extractor.zone_actor(torch.cat((local,global_),dim=-1))
            if self.generator is not None:
                scale=self.policy.log_std.exp()
                if scale.shape!=(2,) or not torch.isfinite(scale).all() or (scale<=0).any():
                    raise ValueError('invalid shared Gaussian standard deviation')
                actions=actions+torch.randn(actions.shape,generator=self.generator,device=actions.device,dtype=actions.dtype)*scale
        return np.clip(actions.flatten().cpu().numpy(),0.,1.)


def validate_shared_targets(source, targets):
    """Allow zone count changes only under identical feature/action conventions."""
    from building_gen.benchmark.environment import _expected_observation_names
    from building_gen.wrapper.observation_variants import CALENDAR_ENCODING
    if not targets or len({t.task_id for t in targets})!=len(targets):
        raise ValueError('targets must be nonempty and unique')
    for target in (source,*targets):
        if target.observation_space.encoding!=CALENDAR_ENCODING:
            raise ValueError('shared transfer currently requires calendar occupancy/bounds encoding')
        for field in ('observation_space','action_space'):
            s=getattr(source,field);t=getattr(target,field)
            if any(getattr(s,k)!=getattr(t,k) for k in ('kind','dtype','encoding','lower_bound','upper_bound')) or t.kind!='box' or t.lower_bound!=0 or t.upper_bound!=1:
                raise ValueError('incompatible shared feature/action normalization')
        s=source.action_profile;t=target.action_profile
        if t.profile_id!='thermostat_v1' or (s.profile_id,s.transform_id,s.channels)!=(t.profile_id,t.transform_id,t.channels):
            raise ValueError('incompatible shared physical action semantics')
        names=_expected_observation_names(target)
        if target.observation_space.shape!=(len(names),) or target.action_space.shape!=(2*len(t.controlled_entities),):
            raise ValueError('incompatible shared task dimensions')
        local,global_=observation_layout(list(names))
        if len(local)!=len(t.controlled_entities):raise ValueError('zone layout mismatch')
        source_names=_expected_observation_names(source)
        _,source_global=observation_layout(list(source_names))
        if [names[i] for i in global_]!=[source_names[i] for i in source_global]:raise ValueError('global feature mismatch')
