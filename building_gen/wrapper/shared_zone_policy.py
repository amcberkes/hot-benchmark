"""Shared zone actor and permutation-invariant pooled building critic."""
import math
import numpy as np
import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


def observation_layout(names):
    zones=[x[:-12] for x in names if x.endswith('_temperature') and x!='outdoor_temperature']
    local=[]
    for z in zones:
        required=[f'{z}_temperature',f'{z}_occupant_count',f'{z}_comfort_minimum_c',f'{z}_comfort_maximum_c',f'{z}_occupied']
        local.append([names.index(x) for x in required])
    used={i for row in local for i in row}
    if not zones:raise ValueError('shared actor requires named zone observations')
    return local,[i for i in range(len(names)) if i not in used]


class SharedExtractor(nn.Module):
    def __init__(self,local_indices,global_indices):
        super().__init__();self.register_buffer('local_indices',th.tensor(local_indices));self.register_buffer('global_indices',th.tensor(global_indices));self.n=len(local_indices)
        width=len(local_indices[0])+len(global_indices)
        self.zone_actor=nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,2))
        self.zone_value=nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        self.latent_dim_pi=2*self.n;self.latent_dim_vf=64
        nn.init.normal_(self.zone_actor[-1].weight,std=.001)
        with th.no_grad():self.zone_actor[-1].bias.copy_(th.tensor([2/3,.2]))
    def zone_inputs(self,features):
        local=features[:,self.local_indices];glob=features[:,self.global_indices].unsqueeze(1).expand(-1,self.n,-1)
        return th.cat([local,glob],dim=-1)
    def forward_actor(self,features):return self.zone_actor(self.zone_inputs(features)).flatten(1)
    def forward_critic(self,features):return self.zone_value(self.zone_inputs(features)).mean(dim=1)
    def forward(self,features):return self.forward_actor(features),self.forward_critic(features)


class SharedZonePolicy(ActorCriticPolicy):
    def __init__(self,*args,local_indices,global_indices,critic_learning_rate_multiplier=1.,**kwargs):
        self.local_indices=local_indices;self.global_indices=global_indices
        if type(critic_learning_rate_multiplier) not in (int,float) or not math.isfinite(critic_learning_rate_multiplier) or critic_learning_rate_multiplier <= 0:
            raise ValueError("critic learning-rate multiplier must be finite and positive")
        self.critic_learning_rate_multiplier=float(critic_learning_rate_multiplier)
        if kwargs.get('use_sde',False):raise ValueError('shared actor does not support SDE')
        super().__init__(*args,**kwargs)
    def _build(self,lr_schedule):
        self.mlp_extractor=SharedExtractor(self.local_indices,self.global_indices).to(self.device)
        if self.action_space.shape!=(2*len(self.local_indices),):raise ValueError('expected two ordered actions per zone')
        self.action_net=nn.Identity();self.log_std=nn.Parameter(th.full((2,),self.log_std_init))
        self.value_net=nn.Linear(64,1)
        parameters=self.parameters()
        if self.critic_learning_rate_multiplier != 1.:
            critic=list(self.mlp_extractor.zone_value.parameters())+list(self.value_net.parameters())
            critic_ids={id(p) for p in critic}
            actor=[p for p in self.parameters() if id(p) not in critic_ids]
            parameters=[dict(params=actor,hot_role='actor'),dict(params=critic,hot_role='critic')]
        self.optimizer=self.optimizer_class(parameters,lr=lr_schedule(1),**self.optimizer_kwargs)
    def _get_action_dist_from_latent(self,latent_pi):
        return self.action_dist.proba_distribution(latent_pi,self.log_std.repeat(len(self.local_indices)))
    def _get_constructor_parameters(self):
        result=super()._get_constructor_parameters();result.update(local_indices=self.local_indices,global_indices=self.global_indices,critic_learning_rate_multiplier=self.critic_learning_rate_multiplier);return result
