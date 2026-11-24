"""PPO actor with an exact calendar thermostat plus learned normalized residuals.

The teacher uses current named calendar features and comfort bounds only. Zero
residual reproduces weekday05:00–23:00 operation and15.56/29.44°C setbacks.
The Gaussian acts in the existing absolute normalized action space; SB3 retains
its ordinary likelihood and clipping semantics. This is a building-wide actor.
"""
import math
import torch
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy


def layout(names):
    zones=[n[:-len('_comfort_minimum_c')] for n in names if n.endswith('_comfort_minimum_c')]
    if not zones:
        raise ValueError('scheduled residual requires named comfort bounds')
    return [[names.index(z+'_comfort_minimum_c'),names.index(z+'_comfort_maximum_c')] for z in zones], [names.index(n) for n in ('hour_sin','hour_cos','weekday_sin','weekday_cos')]


class ScheduledExtractor(nn.Module):
    def __init__(self, width, bounds_indices, clock_indices):
        super().__init__()
        self.register_buffer('bounds_indices',torch.tensor(bounds_indices))
        self.register_buffer('clock_indices',torch.tensor(clock_indices))
        self.latent_dim_pi=2*len(bounds_indices); self.latent_dim_vf=64
        self.actor=nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,self.latent_dim_pi))
        self.critic=nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh())
        nn.init.zeros_(self.actor[-1].weight);nn.init.zeros_(self.actor[-1].bias)

    def teacher(self,x):
        c=x[:,self.clock_indices]*2-1
        minutes=torch.round(torch.remainder(torch.atan2(c[:,0],c[:,1])/(2*math.pi),1)*1440).remainder(1440)
        weekday=torch.round(torch.remainder(torch.atan2(c[:,2],c[:,3])/(2*math.pi),1)*7).remainder(7)+1
        active=((weekday>=2)&(weekday<=6)&(minutes>=300)&(minutes<1380))[:,None]
        bands=x[:,self.bounds_indices]*30+10
        widening=getattr(self,"initial_active_widening_c",0.)
        heat=torch.where(active,bands[:,:,0]-widening,15.56)
        cool=torch.where(active,bands[:,:,1]+widening,29.44)
        return torch.stack(((heat-12)/12,(cool-heat-2)/(30-heat)),dim=-1).flatten(1)

    def forward_actor(self,x):return self.teacher(x)+self.actor(x)
    def forward_critic(self,x):return self.critic(x)
    def forward(self,x):return self.forward_actor(x),self.forward_critic(x)


class ScheduledResidualPolicy(ActorCriticPolicy):
    def __init__(self,*args,bounds_indices,clock_indices,**kwargs):
        self.bounds_indices=bounds_indices;self.clock_indices=clock_indices
        if kwargs.get('use_sde',False):raise ValueError('scheduled residual does not support SDE')
        super().__init__(*args,**kwargs)

    def _build(self,lr_schedule):
        if self.action_space.shape!=(2*len(self.bounds_indices),):raise ValueError('expected two ordered actions per zone')
        self.mlp_extractor=ScheduledExtractor(self.features_dim,self.bounds_indices,self.clock_indices).to(self.device)
        self.action_net=nn.Identity()
        self.log_std=nn.Parameter(torch.full(self.action_space.shape,self.log_std_init))
        self.value_net=nn.Linear(64,1)
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

    def _get_constructor_parameters(self):
        result=super()._get_constructor_parameters()
        result.update(bounds_indices=self.bounds_indices,clock_indices=self.clock_indices)
        return result


class ScheduledSDEExtractor(ScheduledExtractor):
    """Keep the same mean network; append fixed state features for exploration."""
    def forward_actor(self, x):
        return torch.cat((super().forward_actor(x), x), dim=1)


class ScheduledResidualSDEPolicy(ScheduledResidualPolicy):
    """Exact scheduled mean with SB3 state-dependent, persistent exploration.

    Exploration features are normalized observations plus a constant, detached
    from learning. Noise resampling is managed by PPO's sde_sample_freq. The
    ordinary SB3 marginal Gaussian likelihood is used; no post-sampling noise
    or action smoothing is added.
    """
    def __init__(self, *args, bounds_indices, clock_indices, **kwargs):
        self.bounds_indices = bounds_indices
        self.clock_indices = clock_indices
        if not kwargs.get('use_sde', False):
            raise ValueError('scheduled SDE requires use_sde=True')
        if kwargs.get('squash_output', False):
            raise ValueError('scheduled SDE retains unsquashed absolute actions')
        ActorCriticPolicy.__init__(self, *args, **kwargs)

    def _build(self, lr_schedule):
        self.action_count = 2 * len(self.bounds_indices)
        if self.action_space.shape != (self.action_count,):
            raise ValueError('expected two ordered actions per zone')
        self.mlp_extractor = ScheduledSDEExtractor(
            self.features_dim, self.bounds_indices, self.clock_indices
        ).to(self.device)
        self.action_net = nn.Identity()
        self.log_std = nn.Parameter(torch.full(
            (self.features_dim + 1, self.action_count), self.log_std_init
        ))
        self.action_dist.sample_weights(self.log_std)
        self.value_net = nn.Linear(64, 1)
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        mean = latent_pi[:, :self.action_count]
        state = latent_pi[:, self.action_count:].detach()
        features = torch.cat((state, torch.ones_like(state[:, :1])), dim=1)
        features = torch.nn.functional.normalize(features, dim=1)
        return self.action_dist.proba_distribution(mean, self.log_std, features)
