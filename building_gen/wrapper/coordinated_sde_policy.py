"""Persistent full-covariance exploration with a known invertible noise transform.

The transform is part of the probability distribution, not a post-sampling action
edit. A common signed component encourages lower heating and higher cooling
offsets (or the reverse) across zones, retaining independent exploration.
"""
import math
import torch
from stable_baselines3.common.distributions import StateDependentNoiseDistribution
from building_gen.wrapper.scheduled_residual_policy import ScheduledResidualSDEPolicy


class CoordinatedNoiseDistribution(StateDependentNoiseDistribution):
    def __init__(self, action_dim, correlation, cooling_ratio):
        super().__init__(action_dim, full_std=True, epsilon=0.)
        signs=torch.tensor([-1.,1.]*(action_dim//2))
        covariance=(1-correlation)*torch.eye(action_dim)+correlation*torch.outer(signs,signs)
        scales=torch.tensor([1.,cooling_ratio]*(action_dim//2))
        self.mix=scales[:,None]*torch.linalg.cholesky(covariance)

    def proba_distribution(self, mean_actions, log_std, latent_sde):
        self._latent_sde=latent_sde.detach()
        variance=self._latent_sde.square() @ self.get_std(log_std).square()
        mix=self.mix.to(mean_actions)
        scale_tril=mix[None,:,:]*variance.sqrt()[:,None,:]
        self.distribution=torch.distributions.MultivariateNormal(mean_actions,scale_tril=scale_tril)
        return self

    def get_noise(self, latent_sde):
        return super().get_noise(latent_sde) @ self.mix.to(latent_sde).T

    def log_prob(self, actions):return self.distribution.log_prob(actions)
    def entropy(self):return self.distribution.entropy()


class CoordinatedScheduledPolicy(ScheduledResidualSDEPolicy):
    def __init__(self,*args,exploration_correlation=.8,cooling_noise_ratio=3.,initial_active_widening_c=0.,**kwargs):
        if not 0<=exploration_correlation<1 or not math.isfinite(cooling_noise_ratio) or cooling_noise_ratio<=0:
            raise ValueError('invalid exploration covariance')
        if not 0<=initial_active_widening_c<=1:
            raise ValueError('initial active widening must be in [0,1] Celsius')
        if kwargs.get('full_std',True) is not True or kwargs.get('use_expln',False):
            raise ValueError('coordinated exploration requires full_std and exponential std')
        self.exploration_correlation=exploration_correlation
        self.cooling_noise_ratio=cooling_noise_ratio
        self.initial_active_widening_c=initial_active_widening_c
        super().__init__(*args,**kwargs)

    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        self.mlp_extractor.initial_active_widening_c=self.initial_active_widening_c
        self.action_dist=CoordinatedNoiseDistribution(self.action_count,self.exploration_correlation,self.cooling_noise_ratio)
        self.action_dist.sample_weights(self.log_std)

    def _get_constructor_parameters(self):
        result=super()._get_constructor_parameters()
        result.update(exploration_correlation=self.exploration_correlation,cooling_noise_ratio=self.cooling_noise_ratio,initial_active_widening_c=self.initial_active_widening_c)
        return result
