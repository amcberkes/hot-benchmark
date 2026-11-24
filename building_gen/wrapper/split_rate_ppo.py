"""PPO with an explicit critic/actor learning-rate ratio.

SB3 refreshes every optimizer group's rate before each training call. Apply the
ratio afterwards so schedules cannot silently erase it or compound it.
"""
from stable_baselines3 import PPO


class SplitRatePPO(PPO):
    def _update_learning_rate(self, optimizers):
        super()._update_learning_rate(optimizers)
        groups = optimizers if isinstance(optimizers, list) else [optimizers]
        multiplier = self.policy.critic_learning_rate_multiplier
        for optimizer in groups:
            if {g.get('hot_role') for g in optimizer.param_groups} != {'actor', 'critic'}:
                raise ValueError('Split-rate PPO requires explicit actor and critic groups')
            for group in optimizer.param_groups:
                if group['hot_role'] == 'critic':
                    group['lr'] *= multiplier
                    self.logger.record('train/critic_learning_rate', group['lr'])
                else:
                    self.logger.record('train/actor_learning_rate', group['lr'])
