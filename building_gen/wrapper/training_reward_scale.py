"""Explicit fixed PPO training scale; physical diagnostics remain untouched."""
import math
import gymnasium as gym


class TrainingRewardScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
            raise ValueError('training reward scale must be finite and positive')
        super().__init__(env)
        self.scale = float(scale)

    def reward(self, reward):
        return reward * self.scale

    def __getattr__(self, name):
        if name == 'env':
            raise AttributeError(name)
        return getattr(self.env, name)
