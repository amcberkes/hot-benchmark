"""Explicit training-only objective matching a zero-weight comfort penalty."""
import math
import gymnasium as gym


class EnergyOnlyReward(gym.Wrapper):
    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        evidence = info.get('training_reward_components', {})
        if evidence.get('schema_version') != 'hot-weekly-training-reward/1.0.0':
            raise ValueError('energy-only training requires calibrated weekly reward evidence')
        energy = evidence.get('normalized_energy')
        if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not math.isfinite(energy) or energy < 0:
            raise ValueError('invalid normalized weekly energy')
        return observation, -float(energy), terminated, truncated, dict(
            info, unconstrained_training_reward=base_reward, energy_only_training_reward=-float(energy))
