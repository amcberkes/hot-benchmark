"""Training-only energy-weight ablations on authenticated weekly reward terms."""
import math
import gymnasium as gym


def validate_energy_weight(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('training_energy_weight must be finite and nonnegative')
    return float(value)


class TrainingEnergyWeight(gym.Wrapper):
    """Keep occupied comfort, normalization and evaluation semantics unchanged."""
    def __init__(self, env, weight):
        super().__init__(env)
        self.weight = validate_energy_weight(weight)

    def step(self, action):
        observation, base_reward, terminated, truncated, info = self.env.step(action)
        evidence = info.get('training_reward_components', {})
        if evidence.get('schema_version') != 'hot-weekly-training-reward/1.0.0':
            raise ValueError('energy-weight ablation requires calibrated weekly rewards')
        def component(key):
            value = evidence.get(key)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid weekly reward component: {key}')
            return float(value)
        comfort, energy, old_weight = map(component, ('normalized_comfort', 'normalized_energy', 'energy_weight'))
        expected = -(comfort + old_weight * energy)
        if not math.isclose(base_reward, expected, rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('base reward differs from authenticated weekly components')
        reward = -(comfort + self.weight * energy)
        return observation, reward, terminated, truncated, dict(info,
            unmodified_training_reward=base_reward,
            training_objective=dict(schema='hot-training-energy-weight/1', energy_weight=self.weight,
                                    comfort_weight=1., reward=reward))
