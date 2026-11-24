"""PPO with a frozen initial-policy KL penalty on training-rollout states.

The train loop is adapted from Stable-Baselines3 2.7.1 (MIT license below).
KL is KL(initial Gaussian || current Gaussian), summed over action dimensions
and averaged over minibatch states, before physical action clipping. The
coefficient is fixed; this is a policy-preservation diagnostic, not a comfort
constraint. The frozen policy is captured AFTER actor-only initialization.
"""

# The MIT License
#
# Copyright (c) 2019 Antonin Raffin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import copy
import math
import numpy as np
import torch as th
from torch.nn import functional as F
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance


def optimizer_diagnostics(policy):
    """Read gradient norms and parameters without changing optimizer/RNG state."""
    snapshots = {}
    norms = {}
    for name, parameter in policy.named_parameters():
        group = ('actor' if name.startswith(('mlp_extractor.policy_net.', 'action_net.')) or name == 'log_std'
                 else 'critic' if name.startswith(('mlp_extractor.value_net.', 'value_net.')) else 'other')
        if parameter.grad is None:
            continue
        snapshots[name] = (group, parameter.detach().clone())
        norms[group] = norms.get(group, 0.) + float(parameter.grad.detach().square().sum())
    return {f'{group}_gradient_norm': math.sqrt(value) for group,value in norms.items()}, snapshots


def optimizer_displacements(policy, snapshots):
    squared = {}
    for name, parameter in policy.named_parameters():
        if name in snapshots:
            group, before = snapshots[name]
            squared[group] = squared.get(group, 0.) + float((parameter.detach()-before).square().sum())
    return {f'{group}_parameter_step_norm': math.sqrt(value) for group,value in squared.items()}


class AnchoredPPO(PPO):
    def __init__(self, *args, anchor_coefficient=1.0, anchor_occupied_indices=None, **kwargs):
        if isinstance(anchor_coefficient, bool) or not math.isfinite(anchor_coefficient) or anchor_coefficient <= 0:
            raise ValueError("anchor_coefficient must be finite and positive")
        self.anchor_coefficient = float(anchor_coefficient)
        self.anchor_occupied_indices = anchor_occupied_indices
        self.anchor_policy = None
        super().__init__(*args, **kwargs)
        if self.use_sde or not isinstance(self.action_space, spaces.Box):
            raise ValueError("Anchored PPO requires ordinary continuous Gaussian actions")

        if anchor_occupied_indices is not None:
            if not isinstance(anchor_occupied_indices, list) or not anchor_occupied_indices or len(set(anchor_occupied_indices)) != len(anchor_occupied_indices) or any(type(i) is not int or i < 0 or i >= self.observation_space.shape[0] for i in anchor_occupied_indices):
                raise ValueError('anchor occupancy indices must identify distinct observation features')

    def freeze_anchor(self):
        if self.num_timesteps or self._n_updates or self.anchor_policy is not None:
            raise ValueError("Anchor must be frozen exactly once before training")
        self.anchor_policy = copy.deepcopy(self.policy)
        self.anchor_policy.set_training_mode(False)
        for parameter in self.anchor_policy.parameters():
            parameter.requires_grad_(False)

    def anchor_kl(self, observations):
        if self.anchor_policy is None:
            raise ValueError("Missing frozen initial policy")
        with th.no_grad():
            reference = self.anchor_policy.get_distribution(observations).distribution
        current = self.policy.get_distribution(observations).distribution
        kl = th.distributions.kl_divergence(reference, current).sum(dim=-1)
        if self.anchor_occupied_indices is not None:
            flags = observations[:, self.anchor_occupied_indices]
            if not th.all((flags == 0) | (flags == 1)):
                raise ValueError('anchor occupancy features must be binary flags')
            kl = kl * th.any(flags > 0, dim=-1).to(kl.dtype)
        return kl.mean()

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        if self.anchor_policy is None:
            raise ValueError("Freeze the initialized policy before training")
        optimizer_metrics = []
        anchor_losses = []
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                anchor_loss = self.anchor_kl(rollout_data.observations)
                anchor_losses.append(anchor_loss.item())
                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + self.anchor_coefficient * anchor_loss

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                gradient_metrics, snapshots = optimizer_diagnostics(self.policy)
                # Preserve the original joint clipping and optimizer update.
                raw_norm = th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                gradient_metrics['joint_clip_multiplier'] = min(1., self.max_grad_norm/(float(raw_norm)+1e-6))
                gradient_metrics.update(optimizer_displacements(self.policy, snapshots))
                optimizer_metrics.append(gradient_metrics)

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        for key in {key for entry in optimizer_metrics for key in entry}:
            self.logger.record(f'train/{key}', np.mean([entry[key] for entry in optimizer_metrics if key in entry]))
        self.logger.record("train/anchor_kl", np.mean(anchor_losses))
        self.logger.record("train/anchor_coefficient", self.anchor_coefficient)
        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
