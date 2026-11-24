"""Numerically stable probability distributions for bounded control actions.

This module intentionally has no EnergyPlus or Gymnasium dependency so policy
math can be tested without starting a simulator.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal


@lru_cache(maxsize=None)
def _hermgauss(points: int) -> tuple[np.ndarray, np.ndarray]:
    """Cache quadrature constants; policy updates call entropy many times."""

    return np.polynomial.hermite.hermgauss(points)


class LogisticNormal01:
    """A factorized logistic-normal distribution with support in ``[0, 1]``.

    A latent Gaussian sample ``z`` is mapped to an action with
    ``a = sigmoid(z)``.  :meth:`log_prob` includes the transform's Jacobian,
    which is essential when PPO re-evaluates the probability of stored actions.

    Values exactly on a floating-point boundary are moved to the nearest
    representable interior value when evaluating their density.  This makes
    rollout log probabilities finite without changing sampled actions sent to
    the environment.
    """

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        *,
        entropy_quadrature_points: int = 20,
    ) -> None:
        if loc.shape != scale.shape:
            raise ValueError(f"loc and scale must have the same shape, got {loc.shape} and {scale.shape}")
        if entropy_quadrature_points < 1:
            raise ValueError("entropy_quadrature_points must be positive")

        self.loc = loc
        self.scale = scale
        self.base_dist = Normal(loc, scale)
        self.entropy_quadrature_points = entropy_quadrature_points

    @property
    def deterministic_action(self) -> torch.Tensor:
        """Return the squashed latent mode used for deterministic evaluation.

        This is also the action-wise median.  It is intentionally not called the
        logistic-normal density's mode, which can depend on both ``loc`` and
        ``scale``.
        """

        return torch.sigmoid(self.loc)

    def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Draw a differentiable sample in the bounded action space."""

        return torch.sigmoid(self.base_dist.rsample(sample_shape))

    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Draw a non-differentiable sample in the bounded action space."""

        with torch.no_grad():
            return self.rsample(sample_shape)

    @staticmethod
    def _safe_interior(value: torch.Tensor) -> torch.Tensor:
        if not value.is_floating_point():
            raise TypeError("logistic-normal actions must be floating-point tensors")
        if torch.any((value < 0.0) | (value > 1.0)):
            raise ValueError("logistic-normal actions must lie in [0, 1]")

        eps = torch.finfo(value.dtype).eps
        return value.clamp(min=eps, max=1.0 - eps)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Return element-wise action log densities, including the Jacobian."""

        interior = self._safe_interior(value)
        latent = torch.log(interior) - torch.log1p(-interior)
        log_abs_det_jacobian = F.logsigmoid(latent) + F.logsigmoid(-latent)
        return self.base_dist.log_prob(latent) - log_abs_det_jacobian

    def entropy(self) -> torch.Tensor:
        """Estimate element-wise entropy with deterministic Gauss--Hermite quadrature.

        For ``A = sigmoid(Z)``, change of variables gives
        ``H(A) = H(Z) + E[log(sigmoid(Z) * (1 - sigmoid(Z)))]``.  The expectation
        has no simple closed form, so fixed Gauss--Hermite quadrature gives a
        low-variance, differentiable estimate suitable for PPO's entropy bonus.
        """

        nodes_np, weights_np = _hermgauss(self.entropy_quadrature_points)
        nodes = torch.as_tensor(nodes_np, dtype=self.loc.dtype, device=self.loc.device)
        weights = torch.as_tensor(weights_np, dtype=self.loc.dtype, device=self.loc.device)

        broadcast_shape = (self.entropy_quadrature_points,) + (1,) * self.loc.ndim
        latent = self.loc.unsqueeze(0) + math.sqrt(2.0) * self.scale.unsqueeze(0) * nodes.reshape(broadcast_shape)
        log_abs_det_jacobian = F.logsigmoid(latent) + F.logsigmoid(-latent)
        expected_log_jacobian = (weights.reshape(broadcast_shape) * log_abs_det_jacobian).sum(dim=0) / math.sqrt(
            math.pi
        )
        return self.base_dist.entropy() + expected_log_jacobian
