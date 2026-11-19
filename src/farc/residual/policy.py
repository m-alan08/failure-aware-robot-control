"""Residual policies: anything mapping a residual observation to a delta in [-1, 1]^4."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from torch import nn


class ResidualPolicy(Protocol):
    def __call__(self, observation: np.ndarray) -> np.ndarray: ...


class ZeroResidual:
    """Always zero: the controller then reproduces Candidate C exactly."""

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        return np.zeros(4)


class ResidualActor(nn.Module):
    """Tanh-squashed Gaussian actor for SAC (deterministic mean at inference).

    The output layer is zero-initialised, so an untrained actor's deterministic output is
    exactly zero: training starts *at* Candidate C rather than at a random perturbation of it.
    """

    def __init__(self, obs_dim: int, hidden: tuple[int, ...] = (256, 256), act_dim: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.mean = nn.Linear(last, act_dim)
        self.log_std = nn.Linear(last, act_dim)
        for layer in (self.mean, self.log_std):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        self.obs_dim = obs_dim

    def forward(self, obs: torch.Tensor, deterministic: bool = True,
                generator: torch.Generator | None = None) -> torch.Tensor:  # fmt: skip
        h = self.trunk(obs)
        mu = self.mean(h)
        if deterministic:
            return torch.tanh(mu)
        std = self.log_std(h).clamp(-5.0, 2.0).exp()
        eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
        return torch.tanh(mu + std * eps)

    def __call__(self, observation, *args, **kwargs):
        """Tensors: the ordinary differentiable module call (for training). Arrays: the
        deployment interface, one observation in, one deterministic delta out, no gradients."""
        if isinstance(observation, torch.Tensor):
            return super().__call__(observation, *args, **kwargs)
        x = torch.as_tensor(np.asarray(observation, np.float32)).reshape(1, -1)
        if x.shape[1] != self.obs_dim:
            raise ValueError(f"residual observation has {x.shape[1]} dims, expected {self.obs_dim}")
        with torch.no_grad():
            return super().__call__(x)[0].numpy().astype(np.float64)
