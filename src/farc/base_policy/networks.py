"""The SAC actor architecture of the frozen base policies (full precision only).

Migrated from m-alan08/quantrl-fetch src/quantrl/algos/networks.py (commit f2591d0), keeping
only what inference needs: the plain-Linear MLP and the deterministic action tanh(mean).
Parameter names (``network.<i>.weight``) match the original checkpoints. See PROVENANCE.md.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def build_mlp(input_dim: int, hidden_sizes: Sequence[int], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_sizes:
        layers.append(nn.Linear(previous_dim, hidden_dim))
        layers.append(nn.ReLU())
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.network = build_mlp(input_dim, hidden_sizes, 2 * action_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.network(inputs).chunk(2, dim=-1)
        return mean, log_std.clamp(self.log_std_min, self.log_std_max)

    def deterministic_action(self, inputs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(inputs)
        return torch.tanh(mean)
