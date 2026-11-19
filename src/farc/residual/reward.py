"""Residual reward (proposal section 4). A pure function of simulator quantities that are
available to the *trainer* (not to the policy). Not used by any experiment yet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RewardConfig:
    success_bonus: float = 1.0  # sparse task reward, as the base SAC (0 on success, -1 else)
    progress_weight: float = 0.0  # optional dense shaping on object-goal distance (off)
    residual_l2: float = 0.1  # penalty on ||alpha * delta||^2: prefer C unless it helps
    residual_rate: float = 0.05  # penalty on ||delta_t - delta_{t-1}||^2: no chattering


def residual_reward(cfg: RewardConfig, success: bool, distance: float, prev_distance: float,
                    delta: np.ndarray, prev_delta: np.ndarray, alpha: float) -> float:  # fmt: skip
    r = (0.0 if success else -1.0) * cfg.success_bonus
    r += cfg.progress_weight * (prev_distance - distance)
    r -= cfg.residual_l2 * float(np.sum((alpha * np.asarray(delta)) ** 2))
    r -= cfg.residual_rate * float(np.sum((np.asarray(delta) - np.asarray(prev_delta)) ** 2))
    return float(r)
