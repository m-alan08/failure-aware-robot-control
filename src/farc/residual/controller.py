"""Residual controller and simulator integration.

    a_final = clip(a_C + alpha * clip(delta_RL, -1, 1), -1, 1)

- alpha = 0 (or ``ZeroResidual``) returns a_C itself, bitwise: Candidate C exactly.
- The residual policy receives a fresh copy of the observation vector and nothing else; it
  holds no reference to the estimator, supervisor, predictor or session.
- A non-finite or mis-shaped residual output is replaced by zero (fallback to C) and flagged.
- If the task part of the residual observation is invalid, the residual is forced to zero.
- An invalid *controller* observation (non-finite obs or goal) raises
  ``ResidualObservationError``, matching the hardware rule that such a sample ends the trial.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from farc.disturbances.config import SEED_RANGES
from farc.residual.core import CandidateCCore, Proposal
from farc.residual.observation import ResidualObservationBuilder, ResidualObservationError
from farc.residual.policy import ResidualPolicy, ZeroResidual
from farc.residual.reward import RewardConfig, residual_reward


@dataclass(frozen=True)
class ResidualStep:
    command: np.ndarray  # a_final
    a_c: np.ndarray
    delta: np.ndarray  # clipped residual actually applied (before alpha)
    observation: np.ndarray
    proposal: Proposal
    fallback: str | None  # why the residual was zeroed, if it was


class ResidualController:
    def __init__(self, core: CandidateCCore, policy: ResidualPolicy | None = None,
                 alpha: float = 0.0,
                 builder: ResidualObservationBuilder | None = None):  # fmt: skip
        if not (np.isfinite(alpha) and 0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.core, self.alpha = core, float(alpha)
        self.policy = policy if policy is not None else ZeroResidual()
        self.builder = builder or ResidualObservationBuilder()

    def reset(self) -> None:
        self.core.reset()
        self.builder.reset()

    def propose(self, obs: np.ndarray, goal: np.ndarray,
                base_action: np.ndarray | None = None,
                delta: np.ndarray | None = None) -> ResidualStep:  # fmt: skip
        """Compute a_final without committing. ``delta`` overrides the policy (used by the
        training env, where the RL agent supplies the action)."""
        obs, goal = np.asarray(obs, np.float64), np.asarray(goal, np.float64)
        if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(goal))):
            raise ResidualObservationError("non-finite controller observation or goal")
        p = self.core.propose(obs, goal, base_action)
        x = self.builder.build(self.core, p, obs, goal)
        fallback = None
        raw = self.policy(x.copy()) if delta is None else delta
        d = np.asarray(raw, np.float64)
        if d.shape != (4,) or not np.all(np.isfinite(d)):
            d, fallback = np.zeros(4), "invalid_residual"
        elif x[self.builder.spec.task_validity_index] == 0.0:
            d, fallback = np.zeros(4), "invalid_task_observation"
        d = np.clip(d, -1.0, 1.0)
        # alpha = 0: a_C itself, with no floating-point arithmetic on it (bitwise exact)
        command = p.a_c.copy() if self.alpha == 0.0 else np.clip(p.a_c + self.alpha * d, -1, 1)
        return ResidualStep(command, p.a_c, d, x, p, fallback)

    def commit(self, step: ResidualStep) -> None:
        self.core.commit(step.command)
        self.builder.record(step.delta)

    def step(self, obs, goal, base_action=None) -> ResidualStep:
        s = self.propose(obs, goal, base_action)
        self.commit(s)
        return s


def run_residual_episode(session, controller: ResidualController) -> dict[str, Any]:
    """Closed loop on an ``EpisodeSession``-like object (``ScheduledDelaySession`` for the
    simulator). Mirrors ``mechanism.run_episode`` in normal mode. The session's schedule is
    never read by the controller."""
    controller.reset()
    rows = []
    while not session.done:
        view = session.observe()
        s = controller.propose(view.observation, view.goal, view.base_action)
        session.act(s.command)
        controller.commit(s)
        rows.append({"t": view.step, "delay_used": s.proposal.delay,
                     "active": s.proposal.active, "alarm": bool(s.proposal.decision.alarm),
                     "suspended": bool(s.proposal.decision.suspended),
                     "residual_norm": float(np.linalg.norm(s.command - s.a_c)),
                     "fallback": s.fallback})  # fmt: skip
    return {
        "success": session.final_success,
        "final_distance": session.final_distance,
        "rows": rows,
        "commands": [np.asarray(c).copy() for c in session.commands],
    }


def check_training_seed(seed: int) -> int:
    if int(seed) != seed or seed >= SEED_RANGES["heldout"][0]:
        raise ValueError("held-out seeds (4,000,000+) are reserved and refused here")
    return int(seed)


class ResidualEnv(gym.Env):
    """Gymnasium env whose action is the residual delta. Built for a future (unapproved) SAC
    run; ``session_factory(seed)`` returns a fresh ``EpisodeSession``-like object. The policy
    sees only the residual observation; ``info`` carries trainer-side diagnostics."""

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(self, session_factory: Callable[[int], Any], controller: ResidualController,
                 reward: RewardConfig | None = None):  # fmt: skip
        self.factory, self.controller = session_factory, controller
        self.reward_cfg = reward or RewardConfig()
        dim = controller.builder.spec.dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (dim,), np.float64)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float64)
        self.session: Any = None
        self._pending: ResidualStep | None = None

    def _observe(self) -> np.ndarray:
        view = self.session.observe()
        self._view = view
        self._pending = self.controller.propose(view.observation, view.goal, view.base_action,
                                                delta=np.zeros(4))  # fmt: skip
        return self._pending.observation.copy()

    def reset(self, *, seed: int | None = None, options=None):
        if seed is None:
            raise ValueError("ResidualEnv needs an explicit episode seed")
        super().reset(seed=check_training_seed(seed))
        self.session = self.factory(seed)
        self.controller.reset()
        self._distance = float(self.session.final_distance)
        return self._observe(), {}

    def step(self, action):
        assert self._pending is not None
        p = self._pending
        d = np.asarray(action, np.float64)
        if d.shape != (4,) or not np.all(np.isfinite(d)):
            d = np.zeros(4)
        d = np.clip(d, -1.0, 1.0)
        a = self.controller.alpha
        command = p.a_c.copy() if a == 0.0 else np.clip(p.a_c + a * d, -1.0, 1.0)
        prev_delta = self.controller.builder.prev_delta.copy()
        step = ResidualStep(command, p.a_c, d, p.observation, p.proposal, None)
        t = self.session.step
        self.session.act(command)
        self.controller.commit(step)
        success = bool(self.session.successes[t])
        dist = float(self.session.final_distance)
        r = residual_reward(self.reward_cfg, success, dist, self._distance, d, prev_delta, a)
        self._distance = dist
        truncated = bool(self.session.done)
        self._pending = None
        obs = np.zeros(self.controller.builder.spec.dim) if truncated else self._observe()
        info = {"a_c": p.a_c, "a_final": command, "success": success}
        return obs, r, False, truncated, info
