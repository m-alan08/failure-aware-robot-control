"""Deterministic residual corrections on top of the frozen base action.

``a_t = clip(a_base,t + alpha * delta_t, -1, 1)`` on the three position components; the
finger component is the base policy's (blocked in Push). ``delta_t`` has components in
[-1, 1] (one unit = 0.05 m of mocap displacement per step) and depends only on the
controller's view at step t: gripper g = obs[0:3], object o = obs[3:6], goal G, the base
action. u is the horizontal unit vector from o to G.

- ``goal_directed``: delta = u (push harder toward the goal).
- ``alignment``: steer the gripper onto the o->G line at the object's height:
  delta = clip((p - g) / 0.05) with p the projection of g on that line (z = o_z).
- ``reduced``: delta = -a_base (the base action shrunk by a factor 1 - alpha).
- ``random_direction`` (placebo control): a deterministic pseudo-random horizontal unit
  direction, seeded by the observation; same magnitude as ``goal_directed``.
- ``preserve_push``: delta = min(max(a_base_xy . u, 0), 1) u (amplify the base action's own
  component along the push direction; nothing is added when the base action is not pushing).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

STEP = 0.05
STRATEGIES = ("goal_directed", "alignment", "reduced", "preserve_push")


def _push_direction(obj: np.ndarray, goal: np.ndarray) -> np.ndarray | None:
    v = np.asarray(goal, float)[:2] - np.asarray(obj, float)[:2]
    n = float(np.linalg.norm(v))
    return None if n < 1e-9 else v / n


def goal_directed(g, o, goal, base) -> np.ndarray:
    u = _push_direction(o, goal)
    return np.zeros(3) if u is None else np.array([u[0], u[1], 0.0])


def alignment(g, o, goal, base) -> np.ndarray:
    u = _push_direction(o, goal)
    if u is None:
        return np.zeros(3)
    rel = np.asarray(g, float)[:2] - np.asarray(o, float)[:2]
    point = np.asarray(o, float)[:2] + float(rel @ u) * u
    target = np.array([point[0], point[1], float(o[2])])
    return np.clip((target - np.asarray(g, float)) / STEP, -1.0, 1.0)


def reduced(g, o, goal, base) -> np.ndarray:
    return -np.asarray(base, float)[:3]


def preserve_push(g, o, goal, base) -> np.ndarray:
    u = _push_direction(o, goal)
    if u is None:
        return np.zeros(3)
    along = min(max(float(np.asarray(base, float)[:2] @ u), 0.0), 1.0)
    return np.array([along * u[0], along * u[1], 0.0])


def random_direction(g, o, goal, base) -> np.ndarray:
    """PLACEBO: a horizontal unit direction drawn from a generator seeded by the observation
    bytes (deterministic, unrelated to the task geometry); same magnitude as goal_directed."""
    key = np.ascontiguousarray(np.round(np.r_[g, o, goal], 9), np.float64).tobytes()
    angle = np.random.default_rng(list(key)).uniform(0.0, 2.0 * np.pi)
    return np.array([np.cos(angle), np.sin(angle), 0.0])


PLACEBOS = ("random_direction",)

DELTAS: dict[str, Callable[..., np.ndarray]] = {
    "random_direction": random_direction,
    "goal_directed": goal_directed,
    "alignment": alignment,
    "reduced": reduced,
    "preserve_push": preserve_push,
}


@dataclass(frozen=True)
class Candidate:
    strategy: str
    alpha: float
    duration: int

    @property
    def name(self) -> str:
        return f"{self.strategy}_a{self.alpha:g}_d{self.duration}"

    def command(self, g, o, goal, base) -> np.ndarray:
        base = np.asarray(base, float)
        delta = DELTAS[self.strategy](g, o, goal, base)
        out = base.copy()
        out[:3] = np.clip(base[:3] + self.alpha * delta, -1.0, 1.0)
        return out


def candidate_grid(alphas, durations, strategies=STRATEGIES) -> list[Candidate]:
    unknown = set(strategies) - set(DELTAS)
    if unknown:
        raise ValueError(f"unknown strategies {sorted(unknown)}")
    return [Candidate(s, float(a), int(d)) for s in strategies for a in alphas for d in durations]
