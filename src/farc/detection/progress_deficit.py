# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/detector.py
# (introduced in 934063d, last changed in 934063d). See PROVENANCE.md.
"""Offline evaluation of a reactive, deadline-aware progress-deficit detector.

The detector sees only the ground-truth distance between the achieved goal (the gripper for
FetchReach, the object for FetchPush) and the goal that is active for the current decision.
It is evaluated on logged trajectories; it never acts on the environment.

At decision state t (0 <= t < T, where T is the horizon):

- d_t = ||achieved_t - g_t||, where g_t is the goal active for the decision taken at state t.
  For a goal shift at onset o this is the new goal from state o on.
- observed progress over k steps: P_obs = ||achieved_{t-k} - g_t|| - d_t (positive = closer).
- required progress over k steps: P_req = k * max(d_t - eps, 0) / (T - t), the share of the
  remaining distance to the success radius eps that must be covered per k steps to reach it
  by the deadline at a constant rate.
- deficit = P_req - P_obs; the step is flagged when deficit > threshold.
- suppression: no flag while d_t < eps (inside the success radius); the persistence count
  resets.
- reset: the window may not reach back before the most recent goal change (or the reset),
  so the first k states after a goal shift are never flagged and the count resets.
- warning: the first state at which the flag has held for ``persistence`` consecutive
  decision states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Largest end-effector displacement per control step: the Fetch action maps [-1, 1] to a mocap
# position change of at most 0.05 m per axis (gymnasium_robotics fetch_env._set_action).
MAX_AXIS_STEP_DISPLACEMENT = 0.05


@dataclass(frozen=True)
class DetectorConfig:
    window: int
    persistence: int
    deficit_threshold: float = 0.0
    success_threshold: float = 0.05

    def __post_init__(self) -> None:
        if not (isinstance(self.window, int) and self.window >= 1):
            raise ValueError(f"window must be a positive integer, got {self.window!r}")
        if not (isinstance(self.persistence, int) and self.persistence >= 1):
            raise ValueError(f"persistence must be a positive integer, got {self.persistence!r}")
        if not self.success_threshold > 0:
            raise ValueError("success_threshold must be positive")

    @property
    def label(self) -> str:
        return f"k{self.window}_p{self.persistence}_th{self.deficit_threshold:g}"


@dataclass
class DetectorTrace:
    decision_distance: np.ndarray  # (T,) d_t against the goal active for decision t
    observed: np.ndarray  # (T,) NaN where the window is unavailable
    required: np.ndarray  # (T,)
    deficit: np.ndarray  # (T,)
    flagged: np.ndarray  # (T,) bool
    first_warning: int | None  # decision state index, or None


def decision_goals(goals: np.ndarray) -> np.ndarray:
    """Goal active for the decision at each state 0..T-1, from goals logged per state 0..T.

    The logged goal at state t is the goal the simulator held when state t was observed; a goal
    shift at decision o appears from state o + 1. The decision at state t uses goal t + 1.
    """
    goals = np.asarray(goals, dtype=float)
    return goals[1:]


def segment_starts(decision_goal: np.ndarray, atol: float = 1e-9) -> np.ndarray:
    """For each decision state, the first state of the current constant-goal segment."""
    starts = np.zeros(len(decision_goal), dtype=int)
    for t in range(1, len(decision_goal)):
        changed = not np.allclose(decision_goal[t], decision_goal[t - 1], atol=atol)
        starts[t] = t if changed else starts[t - 1]
    return starts


def run_detector(achieved: np.ndarray, goals: np.ndarray, config: DetectorConfig) -> DetectorTrace:
    """``achieved`` and ``goals`` are (T + 1, 3) ground-truth arrays for states 0..T."""
    achieved = np.asarray(achieved, dtype=float)
    goal_for_decision = decision_goals(goals)
    horizon = len(goal_for_decision)
    if achieved.shape[0] != horizon + 1:
        raise ValueError("achieved must have one row per state 0..T")
    starts = segment_starts(goal_for_decision)
    eps = config.success_threshold
    k = config.window
    distance = np.linalg.norm(achieved[:horizon] - goal_for_decision, axis=1)
    observed = np.full(horizon, np.nan)
    required = np.full(horizon, np.nan)
    deficit = np.full(horizon, np.nan)
    flagged = np.zeros(horizon, dtype=bool)
    count = 0
    first_warning = None
    for t in range(horizon):
        if t > 0 and starts[t] != starts[t - 1]:
            count = 0
        if distance[t] < eps or t - k < starts[t]:
            count = 0
            continue
        past = float(np.linalg.norm(achieved[t - k] - goal_for_decision[t]))
        observed[t] = past - distance[t]
        required[t] = k * max(distance[t] - eps, 0.0) / (horizon - t)
        deficit[t] = required[t] - observed[t]
        flagged[t] = deficit[t] > config.deficit_threshold
        count = count + 1 if flagged[t] else 0
        if count >= config.persistence and first_warning is None:
            first_warning = t
    return DetectorTrace(distance, observed, required, deficit, flagged, first_warning)


def kinematically_infeasible(distance_to_go: float, remaining_steps: int) -> bool:
    """True if the remaining distance to the success radius exceeds what the end effector can
    cover in the remaining steps at full speed along a diagonal (sqrt(3) * 0.05 m per step).

    This is a loose necessary condition: for FetchPush the object moves slower than the
    gripper, so many feasible-looking warnings are still too late in practice.
    """
    reach = np.sqrt(3.0) * MAX_AXIS_STEP_DISPLACEMENT * remaining_steps
    return bool(distance_to_go > reach)
