"""Causal per-step features for learned failure prediction on FetchPush.

Logged trajectory schema (``trajectories.csv.gz`` from ``scripts/run_disturbance_eval.py``;
one row per state ``t = 0..T``, ``T = 50``):

| Columns | Meaning at row t |
| --- | --- |
| ``achieved_0..2`` | ground-truth object position (FetchPush ``achieved_goal`` = obs[3:6]) |
| ``gripper_0..2`` | ground-truth grip-site position (obs[0:3]), before observation noise |
| ``goal_0..2`` | goal active in the simulator when state t was observed |
| ``base_action_0..3`` | frozen-policy output at decision step t-1 (row 0: empty) |
| ``executed_action_0..3`` | action after perturbation and clipping at step t-1 (row 0: empty) |
| ``success`` | success after env step t-1 (row 0: success of the initial state) |

Action components 0..2 are gripper displacement commands (x, y, z), component 3 the finger
command (blocked in Push).

Decision step t (``t = 0..T-1``) sees state t. A goal shift scheduled for step s is applied
*before* the policy acts at step s, so the controller already uses the new goal at decision
s, but the log records it only from row s+1 (row s was written after step s-1). The goal
known at decision t is therefore ``goal[t+1]``; this is not future information (the
simulator goal is set before the decision), and it is what the base policy saw.

Causality rules followed here:

- Positions, velocities and progress use rows ``<= t`` only.
- Velocities are backward differences ``x[t] - x[t-1]`` (m per control step), not the
  instantaneous simulator velocities (obs[14:17], obs[20:23]), which are not logged.
- The previous action is the *base* (commanded) action of step t-1. The executed action is
  not used: under action scaling, delay or noise it encodes the disturbance directly.
- Nothing uses ``active_*`` flags, condition names, observation error, rewards, or success.

Missing early history: at t = 0 the velocities and previous action are 0 and
``has_previous`` is 0. Progress windows are truncated to the steps since the current goal
became active (reset at goal changes) and since the start of the episode; with an empty
window the progress is 0 and ``progress_window`` is 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HORIZON = 50
ACTION_DIM = 4

FEATURE_NAMES: tuple[str, ...] = (
    "object_to_goal_x",
    "object_to_goal_y",
    "object_to_goal_z",
    "gripper_to_object_x",
    "gripper_to_object_y",
    "gripper_to_object_z",
    "object_velocity_x",
    "object_velocity_y",
    "object_velocity_z",
    "gripper_velocity_x",
    "gripper_velocity_y",
    "gripper_velocity_z",
    "previous_action_0",
    "previous_action_1",
    "previous_action_2",
    "previous_action_3",
    "object_to_goal_distance",
    "gripper_to_object_distance",
    "remaining_time",
    "object_goal_progress",
    "approach_progress",
    "progress_window",
    "time_since_goal_change",
    "has_previous",
)
FEATURE_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}
NUM_FEATURES = len(FEATURE_NAMES)


@dataclass(frozen=True)
class FeatureConfig:
    progress_window: int = 5  # steps K for the recent progress features
    reset_approach_on_goal_change: bool = True
    goal_change_tolerance: float = 1e-9  # m

    def __post_init__(self) -> None:
        if self.progress_window < 1:
            raise ValueError("progress_window must be at least 1")


def decision_goals(goals: np.ndarray) -> np.ndarray:
    """Goal used by the controller at each decision step ``0..T-1`` (``goals`` is (T+1, 3))."""
    return np.asarray(goals, np.float64)[1:]


def goal_change_steps(goals: np.ndarray, tolerance: float = 1e-9) -> np.ndarray:
    """For each decision step, the decision step at which its goal became active."""
    decision = decision_goals(goals)
    since = np.zeros(len(decision), dtype=np.int64)
    for t in range(1, len(decision)):
        changed = np.max(np.abs(decision[t] - decision[t - 1])) > tolerance
        since[t] = t if changed else since[t - 1]
    return since


def episode_features(
    objects: np.ndarray,
    grippers: np.ndarray,
    goals: np.ndarray,
    base_actions: np.ndarray,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Features for decision steps ``0..T-1``; returns (T, NUM_FEATURES) float64.

    ``objects``, ``grippers``, ``goals``: (T+1, 3) rows of the trajectory log.
    ``base_actions``: (T, 4), the base action of decision step k in row k.
    Row t of the result depends only on states ``<= t``, the goal known at decision t,
    and base actions ``< t``.
    """
    config = config or FeatureConfig()
    objects = np.asarray(objects, np.float64)
    grippers = np.asarray(grippers, np.float64)
    base_actions = np.asarray(base_actions, np.float64)
    horizon = len(objects) - 1
    if grippers.shape != objects.shape or np.asarray(goals).shape != objects.shape:
        raise ValueError("objects, grippers and goals must all be (T+1, 3)")
    if base_actions.shape != (horizon, ACTION_DIM):
        raise ValueError(f"base_actions must be ({horizon}, {ACTION_DIM})")
    goal = decision_goals(goals)
    change = goal_change_steps(goals, config.goal_change_tolerance)
    out = np.zeros((horizon, NUM_FEATURES))
    K = config.progress_window
    for t in range(horizon):
        f = out[t]
        to_goal = goal[t] - objects[t]
        to_object = objects[t] - grippers[t]
        f[0:3] = to_goal
        f[3:6] = to_object
        if t > 0:
            f[6:9] = objects[t] - objects[t - 1]
            f[9:12] = grippers[t] - grippers[t - 1]
            f[12:16] = base_actions[t - 1]
            f[FEATURE_INDEX["has_previous"]] = 1.0
        f[16] = np.linalg.norm(to_goal)
        f[17] = np.linalg.norm(to_object)
        f[FEATURE_INDEX["remaining_time"]] = (horizon - t) / horizon
        f[FEATURE_INDEX["time_since_goal_change"]] = (t - change[t]) / horizon
        window = min(K, t - change[t])
        approach_window = window if config.reset_approach_on_goal_change else min(K, t)
        if window > 0:
            # distance of the earlier object position to the *current* goal: the goal was
            # constant over the window, so this equals the distance logged at that time
            earlier = np.linalg.norm(goal[t] - objects[t - window])
            f[FEATURE_INDEX["object_goal_progress"]] = (earlier - f[16]) / window
        if approach_window > 0:
            s = t - approach_window
            earlier = np.linalg.norm(objects[s] - grippers[s])
            f[FEATURE_INDEX["approach_progress"]] = (earlier - f[17]) / approach_window
        f[FEATURE_INDEX["progress_window"]] = window / K
    return out
