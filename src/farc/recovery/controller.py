"""Deterministic push-recovery controller and online failure triggers (FetchPush-v4).

Recovery controller
-------------------
Inputs, all read from the controller's (possibly noisy) observation at the current decision
step, i.e. what the base policy itself sees: gripper position g = obs[0:3], object position
o = obs[3:6], goal G = desired_goal. No simulator state, contact flags or future information.

Action: Fetch position command a[0:3] in [-1, 1], one unit = 0.05 m of mocap displacement per
step (``fetch_env``: ``pos_ctrl *= 0.05``); a[3] (fingers) is 0, the gripper is blocked in
Push. Proportional rule ``a = clip((target - g) / 0.05, -1, 1)``.

Geometry, with u the horizontal unit vector from o to G:

- pre-push point p = o - standoff * u, at the object's height;
- *behind*: (g - o)·u < -behind_min and the lateral offset from the o→G line < lateral_tol
  and |g_z - o_z| < height_tol (looser tolerances once pushing, for hysteresis).

Modes (memory is reset at every activation):

- HOLD: ||o - G||_xy < hold_radius: command 0.
- PUSH: behind the object: target = the gripper's projection on the o->G line, advanced by
  min(push_lead, ||o - G||): the gripper steers onto the line and stops advancing once the
  object reaches the goal.
- REPOSITION: otherwise. If the gripper is low and the straight path to p passes within
  clearance of the object, first LIFT to o_z + lift_height; at that height move horizontally
  to p; once within reach_tol of p horizontally, DESCEND to the object's height.

Triggered recovery protocol (``InterventionSwitch``)
----------------------------------------------------
- Activation: at decision t, if not active, not in cooldown, the episode has more than
  ``min_steps_left`` steps left, and the trigger fires at t, recovery acts from step t on.
- Termination: (a) success: ||o - G|| < exit_radius for ``exit_steps`` consecutive decisions
  (measured on the controller view) -> base policy from the next step; (b) timeout after
  ``max_duration`` recovery steps -> base policy; (c) the episode ends.
- After termination the trigger is ignored for ``cooldown`` steps, then may re-fire (each
  firing is one intervention).
- Always-on: active at every step, never terminates (HOLD keeps the object at the goal).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

from farc.detection.phase_aware import PhaseDetectorConfig, run_phase_detector
from farc.detection.threshold_sweep import run_minimum
from farc.prediction.dataset import Normalizer
from farc.prediction.evaluation import FailurePredictor, predict_probabilities
from farc.prediction.features import FeatureConfig, episode_features

STEP = 0.05  # metres of mocap displacement per unit action


@dataclass(frozen=True)
class RecoveryConfig:
    standoff: float = 0.06
    behind_min: float = 0.03
    lateral_tol: float = 0.015
    height_tol: float = 0.02
    push_behind_min: float = 0.015
    push_lateral_tol: float = 0.03
    push_lead: float = 0.03
    hold_radius: float = 0.015
    lift_height: float = 0.07
    clearance: float = 0.05
    reach_tol: float = 0.015


REPOSITION, PUSH, HOLD = "reposition", "push", "hold"


@dataclass
class RecoveryController:
    config: RecoveryConfig = field(default_factory=RecoveryConfig)
    pushing: bool = False

    def reset(self) -> None:
        self.pushing = False

    def act(self, gripper: ArrayLike, obj: ArrayLike, goal: ArrayLike) -> tuple[np.ndarray, str]:
        c = self.config
        g, o, G = (np.asarray(v, np.float64) for v in (gripper, obj, goal))
        to_goal = G[:2] - o[:2]
        dist = float(np.linalg.norm(to_goal))
        if dist < c.hold_radius:
            self.pushing = False
            return np.zeros(4), HOLD
        u = to_goal / dist
        rel = g[:2] - o[:2]
        along = float(rel @ u)
        lateral = float(abs(rel[0] * u[1] - rel[1] * u[0]))
        low = abs(g[2] - o[2]) < c.height_tol
        if self.pushing:
            behind = along < -c.push_behind_min and lateral < c.push_lateral_tol and low
        else:
            behind = along < -c.behind_min and lateral < c.lateral_tol and low
        self.pushing = behind
        if behind:
            # advance along the o->G line by at most the distance still to go, and steer onto
            # the line; the gripper stops advancing once the object reaches the goal
            target = np.array([*(o[:2] + (along + min(c.push_lead, dist)) * u), o[2]])
            return _command(target, g), PUSH
        pre = o[:2] - c.standoff * u
        high_z = o[2] + c.lift_height
        if np.linalg.norm(g[:2] - pre) < c.reach_tol:
            target = np.array([*pre, o[2]])  # descend behind the object
        elif (
            g[2] < o[2] + 0.5 * c.lift_height and _segment_distance(g[:2], pre, o[:2]) < c.clearance
        ):
            target = np.array([*g[:2], high_z])  # lift before crossing the object
        else:
            z = high_z if g[2] >= o[2] + 0.5 * c.lift_height else g[2]
            target = np.array([*pre, z])
        return _command(target, g), REPOSITION


def _command(target: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    action = np.zeros(4)
    action[:3] = np.clip((target - gripper) / STEP, -1.0, 1.0)
    return action


def _segment_distance(a: np.ndarray, b: np.ndarray, p: np.ndarray) -> float:
    """Distance from p to the segment a-b (2-D)."""
    ab = b - a
    denom = float(ab @ ab)
    s = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(a + s * ab - p))


# --- online triggers ----------------------------------------------------------------------


class OnlineHistory:
    """Per-episode history of what the controller has seen, in the offline log layout.

    ``record(t, gripper, obj, goal, base_action_prev)`` is called at decision t with the
    controller's view at t: g(t), o(t), the goal used at decision t, and the base policy's
    output at t - 1. Arrays are padded with the latest values, so a feature or deficit of step
    t can be computed with the offline (full-length) functions and never sees later steps.
    """

    def __init__(self, horizon: int):
        self.horizon = horizon
        self.objects = np.zeros((horizon + 1, 3))
        self.grippers = np.zeros((horizon + 1, 3))
        self.goals = np.zeros((horizon + 1, 3))  # offline layout: decision goal t at row t+1
        self.base_actions = np.zeros((horizon, 4))
        self.t = -1

    def record(self, t: int, gripper, obj, goal, base_action_prev=None) -> None:
        if t != self.t + 1:
            raise ValueError("decision steps must be recorded in order")
        self.t = t
        self.objects[t:] = obj
        self.grippers[t:] = gripper
        self.goals[t + 1 :] = goal
        if t == 0:
            self.goals[0] = goal
        elif base_action_prev is not None:
            self.base_actions[t - 1 :] = base_action_prev


class Trigger:
    name = "trigger"

    def reset(self) -> None:  # noqa: B027
        pass

    def score(self, history: OnlineHistory) -> float:
        raise NotImplementedError

    threshold: float = 0.0

    def fires(self, history: OnlineHistory) -> tuple[bool, float]:
        s = self.score(history)
        return bool(s > self.threshold), s


class LearnedTrigger(Trigger):
    """Frozen MLP failure probability of the current step's 24 features."""

    name = "learned"

    def __init__(self, model: FailurePredictor, normalizer: Normalizer, threshold: float,
                 config: FeatureConfig | None = None):  # fmt: skip
        self.model, self.normalizer, self.threshold = model, normalizer, threshold
        self.config = config or FeatureConfig()

    def features(self, history: OnlineHistory) -> np.ndarray:
        h = history
        return episode_features(h.objects, h.grippers, h.goals, h.base_actions, self.config)[h.t]

    def score(self, history: OnlineHistory) -> float:
        x = self.normalizer(self.features(history))[None, None]
        return float(predict_probabilities(self.model, x)[0, 0])


class PhaseAwareTrigger(Trigger):
    """Phase-aware detector (departure monitor off): r_t = min deficit over the last p
    decision states, warning when r_t > threshold."""

    name = "phase_aware"

    def __init__(self, config: PhaseDetectorConfig, threshold: float):
        self.config, self.threshold = config, threshold

    def score(self, history: OnlineHistory) -> float:
        h = history
        trace = run_phase_detector(h.grippers, h.objects, h.goals, self.config)
        return float(run_minimum(trace.deficit, self.config.persistence)[h.t])


@dataclass(frozen=True)
class SwitchConfig:
    max_duration: int = 20
    exit_radius: float = 0.03
    exit_steps: int = 2
    cooldown: int = 5
    min_steps_left: int = 1


@dataclass
class InterventionSwitch:
    """Decides, per decision step, whether recovery or the base policy acts."""

    config: SwitchConfig = field(default_factory=SwitchConfig)
    always_on: bool = False
    active: bool = False
    steps_active: int = 0
    inside_streak: int = 0
    cooldown_left: int = 0
    interventions: list[dict[str, int | str]] = field(default_factory=list)

    def reset(self) -> None:
        self.active, self.steps_active, self.inside_streak, self.cooldown_left = False, 0, 0, 0
        self.interventions = []

    def step(
        self, t: int, horizon: int, fired: bool, object_goal_distance: float
    ) -> tuple[bool, bool]:
        """Returns (recovery acts at t, a new intervention started at t)."""
        if self.always_on:
            if t == 0:
                self.interventions.append({"start": 0, "end": horizon, "reason": "always_on"})
            return True, t == 0
        c = self.config
        if self.active:
            self.inside_streak = (
                self.inside_streak + 1 if object_goal_distance < c.exit_radius else 0
            )
            reason = None
            if self.inside_streak >= c.exit_steps:
                reason = "success"
            elif self.steps_active >= c.max_duration:
                reason = "timeout"
            if reason is not None:
                self.active = False
                self.interventions[-1].update(end=t, reason=reason)
                self.cooldown_left = c.cooldown
        started = False
        if not self.active:
            if self.cooldown_left > 0:
                self.cooldown_left -= 1
            elif fired and horizon - t > c.min_steps_left:
                self.active, self.steps_active, self.inside_streak = True, 0, 0
                self.interventions.append({"start": t, "end": horizon, "reason": "episode_end"})
                started = True
        if self.active:
            self.steps_active += 1
        return self.active, started
