# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/phase_detector.py
# (introduced in 98c903e, last changed in 98c903e). See PROVENANCE.md.
"""Phase-aware reactive failure detector for object pushing (FetchPush).

All quantities are ground truth from logged trajectories: gripper (grip site) position g,
object position o, and the goal G active for the decision at each state (see
``farc.detection.progress_deficit.decision_goals``). Distances are in metres.

Phases, decided at every decision state t:

- MAINTAIN: the object is inside the success radius, ||o - G|| < eps. No progress test.
- MANIPULATE: the gripper is in pushing range of the object. Entered from APPROACH when
  ||g - o|| <= contact_enter for ``hysteresis`` consecutive states; left for APPROACH when
  ||g - o|| > contact_exit for ``hysteresis`` consecutive states (contact_exit > contact_enter).
  This is how the detector returns to APPROACH when the gripper loses the object or
  repositions.
- APPROACH: otherwise.

The contact tracker runs at every state, including MAINTAIN, so leaving the success radius
resumes in whichever of APPROACH or MANIPULATE the geometry indicates.

Progress test (same deadline-aware deficit as the distance-only detector, on a phase-specific
remaining path L):

- APPROACH: L = ||g - p*|| + max(||o - G|| - eps, 0), where p* = o - behind_offset * u is the
  pre-push point behind the object and u is the horizontal unit vector from the object toward
  the goal. Gripper travel toward p* counts as progress, so an unmoved object is not a failure
  by itself, but an approach that is too slow for the deadline still is.
- MANIPULATE: L = max(||o - G|| - eps, 0), the object's remaining distance.

observed = L(state t - k) - L(state t), both with the goal and phase of state t;
required = k * L(t) / (T - t); flagged when required - observed > deficit_threshold. The
window may not reach back across a phase change or a goal change, and the persistence count
resets at those changes.

Departure monitor (goal maintenance): after the object has been inside the success radius of
the current goal, a warning is raised once it has been outside for ``departure_persistence``
consecutive states (None disables the monitor).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from farc.detection.progress_deficit import decision_goals, segment_starts

APPROACH, MANIPULATE, MAINTAIN = "approach", "manipulate", "maintain"


@dataclass(frozen=True)
class PhaseDetectorConfig:
    window: int
    persistence: int
    hysteresis: int = 3
    departure_persistence: int | None = 3
    contact_enter: float = 0.06
    contact_exit: float = 0.09
    behind_offset: float = 0.05
    deficit_threshold: float = 0.0
    success_threshold: float = 0.05

    def __post_init__(self) -> None:
        for name in ("window", "persistence", "hysteresis"):
            value = getattr(self, name)
            if not (isinstance(value, int) and not isinstance(value, bool) and value >= 1):
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.departure_persistence is not None and not (
            isinstance(self.departure_persistence, int) and self.departure_persistence >= 1
        ):
            raise ValueError("departure_persistence must be null or a positive integer")
        if not 0 < self.contact_enter < self.contact_exit:
            raise ValueError("need 0 < contact_enter < contact_exit")
        if self.behind_offset < 0 or self.success_threshold <= 0:
            raise ValueError("behind_offset must be >= 0 and success_threshold > 0")

    @property
    def label(self) -> str:
        departure = "off" if self.departure_persistence is None else self.departure_persistence
        return (
            f"phase_k{self.window}_p{self.persistence}_h{self.hysteresis}_m{departure}"
            f"_th{self.deficit_threshold:g}"
        )


@dataclass
class PhaseTrace:
    phases: list[str]  # (T,) phase at each decision state
    remaining_path: np.ndarray  # (T,) L at t, NaN in MAINTAIN
    deficit: np.ndarray  # (T,) NaN where unavailable
    flagged: np.ndarray  # (T,) bool, deficit flags
    departure_count: np.ndarray  # (T,) consecutive states outside the radius after success
    first_warning: int | None
    warning_kind: str | None  # "deficit" or "departure"
    warning_phase: str | None


def prepush_point(obj: np.ndarray, goal: np.ndarray, offset: float) -> np.ndarray:
    direction = np.asarray(goal, float) - np.asarray(obj, float)
    direction[2] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm < 1e-9:
        return np.asarray(obj, float).copy()
    return np.asarray(obj, float) - offset * direction / norm


def remaining_path(
    gripper: np.ndarray, obj: np.ndarray, goal: np.ndarray, phase: str, config: PhaseDetectorConfig
) -> float:
    object_to_go = max(float(np.linalg.norm(obj - goal)) - config.success_threshold, 0.0)
    if phase == MANIPULATE:
        return object_to_go
    if phase == APPROACH:
        approach = float(np.linalg.norm(gripper - prepush_point(obj, goal, config.behind_offset)))
        return approach + object_to_go
    raise ValueError(f"no remaining path in phase {phase!r}")


def contact_phases(gripper_object_distance: np.ndarray, config: PhaseDetectorConfig) -> list[str]:
    """APPROACH/MANIPULATE with enter/exit thresholds and ``hysteresis`` consecutive states."""
    phases: list[str] = []
    current = MANIPULATE if gripper_object_distance[0] <= config.contact_enter else APPROACH
    streak = 0
    for distance in gripper_object_distance:
        if current == APPROACH:
            streak = streak + 1 if distance <= config.contact_enter else 0
            if streak >= config.hysteresis:
                current, streak = MANIPULATE, 0
        else:
            streak = streak + 1 if distance > config.contact_exit else 0
            if streak >= config.hysteresis:
                current, streak = APPROACH, 0
        phases.append(current)
    return phases


def run_phase_detector(
    gripper: np.ndarray, objects: np.ndarray, goals: np.ndarray, config: PhaseDetectorConfig
) -> PhaseTrace:
    """``gripper``, ``objects`` and ``goals`` are (T + 1, 3) arrays for states 0..T."""
    gripper = np.asarray(gripper, float)
    objects = np.asarray(objects, float)
    goal_for_decision = decision_goals(goals)
    horizon = len(goal_for_decision)
    eps = config.success_threshold
    goal_starts = segment_starts(goal_for_decision)
    gripper_object = np.linalg.norm(gripper[:horizon] - objects[:horizon], axis=1)
    object_goal = np.linalg.norm(objects[:horizon] - goal_for_decision, axis=1)
    contact = contact_phases(gripper_object, config)
    phases = [MAINTAIN if object_goal[t] < eps else contact[t] for t in range(horizon)]

    path = np.full(horizon, np.nan)
    deficit = np.full(horizon, np.nan)
    flagged = np.zeros(horizon, dtype=bool)
    departure = np.zeros(horizon, dtype=int)
    count = 0
    segment_start = 0
    was_inside = False
    departure_streak = 0
    first_warning: int | None = None
    kind: str | None = None
    for t in range(horizon):
        goal = goal_for_decision[t]
        if t > 0 and goal_starts[t] != goal_starts[t - 1]:
            was_inside = False  # success against the old goal does not count
            departure_streak = 0
        if t == 0 or phases[t] != phases[t - 1] or goal_starts[t] != goal_starts[t - 1]:
            segment_start = t
            count = 0
        inside = object_goal[t] < eps
        if inside:
            was_inside = True
            departure_streak = 0
        elif was_inside:
            departure_streak += 1
        departure[t] = departure_streak
        if phases[t] != MAINTAIN:
            path[t] = remaining_path(gripper[t], objects[t], goal, phases[t], config)
            if t - config.window >= segment_start:
                past = remaining_path(
                    gripper[t - config.window], objects[t - config.window], goal, phases[t], config
                )
                required = config.window * path[t] / (horizon - t)
                deficit[t] = required - (past - path[t])
                flagged[t] = deficit[t] > config.deficit_threshold
            count = count + 1 if flagged[t] else 0
        if first_warning is None:
            if count >= config.persistence:
                first_warning, kind = t, "deficit"
            elif (
                config.departure_persistence is not None
                and departure_streak >= config.departure_persistence
            ):
                first_warning, kind = t, "departure"
    return PhaseTrace(
        phases=phases,
        remaining_path=path,
        deficit=deficit,
        flagged=flagged,
        departure_count=departure,
        first_warning=first_warning,
        warning_kind=kind,
        warning_phase=phases[first_warning] if first_warning is not None else None,
    )
