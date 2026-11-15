# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/perturbations.py
# (introduced in 7e7f67a, last changed in 3b3f495). See PROVENANCE.md.
"""Perturbations applied inside a disturbance rollout.

Each perturbation owns its own ``numpy.random.Generator``, derived from
(perturbation_seed, episode_seed, stream id of its type, position among perturbations of the
same type). Noise is drawn at every decision step whether or not the perturbation is active,
so the draw at step t does not depend on onset or duration, and the same standardized noise is
reused across severities (common random numbers for paired comparisons).

Every perturbation records, per decision step, whether it was active and whether it actually
changed its signal; ``check_effect`` uses that to verify the perturbation did what it claims.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from farc.disturbances.config import STREAM_IDS, DisturbanceSpec
from farc.envs.observation_layout import FetchObservationLayout, masked_noise


class PerturbationEffectError(RuntimeError):
    pass


def perturbation_rng(
    perturbation_seed: int, episode_seed: int, disturbance_type: str, position: int
) -> np.random.Generator:
    sequence = np.random.SeedSequence(
        perturbation_seed, spawn_key=(episode_seed, STREAM_IDS[disturbance_type], position)
    )
    return np.random.default_rng(sequence)


class Perturbation:
    def __init__(self, spec: DisturbanceSpec, rng: np.random.Generator, horizon: int):
        self.spec = spec
        self.rng = rng
        self.horizon = horizon
        self.active_steps = np.zeros(horizon, dtype=bool)
        self.changed_steps = np.zeros(horizon, dtype=bool)

    def _record(self, step: int, before: np.ndarray, after: np.ndarray) -> None:
        self.active_steps[step] = self.spec.active(step)
        self.changed_steps[step] = not np.array_equal(before, after)

    def check_effect(self) -> None:
        name = self.spec.type
        outside = self.changed_steps & ~self.active_steps
        if outside.any():
            raise PerturbationEffectError(
                f"{name} changed its signal outside its window at steps "
                f"{np.flatnonzero(outside).tolist()}"
            )
        if self.active_steps.any() and not self.changed_steps.any():
            raise PerturbationEffectError(f"{name} was active but never changed its signal")


class ActionPerturbation(Perturbation):
    def apply(self, step: int, action: np.ndarray, base_history: list[np.ndarray]) -> np.ndarray:
        raise NotImplementedError


class ActionScale(ActionPerturbation):
    def apply(self, step, action, base_history):
        result = action * (1.0 - self.spec.severity) if self.spec.active(step) else action
        self._record(step, action, result)
        return result


class ActionNoise(ActionPerturbation):
    def apply(self, step, action, base_history):
        standard_noise = self.rng.standard_normal(action.shape)
        active = self.spec.active(step)
        result = action + self.spec.severity * standard_noise if active else action
        self._record(step, action, result)
        return result


class ActionDelay(ActionPerturbation):
    """While active, execute the base command issued ``severity`` decision steps earlier.

    Commands from before the episode started are zero (the arm holds still).
    """

    def apply(self, step, action, base_history):
        if self.spec.active(step):
            source = step - int(self.spec.severity)
            result = base_history[source].copy() if source >= 0 else np.zeros_like(action)
        else:
            result = action
        self._record(step, action, result)
        return result


class ObservationNoise(Perturbation):
    """Adds noise to the controller's copy of the observation only.

    The simulator state and the ground-truth achieved and desired goals used for every metric
    are never touched.
    """

    layout: FetchObservationLayout | None = None
    dt: float = 0.0

    def configure(self, layout: FetchObservationLayout, dt: float) -> None:
        self.layout = layout
        self.dt = dt

    def apply(self, step: int, view: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        noisy = {key: value.copy() for key, value in view.items()}
        draws = {key: self.rng.standard_normal(view[key].shape) for key in self.spec.fields}
        if self.spec.active(step):
            for key, draw in draws.items():
                if key == "observation" and self.spec.mask != "full":
                    if self.layout is None:
                        raise RuntimeError("observation mask needs configure(layout, dt)")
                    noise = masked_noise(
                        self.layout, self.spec.mask, self.spec.severity, draw, self.dt
                    )
                else:
                    noise = self.spec.severity * draw
                noisy[key] = noisy[key] + noise
        changed = any(not np.array_equal(view[key], noisy[key]) for key in view)
        self.active_steps[step] = self.spec.active(step)
        self.changed_steps[step] = changed
        return noisy


def sample_goal_with_rng(unwrapped: Any, rng: np.random.Generator) -> np.ndarray:
    """Draw a goal from the environment's own goal distribution with a separate generator.

    ``_sample_goal`` reads ``self.np_random``; the environment's generator is swapped out for
    the call and restored, so the environment's random stream is not advanced.
    """
    original = unwrapped._np_random
    unwrapped._np_random = rng
    try:
        goal = np.asarray(unwrapped._sample_goal(), dtype=np.float64).copy()
    finally:
        unwrapped._np_random = original
    return goal


def goal_in_distribution(unwrapped: Any, goal: np.ndarray, tolerance: float = 1e-9) -> bool:
    """Whether ``goal`` lies in the support of the environment's own goal distribution (the
    distribution the policy was trained on): a box of half-width ``target_range`` around the
    initial gripper position (plus ``target_offset`` with an object, at table height when the
    target is not in the air)."""
    center = np.asarray(unwrapped.initial_gripper_xpos[:3], dtype=np.float64).copy()
    if unwrapped.has_object:
        center = center + unwrapped.target_offset
        if not unwrapped.target_in_the_air and abs(goal[2] - unwrapped.height_offset) > tolerance:
            return False
        offset = np.abs(goal[:2] - center[:2])
    else:
        offset = np.abs(goal - center)
    return bool(np.all(offset <= unwrapped.target_range + tolerance))


class GoalShift(Perturbation):
    """At decision step ``onset``, replace the goal with a new goal sampled from the
    environment's goal distribution, rejecting draws closer than ``severity`` metres to the
    current goal."""

    def __init__(self, spec, rng, horizon):
        super().__init__(spec, rng, horizon)
        self.event: dict[str, Any] | None = None

    def maybe_shift(self, step: int, unwrapped: Any) -> np.ndarray | None:
        self.active_steps[step] = self.spec.active(step)
        if step != self.spec.onset:
            return None
        old_goal = np.asarray(unwrapped.goal, dtype=np.float64).copy()
        attempts = 0
        candidate, distance = old_goal, 0.0
        while distance < self.spec.severity and attempts < self.spec.max_attempts:
            attempts += 1
            candidate = sample_goal_with_rng(unwrapped, self.rng)
            distance = float(np.linalg.norm(candidate - old_goal))
        if distance < self.spec.severity:
            raise PerturbationEffectError(
                f"goal_shift found no goal at least {self.spec.severity} m from the current goal "
                f"in {self.spec.max_attempts} draws"
            )
        if not goal_in_distribution(unwrapped, candidate):
            raise PerturbationEffectError(f"sampled goal {candidate} is outside the goal range")
        unwrapped.goal = candidate.copy()
        self.changed_steps[step] = True
        self.event = {
            "step": step,
            "original_goal": old_goal,
            "new_goal": candidate.copy(),
            "shift_distance": distance,
            "attempts": attempts,
            "in_goal_distribution": True,
        }
        return candidate

    def check_effect(self) -> None:
        super().check_effect()
        if self.event is None:
            raise PerturbationEffectError("goal_shift never fired")
        if self.event["shift_distance"] < self.spec.severity:
            raise PerturbationEffectError("goal_shift moved the goal less than its severity")


ACTION_CLASSES: dict[str, type[ActionPerturbation]] = {
    "action_delay": ActionDelay,
    "action_scale": ActionScale,
    "action_noise": ActionNoise,
}


def build_perturbations(
    disturbances: tuple[DisturbanceSpec, ...],
    perturbation_seed: int,
    episode_seed: int,
    horizon: int,
) -> list[Perturbation]:
    positions: dict[str, int] = {}
    built: list[Perturbation] = []
    for spec in disturbances:
        position = positions.get(spec.type, 0)
        positions[spec.type] = position + 1
        rng = perturbation_rng(perturbation_seed, episode_seed, spec.type, position)
        cls: type[Perturbation]
        if spec.type in ACTION_CLASSES:
            cls = ACTION_CLASSES[spec.type]
        elif spec.type == "observation_noise":
            cls = ObservationNoise
        else:
            cls = GoalShift
        built.append(cls(spec, rng, horizon))
    return built
