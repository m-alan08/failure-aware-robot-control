# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/rollout.py
# (introduced in 7e7f67a, last changed in 09ea4f4). See PROVENANCE.md.
"""One disturbed episode with a frozen policy.

What the controller sees: the ``observation`` and ``desired_goal`` fields, after observation
noise (if any) and normalization by the policy's frozen normalizer. After a goal shift the
controller sees the new goal from the shift step on.

What the metrics use: the simulator's ground-truth ``achieved_goal`` returned by ``env.step``,
the goal active in the simulator (``env.unwrapped.goal``), and ``info["is_success"]``. The
recorded success is checked against ``distance < distance_threshold`` on those ground-truth
values at every step.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from farc.base_policy.normalizer import GoalObservationNormalizer
from farc.disturbances.config import ACTION_TYPES, ConditionSpec
from farc.disturbances.perturbations import (
    ActionPerturbation,
    GoalShift,
    ObservationNoise,
    build_perturbations,
)
from farc.envs.observation_layout import control_period, layout_for, validate_layout

PolicyFunction = Callable[[np.ndarray], np.ndarray]


class RolloutContractError(RuntimeError):
    pass


def normalizer_digest(normalizer: GoalObservationNormalizer) -> str:
    digest = hashlib.sha256()
    state = normalizer.state_dict()
    digest.update(repr(state["obs_clip"]).encode())
    for part in ("observation", "goal"):
        for key in sorted(state[part]):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(np.asarray(state[part][key], np.float64)).tobytes())
    return digest.hexdigest()


class FrozenNormalizer:
    """A read-only copy of a policy normalizer. ``update`` raises, and ``verify`` checks the
    statistics are bit-identical to those it was created with."""

    def __init__(self, normalizer: GoalObservationNormalizer):
        self._normalizer = copy.deepcopy(normalizer)
        self.digest = normalizer_digest(self._normalizer)

    def __call__(self, observations: np.ndarray, goals: np.ndarray) -> np.ndarray:
        return self._normalizer(observations, goals)

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("The base policy normalizer is frozen during disturbance evaluation")

    def verify(self) -> None:
        if normalizer_digest(self._normalizer) != self.digest:
            raise RuntimeError("The frozen normalizer's statistics changed")


@dataclass
class EpisodeTrajectory:
    episode_seed: int
    horizon: int
    distance_threshold: float
    achieved_goals: np.ndarray  # (T+1, g) ground truth
    desired_goals: np.ndarray  # (T+1, g) goal active in the simulator when state k was observed
    original_goal: np.ndarray  # goal at reset
    distances: np.ndarray  # (T+1,) ||achieved - active goal||
    distances_to_original_goal: np.ndarray  # (T+1,) diagnostic only
    base_actions: np.ndarray  # (T, a) policy output on the controller's view
    pre_clip_actions: np.ndarray  # (T, a) after perturbations, before clipping
    executed_actions: np.ndarray  # (T, a) clipped to the action bounds, sent to env.step
    rewards: np.ndarray  # (T,)
    successes: np.ndarray  # (T,) bool, after env step k
    observation_error: np.ndarray  # (T,) ||controller view - ground truth|| at decision step
    active: dict[str, np.ndarray] = field(default_factory=dict)  # label -> (T,) bool
    goal_events: list[dict[str, Any]] = field(default_factory=list)
    initial_success: bool = False
    # (T+1, 3) ground-truth gripper (grip site) position, read from the simulator's
    # observation before any observation noise; layout validated against the simulator
    gripper_positions: np.ndarray | None = None


def _controller_view(observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "observation": np.asarray(observation["observation"], np.float64).copy(),
        "desired_goal": np.asarray(observation["desired_goal"], np.float64).copy(),
    }


def run_disturbed_episode(
    env: gym.Env,
    policy: PolicyFunction,
    normalizer: FrozenNormalizer | GoalObservationNormalizer,
    condition: ConditionSpec,
    episode_seed: int,
    perturbation_seed: int,
) -> EpisodeTrajectory:
    unwrapped: Any = env.unwrapped
    if env.spec is None or env.spec.max_episode_steps is None:
        raise RolloutContractError("environment needs a time limit")
    horizon = int(env.spec.max_episode_steps)
    threshold = float(unwrapped.distance_threshold)
    low = np.asarray(env.action_space.low, np.float64)  # type: ignore[attr-defined]
    high = np.asarray(env.action_space.high, np.float64)  # type: ignore[attr-defined]
    action_dim = low.shape[0]

    perturbations = build_perturbations(
        condition.disturbances, perturbation_seed, episode_seed, horizon
    )
    action_perturbations = sorted(
        (p for p in perturbations if isinstance(p, ActionPerturbation)),
        key=lambda p: ACTION_TYPES.index(p.spec.type),
    )
    observation_perturbations = [p for p in perturbations if isinstance(p, ObservationNoise)]
    goal_perturbations = [p for p in perturbations if isinstance(p, GoalShift)]

    observation, _ = env.reset(seed=episode_seed)
    layout = layout_for(unwrapped, np.asarray(observation["observation"]).shape[0])
    validate_layout(unwrapped, observation["observation"], layout)
    for noise_perturbation in observation_perturbations:
        noise_perturbation.configure(layout, control_period(unwrapped))
    goal_dim = np.asarray(observation["desired_goal"]).shape[0]
    achieved = np.zeros((horizon + 1, goal_dim))
    desired = np.zeros((horizon + 1, goal_dim))
    base_actions = np.zeros((horizon, action_dim))
    pre_clip = np.zeros((horizon, action_dim))
    executed = np.zeros((horizon, action_dim))
    rewards = np.zeros(horizon)
    successes = np.zeros(horizon, dtype=bool)
    observation_error = np.zeros(horizon)
    base_history: list[np.ndarray] = []

    original_goal = np.asarray(unwrapped.goal, np.float64).copy()
    gripper_slice = layout.slices["gripper_position"]
    gripper = np.zeros((horizon + 1, 3))
    gripper[0] = observation["observation"][gripper_slice]
    achieved[0] = observation["achieved_goal"]
    desired[0] = unwrapped.goal
    if not np.allclose(observation["desired_goal"], unwrapped.goal):
        raise RolloutContractError("observation desired_goal differs from the simulator goal")

    for step in range(horizon):
        for shift in goal_perturbations:
            new_goal = shift.maybe_shift(step, unwrapped)
            if new_goal is not None:
                observation = dict(observation)
                observation["desired_goal"] = new_goal.copy()
        truth = _controller_view(observation)
        view = truth
        for noise in observation_perturbations:
            view = noise.apply(step, view)
        observation_error[step] = float(
            np.sqrt(sum(np.sum((view[k] - truth[k]) ** 2) for k in truth))
        )
        inputs = normalizer(view["observation"], view["desired_goal"])
        base = np.asarray(policy(inputs[None])[0], np.float64)
        base_actions[step] = base
        base_history.append(base.copy())
        action = base
        for action_perturbation in action_perturbations:
            action = action_perturbation.apply(step, action, base_history)
        pre_clip[step] = action
        executed[step] = np.clip(action, low, high)

        observation, reward, terminated, truncated, info = env.step(
            executed[step].astype(np.float32)
        )
        last = step == horizon - 1
        if terminated or (truncated and not last) or (last and not truncated):
            raise RolloutContractError(
                f"episode ended unexpectedly at step {step + 1} "
                f"(terminated={terminated}, truncated={truncated}, horizon={horizon})"
            )
        achieved[step + 1] = observation["achieved_goal"]
        gripper[step + 1] = observation["observation"][gripper_slice]
        desired[step + 1] = unwrapped.goal
        if not np.allclose(observation["desired_goal"], unwrapped.goal):
            raise RolloutContractError("observation desired_goal differs from the simulator goal")
        rewards[step] = float(reward)
        successes[step] = bool(info["is_success"])

    validate_layout(unwrapped, observation["observation"], layout)
    distances = np.linalg.norm(achieved - desired, axis=-1)
    recomputed = distances[1:] < threshold
    if not np.array_equal(recomputed, successes):
        raise RolloutContractError(
            "info['is_success'] disagrees with the ground-truth distance to the active goal"
        )
    expected_rewards = -(~successes).astype(float)
    if not np.array_equal(expected_rewards, rewards):
        raise RolloutContractError("rewards are not the sparse -1/0 reward on the active goal")
    for perturbation in perturbations:
        perturbation.check_effect()
    if isinstance(normalizer, FrozenNormalizer):
        normalizer.verify()

    labels: dict[str, np.ndarray] = {}
    for perturbation in perturbations:
        label = perturbation.spec.type
        suffix = 1
        while label in labels:
            suffix += 1
            label = f"{perturbation.spec.type}_{suffix}"
        labels[label] = perturbation.active_steps.copy()
    return EpisodeTrajectory(
        episode_seed=episode_seed,
        horizon=horizon,
        distance_threshold=threshold,
        achieved_goals=achieved,
        desired_goals=desired,
        original_goal=original_goal,
        distances=distances,
        distances_to_original_goal=np.linalg.norm(achieved - original_goal, axis=-1),
        base_actions=base_actions,
        pre_clip_actions=pre_clip,
        executed_actions=executed,
        rewards=rewards,
        successes=successes,
        observation_error=observation_error,
        active=labels,
        goal_events=[shift.event for shift in goal_perturbations if shift.event is not None],
        initial_success=bool(distances[0] < threshold),
        gripper_positions=gripper,
    )
