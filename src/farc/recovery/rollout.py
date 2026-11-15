"""Closed-loop disturbed episode with a frozen base policy, a trigger and a recovery controller.

Follows ``farc.disturbances.rollout.run_disturbed_episode`` step for step (same reset seed,
same perturbation generators, same order: goal shift, observation noise, decision, action
perturbations, clipping), so with recovery never active the executed actions are identical to
Phase 1's. Differences:

- The command passed to the action perturbations is whichever controller acts (base or
  recovery). Actuator disturbances (scale, delay, noise) are properties of the robot, so they
  affect recovery commands too; ``action_delay`` replays the *issued* command history.
- The base policy is evaluated at every step, also while recovery acts, because the learned
  predictor's ``previous_action`` feature is defined as the base policy's output.
- Triggers and the recovery controller see the controller view (after observation noise), the
  same information as the base policy at that step. Metrics use simulator ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

from farc.disturbances.config import ACTION_TYPES, ConditionSpec
from farc.disturbances.perturbations import (
    ActionPerturbation,
    GoalShift,
    ObservationNoise,
    build_perturbations,
)
from farc.disturbances.rollout import (
    FrozenNormalizer,
    PolicyFunction,
    RolloutContractError,
    _controller_view,
)
from farc.envs.observation_layout import control_period, layout_for, validate_layout
from farc.recovery.controller import (
    InterventionSwitch,
    OnlineHistory,
    RecoveryController,
    Trigger,
)

CONFIGURATIONS = ("base_only", "always_on", "reactive_triggered", "learned_triggered")


@dataclass
class ClosedLoopEpisode:
    episode_seed: int
    horizon: int
    final_success: bool
    final_distance: float
    successes: np.ndarray  # (T,) after each step, ground truth
    recovery_active: np.ndarray  # (T,) bool
    modes: list[str]
    scores: np.ndarray  # (T,) trigger score at each decision (NaN without a trigger)
    executed_actions: np.ndarray
    base_actions: np.ndarray
    interventions: list[dict[str, Any]] = field(default_factory=list)
    first_trigger_step: int | None = None


def run_closed_loop_episode(
    env: gym.Env,
    policy: PolicyFunction,
    normalizer: FrozenNormalizer,
    condition: ConditionSpec,
    episode_seed: int,
    perturbation_seed: int,
    switch: InterventionSwitch | None,
    trigger: Trigger | None,
    recovery: RecoveryController | None,
) -> ClosedLoopEpisode:
    unwrapped: Any = env.unwrapped
    if env.spec is None or env.spec.max_episode_steps is None:
        raise RolloutContractError("environment needs a time limit")
    horizon = int(env.spec.max_episode_steps)
    threshold = float(unwrapped.distance_threshold)
    low = np.asarray(env.action_space.low, np.float64)  # type: ignore[attr-defined]
    high = np.asarray(env.action_space.high, np.float64)  # type: ignore[attr-defined]
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
    gs, os_ = layout.slices["gripper_position"], layout.slices["object_position"]
    if switch is not None:
        switch.reset()
    if recovery is not None:
        recovery.reset()
    if trigger is not None:
        trigger.reset()
    history = OnlineHistory(horizon)
    command_history: list[np.ndarray] = []
    successes = np.zeros(horizon, dtype=bool)
    active = np.zeros(horizon, dtype=bool)
    scores = np.full(horizon, np.nan)
    modes: list[str] = []
    executed = np.zeros((horizon, 4))
    base_actions = np.zeros((horizon, 4))
    first_trigger: int | None = None
    distance = float("nan")
    for step in range(horizon):
        for shift in goal_perturbations:
            new_goal = shift.maybe_shift(step, unwrapped)
            if new_goal is not None:
                observation = dict(observation)
                observation["desired_goal"] = new_goal.copy()
        view = _controller_view(observation)
        for noise in observation_perturbations:
            view = noise.apply(step, view)
        g, o, goal = view["observation"][gs], view["observation"][os_], view["desired_goal"]
        base = np.asarray(policy(normalizer(view["observation"], goal)[None])[0], np.float64)
        base_actions[step] = base
        history.record(step, g, o, goal, base_actions[step - 1] if step else None)
        fired = False
        if trigger is not None:
            fired, scores[step] = trigger.fires(history)
            if fired and first_trigger is None:
                first_trigger = step
        use_recovery = False
        if switch is not None:
            use_recovery, started = switch.step(
                step, horizon, fired, float(np.linalg.norm(o - goal))
            )
            if started and recovery is not None:
                recovery.reset()
        if use_recovery:
            if recovery is None:
                raise ValueError("a switch without a recovery controller")
            command, mode = recovery.act(g, o, goal)
        else:
            command, mode = base, "base"
        active[step], modes = use_recovery, [*modes, mode]
        command_history.append(command.copy())
        action = command
        for perturbation in action_perturbations:
            action = perturbation.apply(step, action, command_history)
        executed[step] = np.clip(action, low, high)
        observation, _, terminated, truncated, info = env.step(executed[step].astype(np.float32))
        last = step == horizon - 1
        if terminated or (truncated and not last) or (last and not truncated):
            raise RolloutContractError(f"episode ended unexpectedly at step {step + 1}")
        successes[step] = bool(info["is_success"])
        distance = float(np.linalg.norm(observation["achieved_goal"] - unwrapped.goal))
        if successes[step] != (distance < threshold):
            raise RolloutContractError("info['is_success'] disagrees with the ground truth")
    normalizer.verify()
    return ClosedLoopEpisode(
        episode_seed=episode_seed,
        horizon=horizon,
        final_success=bool(successes[-1]),
        final_distance=distance,
        successes=successes,
        recovery_active=active,
        modes=modes,
        scores=scores,
        executed_actions=executed,
        base_actions=base_actions,
        interventions=[dict(i) for i in switch.interventions] if switch else [],
        first_trigger_step=first_trigger,
    )
