"""A disturbed FetchPush episode that can be snapshotted and restored at any decision step.

The step loop is the one of ``farc.disturbances.rollout.run_disturbed_episode`` (goal shift,
observation noise, decision, action perturbations, clipping, ``env.step``), split into
``observe`` (what the controller sees at decision t) and ``act`` (apply a command).

Everything that determines the future is captured by ``snapshot``:

- the simulator: the whole ``MjData`` (qpos, qvel, act, mocap pose, warm start, time, ...),
  copied with ``mujoco.mj_copyData``; the model is never changed by stepping;
- the environment's Python state: ``unwrapped.goal`` and the ``TimeLimit`` step counter;
- the rollout's state: every perturbation object (its random generator mid-stream, its
  active/changed records, a pending goal-shift event), the issued-command history that
  ``action_delay`` replays, the last observation, the online history used by triggers, and
  the step index.

``restore`` puts all of it back, so two branches from one snapshot see the same simulator
state, goal, generator states and therefore the same future disturbance draws. Branches differ
only through the commands they issue (including what a delay queue later replays).
``state_digest`` hashes the full state for equality checks.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from typing import Any, cast

import gymnasium as gym
import mujoco
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
from farc.recovery.controller import OnlineHistory


def _time_limit(env: gym.Env) -> Any:
    wrapper: Any = env
    while wrapper is not None:
        if hasattr(wrapper, "_elapsed_steps"):
            return wrapper
        wrapper = getattr(wrapper, "env", None)
    raise RolloutContractError("environment has no TimeLimit wrapper")


@dataclass
class Observed:
    """The controller's view at one decision step."""

    step: int
    gripper: np.ndarray
    obj: np.ndarray
    goal: np.ndarray
    base_action: np.ndarray  # frozen-policy output on this view
    observation: np.ndarray | None = None  # the full (possibly noisy) observation vector


@dataclass
class Snapshot:
    data: mujoco.MjData
    goal: np.ndarray
    elapsed_steps: int
    python_state: dict[str, Any]


class EpisodeSession:
    PYTHON_STATE = ("step", "observation", "perturbations", "commands", "history",
                    "successes", "final_distance", "pending_view")  # fmt: skip

    def __init__(
        self,
        env: gym.Env,
        policy: PolicyFunction,
        normalizer: FrozenNormalizer,
        condition: ConditionSpec,
        episode_seed: int,
        perturbation_seed: int,
    ):
        self.env, self.policy, self.normalizer = env, policy, normalizer
        self.raw = cast(Any, env.unwrapped)
        if env.spec is None or env.spec.max_episode_steps is None:
            raise RolloutContractError("environment needs a time limit")
        self.horizon = int(env.spec.max_episode_steps)
        self.threshold = float(self.raw.distance_threshold)
        self.low = np.asarray(env.action_space.low, np.float64)  # type: ignore[attr-defined]
        self.high = np.asarray(env.action_space.high, np.float64)  # type: ignore[attr-defined]
        self.condition, self.episode_seed = condition, episode_seed
        self.perturbations = build_perturbations(
            condition.disturbances, perturbation_seed, episode_seed, self.horizon
        )
        self.observation, _ = env.reset(seed=episode_seed)
        self.layout = layout_for(self.raw, np.asarray(self.observation["observation"]).shape[0])
        validate_layout(self.raw, self.observation["observation"], self.layout)
        for p in self.perturbations:
            if isinstance(p, ObservationNoise):
                p.configure(self.layout, control_period(self.raw))
        self.step = 0
        self.commands: list[np.ndarray] = []
        self.history = OnlineHistory(self.horizon)
        self.successes = np.zeros(self.horizon, dtype=bool)
        self.final_distance = float(np.linalg.norm(
            self.observation["achieved_goal"] - self.raw.goal))  # fmt: skip
        self.pending_view: Observed | None = None

    # --- stepping ----------------------------------------------------------------------

    @property
    def done(self) -> bool:
        return self.step >= self.horizon

    def _ordered(self, kind: type) -> list[Any]:
        items = [p for p in self.perturbations if isinstance(p, kind)]
        if kind is ActionPerturbation:
            items.sort(key=lambda p: ACTION_TYPES.index(p.spec.type))
        return items

    def observe(self) -> Observed:
        """Apply this step's goal shift and observation noise; evaluate the base policy."""
        if self.pending_view is not None:
            return self.pending_view
        if self.done:
            raise RolloutContractError("the episode is over")
        step = self.step
        for shift in self._ordered(GoalShift):
            new_goal = shift.maybe_shift(step, self.raw)
            if new_goal is not None:
                self.observation = dict(self.observation)
                self.observation["desired_goal"] = new_goal.copy()
        view = _controller_view(self.observation)
        for noise in self._ordered(ObservationNoise):
            view = noise.apply(step, view)
        s = self.layout.slices
        g, o = view["observation"][s["gripper_position"]], view["observation"][s["object_position"]]
        goal = view["desired_goal"]
        base = np.asarray(
            self.policy(self.normalizer(view["observation"], goal)[None])[0], np.float64
        )
        prev = self.history.base_actions[step - 1].copy() if step else None
        self.history.record(step, g, o, goal, prev)
        self.history.base_actions[step:] = base  # base output of this step, for feature t+1
        self.pending_view = Observed(
            step, g.copy(), o.copy(), goal.copy(), base, view["observation"].copy()
        )
        return self.pending_view

    def act(self, command: np.ndarray) -> np.ndarray:
        """Issue ``command`` at the current step; returns the executed (perturbed, clipped)
        action."""
        if self.pending_view is None:
            raise RolloutContractError("call observe() before act()")
        step = self.step
        command = np.asarray(command, np.float64).copy()
        self.commands.append(command)
        action = command
        for perturbation in self._ordered(ActionPerturbation):
            action = perturbation.apply(step, action, self.commands)
        executed = np.clip(action, self.low, self.high)
        self.observation, _, terminated, truncated, info = self.env.step(
            executed.astype(np.float32)
        )
        last = step == self.horizon - 1
        if terminated or (truncated and not last) or (last and not truncated):
            raise RolloutContractError(f"episode ended unexpectedly at step {step + 1}")
        self.successes[step] = bool(info["is_success"])
        self.final_distance = float(
            np.linalg.norm(self.observation["achieved_goal"] - self.raw.goal)
        )
        if self.successes[step] != (self.final_distance < self.threshold):
            raise RolloutContractError("info['is_success'] disagrees with the ground truth")
        self.step += 1
        self.pending_view = None
        return executed

    def run_base(self) -> None:
        while not self.done:
            self.act(self.observe().base_action)

    @property
    def final_success(self) -> bool:
        return bool(self.successes[-1])

    # --- branching ---------------------------------------------------------------------

    def snapshot(self) -> Snapshot:
        model, data = self.raw.model, self.raw.data
        saved = mujoco.MjData(model)
        mujoco.mj_copyData(saved, model, data)
        return Snapshot(
            data=saved,
            goal=np.asarray(self.raw.goal).copy(),
            elapsed_steps=int(_time_limit(self.env)._elapsed_steps),
            python_state=copy.deepcopy({k: getattr(self, k) for k in self.PYTHON_STATE}),
        )

    def restore(self, snapshot: Snapshot) -> None:
        model = self.raw.model
        mujoco.mj_copyData(self.raw.data, model, snapshot.data)
        self.raw.goal = snapshot.goal.copy()
        _time_limit(self.env)._elapsed_steps = snapshot.elapsed_steps
        for key, value in copy.deepcopy(snapshot.python_state).items():
            setattr(self, key, value)

    def state_digest(self) -> str:
        """Hash of the simulator state, goal, step counter and every generator state."""
        d = self.raw.data
        digest = hashlib.sha256()
        for array in (d.qpos, d.qvel, d.act, d.mocap_pos, d.mocap_quat, d.qacc_warmstart,
                      d.ctrl, np.array([d.time]), np.asarray(self.raw.goal)):  # fmt: skip
            digest.update(np.ascontiguousarray(array, np.float64).tobytes())
        digest.update(str((self.step, _time_limit(self.env)._elapsed_steps)).encode())
        for p in self.perturbations:
            digest.update(repr(p.rng.bit_generator.state).encode())
        for c in self.commands:
            digest.update(c.tobytes())
        return digest.hexdigest()
