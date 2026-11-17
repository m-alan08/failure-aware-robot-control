"""Phase H: delay-randomised retraining of SAC+HER on FetchPush.

Plan: docs/PHASE_H_RETRAINED_BASELINE_PROPOSAL.md.

- ``training_schedule``: the training delay distribution of section 3.2 (one family per
  episode, uniform; G3/G4 generators with the separate generator seed 20261900).
- ``DelayHistoryWrapper``: applies a ``DelaySchedule`` to executed actions with the replay/drop
  semantics of ``ScheduledDelaySession`` and optionally appends the last ``history`` issued
  commands' position components to the observation (arm R2).
- ``load_retrained``: loads a retrained checkpoint after checking it against
  ``src/farc/retrain/checkpoints.json`` (the frozen-policy manifest is never touched).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from farc.adaptation.experiment import AdaptationConfigError
from farc.base_policy.checkpoint import (
    CheckpointError,
    FrozenPolicy,
    file_sha256,
    read_checkpoint,
)
from farc.base_policy.networks import GaussianActor
from farc.base_policy.normalizer import GoalObservationNormalizer
from farc.delaychange.random_schedules import RandomCell, draw_pulses
from farc.delaychange.schedule import DelaySchedule
from farc.envs.factory import EnvConfig
from farc.supervision.seeds import check_h_seeds

TRAINING_GENERATOR_SEED = 20261900
FAMILIES = ("none", "constant", "persistent", "pulse", "on_off", "change")
ARMS = {"R1": 0, "R2": 4}  # command-history length per arm
POSITION = slice(0, 3)
MANIFEST = Path(__file__).with_name("checkpoints.json")


def training_schedule(run_seed: int, episode: int) -> tuple[str, DelaySchedule]:
    """Section 3.2: one family per episode, uniform; deterministic in (run seed, episode)."""
    rng = np.random.default_rng([TRAINING_GENERATOR_SEED, run_seed, episode])
    family = FAMILIES[int(rng.integers(len(FAMILIES)))]
    key = run_seed * 1_000_000 + episode
    if family == "none":
        return family, DelaySchedule(())
    if family == "constant":
        return family, DelaySchedule(((0, None, int(rng.integers(1, 5))),))
    if family == "persistent":
        cell, index = RandomCell("RP_train", "RP", int(rng.integers(1, 5))), 2
    elif family == "pulse":
        cell, index = RandomCell("RT_train", "RT", int(rng.choice([2, 4]))), 3
    elif family == "on_off":
        cell, index = RandomCell("RI_train", "RI", int(rng.choice([2, 4]))), 4
    else:
        cell, index = RandomCell("RS_train", "RS"), 5
    return family, DelaySchedule(draw_pulses(cell, index, key, TRAINING_GENERATOR_SEED))


def augment(observation: np.ndarray, issued: list[np.ndarray], history: int) -> np.ndarray:
    """Observation followed by the position components of the last ``history`` issued
    commands, most recent first, zeros before the episode start."""
    if history == 0:
        return observation
    parts = [observation]
    for k in range(1, history + 1):
        parts.append(issued[-k][POSITION] if len(issued) >= k else np.zeros(3))
    return np.concatenate(parts).astype(observation.dtype, copy=False)


class DelayHistoryWrapper(gym.Wrapper):
    """Delay schedule on executed actions (replay/drop, as ``ScheduledDelaySession``) and
    optional command history in the observation. ``schedule_for(episode)`` returns the
    episode's ``DelaySchedule``; ``None`` means no delay."""

    def __init__(self, env: gym.Env, history: int, schedule_for=None):
        super().__init__(env)
        self.history, self.schedule_for = history, schedule_for
        self.episode, self.t = -1, 0
        self.issued: list[np.ndarray] = []
        self.schedule = DelaySchedule(())
        self.family = "none"
        self.last_executed: np.ndarray | None = None
        if history:
            space = cast(spaces.Dict, env.observation_space)
            base = cast(spaces.Box, space["observation"])
            extra = 3 * history
            self.observation_space = spaces.Dict({
                **space.spaces,
                "observation": spaces.Box(-np.inf, np.inf, (base.shape[0] + extra,),
                                          dtype=base.dtype)})  # fmt: skip

    def _obs(self, obs: dict) -> dict:
        if not self.history:
            return obs
        return {**obs, "observation": augment(obs["observation"], self.issued, self.history)}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episode += 1
        self.t, self.issued = 0, []
        if self.schedule_for is not None:
            result = self.schedule_for(self.episode)
            self.family, self.schedule = result if isinstance(result, tuple) else ("?", result)
        return self._obs(obs), info

    def step(self, action):
        command = np.asarray(action, np.float64).copy()
        self.issued.append(command)
        source = self.t - self.schedule.delay_at(self.t)
        executed = self.issued[source].copy() if source >= 0 else np.zeros_like(command)
        bounds = cast(spaces.Box, self.env.action_space)
        executed = np.clip(executed, bounds.low, bounds.high)
        self.last_executed = executed
        obs, reward, terminated, truncated, info = self.env.step(executed.astype(np.float32))
        self.t += 1
        return self._obs(obs), reward, terminated, truncated, info


def env_wrapper(arm: str, run_seed: int):
    """The ``env_wrapper`` hook for the ported trainer: training env gets the delay
    distribution (and history for R2); the evaluation env gets history only, no delay."""
    history = ARMS[arm]

    def wrap(env: gym.Env, role: str) -> gym.Env:
        if role == "train":
            return DelayHistoryWrapper(env, history, lambda ep: training_schedule(run_seed, ep))
        return DelayHistoryWrapper(env, history) if history else env

    return wrap


# --- retrained-policy loading -----------------------------------------------------------------


def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"checkpoints": {}}


def load_retrained(path: str | Path, checkpoint_id: str, verify: bool = True) -> FrozenPolicy:
    """As ``farc.base_policy.checkpoint.load_policy`` but against the Phase H manifest.
    ``verify=False`` is for Stage-0 code checks only."""
    path = Path(path)
    sha = file_sha256(path)
    if verify:
        entry = manifest()["checkpoints"].get(checkpoint_id)
        if entry is None or entry["sha256"] != sha:
            raise CheckpointError(f"{checkpoint_id!r}: not in, or not matching, {MANIFEST.name}")
    raw = read_checkpoint(path)
    config = raw["config"]
    known = {f.name for f in fields(EnvConfig)}
    env_config = EnvConfig(**{k: v for k, v in config["env"].items() if k in known})
    dims = dict(raw["dimensions"])
    sac = config["sac"]
    actor = GaussianActor(dims["observation_dim"] + dims["goal_dim"], dims["action_dim"],
                          sac["hidden_sizes"], sac["log_std_min"], sac["log_std_max"])  # fmt: skip
    actor.load_state_dict(raw["agent"]["actor"])
    actor.eval()
    actor.requires_grad_(False)
    normalizer = GoalObservationNormalizer(dims["observation_dim"], dims["goal_dim"],
                                           env_config.obs_clip, env_config.normalized_clip,
                                           env_config.normalizer_eps)  # fmt: skip
    normalizer.load_state_dict(raw["normalizer"])
    return FrozenPolicy(path=path, checkpoint_id=checkpoint_id, sha256=sha,
                        env_config=env_config, training_seed=int(config["seed"]),
                        total_steps=int(raw["total_steps"]), actor=actor,
                        normalizer=normalizer, dimensions=dims, metadata=raw["metadata"],
                        training_config=config)  # fmt: skip


# --- config -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseHConfig:
    name: str
    plan: str
    plan_sha256: str
    training: dict[str, Any]
    training_seeds: tuple[int, ...]
    output: str
    debug_seed_offset: int
    debug_episodes: int
    evaluation_seed_offset: int
    evaluation_episodes: int
    raw: dict[str, Any]

    def seeds(self, role: str) -> list[int]:
        blocks = {
            "debugging": (self.debug_seed_offset, self.debug_episodes),
            "evaluation": (self.evaluation_seed_offset, self.evaluation_episodes),
        }
        off, n = blocks[role]
        return list(range(off, off + n))


def load_phase_h(path: str | Path) -> PhaseHConfig:
    values = json.loads(Path(path).read_text())
    c = PhaseHConfig(name=values["name"], plan=values["plan"], plan_sha256=values["plan_sha256"],
                     training=values["training"],
                     training_seeds=tuple(values["training_seeds"]), output=values["output"],
                     debug_seed_offset=values["debug_seed_offset"],
                     debug_episodes=values["debug_episodes"],
                     evaluation_seed_offset=values["evaluation_seed_offset"],
                     evaluation_episodes=values["evaluation_episodes"], raw=values)  # fmt: skip
    for role in ("debugging", "evaluation"):
        check_h_seeds(c.seeds(role), role)
    if hashlib.sha256(Path(c.plan).read_bytes()).hexdigest() != c.plan_sha256:
        raise AdaptationConfigError("the Phase H plan changed since the config pinned it")
    return c
