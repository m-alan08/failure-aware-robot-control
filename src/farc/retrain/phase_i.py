"""Phase I: warm-started, information-matched delay-aware SAC+HER.

Plan: docs/PHASE_I_STRONG_BASELINE_PROPOSAL.md (approved as written 2026-11-17).

- ``Featurizer``: the extra policy inputs, computed identically in training (inside the env
  wrapper) and in evaluation (inside the episode loop).
  - I-A: the last 10 issued position commands and the last 10 observed gripper displacements
    (divided by K0 * 0.05), most recent first, zero-padded: 60 entries.
  - I-C: the last 4 issued position commands, a one-hot of the Phase E latched delay estimate
    (0..4) and the estimator's current delay-reliability flag: 18 entries.
- ``warm_start``: copies every frozen-checkpoint tensor into the new agent, inserting zero
  columns for the new inputs (after the 25 original observation entries) in each first layer
  and in the matching Adam moments.
- ``HybridNormalizer``: the frozen normaliser, never updated, for the original 25 + 3 entries;
  the extra entries pass through clipped to +/-5.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, cast

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.adaptation.experiment import AdaptationConfigError
from farc.base_policy.checkpoint import (
    CheckpointError,
    FrozenPolicy,
    file_sha256,
    read_checkpoint,
    verify_checkpoint,
)
from farc.base_policy.networks import GaussianActor
from farc.base_policy.normalizer import GoalObservationNormalizer
from farc.delaychange.random_schedules import RandomCell, draw_pulses
from farc.delaychange.schedule import DelaySchedule
from farc.delaychange.stage3 import Latched
from farc.envs.factory import EnvConfig
from farc.paths import checkpoint_path
from farc.retrain.phase_h import DelayHistoryWrapper
from farc.supervision.seeds import check_i_seeds

GENERATOR_SEED = 20262000
ORIGINAL_OBS = 25
K0_STEP = 0.62 * 0.05
ARMS = {"IA": 60, "IC": 18}  # extra input entries per arm
MANIFEST = Path(__file__).with_name("checkpoints_phase_i.json")


# --- training delay distribution (proposal section 4.4) ----------------------------------------


def training_schedule(run_seed: int, episode: int) -> tuple[str, DelaySchedule]:
    rng = np.random.default_rng([GENERATOR_SEED, run_seed, episode])
    group = int(rng.integers(3))
    key = run_seed * 1_000_000 + episode
    if group == 0:
        return "nominal", DelaySchedule(())
    if group == 1:
        d = int(rng.integers(1, 5))
        if rng.integers(2) == 0:
            return "constant", DelaySchedule(((0, None, d),))
        cell: RandomCell = RandomCell("RP_i", "RP", d)
        return "persistent", DelaySchedule(draw_pulses(cell, 2, key, GENERATOR_SEED))
    kind = int(rng.integers(3))
    if kind == 0:
        cell = RandomCell("RT_i", "RT", int(rng.choice([2, 4])))
        return "pulse", DelaySchedule(draw_pulses(cell, 3, key, GENERATOR_SEED))
    if kind == 1:
        cell = RandomCell("RI_i", "RI", int(rng.choice([2, 4])))
        return "on_off", DelaySchedule(draw_pulses(cell, 4, key, GENERATOR_SEED))
    return "change", DelaySchedule(draw_pulses(RandomCell("RS_i", "RS"), 5, key, GENERATOR_SEED))


# --- features (identical in training and evaluation) -------------------------------------------


class Featurizer:
    """Call ``reset()``, then ``observe(obs)`` for every observation (including the first),
    ``features()`` before acting, and ``issued(command)`` after acting."""

    def __init__(self, arm: str, estimator: dict[str, Any]):
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.estimator_config = arm, EstimatorConfig(**estimator)
        self.reset()

    def reset(self) -> None:
        self.g: list[np.ndarray] = []
        self.c: list[np.ndarray] = []
        self.est = ActuatorEstimator(self.estimator_config)
        self.latch = Latched()
        self.decision: tuple[int, bool] = (0, False)

    def observe(self, observation: np.ndarray) -> None:
        g = np.asarray(observation, np.float64)[0:3].copy()
        self.g.append(g)
        if self.arm == "IC":
            self.est.observe(g)
            dec = self.latch.step(self.est)
            self.decision = (int(dec.delay), bool(dec.estimate.delay_reliable))

    def issued(self, command: np.ndarray) -> None:
        c = np.asarray(command, np.float64).copy()
        self.c.append(c)
        if self.arm == "IC":
            self.est.issued(c)

    def _commands(self, n: int) -> list[np.ndarray]:
        return [self.c[-k][0:3] if len(self.c) >= k else np.zeros(3) for k in range(1, n + 1)]

    def features(self) -> np.ndarray:
        if self.arm == "IA":
            disp = [(self.g[-k] - self.g[-k - 1]) / K0_STEP if len(self.g) > k else np.zeros(3)
                    for k in range(1, 11)]  # fmt: skip
            return np.concatenate([*self._commands(10), *disp])
        d, reliable = self.decision
        onehot = np.zeros(5)
        onehot[d] = 1.0
        return np.concatenate([*self._commands(4), onehot, [float(reliable)]])


class PhaseIWrapper(DelayHistoryWrapper):
    """The Phase H delay wrapper (replay/drop, verified bitwise against the session) with the
    Phase I features appended to the observation."""

    def __init__(self, env: gym.Env, arm: str, estimator: dict[str, Any], schedule_for=None):
        super().__init__(env, 0, schedule_for)
        self.featurizer = Featurizer(arm, estimator)
        space = cast(spaces.Dict, env.observation_space)
        base = cast(spaces.Box, space["observation"])
        self.observation_space = spaces.Dict({
            **space.spaces,
            "observation": spaces.Box(-np.inf, np.inf, (base.shape[0] + ARMS[arm],),
                                      dtype=base.dtype)})  # fmt: skip

    def _obs(self, obs: dict) -> dict:
        self.featurizer.observe(obs["observation"])
        extra = self.featurizer.features().astype(obs["observation"].dtype)
        return {**obs, "observation": np.concatenate([obs["observation"], extra])}

    def reset(self, **kwargs):
        self.featurizer.reset()
        return super().reset(**kwargs)

    def step(self, action):
        self.featurizer.issued(np.asarray(action, np.float64))
        return super().step(action)


def env_wrapper(arm: str, run_seed: int, estimator: dict[str, Any]):
    def wrap(env: gym.Env, role: str) -> gym.Env:
        schedule_for = (lambda ep: training_schedule(run_seed, ep)) if role == "train" else None
        return PhaseIWrapper(env, arm, estimator, schedule_for)

    return wrap


# --- normaliser and warm start -----------------------------------------------------------------


class HybridNormalizer:
    """Frozen statistics for the original observation and goal; extra entries clipped only.
    ``update`` is a no-op (proposal section 4.3)."""

    def __init__(self, frozen: GoalObservationNormalizer, extra: int, clip: float = 5.0):
        self.frozen, self.extra, self.clip = copy.deepcopy(frozen), extra, clip

    @property
    def input_dim(self) -> int:
        return ORIGINAL_OBS + self.extra + 3

    def update(self, observations: np.ndarray, goals: np.ndarray) -> None:
        return None

    def __call__(self, observations: np.ndarray, goals: np.ndarray) -> np.ndarray:
        obs = np.asarray(observations)
        base = self.frozen(obs[..., :ORIGINAL_OBS], goals)
        extra = np.clip(obs[..., ORIGINAL_OBS:], -self.clip, self.clip)
        return np.concatenate([base[..., :ORIGINAL_OBS], extra, base[..., ORIGINAL_OBS:]],
                              axis=-1).astype(np.float32)  # fmt: skip

    def state_dict(self) -> dict[str, Any]:
        return {"kind": "phase_i_hybrid", "frozen": self.frozen.state_dict(),
                "extra": self.extra, "clip": self.clip}  # fmt: skip


def _expand(old: torch.Tensor, shape: torch.Size, extra: int) -> torch.Tensor:
    """Insert ``extra`` zero columns after column 25 when only the input width differs."""
    if old.shape == shape:
        return old.clone()
    if old.dim() != 2 or old.shape[0] != shape[0] or shape[1] - old.shape[1] != extra:
        raise ValueError(f"cannot expand {tuple(old.shape)} to {tuple(shape)}")
    zeros = torch.zeros(old.shape[0], extra, dtype=old.dtype)
    return torch.cat([old[:, :ORIGINAL_OBS], zeros, old[:, ORIGINAL_OBS:]], dim=1)


def _expand_tree(old: Any, new: Any, extra: int) -> Any:
    if isinstance(old, torch.Tensor):
        return _expand(old, new.shape, extra) if isinstance(new, torch.Tensor) else old.clone()
    if isinstance(old, dict):
        return {k: _expand_tree(v, new[k] if isinstance(new, dict) and k in new else v, extra)
                for k, v in old.items()}  # fmt: skip
    if isinstance(old, list):
        return [_expand_tree(v, new[i] if isinstance(new, list) and i < len(new) else v, extra)
                for i, v in enumerate(old)]  # fmt: skip
    return copy.deepcopy(old)


def frozen_checkpoint(seed: int) -> dict[str, Any]:
    path = checkpoint_path(f"fetch_push_fp32_seed{seed}")
    verify_checkpoint(path)  # SHA-256 against the frozen manifest; the file is only read
    return read_checkpoint(path)


def warm_start(agent: Any, frozen: dict[str, Any], extra: int) -> None:
    """Load the frozen agent state into ``agent``, expanding first layers and Adam moments."""
    target = agent.state_dict()
    state = frozen["agent"]
    expanded = {}
    for key in ("actor", "critic", "critic_target"):
        expanded[key] = {k: _expand(v, target[key][k].shape, extra) for k, v in state[key].items()}
    expanded["log_alpha"] = state["log_alpha"].clone()
    for key in ("actor_optimizer", "critic_optimizer", "alpha_optimizer"):
        opt = copy.deepcopy(state[key])
        for idx, st in opt["state"].items():
            for name in ("exp_avg", "exp_avg_sq"):
                if name in st:
                    pshape = _param_shape(agent, key, idx)
                    st[name] = _expand(st[name], pshape, extra)
        expanded[key] = opt
    agent.load_state_dict(expanded)


def _param_shape(agent: Any, optimizer_key: str, index: int) -> torch.Size:
    module = {"actor_optimizer": agent.actor, "critic_optimizer": agent.critic}.get(optimizer_key)
    if module is None:
        return agent.log_alpha.shape
    return list(module.parameters())[index].shape


def init_hook(seed: int, extra: int):
    frozen = frozen_checkpoint(seed)
    norm = GoalObservationNormalizer(ORIGINAL_OBS, 3)
    norm.load_state_dict(frozen["normalizer"])

    def hook(agent: Any, _normalizer: Any):
        warm_start(agent, frozen, extra)
        return agent, HybridNormalizer(norm, extra)

    return hook


# --- loading trained Phase I policies -----------------------------------------------------------


def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"checkpoints": {}}


def load_phase_i(path: str | Path, checkpoint_id: str,
                 verify: bool = True) -> tuple[FrozenPolicy, HybridNormalizer]:  # fmt: skip
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
    ns = raw["normalizer"]
    if ns.get("kind") != "phase_i_hybrid":
        raise CheckpointError(f"{path} is not a Phase I checkpoint")
    frozen = GoalObservationNormalizer(ORIGINAL_OBS, 3)
    frozen.load_state_dict(ns["frozen"])
    hybrid = HybridNormalizer(frozen, int(ns["extra"]), float(ns["clip"]))
    policy = FrozenPolicy(path=path, checkpoint_id=checkpoint_id, sha256=sha,
                          env_config=env_config, training_seed=int(config["seed"]),
                          total_steps=int(raw["total_steps"]), actor=actor,
                          normalizer=frozen, dimensions=dims, metadata=raw["metadata"],
                          training_config=config)  # fmt: skip
    return policy, hybrid


# --- config -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseIConfig:
    plan: str
    plan_sha256: str
    training: dict[str, Any]
    training_seeds: tuple[int, ...]
    output: str
    estimator: dict[str, Any]
    raw: dict[str, Any]

    def seeds(self, role: str) -> list[int]:
        blocks = {"debugging": (self.raw["debug_seed_offset"], self.raw["debug_episodes"]),
                  "evaluation": (self.raw["evaluation_seed_offset"],
                                 self.raw["evaluation_episodes"])}  # fmt: skip
        off, n = blocks[role]
        return list(range(off, off + n))


def load_config(path: str | Path) -> PhaseIConfig:
    v = json.loads(Path(path).read_text())
    c = PhaseIConfig(plan=v["plan"], plan_sha256=v["plan_sha256"], training=v["training"],
                     training_seeds=tuple(v["training_seeds"]), output=v["output"],
                     estimator=v["estimator"], raw=v)  # fmt: skip
    for role in ("debugging", "evaluation"):
        check_i_seeds(c.seeds(role), role)
    if hashlib.sha256(Path(c.plan).read_bytes()).hexdigest() != c.plan_sha256:
        raise AdaptationConfigError("the Phase I plan changed since the config pinned it")
    return c
