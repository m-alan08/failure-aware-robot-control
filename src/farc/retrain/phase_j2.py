"""Phase J-2: noise-augmented fine-tuning (docs/PHASE_J2_PROPOSAL.md).

Identical to Phase I except that, in the training role only, the observation the agent sees gets
position noise from the evaluation noise model (``masked_noise``, mask ``positions``): per
episode sigma ~ U[0, 0.01] m, independent per step, from rng([20262300, run seed, episode]).
Goals (and so HER rewards) are untouched.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from farc.adaptation.experiment import AdaptationConfigError
from farc.disturbances.perturbations import masked_noise
from farc.envs.observation_layout import control_period, layout_for
from farc.retrain.phase_i import PhaseIWrapper, training_schedule
from farc.supervision.seeds import check_j2_seeds

GENERATOR_SEED = 20262200
NOISE_SEED = 20262300
SIGMA_MAX = 0.01
ARMS = {"J2A": "IA", "J2C": "IC"}  # J-2 arm -> Phase I feature set
MANIFEST = Path(__file__).with_name("checkpoints_phase_j2.json")


class NoisyPhaseIWrapper(PhaseIWrapper):
    """Phase I wrapper; the base observation is noised before features are computed."""

    def __init__(self, env: gym.Env, arm: str, estimator: dict[str, Any], schedule_for,
                 run_seed: int, sigma_max: float = SIGMA_MAX):  # fmt: skip
        super().__init__(env, arm, estimator, schedule_for)
        self.run_seed, self.sigma_max = run_seed, sigma_max
        self.layout: Any = None
        self.sigma = 0.0
        self.rng = np.random.default_rng(0)

    def _noise(self, obs: dict) -> dict:
        raw = np.asarray(obs["observation"])
        if self.layout is None:
            self.layout = layout_for(self.env.unwrapped, raw.shape[0])
        draw = self.rng.standard_normal(raw.shape[0])
        noise = masked_noise(self.layout, "positions", self.sigma, draw,
                             control_period(self.env.unwrapped))  # fmt: skip
        return {**obs, "observation": (raw + noise).astype(raw.dtype)}

    def _obs(self, obs: dict) -> dict:
        return super()._obs(self._noise(obs))

    def reset(self, **kwargs):
        episode = self.episode + 1
        self.rng = np.random.default_rng([NOISE_SEED, self.run_seed, episode])
        self.sigma = float(self.rng.uniform(0.0, self.sigma_max))
        return super().reset(**kwargs)


def env_wrapper(arm: str, run_seed: int, estimator: dict[str, Any], sigma_max: float = SIGMA_MAX):
    feats = ARMS[arm]

    def wrap(env: gym.Env, role: str) -> gym.Env:
        if role == "train":
            return NoisyPhaseIWrapper(env, feats, estimator,
                                      lambda ep: training_schedule(run_seed, ep, GENERATOR_SEED),
                                      run_seed, sigma_max)  # fmt: skip
        return PhaseIWrapper(env, feats, estimator, None)

    return wrap


def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"checkpoints": {}}


def load_config(path: str | Path) -> dict[str, Any]:
    v = json.loads(Path(path).read_text())
    for role in ("debugging", "evaluation"):
        check_j2_seeds(seeds(v, role), role)
    if hashlib.sha256(Path(v["plan"]).read_bytes()).hexdigest() != v["plan_sha256"]:
        raise AdaptationConfigError("the J-2 plan changed since the config pinned it")
    return v


def seeds(v: dict[str, Any], role: str) -> list[int]:
    blocks = {
        "debugging": (v["debug_seed_offset"], v["debug_episodes"]),
        "evaluation": (v["evaluation_seed_offset"], v["evaluation_episodes"]),
    }
    off, n = blocks[role]
    return list(range(off, off + n))
