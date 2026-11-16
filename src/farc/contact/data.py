"""Seed protection, condition building, rollout logging and transition extraction (Phase F1)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.delay_eval import E2_USED_SEED_RANGES
from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.adaptation.experiment import AdaptationConfigError
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.episodes import Predictor, run_episode
from farc.counterfactual.session import EpisodeSession
from farc.disturbances.config import SEED_RANGES, ConditionSpec, DisturbanceSpec, load_experiment
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve

F1_USED_SEED_RANGES = {
    **E2_USED_SEED_RANGES,
    "phase_e2_debugging": (3_640_000, 3_640_010),
    "phase_e2_evaluation": (3_641_000, 3_641_100),
}
F1_SEED_BLOCKS = {
    "training": (3_650_000, 3_650_200),
    "selection": (3_651_000, 3_651_050),
    "debugging": (3_652_000, 3_652_010),
    "evaluation": (3_653_000, 3_653_100),
    "exploratory": (3_654_000, 3_654_100),  # post-hoc ablations, after the confirmatory run
    "heldout_debugging": (3_655_000, 3_655_010),  # code checks for the held-out runner
}


def check_seeds(seeds: list[int], role: str) -> None:
    """Seeds must be in the validation stage, below the held-out range, disjoint from every
    earlier phase, and inside the block reserved for their role."""
    low, high = SEED_RANGES["validation"]
    if not seeds or not (low <= min(seeds) and max(seeds) < high):
        raise AdaptationConfigError(f"{role} seeds must lie in the validation range")
    held_low, held_high = SEED_RANGES["heldout"]
    if any(held_low <= s < held_high for s in seeds):
        raise AdaptationConfigError("held-out seeds are reserved")
    for name, (a, b) in F1_USED_SEED_RANGES.items():
        if any(a <= s < b for s in seeds):
            raise AdaptationConfigError(f"{role} seeds overlap {name} [{a}, {b})")
    a, b = F1_SEED_BLOCKS[role]
    if not all(a <= s < b for s in seeds):
        raise AdaptationConfigError(f"{role} seeds must lie in [{a}, {b})")


def build_conditions(conditions_from: str, entries) -> list[ConditionSpec]:
    library = {c.name: c for c in load_experiment(resolve(conditions_from)).conditions}
    out = []
    for entry in entries:
        if "from" in entry:
            out.append(ConditionSpec(entry["name"], library[entry["from"]].disturbances))
        else:
            specs = tuple(DisturbanceSpec(**{**d, "fields": tuple(d.get("fields",
                                                                    ("observation",)))})
                          for d in entry["disturbances"])  # fmt: skip
            out.append(ConditionSpec(entry["name"], specs))
    return out


def collect(checkpoint: str, conditions: list[ConditionSpec], seeds: list[int],
            perturbation_seed: int, estimator: dict[str, Any],
            predictors: dict[str, Predictor | None]) -> list[dict[str, Any]]:  # fmt: skip
    """Logged rollouts (every step) for each predictor configuration."""
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    policy = deterministic_policy_function(loaded.actor)
    normalizer = FrozenNormalizer(loaded.normalizer)
    env = make_env(loaded.env_config)
    rows = []
    for condition in conditions:
        for seed in seeds:
            for name, predictor in predictors.items():
                session = EpisodeSession(env, policy, normalizer, condition, seed,
                                         perturbation_seed)  # fmt: skip
                result = run_episode(session, ActuatorEstimator(EstimatorConfig(**estimator)),
                                     predictor, condition, log=True)  # fmt: skip
                for row in result["rows"]:
                    row.update(configuration=name, policy_seed=loaded.training_seed,
                               condition=condition.name, episode_seed=seed,
                               success=result["success"])  # fmt: skip
                rows.extend(result["rows"])
    return rows


EPISODE = ["configuration", "policy_seed", "condition", "episode_seed"]


def columns(prefix: str, n: int) -> list[str]:
    return [f"{prefix}{i}" for i in range(n)]


def transitions(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(true obs t, executed command t, true obs t+1) for every consecutive pair of steps."""
    frame = frame.sort_values([*EPISODE, "t"])
    obs = frame[columns("true", 25)].to_numpy()
    cmd = frame[columns("exec", 3)].to_numpy()
    same = (frame[EPISODE].to_numpy()[1:] == frame[EPISODE].to_numpy()[:-1]).all(axis=1)
    consecutive = same & (np.diff(frame["t"].to_numpy()) == 1)
    return obs[:-1][consecutive], cmd[:-1][consecutive], obs[1:][consecutive]


def horizon_pairs(frame: pd.DataFrame, horizon: int, start: str = "true"):
    """Starting observation, executed commands t..t+h-1, true observation at t+h, and the
    true start observation (for contact flags), for every step with h steps left."""
    starts, cmds, targets, true_starts, index = [], [], [], [], []
    for _, ep in frame.sort_values([*EPISODE, "t"]).groupby(EPISODE, sort=False):
        o_start = ep[columns(start, 25)].to_numpy()
        o_true = ep[columns("true", 25)].to_numpy()
        ex = ep[columns("exec", 3)].to_numpy()
        for t in range(len(ep) - horizon):
            starts.append(o_start[t])
            cmds.append(ex[t : t + horizon])
            targets.append(o_true[t + horizon])
            true_starts.append(o_true[t])
            index.append(ep.index[t])
    return (np.array(starts), np.array(cmds), np.array(targets), np.array(true_starts),
            np.array(index))  # fmt: skip


def contact_flags(true_start: np.ndarray, true_end: np.ndarray, near: float = 0.05,
                  moved: float = 0.002) -> np.ndarray:  # fmt: skip
    """Evaluation-only contact label (pre-registered): gripper within ``near`` of the object
    horizontally at the start, or the object moved more than ``moved`` over the horizon."""
    rel = true_start[:, 6:8]
    displacement = np.linalg.norm(true_end[:, 3:6] - true_start[:, 3:6], axis=1)
    return (np.linalg.norm(rel, axis=1) < near) | (displacement > moved)
