"""Failure-prediction dataset built from logged FetchPush disturbance rollouts.

One example per (episode, decision step t = 0..T-1). The label is the episode's final-step
outcome under the frozen base policy: ``failure = 1`` if the object is not within the
success threshold of the active goal after the last step (``final_success`` in
``episodes.csv``, cross-checked against the last trajectory row).

Splits are grouped by ``episode_seed``: the same seed gives the same initial state and goal
for every condition and every policy seed, and conditions share identical prefixes before
their onset, so all episodes of one seed go to the same split. Normalization statistics are
fitted on training rows only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from farc.prediction.features import (
    ACTION_DIM,
    FEATURE_NAMES,
    NUM_FEATURES,
    FeatureConfig,
    episode_features,
)

ENV_ID = "FetchPush-v4"
VALIDATION_SEEDS = (3_500_000, 4_000_000)  # [low, high)
HELDOUT_SEEDS = (4_000_000, 5_000_000)

_XYZ = ("0", "1", "2")


class HeldOutSeedError(RuntimeError):
    pass


@dataclass
class EpisodeData:
    """Every FetchPush episode, with per-step features (E, T, F) and per-episode metadata."""

    features: np.ndarray  # (E, T, F) float64
    episodes: pd.DataFrame  # one row per episode; ``episode_id`` = row index
    trajectory_hashes: list[str]
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)


def _check_seed(seed: int) -> None:
    if HELDOUT_SEEDS[0] <= seed < HELDOUT_SEEDS[1]:
        raise HeldOutSeedError(f"episode seed {seed} is in the reserved held-out range")
    if not VALIDATION_SEEDS[0] <= seed < VALIDATION_SEEDS[1]:
        raise HeldOutSeedError(f"episode seed {seed} is outside the validation range")


def trajectory_hash(objects: np.ndarray, grippers: np.ndarray, goals: np.ndarray) -> str:
    digest = hashlib.sha256()
    for part in (objects, grippers, goals):
        digest.update(np.ascontiguousarray(np.round(part, 9)).tobytes())
    return digest.hexdigest()


def load_episodes(
    experiment_dir: str | Path, config: FeatureConfig | None = None, env_id: str = ENV_ID
) -> EpisodeData:
    """Load every ``<env_id>_seed<k>/<condition>`` part of a disturbance experiment."""
    config = config or FeatureConfig()
    parts = sorted(Path(experiment_dir, "parts").glob(f"{env_id}_seed*/*"))
    if not parts:
        raise FileNotFoundError(f"no {env_id} parts under {experiment_dir}/parts")
    features: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    hashes: list[str] = []
    for part in parts:
        if not (part / "done.json").exists():
            raise RuntimeError(f"{part} is incomplete (no done.json)")
        policy_seed = int(part.parent.name.rsplit("_seed", 1)[1])
        summary = pd.read_csv(part / "episodes.csv").set_index("episode_seed")
        trajectories = pd.read_csv(part / "trajectories.csv.gz")
        for key, frame in trajectories.groupby("episode_seed", sort=True):
            seed = int(str(key))
            _check_seed(seed)
            frame = frame.sort_values("t")
            if not np.array_equal(frame["t"].to_numpy(), np.arange(len(frame))):
                raise ValueError(f"{part} seed {seed}: rows are not t = 0..T")
            objects = frame[[f"achieved_{a}" for a in _XYZ]].to_numpy(np.float64)
            grippers = frame[[f"gripper_{a}" for a in _XYZ]].to_numpy(np.float64)
            goals = frame[[f"goal_{a}" for a in _XYZ]].to_numpy(np.float64)
            actions = frame[[f"base_action_{a}" for a in range(ACTION_DIM)]].to_numpy(np.float64)
            final_success = bool(summary.loc[seed, "final_success"])
            if bool(frame["success"].iloc[-1]) != final_success:
                raise ValueError(f"{part} seed {seed}: final success disagrees with the log")
            features.append(episode_features(objects, grippers, goals, actions[1:], config))
            hashes.append(trajectory_hash(objects, grippers, goals))
            entry = summary.loc[seed]
            rows.append(
                {
                    "env_id": env_id,
                    "policy_seed": policy_seed,
                    "condition": part.name,
                    "disturbance_type": entry["disturbance_type"],
                    "episode_seed": seed,
                    "final_success": final_success,
                    "failure": int(not final_success),
                    "initial_success": bool(frame["success"].iloc[0]),
                }
            )
    episodes = pd.DataFrame(rows)
    episodes["condition_id"] = pd.Categorical(episodes["condition"]).codes.astype(np.int64)
    episodes["trajectory_hash"] = hashes
    episodes.index.name = "episode_id"
    return EpisodeData(np.stack(features), episodes, hashes, config)


def duplicate_audit(episodes: pd.DataFrame, features: np.ndarray | None = None) -> dict[str, Any]:
    """Exact duplicate state trajectories (object, gripper, goal to 1e-9 m), and, given the
    (E, T, F) features, examples whose feature row is identical to one of another episode
    (shared prefixes, e.g. a condition before its onset vs. undisturbed)."""
    groups = episodes.groupby("trajectory_hash")
    sizes = groups.size()
    dup = sizes[sizes > 1]
    cross_seed = int((groups["episode_seed"].nunique() > 1).sum())
    audit: dict[str, Any] = {
        "episodes": int(len(episodes)),
        "unique_trajectories": int(len(sizes)),
        "duplicate_groups": int(len(dup)),
        "episodes_in_duplicate_groups": int(dup.sum()),
        "groups_spanning_episode_seeds": cross_seed,
    }
    if features is not None:
        flat = np.round(features.reshape(-1, features.shape[-1]), 12)
        seeds = np.repeat(episodes["episode_seed"].to_numpy(), features.shape[1])
        rows = pd.DataFrame({"key": [r.tobytes() for r in flat], "seed": seeds})
        counts = rows.groupby("key")["seed"].agg(["size", "nunique"])
        repeated = counts[counts["size"] > 1]
        audit["examples_with_shared_feature_rows"] = int(repeated["size"].sum())
        audit["shared_feature_rows_spanning_episode_seeds"] = int((repeated["nunique"] > 1).sum())
    return audit


def split_episode_seeds(
    seeds: np.ndarray, validation_fraction: float = 0.2, split_seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint (train, validation) episode-seed sets, a seeded random permutation."""
    unique = np.unique(seeds)
    order = np.random.default_rng(split_seed).permutation(unique)
    n_val = int(round(validation_fraction * len(unique)))
    if not 0 < n_val < len(unique):
        raise ValueError("validation_fraction leaves an empty split")
    return np.sort(order[n_val:]), np.sort(order[:n_val])


@dataclass(frozen=True)
class Normalizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, min_std: float = 1e-6) -> Normalizer:
        flat = features.reshape(-1, features.shape[-1])
        return cls(flat.mean(axis=0), np.maximum(flat.std(axis=0), min_std))

    def __call__(self, features: np.ndarray) -> np.ndarray:
        return (features - self.mean) / self.std

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}


class FailurePredictionDataset(Dataset):
    """Per-step examples of one split.

    Item: dict of tensors
      ``features`` (F,) float32, normalized; ``label`` () float32, 1 = eventual failure;
      ``episode_id``, ``t``, ``condition_id``, ``policy_seed``, ``episode_seed`` () int64.
    """

    def __init__(
        self,
        data: EpisodeData,
        episode_ids: np.ndarray,
        normalizer: Normalizer,
        condition_names: list[str],
    ):
        episode_ids = np.asarray(episode_ids, np.int64)
        episodes = data.episodes.loc[episode_ids]
        raw = data.features[episode_ids]  # (E, T, F)
        n_episodes, horizon, _ = raw.shape
        self.episode_ids = episode_ids
        self.episodes = episodes
        self.normalizer = normalizer
        self.condition_names = condition_names
        self.feature_names = FEATURE_NAMES
        self.horizon = horizon
        self.raw_features = torch.from_numpy(raw.reshape(-1, NUM_FEATURES).astype(np.float32))
        self.features = torch.from_numpy(normalizer(raw).reshape(-1, NUM_FEATURES)).float()

        def per_step(values: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(np.repeat(np.asarray(values, np.int64), horizon))

        self.labels = per_step(episodes["failure"].to_numpy()).float()
        self.episode_id = per_step(episode_ids)
        self.t = torch.from_numpy(np.tile(np.arange(horizon, dtype=np.int64), n_episodes))
        self.condition_id = per_step(episodes["condition_id"].to_numpy())
        self.policy_seed = per_step(episodes["policy_seed"].to_numpy())
        self.episode_seed = per_step(episodes["episode_seed"].to_numpy())

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[index],
            "label": self.labels[index],
            "episode_id": self.episode_id[index],
            "t": self.t[index],
            "condition_id": self.condition_id[index],
            "policy_seed": self.policy_seed[index],
            "episode_seed": self.episode_seed[index],
        }

    def class_balance(self) -> dict[str, float]:
        failures = int(self.labels.sum().item())
        return {
            "examples": len(self),
            "failure_examples": failures,
            "success_examples": len(self) - failures,
            "failure_fraction": failures / len(self),
            "pos_weight": (len(self) - failures) / max(failures, 1),
        }


@dataclass
class FailureDatasets:
    train: FailurePredictionDataset
    validation: FailurePredictionDataset
    normalizer: Normalizer
    data: EpisodeData
    train_seeds: np.ndarray
    validation_seeds: np.ndarray

    def report(self) -> dict[str, Any]:
        episodes = self.data.episodes

        def episode_counts(frame: pd.DataFrame) -> dict[str, int]:
            failed = int(frame["failure"].sum())
            return {
                "episodes": len(frame),
                "successful": len(frame) - failed,
                "failed": failed,
                "episode_seeds": int(frame["episode_seed"].nunique()),
            }

        return {
            "feature_names": list(FEATURE_NAMES),
            "feature_dim": NUM_FEATURES,
            "horizon": int(self.data.features.shape[1]),
            "feature_config": vars(self.data.feature_config),
            "all": episode_counts(episodes),
            "train": {**episode_counts(self.train.episodes), **self.train.class_balance()},
            "validation": {
                **episode_counts(self.validation.episodes),
                **self.validation.class_balance(),
            },
            "duplicates": duplicate_audit(episodes, self.data.features),
            "initially_successful_episodes": int(episodes["initial_success"].sum()),
            "conditions": sorted(episodes["condition"].unique().tolist()),
            "policy_seeds": sorted(int(s) for s in episodes["policy_seed"].unique()),
            "episode_seed_range": [
                int(episodes["episode_seed"].min()),
                int(episodes["episode_seed"].max()),
            ],
            "train_seeds": self.train_seeds.tolist(),
            "validation_seeds": self.validation_seeds.tolist(),
            "normalizer": self.normalizer.state_dict(),
        }


def build_datasets(
    experiment_dir: str | Path,
    validation_fraction: float = 0.2,
    split_seed: int = 0,
    config: FeatureConfig | None = None,
    data: EpisodeData | None = None,
) -> FailureDatasets:
    data = data or load_episodes(experiment_dir, config)
    episodes = data.episodes
    train_seeds, val_seeds = split_episode_seeds(
        episodes["episode_seed"].to_numpy(), validation_fraction, split_seed
    )
    train_ids = episodes.index[episodes["episode_seed"].isin(train_seeds)].to_numpy()
    val_ids = episodes.index[episodes["episode_seed"].isin(val_seeds)].to_numpy()
    train_hashes = set(episodes.loc[train_ids, "trajectory_hash"])
    if train_hashes & set(episodes.loc[val_ids, "trajectory_hash"]):
        raise RuntimeError("an identical trajectory appears in both splits")
    normalizer = Normalizer.fit(data.features[train_ids])
    names = list(pd.Categorical(episodes["condition"]).categories)
    return FailureDatasets(
        FailurePredictionDataset(data, train_ids, normalizer, names),
        FailurePredictionDataset(data, val_ids, normalizer, names),
        normalizer,
        data,
        train_seeds,
        val_seeds,
    )


def save_report(datasets: FailureDatasets, path: str | Path) -> None:
    Path(path).write_text(json.dumps(datasets.report(), indent=2) + "\n")
