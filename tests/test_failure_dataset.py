from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from farc.paths import data_dir
from farc.prediction import FEATURE_NAMES, NUM_FEATURES, FeatureConfig, build_datasets
from farc.prediction.dataset import (
    HeldOutSeedError,
    Normalizer,
    load_episodes,
    split_episode_seeds,
)
from farc.prediction.features import FEATURE_INDEX, episode_features, goal_change_steps

T = 50
rng = np.random.default_rng(0)


def random_episode(
    shift_at: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    objects = 1.3 + 0.01 * rng.standard_normal((T + 1, 3)).cumsum(axis=0)
    grippers = 1.3 + 0.01 * rng.standard_normal((T + 1, 3)).cumsum(axis=0)
    goals = np.tile([1.4, 0.7, 0.42], (T + 1, 1))
    if shift_at is not None:  # logged from row shift_at + 1, as in the rollout
        goals[shift_at + 1 :] = [1.3, 0.8, 0.42]
    actions = rng.uniform(-1, 1, (T, 4))
    return objects, grippers, goals, actions


def col(name: str) -> int:
    return FEATURE_INDEX[name]


def test_feature_indexing():
    objects, grippers, goals, actions = random_episode()
    f = episode_features(objects, grippers, goals, actions)
    assert f.shape == (T, NUM_FEATURES) == (T, len(FEATURE_NAMES))
    t = 7
    np.testing.assert_allclose(f[t, 0:3], goals[t + 1] - objects[t])
    np.testing.assert_allclose(f[t, 3:6], objects[t] - grippers[t])
    np.testing.assert_allclose(f[t, 6:9], objects[t] - objects[t - 1])
    np.testing.assert_allclose(f[t, 9:12], grippers[t] - grippers[t - 1])
    np.testing.assert_allclose(f[t, 12:16], actions[t - 1])
    assert f[t, col("remaining_time")] == pytest.approx((T - t) / T)
    d = np.linalg.norm(goals[1] - objects, axis=1)
    assert f[t, col("object_goal_progress")] == pytest.approx((d[t - 5] - d[t]) / 5)
    a = np.linalg.norm(objects - grippers, axis=1)
    assert f[t, col("approach_progress")] == pytest.approx((a[t - 5] - a[t]) / 5)
    assert FEATURE_NAMES[12] == "previous_action_0"


def test_missing_early_history():
    f = episode_features(*random_episode(), config=FeatureConfig(progress_window=5))
    assert np.all(f[0, 6:16] == 0) and f[0, col("has_previous")] == 0
    assert np.all(f[1:, col("has_previous")] == 1)
    assert f[0, col("object_goal_progress")] == 0 and f[0, col("progress_window")] == 0
    np.testing.assert_allclose(f[:7, col("progress_window")], [0, 0.2, 0.4, 0.6, 0.8, 1, 1])


@pytest.mark.parametrize("t", [0, 10, 25, 26, 49])
def test_causality(t):
    """Perturbing anything after decision t (states > t, goal rows > t+1, actions >= t)
    leaves the features of steps <= t unchanged."""
    objects, grippers, goals, actions = random_episode(shift_at=30)
    base = episode_features(objects, grippers, goals, actions)
    o, g, gl, ac = objects.copy(), grippers.copy(), goals.copy(), actions.copy()
    o[t + 1 :] += 5.0
    g[t + 1 :] -= 3.0
    gl[t + 2 :] += 1.0
    ac[t:] *= -7.0
    changed = episode_features(o, g, gl, ac)
    np.testing.assert_array_equal(base[: t + 1], changed[: t + 1])


def test_goal_shift_resets_progress():
    s = 25
    objects, grippers, goals, actions = random_episode(shift_at=s)
    np.testing.assert_array_equal(goal_change_steps(goals)[[s - 1, s, s + 3]], [0, s, s])
    f = episode_features(objects, grippers, goals, actions)
    np.testing.assert_allclose(f[s, 0:3], goals[s + 1] - objects[s])  # new goal at decision s
    assert f[s, col("time_since_goal_change")] == 0
    assert f[s + 3, col("time_since_goal_change")] == pytest.approx(3 / T)
    assert f[s, col("object_goal_progress")] == 0
    assert f[s, col("approach_progress")] == 0
    assert f[s + 2, col("progress_window")] == pytest.approx(2 / 5)
    new = np.linalg.norm(goals[s + 1] - objects, axis=1)
    assert f[s + 2, col("object_goal_progress")] == pytest.approx((new[s] - new[s + 2]) / 2)
    kept = episode_features(
        *random_episode(shift_at=s),
        FeatureConfig(progress_window=5, reset_approach_on_goal_change=False),
    )
    assert kept[s, col("approach_progress")] != 0


def test_split_integrity():
    seeds = np.repeat(np.arange(3_500_000, 3_500_100), 66)
    train, val = split_episode_seeds(seeds, 0.2, split_seed=3)
    assert len(train) == 80 and len(val) == 20
    assert not set(train) & set(val) and set(train) | set(val) == set(np.unique(seeds))
    np.testing.assert_array_equal(val, split_episode_seeds(seeds, 0.2, split_seed=3)[1])


def test_normalization_uses_training_data_only():
    train = rng.normal(2.0, 3.0, (40, T, NUM_FEATURES))
    norm = Normalizer.fit(train)
    z = norm(train).reshape(-1, NUM_FEATURES)
    np.testing.assert_allclose(z.mean(axis=0), 0, atol=1e-9)
    np.testing.assert_allclose(z.std(axis=0), 1, atol=1e-9)
    constant = np.ones((4, T, NUM_FEATURES))
    assert np.all(np.isfinite(Normalizer.fit(constant)(constant)))


def write_experiment(root: Path, seeds: list[int]) -> None:
    for policy in (0, 1):
        for condition in ("undisturbed", "action_scale_sev0.5_on0_durinf"):
            part = root / "parts" / f"FetchPush-v4_seed{policy}" / condition
            part.mkdir(parents=True)
            (part / "done.json").write_text("{}")
            summaries, rows = [], []
            for seed in seeds:
                objects, grippers, goals, actions = random_episode()
                success = rng.random(T + 1) < 0.5
                for t in range(T + 1):
                    row: dict[str, Any] = {
                        "episode_seed": seed,
                        "t": t,
                        "success": bool(success[t]),
                    }
                    for a in range(3):
                        row[f"achieved_{a}"] = objects[t, a]
                        row[f"gripper_{a}"] = grippers[t, a]
                        row[f"goal_{a}"] = goals[t, a]
                    for a in range(4):
                        row[f"base_action_{a}"] = actions[t - 1, a] if t else np.nan
                    rows.append(row)
                summaries.append(
                    {
                        "episode_seed": seed,
                        "final_success": bool(success[-1]),
                        "disturbance_type": condition.split("_sev")[0],
                    }
                )
            pd.DataFrame(rows).to_csv(part / "trajectories.csv.gz", index=False)
            pd.DataFrame(summaries).to_csv(part / "episodes.csv", index=False)


def test_build_datasets_end_to_end(tmp_path):
    write_experiment(tmp_path, list(range(3_500_000, 3_500_010)))
    datasets = build_datasets(tmp_path, validation_fraction=0.2)
    train, val = datasets.train, datasets.validation
    assert len(train.episodes) == 32 and len(val.episodes) == 8
    assert not set(train.episodes["episode_seed"]) & set(val.episodes["episode_seed"])
    assert len(train) == 32 * T
    flat = datasets.data.features[train.episode_ids].reshape(-1, NUM_FEATURES)
    np.testing.assert_allclose(datasets.normalizer.mean, flat.mean(axis=0))
    item = train[T + 3]
    assert item["features"].shape == (NUM_FEATURES,) and item["t"].item() == 3
    assert item["label"].item() == train.episodes["failure"].iloc[1]
    batch = next(iter(DataLoader(train, batch_size=16, shuffle=False)))
    assert batch["features"].shape == (16, NUM_FEATURES) and batch["label"].dtype == torch.float32
    json.dumps(datasets.report())


def test_heldout_seeds_refused(tmp_path):
    write_experiment(tmp_path, [4_000_001])
    with pytest.raises(HeldOutSeedError):
        load_episodes(tmp_path)


REAL = data_dir() / "experiments" / "phase1_extended"


@pytest.mark.skipif(not (REAL / "parts").exists(), reason="phase1_extended outputs absent")
def test_real_dataset_against_logged_schema():
    datasets = build_datasets(REAL)
    report = datasets.report()
    assert report["all"]["episodes"] == 6600 and report["feature_dim"] == NUM_FEATURES
    assert report["duplicates"]["shared_feature_rows_spanning_episode_seeds"] == 0
    episodes = pd.read_csv(REAL / "episodes.csv")
    push = episodes[episodes["env_id"] == "FetchPush-v4"]
    assert report["all"]["failed"] == int((~push["final_success"].astype(bool)).sum())
