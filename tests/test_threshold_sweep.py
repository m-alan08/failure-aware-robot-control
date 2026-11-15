# Migrated from m-alan08/quantrl-fetch tests/test_threshold_sweep.py
# (introduced in 91e44be, last changed in 3815180). See PROVENANCE.md.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from farc.detection.phase_aware import PhaseDetectorConfig, run_phase_detector
from farc.detection.progress_deficit import DetectorConfig, run_detector
from farc.detection.threshold_sweep import (
    SweepConfigError,
    _stratified_indices,
    detector_names,
    episode_scores,
    exact_curve,
    first_departure,
    lead_score,
    load_sweep_config,
    run_minimum,
    threshold_for_fpr,
)
from farc.paths import data_dir

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "configs/detectors/threshold_sweep_push.json"
SOURCE = data_dir() / "experiments" / "phase1_extended"
T = 50


def random_push_episode(rng: np.random.Generator):
    """A gripper that wanders toward and pushes an object with random speed changes, plus an
    occasional goal shift: enough variety to hit every detector branch."""
    goal = np.array([1.45, 0.75 + rng.uniform(-0.1, 0.1), 0.42])
    obj = np.array([1.3, 0.75, 0.42])
    grip = obj + np.array([-0.2, rng.uniform(-0.05, 0.05), 0.05])
    grippers, objects, goals = [grip.copy()], [obj.copy()], [goal.copy()]
    shift_at = int(rng.integers(15, 45)) if rng.random() < 0.4 else None
    for t in range(T):
        target = obj - 0.04 * (goal - obj) / max(np.linalg.norm(goal - obj), 1e-9)
        grip = grip + rng.uniform(0.0, 0.6) * (target - grip) + rng.normal(0, 0.005, 3)
        if np.linalg.norm(grip - obj) < 0.07:
            obj = obj + rng.uniform(0.0, 0.03) * (goal - obj) / max(
                np.linalg.norm(goal - obj), 1e-9
            )
            obj[2] = 0.42
        if shift_at is not None and t == shift_at:
            goal = goal + np.array([0.0, 0.12, 0.0])
        grippers.append(grip.copy())
        objects.append(obj.copy())
        goals.append(goal.copy())
    return np.array(grippers), np.array(objects), np.array(goals)


def _check_equivalence(gripper, objects, goals, thresholds):
    for k, p in [(5, 3), (10, 5), (3, 2)]:
        distance = run_detector(objects, goals, DetectorConfig(k, p))
        phase = run_phase_detector(gripper, objects, goals, PhaseDetectorConfig(k, p, 3, None))
        departure = first_departure(phase.departure_count, 3)
        for deficit, kind in ((distance.deficit, "distance"), (phase.deficit, "phase")):
            run_min = run_minimum(deficit, p)
            for theta in thresholds:
                if kind == "distance":
                    warning = run_detector(
                        objects, goals, DetectorConfig(k, p, theta)
                    ).first_warning
                    with_departure = None
                else:
                    warning = run_phase_detector(
                        gripper, objects, goals, PhaseDetectorConfig(k, p, 3, None,
                                                                     deficit_threshold=theta)
                    ).first_warning  # fmt: skip
                    with_departure = run_phase_detector(
                        gripper, objects, goals, PhaseDetectorConfig(k, p, 3, 3,
                                                                     deficit_threshold=theta)
                    ).first_warning  # fmt: skip
                for lead in (0, 5, 10):
                    in_time = warning is not None and T - warning > lead
                    assert (lead_score(run_min, lead) > theta) == in_time, (kind, k, p, theta)
                    if with_departure is not None or kind == "phase":
                        expected = with_departure is not None and T - with_departure > lead
                        score = lead_score(run_min, lead, departure)
                        assert (score > theta) == expected, ("departure", k, p, theta, lead)


def test_scores_reproduce_both_detectors_at_every_threshold():
    rng = np.random.default_rng(0)
    thresholds = [-0.05, -0.01, 0.0, 0.002, 0.01, 0.03, 0.1]
    for _ in range(25):
        gripper, objects, goals = random_push_episode(rng)
        _check_equivalence(gripper, objects, goals, thresholds)


@pytest.mark.skipif(not (SOURCE / "parts").exists(), reason="gripper rerun not present")
def test_scores_reproduce_detectors_on_logged_episodes():
    import pandas as pd

    part = SOURCE / "parts/FetchPush-v4_seed0/action_scale_sev0.75_on0_durinf"
    frame = pd.read_csv(part / "trajectories.csv.gz").sort_values(["episode_seed", "t"])
    for _, group in list(frame.groupby("episode_seed"))[:15]:
        gripper = group[[f"gripper_{i}" for i in range(3)]].to_numpy()
        objects = group[[f"achieved_{i}" for i in range(3)]].to_numpy()
        goals = group[[f"goal_{i}" for i in range(3)]].to_numpy()
        _check_equivalence(gripper, objects, goals, [-0.02, 0.0, 0.005, 0.02])


def test_run_minimum_breaks_on_unavailable_states():
    deficit = np.array([np.nan, 0.3, 0.2, np.nan, 0.5, 0.4, 0.6])
    assert np.array_equal(
        run_minimum(deficit, 2), [-np.inf, -np.inf, 0.2, -np.inf, -np.inf, 0.4, 0.4]
    )
    assert np.array_equal(run_minimum(deficit, 1), np.where(np.isnan(deficit), -np.inf, deficit))


def test_lead_score_and_departure():
    run_min = np.full(T, -np.inf)
    run_min[44] = 0.2
    assert lead_score(run_min, 0) == 0.2 and lead_score(run_min, 5) == 0.2
    assert lead_score(run_min, 6) == -np.inf  # state 44 leaves exactly 6 steps, not more
    assert lead_score(run_min, 10, departure_state=30) == np.inf
    assert lead_score(run_min, 25, departure_state=30) == -np.inf
    assert first_departure(np.array([0, 1, 2, 3, 0]), 3) == 3
    assert first_departure(np.array([0, 1, 2]), None) is None


def test_threshold_for_fpr_is_conservative_with_ties():
    successes = np.array([-np.inf, -np.inf, 0.1, 0.1, 0.1, 0.5, 0.9, 1.0, 2.0, 3.0])
    for target in (0.05, 0.1, 0.2, 0.3, 0.6):
        threshold = threshold_for_fpr(successes, target)
        assert np.mean(successes > threshold) <= target + 1e-12
    assert threshold_for_fpr(successes, 0.2) == 1.0  # two allowed: 2.0 and 3.0 warn
    # a tie block cannot be split: allowing 7 warnings on ties at 0.1 drops to the next value
    assert np.mean(successes > threshold_for_fpr(successes, 0.7)) == 0.5


def test_exact_curve_endpoints_and_monotonicity():
    fails = np.array([-np.inf, 0.2, 0.4, np.inf])
    succs = np.array([-np.inf, 0.1, 0.3])
    curve = exact_curve(fails, succs)
    assert curve["detection"].iloc[0] == 0.75 and curve["fpr"].iloc[0] == pytest.approx(2 / 3)
    # the +inf failure (a departure warning) warns at every threshold
    assert curve["detection"].iloc[-1] == 0.25 and curve["fpr"].iloc[-1] == 0.0
    assert (np.diff(curve["detection"]) <= 0).all() and (np.diff(curve["fpr"]) <= 0).all()


def test_unreachable_fpr_with_threshold_independent_warnings():
    successes = np.array([np.inf, np.inf, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    assert np.isnan(threshold_for_fpr(successes, 0.1))  # 2 departures already exceed 1
    assert np.mean(successes > threshold_for_fpr(successes, 0.2)) == 0.2


def test_stratified_resampling_preserves_stratum_sizes():
    strata = np.array(["a"] * 5 + ["b"] * 3)
    indices = _stratified_indices(strata, 50, np.random.default_rng(0))
    assert indices.shape == (50, 8)
    assert ((strata[indices] == "a").sum(axis=1) == 5).all()


def test_config_and_score_keys(tmp_path):
    config = load_sweep_config(CONFIG)
    rng = np.random.default_rng(1)
    gripper, objects, goals = random_push_episode(rng)
    scores = episode_scores({"gripper": gripper, "objects": objects, "goals": goals}, config)
    assert len(scores) == 2 * 2 * 3 * len(config.min_leads)
    values = json.loads(CONFIG.read_text())
    (tmp_path / "bad.json").write_text(json.dumps({**values, "target_fprs": [1.5]}))
    with pytest.raises(SweepConfigError):
        load_sweep_config(tmp_path / "bad.json")


def test_per_seed_matched_envelope_is_the_best_member():
    import pandas as pd

    from farc.detection.threshold_sweep import per_seed_matched

    config = load_sweep_config(CONFIG)
    rng = np.random.default_rng(3)
    rows = []
    for seed in (0, 1):
        for episode in range(40):
            row: dict[str, object] = {
                "training_seed": seed, "disturbance_type": "x", "failure": episode % 3 == 0
            }  # fmt: skip
            for name, *_ in detector_names(config):
                base = rng.normal()
                for lead in config.min_leads:
                    row[f"{name}|{lead}"] = base - 0.1 * lead
            rows.append(row)
    table = per_seed_matched(pd.DataFrame(rows), config)
    part = table[(table["training_seed"] == 0) & (table["min_lead"] == 5)
                 & (table["target_fpr"] == 0.1)]  # fmt: skip
    members = part[part["detector"].str.startswith("distance")]["detection"]
    envelope = part[part["detector"] == "envelope:distance_only"]["detection"].iloc[0]
    assert envelope == members.max()
