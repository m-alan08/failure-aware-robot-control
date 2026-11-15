from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from farc.detection.threshold_sweep import lead_score
from farc.prediction.dataset import Normalizer
from farc.prediction.evaluation import (
    CheckpointError,
    EvaluationData,
    FailurePredictor,
    auroc,
    average_precision,
    bootstrap,
    brier,
    calibration_split,
    expected_calibration_error,
    first_warning,
    lead_maxima,
    load_predictor,
    thresholds,
    verify_normalizer,
    warning_metrics,
)

T = 50
REPO = Path(__file__).resolve().parents[1]


def test_checkpoint_compatibility(tmp_path):
    model = FailurePredictor()
    good = tmp_path / "good.pt"
    torch.save(model.state_dict(), good)
    loaded = load_predictor(good)
    x = torch.randn(3, 24)
    torch.testing.assert_close(loaded(x), model.eval()(x))
    bad = FailurePredictor(23)
    torch.save(bad.state_dict(), tmp_path / "bad.pt")
    with pytest.raises(CheckpointError):
        load_predictor(tmp_path / "bad.pt")
    with pytest.raises(CheckpointError):
        load_predictor(tmp_path / "missing.pt")


def test_verify_normalizer():
    n = Normalizer(np.arange(24.0), np.ones(24))
    verify_normalizer(n, {"normalizer": n.state_dict()})
    with pytest.raises(CheckpointError):
        verify_normalizer(n, {"normalizer": {"mean": [0.0] * 24, "std": [1.0] * 24}})
    with pytest.raises(CheckpointError):
        verify_normalizer(n, {})


def test_committed_report_matches_recomputable_normalizer_shape():
    report = json.loads((REPO / "results/failure_dataset/report.json").read_text())
    assert len(report["normalizer"]["mean"]) == 24 == report["feature_dim"]


def test_strict_lead_criteria():
    """A first warning at t counts for lead > L iff T - t > L; later warnings never rescue a
    late first warning, and step-T-L warnings do not count."""
    scores = np.full((4, T), -1.0)
    scores[0, 39] = 1  # T - t = 11: counts for >10
    scores[1, 40] = 1  # T - t = 10: not >10, counts for >5
    scores[2, 45] = 1  # T - t = 5: not >5
    scores[3, 49] = 1  # last decision step: only "any warning"
    m = warning_metrics(scores, np.ones(4, bool), 0.0, (0, 5, 10))
    assert m["detection_lead0"] == 1.0
    assert m["detection_lead5"] == 0.5
    assert m["detection_lead10"] == 0.25
    np.testing.assert_array_equal(first_warning(scores, 0.0), [39, 40, 45, 49])


def test_lead_maxima_match_first_warning_rule_and_sweep():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(200, T))
    failure = np.ones(200, bool)
    maxima = lead_maxima(scores, (0, 5, 10))
    for theta in (-1.0, 0.5, 2.0):
        m = warning_metrics(scores, failure, theta, (0, 5, 10))
        for lead in (0, 5, 10):
            assert m[f"detection_lead{lead}"] == pytest.approx(np.mean(maxima[lead] > theta))
            assert maxima[lead][7] == lead_score(scores[7], lead)


def test_timestep_metrics_reference_values():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert auroc(y, p) == pytest.approx(0.75)
    assert average_precision(y, p) == pytest.approx((1 + 2 / 3) / 2)
    assert brier(y, p) == pytest.approx(np.mean((p - y) ** 2))
    assert auroc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(0.5)
    assert expected_calibration_error(np.array([0, 1] * 50), np.full(100, 0.5)) == 0.0


def fake_data(seed: int = 0) -> EvaluationData:
    rng = np.random.default_rng(seed)
    seeds = np.arange(3_500_000, 3_500_020)
    rows = [
        {"episode_seed": s, "condition": c, "policy_seed": p, "failure": int(rng.random() < 0.3)}
        for s in seeds
        for c in ("undisturbed", "a", "b")
        for p in (0, 1)
    ]
    episodes = pd.DataFrame(rows)
    episodes.loc[episodes["condition"] == "undisturbed", "failure"] = 0
    calibration, evaluation = calibration_split(seeds, 0.5, 0)
    episodes["part"] = np.where(episodes["episode_seed"].isin(calibration), "calibration",
                                "evaluation")  # fmt: skip
    episodes["disturbed"] = episodes["condition"] != "undisturbed"
    f = episodes["failure"].to_numpy()[:, None]
    good = rng.normal(size=(len(episodes), T)) + 2 * f
    noise = rng.normal(size=(len(episodes), T))
    return EvaluationData(episodes, {"good": good, "noise": noise}, ("good",), calibration,
                          evaluation)  # fmt: skip


def test_calibration_split_is_seed_disjoint():
    seeds = np.arange(20)
    calibration, evaluation = calibration_split(seeds, 0.5, 0)
    assert len(calibration) == len(evaluation) == 10
    assert not set(calibration) & set(evaluation)
    assert set(calibration) | set(evaluation) == set(seeds)


def test_thresholds_use_only_calibration_successes():
    data = fake_data()
    before = thresholds(data, (0.05,))
    e = data.episodes
    evaluation = (e["part"] == "evaluation").to_numpy()
    data.scores["good"][evaluation] += 100.0  # anything on evaluation episodes is ignored
    assert thresholds(data, (0.05,)) == before
    calibration_success = (~evaluation) & e["disturbed"].to_numpy() & (e["failure"] == 0).to_numpy()
    realized = np.mean(
        data.scores["good"][calibration_success].max(axis=1) > before[("good", 0.05)]
    )
    assert realized <= 0.05


def test_paired_bootstrap_differences():
    data = fake_data()
    rates, diffs = bootstrap(data, (0.1,), (0, 5), [("good", "noise")], 200, 0)
    row = diffs[(diffs["metric"] == "detection_lead0")].iloc[0]
    assert row["ci_low"] <= row["difference"] <= row["ci_high"]
    assert row["difference"] > 0 and row["ci_low"] > 0
    same = bootstrap(data, (0.1,), (0,), [("good", "good")], 50, 0)[1]
    assert np.allclose(same[["difference", "ci_low", "ci_high"]].to_numpy(), 0)
    assert set(rates["metric"]) == {"fpr", "detection_lead0", "detection_lead5"}
