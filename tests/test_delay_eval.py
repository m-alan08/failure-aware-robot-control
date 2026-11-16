"""Phase E2: pre-registered delay evaluation."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farc.adaptation.delay_eval import (
    CONFIGURATIONS,
    ShuffledDelay,
    build_conditions,
    derangement,
    derive_tables,
    holm,
    latched_delays,
    load_delay_evaluation,
    paired_difference,
    validate_delay_evaluation,
)
from farc.adaptation.estimator import Estimate
from farc.adaptation.experiment import AdaptationConfigError, true_parameters
from farc.disturbances.perturbations import build_perturbations

CONFIG = "configs/adaptation/phase_e2_delay.json"


def _estimate(step, reliable):
    return Estimate(step, 10, 1, 1.0, 0.99, 5.0, 1e-3, reliable, reliable)


def test_config_loads_with_pinned_plan():
    config = load_delay_evaluation(CONFIG)
    assert len(config.evaluation_seeds) == 100 and max(config.evaluation_seeds) < 4_000_000


def test_changed_plan_is_refused(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text(Path("docs/PHASE_E_DELAY_PREREGISTRATION.md").read_text() + "\nedited\n")
    values = json.loads(Path(CONFIG).read_text())
    values["plan"] = str(plan)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    with pytest.raises(AdaptationConfigError, match="pre-registration changed"):
        load_delay_evaluation(path)


@pytest.mark.parametrize("change", [
    {"evaluation_seed_offset": 3_632_010},  # Phase E pilot evaluation seeds
    {"evaluation_seed_offset": 3_631_000},  # Phase E pilot selection seeds
    {"debug_seed_offset": 3_630_000},  # Phase E pilot debugging seeds
    {"evaluation_seed_offset": 3_999_950},  # crosses into held-out
    {"evaluation_seed_offset": 3_640_005},  # overlaps E2 debugging
])  # fmt: skip
def test_seed_ranges_are_protected(change):
    config = load_delay_evaluation(CONFIG)
    with pytest.raises(AdaptationConfigError):
        validate_delay_evaluation(replace(config, **change), check_plan=False)


def test_conditions_match_the_plan():
    conditions = {c.name: c for c in build_conditions(load_delay_evaluation(CONFIG))}
    assert len(conditions) == 11
    assert true_parameters(conditions["C5_delay3_onset15"], 14) == (1.0, 0)
    assert true_parameters(conditions["C5_delay3_onset15"], 15) == (1.0, 3)
    assert true_parameters(conditions["C6_delay3_steps15to29"], 30) == (1.0, 0)
    assert true_parameters(conditions["C8_delay3_scale0.5"], 5) == (0.5, 3)
    assert true_parameters(conditions["C9_delay1_to3_at20"], 19) == (1.0, 1)
    assert true_parameters(conditions["C9_delay1_to3_at20"], 20) == (1.0, 3)
    assert conditions["C4_delay4"].disturbances[0].severity == 4


def test_delay_schedule_replays_issued_commands():
    """Amendment A2: two consecutive delays execute c[t-1] before step 20, c[t-3] after."""
    condition = {c.name: c for c in build_conditions(load_delay_evaluation(CONFIG))}[
        "C9_delay1_to3_at20"]  # fmt: skip
    perturbations = build_perturbations(condition.disturbances, 1, 2, 50)
    commands: list[np.ndarray] = []
    for t in range(50):
        commands.append(np.full(4, float(t + 1)))
        action = commands[-1]
        for p in perturbations:
            action = p.apply(t, action, commands)
        expected = t - 1 if t < 20 else t - 3
        assert action[0] == (expected + 1 if expected >= 0 else 0.0)
    for p in perturbations:
        p.check_effect()


def test_derangement_has_no_fixed_points_and_is_seeded():
    a, b = derangement(100, 7), derangement(100, 7)
    np.testing.assert_array_equal(a, b)
    assert not np.any(a == np.arange(100)) and sorted(a) == list(range(100))
    with pytest.raises(ValueError):
        derangement(1, 0)


def test_latched_delays_and_shuffled_sham_gating():
    steps = pd.DataFrame({"delay_reliable": [False, True, False, True],
                          "delay_est": [0, 3, 1, 2]})  # fmt: skip
    assert latched_delays(steps) == [None, 3, 3, 2]
    sham = ShuffledDelay([None, 3, 3, 2])
    assert sham.parameters(_estimate(0, False)).delay is None
    assert sham.parameters(_estimate(1, False)).delay is None  # own estimator not ready yet
    assert sham.parameters(_estimate(2, True)).delay == 3  # partner's value, own gate
    assert sham.parameters(_estimate(3, False)).delay == 2  # stays open


def test_holm():
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])


def test_paired_difference_on_known_pairs():
    rows = []
    for seed in range(20):
        rows.append(
            {
                "configuration": "base_only",
                "policy_seed": 0,
                "condition": "c",
                "episode_seed": seed,
                "success": seed < 10,
            }
        )
        rows.append({"configuration": "x", "policy_seed": 0, "condition": "c",
                     "episode_seed": seed, "success": seed < 15})  # fmt: skip
    est, low, high, p, n = paired_difference(pd.DataFrame(rows), "x", "base_only", 2000, 0)
    assert est == pytest.approx(0.25) and n == 20 and low > 0 and p < 0.05


def test_pre_registered_decision_on_synthetic_data():
    config = load_delay_evaluation(CONFIG)
    rng = np.random.default_rng(0)
    episodes, steps = [], []
    for cond in [c["name"] for c in config.conditions]:
        delay = cond in config.delay_conditions
        for seed in range(30):
            base = bool(rng.random() < (0.3 if delay else 0.95))
            for name in CONFIGURATIONS:
                success = base or (delay and name in ("estimated_delay", "oracle_delay",
                                                      "estimated_combined_f0.5"))  # fmt: skip
                episodes.append({"configuration": name, "policy_seed": 0, "condition": cond,
                                 "episode_seed": seed, "success": success,
                                 "near_object_rate": 0.3, "delay_activation_rate": 0.5,
                                 "gain_activation_rate": 0.0, "delay_mismatch_rate": 0.1,
                                 "compensation_saturation_rate": 0.0})  # fmt: skip
                if name in ("base_only", "estimated_delay"):
                    for t in range(5):
                        steps.append({"configuration": name, "policy_seed": 0,
                                      "condition": cond, "episode_seed": seed, "t": t,
                                      "delay_true": 3 if delay and t >= 2 else 0,
                                      "delay_est": 3 if delay and t >= 3 else 0,
                                      "delay_param": 3 if delay and t >= 3 else -1,
                                      "delay_reliable": t >= 1,
                                      "gain_reliable": t >= 1})  # fmt: skip
    tables, summary = derive_tables(pd.DataFrame(episodes), pd.DataFrame(steps),
                                    replace(config, bootstrap_resamples=500))  # fmt: skip
    hyp = tables["hypotheses"].set_index("test")
    assert hyp.loc["H1_pooled_delay_vs_base", "passed"]
    assert hyp.loc["harm_C0_undisturbed", "passed"]
    assert summary["supported"]
    onset = tables["onset_latency"]
    assert (onset["latency_median"] == 1).all()
