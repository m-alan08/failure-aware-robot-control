"""G2 candidates, the Stage-3 grid, calibration rules and analysis."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.adaptation.experiment import AdaptationConfigError
from farc.delaychange.candidates import Sequential, ShortWindow, TwoTimescale, window_fit
from farc.delaychange.stage3 import (
    CANDIDATES,
    EVALUATION,
    Latched,
    calibration_cells,
    calibration_configurations,
    choose_constants,
    detection_metrics,
    evaluation_cells,
    load_g2,
)
from farc.delaychange.stage3_analysis import hypotheses

EST = EstimatorConfig(nominal_gain=0.62)
CONFIG = "configs/delaychange/g2_stage3.json"


def drive(schedule, controllers, steps=50, seed=0, noise=0.0):
    """Synthetic plant: g += 0.62 * 0.05 * executed, executed = c[t - d(t)] (replay/drop).
    Returns per-controller decisions."""
    rng = np.random.default_rng(seed)
    est = ActuatorEstimator(EST)
    g, cmds = np.zeros(3), []
    out: dict[str, list] = {k: [] for k in controllers}
    for t in range(steps):
        est.observe(g + noise * rng.standard_normal(3))
        for k, ctrl in controllers.items():
            out[k].append(ctrl.step(est))
        c = rng.uniform(-1, 1, 4)
        est.issued(c)
        cmds.append(c)
        d = schedule(t)
        executed = cmds[t - d] if t - d >= 0 else np.zeros(4)
        g = g + 0.62 * 0.05 * executed[:3]
    return out


@pytest.mark.parametrize("delay", [0, 2, 4])
def test_window_fit_reproduces_the_frozen_estimate(delay):
    rng = np.random.default_rng(delay)
    est = ActuatorEstimator(EST)
    g, cmds = np.zeros(3), []
    for t in range(30):
        est.observe(g)
        e, _ = window_fit(est, 10)
        assert repr(e) == repr(est.estimate())
        c = rng.uniform(-1, 1, 4)
        est.issued(c)
        cmds.append(c)
        g = g + 0.62 * 0.05 * (cmds[t - delay] if t >= delay else np.zeros(4))[:3]


def test_untriggered_candidates_match_the_latch():
    def const(t):
        return 3

    out = drive(const, {"latch": Latched(), "A10": ShortWindow(10),
                        "B": TwoTimescale(5, consecutive=10**9),
                        "C": Sequential(threshold=1e12, sigma0=0.002)})  # fmt: skip
    ref = [(d.active, d.delay) for d in out["latch"]]
    for k in ("A10", "B", "C"):
        assert [(d.active, d.delay) for d in out[k]] == ref, k


def _first(decisions, predicate, start):
    return next((t for t in range(start, len(decisions)) if predicate(decisions[t])), None)


def test_end_of_delay_is_detected_faster_and_never_with_an_unverified_lag():
    def end(t):
        return 3 if t < 20 else 0

    out = drive(end, {"latch": Latched(), "B": TwoTimescale(4),
                      "C": Sequential(threshold=4.0, sigma0=0.001)})  # fmt: skip
    off = {k: _first(v, lambda d: not d.active, 20) for k, v in out.items()}
    assert off["C"] is not None and off["latch"] is not None and off["C"] < off["latch"]
    assert off["B"] is not None and off["B"] <= off["latch"]
    for k in ("B", "C"):  # while active, only the verified delay is ever applied
        assert all(d.delay == 3 for d in out[k][: off[k]] if d.active)
    assert any(d.alarm for d in out["C"][20:])


def test_sequential_detects_an_onset_and_identifies_from_post_onset_data():
    def onset(t):
        return 0 if t < 20 else 3

    out = drive(onset, {"latch": Latched(), "C": Sequential(threshold=4.0, sigma0=0.001)})
    on = {k: _first(v, lambda d: d.active and d.delay == 3, 20) for k, v in out.items()}
    assert on["C"] is not None and on["C"] <= on["latch"]
    assert on["C"] - 20 >= 4  # re-identification still needs min_transitions post-change


def test_change_between_nonzero_delays_suspends_then_reidentifies():
    def change(t):
        return 1 if t < 20 else 4

    out = drive(change, {"C": Sequential(threshold=4.0, sigma0=0.001)})["C"]
    alarm = _first(out, lambda d: d.alarm, 20)
    assert alarm is not None
    new = _first(out, lambda d: d.active and d.delay == 4, 20)
    assert new is not None and all((not d.active) or d.delay == 1 for d in out[20:new])


def test_two_timescale_waits_for_the_dwell_before_reengaging():
    def end(t):
        return 3 if t < 20 else 0

    out = drive(end, {"B": TwoTimescale(4)})["B"]
    alarms = [t for t, d in enumerate(out) if d.alarm]
    assert alarms
    for t in range(alarms[0], min(alarms[0] + 5, len(out))):
        assert not out[t].active


def test_grids_are_the_preregistered_ones():
    cells = evaluation_cells()
    assert len(cells) == 38
    count = pd.Series([c.family for c in cells]).value_counts().to_dict()
    assert count == {"T": 18, "P": 6, "I": 4, "S": 4, "NG": 4, "H": 2}
    for c in cells:
        if c.family in ("T", "I"):
            assert min(o for o, _, _ in c.pulses) > 0  # no start-up cue
            assert {d for _, _, d in c.pulses} <= {2, 4}
    assert {c.name for c in cells}.isdisjoint({c.name for c in calibration_cells()})
    assert len(calibration_configurations()) == 1 + 4 + 2 + 42
    assert len(EVALUATION) == 11


def test_calibration_rules():
    rows = [
        {
            "configuration": "contact_learned",
            "wrong_reliable_rate": 0.010,
            "suspended_rate": 0.0,
            "false_activation_rate": 0.0,
            "release_latency_median": 9,
        }
    ]
    for w, wrong in ((4, 0.03), (5, 0.014), (6, 0.012), (7, 0.011)):
        rows.append(
            {
                "configuration": f"cand_A_w{w}",
                "wrong_reliable_rate": wrong,
                "suspended_rate": 0,
                "false_activation_rate": 0.0,
                "release_latency_median": w,
            }
        )
    rows += [
        {
            "configuration": "cand_B_wf4",
            "wrong_reliable_rate": 0,
            "suspended_rate": 0.02,
            "false_activation_rate": 0,
            "release_latency_median": 4,
        },
        {
            "configuration": "cand_B_wf5",
            "wrong_reliable_rate": 0,
            "suspended_rate": 0.005,
            "false_activation_rate": 0,
            "release_latency_median": 5,
        },
    ]
    for name, susp, lat in (
        ("cand_C_gauss_h2_s1", 0.02, 1),
        ("cand_C_gauss_h4_s2", 0.005, 2),
        ("cand_C_scalefree_h6_s4", 0.001, 2),
        ("cand_C_gauss_h8_s4", 0.0, 3),
    ):
        rows.append(
            {
                "configuration": name,
                "wrong_reliable_rate": 0,
                "suspended_rate": susp,
                "false_activation_rate": 0,
                "release_latency_median": lat,
            }
        )
    chosen = choose_constants(pd.DataFrame(rows))
    assert chosen["A_window"] == 5 and chosen["B_fast_window"] == 5
    # latency tie at 2: the larger threshold wins
    assert (chosen["C_threshold"], chosen["C_sigma0_mm"], chosen["C_variant"]) == (
        6.0,
        4.0,
        "scalefree",
    )


def test_detection_metrics_definitions():
    rows = []
    for t in range(10):  # persistent d=2, engaged from t=2, one off step at t=5
        rows.append(
            {
                "configuration": "x",
                "policy_seed": 0,
                "condition": "K_d2",
                "episode_seed": 1,
                "t": t,
                "delay_true": 2,
                "active": t >= 2 and t != 5,
                "est_reliable": t >= 2,
                "est_delay": 2 if t != 7 else 1,
            }
        )
    for t in range(10):  # transient end at 4, off at 7 -> latency 3
        rows.append(
            {
                "configuration": "x",
                "policy_seed": 0,
                "condition": "KT_a",
                "episode_seed": 1,
                "t": t,
                "delay_true": 3 if t < 4 else 0,
                "active": 1 <= t < 7,
                "est_reliable": False,
                "est_delay": None,
            }
        )
    for t in range(10):
        rows.append(
            {
                "configuration": "x",
                "policy_seed": 0,
                "condition": "KH_u",
                "episode_seed": 1,
                "t": t,
                "delay_true": 0,
                "active": t == 3,
                "est_reliable": False,
                "est_delay": None,
            }
        )
    m = detection_metrics(pd.DataFrame(rows)).iloc[0]
    assert math.isclose(m["wrong_reliable_rate"], 0.1)
    assert math.isclose(m["suspended_rate"], 1 / 8)
    assert math.isclose(m["false_activation_rate"], 0.1)
    assert m["release_latency_median"] == 3


def test_evaluation_refuses_without_a_pinned_calibration_record(tmp_path):
    values = json.loads(Path(CONFIG).read_text())
    if values["calibration_record"] is None:
        with pytest.raises(AdaptationConfigError, match="calibration record"):
            load_g2(CONFIG)
    load_g2(CONFIG, "calibration")
    values["evaluation_seed_offset"] = 4_000_300
    path = tmp_path / "c.json"
    path.write_text(json.dumps(values))
    with pytest.raises(AdaptationConfigError):
        load_g2(path, "calibration")


def test_primary_is_an_intersection_union_test_with_holm():
    config = replace(load_g2(CONFIG, "calibration"), bootstrap_resamples=400)
    rng = np.random.default_rng(0)
    rows = []
    cells = evaluation_cells()
    effects = {
        ("cand_C_sequential", "T"): 0.3,
        ("cand_C_sequential", "I"): 0.3,
        ("cand_A_short_window", "T"): 0.3,
        ("cand_A_short_window", "I"): 0.3,
        ("cand_A_short_window", "P"): -0.3,
    }  # A trades persistent performance away
    for cell in cells:
        for seed in range(40):
            u = rng.random()
            for name in EVALUATION:
                bonus = effects.get((name, cell.family), 0.0)
                rows.append(
                    {
                        "configuration": name,
                        "policy_seed": 0,
                        "condition": cell.name,
                        "family": cell.family,
                        "episode_seed": seed,
                        "success": bool(u < 0.5 + bonus),
                    }
                )
    prim, sec, gates = hypotheses(pd.DataFrame(rows), config)
    status = dict(zip(prim["candidate"], prim["status"], strict=True))
    assert status["cand_C_sequential"] == "supported"
    assert status["cand_A_short_window"] == "not supported"  # H2 fails
    assert status["cand_B_two_timescale"] == "not supported"  # H1 fails
    assert (prim["p_holm"] >= prim["p_iut"]).all()
    assert set(gates["configuration"]) == set(CANDIDATES)


# --- simulator equivalence ----------------------------------------------------------------------

from farc.paths import checkpoint_path, resolve  # noqa: E402

CHECKPOINT = checkpoint_path("fetch_push_fp32_seed1")
MODEL = resolve("$FARC_DATA_DIR/models/contact_predictor/model.pt")
needs_sim = pytest.mark.skipif(not (CHECKPOINT.exists() and MODEL.exists()),
                               reason="checkpoint or contact model absent")  # fmt: skip


@pytest.fixture(scope="module")
def sim():
    from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
    from farc.contact.dynamics import load_dynamics
    from farc.contact.episodes import learned_predictor, phase_e_predictor
    from farc.disturbances.rollout import FrozenNormalizer
    from farc.envs.factory import make_env

    loaded = load_policy(CHECKPOINT)
    preds = {"learned": learned_predictor(load_dynamics(MODEL)),
             "phase_e": phase_e_predictor(0.62)}  # fmt: skip
    return (make_env(loaded.env_config), deterministic_policy_function(loaded.actor),
            FrozenNormalizer(loaded.normalizer), preds)  # fmt: skip


def _commands(sim, runner, pulses=((6, 16, 3),), seed=3_000_903):
    from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
    from farc.disturbances.config import ConditionSpec

    env, policy, normalizer, preds = sim
    session = ScheduledDelaySession(env, policy, normalizer, ConditionSpec("x", ()), seed, 9,
                                    DelaySchedule(pulses))  # fmt: skip
    runner(session, preds)
    return np.array(session.commands)


@needs_sim
def test_stage3_runner_matches_stage1_diagnostics(sim):
    from farc.delaychange.episodes import DIAGNOSTICS, run_diagnostic_episode
    from farc.delaychange.stage3 import run_g2_episode

    big = {"A_window": 10, "B_fast_window": 4, "C_threshold": 1e12, "C_sigma0_mm": 2,
           "C_variant": "gauss"}  # fmt: skip

    def stage1(name):
        return lambda s, p: run_diagnostic_episode(s, ActuatorEstimator(EST), DIAGNOSTICS[name], p)

    def stage3(name, yoked=None):
        return lambda s, p: run_g2_episode(s, ActuatorEstimator(EST), name, p, big, yoked)

    ref = _commands(sim, stage1("deployed_learned"))
    for name in ("contact_learned", "cand_A_short_window", "cand_C_sequential"):
        np.testing.assert_array_equal(_commands(sim, stage3(name)), ref, err_msg=name)
    np.testing.assert_array_equal(_commands(sim, stage3("sham_C_yoked", np.zeros(50, bool))), ref)
    np.testing.assert_array_equal(
        _commands(sim, stage3("PRIV_true_schedule")), _commands(sim, stage1("true_learned"))
    )
    np.testing.assert_array_equal(
        _commands(sim, stage3("PRIV_release_oracle")), _commands(sim, stage1("release_k0"))
    )
    np.testing.assert_array_equal(
        _commands(sim, stage3("phase_e_predictor")), _commands(sim, stage1("deployed_phase_e"))
    )
    np.testing.assert_array_equal(
        _commands(sim, stage3("base_only")), _commands(sim, stage1("base_only"))
    )
