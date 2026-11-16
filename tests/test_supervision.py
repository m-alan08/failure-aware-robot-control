"""Phase G1 supervisors: causality, dwell, hysteresis, CUSUM, yoking and equivalences."""

from __future__ import annotations

import inspect
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farc.adaptation.estimator import ActuatorEstimator, Estimate, EstimatorConfig, lag_residuals
from farc.adaptation.experiment import AdaptationConfigError
from farc.paths import checkpoint_path, resolve
from farc.supervision import supervisors
from farc.supervision.episodes import CONFIGURATIONS, YOKED, yoke_source
from farc.supervision.evaluation import calibrate_threshold, derive_g1, load_g1, simulate_release
from farc.supervision.supervisors import (
    ContactGate,
    NeverReleaseLatch,
    PredictorSelector,
    ReleaseSupervisor,
    S,
    SupervisorConfig,
    cusum_increment,
)

CONFIG = "configs/supervision/phase_g1.json"
EST = EstimatorConfig(nominal_gain=0.62)


def _estimate(delay, reliable, step=0):
    return Estimate(step=step, transitions=10, delay=delay, gain=1.0, r2=0.9, margin=3.0,
                    excitation=1.0, gain_reliable=False, delay_reliable=reliable)  # fmt: skip


def _estimator(delay, steps=30, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    est = ActuatorEstimator(EST)
    g = np.zeros(3)
    commands = []
    for _ in range(steps):
        est.observe(g + noise * rng.standard_normal(3))
        c = rng.uniform(-1, 1, 4)
        est.issued(c)
        commands.append(c)
        executed = commands[-1 - delay] if len(commands) > delay else np.zeros(4)
        g = g + 0.62 * 0.05 * executed[:3]
    est.observe(g)
    return est


# --- read-only residual accessor --------------------------------------------------------------


@pytest.mark.parametrize("delay", [0, 1, 3])
def test_lag_residuals_match_the_frozen_estimate(delay):
    est = _estimator(delay)
    before = repr(est.estimate())
    transitions, sse = lag_residuals(est)
    assert transitions == est.estimate().transitions == 10
    assert int(np.argmin(sse)) == est.estimate().delay == delay
    assert repr(est.estimate()) == before  # reading changes nothing
    ordered = np.sort(sse)
    assert math.isclose(ordered[1] / max(ordered[0], 1e-15), est.estimate().margin, rel_tol=1e-9)


# --- never-release latch ------------------------------------------------------------------


def test_never_release_ignores_reliable_zero_only_after_engaging():
    latch = NeverReleaseLatch()
    assert latch.parameters(_estimate(0, True)).delay == 0  # before engaging: as the latch
    assert latch.parameters(_estimate(3, True)).delay == 3
    assert latch.parameters(_estimate(0, True)).delay == 3  # the latch would release here
    assert latch.parameters(_estimate(2, True)).delay == 2  # a new d > 0 still updates


# --- supervisor A ---------------------------------------------------------------------------

CFG = SupervisorConfig(cusum_threshold=2.0)


def test_cusum_increment_arithmetic_and_undefined_cases():
    sse = np.array([1.0, 8.0, np.inf, 4.0, 2.0])
    assert math.isclose(cusum_increment(10, sse, 1, CFG), math.log(8.0) - math.log(2.0))
    assert cusum_increment(3, sse, 1, CFG) is None  # too few transitions
    assert cusum_increment(10, sse, 2, CFG) is None  # no displacement at that lag


def _drive(sup, latched, reliable_delay, sse, steps):
    out = []
    for t in range(steps):
        est = _estimate(reliable_delay[t], reliable_delay[t] is not None, t)
        out.append(sup.allow(t, latched[t], est, 10, sse[t]))
    return out


def test_cusum_release_respects_dwell_and_needs_fresh_evidence_to_reengage():
    n = 30
    good, bad = np.array([1.0, 1, 1, 0.1, 1]), np.array([0.1, 1, 1, 1.0, 1])  # lag 0 vs 3
    sse = [good] * 2 + [bad] * 8 + [bad] * 20
    latched = [3] * n
    reliable = [3, None] + [None] * 18 + [3] * 10
    sup = ReleaseSupervisor(CFG, "cusum")
    allowed = _drive(sup, latched, reliable, sse, n)
    # increment = log(10) - log 2 = 1.61 per bad step; S > 2 at t = 3, but dwell blocks until 5
    first_off = allowed.index(False)
    assert first_off == 5
    # stays off without fresh evidence; re-engages at the first fresh reliable d > 0 (t = 20)
    assert not any(allowed[5:20]) and allowed[20]


def test_release_supervisor_resets_when_the_latch_is_off():
    sup = ReleaseSupervisor(CFG, "cusum")
    bad = np.array([0.1, 1, 1, 1.0, 1])
    for t in range(8):
        sup.allow(t, 3, _estimate(None, False, t), 10, bad)
    assert sup.released
    assert sup.allow(8, 0, _estimate(0, True, 8), 10, bad) is False
    assert not sup.released and sup.statistic == 0.0


def test_unset_threshold_never_releases():
    sup = ReleaseSupervisor(SupervisorConfig(cusum_threshold=None), "cusum")
    bad = np.array([0.1, 1, 1, 1.0, 1])
    assert all(sup.allow(t, 3, _estimate(None, False, t), 10, bad) for t in range(40))
    assert sup.statistic > 10


def test_timeout_releases_after_unconfirmed_steps():
    sup = ReleaseSupervisor(SupervisorConfig(timeout_steps=10), "timeout")
    sse = np.ones(5)
    allowed = [sup.allow(t, 3, _estimate(3 if t == 0 else None, t == 0, t), 10, sse)
               for t in range(15)]  # fmt: skip
    assert allowed[:10] == [True] * 10 and allowed[10] is False


def test_yoked_release_copies_the_mask():
    mask = np.zeros(10, bool)
    mask[4:7] = True
    sup = ReleaseSupervisor(CFG, "yoked", mask=mask)
    allowed = [sup.allow(t, 2, _estimate(None, False, t), 0, None) for t in range(10)]
    assert allowed == [not m for m in mask]
    assert yoke_source([5, 6, 7], 7) == 5 and yoke_source([5, 6, 7], 5) == 6
    assert set(YOKED) <= set(CONFIGURATIONS)
    assert CONFIGURATIONS.index("sup_A_yoked") > CONFIGURATIONS.index("sup_A_release")


# --- supervisor B ---------------------------------------------------------------------------


def _obs(gripper, cube):
    o = np.zeros(25)
    o[S["gripper_position"]], o[S["object_position"]] = gripper, cube
    return o


def _run_selector(errors_learned, errors_phase_e, scale=1.0):
    sel = PredictorSelector(SupervisorConfig())
    choices = []
    for t, (el, ee) in enumerate(zip(errors_learned, errors_phase_e, strict=True)):
        truth = _obs(np.zeros(3), np.zeros(3))
        sel.record(
            t,
            1,
            {
                "learned": _obs(np.array([el * scale, 0, 0]), np.zeros(3)),
                "phase_e": _obs(np.array([ee * scale, 0, 0]), np.zeros(3)),
            },
        )
        sel.score(t + 1, truth * scale)
        choices.append(sel.choose(t + 1))
    return choices


def test_selector_switches_only_with_margin_and_dwell():
    n = 30
    choices = _run_selector([0.010] * n, [0.002] * n)
    assert choices[0] == "phase_e"  # first scored step: phase_e clearly better
    switches = [i for i in range(1, n) if choices[i] != choices[i - 1]]
    assert switches == []
    # within the 20% margin nothing switches
    assert set(_run_selector([0.010] * n, [0.0085] * n)) == {"learned"}
    # alternating dominance cannot switch more often than the dwell allows
    alt = [0.01 if (i // 2) % 2 else 0.001 for i in range(n)]
    choices = _run_selector(alt, [0.005] * n)
    changes = [i for i in range(1, n) if choices[i] != choices[i - 1]]
    assert all(b - a >= 5 for a, b in zip(changes[:-1], changes[1:], strict=False))


def test_selector_comparison_is_scale_invariant():
    rng = np.random.default_rng(3)
    el, ee = rng.uniform(0.001, 0.02, 40), rng.uniform(0.001, 0.02, 40)
    assert _run_selector(el, ee) == _run_selector(el, ee, scale=7.0)


def test_selector_scores_only_forecasts_for_the_observed_step():
    sel = PredictorSelector(SupervisorConfig())
    sel.record(0, 3, {"learned": _obs(np.ones(3), np.zeros(3)),
                      "phase_e": _obs(np.zeros(3), np.zeros(3))})  # fmt: skip
    sel.score(2, _obs(np.zeros(3), np.zeros(3)))
    assert sel.ewma["learned"] is None and 3 in sel.forecasts
    sel.score(3, _obs(np.zeros(3), np.zeros(3)))
    learned = sel.ewma["learned"]
    assert learned is not None and math.isclose(learned, math.sqrt(3))
    assert sel.ewma["phase_e"] == 0.0


# --- supervisor C ---------------------------------------------------------------------------


def test_contact_gate_hysteresis_and_dwell():
    gate = ContactGate(SupervisorConfig())

    def at(distance):
        o = np.zeros(25)
        o[S["object_relative_position"]] = [distance, 0, 0]
        return o

    assert gate.choose(0, at(0.10)) == "phase_e"
    assert gate.choose(2, at(0.04)) == "phase_e"  # dwell after the initial decision
    assert gate.choose(5, at(0.04)) == "learned"
    assert gate.choose(10, at(0.055)) == "learned"  # inside the hysteresis band
    assert gate.choose(11, at(0.07)) == "phase_e"


def test_supervisors_never_see_the_simulator():
    import ast

    tree = ast.parse(inspect.getsource(supervisors))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom | ast.Import)
                for a in n.names} | {n.module for n in ast.walk(tree)
                                     if isinstance(n, ast.ImportFrom)}  # fmt: skip
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    for forbidden in (
        "EpisodeSession",
        "true_parameters",
        "mujoco",
        "Oracle",
        "farc.counterfactual.session",
        "farc.adaptation.experiment",
    ):
        assert forbidden not in imported and forbidden not in names


# --- calibration ----------------------------------------------------------------------------


def _trace(increments, fresh_steps=(), seed=1):
    n = len(increments)
    return pd.DataFrame({"configuration": "sup_A_release", "policy_seed": 0,
                         "condition": "C3_delay3", "episode_seed": seed, "t": range(n),
                         "active": True, "cusum_increment": increments,
                         "est_reliable": [t in fresh_steps for t in range(n)],
                         "est_delay": [3.0] * n})  # fmt: skip


def test_simulated_release_counts_dwell_and_waits_for_fresh_evidence():
    ep = _trace([1.0] * 20, fresh_steps=(15,))
    off, active = simulate_release(ep, 2.5, 5)
    # S > 2.5 at t = 2 but dwell blocks to t = 5; off from 5 until the fresh step 15
    assert active == 20 and off == 10
    assert simulate_release(ep, 1000.0, 5)[0] == 0


def test_calibration_picks_the_smallest_threshold_meeting_the_criterion():
    steps = pd.concat([_trace([0.3] * 50, (), seed=s) for s in range(4)])
    h, curve = calibrate_threshold(steps, [1.0, 5.0, 20.0], 5, 0.01)
    assert h == 20.0
    assert list(curve["false_off_rate"])[0] > 0.01


# --- config ---------------------------------------------------------------------------------


def test_config_pins_the_plan_and_requires_a_threshold(tmp_path):
    config = load_g1(CONFIG)
    assert config.seeds == list(range(3_662_000, 3_662_100))
    assert config.seeds_for("calibration")[0] == 3_661_000
    if config.supervisor["cusum_threshold"] is None:
        with pytest.raises(AdaptationConfigError, match="calibrate"):
            config.supervisor_config()
    plan = tmp_path / "plan.md"
    plan.write_text(Path(config.plan).read_text() + "\nedit\n")
    with pytest.raises(AdaptationConfigError, match="changed"):
        load_g1(_write(tmp_path, plan=str(plan)))


def _write(tmp_path, **changes):
    import json

    values = json.loads(Path(CONFIG).read_text())
    values.update(changes)
    path = tmp_path / "g1.json"
    path.write_text(json.dumps(values))
    return path


@pytest.mark.parametrize("key,value", [("evaluation_seed_offset", 4_000_300),
                                       ("evaluation_seed_offset", 3_653_000),
                                       ("calibration_seed_offset", 3_662_000)])  # fmt: skip
def test_config_refuses_heldout_and_used_seeds(tmp_path, key, value):
    with pytest.raises(AdaptationConfigError):
        load_g1(_write(tmp_path, **{key: value}))


# --- analysis on synthetic outcomes -----------------------------------------------------------


def _synthetic(effects):
    config = replace(load_g1(CONFIG), bootstrap_resamples=300, evaluation_episodes=40)
    rng = np.random.default_rng(0)
    episodes, steps = [], []
    for cond in [c["name"] for c in config.conditions]:
        for seed in config.seeds:
            base = bool(rng.random() < 0.5)
            for name in CONFIGURATIONS:
                extra = effects.get((name, cond[0]), 0.0)
                key = {"configuration": name, "policy_seed": 0, "condition": cond,
                       "episode_seed": seed}  # fmt: skip
                episodes.append({**key, "success": base or seed - config.seeds[0] < extra * 40,
                                 "final_distance": 0.0})  # fmt: skip
                for t in range(4):
                    steps.append(
                        {
                            **key,
                            "t": t,
                            "delay_true": 3
                            if cond[0] != "C" or cond.startswith(("C1", "C2", "C3"))
                            else 0,
                            "delay_param": 3 * (t > 0),
                            "active": t > 0 and name != "base_only",
                            "predictor": "learned" if t else "none",
                            "released": False,
                            "est_ns": 1000,
                            "ctrl_ns": 2000,
                            "learned_calls": int(t > 0 and name != "sup_C_contact_gate"),
                            "mlp_steps": 3,
                            "near_true": t % 2 == 0,
                            "err_learned": 0.01 if name == "sup_B_select" else None,
                            "err_phase_e": 0.02 if name == "sup_B_select" else None,
                        }
                    )
    return pd.DataFrame(episodes), pd.DataFrame(steps), config


def test_analysis_runs_and_applies_holm_and_the_call_criterion():
    effects = {("sup_A_release", "T"): 0.5, ("sup_B_select", "C"): 0.5,
               ("sup_B_select", "T"): 0.5}  # fmt: skip
    episodes, steps, config = _synthetic(effects)
    tables, summary = derive_g1(episodes, steps, config)
    assert summary["hypotheses"]["G1"] == "supported"
    assert summary["hypotheses"]["G3"] == "supported"
    assert summary["hypotheses"]["G4"] == "supported"  # equal success, all calls removed
    hyp = tables["hypotheses"].set_index("hypothesis")
    assert (hyp["p_holm"] >= hyp["p_one_sided"]).all()
    assert summary["call_reduction_G4"] == 1.0
    assert set(tables) >= {"latency", "switching", "computation", "safety_gates"}


# --- simulator equivalence --------------------------------------------------------------------

CHECKPOINT = checkpoint_path("fetch_push_fp32_seed2")
MODEL = resolve("$FARC_DATA_DIR/models/contact_predictor/model.pt")
needs_sim = pytest.mark.skipif(not (CHECKPOINT.exists() and MODEL.exists()),
                               reason="FetchPush checkpoint or contact model absent")  # fmt: skip


@pytest.fixture(scope="module")
def sim():
    from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
    from farc.contact.data import build_conditions
    from farc.contact.dynamics import load_dynamics
    from farc.contact.episodes import learned_predictor, phase_e_predictor
    from farc.disturbances.rollout import FrozenNormalizer
    from farc.envs.factory import make_env

    loaded = load_policy(CHECKPOINT)
    config = load_g1(CONFIG)
    conditions = {c.name: c for c in build_conditions(config.conditions_from, config.conditions)}
    model = load_dynamics(MODEL, config.model_sha256)
    preds = {"learned": learned_predictor(model), "phase_e": phase_e_predictor(0.62)}
    return (make_env(loaded.env_config), deterministic_policy_function(loaded.actor),
            FrozenNormalizer(loaded.normalizer), conditions, preds)  # fmt: skip


def _session(sim, condition, seed=3_000_778):
    from farc.counterfactual.session import EpisodeSession

    env, policy, normalizer, conditions, _ = sim
    return EpisodeSession(env, policy, normalizer, conditions[condition], seed, 5)


@needs_sim
@pytest.mark.parametrize("name,f1", [("base_only", None), ("phase_e_predictor", "phase_e"),
                                     ("contact_learned", "learned"),
                                     ("oracle_sim_predictor", "oracle")])  # fmt: skip
def test_reproduces_the_f1_episode_loop(sim, name, f1):
    from farc.contact.episodes import run_episode
    from farc.supervision.episodes import run_supervised_episode

    preds, conditions = sim[4], sim[3]
    cond = conditions["C3_delay3"]
    a = _session(sim, "C3_delay3")
    run_supervised_episode(a, ActuatorEstimator(EST), name, preds, cond, SupervisorConfig())
    b = _session(sim, "C3_delay3")
    predictor = preds.get(f1) if f1 in preds else None
    run_episode(b, ActuatorEstimator(EST), predictor, cond, use_oracle=f1 == "oracle")
    np.testing.assert_array_equal(np.array(a.commands), np.array(b.commands))


@needs_sim
def test_supervisors_without_triggers_match_contact_learned(sim):
    from farc.supervision.episodes import run_supervised_episode

    preds, conditions = sim[4], sim[3]
    cond = conditions["T1_delay3_steps0to14"]
    ref = _session(sim, "T1_delay3_steps0to14")
    run_supervised_episode(ref, ActuatorEstimator(EST), "contact_learned", preds, cond,
                           SupervisorConfig())  # fmt: skip
    a = _session(sim, "T1_delay3_steps0to14")
    run_supervised_episode(a, ActuatorEstimator(EST), "sup_A_release", preds, cond,
                           SupervisorConfig(cusum_threshold=None))  # fmt: skip
    np.testing.assert_array_equal(np.array(a.commands), np.array(ref.commands))
    y = _session(sim, "T1_delay3_steps0to14")
    run_supervised_episode(y, ActuatorEstimator(EST), "sup_A_yoked", preds, cond,
                           SupervisorConfig(), yoked=np.zeros(50, bool))  # fmt: skip
    np.testing.assert_array_equal(np.array(y.commands), np.array(ref.commands))
    yb = _session(sim, "T1_delay3_steps0to14")
    run_supervised_episode(
        yb,
        ActuatorEstimator(EST),
        "sup_B_yoked",
        preds,
        cond,
        SupervisorConfig(),
        yoked=np.array(["learned"] * 50, object),
    )
    np.testing.assert_array_equal(np.array(yb.commands), np.array(ref.commands))


@needs_sim
def test_deployable_predictors_get_only_view_and_issued_queue(sim):
    from farc.supervision.episodes import run_supervised_episode

    conditions = sim[3]
    seen = []

    def spy(name):
        def predict(observation, queued, gain):
            seen.append((type(observation), [type(q) for q in queued], type(gain)))
            return sim[4][name](observation, queued, gain)

        return predict

    preds = {"learned": spy("learned"), "phase_e": spy("phase_e")}
    session = _session(sim, "C6_delay3_posnoise0.005")
    run_supervised_episode(session, ActuatorEstimator(EST), "sup_B_select", preds,
                           conditions["C6_delay3_posnoise0.005"],
                           SupervisorConfig())  # fmt: skip
    assert seen
    for obs_type, queue_types, gain_type in seen:
        assert obs_type is np.ndarray and set(queue_types) <= {np.ndarray}
        assert gain_type is float
