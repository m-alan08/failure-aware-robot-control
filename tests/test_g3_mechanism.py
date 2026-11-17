"""Phase G3: seeds, schedules, EventReplay, the runner and the pre-written analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.adaptation.experiment import AdaptationConfigError
from farc.delaychange.candidates import Sequential
from farc.delaychange.mechanism import (
    CONFIGURATIONS,
    EventReplay,
    load_g3,
    oracle_events,
    rotate,
    run_episode,
)
from farc.delaychange.mechanism_analysis import (
    INTERACTION,
    IntegrityError,
    contrast,
    derive_g3,
    divergence,
    hypotheses,
    label,
)
from farc.delaychange.random_schedules import draw_pulses, g3_cells, schedule_for, transitions
from farc.delaychange.schedule import DelaySchedule
from farc.delaychange.stage3 import Latched
from farc.supervision.seeds import check_g3_seeds

EST = EstimatorConfig(nominal_gain=0.62)
CONFIG = "configs/delaychange/g3_mechanism.json"
C_ARGS = (12.0, 0.004, "gauss")


# --- seeds ------------------------------------------------------------------------------------


def test_seed_blocks_are_fresh_and_below_heldout():
    config = load_g3(CONFIG)
    assert config.seeds[0] == 3_681_000 and config.seeds[-1] == 3_681_999
    assert load_g3(CONFIG, "debugging").seeds == list(range(3_680_000, 3_680_010))
    for bad in ([3_674_050], [3_673_000], [4_000_300], [4_000_000], [3_682_000]):
        with pytest.raises(AdaptationConfigError):
            check_g3_seeds(bad, "evaluation")


# --- schedules --------------------------------------------------------------------------------


def test_schedules_are_deterministic_bounded_and_independent_of_policy():
    cells = g3_cells()
    for i, cell in enumerate(cells):
        for seed in range(3_681_000, 3_681_300):
            p = draw_pulses(cell, i, seed)
            assert p == draw_pulses(cell, i, seed)
            s = DelaySchedule(p)  # validates non-overlap
            tr = transitions(s)
            if cell.family == "RT":
                ((onset, dur, d),) = p
                assert dur is not None
                assert 4 <= onset <= 34 and 3 <= dur <= 16 and onset + dur <= 46
                assert [k for _, k in tr] == ["onset", "end"]
            elif cell.family == "RI":
                assert 4 <= p[0][0] <= 12 and len(p) >= 2
                for (o1, d1, _), (o2, _, _) in zip(p[:-1], p[1:], strict=True):
                    assert d1 is not None
                    assert 3 <= d1 <= 10 and 3 <= o2 - (o1 + d1) <= 10
                assert all(o <= 44 for o, _, _ in p)
            elif cell.family == "RS":
                change = p[0][1]
                assert change is not None
                assert 12 <= change <= 32 and [k for _, k in tr] == ["change"]
            elif cell.family == "RP":
                assert 0 <= p[0][0] <= 20 and p[0][1] is None
            else:
                assert p == ()
    # schedules vary across episode seeds (randomised), e.g. RT onsets
    onsets = {draw_pulses(cells[0], 0, s)[0][0] for s in range(3_681_000, 3_681_100)}
    assert len(onsets) > 15


def test_oracle_events_sit_at_true_transitions():
    s = DelaySchedule(((10, 5, 3),))
    assert oracle_events(s) == {10: 10, 15: 15}
    assert s.delay_at(9) == 0 and s.delay_at(10) == 3 and s.delay_at(15) == 0


def test_rotation_keeps_count_and_is_seeded():
    m = np.zeros(50, bool)
    m[20:26] = True
    r = rotate(m, 1, 2, 3)
    assert r.sum() == 6 and not np.array_equal(r, m)
    assert np.array_equal(r, rotate(m, 1, 2, 3))


# --- EventReplay on the synthetic plant (open loop, so trajectories are identical) ------------


def drive(schedule, ctrl, steps=50, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    est = ActuatorEstimator(EST)
    g, cmds, out = np.zeros(3), [], []
    for t in range(steps):
        est.observe(g + noise * rng.standard_normal(3))
        dec = ctrl.step(est)
        out.append((dec.active, dec.delay, dec.alarm, dec.suspended,
                    getattr(ctrl, "reset_start", -1)))  # fmt: skip
        c = rng.uniform(-1, 1, 4)
        est.issued(c)
        cmds.append(c)
        d = schedule(t)
        g = g + 0.62 * 0.05 * (cmds[t - d] if t - d >= 0 else np.zeros(4))[:3]
    return out


SCHEDULES = {
    "end": lambda t: 3 if t < 20 else 0,
    "onset": lambda t: 0 if t < 20 else 3,
    "change": lambda t: 1 if t < 20 else 4,
    "pulse": lambda t: 2 if 12 <= t < 30 else 0,
}


@pytest.mark.parametrize("name", list(SCHEDULES))
def test_replaying_c_events_reproduces_c(name):
    sched = SCHEDULES[name]
    c = drive(sched, Sequential(4.0, 0.001), noise=1e-4)
    events = {t: r[4] for t, r in enumerate(c) if r[2]}
    assert events, "the scenario must make C alarm"
    rep = drive(sched, EventReplay(4.0, 0.001, events=events, pause=True, reset=True), noise=1e-4)
    assert rep == c


def test_empty_replay_is_the_latch():
    for sched in SCHEDULES.values():
        lat = [(a, d) for a, d, *_ in drive(sched, Latched())]
        rep = [(a, d) for a, d, *_ in drive(sched, EventReplay(*C_ARGS))]
        assert rep == lat


def test_pause_only_and_reset_only_switches():
    sched = SCHEDULES["change"]
    c = drive(sched, Sequential(4.0, 0.001), noise=1e-4)
    events = {t: r[4] for t, r in enumerate(c) if r[2]}
    reset_only = drive(sched, EventReplay(4.0, 0.001, events=events, pause=False), noise=1e-4)
    assert not any(r[3] for r in reset_only)  # never paused
    assert [r[4] for r in reset_only][-1] == c[-1][4]  # same reset
    pause_only = drive(sched, EventReplay(4.0, 0.001, events=events, reset=False), noise=1e-4)
    assert all(r[4] == 0 for r in pause_only)  # never reset


# --- analysis ---------------------------------------------------------------------------------


def _episodes(effects: dict[str, float], fams=("RT", "RI", "RS", "RP"), n=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for fam in fams:
        for s in range(n):
            u = rng.random()
            for name in CONFIGURATIONS:
                rows.append({"configuration": name, "policy_seed": 0, "condition": fam + "_x",
                             "family": fam, "episode_seed": s,
                             "success": int(u < 0.5 + effects.get(name, 0.0))})  # fmt: skip
    return pd.DataFrame(rows)


def test_interaction_contrast_and_labels():
    ep = _episodes({"C_full": 0.3, "C_pause": 0.1, "C_reset": 0.1})
    r = contrast(ep, INTERACTION, 0.0, 500, 1, two_sided=True)
    w = ep.pivot_table(index=["policy_seed", "condition", "episode_seed"],
                       columns="configuration", values="success")  # fmt: skip
    manual = (w["C_full"] - w["C_reset"] - w["C_pause"] + w["contact_learned"]).mean()
    assert r["estimate"] == pytest.approx(manual)
    assert label({"estimate": 0.02, "ci_low": 0.01, "ci_high": 0.03, "p_holm": 0.001},
                 0.01, 0.025) == "meaningful"  # fmt: skip
    assert label({"estimate": 0.005, "ci_low": 0.001, "ci_high": 0.009, "p_holm": 0.001},
                 0.01, 0.025) == "significant but small"  # fmt: skip
    assert label({"estimate": 0.0, "ci_low": -0.005, "ci_high": 0.005, "p_holm": 0.5},
                 0.01, 0.025) == "negligible"  # fmt: skip
    assert label({"estimate": 0.0, "ci_low": -0.03, "ci_high": 0.03, "p_holm": 0.5},
                 0.01, 0.025) == "inconclusive"  # fmt: skip


def test_gatekeeper_failure_makes_mechanism_tests_descriptive():
    ep = _episodes({"C_pause": 0.3})
    gate, prim, _ = hypotheses(ep, 0.01, 0.025, 500, 1, 0.02)
    assert not gate["passed"].iloc[0]
    assert set(prim["label"]) == {"descriptive (gatekeeper failed)"}
    ep = _episodes({"C_full": 0.3, "C_pause": 0.3})
    gate, prim, _ = hypotheses(ep, 0.01, 0.025, 500, 1, 0.02)
    assert gate["passed"].iloc[0]
    assert prim.set_index("test").loc["M1", "label"] == "meaningful"
    assert (prim["p_holm"] >= prim["p"]).all()


def _steps(alarms: dict[str, list[int]], shadows: dict[str, list[int]]):
    rows = []
    for name in CONFIGURATIONS:
        for t in range(50):
            rows.append({"configuration": name, "policy_seed": 0, "condition": "RT_d2",
                         "family": "RT", "episode_seed": 1, "t": t,
                         "delay_true": 2 if 10 <= t < 20 else 0, "delay_used": 0,
                         "active": False, "suspended": False, "ctrl_ns": 1, "learned_calls": 0,
                         "alarm": t in alarms.get(name, []),
                         "shadow_alarm": t in shadows.get(name, [])})  # fmt: skip
    return pd.DataFrame(rows)


def test_divergence_and_integrity_stop():
    same = {n: [11, 21] for n in CONFIGURATIONS}
    steps = _steps({"C_full": [11, 21]}, same)
    div, _ = divergence(steps)
    assert (div["first_alarm_identical"] == 1.0).all()
    assert (div["full_agreement_share"] == 1.0).all()
    shifted = {**same, "C_reset": [12, 21]}
    div, _ = divergence(_steps({"C_full": [11, 21]}, shifted))
    assert div.set_index("configuration").loc["C_reset", "first_alarm_identical"] == 0.0
    config = load_g3(CONFIG)
    episodes = _episodes({}, fams=("RT",), n=2)
    with pytest.raises(IntegrityError):
        derive_g3(episodes, _steps({"C_full": [11, 21]}, shifted), config)


# --- simulator --------------------------------------------------------------------------------

from farc.paths import checkpoint_path, resolve  # noqa: E402

CHECKPOINT = checkpoint_path("fetch_push_fp32_seed1")
MODEL = resolve("$FARC_DATA_DIR/models/contact_predictor/model.pt")
needs_sim = pytest.mark.skipif(not (CHECKPOINT.exists() and MODEL.exists()),
                               reason="checkpoint or contact model absent")  # fmt: skip


@pytest.fixture(scope="module")
def sim():
    from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
    from farc.contact.dynamics import load_dynamics
    from farc.contact.episodes import learned_predictor
    from farc.disturbances.rollout import FrozenNormalizer
    from farc.envs.factory import make_env

    loaded = load_policy(CHECKPOINT)
    preds = {"learned": learned_predictor(load_dynamics(MODEL), move_object=True)}
    return (make_env(loaded.env_config), deterministic_policy_function(loaded.actor),
            FrozenNormalizer(loaded.normalizer), preds)  # fmt: skip


def _run(sim, schedule, ctrl, seed=3_000_903, **kw):
    from farc.delaychange.schedule import ScheduledDelaySession
    from farc.disturbances.config import ConditionSpec

    env, policy, normalizer, preds = sim
    session = ScheduledDelaySession(env, policy, normalizer, ConditionSpec("x", ()), seed, 9,
                                    schedule)  # fmt: skip
    result = run_episode(session, ActuatorEstimator(EST), ctrl, preds, **kw)
    return np.array(session.commands), result["rows"]


@needs_sim
def test_simulator_replay_equivalence_and_first_alarm_identity(sim):
    cell = g3_cells()[2]  # RI_d2: several transitions
    found = False
    for seed in range(3_000_900, 3_000_906):
        sched = schedule_for(cell, 2, seed)
        c_cmd, c_rows = _run(sim, sched, Sequential(*C_ARGS), seed)
        events = {r["t"]: r["reset_start"] for r in c_rows if r["alarm"]}
        r_cmd, _ = _run(sim, sched, EventReplay(*C_ARGS, events=events), seed)
        np.testing.assert_array_equal(r_cmd, c_cmd)
        e_cmd, _ = _run(sim, sched, EventReplay(*C_ARGS), seed)
        l_cmd, _ = _run(sim, sched, Latched(), seed)
        np.testing.assert_array_equal(e_cmd, l_cmd)
        _, rr = _run(sim, sched, EventReplay(*C_ARGS, events=events, pause=False), seed,
                     shadow=Sequential(*C_ARGS))  # fmt: skip
        c_first = next((r["t"] for r in c_rows if r["alarm"]), None)
        s_first = next((r["t"] for r in rr if r["shadow_alarm"]), None)
        assert c_first == s_first
        found |= bool(events)
    assert found, "at least one episode must exercise C's alarms"
