"""G2 Stage 1: delay schedules, transition semantics and diagnostic configurations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.delaychange.episodes import DIAGNOSTICS, branch, run_diagnostic_episode
from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
from farc.delaychange.stage1 import decision_rules, load_stage1, stage1_cells
from farc.disturbances.config import ConditionSpec, DisturbanceSpec
from farc.paths import checkpoint_path, resolve

EST = EstimatorConfig(nominal_gain=0.62)
CONFIG = "configs/delaychange/g2_stage1.json"


def test_schedule_basics():
    s = DelaySchedule(((6, 4, 3), (20, None, 2)))
    assert [s.delay_at(t) for t in (5, 6, 9, 10, 19, 20, 49)] == [0, 3, 3, 0, 0, 2, 2]
    assert s.ends(50) == [10]
    with pytest.raises(ValueError, match="overlap"):
        DelaySchedule(((0, 10, 3), (5, 3, 1)))
    with pytest.raises(ValueError):
        DelaySchedule(((0, 0, 3),))


def test_command_horizon_and_placeholders_under_replay_drop():
    s = DelaySchedule(((10, 5, 3),))  # steps 10..14 execute c[t-3]; 15 executes c[15]
    assert s.command_horizon(5, 50) == 0  # before the delay: immediate
    assert s.command_horizon(7, 50) == 0  # executes at once (and is replayed at step 10)
    assert s.command_horizon(11, 50) == 3  # executes at 14
    assert s.command_horizon(12, 50) is None  # dropped when the delay ends at 15
    assert s.command_horizon(20, 50) == 0
    # stale look-ahead of 3 after the end: all three placeholders execute immediately
    assert s.placeholders(20, 3) == 3
    assert s.placeholders(11, 3) == 0  # inside the delay none of them execute in time


def test_stage1_matrix_is_the_preregistered_one():
    cells = {c.name: c for c in stage1_cells()}
    transient = [c for c in cells.values() if c.group == "transient"]
    assert len(transient) == 6 and {c.pulses[0][1] for c in transient} == {4, 8, 16}
    assert {c.pulses[0][0] for c in transient} == {6, 18}
    assert all(c.semantics == "drain_hold" for c in cells.values() if c.group.endswith("drain"))
    assert all(n in DIAGNOSTICS for c in cells.values() for n in c.configurations)
    n = sum(len(c.configurations) for c in cells.values())
    assert n == 130  # x 150 episodes = 19,500 (proposal: about 20,000)
    assert not DIAGNOSTICS["deployed_learned"].privileged
    assert not DIAGNOSTICS["never_release"].privileged
    for name in ("true_learned", "release_k0", "schedule_sim", "latched_sim", "forced_on_L3"):
        assert DIAGNOSTICS[name].privileged
    assert branch(20).privileged


def test_config_seeds_and_plan():
    config = load_stage1(CONFIG)
    assert config.seeds == list(range(3_671_000, 3_671_050))
    assert load_stage1(CONFIG, debug=True).seeds[0] == 3_670_000


def _synthetic(base, deployed, ks, drain_base, drain_dep, seeds=60):
    rows = []
    rng = np.random.default_rng(0)
    for seed in range(seeds):
        for cond in ("P3_a", "P3_b"):
            u = rng.random()
            values = {"base_only": u < base, "deployed_learned": u < deployed,
                      **{f"release_k{k}": u < v for k, v in ks.items()}}  # fmt: skip
            for name, ok in values.items():
                rows.append(
                    {
                        "configuration": name,
                        "policy_seed": 0,
                        "condition": cond,
                        "episode_seed": seed,
                        "group": "transient",
                        "success": ok,
                    }
                )
            for name, ok in (("base_only", u < drain_base), ("deployed_learned", u < drain_dep)):
                rows.append({"configuration": name, "policy_seed": 0,
                             "condition": cond + "_drain", "episode_seed": seed,
                             "group": "transient_drain", "success": ok})  # fmt: skip
    return pd.DataFrame(rows)


def test_decision_rules_detect_stale_cause_and_pipeline_dependence():
    ks = {0: 0.97, 1: 0.95, 2: 0.93, 3: 0.9, 5: 0.86, 7: 0.84}
    rules = decision_rules(_synthetic(0.98, 0.8, ks, 0.98, 0.97), 400, 1)
    assert rules["stale_compensation_is_main_cause"]
    assert rules["pipeline_dependence_flagged"]  # harm shrinks from -18 to -1 points
    flat = {k: 0.82 for k in ks}
    rules = decision_rules(_synthetic(0.98, 0.8, flat, 0.98, 0.8), 400, 1)
    assert not rules["stale_compensation_is_main_cause"]
    assert not rules["pipeline_dependence_flagged"]


# --- simulator --------------------------------------------------------------------------------

CHECKPOINT = checkpoint_path("fetch_push_fp32_seed2")
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
    model = load_dynamics(MODEL)
    preds = {"learned": learned_predictor(model), "phase_e": phase_e_predictor(0.62)}
    return (make_env(loaded.env_config), deterministic_policy_function(loaded.actor),
            FrozenNormalizer(loaded.normalizer), preds, make_env(loaded.env_config))  # fmt: skip


def _scheduled(sim, pulses, semantics="replay_drop", seed=3_000_901):
    env, policy, normalizer = sim[:3]
    return ScheduledDelaySession(env, policy, normalizer, ConditionSpec("x", ()), seed, 7,
                                 DelaySchedule(pulses), semantics)  # fmt: skip


@needs_sim
@pytest.mark.parametrize("pulses", [((6, 8, 3),), ((0, None, 3),), ((6, 10, 3), (26, 10, 3)),
                                    ((0, 20, 1), (20, None, 3))])  # fmt: skip
def test_replay_drop_reproduces_the_existing_pipeline(sim, pulses):
    from farc.counterfactual.session import EpisodeSession

    policy, normalizer, other_env = sim[1], sim[2], sim[4]
    specs = tuple(DisturbanceSpec(type="action_delay", severity=d, onset=o, duration=L)
                  for o, L, d in pulses)  # fmt: skip
    old = EpisodeSession(other_env, policy, normalizer, ConditionSpec("x", specs), 3_000_901, 7)
    new = _scheduled(sim, pulses)
    rng = np.random.default_rng(0)
    while not old.done:
        old.observe(), new.observe()
        c = rng.uniform(-1, 1, 4)
        np.testing.assert_array_equal(old.act(c), new.act(c))


@needs_sim
def test_drain_hold_holds_at_onset_and_merges_at_end(sim):
    s = _scheduled(sim, ((5, 4, 3),), "drain_hold")
    cmds = [np.full(4, 0.01 * (i + 1)) for i in range(15)]
    executed = []
    for c in cmds:
        s.observe()
        executed.append(s.act(c))
    first = np.array(executed)[:, 0]
    expected = [
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,  # no delay
        0.05,
        0.05,
        0.05,  # hold the last executed command while the queue fills
        0.06,  # step 8: c[5] (delay 3)
        0.07 + 0.08 + 0.09 + 0.10,  # step 9: delay ends, queued c[6..8] merged with c[9]
        0.11,
        0.12,
        0.13,
        0.14,
        0.15,
    ]
    np.testing.assert_allclose(first, expected)


@needs_sim
def test_snapshot_restores_the_drain_queue_exactly(sim):
    from farc.contact.episodes import oracle_prediction

    s = _scheduled(sim, ((5, 6, 3),), "drain_hold")
    rng = np.random.default_rng(1)
    for _ in range(7):
        s.observe()
        s.act(rng.uniform(-1, 1, 4))
    s.observe()
    queue = [q.copy() for q in s.queue]
    digest = s.state_digest()
    oracle_prediction(s, 3)
    assert s.state_digest() == digest
    assert len(s.queue) == len(queue) and all(
        np.array_equal(a, b) for a, b in zip(s.queue, queue, strict=True)
    )


@needs_sim
def test_deployed_learned_reproduces_the_f1_episode_loop(sim):
    from farc.contact.episodes import run_episode
    from farc.counterfactual.session import EpisodeSession

    env, policy, normalizer, preds = sim[4], sim[1], sim[2], sim[3]
    spec = ConditionSpec("c3", (DisturbanceSpec(type="action_delay", severity=3),))
    a = EpisodeSession(env, policy, normalizer, spec, 3_000_902, 7)
    run_episode(a, ActuatorEstimator(EST), preds["learned"], spec)
    b = _scheduled(sim, ((0, None, 3),), seed=3_000_902)
    run_diagnostic_episode(b, ActuatorEstimator(EST), DIAGNOSTICS["deployed_learned"], preds)
    np.testing.assert_array_equal(np.array(a.commands), np.array(b.commands))


@needs_sim
def test_privileged_overrides_do_what_they_say(sim):
    preds = sim[3]
    pulses = ((6, 8, 3),)  # delay ends at step 14
    r = run_diagnostic_episode(_scheduled(sim, pulses), ActuatorEstimator(EST),
                               DIAGNOSTICS["release_k0"], preds)["rows"]  # fmt: skip
    assert not any(row["active"] for row in r if row["t"] >= 14)
    r = run_diagnostic_episode(_scheduled(sim, pulses), ActuatorEstimator(EST),
                               DIAGNOSTICS["true_learned"], preds)["rows"]  # fmt: skip
    assert [row["delay_used"] for row in r] == [row["delay_true"] for row in r]
    r = run_diagnostic_episode(_scheduled(sim, ()), ActuatorEstimator(EST),
                               DIAGNOSTICS["forced_on_L3"], preds)["rows"]  # fmt: skip
    assert [row["t"] for row in r if row["active"]] == [18, 19, 20]
    r = run_diagnostic_episode(_scheduled(sim, ((6, None, 3),)), ActuatorEstimator(EST),
                               DIAGNOSTICS["forced_off_L3"], preds)["rows"]  # fmt: skip
    assert not any(row["active"] for row in r if 18 <= row["t"] < 21)
    assert all(row["active"] for row in r if 6 <= row["t"] < 18 or row["t"] >= 21)
