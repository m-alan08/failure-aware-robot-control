"""Residual RL infrastructure tests: training mixture, SAC runner, selection, analysis.

SOFTWARE ONLY. Synthetic plants (``StubSession``) and unregistered seeds 0-99,999; the
simulator tests use FetchPush with a scripted stand-in policy (no frozen policy) on seeds
0-2. No registered seed is used, nothing is trained beyond a few hundred stub steps, and no
number here is an experimental result.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_residual import SCHEDULES, StubSession, _core

import farc.supervision.seeds as seeds_mod
from farc.delaychange.schedule import DelaySchedule
from farc.disturbances.config import SEED_RANGES, ConditionSpec
from farc.residual import analysis as A
from farc.residual.controller import ResidualController, ResidualEnv
from farc.residual.disturbances import ObjectDynamics
from farc.residual.training import (
    SMOKE_SEEDS,
    ReplayBuffer,
    ResidualSAC,
    RunRefused,
    SeedsExhausted,
    TrainConfig,
    alpha_schedule,
    check_run,
    load_plan_config,
    smoke_config,
    train,
)
from farc.residual.training_env import (
    DELAY_CELL_INDEX_BASE,
    SHARES,
    AR1PositionNoise,
    TrainingSessionFactory,
    check_shares,
    draw_training,
    without,
)

PLAN = load_plan_config()

# --- seed isolation -----------------------------------------------------------------------------


def _registered_blocks():
    out = dict(SEED_RANGES)
    for name in dir(seeds_mod):
        value = getattr(seeds_mod, name)
        if isinstance(value, dict):
            for k, v in value.items():
                if (isinstance(v, tuple) and len(v) == 2
                        and all(isinstance(x, int) for x in v)):  # fmt: skip
                    out[f"{name}.{k}"] = v
    return out


def test_smoke_seeds_disjoint_from_every_registered_block():
    blocks = _registered_blocks()
    assert len(blocks) > 10 and "R_SEED_BLOCKS.training" in blocks
    lo, hi = SMOKE_SEEDS
    for name, (a, b) in blocks.items():
        assert hi <= a or b <= lo, f"smoke seeds overlap {name}"


def test_plan_config_still_unapproved_and_pinned():
    assert PLAN["approved"] is False
    digest = hashlib.sha256(Path(PLAN["plan"]).read_bytes()).hexdigest()
    assert digest == PLAN["plan_sha256"], "the draft pre-registration must not change"


def _registered_cfg(**kw):
    lo, _ = seeds_mod.R_SEED_BLOCKS["training"]
    slo, _ = seeds_mod.R_SEED_BLOCKS["selection"]
    base = TrainConfig(arm="R", alpha_target=0.1, network_seed=0, policy_index=0,
                       episode_seeds=tuple(range(lo, lo + 20_000 // 2)),
                       nominal_eval_seeds=tuple(range(slo, slo + 100)))  # fmt: skip
    return replace(base, **kw)


def test_registered_run_refused_while_unapproved():
    with pytest.raises(RunRefused, match="not approved"):
        check_run(_registered_cfg(), PLAN)


def _approved_copy(tmp_path, plan_text="draft plan"):
    (tmp_path / "plan.md").write_text(plan_text)
    cfg = dict(PLAN, approved=True, plan="plan.md",
               plan_sha256=hashlib.sha256(plan_text.encode()).hexdigest())  # fmt: skip
    return cfg


def test_registered_validation_logic_on_a_hypothetical_approved_copy(tmp_path):
    """Exercises the checks a real approval would unlock, on a temporary copy only."""
    plan = _approved_copy(tmp_path)
    check_run(_registered_cfg(), plan, tmp_path)  # passes
    (tmp_path / "plan.md").write_text("edited")
    with pytest.raises(RunRefused, match="changed"):
        check_run(_registered_cfg(), plan, tmp_path)
    plan = _approved_copy(tmp_path)
    bad = [
        _registered_cfg(alpha_target=0.3),
        _registered_cfg(network_seed=5),
        _registered_cfg(network_seed=1, policy_index=0),
        _registered_cfg(env_steps=2_000_000),
        _registered_cfg(target_entropy=-4.0),
        _registered_cfg(episode_seeds=(3_710_005, 3_710_006)),  # not in order from the start
        _registered_cfg(episode_seeds=(3_721_000,)),  # selection seed used for training
        _registered_cfg(nominal_eval_seeds=(3_722_000,)),  # evaluation seed at selection
        _registered_cfg(nominal_eval_seeds=(4_000_600,)),  # held-out
    ]
    for cfg in bad:
        with pytest.raises((RunRefused, ValueError)):
            check_run(cfg, plan, tmp_path)


@pytest.mark.parametrize("seed", [100_000, 3_710_000, 3_720_000, 3_721_000, 3_722_000,
                                  4_000_000, 4_000_600, -1])  # fmt: skip
def test_smoke_runs_refuse_registered_and_heldout_seeds(seed):
    with pytest.raises(RunRefused):
        check_run(smoke_config(episode_seeds=(0, seed)), PLAN)
    with pytest.raises(RunRefused):
        check_run(smoke_config(nominal_eval_seeds=(seed,)), PLAN)


@pytest.mark.parametrize("kw", [{"arm": "Z"}, {"mode": "x"}, {"alpha_target": 0.0},
                                {"alpha_target": 1.5}, {"alpha_start": 0.5},
                                {"policy_index": 3}, {"batch_size": 0},
                                {"zero_data_steps": 4}])  # fmt: skip
def test_config_validation(kw):
    with pytest.raises(RunRefused):
        check_run(smoke_config(**kw), PLAN)


def test_env_still_refuses_heldout_seed():
    env = ResidualEnv(lambda s: StubSession(int(s), SCHEDULES[0]), ResidualController(_core()))
    with pytest.raises(ValueError):
        env.reset(seed=4_000_600)


# --- training mixture ---------------------------------------------------------------------------


def test_draws_are_pure_functions_of_the_seeds():
    for s in range(50):
        assert draw_training(s).to_dict() == draw_training(s).to_dict()
    other = [draw_training(s, generator_seed=1).to_dict() for s in range(50)]
    assert other != [draw_training(s).to_dict() for s in range(50)]


def test_family_shares_and_parameter_ranges():
    draws = [draw_training(s) for s in range(20_000)]
    freq = Counter(d.family for d in draws)
    for f, share in SHARES.items():
        assert abs(freq[f] / len(draws) - share) < 0.015
    for d in draws:
        if d.family == "T-N":
            assert d.is_nominal
        elif d.family == "T-M":
            assert 0.5 <= d.dynamics.mass_scale <= 2.0
            assert 0.5 <= d.dynamics.friction_scale <= 1.5
            assert d.obs_sigma == 0 and not d.extra and not d.schedule.pulses
        elif d.family == "T-O":
            assert 0 <= d.obs_sigma <= 0.01 and d.dynamics.is_nominal
        elif d.family == "T-A":
            assert 0 <= d.action_noise <= 0.2 and 0 <= d.action_scale <= 0.4
            assert {s.type for s in d.extra} <= {"action_noise", "action_scale"}
        else:
            assert d.delay_family in ("RT", "RI", "RS", "RP") and 1 <= d.delay <= 4
            assert d.schedule.pulses and max(p[2] for p in d.schedule.pulses) <= 4
    assert DELAY_CELL_INDEX_BASE > 8  # disjoint from G3/G4 cell indices 0-8


def test_share_validation_and_arm_variants():
    with pytest.raises(ValueError):
        check_shares({"T-N": 0.5})
    with pytest.raises(ValueError):
        check_shares({"T-X": 1.0})
    a2 = without(SHARES, "T-D")
    assert "T-D" not in a2 and abs(sum(a2.values()) - 1) < 1e-12
    assert a2["T-N"] == pytest.approx(0.25 / 0.75)
    assert {d.family for d in (draw_training(s, a2) for s in range(2000))} == set(a2)


# --- simulator wiring (FetchPush, scripted stand-in policy, seeds 0-2) --------------------------

gym = pytest.importorskip("gymnasium")
pytest.importorskip("gymnasium_robotics")


def _scripted(normalized):
    """Stand-in for a frozen policy: push the block toward the goal (not a trained model)."""
    x = np.asarray(normalized, np.float64)[0]
    grip, block, goal = x[0:3], x[3:6], x[-3:]
    push = goal - block
    target = block - 0.05 * push / (np.linalg.norm(push) + 1e-9)
    a = np.zeros(4)
    a[:3] = np.clip(10 * (target - grip), -1, 1)
    return a[None]


def _identity_normalizer(obs, goal):
    return np.concatenate([np.asarray(obs, np.float64), np.asarray(goal, np.float64)], -1)


@pytest.fixture(scope="module")
def env():
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    e = gym.make("FetchPush-v4")
    yield e
    e.close()


def _builder(env):
    from farc.delaychange.schedule import ScheduledDelaySession

    def build(seed, condition, schedule):
        return ScheduledDelaySession(env, _scripted, _identity_normalizer, condition, seed,
                                     20261900, schedule, "replay_drop")  # fmt: skip

    return build


def _run(session, commands=None):
    obs, cmds = [], []
    while not session.done:
        v = session.observe()
        obs.append(v.observation.copy())
        c = v.base_action if commands is None else commands[session.step]
        cmds.append(np.asarray(c).copy())
        session.act(c)
    return np.array(obs), np.array(cmds), session.final_distance


def _model_values(env):
    m = env.unwrapped.model
    return m.body_mass.copy(), m.body_inertia.copy(), m.geom_friction.copy()


def _same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def test_nominal_family_is_bitwise_the_plain_session(env):
    build = _builder(env)
    f = TrainingSessionFactory(build, {"T-N": 1.0})
    for seed in range(3):
        ref = _run(build(seed, ConditionSpec("nominal", ()), DelaySchedule(())))
        got = _run(f(seed))
        assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])
        assert ref[2] == got[2]


def test_zero_strength_tm_and_to_are_bitwise_identity(env):
    from farc.envs.observation_layout import control_period
    from farc.residual.disturbances import ObjectDynamicsPatch
    from farc.residual.training_env import DisturbedSession, TrainingDraw

    build = _builder(env)
    ref = _run(build(1, ConditionSpec("n", ()), DelaySchedule(())))
    s = build(1, ConditionSpec("n", ()), DelaySchedule(()))
    noise = AR1PositionNoise(0.0, np.random.default_rng(0), s.horizon)
    noise.configure(s.layout, control_period(s.raw))
    s.perturbations.append(noise)
    patch = ObjectDynamicsPatch(s.raw.model, ObjectDynamics(1.0, 1.0))
    got = _run(DisturbedSession(s, patch, TrainingDraw("T-M", 1)))
    assert np.array_equal(ref[0], got[0]) and np.array_equal(ref[1], got[1])


def test_tm_patch_restored_at_episode_end_and_when_abandoned(env):
    before = _model_values(env)
    f = TrainingSessionFactory(_builder(env), {"T-M": 1.0})
    s = f(0)
    assert s.patched and not _same(before, _model_values(env))
    _run(s)
    assert not s.patched and _same(before, _model_values(env))
    s = f(1)  # abandon mid-episode
    s.act(s.observe().base_action)
    assert s.patched
    f(2).close()  # building the next episode restores the abandoned patch first
    assert _same(before, _model_values(env))


def test_tm_changes_dynamics_under_a_fixed_command_sequence(env):
    build = _builder(env)
    f = TrainingSessionFactory(build, {"T-M": 1.0})
    seed = next(s for s in range(100) if draw_training(s, {"T-M": 1.0}).dynamics.mass_scale > 1.5)
    ref = _run(build(seed, ConditionSpec("n", ()), DelaySchedule(())))
    got = _run(f(seed), commands=ref[1])
    block = slice(3, 6)
    assert np.abs(ref[0][-1, block] - ref[0][0, block]).max() > 1e-3, "the push must move the block"
    assert not np.array_equal(ref[0], got[0])


def test_object_geom_friction_below_one_is_a_noop_documented_finding(env):
    """§3 read literally scales only the object geom's friction. MuJoCo uses the maximum of
    the two geoms' friction, and table/fingers have 1.0, so scales < 1 do nothing at all.
    The proposed ``object_table`` mode does change the dynamics. Pending a decision."""
    from farc.residual.disturbances import ObjectDynamicsPatch
    from farc.residual.training_env import DisturbedSession, TrainingDraw

    build = _builder(env)
    ref = _run(build(0, ConditionSpec("n", ()), DelaySchedule(())))
    before = _model_values(env)
    out = {}
    for mode in ("object_geom", "object_table"):
        s = build(0, ConditionSpec("n", ()), DelaySchedule(()))
        patch = ObjectDynamicsPatch(s.raw.model, ObjectDynamics(1.0, 0.5), friction_mode=mode)
        out[mode] = _run(DisturbedSession(s, patch, TrainingDraw("T-M", 0)), commands=ref[1])
        assert _same(before, _model_values(env))
    assert np.array_equal(out["object_geom"][0], ref[0])  # the no-op
    assert not np.array_equal(out["object_table"][0], ref[0])


def test_to_noise_reaches_only_the_controller_view(env):
    build = _builder(env)
    seed = next(s for s in range(200) if draw_training(s, {"T-O": 1.0}).obs_sigma > 0.005)
    ref = _run(build(seed, ConditionSpec("n", ()), DelaySchedule(())))
    f = TrainingSessionFactory(build, {"T-O": 1.0})
    s = f(seed)
    got = _run(s, commands=ref[1])  # same executed commands, open loop
    assert got[2] == ref[2]  # ground-truth distance untouched
    lay = s.layout.slices
    diff = got[0] - ref[0]
    assert np.abs(diff[:, lay["gripper_position"]]).max() > 0
    assert not diff[:, lay["finger_position"]].any()
    assert not diff[:, lay["gripper_velocity"]].any()


def test_ta_and_td_use_the_existing_disturbances(env):
    build = _builder(env)
    for fam in ("T-A", "T-D"):
        seed = 0
        d = draw_training(seed, {fam: 1.0})
        s = TrainingSessionFactory(build, {fam: 1.0})(seed)
        if fam == "T-A":
            assert {p.spec.type for p in s.perturbations} == {x.type for x in d.extra}
        else:
            assert s.schedule == d.schedule and s.semantics == "replay_drop"
        _run(s)


# --- SAC runner (synthetic StubSession plant) ---------------------------------------------------


def _stub_env():
    return ResidualEnv(lambda s: StubSession(int(s), SCHEDULES[2]),
                       ResidualController(_core(), alpha=0.05))  # fmt: skip


def _flat(agent):
    return torch.cat([p.detach().flatten() for p in agent.actor.parameters()])


def test_untrained_agent_is_exactly_candidate_c():
    agent = ResidualSAC(smoke_config())
    x = np.random.default_rng(0).standard_normal(65)
    assert np.array_equal(agent.act(x, True), np.zeros(4))


def test_alpha_ramp():
    cfg = smoke_config(alpha_target=0.2, alpha_ramp_steps=100)
    assert alpha_schedule(0, cfg) == 0.05
    assert alpha_schedule(50, cfg) == pytest.approx(0.125)
    assert alpha_schedule(100, cfg) == alpha_schedule(10_000, cfg) == 0.2


def test_smoke_run_is_deterministic_and_logs_everything(tmp_path):
    results, weights, logs = [], [], []
    for k in range(2):
        cfg = smoke_config()
        r = train(cfg, _stub_env(), lambda agent, alpha: 1.0, 1.0, tmp_path / str(k), PLAN)
        state = torch.load(r.final_checkpoint, weights_only=False)
        agent = ResidualSAC(cfg)
        agent.load_state_dict(state["agent"])
        results.append(r)
        weights.append(_flat(agent))
        logs.append([json.loads(x) for x in Path(r.log).read_text().splitlines()])
    assert results[0].status == "completed" and results[0].steps == 400
    assert torch.equal(weights[0], weights[1])
    assert logs[0] == logs[1]
    kinds = Counter(x["kind"] for x in logs[0])
    assert kinds["config"] == 1 and kinds["critic_warmup_done"] == 1 and kinds["end"] == 1
    assert kinds["nominal_eval"] == 4 and kinds["episode"] == 8
    assert (tmp_path / "0" / "checkpoint_0.pt").exists()
    assert not torch.equal(weights[0], _flat(ResidualSAC(smoke_config())))  # it did update


def test_zero_data_phase_and_critic_only_warmup(tmp_path):
    cfg = smoke_config(env_steps=60, zero_data_steps=60, eval_interval=1000)
    env = _stub_env()
    seen = []
    orig = env.step

    def spy(a):
        seen.append(np.asarray(a).copy())
        return orig(a)

    env.step = spy
    r = train(cfg, env, lambda agent, alpha: 1.0, 1.0, tmp_path, PLAN)
    assert all(not a.any() for a in seen) and len(seen) == 60
    agent = ResidualSAC(cfg)
    agent.load_state_dict(torch.load(r.final_checkpoint, weights_only=False)["agent"])
    assert torch.equal(_flat(agent), _flat(ResidualSAC(cfg)))  # actor never updated


def test_rollback_rule_halves_lr_then_fails_run(tmp_path):
    cfg = smoke_config()
    r = train(cfg, _stub_env(), lambda agent, alpha: 0.0, 0.9, tmp_path, PLAN)
    assert r.status == "failed_rollback" and r.rollbacks == 2 and r.steps == 200
    events = [json.loads(x) for x in Path(r.log).read_text().splitlines() if '"rollback"' in x]
    assert [e["actor_lr"] for e in events] == [cfg.actor_lr / 2, cfg.actor_lr / 4]
    final = torch.load(r.final_checkpoint, weights_only=False)
    assert final["status"] == "failed_rollback"


def test_rollback_threshold_is_two_points(tmp_path):
    ok = train(smoke_config(env_steps=100), _stub_env(), lambda a, al: 0.88, 0.90,
               tmp_path / "a", PLAN)  # fmt: skip
    assert ok.rollbacks == 0
    bad = train(smoke_config(env_steps=100), _stub_env(), lambda a, al: 0.879, 0.90,
                tmp_path / "b", PLAN)  # fmt: skip
    assert bad.rollbacks == 1


def test_seed_exhaustion_is_an_error_not_a_silent_wrap(tmp_path):
    with pytest.raises(SeedsExhausted):
        train(smoke_config(episode_seeds=(0, 1)), _stub_env(), lambda a, al: 1.0, 1.0,
              tmp_path, PLAN)  # fmt: skip


def test_replay_buffer_wraps():
    b = ReplayBuffer(3, 2)
    for i in range(5):
        b.add(np.full(2, i), np.zeros(4), float(i), np.zeros(2), False)
    assert b.size == 3 and sorted(b.rew.tolist()) == [2.0, 3.0, 4.0]


# --- selection and analysis (synthetic tables) --------------------------------------------------


def _table(effects: dict[tuple[str, str], float], seeds=200, policies=3, network_seeds=3,
           base=0.5, rng_seed=0, alpha=np.nan):  # fmt: skip
    """effects[(config, family)] = success-probability shift vs C. Families map to groups."""
    group = {"nominal": "N", "RT": "D", "RP": "D", "RI": "D", "T-M": "M", "T-O": "M",
             "T-A": "M", "X-DH": "X", "X-G": "X"}  # fmt: skip
    rng = np.random.default_rng(rng_seed)
    rows = []
    configs = sorted({c for c, _ in effects} | {"C"})
    for fam, g in group.items():
        for p in range(policies):
            for s in range(seeds):
                u = np.full(network_seeds, rng.random())  # paired: common random numbers
                for cfg in configs:
                    prob = base + effects.get((cfg, fam), 0.0)
                    nss = [-1] if cfg == "C" else range(network_seeds)
                    for i, ns in enumerate(nss):
                        rows.append({"config": cfg, "alpha": alpha if cfg != "C" else np.nan,
                                     "network_seed": ns, "policy": p, "episode_seed": s,
                                     "cell": fam, "group": g, "family": fam,
                                     "success": bool(u[i] < prob)})  # fmt: skip
    return pd.DataFrame(rows)


def _all(config, shift):
    fams = ("nominal", "RT", "RP", "RI", "T-M", "T-O", "T-A", "X-DH", "X-G")
    return {(config, f): shift for f in fams}


def test_gatekeeping_claim_wording():
    strong = {**_all("R", 0.0), **{("R", f): 0.1 for f in ("T-M", "T-O", "T-A", "X-DH", "X-G")}}
    res = A.development_analysis(_table(strong), resamples=2000)
    assert res["steps_passed"] == 4 and res["claim"] == A.CLAIMS[4]
    no_x = {**strong, ("R", "X-DH"): 0.0, ("R", "X-G"): 0.0}
    res = A.development_analysis(_table(no_x), resamples=2000)
    assert res["steps_passed"] == 3 and res["claim"] == A.CLAIMS[3]
    small = {**_all("R", 0.0), **{("R", f): 0.01 for f in ("T-M", "T-O", "T-A")}}
    res = A.development_analysis(_table(small, seeds=300), resamples=2000)
    assert res["steps_passed"] == 2 and res["claim"] == A.CLAIMS[2]
    harm = {**strong, ("R", "nominal"): -0.2}
    res = A.development_analysis(_table(harm), resamples=2000)
    assert res["claim_draft_wording"] == A.CLAIMS[0]
    assert [s["tested"] for s in res["primary"]] == [True, False, False, False]


def test_secondaries_and_safety_counts():
    eff = {**_all("R", 0.0), **{("R", f): 0.1 for f in ("T-M", "T-O", "T-A")}, **_all("A1", 0.0)}
    res = A.development_analysis(_table(eff), resamples=1000)
    s1 = next(s for s in res["secondary"] if s["name"].startswith("S1"))
    assert s1["estimate"] > 0.05 and "p_holm" in s1
    assert all(s.get("missing") for s in res["secondary"] if s["name"][:2] in ("S2", "S3", "S4"))
    assert {"family", "harmed", "rescued"} <= set(res["safety"].columns)


def test_holm():
    assert A.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_unbalanced_design_is_rejected():
    df = _table(_all("R", 0.0), seeds=10)
    with pytest.raises(A.AnalysisInputError):
        A.development_analysis(df.drop(index=df.index[-1]), resamples=100)


def _selection(shifts: dict[float, dict[str, float]]):
    parts: list[pd.DataFrame] = []
    for a, fam_shift in shifts.items():
        t = _table({("R", f): v for f, v in fam_shift.items()} | {("R", "nominal"): 0.0}
                   if "nominal" not in fam_shift else {("R", f): v for f, v in fam_shift.items()},
                   seeds=100, alpha=a, rng_seed=1)  # fmt: skip
        parts.append(t if not parts else t[t.config == "R"])
    return pd.concat(parts, ignore_index=True)


def test_selection_rule():
    df = _selection({0.05: {"T-M": 0.05}, 0.1: {"T-M": 0.2}, 0.2: {"T-M": 0.2, "nominal": -0.3}})
    sel = A.select_alpha(df, [0.05, 0.1, 0.2])
    passes = dict(zip(sel.table.alpha, sel.table.passes_nominal, strict=True))
    assert sel.alpha == 0.1 and not passes[0.2]
    none = _selection({a: {"nominal": -0.3} for a in (0.05, 0.1, 0.2)})
    assert A.select_alpha(none, [0.05, 0.1, 0.2]).alpha is None


def test_selection_ties_go_to_smaller_alpha():
    df = _selection({0.05: {"T-M": 0.2}, 0.1: {"T-M": 0.2}, 0.2: {"T-M": 0.2}})
    sel = A.select_alpha(df, [0.05, 0.1, 0.2], tie=0.05)
    assert sel.alpha == 0.05


def test_failed_runs_enter_as_c():
    df = _table({**_all("R", 0.3)}, seeds=20)
    sub = A.substitute_failed(df, [("R", np.nan, 0), ("R", np.nan, 1), ("R", np.nan, 2)])
    c = sub[sub.config == "C"].set_index(A.KEYS).success
    for ns in range(3):
        r = sub[(sub.config == "R") & (sub.network_seed == ns)].set_index(A.KEYS).success
        assert (r == c.reindex(r.index)).all()
