"""Protocol-correction tests: SAC algorithm checks, boundary fixes, B2, episode generation,
seed-block proposals and the degradation/non-inferiority distinction.

SOFTWARE ONLY: synthetic plants, unregistered seeds 0-99,999, FetchPush with a scripted
stand-in policy. No registered seed, no training beyond a few hundred stub steps, no result.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from test_residual import SCHEDULES, StubSession, _core, _normalizer, _policy
from test_residual_training import PLAN, _builder, _run, _stub_env, _table, env  # noqa: F401

from farc.residual import analysis as A
from farc.residual import proposals as P
from farc.residual.base_core import B2_SPEC, SACBaseCore, check_b2_spec
from farc.residual.controller import ResidualController, ResidualEnv, run_residual_episode
from farc.residual.episodes import (
    check_role_seeds,
    development_cells,
    generate,
    selection_cells,
)
from farc.residual.observation import ObservationSpec, ResidualObservationBuilder
from farc.residual.policy import ResidualActor, ZeroResidual
from farc.residual.training import (
    ReplayBuffer,
    ResidualSAC,
    RunRefused,
    SeedsExhausted,
    smoke_config,
    train,
)

# --- final-step boundary and truncation -----------------------------------------------------


def test_budget_that_fits_the_seed_list_exactly_does_not_raise(tmp_path):
    """400 steps = 8 episodes of 50: 8 seeds must suffice (no reset after the final step)."""
    cfg = smoke_config(episode_seeds=tuple(range(8)))
    r = train(cfg, _stub_env(), lambda a, al: 1.0, 1.0, tmp_path, PLAN)
    end = json.loads(Path(r.log).read_text().splitlines()[-1])
    assert r.status == "completed" and r.episodes == 8 and end["seeds_used"] == 8
    with pytest.raises(SeedsExhausted):
        train(smoke_config(episode_seeds=tuple(range(7))), _stub_env(), lambda a, al: 1.0, 1.0,
              tmp_path / "b", PLAN)  # fmt: skip


def test_time_limit_transitions_are_stored_as_terminal(tmp_path, monkeypatch):
    stored = []
    orig = ReplayBuffer.add

    def spy(self, o, a, r, o2, terminal):
        stored.append((bool(terminal), bool(np.any(o2))))
        return orig(self, o, a, r, o2, terminal)

    monkeypatch.setattr(ReplayBuffer, "add", spy)
    train(smoke_config(env_steps=100, eval_interval=1000), _stub_env(), lambda a, al: 1.0, 1.0,
          tmp_path, PLAN)  # fmt: skip
    terminal = [i for i, (t, _) in enumerate(stored) if t]
    assert terminal == [49, 99]
    # the placeholder next-observation (all zeros) only ever appears on terminal rows
    assert all(t for t, nonzero in stored if not nonzero)


def test_terminal_rows_do_not_bootstrap():
    agent = ResidualSAC(smoke_config())
    n = 8
    b = {"obs": torch.zeros(n, 65), "act": torch.zeros(n, 4), "rew": torch.full((n,), -1.0),
         "next_obs": torch.randn(n, 65), "done": torch.ones(n)}  # fmt: skip
    assert torch.equal(agent.td_target(b, torch.tensor(0.5)), b["rew"])
    b["done"] = torch.zeros(n)
    t = agent.td_target(b, torch.tensor(0.0))
    with torch.no_grad():
        # zero-initialised critic heads are not zero, so the bootstrap term must matter
        assert not torch.equal(t, b["rew"])


# --- SAC algorithm checks ---------------------------------------------------------------------


def test_tanh_gaussian_log_prob_matches_closed_form():
    agent = ResidualSAC(smoke_config())
    torch.manual_seed(3)
    with torch.no_grad():
        for layer in (agent.actor.mean, agent.actor.log_std):
            layer.weight.normal_(0, 0.3)
            layer.bias.normal_(0, 0.3)
    x = torch.randn(256, 65)
    g = torch.Generator().manual_seed(1)
    a, logp = agent._sample(x, g)
    g = torch.Generator().manual_seed(1)
    h = agent.actor.trunk(x)
    mu, log_std = agent.actor.mean(h), agent.actor.log_std(h).clamp(-5, 2)
    eps = torch.randn(mu.shape, generator=g)
    pre = mu + log_std.exp() * eps
    ref = torch.distributions.Normal(mu, log_std.exp()).log_prob(pre)
    ref = (ref - torch.log1p(-(torch.tanh(pre) ** 2))).sum(-1)  # naive form, fine if |pre| < 3
    ok = (pre.abs() < 3).all(-1)
    assert ok.sum() > 100
    assert torch.allclose(a, torch.tanh(pre))
    assert torch.allclose(logp[ok], ref[ok], atol=1e-4)
    assert torch.isfinite(logp).all()  # the implementation's stable form never overflows


def test_entropy_coefficient_moves_toward_target():
    """With entropy above the target (-8 for a 4-dim action), the coefficient must fall."""
    cfg = smoke_config()
    agent = ResidualSAC(cfg)
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(512, 65)
    for _ in range(512):
        buf.add(rng.standard_normal(65), rng.uniform(-1, 1, 4), -1.0, rng.standard_normal(65),
                False)  # fmt: skip
    start = agent.log_entropy_coef.item()
    out = {}
    for _ in range(20):
        out = agent.update(buf.sample(rng, 64), actor=True)
    assert out["entropy"] > cfg.target_entropy
    assert agent.log_entropy_coef.item() < start


def test_critic_only_updates_leave_actor_and_entropy_untouched():
    agent = ResidualSAC(smoke_config())
    rng = np.random.default_rng(0)
    buf = ReplayBuffer(64, 65)
    for _ in range(64):
        buf.add(rng.standard_normal(65), np.zeros(4), -1.0, rng.standard_normal(65), False)
    a0 = [p.clone() for p in agent.actor.parameters()]
    e0 = agent.log_entropy_coef.item()
    c0 = [p.clone() for p in agent.critic.parameters()]
    for _ in range(5):
        agent.update(buf.sample(rng, 16), actor=False)
    assert all(torch.equal(x, y) for x, y in zip(a0, agent.actor.parameters(), strict=True))
    assert agent.log_entropy_coef.item() == e0
    assert not all(torch.equal(x, y) for x, y in zip(c0, agent.critic.parameters(), strict=True))


def test_rollback_restores_networks_but_not_the_noise_stream(tmp_path):
    cfg = smoke_config()
    r = train(cfg, _stub_env(), lambda a, al: 0.0, 0.9, tmp_path, PLAN)
    ck0 = torch.load(tmp_path / "checkpoint_0.pt", weights_only=False)
    final = torch.load(r.final_checkpoint, weights_only=False)
    for k, v in ck0["agent"]["actor"].items():
        assert torch.equal(v, final["agent"]["actor"][k])  # rolled back to C exactly
    assert not torch.equal(ck0["agent"]["generator"], final["agent"]["generator"])
    assert {"replay_rng", "buffer_size", "seeds_used"} <= set(final)


def test_checkpoints_reproduce_bitwise(tmp_path):
    for k in range(2):
        train(smoke_config(), _stub_env(), lambda a, al: 1.0, 1.0, tmp_path / str(k), PLAN)
    for name in ("checkpoint_100.pt", "checkpoint_400.pt", "final.pt"):
        s0 = torch.load(tmp_path / "0" / name, weights_only=False)["agent"]
        s1 = torch.load(tmp_path / "1" / name, weights_only=False)["agent"]
        for part in ("actor", "critic", "critic_target"):
            assert all(torch.equal(s0[part][k], s1[part][k]) for k in s0[part])


# --- B2 / B0 ----------------------------------------------------------------------------------


def _b2_controller(policy=None, alpha=0.0):
    return ResidualController(cast(Any, SACBaseCore(_policy, _normalizer)), policy, alpha,
                              ResidualObservationBuilder(B2_SPEC))  # fmt: skip


@pytest.mark.parametrize("schedule", range(4))
def test_b2_with_zero_residual_is_frozen_sac_exactly(schedule):
    for seed in range(3):
        ref = StubSession(seed, SCHEDULES[schedule])
        while not ref.done:
            ref.act(np.clip(ref.observe().base_action, -1, 1))
        for pol, alpha in ((ZeroResidual(), 0.0), (ResidualActor(B2_SPEC.dim), 0.2)):
            s = StubSession(seed, SCHEDULES[schedule])
            out = run_residual_episode(s, _b2_controller(pol, alpha))
            assert np.array_equal(np.array(s.commands), np.array(ref.commands))
            assert not any(r["active"] or r["alarm"] for r in out["rows"])


def test_b2_observation_is_41_dims_and_has_no_c_signals():
    assert B2_SPEC.dim == 41
    with pytest.raises(ValueError):
        check_b2_spec(ObservationSpec())
    renv = ResidualEnv(lambda s: StubSession(int(s), SCHEDULES[1]), _b2_controller(alpha=0.1))
    obs, _ = renv.reset(seed=0)
    assert obs.shape == (41,)
    obs, *_ = renv.step(np.ones(4))
    assert obs.shape == (41,)


def test_b2_residual_changes_commands_when_nonzero():
    s = StubSession(0, SCHEDULES[0])
    run_residual_episode(s, _b2_controller(lambda x: np.ones(4), 0.1))
    ref = StubSession(0, SCHEDULES[0])
    while not ref.done:
        ref.act(np.clip(ref.observe().base_action, -1, 1))
    assert not np.array_equal(np.array(s.commands), np.array(ref.commands))


# --- episode generation -----------------------------------------------------------------------


def test_cell_grids_match_the_draft():
    sel = selection_cells()
    assert [c.name for c in sel] == ["nominal", "T-M", "T-O", "T-A", "T-D_RT", "T-D_RP"]
    dev = development_cells()
    groups = {g: [c for c in dev if c.group == g] for g in "NDMX"}
    assert len(groups["N"]) == 1 and len(groups["M"]) == 7 and len(groups["X"]) == 4
    assert len(groups["D"]) == 41 and len(development_cells(include_h_in_d=False)) == 51
    xdh = [c for c in groups["X"] if c.family == "X-DH"]
    assert all(c.make(5).semantics == "drain_hold" for c in xdh)
    assert all(c.make(5).semantics == "replay_drop" for c in groups["D"])
    xg = next(c for c in groups["X"] if c.family == "X-G").make(0).extra[0]
    assert (xg.type, xg.severity, xg.onset) == ("goal_shift", 0.05, 20)
    for c in sel[1:]:
        s1, s2 = c.make(11), c.make(11)
        assert s1 == s2  # pure function of the seed


@pytest.mark.parametrize("role,seeds", [("selection", [3_721_000]), ("evaluation", [3_722_000]),
                                        ("smoke", [3_722_000]), ("smoke", [4_000_600]),
                                        ("bogus", [0])])  # fmt: skip
def test_generation_refuses_registered_or_bad_seeds(role, seeds):
    with pytest.raises(RunRefused):
        check_role_seeds(role, seeds, PLAN)


def _stub_build(seed, condition, schedule, semantics, policy):
    s: Any = StubSession(seed, schedule)
    s.raw, s.layout, s.perturbations = None, None, []  # what disturb() may touch
    return s


def _stub_run(config, ns, policy, session):
    if config == "C":
        ctl = ResidualController(_core(), ZeroResidual(), 0.0)
    elif config == "B0":
        ctl = _b2_controller()
    else:
        ctl = ResidualController(_core(), lambda x, ns=ns: np.full(4, 0.1 * (ns + 1)), 0.1)
    return {"success": run_residual_episode(session, ctl)["success"]}


def test_generate_writes_an_analysable_table():
    import pandas as pd

    cells = [c for c in development_cells() if c.group in ("N", "D")][:4]
    rows = generate(cells, list(range(5)), [0, 1], {"C": [-1], "B0": [-1], "R": [0, 1, 2]},
                    _stub_build, _stub_run, "smoke", alpha=0.1)  # fmt: skip
    df = pd.DataFrame(rows)
    assert len(df) == 4 * 5 * 2 * (1 + 1 + 3)
    assert set(df.columns) >= {"config", "alpha", "network_seed", "policy", "episode_seed",
                               "cell", "group", "family", "success"}  # fmt: skip
    d = A._paired(df, df.config == "R", df.config == "C")
    assert len(d) == 4 * 5 * 2
    zero = pd.DataFrame(generate(cells, [0, 1], [0], {"C": [-1], "R0": [-1]}, _stub_build,
                                 lambda c, n, p, s: _stub_run("C", n, p, s), "smoke"))  # fmt: skip
    a = zero[zero.config == "C"].success.to_numpy()
    b = zero[zero.config == "R0"].success.to_numpy()
    assert np.array_equal(a, b)


def test_generate_applies_tm_and_restores_in_the_simulator(env):  # noqa: F811  (pytest fixture)
    build = _builder(env)

    def b(seed, condition, schedule, semantics, policy):
        from test_residual_training import _identity_normalizer, _scripted

        from farc.delaychange.schedule import ScheduledDelaySession

        return ScheduledDelaySession(env, _scripted, _identity_normalizer, condition, seed,
                                     20261900, schedule, semantics)  # fmt: skip

    m = env.unwrapped.model
    before = m.body_mass.copy(), m.geom_friction.copy()
    cells = [c for c in development_cells() if c.group in ("N", "M")]
    seen = {}

    def run(config, ns, policy, s):
        seen[s.draw.cell] = (float(m.body_mass[-1]), s.patched)
        _run(s)
        return {"success": s.final_success}

    generate(cells, [0], [0], {"C": [-1]}, b, run, "smoke", friction_mode="object_table")
    assert np.array_equal(before[0], m.body_mass) and np.array_equal(before[1], m.geom_friction)
    assert seen["nominal"][1] is False and seen["TM_mass2"][1] is True
    assert seen["TO_1cm"][1] is False
    del build


# --- seed-block proposal ----------------------------------------------------------------------


def test_proposed_blocks_are_disjoint_and_sufficient():
    assert P.proposal_conflicts() == []
    seeds = P.proposed_training_seeds()
    assert len(seeds) == P.training_episodes_needed() == 20_000
    assert len(set(seeds)) == len(seeds)
    a, b = P.PROPOSED_R_SEED_BLOCKS["training_monitor"]
    assert b - a == 100
    a, b = P.PROPOSED_R_SEED_BLOCKS["evaluation_extension"]
    assert b - a == 300


def test_proposed_blocks_are_not_registered_or_usable():
    from farc.supervision.seeds import R_SEED_BLOCKS, check_r_seeds

    assert not set(P.PROPOSED_R_SEED_BLOCKS) & set(R_SEED_BLOCKS)
    with pytest.raises(ValueError):
        check_r_seeds([3_730_000], "training")
    with pytest.raises(RunRefused):
        check_role_seeds("smoke", [3_740_000], PLAN)


def test_proposal_checker_detects_an_overlap(monkeypatch):
    monkeypatch.setitem(P.PROPOSED_R_SEED_BLOCKS, "bad", (3_719_990, 3_720_005))
    problems = P.proposal_conflicts()
    assert any("bad overlaps" in p and "training" in p for p in problems)


# --- non-inferiority vs degradation -----------------------------------------------------------


def _all(config, shift):
    fams = ("nominal", "RT", "RP", "RI", "T-M", "T-O", "T-A", "X-DH", "X-G")
    return {(config, f): shift for f in fams}


def _noisy_table(nominal_shift: float, seeds: int, rng_seed: int = 0):
    """Uncoupled C and R draws (independent per network seed): a wide-CI design."""
    import pandas as pd

    groups = {"nominal": "N", "RT": "D", "T-M": "M", "X-G": "X"}
    rng = np.random.default_rng(rng_seed)
    rows = []
    for fam, g in groups.items():
        for p in range(3):
            for s in range(seeds):
                rows.append({"config": "C", "alpha": np.nan, "network_seed": -1, "policy": p,
                             "episode_seed": s, "cell": fam, "group": g, "family": fam,
                             "success": bool(rng.random() < 0.9)})  # fmt: skip
                shift = nominal_shift if fam == "nominal" else 0.0
                for ns in range(3):
                    rows.append({"config": "R", "alpha": np.nan, "network_seed": ns,
                                 "policy": p, "episode_seed": s, "cell": fam, "group": g,
                                 "family": fam,
                                 "success": bool(rng.random() < 0.9 + shift)})  # fmt: skip
    return pd.DataFrame(rows)


def test_failing_non_inferiority_is_not_reported_as_degradation():
    # No true difference, few seeds, independent draws: the CI straddles -1 point and 0.
    res = A.development_analysis(_noisy_table(0.0, seeds=40, rng_seed=1), resamples=2000)
    step1 = res["primary"][0]
    assert step1["lower"] < -0.01 < 0 < step1["upper"]
    assert step1["passed"] is False and step1["degradation_shown"] is False
    assert res["claim_draft_wording"] == A.CLAIMS[0]  # what the draft would have said
    assert "inconclusive" in res["claim"]


def test_clear_harm_is_reported_as_degradation():
    eff = {**_all("R", 0.0), ("R", "nominal"): -0.2}
    res = A.development_analysis(_table(eff), resamples=2000)
    assert res["primary"][0]["degradation_shown"] is True
    assert res["claim"].startswith("the residual degrades C on N")
    assert [s["tested"] for s in res["primary"]] == [True, False, False, False]


def test_margins_are_parameters():
    eff = {**_all("R", 0.0), **{("R", f): 0.1 for f in ("T-M", "T-O", "T-A", "X-DH", "X-G")}}
    res = A.development_analysis(_table(eff), resamples=500, ni_margin=0.02, practical=0.03)
    assert res["primary"][0]["margin"] == -0.02 and res["primary"][2]["practical"] == 0.03
    assert math.isclose(res["primary"][1]["margin"], -0.02)


def test_rollback_stop_rule():
    assert [A.rollback_stop(k, 3) for k in range(4)] == [False, False, True, True]
    with pytest.raises(A.AnalysisInputError):
        A.rollback_stop(4, 3)
