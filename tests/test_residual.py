"""Residual RL architecture on top of frozen Candidate C (no training, no simulator).

Every trajectory here is SYNTHETIC: a linear stub plant and a fixed random stand-in policy,
generated from fixed seeds to exercise code paths. Nothing is a simulator episode, an
evaluation result or a physical measurement, and no experimental seed block is used.
"""

from __future__ import annotations

import copy
import itertools
from typing import Any

import numpy as np
import pytest
import torch

from farc.adaptation.estimator import STEP, ActuatorEstimator, EstimatorConfig
from farc.contact.episodes import phase_e_predictor
from farc.counterfactual.session import Observed
from farc.delaychange.mechanism import run_episode as reference_episode
from farc.delaychange.mechanism import sequential
from farc.delaychange.schedule import DelaySchedule
from farc.hardware.controller import C_CONSTANTS, ESTIMATOR, HardwareController
from farc.hardware.fifo import CommandFIFO
from farc.residual.controller import (
    ResidualController,
    ResidualEnv,
    check_training_seed,
    run_residual_episode,
)
from farc.residual.core import CandidateCCore
from farc.residual.observation import (
    OPTIONAL,
    ObservationSpec,
    ResidualObservationBuilder,
    ResidualObservationError,
)
from farc.residual.policy import ResidualActor, ZeroResidual
from farc.residual.reward import RewardConfig, residual_reward

_W = np.random.default_rng(7).standard_normal((28, 4)) * 0.5
K0 = ESTIMATOR["nominal_gain"]


def _policy(x):
    return np.tanh(np.asarray(x, np.float64) @ _W).astype(np.float32)


def _normalizer(obs, goal):
    return np.concatenate([np.asarray(obs, np.float64), np.asarray(goal, np.float64)], axis=-1)


PRED = phase_e_predictor(K0)


class StubSession:
    """SYNTHETIC linear plant with replay_drop delay semantics (no physics engine)."""

    def __init__(self, seed: int, schedule: DelaySchedule, horizon: int = 50, noise=1e-4):
        self.rng = np.random.default_rng(seed)
        self.schedule, self.horizon, self.noise = schedule, horizon, noise
        self.policy, self.normalizer = _policy, _normalizer
        self.obs = np.zeros(25)
        self.obs[0:3] = [1.3, 0.75, 0.42] + self.rng.normal(0, 0.02, 3)
        self.obs[3:6] = [1.35, 0.7, 0.425]
        self.goal = np.array([1.4, 0.8, 0.42]) + self.rng.normal(0, 0.02, 3)
        self.step = 0
        self.commands: list[np.ndarray] = []
        self.pending_view: Observed | None = None
        self.successes = np.zeros(horizon, bool)
        self.final_distance = self._dist()
        self.condition = None

    def _dist(self) -> float:
        return float(np.linalg.norm(self.obs[0:3] - self.goal))

    @property
    def done(self):
        return self.step >= self.horizon

    @property
    def final_success(self):
        return bool(self.successes[-1])

    def observe(self):
        if self.pending_view is None:
            o = self.obs + self.rng.normal(0, self.noise, 25)
            base = np.asarray(self.policy(self.normalizer(o, self.goal)[None])[0], np.float64)
            self.pending_view = Observed(self.step, o[0:3].copy(), o[3:6].copy(),
                                         self.goal.copy(), base, o.copy())  # fmt: skip
        return self.pending_view

    def act(self, command):
        c = np.asarray(command, np.float64).copy()
        self.commands.append(c)
        src = self.step - self.schedule.delay_at(self.step)
        ex = np.clip(self.commands[src] if src >= 0 else np.zeros(4), -1, 1)
        self.obs[0:3] += K0 * STEP * ex[:3]
        self.obs[20:23] = K0 * STEP * ex[:3]
        self.obs[6:9] = self.obs[3:6] - self.obs[0:3]
        self.final_distance = self._dist()
        self.successes[self.step] = self.final_distance < 0.05
        self.step += 1
        self.pending_view = None
        return ex


SCHEDULES = [DelaySchedule(()), DelaySchedule(((0, None, 3),)),
             DelaySchedule(((5, 10, 4), (15, 10, 1), (30, None, 2))),
             DelaySchedule(((0, 20, 1), (20, None, 4)))]  # fmt: skip


def _core():
    return CandidateCCore(_policy, _normalizer, PRED)


class RandomResidual:
    def __init__(self, seed=0, scale=3.0):
        self.rng, self.scale = np.random.default_rng(seed), scale

    def __call__(self, x):
        return self.rng.normal(0, self.scale, 4)


def _c_state(core: CandidateCCore):
    c = core.ctrl
    return (
        c.trusted,
        c.suspended,
        c.pending,
        c.reset_start,
        c.sigma2,
        c.gain_estimate,
        dict(c.stats),
        dict(c.starts),
        c.origin,
        [p.tolist() for p in core.est.positions],
        [x.tolist() for x in core.est.commands],
    )


def _states(n, seed):
    rng = np.random.default_rng(seed)
    g = np.array([1.3, 0.75, 0.42]) + np.cumsum(rng.normal(0, 0.01, (n, 3)), 0)
    out = []
    for t in range(n):
        o = rng.normal(0, 0.05, 25)
        o[0:3] = g[t]
        out.append(o)
    return out


# --- zero-residual equivalence ------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(5))
def test_zero_residual_matches_hardware_controller_open_loop(seed):
    goal = np.array([1.4, 0.8, 0.42])
    ref = HardwareController("C", _policy, _normalizer, {"learned": PRED})
    res = ResidualController(_core(), ZeroResidual(), alpha=0.0)
    res.reset()
    for o in _states(60, seed):
        a = ref.step(o, goal).command
        s = res.step(o, goal)
        np.testing.assert_array_equal(s.command, a)
        assert res.core.ctrl.trusted == ref.ctrl.trusted
        assert res.core.ctrl.stats == ref.ctrl.stats
        assert res.core.ctrl.reset_start == ref.ctrl.reset_start


def _reference_commands(seed, schedule):
    session = StubSession(seed, schedule)
    est = ActuatorEstimator(EstimatorConfig(**ESTIMATOR))
    stub: Any = session
    rows = reference_episode(stub, est, sequential(C_CONSTANTS), {"learned": PRED})["rows"]
    return np.array(session.commands), rows


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize(
    "policy,alpha",
    [(ZeroResidual(), 0.0), (ZeroResidual(), 0.5), (RandomResidual(), 0.0),
     (ResidualActor(ObservationSpec().dim), 1.0)],  # untrained actor: zero-initialised output
)  # fmt: skip
def test_zero_residual_reproduces_candidate_c_closed_loop(schedule, policy, alpha):
    for seed in range(3):
        ref, rows = _reference_commands(seed, schedule)
        session = StubSession(seed, schedule)
        out = run_residual_episode(session, ResidualController(_core(), policy, alpha))
        np.testing.assert_array_equal(np.array(out["commands"]), ref)
        assert [r["delay_used"] for r in out["rows"]] == [r["delay_used"] for r in rows]
        assert [r["alarm"] for r in out["rows"]] == [r["alarm"] for r in rows]


def test_equivalence_fixture_exercises_compensation():
    """Guard against a vacuous equivalence test: C must actually compensate and alarm."""
    _, rows = _reference_commands(0, SCHEDULES[3])
    assert any(r["active"] for r in rows)
    assert any(r["alarm"] for r in rows)


def test_zero_mode_env_reproduces_candidate_c():
    schedule = SCHEDULES[2]
    ref, _ = _reference_commands(1, schedule)
    env = ResidualEnv(lambda s: StubSession(s, schedule), ResidualController(_core(), alpha=0.0))
    env.reset(seed=1)
    done = False
    while not done:
        _, r, _, done, _ = env.step(np.ones(4))
        assert np.isfinite(r)
    np.testing.assert_array_equal(np.array(env.session.commands), ref)


# --- action bounds ------------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 1.0])
def test_action_bounds(alpha):
    res = ResidualController(_core(), RandomResidual(3, scale=10.0), alpha)
    res.reset()
    goal = np.array([1.4, 0.8, 0.42])
    for o in _states(50, 3):
        s = res.step(o, goal)
        assert np.all(np.abs(s.command) <= 1.0)
        assert np.all(np.abs(s.delta) <= 1.0)
        assert np.all(np.abs(s.command - s.a_c) <= alpha + 1e-12)


@pytest.mark.parametrize("alpha", [-0.1, 1.5, float("nan"), float("inf")])
def test_invalid_alpha_rejected(alpha):
    with pytest.raises(ValueError):
        ResidualController(_core(), alpha=alpha)


def test_actor_output_is_bounded():
    actor = ResidualActor(10)
    with torch.no_grad():
        for p in actor.parameters():
            p.normal_(0, 5.0)
    x = torch.randn(64, 10) * 100
    assert torch.all(actor(x).abs() <= 1.0)
    assert torch.all(actor(x, deterministic=False).abs() <= 1.0)


# --- observation dimensions ---------------------------------------------------------------------


@pytest.mark.parametrize("flags", list(itertools.product([False, True], repeat=4)))
def test_observation_dimensions(flags):
    spec = ObservationSpec(**dict(zip(OPTIONAL, flags, strict=True)))
    expected = 28 + 12 + 9 * flags[0] + 8 * flags[1] + 4 * flags[2] + flags[3] + 1 + sum(flags)
    assert spec.dim == expected
    res = ResidualController(_core(), alpha=0.1, builder=ResidualObservationBuilder(spec))
    res.reset()
    for o in _states(20, 0):
        s = res.step(o, np.array([1.4, 0.8, 0.42]))
        assert s.observation.shape == (spec.dim,)
        assert np.all(np.isfinite(s.observation))
        assert s.observation[spec.task_validity_index] == 1.0


def test_actor_rejects_wrong_dimension():
    with pytest.raises(ValueError):
        ResidualActor(ObservationSpec().dim)(np.zeros(ObservationSpec().dim + 1))


def test_observation_contains_no_privileged_inputs():
    """The builder's inputs are the core, its proposal, obs and goal: no session, schedule,
    seed or simulator object is accepted."""
    import inspect

    params = list(inspect.signature(ResidualObservationBuilder.build).parameters)
    assert params == ["self", "core", "p", "obs", "goal"]
    assert not hasattr(_core(), "session")


# --- estimator and supervisor isolation ---------------------------------------------------------


def test_observation_build_is_read_only():
    core = _core()
    builder = ResidualObservationBuilder()
    goal = np.array([1.4, 0.8, 0.42])
    for o in _states(40, 2):
        p = core.propose(o, goal)
        before = copy.deepcopy(_c_state(core))
        builder.build(core, p, o, goal)
        assert _c_state(core) == before
        core.commit(p.a_c)


class Mutator:
    def __call__(self, x):
        x[:] = np.nan  # tries to corrupt the observation it is handed
        return np.full(4, 0.3)


def test_residual_only_reaches_c_through_the_issued_command():
    """With a non-zero residual, C's state equals that of an independent reference Candidate C
    fed the same observed positions and the same issued (a_final) commands."""
    goal = np.array([1.4, 0.8, 0.42])
    res = ResidualController(_core(), RandomResidual(5, 2.0), alpha=0.3)
    res.reset()
    est, ctrl = ActuatorEstimator(EstimatorConfig(**ESTIMATOR)), sequential(C_CONSTANTS)
    for o in _states(60, 4):
        s = res.step(o, goal)
        est.observe(o[0:3])
        ref = ctrl.step(est)
        est.issued(s.command)
        assert ref.delay == s.proposal.delay and ref.alarm == s.proposal.decision.alarm
        assert ctrl.stats == res.core.ctrl.stats and ctrl.trusted == res.core.ctrl.trusted


def test_policy_cannot_mutate_controller_inputs():
    res = ResidualController(_core(), Mutator(), alpha=0.2)
    res.reset()
    goal = np.array([1.4, 0.8, 0.42])
    for o in _states(10, 0):
        s = res.step(o, goal)
        assert np.all(np.isfinite(s.observation))
        assert s.fallback is None
    w = _W.copy()
    np.testing.assert_array_equal(w, _W)


# --- delay-buffer interaction -------------------------------------------------------------------


def test_queued_commands_and_delay_queue_carry_a_final():
    schedule = SCHEDULES[1]
    seen = []

    def spy(obs, queued, gain):
        seen.append([q.copy() for q in queued])
        return PRED(obs, queued, gain)

    core = CandidateCCore(_policy, _normalizer, spy)
    res = ResidualController(core, RandomResidual(1, 1.0), alpha=0.2)
    session = StubSession(0, schedule)
    out = run_residual_episode(session, res)
    finals = np.array(out["commands"])
    np.testing.assert_array_equal(np.array(core.issued), finals)
    assert seen, "Candidate C never compensated in the fixture"
    for q in seen:  # every queued command is an issued a_final (or the pre-episode zero)
        for c in q:
            assert np.all(c == 0) or any(np.array_equal(c, f) for f in finals)
    fifo = CommandFIFO(schedule=schedule)
    for t, c in enumerate(finals):
        o = fifo.push(c, t + 1)
        expect = finals[t - 3] if t >= 3 else np.zeros(4)
        np.testing.assert_array_equal(o.executed, expect)


def test_prediction_error_uses_only_past_predictions():
    builder = ResidualObservationBuilder()
    res = ResidualController(_core(), alpha=0.0, builder=builder)
    session = StubSession(0, SCHEDULES[1])
    res.reset()
    idx = ObservationSpec().dim - 4 - 1  # prediction group's "valid" flag (before 4 validity bits)
    valid_at = []
    while not session.done:
        v = session.observe()
        s = res.propose(v.observation, v.goal, v.base_action)
        valid_at.append(s.observation[idx])
        assert all(k > v.step for k in builder._targets)  # only future targets are pending
        session.act(s.command)
        res.commit(s)
    assert valid_at[0] == 0.0 and max(valid_at) == 1.0


# --- deterministic resets -----------------------------------------------------------------------


def test_deterministic_resets():
    res = ResidualController(_core(), ResidualActor(ObservationSpec().dim), alpha=0.5)
    torch.manual_seed(0)
    with torch.no_grad():
        res.policy.mean.weight.normal_(0, 0.1)  # type: ignore[attr-defined]
    runs = []
    for _ in range(2):
        session = StubSession(11, SCHEDULES[2])
        out = run_residual_episode(session, res)
        runs.append(np.array(out["commands"]))
    np.testing.assert_array_equal(runs[0], runs[1])
    res.reset()
    assert res.core.t == 0 and res.core.issued == [] and res.core.est.positions == []
    assert res.core.ctrl.stats == {} and res.core.ctrl.trusted == 0
    assert np.all(res.builder.prev_delta == 0) and res.builder._targets == {}


def test_env_reset_is_deterministic_and_stochastic_actor_seedable():
    env = ResidualEnv(lambda s: StubSession(s, SCHEDULES[3]),
                      ResidualController(_core(), alpha=0.1))  # fmt: skip
    a, _ = env.reset(seed=5)
    env.step(np.full(4, 0.7))
    b, _ = env.reset(seed=5)
    np.testing.assert_array_equal(a, b)
    actor = ResidualActor(8)
    x = torch.randn(3, 8)
    g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(1)
    assert torch.equal(actor(x, False, g1), actor(x, False, g2))


# --- missing or invalid observations ------------------------------------------------------------


@pytest.mark.parametrize("bad", ["nan_obs", "inf_goal", "short_obs"])
def test_invalid_controller_observation_raises(bad):
    res = ResidualController(_core(), alpha=0.1)
    res.reset()
    o, g = _states(1, 0)[0], np.array([1.4, 0.8, 0.42])
    if bad == "nan_obs":
        o[3] = np.nan
    elif bad == "inf_goal":
        g[0] = np.inf
    else:
        o = o[:20]
    with pytest.raises((ResidualObservationError, ValueError)):
        res.step(o, g)


@pytest.mark.parametrize("out", [np.full(4, np.nan), np.zeros(3), np.array([0, 0, np.inf, 0])])
def test_invalid_residual_falls_back_to_candidate_c(out):
    res = ResidualController(_core(), lambda x: out, alpha=0.5)
    res.reset()
    s = res.step(_states(1, 0)[0], np.array([1.4, 0.8, 0.42]))
    assert s.fallback == "invalid_residual"
    np.testing.assert_array_equal(s.command, s.a_c)


def test_invalid_task_features_force_zero_residual():
    core = CandidateCCore(_policy, lambda o, g: _normalizer(o, g) * np.nan, PRED)
    # Candidate C itself keeps a valid policy input; only the residual's task features break.
    core._act = lambda o, g: np.asarray(_policy(_normalizer(o, g)[None])[0], np.float64)  # type: ignore[method-assign,assignment]
    res = ResidualController(core, lambda x: np.ones(4), alpha=0.5)
    res.reset()
    s = res.step(_states(1, 0)[0], np.array([1.4, 0.8, 0.42]))
    assert s.fallback == "invalid_task_observation"
    np.testing.assert_array_equal(s.command, s.a_c)
    assert s.observation[ObservationSpec().task_validity_index] == 0.0
    assert np.all(np.isfinite(s.observation))


def test_protocol_errors():
    core = _core()
    o, g = _states(1, 0)[0], np.array([1.4, 0.8, 0.42])
    with pytest.raises(RuntimeError):
        core.commit(np.zeros(4))
    p = core.propose(o, g)
    with pytest.raises(RuntimeError):
        core.propose(o, g)
    for bad in (np.full(4, 2.0), np.full(4, np.nan), np.zeros(3)):
        with pytest.raises(ValueError):
            core.commit(bad)
    core.commit(p.a_c)


@pytest.mark.parametrize("seed", [None, 4_000_000, 4_000_650, 1.5])
def test_env_refuses_heldout_or_missing_seeds(seed):
    env = ResidualEnv(lambda s: StubSession(int(s), SCHEDULES[0]), ResidualController(_core()))
    with pytest.raises(ValueError):
        env.reset(seed=seed)
    if seed is not None and seed != 1.5:
        with pytest.raises(ValueError):
            check_training_seed(seed)


def test_reward_terms():
    cfg = RewardConfig()
    z = np.zeros(4)
    assert residual_reward(cfg, True, 0.0, 0.0, z, z, 0.1) == 0.0
    assert residual_reward(cfg, False, 0.0, 0.0, z, z, 0.1) == -1.0
    assert residual_reward(cfg, True, 0.0, 0.0, np.ones(4), z, 0.5) < 0.0
