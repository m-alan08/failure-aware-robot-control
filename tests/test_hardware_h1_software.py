"""H1 pre-execution software: shadow isolation and the locked analysis.

Every input built here is SYNTHETIC: generated in the test from fixed seeds to exercise the
code paths. None of it is, or stands in for, a physical measurement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from farc.contact.episodes import phase_e_predictor
from farc.delaychange.schedule import DelaySchedule
from farc.hardware import h1_analysis as A
from farc.hardware.config import replay_config
from farc.hardware.controller import HardwareController
from farc.hardware.fifo import CommandFIFO
from farc.hardware.logging import StepLogger, verify_log
from farc.hardware.observation import ObservationAdapter, RawState, raw_from_sim_observation
from farc.hardware.pipeline import HardwarePipeline
from farc.hardware.shadow import ShadowSupervisor
from farc.paths import checkpoint_path, resolve

P = 40_000_000
SYNTHETIC = "SYNTHETIC test fixture - not a physical measurement"


# --- synthetic controller pieces (deterministic stand-ins for the policy) ---------------------

_W = np.random.default_rng(7).standard_normal((28, 4)) * 0.5


def _policy(x):
    return np.tanh(np.asarray(x, np.float64) @ _W).astype(np.float32)


def _normalizer(obs, goal):
    return np.concatenate([np.asarray(obs, np.float64), np.asarray(goal, np.float64)])


def _preds():
    kin = phase_e_predictor(0.62)
    return {"phase_e": kin, "learned": kin}  # learned stand-in: same signature, no model


def _states(n: int, seed: int) -> list[RawState]:
    """SYNTHETIC gripper/object trajectory (random walk), one sample per 40 ms."""
    rng = np.random.default_rng(seed)
    g = np.array([1.3, 0.75, 0.42]) + np.cumsum(rng.normal(0, 0.01, (n, 3)), 0)
    o = np.array([1.35, 0.7, 0.425]) + np.cumsum(rng.normal(0, 0.002, (n, 3)), 0)
    out = []
    for t in range(n):
        ns = (t + 1) * P
        out.append(RawState(ns, ns, ns, g[t], rng.normal(0, 0.1, 3), o[t],
                            np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3),
                            np.array([0.02, 0.02]), np.zeros(2)))  # fmt: skip
    return out


def _run(name: str, shadow: bool, states: list[RawState | None], schedule: DelaySchedule,
         policy=_policy, normalizer=_normalizer, preds=None, goal=None):  # fmt: skip
    pipe = HardwarePipeline(ObservationAdapter(replay_config()),
                            HardwareController(name, policy, normalizer, preds or _preds()),
                            CommandFIFO(), ShadowSupervisor() if shadow else None)  # fmt: skip
    pipe.reset(schedule)
    g = np.array([1.4, 0.8, 0.42]) if goal is None else goal
    return [pipe.cycle(s, g) for s in states]


SCHEDULES = [DelaySchedule(()), DelaySchedule(((0, None, 3),)),
             DelaySchedule(((5, 10, 4), (15, 10, 1), (30, None, 2))),
             DelaySchedule(((0, 20, 1), (20, None, 4)))]  # fmt: skip


@pytest.mark.parametrize("name", ["F", "K", "CA", "C"])
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_shadow_does_not_change_active_commands(name, schedule):
    states: list[RawState | None] = list(_states(50, 3))
    off, on = _run(name, False, states, schedule), _run(name, True, states, schedule)
    for a, b in zip(off, on, strict=True):
        assert np.array_equal(a.command, b.command)
        assert (a.executed is None) == (b.executed is None)
        if a.executed is not None:
            assert np.array_equal(a.executed, b.executed)
        assert a.record["safety_event"] == b.record["safety_event"]
        assert a.record["shadow_alarm"] is None and b.record["shadow_alarm"] is not None


@pytest.mark.parametrize("kind", ["missing", "stale"])
def test_shadow_isolation_through_safety_stops(kind):
    states: list[RawState | None] = list(_states(50, 4))
    if kind == "missing":
        states[17] = None
    else:
        s = states[17]
        assert s is not None
        states[17] = RawState(s.decision_ns + 3 * P, s.gripper_stamp_ns, s.object_stamp_ns,
                              s.gripper_position, s.gripper_velocity, s.object_position,
                              s.object_quaternion, s.object_velocity,
                              s.object_angular_velocity, s.finger_position,
                              s.finger_velocity)  # fmt: skip
    for name in ("K", "C"):
        off = _run(name, False, states, SCHEDULES[2])
        on = _run(name, True, states, SCHEDULES[2])
        for t, (a, b) in enumerate(zip(off, on, strict=True)):
            assert np.array_equal(a.command, b.command)
            assert a.record["safety_event"] == b.record["safety_event"]
            if t >= 17:
                assert a.record["safety_event"] == f"observation_{kind}"
                assert b.executed is None and b.record["shadow_alarm"] is None


def test_shadow_reproduces_active_candidate_c_decisions():
    out = _run("C", True, list(_states(50, 5)), SCHEDULES[2])
    for o in out:
        r = o.record
        assert r["shadow_alarm"] == r["alarm"] and r["shadow_suspended"] == r["suspended"]
        assert r["shadow_trusted"] == r["cusum_trusted"]
        assert r["shadow_cusum_stats"] == r["cusum_stats"]


def test_shadow_fields_pass_log_schema(tmp_path):
    out = _run("CA", True, list(_states(10, 6)), SCHEDULES[1])
    with StepLogger(tmp_path / "s.jsonl", {"fixture": SYNTHETIC}) as log:
        for o in out:
            log.write(o.record)
    assert verify_log(tmp_path / "s.jsonl") == 10


F1_STEPS = Path(str(resolve("$FARC_DATA_DIR/experiments/phase_f1_eval/steps.csv.gz")))
has_data = checkpoint_path("fetch_push_fp32_seed1").exists() and F1_STEPS.exists()


@pytest.mark.skipif(not has_data, reason="checkpoints or F1 records absent")
def test_shadow_isolation_on_recorded_trajectories():
    import sys

    sys.path.insert(0, "scripts")
    import hardware_h0_replay as h

    from farc.hardware.replay import load_f1, reconstruct_goal

    _, policy, normalizer, env, preds = h.setup(1)
    eps = [e for e in load_f1(("base_only",)) if e.policy_seed == 1
           and e.condition in ("C3_delay3", "C5_delay3_onset15")][:4]  # fmt: skip
    for e in eps:
        goal = reconstruct_goal(env, e)
        states: list[RawState | None] = [raw_from_sim_observation(v, (t + 1) * P)
                                         for t, v in enumerate(e.view)]  # fmt: skip
        states[30] = None  # a missing sample mid-episode
        for name in ("F", "K", "CA", "C"):
            off = _run(name, False, states, SCHEDULES[2], policy, normalizer, preds, goal)
            on = _run(name, True, states, SCHEDULES[2], policy, normalizer, preds, goal)
            for a, b in zip(off, on, strict=True):
                assert np.array_equal(a.command, b.command)
                if a.executed is not None:
                    assert b.executed is not None and np.array_equal(a.executed, b.executed)


# --- latency (section 4.2) ---------------------------------------------------------------------


def _latency_fixture(lag_ms: int, seed: int, noise: float = 0.02):
    """SYNTHETIC: 25 Hz commands with timestamp jitter; velocity sampled at 500 Hz responding
    to the command lag_ms later (zero-order hold) plus noise."""
    rng = np.random.default_rng(seed)
    n = 50
    cmd_t = np.cumsum(rng.integers(39_000_000, 41_000_000, n)).astype(np.int64)
    cmd = rng.uniform(-1, 1, (n, 2))
    vel_t = np.arange(cmd_t[0], cmd_t[-1] + 400_000_000, 2_000_000, dtype=np.int64)
    idx = np.searchsorted(cmd_t, vel_t - lag_ms * 1_000_000, side="right") - 1
    vel = np.where(idx[:, None] >= 0, cmd[np.clip(idx, 0, n - 1)], 0.0)
    vel = vel + rng.normal(0, noise, vel.shape)
    return cmd_t, cmd, vel_t, vel


@pytest.mark.parametrize("lag", [0, 23, 45, 87, 160, 250, 290])
def test_latency_recovers_transport_lag_with_zero_order_hold(lag):
    """H1-A1: commands are treated as a zero-order hold, so a SYNTHETIC hold-command plant
    gives back its transport lag (the pre-amendment linear method read about +20 ms)."""
    lat = A.episode_latency(*_latency_fixture(lag, lag))
    assert abs(lat.pooled_ms - lag) <= 2
    assert all(abs(x - lag) <= 3 for x in lat.per_axis_ms)
    assert lat.peak_correlation > 0.85


@pytest.mark.parametrize("lag,sensor", [(25, 15), (60, 30), (100, 8)])
def test_latency_with_receipt_stamps_includes_sensor_latency(lag, sensor):
    """H1-A1: velocity samples stamped at controller receipt (measurement + S) give L + S,
    the delay the estimator experiences. SYNTHETIC."""
    ct, c, vt, v = _latency_fixture(lag, lag + sensor)
    lat = A.episode_latency(ct, c, vt + sensor * 1_000_000, v)
    assert abs(lat.pooled_ms - (lag + sensor)) <= 2


def test_latency_on_smooth_signals():
    """SYNTHETIC smooth command sampled every 5 ms: within the hold quantisation (<= 3 ms)."""
    t = np.arange(0, 3_000_000_000, 5_000_000, dtype=np.int64)
    c = np.c_[np.sin(t / 1e9 * 7.0), np.cos(t / 1e9 * 5.0) * np.sin(t / 1e9 * 3.1)]
    lag = 63
    vt = t + 1_000_000
    u = vt / 1e9 - lag / 1e3
    v = np.c_[np.sin(u * 7.0), np.cos(u * 5.0) * np.sin(u * 3.1)]
    assert abs(A.episode_latency(t, c, vt, v).pooled_ms - lag) <= 3


def test_latency_input_errors():
    ct, c, vt, v = _latency_fixture(40, 1)
    with pytest.raises(ValueError):
        A.episode_latency(ct[::-1], c, vt, v)
    with pytest.raises(ValueError):
        A.episode_latency(ct[:5], c[:5], vt, v)


# --- exact bounds ------------------------------------------------------------------------------


def test_clopper_pearson_values():
    assert A.clopper_pearson_lower(18, 20) == pytest.approx(0.7174, abs=5e-4)
    assert A.clopper_pearson_lower(20, 20) == pytest.approx(0.05 ** (1 / 20), abs=1e-6)
    assert A.clopper_pearson_lower(0, 20) == 0.0
    lo, hi = A.clopper_pearson_interval(10, 20)
    assert lo == pytest.approx(0.2720, abs=5e-4) and hi == pytest.approx(0.7280, abs=5e-4)


def test_preregistered_constants_unchanged():
    assert (A.FEASIBILITY_MIN_CORRECT, A.Q1_EPISODES_PER_DELAY, A.WRONG_LATCH_MAX) == (18, 20, 1)
    assert (A.Q5_EPISODES, A.Q5_MIN_RELIABLE, A.Q5_MIN_AGREEMENT) == (20, 15, 0.80)
    assert (A.HALF_STEP_BAND_MS, A.STEP_MS, A.LAG_RANGE_MS) == (8.0, 40.0, (0, 300))
    assert (A.ADDITIVITY_POINT_MS, A.ADDITIVITY_CI_MS) == (10.0, 20.0)
    assert (A.BOOTSTRAP_RESAMPLES, A.BOOTSTRAP_SEED) == (2_000, 20264101)
    assert A.GAIN_RANGE == (0.7, 1.3) and A.NOISE_FLAG_M == 0.0025 and A.SAFETY_STOP_LIMIT == 3
    doc = Path("docs/HARDWARE_H1_PREREGISTRATION.md").read_text()
    for s in (
        "≥ 18 / 20 correct",
        "≤ 1 / 20",
        "|Δ_d| ≤ 10 ms",
        "±20 ms",
        "seed 20264101",
        "±8 ms of 20 ms",
        "at least 15 / 20",
        "at least 80%",
        "[0.7, 1.3]",
        "2.5 mm",
    ):
        assert s in doc, s


# --- Q5 / Q1 fixtures ----------------------------------------------------------------------------


def _ep(block: str, seq: int, d: int, final: int, *, reliable: bool = True, lat: float = 45.0,
        series: tuple[int, ...] | None = None, **kw: Any) -> A.EpisodeResult:  # fmt: skip
    return A.EpisodeResult(block, seq, d, series if series is not None else (0,) * 49 + (final,),
                           reliable, final, kw.pop("gain", 1.0), lat,
                           kw.pop("fifo_exact", True), **kw)  # fmt: skip


def _q5(lag: int = 1, lat: float = 45.0, n_reliable: int = 20, n_other: int = 0):
    eps = []
    for i in range(20):
        if i < n_reliable - n_other:
            eps.append(_ep("Q5", i, 0, lag, lat=lat))
        elif i < n_reliable:
            eps.append(_ep("Q5", i, 0, lag + 1, lat=lat))
        else:
            eps.append(_ep("Q5", i, 0, 0, reliable=False, lat=lat))
    return eps


def test_q5_resolves_and_freezes_expected_answers():
    r = A.resolve_q5(_q5())
    assert r.resolved and r.d0 == 1 and r.d0_ts == 1 and not r.ambiguous
    assert r.expected == {0: 1, 1: 2, 2: 3, 3: 4, 4: 5}


@pytest.mark.parametrize("kw,reason", [
    ({"n_reliable": 14}, "(a)"),
    ({"n_other": 5}, "(b)"),  # 15/20 agreement = 0.75
    ({"n_other": 10}, "(b)"),  # tie: no unique mode
    ({"lat": 60.0}, "(c)"),  # 60 mod 40 = 20: half-step
    ({"lat": 52.0}, "(c)"),  # 12 ms from a boundary centre: inside +-8? no: |12-20| = 8 -> in
    ({"lat": 85.0}, "(d)"),  # d0_ts = 2, estimator mode 1
])  # fmt: skip
def test_q5_unresolved_conditions(kw, reason):
    r = A.resolve_q5(_q5(**kw))
    assert not r.resolved and any(x.startswith(reason) for x in r.reasons), r.reasons
    assert r.expected == {}


def test_q5_band_edges_and_aborted_episodes():
    assert A.resolve_q5(_q5(lat=51.9)).resolved  # |11.9 - 20| = 8.1 > 8: not ambiguous
    eps = _q5()
    for i in range(6):
        eps[i] = _ep("Q5", i, 0, 1, aborted=True, abort_reason="observation_stale")
    r = A.resolve_q5(eps)
    assert not r.resolved and r.reliable == 14  # aborted episodes are not reliable


def _q1_block(d: int, expected: int, *, correct: int = 20, wrong: int = 0, abstain: int = 0,
              aborted: int = 0, lat_shift: float = 0.0, l0: float = 45.0,
              fifo_exact: bool = True) -> list[A.EpisodeResult]:  # fmt: skip
    eps = []
    other = expected + 1 if expected < 4 else expected - 1
    other = other if other > 0 else 2
    kinds = ["c"] * correct + ["w"] * wrong + ["a"] * abstain + ["x"] * aborted
    kinds += ["a"] * (20 - len(kinds))
    for i, k in enumerate(kinds):
        final = {"c": expected, "w": other, "a": 0, "x": expected}[k]
        eps.append(_ep("Q1", i, d, final, lat=l0 + d * 40 + lat_shift, fifo_exact=fifo_exact,
                       aborted=k == "x",
                       abort_reason="watchdog" if k == "x" else None))  # fmt: skip
    return eps


def test_q1_full_pass_and_ineligible_delay():
    eps = _q5() + [e for d in range(5) for e in _q1_block(d, d + 1)]
    rep = A.analyse(eps)
    delays = {r.delay: r for r in rep["q1"]["delays"]}
    assert all(delays[d].status == "passed" for d in range(4))
    assert delays[4].status == "ineligible" and delays[4].expected == 5
    assert rep["q1"]["status"] == "passed"
    assert delays[0].one_sided_lower == pytest.approx(0.05 ** (1 / 20), abs=1e-6)


@pytest.mark.parametrize("kw,status", [
    ({"correct": 18, "abstain": 2}, "passed"),
    ({"correct": 17, "abstain": 3}, "failed"),
    ({"correct": 18, "wrong": 2}, "failed"),
    ({"correct": 18, "wrong": 1, "abstain": 1}, "passed"),
    ({"correct": 18, "aborted": 2}, "passed"),
    ({"correct": 17, "aborted": 3}, "failed"),  # aborted = not correct
    ({"lat_shift": 15.0}, "not_interpretable"),  # additivity violated despite 20/20
    ({"fifo_exact": False}, "not_interpretable"),
])  # fmt: skip
def test_q1_per_delay_criteria(kw, status):
    q5 = _q5()
    r = A.analyse_delay(2, 3, q5, _q1_block(2, 3, **kw))
    assert r.status == status, r.reasons
    if "abstain" in kw:
        assert r.abstained == kw["abstain"] and r.wrong_latch == kw.get("wrong", 0)


def test_additivity_ci_rule():
    q5 = _q5()
    rng = np.random.default_rng(0)
    noisy = [_ep("Q1", i, 1, 2, lat=45 + 40 + float(rng.normal(0, 40))) for i in range(20)]
    a = A.additivity(q5, noisy, 1)
    assert a.ci_ms is not None and not a.passed  # CI wider than +-20 ms
    # deterministic: same inputs, same bootstrap
    assert A.additivity(q5, noisy, 1).ci_ms == a.ci_ms
    unpaired = [_ep("Q1", 100 + i, 1, 2, lat=85.0) for i in range(20)]
    assert A.additivity(q5, unpaired, 1).reason == "no shared sequence ids for pairing"


def test_zero_total_delay_false_positive_and_transient_latch():
    q5 = _q5(lag=0, lat=5.0)
    assert A.resolve_q5(q5).d0 == 0
    eps = [_ep("Q1", i, 0, 0, lat=5.0) for i in range(17)]
    eps.append(_ep("Q1", 17, 0, 0, lat=5.0, series=(0,) * 10 + (2,) * 5 + (0,) * 35))  # transient
    eps.append(_ep("Q1", 18, 0, 3, lat=5.0))  # false positive (final latched > 0)
    eps.append(_ep("Q1", 19, 0, 1, lat=5.0))  # false positive
    r = A.analyse_delay(0, 0, q5, eps)
    assert r.wrong_latch == 2 and r.transient_false_latch == 3 and r.correct == 18
    assert r.status == "failed" and any("wrong latches" in x for x in r.reasons)


def test_q1_not_run_when_q5_unresolved_and_safety_accounting():
    eps = _q5(n_reliable=10) + _q1_block(1, 2)
    rep = A.analyse(eps)
    assert rep["q1"]["status"] == "not_run"
    eps = _q5() + _q1_block(1, 2, correct=17, aborted=3)
    s = A.safety_accounting(eps)
    assert s["safety_stops"] == 3 and s["session_must_stop"] and s["by_reason"] == {"watchdog": 3}


def test_sensor_noise_and_gain_summaries():
    rng = np.random.default_rng(11)
    t = np.arange(0, 30_000_000_000, 10_000_000, dtype=np.int64)  # SYNTHETIC 100 Hz, 30 s
    drift = np.outer(t / 1e9, [1e-4, -2e-4, 0.0])
    quiet = A.sensor_noise(t, drift + rng.normal(0, 0.001, (len(t), 3)), "gripper")
    assert all(s == pytest.approx(0.001, rel=0.05) for s in quiet["std_m"])
    assert all(abs(a) < 0.05 for a in quiet["lag1_autocorr"])
    assert quiet["rate_hz"] == pytest.approx(100.0) and not quiet["flag_over_2p5mm"]
    loud = A.sensor_noise(t, rng.normal(0, 0.003, (len(t), 3)), "gripper")
    assert loud["flag_over_2p5mm"]
    eps = [_ep("Q1", i, 1, 2, gain=0.6) for i in range(20)]
    eps += [_ep("Q1", i, 2, 3, gain=1.0) for i in range(20)]
    g = A.gain_summary(eps)
    assert g[1]["outside_range"] and not g[2]["outside_range"]
