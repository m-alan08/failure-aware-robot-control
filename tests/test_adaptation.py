"""Phase E: online actuator identification and adaptive compensation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from farc.adaptation.compensation import (
    CompensationConfig,
    Fixed,
    LatchedEstimate,
    Parameters,
    compensate,
    predicted_observation,
)
from farc.adaptation.estimator import STEP, ActuatorEstimator, EstimatorConfig
from farc.adaptation.experiment import (
    AdaptationConfigError,
    configurations,
    load_adaptation_experiment,
    true_parameters,
    validate_adaptation_experiment,
)
from farc.counterfactual.artifacts import ArtifactError, analyze_existing, write_raw
from farc.disturbances.config import ConditionSpec, DisturbanceSpec
from farc.envs.observation_layout import layout_for
from farc.paths import checkpoint_path

K0 = 0.62
CONFIG = "configs/adaptation/phase_e_pilot.json"


def synthetic(gain, delay, steps=40, noise=0.0, seed=0, commands=None, gain_after=None):
    """Plant: g[t+1] = g[t] + K0 * gain * STEP * c[t - delay] (+ observation noise).
    Commands are the issued ones; ``gain_after`` = (t0, g) changes the gain from step t0."""
    rng = np.random.default_rng(seed)
    if commands is None:
        phase = rng.uniform(0, 2 * np.pi, 2)
        commands = np.stack([np.r_[np.sin(0.4 * t + phase), 0.0] for t in range(steps)])
    true = np.zeros(3)
    positions, issued = [], []
    for t in range(steps):
        positions.append(true + noise * rng.standard_normal(3))
        issued.append(commands[t])
        g = gain if gain_after is None or t < gain_after[0] else gain_after[1]
        executed = commands[t - delay] if t - delay >= 0 else np.zeros(3)
        true = true + K0 * g * STEP * np.r_[executed[:2], 0.0]
    return positions, issued


def run_estimator(positions, issued, config=None):
    est = ActuatorEstimator(config or EstimatorConfig(nominal_gain=K0))
    estimates = []
    for p, c in zip(positions, issued, strict=True):
        est.observe(p)
        estimates.append(est.estimate())
        est.issued(c)
    return estimates


# --- identification on controlled synthetic signals -----------------------------------------


@pytest.mark.parametrize("gain", [1.0, 0.5, 0.15])
@pytest.mark.parametrize("delay", [0, 1, 3])
def test_recovers_gain_and_delay(gain, delay):
    estimates = run_estimator(*synthetic(gain, delay))
    late = [e for e in estimates[15:] if e.gain_reliable and e.delay_reliable]
    assert len(late) > 20
    assert all(e.delay == delay for e in late)
    assert all(abs(e.gain - gain) < 1e-6 for e in late)


def test_noisy_signal_lowers_fit_and_abstains():
    estimates = run_estimator(*synthetic(0.5, 0, noise=0.02))
    assert np.mean([e.gain_reliable for e in estimates]) < 0.2


def test_no_excitation_abstains():
    estimates = run_estimator(*synthetic(0.5, 1, commands=np.zeros((40, 3))))
    assert not any(e.gain_reliable or e.delay_reliable for e in estimates)
    small = np.full((40, 3), 0.05)
    estimates = run_estimator(*synthetic(0.5, 1, commands=small))
    assert not any(e.gain_reliable for e in estimates)


def test_constant_command_leaves_delay_unidentified():
    """Every lag explains a constant command equally well: the delay must abstain."""
    estimates = run_estimator(*synthetic(0.5, 2, commands=np.tile([0.8, -0.6, 0.0], (40, 1))))
    assert not any(e.delay_reliable for e in estimates[15:])


def test_too_few_transitions_abstains():
    estimates = run_estimator(*synthetic(0.5, 0))
    assert not any(e.gain_reliable for e in estimates[:4])


# --- causality ------------------------------------------------------------------------------


def test_estimate_depends_only_on_the_past():
    """Changing the plant after step t cannot change any estimate up to t."""
    a = run_estimator(*synthetic(1.0, 1))
    b = run_estimator(*synthetic(1.0, 1, gain_after=(25, 0.3)))
    assert list(map(repr, a[:26])) == list(
        map(repr, b[:26])
    )  # the step-25 estimate uses transitions up to position 25
    assert repr(a[30]) != repr(b[30])


def test_streaming_matches_prefix_and_enforces_order():
    positions, issued = synthetic(0.5, 1)
    full = run_estimator(positions, issued)
    prefix = run_estimator(positions[:12], issued[:12])
    assert list(map(repr, full[:12])) == list(map(repr, prefix))
    est = ActuatorEstimator(EstimatorConfig(nominal_gain=K0))
    est.observe(np.zeros(3))
    with pytest.raises(RuntimeError):
        est.observe(np.zeros(3))
    with pytest.raises(RuntimeError):
        ActuatorEstimator(EstimatorConfig(nominal_gain=K0)).issued(np.zeros(4))


def test_latched_estimate_keeps_last_reliable_value():
    estimates = run_estimator(*synthetic(0.5, 0))
    source = LatchedEstimate()
    params = [source.parameters(e) for e in estimates]
    assert params[0] == Parameters(None, None)  # abstains before the first reliable estimate
    assert params[-1].gain == pytest.approx(0.5)
    assert params[-1].delay == 0


def test_sham_gates_on_reliability_but_ignores_values():
    estimates = run_estimator(*synthetic(1.0, 0))
    sham = Fixed(0.5, None, gate_on_reliability=True)
    params = [sham.parameters(e) for e in estimates]
    assert params[0].gain is None and params[-1].gain == 0.5


# --- compensation bounds ------------------------------------------------------------------


def _layout():
    class Raw:
        has_object = True

    return layout_for(Raw(), 25)


def _comp(config, params, base, policy_at=None, history=None):
    obs = np.arange(25, dtype=float)
    return compensate(config, params, np.asarray(base, float), policy_at or (lambda o: base),
                      obs, _layout(), history or [], K0)  # fmt: skip


@pytest.mark.parametrize("gain", [0.05, 0.15, 0.5, 0.8])
def test_gain_compensation_is_clipped_and_floored(gain):
    base = np.array([0.3, -0.9, 0.1, 0.7])
    d = _comp(CompensationConfig(True, False, gain_floor=0.25), Parameters(gain, None), base)
    assert np.all(np.abs(d.command) <= 1.0)
    expected = np.clip(base[:3] / max(gain, 0.25), -1, 1)
    np.testing.assert_allclose(d.command[:3], expected)
    assert d.command[3] == base[3]  # finger untouched
    assert d.gain_active and d.saturated == bool(np.any(np.abs(base[:3] / max(gain, 0.25)) > 1))


def test_gain_near_nominal_or_unknown_is_not_compensated():
    base = np.array([0.3, -0.2, 0.1, 0.0])
    for params in (Parameters(0.9, None), Parameters(None, None)):
        d = _comp(CompensationConfig(True, False), params, base)
        np.testing.assert_array_equal(d.command, base)
        assert not d.gain_active


def test_predicted_observation_shifts_gripper_and_relative_object():
    layout = _layout()
    obs = np.zeros(25)
    queued = [np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0, 0.0])]  # 2.0 is clipped
    pred = predicted_observation(obs, layout, queued, K0, 0.5)
    delta = K0 * 0.5 * STEP * np.array([1.0, 1.0, 0.0])
    np.testing.assert_allclose(pred[0:3], delta)
    np.testing.assert_allclose(pred[6:9], -delta)
    assert np.all(pred[3:6] == 0) and np.all(pred[9:] == 0)


def test_delay_compensation_queries_policy_at_predicted_state():
    seen = []
    history = [np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0]), np.array([0.5, 0, 0, 0])]

    def policy_at(observation):
        seen.append(observation)
        return np.array([0.2, 0, 0, 0])

    d = _comp(CompensationConfig(False, True), Parameters(None, 2), np.zeros(4),
              policy_at=policy_at, history=history)  # fmt: skip
    assert d.delay_active and len(seen) == 1
    expected = K0 * STEP * 1.5  # the last two issued commands are still queued
    assert seen[0][0] - 0.0 == pytest.approx(expected)
    np.testing.assert_allclose(d.command, [0.2, 0, 0, 0])


# --- ground truth is only used by the oracle --------------------------------------------


def test_true_parameters_respect_the_window():
    c = ConditionSpec("x", (DisturbanceSpec("action_scale", 0.5, onset=15, duration=10),))
    assert true_parameters(c, 14) == (1.0, 0)
    assert true_parameters(c, 15) == (0.5, 0)
    assert true_parameters(c, 25) == (1.0, 0)


def test_only_the_oracle_reads_true_parameters():
    config = load_adaptation_experiment(CONFIG)
    kinds = {name: kind for name, (_, kind) in configurations(config).items()}
    assert [n for n, k in kinds.items() if k == "oracle"] == ["oracle_combined"]
    assert all(n.startswith(("estimated", "base")) for n, k in kinds.items()
               if k in ("estimate", "none"))  # fmt: skip


# --- seed-range protection ------------------------------------------------------------------


def test_pilot_config_is_valid_and_disjoint():
    config = load_adaptation_experiment(CONFIG)
    assert not set(config.selection_seeds) & set(config.evaluation_seeds)
    assert max(config.selection_seeds + config.evaluation_seeds) < 4_000_000


@pytest.mark.parametrize("change", [
    {"evaluation_seed_offset": 4_000_000},
    {"evaluation_seed_offset": 3_999_990, "evaluation_episodes": 30},  # crosses into held-out
    {"selection_seed_offset": 3_630_005},  # Phase E debugging seeds
    {"selection_seed_offset": 3_621_030},  # Phase D
    {"evaluation_seed_offset": 3_631_010},  # overlaps selection
    {"selection_seed_offset": 3_400_000},  # outside the validation stage
])  # fmt: skip
def test_seed_ranges_are_protected(change):
    config = load_adaptation_experiment(CONFIG)
    with pytest.raises(AdaptationConfigError):
        validate_adaptation_experiment(replace(config, **change))


# --- raw artifacts --------------------------------------------------------------------------

RAW = ("episodes.csv.gz", "steps.csv.gz", "metadata.json")


def _digest(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_analysis_never_touches_raw_files_even_on_failure(tmp_path):
    episodes = pd.DataFrame({"configuration": ["base_only"], "success": [True]})
    steps = pd.DataFrame({"t": [0, 1]})
    write_raw(tmp_path, {RAW[0]: episodes, RAW[1]: steps}, {"seconds": 1.0})
    raw = {n: (tmp_path / n).read_bytes() for n in RAW}

    def derive(e, s):
        return {"summary": pd.DataFrame({"n": [len(e), len(s)]})}, {"ok": True}

    analyze_existing(tmp_path, derive, None, ("summary",), RAW)
    assert json.loads((tmp_path / "analysis.json").read_text())["ok"]

    def broken(e, s):
        raise ValueError("bug")

    before = _digest(tmp_path)
    with pytest.raises(ValueError):
        analyze_existing(tmp_path, broken, None, ("summary",), RAW)
    assert _digest(tmp_path) == before
    with pytest.raises(ArtifactError):  # only the designated derived tables may be written
        analyze_existing(tmp_path, lambda e, s: ({"other": e}, {}), None, ("summary",), RAW)
    assert _digest(tmp_path) == before
    with pytest.raises(ArtifactError, match="already exist"):
        write_raw(tmp_path, {RAW[0]: episodes, RAW[1]: steps}, {})
    assert {n: (tmp_path / n).read_bytes() for n in RAW} == raw


# --- exact replay in the simulator ----------------------------------------------------------

CHECKPOINT = checkpoint_path("fetch_push_fp32_seed2")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="FetchPush checkpoint absent")
def test_every_configuration_replays_exactly():
    from farc.adaptation.experiment import check_replay

    config = replace(load_adaptation_experiment(CONFIG),
                     checkpoints=(str(CHECKPOINT),))  # fmt: skip
    names = ["action_delay_sev3_on0_durinf", "action_scale_sev0.5_on15_dur10"]
    check_replay(config, str(CHECKPOINT), names, 3_000_777)
