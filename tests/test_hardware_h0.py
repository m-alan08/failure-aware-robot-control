"""Stage H0 tests: observation adapter, FIFO, controller equivalence, logging, safety, timing.

No simulator stepping and no hardware. Tests that need the frozen checkpoints or the archived
F1 records skip when those are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces
from gymnasium_robotics.utils import rotations

from farc.delaychange.schedule import DelaySchedule
from farc.envs.observation_layout import OBJECT
from farc.hardware.config import CalibrationMissing, HardwareConfig, load_config, replay_config
from farc.hardware.controller import C_CONSTANTS, ESTIMATOR
from farc.hardware.fifo import CommandFIFO
from farc.hardware.logging import StepLogger, empty_record, validate_record, verify_log
from farc.hardware.observation import (
    ObservationAdapter,
    ObservationError,
    RawState,
    raw_from_sim_observation,
)
from farc.paths import checkpoint_path, resolve
from farc.retrain.phase_h import DelayHistoryWrapper

S = OBJECT.slices
P = 40_000_000
has_ckpt = checkpoint_path("fetch_push_fp32_seed1").exists()
has_f1 = Path(str(resolve("$FARC_DATA_DIR/experiments/phase_f1_eval/steps.csv.gz"))).exists()
needs_data = pytest.mark.skipif(not (has_ckpt and has_f1), reason="checkpoints or F1 absent")


def _state(t: int = 1, **kw) -> RawState:
    base = dict(decision_ns=t * P, gripper_stamp_ns=t * P, object_stamp_ns=t * P,
                gripper_position=np.array([1.3, 0.75, 0.42]),
                gripper_velocity=np.array([0.1, -0.2, 0.0]),
                object_position=np.array([1.35, 0.7, 0.425]),
                object_quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
                object_velocity=np.array([0.05, 0.0, 0.0]),
                object_angular_velocity=np.array([0.0, 0.0, 0.3]),
                finger_position=np.array([0.02, 0.02]),
                finger_velocity=np.array([0.0, 0.0]))  # fmt: skip
    base.update(kw)
    return RawState(**base)


def _rot(seed: int) -> np.ndarray:
    q = np.random.default_rng(seed).standard_normal(4)
    return rotations.quat2mat(q / np.linalg.norm(q))


def _config(R: np.ndarray, t) -> HardwareConfig:
    base = replay_config()
    cal = base.calibration.__class__(**{**base.calibration.__dict__,
                                        "world_from_base_rotation": R.tolist(),
                                        "world_from_base_translation": list(t)})  # fmt: skip
    return HardwareConfig(platform="franka", calibration=cal)


# --- observation adapter -------------------------------------------------------------------------


def test_observation_dimensions_and_ordering_identity():
    obs = ObservationAdapter(replay_config()).observation(_state())
    assert obs.shape == (25,)
    s, dt = _state(), 0.04
    np.testing.assert_array_equal(obs[0:3], s.gripper_position)
    np.testing.assert_array_equal(obs[3:6], s.object_position)
    np.testing.assert_array_equal(obs[6:9], s.object_position - s.gripper_position)
    np.testing.assert_array_equal(obs[9:11], s.finger_position)
    np.testing.assert_array_equal(obs[11:14], np.zeros(3))  # identity orientation
    np.testing.assert_allclose(obs[14:17], (s.object_velocity - s.gripper_velocity) * dt)
    np.testing.assert_array_equal(obs[17:20], s.object_angular_velocity * dt)
    np.testing.assert_array_equal(obs[20:23], s.gripper_velocity * dt)
    np.testing.assert_array_equal(obs[23:25], np.zeros(2))
    # slices agree with the simulator layout table
    assert [S[k] for k in ("gripper_position", "object_relative_velocity", "finger_velocity")] == [
        slice(0, 3),
        slice(14, 17),
        slice(23, 25),
    ]


def test_coordinate_transformation():
    R, t = _rot(3), np.array([0.5, -0.2, 0.1])
    a = ObservationAdapter(_config(R, t))
    s = _state(object_quaternion=rotations.mat2quat(_rot(4)))
    obs = a.observation(s)
    np.testing.assert_allclose(obs[0:3], R @ s.gripper_position + t, atol=1e-15)
    np.testing.assert_allclose(obs[3:6], R @ s.object_position + t, atol=1e-15)
    np.testing.assert_allclose(obs[6:9], R @ (s.object_position - s.gripper_position), atol=1e-15)
    np.testing.assert_allclose(obs[20:23], R @ s.gripper_velocity * 0.04, atol=1e-15)
    np.testing.assert_allclose(obs[17:20], R @ s.object_angular_velocity * 0.04, atol=1e-15)
    np.testing.assert_allclose(obs[11:14], rotations.mat2euler(R @ _rot(4)), atol=1e-12)
    np.testing.assert_allclose(a.goal_from_base([0.1, 0.2, 0.3]), R @ [0.1, 0.2, 0.3] + t)
    with pytest.raises(ValueError):
        ObservationAdapter(_config(-np.eye(3), t))  # improper rotation


def test_marker_offset_is_applied_in_object_frame():
    base = replay_config()
    cal = base.calibration.__class__(**{**base.calibration.__dict__,
                                        "object_marker_offset_m": [0.0, 0.0, -0.025]})  # fmt: skip
    a = ObservationAdapter(HardwareConfig(platform="franka", calibration=cal))
    quat = rotations.euler2quat(np.array([np.pi, 0.0, 0.0]))  # upside down: offset flips
    obs = a.observation(_state(object_quaternion=quat))
    np.testing.assert_allclose(obs[3:6], _state().object_position + [0, 0, 0.025], atol=1e-12)


def test_round_trip_from_simulator_observation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        o = rng.uniform(-1, 1, 25)
        o[6:9] = o[3:6] - o[0:3]
        o[11:14] = rng.uniform(-1, 1, 3)  # away from gimbal lock
        back = ObservationAdapter(replay_config()).observation(raw_from_sim_observation(o, P))
        np.testing.assert_allclose(back, o, rtol=0, atol=1e-13)


@pytest.mark.parametrize("bad,kind", [
    ({"gripper_position": np.array([np.nan, 0, 0])}, "non_finite"),
    ({"object_position": np.zeros(2)}, "shape"),
    ({"object_quaternion": np.array([2.0, 0, 0, 0])}, "quaternion"),
    ({"gripper_stamp_ns": 0, "decision_ns": 2 * P}, "stale"),
    ({"object_stamp_ns": 3 * P}, "future_stamp"),
])  # fmt: skip
def test_invalid_observations_raise(bad, kind):
    with pytest.raises(ObservationError) as err:
        ObservationAdapter(replay_config()).observation(_state(**bad))
    assert err.value.kind == kind


def test_repeated_and_skewed_samples_raise():
    a = ObservationAdapter(replay_config())
    a.observation(_state(1))
    with pytest.raises(ObservationError) as err:
        a.observation(_state(1, decision_ns=2 * P))
    assert err.value.kind == "repeated"
    with pytest.raises(ObservationError) as err:
        ObservationAdapter(replay_config()).observation(_state(2, object_stamp_ns=P + P // 2))
    assert err.value.kind == "skew"


def test_template_placeholders_refuse_to_run():
    cfg = load_config("configs/hardware/h0_template.json")
    assert set(cfg.calibration.missing()) == set(cfg.calibration.__dict__)
    with pytest.raises(CalibrationMissing):
        cfg.require_calibrated()
    with pytest.raises(CalibrationMissing):
        ObservationAdapter(cfg)
    with pytest.raises(ValueError):
        HardwareConfig(platform="franka", control_period_s=0.02)


# --- FIFO --------------------------------------------------------------------------------------


class _Recorder(gym.Env):
    """Stub environment (no physics): records executed actions."""

    def __init__(self):
        self.action_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.observation_space = spaces.Dict({"observation": spaces.Box(-1, 1, (1,))})
        self.executed: list[np.ndarray] = []

    def reset(self, *, seed=None, options=None):
        self.executed = []
        return {"observation": np.zeros(1)}, {}

    def step(self, action):
        self.executed.append(np.asarray(action, np.float64))
        return {"observation": np.zeros(1)}, 0.0, False, False, {}


def _schedules():
    yield DelaySchedule(())
    for d in range(1, 5):
        yield DelaySchedule(((0, None, d),))
    yield DelaySchedule(((5, 6, 3), (20, None, 1)))
    yield DelaySchedule(((0, 10, 4), (10, 10, 1), (20, 10, 4)))
    yield DelaySchedule(((3, 2, 2), (7, 3, 4), (12, None, 2)))


@pytest.mark.parametrize("schedule", list(_schedules()))
def test_fifo_matches_simulator_delay_semantics(schedule):
    rng = np.random.default_rng(1)
    cmds = rng.uniform(-1.3, 1.3, (50, 4))  # includes out-of-bound commands (clipping)
    env = _Recorder()
    wrapper = DelayHistoryWrapper(env, 0, lambda _ep: schedule)
    wrapper.reset()
    fifo = CommandFIFO(schedule=schedule)
    for t, c in enumerate(cmds):
        wrapper.step(c)
        out = fifo.push(c, (t + 1) * P)
        # the wrapper casts to float32 at env.step (the simulator boundary); the FIFO keeps
        # float64 for the robot interface. Equal after the same cast; same source and clip.
        np.testing.assert_array_equal(out.executed.astype(np.float32), env.executed[-1])
        assert out.delay == schedule.delay_at(t)


def test_fifo_ordering_initial_zeros_changes_and_reset():
    fifo = CommandFIFO()
    fifo.set_delay(2)
    outs = [fifo.push(np.full(4, float(i) / 10), (i + 1) * P) for i in range(5)]
    assert [o.source_step for o in outs] == [-2, -1, 0, 1, 2]
    np.testing.assert_array_equal(outs[0].executed, np.zeros(4))
    np.testing.assert_array_equal(outs[2].executed, np.full(4, 0.0))
    np.testing.assert_array_equal(outs[4].executed, np.full(4, 0.2))
    assert outs[4].source_issued_ns == 3 * P
    fifo.set_delay(4)  # increase: replays an older command
    o = fifo.push(np.full(4, 0.5), 6 * P)
    assert o.source_step == 1
    fifo.set_delay(0)  # decrease: drops the queue, executes the newest
    o = fifo.push(np.full(4, 0.6), 7 * P)
    np.testing.assert_array_equal(o.executed, np.full(4, 0.6))
    fifo.reset()
    fifo.set_delay(1)
    o = fifo.push(np.ones(4) * 0.3, 8 * P)
    np.testing.assert_array_equal(o.executed, np.zeros(4))  # nothing carried over a reset
    for bad in (-1, 5, 1.5):
        with pytest.raises(ValueError):
            fifo.set_delay(bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fifo.push(np.ones(4), 8 * P)  # non-increasing timestamp
    with pytest.raises(ValueError):
        fifo.push(np.array([np.nan, 0, 0, 0]), 9 * P)
    with pytest.raises(RuntimeError):
        CommandFIFO(schedule=DelaySchedule(())).set_delay(1)


# --- logging -----------------------------------------------------------------------------------


def _rec(t: int) -> dict:
    r = empty_record()
    r.update({"t": t, "configuration": "C", "mono_ns": t, "command": [0.0, 0.1, 0.2, 0.0]})
    return r


def test_logging_integrity_and_write_once(tmp_path):
    path = tmp_path / "log.jsonl"
    with StepLogger(path, {"run": "test"}) as log:
        for t in range(5):
            log.write(_rec(t))
    assert verify_log(path) == 5
    with pytest.raises(FileExistsError):
        StepLogger(path, {})
    lines = path.read_bytes().splitlines(keepends=True)
    tampered = lines[3].replace(b'"t": 2', b'"t": 9')
    (tmp_path / "bad.jsonl").write_bytes(b"".join(lines[:3] + [tampered] + lines[4:]))
    with pytest.raises(ValueError):
        verify_log(tmp_path / "bad.jsonl")
    (tmp_path / "gap.jsonl").write_bytes(b"".join(lines[:2] + lines[3:]))
    with pytest.raises(ValueError):
        verify_log(tmp_path / "gap.jsonl")


def test_schema_rejects_missing_extra_and_wrong_types():
    r = {**_rec(0), "seq": 0, "prev_sha256": "x"}
    validate_record(r)
    for broken in ({k: v for k, v in r.items() if k != "alarm"}, {**r, "surprise": 1},
                   {**r, "t": "0"}, {**r, "active": 1}, {**r, "learned_calls": True}):  # fmt: skip
        with pytest.raises(ValueError):
            validate_record(broken)


# --- controller (needs checkpoints and F1 records) --------------------------------------------


def test_frozen_constants_match_g4_config():
    v = json.loads(Path("configs/delaychange/g4_heldout.json").read_text())
    assert v["estimator"] == ESTIMATOR and v["constants"] == C_CONSTANTS


@pytest.fixture(scope="module")
def replay():
    import sys

    sys.path.insert(0, "scripts")
    import hardware_h0_replay as h

    from farc.hardware.replay import load_f1, reconstruct_goal

    _, policy, normalizer, env, preds = h.setup(1)
    eps = [e for e in load_f1() if e.policy_seed == 1 and e.condition in
           ("C3_delay3", "C5_delay3_onset15", "C8_posnoise0.01")]  # fmt: skip
    picked = []
    for cfg in ("base_only", "phase_e_predictor", "contact_learned"):
        for cond in ("C3_delay3", "C5_delay3_onset15", "C8_posnoise0.01"):
            picked += [e for e in eps if e.configuration == cfg and e.condition == cond][:2]
    for e in picked:
        e.goal = reconstruct_goal(env, e)
    return h, policy, normalizer, preds, picked


@needs_data
def test_closed_loop_reproduces_recorded_commands_bitwise(replay):
    h, policy, normalizer, preds, eps = replay
    for e in eps:
        name = h.RECORDED[e.configuration]
        for path in ("direct", "adapter"):
            r = h.closed_loop(e, name, policy, normalizer, preds, path)
            assert r["cmd_exact_steps"] == r["steps"], (e.configuration, e.condition, path)
            assert r["delay_mismatch"] == 0 and r["active_mismatch"] == 0
            assert r["gain_max_abs"] == 0.0 and r["pred_max_abs"] == 0.0


@needs_data
def test_open_loop_matches_original_loops_for_all_configurations(replay):
    h, policy, normalizer, preds, eps = replay
    alarms = 0
    for e in [x for x in eps if x.configuration == "base_only"]:
        for cfg in ("F", "K", "CA", "C"):
            r = h.open_loop(e, cfg, policy, normalizer, preds)
            assert r["cmd_bitwise"] and r["pred_bitwise"] and not r["mismatch"], (cfg, e.condition)
            alarms += r["alarms"] if cfg == "C" else 0
    assert alarms > 0  # the supervisor's switching path was exercised


@needs_data
def test_deterministic_replay_and_timing_fields(replay, tmp_path):
    from farc.hardware.controller import HardwareController
    from farc.hardware.pipeline import HardwarePipeline

    _, policy, normalizer, preds, eps = replay
    e = eps[0]
    runs = []
    for k in range(2):
        pipe = HardwarePipeline(ObservationAdapter(replay_config()),
                                HardwareController("C", policy, normalizer, preds),
                                CommandFIFO(schedule=DelaySchedule(((0, None, 3),))))  # fmt: skip
        pipe.reset()
        with StepLogger(tmp_path / f"run{k}.jsonl", {"episode_seed": e.episode_seed}) as log:
            recs = []
            for t in range(len(e.view)):
                rec = pipe.cycle(raw_from_sim_observation(e.view[t], (t + 1) * P), e.goal).record
                log.write(rec)
                recs.append(rec)
        assert verify_log(tmp_path / f"run{k}.jsonl") == len(e.view)
        for rec in recs:
            parts = [rec[k] for k in ("lat_base_policy_ns", "lat_estimator_observe_ns",
                                      "lat_identification_supervision_ns", "lat_predictor_ns",
                                      "lat_compensated_policy_ns")]  # fmt: skip
            assert all(isinstance(x, int) and x >= 0 for x in parts)
            assert sum(parts) <= rec["lat_controller_total_ns"] <= rec["lat_pipeline_total_ns"]
        runs.append([{k: v for k, v in r.items() if not k.startswith("lat_") and k != "mono_ns"}
                     for r in recs])  # fmt: skip
    assert runs[0] == runs[1]


@needs_data
def test_pipeline_safety_stop_on_missing_observation(replay):
    from farc.hardware.controller import HardwareController
    from farc.hardware.pipeline import HardwarePipeline

    _, policy, normalizer, preds, eps = replay
    e = eps[0]
    pipe = HardwarePipeline(ObservationAdapter(replay_config()),
                            HardwareController("CA", policy, normalizer, preds),
                            CommandFIFO())  # fmt: skip
    pipe.reset()
    pipe.cycle(raw_from_sim_observation(e.view[0], P), e.goal)
    out = pipe.cycle(None, e.goal)
    assert out.record["safety_event"] == "observation_missing" and out.executed is None
    np.testing.assert_array_equal(out.command, np.zeros(4))
    later = pipe.cycle(raw_from_sim_observation(e.view[1], 3 * P), e.goal)
    assert later.record["safety_event"] == "observation_missing" and later.executed is None
    assert pipe.controller.t == 1  # the controller never stepped after the stop
