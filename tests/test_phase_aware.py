# Migrated from m-alan08/quantrl-fetch tests/test_phase_detector.py
# (introduced in 98c903e, last changed in ff20609). See PROVENANCE.md.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from farc.detection.phase_aware import (
    APPROACH,
    MAINTAIN,
    MANIPULATE,
    PhaseDetectorConfig,
    contact_phases,
    prepush_point,
    remaining_path,
    run_phase_detector,
)
from farc.detection.phase_aware_eval import (
    PhaseEvalConfigError,
    episode_context,
    load_phase_evaluation,
    warning_row,
)

T = 50
REPO = Path(__file__).resolve().parents[1]
GOAL = np.array([1.5, 0.0, 0.42])
START = np.array([1.2, 0.0, 0.42])  # object 0.30 m from the goal


def episode(gripper_x, object_x, goal=GOAL):
    """Gripper and object move along x at table height; the goal is fixed unless given per
    state."""
    gripper = np.tile(START, (T + 1, 1))
    gripper[:, 0] = gripper_x
    objects = np.tile(START, (T + 1, 1))
    objects[:, 0] = object_x
    goals = np.tile(goal, (T + 1, 1)) if np.ndim(goal) == 1 else goal
    return gripper, objects, goals


def pushed_episode(push_start=5, speed=0.02):
    """Gripper starts 0.2 m behind the object, reaches the pre-push point, then pushes."""
    t = np.arange(T + 1)
    object_x = START[0] + np.clip(t - push_start, 0, None) * speed
    object_x = np.minimum(object_x, GOAL[0] - 0.01)
    approach = np.minimum(START[0] - 0.20 + t * 0.04, START[0] - 0.05)
    gripper_x = np.where(t < push_start, approach, object_x - 0.05)
    return episode(gripper_x, object_x)


# ---------------------------------------------------------------- geometry and phases


def test_prepush_point_is_behind_the_object_at_object_height():
    point = prepush_point(START, GOAL, 0.05)
    assert np.allclose(point, START - np.array([0.05, 0.0, 0.0]))
    raised_goal = GOAL + np.array([0.0, 0.0, 0.3])
    assert np.isclose(prepush_point(START, raised_goal, 0.05)[2], START[2])
    assert np.allclose(prepush_point(GOAL, GOAL, 0.05), GOAL)


def test_remaining_path_by_phase():
    config = PhaseDetectorConfig(5, 3)
    gripper = START - np.array([0.25, 0.0, 0.0])
    assert np.isclose(remaining_path(gripper, START, GOAL, MANIPULATE, config), 0.25)
    assert np.isclose(remaining_path(gripper, START, GOAL, APPROACH, config), 0.20 + 0.25)
    with pytest.raises(ValueError):
        remaining_path(gripper, START, GOAL, MAINTAIN, config)


def test_contact_hysteresis_and_return_to_approach():
    distances = np.array([0.2, 0.05, 0.2, 0.05, 0.05, 0.05, 0.08, 0.08, 0.1, 0.1, 0.1, 0.1])
    no_hysteresis = contact_phases(distances, PhaseDetectorConfig(1, 1, hysteresis=1))
    assert no_hysteresis[:4] == [APPROACH, MANIPULATE, APPROACH, MANIPULATE]
    assert no_hysteresis[6:8] == [MANIPULATE, MANIPULATE]  # 8 cm is between enter and exit
    assert no_hysteresis[8] == APPROACH  # past the 9 cm exit
    three = contact_phases(distances, PhaseDetectorConfig(1, 1, hysteresis=3))
    assert three[:5] == [APPROACH] * 5  # one close state is not enough
    assert three[5] == MANIPULATE  # third consecutive close state
    assert three[8:10] == [MANIPULATE, MANIPULATE]
    assert three[10] == APPROACH  # third consecutive far state


# ---------------------------------------------------------------- warnings


def test_on_time_push_does_not_warn_although_the_object_is_still_at_first():
    gripper, objects, goals = pushed_episode(push_start=5)
    for k, p in [(5, 3), (10, 5)]:
        trace = run_phase_detector(gripper, objects, goals, PhaseDetectorConfig(k, p))
        assert trace.first_warning is None, (k, p, trace.first_warning)
        assert trace.phases[0] == APPROACH and MANIPULATE in trace.phases


def test_stalled_approach_far_from_the_object_still_warns():
    gripper_x = np.full(T + 1, START[0] - 0.4)  # gripper never moves
    gripper, objects, goals = episode(gripper_x, np.full(T + 1, START[0]))
    trace = run_phase_detector(gripper, objects, goals, PhaseDetectorConfig(5, 3))
    assert trace.first_warning == 5 + 3 - 1
    assert trace.warning_kind == "deficit" and trace.warning_phase == APPROACH


def test_stalled_object_in_contact_warns_in_manipulation():
    gripper_x = np.full(T + 1, START[0] - 0.04)
    gripper, objects, goals = episode(gripper_x, np.full(T + 1, START[0]))
    trace = run_phase_detector(gripper, objects, goals, PhaseDetectorConfig(5, 3, hysteresis=1))
    assert set(trace.phases) == {MANIPULATE}
    assert trace.first_warning == 7 and trace.warning_phase == MANIPULATE


def test_window_resets_at_phase_changes():
    gripper, objects, goals = pushed_episode(push_start=5)
    config = PhaseDetectorConfig(5, 3, hysteresis=1)
    trace = run_phase_detector(gripper, objects, goals, config)
    change = trace.phases.index(MANIPULATE)
    assert np.isnan(trace.deficit[change : change + 5]).all()
    assert not np.isnan(trace.deficit[change + 5])


def test_maintenance_departure_warning_and_its_ablation():
    t = np.arange(T + 1)
    # object pushed into the radius by state 15, then knocked out from state 30 on
    object_x = np.where(t < 15, START[0] + t * 0.02, GOAL[0] - 0.01)
    object_x = np.where(t >= 30, GOAL[0] - 0.09, object_x)
    gripper, objects, goals = episode(object_x - 0.05, object_x)
    with_monitor = run_phase_detector(
        gripper, objects, goals, PhaseDetectorConfig(10, 5, departure_persistence=3)
    )
    assert with_monitor.phases[20] == MAINTAIN and with_monitor.phases[30] != MAINTAIN
    assert with_monitor.departure_count[29] == 0 and with_monitor.departure_count[32] == 3
    assert with_monitor.first_warning == 32 and with_monitor.warning_kind == "departure"
    without = run_phase_detector(
        gripper, objects, goals, PhaseDetectorConfig(10, 5, departure_persistence=None)
    )
    assert without.warning_kind != "departure"
    assert without.first_warning is None or without.first_warning > 32


def test_goal_shift_clears_success_history_and_window():
    t = np.arange(T + 1)
    object_x = np.where(t < 15, START[0] + t * 0.02, GOAL[0] - 0.01)
    goals = np.tile(GOAL, (T + 1, 1))
    new_goal = GOAL + np.array([0.0, 0.2, 0.0])
    goals[26:] = new_goal  # shift decided at state 25
    gripper, objects, goals = episode(object_x - 0.05, object_x, goal=goals)
    trace = run_phase_detector(
        gripper, objects, goals, PhaseDetectorConfig(5, 3, departure_persistence=1)
    )
    # outside the new goal's radius, but never inside it: no departure warning
    assert trace.departure_count[25:].max() == 0
    assert np.isnan(trace.deficit[25:30]).all()
    assert trace.warning_kind == "deficit"
    assert trace.first_warning is not None and trace.first_warning >= 25 + 5


def test_config_validation():
    with pytest.raises(ValueError):
        PhaseDetectorConfig(5, 3, contact_enter=0.1, contact_exit=0.05)
    with pytest.raises(ValueError):
        PhaseDetectorConfig(5, 3, hysteresis=0)
    assert PhaseDetectorConfig(10, 5, 3, None).label == "phase_k10_p5_h3_moff_th0"
    evaluation = load_phase_evaluation(REPO / "configs/detectors/phase_aware_push.json")
    assert len(evaluation.phase_configs()) == 16 and len(evaluation.baselines()) == 4
    values = json.loads((REPO / "configs/detectors/phase_aware_push.json").read_text())
    values["typo"] = 1
    bad = REPO / "tests" / "_bad_phase_config.json"
    try:
        bad.write_text(json.dumps(values))
        with pytest.raises(PhaseEvalConfigError):
            load_phase_evaluation(bad)
    finally:
        bad.unlink(missing_ok=True)


def test_context_and_warning_row_timing():
    gripper, objects, goals = pushed_episode(push_start=5)
    evaluation = load_phase_evaluation(REPO / "configs/detectors/phase_aware_push.json")
    arrays = {"gripper": gripper, "objects": objects, "goals": goals}
    context = episode_context(arrays, evaluation)
    assert context["first_object_move_state"] == 6
    distance = np.linalg.norm(objects - goals, axis=1)
    assert context["first_success_state"] == int(np.flatnonzero(distance[1:] < 0.05)[0]) + 1
    row = warning_row("x", "phase_aware", 3, context, evaluation, 0.01, onset=10, extra={})
    assert row["lead_time"] == 47 and row["warning_minus_first_move"] == -3
    assert row["warned_before_object_moved"] and row["warned_before_onset"]
    assert row["reference_phase_at_warning"] == APPROACH
    # 0.25 m to go in 47 steps needs 5.3 mm per step: on time against a 10 mm reference...
    assert not row["late_by_object_speed"] and not row["late_by_lead"]
    # ...and late against a 5 mm reference
    late = warning_row("x", "phase_aware", 3, context, evaluation, 0.005, onset=10, extra={})
    assert late["late_by_object_speed"]
    quiet = warning_row("x", "phase_aware", None, context, evaluation, 0.01, None, {})
    assert quiet["lead_time"] is None and quiet["warned"] is False


def test_compare_detectors_pairs_episodes():
    import pandas as pd

    from farc.detection.phase_aware_eval import compare_detectors

    rows = []
    for seed in (0, 1):
        for episode, (failure, a, b) in enumerate(
            [(True, True, False), (True, True, True), (False, False, True), (False, False, False)]
        ):
            for detector, warned in (("A", a), ("B", b)):
                rows.append(
                    {
                        "env_id": "e",
                        "training_seed": seed,
                        "condition": "c",
                        "episode_seed": episode,
                        "disturbance_type": "x",
                        "failure": failure,
                        "detector": detector,
                        "warned": warned,
                    }
                )
    row = compare_detectors(pd.DataFrame(rows), "A", "B", 200, 0)
    assert row["detection_first"] == 1.0 and row["detection_second"] == 0.5
    assert row["detection_difference"] == 0.5 and row["detection_only_first"] == 2
    assert row["fpr_difference"] == -0.5 and row["fpr_only_second"] == 2
    with pytest.raises(KeyError):
        compare_detectors(pd.DataFrame(rows), "A", "missing", 10, 0)
    frame = pd.DataFrame(rows)
    frame["lead_time"] = [3.0 if d == "A" else 20.0 for d in frame["detector"]]
    late = compare_detectors(frame, "A", "B", 200, 0, min_lead=5)
    assert late["detection_first"] == 0.0 and late["detection_second"] == 0.5
