# Migrated from m-alan08/quantrl-fetch tests/test_detector.py
# (introduced in 934063d, last changed in 934063d). See PROVENANCE.md.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farc.detection.progress_deficit import (
    DetectorConfig,
    decision_goals,
    kinematically_infeasible,
    run_detector,
    segment_starts,
)
from farc.detection.progress_deficit_eval import (
    DetectorConfigError,
    _rates,
    evaluate_episode,
    load_detector_experiment,
    load_trajectories,
)
from farc.paths import data_dir

T = 50
GOAL = np.array([1.0, 0.0, 0.0])
REPO = Path(__file__).resolve().parents[1]
SOURCE = data_dir() / "experiments" / "phase1_extended"


def path_along_x(positions: np.ndarray, goals: np.ndarray | None = None):
    """Episode where the achieved goal moves along x; positions has T + 1 entries."""
    achieved = np.zeros((T + 1, 3))
    achieved[:, 0] = positions
    if goals is None:
        goals = np.tile(GOAL, (T + 1, 1))
    return achieved, goals


def test_fast_approach_never_warns():
    # covers 1 m in 10 steps, then holds inside the success radius
    achieved, goals = path_along_x(np.minimum(np.arange(T + 1) * 0.1, 1.0))
    trace = run_detector(achieved, goals, DetectorConfig(3, 2))
    assert trace.first_warning is None and not trace.flagged.any()


def test_stalled_far_from_goal_warns_after_window_plus_persistence():
    achieved, goals = path_along_x(np.zeros(T + 1))
    for k, p in [(3, 2), (5, 3), (10, 5)]:
        trace = run_detector(achieved, goals, DetectorConfig(k, p))
        assert np.isnan(trace.deficit[:k]).all()  # not enough history
        assert trace.flagged[k:].all()
        assert trace.first_warning == k + p - 1


def test_required_and_observed_progress_values():
    positions = np.zeros(T + 1)
    positions[1:] = 0.01 * np.arange(1, T + 1)  # 1 cm per step
    achieved, goals = path_along_x(positions)
    trace = run_detector(achieved, goals, DetectorConfig(5, 1))
    t = 20
    distance = 1.0 - 0.2
    assert np.isclose(trace.observed[t], 0.05)
    assert np.isclose(trace.required[t], 5 * (distance - 0.05) / (T - t))
    assert np.isclose(trace.deficit[t], trace.required[t] - trace.observed[t])
    # 1 cm per step needs (0.75 / 30) * 5 = 0.125 m per 5 steps: too slow, so flagged
    assert trace.flagged[t]


def test_constant_rate_is_judged_against_the_deadline():
    # 2 cm per step reaches the 5 cm radius (0.95 m) by step 47.5: always on schedule
    on_time, goals = path_along_x(0.02 * np.arange(T + 1))
    trace = run_detector(on_time, goals, DetectorConfig(5, 1))
    assert not trace.flagged.any()
    # 1.8 cm per step ends 0.1 m short: behind schedule at every available state
    late, goals = path_along_x(0.018 * np.arange(T + 1))
    trace = run_detector(late, goals, DetectorConfig(5, 1))
    assert trace.flagged[5:].all() and trace.first_warning == 5


def test_suppressed_inside_success_radius():
    positions = np.full(T + 1, 0.97)  # 3 cm from the goal and not moving
    achieved, goals = path_along_x(positions)
    trace = run_detector(achieved, goals, DetectorConfig(3, 2))
    assert trace.first_warning is None and not trace.flagged.any()


def test_persistence_requires_consecutive_flags():
    # alternate stalling and big jumps so flags never hold for two consecutive states
    positions = np.zeros(T + 1)
    for t in range(1, T + 1):
        positions[t] = positions[t - 1] + (0.06 if t % 2 else 0.0)
    positions = np.minimum(positions, 0.4)
    achieved, goals = path_along_x(positions)
    trace = run_detector(achieved, goals, DetectorConfig(1, 2))
    # the cap at 0.4 m is reached at state 13; before it, flags alternate
    assert trace.flagged[1:13].any() and not trace.flagged[1:13].all()
    run = 0
    for flag in trace.flagged[:13]:
        run = run + 1 if flag else 0
        assert run < 2
    # after the cap the arm stalls, flags become consecutive, and the warning follows
    assert trace.first_warning is not None and trace.first_warning >= 13


def test_goal_shift_resets_window_and_uses_new_goal():
    onset = 25
    achieved, goals = path_along_x(np.minimum(np.arange(T + 1) * 0.1, 1.0))
    new_goal = np.array([1.0, 0.5, 0.0])
    goals[onset + 1 :] = new_goal  # logged from the state after the shift decision
    decision = decision_goals(goals)
    assert np.allclose(decision[onset - 1], GOAL) and np.allclose(decision[onset], new_goal)
    assert segment_starts(decision)[onset] == onset
    k = 3
    trace = run_detector(achieved, goals, DetectorConfig(k, 2))
    # decision distance at the shift state is to the new goal, not the old one
    assert np.isclose(trace.decision_distance[onset], 0.5)
    # the window does not reach across the shift
    assert np.isnan(trace.deficit[onset : onset + k]).all()
    assert not np.isnan(trace.deficit[onset + k])
    # the arm never moves toward the new goal: warning k + p - 1 states after the shift
    assert trace.first_warning == onset + k + 1


def test_goal_shift_resets_persistence_count():
    achieved, goals = path_along_x(np.zeros(T + 1))  # stalled from the start
    goals[13:] = np.array([0.0, 1.0, 0.0])  # shift decided at state 12
    trace = run_detector(achieved, goals, DetectorConfig(3, 5))
    # before the shift the count reaches 3, 4 (states 3..11 give 9 flags -> warning at 7)
    assert trace.first_warning == 7
    late = run_detector(achieved, goals, DetectorConfig(3, 12))
    # persistence 12 would be reached at state 14 without a reset; the shift at 12 resets it,
    # and the window is unavailable until 15, so the warning comes at 15 + 11
    assert late.first_warning == 26


def test_kinematic_infeasibility_bound():
    assert not kinematically_infeasible(0.0, 1)
    assert not kinematically_infeasible(np.sqrt(3) * 0.05 * 4 - 1e-6, 4)
    assert kinematically_infeasible(np.sqrt(3) * 0.05 * 4 + 1e-6, 4)


def test_evaluate_episode_fields():
    achieved, goals = path_along_x(np.zeros(T + 1))
    arrays = {"achieved": achieved, "goals": goals, "actions": np.full((T, 3), 0.5)}
    row = evaluate_episode(arrays, DetectorConfig(3, 2), onset=10, has_object=True)
    assert row["warned"] and row["warning_state"] == 4
    assert row["lead_time"] == 46 and row["warning_minus_onset"] == -6
    assert row["warned_before_onset"] is True
    assert row["object_unmoved_at_warning"] is True
    assert row["gripper_commanded_in_window"] is True
    reach = evaluate_episode(arrays, DetectorConfig(3, 2), onset=None, has_object=False)
    assert reach["object_unmoved_at_warning"] is None and reach["warned_before_onset"] is None


def test_rates_count_detections_and_false_positives():
    frame = pd.DataFrame(
        {
            "failure": [True, True, True, False, False],
            "warned": [True, False, True, True, False],
            "lead_time": [10, None, 3, 40, None],
            "warning_state": [40, None, 47, 10, None],
            "infeasible_at_warning": [False, None, True, False, None],
            "warned_before_onset": [False, None, False, True, None],
            "object_unmoved_at_warning": [None] * 5,
        }
    )
    stats = _rates(frame, 5)
    assert stats["detection_rate"] == 2 / 3 and stats["missed"] == 1
    assert stats["false_positive_rate"] == 0.5
    assert stats["median_lead_time"] == 6.5 and stats["detected_lead_le_5"] == 1
    assert stats["detected_kinematically_infeasible"] == 1
    assert stats["false_positives_before_onset"] == 1


def test_config_validation(tmp_path):
    config = load_detector_experiment(REPO / "configs/detectors/progress_deficit_grid.json")
    assert [c.label for c in config.configs()][:2] == ["k3_p2_th0", "k3_p3_th0"]
    assert len(config.configs()) == 9
    values = json.loads((REPO / "configs/detectors/progress_deficit_grid.json").read_text())
    (tmp_path / "bad.json").write_text(json.dumps({**values, "windows": [0]}))
    with pytest.raises(DetectorConfigError):
        load_detector_experiment(tmp_path / "bad.json")
    (tmp_path / "typo.json").write_text(json.dumps({**values, "window": [3]}))
    with pytest.raises(DetectorConfigError, match="unknown"):
        load_detector_experiment(tmp_path / "typo.json")


@pytest.mark.skipif(not (SOURCE / "parts").exists(), reason="phase1_extended outputs not present")
def test_detector_distances_match_logged_distances(tmp_path):
    part = SOURCE / "parts/FetchPush-v4_seed0/goal_shift_sev0.1_on40_durinf"
    link = tmp_path / "parts/FetchPush-v4_seed0"
    link.mkdir(parents=True)
    (link / part.name).symlink_to(part)
    episodes = load_trajectories(tmp_path)
    assert len(episodes) == 100
    logged = pd.read_csv(part / "trajectories.csv.gz")
    for (_, _, _, seed), arrays in list(episodes.items())[:10]:
        trace = run_detector(arrays["achieved"], arrays["goals"], DetectorConfig(3, 2))
        distance = logged[logged["episode_seed"] == seed]["distance"].to_numpy()
        # identical except the shift state, which is logged against the old goal
        same = np.r_[np.arange(40), np.arange(41, T)]
        assert np.allclose(trace.decision_distance[same], distance[same])
        assert not np.isclose(trace.decision_distance[40], distance[40])
        assert np.isclose(
            trace.decision_distance[40],
            np.linalg.norm(arrays["achieved"][40] - arrays["goals"][41]),
        )
