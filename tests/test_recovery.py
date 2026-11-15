from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from farc.detection.phase_aware import PhaseDetectorConfig, run_phase_detector
from farc.detection.threshold_sweep import run_minimum
from farc.paths import checkpoint_path
from farc.prediction.dataset import Normalizer
from farc.prediction.evaluation import FailurePredictor
from farc.prediction.features import NUM_FEATURES, episode_features
from farc.recovery.controller import (
    HOLD,
    PUSH,
    REPOSITION,
    InterventionSwitch,
    LearnedTrigger,
    OnlineHistory,
    PhaseAwareTrigger,
    RecoveryController,
    SwitchConfig,
)
from farc.recovery.experiment import (
    RecoveryConfigError,
    cluster_bootstrap_difference,
    load_recovery_experiment,
    paired,
    summary,
)
from farc.recovery.rollout import CONFIGURATIONS

T = 50
Z = 0.425


# --- recovery controller ------------------------------------------------------------------


def test_hold_at_goal():
    action, mode = RecoveryController().act([1.2, 0.7, Z], [1.3, 0.7, Z], [1.305, 0.7, Z])
    assert mode == HOLD and np.all(action == 0)


def test_push_when_behind_and_stop_at_goal():
    c = RecoveryController()
    action, mode = c.act([1.24, 0.7, Z], [1.30, 0.7, Z], [1.50, 0.7, Z])
    assert mode == PUSH and action[0] > 0 and abs(action[1]) < 1e-12 and action[3] == 0
    # near the goal the gripper target advances by only the remaining distance
    near, _ = c.act([1.44, 0.7, Z], [1.48, 0.7, Z], [1.50, 0.7, Z])
    assert near[0] == pytest.approx(0.02 / 0.05)


def test_lift_before_crossing_and_descend_behind():
    c = RecoveryController()
    goal, obj = np.array([1.5, 0.7, Z]), np.array([1.3, 0.7, Z])
    action, mode = c.act([1.36, 0.7, Z], obj, goal)  # gripper on the goal side of the object
    assert mode == REPOSITION and action[2] > 0 and np.allclose(action[:2], 0)
    high = Z + 0.07
    action, _ = c.act([1.36, 0.7, high], obj, goal)  # lifted: travel to the pre-push point
    assert action[0] < 0 and abs(action[2]) < 1e-12
    action, _ = c.act([1.24, 0.7, high], obj, goal)  # above the pre-push point: descend
    assert action[2] < 0 and np.allclose(action[:2], 0)


def test_controller_is_deterministic_and_bounded():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g, o, G = (rng.uniform([1.0, 0.4, 0.4], [1.6, 1.0, 0.6]) for _ in range(3))
        a1, m1 = RecoveryController().act(g, o, G)
        a2, m2 = RecoveryController().act(g, o, G)
        assert m1 == m2 and np.array_equal(a1, a2) and np.all(np.abs(a1) <= 1)


# --- online feature timing ----------------------------------------------------------------


def synthetic(shift_at: int = 25):
    rng = np.random.default_rng(3)
    objects = 1.3 + 0.01 * rng.standard_normal((T + 1, 3)).cumsum(axis=0)
    grippers = 1.25 + 0.01 * rng.standard_normal((T + 1, 3)).cumsum(axis=0)
    decision_goal = np.tile([1.45, 0.75, Z], (T, 1))
    decision_goal[shift_at:] = [1.35, 0.85, Z]  # the controller sees the new goal at shift_at
    logged_goals = np.vstack([decision_goal[:1], decision_goal])  # appears in row shift_at+1
    base = rng.uniform(-1, 1, (T, 4))
    return objects, grippers, decision_goal, logged_goals, base


def online_history(objects, grippers, decision_goal, base, upto):
    h = OnlineHistory(T)
    for t in range(upto + 1):
        h.record(t, grippers[t], objects[t], decision_goal[t], base[t - 1] if t else None)
    return h


@pytest.mark.parametrize("t", [0, 1, 24, 25, 26, 30, 49])
def test_online_features_equal_offline_and_ignore_the_future(t):
    objects, grippers, decision_goal, logged, base = synthetic()
    offline = episode_features(objects, grippers, logged, base)
    h = online_history(objects, grippers, decision_goal, base, t)
    trigger = LearnedTrigger(FailurePredictor(), Normalizer(np.zeros(24), np.ones(24)), 0.5)
    np.testing.assert_allclose(trigger.features(h), offline[t], atol=1e-12)
    # the history holds nothing from after t (padded with the latest values)
    assert np.all(h.objects[t:] == objects[t]) and np.all(h.goals[t + 1 :] == decision_goal[t])


@pytest.mark.parametrize("t", [4, 10, 25, 31, 49])
def test_online_phase_score_equals_offline(t):
    objects, grippers, decision_goal, logged, base = synthetic()
    config = PhaseDetectorConfig(5, 5, 3, None)
    offline = run_minimum(run_phase_detector(grippers, objects, logged, config).deficit, 5)
    h = online_history(objects, grippers, decision_goal, base, t)
    score = PhaseAwareTrigger(config, 0.0).score(h)
    assert score == offline[t] or (np.isneginf(score) and np.isneginf(offline[t]))


def test_learned_trigger_score_matches_model():
    torch.manual_seed(0)
    model = FailurePredictor().eval()
    objects, grippers, decision_goal, _, base = synthetic()
    h = online_history(objects, grippers, decision_goal, base, 10)
    norm = Normalizer(np.full(NUM_FEATURES, 0.1), np.full(NUM_FEATURES, 2.0))
    trigger = LearnedTrigger(model, norm, 0.5)
    x = torch.tensor(norm(trigger.features(h)), dtype=torch.float32)
    assert trigger.score(h) == pytest.approx(torch.sigmoid(model(x[None]))[0].item(), rel=1e-6)


def test_history_must_be_recorded_in_order():
    h = OnlineHistory(T)
    h.record(0, np.zeros(3), np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError):
        h.record(2, np.zeros(3), np.zeros(3), np.zeros(3))


# --- intervention switching ---------------------------------------------------------------


def drive(switch, fires, distances):
    acts = []
    for t in range(len(fires)):
        acts.append(switch.step(t, len(fires), fires[t], distances[t])[0])
    return np.array(acts)


def test_timeout_cooldown_and_refire():
    s = InterventionSwitch(SwitchConfig(max_duration=3, cooldown=2, exit_steps=2))
    fires = [False, True, True, True, True, True, True, True, True, True]
    acts = drive(s, fires, [1.0] * 10)
    # active 1..3 (3 steps), timeout at 4, cooldown 4..5, refire at 6..8, timeout at 9
    np.testing.assert_array_equal(acts, [0, 1, 1, 1, 0, 0, 1, 1, 1, 0])
    assert [i["reason"] for i in s.interventions] == ["timeout", "timeout"]
    assert [(i["start"], i["end"]) for i in s.interventions] == [(1, 4), (6, 9)]


def test_success_exit_returns_to_base():
    s = InterventionSwitch(SwitchConfig(max_duration=20, exit_radius=0.03, exit_steps=2))
    distances = [0.2, 0.2, 0.1, 0.02, 0.02, 0.02, 0.02]
    acts = drive(s, [True] + [False] * 6, distances)
    np.testing.assert_array_equal(acts, [1, 1, 1, 1, 0, 0, 0])
    assert s.interventions == [{"start": 0, "end": 4, "reason": "success"}]


def test_no_activation_without_time_left_and_episode_end_accounting():
    s = InterventionSwitch(SwitchConfig(min_steps_left=3))
    acts = drive(s, [False] * 7 + [True] * 3, [1.0] * 10)
    assert not acts.any() and s.interventions == []
    s = InterventionSwitch(SwitchConfig(max_duration=50))
    drive(s, [False] * 5 + [True] + [False] * 4, [1.0] * 10)
    assert s.interventions == [{"start": 5, "end": 10, "reason": "episode_end"}]


def test_always_on_and_reset():
    s = InterventionSwitch(always_on=True)
    assert drive(s, [False] * 5, [0.0] * 5).all() and len(s.interventions) == 1
    s.reset()
    assert s.interventions == [] and not s.active


# --- episode accounting -------------------------------------------------------------------


def frame():
    rows = []
    outcomes = {  # (base, always, reactive, learned) success per episode, activation of each
        0: ((1, 1, 1, 0), (0, 1, 0, 1)),
        1: ((0, 1, 0, 1), (0, 1, 1, 1)),
        2: ((0, 0, 0, 0), (0, 1, 0, 0)),
        3: ((1, 0, 1, 1), (0, 1, 1, 0)),
    }
    for seed, (success, activated) in outcomes.items():
        for kind, s, a in zip(CONFIGURATIONS, success, activated, strict=True):
            rows.append({"configuration": kind, "policy_seed": 0, "condition": "c",
                         "episode_seed": seed, "final_success": bool(s), "activated": bool(a),
                         "interventions": int(a), "recovery_steps": 5 * a,
                         "first_activation_step": 3.0 if a else np.nan, "exits_success": 0,
                         "exits_timeout": a, "exits_episode_end": 0})  # fmt: skip
    return pd.DataFrame(rows)


def test_summary_counts():
    data = paired(frame())
    learned = summary(data[data["configuration"] == "learned_triggered"])
    assert learned["success_rate"] == 0.5 and learned["base_success_rate"] == 0.5
    assert learned["recovered"] == 1 and learned["harmed"] == 1
    assert learned["activated_on_base_success"] == 1 and learned["activated_on_base_failure"] == 1
    assert learned["unnecessary_intervention_rate"] == 0.5
    reactive = summary(data[data["configuration"] == "reactive_triggered"])
    assert reactive["recovered"] == 0 and reactive["harmed"] == 0
    assert reactive["activated_on_base_success"] == 1


def test_pairing_requires_every_configuration():
    broken = frame()
    broken = broken[~((broken["episode_seed"] == 2) & (broken["configuration"] == "always_on"))]
    with pytest.raises(ValueError):
        paired(broken)


def test_cluster_bootstrap_difference():
    data = paired(frame())
    est, low, high = cluster_bootstrap_difference(
        data, "always_on", "base_only", 500, np.random.default_rng(0)
    )
    assert est == 0.0 and low <= est <= high
    est, low, high = cluster_bootstrap_difference(
        data, "base_only", "base_only", 100, np.random.default_rng(0)
    )
    assert est == low == high == 0.0


def test_config_refuses_used_and_heldout_seeds(tmp_path):
    base = json.loads(Path("configs/recovery/phase_c_development.json").read_text())
    config = load_recovery_experiment("configs/recovery/phase_c_development.json")
    assert config.episode_seeds[0] == 3_610_000 and asdict(config)["episodes"] == 100
    for offset in (3_500_050, 3_600_000, 4_000_000):
        path = tmp_path / f"{offset}.json"
        path.write_text(json.dumps({**base, "episode_seed_offset": offset}))
        with pytest.raises(RecoveryConfigError):
            load_recovery_experiment(path)


# --- integration with the frozen policy -----------------------------------------------------

PUSH_CHECKPOINT = checkpoint_path("fetch_push_fp32_seed0")


@pytest.mark.skipif(not PUSH_CHECKPOINT.exists(), reason="FetchPush checkpoint absent")
@pytest.mark.parametrize(
    "condition_name",
    [
        "action_delay_sev3_on0_durinf",
        "goal_shift_sev0.1_on25_durinf",
        "observation_noise_sev0.01_on0_durinf",
    ],  # fmt: skip
)
def test_base_only_reproduces_phase1_rollout(condition_name):
    from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
    from farc.disturbances.config import load_experiment
    from farc.disturbances.rollout import FrozenNormalizer, run_disturbed_episode
    from farc.envs.factory import make_env
    from farc.recovery.rollout import run_closed_loop_episode

    conditions = {c.name: c for c in load_experiment(
        "configs/disturbances/phase1_extended.json").conditions}  # fmt: skip
    loaded = load_policy(PUSH_CHECKPOINT)
    policy = deterministic_policy_function(loaded.actor)
    env = make_env(loaded.env_config)
    frozen = FrozenNormalizer(loaded.normalizer)
    seed = 3_000_123  # pilot range
    reference = run_disturbed_episode(env, policy, frozen, conditions[condition_name], seed, 7)
    closed = run_closed_loop_episode(env, policy, frozen, conditions[condition_name], seed, 7,
                                     None, None, None)  # fmt: skip
    np.testing.assert_array_equal(closed.executed_actions, reference.executed_actions)
    np.testing.assert_array_equal(closed.successes, reference.successes)
    assert not closed.recovery_active.any()
