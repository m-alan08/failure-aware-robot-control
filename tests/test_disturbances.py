# Migrated from m-alan08/quantrl-fetch tests/test_disturbances.py
# (introduced in 7621e7c, last changed in 09ea4f4). See PROVENANCE.md.
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from farc.base_policy.normalizer import GoalObservationNormalizer
from farc.disturbances.config import (
    ConditionSpec,
    DisturbanceConfigError,
    DisturbanceSpec,
    expand_condition_entry,
    load_experiment,
    validate_disturbance,
)
from farc.disturbances.experiment import episode_row, summarize
from farc.disturbances.metrics import (
    first_success_step,
    kaplan_meier_median,
    recovery_outcome,
)
from farc.disturbances.perturbations import (
    PerturbationEffectError,
    perturbation_rng,
    sample_goal_with_rng,
)
from farc.disturbances.rollout import FrozenNormalizer, run_disturbed_episode
from farc.envs.factory import EnvConfig, make_env
from farc.paths import checkpoint_path

REPO = Path(__file__).resolve().parents[1]
SEED = 3_000_000
PSEED = 7


def proportional_policy(inputs: np.ndarray) -> np.ndarray:
    """Scripted FetchReach controller on an identity normalizer: inputs = [obs(10), goal(3)],
    obs[:3] is the gripper position. Moves toward the goal it is shown."""
    gripper = inputs[:, :3]
    goal = inputs[:, -3:]
    action = np.zeros((inputs.shape[0], 4))
    action[:, :3] = np.clip(10.0 * (goal - gripper), -1.0, 1.0)
    return action


@pytest.fixture(scope="module")
def reach_env():
    env = make_env(EnvConfig(env_id="FetchReach-v4"))
    yield env
    env.close()


@pytest.fixture()
def identity_normalizer():
    return FrozenNormalizer(GoalObservationNormalizer(10, 3, eps=1.0, normalized_clip=200.0))


def condition(*specs: DisturbanceSpec) -> ConditionSpec:
    return ConditionSpec("test", tuple(specs))


def rollout(env, normalizer, *specs, seed=SEED, pseed=PSEED, policy=proportional_policy):
    return run_disturbed_episode(env, policy, normalizer, condition(*specs), seed, pseed)


# ---------------------------------------------------------------- metrics


def test_first_success_step_is_one_indexed():
    assert first_success_step(np.array([0, 0, 1, 1], bool)) == 3
    assert first_success_step(np.zeros(4, bool)) is None


def test_recovery_counts_only_states_after_onset_and_censors():
    successes = np.array([1, 1, 0, 0, 0, 1, 1, 0], bool)
    outcome = recovery_outcome(successes, onset=2, horizon=8)
    assert outcome.recovered and outcome.recovery_time == 4  # first success at state 6
    assert not outcome.censored and outcome.censor_time == 6
    censored = recovery_outcome(np.array([1, 1, 0, 0], bool), onset=2, horizon=4)
    assert censored.censored and not censored.recovered
    assert censored.recovery_time is None and censored.censor_time == 2


def test_kaplan_meier_median_handles_censoring():
    assert kaplan_meier_median(np.array([1, 2, 3, 4]), np.array([1, 1, 1, 1], bool)) == 2.0
    # most of the sample censored: survival never reaches 0.5
    assert kaplan_meier_median(np.array([5, 9, 9, 9]), np.array([1, 0, 0, 0], bool)) is None
    # early censoring shrinks the risk set: at t=5 only {5, 9} are at risk, S drops to 0.5
    assert kaplan_meier_median(np.array([5, 1, 1, 9]), np.array([1, 0, 0, 0], bool)) == 5.0
    # censoring removes subjects from the risk set without counting them as events
    assert kaplan_meier_median(np.array([1, 2, 3, 4]), np.array([0, 1, 1, 1], bool)) == 3.0


# ---------------------------------------------------------------- config


def test_config_rejects_bad_disturbances():
    with pytest.raises(DisturbanceConfigError):
        validate_disturbance(DisturbanceSpec("action_scale", 1.5))
    with pytest.raises(DisturbanceConfigError):
        validate_disturbance(DisturbanceSpec("goal_shift", 0.1, onset=10, duration=5))
    with pytest.raises(DisturbanceConfigError):
        validate_disturbance(DisturbanceSpec("action_delay", 1.5))
    with pytest.raises(DisturbanceConfigError):
        validate_disturbance(DisturbanceSpec("observation_noise", 0.1, fields=("achieved_goal",)))
    with pytest.raises(DisturbanceConfigError):
        validate_disturbance(DisturbanceSpec("action_noise", 0.1, onset=50), horizon=50)
    with pytest.raises(DisturbanceConfigError, match="unknown keys"):
        expand_condition_entry({"type": "action_noise", "severity": 0.1, "sigma": 1}, "x")


def test_sweep_entry_expands_and_names_conditions():
    conditions = expand_condition_entry(
        {"type": "goal_shift", "severities": [0.05, 0.1], "onsets": [10, 25]}, "x"
    )
    assert [c.name for c in conditions] == [
        "goal_shift_sev0.05_on10_durinf",
        "goal_shift_sev0.05_on25_durinf",
        "goal_shift_sev0.1_on10_durinf",
        "goal_shift_sev0.1_on25_durinf",
    ]


def test_pilot_config_loads_and_keeps_seed_ranges_apart(tmp_path):
    config = load_experiment(REPO / "configs/disturbances/pilot.json")
    assert config.stage == "pilot" and config.episodes == 20
    assert {"undisturbed"} <= {c.name for c in config.conditions}
    types = {d.type for c in config.conditions for d in c.disturbances}
    assert types == {
        "action_scale",
        "action_noise",
        "observation_noise",
        "goal_shift",
        "action_delay",
    }
    # pilot seeds may not reach into the held-out range
    with pytest.raises(DisturbanceConfigError, match="pilot episode seeds"):
        load_experiment(
            REPO / "configs/disturbances/pilot.json", {"episode_seed_offset": 3_999_990}
        )
    with pytest.raises(DisturbanceConfigError, match="pilot episode seeds"):
        load_experiment(
            REPO / "configs/disturbances/pilot.json", {"episode_seed_offset": 4_000_000}
        )
    held = load_experiment(
        REPO / "configs/disturbances/pilot.json",
        {"stage": "heldout", "episode_seed_offset": 4_000_000},
    )
    assert held.episode_seeds[0] == 4_000_000
    bad = json.loads((REPO / "configs/disturbances/pilot.json").read_text())
    bad["typo"] = 1
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(DisturbanceConfigError, match="unknown keys"):
        load_experiment(tmp_path / "bad.json")


# ---------------------------------------------------------------- RNG


def test_perturbation_streams_are_independent_and_reproducible():
    a = perturbation_rng(PSEED, SEED, "action_noise", 0).standard_normal(5)
    assert np.array_equal(a, perturbation_rng(PSEED, SEED, "action_noise", 0).standard_normal(5))
    for other in (
        perturbation_rng(PSEED, SEED, "observation_noise", 0),
        perturbation_rng(PSEED, SEED + 1, "action_noise", 0),
        perturbation_rng(PSEED + 1, SEED, "action_noise", 0),
        perturbation_rng(PSEED, SEED, "action_noise", 1),
    ):
        assert not np.array_equal(a, other.standard_normal(5))


# ---------------------------------------------------------------- rollouts


def test_undisturbed_rollout_is_deterministic_and_sparse(reach_env, identity_normalizer):
    first = rollout(reach_env, identity_normalizer)
    second = rollout(reach_env, identity_normalizer)
    assert first.horizon == 50 and first.successes.shape == (50,)
    for name in ("achieved_goals", "desired_goals", "executed_actions", "rewards"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert np.array_equal(first.base_actions, first.executed_actions)
    assert set(np.unique(first.rewards)) <= {-1.0, 0.0}
    assert first.successes[-1]  # the scripted controller solves FetchReach


def test_disturbed_run_is_deterministic_and_same_reset(reach_env, identity_normalizer):
    noise = DisturbanceSpec("action_noise", 0.3)
    a = rollout(reach_env, identity_normalizer, noise)
    b = rollout(reach_env, identity_normalizer, noise)
    c = rollout(reach_env, identity_normalizer, noise, pseed=PSEED + 1)
    assert np.array_equal(a.executed_actions, b.executed_actions)
    assert not np.array_equal(a.executed_actions, c.executed_actions)
    # the perturbation seed never changes the episode's initial state or goal
    assert np.array_equal(a.achieved_goals[0], c.achieved_goals[0])
    assert np.array_equal(a.original_goal, c.original_goal)


def test_action_noise_draws_do_not_depend_on_other_perturbations(reach_env, identity_normalizer):
    noise = DisturbanceSpec("action_noise", 0.2, onset=0, duration=5)
    alone = rollout(reach_env, identity_normalizer, noise)
    together = rollout(
        reach_env, identity_normalizer, noise, DisturbanceSpec("observation_noise", 0.01)
    )
    # under observation noise the base action differs, but the added noise is the same draw
    assert np.allclose(
        alone.pre_clip_actions[:5] - alone.base_actions[:5],
        together.pre_clip_actions[:5] - together.base_actions[:5],
    )


def test_action_scale_timing_and_magnitude(reach_env, identity_normalizer):
    spec = DisturbanceSpec("action_scale", 0.5, onset=5, duration=3)
    trajectory = rollout(reach_env, identity_normalizer, spec)
    changed = np.any(trajectory.executed_actions != trajectory.base_actions, axis=1)
    assert np.flatnonzero(changed).tolist() == [5, 6, 7]
    assert np.allclose(trajectory.executed_actions[5:8], 0.5 * trajectory.base_actions[5:8])
    assert trajectory.active["action_scale"].tolist() == [5 <= t < 8 for t in range(50)]


def test_action_noise_is_clipped_to_bounds(reach_env, identity_normalizer):
    trajectory = rollout(reach_env, identity_normalizer, DisturbanceSpec("action_noise", 2.0))
    assert np.all(np.abs(trajectory.executed_actions) <= 1.0)
    assert np.any(np.abs(trajectory.pre_clip_actions) > 1.0)
    assert np.array_equal(
        trajectory.executed_actions, np.clip(trajectory.pre_clip_actions, -1.0, 1.0)
    )


def test_action_delay_executes_earlier_commands(reach_env, identity_normalizer):
    trajectory = rollout(
        reach_env, identity_normalizer, DisturbanceSpec("action_delay", 3, onset=10, duration=10)
    )
    executed, base = trajectory.executed_actions, trajectory.base_actions
    assert np.array_equal(executed[10:20], base[7:17])
    assert np.array_equal(executed[:10], base[:10])
    assert np.array_equal(executed[20:], base[20:])
    start = rollout(reach_env, identity_normalizer, DisturbanceSpec("action_delay", 2))
    assert np.array_equal(start.executed_actions[:2], np.zeros((2, 4)))


def test_observation_noise_only_reaches_the_controller(reach_env, identity_normalizer):
    seen = []

    def recording_policy(inputs):
        seen.append(inputs[0].copy())
        return proportional_policy(inputs)

    clean = rollout(reach_env, identity_normalizer)
    trajectory = rollout(
        reach_env,
        identity_normalizer,
        DisturbanceSpec("observation_noise", 0.02, onset=10, duration=5),
        policy=recording_policy,
    )
    assert np.flatnonzero(trajectory.observation_error > 0).tolist() == list(range(10, 15))
    # the controller's goal input stays clean (only 'observation' is noised)
    assert all(np.allclose(s[-3:], trajectory.original_goal) for s in seen)
    # identical behaviour before onset
    assert np.array_equal(clean.executed_actions[:10], trajectory.executed_actions[:10])
    # metrics use ground truth: logged achieved goal is the simulator's gripper site position
    unwrapped = reach_env.unwrapped
    site = unwrapped._utils.get_site_xpos(unwrapped.model, unwrapped.data, "robot0:grip")
    assert np.allclose(trajectory.achieved_goals[-1], site)
    assert np.allclose(
        trajectory.distances,
        np.linalg.norm(trajectory.achieved_goals - trajectory.desired_goals, axis=1),
    )


def test_goal_shift_uses_new_goal_and_does_not_count_old_goal(reach_env, identity_normalizer):
    spec = DisturbanceSpec("goal_shift", 0.1, onset=30)
    trajectory = rollout(reach_env, identity_normalizer, spec)
    (event,) = trajectory.goal_events
    old, new = event["original_goal"], event["new_goal"]
    assert np.allclose(old, trajectory.original_goal)
    assert np.linalg.norm(new - old) >= 0.1
    assert np.allclose(trajectory.desired_goals[:31], old)  # states 0..30 observed before shift
    assert np.allclose(trajectory.desired_goals[31:], new)
    # the robot was at the old goal when the goal moved, so the step right after the shift is
    # a failure against the new goal even though it is still near the old one
    assert trajectory.successes[29]
    assert trajectory.distances_to_original_goal[31] < 0.05
    assert not trajectory.successes[30]
    row = episode_row(trajectory, ConditionSpec("shift", (spec,)), "FetchReach-v4", "x")
    assert row["success_at_onset"] is False
    assert row["success_at_old_goal_after_shift"] is True
    assert row["recovered"] and row["recovery_time"] >= 2
    first_new = int(np.flatnonzero(trajectory.successes[30:])[0]) + 31
    assert row["recovery_time"] == first_new - 30
    # new goal lies in the environment's goal distribution for FetchReach
    unwrapped = reach_env.unwrapped
    offset = np.abs(new - unwrapped.initial_gripper_xpos[:3])
    assert np.all(offset <= unwrapped.target_range + 1e-9)


def test_goal_sampling_leaves_environment_rng_untouched(reach_env):
    reach_env.reset(seed=SEED)
    unwrapped = reach_env.unwrapped
    before = unwrapped.np_random.bit_generator.state
    sample_goal_with_rng(unwrapped, np.random.default_rng(0))
    assert unwrapped.np_random.bit_generator.state == before


def test_normalizer_is_frozen(identity_normalizer, reach_env):
    with pytest.raises(RuntimeError, match="frozen"):
        identity_normalizer.update(np.zeros((1, 10)), np.zeros((1, 3)))
    rollout(reach_env, identity_normalizer, DisturbanceSpec("observation_noise", 0.05))
    identity_normalizer.verify()


def test_effect_check_rejects_a_perturbation_that_does_nothing(reach_env, identity_normalizer):
    def zero_policy(inputs):
        return np.zeros((inputs.shape[0], 4))

    with pytest.raises(PerturbationEffectError, match="never changed"):
        rollout(
            reach_env, identity_normalizer, DisturbanceSpec("action_scale", 0.5), policy=zero_policy
        )


def test_summary_metrics_on_rollouts(reach_env, identity_normalizer):
    rows = []
    baseline = ConditionSpec("undisturbed", ())
    shift = ConditionSpec("shift", (DisturbanceSpec("goal_shift", 0.1, onset=45),))
    for seed in range(SEED, SEED + 4):
        for cond in (baseline, shift):
            trajectory = run_disturbed_episode(
                reach_env, proportional_policy, identity_normalizer, cond, seed, PSEED
            )
            rows.append(episode_row(trajectory, cond, "FetchReach-v4", "x"))
    import pandas as pd

    episodes = pd.DataFrame(rows)
    summary = summarize(episodes, 200, 0.95).set_index("condition")
    base = episodes[episodes["condition"] == "undisturbed"]
    assert summary.loc["undisturbed", "success_rate"] == base["final_success"].mean()
    assert summary.loc["undisturbed", "median_final_distance"] == base["final_distance"].median()
    part = episodes[episodes["condition"] == "shift"]
    eligible = part[part["recovery_eligible"].astype(bool)]
    assert summary.loc["shift", "recovery_eligible"] == len(eligible)
    assert summary.loc["shift", "recovery_rate"] == eligible["recovered"].astype(bool).mean()
    lost = int((base["final_success"].to_numpy() & ~part["final_success"].to_numpy()).sum())
    assert summary.loc["shift", "paired_lost_vs_baseline"] == lost


@pytest.mark.skipif(
    not checkpoint_path("fetch_reach_fp32_seed0").exists(),
    reason="trained FetchReach checkpoint not present",
)
def test_undisturbed_matches_existing_evaluation():
    from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
    from farc.base_policy.reference_eval import run_episode

    loaded = load_policy(checkpoint_path("fetch_reach_fp32_seed0"))
    policy = deterministic_policy_function(loaded.actor)
    env = make_env(loaded.env_config)
    frozen = FrozenNormalizer(loaded.normalizer)
    for seed in (SEED, SEED + 1, SEED + 2):
        reference = run_episode(env, policy, loaded.normalizer, seed)
        trajectory = run_disturbed_episode(
            env, policy, frozen, ConditionSpec("undisturbed", ()), seed, PSEED
        )
        assert reference.success == float(trajectory.successes[-1])
        assert reference.episode_return == trajectory.rewards.sum()
        assert np.isclose(reference.final_distance, trajectory.distances[-1])
    env.close()


# ---------------------------------------------------------------- observation masks

from farc.disturbances.perturbations import (  # noqa: E402
    ObservationNoise,
    goal_in_distribution,
)
from farc.envs.observation_layout import (  # noqa: E402
    OBJECT,
    REACH,
    control_period,
    layout_for,
    validate_layout,
)


@pytest.fixture(scope="module")
def push_env():
    env = make_env(EnvConfig(env_id="FetchPush-v4"))
    yield env
    env.close()


def _noise_on(env, mask, severity=0.01, seed=SEED, step=0):
    observation, _ = env.reset(seed=seed)
    unwrapped = env.unwrapped
    layout = layout_for(unwrapped, observation["observation"].shape[0])
    spec = DisturbanceSpec("observation_noise", severity, mask=mask)
    perturbation = ObservationNoise(spec, perturbation_rng(PSEED, seed, spec.type, 0), 50)
    perturbation.configure(layout, control_period(unwrapped))
    view = {
        "observation": observation["observation"].astype(np.float64),
        "desired_goal": observation["desired_goal"].astype(np.float64),
    }
    noisy = perturbation.apply(step, view)
    return view, noisy, layout, control_period(unwrapped)


@pytest.mark.parametrize("env_name", ["reach_env", "push_env"])
def test_layout_matches_simulator_during_motion(env_name, request):
    env = request.getfixturevalue(env_name)
    observation, _ = env.reset(seed=SEED)
    layout = layout_for(env.unwrapped, observation["observation"].shape[0])
    assert layout is (OBJECT if env_name == "push_env" else REACH)
    assert sum(s.stop - s.start for s in layout.slices.values()) == layout.size
    rng = np.random.default_rng(0)
    for _ in range(10):
        observation, *_ = env.step(rng.uniform(-1, 1, 4).astype(np.float32))
        validate_layout(env.unwrapped, observation["observation"], layout)
    assert np.isclose(control_period(env.unwrapped), 0.04)


def test_position_mask_on_reach_touches_only_gripper_position(reach_env):
    view, noisy, layout, _ = _noise_on(reach_env, "positions")
    changed = np.flatnonzero(noisy["observation"] != view["observation"])
    assert changed.tolist() == [0, 1, 2]
    assert np.array_equal(noisy["desired_goal"], view["desired_goal"])


def test_position_mask_on_push_keeps_relative_position_consistent(push_env):
    view, noisy, layout, _ = _noise_on(push_env, "positions")
    changed = set(np.flatnonzero(noisy["observation"] != view["observation"]).tolist())
    assert changed == set(range(0, 9))  # gripper, object, and the derived relative position
    obs = noisy["observation"]
    assert np.allclose(obs[6:9], obs[3:6] - obs[0:3])
    assert np.array_equal(noisy["desired_goal"], view["desired_goal"])


def test_velocity_mask_uses_metres_per_second(push_env):
    view, noisy, layout, dt = _noise_on(push_env, "velocities", severity=0.5)
    delta = noisy["observation"] - view["observation"]
    changed = set(np.flatnonzero(delta).tolist())
    assert changed == set(range(14, 17)) | set(range(20, 25))
    draw = perturbation_rng(PSEED, SEED, "observation_noise", 0).standard_normal(25)
    assert np.allclose(delta[20:23], 0.5 * dt * draw[20:23])
    assert np.allclose(delta[14:17], 0.5 * dt * draw[14:17] - delta[20:23])


def test_masks_share_draws_and_are_reproducible(push_env):
    _, full, _, _ = _noise_on(push_env, "full", severity=0.01)
    view, positions, _, _ = _noise_on(push_env, "positions", severity=0.01)
    _, again, _, _ = _noise_on(push_env, "positions", severity=0.01)
    assert np.array_equal(positions["observation"], again["observation"])
    # the gripper and object position noise is the same draw the full mask uses there
    assert np.allclose(positions["observation"][0:6], full["observation"][0:6])
    _, other_seed, _, _ = _noise_on(push_env, "positions", severity=0.01, seed=SEED + 1)
    assert not np.allclose(other_seed["observation"][0:3] - view["observation"][0:3], 0)


def test_masked_noise_never_reaches_ground_truth(push_env):
    normalizer = FrozenNormalizer(GoalObservationNormalizer(25, 3, eps=1.0, normalized_clip=200.0))
    seen = []

    def policy(inputs):
        seen.append(inputs[0].copy())
        return np.tile(np.array([0.3, -0.2, 0.0, 0.0]), (inputs.shape[0], 1))

    spec = DisturbanceSpec("observation_noise", 0.02, onset=5, duration=10, mask="positions")
    trajectory = run_disturbed_episode(push_env, policy, normalizer, condition(spec), SEED, PSEED)
    clean = run_disturbed_episode(push_env, policy, normalizer, ConditionSpec("u", ()), SEED, PSEED)
    # the policy ignores its input here, so the true trajectory must be identical
    assert np.array_equal(trajectory.achieved_goals, clean.achieved_goals)
    assert np.array_equal(trajectory.desired_goals, clean.desired_goals)
    assert np.flatnonzero(trajectory.observation_error > 0).tolist() == list(range(5, 15))
    unwrapped = push_env.unwrapped
    site = unwrapped._utils.get_site_xpos(unwrapped.model, unwrapped.data, "object0")
    assert np.allclose(trajectory.achieved_goals[-1], site)
    assert all(np.allclose(s[-3:], trajectory.original_goal) for s in seen)
    normalizer.verify()


def test_mask_validation():
    with pytest.raises(DisturbanceConfigError, match="mask"):
        validate_disturbance(DisturbanceSpec("observation_noise", 0.01, mask="angles"))
    with pytest.raises(DisturbanceConfigError, match="only applies"):
        validate_disturbance(
            DisturbanceSpec(
                "observation_noise", 0.01, mask="positions", fields=("observation", "desired_goal")
            )
        )
    (entry,) = expand_condition_entry(
        {"type": "observation_noise", "severity": 0.01, "mask": "positions"}, "x"
    )
    assert entry.name == "observation_noise_positions_sev0.01_on0_durinf"


def test_goal_shift_goals_lie_in_the_training_goal_range(push_env):
    push_env.reset(seed=SEED)
    unwrapped = push_env.unwrapped
    rng = np.random.default_rng(1)
    goals = [sample_goal_with_rng(unwrapped, rng) for _ in range(200)]
    assert all(goal_in_distribution(unwrapped, g) for g in goals)
    outside = goals[0].copy()
    outside[0] += 1.0
    assert not goal_in_distribution(unwrapped, outside)
    raised = goals[0].copy()
    raised[2] += 0.1
    assert not goal_in_distribution(unwrapped, raised)


def test_validation_seed_range_is_separate():
    path = REPO / "configs/disturbances/pilot.json"
    with pytest.raises(DisturbanceConfigError, match="pilot episode seeds"):
        load_experiment(path, {"episode_seed_offset": 3_500_000})
    config = load_experiment(path, {"stage": "validation", "episode_seed_offset": 3_500_000})
    assert config.episode_seeds[0] == 3_500_000
    with pytest.raises(DisturbanceConfigError, match="validation episode seeds"):
        load_experiment(path, {"stage": "validation", "episode_seed_offset": 3_999_990})


# ---------------------------------------------------------------- recovery regimes, runner

from farc.disturbances.experiment import ResumeConfigMismatch, run_experiment  # noqa: E402
from farc.statistics import hierarchical_bootstrap_interval  # noqa: E402


def test_transient_recovery_is_measured_after_the_window(reach_env, identity_normalizer):
    # the scripted controller is at the goal by step 15; a full-strength scale (zero action)
    # can't move it away, so use heavy noise to push it out during steps 15..24
    spec = DisturbanceSpec("action_noise", 1.0, onset=15, duration=10)
    cond = ConditionSpec("mid", (spec,))
    assert cond.regime == "transient" and cond.window_end == 25
    rows = []
    for seed in range(SEED, SEED + 8):
        trajectory = run_disturbed_episode(
            reach_env, proportional_policy, identity_normalizer, cond, seed, PSEED
        )
        row = episode_row(trajectory, cond, "FetchReach-v4", "x")
        rows.append(row)
        assert row["recovery_reference_state"] == 25
        assert row["success_at_reference"] == bool(trajectory.distances[25] < 0.05)
        assert row["recovery_eligible"] == (not row["success_at_reference"])
        if row["recovery_eligible"]:
            hits = np.flatnonzero(trajectory.successes[25:])
            if hits.size:
                assert row["recovered"] and row["recovery_time"] == hits[0] + 1
            else:
                assert row["recovery_censored"] and row["censor_time"] == 25
        else:
            assert row["recovered"] is None  # not counted as a recovery
        window = trajectory.successes[15:25]
        assert row["lost_success_in_window"] == bool(row["success_at_onset"] and not window.all())
    assert any(r["recovery_eligible"] for r in rows)


def test_permanent_disturbance_excludes_episodes_successful_at_onset(
    reach_env, identity_normalizer
):
    # onset 20 is after the scripted controller reached the goal, so no episode is eligible
    cond = ConditionSpec("late", (DisturbanceSpec("action_scale", 0.5, onset=20),))
    assert cond.regime == "permanent"
    trajectory = run_disturbed_episode(
        reach_env, proportional_policy, identity_normalizer, cond, SEED, PSEED
    )
    row = episode_row(trajectory, cond, "FetchReach-v4", "x")
    assert row["success_at_onset"] and row["recovery_eligible"] is False
    assert row["recovered"] is None


def test_hierarchical_bootstrap_brackets_the_pooled_mean():
    rng = np.random.default_rng(0)
    groups = [np.ones(50), np.zeros(50), np.r_[np.ones(25), np.zeros(25)]]
    low, high = hierarchical_bootstrap_interval(groups, 4000, 0.95, rng)
    assert low < 0.5 < high and low >= 0 and high <= 1
    assert hierarchical_bootstrap_interval([np.ones(10)] * 3, 500, 0.95, rng) == (1.0, 1.0)
    assert np.isnan(hierarchical_bootstrap_interval([np.array([])], 10, 0.95, rng)[0])


@pytest.mark.skipif(
    not checkpoint_path("fetch_reach_fp32_seed0").exists(),
    reason="trained FetchReach checkpoint not present",
)
def test_runner_resumes_at_condition_level(tmp_path):
    values = json.loads((REPO / "configs/disturbances/phase1_extended.json").read_text())
    values.update(
        episodes=2,
        checkpoints=["$FARC_DATA_DIR/checkpoints/fetch_reach_fp32_seed0/checkpoint.pt"],
        output_dir=str(tmp_path),
        bootstrap_resamples=50,
        conditions=[
            {"type": "none", "name": "undisturbed"},
            {"type": "action_noise", "severity": 0.3},
            {"type": "observation_noise", "mask": "positions", "severity": 0.01},
        ],
    )
    from farc.disturbances.config import experiment_from_dict

    config = experiment_from_dict(values)
    out = tmp_path / "run"
    first = run_experiment(config, out, verbose=False)
    episodes = pd_read(first["episodes"])
    part = next((out / "parts").glob("*/action_noise*/done.json"))
    part.unlink()  # simulate an interruption before this condition finished
    second = run_experiment(config, out, verbose=False, resume=True)
    assert pd_read(second["episodes"]).equals(episodes)
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["parts_run"] == 1 and metadata["parts_reused"] == 2
    assert metadata["checkpoints"][0]["original_evaluator_check"]["identical"]
    with pytest.raises(ResumeConfigMismatch):
        run_experiment(config, out, verbose=False)  # resume not requested
    changed = experiment_from_dict({**values, "perturbation_seed": 1})
    with pytest.raises(ResumeConfigMismatch):
        run_experiment(changed, out, verbose=False, resume=True)


def pd_read(path):
    import pandas as pd

    return pd.read_csv(path)


def test_gripper_positions_are_ground_truth_and_ignore_observation_noise(push_env):
    normalizer = FrozenNormalizer(GoalObservationNormalizer(25, 3, eps=1.0, normalized_clip=200.0))

    def policy(inputs):
        return np.tile(np.array([0.4, 0.2, -0.1, 0.0]), (inputs.shape[0], 1))

    clean = run_disturbed_episode(push_env, policy, normalizer, ConditionSpec("u", ()), SEED, PSEED)
    assert clean.gripper_positions is not None
    observation, _ = push_env.reset(seed=SEED)
    assert np.allclose(clean.gripper_positions[0], observation["observation"][0:3])
    noisy = run_disturbed_episode(
        push_env,
        policy,
        normalizer,
        condition(DisturbanceSpec("observation_noise", 0.05, mask="positions")),
        SEED,
        PSEED,
    )
    assert noisy.gripper_positions is not None
    unwrapped = push_env.unwrapped
    site = unwrapped._utils.get_site_xpos(unwrapped.model, unwrapped.data, "robot0:grip")
    assert np.allclose(noisy.gripper_positions[-1], site)
    # the policy ignores its input, so the noise must not show up in the logged positions
    assert np.array_equal(noisy.gripper_positions, clean.gripper_positions)
    assert clean.gripper_positions.shape == (51, 3)
    assert np.linalg.norm(clean.gripper_positions[-1] - clean.gripper_positions[0]) > 0.05
