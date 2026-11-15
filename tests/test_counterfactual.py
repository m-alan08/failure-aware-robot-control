from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farc.counterfactual.experiment import (
    CounterfactualConfigError,
    candidate_metrics,
    candidate_table,
    clustered_interval,
    load_counterfactual_experiment,
    oracle_table,
    split_seeds,
)
from farc.counterfactual.residuals import (
    STRATEGIES,
    Candidate,
    alignment,
    candidate_grid,
    goal_directed,
    preserve_push,
    reduced,
)
from farc.paths import checkpoint_path

Z = 0.425
PILOT = "configs/counterfactual/phase_d_pilot.json"


# --- residual strategies ------------------------------------------------------------------


def test_residual_directions():
    g, o, goal = np.array([1.2, 0.7, Z]), np.array([1.3, 0.7, Z]), np.array([1.3, 0.9, Z])
    np.testing.assert_allclose(goal_directed(g, o, goal, np.zeros(4)), [0, 1, 0])
    # gripper 0.1 m off the o->G line (which is x = 1.3): steer +x, saturated
    np.testing.assert_allclose(alignment(g, o, goal, np.zeros(4)), [1, 0, 0])
    base = np.array([0.3, -0.2, 0.1, 0.5])
    np.testing.assert_allclose(reduced(g, o, goal, base), [-0.3, 0.2, -0.1])
    np.testing.assert_allclose(preserve_push(g, o, goal, np.array([0.3, 0.4, 0.1, 0])), [0, 0.4, 0])
    # nothing is added when the base action points away from the goal
    np.testing.assert_allclose(preserve_push(g, o, goal, np.array([0, -0.4, 0, 0])), [0, 0, 0])
    assert np.all(goal_directed(g, o, o, np.zeros(4)) == 0)  # object at the goal


def test_action_equation_and_bounds():
    rng = np.random.default_rng(0)
    for candidate in candidate_grid((0.25, 0.5, 1.0), (3,)):
        for _ in range(100):
            g, o, goal = (rng.uniform([1.0, 0.4, 0.4], [1.6, 1.0, 0.6]) for _ in range(3))
            base = rng.uniform(-1, 1, 4)
            a = candidate.command(g, o, goal, base)
            assert np.all(np.abs(a) <= 1) and a[3] == base[3]
            from farc.counterfactual.residuals import DELTAS

            delta = DELTAS[candidate.strategy](g, o, goal, base)
            assert np.all(np.abs(delta) <= 1 + 1e-12)
            np.testing.assert_allclose(a[:3], np.clip(base[:3] + candidate.alpha * delta, -1, 1))
    assert Candidate("reduced", 1.0, 3).command(g, o, goal, base)[:3].tolist() == [0, 0, 0]
    assert len(candidate_grid((0.25, 0.5), (3, 6))) == 4 * len(STRATEGIES)


# --- accounting ---------------------------------------------------------------------------


def branch_frame():
    rows = []
    # (episode_seed, base_success, branch_success, base_dist, branch_dist)
    data = [
        (1, True, True, 0.02, 0.01),
        (1, True, False, 0.02, 0.08),
        (2, False, True, 0.2, 0.03),
        (2, False, False, 0.2, 0.25),
        (3, False, False, 0.3, 0.1),
        (3, True, True, 0.01, 0.01),
    ]
    for i, (seed, a, b, da, db) in enumerate(data):
        for candidate, flip in (("x_a0.5_d3", False), ("y_a1_d3", True)):
            rows.append(
                {
                    "policy_seed": 0,
                    "condition": "c",
                    "episode_seed": seed,
                    "t": i,
                    "disturbance_type": "d",
                    "candidate": candidate,
                    "strategy": candidate[0],
                    "alpha": 0.5,
                    "duration": 3,
                    "base_success": a,
                    "success": (not b) if flip else b,
                    "base_final_distance": da,
                    "final_distance": db,
                    "applied_steps": 3,
                    "mean_residual_norm": 0.1,
                }
            )
    return pd.DataFrame(rows)  # fmt: skip


def test_candidate_metrics():
    frame = branch_frame()
    m = candidate_metrics(frame[frame["candidate"] == "x_a0.5_d3"])
    assert m["states"] == 6 and m["base_success_rate"] == 0.5 and m["branch_success_rate"] == 0.5
    assert m["rescued"] == 1 and m["rescued_fraction"] == pytest.approx(1 / 3)
    assert m["harmed"] == 1 and m["harmed_fraction"] == pytest.approx(1 / 3)
    assert m["mean_distance_improvement"] == pytest.approx(
        np.mean([0.01, -0.06, 0.17, -0.05, 0.2, 0])
    )


def test_oracle_and_clustered_interval():
    frame = branch_frame()
    oracle = oracle_table(frame).set_index("group").loc["pooled"]
    assert oracle["base_failures"] == 3 and oracle["oracle_rescuable_failures"] == 3
    # state 1 (base success): candidate x fails, candidate y succeeds -> not harmed by every one
    assert oracle["base_successes_harmed_by_every_candidate"] == 0
    assert oracle["base_successes_harmed_by_some_candidate"] == 3
    sub = frame[frame["candidate"] == "x_a0.5_d3"].assign(
        diff=lambda f: f["success"].astype(float) - f["base_success"].astype(float)
    )
    est, low, high = clustered_interval(sub, "diff", 500, np.random.default_rng(0))
    assert est == 0.0 and low <= est <= high
    table = candidate_table(frame, [], 200, 0)
    assert set(table["candidate"]) == {"x_a0.5_d3", "y_a1_d3"}


def test_selection_split_is_seed_disjoint():
    seeds = np.repeat(np.arange(3_621_000, 3_621_060), 7)
    selection, evaluation = split_seeds(seeds, 0.5, 0)
    assert len(selection) == len(evaluation) == 30 and not set(selection) & set(evaluation)


def test_config_seed_guards(tmp_path):
    base = json.loads(Path(PILOT).read_text())
    assert load_counterfactual_experiment(PILOT).episode_seeds[0] == 3_620_000
    full = load_counterfactual_experiment("configs/counterfactual/phase_d_full.json")
    assert not set(full.episode_seeds) & set(load_counterfactual_experiment(PILOT).episode_seeds)
    for offset in (3_500_000, 3_600_010, 3_610_050, 4_000_000):
        path = tmp_path / f"{offset}.json"
        path.write_text(json.dumps({**base, "episode_seed_offset": offset}))
        with pytest.raises(CounterfactualConfigError):
            load_counterfactual_experiment(path)


# --- exact restoration (needs the frozen policy) -------------------------------------------

CHECKPOINT = checkpoint_path("fetch_push_fp32_seed2")
needs_policy = pytest.mark.skipif(not CHECKPOINT.exists(), reason="FetchPush checkpoint absent")
CONDITIONS = ["action_delay_sev3_on0_durinf", "action_noise_sev1_on0_durinf",
              "goal_shift_sev0.1_on25_durinf", "observation_noise_sev0.01_on0_durinf",
              "action_scale_sev0.5_on15_dur10"]  # fmt: skip


@pytest.fixture(scope="module")
def policy_env():
    from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
    from farc.disturbances.config import load_experiment
    from farc.disturbances.rollout import FrozenNormalizer
    from farc.envs.factory import make_env

    loaded = load_policy(CHECKPOINT)
    conditions = {c.name: c for c in load_experiment(
        "configs/disturbances/phase1_extended.json").conditions}  # fmt: skip
    return (make_env(loaded.env_config), deterministic_policy_function(loaded.actor),
            FrozenNormalizer(loaded.normalizer), conditions)  # fmt: skip


@needs_policy
@pytest.mark.parametrize("name", CONDITIONS)
def test_session_matches_phase1_rollout(policy_env, name):
    from farc.counterfactual.session import EpisodeSession
    from farc.disturbances.rollout import run_disturbed_episode

    env, policy, frozen, conditions = policy_env
    reference = run_disturbed_episode(env, policy, frozen, conditions[name], 3_000_777, 5)
    session = EpisodeSession(env, policy, frozen, conditions[name], 3_000_777, 5)
    executed = []
    while not session.done:
        executed.append(session.act(session.observe().base_action))
    np.testing.assert_array_equal(np.array(executed), reference.executed_actions)
    np.testing.assert_array_equal(session.successes, reference.successes)


@needs_policy
@pytest.mark.parametrize("name", CONDITIONS)
@pytest.mark.parametrize("branch_at", [0, 24, 25, 26, 49])
def test_restore_replays_exactly_after_a_different_branch(policy_env, name, branch_at):
    """Branch A after restore == unbranched base run, even after running a different branch
    (which advances generators, the delay queue and the goal-shift state) in between."""
    from farc.counterfactual.session import EpisodeSession

    env, policy, frozen, conditions = policy_env
    session = EpisodeSession(env, policy, frozen, conditions[name], 3_000_778, 5)
    snap, digest_at_snapshot, executed = None, None, []
    while not session.done:
        view = session.observe()
        if view.step == branch_at:
            snap, digest_at_snapshot = session.snapshot(), session.state_digest()
        executed.append(session.act(view.base_action))
    final = (session.successes.copy(), session.final_distance, session.state_digest())

    assert snap is not None
    session.restore(snap)
    assert session.state_digest() == digest_at_snapshot
    while not session.done:  # a very different branch
        session.observe()
        session.act(np.array([1.0, -1.0, 1.0, 0.0]))
    session.restore(snap)
    assert session.state_digest() == digest_at_snapshot
    replay = executed[:branch_at]
    while not session.done:
        replay.append(session.act(session.observe().base_action))
    np.testing.assert_array_equal(np.array(replay), np.array(executed))
    assert (session.final_distance, session.state_digest()) == final[1:]
    np.testing.assert_array_equal(session.successes, final[0])


@needs_policy
@pytest.mark.parametrize(
    "name", ["action_noise_sev1_on0_durinf", "observation_noise_sev0.01_on0_durinf"]
)
def test_branches_share_future_disturbance_draws(policy_env, name):
    """Noise generators are in the same state at every later step of both branches, so both
    branches receive the same draws whatever they command."""
    from farc.counterfactual.session import EpisodeSession

    env, policy, frozen, conditions = policy_env
    session = EpisodeSession(env, policy, frozen, conditions[name], 3_000_779, 5)
    for _ in range(10):
        session.act(session.observe().base_action)
    session.observe()
    snap = session.snapshot()
    states = []
    for command in (np.zeros(4), np.array([1.0, 1.0, -1.0, 0.0])):
        session.restore(snap)
        seen = []
        for _ in range(8):
            session.observe()
            seen.append([repr(p.rng.bit_generator.state) for p in session.perturbations])
            session.act(command)
        states.append(seen)
    assert states[0] == states[1]


@needs_policy
def test_delay_queue_replays_issued_commands(policy_env):
    from farc.counterfactual.session import EpisodeSession

    env, policy, frozen, conditions = policy_env
    session = EpisodeSession(env, policy, frozen, conditions["action_delay_sev3_on0_durinf"],
                             3_000_780, 5)  # fmt: skip
    for _ in range(10):
        session.act(session.observe().base_action)
    session.observe()
    snap = session.snapshot()
    marker = np.array([0.5, -0.5, 0.25, 0.0])
    executed = [session.act(marker)]
    for _ in range(3):
        session.observe()
        executed.append(session.act(np.zeros(4)))
    np.testing.assert_array_equal(executed[3], marker)  # issued 3 steps earlier
    session.restore(snap)
    assert len(session.commands) == 10


def test_placebo_direction_is_deterministic_unit_and_horizontal():
    from farc.counterfactual.residuals import PLACEBOS, random_direction

    rng = np.random.default_rng(1)
    angles = []
    for _ in range(50):
        g, o, goal = (rng.uniform(1.0, 1.5, 3) for _ in range(3))
        d = random_direction(g, o, goal, np.zeros(4))
        np.testing.assert_array_equal(d, random_direction(g, o, goal, np.zeros(4)))
        assert d[2] == 0 and np.linalg.norm(d) == pytest.approx(1.0)
        angles.append(np.arctan2(d[1], d[0]))
    assert np.std(angles) > 1.0  # spread over directions, not aligned with the task
    assert candidate_grid((0.5,), (3,), PLACEBOS)[0].name == "random_direction_a0.5_d3"
    with pytest.raises(ValueError):
        candidate_grid((0.5,), (3,), ("nope",))
