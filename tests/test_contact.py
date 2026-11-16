"""Phase F1: contact-aware motion prediction."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from farc.adaptation.experiment import AdaptationConfigError
from farc.contact.data import check_seeds, contact_flags
from farc.contact.dynamics import (
    S,
    load_dynamics,
    rollout,
    rollout_batch,
    static_object_batch,
    train_dynamics,
    with_state,
)
from farc.contact.episodes import (
    learned_gripper_consistent,
    learned_gripper_position_only,
    learned_predictor,
    oracle_prediction,
    phase_e_predictor,
    queued_commands,
    run_episode,
)
from farc.contact.evaluation import load_contact_evaluation
from farc.counterfactual.session import EpisodeSession
from farc.paths import checkpoint_path

CONFIG = "configs/contact/phase_f1_eval.json"
K0 = 0.62


def synthetic_transitions(n=20000, seed=0):
    """Gripper moves 0.62*0.05*command; object moves with the gripper when within 4 cm."""
    rng = np.random.default_rng(seed)
    obs = np.zeros((n, 25))
    obs[:, S["gripper_position"]] = rng.uniform(-0.1, 0.1, (n, 3))
    obs[:, S["object_position"]] = obs[:, S["gripper_position"]] + rng.uniform(-0.06, 0.06, (n, 3))
    obs = with_state(
        obs,
        np.concatenate(
            [obs[:, S["gripper_position"]], obs[:, S["object_position"]], np.zeros((n, 12))], 1
        ),
    )
    cmd = rng.uniform(-1, 1, (n, 3))
    nxt = obs.copy()
    move = K0 * 0.05 * cmd
    nxt[:, S["gripper_position"]] += move
    touching = np.linalg.norm(obs[:, S["object_relative_position"]], axis=1) < 0.04
    nxt[touching, 3:6] += move[touching]
    nxt[:, S["object_relative_position"]] = nxt[:, 3:6] - nxt[:, 0:3]
    return obs, cmd, nxt


@pytest.fixture(scope="module")
def model():
    obs, cmd, nxt = synthetic_transitions()
    return train_dynamics(obs, cmd, nxt, width=128, weight_decay=0.0, epochs=60, seed=0)


# --- action queue ---------------------------------------------------------------------------


def test_queued_commands_order_and_padding():
    history = [np.full(4, float(i)) for i in range(5)]
    assert [c[0] for c in queued_commands(history, 3)] == [2.0, 3.0, 4.0]
    padded = queued_commands(history[:1], 3)
    assert [c[0] for c in padded] == [0.0, 0.0, 0.0]  # zeros before the episode
    assert len(queued_commands([], 2)) == 2


# --- observation consistency ------------------------------------------------------------------


def test_with_state_keeps_relative_position_consistent():
    obs = np.arange(25, dtype=float)
    state = np.arange(18, dtype=float) * 0.1
    out = with_state(obs, state)
    np.testing.assert_allclose(out[S["object_relative_position"]],
                               out[S["object_position"]] - out[S["gripper_position"]])  # fmt: skip
    np.testing.assert_array_equal(out[S["finger_position"]], obs[S["finger_position"]])
    np.testing.assert_array_equal(out[S["finger_velocity"]], obs[S["finger_velocity"]])


def test_rollout_outputs_are_consistent_and_ablation_holds_object(model):
    obs, _, _ = synthetic_transitions(10, seed=3)
    commands = [np.array([0.5, -0.2, 0.0, 0.0])] * 3
    for move_object in (True, False):
        out = rollout(model, obs[0], commands, 1.0, move_object)
        relative = out[S["object_position"]] - out[S["gripper_position"]]
        np.testing.assert_allclose(out[S["object_relative_position"]], relative)
        if not move_object:
            for key in ("object_position", "object_rotation", "object_relative_velocity",
                        "object_angular_velocity"):  # fmt: skip
                np.testing.assert_array_equal(out[S[key]], obs[0][S[key]])


def test_learned_model_beats_static_on_synthetic_contact(model):
    obs, cmd, _ = synthetic_transitions(2000, seed=9)
    executed = np.repeat(cmd[:, None, :], 2, axis=1)
    truth = obs.copy()
    for _ in range(2):  # the synthetic plant, two steps
        move = K0 * 0.05 * cmd
        touching = np.linalg.norm(truth[:, S["object_relative_position"]], axis=1) < 0.04
        truth[:, 0:3] += move
        truth[touching, 3:6] += move[touching]
        truth[:, 6:9] = truth[:, 3:6] - truth[:, 0:3]
    learned = rollout_batch(model, obs, executed)
    static = static_object_batch(obs, executed, K0)
    moved = np.linalg.norm(truth[:, 3:6] - obs[:, 3:6], axis=1) > 0.002

    def err(p):
        return np.linalg.norm(p[moved, 3:6] - truth[moved, 3:6], axis=1).mean()

    assert moved.sum() > 100 and err(learned) < 0.5 * err(static)
    single = rollout(model, obs[0], [np.r_[cmd[0], 0.0]] * 2)
    np.testing.assert_allclose(single, learned[0], atol=1e-6)


def test_model_hash_is_enforced(model, tmp_path):
    digest = model.save(tmp_path / "m.pt")
    loaded = load_dynamics(tmp_path / "m.pt", digest)
    x = synthetic_transitions(5)[0]
    np.testing.assert_allclose(loaded.step(x, np.zeros((5, 3))), model.step(x, np.zeros((5, 3))))
    with pytest.raises(ValueError, match="hash"):
        load_dynamics(tmp_path / "m.pt", "0" * 64)


# --- no privileged state in deployable predictors -------------------------------------------


@pytest.mark.parametrize("make", [lambda m: phase_e_predictor(K0),
                                  lambda m: learned_predictor(m, True),
                                  lambda m: learned_predictor(m, False),
                                  lambda m: learned_gripper_position_only(m),
                                  lambda m: learned_gripper_consistent(m)])  # fmt: skip
def test_deployable_predictors_take_only_observation_queue_and_gain(make, model):
    predictor = make(model)
    assert list(inspect.signature(predictor).parameters) == ["observation", "queued", "gain"]
    closure = inspect.getclosurevars(predictor)
    captured = [*closure.nonlocals.values(), *closure.globals.values()]
    assert not any(isinstance(v, EpisodeSession) for v in captured)
    obs = synthetic_transitions(1)[0][0]
    before = obs.copy()
    predictor(obs, [np.zeros(4)], 1.0)
    np.testing.assert_array_equal(obs, before)  # input not mutated


def test_oracle_and_predictor_are_exclusive():
    with pytest.raises(ValueError):
        run_episode(None, None, phase_e_predictor(K0), None, use_oracle=True)  # type: ignore[arg-type]


def test_contact_flags():
    start = np.zeros((3, 25))
    end = np.zeros((3, 25))
    start[0, 6:8] = [0.03, 0.0]  # near
    start[1, 6:8] = [0.2, 0.0]
    end[1, 3] = 0.005  # far but the object moved 5 mm
    start[2, 6:8] = [0.2, 0.0]  # far and still
    assert contact_flags(start, end).tolist() == [True, True, False]


# --- seeds and pre-registration ---------------------------------------------------------------


def test_config_pins_plan_and_model():
    config = load_contact_evaluation(CONFIG)
    assert len(config.model_sha256) == 64 and min(config.seeds) == 3_653_000


@pytest.mark.parametrize("seeds,role", [
    ([3_653_000], "training"),  # evaluation seed used for training
    ([3_650_000], "evaluation"),  # training seed used for evaluation
    ([3_641_050], "evaluation"),  # Phase E2 evaluation seeds
    ([4_000_001], "evaluation"),  # held out
    ([3_651_049, 3_651_050], "selection"),  # leaves the selection block
])  # fmt: skip
def test_seed_roles_are_protected(seeds, role):
    with pytest.raises(AdaptationConfigError):
        check_seeds(seeds, role)


def test_changed_plan_is_refused(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text(Path("docs/PHASE_F1_PREREGISTRATION.md").read_text() + "\nedit\n")
    values = json.loads(Path(CONFIG).read_text())
    values["plan"] = str(plan)
    (tmp_path / "c.json").write_text(json.dumps(values))
    with pytest.raises(AdaptationConfigError, match="pre-registration"):
        load_contact_evaluation(tmp_path / "c.json")


# --- simulator: causality, exact oracle restore, Phase E equivalence --------------------------

CHECKPOINT = checkpoint_path("fetch_push_fp32_seed2")
needs_policy = pytest.mark.skipif(not CHECKPOINT.exists(), reason="FetchPush checkpoint absent")


@pytest.fixture(scope="module")
def sim():
    from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
    from farc.contact.data import build_conditions
    from farc.disturbances.rollout import FrozenNormalizer
    from farc.envs.factory import make_env

    loaded = load_policy(CHECKPOINT)
    config = load_contact_evaluation(CONFIG)
    conditions = {c.name: c for c in build_conditions(config.conditions_from, config.conditions)}
    return (make_env(loaded.env_config), deterministic_policy_function(loaded.actor),
            FrozenNormalizer(loaded.normalizer), conditions, config)  # fmt: skip


def _session(sim, name, seed=3_000_777):
    env, policy, normalizer, conditions, config = sim
    return EpisodeSession(env, policy, normalizer, conditions[name], seed, 5), conditions[name]


@needs_policy
def test_oracle_restores_exactly_and_predicts_the_realised_future(sim):
    session, _ = _session(sim, "C3_delay3")
    futures = {}
    while not session.done:
        view = session.observe()
        if view.step in (10, 25):
            digest = session.state_digest()
            futures[view.step] = oracle_prediction(session, 3)
            assert session.state_digest() == digest
            assert session.pending_view is not None and session.pending_view.step == view.step
            np.testing.assert_array_equal(session.pending_view.base_action, view.base_action)
        session.act(view.base_action)
        if view.step - 2 in futures:  # observation after executing steps t..t+2
            np.testing.assert_array_equal(session.observation["observation"],
                                          futures[view.step - 2])  # fmt: skip


@needs_policy
def test_deployable_predictor_sees_view_and_issued_queue_only(sim):
    session, condition = _session(sim, "C6_delay3_posnoise0.005")
    from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig

    calls = []

    def spy(observation, queued, gain):
        calls.append((session.step, observation.copy(), [q.copy() for q in queued],
                      [c.copy() for c in session.commands],
                      np.asarray(session.observation["observation"]).copy()))  # fmt: skip
        return observation

    run_episode(session, ActuatorEstimator(EstimatorConfig(**sim[4].estimator)), spy, condition)
    assert calls
    for t, observation, queued, commands, true_obs in calls:
        assert len(commands) == t  # only commands already issued (causal)
        d = len(queued)
        expected = queued_commands(commands, d)
        for a, b in zip(queued, expected, strict=True):
            np.testing.assert_array_equal(a, b)
        assert not np.array_equal(observation, true_obs)  # the noisy view, not the true state


@needs_policy
def test_phase_e_predictor_reproduces_the_phase_e_compensator(sim):
    from farc.adaptation.compensation import CompensationConfig, LatchedEstimate
    from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
    from farc.adaptation.experiment import run_episode as phase_e_episode

    est = EstimatorConfig(**sim[4].estimator)
    for name in ("C3_delay3", "C7_delay3_scale0.5"):
        a_session, condition = _session(sim, name)
        a = run_episode(a_session, ActuatorEstimator(est), phase_e_predictor(K0), condition)
        b_session, _ = _session(sim, name)
        b = phase_e_episode(b_session, ActuatorEstimator(est), CompensationConfig(False, True),
                            LatchedEstimate(), condition)  # fmt: skip
        assert a["final_distance"] == b["final_distance"]
        np.testing.assert_array_equal(np.array(a_session.commands), np.array(b_session.commands))


def test_exploratory_ablations_keep_object_static_and_consistent(model):
    obs = synthetic_transitions(3, seed=4)[0][0]
    queued = [np.array([0.7, 0.1, 0.0, 0.0])] * 3
    for predictor in (learned_gripper_position_only(model), learned_gripper_consistent(model)):
        out = predictor(obs, queued, 1.0)
        np.testing.assert_array_equal(out[S["object_position"]], obs[S["object_position"]])
        relative = out[S["object_position"]] - out[S["gripper_position"]]
        np.testing.assert_allclose(out[S["object_relative_position"]], relative)
    out = learned_gripper_consistent(model)(obs, queued, 1.0)
    np.testing.assert_allclose(out[S["object_relative_velocity"]], -out[S["gripper_velocity"]])
    out = learned_gripper_position_only(model)(obs, queued, 1.0)
    np.testing.assert_array_equal(out[S["gripper_velocity"]], obs[S["gripper_velocity"]])
