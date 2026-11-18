"""Phase I Stage-0 checks: seeds, distribution, features, normaliser and warm-start equivalence."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from farc.adaptation.experiment import AdaptationConfigError
from farc.paths import checkpoint_path
from farc.retrain.phase_i import (
    ARMS,
    K0_STEP,
    Featurizer,
    HybridNormalizer,
    load_config,
    training_schedule,
)
from farc.supervision.seeds import check_i_seeds

CONFIG = "configs/retrain/phase_i.json"
EST = {"nominal_gain": 0.62, "window": 10, "max_lag": 4, "min_transitions": 4,
       "min_excitation": 0.000225, "min_r2": 0.8, "min_margin": 2.0}  # fmt: skip
needs_ckpt = pytest.mark.skipif(not checkpoint_path("fetch_push_fp32_seed0").exists(),
                                reason="frozen checkpoint absent")  # fmt: skip


def test_config_seeds_and_guard():
    c = load_config(CONFIG)
    assert c.seeds("debugging") == list(range(3_700_100, 3_700_110))
    assert c.seeds("evaluation") == list(range(3_702_000, 3_702_300))
    assert c.training["train"]["random_steps"] == 0
    assert c.training["train"]["total_steps"] == 1_000_000
    for bad in ([4_000_600], [3_701_000], [3_700_005], [3_702_300], [4_000_899]):
        with pytest.raises(AdaptationConfigError):
            check_i_seeds(bad, "evaluation")


def test_training_distribution_thirds():
    groups = {"nominal": 0, "persistent": 0, "transient": 0}
    for e in range(9000):
        fam, _ = training_schedule(0, e)
        g = ("nominal" if fam == "nominal" else
             "persistent" if fam in ("constant", "persistent") else "transient")  # fmt: skip
        groups[g] += 1
    assert all(abs(n - 3000) < 200 for n in groups.values()), groups
    assert training_schedule(1, 3)[1] == training_schedule(1, 3)[1]


def test_featurizer_ia_layout_and_scaling():
    f = Featurizer("IA", EST)
    obs = np.zeros(25)
    f.observe(obs)
    np.testing.assert_array_equal(f.features(), np.zeros(60))
    f.issued(np.array([0.5, -0.5, 1.0, 0.0]))
    obs2 = obs.copy()
    obs2[0:3] = [K0_STEP, 0, -K0_STEP]
    f.observe(obs2)
    x = f.features()
    assert x.shape == (ARMS["IA"],)
    np.testing.assert_array_equal(x[0:3], [0.5, -0.5, 1.0])  # most recent command first
    np.testing.assert_allclose(x[30:33], [1.0, 0.0, -1.0])  # most recent displacement first
    assert not x[3:30].any() and not x[33:].any()


def test_featurizer_ic_layout():
    f = Featurizer("IC", EST)
    f.observe(np.zeros(25))
    x = f.features()
    assert x.shape == (ARMS["IC"],)
    np.testing.assert_array_equal(x[12:17], [1, 0, 0, 0, 0])  # no delay latched yet
    assert x[17] == 0.0


@needs_ckpt
def test_hybrid_normalizer_matches_frozen_on_original_entries():
    from farc.base_policy.normalizer import GoalObservationNormalizer
    from farc.retrain.phase_i import frozen_checkpoint

    frozen = frozen_checkpoint(0)
    norm = GoalObservationNormalizer(25, 3)
    norm.load_state_dict(frozen["normalizer"])
    hyb = HybridNormalizer(norm, 60)
    rng = np.random.default_rng(0)
    obs, goal = rng.normal(size=(7, 85)), rng.normal(size=(7, 3))
    out = hyb(obs, goal)
    ref = norm(obs[:, :25], goal)
    np.testing.assert_array_equal(out[:, :25], ref[:, :25])
    np.testing.assert_array_equal(out[:, 85:], ref[:, 25:])
    np.testing.assert_array_equal(out[:, 25:85], np.clip(obs[:, 25:], -5, 5).astype(np.float32))
    before = hyb.state_dict()
    hyb.update(obs, goal)
    assert (
        hyb.state_dict()["frozen"]["observation"]["count"]
        == before["frozen"]["observation"]["count"]
    )


@needs_ckpt
@pytest.mark.parametrize("arm", ["IA", "IC"])
def test_warm_start_reproduces_the_frozen_actor_and_critics(arm):
    from farc.retrain.phase_i import frozen_checkpoint, warm_start
    from farc.retrain.qrl.algos.networks import GaussianActor, TwinQCritic
    from farc.retrain.qrl.algos.sac import SACAgent
    from farc.retrain.qrl.config import SACConfig

    extra = ARMS[arm]
    frozen = frozen_checkpoint(0)
    sac = SACConfig(hidden_sizes=[256, 256, 256], batch_size=512, gamma=0.98, tau=0.005)
    agent = SACAgent(28 + extra, 4, sac)
    warm_start(agent, frozen, extra)
    ref_actor = GaussianActor(28, 4, [256, 256, 256])
    ref_actor.load_state_dict(frozen["agent"]["actor"])
    ref_critic = TwinQCritic(28, 4, [256, 256, 256])
    ref_critic.load_state_dict(frozen["agent"]["critic"])
    g = torch.Generator().manual_seed(1)
    base = torch.randn(64, 28, generator=g)
    extras = torch.randn(64, extra, generator=g) * 3
    x = torch.cat([base[:, :25], extras, base[:, 25:]], dim=1)
    a = torch.rand(64, 4, generator=g) * 2 - 1
    with torch.no_grad():
        new_mean, _ = agent.actor(x)
        ref_mean, _ = ref_actor(base)
        q_new = agent.critic(x, a)
        q_ref = ref_critic(base, a)
    # zero columns contribute exactly 0.0; summation order may still differ in the last bit
    assert torch.allclose(new_mean, ref_mean, atol=1e-6, rtol=0)
    assert torch.allclose(q_new[0], q_ref[0], atol=1e-5, rtol=0)
    assert torch.equal(agent.log_alpha.detach(), frozen["agent"]["log_alpha"])
    w = agent.actor.state_dict()["network.0.weight"]
    assert torch.count_nonzero(w[:, 25 : 25 + extra]) == 0
    st = agent.actor_optimizer.state_dict()["state"][0]
    assert (
        st["exp_avg"].shape == w.shape
        and torch.count_nonzero(st["exp_avg"][:, 25 : 25 + extra]) == 0
    )


@needs_ckpt
@pytest.mark.parametrize("arm", ["IA", "IC"])
def test_wrapper_features_equal_evaluation_features(arm):
    """Training-wrapper features and evaluation-loop features on the same trajectory."""
    from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
    from farc.delaychange.schedule import ScheduledDelaySession
    from farc.disturbances.config import ConditionSpec
    from farc.disturbances.rollout import FrozenNormalizer
    from farc.envs.factory import make_env
    from farc.retrain.phase_i import PhaseIWrapper

    loaded = load_policy(checkpoint_path("fetch_push_fp32_seed1"))
    policy = deterministic_policy_function(loaded.actor)
    seed = 3_700_100
    for ep in range(3):
        fam, schedule = training_schedule(5, ep)
        session = ScheduledDelaySession(make_env(loaded.env_config), policy,
                                        FrozenNormalizer(loaded.normalizer),
                                        ConditionSpec("x", ()), seed + ep, 1, schedule)  # fmt: skip
        feat = Featurizer(arm, EST)
        eval_features, commands = [], []
        while not session.done:
            view = session.observe()
            assert view.observation is not None
            feat.observe(view.observation)
            eval_features.append(feat.features())
            command = np.clip(view.base_action, -1, 1)
            commands.append(command)
            feat.issued(command)
            session.act(command)
        env = PhaseIWrapper(
            make_env(loaded.env_config), arm, EST, lambda _, f=fam, s=schedule: (f, s)
        )
        obs, _ = env.reset(seed=seed + ep)
        np.testing.assert_array_equal(obs["observation"][25:], eval_features[0])
        for t, command in enumerate(commands[:-1]):
            obs, *_ = env.step(command)
            np.testing.assert_array_equal(obs["observation"][25:], eval_features[t + 1],
                                          err_msg=f"{arm} {fam} t={t + 1}")  # fmt: skip
