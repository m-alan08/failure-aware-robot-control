"""Unit tests for the T-M and T-O families (residual RL pre-registration §3).

These are software tests only. They use seeds 0-2, which lie outside every registered seed
block, and never touch the R debugging, selection, evaluation or held-out blocks. The
pre-registered §3 checks proper are run on R debugging seeds only after plan approval.
"""

from __future__ import annotations

import numpy as np
import pytest

from farc.envs.observation_layout import OBJECT
from farc.residual.disturbances import (
    CorrelatedObservationNoise,
    ObjectDynamics,
    ObjectDynamicsPatch,
)


def test_object_dynamics_sampling_ranges():
    rng = np.random.default_rng(0)
    draws = [ObjectDynamics.sample(rng) for _ in range(2000)]
    mass = np.array([d.mass_scale for d in draws])
    fric = np.array([d.friction_scale for d in draws])
    assert mass.min() >= 0.5 and mass.max() <= 2.0
    assert fric.min() >= 0.5 and fric.max() <= 1.5
    assert ObjectDynamics().is_nominal
    with pytest.raises(ValueError):
        ObjectDynamics(0.0, 1.0)


def test_ar1_noise_autocorrelation_and_scale():
    rng = np.random.default_rng(1)
    gen = CorrelatedObservationNoise(OBJECT, sigma=0.01, rng=rng)
    series = np.array([gen.noise() for _ in range(20_000)])
    x = series[:, 0]
    lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(lag1 - 0.8) <= 0.05  # pre-registered effect check
    assert abs(x.std() - 0.01) < 0.001
    pos = OBJECT.slices
    rel = series[:, pos["object_relative_position"]]
    expected = series[:, pos["object_position"]] - series[:, pos["gripper_position"]]
    assert np.array_equal(rel, expected)
    untouched = series[:, pos["finger_position"]]
    assert not untouched.any()


def test_ar1_zero_sigma_is_identity_and_input_untouched():
    gen = CorrelatedObservationNoise(OBJECT, sigma=0.0, rng=np.random.default_rng(2))
    obs = np.random.default_rng(3).standard_normal(OBJECT.size)
    keep = obs.copy()
    for _ in range(5):
        out = gen.apply(obs)
        assert np.array_equal(out, obs) and out is not obs
    assert np.array_equal(obs, keep)


def test_ar1_common_random_numbers_across_sigma():
    a = CorrelatedObservationNoise(OBJECT, 0.005, np.random.default_rng(4))
    b = CorrelatedObservationNoise(OBJECT, 0.010, np.random.default_rng(4))
    for _ in range(10):
        assert np.allclose(2 * a.noise(), b.noise())


# ------------------------------------------------------------ simulator (unit seeds only) --

gym = pytest.importorskip("gymnasium")
pytest.importorskip("gymnasium_robotics")


def _make_env():
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    return gym.make("FetchPush-v4")


def _scripted_commands(seed: int = 0, steps: int = 40) -> np.ndarray:
    """Record a scripted push on the nominal model: approach behind the block, then push."""
    env = _make_env()
    obs, _ = env.reset(seed=seed)
    commands = []
    for t in range(steps):
        o = obs["observation"]
        grip, block = o[0:3], o[3:6]
        target = block + ([-0.05, 0.0, 0.0] if t < 15 else [0.08, 0.0, 0.0])
        a = np.zeros(4)
        a[:3] = np.clip(10.0 * (target - grip), -1.0, 1.0)
        commands.append(a)
        obs, *_ = env.step(a)
    env.close()
    return np.array(commands)


COMMANDS = None


def _rollout(dynamics: ObjectDynamics | None, seed: int = 0):
    global COMMANDS
    if COMMANDS is None:
        COMMANDS = _scripted_commands(seed)
    env = _make_env()
    obs, _ = env.reset(seed=seed)
    model = env.unwrapped.model
    actions = COMMANDS  # one fixed command sequence, replayed open loop
    traj = [obs["observation"][3:6].copy()]

    def run():
        for a in actions:
            o, *_ = env.step(a)
            traj.append(o["observation"][3:6].copy())

    if dynamics is None:
        run()
    else:
        before = (model.body_mass.copy(), model.body_inertia.copy(), model.geom_friction.copy())
        with ObjectDynamicsPatch(model, dynamics):
            run()
        after = (model.body_mass, model.body_inertia, model.geom_friction)
        assert all(np.array_equal(x, y) for x, y in zip(before, after, strict=True))
    env.close()
    return np.array(traj)


def test_object_dynamics_nominal_is_bitwise_identity():
    assert np.array_equal(_rollout(None), _rollout(ObjectDynamics()))


def test_object_dynamics_changes_object_trajectory():
    base = _rollout(None)
    assert np.abs(base[-1] - base[0]).max() > 1e-3, "fixture must actually push the object"
    heavy = _rollout(ObjectDynamics(2.0, 1.5))
    light = _rollout(ObjectDynamics(0.5, 0.5))
    assert not np.array_equal(base, heavy) and not np.array_equal(base, light)
