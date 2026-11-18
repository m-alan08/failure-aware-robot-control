"""Phase J diagnostics: helpers behave as the report states (no data needed)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SPEC = importlib.util.spec_from_file_location("phase_j", Path("scripts/phase_j_diagnostics.py"))
assert SPEC is not None and SPEC.loader is not None
pj = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pj)


def test_per_step_difference_noise_matches_the_analytic_value():
    rng = np.random.default_rng(0)
    clean = np.cumsum(rng.normal(0, 0.01, (20000, 3)), axis=0)
    noisy = clean + rng.normal(0, 0.01, clean.shape)
    rep = "A per-step difference (deployed)"
    err = (pj._rep(rep, noisy) - pj._rep(rep, clean))[1:]
    assert abs(err.std() - np.sqrt(2) * 0.01 / pj.K0_STEP) < 0.01
    lag1 = np.mean([np.corrcoef(err[:-1, j], err[1:, j])[0, 1] for j in range(3)])
    assert abs(lag1 + 0.5) < 0.02


def test_wider_differences_reduce_noise_as_one_over_m():
    rng = np.random.default_rng(1)
    clean = np.zeros((20000, 3))
    noisy = rng.normal(0, 0.01, clean.shape)
    s1 = np.nanstd(pj._rep("A per-step difference (deployed)", noisy))
    s5 = np.nanstd(pj._rep("C 5-step difference / 5", noisy))
    assert abs(s5 / s1 - 0.2) < 0.01


def test_featurize_uses_the_deployed_featurizer():
    obs = np.zeros((5, 25))
    obs[:, 0] = np.arange(5) * pj.K0_STEP
    cmd = np.ones((5, 4)) * 0.5
    f = pj.featurize("IA", obs, cmd)
    assert f.shape == (5, 60)
    np.testing.assert_allclose(f[4, 30], 1.0)  # most recent displacement, scaled
    np.testing.assert_allclose(f[4, 0:3], 0.5)  # most recent command
