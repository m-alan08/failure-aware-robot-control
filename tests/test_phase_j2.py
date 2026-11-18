"""Phase J-2 Stage-0 checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from farc.adaptation.experiment import AdaptationConfigError
from farc.paths import checkpoint_path
from farc.retrain.phase_i import training_schedule
from farc.retrain.phase_j2 import load_config, seeds
from farc.supervision.seeds import check_j2_seeds

CONFIG = "configs/retrain/phase_j2.json"
needs_ckpt = pytest.mark.skipif(not checkpoint_path("fetch_push_fp32_seed1").exists(),
                                reason="checkpoint absent")  # fmt: skip


def test_config_and_seed_blocks():
    v = load_config(CONFIG)
    assert seeds(v, "evaluation") == list(range(3_704_000, 3_704_300))
    phase_i = json.loads(Path("configs/retrain/phase_i.json").read_text())
    assert v["training"] == phase_i["training"]  # identical to Phase I
    for bad in ([4_000_600], [3_703_000], [3_700_200], [3_704_300]):
        with pytest.raises(AdaptationConfigError):
            check_j2_seeds(bad, "evaluation")


def test_phase_i_schedule_default_unchanged_and_j2_seed_differs():
    a = [training_schedule(0, e) for e in range(50)]
    assert a == [training_schedule(0, e, 20262000) for e in range(50)]
    assert a != [training_schedule(0, e, 20262200) for e in range(50)]


@needs_ckpt
def test_noisy_wrapper_with_zero_sigma_equals_phase_i_and_noise_matches_model():
    from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
    from farc.envs.factory import make_env
    from farc.retrain.phase_i import PhaseIWrapper
    from farc.retrain.phase_j2 import NoisyPhaseIWrapper

    est = load_config(CONFIG)["estimator"]
    loaded = load_policy(checkpoint_path("fetch_push_fp32_seed1"))
    sched = training_schedule(1, 0, 20262200)
    plain = PhaseIWrapper(make_env(loaded.env_config), "IA", est, lambda _: sched)
    zero = NoisyPhaseIWrapper(make_env(loaded.env_config), "IA", est, lambda _: sched, 1, 0.0)
    o1, _ = plain.reset(seed=3_700_300)
    o2, _ = zero.reset(seed=3_700_300)
    np.testing.assert_array_equal(o1["observation"], o2["observation"])
    policy = deterministic_policy_function(loaded.actor)
    for _ in range(20):
        a = policy(loaded.normalizer(o1["observation"][:25], o1["desired_goal"])[None])[0]
        o1, *_ = plain.step(a)
        o2, *_ = zero.step(a)
        np.testing.assert_array_equal(o1["observation"], o2["observation"])
    # noise statistics: positions only, relative recomputed, velocities untouched
    noisy = NoisyPhaseIWrapper(make_env(loaded.env_config), "IA", est, lambda _: sched, 1, 0.01)
    noisy.sigma_max = 0.01
    diffs = []
    for ep in range(30):
        obs, _ = noisy.reset(seed=3_700_300 + ep % 10)
        raw = cast(Any, noisy.env.unwrapped)._get_obs()["observation"]
        d = obs["observation"][:25] - raw
        diffs.append(d)
        assert np.allclose(d[6:9], d[3:6] - d[0:3], atol=1e-6)
        assert np.all(d[20:25] == 0) and np.all(d[9:20] == 0)
        assert 0.0 <= noisy.sigma <= 0.01
    assert np.std(np.array(diffs)[:, 0:3]) > 0.002


def test_evaluation_cells_have_no_duplicates():
    from farc.retrain.evaluation_j2 import DUPLICATES, j2_cells

    cells = j2_cells()
    assert len(cells) == 46 and not {c.name for c in cells} & set(DUPLICATES)
    specs = [(c.extra, c.fixed.pulses if c.fixed is not None else ("random", c.index))
             for c in cells]  # fmt: skip
    assert len({repr(s) for s in specs}) == len(specs)
