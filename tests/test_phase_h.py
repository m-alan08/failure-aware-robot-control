"""Phase H: seeds, the training delay distribution, the delay/history wrapper and the port."""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from farc.adaptation.experiment import AdaptationConfigError
from farc.delaychange.random_schedules import draw_pulses, g3_cells
from farc.retrain.phase_h import FAMILIES, augment, load_phase_h, training_schedule
from farc.supervision.seeds import check_h_seeds

CONFIG = "configs/retrain/phase_h.json"


def test_seed_blocks_and_guard():
    c = load_phase_h(CONFIG)
    assert c.seeds("debugging") == list(range(3_700_000, 3_700_010))
    assert c.seeds("evaluation") == list(range(3_701_000, 3_701_300))
    for bad in ([4_000_600], [4_000_000], [3_690_005], [3_681_500], [3_701_300]):
        with pytest.raises(AdaptationConfigError):
            check_h_seeds(bad, "evaluation")


def test_training_config_is_the_upstream_fetch_push_config():
    c = load_phase_h(CONFIG)
    t = c.training
    assert t["sac"] == {"hidden_sizes": [256, 256, 256], "batch_size": 512, "gamma": 0.98,
                        "tau": 0.005}  # fmt: skip
    assert t["her"] == {"replay_k": 4, "buffer_episodes": 20000}
    assert t["train"]["total_steps"] == 1_000_000 and t["train"]["updates_per_step"] == 0.5


def test_training_distribution_is_deterministic_uniform_and_separate_from_g3():
    draws = [training_schedule(0, e) for e in range(6000)]
    assert draws[17][1] == training_schedule(0, 17)[1]
    counts = {f: sum(d[0] == f for d in draws) for f in FAMILIES}
    assert all(abs(n - 1000) < 120 for n in counts.values()), counts
    assert [training_schedule(0, e) for e in range(50)] != [training_schedule(1, e)
                                                            for e in range(50)]  # fmt: skip
    # the default G3/G4 generator is unchanged by the new parameter
    cell = g3_cells()[0]
    assert draw_pulses(cell, 0, 3_681_000) == draw_pulses(cell, 0, 3_681_000, 20261700)
    assert draw_pulses(cell, 0, 3_681_000) != draw_pulses(cell, 0, 3_681_000, 20261900)


def test_augment_orders_most_recent_first_with_zero_padding():
    obs = np.arange(25.0)
    issued = [np.array([1, 2, 3, 9.0]), np.array([4, 5, 6, 9.0])]
    out = augment(obs, issued, 4)
    assert out.shape == (37,)
    np.testing.assert_array_equal(out[25:31], [4, 5, 6, 1, 2, 3])
    np.testing.assert_array_equal(out[31:], np.zeros(6))
    assert augment(obs, issued, 0) is obs


def test_ported_trainer_differs_from_upstream_only_by_the_marked_hook():
    src = Path("src/farc/retrain/qrl/training.py").read_text()
    marked = [line for line in src.splitlines()
              if "PHASE H CHANGE" in line and not line.lstrip().startswith("#")]  # fmt: skip
    assert len(marked) == 2
    fn = next(
        n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "train"
    )
    assert [a.arg for a in fn.args.args][-1] == "env_wrapper"


# --- simulator --------------------------------------------------------------------------------

from farc.paths import checkpoint_path  # noqa: E402

CHECKPOINT = checkpoint_path("fetch_push_fp32_seed1")
needs_sim = pytest.mark.skipif(not CHECKPOINT.exists(), reason="checkpoint absent")


@needs_sim
@pytest.mark.parametrize("episode", [0, 1, 2, 3, 4, 5, 6, 7])
def test_wrapper_executes_exactly_what_the_session_executes(episode):
    """Same initial state and the same issued commands (the frozen policy's, through the
    session): the wrapper must execute identical actions and produce identical observations."""
    from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
    from farc.delaychange.schedule import ScheduledDelaySession
    from farc.disturbances.config import ConditionSpec
    from farc.disturbances.rollout import FrozenNormalizer
    from farc.envs.factory import make_env
    from farc.retrain.phase_h import DelayHistoryWrapper

    family, schedule = training_schedule(7, episode)
    loaded = load_policy(CHECKPOINT)
    policy = deterministic_policy_function(loaded.actor)
    seed = 3_700_000 + episode  # Phase H debugging block
    session = ScheduledDelaySession(make_env(loaded.env_config), policy,
                                    FrozenNormalizer(loaded.normalizer),
                                    ConditionSpec("x", ()), seed, 1, schedule)  # fmt: skip
    issued, executed_s, obs_s = [], [], []
    while not session.done:
        view = session.observe()
        command = np.clip(view.base_action, -1, 1)
        issued.append(command)
        executed_s.append(session.act(command))
        obs_s.append(np.asarray(session.observation["observation"]).copy())
    env = DelayHistoryWrapper(make_env(loaded.env_config), 4, lambda _: (family, schedule))
    env.reset(seed=seed)
    for t, command in enumerate(issued):
        obs, *_ = env.step(command)
        np.testing.assert_array_equal(env.last_executed, executed_s[t], err_msg=f"t={t}")
        np.testing.assert_array_equal(obs["observation"][:25], obs_s[t], err_msg=f"{family} t={t}")
        np.testing.assert_array_equal(obs["observation"][25:28], command[:3])
