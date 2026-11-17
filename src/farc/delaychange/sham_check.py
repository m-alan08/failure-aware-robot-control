"""G2 exploratory sham check (docs/PHASE_G2_SHAM_CHECK.md): C against a random-timing sham."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.dynamics import load_dynamics
from farc.contact.episodes import learned_predictor, phase_e_predictor
from farc.delaychange.schedule import ScheduledDelaySession
from farc.delaychange.stage3 import Cell, G2Config, evaluation_cells, run_g2_episode
from farc.delaychange.stage3_analysis import paired
from farc.disturbances.config import ConditionSpec
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.supervision.seeds import G2_SEED_BLOCKS, check_g2_seeds

SHAM_SEED = 20261604
CONFIGURATIONS = ("contact_learned", "cand_C_sequential", "sham_C_yoked", "sham_C_random")
SOURCE = "cand_C_sequential"


def check_cells() -> list[Cell]:
    return [c for c in evaluation_cells() if c.family in ("T", "I")]


def exploratory_seeds() -> list[int]:
    seeds = list(range(*G2_SEED_BLOCKS["exploratory"]))
    check_g2_seeds(seeds, "exploratory")
    return seeds


def random_rotation(mask: np.ndarray, policy_seed: int, cell_index: int,
                    episode_seed: int) -> np.ndarray:  # fmt: skip
    """C's mask rotated circularly by a seeded offset in 1 .. n-1: same count, new timing."""
    n = len(mask)
    if n < 2:
        return mask.copy()
    rng = np.random.default_rng([SHAM_SEED, policy_seed, cell_index, episode_seed])
    return np.roll(mask, int(rng.integers(1, n)))


def run_task(args: tuple[G2Config, str, Cell]):
    config, checkpoint, cell = args
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    actor = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    policy = deterministic_policy_function(loaded.actor)
    normalizer, env = FrozenNormalizer(loaded.normalizer), make_env(loaded.env_config)
    est_cfg = EstimatorConfig(**config.estimator)
    model = load_dynamics(resolve(config.model), config.model_sha256)
    preds = {"learned": learned_predictor(model, move_object=True),
             "phase_e": phase_e_predictor(est_cfg.nominal_gain)}  # fmt: skip
    seeds = exploratory_seeds()
    cell_index = [c.name for c in evaluation_cells()].index(cell.name)
    masks: dict[int, np.ndarray] = {}
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for sham_phase in (False, True):
        for seed in seeds:
            for name in CONFIGURATIONS:
                if name.startswith("sham_") != sham_phase:
                    continue
                mask = None
                if name == "sham_C_yoked":  # as in Stage 3: the next seed of the same condition
                    mask = masks[seeds[(seeds.index(seed) + 1) % len(seeds)]]
                elif name == "sham_C_random":
                    mask = random_rotation(masks[seed], loaded.training_seed, cell_index, seed)
                session = ScheduledDelaySession(
                    env, policy, normalizer, ConditionSpec(cell.name, cell.extra), seed,
                    config.perturbation_seed, cell.schedule, "replay_drop")  # fmt: skip
                # Both shams run the contact-aware controller with a forced-off mask; the
                # Stage-3 runner takes the mask under the yoked name.
                runner = "sham_C_yoked" if name.startswith("sham_") else name
                result = run_g2_episode(session, ActuatorEstimator(est_cfg), runner, preds,
                                        config.constants, mask)  # fmt: skip
                if name == SOURCE:
                    masks[seed] = np.array([r["suspended"] for r in result["rows"]])
                key = {
                    "configuration": name,
                    "policy_seed": loaded.training_seed,
                    "condition": cell.name,
                    "family": cell.family,
                    "episode_seed": seed,
                }
                episodes.append({**key, "success": result["success"],
                                 "final_distance": result["final_distance"],
                                 "suspended_steps": int(mask.sum()) if mask is not None
                                 else int(masks[seed].sum()) if name == SOURCE else 0})  # fmt: skip
                steps.extend({**key, **{k: r[k] for k in ("t", "delay_true", "delay_used",
                                                          "active", "suspended")}}
                             for r in result["rows"])  # fmt: skip
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor[k]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes, steps


COMPARISONS = (
    ("E1", "cand_C_sequential", "sham_C_random"),
    ("E2", "sham_C_yoked", "sham_C_random"),
    ("E3", "sham_C_random", "contact_learned"),
    ("E4", "cand_C_sequential", "sham_C_yoked"),
    ("E5", "cand_C_sequential", "contact_learned"),
)


def derive_check(episodes: pd.DataFrame, steps: pd.DataFrame, config: G2Config):
    n, s = config.bootstrap_resamples, config.bootstrap_seed
    rows = []
    for label, a, b in COMPARISONS:
        scopes = [("T+I", episodes)]
        if label == "E1":
            scopes += [(f, episodes[episodes["family"] == f]) for f in ("T", "I")]
        for scope, frame in scopes:
            rows.append({"test": label, "scope": scope, "configuration": a, "reference": b,
                         **paired(frame, a, b, 0.0, n, s)})  # fmt: skip
    table = pd.DataFrame(rows)
    e1 = table[(table["test"] == "E1") & (table["scope"] == "T+I")].iloc[0]
    # Descriptive: how many steps each controller had compensation on while a delay was present.
    st = steps.assign(
        on_delay=steps["active"].astype(bool) & (steps["delay_true"] > 0),
        on_no_delay=steps["active"].astype(bool) & (steps["delay_true"] == 0),
    )
    activity = (st.groupby(["configuration", "family"])[["on_delay", "on_no_delay"]].mean()
                .reset_index())  # fmt: skip
    success = episodes.groupby(["configuration", "family"])["success"].mean().unstack() * 100
    suspended = episodes.groupby("configuration")["suspended_steps"].agg(["mean", "sum"])
    summary = {
        "E1_estimate": float(e1["estimate"]), "E1_ci": [float(e1["ci_low"]),
                                                         float(e1["ci_high"])],
        "E1_p_one_sided": float(e1["p_one_sided"]),
        "E1_result": "timing matters" if e1["p_one_sided"] <= config.alpha_one_sided
        else "not shown",
        "episodes": len(episodes),
        "seeds": [min(exploratory_seeds()), max(exploratory_seeds())],
    }  # fmt: skip
    return {
        "comparisons": table,
        "success_by_family": success.reset_index(),
        "suspended_steps": suspended.reset_index(),
        "activity": activity,
    }, summary


def check_config(config: G2Config) -> G2Config:
    """The Stage-3 config with its pinned constants; only the seeds differ (exploratory)."""
    return replace(config, output=config.output + "_sham_check")


DERIVED = ("comparisons", "success_by_family", "suspended_steps", "activity")
