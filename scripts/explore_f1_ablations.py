"""EXPLORATORY (post hoc) Phase F1 ablations on fresh seeds 3,654,000-099.

The pre-registered gripper-only ablation produced an internally inconsistent observation
(updated gripper velocity, stale object-relative velocity), so it cannot attribute the
contact-aware gain. This script runs two properly built ablations next to the Phase E and
contact-aware predictors on a fresh seed block. It is not confirmatory, and its results do
not change the pre-registered verdict.
"""

from __future__ import annotations

import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import torch

from farc.adaptation.delay_eval import paired_difference
from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.data import build_conditions, check_seeds
from farc.contact.dynamics import load_dynamics
from farc.contact.episodes import (
    learned_gripper_consistent,
    learned_gripper_position_only,
    learned_predictor,
    phase_e_predictor,
    run_episode,
)
from farc.contact.evaluation import load_contact_evaluation
from farc.counterfactual.artifacts import METADATA, analyze_existing, write_raw
from farc.counterfactual.session import EpisodeSession
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256

SEEDS = list(range(3_654_000, 3_654_100))
CONFIGS = ("phase_e_predictor", "contact_learned", "learned_gripper_position_only",
           "learned_gripper_consistent")  # fmt: skip
OUT = "$FARC_DATA_DIR/experiments/phase_f1_exploratory_ablations"
RAW = ("episodes.csv.gz", METADATA)


def run(checkpoint: str) -> list[dict]:
    config = load_contact_evaluation("configs/contact/phase_f1_eval.json")
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    policy = deterministic_policy_function(loaded.actor)
    normalizer = FrozenNormalizer(loaded.normalizer)
    env = make_env(loaded.env_config)
    model = load_dynamics(resolve(config.model), config.model_sha256)
    k0 = EstimatorConfig(**config.estimator).nominal_gain
    preds = {"phase_e_predictor": phase_e_predictor(k0),
             "contact_learned": learned_predictor(model, True),
             "learned_gripper_position_only": learned_gripper_position_only(model),
             "learned_gripper_consistent": learned_gripper_consistent(model)}  # fmt: skip
    rows = []
    conditions = [c for c in build_conditions(config.conditions_from, config.conditions)
                  if c.name in config.delay_conditions]  # fmt: skip
    for condition in conditions:
        for seed in SEEDS:
            for name in CONFIGS:
                session = EpisodeSession(env, policy, normalizer, condition, seed,
                                         config.perturbation_seed)  # fmt: skip
                r = run_episode(session, ActuatorEstimator(EstimatorConfig(**config.estimator)),
                                preds[name], condition)  # fmt: skip
                rows.append(
                    {
                        "configuration": name,
                        "policy_seed": loaded.training_seed,
                        "condition": condition.name,
                        "episode_seed": seed,
                        "success": r["success"],
                        "final_distance": r["final_distance"],
                    }
                )
    return rows


def derive(episodes: pd.DataFrame):
    rows = []
    pairs = [(a, "phase_e_predictor") for a in CONFIGS[1:]] + [
        (a, "contact_learned") for a in CONFIGS[2:]]  # fmt: skip
    for scope, frame in [("pooled", episodes), *episodes.groupby("condition")]:
        for a, b in pairs:
            est, low, high, p, n = paired_difference(frame, a, b, 10_000, 20261503)
            rows.append({"scope": scope, "configuration": a, "reference": b, "estimate": est,
                         "ci_low": low, "ci_high": high, "episodes": n})  # fmt: skip
    rates = episodes.groupby(["condition", "configuration"])["success"].mean().unstack()
    return {"differences": pd.DataFrame(rows), "success_rates": rates.reset_index()}, {
        "label": "EXPLORATORY, post hoc",
        "seeds": [SEEDS[0], SEEDS[-1]],
    }


def main() -> None:
    check_seeds(SEEDS, "exploratory")
    config = load_contact_evaluation("configs/contact/phase_f1_eval.json")
    out = resolve(OUT)
    if not (out / RAW[0]).exists():
        started = time.time()
        with Pool(3) as pool:
            parts = pool.map(run, list(config.checkpoints))
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=False).stdout.strip()  # fmt: skip
        write_raw(out, {RAW[0]: pd.DataFrame([r for p in parts for r in p])},
                  {"label": "EXPLORATORY post-hoc ablations", "git_commit": commit,
                   "model_sha256": config.model_sha256, "seeds": SEEDS,
                   "checkpoint_sha256": {c: file_sha256(resolve(c)) for c in config.checkpoints},
                   "seconds": round(time.time() - started, 1)})  # fmt: skip
    tables, _ = analyze_existing(out, derive, Path("results/phase_f1_exploratory_ablations"),
                                 ("differences", "success_rates"), RAW)  # fmt: skip
    d = tables["differences"]
    print(d[d.scope == "pooled"].round(4).to_string(index=False))
    print(tables["success_rates"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
