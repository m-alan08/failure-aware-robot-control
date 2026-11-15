"""Phase D: counterfactual residual corrections from exact simulator snapshots.

For each episode the frozen base policy is run once (branch A). At selected decision steps the
full state is snapshotted after the controller has observed step t; for every candidate the
snapshot is restored, the residual is applied for ``duration`` steps (or until the episode
ends), and the base policy continues to the end (branch B). Branch A from any state on the
base trajectory is the base episode itself, which ``verify_branch_a`` checks by replay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.counterfactual.residuals import STRATEGIES, Candidate, candidate_grid
from farc.counterfactual.session import EpisodeSession
from farc.disturbances.config import SEED_RANGES, ConditionSpec, load_experiment
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.prediction.dataset import Normalizer
from farc.prediction.evaluation import load_predictor
from farc.recovery.controller import LearnedTrigger

ENV_ID = "FetchPush-v4"
USED_SEED_RANGES = {
    "phase1_and_phase_b_validation": (3_500_000, 3_500_100),
    "phase_c_controller_debugging": (3_600_000, 3_600_030),
    "phase_c_development": (3_610_000, 3_610_100),
}


class CounterfactualConfigError(ValueError):
    pass


@dataclass(frozen=True)
class CounterfactualExperiment:
    name: str
    conditions_from: str
    checkpoints: tuple[str, ...]
    episode_seed_offset: int
    episodes: int
    perturbation_seed: int
    branch_steps: tuple[int, ...]
    include_first_warning: bool
    predictor: str
    learned_threshold: float
    alphas: tuple[float, ...]
    durations: tuple[int, ...]
    selection_fraction: float
    split_seed: int
    bootstrap_resamples: int
    bootstrap_seed: int
    output: str
    excluded_seed_ranges: tuple[tuple[int, int], ...] = ()
    strategies: tuple[str, ...] = STRATEGIES

    @property
    def episode_seeds(self) -> list[int]:
        return list(range(self.episode_seed_offset, self.episode_seed_offset + self.episodes))


def load_counterfactual_experiment(path: str | Path) -> CounterfactualExperiment:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    for key in ("checkpoints", "branch_steps", "alphas", "durations", "strategies"):
        if key in values:
            values[key] = tuple(values[key])
    values["excluded_seed_ranges"] = tuple(tuple(r) for r in values.get("excluded_seed_ranges", ()))
    config = CounterfactualExperiment(**values)
    seeds = config.episode_seeds
    low, high = SEED_RANGES["validation"]
    if not (low <= seeds[0] and seeds[-1] < high):
        raise CounterfactualConfigError(f"seeds must lie in the validation range [{low}, {high})")
    ranges = {**USED_SEED_RANGES, **{f"excluded_{i}": r for i, r in
              enumerate(config.excluded_seed_ranges)}}  # fmt: skip
    for name, (a, b) in ranges.items():
        if seeds[0] < b and a <= seeds[-1]:
            raise CounterfactualConfigError(f"seeds overlap {name} [{a}, {b})")
    if not all(0 < a <= 1 for a in config.alphas) or not all(d >= 1 for d in config.durations):
        raise CounterfactualConfigError("alphas must be in (0, 1] and durations >= 1")
    return config


def push_conditions(config: CounterfactualExperiment) -> list[ConditionSpec]:
    source = load_experiment(resolve(config.conditions_from))
    return [c for c in source.conditions if c.applies_to(ENV_ID)]


def run_branch(session: EpisodeSession, candidate: Candidate) -> dict[str, float]:
    """From the current (observed) state: residual for ``duration`` steps, then base."""
    applied, magnitude = 0, 0.0
    while not session.done:
        view = session.observe()
        if applied < candidate.duration:
            command = candidate.command(view.gripper, view.obj, view.goal, view.base_action)
            magnitude += float(np.linalg.norm(command[:3] - view.base_action[:3]))
            applied += 1
        else:
            command = view.base_action
        session.act(command)
    return {
        "success": session.final_success,
        "final_distance": session.final_distance,
        "applied_steps": applied,
        "mean_residual_norm": magnitude / max(applied, 1),
    }


def run_episode(
    session: EpisodeSession,
    candidates: list[Candidate],
    branch_steps: tuple[int, ...],
    trigger: LearnedTrigger | None,
    include_first_warning: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Base run with snapshots, then every candidate from every snapshot."""
    snapshots, states = [], []
    first_warning = None
    while not session.done:
        view = session.observe()
        t = view.step
        score = trigger.score(session.history) if trigger is not None else float("nan")
        warned = trigger is not None and score > trigger.threshold
        if warned and first_warning is None:
            first_warning = t
        is_grid = t in branch_steps
        is_warning = include_first_warning and warned and first_warning == t
        if is_grid or is_warning:
            snapshots.append(session.snapshot())
            distance = float(np.linalg.norm(view.obj - view.goal))
            states.append({"t": t, "grid_state": is_grid, "first_warning_state": is_warning,
                           "score": score, "warning_now": warned,
                           "warned_so_far": first_warning is not None,
                           "view_object_goal_distance": distance})  # fmt: skip
        session.act(view.base_action)
    base = {"success": session.final_success, "final_distance": session.final_distance,
            "first_warning_step": first_warning}  # fmt: skip
    rows = []
    for snap, state in zip(snapshots, states, strict=True):
        for candidate in candidates:
            session.restore(snap)
            outcome = run_branch(session, candidate)
            rows.append({**state, "candidate": candidate.name, "strategy": candidate.strategy,
                         "alpha": candidate.alpha, "duration": candidate.duration,
                         "base_success": base["success"],
                         "base_final_distance": base["final_distance"], **outcome})  # fmt: skip
    return base, rows


def verify_branch_a(session: EpisodeSession, steps: tuple[int, ...]) -> None:
    """Restoring a snapshot and continuing the base policy must reproduce the base episode."""
    snaps = {}
    while not session.done:
        view = session.observe()
        if view.step in steps:
            snaps[view.step] = session.snapshot()
        session.act(view.base_action)
    reference = (session.successes.copy(), session.final_distance, session.state_digest())
    for snap in snaps.values():
        session.restore(snap)
        session.run_base()
        again = (session.successes, session.final_distance, session.state_digest())
        if not (np.array_equal(reference[0], again[0]) and reference[1:] == again[1:]):
            raise RuntimeError("branch A replay differs from the base episode")


def run_policy(args: tuple[CounterfactualExperiment, str, Normalizer]) -> list[dict[str, Any]]:
    config, checkpoint, normalizer = args
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    actor_state = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    policy = deterministic_policy_function(loaded.actor)
    frozen = FrozenNormalizer(loaded.normalizer)
    env = make_env(loaded.env_config)
    trigger = LearnedTrigger(load_predictor(resolve(config.predictor)), normalizer,
                             config.learned_threshold)  # fmt: skip
    candidates = candidate_grid(config.alphas, config.durations, config.strategies)
    rows: list[dict[str, Any]] = []
    for condition in push_conditions(config):
        for seed in config.episode_seeds:
            if seed == config.episode_seeds[0]:  # exact-replay check on every condition
                verify_branch_a(
                    EpisodeSession(env, policy, frozen, condition, seed, config.perturbation_seed),
                    config.branch_steps,
                )
            session = EpisodeSession(env, policy, frozen, condition, seed, config.perturbation_seed)
            base, branch_rows = run_episode(session, candidates, config.branch_steps, trigger,
                                            config.include_first_warning)  # fmt: skip
            label = {"policy_seed": loaded.training_seed, "condition": condition.name,
                     "disturbance_type": condition.disturbances[0].type
                     if condition.disturbances else "none", "episode_seed": seed,
                     "first_warning_step": base["first_warning_step"]}  # fmt: skip
            rows.extend({**label, **r} for r in branch_rows)
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor_state[k]):
            raise RuntimeError("the frozen actor changed")
    return rows


# --- analysis -----------------------------------------------------------------------------

STATE_KEYS = ["policy_seed", "condition", "episode_seed", "t"]


def split_seeds(seeds: np.ndarray, selection_fraction: float, seed: int):
    unique = np.unique(seeds)
    order = np.random.default_rng(seed).permutation(unique)
    n = int(round(selection_fraction * len(unique)))
    return np.sort(order[:n]), np.sort(order[n:])


def candidate_metrics(frame: pd.DataFrame) -> dict[str, float]:
    a, b = frame["base_success"].to_numpy(bool), frame["success"].to_numpy(bool)
    return {
        "states": len(frame),
        "base_success_rate": a.mean(),
        "branch_success_rate": b.mean(),
        "success_difference": b.mean() - a.mean(),
        "base_failures": int((~a).sum()),
        "rescued": int((~a & b).sum()),
        "rescued_fraction": (~a & b).sum() / max((~a).sum(), 1),
        "base_successes": int(a.sum()),
        "harmed": int((a & ~b).sum()),
        "harmed_fraction": (a & ~b).sum() / max(a.sum(), 1),
        "mean_distance_improvement": float(
            (frame["base_final_distance"] - frame["final_distance"]).mean()
        ),
        "mean_applied_steps": float(frame["applied_steps"].mean()),
        "mean_residual_norm": float(frame["mean_residual_norm"].mean()),
    }


def clustered_interval(
    frame: pd.DataFrame, value: str, resamples: int, rng
) -> tuple[float, float, float]:
    """Mean of ``value`` with an episode-seed cluster bootstrap (a seed brings all of its
    policies, conditions and branch states)."""
    groups = frame.groupby("episode_seed")[value]
    sums, counts = groups.sum().to_numpy(float), groups.count().to_numpy(float)
    estimate = sums.sum() / counts.sum()
    picks = rng.integers(0, len(sums), size=(resamples, len(sums)))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(estimate), float(low), float(high)


def candidate_table(frame: pd.DataFrame, by: list[str], resamples: int, seed: int) -> pd.DataFrame:
    frame = frame.assign(
        diff=frame["success"].astype(float) - frame["base_success"].astype(float),
        dist_gain=frame["base_final_distance"] - frame["final_distance"],
    )
    rows = []
    for key, sub in frame.groupby(["candidate", *by]):
        key = key if isinstance(key, tuple) else (key,)
        rng = np.random.default_rng(seed)
        est, low, high = clustered_interval(sub, "diff", resamples, rng)
        _, dlow, dhigh = clustered_interval(sub, "dist_gain", resamples, rng)
        first = sub.iloc[0]
        rows.append({**dict(zip(["candidate", *by], key, strict=True)),
                     "strategy": first["strategy"], "alpha": first["alpha"],
                     "duration": first["duration"], **candidate_metrics(sub),
                     "diff_ci_low": low, "diff_ci_high": high,
                     "distance_gain_ci_low": dlow, "distance_gain_ci_high": dhigh})  # fmt: skip
    return pd.DataFrame(rows)


def oracle_table(frame: pd.DataFrame) -> pd.DataFrame:
    """ORACLE (uses counterfactual outcomes): per state, success if any candidate succeeds."""
    per_state = frame.groupby(STATE_KEYS + ["disturbance_type"]).agg(
        base_success=("base_success", "first"), any_success=("success", "max"),
        all_success=("success", "min"),
    ).reset_index()  # fmt: skip
    rows: list[dict[str, Any]] = []
    for key, sub in [("pooled", per_state), *per_state.groupby("disturbance_type")]:
        a = sub["base_success"].to_numpy(bool)
        rows.append({"group": key, "states": len(sub), "base_success_rate": a.mean(),
                     "oracle_any_candidate_success_rate": sub["any_success"].mean(),
                     "base_failures": int((~a).sum()),
                     "oracle_rescuable_failures": int((~a & sub["any_success"]).sum()),
                     "base_successes_harmed_by_every_candidate":
                         int((a & ~sub["any_success"].astype(bool)).sum()),
                     "base_successes_harmed_by_some_candidate":
                         int((a & ~sub["all_success"].astype(bool)).sum())})  # fmt: skip
    return pd.DataFrame(rows)
