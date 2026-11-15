# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/experiment.py
# (introduced in 0b6b61d, last changed in 09ea4f4). See PROVENANCE.md.
"""Run a disturbance experiment over checkpoints x conditions x fixed episode seeds.

Work is saved per (checkpoint, condition) under ``parts/`` with a ``done.json`` marker written
last, so an interrupted run resumes at the condition level. Episode rows are combined at the
end; full per-step trajectories stay in the parts (one gzip CSV per condition).
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.base_policy.checkpoint import deterministic_policy_function, file_sha256, load_policy
from farc.base_policy.reference_eval import evaluate_policy
from farc.disturbances.config import ConditionSpec, ExperimentConfig
from farc.disturbances.metrics import first_success_step, kaplan_meier_median, recovery_outcome
from farc.disturbances.rollout import (
    EpisodeTrajectory,
    FrozenNormalizer,
    normalizer_digest,
    run_disturbed_episode,
)
from farc.envs.factory import check_goal_env, env_dimensions, make_env
from farc.paths import resolve
from farc.statistics import bootstrap_confidence_interval, hierarchical_bootstrap_interval
from farc.utils.run_info import collect_run_metadata
from farc.utils.seeding import seed_everything

STAGE_NOTES = {
    "pilot": "Exploratory pilot: few episodes, severities fixed before running. Not evidence.",
    "validation": (
        "Validation-range baseline characterization. Severity grid fixed before running. "
        "Final claims need the reserved held-out range."
    ),
    "heldout": "Held-out test range.",
}
GROUP_KEYS = [
    "env_id",
    "condition",
    "disturbance_type",
    "mask",
    "severity",
    "onset",
    "duration",
    "regime",
]


class ResumeConfigMismatch(RuntimeError):
    pass


def config_fingerprint(config: ExperimentConfig) -> str:
    values = config.to_dict()
    values.pop("output_dir")
    values.pop("name")
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def _primary(condition: ConditionSpec) -> dict[str, Any]:
    if condition.is_baseline:
        values: dict[str, Any] = {
            "disturbance_type": "none",
            "severity": 0.0,
            "onset": None,
            "duration": None,
            "mask": None,
        }
    elif len(condition.disturbances) == 1:
        spec = condition.disturbances[0]
        values = {
            "disturbance_type": spec.type,
            "severity": float(spec.severity),
            "onset": spec.onset,
            "duration": spec.duration,
            "mask": spec.mask if spec.type == "observation_noise" else None,
        }
    else:
        values = {
            "disturbance_type": "+".join(d.type for d in condition.disturbances),
            "severity": float("nan"),
            "onset": condition.onset,
            "duration": None,
            "mask": None,
        }
    values["regime"] = condition.regime
    return values


def _state_success(trajectory: EpisodeTrajectory, state: int) -> bool:
    """Success at ``state`` against the goal that is active for decisions taken from it (for
    the onset of a goal shift: the new goal)."""
    goal = trajectory.desired_goals[min(state + 1, trajectory.horizon)]
    distance = float(np.linalg.norm(trajectory.achieved_goals[state] - goal))
    return distance < trajectory.distance_threshold


def episode_row(
    trajectory: EpisodeTrajectory,
    condition: ConditionSpec,
    env_id: str,
    checkpoint: str,
    training_seed: int | None = None,
) -> dict[str, Any]:
    """Per-episode outcomes.

    Recovery uses a reference state r and only episodes that are failing at r
    (``recovery_eligible``); the event is the first success at a state after r.
    - permanent disturbance (no end, including goal shifts): r = onset, so the outcome is
      success reached under the disturbance, by episodes not already successful at onset.
    - transient disturbance: r = the state after the last disturbed decision, so the outcome is
      recovery after the disturbance has ended, by episodes failing when it ended.
    """
    horizon = trajectory.horizon
    first = first_success_step(trajectory.successes)
    row: dict[str, Any] = {
        "env_id": env_id,
        "checkpoint": checkpoint,
        "training_seed": training_seed,
        "condition": condition.name,
        **_primary(condition),
        "disturbances_json": json.dumps([d.to_dict() for d in condition.disturbances]),
        "episode_seed": trajectory.episode_seed,
        "horizon": horizon,
        "initial_distance": float(trajectory.distances[0]),
        "final_distance": float(trajectory.distances[-1]),
        "final_success": bool(trajectory.successes[-1]),
        "ever_success": first is not None,
        "first_success_step": first,
        "episode_return": float(trajectory.rewards.sum()),
        "clipped_action_fraction": float(
            np.mean(np.any(trajectory.pre_clip_actions != trajectory.executed_actions, axis=1))
        ),
        "mean_action_deviation": float(
            np.mean(np.linalg.norm(trajectory.executed_actions - trajectory.base_actions, axis=1))
        ),
        "mean_observation_error": float(np.mean(trajectory.observation_error)),
        "original_goal": json.dumps(trajectory.original_goal.round(6).tolist()),
        "final_goal": json.dumps(trajectory.desired_goals[-1].round(6).tolist()),
    }
    onset = condition.onset
    empty = {
        "success_at_onset": None,
        "recovery_reference_state": None,
        "success_at_reference": None,
        "recovery_eligible": None,
        "recovered": None,
        "recovery_time": None,
        "recovery_censored": None,
        "censor_time": None,
        "lost_success_in_window": None,
        "retained_success_through_window": None,
    }
    row.update(empty)
    if onset is not None:
        at_onset = _state_success(trajectory, onset)
        reference = onset if condition.window_end is None else condition.window_end
        row["success_at_onset"] = at_onset
        if reference < horizon:
            at_reference = _state_success(trajectory, reference)
            row["recovery_reference_state"] = reference
            row["success_at_reference"] = at_reference
            row["recovery_eligible"] = not at_reference
            if not at_reference:
                outcome = recovery_outcome(trajectory.successes, reference, horizon)
                row.update(
                    recovered=outcome.recovered,
                    recovery_time=outcome.recovery_time,
                    recovery_censored=outcome.censored,
                    censor_time=outcome.censor_time,
                )
        if condition.window_end is not None:
            window = trajectory.successes[onset : condition.window_end]  # states onset+1..end
            row["lost_success_in_window"] = bool(at_onset and not window.all())
            row["retained_success_through_window"] = bool(at_onset and window.all())
    event = trajectory.goal_events[0] if trajectory.goal_events else None
    if event is not None:
        step = event["step"]
        old_goal_successes = trajectory.distances_to_original_goal[step + 1 :] < (
            trajectory.distance_threshold
        )
        row.update(
            goal_shift_step=step,
            goal_shift_distance=event["shift_distance"],
            goal_shift_attempts=event["attempts"],
            goal_shift_in_goal_distribution=event.get("in_goal_distribution"),
            shifted_goal=json.dumps(np.round(event["new_goal"], 6).tolist()),
            success_at_old_goal_after_shift=bool(old_goal_successes.any()),
        )
    return row


def trajectory_rows(
    trajectory: EpisodeTrajectory, condition: ConditionSpec, env_id: str
) -> list[dict[str, Any]]:
    rows = []
    for state in range(trajectory.horizon + 1):
        row: dict[str, Any] = {
            "env_id": env_id,
            "condition": condition.name,
            "episode_seed": trajectory.episode_seed,
            "t": state,
            "distance": trajectory.distances[state],
            "distance_to_original_goal": trajectory.distances_to_original_goal[state],
        }
        for axis in range(trajectory.achieved_goals.shape[1]):
            row[f"achieved_{axis}"] = trajectory.achieved_goals[state, axis]
            row[f"goal_{axis}"] = trajectory.desired_goals[state, axis]
        if trajectory.gripper_positions is not None:
            for axis in range(3):
                row[f"gripper_{axis}"] = trajectory.gripper_positions[state, axis]
        if state == 0:
            row["success"] = trajectory.initial_success
        else:
            step = state - 1
            row["success"] = bool(trajectory.successes[step])
            row["reward"] = trajectory.rewards[step]
            for axis in range(trajectory.base_actions.shape[1]):
                row[f"base_action_{axis}"] = trajectory.base_actions[step, axis]
                row[f"executed_action_{axis}"] = trajectory.executed_actions[step, axis]
                row[f"pre_clip_action_{axis}"] = trajectory.pre_clip_actions[step, axis]
            row["observation_error"] = trajectory.observation_error[step]
            for label, active in trajectory.active.items():
                row[f"active_{label}"] = bool(active[step])
        rows.append(row)
    return rows


def _group_stats(group: pd.DataFrame, resamples: int, confidence: float) -> dict[str, Any]:
    success = group["final_success"].to_numpy(dtype=float)
    low, high = bootstrap_confidence_interval(success, resamples, confidence)
    firsts = group["first_success_step"].dropna().to_numpy(dtype=float)
    stats: dict[str, Any] = {
        "episodes": len(group),
        "success_rate": success.mean(),
        "success_ci_low": low,
        "success_ci_high": high,
        "ever_success_rate": group["ever_success"].astype(bool).mean(),
        "mean_final_distance": group["final_distance"].mean(),
        "median_final_distance": group["final_distance"].median(),
        "median_first_success_step": float(np.median(firsts)) if firsts.size else None,
        "mean_return": group["episode_return"].mean(),
        "mean_clipped_action_fraction": group["clipped_action_fraction"].mean(),
    }
    if group["recovery_reference_state"].notna().all():
        eligible = group[group["recovery_eligible"].astype(bool)]
        recovered = eligible["recovered"].astype(bool).to_numpy()
        stats.update(
            success_at_onset_count=int(group["success_at_onset"].astype(bool).sum()),
            recovery_eligible=len(eligible),
            recovered_count=int(recovered.sum()),
            recovery_rate=recovered.mean() if len(eligible) else None,
        )
        if len(eligible):
            r_low, r_high = bootstrap_confidence_interval(
                recovered.astype(float), resamples, confidence
            )
            times = np.where(
                recovered, eligible["recovery_time"].fillna(0), eligible["censor_time"]
            ).astype(float)
            recovered_times = eligible.loc[recovered, "recovery_time"].to_numpy(dtype=float)
            stats.update(
                recovery_ci_low=r_low,
                recovery_ci_high=r_high,
                km_median_recovery_time=kaplan_meier_median(times, recovered),
                median_recovery_time_recovered_only=float(np.median(recovered_times))
                if recovered_times.size
                else None,
            )
    if group["lost_success_in_window"].notna().any():
        stats["lost_success_in_window_count"] = int(group["lost_success_in_window"].sum())
        stats["retained_success_through_window_count"] = int(
            group["retained_success_through_window"].sum()
        )
    if "goal_shift_distance" in group and group["goal_shift_distance"].notna().any():
        stats["min_goal_shift_distance"] = group["goal_shift_distance"].min()
        stats["mean_goal_shift_distance"] = group["goal_shift_distance"].mean()
        stats["old_goal_success_after_shift_count"] = int(
            group["success_at_old_goal_after_shift"].astype(bool).sum()
        )
    return stats


def _with_baseline(episodes: pd.DataFrame) -> pd.DataFrame:
    baseline = episodes[episodes["disturbance_type"] == "none"][
        ["env_id", "training_seed", "episode_seed", "final_success"]
    ].rename(columns={"final_success": "baseline_final_success"})
    keys = ["env_id", "training_seed", "episode_seed"]
    merged = episodes.merge(baseline, on=keys, how="left", validate="many_to_one")
    return merged


def summarize(episodes: pd.DataFrame, resamples: int, confidence: float) -> pd.DataFrame:
    """One row per (training seed, condition)."""
    merged = _with_baseline(episodes)
    rows = []
    keys = [*GROUP_KEYS, "training_seed"]
    for key, group in merged.groupby(keys, dropna=False, sort=False):
        row: dict[str, Any] = dict(zip(keys, key, strict=True))
        row.update(_group_stats(group, resamples, confidence))
        if group["baseline_final_success"].notna().all():
            ours = group["final_success"].astype(bool)
            theirs = group["baseline_final_success"].astype(bool)
            row.update(
                paired_lost_vs_baseline=int((theirs & ~ours).sum()),
                paired_gained_vs_baseline=int((~theirs & ours).sum()),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate(episodes: pd.DataFrame, resamples: int, confidence: float, seed: int) -> pd.DataFrame:
    """One row per condition, pooled over training seeds, with per-seed ranges and
    hierarchical (seed, then episode) bootstrap intervals."""
    rng = np.random.default_rng(seed)
    merged = _with_baseline(episodes)
    rows = []
    for key, group in merged.groupby(GROUP_KEYS, dropna=False, sort=False):
        row: dict[str, Any] = dict(zip(GROUP_KEYS, key, strict=True))
        row.update(_group_stats(group, resamples, confidence))
        # success_ci_* here is an episode-level bootstrap over the pooled episodes
        seeds = sorted(group["training_seed"].dropna().unique())
        per_seed = group.groupby("training_seed")["final_success"].mean()
        row.update(
            training_seeds=len(seeds),
            seed_success_min=per_seed.min(),
            seed_success_max=per_seed.max(),
        )
        by_seed = [group[group["training_seed"] == s] for s in seeds]
        low, high = hierarchical_bootstrap_interval(
            [g["final_success"].to_numpy(dtype=float) for g in by_seed], resamples, confidence, rng
        )
        row.update(success_hier_ci_low=low, success_hier_ci_high=high)
        if row.get("recovery_eligible"):
            groups = [
                g.loc[g["recovery_eligible"].astype(bool), "recovered"].to_numpy(dtype=float)
                for g in by_seed
            ]
            low, high = hierarchical_bootstrap_interval(groups, resamples, confidence, rng)
            row.update(recovery_hier_ci_low=low, recovery_hier_ci_high=high)
            per_seed_recovery = [g.mean() for g in groups if g.size]
            row.update(
                seed_recovery_min=min(per_seed_recovery), seed_recovery_max=max(per_seed_recovery)
            )
        if group["baseline_final_success"].notna().all() and row["disturbance_type"] != "none":
            differences = [
                g["final_success"].to_numpy(dtype=float)
                - g["baseline_final_success"].to_numpy(dtype=float)
                for g in by_seed
            ]
            low, high = hierarchical_bootstrap_interval(differences, resamples, confidence, rng)
            ours = group["final_success"].astype(bool)
            theirs = group["baseline_final_success"].astype(bool)
            row.update(
                paired_success_difference=float(np.concatenate(differences).mean()),
                paired_difference_hier_ci_low=low,
                paired_difference_hier_ci_high=high,
                paired_lost_vs_baseline=int((theirs & ~ours).sum()),
                paired_gained_vs_baseline=int((~theirs & ours).sum()),
            )
        rows.append(row)
    return pd.DataFrame(rows)


def distance_curve(trajectories: list[EpisodeTrajectory]) -> pd.DataFrame:
    distances = np.stack([t.distances for t in trajectories])
    return pd.DataFrame(
        {
            "t": np.arange(distances.shape[1]),
            "mean": distances.mean(axis=0),
            "median": np.median(distances, axis=0),
            "q25": np.quantile(distances, 0.25, axis=0),
            "q75": np.quantile(distances, 0.75, axis=0),
            "episodes": distances.shape[0],
        }
    )


def _check_original_evaluator(
    env: Any,
    policy: Any,
    loaded: Any,
    seeds: list[int],
    rows: list[dict[Any, Any]],
) -> dict[str, Any]:
    """Re-run the repository's original evaluator on the same seeds and require identical
    per-episode success, return and final distance to the undisturbed rows."""
    _, outcomes = evaluate_policy(env, policy, loaded.normalizer, seeds, 1)
    by_seed = {row["episode_seed"]: row for row in rows}
    mismatches = [
        outcome.seed
        for outcome in outcomes
        if by_seed[outcome.seed]["final_success"] != bool(outcome.success)
        or by_seed[outcome.seed]["episode_return"] != outcome.episode_return
        or not np.isclose(by_seed[outcome.seed]["final_distance"], outcome.final_distance)
    ]
    if mismatches:
        raise RuntimeError(f"undisturbed rollouts differ from the original evaluator: {mismatches}")
    return {"episodes": len(outcomes), "identical": True}


def _part_name(env_id: str, training_seed: int) -> str:
    return f"{env_id}_seed{training_seed}"


def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    verbose: bool = True,
    resume: bool = False,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    fingerprint = config_fingerprint(config)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if not resume:
            raise ResumeConfigMismatch(f"{output_dir} holds a run; pass resume to continue it")
        if saved["fingerprint"] != fingerprint:
            raise ResumeConfigMismatch(
                f"{output_dir} was started with a different configuration; refusing to mix them"
            )
    else:
        manifest_path.write_text(
            json.dumps({"fingerprint": fingerprint, "experiment": config.to_dict()}, indent=2)
            + "\n"
        )
    seed_everything(config.perturbation_seed, deterministic=True)
    torch.set_num_threads(config.torch_threads)
    started = time.perf_counter()
    metadata: dict[str, Any] = {
        "experiment": config.to_dict(),
        "fingerprint": fingerprint,
        "note": STAGE_NOTES[config.stage],
        "run": collect_run_metadata(config.perturbation_seed),
        "device": "cpu",
        "python": platform.python_version(),
        "policy": "deterministic actor output tanh(mean), frozen weights and normalizer",
        "episode_seeds": [config.episode_seeds[0], config.episode_seeds[-1] + 1],
        "checkpoints": [],
        "parts_reused": 0,
        "parts_run": 0,
    }
    parts_dir = output_dir / "parts"
    for checkpoint_path in config.checkpoints:
        resolved_checkpoint = resolve(checkpoint_path)
        loaded = load_policy(resolved_checkpoint)  # verifies the manifest SHA-256
        sha_before = loaded.sha256
        actor_state = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
        loaded.actor.requires_grad_(False)
        frozen = FrozenNormalizer(loaded.normalizer)
        original_digest = normalizer_digest(loaded.normalizer)
        policy = deterministic_policy_function(loaded.actor)
        env = make_env(loaded.env_config)
        check_goal_env(env)
        dimensions = env_dimensions(env)
        if dimensions.episode_length != loaded.dimensions["episode_length"]:
            raise RuntimeError("environment horizon differs from the checkpoint's horizon")
        env_id = loaded.env_config.env_id
        training_seed = loaded.training_seed
        checkpoint_started = time.perf_counter()
        evaluator_check = None
        for condition in config.conditions:
            if not condition.applies_to(env_id):
                continue
            part = parts_dir / _part_name(env_id, training_seed) / condition.name
            done = part / "done.json"
            if done.exists():
                marker = json.loads(done.read_text())
                if marker["fingerprint"] != fingerprint or marker["checkpoint_sha256"] != (
                    sha_before
                ):
                    raise ResumeConfigMismatch(f"{part} was produced by a different run")
                metadata["parts_reused"] += 1
                if condition.is_baseline and evaluator_check is None:
                    saved_rows = pd.read_csv(part / "episodes.csv").to_dict("records")
                    evaluator_check = _check_original_evaluator(
                        env, policy, loaded, config.episode_seeds, saved_rows
                    )
                continue
            part.mkdir(parents=True, exist_ok=True)
            condition_started = time.perf_counter()
            trajectories = []
            rows: list[dict[Any, Any]] = []
            steps: list[dict[str, Any]] = []
            for episode_seed in config.episode_seeds:
                trajectory = run_disturbed_episode(
                    env, policy, frozen, condition, episode_seed, config.perturbation_seed
                )
                trajectories.append(trajectory)
                rows.append(
                    episode_row(trajectory, condition, env_id, checkpoint_path, training_seed)
                )
                if config.save_trajectories:
                    steps.extend(trajectory_rows(trajectory, condition, env_id))
            if condition.is_baseline:
                evaluator_check = _check_original_evaluator(
                    env, policy, loaded, config.episode_seeds, rows
                )
            pd.DataFrame(rows).to_csv(part / "episodes.csv", index=False)
            curve = distance_curve(trajectories)
            curve.insert(0, "condition", condition.name)
            curve.insert(0, "training_seed", training_seed)
            curve.insert(0, "env_id", env_id)
            curve.to_csv(part / "distance_curve.csv", index=False)
            if config.save_trajectories:
                pd.DataFrame(steps).to_csv(part / "trajectories.csv.gz", index=False)
            seconds = time.perf_counter() - condition_started
            done.write_text(
                json.dumps(
                    {
                        "fingerprint": fingerprint,
                        "checkpoint_sha256": sha_before,
                        "condition": condition.to_dict(),
                        "seconds": round(seconds, 2),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    },
                    indent=2,
                )
                + "\n"
            )
            metadata["parts_run"] += 1
            if verbose:
                rate = np.mean([r["final_success"] for r in rows])
                print(
                    f"[{env_id} seed {training_seed}] {condition.name:<48s} "
                    f"success {rate:.2f}  {seconds:5.1f}s",
                    flush=True,
                )
        env.close()
        frozen.verify()
        if normalizer_digest(loaded.normalizer) != original_digest:
            raise RuntimeError("the loaded policy normalizer changed during evaluation")
        for key, value in loaded.actor.state_dict().items():
            if not torch.equal(value, actor_state[key]):
                raise RuntimeError(f"actor parameter {key} changed during evaluation")
        if file_sha256(resolved_checkpoint) != sha_before:
            raise RuntimeError(f"{checkpoint_path} changed on disk during evaluation")
        metadata["checkpoints"].append(
            {
                "path": checkpoint_path,
                "sha256": sha_before,
                "env_id": env_id,
                "training_seed": training_seed,
                "checkpoint_id": loaded.checkpoint_id,
                "total_steps": loaded.total_steps,
                "dimensions": loaded.dimensions,
                "normalizer_sha256": original_digest,
                "original_evaluator_check": evaluator_check,
                "evaluation_seconds_this_invocation": round(
                    time.perf_counter() - checkpoint_started, 2
                ),
            }
        )

    episode_frames, curve_frames = [], []
    for loaded_part in sorted(parts_dir.glob("*/*/done.json")):
        episode_frames.append(pd.read_csv(loaded_part.parent / "episodes.csv"))
        curve_frames.append(pd.read_csv(loaded_part.parent / "distance_curve.csv"))
    episodes = pd.concat(episode_frames, ignore_index=True)
    order: dict[str, int] = {}
    for index, item in enumerate(config.conditions):
        order.setdefault(item.name, index)
    episodes = episodes.sort_values(
        ["env_id", "training_seed", "condition", "episode_seed"],
        key=lambda c: c.map(order) if c.name == "condition" else c,
    ).reset_index(drop=True)
    for column in ("onset", "duration"):
        episodes[column] = episodes[column].astype("Int64")
    per_seed = summarize(episodes, config.bootstrap_resamples, config.confidence)
    pooled = aggregate(
        episodes, config.bootstrap_resamples, config.confidence, config.perturbation_seed
    )
    per_seed.insert(0, "stage", config.stage)
    pooled.insert(0, "stage", config.stage)
    curves = pd.concat(curve_frames, ignore_index=True)
    paths = {
        "episodes": output_dir / "episodes.csv",
        "summary_per_seed": output_dir / "summary_per_seed.csv",
        "summary": output_dir / "summary.csv",
        "distance_curves": output_dir / "distance_curves.csv",
        "metadata": output_dir / "metadata.json",
    }
    episodes.to_csv(paths["episodes"], index=False)
    per_seed.to_csv(paths["summary_per_seed"], index=False)
    pooled.to_csv(paths["summary"], index=False)
    curves.to_csv(paths["distance_curves"], index=False)

    from farc.disturbances.plots import plot_all

    paths.update(plot_all(pooled, per_seed, curves, output_dir, config.stage))
    metadata["wall_seconds_this_invocation"] = round(time.perf_counter() - started, 2)
    metadata["compute_seconds_all_parts"] = round(
        sum(json.loads(p.read_text())["seconds"] for p in parts_dir.glob("*/*/done.json")), 2
    )
    metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
    metadata["files"] = {key: path.name for key, path in paths.items()}
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return paths
