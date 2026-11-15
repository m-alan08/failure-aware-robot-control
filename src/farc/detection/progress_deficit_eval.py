# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/detector_eval.py
# (introduced in 934063d, last changed in 934063d). See PROVENANCE.md.
"""Run the progress-deficit detector over a finished disturbance experiment's trajectories.

Episode labels come from the experiment's own episode rows: a failure is an episode that is
not successful at the final step (the evaluation's success definition). A warning on a failed
episode is a detection; a warning on a successful episode is a false positive.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from farc.detection.progress_deficit import (
    DetectorConfig,
    kinematically_infeasible,
    run_detector,
)
from farc.paths import resolve
from farc.statistics import hierarchical_bootstrap_interval

TRAJECTORY_COLUMNS = [
    "env_id", "condition", "episode_seed", "t",
    "achieved_0", "achieved_1", "achieved_2", "goal_0", "goal_1", "goal_2",
    "executed_action_0", "executed_action_1", "executed_action_2",
]  # fmt: skip
EPISODE_KEYS = ["env_id", "training_seed", "condition", "episode_seed"]
CONDITION_KEYS = [
    "env_id", "condition", "disturbance_type", "mask", "severity", "onset", "duration", "regime",
]  # fmt: skip
# The object counts as not yet moved while it is within this distance of its start position.
OBJECT_STATIONARY_TOLERANCE = 1e-3


class DetectorConfigError(ValueError):
    pass


@dataclass(frozen=True)
class DetectorExperiment:
    name: str
    source: str
    windows: tuple[int, ...]
    persistence: tuple[int, ...]
    deficit_thresholds: tuple[float, ...]
    success_threshold: float
    late_lead_steps: int
    bootstrap_resamples: int
    bootstrap_seed: int

    def configs(self) -> list[DetectorConfig]:
        return [
            DetectorConfig(k, p, threshold, self.success_threshold)
            for threshold in self.deficit_thresholds
            for k in self.windows
            for p in self.persistence
        ]


KEYS = {
    "name", "description", "source", "windows", "persistence", "deficit_thresholds",
    "success_threshold", "late_lead_steps", "bootstrap_resamples", "bootstrap_seed",
}  # fmt: skip


def load_detector_experiment(path: str | Path) -> DetectorExperiment:
    values = json.loads(Path(path).read_text())
    unknown = set(values) - KEYS
    if unknown:
        raise DetectorConfigError(f"unknown keys {sorted(unknown)}")
    missing = KEYS - {"description"} - set(values)
    if missing:
        raise DetectorConfigError(f"missing keys {sorted(missing)}")

    def positive_ints(name: str) -> tuple[int, ...]:
        items = values[name]
        if not (
            isinstance(items, list)
            and items
            and all(isinstance(i, int) and not isinstance(i, bool) and i >= 1 for i in items)
        ):
            raise DetectorConfigError(f"{name} must be a non-empty list of positive integers")
        return tuple(items)

    thresholds = values["deficit_thresholds"]
    if not (
        isinstance(thresholds, list)
        and thresholds
        and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in thresholds)
    ):
        raise DetectorConfigError("deficit_thresholds must be a non-empty list of numbers")
    experiment = DetectorExperiment(
        name=values["name"],
        source=values["source"],
        windows=positive_ints("windows"),
        persistence=positive_ints("persistence"),
        deficit_thresholds=tuple(float(x) for x in thresholds),
        success_threshold=float(values["success_threshold"]),
        late_lead_steps=int(values["late_lead_steps"]),
        bootstrap_resamples=int(values["bootstrap_resamples"]),
        bootstrap_seed=int(values["bootstrap_seed"]),
    )
    experiment.configs()  # validates every combination
    return experiment


def load_trajectories(source: Path) -> dict[tuple, dict[str, np.ndarray]]:
    """Per-episode ground-truth arrays from every finished part of an experiment."""
    episodes: dict[tuple, dict[str, np.ndarray]] = {}
    for done in sorted(source.glob("parts/*/*/done.json")):
        part = done.parent
        training_seed = int(part.parent.name.rsplit("_seed", 1)[1])
        frame = pd.read_csv(part / "trajectories.csv.gz", usecols=TRAJECTORY_COLUMNS)
        frame = frame.sort_values(["episode_seed", "t"])
        for (env_id, condition, seed), group in frame.groupby(
            ["env_id", "condition", "episode_seed"], sort=False
        ):
            if not np.array_equal(group["t"].to_numpy(), np.arange(len(group))):
                raise ValueError(f"{part}: episode {seed} has missing or repeated states")
            actions = group[[f"executed_action_{i}" for i in range(3)]].to_numpy()[1:]
            episodes[(env_id, training_seed, condition, int(seed))] = {  # type: ignore[call-overload]
                "achieved": group[[f"achieved_{i}" for i in range(3)]].to_numpy(),
                "goals": group[[f"goal_{i}" for i in range(3)]].to_numpy(),
                "actions": actions,
            }
    return episodes


def evaluate_episode(
    arrays: dict[str, np.ndarray], config: DetectorConfig, onset: int | None, has_object: bool
) -> dict[str, Any]:
    trace = run_detector(arrays["achieved"], arrays["goals"], config)
    horizon = len(trace.decision_distance)
    warning = trace.first_warning
    row: dict[str, Any] = {
        "detector": config.label,
        "window": config.window,
        "persistence": config.persistence,
        "deficit_threshold": config.deficit_threshold,
        "warned": warning is not None,
        "warning_state": warning,
        "lead_time": None,
        "warning_minus_onset": None,
        "warned_before_onset": None,
        "distance_at_warning": None,
        "infeasible_at_warning": None,
        "object_unmoved_at_warning": None,
        "gripper_commanded_in_window": None,
        "flagged_states": int(trace.flagged.sum()),
    }
    if warning is None:
        return row
    distance = float(trace.decision_distance[warning])
    row.update(
        lead_time=horizon - warning,
        distance_at_warning=distance,
        infeasible_at_warning=kinematically_infeasible(
            distance - config.success_threshold, horizon - warning
        ),
    )
    if onset is not None:
        row["warning_minus_onset"] = warning - onset
        row["warned_before_onset"] = warning < onset
    if has_object:
        achieved = arrays["achieved"]
        moved = float(np.linalg.norm(achieved[warning] - achieved[0]))
        row["object_unmoved_at_warning"] = moved < OBJECT_STATIONARY_TOLERANCE
        window = arrays["actions"][max(0, warning - config.window) : warning]
        row["gripper_commanded_in_window"] = bool(np.abs(window).max(initial=0.0) > 1e-3)
    return row


def evaluate(
    experiment: DetectorExperiment, episodes_frame: pd.DataFrame, trajectories: dict
) -> pd.DataFrame:
    rows = []
    configs = experiment.configs()
    for record in episodes_frame.to_dict("records"):
        key = tuple(record[k] for k in EPISODE_KEYS)
        if key not in trajectories:
            raise KeyError(f"no trajectory for episode {key}")
        onset = None if pd.isna(record["onset"]) else int(record["onset"])
        has_object = record["env_id"] != "FetchReach-v4"
        base = {k: record[k] for k in [*EPISODE_KEYS, *CONDITION_KEYS[2:]]}
        base.update(
            final_success=bool(record["final_success"]),
            ever_success=bool(record["ever_success"]),
            failure=not bool(record["final_success"]),
        )
        for config in configs:
            rows.append({**base, **evaluate_episode(trajectories[key], config, onset, has_object)})
    return pd.DataFrame(rows)


def _rates(group: pd.DataFrame, late_lead_steps: int) -> dict[str, Any]:
    failures = group[group["failure"]]
    successes = group[~group["failure"]]
    detected = failures[failures["warned"]]
    false_positive = successes[successes["warned"]]
    leads = detected["lead_time"].to_numpy(dtype=float)
    stats: dict[str, Any] = {
        "episodes": len(group),
        "failures": len(failures),
        "detected": len(detected),
        "missed": len(failures) - len(detected),
        "detection_rate": len(detected) / len(failures) if len(failures) else None,
        "successes": len(successes),
        "false_positives": len(false_positive),
        "false_positive_rate": len(false_positive) / len(successes) if len(successes) else None,
        "median_lead_time": float(np.median(leads)) if leads.size else None,
        "q25_lead_time": float(np.quantile(leads, 0.25)) if leads.size else None,
        "q75_lead_time": float(np.quantile(leads, 0.75)) if leads.size else None,
        "median_warning_state_detected": float(detected["warning_state"].median())
        if len(detected)
        else None,
        f"detected_lead_le_{late_lead_steps}": int((leads <= late_lead_steps).sum()),
        "detected_kinematically_infeasible": int(detected["infeasible_at_warning"].sum()),
        "detected_before_onset": int(detected["warned_before_onset"].fillna(False).sum()),
        "false_positives_before_onset": int(
            false_positive["warned_before_onset"].fillna(False).sum()
        ),
    }
    if group["object_unmoved_at_warning"].notna().any():
        stats["false_positives_object_unmoved"] = int(
            false_positive["object_unmoved_at_warning"].fillna(False).sum()
        )
        stats["detected_object_unmoved"] = int(
            detected["object_unmoved_at_warning"].fillna(False).sum()
        )
    return stats


def summarize_by_condition(results: pd.DataFrame, late_lead_steps: int) -> pd.DataFrame:
    keys = ["env_id", "detector", "window", "persistence", "deficit_threshold", *CONDITION_KEYS[1:]]
    rows = []
    for key, group in results.groupby(keys, dropna=False, sort=False):
        rows.append({**dict(zip(keys, key, strict=True)), **_rates(group, late_lead_steps)})
    return pd.DataFrame(rows)


def summarize_per_seed(results: pd.DataFrame, late_lead_steps: int) -> pd.DataFrame:
    keys = ["env_id", "detector", "training_seed", "condition"]
    rows = []
    for key, group in results.groupby(keys, dropna=False, sort=False):
        rows.append({**dict(zip(keys, key, strict=True)), **_rates(group, late_lead_steps)})
    return pd.DataFrame(rows)


def summarize_overall(
    results: pd.DataFrame, late_lead_steps: int, resamples: int, seed: int
) -> pd.DataFrame:
    """Per task and detector over every disturbed condition, with the undisturbed episodes
    reported separately (they are all successes, so they only give false positives)."""
    rng = np.random.default_rng(seed)
    keys = ["env_id", "detector", "window", "persistence", "deficit_threshold"]
    rows = []
    for key, group in results.groupby(keys, sort=False):
        disturbed = group[group["disturbance_type"] != "none"]
        undisturbed = group[group["disturbance_type"] == "none"]
        row = {**dict(zip(keys, key, strict=True)), **_rates(disturbed, late_lead_steps)}
        seeds = sorted(disturbed["training_seed"].unique())
        failures = [
            disturbed[(disturbed["training_seed"] == s) & disturbed["failure"]]["warned"].to_numpy(
                dtype=float
            )
            for s in seeds
        ]
        successes = [
            disturbed[(disturbed["training_seed"] == s) & ~disturbed["failure"]]["warned"].to_numpy(
                dtype=float
            )
            for s in seeds
        ]
        row["detection_hier_ci_low"], row["detection_hier_ci_high"] = (
            hierarchical_bootstrap_interval(failures, resamples, 0.95, rng)
        )
        row["fpr_hier_ci_low"], row["fpr_hier_ci_high"] = hierarchical_bootstrap_interval(
            successes, resamples, 0.95, rng
        )
        row["undisturbed_episodes"] = len(undisturbed)
        row["undisturbed_false_positives"] = int(undisturbed["warned"].sum())
        row["undisturbed_false_positive_rate"] = (
            float(undisturbed["warned"].mean()) if len(undisturbed) else None
        )
        rows.append(row)
    return pd.DataFrame(rows)


def run_detector_evaluation(
    experiment: DetectorExperiment, output_dir: Path, config_path: Path | None = None
) -> dict[str, Path]:
    source = resolve(experiment.source)
    episodes_frame = pd.read_csv(source / "episodes.csv")
    source_metadata = json.loads((source / "metadata.json").read_text())
    trajectories = load_trajectories(source)
    if len(trajectories) != len(episodes_frame):
        raise ValueError(f"{len(trajectories)} trajectories for {len(episodes_frame)} episode rows")
    results = evaluate(experiment, episodes_frame, trajectories)
    by_condition = summarize_by_condition(results, experiment.late_lead_steps)
    per_seed = summarize_per_seed(results, experiment.late_lead_steps)
    overall = summarize_overall(
        results,
        experiment.late_lead_steps,
        experiment.bootstrap_resamples,
        experiment.bootstrap_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "episodes": output_dir / "detector_episodes.csv.gz",
        "by_condition": output_dir / "detector_by_condition.csv",
        "per_seed": output_dir / "detector_per_seed.csv",
        "overall": output_dir / "detector_overall.csv",
        "metadata": output_dir / "detector_metadata.json",
    }
    results.to_csv(paths["episodes"], index=False)
    by_condition.to_csv(paths["by_condition"], index=False)
    per_seed.to_csv(paths["per_seed"], index=False)
    overall.to_csv(paths["overall"], index=False)
    from farc.detection.progress_deficit_plots import plot_detector_results
    from farc.utils.run_info import git_state

    paths.update(plot_detector_results(overall, by_condition, results, output_dir))
    episodes_digest = hashlib.sha256((source / "episodes.csv").read_bytes()).hexdigest()
    metadata = {
        "experiment": asdict(experiment),
        "config_file": str(config_path) if config_path else None,
        "git": git_state(),
        "source": str(source),
        "source_episodes_sha256": episodes_digest,
        "source_git": source_metadata["run"]["git"],
        "source_stage": source_metadata["experiment"]["stage"],
        "source_episode_seeds": source_metadata["episode_seeds"],
        "episodes": len(episodes_frame),
        "detectors": [c.label for c in experiment.configs()],
        "labels": "failure = not successful at the final step (50); from the source episodes",
        "files": {k: p.name for k, p in paths.items()},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return paths
