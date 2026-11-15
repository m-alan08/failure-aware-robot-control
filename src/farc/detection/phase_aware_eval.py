# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/phase_eval.py
# (introduced in 98c903e, last changed in ff20609). See PROVENANCE.md.
"""Matched comparison of the phase-aware detector with the distance-only detector (FetchPush).

Both detectors run on exactly the same logged episodes. Episode labels come from the source
run: failure = not successful at step 50.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from farc.detection.phase_aware import (
    APPROACH,
    MAINTAIN,
    PhaseDetectorConfig,
    contact_phases,
    run_phase_detector,
)
from farc.detection.progress_deficit import DetectorConfig, decision_goals, run_detector
from farc.paths import resolve
from farc.statistics import hierarchical_bootstrap_interval

EPISODE_KEYS = ["env_id", "training_seed", "condition", "episode_seed"]
CONDITION_COLUMNS = ["disturbance_type", "mask", "severity", "onset", "duration", "regime"]
STATE_COLUMNS = [
    "env_id", "condition", "episode_seed", "t",
    "achieved_0", "achieved_1", "achieved_2", "goal_0", "goal_1", "goal_2",
    "gripper_0", "gripper_1", "gripper_2",
    "executed_action_0", "executed_action_1", "executed_action_2", "executed_action_3",
]  # fmt: skip
OBJECT_MOVED_TOLERANCE = 1e-3


class PhaseEvalConfigError(ValueError):
    pass


@dataclass(frozen=True)
class PhaseEvaluation:
    name: str
    source: str
    reference_source: str | None
    env_id: str
    baseline_windows: tuple[int, ...]
    baseline_persistence: tuple[int, ...]
    windows: tuple[int, ...]
    persistence: tuple[int, ...]
    hysteresis: tuple[int, ...]
    departure_persistence: tuple[int | None, ...]
    contact_enter: float
    contact_exit: float
    behind_offset: float
    deficit_threshold: float
    success_threshold: float
    reference_hysteresis: int
    late_lead_steps: int
    speed_reference_quantile: float
    bootstrap_resamples: int
    bootstrap_seed: int

    def baselines(self) -> list[DetectorConfig]:
        return [
            DetectorConfig(k, p, self.deficit_threshold, self.success_threshold)
            for k, p in itertools.product(self.baseline_windows, self.baseline_persistence)
        ]

    def phase_configs(self) -> list[PhaseDetectorConfig]:
        return [
            PhaseDetectorConfig(
                window=k,
                persistence=p,
                hysteresis=h,
                departure_persistence=m,
                contact_enter=self.contact_enter,
                contact_exit=self.contact_exit,
                behind_offset=self.behind_offset,
                deficit_threshold=self.deficit_threshold,
                success_threshold=self.success_threshold,
            )
            for k, p, h, m in itertools.product(
                self.windows, self.persistence, self.hysteresis, self.departure_persistence
            )
        ]


def load_phase_evaluation(path: str | Path) -> PhaseEvaluation:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    fields = set(PhaseEvaluation.__dataclass_fields__)
    unknown, missing = set(values) - fields, fields - set(values)
    if unknown or missing:
        raise PhaseEvalConfigError(f"unknown keys {sorted(unknown)}, missing {sorted(missing)}")
    for key in (
        "baseline_windows",
        "baseline_persistence",
        "windows",
        "persistence",
        "hysteresis",
        "departure_persistence",
    ):
        if not isinstance(values[key], list) or not values[key]:
            raise PhaseEvalConfigError(f"{key} must be a non-empty list")
        values[key] = tuple(values[key])
    evaluation = PhaseEvaluation(**values)
    evaluation.baselines()
    evaluation.phase_configs()  # validates every combination
    return evaluation


def load_states(source: Path, env_id: str) -> dict[tuple, dict[str, np.ndarray]]:
    episodes: dict[tuple, dict[str, np.ndarray]] = {}
    for done in sorted(source.glob(f"parts/{env_id}_seed*/*/done.json")):
        part = done.parent
        training_seed = int(part.parent.name.rsplit("_seed", 1)[1])
        frame = pd.read_csv(part / "trajectories.csv.gz", usecols=STATE_COLUMNS)
        frame = frame.sort_values(["episode_seed", "t"])
        for (_, condition, seed), group in frame.groupby(
            ["env_id", "condition", "episode_seed"], sort=False
        ):
            if not np.array_equal(group["t"].to_numpy(), np.arange(len(group))):
                raise ValueError(f"{part}: episode {seed} has missing or repeated states")
            key = (env_id, training_seed, condition, int(seed))  # type: ignore[call-overload]
            episodes[key] = {
                "objects": group[[f"achieved_{i}" for i in range(3)]].to_numpy(),
                "goals": group[[f"goal_{i}" for i in range(3)]].to_numpy(),
                "gripper": group[[f"gripper_{i}" for i in range(3)]].to_numpy(),
                "actions": group[[f"executed_action_{i}" for i in range(4)]].to_numpy()[1:],
            }
    return episodes


def check_matched(
    states: dict[tuple, dict[str, np.ndarray]], reference: Path, env_id: str
) -> dict[str, Any]:
    """Require the rerun to reproduce the reference run's object, goal and action trajectories
    exactly, so both detectors and the earlier results refer to the same episodes."""
    checked = 0
    for done in sorted(reference.glob(f"parts/{env_id}_seed*/*/done.json")):
        part = done.parent
        training_seed = int(part.parent.name.rsplit("_seed", 1)[1])
        columns = [c for c in STATE_COLUMNS if not c.startswith("gripper_")]
        frame = pd.read_csv(part / "trajectories.csv.gz", usecols=columns)
        frame = frame.sort_values(["episode_seed", "t"])
        for (_, condition, seed), group in frame.groupby(
            ["env_id", "condition", "episode_seed"], sort=False
        ):
            key = (env_id, training_seed, condition, int(seed))  # type: ignore[call-overload]
            ours = states.get(key)
            if ours is None:
                raise ValueError(f"episode {key} is missing from the rerun")
            same = (
                np.array_equal(ours["objects"], group[[f"achieved_{i}" for i in range(3)]])
                and np.array_equal(ours["goals"], group[[f"goal_{i}" for i in range(3)]])
                and np.array_equal(
                    ours["actions"],
                    group[[f"executed_action_{i}" for i in range(4)]].to_numpy()[1:],
                )
            )
            if not same:
                raise ValueError(f"episode {key} differs between the rerun and {reference}")
            checked += 1
    if checked != len(states):
        raise ValueError(f"matched {checked} reference episodes, the rerun has {len(states)}")
    return {"reference": str(reference), "episodes_identical": checked}


def object_speed_reference(
    states: dict[tuple, dict[str, np.ndarray]], episodes: pd.DataFrame, quantile: float
) -> float:
    """A reference for how fast the undisturbed base policy moves the object: the given
    quantile of per-step object displacement over undisturbed steps where the object moves."""
    speeds = []
    for record in episodes[episodes["disturbance_type"] == "none"].to_dict("records"):
        objects = states[tuple(record[k] for k in EPISODE_KEYS)]["objects"]
        steps = np.linalg.norm(np.diff(objects, axis=0), axis=1)
        speeds.append(steps[steps > OBJECT_MOVED_TOLERANCE])
    return float(np.quantile(np.concatenate(speeds), quantile))


def _first(condition: np.ndarray) -> int | None:
    hits = np.flatnonzero(condition)
    return int(hits[0]) if hits.size else None


def episode_context(arrays: dict[str, np.ndarray], evaluation: PhaseEvaluation) -> dict[str, Any]:
    objects, goals = arrays["objects"], arrays["goals"]
    moved = np.linalg.norm(objects - objects[0], axis=1) >= OBJECT_MOVED_TOLERANCE
    success = np.linalg.norm(objects - goals, axis=1) < evaluation.success_threshold
    goal_for_decision = decision_goals(goals)
    horizon = len(goal_for_decision)
    reference = PhaseDetectorConfig(
        1, 1, hysteresis=evaluation.reference_hysteresis,
        contact_enter=evaluation.contact_enter, contact_exit=evaluation.contact_exit,
    )  # fmt: skip
    contact = contact_phases(
        np.linalg.norm(arrays["gripper"][:horizon] - objects[:horizon], axis=1), reference
    )
    inside = np.linalg.norm(objects[:horizon] - goal_for_decision, axis=1) < (
        evaluation.success_threshold
    )
    later_success = _first(success[1:])
    return {
        "first_object_move_state": _first(moved),
        "first_success_state": None if later_success is None else later_success + 1,
        "reference_phases": [MAINTAIN if inside[t] else contact[t] for t in range(horizon)],
        "object_goal": np.linalg.norm(objects[:horizon] - goal_for_decision, axis=1),
        "horizon": horizon,
    }


def warning_row(
    detector: str,
    family: str,
    warning: int | None,
    context: dict[str, Any],
    evaluation: PhaseEvaluation,
    speed_reference: float,
    onset: int | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "detector": detector,
        "family": family,
        "warned": warning is not None,
        "warning_state": warning,
        "lead_time": None,
        "warning_minus_first_move": None,
        "warning_minus_first_success": None,
        "warned_before_object_moved": None,
        "reference_phase_at_warning": None,
        "late_by_lead": None,
        "late_by_object_speed": None,
        "warned_before_onset": None,
        **extra,
    }
    if warning is None:
        return row
    horizon = context["horizon"]
    first_move = context["first_object_move_state"]
    first_success = context["first_success_state"]
    to_go = max(float(context["object_goal"][warning]) - evaluation.success_threshold, 0.0)
    row.update(
        lead_time=horizon - warning,
        warning_minus_first_move=None if first_move is None else warning - first_move,
        warning_minus_first_success=None if first_success is None else warning - first_success,
        warned_before_object_moved=first_move is None or warning < first_move,
        reference_phase_at_warning=context["reference_phases"][warning],
        late_by_lead=horizon - warning <= evaluation.late_lead_steps,
        late_by_object_speed=to_go / (horizon - warning) > speed_reference,
        warned_before_onset=None if onset is None else warning < onset,
    )
    return row


def evaluate(
    evaluation: PhaseEvaluation,
    episodes: pd.DataFrame,
    states: dict[tuple, dict[str, np.ndarray]],
    speed_reference: float,
) -> pd.DataFrame:
    rows = []
    baselines = evaluation.baselines()
    phase_configs = evaluation.phase_configs()
    for record in episodes.to_dict("records"):
        arrays = states[tuple(record[k] for k in EPISODE_KEYS)]
        context = episode_context(arrays, evaluation)
        onset = None if pd.isna(record["onset"]) else int(record["onset"])
        base = {k: record[k] for k in [*EPISODE_KEYS, *CONDITION_COLUMNS]}
        base.update(
            failure=not bool(record["final_success"]),
            ever_success=bool(record["ever_success"]),
            first_object_move_state=context["first_object_move_state"],
            first_success_state=context["first_success_state"],
        )
        for config in baselines:
            trace = run_detector(arrays["objects"], arrays["goals"], config)
            rows.append({**base, **warning_row(
                f"distance_k{config.window}_p{config.persistence}", "distance_only",
                trace.first_warning, context, evaluation, speed_reference, onset,
                {"window": config.window, "persistence": config.persistence,
                 "hysteresis": None, "departure_persistence": None, "warning_kind": None,
                 "warning_phase": None},
            )})  # fmt: skip
        for phase_config in phase_configs:
            phase_trace = run_phase_detector(
                arrays["gripper"], arrays["objects"], arrays["goals"], phase_config
            )
            rows.append({**base, **warning_row(
                phase_config.label, "phase_aware", phase_trace.first_warning, context,
                evaluation, speed_reference, onset,
                {"window": phase_config.window, "persistence": phase_config.persistence,
                 "hysteresis": phase_config.hysteresis,
                 "departure_persistence": phase_config.departure_persistence,
                 "warning_kind": phase_trace.warning_kind,
                 "warning_phase": phase_trace.warning_phase},
            )})  # fmt: skip
    return pd.DataFrame(rows)


def rates(group: pd.DataFrame) -> dict[str, Any]:
    failures = group[group["failure"]]
    successes = group[~group["failure"]]
    detected = failures[failures["warned"]]
    false_alarms = successes[successes["warned"]]
    leads = detected["lead_time"].to_numpy(dtype=float)

    def count(frame: pd.DataFrame, column: str, value: Any = True) -> int:
        return int((frame[column] == value).sum())

    return {
        "episodes": len(group),
        "failures": len(failures),
        "detected": len(detected),
        "missed": len(failures) - len(detected),
        "detection_rate": len(detected) / len(failures) if len(failures) else None,
        "successes": len(successes),
        "false_positives": len(false_alarms),
        "false_positive_rate": len(false_alarms) / len(successes) if len(successes) else None,
        "median_lead_time": float(np.median(leads)) if leads.size else None,
        "q25_lead_time": float(np.quantile(leads, 0.25)) if leads.size else None,
        "q75_lead_time": float(np.quantile(leads, 0.75)) if leads.size else None,
        "median_warning_minus_first_move_detected": detected["warning_minus_first_move"].median(),
        "median_warning_minus_first_move_false_alarms": false_alarms[
            "warning_minus_first_move"
        ].median(),
        "median_warning_minus_first_success_false_alarms": false_alarms[
            "warning_minus_first_success"
        ].median(),
        "detected_late_by_lead": count(detected, "late_by_lead"),
        "detected_late_by_object_speed": count(detected, "late_by_object_speed"),
        "false_alarms_in_approach": count(false_alarms, "reference_phase_at_warning", APPROACH),
        "false_alarms_before_object_moved": count(false_alarms, "warned_before_object_moved"),
        "detected_in_approach": count(detected, "reference_phase_at_warning", APPROACH),
        "detected_before_object_moved": count(detected, "warned_before_object_moved"),
        "departure_warnings_detected": count(detected, "warning_kind", "departure"),
        "departure_warnings_false_alarms": count(false_alarms, "warning_kind", "departure"),
    }


def summarize(results: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for key, group in results.groupby(keys, dropna=False, sort=False):
        rows.append({**dict(zip(keys, key, strict=True)), **rates(group)})
    return pd.DataFrame(rows)


DETECTOR_KEYS = ["detector", "family", "window", "persistence", "hysteresis",
                 "departure_persistence"]  # fmt: skip


def overall(results: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for key, group in results.groupby(DETECTOR_KEYS, dropna=False, sort=False):
        disturbed = group[group["disturbance_type"] != "none"]
        undisturbed = group[group["disturbance_type"] == "none"]
        row = {**dict(zip(DETECTOR_KEYS, key, strict=True)), **rates(disturbed)}
        seeds = sorted(disturbed["training_seed"].unique())
        for label, failed in (("detection", True), ("fpr", False)):
            groups = [
                disturbed[(disturbed["training_seed"] == s) & (disturbed["failure"] == failed)][
                    "warned"
                ].to_numpy(dtype=float)
                for s in seeds
            ]
            row[f"{label}_hier_ci_low"], row[f"{label}_hier_ci_high"] = (
                hierarchical_bootstrap_interval(groups, resamples, 0.95, rng)
            )
        row["undisturbed_false_positives"] = int(undisturbed["warned"].sum())
        row["undisturbed_false_positive_rate"] = float(undisturbed["warned"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_comparison(results: pd.DataFrame, resamples: int, seed: int) -> pd.DataFrame:
    """Each phase-aware configuration against the distance-only detector with the same k and p,
    episode by episode, over the disturbed conditions."""
    rng = np.random.default_rng(seed)
    disturbed = results[results["disturbance_type"] != "none"]
    keys = EPISODE_KEYS
    rows = []
    for (detector, window, persistence), group in disturbed[
        disturbed["family"] == "phase_aware"
    ].groupby(["detector", "window", "persistence"], sort=False):
        baseline_name = f"distance_k{window}_p{persistence}"
        baseline = disturbed[disturbed["detector"] == baseline_name][[*keys, "warned"]]
        merged = group.merge(baseline, on=keys, suffixes=("", "_baseline"), validate="one_to_one")
        row: dict[str, Any] = {"detector": detector, "baseline": baseline_name}
        for label, failed in (("failures", True), ("successes", False)):
            part = merged[merged["failure"] == failed]
            ours, theirs = part["warned"].astype(bool), part["warned_baseline"].astype(bool)
            row[f"{label}"] = len(part)
            row[f"{label}_both_warned"] = int((ours & theirs).sum())
            row[f"{label}_only_phase_warned"] = int((ours & ~theirs).sum())
            row[f"{label}_only_baseline_warned"] = int((~ours & theirs).sum())
            differences = [
                (part[part["training_seed"] == s]["warned"].astype(float)
                 - part[part["training_seed"] == s]["warned_baseline"].astype(float)).to_numpy()
                for s in sorted(part["training_seed"].unique())
            ]  # fmt: skip
            name = "detection" if failed else "fpr"
            row[f"{name}_difference"] = float(ours.mean() - theirs.mean())
            row[f"{name}_difference_hier_ci_low"], row[f"{name}_difference_hier_ci_high"] = (
                hierarchical_bootstrap_interval(differences, resamples, 0.95, rng)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run_phase_evaluation(
    evaluation: PhaseEvaluation, output_dir: Path, config_path: Path | None = None
) -> dict[str, Path]:
    source = resolve(evaluation.source)
    episodes = pd.read_csv(source / "episodes.csv")
    episodes = episodes[episodes["env_id"] == evaluation.env_id].reset_index(drop=True)
    states = load_states(source, evaluation.env_id)
    if len(states) != len(episodes):
        raise ValueError(f"{len(states)} trajectories for {len(episodes)} episodes")
    matched = None
    if evaluation.reference_source:
        matched = check_matched(states, resolve(evaluation.reference_source), evaluation.env_id)
    speed_reference = object_speed_reference(states, episodes, evaluation.speed_reference_quantile)
    results = evaluate(evaluation, episodes, states, speed_reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "episodes": results,
        "overall": overall(results, evaluation.bootstrap_resamples, evaluation.bootstrap_seed),
        "by_condition": summarize(results, [*DETECTOR_KEYS, "condition", *CONDITION_COLUMNS]),
        "per_seed": summarize(results, ["detector", "training_seed", "condition"]),
        "paired": paired_comparison(
            results, evaluation.bootstrap_resamples, evaluation.bootstrap_seed + 1
        ),
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        suffix = ".csv.gz" if name == "episodes" else ".csv"
        paths[name] = output_dir / f"phase_detector_{name}{suffix}"
        table.to_csv(paths[name], index=False)
    from farc.detection.phase_aware_plots import plot_phase_results
    from farc.utils.run_info import git_state

    paths.update(plot_phase_results(tables["overall"], tables["paired"], results, output_dir))
    source_metadata = json.loads((source / "metadata.json").read_text())
    metadata = {
        "evaluation": asdict(evaluation),
        "config_file": str(config_path) if config_path else None,
        "git": git_state(),
        "source": str(source),
        "source_git": source_metadata["run"]["git"],
        "source_stage": source_metadata["experiment"]["stage"],
        "source_episode_seeds": source_metadata["episode_seeds"],
        "source_episodes_sha256": hashlib.sha256(
            (source / "episodes.csv").read_bytes()
        ).hexdigest(),
        "matched_reference": matched,
        "object_speed_reference_m_per_step": speed_reference,
        "episodes": len(episodes),
        "labels": "failure = not successful at the final step (50)",
        "files": {k: p.name for k, p in paths.items()},
    }
    paths["metadata"] = output_dir / "phase_detector_metadata.json"
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return paths


def compare_detectors(
    results: pd.DataFrame,
    first: str,
    second: str,
    resamples: int,
    seed: int,
    min_lead: int | None = None,
) -> dict[str, Any]:
    """Episode-paired difference (first minus second) in detection rate and false-positive rate
    over the disturbed episodes, with seed-then-episode bootstrap intervals. Choosing which
    pairs to compare after seeing aggregate results makes these comparisons post hoc.

    With ``min_lead``, a warning counts only if more than ``min_lead`` steps remain when it
    fires (warnings with less time left are treated as no warning)."""
    rng = np.random.default_rng(seed)
    disturbed = results[results["disturbance_type"] != "none"].copy()
    if min_lead is not None:
        disturbed["warned"] = disturbed["warned"].astype(bool) & (
            disturbed["lead_time"].fillna(0) > min_lead
        )
    a = disturbed[disturbed["detector"] == first][[*EPISODE_KEYS, "failure", "warned"]]
    b = disturbed[disturbed["detector"] == second][[*EPISODE_KEYS, "warned"]]
    if a.empty or b.empty:
        raise KeyError(f"unknown detector {first if a.empty else second!r}")
    merged = a.merge(b, on=EPISODE_KEYS, suffixes=("", "_other"), validate="one_to_one")
    row: dict[str, Any] = {"first": first, "second": second, "min_lead": min_lead}
    for name, failed in (("detection", True), ("fpr", False)):
        part = merged[merged["failure"] == failed]
        ours, theirs = part["warned"].astype(bool), part["warned_other"].astype(bool)
        differences = [
            (part[part["training_seed"] == s]["warned"].astype(float)
             - part[part["training_seed"] == s]["warned_other"].astype(float)).to_numpy()
            for s in sorted(part["training_seed"].unique())
        ]  # fmt: skip
        low, high = hierarchical_bootstrap_interval(differences, resamples, 0.95, rng)
        row.update(
            {
                f"{name}_first": float(ours.mean()),
                f"{name}_second": float(theirs.mean()),
                f"{name}_difference": float(ours.mean() - theirs.mean()),
                f"{name}_difference_ci_low": low,
                f"{name}_difference_ci_high": high,
                f"{name}_only_first": int((ours & ~theirs).sum()),
                f"{name}_only_second": int((~ours & theirs).sum()),
            }
        )
    return row
