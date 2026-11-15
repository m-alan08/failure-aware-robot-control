# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/threshold_sweep.py
# (introduced in 91e44be, last changed in 3815180). See PROVENANCE.md.
"""Threshold-free comparison of the distance-only and phase-aware detectors (FetchPush).

Both detectors warn at the first state t where the deficit has exceeded the threshold theta at
p consecutive decision states. A state whose deficit is unavailable (window not yet filled
after a reset, inside the success radius, or in the maintain phase) breaks the run, and every
reset is followed by at least one unavailable state, so the persistence count of the original
detectors equals the length of the current run of consecutive available deficits above theta.

Let r_t be the minimum deficit over states t - p + 1 .. t (minus infinity if any of them is
unavailable). The first warning at threshold theta is the first t with r_t > theta, so

- the detector warns at all iff max_t r_t > theta;
- its first warning leaves more than L steps (T - t > L) iff max_{t < T - L} r_t > theta.

These maxima are per-episode scores: every threshold's outcome follows from them exactly, so
the detection-vs-FPR curve is computed over all thresholds rather than a grid. The phase-aware
detector's departure monitor does not depend on theta; when it is on, a departure warning
early enough for the lead requirement sets the score to +infinity.

Actionable trade-off: detection counts a failure only if the first warning leaves more than L
steps; false positives count every warning on a successful episode, whenever it fires.
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

from farc.detection.phase_aware import PhaseDetectorConfig, run_phase_detector
from farc.detection.phase_aware_eval import EPISODE_KEYS, load_states
from farc.detection.progress_deficit import DetectorConfig, run_detector
from farc.paths import resolve

CONDITION_COLUMNS = ["condition", "disturbance_type", "severity", "onset", "duration", "regime"]


class SweepConfigError(ValueError):
    pass


def run_minimum(deficit: np.ndarray, persistence: int) -> np.ndarray:
    """r_t: minimum deficit over the last ``persistence`` states, -inf if any is unavailable or
    fewer than ``persistence`` states exist."""
    values = np.where(np.isnan(deficit), -np.inf, np.asarray(deficit, float))
    result = np.full(len(values), -np.inf)
    for t in range(persistence - 1, len(values)):
        result[t] = values[t - persistence + 1 : t + 1].min()
    return result


def lead_score(run_min: np.ndarray, min_lead: int, departure_state: int | None = None) -> float:
    """Largest threshold at which the first warning leaves more than ``min_lead`` steps
    (strictly: the detector warns in time iff threshold < score)."""
    horizon = len(run_min)
    last = horizon - min_lead  # states t < last leave more than min_lead steps
    if departure_state is not None and departure_state < last:
        return float("inf")
    window = run_min[: max(last, 0)]
    return float(window.max()) if window.size else float("-inf")


def first_departure(departure_count: np.ndarray, persistence: int | None) -> int | None:
    if persistence is None:
        return None
    hits = np.flatnonzero(departure_count >= persistence)
    return int(hits[0]) if hits.size else None


@dataclass(frozen=True)
class SweepConfig:
    name: str
    source: str
    env_id: str
    windows: tuple[int, ...]
    persistence: tuple[int, ...]
    phase_hysteresis: int
    phase_departure_persistence: tuple[int | None, ...]
    contact_enter: float
    contact_exit: float
    behind_offset: float
    success_threshold: float
    min_leads: tuple[int, ...]
    target_fprs: tuple[float, ...]
    breakdown_fpr: float
    bootstrap_resamples: int
    bootstrap_seed: int


def load_sweep_config(path: str | Path) -> SweepConfig:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    fields = set(SweepConfig.__dataclass_fields__)
    unknown, missing = set(values) - fields, fields - set(values)
    if unknown or missing:
        raise SweepConfigError(f"unknown keys {sorted(unknown)}, missing {sorted(missing)}")
    for key in ("windows", "persistence", "phase_departure_persistence", "min_leads",
                "target_fprs"):  # fmt: skip
        if not isinstance(values[key], list) or not values[key]:
            raise SweepConfigError(f"{key} must be a non-empty list")
        values[key] = tuple(values[key])
    config = SweepConfig(**values)
    if not all(0 < f < 1 for f in config.target_fprs) or not 0 < config.breakdown_fpr < 1:
        raise SweepConfigError("target FPRs must lie in (0, 1)")
    if not all(isinstance(lead, int) and lead >= 0 for lead in config.min_leads):
        raise SweepConfigError("min_leads must be non-negative integers")
    return config


def detector_names(config: SweepConfig) -> list[tuple[str, str, int, int, int | None]]:
    names: list[tuple[str, str, int, int, int | None]] = []
    for k, p in itertools.product(config.windows, config.persistence):
        names.append((f"distance_k{k}_p{p}", "distance_only", k, p, None))
        for m in config.phase_departure_persistence:
            label = "off" if m is None else m
            names.append(
                (f"phase_k{k}_p{p}_h{config.phase_hysteresis}_m{label}", "phase_aware", k, p, m)
            )
    return names


def episode_scores(arrays: dict[str, np.ndarray], config: SweepConfig) -> dict[str, float]:
    scores: dict[str, float] = {}
    for k, p in itertools.product(config.windows, config.persistence):
        distance = run_detector(
            arrays["objects"], arrays["goals"], DetectorConfig(k, p, 0.0, config.success_threshold)
        )
        run_min = run_minimum(distance.deficit, p)
        for lead in config.min_leads:
            scores[f"distance_k{k}_p{p}|{lead}"] = lead_score(run_min, lead)
        phase = run_phase_detector(
            arrays["gripper"],
            arrays["objects"],
            arrays["goals"],
            PhaseDetectorConfig(
                k, p, config.phase_hysteresis, None, config.contact_enter, config.contact_exit,
                config.behind_offset, 0.0, config.success_threshold,
            ),
        )  # fmt: skip
        phase_min = run_minimum(phase.deficit, p)
        for m in config.phase_departure_persistence:
            departure = first_departure(phase.departure_count, m)
            label = "off" if m is None else m
            for lead in config.min_leads:
                key = f"phase_k{k}_p{p}_h{config.phase_hysteresis}_m{label}|{lead}"
                scores[key] = lead_score(phase_min, lead, departure)
    return scores


def compute_scores(
    config: SweepConfig, episodes: pd.DataFrame, states: dict[tuple, dict[str, np.ndarray]]
) -> pd.DataFrame:
    rows = []
    for record in episodes.to_dict("records"):
        arrays = states[tuple(record[k] for k in EPISODE_KEYS)]
        row = {k: record[k] for k in [*EPISODE_KEYS, *CONDITION_COLUMNS[1:]]}
        row["failure"] = not bool(record["final_success"])
        row.update(episode_scores(arrays, config))
        rows.append(row)
    return pd.DataFrame(rows)


def threshold_for_fpr(success_any: np.ndarray, target: float) -> float:
    """Smallest threshold whose false-positive rate is at most ``target``: warn iff
    score > threshold, allowing floor(target * n) successful episodes to warn.

    Scores of +inf (threshold-independent departure warnings) always warn; if there are more
    of them than allowed, no threshold reaches the target and NaN is returned."""
    ordered = np.sort(success_any)
    allowed = int(np.floor(target * len(ordered)))
    if int(np.isposinf(ordered).sum()) > allowed:
        return float("nan")
    return float(ordered[len(ordered) - allowed - 1])


def rates_at(threshold: float, failure_scores: np.ndarray, success_any: np.ndarray):
    if np.isnan(threshold):
        return float("nan"), float("nan")
    return float(np.mean(failure_scores > threshold)), float(np.mean(success_any > threshold))


def exact_curve(failure_scores: np.ndarray, success_any: np.ndarray) -> pd.DataFrame:
    """Detection and FPR at every distinct threshold (all finite score values, plus the
    extremes)."""
    candidates = np.unique(np.concatenate([failure_scores, success_any]))
    candidates = candidates[np.isfinite(candidates)]
    # +inf scores warn at every threshold, so the curve ends at the largest finite threshold
    thresholds = np.concatenate([[-np.inf], candidates if candidates.size else [0.0]])
    fail_sorted = np.sort(failure_scores)
    succ_sorted = np.sort(success_any)
    detection = 1.0 - np.searchsorted(fail_sorted, thresholds, side="right") / len(fail_sorted)
    fpr = 1.0 - np.searchsorted(succ_sorted, thresholds, side="right") / len(succ_sorted)
    return pd.DataFrame({"threshold": thresholds, "detection": detection, "fpr": fpr})


def detection_at_fpr(failure_scores, success_any, target) -> tuple[float, float, float]:
    threshold = threshold_for_fpr(success_any, target)
    detection, fpr = rates_at(threshold, failure_scores, success_any)
    return threshold, detection, fpr


def _stratified_indices(strata: np.ndarray, resamples: int, rng: np.random.Generator):
    """(resamples, n) episode indices, resampled with replacement within each stratum."""
    columns = []
    for value in np.unique(strata):
        members = np.flatnonzero(strata == value)
        columns.append(members[rng.integers(0, len(members), size=(resamples, len(members)))])
    return np.concatenate(columns, axis=1)


def matched_tables(
    scores: pd.DataFrame, config: SweepConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detection at matched FPR for every detector and lead requirement, family envelopes,
    and paired phase-minus-distance differences, with stratified episode bootstrap
    intervals (strata: training seed x condition). Thresholds are re-chosen in every resample."""
    disturbed = scores[scores["disturbance_type"] != "none"].reset_index(drop=True)
    failure = disturbed["failure"].to_numpy(bool)
    strata = (disturbed["training_seed"].astype(str) + "|" + disturbed["condition"]).to_numpy()
    rng = np.random.default_rng(config.bootstrap_seed)
    indices = _stratified_indices(strata, config.bootstrap_resamples, rng)
    detectors = detector_names(config)

    def det_at(name: str, lead: int, target: float, index: np.ndarray | None) -> float:
        lead_scores = disturbed[f"{name}|{lead}"].to_numpy(float)
        any_scores = disturbed[f"{name}|0"].to_numpy(float)
        fail = failure if index is None else failure[index]
        s_lead = lead_scores if index is None else lead_scores[index]
        s_any = any_scores if index is None else any_scores[index]
        return detection_at_fpr(s_lead[fail], s_any[~fail], target)[1]

    point: dict[tuple, float] = {}
    boot: dict[tuple, np.ndarray] = {}
    for name, *_ in detectors:
        for lead in config.min_leads:
            for target in config.target_fprs:
                point[(name, lead, target)] = det_at(name, lead, target, None)
                boot[(name, lead, target)] = np.array(
                    [det_at(name, lead, target, index) for index in indices]
                )
    rows = []
    for name, family, k, p, m in detectors:
        for lead in config.min_leads:
            for target in config.target_fprs:
                any_scores = disturbed[f"{name}|0"].to_numpy(float)
                threshold, _, actual_fpr = detection_at_fpr(
                    disturbed[f"{name}|{lead}"].to_numpy(float)[failure],
                    any_scores[~failure],
                    target,
                )
                low, high = np.nanquantile(boot[(name, lead, target)], [0.025, 0.975])
                rows.append({
                    "detector": name, "family": family, "window": k, "persistence": p,
                    "departure_persistence": m, "min_lead": lead, "target_fpr": target,
                    "threshold": threshold, "actual_fpr": actual_fpr,
                    "detection": point[(name, lead, target)], "ci_low": low, "ci_high": high,
                })  # fmt: skip
    matched = pd.DataFrame(rows)

    families = {
        "distance_only": [d[0] for d in detectors if d[1] == "distance_only"],
        "phase_aware_m_off": [d[0] for d in detectors if d[1] == "phase_aware" and d[4] is None],
        "phase_aware_all": [d[0] for d in detectors if d[1] == "phase_aware"],
    }
    envelope_rows = []
    envelope_boot: dict[tuple, np.ndarray] = {}
    for family, members in families.items():
        if not members:
            continue
        for lead in config.min_leads:
            for target in config.target_fprs:
                values = [point[(m, lead, target)] for m in members]
                if all(np.isnan(values)):
                    continue
                best = members[int(np.nanargmax(values))]
                stacked = np.array([boot[(m, lead, target)] for m in members])
                samples = np.where(
                    np.isnan(stacked).all(axis=0), np.nan,
                    np.nanmax(np.where(np.isnan(stacked), -np.inf, stacked), axis=0),
                )  # fmt: skip
                envelope_boot[(family, lead, target)] = samples
                low, high = np.nanquantile(samples, [0.025, 0.975])
                envelope_rows.append({
                    "family": family, "min_lead": lead, "target_fpr": target,
                    "detection": float(np.nanmax(values)), "best_member": best, "ci_low": low,
                    "ci_high": high,
                })  # fmt: skip
    envelopes = pd.DataFrame(envelope_rows)

    paired_rows = []
    pairs = [
        (f"phase_k{k}_p{p}_h{config.phase_hysteresis}_m{'off' if m is None else m}",
         f"distance_k{k}_p{p}")
        for k, p in itertools.product(config.windows, config.persistence)
        for m in config.phase_departure_persistence
    ]  # fmt: skip
    for first, second in pairs:
        for lead in config.min_leads:
            for target in config.target_fprs:
                diff = boot[(first, lead, target)] - boot[(second, lead, target)]
                if np.isnan(diff).all():
                    low = high = float("nan")
                else:
                    low, high = np.nanquantile(diff, [0.025, 0.975])
                paired_rows.append({
                    "first": first, "second": second, "min_lead": lead, "target_fpr": target,
                    "difference": point[(first, lead, target)] - point[(second, lead, target)],
                    "ci_low": low, "ci_high": high,
                })  # fmt: skip
    for first, second in (("phase_aware_m_off", "distance_only"),
                          ("phase_aware_all", "distance_only")):  # fmt: skip
        for lead in config.min_leads:
            for target in config.target_fprs:
                if (first, lead, target) not in envelope_boot:
                    continue
                diff = envelope_boot[(first, lead, target)] - envelope_boot[(second, lead, target)]
                low, high = np.nanquantile(diff, [0.025, 0.975])
                a = envelopes.query(
                    "family == @first and min_lead == @lead and target_fpr == @target"
                )
                b = envelopes.query(
                    "family == @second and min_lead == @lead and target_fpr == @target"
                )
                paired_rows.append({
                    "first": f"envelope:{first}", "second": f"envelope:{second}",
                    "min_lead": lead, "target_fpr": target,
                    "difference": float(a["detection"].iloc[0] - b["detection"].iloc[0]),
                    "ci_low": low, "ci_high": high,
                })  # fmt: skip
    return matched, envelopes, pd.DataFrame(paired_rows)


def by_condition(scores: pd.DataFrame, config: SweepConfig) -> pd.DataFrame:
    """Per-condition rates for every detector at the threshold that gives each detector the
    pooled ``breakdown_fpr`` on disturbed successes."""
    disturbed = scores[scores["disturbance_type"] != "none"]
    failure = disturbed["failure"].to_numpy(bool)
    rows = []
    for name, family, k, p, m in detector_names(config):
        threshold = threshold_for_fpr(
            disturbed[f"{name}|0"].to_numpy(float)[~failure], config.breakdown_fpr
        )
        if np.isnan(threshold):
            continue  # this detector cannot reach the breakdown FPR (departure warnings)
        for key, group in scores.groupby(CONDITION_COLUMNS, dropna=False, sort=False):
            fails = group[group["failure"]]
            succs = group[~group["failure"]]
            row: dict[str, Any] = {
                "detector": name, "family": family, "window": k, "persistence": p,
                "departure_persistence": m, "pooled_fpr_target": config.breakdown_fpr,
                "threshold": threshold, **dict(zip(CONDITION_COLUMNS, key, strict=True)),
                "failures": len(fails), "successes": len(succs),
                "fpr": float((succs[f"{name}|0"] > threshold).mean()) if len(succs) else None,
            }  # fmt: skip
            for lead in config.min_leads:
                row[f"detection_lead_gt_{lead}"] = (
                    float((fails[f"{name}|{lead}"] > threshold).mean()) if len(fails) else None
                )
            rows.append(row)
    return pd.DataFrame(rows)


def per_seed_matched(scores: pd.DataFrame, config: SweepConfig) -> pd.DataFrame:
    """Matched-FPR detection within each training seed (threshold chosen within the seed),
    for every detector and the two family envelopes. Point estimates only: a consistency
    check across the three trained policies, which the episode bootstrap does not resample."""
    disturbed = scores[scores["disturbance_type"] != "none"]
    detectors = detector_names(config)
    rows = []
    for seed, group in disturbed.groupby("training_seed"):
        failure = group["failure"].to_numpy(bool)
        values: dict[tuple, float] = {}
        for name, *_ in detectors:
            any_scores = group[f"{name}|0"].to_numpy(float)
            for lead in config.min_leads:
                lead_scores = group[f"{name}|{lead}"].to_numpy(float)
                for target in config.target_fprs:
                    value = detection_at_fpr(lead_scores[failure], any_scores[~failure], target)[1]
                    values[(name, lead, target)] = value
                    rows.append({"training_seed": seed, "detector": name, "min_lead": lead,
                                 "target_fpr": target, "detection": value})  # fmt: skip
        for family, members in (
            ("envelope:distance_only", [d[0] for d in detectors if d[1] == "distance_only"]),
            ("envelope:phase_aware_m_off",
             [d[0] for d in detectors if d[1] == "phase_aware" and d[4] is None]),
        ):  # fmt: skip
            for lead in config.min_leads:
                for target in config.target_fprs:
                    best = np.nanmax([values[(m, lead, target)] for m in members])
                    rows.append({"training_seed": seed, "detector": family, "min_lead": lead,
                                 "target_fpr": target, "detection": float(best)})  # fmt: skip
    return pd.DataFrame(rows)


def curves(scores: pd.DataFrame, config: SweepConfig) -> pd.DataFrame:
    disturbed = scores[scores["disturbance_type"] != "none"]
    failure = disturbed["failure"].to_numpy(bool)
    frames = []
    for name, family, *_ in detector_names(config):
        success_any = disturbed[f"{name}|0"].to_numpy(float)[~failure]
        for lead in config.min_leads:
            curve = exact_curve(disturbed[f"{name}|{lead}"].to_numpy(float)[failure], success_any)
            curve.insert(0, "min_lead", lead)
            curve.insert(0, "family", family)
            curve.insert(0, "detector", name)
            frames.append(curve)
    return pd.concat(frames, ignore_index=True)


def curve_on_grid(full: pd.DataFrame, step: float = 0.005) -> pd.DataFrame:
    """Best detection reachable at FPR <= f on a regular FPR grid (compact copy of the curves)."""
    grid = np.round(np.arange(0.0, 1.0 + step / 2, step), 6)
    rows = []
    for (detector, family, lead), group in full.groupby(["detector", "family", "min_lead"]):
        for f in grid:
            allowed = group[group["fpr"] <= f + 1e-12]
            rows.append({"detector": detector, "family": family, "min_lead": lead, "fpr": f,
                         "detection": float(allowed["detection"].max())})  # fmt: skip
    return pd.DataFrame(rows)


def run_sweep(config: SweepConfig, output_dir: Path, config_path: Path | None = None):
    source = resolve(config.source)
    episodes = pd.read_csv(source / "episodes.csv")
    episodes = episodes[episodes["env_id"] == config.env_id].reset_index(drop=True)
    states = load_states(source, config.env_id)
    if len(states) != len(episodes):
        raise ValueError(f"{len(states)} trajectories for {len(episodes)} episodes")
    scores = compute_scores(config, episodes, states)
    matched, envelopes, paired = matched_tables(scores, config)
    full_curves = curves(scores, config)
    tables = {
        "scores": scores,
        "matched": matched,
        "envelopes": envelopes,
        "paired": paired,
        "by_condition": by_condition(scores, config),
        "per_seed": per_seed_matched(scores, config),
        "curves": full_curves,
        "curves_grid": curve_on_grid(full_curves),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        suffix = ".csv.gz" if name in ("scores", "curves") else ".csv"
        paths[name] = output_dir / f"sweep_{name}{suffix}"
        table.to_csv(paths[name], index=False)
    from farc.detection.threshold_sweep_plots import plot_sweep
    from farc.utils.run_info import git_state

    paths.update(plot_sweep(tables["curves_grid"], matched, envelopes, paired,
                            tables["by_condition"], config, output_dir))  # fmt: skip
    source_metadata = json.loads((source / "metadata.json").read_text())
    metadata = {
        "config": asdict(config),
        "config_file": str(config_path) if config_path else None,
        "git": git_state(),
        "source": str(source),
        "source_git": source_metadata["run"]["git"],
        "source_stage": source_metadata["experiment"]["stage"],
        "source_episode_seeds": source_metadata["episode_seeds"],
        "source_episodes_sha256": hashlib.sha256(
            (source / "episodes.csv").read_bytes()
        ).hexdigest(),
        "episodes": len(episodes),
        "disturbed_failures": int(scores[scores["disturbance_type"] != "none"]["failure"].sum()),
        "disturbed_successes": int(
            (~scores[scores["disturbance_type"] != "none"]["failure"]).sum()
        ),
        "bootstrap": "stratified episode resampling within training seed x condition; "
        "thresholds re-chosen in every resample; training seeds are not resampled",
        "status": "exploratory: validation trajectories already informed detector design",
        "files": {k: p.name for k, p in paths.items()},
    }
    paths["metadata"] = output_dir / "sweep_metadata.json"
    paths["metadata"].write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    return paths
