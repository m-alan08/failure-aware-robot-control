"""Phase B evaluation of learned failure predictors against the reactive detectors.

Everything is evaluated on the episode-seed-disjoint validation split of the dataset
(``build_datasets(split_seed=0, validation_fraction=0.2)``). That split's 20 episode seeds are
divided again, by seed, into a *calibration* half (thresholds are chosen there) and an
*evaluation* half (every reported warning metric is computed there).

Per-step scores, decision steps t = 0..T-1 (T = 50, so T - t >= 1 steps remain):

- learned predictor: sigmoid of the MLP logit on the normalized features of step t;
- reactive detectors: r_t, the minimum deficit over the last p decision states
  (``threshold_sweep.run_minimum``), so "first t with r_t > theta" is exactly the original
  detector's first warning at threshold theta.

Every detector warns at the first step whose score exceeds the threshold theta.

Episode-level warning metrics (``disturbed`` episodes pooled, as in the Phase 1 sweep):

- FPR: fraction of successful episodes with any warning at t <= T-1;
- detection (lead > L): fraction of failed episodes whose *first* warning is at a step t with
  T - t > L, i.e. t < T - L. L = 0 is "any warning"; L = 5 and L = 10 are the strict lead
  criteria (t <= 44 and t <= 39). A first warning at or after step T - L does not count, and a
  later warning cannot rescue it, because only the first warning is used.

Because the first warning at theta is the first t with s_t > theta, a failure is detected with
lead > L iff max_{t < T-L} s_t > theta, so all rates follow from three per-episode maxima.

Thresholds: for each target FPR, the smallest threshold whose FPR on the *calibration*
successes is at most the target (``threshold_sweep.threshold_for_fpr``). The realized FPR on
the evaluation half can be above or below the target; it is reported.

Uncertainty: a two-sample cluster bootstrap. In each resample the calibration seeds and the
evaluation seeds are drawn with replacement, separately; each drawn seed brings all its
episodes (every condition and every policy seed), so the dependence between episodes that
share an initial state is kept. Thresholds are re-chosen on each calibration resample and
differences are paired on the same evaluation resample.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from farc.detection.phase_aware import PhaseDetectorConfig, run_phase_detector
from farc.detection.progress_deficit import DetectorConfig, run_detector
from farc.detection.threshold_sweep import run_minimum, threshold_for_fpr
from farc.prediction.dataset import FailureDatasets, Normalizer, split_episode_seeds
from farc.prediction.features import NUM_FEATURES

HIDDEN = (128, 64)


class CheckpointError(RuntimeError):
    pass


class FailurePredictor(nn.Module):
    """The trained architecture (24 -> 128 -> 64 -> 1, ReLU), as in
    ``scripts/train_failure_predictor.py``. Parameter names must match the checkpoints."""

    def __init__(self, input_dim: int = NUM_FEATURES):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, HIDDEN[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN[0], HIDDEN[1]),
            nn.ReLU(),
            nn.Linear(HIDDEN[1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


EXPECTED_SHAPES = {
    "network.0.weight": (HIDDEN[0], NUM_FEATURES),
    "network.0.bias": (HIDDEN[0],),
    "network.2.weight": (HIDDEN[1], HIDDEN[0]),
    "network.2.bias": (HIDDEN[1],),
    "network.4.weight": (1, HIDDEN[1]),
    "network.4.bias": (1,),
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_predictor(path: str | Path) -> FailurePredictor:
    """Load a state-dict checkpoint, refusing anything that does not match the architecture."""
    path = Path(path)
    if not path.exists():
        raise CheckpointError(f"checkpoint {path} does not exist")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise CheckpointError(f"{path} is not a state dict")
    shapes = {k: tuple(v.shape) for k, v in state.items()}
    if shapes != EXPECTED_SHAPES:
        raise CheckpointError(f"{path}: parameters {shapes} != expected {EXPECTED_SHAPES}")
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise CheckpointError(f"{path} has non-finite parameters")
    model = FailurePredictor()
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def verify_normalizer(normalizer: Normalizer, reference: dict[str, Any], atol: float = 1e-9):
    """Check the recomputed training-set normalizer against a saved dataset report."""
    saved = reference.get("normalizer")
    if saved is None:
        raise CheckpointError("the dataset report has no normalizer statistics")
    for key in ("mean", "std"):
        ours = getattr(normalizer, key)
        theirs = np.asarray(saved[key])
        if ours.shape != theirs.shape or not np.allclose(ours, theirs, rtol=0, atol=atol):
            raise CheckpointError(f"training-set normalizer {key} differs from the report")


@torch.no_grad()
def predict_probabilities(model: FailurePredictor, features: np.ndarray) -> np.ndarray:
    """(E, T, F) normalized features -> (E, T) failure probabilities."""
    x = torch.from_numpy(np.asarray(features, np.float32))
    return torch.sigmoid(model(x)).double().numpy()


@dataclass(frozen=True)
class ReactiveSpec:
    name: str
    family: str  # "distance_only" or "phase_aware"
    window: int
    persistence: int
    hysteresis: int = 3


def reactive_scores(
    spec: ReactiveSpec,
    objects: np.ndarray,
    grippers: np.ndarray,
    goals: np.ndarray,
    geometry: dict[str, float],
) -> np.ndarray:
    """Per-decision-step run minimum r_t (length T) of one reactive detector."""
    if spec.family == "distance_only":
        trace = run_detector(
            objects,
            goals,
            DetectorConfig(spec.window, spec.persistence, 0.0, geometry["success_threshold"]),
        )
        deficit = trace.deficit
    elif spec.family == "phase_aware":
        phase = run_phase_detector(
            grippers,
            objects,
            goals,
            PhaseDetectorConfig(
                spec.window,
                spec.persistence,
                spec.hysteresis,
                None,  # departure monitor off: the threshold-dependent family
                geometry["contact_enter"],
                geometry["contact_exit"],
                geometry["behind_offset"],
                0.0,
                geometry["success_threshold"],
            ),
        )
        deficit = phase.deficit
    else:
        raise ValueError(f"unknown family {spec.family!r}")
    return run_minimum(deficit, spec.persistence)


def lead_maxima(scores: np.ndarray, leads: tuple[int, ...]) -> dict[int, np.ndarray]:
    """For (E, T) scores, max over t < T - L of each episode (-inf if empty)."""
    horizon = scores.shape[1]
    out = {}
    for lead in leads:
        last = horizon - lead
        out[lead] = scores[:, :last].max(axis=1) if last > 0 else np.full(len(scores), -np.inf)
    return out


def first_warning(scores: np.ndarray, threshold: float) -> np.ndarray:
    """First step with score > threshold per episode, NaN if none (or threshold NaN)."""
    if np.isnan(threshold):
        return np.full(len(scores), np.nan)
    hit = scores > threshold
    first = hit.argmax(axis=1).astype(float)
    first[~hit.any(axis=1)] = np.nan
    return first


def warning_metrics(
    scores: np.ndarray, failure: np.ndarray, threshold: float, leads: tuple[int, ...]
) -> dict[str, float]:
    horizon = scores.shape[1]
    failure = np.asarray(failure, bool)
    first = first_warning(scores, threshold)
    warned = ~np.isnan(first)
    out: dict[str, float] = {
        "episodes": float(len(failure)),
        "failures": float(failure.sum()),
        "successes": float((~failure).sum()),
        "fpr": float(warned[~failure].mean()) if (~failure).any() else float("nan"),
    }
    for lead in leads:
        early = warned & (horizon - np.nan_to_num(first, nan=horizon) > lead)
        out[f"detection_lead{lead}"] = (
            float(early[failure].mean()) if failure.any() else float("nan")
        )
    detected = first[failure & warned]
    out["median_first_warning_t"] = float(np.median(detected)) if detected.size else float("nan")
    out["mean_first_warning_t"] = float(np.mean(detected)) if detected.size else float("nan")
    out["median_steps_remaining"] = (
        float(np.median(horizon - detected)) if detected.size else float("nan")
    )
    false = first[~failure & warned]
    out["median_false_warning_t"] = float(np.median(false)) if false.size else float("nan")
    return out


# --- timestep-level metrics -------------------------------------------------------------


def auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUROC with average ranks for ties."""
    labels = np.asarray(labels, bool)
    n_pos, n_neg = int(labels.sum()), int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(np.asarray(scores, float)).rank(method="average").to_numpy()
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC as average precision (step-wise, ties grouped)."""
    labels = np.asarray(labels, bool)
    if not labels.any():
        return float("nan")
    order = np.argsort(-np.asarray(scores, float), kind="mergesort")
    s, y = np.asarray(scores, float)[order], labels[order]
    distinct = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / labels.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities) - np.asarray(labels, float)) ** 2))


def reliability(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    p = np.asarray(probabilities, float)
    y = np.asarray(labels, float)
    index = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        rows.append(
            {
                "bin_low": b / bins,
                "bin_high": (b + 1) / bins,
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()) if mask.any() else float("nan"),
                "observed_failure_rate": float(y[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10):
    table = reliability(labels, probabilities, bins)
    filled = table[table["count"] > 0]
    gap = (filled["mean_predicted"] - filled["observed_failure_rate"]).abs()
    return float((gap * filled["count"]).sum() / filled["count"].sum())


def timestep_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels).ravel()
    probabilities = np.asarray(probabilities).ravel()
    return {
        "examples": float(labels.size),
        "failure_fraction": float(np.mean(labels)),
        "auroc": auroc(labels, probabilities),
        "pr_auc": average_precision(labels, probabilities),
        "brier": brier(labels, probabilities),
        "brier_constant_baseline": brier(labels, np.full(labels.size, np.mean(labels))),
        "ece_10bin": expected_calibration_error(labels, probabilities),
        "mean_predicted": float(np.mean(probabilities)),
    }


# --- the evaluation ----------------------------------------------------------------------


@dataclass
class EvaluationData:
    """Validation episodes with per-step scores of every detector, aligned by row."""

    episodes: pd.DataFrame  # one row per validation episode; column ``part`` in {calibration,
    # evaluation}; ``disturbed`` bool
    scores: dict[str, np.ndarray]  # detector -> (E, T)
    learned: tuple[str, ...]
    calibration_seeds: np.ndarray
    evaluation_seeds: np.ndarray


def calibration_split(
    validation_seeds: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """(calibration, evaluation) episode seeds; the calibration part has ``fraction``."""
    evaluation, calibration = split_episode_seeds(validation_seeds, 1 - fraction, seed)
    return calibration, evaluation


def assemble(
    datasets: FailureDatasets,
    predictors: dict[str, FailurePredictor],
    reactive: list[ReactiveSpec],
    states: dict[tuple, dict[str, np.ndarray]],
    geometry: dict[str, float],
    calibration_fraction: float,
    calibration_seed: int,
) -> EvaluationData:
    val = datasets.validation
    episodes = val.episodes.reset_index().copy()
    horizon = val.horizon
    normalized = val.features.numpy().reshape(len(episodes), horizon, NUM_FEATURES)
    scores = {name: predict_probabilities(m, normalized) for name, m in predictors.items()}
    for spec in reactive:
        rows = []
        for record in episodes.to_dict("records"):
            key = (record["env_id"], record["policy_seed"], record["condition"],
                   record["episode_seed"])  # fmt: skip
            a = states[key]
            rows.append(reactive_scores(spec, a["objects"], a["gripper"], a["goals"], geometry))
        scores[spec.name] = np.stack(rows)
        if scores[spec.name].shape != (len(episodes), horizon):
            raise ValueError(f"{spec.name}: scores are not aligned with the decision steps")
    calibration, evaluation = calibration_split(
        datasets.validation_seeds, calibration_fraction, calibration_seed
    )
    episodes["part"] = np.where(
        episodes["episode_seed"].isin(calibration), "calibration", "evaluation"
    )
    episodes["disturbed"] = episodes["condition"] != "undisturbed"
    return EvaluationData(episodes, scores, tuple(predictors), calibration, evaluation)


def thresholds(
    data: EvaluationData, target_fprs: tuple[float, ...], mask: np.ndarray | None = None
) -> dict[tuple[str, float], float]:
    """Thresholds from calibration-part disturbed successes (or the given episode mask)."""
    e = data.episodes
    if mask is None:
        mask = (e["part"] == "calibration").to_numpy() & e["disturbed"].to_numpy()
    successes = mask & (e["failure"].to_numpy() == 0)
    out = {}
    for name, s in data.scores.items():
        any_warning = s[successes].max(axis=1)
        for target in target_fprs:
            out[(name, target)] = threshold_for_fpr(any_warning, target)
    return out


def metric_table(
    data: EvaluationData,
    chosen: dict[tuple[str, float], float],
    leads: tuple[int, ...],
    group_by: str | None,
    part: str = "evaluation",
    disturbed_only: bool = True,
) -> pd.DataFrame:
    e = data.episodes
    base = (e["part"] == part).to_numpy().copy()
    if disturbed_only:
        base &= e["disturbed"].to_numpy()
    groups: list[tuple[Any, np.ndarray]] = (
        [("pooled", base)]
        if group_by is None
        else [
            (g, base & (e[group_by] == g).to_numpy())
            for g in sorted(e.loc[base, group_by].unique())
        ]
    )
    rows = []
    for (name, target), theta in chosen.items():
        for group, mask in groups:
            m = warning_metrics(
                data.scores[name][mask], e["failure"].to_numpy()[mask], theta, leads
            )
            rows.append({"detector": name, "target_fpr": target, "threshold": theta,
                         "group": group, **m})  # fmt: skip
    table = pd.DataFrame(rows)
    if group_by is not None:
        table = table.rename(columns={"group": group_by})
    return table


def _seed_index(episodes: pd.DataFrame, mask: np.ndarray) -> dict[int, np.ndarray]:
    sub = episodes[mask]
    return {int(s): sub.index[sub["episode_seed"] == s].to_numpy() for s in
            sorted(sub["episode_seed"].unique())}  # fmt: skip


def bootstrap(
    data: EvaluationData,
    target_fprs: tuple[float, ...],
    leads: tuple[int, ...],
    pairs: list[tuple[str, str]],
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster-bootstrap intervals for detection rates, FPRs, and paired differences."""
    e = data.episodes
    disturbed = e["disturbed"].to_numpy()
    failure = e["failure"].to_numpy().astype(bool)
    calib = _seed_index(e, (e["part"] == "calibration").to_numpy() & disturbed)
    evals = _seed_index(e, (e["part"] == "evaluation").to_numpy() & disturbed)
    maxima = {name: lead_maxima(s, leads) for name, s in data.scores.items()}
    names = list(data.scores)
    rng = np.random.default_rng(seed)
    cal_keys, eval_keys = np.array(list(calib)), np.array(list(evals))
    # stat[(name, target, metric)] -> list over resamples
    stats: dict[tuple[str, float, str], list[float]] = {}
    for _ in range(resamples):
        ci = np.concatenate([calib[s] for s in rng.choice(cal_keys, len(cal_keys))])
        ei = np.concatenate([evals[s] for s in rng.choice(eval_keys, len(eval_keys))])
        cal_success = ci[~failure[ci]]
        ev_fail, ev_success = ei[failure[ei]], ei[~failure[ei]]
        for name in names:
            any_cal = maxima[name][0][cal_success]
            for target in target_fprs:
                theta = threshold_for_fpr(any_cal, target)
                values = {"fpr": float(np.mean(maxima[name][0][ev_success] > theta))}
                for lead in leads:
                    values[f"detection_lead{lead}"] = float(
                        np.mean(maxima[name][lead][ev_fail] > theta)
                    )
                if np.isnan(theta):
                    values = dict.fromkeys(values, float("nan"))
                for metric, v in values.items():
                    stats.setdefault((name, target, metric), []).append(v)
    alpha = (1 - confidence) / 2

    def interval(values: np.ndarray) -> tuple[float, float]:
        values = values[~np.isnan(values)]
        if not values.size:
            return float("nan"), float("nan")
        low, high = np.quantile(values, [alpha, 1 - alpha])
        return float(low), float(high)

    chosen = thresholds(data, target_fprs)
    point = metric_table(data, chosen, leads, None)
    point_lookup = point.set_index(["detector", "target_fpr"])

    def estimate(name: str, target: float, metric: str) -> float:
        return float(point_lookup.loc[(name, target), metric])  # type: ignore[arg-type]

    rate_rows = []
    for (name, target, metric), draws in stats.items():
        low, high = interval(np.asarray(draws))
        rate_rows.append({"detector": name, "target_fpr": target, "metric": metric,
                          "estimate": estimate(name, target, metric),
                          "ci_low": low, "ci_high": high})  # fmt: skip
    diff_rows = []
    metrics = ["fpr", *[f"detection_lead{lead}" for lead in leads]]
    for a, b in pairs:
        for target in target_fprs:
            for metric in metrics:
                d = np.asarray(stats[(a, target, metric)]) - np.asarray(stats[(b, target, metric)])
                low, high = interval(d)
                est = estimate(a, target, metric) - estimate(b, target, metric)
                diff_rows.append({"detector": a, "baseline": b, "target_fpr": target,
                                  "metric": metric, "difference": est, "ci_low": low,
                                  "ci_high": high,
                                  "fraction_resamples_above_0": float(np.mean(d > 0))})  # fmt: skip
    return pd.DataFrame(rate_rows), pd.DataFrame(diff_rows)


def timestep_tables(
    data: EvaluationData,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Timestep-level metrics of the learned predictors on the evaluation part (all
    conditions, every decision step) with seed-cluster bootstrap intervals, and the
    reliability table."""
    e = data.episodes
    rows, reliability_rows = [], []
    alpha = (1 - confidence) / 2
    for part in ("evaluation", "calibration"):
        mask = (e["part"] == part).to_numpy()
        index = _seed_index(e, mask)
        keys = np.array(list(index))
        horizon = next(iter(data.scores.values())).shape[1]
        for name in data.learned:
            probs = data.scores[name]
            labels = np.repeat(e["failure"].to_numpy()[:, None], horizon, axis=1)
            point = timestep_metrics(labels[mask], probs[mask])
            if part == "evaluation":
                rng = np.random.default_rng(seed)
                draws: dict[str, list[float]] = {k: [] for k in ("auroc", "pr_auc", "brier")}
                fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
                    "auroc": auroc, "pr_auc": average_precision, "brier": brier,
                }  # fmt: skip
                for _ in range(resamples):
                    ei = np.concatenate([index[s] for s in rng.choice(keys, len(keys))])
                    y, p = labels[ei].ravel(), probs[ei].ravel()
                    for k, fn in fns.items():
                        draws[k].append(fn(y, p))
                for k, values in draws.items():
                    low, high = np.quantile(values, [alpha, 1 - alpha])
                    point[f"{k}_ci_low"], point[f"{k}_ci_high"] = float(low), float(high)
                table = reliability(labels[mask].ravel(), probs[mask].ravel())
                table.insert(0, "predictor", name)
                reliability_rows.append(table)
            rows.append({"predictor": name, "part": part, **point})
    return pd.DataFrame(rows), pd.concat(reliability_rows, ignore_index=True)


def timestep_by_remaining(data: EvaluationData, edges: tuple[int, ...] = (0, 10, 20, 30, 40, 50)):
    """AUROC and Brier per remaining-time band on the evaluation part."""
    e = data.episodes
    mask = (e["part"] == "evaluation").to_numpy()
    horizon = next(iter(data.scores.values())).shape[1]
    remaining = horizon - np.arange(horizon)
    labels = e["failure"].to_numpy()[mask]
    rows = []
    for name in data.learned:
        probs = data.scores[name][mask]
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            cols = (remaining > low) & (remaining <= high)
            y = np.repeat(labels[:, None], cols.sum(), axis=1).ravel()
            p = probs[:, cols].ravel()
            rows.append({"predictor": name, "steps_remaining": f"({low}, {high}]",
                         "auroc": auroc(y, p), "brier": brier(y, p),
                         "mean_predicted": float(p.mean()),
                         "failure_fraction": float(y.mean())})  # fmt: skip
    return pd.DataFrame(rows)
