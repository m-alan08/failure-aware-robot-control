"""Phase C closed-loop recovery experiment: runner and paired analysis."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.detection.phase_aware import PhaseDetectorConfig
from farc.disturbances.config import SEED_RANGES, ConditionSpec, load_experiment
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.prediction.dataset import Normalizer
from farc.prediction.evaluation import load_predictor
from farc.recovery.controller import (
    InterventionSwitch,
    LearnedTrigger,
    PhaseAwareTrigger,
    RecoveryConfig,
    RecoveryController,
    SwitchConfig,
    Trigger,
)
from farc.recovery.rollout import CONFIGURATIONS, ClosedLoopEpisode, run_closed_loop_episode

ENV_ID = "FetchPush-v4"
# seeds that informed earlier phases or controller design; the experiment must avoid them
USED_SEED_RANGES = {
    "phase1_and_phase_b_validation": (3_500_000, 3_500_100),
    "recovery_controller_debugging": (3_600_000, 3_600_030),
}


class RecoveryConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RecoveryExperiment:
    name: str
    conditions_from: str
    checkpoints: tuple[str, ...]
    episode_seed_offset: int
    episodes: int
    perturbation_seed: int
    predictor: str
    learned_threshold: float
    reactive_threshold: float
    phase_detector: dict[str, Any]
    switch: dict[str, Any]
    recovery: dict[str, Any]
    bootstrap_resamples: int
    bootstrap_seed: int
    output: str

    @property
    def episode_seeds(self) -> list[int]:
        return list(range(self.episode_seed_offset, self.episode_seed_offset + self.episodes))


def load_recovery_experiment(path: str | Path) -> RecoveryExperiment:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    values["checkpoints"] = tuple(values["checkpoints"])
    return validate_recovery_experiment(RecoveryExperiment(**values))


def validate_recovery_experiment(config: RecoveryExperiment) -> RecoveryExperiment:
    """Seed-range checks; call again after any override of the episode seeds."""
    seeds = config.episode_seeds
    low, high = SEED_RANGES["validation"]
    if not (low <= seeds[0] and seeds[-1] < high):
        raise RecoveryConfigError(f"episode seeds must lie in the validation range [{low}, {high})")
    held_low, held_high = SEED_RANGES["heldout"]
    if any(held_low <= s < held_high for s in seeds):
        raise RecoveryConfigError("held-out seeds are reserved")
    for name, (a, b) in USED_SEED_RANGES.items():
        if seeds[0] < b and a <= seeds[-1]:
            raise RecoveryConfigError(f"episode seeds overlap {name} [{a}, {b})")
    return config


def push_conditions(config: RecoveryExperiment) -> list[ConditionSpec]:
    source = load_experiment(resolve(config.conditions_from))
    return [c for c in source.conditions if c.applies_to(ENV_ID)]


def build_trigger(kind: str, config: RecoveryExperiment, normalizer: Normalizer) -> Trigger | None:
    if kind == "learned_triggered":
        model = load_predictor(resolve(config.predictor))
        return LearnedTrigger(model, normalizer, config.learned_threshold)
    if kind == "reactive_triggered":
        detector = PhaseDetectorConfig(**{**config.phase_detector, "departure_persistence": None})
        return PhaseAwareTrigger(detector, config.reactive_threshold)
    return None


def build_switch(kind: str, config: RecoveryExperiment) -> InterventionSwitch | None:
    if kind == "base_only":
        return None
    return InterventionSwitch(SwitchConfig(**config.switch), always_on=kind == "always_on")


def episode_row(e: ClosedLoopEpisode, kind: str, condition: ConditionSpec, policy_seed: int):
    interventions = e.interventions
    starts = [int(i["start"]) for i in interventions]
    return {
        "configuration": kind,
        "policy_seed": policy_seed,
        "condition": condition.name,
        "disturbance_type": condition.disturbances[0].type if condition.disturbances else "none",
        "episode_seed": e.episode_seed,
        "final_success": e.final_success,
        "final_distance": e.final_distance,
        "activated": bool(e.recovery_active.any()),
        "interventions": len(interventions),
        "recovery_steps": int(e.recovery_active.sum()),
        "first_activation_step": starts[0] if starts else np.nan,
        "first_trigger_step": e.first_trigger_step if e.first_trigger_step is not None else np.nan,
        "exits_success": sum(i["reason"] == "success" for i in interventions),
        "exits_timeout": sum(i["reason"] == "timeout" for i in interventions),
        "exits_episode_end": sum(i["reason"] == "episode_end" for i in interventions),
        "intervention_log": json.dumps(interventions),
    }


def run_policy(args: tuple[RecoveryExperiment, str, Normalizer]) -> list[dict[str, Any]]:
    """All configurations and conditions for one base-policy checkpoint."""
    config, checkpoint, normalizer = args
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))  # verifies the manifest SHA-256
    actor_state = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    policy = deterministic_policy_function(loaded.actor)
    frozen = FrozenNormalizer(loaded.normalizer)
    env = make_env(loaded.env_config)
    recovery = RecoveryController(RecoveryConfig(**config.recovery))
    rows = []
    for kind in CONFIGURATIONS:
        trigger = build_trigger(kind, config, normalizer)
        for condition in push_conditions(config):
            for seed in config.episode_seeds:
                e = run_closed_loop_episode(
                    env, policy, frozen, condition, seed, config.perturbation_seed,
                    build_switch(kind, config), trigger, recovery,
                )  # fmt: skip
                rows.append(episode_row(e, kind, condition, loaded.training_seed))
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor_state[k]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return rows


# --- analysis -----------------------------------------------------------------------------

KEYS = ["policy_seed", "condition", "episode_seed"]


def paired(episodes: pd.DataFrame) -> pd.DataFrame:
    """One row per (policy, condition, episode seed, configuration) with the base-only
    outcome of the same episode attached."""
    base = episodes[episodes["configuration"] == "base_only"].set_index(KEYS)["final_success"]
    out = episodes.join(base.rename("base_success"), on=KEYS)
    if out["base_success"].isna().any():
        raise ValueError("some episodes have no base-only counterpart")
    counts = out.groupby(KEYS)["configuration"].nunique()
    if (counts != len(CONFIGURATIONS)).any():
        raise ValueError("every episode must be run under every configuration")
    out["base_success"] = out["base_success"].astype(bool)
    return out


def summary(frame: pd.DataFrame) -> dict[str, float]:
    s, b, act = (frame[c].to_numpy(bool) for c in ("final_success", "base_success", "activated"))
    active = frame[act]

    def if_active(column: str, how: str) -> float:
        return float(getattr(active[column], how)()) if len(active) else float("nan")

    return {
        "episodes": len(frame),
        "success_rate": s.mean(),
        "base_success_rate": b.mean(),
        "success_difference": s.mean() - b.mean(),
        "activation_rate": act.mean(),
        "recovered": int((~b & s).sum()),  # base fails, this configuration succeeds
        "harmed": int((b & ~s).sum()),  # base succeeds, this configuration fails
        "base_failures": int((~b).sum()),
        "recovered_fraction_of_base_failures": (~b & s).sum() / max((~b).sum(), 1),
        "activated_on_base_success": int((act & b).sum()),
        "unnecessary_intervention_rate": (act & b).sum() / max(b.sum(), 1),
        "activated_on_base_failure": int((act & ~b).sum()),
        "mean_interventions_if_activated": if_active("interventions", "mean"),
        "mean_recovery_steps_if_activated": if_active("recovery_steps", "mean"),
        "median_first_activation_step": if_active("first_activation_step", "median"),
        "exits_success": int(frame["exits_success"].sum()),
        "exits_timeout": int(frame["exits_timeout"].sum()),
        "exits_episode_end": int(frame["exits_episode_end"].sum()),
    }  # fmt: skip


def cluster_bootstrap_difference(
    frame: pd.DataFrame, a: str, b: str, resamples: int, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Paired success difference a - b with episode seeds as clusters: a resampled seed brings
    every (policy, condition) episode of that seed, under both configurations."""
    wide = frame.pivot_table(index=KEYS, columns="configuration", values="final_success",
                             aggfunc="first").astype(float)  # fmt: skip
    diff = (wide[a] - wide[b]).groupby(level="episode_seed")
    sums, counts = diff.sum().to_numpy(), diff.count().to_numpy()
    estimate = sums.sum() / counts.sum()
    picks = rng.integers(0, len(sums), size=(resamples, len(sums)))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(estimate), float(low), float(high)


COMPARISONS = [
    ("always_on", "base_only"),
    ("reactive_triggered", "base_only"),
    ("learned_triggered", "base_only"),
    ("learned_triggered", "reactive_triggered"),
    ("learned_triggered", "always_on"),
]


def analyze(episodes: pd.DataFrame, resamples: int, seed: int) -> dict[str, pd.DataFrame]:
    data = paired(episodes)
    tables: dict[str, pd.DataFrame] = {}
    for name, by in (("pooled", []), ("by_condition", ["condition"]),
                     ("by_policy_seed", ["policy_seed"]),
                     ("by_disturbance_type", ["disturbance_type"])):  # fmt: skip
        groups = [((), data)] if not by else list(data.groupby(by))
        rows, diffs = [], []
        for key, frame in groups:
            label = dict(zip(by, key if isinstance(key, tuple) else (key,), strict=True))
            for kind, sub in frame.groupby("configuration"):
                rows.append({**label, "configuration": kind, **summary(sub)})
            rng = np.random.default_rng(seed)
            for a, b in COMPARISONS:
                est, low, high = cluster_bootstrap_difference(frame, a, b, resamples, rng)
                diffs.append({**label, "configuration": a, "baseline": b,
                              "success_difference": est,
                              "ci_low": low, "ci_high": high})  # fmt: skip
        tables[f"summary_{name}"] = pd.DataFrame(rows)
        tables[f"differences_{name}"] = pd.DataFrame(diffs)
    # oracle selection (privileged, post hoc): always-on on episodes the base fails, base
    # policy otherwise. It bounds only the per-episode choice between base-only and always-on;
    # a trigger that intervenes later or for part of an episode can do better on some episodes.
    wide = data.pivot_table(index=KEYS, columns="configuration", values="final_success",
                            aggfunc="first").astype(bool)  # fmt: skip
    oracle = np.where(wide["base_only"], True, wide["always_on"])
    tables["oracle"] = pd.DataFrame([{
        "configuration": "oracle_always_on_on_base_failures",
        "success_rate": float(oracle.mean()),
        "base_success_rate": float(wide["base_only"].mean()),
    }])  # fmt: skip
    return tables


def rollout_provenance(out: Path, analyze_only: bool, fresh: dict[str, Any]) -> dict[str, Any]:
    """Provenance of the run that produced ``episodes.csv``. A fresh run returns ``fresh``; an
    ``--analyze-only`` run keeps what the previous metadata recorded instead of overwriting it."""
    if not analyze_only:
        return fresh
    path = out / "metadata.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    if "rollout" in previous:
        return dict(previous["rollout"])
    legacy = {k: previous[k] for k in ("seconds", "episodes_rows") if k in previous}
    return {"unknown": "no rollout provenance was recorded", "previous_metadata": legacy}


def write_metadata(config: RecoveryExperiment, out: Path, extra: dict[str, Any]) -> None:
    (out / "metadata.json").write_text(
        json.dumps({"config": asdict(config), **extra}, indent=2, default=str) + "\n"
    )
