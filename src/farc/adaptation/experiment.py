"""Phase E pilot: online actuator identification and adaptive compensation on FetchPush-v4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.compensation import (
    CompensationConfig,
    Fixed,
    LatchedEstimate,
    Parameters,
    ParameterSource,
    compensate,
)
from farc.adaptation.estimator import ActuatorEstimator, Estimate, EstimatorConfig
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.counterfactual.session import EpisodeSession
from farc.disturbances.config import SEED_RANGES, ConditionSpec, load_experiment
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve

ENV_ID = "FetchPush-v4"
# every validation-range seed block an earlier phase (or Phase E debugging) has looked at
USED_SEED_RANGES = {
    "phase1_and_phase_b_validation": (3_500_000, 3_500_100),
    "phase_c_controller_debugging": (3_600_000, 3_600_030),
    "phase_c_development": (3_610_000, 3_610_100),
    "phase_d_pilot": (3_620_000, 3_620_005),
    "phase_d_full_and_placebo": (3_621_000, 3_621_060),
    "phase_e_debugging": (3_630_000, 3_630_010),
}


class AdaptationConfigError(ValueError):
    pass


@dataclass(frozen=True)
class AdaptationExperiment:
    name: str
    conditions_from: str
    conditions: tuple[str, ...]
    checkpoints: tuple[str, ...]
    selection_seed_offset: int
    selection_episodes: int
    evaluation_seed_offset: int
    evaluation_episodes: int
    perturbation_seed: int
    estimator: dict[str, Any]
    gain_floors: tuple[float, ...]
    activate_below: float
    sham_gain: float
    sham_delay: int
    bootstrap_resamples: int
    bootstrap_seed: int
    output: str

    @property
    def selection_seeds(self) -> list[int]:
        o = self.selection_seed_offset
        return list(range(o, o + self.selection_episodes))

    @property
    def evaluation_seeds(self) -> list[int]:
        o = self.evaluation_seed_offset
        return list(range(o, o + self.evaluation_episodes))


def validate_seeds(seeds: list[int], label: str) -> None:
    low, high = SEED_RANGES["validation"]
    if not seeds or not (low <= min(seeds) and max(seeds) < high):
        raise AdaptationConfigError(f"{label} seeds must lie in the validation range")
    held_low, held_high = SEED_RANGES["heldout"]
    if any(held_low <= s < held_high for s in seeds):
        raise AdaptationConfigError("held-out seeds are reserved")
    for name, (a, b) in USED_SEED_RANGES.items():
        if any(a <= s < b for s in seeds):
            raise AdaptationConfigError(f"{label} seeds overlap {name} [{a}, {b})")


def validate_adaptation_experiment(config: AdaptationExperiment) -> AdaptationExperiment:
    validate_seeds(config.selection_seeds, "selection")
    validate_seeds(config.evaluation_seeds, "evaluation")
    if set(config.selection_seeds) & set(config.evaluation_seeds):
        raise AdaptationConfigError("selection and evaluation seeds overlap")
    return config


def load_adaptation_experiment(path: str | Path) -> AdaptationExperiment:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    for key in ("conditions", "checkpoints", "gain_floors"):
        values[key] = tuple(values[key])
    return validate_adaptation_experiment(AdaptationExperiment(**values))


def selected_conditions(config: AdaptationExperiment) -> list[ConditionSpec]:
    source = {c.name: c for c in load_experiment(resolve(config.conditions_from)).conditions}
    missing = [n for n in config.conditions if n not in source]
    if missing:
        raise AdaptationConfigError(f"unknown conditions {missing}")
    return [source[n] for n in config.conditions if source[n].applies_to(ENV_ID)]


# --- ground truth (evaluation only; reaches only the ORACLE configuration) -----------------


def true_parameters(condition: ConditionSpec, step: int) -> tuple[float, int]:
    gain, delay = 1.0, 0
    for d in condition.disturbances:
        if d.type == "action_scale" and d.active(step):
            gain *= 1.0 - d.severity
        if d.type == "action_delay" and d.active(step):
            delay = int(d.severity)
    return gain, delay


class Oracle:
    """PRIVILEGED: the injected disturbance's true parameters at the current step."""

    def __init__(self, condition: ConditionSpec):
        self.condition = condition

    def parameters(self, estimate: Estimate) -> Parameters:
        gain, delay = true_parameters(self.condition, estimate.step)
        return Parameters(gain if gain < 1.0 else None, delay or None)


# --- configurations ------------------------------------------------------------------------


def configurations(config: AdaptationExperiment) -> dict[str, tuple[CompensationConfig, str]]:
    """name -> (compensation config, parameter source kind)."""
    out: dict[str, tuple[CompensationConfig, str]] = {
        "base_only": (CompensationConfig(False, False), "none"),
    }
    for floor in config.gain_floors:
        out[f"estimated_gain_f{floor:g}"] = (
            CompensationConfig(True, False, floor, config.activate_below),
            "estimate",
        )
        out[f"estimated_combined_f{floor:g}"] = (
            CompensationConfig(True, True, floor, config.activate_below), "estimate")  # fmt: skip
    floor = min(config.gain_floors)
    out["estimated_delay"] = (CompensationConfig(False, True, floor, config.activate_below),
                              "estimate")  # fmt: skip
    out["oracle_combined"] = (CompensationConfig(True, True, floor, 1.0), "oracle")
    out["sham_gain"] = (CompensationConfig(True, False, floor, 1.0), "sham_gain")
    out["sham_delay"] = (CompensationConfig(False, True, floor, 1.0), "sham_delay")
    return out


def parameter_source(kind: str, config: AdaptationExperiment, condition) -> ParameterSource:
    if kind == "estimate":
        return LatchedEstimate()
    if kind == "oracle":
        return Oracle(condition)
    if kind == "sham_gain":
        return Fixed(config.sham_gain, None, gate_on_reliability=True)
    if kind == "sham_delay":
        return Fixed(None, config.sham_delay, gate_on_reliability=True)
    return Fixed(None, None)


def run_episode(session: EpisodeSession, estimator: ActuatorEstimator,
                compensation: CompensationConfig, source: ParameterSource,
                condition: ConditionSpec) -> dict[str, Any]:  # fmt: skip
    """One closed-loop episode. The estimator sees only the controller's view and the
    commands issued; ``condition`` is used after the fact for estimation-error metrics."""
    steps = []
    digest = hashlib.sha256()
    while not session.done:
        view = session.observe()
        estimator.observe(view.gripper)
        estimate = estimator.estimate()
        params = source.parameters(estimate)

        def policy_at(observation: np.ndarray, goal=view.goal) -> np.ndarray:
            normalized = session.normalizer(observation, goal)[None]
            return np.asarray(session.policy(normalized)[0], np.float64)

        assert view.observation is not None
        decision = compensate(compensation, params, view.base_action, policy_at,
                              view.observation, session.layout, list(session.commands),
                              estimator.config.nominal_gain)  # fmt: skip
        session.act(decision.command)
        estimator.issued(decision.command)
        digest.update(decision.command.tobytes())
        gain_true, delay_true = true_parameters(condition, max(estimate.step - 1, 0))
        steps.append({
            "t": estimate.step, "gain_est": estimate.gain, "delay_est": estimate.delay,
            "gain_reliable": estimate.gain_reliable, "delay_reliable": estimate.delay_reliable,
            "r2": estimate.r2, "gain_true": gain_true, "delay_true": delay_true,
            "gain_active": decision.gain_active, "delay_active": decision.delay_active,
            "saturated": decision.saturated,
            "base_saturated": bool(np.any(np.abs(view.base_action[:3]) >= 0.999)),
        })  # fmt: skip
    frame = pd.DataFrame(steps)
    return {"success": session.final_success, "final_distance": session.final_distance,
            "command_digest": digest.hexdigest(), "steps": frame}  # fmt: skip


def _estimation_summary(steps: pd.DataFrame) -> dict[str, float]:
    """Accuracy of the online estimates against the true values, at steps where the true
    values were constant over the whole estimation window (evaluation only)."""
    g = steps[steps["gain_reliable"]]
    dl = steps[steps["delay_reliable"]]
    first = steps.index[steps["gain_reliable"]]
    first_d = steps.index[steps["delay_reliable"]]
    return {
        "gain_abstention": 1.0 - steps["gain_reliable"].mean(),
        "delay_abstention": 1.0 - steps["delay_reliable"].mean(),
        "gain_abs_error": float((g["gain_est"] - g["gain_true"]).abs().mean()) if len(g)
        else np.nan,
        "gain_rel_error": float(((g["gain_est"] - g["gain_true"]) / g["gain_true"]).abs()
                                .mean()) if len(g) else np.nan,
        "delay_accuracy": float((dl["delay_est"] == dl["delay_true"]).mean()) if len(dl)
        else np.nan,
        "first_gain_reliable_step": float(steps.loc[first[0], "t"]) if len(first) else np.nan,
        "first_delay_reliable_step": float(steps.loc[first_d[0], "t"]) if len(first_d)
        else np.nan,
        "gain_activation_rate": steps["gain_active"].mean(),
        "delay_activation_rate": steps["delay_active"].mean(),
        "compensation_saturation_rate": steps["saturated"].mean(),
        "base_saturation_rate": steps["base_saturated"].mean(),
    }  # fmt: skip


def run_policy(args: tuple[AdaptationExperiment, str]) -> tuple[list[dict], list[dict]]:
    """All configurations, conditions and seeds for one frozen policy checkpoint."""
    config, checkpoint = args
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))  # verifies the manifest SHA-256
    actor_state = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    policy = deterministic_policy_function(loaded.actor)
    normalizer = FrozenNormalizer(loaded.normalizer)
    env = make_env(loaded.env_config)
    estimator_config = EstimatorConfig(**config.estimator)
    seeds = [("selection", s) for s in config.selection_seeds] + [
        ("evaluation", s) for s in config.evaluation_seeds]  # fmt: skip
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for condition in selected_conditions(config):
        severity = condition.disturbances[0].severity if condition.disturbances else 0.0
        kind = condition.disturbances[0].type if condition.disturbances else "none"
        for split, seed in seeds:
            for name, (compensation, source_kind) in configurations(config).items():
                session = EpisodeSession(env, policy, normalizer, condition, seed,
                                         config.perturbation_seed)  # fmt: skip
                source = parameter_source(source_kind, config, condition)
                result = run_episode(session, ActuatorEstimator(estimator_config), compensation,
                                     source, condition)  # fmt: skip
                key = {"configuration": name, "policy_seed": loaded.training_seed,
                       "condition": condition.name, "disturbance_type": kind,
                       "severity": severity, "split": split, "episode_seed": seed}  # fmt: skip
                episodes.append({**key, "success": result["success"],
                                 "final_distance": result["final_distance"],
                                 "command_digest": result["command_digest"],
                                 **_estimation_summary(result["steps"])})  # fmt: skip
                if name == "base_only":
                    steps.extend({**key, **row} for row in result["steps"].to_dict("records"))
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor_state[k]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes, steps


def check_replay(config: AdaptationExperiment, checkpoint: str, condition_names: list[str],
                 seed: int) -> None:  # fmt: skip
    """Running the same episode twice (any configuration) must give identical commands."""
    loaded = load_policy(resolve(checkpoint))
    policy = deterministic_policy_function(loaded.actor)
    normalizer = FrozenNormalizer(loaded.normalizer)
    env = make_env(loaded.env_config)
    conditions = {c.name: c for c in selected_conditions(config)}
    for name in condition_names:
        for conf_name, (compensation, kind) in configurations(config).items():
            digests = set()
            for _ in range(2):
                session = EpisodeSession(env, policy, normalizer, conditions[name], seed,
                                         config.perturbation_seed)  # fmt: skip
                r = run_episode(session, ActuatorEstimator(EstimatorConfig(**config.estimator)),
                                compensation, parameter_source(kind, config, conditions[name]),
                                conditions[name])  # fmt: skip
                digests.add((r["command_digest"], r["final_distance"]))
            if len(digests) != 1:
                raise RuntimeError(f"replay differs for {name} / {conf_name}")


# --- analysis -------------------------------------------------------------------------------

KEYS = ["policy_seed", "condition", "episode_seed"]
ACTUATOR_TYPES = ("action_scale", "action_delay")


def paired(episodes: pd.DataFrame) -> pd.DataFrame:
    base = episodes[episodes["configuration"] == "base_only"].set_index(KEYS)
    out = episodes.join(base["success"].rename("base_success"), on=KEYS)
    if out["base_success"].isna().any():
        raise ValueError("every episode needs a base-only counterpart")
    out["base_success"] = out["base_success"].astype(bool)
    out["success"] = out["success"].astype(bool)
    return out


def _cluster_ci(frame: pd.DataFrame, resamples: int, seed: int) -> tuple[float, float, float]:
    diff = frame["success"].astype(float) - frame["base_success"].astype(float)
    groups = diff.groupby(frame["episode_seed"])
    sums, counts = groups.sum().to_numpy(), groups.count().to_numpy()
    estimate = sums.sum() / counts.sum()
    picks = np.random.default_rng(seed).integers(0, len(sums), size=(resamples, len(sums)))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(estimate), float(low), float(high)


def performance_table(frame: pd.DataFrame, by: list[str], resamples: int, seed: int):
    rows = []
    for key, sub in frame.groupby(["configuration", *by]):
        key = key if isinstance(key, tuple) else (key,)
        a, b = sub["base_success"].to_numpy(bool), sub["success"].to_numpy(bool)
        est, low, high = _cluster_ci(sub, resamples, seed)
        rows.append({**dict(zip(["configuration", *by], key, strict=True)),
                     "episodes": len(sub), "success_rate": b.mean(),
                     "base_success_rate": a.mean(), "success_difference": est,
                     "ci_low": low, "ci_high": high,
                     "rescued": int((~a & b).sum()), "base_failures": int((~a).sum()),
                     "harmed": int((a & ~b).sum()), "base_successes": int(a.sum()),
                     "gain_activation_rate": sub["gain_activation_rate"].mean(),
                     "delay_activation_rate": sub["delay_activation_rate"].mean(),
                     "compensation_saturation_rate": sub["compensation_saturation_rate"].mean(),
                     "base_saturation_rate": sub["base_saturation_rate"].mean()})  # fmt: skip
    return pd.DataFrame(rows)


def estimator_table(steps: pd.DataFrame) -> pd.DataFrame:
    """Per condition and phase of the episode, from the base-only runs (the estimator observes
    the unmodified base policy there)."""
    steps = steps.assign(phase=pd.cut(steps["t"], [-1, 4, 14, 29, 49],
                                      labels=["t0-4", "t5-14", "t15-29", "t30-49"]))  # fmt: skip
    rows = []
    for (cond, phase), sub in steps.groupby(["condition", "phase"], observed=True):
        g, d = sub[sub["gain_reliable"]], sub[sub["delay_reliable"]]
        rows.append({"condition": cond, "phase": phase, "steps": len(sub),
                     "gain_true_median": sub["gain_true"].median(),
                     "gain_reliable_rate": sub["gain_reliable"].mean(),
                     "gain_est_median": g["gain_est"].median() if len(g) else np.nan,
                     "gain_abs_error_median": (g["gain_est"] - g["gain_true"]).abs().median()
                     if len(g) else np.nan,
                     "delay_true": int(sub["delay_true"].max()),
                     "delay_reliable_rate": sub["delay_reliable"].mean(),
                     "delay_accuracy_when_reliable": (d["delay_est"] == d["delay_true"]).mean()
                     if len(d) else np.nan,
                     "delay_accuracy_unconditional": (sub["delay_est"] == sub["delay_true"])
                     .mean(),
                     "r2_median": sub["r2"].median()})  # fmt: skip
    return pd.DataFrame(rows)


DERIVED_TABLES = (
    "selection_actuator_pooled",
    "evaluation_by_condition",
    "evaluation_actuator_pooled",
    "evaluation_by_policy_seed",
    "estimator_accuracy",
)


def derive_tables(episodes: pd.DataFrame, steps: pd.DataFrame, config: AdaptationExperiment):
    data = paired(episodes)
    n, s = config.bootstrap_resamples, config.bootstrap_seed
    sel = data[data["split"] == "selection"]
    ev = data[data["split"] == "evaluation"]
    actuator = sel["disturbance_type"].isin(ACTUATOR_TYPES)
    selection = performance_table(sel[actuator], [], n, s)
    gain_rows = selection[selection["configuration"].str.startswith("estimated_gain_f")]
    # the floor with the best selection-seed difference; ties go to the higher (more
    # conservative) floor
    best = gain_rows.sort_values(["success_difference", "configuration"],
                                 ascending=[False, False]).iloc[0]  # fmt: skip
    floor = best["configuration"].removeprefix("estimated_gain_f")
    ev_act = ev[ev["disturbance_type"].isin(ACTUATOR_TYPES)]
    tables = {
        "selection_actuator_pooled": selection,
        "evaluation_by_condition": performance_table(ev, ["condition"], n, s),
        "evaluation_actuator_pooled": performance_table(ev_act, [], n, s),
        "evaluation_by_policy_seed": performance_table(ev_act, ["policy_seed"], n, s),
        "estimator_accuracy": estimator_table(steps[steps["split"] == "evaluation"]),
    }
    primary = ["base_only", f"estimated_gain_f{floor}", "estimated_delay",
               f"estimated_combined_f{floor}"]  # fmt: skip
    summary = {"selected_gain_floor": float(floor),
               "primary_configurations": primary,
               "oracle_configurations": ["oracle_combined"],
               "sham_configurations": ["sham_gain", "sham_delay"],
               "episodes": len(episodes), "step_rows": len(steps)}  # fmt: skip
    return tables, summary
