"""Phase F1 evaluation (pre-registered in ``docs/PHASE_F1_PREREGISTRATION.md``)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.delay_eval import holm, paired_difference
from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.adaptation.experiment import AdaptationConfigError
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.data import EPISODE, build_conditions, check_seeds, contact_flags, horizon_pairs
from farc.contact.dynamics import S, load_dynamics, rollout_batch, static_object_batch
from farc.contact.episodes import Predictor, learned_predictor, phase_e_predictor, run_episode
from farc.counterfactual.session import EpisodeSession
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve

CONFIGURATIONS = ("base_only", "phase_e_predictor", "contact_learned", "learned_gripper_only",
                  "oracle_sim_predictor")  # fmt: skip
LOGGED = ("base_only", "phase_e_predictor", "contact_learned")


@dataclass(frozen=True)
class ContactEvaluation:
    name: str
    plan: str
    plan_sha256: str
    model: str
    model_sha256: str
    conditions_from: str
    conditions: tuple[dict[str, Any], ...]
    delay_conditions: tuple[str, ...]
    harm_conditions: tuple[str, ...]
    checkpoints: tuple[str, ...]
    debug_seed_offset: int
    debug_episodes: int
    evaluation_seed_offset: int
    evaluation_episodes: int
    perturbation_seed: int
    estimator: dict[str, Any]
    noninferiority_margin: float
    harm_ci_floor: float
    harm_max_fraction: float
    bootstrap_resamples: int
    bootstrap_seed: int
    output: str
    debug_run: bool = field(default=False, compare=False)

    @property
    def seeds(self) -> list[int]:
        if self.debug_run:
            return list(range(self.debug_seed_offset, self.debug_seed_offset + self.debug_episodes))
        o = self.evaluation_seed_offset
        return list(range(o, o + self.evaluation_episodes))


def load_contact_evaluation(path: str | Path, check_plan: bool = True) -> ContactEvaluation:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    for key in ("conditions", "delay_conditions", "harm_conditions", "checkpoints"):
        values[key] = tuple(values[key])
    config = ContactEvaluation(**values)
    debug = list(range(config.debug_seed_offset, config.debug_seed_offset + config.debug_episodes))
    check_seeds(debug, "debugging")
    check_seeds(ContactEvaluation(**{**values, "debug_run": False}).seeds, "evaluation")
    names = {c["name"] for c in config.conditions}
    if (set(config.delay_conditions) | set(config.harm_conditions)) - names:
        raise AdaptationConfigError("unknown condition in delay or harm groups")
    plan_hash = hashlib.sha256(Path(config.plan).read_bytes()).hexdigest()
    if check_plan and plan_hash != config.plan_sha256:
        raise AdaptationConfigError("the pre-registration changed; record an amendment")
    return config


def predictors(config: ContactEvaluation) -> dict[str, Predictor | None]:
    model = load_dynamics(resolve(config.model), config.model_sha256)
    k0 = EstimatorConfig(**config.estimator).nominal_gain
    return {"base_only": None, "phase_e_predictor": phase_e_predictor(k0),
            "contact_learned": learned_predictor(model, move_object=True),
            "learned_gripper_only": learned_predictor(model, move_object=False),
            "oracle_sim_predictor": None}  # fmt: skip


def run_policy(args: tuple[ContactEvaluation, str]) -> tuple[list[dict], list[dict]]:
    config, checkpoint = args
    return run_configurations(config, checkpoint, CONFIGURATIONS, LOGGED, predictors(config))


def run_configurations(
    config: ContactEvaluation,
    checkpoint: str,
    configurations: tuple[str, ...],
    logged: tuple[str, ...],
    preds: dict[str, Predictor | None],
) -> tuple[list[dict], list[dict]]:
    """Every configuration on every condition and seed for one frozen policy. The name
    ``oracle_sim_predictor`` selects the privileged simulator oracle."""
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    actor_state = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    policy = deterministic_policy_function(loaded.actor)
    normalizer = FrozenNormalizer(loaded.normalizer)
    env = make_env(loaded.env_config)
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for condition in build_conditions(config.conditions_from, config.conditions):
        for seed in config.seeds:
            for name in configurations:
                session = EpisodeSession(env, policy, normalizer, condition, seed,
                                         config.perturbation_seed)  # fmt: skip
                estimator = ActuatorEstimator(EstimatorConfig(**config.estimator))
                result = run_episode(
                    session,
                    estimator,
                    preds[name],
                    condition,
                    use_oracle=name == "oracle_sim_predictor",
                    log=True,
                )
                key = {"configuration": name, "policy_seed": loaded.training_seed,
                       "condition": condition.name, "episode_seed": seed}  # fmt: skip
                rows = pd.DataFrame(result["rows"])
                near = np.linalg.norm(rows[["true6", "true7"]].to_numpy(), axis=1) < 0.05
                episodes.append({**key, "success": result["success"],
                                 "final_distance": result["final_distance"],
                                 "active_rate": rows["active"].mean(),
                                 "near_object_rate": float(near.mean())})  # fmt: skip
                if name in logged:
                    steps.extend({**key, **r} for r in result["rows"])
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor_state[k]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes, steps


# --- analysis -------------------------------------------------------------------------------

DERIVED_TABLES = ("hypotheses", "by_condition", "prediction_accuracy", "closed_loop_prediction",
                  "contact_split", "harm_cases")  # fmt: skip


def _seed_bootstrap(values: pd.Series, resamples: int, seed: int):
    """Mean of per-seed values with a seed bootstrap (values indexed by episode seed)."""
    v = values.to_numpy(float)
    picks = np.random.default_rng(seed).integers(0, len(v), size=(resamples, len(v)))
    draws = v[picks].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(v.mean()), float(low), float(high)


def offline_accuracy(steps: pd.DataFrame, config: ContactEvaluation) -> pd.DataFrame:
    """Per-step errors at the true horizon from base-only trajectories (view as input, true
    executed queue, true target): static (Phase E), learned, learned gripper-only."""
    model = load_dynamics(resolve(config.model), config.model_sha256)
    k0 = EstimatorConfig(**config.estimator).nominal_gain
    base = steps[(steps["configuration"] == "base_only")
                 & steps["condition"].isin(config.delay_conditions)]  # fmt: skip
    out = []
    for h in sorted(int(v) for v in base["delay_true"].unique()):
        if h < 1:
            continue
        sub = base[base["delay_true"] == h]
        start, cmds, target, true_start, index = horizon_pairs(sub, h, start="view")
        if not len(start):
            continue
        preds = {
            "static": static_object_batch(start, cmds, k0),
            "learned": rollout_batch(model, start, cmds),
            "learned_gripper_only": rollout_batch(model, start, cmds, move_object=False),
        }
        contact = contact_flags(true_start, target)
        frame = sub.loc[index, EPISODE + ["t"]].reset_index(drop=True)
        frame["horizon"] = int(h)
        frame["contact"] = contact
        for name, pred in preds.items():
            for key in ("gripper_position", "object_position"):
                frame[f"{name}_{key}_error"] = np.linalg.norm(
                    pred[:, S[key]] - target[:, S[key]], axis=1)  # fmt: skip
        out.append(frame)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def closed_loop_prediction(steps: pd.DataFrame) -> pd.DataFrame:
    """Error of the prediction actually used at t versus the true state at t + d_hat, inside
    each compensated configuration's own closed loop."""
    rows = []
    for name in ("phase_e_predictor", "contact_learned"):
        sub = steps[steps["configuration"] == name].sort_values(EPISODE + ["t"])
        for (_, _, cond, _), ep in sub.groupby(EPISODE, sort=False):
            ep = ep.reset_index(drop=True)
            true_g = ep[["true0", "true1", "true2"]].to_numpy()
            true_o = ep[["true3", "true4", "true5"]].to_numpy()
            near = np.linalg.norm(ep[["true6", "true7"]].to_numpy(), axis=1) < 0.05
            delays = ep["delay_param"].to_numpy(int)
            pred_g = ep[[f"pred_gripper_position{i}" for i in range(3)]].to_numpy(float)
            pred_o = ep[[f"pred_object_position{i}" for i in range(3)]].to_numpy(float)
            for t in np.flatnonzero(ep["active"].to_numpy(bool)):
                d = int(delays[t])
                if t + d >= len(ep):
                    continue
                pg, po = pred_g[t], pred_o[t]
                moved = np.linalg.norm(true_o[t + d] - true_o[t]) > 0.002
                rows.append({"configuration": name, "condition": cond,
                             "contact": bool(near[t] or moved),
                             "gripper_error": float(np.linalg.norm(pg - true_g[t + d])),
                             "object_error":
                                 float(np.linalg.norm(po - true_o[t + d]))})  # fmt: skip
    frame = pd.DataFrame(rows)
    return (frame.groupby(["configuration", "condition", "contact"])
            .agg(steps=("object_error", "size"), gripper_error_mean=("gripper_error", "mean"),
                 object_error_mean=("object_error", "mean"),
                 object_error_p90=("object_error", lambda x: x.quantile(0.9)))
            .reset_index())  # fmt: skip


def derive_tables(episodes: pd.DataFrame, steps: pd.DataFrame, config: ContactEvaluation):
    n, s = config.bootstrap_resamples, config.bootstrap_seed
    delay = episodes[episodes["condition"].isin(config.delay_conditions)]

    def diff(frame, a, b):
        return dict(zip(["estimate", "ci_low", "ci_high", "p", "episodes"],
                        paired_difference(frame, a, b, n, s), strict=True))  # fmt: skip

    accuracy = offline_accuracy(steps, config)
    contact_acc = accuracy[accuracy["contact"]]
    gap = contact_acc["static_object_position_error"] - contact_acc["learned_object_position_error"]
    per_seed = gap.groupby(contact_acc["episode_seed"]).mean()
    hp = _seed_bootstrap(per_seed, n, s)
    hc = diff(delay, "contact_learned", "phase_e_predictor")
    if hc["ci_low"] > 0:
        verdict = "improves"
    elif hc["ci_high"] < 0:
        verdict = "worse"
    elif hc["ci_low"] > -config.noninferiority_margin:
        verdict = "non-inferior"
    else:
        verdict = "inconclusive"
    rows = [
        {"test": "HP_contact_object_error_static_minus_learned_m", "estimate": hp[0],
         "ci_low": hp[1], "ci_high": hp[2], "criterion": "ci_low > 0", "passed": hp[1] > 0},
        {"test": "HC_primary_contact_learned_minus_phase_e", **hc,
         "criterion": "improves if ci_low > 0", "passed": hc["ci_low"] > 0,
         "verdict": verdict},
        {"test": "HA_contact_learned_minus_gripper_only",
         **diff(delay, "contact_learned", "learned_gripper_only"), "criterion": "secondary"},
        {"test": "HO_oracle_minus_phase_e",
         **diff(delay, "oracle_sim_predictor", "phase_e_predictor"), "criterion": "secondary"},
    ]  # fmt: skip
    for cond in config.harm_conditions:
        sub = episodes[episodes["condition"] == cond]
        r = diff(sub, "contact_learned", "base_only")
        w = sub.pivot_table(index=["policy_seed", "episode_seed"], columns="configuration",
                            values="success", aggfunc="first").astype(bool)  # fmt: skip
        harmed = float(
            (w["base_only"] & ~w["contact_learned"]).sum() / max(w["base_only"].sum(), 1)
        )
        rows.append({"test": f"harm_{cond}", **r, "harmed_fraction": harmed,
                     "criterion": f"ci_low > {config.harm_ci_floor} and harmed <= "
                                  f"{config.harm_max_fraction}",
                     "passed": r["ci_low"] > config.harm_ci_floor
                     and harmed <= config.harm_max_fraction})  # fmt: skip
    hypotheses = pd.DataFrame(rows)

    by_condition = []
    for cond_key, sub in episodes.groupby("condition"):
        w = sub.pivot_table(index=["policy_seed", "episode_seed"], columns="configuration",
                            values="success", aggfunc="first").astype(bool)  # fmt: skip
        for name in CONFIGURATIONS[1:]:
            for ref in ("base_only", "phase_e_predictor"):
                if name == ref:
                    continue
                r = diff(sub, name, ref)
                a, b = w[ref], w[name]
                by_condition.append({"condition": cond_key, "configuration": name,
                                     "reference": ref, **r, "success_rate": b.mean(),
                                     "reference_success_rate": a.mean(),
                                     "rescued": int((~a & b).sum()),
                                     "harmed": int((a & ~b).sum()),
                                     "active_rate": sub[sub["configuration"] == name]
                                     ["active_rate"].mean()})  # fmt: skip
    table = pd.DataFrame(by_condition)
    mask = ((table["configuration"] == "contact_learned")
            & (table["reference"] == "phase_e_predictor")
            & table["condition"].isin(config.delay_conditions))  # fmt: skip
    table["p_holm_primary"] = np.nan
    table.loc[mask, "p_holm_primary"] = holm(table.loc[mask, "p"].tolist())

    acc_rows = []
    for (acc_cond, h, contact), sub in accuracy.groupby(["condition", "horizon", "contact"]):
        row = {"condition": acc_cond, "horizon": h, "contact": contact, "steps": len(sub)}
        for col in sub.columns:
            if col.endswith("_error"):
                row[f"{col}_mean"] = sub[col].mean()
        acc_rows.append(row)
    for contact, sub in accuracy.groupby("contact"):
        row = {"condition": "ALL_delay", "horizon": -1, "contact": contact, "steps": len(sub)}
        for col in sub.columns:
            if col.endswith("_error"):
                row[f"{col}_mean"] = sub[col].mean()
        acc_rows.append(row)

    # exploratory: contact-heavy vs light episodes (median split of base-only contact share)
    base_contact = (accuracy.groupby(["policy_seed", "condition", "episode_seed"])["contact"]
                    .mean().rename("contact_share"))  # fmt: skip
    joined = delay.join(base_contact, on=["policy_seed", "condition", "episode_seed"])
    median = float(base_contact.median())
    split = []
    for label, sub in (
        ("contact_heavy", joined[joined["contact_share"] > median]),
        ("contact_light", joined[joined["contact_share"] <= median]),
    ):
        for name in ("contact_learned", "oracle_sim_predictor", "learned_gripper_only"):
            split.append({"group": label, "median_contact_share": median,
                          "configuration": name, "reference": "phase_e_predictor",
                          **diff(sub, name, "phase_e_predictor")})  # fmt: skip

    w = delay.pivot_table(
        index=["policy_seed", "condition", "episode_seed"],
        columns="configuration",
        values="success",
        aggfunc="first",
    ).astype(bool)
    harmed_frame = w[w["phase_e_predictor"] & ~w["contact_learned"]].reset_index()
    harmed_frame = harmed_frame.join(base_contact, on=["policy_seed", "condition", "episode_seed"])
    harm_cases = harmed_frame[["policy_seed", "condition", "episode_seed", "contact_share"]]

    tables = {"hypotheses": hypotheses, "by_condition": table,
              "prediction_accuracy": pd.DataFrame(acc_rows),
              "closed_loop_prediction": closed_loop_prediction(steps),
              "contact_split": pd.DataFrame(split), "harm_cases": harm_cases}  # fmt: skip
    summary = {"verdict_primary": verdict, "plan_sha256": config.plan_sha256,
               "model_sha256": config.model_sha256, "episodes": len(episodes),
               "step_rows": len(steps)}  # fmt: skip
    return tables, summary
