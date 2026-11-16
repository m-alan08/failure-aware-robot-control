"""One-shot held-out confirmation of Phases E and F1 (``docs/HELDOUT_CONFIRMATION_PLAN.md``).

This is the only code path that accepts held-out seeds, and it accepts only the approved slice
4,000,000-4,000,299. Every other loader keeps refusing the whole held-out range.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from farc.adaptation.delay_eval import paired_difference
from farc.adaptation.estimator import EstimatorConfig
from farc.adaptation.experiment import AdaptationConfigError
from farc.contact.data import check_seeds
from farc.contact.dynamics import load_dynamics
from farc.contact.episodes import (
    Predictor,
    learned_gripper_consistent,
    learned_predictor,
    phase_e_predictor,
)
from farc.contact.evaluation import (
    ContactEvaluation,
    closed_loop_prediction,
    offline_accuracy,
    run_configurations,
)
from farc.paths import resolve

APPROVED_SLICE = (4_000_000, 4_000_300)
CONFIGURATIONS = ("base_only", "phase_e_predictor", "contact_learned",
                  "learned_gripper_consistent", "oracle_sim_predictor")  # fmt: skip
LOGGED = ("base_only", "phase_e_predictor", "contact_learned")
# fixed-sequence gatekeeping family (plan order)
HYPOTHESES = (
    ("H1", "phase_e_predictor", "base_only"),
    ("H2", "contact_learned", "phase_e_predictor"),
    ("H3", "contact_learned", "learned_gripper_consistent"),
)
EXTRA_KEYS = ("heldout_approved", "h1_min_effect", "development_estimates")


@dataclass(frozen=True)
class Heldout:
    contact: ContactEvaluation  # shared fields (conditions, model, estimator, statistics)
    heldout_approved: str
    h1_min_effect: float
    development_estimates: dict[str, float]


def load_heldout(path: str | Path, debug: bool = False) -> Heldout:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    extras = {k: values.pop(k) for k in EXTRA_KEYS}
    for key in ("conditions", "delay_conditions", "harm_conditions", "checkpoints"):
        values[key] = tuple(values[key])
    contact = ContactEvaluation(**values, debug_run=debug)
    digest = hashlib.sha256(Path(contact.plan).read_bytes()).hexdigest()
    if digest != contact.plan_sha256:
        raise AdaptationConfigError("the held-out plan changed since the config pinned it")
    if not str(extras["heldout_approved"]).strip():
        raise AdaptationConfigError("the held-out run needs an explicit recorded approval")
    if set(extras["development_estimates"]) != {h for h, _, _ in HYPOTHESES}:
        raise AdaptationConfigError("development estimates must cover H1-H3")
    check_seeds(replace(contact, debug_run=True).seeds, "heldout_debugging")
    check_heldout_slice(replace(contact, debug_run=False).seeds)
    return Heldout(contact, str(extras["heldout_approved"]), float(extras["h1_min_effect"]),
                   {k: float(v) for k, v in extras["development_estimates"].items()})  # fmt: skip


def check_heldout_slice(seeds: list[int]) -> None:
    low, high = APPROVED_SLICE
    if not seeds or not all(low <= s < high for s in seeds):
        raise AdaptationConfigError(
            f"held-out seeds must lie in the approved slice [{low}, {high})"
        )


def heldout_predictors(contact: ContactEvaluation) -> dict[str, Predictor | None]:
    model = load_dynamics(resolve(contact.model), contact.model_sha256)
    k0 = EstimatorConfig(**contact.estimator).nominal_gain
    return {"base_only": None, "phase_e_predictor": phase_e_predictor(k0),
            "contact_learned": learned_predictor(model, move_object=True),
            "learned_gripper_consistent": learned_gripper_consistent(model),
            "oracle_sim_predictor": None}  # fmt: skip


def run_policy(args: tuple[Heldout, str]) -> tuple[list[dict], list[dict]]:
    spec, checkpoint = args
    return run_configurations(spec.contact, checkpoint, CONFIGURATIONS, LOGGED,
                              heldout_predictors(spec.contact))  # fmt: skip


# --- pre-registered analysis ----------------------------------------------------------------

DERIVED_TABLES = ("gatekeeping", "safety_gates", "by_condition", "by_policy_seed",
                  "prediction_accuracy", "closed_loop_prediction")  # fmt: skip


def _outcomes(frame: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
    w = frame.pivot_table(index=["policy_seed", "condition", "episode_seed"],
                          columns="configuration", values="success", aggfunc="first")  # fmt: skip
    w = w.astype(bool)
    ref, new = w[b], w[a]
    return {"success_rate": new.mean(), "reference_success_rate": ref.mean(),
            "rescued": int((~ref & new).sum()), "harmed": int((ref & ~new).sum()),
            "reference_successes": int(ref.sum()),
            "harmed_fraction": float((ref & ~new).sum() / max(ref.sum(), 1))}  # fmt: skip


def derive_heldout(episodes: pd.DataFrame, steps: pd.DataFrame, spec: Heldout):
    c = spec.contact
    n, s = c.bootstrap_resamples, c.bootstrap_seed
    delay = episodes[episodes["condition"].isin(c.delay_conditions)]

    def diff(frame, a, b):
        est, low, high, p, count = paired_difference(frame, a, b, n, s)
        return {"estimate": est, "ci_low": low, "ci_high": high, "p_two_sided": p,
                "episodes": count}  # fmt: skip

    gate_open, rows = True, []
    for name, a, b in HYPOTHESES:
        r = diff(delay, a, b)
        bound = spec.h1_min_effect if name == "H1" else 0.0
        dev = spec.development_estimates[name]
        status = ("supported" if r["ci_low"] > bound else "not supported") if gate_open else (
            "not tested (gate closed)")  # fmt: skip
        rows.append({"hypothesis": name, "configuration": a, "reference": b, **r,
                     "bound": bound, "status": status, "development_estimate": dev,
                     "shrinkage_flag": bool(r["estimate"] < 0.5 * dev)})  # fmt: skip
        gate_open = gate_open and r["ci_low"] > bound
    gatekeeping = pd.DataFrame(rows)

    gates = []
    for name in ("contact_learned", "phase_e_predictor"):
        for cond in c.harm_conditions:
            sub = episodes[episodes["condition"] == cond]
            r, o = diff(sub, name, "base_only"), _outcomes(sub, name, "base_only")
            gates.append({"configuration": name, "condition": cond, **r, **o,
                          "passed": r["ci_low"] > c.harm_ci_floor
                          and o["harmed_fraction"] <= c.harm_max_fraction,
                          "binding": name == "contact_learned"})  # fmt: skip
    safety = pd.DataFrame(gates)

    pairs = [("phase_e_predictor", "base_only"), ("contact_learned", "base_only"),
             ("contact_learned", "phase_e_predictor"),
             ("contact_learned", "learned_gripper_consistent"),
             ("oracle_sim_predictor", "phase_e_predictor")]  # fmt: skip
    by_condition = []
    for cond_key, sub in episodes.groupby("condition"):
        for a, b in pairs:
            by_condition.append({"condition": cond_key, "configuration": a, "reference": b,
                                 **diff(sub, a, b), **_outcomes(sub, a, b)})  # fmt: skip
    by_policy = []
    for pol, sub in delay.groupby("policy_seed"):
        for a, b in pairs:
            by_policy.append({"policy_seed": pol, "configuration": a, "reference": b,
                              **diff(sub, a, b)})  # fmt: skip

    accuracy = offline_accuracy(steps, c)
    acc_rows = []
    for (contact, h), sub in accuracy.groupby(["contact", "horizon"]):
        acc_rows.append({"contact": contact, "horizon": h, "steps": len(sub),
                         **{f"{col}_mean": sub[col].mean() for col in sub.columns
                            if col.endswith("_error")}})  # fmt: skip
    for contact, sub in accuracy.groupby("contact"):
        acc_rows.append({"contact": contact, "horizon": -1, "steps": len(sub),
                         **{f"{col}_mean": sub[col].mean() for col in sub.columns
                            if col.endswith("_error")}})  # fmt: skip

    oracle = diff(delay, "oracle_sim_predictor", "phase_e_predictor")["estimate"]
    h2 = gatekeeping.loc[gatekeeping["hypothesis"] == "H2", "estimate"].iloc[0]
    tables = {"gatekeeping": gatekeeping, "safety_gates": safety,
              "by_condition": pd.DataFrame(by_condition),
              "by_policy_seed": pd.DataFrame(by_policy),
              "prediction_accuracy": pd.DataFrame(acc_rows),
              "closed_loop_prediction": closed_loop_prediction(steps)}  # fmt: skip
    binding = safety[safety["binding"]]
    summary = {
        "hypotheses": dict(zip(gatekeeping["hypothesis"], gatekeeping["status"], strict=True)),
        "safety_gates_passed": bool(binding["passed"].all()),
        "share_of_oracle_gain_over_phase_e": float(h2 / oracle) if abs(oracle) > 1e-12
        else float("nan"),
        "plan_sha256": c.plan_sha256, "model_sha256": c.model_sha256,
        "approval": spec.heldout_approved, "episodes": len(episodes), "step_rows": len(steps),
        "seeds": [int(min(c.seeds)), int(max(c.seeds))],
    }  # fmt: skip
    return tables, summary
