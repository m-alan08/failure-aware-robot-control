"""Pre-registered G4 analysis (docs/PHASE_G4_HELDOUT_PROPOSAL.md, section 4)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from farc.adaptation.delay_eval import holm
from farc.delaychange.mechanism_analysis import step_metrics, two

TI_R = ("RT", "RI")
TI_F = ("F-T", "F-I")
PERSISTENT = ("RP", "F-P")
GATE_FAMILIES = ("RT", "RI", "RS", "RP", "F-T", "F-I", "F-P", "NG")


def fam(episodes: pd.DataFrame, families: tuple[str, ...]) -> pd.DataFrame:
    return episodes[episodes["family"].isin(families)]


def _row(name: str, frame: pd.DataFrame, a: str, b: str, bound: float, n: int, s: int,
         scope: str) -> dict[str, Any]:  # fmt: skip
    return {"test": name, "scope": scope, "configuration": a, "reference": b,
            **two(frame, a, b, bound, n, s)}  # fmt: skip


def primary(episodes: pd.DataFrame, c) -> tuple[pd.DataFrame, dict[str, Any]]:
    n, s, alpha = c.bootstrap_resamples, c.bootstrap_seed, c.alpha_one_sided
    p1 = _row("P1", fam(episodes, TI_R), "C", "contact_learned", 0.0, n, s, "R: RT+RI")
    p2 = _row("P2", fam(episodes, PERSISTENT), "C", "contact_learned",
              -c.persistent_noninferiority, n, s, "R+F: persistent")  # fmt: skip
    p3 = _row("P3", fam(episodes, TI_R), "C", "sham_C_cross_yoked", 0.0, n, s, "R: RT+RI")
    step1_p = max(p1["p"], p2["p"])
    step1 = step1_p <= alpha
    step2 = bool(step1 and p3["p"] <= alpha)
    if not step1:
        p1_label = "not confirmed"
    elif p1["estimate"] >= c.practical_threshold:
        p1_label = "confirmed and practically meaningful"
    else:
        p1_label = "confirmed, below practical threshold"
    m = c.specificity_minimum
    if not step1:
        p3_label = "not tested (step 1 failed)"
    elif step2 and p3["estimate"] >= m:
        p3_label = "meaningful"
    elif step2:
        p3_label = "significant but small"
    elif -m < p3["ci_low"] and p3["ci_high"] < m:
        p3_label = "negligible"
    else:
        p3_label = "inconclusive"
    table = pd.DataFrame([p1, p2, p3])
    table["step"] = [1, 1, 2]
    table["label"] = [p1_label, "passed" if p2["p"] <= alpha else "failed", p3_label]
    return table, {"step1_p_iut": step1_p, "step1_passed": bool(step1),
                   "step2_passed": step2, "P1_label": p1_label, "P3_label": p3_label}  # fmt: skip


def secondary(episodes: pd.DataFrame, c) -> pd.DataFrame:
    n, s = c.bootstrap_resamples, c.bootstrap_seed
    sac, rob = -c.sac_noninferiority, -c.robustness_noninferiority
    rows = [
        _row("S1", fam(episodes, TI_R), "C", "base_only", sac, n, s, "R: RT+RI"),
        _row("S2", fam(episodes, TI_F), "C", "base_only", sac, n, s, "F: T+I"),
        _row("S3", fam(episodes, PERSISTENT), "C", "base_only", 0.0, n, s, "R+F: persistent"),
        _row("S4", fam(episodes, TI_F), "C", "contact_learned", 0.0, n, s, "F: T+I"),
        _row("S5", fam(episodes, ("RS",)), "C", "contact_learned", 0.0, n, s, "R: RS"),
        _row("S6", fam(episodes, TI_R), "C", "sham_C_random_pause", 0.0, n, s, "R: RT+RI"),
        _row("S7", fam(episodes, ("NG",)), "C", "contact_learned", rob, n, s, "F: NG"),
    ]
    table = pd.DataFrame(rows)
    table["p_holm"] = holm(table["p"].tolist())
    table["result"] = ["supported" if p <= c.alpha_one_sided else "not supported"
                       for p in table["p_holm"]]  # fmt: skip
    return table


def gates(episodes: pd.DataFrame, c) -> pd.DataFrame:
    n, s = c.bootstrap_resamples, c.bootstrap_seed
    rows = []
    for cond in ("H_undisturbed", "H_noise0.01"):
        r = _row("no_delay", episodes[episodes["condition"] == cond], "C", "base_only",
                 c.harm_ci_floor, n, s, cond)  # fmt: skip
        r["harmed_fraction"] = r["harmed"] / max(r["reference_successes"], 1)
        r["passed"] = r["ci_low"] > c.harm_ci_floor and r["harmed_fraction"] <= c.harm_max_fraction
        rows.append(r)
    for f in GATE_FAMILIES:
        r = _row("family_harm", fam(episodes, (f,)), "C", "contact_learned",
                 c.family_harm_floor, n, s, f)  # fmt: skip
        r["passed"] = r["ci_low"] > c.family_harm_floor
        rows.append(r)
    return pd.DataFrame(rows)


def descriptive(episodes: pd.DataFrame, c) -> pd.DataFrame:
    n, s = c.bootstrap_resamples, c.bootstrap_seed
    rows = [_row("G2_same_condition_sham", fam(episodes, TI_F), "C", "sham_C_same_condition",
                 0.0, n, s, "F: T+I")]  # fmt: skip
    for cond, frame in episodes.groupby("condition"):
        for ref in ("contact_learned", "base_only"):
            rows.append(_row("per_condition", frame, "C", ref, 0.0, n, s, str(cond)))
    for pol, frame in fam(episodes, TI_R).groupby("policy_seed"):
        rows.append(_row("P1_per_policy", frame, "C", "contact_learned", 0.0, n, s,
                         f"policy {pol}"))  # fmt: skip
    for f in ("RT", "RI", "RS", "RP", "H", "F-P", "F-T", "F-I", "NG"):
        frame = fam(episodes, (f,))
        for a in sorted(set(frame["configuration"]) - {"contact_learned"}):
            rows.append(_row("per_family", frame, a, "contact_learned", 0.0, n, s, f))
    return pd.DataFrame(rows)


DERIVED = ("primary", "secondary", "gates", "descriptive", "success", "latency", "rates",
           "overhead")  # fmt: skip


def derive_g4(episodes: pd.DataFrame, steps: pd.DataFrame, c) -> tuple[dict, dict]:
    prim, steps_summary = primary(episodes, c)
    gate = gates(episodes, c)
    gates_ok = bool(gate["passed"].all())
    confirmed = steps_summary["step1_passed"] and gates_ok
    if not confirmed:
        verdict = "C not confirmed"
    elif steps_summary["step2_passed"]:
        verdict = "C confirmed, with episode-specific detection confirmed"
    else:
        verdict = "C confirmed; episode-specific detection not confirmed"
    success = episodes.groupby(["family", "configuration"])["success"].mean().unstack() * 100
    tables = {"primary": prim, "secondary": secondary(episodes, c), "gates": gate,
              "descriptive": descriptive(episodes, c), "success": success.reset_index(),
              **step_metrics(steps)}  # fmt: skip
    summary = {"verdict": verdict, "gates_passed": gates_ok, **steps_summary,
               "primary": {r["test"]: {"estimate": round(r["estimate"], 4),
                                       "ci": [round(r["ci_low"], 4), round(r["ci_high"], 4)],
                                       "p": r["p"], "label": r["label"]}
                           for r in prim.to_dict("records")},
               "episodes": len(episodes), "plan_sha256": c.plan_sha256,
               "constants": c.constants}  # fmt: skip
    return tables, summary
