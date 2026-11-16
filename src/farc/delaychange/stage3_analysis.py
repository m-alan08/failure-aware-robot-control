"""Pre-registered Stage-3 analysis (docs/PHASE_G2_PROPOSAL.md, sections 5.4-6)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from farc.adaptation.delay_eval import holm
from farc.delaychange.stage3 import CANDIDATES, KEYS, G2Config

FAMILY_SETS = {"T+I": ("T", "I"), "P": ("P",), "S": ("S",), "NG": ("NG",), "H": ("H",)}


def paired(frame: pd.DataFrame, a: str, b: str, bound: float, n: int, seed: int) -> dict:
    """Paired success difference a - b with an episode-seed cluster bootstrap; the one-sided
    p-value is P(draw <= bound)."""
    w = frame.pivot_table(index=KEYS, columns="configuration", values="success",
                          aggfunc="first").astype(float)  # fmt: skip
    if a not in w or b not in w:
        return {"estimate": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_one_sided": np.nan,
                "bound": bound, "episodes": 0, "rescued": 0, "harmed": 0,
                "reference_successes": 0}  # fmt: skip
    w = w[[a, b]].dropna()
    d = (w[a] - w[b]).groupby(level="episode_seed")
    sums, counts = d.sum().to_numpy(), d.count().to_numpy()
    est = sums.sum() / counts.sum()
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n, len(sums)))
    draws = sums[idx].sum(1) / counts[idx].sum(1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"estimate": float(est), "ci_low": float(low), "ci_high": float(high),
            "p_one_sided": float((draws <= bound).mean()), "bound": bound,
            "episodes": len(w), "rescued": int(((w[b] == 0) & (w[a] == 1)).sum()),
            "harmed": int(((w[b] == 1) & (w[a] == 0)).sum()),
            "reference_successes": int((w[b] == 1).sum())}  # fmt: skip


def _family(episodes: pd.DataFrame, key: str) -> pd.DataFrame:
    return episodes[episodes["family"].isin(FAMILY_SETS[key])]


def hypotheses(episodes: pd.DataFrame, config: G2Config):
    n, s = config.bootstrap_resamples, config.bootstrap_seed
    present = [c for c in CANDIDATES if c in set(episodes["configuration"])]
    primary = []
    for x in present:
        h1 = paired(_family(episodes, "T+I"), x, "contact_learned", 0.0, n, s)
        h2 = paired(_family(episodes, "P"), x, "contact_learned",
                    -config.persistent_noninferiority, n, s)  # fmt: skip
        primary.append({"candidate": x,
                        **{f"H1_{k}": v for k, v in h1.items()},
                        **{f"H2_{k}": v for k, v in h2.items()},
                        "p_iut": max(h1["p_one_sided"], h2["p_one_sided"]),
                        "H1_practically_meaningful":
                            h1["estimate"] >= config.transient_margin_practical})  # fmt: skip
    prim = pd.DataFrame(primary)
    if len(prim):
        prim["p_holm"] = holm(prim["p_iut"].tolist())
        prim["status"] = np.where(prim["p_holm"] <= config.alpha_one_sided, "supported",
                                  "not supported")  # fmt: skip

    tests = []
    for x in present:
        tests.append(("S1_changes", x, "contact_learned", "S", 0.0))
    for x, sham in (("cand_C_sequential", "sham_C_yoked"), ("cand_B_two_timescale",
                                                            "sham_B_yoked")):  # fmt: skip
        if x in present:
            tests.append(("S2_specificity", x, sham, "T+I", 0.0))
    if "cand_C_sequential" in present:
        tests.append(("S3_beyond_G1", "cand_C_sequential", "sup_A_release", "T+I", 0.0))
    for x in present:
        tests.append(("S4_vs_frozen_SAC", x, "base_only", "T+I", -config.sac_noninferiority))
    for x in present:
        tests.append(("S5_robustness", x, "contact_learned", "NG",
                      -config.robustness_noninferiority))  # fmt: skip
    sec = pd.DataFrame([{"test": t, "configuration": a, "reference": b, "family": f,
                         **paired(_family(episodes, f), a, b, bound, n, s)}
                        for t, a, b, f, bound in tests])  # fmt: skip
    if len(sec):
        sec["p_holm"] = holm(sec["p_one_sided"].fillna(1.0).tolist())
        sec["status"] = np.where(sec["p_holm"] <= config.alpha_one_sided, "supported",
                                 "not supported")  # fmt: skip

    gates = []
    for x in present:
        for cond in ("H_undisturbed", "H_noise0.01"):
            sub = episodes[episodes["condition"] == cond]
            r = paired(sub, x, "base_only", config.harm_ci_floor, n, s)
            frac = r["harmed"] / max(r["reference_successes"], 1)
            gates.append({"configuration": x, "condition": cond, **r, "harmed_fraction": frac,
                          "passed": r["ci_low"] > config.harm_ci_floor
                          and frac <= config.harm_max_fraction})  # fmt: skip
    return prim, sec, pd.DataFrame(gates)


def comparisons(episodes: pd.DataFrame, config: G2Config) -> pd.DataFrame:
    """Descriptive: every configuration vs frozen SAC and vs contact-aware, per family and per
    condition, with rescued and harmed counts."""
    n, s = config.bootstrap_resamples, config.bootstrap_seed
    rows = []
    names = sorted(set(episodes["configuration"]) - {"base_only"})
    for key in FAMILY_SETS:
        frame = _family(episodes, key)
        for a in names:
            for b in ("base_only", "contact_learned"):
                if a != b:
                    rows.append({"scope": key, "configuration": a, "reference": b,
                                 **paired(frame, a, b, 0.0, n, s)})  # fmt: skip
    for cond, frame in episodes.groupby("condition"):
        for a in names:
            for b in ("base_only", "contact_learned"):
                if a != b:
                    rows.append({"scope": cond, "configuration": a, "reference": b,
                                 **paired(frame, a, b, 0.0, n, s)})  # fmt: skip
    return pd.DataFrame(rows)


def step_metrics(steps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Section 6.2: detection, alarms, stale duration, false alarms/releases/activations,
    estimator accuracy, switching and overhead."""
    transitions, episodes_rows = [], []
    for (name, family, _p, cond, _s), ep in steps.sort_values("t").groupby(
            ["configuration", "family", "policy_seed", "condition", "episode_seed"]):  # fmt: skip
        dt = ep["delay_true"].to_numpy(int)
        act = ep["active"].to_numpy(bool)
        eff = np.where(act, ep["delay_used"].to_numpy(int), 0)
        alarm = ep["alarm"].to_numpy(bool)
        for t in range(1, len(dt)):
            if dt[t] == dt[t - 1]:
                continue
            nxt = next((u for u in range(t + 1, len(dt)) if dt[u] != dt[t]), len(dt))
            right = eff[t:nxt] == dt[t:nxt]
            hit = np.flatnonzero(right)
            al = np.flatnonzero(alarm[t:nxt])
            transitions.append({
                "configuration": name, "family": family, "condition": cond,
                "kind": "end" if dt[t] == 0 else "onset" if dt[t - 1] == 0 else "change",
                "detection_latency": float(hit[0]) if len(hit) else np.nan,
                "never_correct": not len(hit), "segment": int(nxt - t),
                "mismatched_steps": int((~right).sum()),
                "alarm_latency": float(al[0]) if len(al) else np.nan})  # fmt: skip
        engaged = np.maximum.accumulate(act) & (dt > 0)
        mode = act.astype(int) * 10 + eff
        episodes_rows.append({
            "configuration": name, "family": family, "condition": cond,
            "steps": len(dt), "correct_state_steps": int((eff == dt).sum()),
            "no_delay_steps": int((dt == 0).sum()), "false_active": int((act & (dt == 0)).sum()),
            "engaged_delay_steps": int(engaged.sum()),
            "off_after_engagement": int((engaged & ~act).sum()),
            "alarms": int(alarm.sum()), "active_steps": int(act.sum()),
            "switches": int((mode[1:] != mode[:-1]).sum()),
            "learned_calls": int(ep["learned_calls"].sum())})  # fmt: skip
    tr, ep = pd.DataFrame(transitions), pd.DataFrame(episodes_rows)
    latency = (tr.groupby(["configuration", "family", "kind"])
               .agg(transitions=("segment", "size"), never_correct=("never_correct", "sum"),
                    detection_median=("detection_latency", "median"),
                    detection_p90=("detection_latency", lambda x: x.quantile(0.9)),
                    alarm_median=("alarm_latency", "median"),
                    stale_or_mismatched_steps_mean=("mismatched_steps", "mean"))
               .reset_index())  # fmt: skip
    sums = ep.groupby(["configuration", "family"]).sum(numeric_only=True)
    rates = pd.DataFrame({
        "estimator_accuracy": sums["correct_state_steps"] / sums["steps"],
        "false_activation_rate": sums["false_active"] / sums["no_delay_steps"].clip(lower=1),
        "false_release_rate": sums["off_after_engagement"]
        / sums["engaged_delay_steps"].clip(lower=1),
        "alarms_per_1000_active_steps": 1000 * sums["alarms"] / sums["active_steps"].clip(lower=1),
    }).reset_index()  # fmt: skip
    switching = (ep.groupby(["configuration", "family"])
                 .agg(switches_mean=("switches", "mean"), switches_max=("switches", "max"),
                      episodes_ge6=("switches", lambda x: int((x >= 6).sum())),
                      learned_calls_per_episode=("learned_calls", "mean"))
                 .reset_index())  # fmt: skip
    overhead = (steps.groupby("configuration")["ctrl_ns"]
                .agg(median_us=lambda x: x.median() / 1e3, p99_us=lambda x: x.quantile(0.99) / 1e3)
                .reset_index())  # fmt: skip
    return {"latency": latency, "rates": rates, "switching": switching, "overhead": overhead}


DERIVED = ("primary", "secondary", "safety_gates", "comparisons", "latency", "rates",
           "switching", "overhead")  # fmt: skip


def derive_g2(episodes: pd.DataFrame, steps: pd.DataFrame, config: G2Config):
    prim, sec, gates = hypotheses(episodes, config)
    tables: dict[str, pd.DataFrame] = {"primary": prim, "secondary": sec, "safety_gates": gates,
                                       "comparisons": comparisons(episodes, config),
                                       **step_metrics(steps)}  # fmt: skip
    summary: dict[str, Any] = {
        "primary": ({r.candidate: r.status for r in prim.itertuples()} if len(prim) else {}),
        "safety_gates_passed": ({k: bool(v) for k, v in
                                 gates.groupby("configuration")["passed"].all().items()}
                                if len(gates) else {}),
        "constants": config.constants, "plan_sha256": config.plan_sha256,
        "calibration_record_sha256": config.calibration_record_sha256,
        "episodes": len(episodes), "step_rows": len(steps),
        "seeds": [min(config.seeds), max(config.seeds)],
    }  # fmt: skip
    return tables, summary
