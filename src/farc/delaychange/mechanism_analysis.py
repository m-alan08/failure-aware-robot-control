"""Pre-written G3 analysis (docs/PHASE_G3_PROPOSAL.md, sections 4.4-6)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from farc.adaptation.delay_eval import holm
from farc.delaychange.mechanism import (
    FIRST_ALARM_IDENTICAL,
    KEYS,
    NO_SHADOW,
    PRIVILEGED,
)
from farc.delaychange.stage3_analysis import paired

PRIMARY_SCOPE = ("RT", "RI")


class IntegrityError(RuntimeError):
    """A replayed run's shadow detector did not reproduce C's first alarm."""


def scope(episodes: pd.DataFrame, families: tuple[str, ...]) -> pd.DataFrame:
    return episodes[episodes["family"].isin(families)]


def contrast(frame: pd.DataFrame, weights: dict[str, float], bound: float, n: int, seed: int,
             two_sided: bool = False) -> dict[str, Any]:  # fmt: skip
    """sum_c w_c * success_c per pair, with an episode-seed cluster bootstrap. One-sided p is
    P(draw <= bound); two-sided p is 2 * min(P(draw <= 0), P(draw >= 0))."""
    w = frame.pivot_table(index=KEYS, columns="configuration", values="success",
                          aggfunc="first").astype(float)  # fmt: skip
    w = w[list(weights)].dropna()
    total = pd.Series(0.0, index=w.index)
    for c, v in weights.items():
        total = total + w[c] * v
    d = total.groupby(level="episode_seed")
    sums, counts = d.sum().to_numpy(), d.count().to_numpy()
    est = sums.sum() / counts.sum()
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n, len(sums)))
    draws = sums[idx].sum(1) / counts[idx].sum(1)
    low, high = np.quantile(draws, [0.025, 0.975])
    p = (min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean())) if two_sided
         else float((draws <= bound).mean()))  # fmt: skip
    return {"estimate": float(est), "ci_low": float(low), "ci_high": float(high),
            "p": float(p), "bound": bound, "episodes": len(w)}  # fmt: skip


def two(frame, a, b, bound, n, s) -> dict[str, Any]:
    r = paired(frame, a, b, bound, n, s)
    return {"estimate": r["estimate"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
            "p": r["p_one_sided"], "bound": bound, "episodes": r["episodes"],
            "rescued": r["rescued"], "harmed": r["harmed"],
            "reference_successes": r["reference_successes"]}  # fmt: skip


INTERACTION = {"C_full": 1.0, "C_reset": -1.0, "C_pause": -1.0, "contact_learned": 1.0}
ORACLE_INTERACTION = {"ORACLE_full": 1.0, "ORACLE_reset": -1.0, "ORACLE_pause": -1.0,
                      "contact_learned": 1.0}  # fmt: skip


def label(row: dict, mme: float, alpha: float, holm_key: str = "p_holm") -> str:
    sig = row.get(holm_key, 1.0) <= alpha
    est = abs(row["estimate"]) if row.get("two_sided") else row["estimate"]
    if sig and est >= mme:
        return "meaningful"
    if sig:
        return "significant but small"
    if -mme < row["ci_low"] and row["ci_high"] < mme:
        return "negligible"
    return "inconclusive"


def _test(name, frame, a, b, n, s, bound=0.0, two_sided=False, weights=None) -> dict[str, Any]:
    if weights is not None:
        r = contrast(frame, weights, bound, n, s, two_sided)
        r.update(rescued=np.nan, harmed=np.nan)
    else:
        r = two(frame, a, b, bound, n, s)
    return {"test": name, "configuration": a, "reference": b, "two_sided": two_sided, **r}


def hypotheses(episodes: pd.DataFrame, mme: float, alpha: float, n: int, s: int,
               sac_margin: float):  # fmt: skip
    prim = scope(episodes, PRIMARY_SCOPE)
    gate = _test("G", prim, "C_full", "contact_learned", n, s)
    gate["passed"] = gate["p"] <= alpha
    rows = [
        _test("M1", prim, "C_pause", "sham_pause_random", n, s),
        _test("M2", prim, "C_reset", "contact_learned", n, s),
        _test("M3", prim, "interaction", "", n, s, two_sided=True, weights=INTERACTION),
        _test("M4", prim, "C_full", "sham_full_yoked", n, s),
    ]
    primary = pd.DataFrame(rows)
    primary["p_holm"] = holm(primary["p"].tolist())
    primary["confirmatory"] = bool(gate["passed"])
    primary["label"] = [
        label(r, mme, alpha) if gate["passed"] else "descriptive (gatekeeper failed)"
        for r in primary.to_dict("records")
    ]
    rs = scope(episodes, ("RS",))
    sec = pd.DataFrame([
        {**_test("M1", rs, "C_pause", "sham_pause_random", n, s), "scope": "RS"},
        {**_test("M2", rs, "C_reset", "contact_learned", n, s), "scope": "RS"},
        {**_test("M4", rs, "C_full", "sham_full_yoked", n, s), "scope": "RS"},
        {**_test("pause_main", prim, "C_pause", "contact_learned", n, s), "scope": "RT+RI"},
        {**_test("vs_frozen_SAC", prim, "C_full", "base_only", n, s, bound=-sac_margin),
         "scope": "RT+RI"},
        {**_test("yoked_timing", prim, "sham_full_yoked", "contact_learned", n, s),
         "scope": "RT+RI"},
    ])  # fmt: skip
    sec["p_holm"] = holm(sec["p"].tolist())
    sec["label"] = [label(r, mme, alpha) for r in sec.to_dict("records")]
    return pd.DataFrame([gate]), primary, sec


def oracle_diagnostics(episodes: pd.DataFrame, n: int, s: int) -> pd.DataFrame:
    """PRIVILEGED references only; no tests."""
    prim = scope(episodes, PRIMARY_SCOPE)
    rows = []
    for name, fam in (("RT+RI", PRIMARY_SCOPE), ("RS", ("RS",))):
        f = scope(episodes, fam) if name == "RS" else prim
        for a, b in (("ORACLE_full", "C_full"), ("ORACLE_full", "contact_learned"),
                     ("ORACLE_pause", "contact_learned"), ("ORACLE_reset", "contact_learned"),
                     ("PRIV_true_schedule", "contact_learned"),
                     ("PRIV_true_schedule", "C_full")):  # fmt: skip
            rows.append({"scope": name, **_test("oracle", f, a, b, n, s)})
        inter = _test("oracle", f, "oracle_interaction", "", n, s, two_sided=True,
                      weights=ORACLE_INTERACTION)  # fmt: skip
        rows.append({"scope": name, **inter})
    return pd.DataFrame(rows)


def decomposition(episodes: pd.DataFrame, n: int, s: int) -> pd.DataFrame:
    """Every configuration against contact-aware and frozen SAC, per family (descriptive)."""
    rows = []
    names = sorted(set(episodes["configuration"]) - {"contact_learned"})
    for fam in ("RT", "RI", "RS", "RP", "H"):
        f = scope(episodes, (fam,))
        for a in names:
            rows.append({"scope": fam, **_test("vs_contact", f, a, "contact_learned", n, s)})
    prim = scope(episodes, PRIMARY_SCOPE)
    for a in names:
        rows.append({"scope": "RT+RI", **_test("vs_contact", prim, a, "contact_learned", n, s)})
    return pd.DataFrame(rows)


def safety(episodes: pd.DataFrame, n: int, s: int, persistent_margin: float, floor: float,
           max_fraction: float) -> pd.DataFrame:  # fmt: skip
    rows = []
    rp = scope(episodes, ("RP",))
    for a in ("C_full", "C_reset"):
        r = _test("persistent", rp, a, "contact_learned", n, s, bound=-persistent_margin)
        r["passed"] = r["ci_low"] > -persistent_margin
        rows.append(r)
    for cond in ("H_undisturbed", "H_noise0.01"):
        sub = episodes[episodes["condition"] == cond]
        for a in sorted(set(sub["configuration"]) - set(PRIVILEGED) - {"base_only"}):
            r = _test(cond, sub, a, "base_only", n, s, bound=floor)
            frac = r["harmed"] / max(r["reference_successes"], 1)
            r["harmed_fraction"] = frac
            r["passed"] = r["ci_low"] > floor and frac <= max_fraction
            rows.append(r)
    return pd.DataFrame(rows)


# --- step-level measures (section 5) and alarm divergence (section 4.4) -----------------------


def _transitions(dt: np.ndarray) -> list[tuple[int, str]]:
    return [(t, "end" if dt[t] == 0 else "onset" if dt[t - 1] == 0 else "change")
            for t in range(1, len(dt)) if dt[t] != dt[t - 1]]  # fmt: skip


def step_metrics(steps: pd.DataFrame) -> dict[str, pd.DataFrame]:
    trans, eps = [], []
    group = ["configuration", "family", "policy_seed", "condition", "episode_seed"]
    for (name, family, _p, _c, _s), ep in steps.groupby(group, sort=False):
        dt = ep["delay_true"].to_numpy(int)
        act = ep["active"].to_numpy(bool)
        eff = np.where(act, ep["delay_used"].to_numpy(int), 0)
        alarm = ep["alarm"].to_numpy(bool)
        tr = _transitions(dt)
        times = [t for t, _ in tr]
        for i, (t, kind) in enumerate(tr):
            nxt = times[i + 1] if i + 1 < len(times) else len(dt)
            if kind == "end":
                off = np.flatnonzero(~act[t:nxt])
                lat = float(off[0]) if len(off) else np.nan
            else:
                ok = np.flatnonzero(act[t:nxt] & (eff[t:nxt] == dt[t]))
                lat = float(ok[0]) if len(ok) else np.nan
            al = np.flatnonzero(alarm[t : min(t + 6, nxt)])
            trans.append({"configuration": name, "family": family, "kind": kind,
                          "latency": lat, "never": bool(np.isnan(lat)),
                          "alarm_within_5": bool(len(al))})  # fmt: skip
        prev = np.zeros(len(dt), int)  # the previous true delay, for "stale"
        for t, _ in tr:
            prev[t:] = dt[t - 1]
        wrong = act & (eff != dt)
        stale = wrong & (eff == prev) & (prev != dt)
        recent = np.zeros(len(dt), bool)
        for t in times:
            recent[t : t + 6] = True
        offs: list[float] = []
        for a in np.flatnonzero(alarm):
            before = [int(a) - t for t in times if t <= a]
            offs.append(float(min(before)) if before else np.nan)
        eps.append({"configuration": name, "family": family, "steps": len(dt),
                    "incorrect": int(wrong.sum()), "stale": int(stale.sum()),
                    "missed": int(((dt > 0) & ~act).sum()),
                    "delayed": int((dt > 0).sum()), "free": int((dt == 0).sum()),
                    "on_delayed": int((act & (dt > 0)).sum()),
                    "on_free": int((act & (dt == 0)).sum()),
                    "alarms": int(alarm.sum()), "quiet_steps": int((~recent).sum()),
                    "false_alarms": int((alarm & ~recent).sum()),
                    "alarm_offset_median": float(np.nanmedian(offs)) if offs and
                    not np.all(np.isnan(offs)) else np.nan,
                    "switches": int((np.diff(act.astype(int) * 10 + eff) != 0).sum())})  # fmt: skip
    tr_df, ep_df = pd.DataFrame(trans), pd.DataFrame(eps)
    latency = (tr_df.groupby(["configuration", "family", "kind"])
               .agg(transitions=("latency", "size"), never=("never", "sum"),
                    latency_mean=("latency", "mean"), latency_median=("latency", "median"),
                    alarm_recall_5=("alarm_within_5", "mean"))
               .reset_index())  # fmt: skip
    sums = ep_df.groupby(["configuration", "family"]).sum(numeric_only=True)
    rates = pd.DataFrame({
        "incorrect_per_100": 100 * sums["incorrect"] / sums["steps"],
        "stale_per_100": 100 * sums["stale"] / sums["steps"],
        "missed_share_of_delayed": sums["missed"] / sums["delayed"].clip(lower=1),
        "active_share_delayed": sums["on_delayed"] / sums["delayed"].clip(lower=1),
        "active_share_free": sums["on_free"] / sums["free"].clip(lower=1),
        "alarms_per_episode": sums["alarms"] / ep_df.groupby(["configuration", "family"]).size(),
        "false_alarms_per_1000_quiet": 1000 * sums["false_alarms"]
        / sums["quiet_steps"].clip(lower=1),
        "switches_per_episode": sums["switches"] / ep_df.groupby(["configuration",
                                                                  "family"]).size(),
    }).reset_index()  # fmt: skip
    offsets = (ep_df.groupby(["configuration", "family"])["alarm_offset_median"].median()
               .rename("alarm_offset_median_of_medians").reset_index())  # fmt: skip
    overhead = (steps.groupby("configuration")["ctrl_ns"]
                .agg(median_us=lambda x: x.median() / 1e3, p99_us=lambda x: x.quantile(0.99) / 1e3)
                .reset_index())  # fmt: skip
    return {"latency": latency, "rates": rates.merge(offsets), "overhead": overhead}


def divergence(steps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shadow alarms in each run against C_full's own alarms in the same pair (section 4.4).
    Returns the per-configuration summary and per-pair agreement flags."""
    keyed = steps[["configuration", *KEYS, "t", "alarm", "shadow_alarm"]]
    c = (keyed[(keyed["configuration"] == "C_full") & keyed["alarm"]]
         .groupby(KEYS)["t"].apply(list).to_dict())  # fmt: skip
    rows, flags = [], []
    shadowed = keyed[~keyed["configuration"].isin(NO_SHADOW)]
    sh = (shadowed[shadowed["shadow_alarm"]].groupby(["configuration", *KEYS])["t"]
          .apply(list).to_dict())  # fmt: skip
    pairs = keyed[keyed["configuration"] == "C_full"][KEYS].drop_duplicates()
    for name in sorted(set(shadowed["configuration"])):
        first_same = later = later_matched = extra = full = n = 0
        for key in pairs.itertuples(index=False):
            k = tuple(key)
            ca = c.get(k, [])
            sa = sh.get((name, *k), [])
            n += 1
            first_same += ca[:1] == sa[:1]
            later += max(len(ca) - 1, 0)
            later_matched += sum(any(abs(a - b) <= 1 for b in sa) for a in ca[1:])
            unmatched = sum(not any(abs(a - b) <= 1 for b in ca) for a in sa)
            extra += unmatched
            agree = (ca[:1] == sa[:1] and all(any(abs(a - b) <= 1 for b in sa) for a in ca)
                     and unmatched == 0)  # fmt: skip
            full += agree
            flags.append({"configuration": name, **dict(zip(KEYS, k, strict=True)),
                          "agree": bool(agree), "c_events": len(ca)})  # fmt: skip
        rows.append({"configuration": name, "pairs": n, "first_alarm_identical": first_same / n,
                     "later_c_events": later,
                     "later_matched_share": later_matched / later if later else np.nan,
                     "unmatched_shadow_alarms_per_episode": extra / n,
                     "full_agreement_share": full / n})  # fmt: skip
    return pd.DataFrame(rows), pd.DataFrame(flags)


DERIVED = ("gatekeeper", "primary", "secondary", "oracle", "decomposition", "safety",
           "divergence", "sensitivity", "latency", "rates", "overhead")  # fmt: skip


def derive_g3(episodes: pd.DataFrame, steps: pd.DataFrame, config) -> tuple[dict, dict]:
    n, s = config.bootstrap_resamples, config.bootstrap_seed
    div, flags = divergence(steps)
    bad = div[div["configuration"].isin(FIRST_ALARM_IDENTICAL)
              & (div["first_alarm_identical"] < 1.0)]  # fmt: skip
    if len(bad):
        raise IntegrityError("shadow first alarm differs from C's:\n" + bad.to_string())
    gate, prim, sec = hypotheses(episodes, config.minimum_meaningful, config.alpha_one_sided,
                                 n, s, config.sac_noninferiority)  # fmt: skip
    # Sensitivity (descriptive, selection-biased): pairs where all three replays fully agree.
    agree = (flags[flags["configuration"].isin(FIRST_ALARM_IDENTICAL)]
             .groupby(KEYS)["agree"].all())  # fmt: skip
    keep = agree[agree].reset_index()[KEYS]
    sub = scope(episodes.merge(keep, on=KEYS), PRIMARY_SCOPE)
    sens = pd.DataFrame([
        _test("M1", sub, "C_pause", "sham_pause_random", n, s),
        _test("M2", sub, "C_reset", "contact_learned", n, s),
        _test("M3", sub, "interaction", "", n, s, two_sided=True, weights=INTERACTION),
        _test("M4", sub, "C_full", "sham_full_yoked", n, s),
    ])  # fmt: skip
    tables = {"gatekeeper": gate, "primary": prim, "secondary": sec,
              "oracle": oracle_diagnostics(episodes, n, s),
              "decomposition": decomposition(episodes, n, s),
              "safety": safety(episodes, n, s, config.persistent_noninferiority,
                               config.harm_ci_floor, config.harm_max_fraction),
              "divergence": div, "sensitivity": sens, **step_metrics(steps)}  # fmt: skip
    summary = {
        "gatekeeper_passed": bool(gate["passed"].iloc[0]),
        "gatekeeper_estimate": float(gate["estimate"].iloc[0]),
        "primary": {r.test: {"estimate": round(r.estimate, 4), "ci": [round(r.ci_low, 4),
                             round(r.ci_high, 4)], "p_holm": r.p_holm, "label": r.label}
                    for r in prim.itertuples()},
        "safety_all_passed": bool(tables["safety"]["passed"].all()),
        "episodes": len(episodes), "step_rows": len(steps),
        "sensitivity_pairs": len(keep), "constants": config.constants,
        "plan_sha256": config.plan_sha256,
    }  # fmt: skip
    return tables, summary
