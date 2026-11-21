"""Selection (§5) and pre-written development analysis (§6) for residual RL. NOT RUN.

Written before any training or selection seed is used, from the DRAFT pre-registration
(``docs/RESIDUAL_RL_PREREGISTRATION.md``, not approved). Functions are pure: they take a long
episode table and return numbers; nothing here reads seeds or runs episodes.

Episode table (one row per episode):

| column | meaning |
|---|---|
| ``config`` | ``C``, ``B0``, ``R``, ``A1``, ``A2``, ``A3``, ``B2`` (``P`` is ignored here) |
| ``alpha`` | residual scale (selection only; NaN for C) |
| ``network_seed`` | residual network seed (NaN or -1 for non-residual configs) |
| ``policy`` | frozen SAC policy id |
| ``episode_seed`` | the bootstrap cluster |
| ``cell`` | cell name |
| ``group`` | ``N``, ``D``, ``M`` or ``X`` (development); unused at selection |
| ``family`` | e.g. ``nominal``, ``T-M``, ``T-O``, ``T-A``, ``RT``, ``RP``, ``RI`` |
| ``success`` | bool |

Units are fractions (0.01 = 1 point). Pairing: the residual effect is averaged over network
seeds within each (policy, episode seed, cell) *before* differencing (§6 "Unit"); the
bootstrap resamples episode seeds (clusters) with all their policies and cells.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

KEYS = ["policy", "episode_seed", "cell"]
EFFICACY_FAMILIES = ("T-M", "T-O", "T-A")
CLAIMS = {
    4: "residual RL adds robustness beyond C, including to disturbance families not seen in "
       "training",
    3: "residual RL improves performance on the disturbance families it was trained on; "
       "generalisation not shown",
    2: "null for Q1",
    0: "the residual degrades C",
}  # fmt: skip


class AnalysisInputError(ValueError):
    pass


def substitute_failed(df: pd.DataFrame, failed: Sequence[tuple[str, float, int]]) -> pd.DataFrame:
    """Runs stopped by the rollback rule enter analysis as C (delta = 0), not as excluded
    (§4). ``failed`` lists (config, alpha, network_seed); their success becomes C's."""
    out = df.copy()
    c = df[df.config == "C"].set_index(KEYS)["success"]
    for config, alpha, ns in failed:
        m = (out.config == config) & (out.network_seed == ns)
        if not np.isnan(alpha):
            m &= out.alpha == alpha
        if not m.any():
            raise AnalysisInputError(f"no rows for failed run {(config, alpha, ns)}")
        out.loc[m, "success"] = c.reindex(pd.MultiIndex.from_frame(out.loc[m, KEYS])).to_numpy()
    return out


def _paired(df: pd.DataFrame, treat: pd.Series, control: pd.Series) -> pd.DataFrame:
    """Per (policy, episode_seed, cell): network-seed-averaged treatment minus control."""
    t = df[treat].groupby(KEYS).success.agg(["mean", "size"])
    c = df[control].groupby(KEYS).success.agg(["mean", "size"])
    if t.empty or c.empty:
        raise AnalysisInputError("a contrast has no rows")
    if set(t.index) != set(c.index):
        raise AnalysisInputError("treatment and control cover different (policy, seed, cell)")
    if t["size"].nunique() != 1 or c["size"].nunique() != 1:
        raise AnalysisInputError("unbalanced design: network-seed counts differ across pairs")
    d = (t["mean"] - c["mean"].reindex(t.index)).rename("diff").reset_index()
    return d


def _cluster_means(d: pd.DataFrame) -> pd.Series:
    sizes = d.groupby("episode_seed").size()
    if sizes.nunique() != 1:
        raise AnalysisInputError("unbalanced design: episode seeds have different cell counts")
    return d.groupby("episode_seed")["diff"].mean().sort_index()


@dataclass(frozen=True)
class Contrast:
    name: str
    estimate: float
    lower: float  # one-sided 97.5% lower bound (2.5th bootstrap percentile)
    upper: float  # 97.5th percentile (reported, not tested)
    p_one_sided: float  # bootstrap P(estimate <= 0), with +1 correction
    n_clusters: int


def bootstrap(cluster_means: pd.Series, resamples: int, seed: int) -> Contrast:
    x = cluster_means.to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (resamples, len(x)))
    boot = x[idx].mean(axis=1)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = (1 + int(np.sum(boot <= 0))) / (resamples + 1)
    return Contrast("", float(x.mean()), float(lo), float(hi), p, len(x))


def contrast(df: pd.DataFrame, name: str, treat: str, control: str, group: str | None = None,
             family: str | None = None, resamples: int = 10_000,
             seed: int = 20261900) -> Contrast:  # fmt: skip
    m = pd.Series(True, index=df.index)
    if group is not None:
        m &= df.group == group
    if family is not None:
        m &= df.family == family
    d = _paired(df, m & (df.config == treat), m & (df.config == control))
    c = bootstrap(_cluster_means(d), resamples, seed)
    return Contrast(name, c.estimate, c.lower, c.upper, c.p_one_sided, c.n_clusters)


# --- §5 selection ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    alpha: float | None  # None: no arm passes; the experiment stops with a null result (§8.2)
    table: pd.DataFrame


def select_alpha(df: pd.DataFrame, alphas: Sequence[float], nominal_margin: float = 0.01,
                 tie: float = 0.005) -> Selection:  # fmt: skip
    """§5 rule on the selection block. ``df`` holds C rows and R rows with an ``alpha``."""
    rows = []
    for a in alphas:
        sub = df[(df.config == "C") | ((df.config == "R") & (df.alpha == a))]
        if not (sub.config == "R").any():
            raise AnalysisInputError(f"no selection rows for alpha={a}")
        nom = _paired(sub, (sub.family == "nominal") & (sub.config == "R"),
                      (sub.family == "nominal") & (sub.config == "C"))["diff"].mean()  # fmt: skip
        eff_mask = sub.family.isin(EFFICACY_FAMILIES)
        eff = _paired(sub, eff_mask & (sub.config == "R"),
                      eff_mask & (sub.config == "C"))["diff"].mean()  # fmt: skip
        rows.append({"alpha": a, "nominal_diff": nom, "efficacy_diff": eff,
                     "passes_nominal": bool(nom >= -nominal_margin - 1e-12)})  # fmt: skip
    table = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
    ok = table[table.passes_nominal]
    if ok.empty:
        return Selection(None, table)
    best = ok.efficacy_diff.max()
    chosen = ok[ok.efficacy_diff >= best - tie - 1e-12].alpha.min()  # ties -> smaller alpha
    return Selection(float(chosen), table)


# --- §6 development analysis ----------------------------------------------------------------


PRIMARY = (("nominal protection", "N", -0.01, None), ("delay protection", "D", -0.01, None),
           ("efficacy (Q1)", "M", 0.0, 0.02), ("generalisation (Q2)", "X", 0.0, None))  # fmt: skip
SECONDARY = (("S1 R-A1 on M", "R", "A1", "M", None), ("S2 R-B2 on M", "R", "B2", "M", None),
             ("S3 R-A2 on D", "R", "A2", "D", None), ("S4 A3-C on D", "A3", "C", "D", None),
             ("S5 R-C on D (RI)", "R", "C", "D", "RI"))  # fmt: skip


def holm(pvalues: Sequence[float]) -> list[float]:
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def development_analysis(df: pd.DataFrame, resamples: int = 10_000, seed: int = 20261900,
                         ni_margin: float = 0.01, practical: float = 0.02) -> dict:  # fmt: skip
    """Fixed-sequence gatekeeping (one-sided alpha 0.025), claim wording, Holm secondaries,
    and the always-reported safety counts. Every bootstrap uses the same seed, so all
    contrasts share resampling indices when they share clusters."""
    for g in ("N", "D", "M", "X"):
        if not (df.group == g).any():
            raise AnalysisInputError(f"group {g} is missing")
    steps, passed = [], 0
    gate_open = True
    primary = [(n, g, -ni_margin if m < 0 else m, None if p is None else practical)
               for n, g, m, p in PRIMARY]  # fmt: skip
    for i, (name, group, margin, prac) in enumerate(primary, start=1):
        c = contrast(df, name, "R", "C", group=group, resamples=resamples, seed=seed)
        ok = c.lower > margin and (prac is None or c.estimate >= prac - 1e-12)
        tested = gate_open
        # Degradation needs its own evidence: the upper one-sided 97.5% bound below 0.
        # Failing to show non-inferiority (lower bound <= -margin) is NOT that evidence.
        degraded = bool(c.upper < 0)
        steps.append({"step": i, "claim": name, "group": group, "estimate": c.estimate,
                      "lower": c.lower, "upper": c.upper, "margin": margin,
                      "practical": prac, "tested": tested,
                      "passed": bool(ok) if tested else None,
                      "degradation_shown": degraded if tested else None,
                      "degradation_beyond_margin": bool(c.upper < margin) if tested and i <= 2
                      else None})  # fmt: skip
        if tested and ok:
            passed = i
        gate_open = gate_open and ok
    # The draft's wording, reproduced literally (it reports any failure of step 1 or 2 as
    # degradation), and the proposed corrected wording (docs/RESIDUAL_RL_PROTOCOL_REVIEW.md,
    # S-4). Both are always returned; neither is chosen silently.
    claim_draft = CLAIMS[min(passed, 4)] if passed >= 2 else CLAIMS[0]
    if passed >= 2:
        claim = claim_draft
    else:
        failed = steps[passed]  # the first protection step that failed
        if failed["degradation_shown"]:
            claim = f"the residual degrades C on {failed['group']} (upper bound < 0)"
        else:
            claim = (f"non-inferiority to C not shown on {failed['group']}; degradation not "
                     "shown either (inconclusive)")  # fmt: skip
    secondary: list[dict[str, Any]] = []
    present = set(df.config)
    for name, treat, control, group, family in SECONDARY:
        rows_exist = (df.group == group) & (df.family == family if family else True)
        if treat not in present or control not in present or not rows_exist.any():
            secondary.append({"name": name, "missing": True})
            continue
        c = contrast(df, name, treat, control, group=group, family=family,
                     resamples=resamples, seed=seed)  # fmt: skip
        secondary.append({"name": name, "estimate": c.estimate, "lower": c.lower,
                          "upper": c.upper, "p": c.p_one_sided})  # fmt: skip
    with_p = [s for s in secondary if "p" in s]
    for s, adj in zip(with_p, holm([s["p"] for s in with_p]), strict=True):
        s["p_holm"] = adj
        s["descriptive_only"] = passed < 3
    return {"primary": steps, "steps_passed": passed, "claim": claim,
            "claim_draft_wording": claim_draft,
            "secondary": secondary, "safety": harmed_rescued(df)}  # fmt: skip


def harmed_rescued(df: pd.DataFrame, treat: str = "R") -> pd.DataFrame:
    """Per family: episodes C solved and R lost (harmed), and the reverse (rescued), counted
    per network seed."""
    c = df[df.config == "C"].set_index(KEYS)["success"]
    r = df[df.config == treat].copy()
    r["c_success"] = c.reindex(pd.MultiIndex.from_frame(r[KEYS])).to_numpy()
    r["harmed"] = r.c_success.astype(bool) & ~r.success.astype(bool)
    r["rescued"] = ~r.c_success.astype(bool) & r.success.astype(bool)
    return r.groupby("family")[["harmed", "rescued"]].sum().astype(int).reset_index()


def rollback_stop(failed_runs: int, total_runs: int) -> bool:
    """§8.3: the experiment stops if MORE than one third of the R-alpha* runs fail by the
    rollback rule (with 3 runs: 2 or 3 failures stop it, 1 does not)."""
    if total_runs < 1 or not 0 <= failed_runs <= total_runs:
        raise AnalysisInputError("need 0 <= failed_runs <= total_runs and total_runs >= 1")
    return 3 * failed_runs > total_runs
