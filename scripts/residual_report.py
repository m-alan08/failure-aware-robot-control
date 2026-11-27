"""Descriptive tables for the residual-RL development evaluation (after the pre-written analysis).

    PYTHONPATH=src python scripts/residual_report.py results/residual_rl

Reads ``evaluation/evaluation_episodes.csv.gz`` (failed runs substituted as C, exactly as the
analysis does), ``evaluation/evaluation_privileged_P.csv.gz``, ``evaluation/analysis.json`` and
the training logs. Writes ``report/`` with:

- success by configuration x group and configuration x family (network-seed mean);
- per-network R - C on each group with the cluster bootstrap (training-seed variability);
- paired harmed/rescued counts R vs C and R vs B0 per family;
- training summaries per run (status, rollbacks, final nominal check, residual magnitude).

Everything here is descriptive; the confirmatory result is ``analysis.json`` alone.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from farc.residual.analysis import _cluster_means, _paired, bootstrap, substitute_failed

GROUPS = ["N", "D", "M", "X", "N_rep", "O_rep"]


def mdt(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without the optional ``tabulate`` dependency."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]

    def cell(v) -> str:
        if isinstance(v, float):
            return "" if v != v else f"{v:.2f}"
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join(lines)


def _contrast(df, treat_mask, control_mask, plan):
    d = _paired(df, treat_mask, control_mask)
    return bootstrap(_cluster_means(d), plan["bootstrap_resamples"], plan["bootstrap_seed"])


def main(root: Path) -> None:
    plan = json.loads(Path("configs/residual/residual_rl.json").read_text())
    ev = root / "evaluation"
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ev / "evaluation_episodes.csv.gz")
    failed = json.loads((ev / "failed.json").read_text())
    df = substitute_failed(df, [tuple(f) for f in failed])
    md = ["# Residual RL development evaluation: descriptive report", ""]
    md.append(f"Episodes: {len(df):,}. Failed runs substituted as C: {failed or 'none'}.")
    md.append("")
    # 1. success by config x group (mean over network seeds, policies, seeds, cells)
    t = df.groupby(["config", "group"]).success.mean().unstack()[GROUPS] * 100
    order = [c for c in ["B0", "C", "R", "A1", "A2", "A3", "B2"] if c in t.index]
    t = t.loc[order]
    p_path = ev / "evaluation_privileged_P.csv.gz"
    if p_path.exists():
        p = pd.read_csv(p_path)
        t.loc["P (D only, privileged)"] = {"D": 100 * p.success.mean()}
    md += ["## Success (%) by configuration and group", "", mdt(t.round(2)), ""]
    fam = df.groupby(["family", "config"]).success.mean().unstack()[order] * 100
    if p_path.exists():
        fam["P"] = p.groupby("family").success.mean() * 100
    md += ["## Success (%) by family", "", mdt(fam.round(2)), ""]
    # 2. effects with uncertainty, pooled and per network
    rows = []
    for g in GROUPS:
        m = df.group == g
        for treat in [c for c in order if c not in ("C",)]:
            c = _contrast(df, m & (df.config == treat), m & (df.config == "C"), plan)
            rows.append({"contrast": f"{treat} - C", "group": g, "network": "pooled",
                         "estimate": 100 * c.estimate, "lo95": 100 * c.lower,
                         "hi95": 100 * c.upper})  # fmt: skip
            if treat in ("R", "A1", "A2", "A3", "B2"):
                for ns in sorted(df[df.config == treat].network_seed.unique()):
                    tm = m & (df.config == treat) & (df.network_seed == ns)
                    c = _contrast(df, tm, m & (df.config == "C"), plan)
                    rows.append({"contrast": f"{treat} - C", "group": g, "network": int(ns),
                                 "estimate": 100 * c.estimate, "lo95": 100 * c.lower,
                                 "hi95": 100 * c.upper})  # fmt: skip
        c = _contrast(df, m & (df.config == "R"), m & (df.config == "B0"), plan)
        rows.append(
            {
                "contrast": "R - B0",
                "group": g,
                "network": "pooled",
                "estimate": 100 * c.estimate,
                "lo95": 100 * c.lower,
                "hi95": 100 * c.upper,
            }
        )
    eff = pd.DataFrame(rows)
    eff.to_csv(out / "effects.csv", index=False)
    md += ["## Paired effects (points; 95% episode-seed cluster bootstrap, descriptive)", "",
           mdt(eff[eff.network == "pooled"].round(2), index=False), "",
           "### Per network (training-seed variability)", "",
           mdt(eff[eff.network != "pooled"].round(2), index=False), ""]  # fmt: skip
    # 3. harmed / rescued per family vs C and vs B0
    key = ["policy", "episode_seed", "cell"]
    hr = []
    for ref in ("C", "B0"):
        base = df[df.config == ref].set_index(key)["success"].astype(bool)
        r = df[df.config == "R"].copy()
        r["ref"] = base.reindex(pd.MultiIndex.from_frame(r[key])).to_numpy()
        r["harmed"] = r.ref & ~r.success.astype(bool)
        r["rescued"] = ~r.ref & r.success.astype(bool)
        hg = r.groupby("family")[["harmed", "rescued"]].sum()
        hg["n"] = r.groupby("family").size()
        hg.insert(0, "vs", ref)
        hr.append(hg.reset_index())
    hr_df = pd.concat(hr)
    md += ["## Harmed / rescued episodes, R vs reference (counted per network seed)", "",
           mdt(hr_df, index=False), ""]  # fmt: skip
    # 4. residual magnitude and fallbacks
    if "residual_norm_mean" in df:
        mag = df[df.config.isin(["R", "A1", "A2", "A3", "B2"])].groupby(["config", "group"])
        md += ["## Mean per-step |a_final - a_C| by group", "",
               mdt(mag.residual_norm_mean.mean().unstack().round(4)), "",
               f"Fallbacks: {int(df.get('fallbacks', pd.Series([0])).sum())}", ""]  # fmt: skip
    # 5. C alarm rates with and without the residual (closed-loop coupling, §6 safety)
    if "alarms" in df:
        al = df[df.config.isin(["C", "R"])].groupby(["group", "config"]).alarms
        md += ["## Mean C alarms per episode", "", mdt(al.mean().unstack().round(3)), ""]
    # 6. training summaries
    tr = []
    for d in sorted(root.glob("*_n[0-2]")):
        rows_ = [json.loads(x) for x in (d / "log.jsonl").read_text().splitlines()]
        end: dict = next((r for r in rows_ if r["kind"] == "end"), {})
        evs = [r for r in rows_ if r["kind"] == "nominal_eval"]
        eps = pd.DataFrame([r for r in rows_ if r["kind"] == "episode"]).tail(2000)
        tr.append({"run": d.name, "status": end.get("status"), "rollbacks": end.get("rollbacks"),
                   "c_nominal": evs[0]["c_success"] if evs else None,
                   "nominal_checks": " ".join(f"{e['success']:.2f}" for e in evs),
                   "train_success_last2k": eps.success.mean(),
                   "delta_abs_last2k": eps.delta_abs.mean(),
                   "delta_sat_last2k": eps.delta_sat.mean(),
                   "correction_last2k": eps.correction_abs.mean()})  # fmt: skip
    md += ["## Training runs", "", mdt(pd.DataFrame(tr).round(4), index=False), ""]
    analysis = json.loads((ev / "analysis.json").read_text())
    pts = ["estimate", "lower", "upper", "margin", "practical"]
    prim = pd.DataFrame(analysis["primary"]).drop(columns=["per_network"])
    sec = pd.DataFrame(analysis["secondary"])
    for t_ in (prim, sec):  # fractions -> points for display only
        for col in [c for c in pts if c in t_]:
            t_[col] = pd.to_numeric(t_[col], errors="coerce") * 100
    per_net = {k: round(100 * v, 2) for k, v in
               (analysis["primary"][2].get("per_network") or {}).items()}  # fmt: skip
    md += ["## Pre-written analysis (confirmatory; points)", "",
           f"Claim: **{analysis['claim']}**", "", mdt(prim, index=False), "",
           f"Per-network efficacy estimates (points): {per_net}", "",
           mdt(sec, index=False), ""]  # fmt: skip
    (out / "report.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
