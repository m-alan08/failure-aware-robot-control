"""Statistical design audit for the residual-RL plan (protocol review, not an experiment).

Inputs: the EXISTING G4 held-out episode table (already collected; no new episodes, no new
seeds) for C success rates, paired discordance between close variants of C, and the
within-seed correlation across policies. Everything else is SYNTHETIC simulation of the §6
design: paired binary outcomes per (episode seed, policy, cell), residual averaged over 3
network seeds, episode-seed cluster percentile bootstrap, one-sided alpha 0.025.

Reported:
1. G4 inputs.
2. Rejection rate of H0: delta <= -margin when delta = -margin exactly (type-I error) and when
   delta = 0 (power), per group (N, D, M), margin (1 or 2 points), discordance and seeds.
3. Bootstrap coverage when the 3 trained networks differ in their true effect (a run-level
   variance component the bootstrap does not resample).
4. The §5 nominal filter: probability that an arm with a given true nominal harm passes.
5. Normal-approximation pair counts for 80/90% power.

Two simulator artefacts were found and fixed during review (both made the bootstrap look
wrong when it was not): rescaling success probabilities by each run's realised mean coupled
the clusters; setting harm/rescue rates from the unclipped mean moved the true effect off the
boundary. Output: ``results/residual_protocol_review/power_audit.json``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

G4 = "~/farc-data/experiments/phase_g4_heldout/episodes.csv.gz"
Z975, Z80, Z90 = 1.959964, 0.841621, 1.281552


def g4_inputs(path: str) -> dict:
    d = pd.read_csv(Path(path).expanduser())
    w = d.pivot_table(
        index=["arm", "condition", "family", "policy_seed", "episode_seed"],
        columns="configuration",
        values="success",
        aggfunc="first",
    ).reset_index()
    out: dict = {"episodes": len(d), "families": {}}
    for fam, g in w.groupby("family"):
        out["families"][fam] = {
            "pairs": len(g),
            "C_success": float(g.C.mean()),
            "discordance_C_vs_contact_learned": float((g.contact_learned != g.C).mean()),
            "discordance_C_vs_base_only": float((g.base_only != g.C).mean()),
        }
    corr = []
    for _, g in w.groupby("condition"):
        p = g.pivot(index="episode_seed", columns="policy_seed", values="C").astype(float)
        c = p.corr().to_numpy()[np.triu_indices(3, 1)]
        corr.extend(c[np.isfinite(c)].tolist())
    out["within_seed_policy_corr_median"] = float(np.median(corr))
    out["within_seed_policy_corr_max"] = float(np.max(corr))
    return out


def simulate(
    rng: np.random.Generator,
    n_seeds: int,
    n_cells: int,
    p_c: float,
    harm: float,
    rescue: float,
    margin: float,
    sims: int,
    boot: int,
    n_policies: int = 3,
    n_nets: int = 3,
    shared: float = 0.5,
    seed_sd: float = 1.0,
    net_sd: float = 0.0,
) -> tuple[float, float, float]:
    """(rejection rate of H0: delta <= -margin, coverage of the 95% interval for the true
    mean-network delta, mean estimate). ``harm``/``rescue`` are unconditional per-pair rates
    P(C=1, R=0) and P(C=0, R=1) for an average network; ``net_sd`` spreads harm across the
    networks (lognormal, mean 1); ``shared`` is the probability that a network's discordance
    draw is the common one (correlation across network seeds); ``seed_sd`` is the logit-scale
    sd of a per-seed effect shared by policies and cells."""
    logit = math.log(p_c / (1 - p_c))

    def draw_pc(n: int) -> np.ndarray:
        z = rng.normal(0, seed_sd, (n, 1, 1))
        e = rng.normal(0, 0.3, (n, n_policies, n_cells))
        return 1 / (1 + np.exp(-(logit + z + e)))

    # A FIXED scale keeps E[C success] near p_c without coupling the simulated clusters.
    # E[pc] does not depend on the cell/policy layout, so it is estimated from 2M single
    # draws: with only a few thousand seeds the seed-level effect left an error of ~0.002 in
    # E[pc], which biased the rescue term (an artefact, caught in review).
    z = rng.normal(0, seed_sd, 2_000_000) + rng.normal(0, 0.3, 2_000_000)
    raw = 1 / (1 + np.exp(-(logit + z)))
    fixed_scale = p_c / raw.mean()
    p_pop = float(np.clip(raw * fixed_scale, 1e-6, 1 - 1e-6).mean())
    # Rates from the CLIPPED population mean, so the true effect is exactly rescue - harm.
    q1, q0 = harm / p_pop, rescue / (1 - p_pop)
    if q1 > 1 or q0 > 1:
        raise ValueError("harm or rescue rate infeasible for this success rate")
    true_delta = rescue - harm
    rej = cover = 0
    est = []
    for _ in range(sims):
        pc = np.clip(draw_pc(n_seeds) * fixed_scale, 1e-6, 1 - 1e-6)
        c = rng.random(pc.shape) < pc
        if net_sd:
            scale = np.exp(rng.normal(0, net_sd, n_nets) - net_sd**2 / 2)
        else:
            scale = np.ones(n_nets)
        common = rng.random(pc.shape)
        r_sum = np.zeros(pc.shape)
        for k in range(n_nets):
            own = rng.random(pc.shape)
            u = np.where(rng.random(pc.shape) < shared, common, own)
            flip = np.where(c, u < min(1.0, q1 * scale[k]), u < q0)
            r_sum += np.where(flip, ~c, c)
        diff = r_sum / n_nets - c
        x = diff.mean(axis=(1, 2))  # cluster (episode-seed) means
        idx = rng.integers(0, n_seeds, (boot, n_seeds))
        b = x[idx].mean(axis=1)
        lo, hi = np.quantile(b, [0.025, 0.975])
        rej += int(lo > -margin)
        cover += int(lo <= true_delta <= hi)
        est.append(float(x.mean()))
    return rej / sims, cover / sims, float(np.mean(est))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--g4", default=G4)
    ap.add_argument("--sims", type=int, default=400)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results/residual_protocol_review")
    a = ap.parse_args()
    rng = np.random.default_rng(20261901)
    inputs = g4_inputs(a.g4)
    # N: 1 cell, C ~ 99.7% (G4 H_undisturbed); D: 41 G4 cells, C ~ 93%; M: 7 cells, C ~ 85%
    # (assumed; T-M/T-O/T-A success under C is unknown until evaluated).
    groups = {"N": (1, 0.997), "D": (41, 0.93), "M": (7, 0.85)}
    # null scenarios: harm = rescue = discordance / 2. At N the ceiling caps rescue at 0.3%.
    null_disc = {"N": [0.0, 0.002, 0.006], "D": [0.02, 0.05, 0.10], "M": [0.02, 0.05, 0.10]}
    rows = []
    for g, (cells, pc) in groups.items():
        for margin in (0.01, 0.02):
            for n in (300, 600, 1200):
                for d in null_disc[g]:
                    r = min(d / 2, 0.9 * (1 - pc))
                    pw, cov, _ = simulate(rng, n, cells, pc, r, r, margin, a.sims, a.boot)
                    rows.append({"group": g, "margin": margin, "n_seeds": n,
                                 "scenario": "delta=0", "discordance": 2 * r,
                                 "reject_rate": pw, "coverage": cov})  # fmt: skip
                    t1, cov, _ = simulate(rng, n, cells, pc, r + margin, r, margin, a.sims, a.boot)
                    rows.append({"group": g, "margin": margin, "n_seeds": n,
                                 "scenario": "delta=-margin", "discordance": 2 * r + margin,
                                 "reject_rate": t1, "coverage": cov})  # fmt: skip
    df = pd.DataFrame(rows)
    run_level = []
    for net_sd in (0.0, 0.5, 1.0):
        _, cov, _ = simulate(
            rng, 300, 7, 0.85, 0.03, 0.03, 0.02, a.sims, a.boot, net_sd=net_sd, shared=0.0
        )
        run_level.append({"net_sd": net_sd, "coverage_of_mean_network_effect": cov})
    filt = []
    for harm in (0.0, 0.005, 0.01, 0.02, 0.03):
        passes = 0
        for _ in range(a.sims):
            c = rng.random((100, 3)) < 0.997
            r_mean = np.mean(
                [np.where(c & (rng.random((100, 3)) < harm / 0.997), False, c) for _ in range(3)],
                axis=0,
            )
            passes += int((r_mean - c).mean() >= -0.01)
        filt.append({"true_nominal_harm": harm, "p_pass_filter": passes / a.sims})

    def n_required(var_d: float, margin: float, z_b: float) -> int:
        return math.ceil((Z975 + z_b) ** 2 * var_d / margin**2)

    sizes = [
        {"group": g, "discordance": disc, "margin": m,
         "pairs_80pct": n_required(disc, m, Z80), "pairs_90pct": n_required(disc, m, Z90)}
        for g, disc in (("N", 0.002), ("N", 0.006), ("D", 0.05), ("M", 0.05), ("M", 0.10))
        for m in (0.01, 0.02)
    ]  # fmt: skip
    out = {
        "g4_inputs": inputs,
        "grid": df.to_dict(orient="records"),
        "run_level_variance": run_level,
        "selection_nominal_filter": filt,
        "normal_approx_pairs_single_network_independent": sizes,
        "settings": {"sims": a.sims, "boot": a.boot, "rng_seed": 20261901},
        "note": "synthetic simulation calibrated on existing G4 data; not a result",
    }
    p = Path(a.out)
    p.mkdir(parents=True, exist_ok=True)
    (p / "power_audit.json").write_text(json.dumps(out, indent=2) + "\n")
    df.to_csv(p / "power_audit_grid.csv", index=False)
    print(df.round(3).to_string(index=False))
    print(json.dumps({k: out[k] for k in ("run_level_variance", "selection_nominal_filter")}))


if __name__ == "__main__":
    main()
