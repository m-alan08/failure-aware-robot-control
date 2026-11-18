"""Phase J diagnostic audit (docs/PHASE_J_DIAGNOSTIC_REPORT.md). Read-only; no simulation.

Inputs, all archived and development-only (no held-out data is read):
- Phase I Stage-2 raw episodes (``phase_i_evaluation``, seeds 3,702,000-299): per-seed gate.
- Phase F1 evaluation step records (``phase_f1_eval``, seeds 3,653,000-099): real noise-free
  (``true*``) and noisy (``view*``) observations and issued commands (``cmd*``) of frozen-SAC
  episodes under no noise (C0), 1 cm position noise (C8) and delay 3 + 0.5 cm noise (C6).
- Phase E2 step records (``phase_e2_delay``, seeds 3,641,000-099): the estimator's own logged
  outputs in the simulator, for the estimator-reliability analysis.
- No policy is run. An open-loop action-sensitivity probe was considered and dropped: the
  archived F1 step records do not contain the desired goal, so the policy's input could not be
  reconstructed without inventing it.

Features are computed with the deployed ``farc.retrain.phase_i.Featurizer`` code.

    uv run python scripts/phase_j_diagnostics.py --out results/phase_j_diagnostics
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from farc.delaychange.mechanism_analysis import two
from farc.paths import resolve
from farc.retrain.phase_i import K0_STEP, Featurizer

EST = {"nominal_gain": 0.62, "window": 10, "max_lag": 4, "min_transitions": 4,
       "min_excitation": 0.000225, "min_r2": 0.8, "min_margin": 2.0}  # fmt: skip
SIGMA = 0.01  # C8 / H_noise0.01 / E_C8: position noise standard deviation (metres)
NOISE_CONDS = ("H_noise0.01", "E_C8_noise0.01")
NOMINAL_CONDS = ("H_undisturbed", "E_C0_undisturbed")


def per_seed_gate(n: int = 10_000, s: int = 20262000) -> pd.DataFrame:
    e = pd.read_csv(resolve("$FARC_DATA_DIR/experiments/phase_i_evaluation/episodes.csv.gz"))
    rows = []
    for scope, conds in (("noise", NOISE_CONDS), ("nominal", NOMINAL_CONDS)):
        sub = e[e["condition"].isin(conds)]
        for k in (0, 1, 2, "pooled"):
            frame = sub if k == "pooled" else sub[sub["policy_seed"] == k]
            for arm in ("phase_i_IA", "phase_i_IC", "retrained_R2"):
                r = two(frame, arm, "base_only", -0.02, n, s)
                succ = frame[frame["configuration"] == arm]["success"].mean()
                base = frame[frame["configuration"] == "base_only"]["success"].mean()
                rows.append({"scope": scope, "seed": k, "configuration": arm,
                             "success": succ, "frozen_sac_success": base, **r})  # fmt: skip
    return pd.DataFrame(rows)


def load_trajectories(conditions: tuple[str, ...]) -> pd.DataFrame:
    cols = (["configuration", "policy_seed", "condition", "episode_seed", "t", "delay_true"]
            + [f"true{i}" for i in range(25)] + [f"view{i}" for i in range(25)]
            + [f"cmd{i}" for i in range(3)])  # fmt: skip
    s = pd.read_csv(resolve("$FARC_DATA_DIR/experiments/phase_f1_eval/steps.csv.gz"),
                    usecols=cols)  # fmt: skip
    s = s[(s["configuration"] == "base_only") & s["condition"].isin(conditions)]
    return s.sort_values(["policy_seed", "condition", "episode_seed", "t"])


def episodes(s: pd.DataFrame):
    for key, ep in s.groupby(["policy_seed", "condition", "episode_seed"], sort=False):
        true = ep[[f"true{i}" for i in range(25)]].to_numpy(float)
        view = ep[[f"view{i}" for i in range(25)]].to_numpy(float)
        cmd = np.c_[ep[[f"cmd{i}" for i in range(3)]].to_numpy(float), np.zeros(len(ep))]
        yield key, true, view, cmd, ep["t"].to_numpy(int)


def featurize(arm: str, obs: np.ndarray, cmd: np.ndarray) -> np.ndarray:
    """Deployed Featurizer on an archived trajectory: features at every step."""
    f = Featurizer(arm, EST)
    out = []
    for t in range(len(obs)):
        f.observe(obs[t])
        out.append(f.features())
        f.issued(cmd[t])
    return np.array(out)


def noise_model_check(s: pd.DataFrame) -> dict[str, Any]:
    n_g, lag1 = [], []
    for _, true, view, _, _ in episodes(s):
        e = view[:, 0:3] - true[:, 0:3]
        n_g.append(e)
        if len(e) > 2:
            lag1.append([np.corrcoef(e[:-1, j], e[1:, j])[0, 1] for j in range(3)])
    n = np.concatenate(n_g)
    rel = (
        s[[f"view{i}" for i in range(6, 9)]].to_numpy()
        - s[[f"true{i}" for i in range(6, 9)]].to_numpy()
    )
    obj = (
        s[[f"view{i}" for i in range(3, 6)]].to_numpy()
        - s[[f"true{i}" for i in range(3, 6)]].to_numpy()
    )
    vel = (
        s[[f"view{i}" for i in range(20, 23)]].to_numpy()
        - s[[f"true{i}" for i in range(20, 23)]].to_numpy()
    )
    return {"gripper_noise_std_per_axis": n.std(0).round(5).tolist(),
            "gripper_noise_mean_per_axis": n.mean(0).round(6).tolist(),
            "lag1_autocorrelation_per_axis": np.nanmean(lag1, 0).round(4).tolist(),
            "relative_position_noise_equals_object_minus_gripper":
                bool(np.allclose(rel, obj - n, atol=1e-9)),
            "gripper_velocity_noise_max_abs": float(np.abs(vel).max()),
            "steps": int(len(n))}  # fmt: skip


def feature_noise(s: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """I-A displacement entries (scaled by 1/(K0*0.05)) from noisy vs noise-free positions."""
    noisy, clean, tt = [], [], []
    for _, true, view, cmd, t in episodes(s):
        noisy.append(featurize("IA", view, cmd)[:, 30:60])
        clean.append(featurize("IA", true, cmd)[:, 30:60])
        tt.append(t)
    fn, fc, t = np.concatenate(noisy), np.concatenate(clean), np.concatenate(tt)
    err = fn - fc
    rows = []
    for lag in range(10):
        cols = slice(3 * lag, 3 * lag + 3)
        valid = t > lag  # entries that are not zero-padded
        e, c = err[valid, cols], fc[valid, cols]
        nxt = slice(3 * (lag + 1), 3 * (lag + 1) + 3) if lag < 9 else None
        if nxt is None:
            corr = np.nan
        else:
            later = err[t > lag + 1]
            corr = np.mean([np.corrcoef(later[:, cols][:, j], later[:, nxt][:, j])[0, 1]
                            for j in range(3)])  # fmt: skip
        rows.append({"displacement_lag": lag + 1, "noise_std_measured": float(e.std()),
                     "noise_std_analytic": float(np.sqrt(2) * SIGMA / K0_STEP),
                     "clean_signal_std": float(c.std()),
                     "clean_signal_median_abs": float(np.median(np.abs(c))),
                     "snr_std_ratio": float(c.std() / e.std()),
                     "corr_with_next_lag_noise": float(corr),
                     "clip_fraction_noisy": float(np.mean(np.abs(fn[valid, cols]) > 5)),
                     "clip_fraction_clean": float(np.mean(np.abs(c) > 5))})  # fmt: skip
    # signal by episode stage (first displacement entry)
    stage = pd.DataFrame({"t": t, "clean_abs": np.abs(fc[:, 0:3]).mean(1),
                          "noise_abs": np.abs(err[:, 0:3]).mean(1)})  # fmt: skip
    stage["stage"] = pd.cut(stage["t"], [-1, 9, 19, 29, 39, 49],
                            labels=["0-9", "10-19", "20-29", "30-39", "40-49"])  # fmt: skip
    by_stage = (
        stage.groupby("stage", observed=True)[["clean_abs", "noise_abs"]].mean().reset_index()
    )
    return pd.DataFrame(rows), by_stage


def filtered_alternatives(s: pd.DataFrame) -> pd.DataFrame:
    """Diagnostic only: noise and fidelity of alternative displacement representations,
    computed on the same archived trajectories. No policy uses these."""
    reps: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {k: [] for k in (
        "A per-step difference (deployed)", "B 3-step moving average of differences",
        "B exponential filter (alpha 0.3) of differences", "C 2-step difference / 2",
        "C 3-step difference / 3", "C 5-step difference / 5")}  # fmt: skip
    for _, true, view, _, _ in episodes(s):
        for name in reps:
            reps[name].append((_rep(name, view[:, 0:3]), _rep(name, true[:, 0:3])))
    ref = []  # the true per-step displacement, the quantity the deployed feature estimates
    for _, true, _, _, _ in episodes(s):
        ref.append(_rep("A per-step difference (deployed)", true[:, 0:3]))
    r = np.concatenate(ref)
    rows = []
    for name, pairs in reps.items():
        noisy = np.concatenate([p[0] for p in pairs])
        clean = np.concatenate([p[1] for p in pairs])
        ok = ~np.isnan(noisy).any(1) & ~np.isnan(r).any(1)
        rows.append({"representation": name,
                     "noise_std": float((noisy[ok] - clean[ok]).std()),
                     "corr_noisy_with_true_step_displacement": float(np.mean(
                         [np.corrcoef(noisy[ok][:, j], r[ok][:, j])[0, 1] for j in range(3)])),
                     "corr_clean_with_true_step_displacement": float(np.mean(
                         [np.corrcoef(clean[ok][:, j], r[ok][:, j])[0, 1]
                          for j in range(3)]))})  # fmt: skip
    return pd.DataFrame(rows)


def _rep(name: str, g: np.ndarray) -> np.ndarray:
    d = np.full_like(g, np.nan)
    d[1:] = (g[1:] - g[:-1]) / K0_STEP
    if name.startswith("A "):
        return d
    if name.startswith("B 3-step"):
        out = np.full_like(g, np.nan)
        for t in range(3, len(g)):
            out[t] = d[t - 2 : t + 1].mean(0)
        return out
    if name.startswith("B exponential"):
        out = np.full_like(g, np.nan)
        acc = None
        for t in range(1, len(g)):
            acc = d[t] if acc is None else 0.3 * d[t] + 0.7 * acc
            out[t] = acc
        return out
    m = int(name.split()[1].split("-")[0])
    out = np.full_like(g, np.nan)
    out[m:] = (g[m:] - g[:-m]) / (m * K0_STEP)
    return out


def estimator_reliability() -> pd.DataFrame:
    """The estimator's own logged outputs in Phase E2 (simulator, frozen-SAC episodes)."""
    s = pd.read_csv(resolve("$FARC_DATA_DIR/experiments/phase_e2_delay/steps.csv.gz"),
                    usecols=["configuration", "condition", "policy_seed", "episode_seed", "t",
                             "delay_est", "delay_reliable", "delay_true"])  # fmt: skip
    s = s[s["configuration"] == "base_only"]
    rows = []
    for cond, sub in s.groupby("condition"):
        rel = sub["delay_reliable"].astype(bool)
        rows.append({"condition": cond, "steps": len(sub),
                     "reliable_share": float(rel.mean()),
                     "reliable_and_correct_share_of_reliable": float(
                         (sub.loc[rel, "delay_est"] == sub.loc[rel, "delay_true"]).mean())
                     if rel.any() else np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def ic_features(s: pd.DataFrame) -> pd.DataFrame:
    """I-C's estimate features (one-hot of latched d-hat, reliability flag) reconstructed
    offline with the deployed Featurizer, from noisy views vs noise-free positions."""
    rows = []
    for (_, cond, _), true, view, cmd, _ in episodes(s):
        for label, obs in (("noisy view", view), ("noise-free", true)):
            f = featurize("IC", obs, cmd)
            dhat = f[:, 12:17].argmax(1)
            rows.append({"condition": cond, "input": label, "reliable_share": f[:, 17].mean(),
                         "latched_nonzero_share": (dhat > 0).mean(),
                         "latched_dhat_mode": int(np.bincount(dhat).argmax())})  # fmt: skip
    return (pd.DataFrame(rows).groupby(["condition", "input"])
            [["reliable_share", "latched_nonzero_share"]].mean().reset_index())  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/phase_j_diagnostics")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    per_seed_gate().to_csv(out / "per_seed_gate.csv", index=False)
    c8 = load_trajectories(("C8_posnoise0.01",))
    (out / "noise_model_check.json").write_text(json.dumps(noise_model_check(c8), indent=2) + "\n")
    fn, stage = feature_noise(c8)
    fn.to_csv(out / "ia_feature_noise.csv", index=False)
    stage.to_csv(out / "ia_feature_noise_by_stage.csv", index=False)
    filtered_alternatives(c8).to_csv(out / "filtered_alternatives.csv", index=False)
    estimator_reliability().to_csv(out / "estimator_reliability_e2.csv", index=False)
    mixed = load_trajectories(("C0_undisturbed", "C8_posnoise0.01", "C3_delay3",
                               "C6_delay3_posnoise0.005"))  # fmt: skip
    ic_features(mixed).to_csv(out / "ic_features.csv", index=False)
    print("written to", out)


if __name__ == "__main__":
    main()
