"""Phase J-3 learning-dynamics audit (docs/PHASE_J3_LEARNING_DYNAMICS_AUDIT.md).

Read-only. No training, no simulator episodes, no new seeds. Inputs:
- committed training logs ``results/phase_{i,j2}_training/*/progress.csv`` (what the trainer
  logged: critic/actor/alpha loss, alpha, entropy, mean Q, training success, and periodic
  noise-free, delay-free evaluation success on 50 episodes);
- saved final checkpoints: frozen SAC, Phase I and J-2 (read only, SHA-256 checked);
- archived Phase F1 frozen-SAC step records (development seeds 3,653,000-099): noise-free
  positions and issued commands under constant delays 0-4. Synthetic Gaussian position noise is
  added *offline* to those recorded positions; nothing is re-simulated, so closed-loop effects
  of noise are not represented.

    uv run python scripts/phase_j3_audit.py --out results/phase_j3_audit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from farc.base_policy.checkpoint import file_sha256
from farc.paths import resolve
from farc.retrain.phase_i import K0_STEP, ORIGINAL_OBS, Featurizer, _expand, frozen_checkpoint

EST = {"nominal_gain": 0.62, "window": 10, "max_lag": 4, "min_transitions": 4,
       "min_excitation": 0.000225, "min_r2": 0.8, "min_margin": 2.0}  # fmt: skip
PAIRS = {"IA": "J2A", "IC": "J2C"}
EXTRA = {"IA": 60, "IC": 18}
SIGMAS = (0.0, 0.0025, 0.005, 0.0075, 0.01)
NOISE_SEED = 20263300  # offline synthetic-noise stream (not an episode seed)


# --- 1. training logs ---------------------------------------------------------------------------


def _runs(arm_i: str, arm_j: str, seed: int):
    yield "I", arm_i, f"results/phase_i_training/phase_i_{arm_i}_seed{seed}"
    yield "J2", arm_j, f"results/phase_j2_training/phase_j2_{arm_j}_seed{seed}"


def logs() -> pd.DataFrame:
    rows = []
    for arm_i, arm_j in PAIRS.items():
        for seed in range(3):
            for phase, arm, d in _runs(arm_i, arm_j, seed):
                p = pd.read_csv(Path(d) / "progress.csv")
                first, last = p.head(5), p.tail(10)
                rows.append({
                    "pair": arm_i, "phase": phase, "arm": arm, "seed": seed,
                    "eval_first_20k": p["eval_success_rate"].iloc[0],
                    "eval_min": p["eval_success_rate"].min(),
                    "eval_min_step": int(p.loc[p["eval_success_rate"].idxmin(), "step"]),
                    "eval_mean_first5": first["eval_success_rate"].mean(),
                    "eval_mean_last10": last["eval_success_rate"].mean(),
                    "train_success_last10": last["train_success"].mean(),
                    "critic_loss_first5": first["critic_loss"].mean(),
                    "critic_loss_last10": last["critic_loss"].mean(),
                    "critic_loss_max": p["critic_loss"].max(),
                    "q_mean_last10": last["q_mean"].mean(),
                    "actor_loss_last10": last["actor_loss"].mean(),
                    "alpha_last10": last["alpha"].mean(),
                    "entropy_last10": last["entropy"].mean(),
                })  # fmt: skip
    return pd.DataFrame(rows)


def curves() -> pd.DataFrame:
    out = []
    for arm_i, arm_j in PAIRS.items():
        for seed in range(3):
            for phase, arm, d in _runs(arm_i, arm_j, seed):
                p = pd.read_csv(Path(d) / "progress.csv")
                p = p[["step", "eval_success_rate", "train_success", "critic_loss", "q_mean",
                       "alpha", "entropy"]].assign(phase=phase, arm=arm, seed=seed)  # fmt: skip
                out.append(p)
    return pd.concat(out)


# --- 2. parameter drift -------------------------------------------------------------------------


def _ckpt(kind: str, arm: str, seed: int) -> dict:
    manifest = json.loads(Path(f"src/farc/retrain/checkpoints_phase_{kind}.json").read_text())
    cid = f"phase_{kind}_{arm}_seed{seed}"
    entry = manifest["checkpoints"][cid]
    path = resolve(entry["path_template"].replace("{id}", cid))
    if file_sha256(path) != entry["sha256"]:
        raise RuntimeError(f"{cid}: checkpoint hash mismatch")
    return torch.load(path, weights_only=False, map_location="cpu")["agent"]


def drift() -> pd.DataFrame:
    rows = []
    for arm_i, arm_j in PAIRS.items():
        extra = EXTRA[arm_i]
        for seed in range(3):
            frozen = frozen_checkpoint(seed)["agent"]
            for phase, arm, kind in (("I", arm_i, "i"), ("J2", arm_j, "j2")):
                trained = _ckpt(kind, arm, seed)
                for net in ("actor", "critic"):
                    for name, w in trained[net].items():
                        w = w.float()
                        w0 = _expand(frozen[net][name], w.shape, extra).float()
                        row = {"pair": arm_i, "phase": phase, "arm": arm, "seed": seed,
                               "net": net, "param": name,
                               "rel_change": float((w - w0).norm() / w0.norm())}  # fmt: skip
                        if name.endswith("0.weight") and w.dim() == 2:
                            orig = slice(0, ORIGINAL_OBS)
                            new = slice(ORIGINAL_OBS, ORIGINAL_OBS + extra)
                            rest = slice(ORIGINAL_OBS + extra, w.shape[1])
                            ncols = lambda m, s: float(m[:, s].norm(dim=0).mean())  # noqa: E731
                            row.update({
                                "orig_cols_rel_change": float((w[:, orig] - w0[:, orig]).norm()
                                                              / w0[:, orig].norm()),
                                "goal_action_cols_rel_change": float(
                                    (w[:, rest] - w0[:, rest]).norm()
                                                                     / w0[:, rest].norm()),
                                "orig_col_norm_mean": ncols(w, orig),
                                "new_col_norm_mean": ncols(w, new),
                                "new_over_orig_col_norm": ncols(w, new) / ncols(w, orig),
                            })  # fmt: skip
                            if arm_i == "IA":  # commands vs displacement sub-blocks
                                row["new_cmd_col_norm"] = ncols(w, slice(25, 55))
                                row["new_disp_col_norm"] = ncols(w, slice(55, 85))
                            else:
                                row["new_cmd_col_norm"] = ncols(w, slice(25, 37))
                                row["new_onehot_col_norm"] = ncols(w, slice(37, 42))
                                row["new_reliable_col_norm"] = ncols(w, slice(42, 43))
                        rows.append(row)
    return pd.DataFrame(rows)


# --- 3. delay information under noise (offline, archived trajectories) --------------------------


def trajectories() -> pd.DataFrame:
    cols = (["configuration", "policy_seed", "condition", "episode_seed", "t"]
            + [f"true{i}" for i in range(3)] + [f"cmd{i}" for i in range(3)])  # fmt: skip
    s = pd.read_csv(resolve("$FARC_DATA_DIR/experiments/phase_f1_eval/steps.csv.gz"),
                    usecols=cols)  # fmt: skip
    conds = ("C0_undisturbed", "C1_delay1", "C2_delay2", "C3_delay3", "C4_delay4")
    s = s[(s["configuration"] == "base_only") & s["condition"].isin(conds)]
    return s.sort_values(["policy_seed", "condition", "episode_seed", "t"])


def _episodes(s: pd.DataFrame):
    for (seed, cond, ep_seed), ep in s.groupby(["policy_seed", "condition", "episode_seed"],
                                                sort=False):  # fmt: skip
        g = ep[[f"true{i}" for i in range(3)]].to_numpy(float)
        cmd = np.c_[ep[[f"cmd{i}" for i in range(3)]].to_numpy(float), np.zeros(len(ep))]
        yield int(cond[1]), seed, ep_seed, g, cmd


def lag_identification(s: pd.DataFrame) -> pd.DataFrame:
    """Best-case (oracle-structure) lag identification from a 10-step window of the I-A inputs:
    for lag L in 0..5, least-squares gain fit of displacement_t on command_{t-1-L}; pick the L
    with the smallest residual. Accuracy is scored against the lag chosen on noise-free data for
    d = 0 plus d (the offset is calibrated, not assumed). Also reports the per-entry SNR."""
    rows = []
    rng = np.random.default_rng(NOISE_SEED)
    eps = list(_episodes(s))
    picks: dict[tuple[float, int], list[int]] = {}
    snr: dict[float, list[float]] = {sg: [] for sg in SIGMAS}
    for d, _, _, g, cmd in eps:
        for sg in SIGMAS:
            gn = g + sg * rng.standard_normal(g.shape)
            disp = np.diff(gn, axis=0) / K0_STEP  # disp[t-1] = (g_t - g_{t-1}) / K0
            clean = np.diff(g, axis=0) / K0_STEP
            if sg > 0:
                snr[sg].append(float(clean.std() / (disp - clean).std()))
            c = cmd[:, :3]
            for t in range(16, len(g)):
                y = disp[t - 10 : t]  # 10 most recent displacements
                best, res = 0, np.inf
                for lag in range(6):
                    idx = np.arange(t - 10, t) - lag
                    x = c[idx]
                    k = float((x * y).sum() / max((x * x).sum(), 1e-12))
                    r = float(((y - k * x) ** 2).sum())
                    if r < res:
                        best, res = lag, r
                picks.setdefault((sg, d), []).append(best)
    base = int(np.bincount(picks[(0.0, 0)]).argmax())
    for (sg, d), v in sorted(picks.items()):
        v = np.array(v)
        rows.append({"sigma_m": sg, "true_delay": d,
                     "accuracy": float((v == base + d).mean()),
                     "within_1": float((np.abs(v - (base + d)) <= 1).mean()),
                     "windows": len(v),
                     "displacement_snr": float(np.mean(snr[sg])) if sg > 0 else np.inf,
                     })  # fmt: skip
    return pd.DataFrame(rows)


def estimator_under_noise(s: pd.DataFrame) -> pd.DataFrame:
    """I-C's features (deployed Featurizer) from recorded positions plus synthetic noise."""
    rng = np.random.default_rng(NOISE_SEED + 1)
    rows = []
    for d, _, _, g, cmd in _episodes(s):
        for sg in SIGMAS:
            obs = np.zeros((len(g), 25))
            obs[:, 0:3] = g + sg * rng.standard_normal(g.shape)
            f = Featurizer("IC", EST)
            dh, rel = [], []
            for t in range(len(obs)):
                f.observe(obs[t])
                feat = f.features()
                dh.append(int(feat[12:17].argmax()))
                rel.append(feat[17])
                f.issued(cmd[t])
            dh, rel = np.array(dh), np.array(rel, bool)
            rows.append({"sigma_m": sg, "true_delay": d, "reliable_share": rel.mean(),
                         "latched_correct": (dh == d).mean(),
                         "latched_zero": (dh == 0).mean()})  # fmt: skip
    return pd.DataFrame(rows).groupby(["sigma_m", "true_delay"]).mean().reset_index()


def analytic() -> pd.DataFrame:
    """Noise on one I-A displacement entry: sqrt(2) sigma / K0_STEP; adjacent entries -0.5."""
    rows = [{"sigma_m": sg, "displacement_noise_std": np.sqrt(2) * sg / K0_STEP,
             "clip_prob_at_5": float(2 * (1 - _phi(5 / (np.sqrt(2) * sg / K0_STEP))))
             if sg > 0 else 0.0} for sg in SIGMAS]  # fmt: skip
    # J-2 training mixture sigma ~ U[0, 1 cm]: E[sigma^2] = (1 cm)^2 / 3
    rows.append({"sigma_m": "U[0,0.01] mixture",
                 "displacement_noise_std": np.sqrt(2 * 0.01**2 / 3) / K0_STEP,
                 "clip_prob_at_5": np.nan})  # fmt: skip
    return pd.DataFrame(rows)


def _phi(x: float) -> float:
    from math import erf, sqrt

    return 0.5 * (1 + erf(x / sqrt(2)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/phase_j3_audit")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    logs().to_csv(out / "training_log_summary.csv", index=False)
    curves().to_csv(out / "training_curves.csv", index=False)
    drift().to_csv(out / "parameter_drift.csv", index=False)
    analytic().to_csv(out / "displacement_noise_analytic.csv", index=False)
    s = trajectories()
    lag_identification(s).to_csv(out / "lag_identification.csv", index=False)
    estimator_under_noise(s).to_csv(out / "estimator_under_noise.csv", index=False)
    print("written", sorted(x.name for x in out.iterdir()))


if __name__ == "__main__":
    main()
