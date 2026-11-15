# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/plots.py
# (introduced in 0b6b61d, last changed in fad4381). See PROVENANCE.md.
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAMILY_COLORS = {
    "action_scale": "#1f77b4",
    "action_scale, transient": "#6baed6",
    "action_noise": "#d62728",
    "action_noise, transient": "#fc9272",
    "action_delay": "#9467bd",
    "goal_shift": "#2ca02c",
    "observation_noise (positions)": "#ff7f0e",
    "observation_noise (velocities)": "#8c564b",
    "observation_noise (full)": "#7f7f7f",
}


def family(row: pd.Series) -> str:
    kind = str(row["disturbance_type"])
    if kind == "observation_noise":
        return f"observation_noise ({row['mask']})"
    if row["regime"] == "transient":
        return f"{kind}, transient"
    return kind


def _label(row: pd.Series) -> str:
    severity = f"{row['severity']:g}"
    if row["disturbance_type"] == "goal_shift":
        return f"goal shift >= {severity} m at step {int(row['onset'])}"
    if row["regime"] == "transient":
        return f"{row['disturbance_type']} {severity}, steps {int(row['onset'])}-" + str(
            int(row["onset"]) + int(row["duration"]) - 1
        )
    unit = {
        "positions": " m",
        "velocities": " m/s",
        "full": " raw",
    }.get(str(row["mask"]), "")
    if row["disturbance_type"] == "observation_noise":
        return f"obs noise {row['mask']} {severity}{unit}"
    if row["disturbance_type"] == "action_delay":
        return f"action delay {severity} steps"
    return f"{row['disturbance_type']} {severity}"


def _dot_panel(axis, frame, per_seed, value, low, high, seed_value, baseline, title, xlabel):
    frame = frame.reset_index(drop=True)
    positions = np.arange(len(frame))[::-1]
    for y, (_, row) in zip(positions, frame.iterrows(), strict=True):
        color = FAMILY_COLORS.get(family(row), "black")
        if pd.notna(row.get(value)):
            axis.errorbar(
                row[value],
                y,
                xerr=[[row[value] - row[low]], [row[high] - row[value]]],
                fmt="o",
                color=color,
                capsize=2,
                ms=5,
            )
        seeds = per_seed[
            (per_seed["env_id"] == row["env_id"]) & (per_seed["condition"] == row["condition"])
        ]
        if seed_value in seeds:
            axis.scatter(seeds[seed_value], [y] * len(seeds), marker="|", color="k", s=40, lw=1)
    if baseline is not None:
        axis.axvline(baseline, color="gray", ls="--", lw=1)
    axis.set_yticks(positions)
    axis.set_yticklabels([_label(r) for _, r in frame.iterrows()], fontsize=7)
    axis.set_xlim(-0.03, 1.03)
    axis.set_title(title, fontsize=9)
    axis.set_xlabel(xlabel, fontsize=8)
    axis.grid(axis="x", alpha=0.3)


def plot_success_overview(pooled, per_seed, path: Path, stage: str) -> None:
    envs = list(dict.fromkeys(pooled["env_id"]))
    counts = [int((pooled["env_id"] == e).sum()) for e in envs]
    figure, axes = plt.subplots(
        1, len(envs), figsize=(6.0 * len(envs), 0.32 * max(counts) + 1.6), squeeze=False
    )
    for axis, env_id in zip(axes[0], envs, strict=True):
        frame = pooled[(pooled["env_id"] == env_id) & (pooled["disturbance_type"] != "none")]
        base = pooled[(pooled["env_id"] == env_id) & (pooled["disturbance_type"] == "none")]
        baseline = float(base["success_rate"].iloc[0]) if len(base) else None
        _dot_panel(
            axis,
            frame,
            per_seed,
            "success_rate",
            "success_hier_ci_low",
            "success_hier_ci_high",
            "success_rate",
            baseline,
            f"{env_id}: final-step success",
            "success rate (dots: pooled; bars: seed+episode bootstrap 95%; ticks: per seed)",
        )
    figure.suptitle(f"Success under each disturbance ({stage}; dashed = undisturbed)", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_recovery_overview(pooled, per_seed, path: Path, stage: str) -> None:
    pooled = pooled[pooled["recovery_eligible"].fillna(0) > 0]
    envs = list(dict.fromkeys(pooled["env_id"]))
    if not envs:
        return
    counts = [int((pooled["env_id"] == e).sum()) for e in envs]
    figure, axes = plt.subplots(
        1, len(envs), figsize=(6.0 * len(envs), 0.32 * max(counts) + 1.6), squeeze=False
    )
    for axis, env_id in zip(axes[0], envs, strict=True):
        frame = pooled[pooled["env_id"] == env_id].reset_index(drop=True)
        _dot_panel(
            axis,
            frame,
            per_seed,
            "recovery_rate",
            "recovery_hier_ci_low",
            "recovery_hier_ci_high",
            "recovery_rate",
            None,
            f"{env_id}: success after the reference state, among episodes failing there",
            "probability (permanent: from onset; transient: after the window ends)",
        )
        positions = np.arange(len(frame))[::-1]
        for y, (_, row) in zip(positions, frame.iterrows(), strict=True):
            axis.text(1.02, y, f"n={int(row['recovery_eligible'])}", fontsize=6, va="center")
    figure.suptitle(f"Recovery ({stage})", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_distance_trajectories(pooled, curves, path: Path, stage: str) -> None:
    info = pooled.set_index(["env_id", "condition"])
    mean_curves = curves.groupby(["env_id", "condition", "t"])["mean"].mean().sort_index()
    envs = list(dict.fromkeys(pooled["env_id"]))
    families = [f for f in FAMILY_COLORS if f in set(pooled.apply(family, axis=1))]
    figure, axes = plt.subplots(
        len(envs),
        len(families),
        figsize=(3.0 * len(families), 2.8 * len(envs)),
        squeeze=False,
        sharey="row",
    )
    for row_index, env_id in enumerate(envs):
        env_info = info.loc[env_id]
        baseline_names = env_info[env_info["disturbance_type"] == "none"].index
        for column, name in enumerate(families):
            axis = axes[row_index, column]
            for base in baseline_names:
                curve = mean_curves.loc[(env_id, base)]
                axis.plot(curve.index, curve.to_numpy() * 100, color="gray", ls="--", lw=1)
            members = [c for c in env_info.index if family(env_info.loc[c].rename(None)) == name]
            shades = plt.cm.viridis(np.linspace(0.1, 0.85, max(len(members), 1)))
            for shade, condition in zip(shades, members, strict=False):
                row = env_info.loc[condition]
                curve = mean_curves.loc[(env_id, condition)]
                axis.plot(
                    curve.index,
                    curve.to_numpy() * 100,
                    color=shade,
                    lw=1.2,
                    label=_label(pd.Series({**row.to_dict(), "condition": condition})),
                )
                if row["regime"] == "transient":
                    axis.axvspan(
                        row["onset"], row["onset"] + row["duration"], color="k", alpha=0.05
                    )
                elif pd.notna(row["onset"]) and row["onset"] > 0:
                    axis.axvline(row["onset"], color="k", lw=0.5, alpha=0.4)
            axis.axhline(5.0, color="red", lw=0.8, alpha=0.6)
            axis.set_title(f"{env_id}\n{name}", fontsize=8)
            axis.set_xlabel("state t", fontsize=8)
            if column == 0:
                axis.set_ylabel("mean distance to active goal (cm)", fontsize=8)
            axis.legend(fontsize=5)
    figure.suptitle(
        f"Distance to the active goal ({stage}; mean over seeds and episodes; gray dashed = "
        "undisturbed; red = 5 cm success radius)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_all(pooled, per_seed, curves, output_dir: Path, stage: str) -> dict[str, Path]:
    paths = {
        "success_plot": output_dir / "success_overview.png",
        "recovery_plot": output_dir / "recovery_overview.png",
        "distance_plot": output_dir / "distance_trajectories.png",
    }
    plot_success_overview(pooled, per_seed, paths["success_plot"], stage)
    plot_recovery_overview(pooled, per_seed, paths["recovery_plot"], stage)
    plot_distance_trajectories(pooled, curves, paths["distance_plot"], stage)
    return {key: path for key, path in paths.items() if path.exists()}
