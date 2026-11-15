# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/detector_plots.py
# (introduced in 934063d, last changed in 934063d). See PROVENANCE.md.
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_tradeoff(overall: pd.DataFrame, path: Path) -> None:
    envs = list(dict.fromkeys(overall["env_id"]))
    figure, axes = plt.subplots(1, len(envs), figsize=(5.2 * len(envs), 4.4), squeeze=False)
    for axis, env_id in zip(axes[0], envs, strict=True):
        frame = overall[overall["env_id"] == env_id]
        for persistence, group in frame.groupby("persistence"):
            axis.errorbar(
                group["false_positive_rate"],
                group["detection_rate"],
                xerr=[
                    group["false_positive_rate"] - group["fpr_hier_ci_low"],
                    group["fpr_hier_ci_high"] - group["false_positive_rate"],
                ],
                yerr=[
                    group["detection_rate"] - group["detection_hier_ci_low"],
                    group["detection_hier_ci_high"] - group["detection_rate"],
                ],
                fmt="o",
                capsize=2,
                label=f"persistence {persistence}",
            )
            for _, row in group.iterrows():
                axis.annotate(
                    f"k={row['window']}",
                    (row["false_positive_rate"], row["detection_rate"]),
                    fontsize=7,
                    xytext=(4, -8),
                    textcoords="offset points",
                )
        axis.plot([0, 1], [0, 1], color="gray", lw=0.6, ls=":")
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("false-positive rate (successful disturbed episodes warned)")
        axis.set_ylabel("detection rate (failed episodes warned)")
        axis.set_title(f"{env_id}: all disturbed conditions", fontsize=9)
        axis.legend(fontsize=7)
    figure.suptitle(
        "Progress-deficit detector, threshold 0 (bars: seed+episode bootstrap 95%)", fontsize=10
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_condition_heatmaps(by_condition: pd.DataFrame, path: Path) -> None:
    envs = list(dict.fromkeys(by_condition["env_id"]))
    metrics = [("detection_rate", "detection rate"), ("false_positive_rate", "false-positive rate")]
    detectors = list(dict.fromkeys(by_condition["detector"]))
    conditions = {
        e: list(dict.fromkeys(by_condition[by_condition["env_id"] == e]["condition"])) for e in envs
    }
    height = max(len(c) for c in conditions.values())
    figure, axes = plt.subplots(
        len(envs), 2, figsize=(13, 0.3 * height * len(envs) + 2.0), squeeze=False
    )
    for row_index, env_id in enumerate(envs):
        frame = by_condition[by_condition["env_id"] == env_id]
        for column, (metric, title) in enumerate(metrics):
            table = frame.pivot(index="condition", columns="detector", values=metric)
            table = table.reindex(index=conditions[env_id], columns=detectors).astype(float)
            axis = axes[row_index, column]
            image = axis.imshow(table.to_numpy(), vmin=0, vmax=1, cmap="viridis", aspect="auto")
            for (i, j), value in np.ndenumerate(table.to_numpy()):
                text = "-" if np.isnan(value) else f"{value:.2f}"
                axis.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=5.5,
                    color="white" if not np.isnan(value) and value < 0.6 else "black",
                )
            axis.set_xticks(range(len(detectors)))
            axis.set_xticklabels(detectors, rotation=60, fontsize=6)
            axis.set_yticks(range(len(table.index)))
            axis.set_yticklabels(table.index, fontsize=6)
            axis.set_title(f"{env_id}: {title} ('-' = no failures or no successes)", fontsize=8)
            figure.colorbar(image, ax=axis, fraction=0.03)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_lead_times(results: pd.DataFrame, path: Path) -> None:
    detected = results[results["failure"] & results["warned"]]
    envs = list(dict.fromkeys(results["env_id"]))
    detectors = list(dict.fromkeys(results["detector"]))
    figure, axes = plt.subplots(1, len(envs), figsize=(5.5 * len(envs), 3.8), squeeze=False)
    for axis, env_id in zip(axes[0], envs, strict=True):
        frame = detected[detected["env_id"] == env_id]
        data = [frame[frame["detector"] == d]["lead_time"].to_numpy(float) for d in detectors]
        axis.boxplot(data, showfliers=False)
        axis.set_xticks(range(1, len(detectors) + 1))
        axis.set_xticklabels(detectors, rotation=60, fontsize=7)
        axis.set_ylabel("lead time: steps left at warning")
        axis.set_ylim(0, 50)
        axis.set_title(f"{env_id}: detected failures", fontsize=9)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_detector_results(overall, by_condition, results, output_dir: Path) -> dict[str, Path]:
    paths = {
        "tradeoff_plot": output_dir / "detector_tradeoff.png",
        "heatmap_plot": output_dir / "detector_by_condition.png",
        "lead_time_plot": output_dir / "detector_lead_times.png",
    }
    plot_tradeoff(overall, paths["tradeoff_plot"])
    plot_condition_heatmaps(by_condition, paths["heatmap_plot"])
    plot_lead_times(results, paths["lead_time_plot"])
    return paths
