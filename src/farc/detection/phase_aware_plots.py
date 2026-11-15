# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/phase_plots.py
# (introduced in 98c903e, last changed in 98c903e). See PROVENANCE.md.
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _errorbars(axis, frame, label, color, marker):
    axis.errorbar(
        frame["false_positive_rate"],
        frame["detection_rate"],
        xerr=[
            frame["false_positive_rate"] - frame["fpr_hier_ci_low"],
            frame["fpr_hier_ci_high"] - frame["false_positive_rate"],
        ],
        yerr=[
            frame["detection_rate"] - frame["detection_hier_ci_low"],
            frame["detection_hier_ci_high"] - frame["detection_rate"],
        ],
        fmt=marker,
        color=color,
        capsize=2,
        ms=6,
        label=label,
    )


def plot_tradeoff(overall: pd.DataFrame, path: Path) -> None:
    pairs = list(dict.fromkeys(zip(overall["window"], overall["persistence"], strict=True)))
    figure, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4.2), squeeze=False)
    styles = {
        (1, "off"): ("#9ecae1", "o"), (1, 3): ("#3182bd", "o"),
        (3, "off"): ("#fdae6b", "D"), (3, 3): ("#e6550d", "D"),
    }  # fmt: skip
    for axis, (window, persistence) in zip(axes[0], pairs, strict=True):
        frame = overall[(overall["window"] == window) & (overall["persistence"] == persistence)]
        base = frame[frame["family"] == "distance_only"]
        _errorbars(axis, base, "distance-only", "black", "s")
        phase = frame[frame["family"] == "phase_aware"]
        for (hysteresis, departure), sub in phase.groupby(
            ["hysteresis", "departure_persistence"], dropna=False
        ):
            m = "off" if pd.isna(departure) else int(departure)
            color, marker = styles.get((int(hysteresis), m), ("gray", "^"))
            _errorbars(axis, sub, f"phase h={int(hysteresis)} departure m={m}", color, marker)
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
        axis.plot([0, 1], [0, 1], color="gray", lw=0.6, ls=":")
        axis.set_title(f"k={window}, p={persistence}", fontsize=9)
        axis.set_xlabel("false-positive rate (disturbed successes)", fontsize=8)
        axis.set_ylabel("detection rate (failures)", fontsize=8)
        axis.legend(fontsize=6, loc="lower right")
    figure.suptitle(
        "FetchPush: distance-only vs phase-aware detector on matched episodes "
        "(bars: seed+episode bootstrap 95%)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_paired(paired: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 0.32 * len(paired) + 1.8), sharey=True)
    positions = np.arange(len(paired))[::-1]
    for axis, name, title in (
        (axes[0], "detection", "detection rate: phase-aware minus distance-only"),
        (axes[1], "fpr", "false-positive rate: phase-aware minus distance-only"),
    ):
        value = paired[f"{name}_difference"].to_numpy(float)
        low = paired[f"{name}_difference_hier_ci_low"].to_numpy(float)
        high = paired[f"{name}_difference_hier_ci_high"].to_numpy(float)
        axis.errorbar(value, positions, xerr=[value - low, high - value], fmt="o", capsize=2)
        axis.axvline(0, color="gray", lw=0.8)
        axis.set_title(title, fontsize=9)
        axis.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels(paired["detector"], fontsize=7)
    figure.suptitle(
        "Paired differences on the same disturbed episodes (95% seed+episode "
        "bootstrap; not a significance test)",
        fontsize=10,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_timing(results: pd.DataFrame, path: Path) -> None:
    disturbed = results[(results["disturbance_type"] != "none") & results["warned"]]
    detectors = list(dict.fromkeys(disturbed["detector"]))
    figure, axes = plt.subplots(2, 1, figsize=(max(8, 0.45 * len(detectors)), 7), sharex=True)
    for axis, failed, title in (
        (axes[0], True, "detected failures"),
        (axes[1], False, "false alarms (successful episodes)"),
    ):
        part = disturbed[disturbed["failure"] == failed]
        data = [
            part[part["detector"] == d]["warning_minus_first_move"].dropna().to_numpy(float)
            for d in detectors
        ]
        axis.boxplot(data, showfliers=False)
        axis.axhline(0, color="red", lw=0.8)
        axis.set_ylabel("warning state − first object movement", fontsize=8)
        axis.set_title(
            f"FetchPush disturbed episodes: {title} (below 0 = before the object moved)", fontsize=9
        )
    axes[1].set_xticks(range(1, len(detectors) + 1))
    axes[1].set_xticklabels(detectors, rotation=70, fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_phase_results(overall, paired, results, output_dir: Path) -> dict[str, Path]:
    paths = {
        "tradeoff_plot": output_dir / "phase_detector_tradeoff.png",
        "paired_plot": output_dir / "phase_detector_paired.png",
        "timing_plot": output_dir / "phase_detector_timing.png",
    }
    plot_tradeoff(overall, paths["tradeoff_plot"])
    plot_paired(paired, paths["paired_plot"])
    plot_timing(results, paths["timing_plot"])
    return paths
