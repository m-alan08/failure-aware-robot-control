# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/sweep_plots.py
# (introduced in 91e44be, last changed in 91e44be). See PROVENANCE.md.
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FAMILY_STYLE = {
    "distance_only": ("black", "-"),
    "phase_aware_m_off": ("#e6550d", "-"),
    "phase_aware_all": ("#3182bd", "--"),
}


def _detector_color(name: str) -> tuple[str, str]:
    if name.startswith("distance"):
        return "#636363", "-"
    if name.endswith("moff"):
        return "#fd8d3c", "-"
    return "#6baed6", ":"


def plot_curves(grid: pd.DataFrame, leads, path: Path, max_fpr: float = 0.5) -> None:
    figure, axes = plt.subplots(1, len(leads), figsize=(5.0 * len(leads), 4.4), squeeze=False)
    for axis, lead in zip(axes[0], leads, strict=True):
        frame = grid[(grid["min_lead"] == lead) & (grid["fpr"] <= max_fpr)]
        for name, group in frame.groupby("detector"):
            color, style = _detector_color(str(name))
            axis.plot(group["fpr"], group["detection"], color=color, ls=style, lw=0.9, alpha=0.8)
        envelopes = {
            "distance_only": frame[frame["family"] == "distance_only"],
            "phase_aware_m_off": frame[frame["detector"].str.endswith("moff")],
        }
        for family, part in envelopes.items():
            best = part.groupby("fpr")["detection"].max()
            color, style = FAMILY_STYLE[family]
            axis.plot(best.index, best.to_numpy(), color=color, ls=style, lw=2.2,
                      label=f"best {family.replace('_', ' ')}")  # fmt: skip
        axis.set_xlim(0, max_fpr)
        axis.set_ylim(0, 1.0)
        axis.set_xlabel("FPR: successful disturbed episodes with any warning", fontsize=8)
        axis.set_ylabel(
            f"detection: failures warned with > {lead} steps left" if lead
            else "detection: failures warned at any time",
            fontsize=8,
        )  # fmt: skip
        axis.set_title(f"lead requirement: > {lead} steps" if lead else "no lead requirement",
                       fontsize=9)  # fmt: skip
        axis.grid(alpha=0.3)
        axis.legend(fontsize=7, loc="lower right")
    figure.suptitle(
        "FetchPush threshold sweep (thin: each k,p; gray distance-only, orange phase m=off, "
        "blue dotted phase m=3; thick: best of family). Exploratory, validation seeds.",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_paired(paired: pd.DataFrame, path: Path) -> None:
    envelope = paired[paired["first"] == "envelope:phase_aware_m_off"]
    leads = sorted(envelope["min_lead"].unique())
    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    offsets = np.linspace(-0.25, 0.25, len(leads))
    targets = sorted(envelope["target_fpr"].unique())
    for offset, lead in zip(offsets, leads, strict=True):
        part = envelope[envelope["min_lead"] == lead].sort_values("target_fpr")
        x = np.arange(len(part)) + offset
        axis.errorbar(
            x, part["difference"],
            yerr=[part["difference"] - part["ci_low"], part["ci_high"] - part["difference"]],
            fmt="o", capsize=3, label=f"lead > {lead}" if lead else "any warning",
        )  # fmt: skip
    axis.axhline(0, color="gray", lw=0.8)
    axis.set_xticks(range(len(targets)))
    axis.set_xticklabels([f"FPR {t:.0%}" for t in targets])
    axis.set_ylabel("detection: best phase (m=off) − best distance-only")
    axis.set_title(
        "Matched-FPR difference, best of each family (95% stratified episode bootstrap, "
        "envelope re-selected per resample)",
        fontsize=8,
    )
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_by_condition(table: pd.DataFrame, lead: int, path: Path) -> None:
    detectors = [d for d in dict.fromkeys(table["detector"]) if not d.endswith("m3")]
    column = f"detection_lead_gt_{lead}"
    disturbed = table[table["disturbance_type"] != "none"]
    pivot = disturbed.pivot_table(index="condition", columns="detector", values=column,
                                  sort=False)[detectors]  # fmt: skip
    fpr = table.pivot_table(index="condition", columns="detector", values="fpr", sort=False)[
        detectors
    ]
    figure, axes = plt.subplots(1, 2, figsize=(14, 0.32 * len(fpr) + 2.2))
    for axis, data, title in (
        (axes[0], pivot, f"detection with > {lead} steps left"),
        (axes[1], fpr, "false-positive rate"),
    ):
        values = data.to_numpy(dtype=float)
        image = axis.imshow(values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        for (i, j), value in np.ndenumerate(values):
            dark = not np.isnan(value) and value < 0.6
            label = "-" if np.isnan(value) else f"{value:.2f}"
            axis.text(j, i, label, ha="center", va="center", fontsize=5.5,
                      color="white" if dark else "black")  # fmt: skip
        axis.set_xticks(range(len(data.columns)))
        axis.set_xticklabels(data.columns, rotation=60, fontsize=6)
        axis.set_yticks(range(len(data.index)))
        axis.set_yticklabels(data.index, fontsize=6)
        axis.set_title(title, fontsize=8)
        figure.colorbar(image, ax=axis, fraction=0.03)
    pooled = table["pooled_fpr_target"].iloc[0]
    figure.suptitle(
        f"Per condition, each detector at the threshold giving it {pooled:.0%} pooled FPR on "
        "disturbed successes ('-': no failures)",
        fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def plot_sweep(grid, matched, envelopes, paired, by_condition, config, output_dir: Path):
    paths = {
        "curves_plot": output_dir / "sweep_curves.png",
        "paired_plot": output_dir / "sweep_matched_difference.png",
        "condition_plot": output_dir / "sweep_by_condition.png",
    }
    plot_curves(grid, list(config.min_leads), paths["curves_plot"])
    plot_paired(paired, paths["paired_plot"])
    lead = 5 if 5 in config.min_leads else config.min_leads[-1]
    plot_by_condition(by_condition, lead, paths["condition_plot"])
    return paths
