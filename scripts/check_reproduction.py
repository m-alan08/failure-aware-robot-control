"""Compare regenerated Phase 1 outputs under $FARC_DATA_DIR with the reference results.

The reference tables in results/ were produced in m-alan08/quantrl-fetch (see each
metadata.json and PROVENANCE.md). A table passes only if it is identical after reading both
CSV files (same columns, same row order, same values).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from farc.paths import data_dir

REPO = Path(__file__).resolve().parents[1]
CHECKS = {
    "phase1_extended": (
        "experiments/phase1_extended",
        ["summary.csv", "summary_per_seed.csv", "distance_curves.csv"],
    ),
    "progress_deficit_grid": (
        "experiments/phase1_extended/detectors/progress_deficit_grid",
        ["detector_overall.csv", "detector_by_condition.csv", "detector_per_seed.csv"],
    ),
    "phase_aware_push": (
        "experiments/phase1_extended/detectors/phase_aware_push",
        [
            "phase_detector_overall.csv",
            "phase_detector_paired.csv",
            "phase_detector_by_condition.csv",
            "phase_detector_per_seed.csv",
            "post_hoc_comparisons.csv",
        ],
    ),
    "threshold_sweep_push": (
        "experiments/phase1_extended/detectors/threshold_sweep_push",
        [
            "sweep_matched.csv",
            "sweep_envelopes.csv",
            "sweep_paired.csv",
            "sweep_by_condition.csv",
            "sweep_per_seed.csv",
            "sweep_curves_grid.csv",
        ],
    ),
}


def compare(fresh_path: Path, reference_path: Path) -> str | None:
    if not fresh_path.exists():
        return "missing (not regenerated yet)"
    fresh = pd.read_csv(fresh_path, low_memory=False)
    reference = pd.read_csv(reference_path, low_memory=False)
    if list(fresh.columns) != list(reference.columns):
        return f"columns differ: {sorted(set(fresh.columns) ^ set(reference.columns))}"
    try:
        pd.testing.assert_frame_equal(fresh, reference, check_exact=True, check_dtype=False)
    except AssertionError as error:
        return str(error).splitlines()[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", choices=sorted(CHECKS), default=None)
    arguments = parser.parse_args()
    failures = 0
    for name in arguments.only or CHECKS:
        output_dir, files = CHECKS[name]
        for file_name in files:
            problem = compare(
                data_dir() / output_dir / file_name, REPO / "results" / name / file_name
            )
            status = "identical" if problem is None else f"DIFFERS: {problem}"
            failures += problem is not None
            print(f"{name}/{file_name}: {status}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
