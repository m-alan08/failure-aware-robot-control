# Migrated from m-alan08/quantrl-fetch scripts/compare_detectors.py
# (introduced in c39a4ab, last changed in ff20609). See PROVENANCE.md.
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from farc.detection.phase_aware_eval import compare_detectors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Episode-paired comparison of two detectors from a saved phase evaluation."
    )
    parser.add_argument("--episodes", required=True, help="phase_detector_episodes.csv.gz")
    parser.add_argument("--pair", nargs=2, action="append", required=True, metavar=("A", "B"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument(
        "--min-lead",
        type=int,
        action="append",
        default=None,
        help="Also compare counting only warnings with more than this many steps left (repeatable)",
    )
    arguments = parser.parse_args()
    results = pd.read_csv(arguments.episodes, low_memory=False)
    leads = [None, *(arguments.min_lead or [])]
    rows = [
        compare_detectors(results, a, b, arguments.resamples, arguments.seed + index, lead)
        for index, ((a, b), lead) in enumerate(
            (pair, lead) for lead in leads for pair in arguments.pair
        )
    ]
    frame = pd.DataFrame(rows)
    Path(arguments.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(arguments.out, index=False)
    print(frame.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
