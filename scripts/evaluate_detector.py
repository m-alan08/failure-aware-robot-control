# Migrated from m-alan08/quantrl-fetch scripts/evaluate_detector.py
# (introduced in 934063d, last changed in 934063d). See PROVENANCE.md.
from __future__ import annotations

import argparse
from pathlib import Path

from farc.detection.progress_deficit_eval import load_detector_experiment, run_detector_evaluation
from farc.paths import resolve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the progress-deficit detector offline on saved trajectories."
    )
    parser.add_argument("--config", default="configs/detectors/progress_deficit_grid.json")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <source>/detectors/<name>",
    )
    arguments = parser.parse_args()
    experiment = load_detector_experiment(arguments.config)
    output_dir = (
        Path(arguments.output_dir).expanduser()
        if arguments.output_dir
        else resolve(experiment.source) / "detectors" / experiment.name
    )
    for key, path in run_detector_evaluation(
        experiment, output_dir, Path(arguments.config)
    ).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
