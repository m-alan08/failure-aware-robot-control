# Migrated from m-alan08/quantrl-fetch scripts/evaluate_phase_detector.py
# (introduced in 98c903e, last changed in 98c903e). See PROVENANCE.md.
from __future__ import annotations

import argparse
from pathlib import Path

from farc.detection.phase_aware_eval import load_phase_evaluation, run_phase_evaluation
from farc.paths import resolve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the phase-aware and distance-only detectors on matched episodes."
    )
    parser.add_argument("--config", default="configs/detectors/phase_aware_push.json")
    parser.add_argument("--output-dir", default=None, help="Default: <source>/detectors/<name>")
    arguments = parser.parse_args()
    evaluation = load_phase_evaluation(arguments.config)
    output_dir = (
        Path(arguments.output_dir).expanduser()
        if arguments.output_dir
        else resolve(evaluation.source) / "detectors" / evaluation.name
    )
    paths = run_phase_evaluation(evaluation, output_dir, Path(arguments.config))
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
