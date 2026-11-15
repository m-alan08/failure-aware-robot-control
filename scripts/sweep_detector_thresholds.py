# Migrated from m-alan08/quantrl-fetch scripts/sweep_detector_thresholds.py
# (introduced in 91e44be, last changed in 91e44be). See PROVENANCE.md.
from __future__ import annotations

import argparse
from pathlib import Path

from farc.detection.threshold_sweep import load_sweep_config, run_sweep
from farc.paths import resolve


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Threshold-free matched-FPR comparison of the FetchPush detectors."
    )
    parser.add_argument("--config", default="configs/detectors/threshold_sweep_push.json")
    parser.add_argument("--output-dir", default=None, help="Default: <source>/detectors/<name>")
    arguments = parser.parse_args()
    config = load_sweep_config(arguments.config)
    output_dir = (
        Path(arguments.output_dir).expanduser()
        if arguments.output_dir
        else resolve(config.source) / "detectors" / config.name
    )
    for key, path in run_sweep(config, output_dir, Path(arguments.config)).items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
