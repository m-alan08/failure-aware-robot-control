"""Build the FetchPush failure-prediction splits and write a dataset report (JSON)."""

from __future__ import annotations

import argparse
import json

from farc.paths import resolve
from farc.prediction import FeatureConfig, build_datasets
from farc.prediction.dataset import save_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="$FARC_DATA_DIR/experiments/phase1_extended")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--progress-window", type=int, default=5)
    parser.add_argument("--output", default="results/failure_dataset/report.json")
    args = parser.parse_args()
    datasets = build_datasets(
        resolve(args.experiment),
        args.validation_fraction,
        args.split_seed,
        FeatureConfig(progress_window=args.progress_window),
    )
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_report(datasets, output)
    report = datasets.report()
    for key in ("all", "train", "validation", "duplicates"):
        print(key, json.dumps(report[key]))
    print("feature_dim", report["feature_dim"], "horizon", report["horizon"])


if __name__ == "__main__":
    main()
