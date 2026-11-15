"""Phase B evaluation: learned failure predictors vs the reactive detectors.

Stops with an error if a checkpoint or the dataset report (training normalizer) is missing;
nothing is substituted or retrained.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from farc.detection.phase_aware_eval import load_states
from farc.paths import resolve
from farc.prediction import FeatureConfig, build_datasets
from farc.prediction.evaluation import (
    CheckpointError,
    ReactiveSpec,
    assemble,
    bootstrap,
    file_sha256,
    load_predictor,
    metric_table,
    thresholds,
    timestep_by_remaining,
    timestep_tables,
    verify_normalizer,
)

COMPACT = [
    "split.json", "warning_pooled.csv", "warning_by_condition.csv", "warning_by_policy_seed.csv",
    "warning_intervals.csv", "warning_differences.csv", "timestep_metrics.csv",
    "timestep_by_remaining.csv", "reliability.csv", "metadata.json",
]  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prediction/phase_b_evaluation.json")
    parser.add_argument("--resamples", type=int, help="override (for quick runs)")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    resamples = args.resamples or config["bootstrap_resamples"]
    ts_resamples = args.resamples or config["timestep_bootstrap_resamples"]
    started = time.time()

    paths = {name: resolve(p) for name, p in config["predictors"].items()}
    report_path = resolve(config["dataset_report"])
    missing = [str(p) for p in [*paths.values(), report_path] if not p.exists()]
    if missing:
        raise CheckpointError(f"required files are missing: {missing}")
    predictors = {name: load_predictor(path) for name, path in paths.items()}

    source = resolve(config["source"])
    datasets = build_datasets(
        source,
        config["validation_fraction"],
        config["split_seed"],
        FeatureConfig(progress_window=config["progress_window"]),
    )
    verify_normalizer(datasets.normalizer, json.loads(report_path.read_text()))
    reactive = [ReactiveSpec(**spec) for spec in config["reactive"]]
    states = load_states(source, "FetchPush-v4")
    data = assemble(
        datasets, predictors, reactive, states, config["geometry"],
        config["calibration_fraction"], config["calibration_seed"],
    )  # fmt: skip
    fprs, leads = tuple(config["target_fprs"]), tuple(config["leads"])
    chosen = thresholds(data, fprs)

    out = resolve(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    e = data.episodes
    split = {
        "dataset_train_seeds": datasets.train_seeds.tolist(),
        "validation_seeds": datasets.validation_seeds.tolist(),
        "calibration_seeds": data.calibration_seeds.tolist(),
        "evaluation_seeds": data.evaluation_seeds.tolist(),
    }
    for part in ("calibration", "evaluation"):
        sub = e[e["part"] == part]
        dist = sub[sub["disturbed"]]
        split[part] = {
            "episodes": len(sub), "disturbed_episodes": len(dist),
            "disturbed_failures": int(dist["failure"].sum()),
            "disturbed_successes": int((dist["failure"] == 0).sum()),
        }  # fmt: skip
    (out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    metric_table(data, chosen, leads, None).to_csv(out / "warning_pooled.csv", index=False)
    metric_table(data, chosen, leads, "condition", disturbed_only=False).to_csv(
        out / "warning_by_condition.csv", index=False
    )
    metric_table(data, chosen, leads, "policy_seed").to_csv(
        out / "warning_by_policy_seed.csv", index=False
    )
    pairs = [tuple(p) for p in config["pairs"]]
    rates, diffs = bootstrap(data, fprs, leads, pairs, resamples, config["bootstrap_seed"])
    rates.to_csv(out / "warning_intervals.csv", index=False)
    diffs.to_csv(out / "warning_differences.csv", index=False)
    ts, rel = timestep_tables(data, ts_resamples, config["bootstrap_seed"])
    ts.to_csv(out / "timestep_metrics.csv", index=False)
    rel.to_csv(out / "reliability.csv", index=False)
    timestep_by_remaining(data).to_csv(out / "timestep_by_remaining.csv", index=False)
    metadata = {
        "config": config,
        "bootstrap_resamples": resamples,
        "timestep_bootstrap_resamples": ts_resamples,
        "checkpoint_sha256": {name: file_sha256(p) for name, p in paths.items()},
        "dataset_report_sha256": file_sha256(report_path),
        "thresholds": {f"{n}|{t}": v for (n, t), v in chosen.items()},
        "seconds": round(time.time() - started, 1),
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = resolve(config["results"])
    results.mkdir(parents=True, exist_ok=True)
    for name in COMPACT:
        shutil.copy(out / name, results / name)
    print(f"wrote {out} and {results} in {metadata['seconds']} s")


if __name__ == "__main__":
    main()
