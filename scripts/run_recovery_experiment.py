"""Phase C: closed-loop failure-triggered recovery on FetchPush-v4 (development seeds).

Stops if the predictor checkpoint, the dataset report or the Phase B thresholds do not match;
nothing is substituted or retrained.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from farc.paths import resolve
from farc.prediction import build_datasets
from farc.prediction.evaluation import (
    CheckpointError,
    file_sha256,
    load_predictor,
    verify_normalizer,
)
from farc.recovery.experiment import (
    analyze,
    load_recovery_experiment,
    run_policy,
    validate_recovery_experiment,
    write_metadata,
)

PHASE_B = Path("results/phase_b_evaluation/metadata.json")
DATASET_REPORT = Path("results/failure_dataset/report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/recovery/phase_c_development.json")
    parser.add_argument("--episodes", type=int, help="override, for quick runs")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    config = load_recovery_experiment(args.config)
    if args.episodes:
        config = validate_recovery_experiment(
            replace(config, episodes=args.episodes)
        )  # the override must not reach used or held-out seeds
    out = resolve(config.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    phase_b = json.loads(PHASE_B.read_text())["thresholds"]
    expected = {"learned": phase_b["learned_unweighted|0.05"],
                "reactive": phase_b["phase_k5_p5_h3|0.05"]}  # fmt: skip
    if (config.learned_threshold, config.reactive_threshold) != (
        expected["learned"],
        expected["reactive"],
    ):
        raise CheckpointError(f"thresholds differ from the Phase B 5% operating points {expected}")
    load_predictor(resolve(config.predictor))  # shape-checks the checkpoint
    datasets = build_datasets(resolve("$FARC_DATA_DIR/experiments/phase1_extended"))
    verify_normalizer(datasets.normalizer, json.loads(DATASET_REPORT.read_text()))

    episodes_path = out / "episodes.csv"
    if not args.analyze_only:
        with Pool(len(config.checkpoints)) as pool:
            parts = pool.map(
                run_policy, [(config, c, datasets.normalizer) for c in config.checkpoints]
            )
        pd.DataFrame([row for part in parts for row in part]).to_csv(episodes_path, index=False)
    episodes = pd.read_csv(episodes_path)
    tables = analyze(episodes, config.bootstrap_resamples, config.bootstrap_seed)
    results = Path("results") / config.name
    results.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        shutil.copy(out / f"{name}.csv", results / f"{name}.csv")
    write_metadata(config, out, {
        "predictor_sha256": file_sha256(resolve(config.predictor)),
        "phase_b_thresholds": expected,
        "episodes_rows": len(episodes),
        "seconds": round(time.time() - started, 1),
    })  # fmt: skip
    shutil.copy(out / "metadata.json", results / "metadata.json")
    print(tables["summary_pooled"].round(3).to_string())
    print(tables["differences_pooled"].round(3).to_string())


if __name__ == "__main__":
    main()
