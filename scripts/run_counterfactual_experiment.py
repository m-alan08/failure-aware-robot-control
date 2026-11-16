"""Phase D: counterfactual residual-correction feasibility on FetchPush-v4.

``--analyze-only`` reads the raw files of an earlier run and rewrites only the derived
tables and ``analysis.json`` (see ``farc.counterfactual.artifacts``); a rollout run refuses to
replace existing raw files unless ``--overwrite-raw`` is given.

Every branch restarts from an exact simulator snapshot; the run aborts if a restored base
continuation does not reproduce the base episode. Candidate selection uses the selection seeds
only; reported confirmatory numbers come from the disjoint evaluation seeds.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from farc.counterfactual.artifacts import (
    analyze_existing,
    publish_metadata,
    write_raw,
)
from farc.counterfactual.experiment import (
    derive_tables,
    load_counterfactual_experiment,
    run_policy,
    split_seeds,
)
from farc.paths import resolve
from farc.prediction import build_datasets
from farc.prediction.evaluation import (
    CheckpointError,
    file_sha256,
    load_predictor,
    verify_normalizer,
)

PHASE_B = Path("results/phase_b_evaluation/metadata.json")
DATASET_REPORT = Path("results/failure_dataset/report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/counterfactual/phase_d_pilot.json")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--overwrite-raw", action="store_true",
                        help="allow a rollout run to replace existing raw files")  # fmt: skip
    args = parser.parse_args()
    config = load_counterfactual_experiment(args.config)
    out = resolve(config.output)
    results = Path("results") / config.name
    phase_b = json.loads(PHASE_B.read_text())
    threshold = phase_b["thresholds"]["learned_unweighted|0.05"]
    if config.learned_threshold != threshold:
        raise CheckpointError(f"threshold differs from the Phase B 5% point {threshold}")
    predictor_sha = file_sha256(resolve(config.predictor))
    if predictor_sha != phase_b["checkpoint_sha256"]["learned_unweighted"]:
        raise CheckpointError("the predictor is not the one Phase B calibrated the threshold on")

    if not args.analyze_only:
        if (out / "branches.csv.gz").exists() and not args.overwrite_raw:
            raise SystemExit(f"{out} already holds raw results; use --analyze-only or "
                             "--overwrite-raw")  # fmt: skip
        started = time.time()
        load_predictor(resolve(config.predictor))
        datasets = build_datasets(resolve("$FARC_DATA_DIR/experiments/phase1_extended"))
        verify_normalizer(datasets.normalizer, json.loads(DATASET_REPORT.read_text()))
        with Pool(len(config.checkpoints)) as pool:
            parts = pool.map(run_policy, [(config, c, datasets.normalizer)
                                          for c in config.checkpoints])  # fmt: skip
        frame = pd.DataFrame([r for part in parts for r in part])
        seeds = frame["episode_seed"].to_numpy()
        selection, evaluation = split_seeds(seeds, config.selection_fraction, config.split_seed)
        write_raw(out, frame, {
            "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
            "selection_seeds": selection.tolist(), "evaluation_seeds": evaluation.tolist(),
            "branch_rows": len(frame), "predictor_sha256": predictor_sha,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds": round(time.time() - started, 1),
        }, overwrite=args.overwrite_raw)  # fmt: skip
        publish_metadata(out, results)

    tables, summary = analyze_existing(out, lambda f: derive_tables(f, config), results)
    cols = ["candidate", "states", "success_difference", "diff_ci_low", "diff_ci_high",
            "rescued_fraction", "harmed_fraction", "mean_distance_improvement"]  # fmt: skip
    print("selected on selection seeds:", summary["selected_candidate_on_selection_seeds"])
    print(tables["evaluation_candidates"].sort_values("success_difference", ascending=False)[cols]
          .round(4).to_string())  # fmt: skip
    print(tables["oracle_evaluation"].round(3).to_string())


if __name__ == "__main__":
    main()
