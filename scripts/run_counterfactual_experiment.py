"""Phase D: counterfactual residual-correction feasibility on FetchPush-v4.

Every branch restarts from an exact simulator snapshot; the run aborts if a restored base
continuation does not reproduce the base episode. Candidate selection uses the selection seeds
only; reported confirmatory numbers come from the disjoint evaluation seeds.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from farc.counterfactual.experiment import (
    candidate_table,
    load_counterfactual_experiment,
    oracle_table,
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
    args = parser.parse_args()
    config = load_counterfactual_experiment(args.config)
    out = resolve(config.output)
    out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    threshold = json.loads(PHASE_B.read_text())["thresholds"]["learned_unweighted|0.05"]
    if config.learned_threshold != threshold:
        raise CheckpointError(f"threshold differs from the Phase B 5% point {threshold}")
    load_predictor(resolve(config.predictor))
    datasets = build_datasets(resolve("$FARC_DATA_DIR/experiments/phase1_extended"))
    verify_normalizer(datasets.normalizer, json.loads(DATASET_REPORT.read_text()))

    branches_path = out / "branches.csv.gz"
    if not args.analyze_only:
        with Pool(len(config.checkpoints)) as pool:
            parts = pool.map(run_policy, [(config, c, datasets.normalizer)
                                          for c in config.checkpoints])  # fmt: skip
        pd.DataFrame([r for part in parts for r in part]).to_csv(branches_path, index=False)
    frame = pd.read_csv(branches_path)
    selection, evaluation = split_seeds(frame["episode_seed"].to_numpy(),
                                        config.selection_fraction, config.split_seed)  # fmt: skip
    sel = frame[frame["episode_seed"].isin(selection)]
    ev = frame[frame["episode_seed"].isin(evaluation)]
    n, s = config.bootstrap_resamples, config.bootstrap_seed
    tables = {
        "selection_candidates": candidate_table(sel, [], n, s),
        "evaluation_candidates": candidate_table(ev, [], n, s),
        "evaluation_by_disturbance_type": candidate_table(ev, ["disturbance_type"], n, s),
        "evaluation_by_step": candidate_table(ev, ["t"], n, s),
        "evaluation_by_policy_seed": candidate_table(ev, ["policy_seed"], n, s),
        "evaluation_by_warning": candidate_table(ev, ["warning_now"], n, s),
        "evaluation_by_base_outcome": candidate_table(ev, ["base_success"], n, s),
        "oracle_evaluation": oracle_table(ev),
    }
    chosen = tables["selection_candidates"].sort_values(
        ["success_difference", "harmed_fraction"], ascending=[False, True]).iloc[0]  # fmt: skip
    results = Path("results") / config.name
    results.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
        shutil.copy(out / f"{name}.csv", results / f"{name}.csv")
    states = frame.drop_duplicates(["policy_seed", "condition", "episode_seed", "t"])
    metadata = {
        "config": {k: getattr(config, k) for k in config.__dataclass_fields__},
        "selection_seeds": selection.tolist(), "evaluation_seeds": evaluation.tolist(),
        "selected_candidate_on_selection_seeds": str(chosen["candidate"]),
        "branch_states": len(states), "branch_rows": len(frame),
        "first_warning_states": int(states["first_warning_state"].sum()),
        "predictor_sha256": file_sha256(resolve(config.predictor)),
        "seconds": round(time.time() - started, 1),
    }  # fmt: skip
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n")
    shutil.copy(out / "metadata.json", results / "metadata.json")
    cols = ["candidate", "states", "success_difference", "diff_ci_low", "diff_ci_high",
            "rescued_fraction", "harmed_fraction", "mean_distance_improvement"]  # fmt: skip
    print("selected on selection seeds:", chosen["candidate"])
    print(tables["evaluation_candidates"].sort_values("success_difference", ascending=False)[cols]
          .round(4).to_string())  # fmt: skip
    print(tables["oracle_evaluation"].round(3).to_string())


if __name__ == "__main__":
    main()
