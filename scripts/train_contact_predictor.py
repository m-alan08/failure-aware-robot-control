"""Phase F1: collect dynamics data (training and model-selection seeds), train the
pre-registered candidate grid, select by open-loop prediction error only, and freeze.

Raw rollouts are write-once (``farc.counterfactual.artifacts``); ``--reuse-data`` trains from
the existing raw files without rewriting them.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from farc.adaptation.estimator import EstimatorConfig
from farc.contact.data import (
    build_conditions,
    check_seeds,
    collect,
    contact_flags,
    horizon_pairs,
    transitions,
)
from farc.contact.dynamics import S, rollout_batch, static_object_batch, train_dynamics
from farc.contact.episodes import phase_e_predictor
from farc.counterfactual.artifacts import METADATA, sha256, write_raw
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256

RAW = ("training.csv.gz", "selection.csv.gz", METADATA)


def _collect(checkpoint, config, seeds):
    conditions = build_conditions(config["conditions_from"], config["conditions"])
    k0 = EstimatorConfig(**config["estimator"]).nominal_gain
    return collect(checkpoint, conditions, seeds, config["perturbation_seed"],
                   config["estimator"], {"base_only": None,
                                         "phase_e_predictor": phase_e_predictor(k0)})  # fmt: skip


def position_errors(model, frame, horizon, k0):
    start, cmds, target, true_start, _ = horizon_pairs(frame, horizon)
    learned = rollout_batch(model, start, cmds)
    static = static_object_batch(start, cmds, k0)
    contact = contact_flags(true_start, target)
    out = {}
    for name, pred in (("learned", learned), ("static", static)):
        for key in ("gripper_position", "object_position"):
            err = np.linalg.norm(pred[:, S[key]] - target[:, S[key]], axis=1)
            out[(name, key)] = err
    return out, contact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/contact/phase_f1_dynamics.json")
    parser.add_argument("--reuse-data", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    train_seeds = list(
        range(
            config["training_seed_offset"],
            config["training_seed_offset"] + config["training_episodes"],
        )
    )
    sel_seeds = list(
        range(
            config["selection_seed_offset"],
            config["selection_seed_offset"] + config["selection_episodes"],
        )
    )
    check_seeds(train_seeds, "training")
    check_seeds(sel_seeds, "selection")
    out = resolve(config["data_output"])
    k0 = EstimatorConfig(**config["estimator"]).nominal_gain
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            check=False).stdout.strip()  # fmt: skip
    if not args.reuse_data:
        started = time.time()
        frames = {}
        for name, seeds in (("training.csv.gz", train_seeds), ("selection.csv.gz", sel_seeds)):
            with Pool(len(config["checkpoints"])) as pool:
                parts = pool.map(partial(_collect, config=config, seeds=seeds),
                                 config["checkpoints"])  # fmt: skip
            frames[name] = pd.DataFrame([r for part in parts for r in part])
        write_raw(out, frames, {"config": config, "config_sha256": file_sha256(Path(args.config)),
                                "git_commit": commit,
                                "checkpoint_sha256": {c: file_sha256(resolve(c))
                                                      for c in config["checkpoints"]},
                                "seconds": round(time.time() - started, 1)})  # fmt: skip
    raw_hashes = {n: sha256(out / n) for n in RAW}
    train = pd.read_csv(out / RAW[0])
    sel = pd.read_csv(out / RAW[1])
    obs, cmd, nxt = transitions(train)
    rows, models = [], {}
    for width in config["widths"]:
        for wd in config["weight_decays"]:
            model = train_dynamics(obs, cmd, nxt, width, wd, config["epochs"], config["train_seed"])
            score, row = [], {"width": width, "weight_decay": wd}
            for h in config["horizons"]:
                errors, contact = position_errors(model, sel, h, k0)
                weight = np.where(contact, config["contact_weight"], 1.0)
                both = (
                    errors[("learned", "gripper_position")] + errors[("learned", "object_position")]
                )
                score.append(float((both * weight).sum() / weight.sum()))
                for (name, key), err in errors.items():
                    row[f"h{h}_{name}_{key}_contact"] = float(err[contact].mean())
                    row[f"h{h}_{name}_{key}_free"] = float(err[~contact].mean())
            row["selection_score"] = float(np.mean(score))
            rows.append(row)
            models[(width, wd)] = model
    table = pd.DataFrame(rows).sort_values("selection_score")
    best = table.iloc[0]
    model = models[(int(best["width"]), float(best["weight_decay"]))]
    model_dir = resolve(config["model_output"])
    if (model_dir / "model.pt").exists():
        raise SystemExit(f"{model_dir}/model.pt exists; the frozen predictor is never overwritten")
    digest = model.save(model_dir / "model.pt")
    results = Path("results") / config["name"]
    results.mkdir(parents=True, exist_ok=True)
    table.to_csv(results / "model_selection.csv", index=False)
    summary = {
        "selected": {"width": int(best["width"]), "weight_decay": float(best["weight_decay"])},
        "model_sha256": digest,
        "model_path": str(model_dir / "model.pt"),
        "training_transitions": int(len(obs)),
        "raw_sha256": raw_hashes,
        "git_commit": commit,
        "config_sha256": file_sha256(Path(args.config)),
    }
    (model_dir / "model.json").write_text(json.dumps(summary, indent=2) + "\n")
    (results / "model.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(table.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
