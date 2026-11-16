"""Phase E pilot: online actuator identification and adaptive compensation on FetchPush-v4.

A rollout run writes the raw files (``episodes.csv.gz``, ``steps.csv.gz``, ``metadata.json``)
once and refuses to replace them without ``--overwrite-raw``. ``--analyze-only`` reads them and
rewrites only the derived tables and ``analysis.json`` (``farc.counterfactual.artifacts``).
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from farc.adaptation.experiment import (
    DERIVED_TABLES,
    check_replay,
    derive_tables,
    load_adaptation_experiment,
    run_policy,
)
from farc.counterfactual.artifacts import METADATA, analyze_existing, publish_metadata, write_raw
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256

RAW = ("episodes.csv.gz", "steps.csv.gz", METADATA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adaptation/phase_e_pilot.json")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--overwrite-raw", action="store_true")
    args = parser.parse_args()
    config = load_adaptation_experiment(args.config)
    out = resolve(config.output)
    results = Path("results") / config.name

    if not args.analyze_only:
        if (out / RAW[0]).exists() and not args.overwrite_raw:
            raise SystemExit(f"{out} already holds raw results; use --analyze-only")
        started = time.time()
        check_replay(config, config.checkpoints[0],
                     [config.conditions[0], config.conditions[1], config.conditions[4]],
                     config.selection_seeds[0])  # fmt: skip
        with Pool(len(config.checkpoints)) as pool:
            parts = pool.map(run_policy, [(config, c) for c in config.checkpoints])
        episodes = pd.DataFrame([r for part in parts for r in part[0]])
        steps = pd.DataFrame([r for part in parts for r in part[1]])
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=False).stdout.strip()  # fmt: skip
        write_raw(out, {"episodes.csv.gz": episodes, "steps.csv.gz": steps}, {
            "config": asdict(config),
            "config_file": args.config,
            "config_sha256": file_sha256(Path(args.config)),
            "git_commit": commit,
            "checkpoint_sha256": {c: file_sha256(resolve(c)) for c in config.checkpoints},
            "selection_seeds": config.selection_seeds,
            "evaluation_seeds": config.evaluation_seeds,
            "replay_check": "passed (every configuration, 3 conditions, run twice)",
            "episodes_rows": len(episodes), "step_rows": len(steps),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds": round(time.time() - started, 1),
        }, overwrite=args.overwrite_raw)  # fmt: skip
        publish_metadata(out, results)

    tables, summary = analyze_existing(
        out, lambda e, s: derive_tables(e, s, config), results, DERIVED_TABLES, RAW
    )
    print(summary)
    cols = ["configuration", "success_rate", "base_success_rate", "success_difference",
            "ci_low", "ci_high", "rescued", "harmed"]  # fmt: skip
    print(tables["evaluation_actuator_pooled"][cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
