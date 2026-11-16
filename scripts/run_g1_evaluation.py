"""Phase G1 supervisory switching (docs/PHASE_G1_PROPOSAL.md, approved 2026-11-16).

- ``--debug``: debugging seeds 3,660,000-009, every configuration and condition, with a
  debug-only CUSUM threshold so the release path runs. Code checks only; outcomes are not
  analysed.
- ``--calibrate``: calibration seeds 3,661,000-049 on C1-C4, the never-releasing CUSUM trace
  only. Sets the threshold by the pre-registered false-release criterion. Task success is not
  written or read.
- Default: the pre-registered evaluation on 3,662,000-099, run once. Refuses a dirty tree, an
  unset threshold, a changed plan or an existing output directory.
- ``--analyze-only``: re-derives the tables from the write-once raw files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from farc.counterfactual.artifacts import METADATA, analyze_existing, publish_metadata, write_raw
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256
from farc.supervision.episodes import CONFIGURATIONS
from farc.supervision.evaluation import (
    CALIBRATION_CONFIGURATION,
    DERIVED_TABLES,
    calibrate_threshold,
    derive_g1,
    load_g1,
    run_task,
    tasks,
)

CONFIG = "configs/supervision/phase_g1.json"
RAW = ("episodes.csv.gz", "steps.csv.gz", METADATA)
DEBUG_THRESHOLD = 3.0  # debug runs only, so the release path executes; never analysed


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def _run(jobs: list[tuple], processes: int):
    with Pool(processes) as pool:
        parts = pool.map(run_task, jobs)
    episodes = pd.DataFrame([r for part in parts for r in part[0]])
    steps = pd.DataFrame([r for part in parts for r in part[1]])
    return episodes, steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true")
    mode.add_argument("--calibrate", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    role = "debugging" if args.debug else "calibration" if args.calibrate else "evaluation"
    config = load_g1(args.config, role=role)
    meta = {
        "config_file": args.config,
        "config_sha256": file_sha256(Path(args.config)),
        "plan_sha256": config.plan_sha256,
        "model_sha256": config.model_sha256,
        "git_commit": _git("rev-parse", "HEAD"),
        "role": role,
        "checkpoint_sha256": {c: file_sha256(resolve(c)) for c in config.checkpoints},
    }

    if args.debug:
        sup = {**config.supervisor, "cusum_threshold": DEBUG_THRESHOLD}
        config = replace(config, supervisor=sup)
        out = resolve(config.output + "_debug")
        if (out / RAW[0]).exists():
            raise SystemExit(f"{out} already holds a debug run")
        conditions = tuple(c["name"] for c in config.conditions)
        started = time.time()
        episodes, steps = _run(tasks(config, conditions, CONFIGURATIONS, True), args.processes)
        write_raw(
            out,
            {RAW[0]: episodes, RAW[1]: steps},
            {
                **meta,
                "config": asdict(config),
                "seeds": config.seeds,
                "debug_threshold": DEBUG_THRESHOLD,
                "episodes_rows": len(episodes),
                "step_rows": len(steps),
                "seconds": round(time.time() - started, 1),
            },
        )
        print(f"debug run written to {out}: {len(episodes)} episodes, {len(steps)} steps; "
              "outcomes are not analysed")  # fmt: skip
        return

    if args.calibrate:
        out = resolve(config.output + "_calibration")
        if out.exists():
            raise SystemExit(f"{out} exists: calibration is one-shot")
        if config.supervisor.get("cusum_threshold") is not None:
            raise SystemExit("the threshold is already set")
        jobs = tasks(config, config.calibration_conditions, (CALIBRATION_CONFIGURATION,), False)
        started = time.time()
        _, steps = _run(jobs, args.processes)
        keep = ["configuration", "policy_seed", "condition", "episode_seed", "t", "delay_true",
                "delay_param", "active", "est_delay", "est_reliable", "transitions",
                "cusum_increment", "cusum_statistic"]  # fmt: skip
        steps = steps[keep]  # success is never written for calibration
        sup = config.supervisor
        h, curve = calibrate_threshold(steps, sup["cusum_threshold_grid"], sup["min_dwell"],
                                       sup["false_release_max"])  # fmt: skip
        record = {**meta, "seeds": config.seeds, "conditions": config.calibration_conditions,
                  "criterion": f"smallest grid threshold with false-off share of active steps "
                               f"<= {sup['false_release_max']} under constant delay",
                  "chosen_threshold": h, "step_rows": len(steps),
                  "seconds": round(time.time() - started, 1)}  # fmt: skip
        write_raw(out, {RAW[1]: steps}, record)
        results = Path("results") / (config.name + "_calibration")
        results.mkdir(parents=True, exist_ok=True)
        curve.to_csv(results / "threshold_curve.csv", index=False)
        (results / "calibration.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(curve.to_string(index=False))
        print(f"chosen threshold: {h}")
        return

    out = resolve(config.output)
    results = Path("results") / config.name
    if not args.analyze_only:
        if config.supervisor.get("cusum_threshold") is None:
            raise SystemExit("cusum_threshold is unset: run --calibrate and commit it first")
        dirty = _git("status", "--porcelain")
        if dirty:
            raise SystemExit(f"refusing the evaluation with a dirty tree:\n{dirty}")
        if out.exists():
            raise SystemExit(f"{out} exists: the evaluation is one-shot")
        out.mkdir(parents=True)
        (out / "LOCK.json").write_text(json.dumps(
            {**meta, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            indent=2) + "\n")  # fmt: skip
        conditions = tuple(c["name"] for c in config.conditions)
        started = time.time()
        episodes, steps = _run(tasks(config, conditions, CONFIGURATIONS, True), args.processes)
        write_raw(out, {RAW[0]: episodes, RAW[1]: steps},
                  {**meta, "config": asdict(config), "seeds": config.seeds,
                   "episodes_rows": len(episodes), "step_rows": len(steps),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "seconds": round(time.time() - started, 1)})  # fmt: skip
        publish_metadata(out, results)

    tables, summary = analyze_existing(
        out, lambda e, s: derive_g1(e, s, config), results, DERIVED_TABLES, RAW
    )
    print(json.dumps(summary, indent=2, default=str))
    print(tables["hypotheses"].round(4).to_string(index=False))
    print(tables["safety_gates"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
