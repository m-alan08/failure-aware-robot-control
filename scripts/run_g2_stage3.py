"""G2 Stages 2-3 (docs/PHASE_G2_PROPOSAL.md, section 5).

- ``--debug``: debugging seeds 3,670,000-009, every evaluation configuration on every cell, with
  placeholder constants. Code checks only; outcomes are not analysed.
- ``--calibrate``: calibration seeds 3,672,000-099 on the calibration cells. Constants are chosen
  by detection criteria only, and task success is not written.
- Default: the pre-registered evaluation on 3,673,000-099, run once. Refuses a dirty tree, a
  changed plan, an unpinned calibration record or an existing output directory.
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
from farc.delaychange.stage3 import (
    EVALUATION,
    YOKED,
    calibration_cells,
    calibration_configurations,
    choose_constants,
    detection_metrics,
    evaluation_cells,
    load_g2,
    run_task,
)
from farc.delaychange.stage3_analysis import DERIVED, derive_g2
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256

CONFIG = "configs/delaychange/g2_stage3.json"
RAW = ("episodes.csv.gz", "steps.csv.gz", METADATA)
DEBUG_CONSTANTS = {
    "A_window": 5,
    "B_fast_window": 4,
    "C_threshold": 4.0,
    "C_sigma0_mm": 2.0,
    "C_variant": "gauss",
}  # code checks only; never analysed


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def evaluation_configurations(constants: dict) -> tuple[str, ...]:
    """Candidates that failed calibration (no admissible setting) are not run (A8)."""
    drop = set()
    if constants.get("A_window") is None:
        drop.add("cand_A_short_window")
    if constants.get("B_fast_window") is None:
        drop |= {"cand_B_two_timescale", "sham_B_yoked"}
    if constants.get("C_threshold") is None:
        drop |= {"cand_C_sequential", "sham_C_yoked"}
    names = tuple(c for c in EVALUATION if c not in drop)
    assert all(YOKED[s] in names for s in names if s in YOKED)
    return names


def _run(jobs, processes):
    with Pool(processes) as pool:
        parts = pool.map(run_task, jobs)
    return (pd.DataFrame([r for p in parts for r in p[0]]),
            pd.DataFrame([r for p in parts for r in p[1]]))  # fmt: skip


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
    config = load_g2(args.config, role)
    meta = {
        "config_file": args.config,
        "config_sha256": file_sha256(Path(args.config)),
        "plan_sha256": config.plan_sha256,
        "git_commit": _git("rev-parse", "HEAD"),
        "role": role,
        "seeds": config.seeds,
        "checkpoint_sha256": {c: file_sha256(resolve(c)) for c in config.checkpoints},
    }

    if args.debug:
        config = replace(config, constants=DEBUG_CONSTANTS)
        out = resolve(config.output + "_debug")
        if out.exists():
            raise SystemExit(f"{out} exists")
        names = evaluation_configurations(config.constants)
        jobs = [(config, c, cell, names) for c in config.checkpoints for cell in evaluation_cells()]
        started = time.time()
        episodes, steps = _run(jobs, args.processes)
        write_raw(
            out,
            {RAW[0]: episodes, RAW[1]: steps},
            {
                **meta,
                "constants": DEBUG_CONSTANTS,
                "episodes_rows": len(episodes),
                "step_rows": len(steps),
                "seconds": round(time.time() - started, 1),
            },
        )
        print(f"debug run: {len(episodes)} episodes, {len(steps)} steps; not analysed")
        return

    if args.calibrate:
        out = resolve(config.output + "_calibration")
        if out.exists():
            raise SystemExit(f"{out} exists: calibration is one-shot")
        if _git("status", "--porcelain"):
            raise SystemExit("refusing calibration with a dirty tree")
        names = calibration_configurations()
        jobs = [
            (config, c, cell, names) for c in config.checkpoints for cell in calibration_cells()
        ]
        started = time.time()
        _, steps = _run(jobs, args.processes)  # episode outcomes are discarded unread
        metrics = detection_metrics(steps)
        chosen = choose_constants(metrics)
        keep = ["configuration", "policy_seed", "condition", "family", "episode_seed", "t",
                "delay_true", "delay_used", "active", "alarm", "suspended", "est_delay",
                "est_reliable"]  # fmt: skip
        record = {**meta, "chosen": chosen, "seconds": round(time.time() - started, 1),
                  "step_rows": len(steps)}  # fmt: skip
        write_raw(out, {RAW[1]: steps[keep]}, record)
        results = Path("results") / (config.name + "_calibration")
        results.mkdir(parents=True, exist_ok=True)
        metrics.to_csv(results / "detection_metrics.csv", index=False)
        (results / "calibration.json").write_text(json.dumps(record, indent=2, default=str) + "\n")
        print(metrics.round(4).to_string(index=False))
        print(json.dumps(chosen, indent=2))
        return

    out = resolve(config.output)
    results = Path("results") / config.name
    if not args.analyze_only:
        if _git("status", "--porcelain"):
            raise SystemExit("refusing the evaluation with a dirty tree")
        if out.exists():
            raise SystemExit(f"{out} exists: the evaluation is one-shot")
        out.mkdir(parents=True)
        (out / "LOCK.json").write_text(
            json.dumps(
                {
                    **meta,
                    "constants": config.constants,
                    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
                indent=2,
            )
            + "\n"
        )
        names = evaluation_configurations(config.constants)
        jobs = [(config, c, cell, names) for c in config.checkpoints for cell in evaluation_cells()]
        started = time.time()
        episodes, steps = _run(jobs, args.processes)
        write_raw(out, {RAW[0]: episodes, RAW[1]: steps},
                  {**meta, "config": asdict(config), "configurations": names,
                   "episodes_rows": len(episodes), "step_rows": len(steps),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "seconds": round(time.time() - started, 1)})  # fmt: skip
        publish_metadata(out, results)
    tables, summary = analyze_existing(out, lambda e, s: derive_g2(e, s, config), results,
                                       DERIVED, RAW)  # fmt: skip
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
