"""Phase G3 mechanism isolation (docs/PHASE_G3_PROPOSAL.md).

- ``--debug``: debugging seeds 3,680,000-009. Every configuration on every cell, plus the
  bitwise replay check. Code checks only; outcomes are not analysed.
- Default: the evaluation on 3,681,000-999, run once. Refuses a dirty tree, a changed plan or
  an existing output directory.
- ``--analyze-only``: re-derives the tables from the write-once raw files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from farc.counterfactual.artifacts import METADATA, analyze_existing, publish_metadata, write_raw
from farc.delaychange.mechanism import CONFIGURATIONS, load_g3, replay_check, run_task
from farc.delaychange.mechanism_analysis import DERIVED, derive_g3, divergence
from farc.delaychange.random_schedules import g3_cells
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256

CONFIG = "configs/delaychange/g3_mechanism.json"
RAW = ("episodes.csv.gz", "steps.csv.gz", METADATA)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def _run(jobs, processes):
    with Pool(processes) as pool:
        parts = pool.map(run_task, jobs)
    return (pd.DataFrame([r for p in parts for r in p[0]]),
            pd.DataFrame([r for p in parts for r in p[1]]))  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    role = "debugging" if args.debug else "evaluation"
    config = load_g3(CONFIG, role)
    meta = {
        "config_file": CONFIG,
        "config_sha256": file_sha256(Path(CONFIG)),
        "plan_sha256": config.plan_sha256,
        "git_commit": _git("rev-parse", "HEAD"),
        "role": role,
        "seeds": [min(config.seeds), max(config.seeds)],
        "configurations": CONFIGURATIONS,
        "checkpoint_sha256": {c: file_sha256(resolve(c)) for c in config.checkpoints},
    }
    jobs = [(config, c, cell) for c in config.checkpoints for cell in g3_cells()]

    if args.debug:
        out = resolve(config.output + "_debug")
        if out.exists():
            raise SystemExit(f"{out} exists")
        with Pool(args.processes) as pool:
            checks = pd.DataFrame([r for p in pool.map(replay_check, jobs) for r in p])
        episodes, steps = _run(jobs, args.processes)
        div, _ = divergence(steps)
        write_raw(out, {RAW[0]: episodes, RAW[1]: steps, "replay_check.csv.gz": checks},
                  {**meta, "episodes_rows": len(episodes), "step_rows": len(steps)})  # fmt: skip
        print(checks.groupby("condition")[["replay_identical", "empty_replay_is_contact"]].all())
        print("C events in check episodes:", int(checks["c_events"].sum()))
        print(div.round(3).to_string(index=False))
        print(f"debug run: {len(episodes)} episodes; outcomes not analysed")
        if not (checks["replay_identical"].all() and checks["empty_replay_is_contact"].all()):
            raise SystemExit("REPLAY CHECK FAILED")
        return

    out, results = resolve(config.output), Path("results") / config.name
    if not args.analyze_only:
        if _git("status", "--porcelain"):
            raise SystemExit("refusing the evaluation with a dirty tree")
        if out.exists():
            raise SystemExit(f"{out} exists: the evaluation is one-shot")
        out.mkdir(parents=True)
        (out / "LOCK.json").write_text(json.dumps(
            {**meta, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
            indent=2) + "\n")  # fmt: skip
        started = time.time()
        episodes, steps = _run(jobs, args.processes)
        write_raw(out, {RAW[0]: episodes, RAW[1]: steps},
                  {**meta, "config": asdict(config), "episodes_rows": len(episodes),
                   "step_rows": len(steps), "seconds": round(time.time() - started, 1),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})  # fmt: skip
        publish_metadata(out, results)
    _, summary = analyze_existing(out, lambda e, s: derive_g3(e, s, config), results, DERIVED,
                                  RAW)  # fmt: skip
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
