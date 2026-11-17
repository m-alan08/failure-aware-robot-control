"""Phase G4 held-out confirmation of Candidate C (docs/PHASE_G4_HELDOUT_PROPOSAL.md).

- ``--debug``: debugging seeds 3,690,000-009 on every cell, plus the bitwise equivalence check
  against the G2 Stage-3 runner. Code checks only; outcomes are not analysed.
- Default: the confirmation on held-out seeds 4,000,300-599, run once. Refuses a dirty tree, a
  changed plan, an existing output directory or any other seed.
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
from farc.delaychange.heldout import ARM_F, ARM_R, equivalence_check, g4_cells, load_g4, run_task
from farc.delaychange.heldout_analysis import DERIVED, derive_g4
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256

CONFIG = "configs/delaychange/g4_heldout.json"
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
    role = "debugging" if args.debug else "confirmation"
    config = load_g4(CONFIG, role)
    meta = {
        "config_file": CONFIG,
        "config_sha256": file_sha256(Path(CONFIG)),
        "plan_sha256": config.plan_sha256,
        "git_commit": _git("rev-parse", "HEAD"),
        "role": role,
        "seeds": [min(config.seeds), max(config.seeds)],
        "arms": {"R": ARM_R, "F": ARM_F},
        "checkpoint_sha256": {c: file_sha256(resolve(c)) for c in config.checkpoints},
    }
    jobs = [(config, c, cell) for c in config.checkpoints for cell in g4_cells()]

    if args.debug:
        out = resolve(config.output + "_debug")
        if out.exists():
            raise SystemExit(f"{out} exists")
        with Pool(args.processes) as pool:
            checks = pd.DataFrame([r for p in pool.map(equivalence_check, jobs) for r in p])
        episodes, steps = _run(jobs, args.processes)
        write_raw(out, {RAW[0]: episodes, RAW[1]: steps, "equivalence.csv.gz": checks},
                  {**meta, "episodes_rows": len(episodes), "step_rows": len(steps)})  # fmt: skip
        cols = ["C_identical", "contact_identical", "sham_identical"]
        print(checks.groupby("arm")[cols].all(), "episodes where C paused:",
              int(checks["c_paused"].sum()), "of", len(checks))  # fmt: skip
        print(f"debug run: {len(episodes)} episodes; outcomes not analysed")
        if not checks[cols].all().all():
            raise SystemExit("EQUIVALENCE CHECK FAILED")
        return

    out, results = resolve(config.output), Path("results") / config.name
    if not args.analyze_only:
        if _git("status", "--porcelain"):
            raise SystemExit("refusing the confirmation with a dirty tree")
        if out.exists():
            raise SystemExit(f"{out} exists: the confirmation is one-shot")
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
    _, summary = analyze_existing(out, lambda e, s: derive_g4(e, s, config), results, DERIVED,
                                  RAW)  # fmt: skip
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
