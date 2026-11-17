"""G2 Stage 1: mechanism isolation (docs/PHASE_G2_PROPOSAL.md, section 3).

- ``--debug``: debugging seeds 3,670,000-009; code checks only, outcomes not analysed.
- Default: the mechanism block 3,671,000-049, run once (refuses a dirty tree, a changed plan
  or an existing output directory), followed by the point-of-no-return branches on the
  episodes where frozen SAC succeeds and the deployed compensator fails.
- ``--analyze-only``: re-derive the tables from the write-once raw files.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from farc.counterfactual.artifacts import METADATA, analyze_existing, publish_metadata, write_raw
from farc.delaychange.stage1 import (
    DERIVED,
    branch_targets,
    derive_branches,
    derive_stage1,
    load_stage1,
    run_branch_task,
    run_task,
    stage1_cells,
)
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256

CONFIG = "configs/delaychange/g2_stage1.json"
RAW = ("episodes.csv.gz", "steps.csv.gz", "branches.csv.gz", METADATA)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    config = load_stage1(args.config, debug=args.debug)
    if args.debug:
        config = replace(config, name=config.name + "_debug", output=config.output + "_debug")
    out = resolve(config.output)
    results = Path("results") / config.name

    if not args.analyze_only:
        if not args.debug and _git("status", "--porcelain"):
            raise SystemExit("refusing the Stage-1 run with a dirty tree")
        if out.exists():
            raise SystemExit(f"{out} exists: Stage 1 is one-shot")
        out.mkdir(parents=True)
        meta = {"config_file": args.config, "config_sha256": file_sha256(Path(args.config)),
                "plan_sha256": config.plan_sha256, "model_sha256": config.model_sha256,
                "git_commit": _git("rev-parse", "HEAD"), "debug_run": args.debug,
                "seeds": config.seeds, "config": asdict(config),
                "checkpoint_sha256": {c: file_sha256(resolve(c)) for c in config.checkpoints},
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
        (out / "LOCK.json").write_text(json.dumps(meta, indent=2) + "\n")
        started = time.time()
        jobs = [(config, c, cell.name) for c in config.checkpoints for cell in stage1_cells()]
        with Pool(args.processes) as pool:
            parts = pool.map(run_task, jobs)
        episodes = pd.DataFrame([r for p in parts for r in p[0]])
        steps = pd.DataFrame([r for p in parts for r in p[1]])
        targets = branch_targets(episodes)
        by_policy: dict[int, list[tuple[str, int]]] = defaultdict(list)
        for pol, cond, seed in targets[["policy_seed", "condition", "episode_seed"]].to_numpy():
            by_policy[int(pol)].append((str(cond), int(seed)))
        checkpoint_of = {int(Path(c).parent.name.rsplit("seed", 1)[1]): c
                         for c in config.checkpoints}  # fmt: skip
        branch_jobs = [(config, checkpoint_of[p], chunk[i::args.processes])
                       for p, chunk in by_policy.items() for i in range(args.processes)
                       if chunk[i::args.processes]]  # fmt: skip
        with Pool(args.processes) as pool:
            branch_parts = pool.map(run_branch_task, branch_jobs)
        branches = pd.DataFrame(
            [r for p in branch_parts for r in p],
            columns=[*episodes.columns, "switch_step", "end_step"],
        )
        write_raw(out, {RAW[0]: episodes, RAW[1]: steps, RAW[2]: branches},
                  {**meta, "episodes_rows": len(episodes), "step_rows": len(steps),
                   "branch_rows": len(branches), "branch_targets": len(targets),
                   "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "seconds": round(time.time() - started, 1)})  # fmt: skip
        if args.debug:
            print(f"debug run: {len(episodes)} episodes, {len(steps)} steps, "
                  f"{len(branches)} branch rollouts; outcomes not analysed")  # fmt: skip
            return
        publish_metadata(out, results)

    def derive(e, s, b):
        tables, summary = derive_stage1(e, s, config)
        tables["branches"] = derive_branches(b) if len(b) else pd.DataFrame()
        return tables, summary

    tables, summary = analyze_existing(out, derive, results, (*DERIVED, "branches"), RAW)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
