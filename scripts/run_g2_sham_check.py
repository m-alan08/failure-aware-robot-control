"""G2 exploratory sham check (docs/PHASE_G2_SHAM_CHECK.md): C against a random-timing sham on
the exploratory seeds 3,674,000-099. One run; refuses a dirty tree or an existing output.
``--analyze-only`` re-derives the tables from the write-once raw files."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from farc.counterfactual.artifacts import METADATA, analyze_existing, publish_metadata, write_raw
from farc.delaychange.sham_check import (
    CONFIGURATIONS,
    DERIVED,
    check_cells,
    check_config,
    derive_check,
    exploratory_seeds,
    run_task,
)
from farc.delaychange.stage3 import load_g2
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256

CONFIG = "configs/delaychange/g2_stage3.json"
PLAN = "docs/PHASE_G2_SHAM_CHECK.md"
RAW = ("episodes.csv.gz", "steps.csv.gz", METADATA)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = parser.parse_args()
    config = check_config(load_g2(CONFIG, "evaluation"))  # pinned constants, verified
    out, results = resolve(config.output), Path("results") / "phase_g2_sham_check"
    if not args.analyze_only:
        if _git("status", "--porcelain"):
            raise SystemExit("refusing the check with a dirty tree")
        if out.exists():
            raise SystemExit(f"{out} exists: the check is one-shot")
        out.mkdir(parents=True)
        meta = {"plan": PLAN, "plan_sha256": file_sha256(Path(PLAN)),
                "config_sha256": file_sha256(Path(CONFIG)), "git_commit": _git("rev-parse", "HEAD"),
                "role": "exploratory", "seeds": exploratory_seeds(),
                "configurations": CONFIGURATIONS, "constants": config.constants,
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}  # fmt: skip
        (out / "LOCK.json").write_text(json.dumps(meta, indent=2) + "\n")
        jobs = [(config, c, cell) for c in config.checkpoints for cell in check_cells()]
        started = time.time()
        with Pool(args.processes) as pool:
            parts = pool.map(run_task, jobs)
        episodes = pd.DataFrame([r for p in parts for r in p[0]])
        steps = pd.DataFrame([r for p in parts for r in p[1]])
        write_raw(out, {RAW[0]: episodes, RAW[1]: steps},
                  {**meta, "episodes_rows": len(episodes), "step_rows": len(steps),
                   "seconds": round(time.time() - started, 1)})  # fmt: skip
        publish_metadata(out, results)
    _, summary = analyze_existing(out, lambda e, s: derive_check(e, s, config), results,
                                  DERIVED, RAW)  # fmt: skip
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
