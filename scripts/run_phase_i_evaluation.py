"""Phase I Stage-2 evaluation (docs/PHASE_I_STRONG_BASELINE_PROPOSAL.md, sections 4.1 and 6).

- ``--debug --phase-i-root <DIR>``: debugging seeds 3,700,100-109 with unregistered Stage-0
  checkpoints; code checks only, outcomes not analysed.
- Default: evaluation seeds 3,702,000-299, run once, with registered checkpoints only (Phase H
  R2 from ``checkpoints.json``, Phase I from ``checkpoints_phase_i.json``). Refuses a dirty tree
  or an existing output.
- ``--analyze-only``: re-derives the tables from the write-once raw files.
"""

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
from farc.delaychange.heldout import load_g4
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256
from farc.retrain import phase_i
from farc.retrain.evaluation_i import (
    DERIVED,
    ContextI,
    cells,
    derive_i,
    phase_i_paths,
    r2_paths,
    run_task,
)
from farc.retrain.phase_h import load_phase_h

CONFIG = "configs/retrain/phase_i.json"
RAW = ("episodes.csv.gz", METADATA)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    p.add_argument("--phase-i-root", default=None, help="debug only")
    p.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = p.parse_args()
    c = phase_i.load_config(CONFIG)
    h = load_phase_h("configs/retrain/phase_h.json")
    g4 = load_g4("configs/delaychange/g4_heldout.json", "debugging")
    role = "debugging" if a.debug else "evaluation"
    root = a.phase_i_root if a.debug and a.phase_i_root else str(resolve(c.output))
    suffix = "_stage0" if a.debug and a.phase_i_root else ""
    ctx = ContextI(frozen_checkpoints=g4.checkpoints, r2_paths=r2_paths(str(resolve(h.output))),
                   phase_i_paths=phase_i_paths(root, suffix), verify=not a.debug,
                   model=g4.model, model_sha256=g4.model_sha256, estimator=c.estimator,
                   constants=g4.constants, seeds=tuple(c.seeds(role)),
                   perturbation_seed=int(c.raw["perturbation_seed"]))  # fmt: skip
    out = resolve("$FARC_DATA_DIR/experiments/phase_i_evaluation" + ("_debug" if a.debug else ""))
    results = Path("results/phase_i_evaluation")
    meta = {"plan_sha256": c.plan_sha256, "config_sha256": file_sha256(Path(CONFIG)),
            "git_commit": _git("rev-parse", "HEAD"), "role": role,
            "seeds": [min(ctx.seeds), max(ctx.seeds)],
            "phase_i_manifest": "debug" if a.debug else phase_i.manifest()}  # fmt: skip
    jobs = [(ctx, k, cell) for k in range(3) for cell in cells()]
    if not a.analyze_only:
        if not a.debug and _git("status", "--porcelain"):
            raise SystemExit("refusing the evaluation with a dirty tree")
        if out.exists():
            raise SystemExit(f"{out} exists: the evaluation is one-shot")
        out.mkdir(parents=True)
        if not a.debug:
            (out / "LOCK.json").write_text(json.dumps(meta, indent=2) + "\n")
        started = time.time()
        with Pool(a.processes) as pool:
            parts = pool.map(run_task, jobs)
        e = pd.DataFrame([r for part in parts for r in part])
        write_raw(out, {RAW[0]: e}, {**meta, "episodes_rows": len(e),
                                     "seconds": round(time.time() - started, 1)})  # fmt: skip
        if a.debug:
            print(f"debug run: {len(e)} episodes; outcomes not analysed")
            print(e.groupby("configuration").size().to_string())
            return
        publish_metadata(out, results)
    _, summary = analyze_existing(out, lambda e: derive_i(e), results, DERIVED, RAW)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
