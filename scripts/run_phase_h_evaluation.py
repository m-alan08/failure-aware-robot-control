"""Phase H Stage-2 evaluation (docs/PHASE_H_RETRAINED_BASELINE_PROPOSAL.md, sections 3.4-4).

- ``--debug --retrained-root <DIR>``: debugging seeds 3,700,000-009 with unregistered (Stage-0)
  checkpoints; code checks only, outcomes not analysed.
- Default: evaluation seeds 3,701,000-299, run once, with checkpoints registered in
  ``src/farc/retrain/checkpoints.json``. Refuses a dirty tree or an existing output.
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
from farc.retrain.evaluation import (
    DERIVED,
    Context,
    derive_h,
    h_cells,
    retrained_paths,
    run_task,
)
from farc.retrain.phase_h import load_phase_h, manifest

CONFIG = "configs/retrain/phase_h.json"
G4_CONFIG = "configs/delaychange/g4_heldout.json"
RAW = ("episodes.csv.gz", "steps.csv.gz", METADATA)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    p.add_argument("--retrained-root", default=None, help="debug only")
    p.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = p.parse_args()
    c = load_phase_h(CONFIG)
    g4 = load_g4(G4_CONFIG, "debugging")  # frozen checkpoints, F1 model, estimator, constants
    role = "debugging" if a.debug else "evaluation"
    root = a.retrained_root if a.debug and a.retrained_root else str(resolve(c.output))
    suffix = "_stage0" if a.debug and a.retrained_root else ""
    ctx = Context(frozen_checkpoints=g4.checkpoints, retrained=retrained_paths(root, suffix),
                  verify_retrained=not a.debug, model=g4.model, model_sha256=g4.model_sha256,
                  estimator=g4.estimator, constants=g4.constants,
                  seeds=tuple(c.seeds(role)))  # fmt: skip
    out = resolve("$FARC_DATA_DIR/experiments/phase_h_evaluation" + ("_debug" if a.debug else ""))
    results = Path("results/phase_h_evaluation")
    meta = {"plan_sha256": c.plan_sha256, "config_sha256": file_sha256(Path(CONFIG)),
            "git_commit": _git("rev-parse", "HEAD"), "role": role,
            "seeds": [min(ctx.seeds), max(ctx.seeds)],
            "retrained_manifest": "debug (unregistered)" if a.debug else manifest()}  # fmt: skip
    jobs = [(ctx, k, cell) for k in range(3) for cell in h_cells()]
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
        e = pd.DataFrame([r for p_ in parts for r in p_[0]])
        s = pd.DataFrame([r for p_ in parts for r in p_[1]])
        write_raw(out, {RAW[0]: e, RAW[1]: s},
                  {**meta, "episodes_rows": len(e), "step_rows": len(s),
                   "seconds": round(time.time() - started, 1)})  # fmt: skip
        if a.debug:
            print(f"debug run: {len(e)} episodes; outcomes not analysed")
            print(e.groupby("configuration").size().to_string())
            return
        publish_metadata(out, results)
    _, summary = analyze_existing(out, lambda e, s: derive_h(e, s), results, DERIVED, RAW)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
