"""Phase J-1 privileged-input ablation (docs/PHASE_J1_PROTOCOL.md). Evaluation only.

- ``--debug``: debugging seeds 3,700,200-209; code and integrity checks only.
- Default: evaluation seeds 3,703,000-299, run once. Refuses a dirty tree or an existing output.
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
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256
from farc.retrain.ablation_j1 import DERIVED, derive_j1, j1_cells, load_j1, run_task

CONFIG = "configs/retrain/phase_j1.json"
RAW = ("episodes.csv.gz", METADATA)


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--debug", action="store_true")
    mode.add_argument("--analyze-only", action="store_true")
    p.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = p.parse_args()
    c = load_j1(CONFIG)
    role = "debugging" if a.debug else "evaluation"
    seeds = tuple(c.seeds(role))
    out = resolve(c.raw["output"] + ("_debug" if a.debug else ""))
    results = Path("results/phase_j1_ablation")
    root = str(resolve("$FARC_DATA_DIR/models/retrained_phase_i"))
    if not a.analyze_only:
        if not a.debug and _git("status", "--porcelain"):
            raise SystemExit("refusing the run with a dirty tree")
        if out.exists():
            raise SystemExit(f"{out} exists: the run is one-shot")
        out.mkdir(parents=True)
        meta = {"protocol_sha256": c.raw["protocol_sha256"],
                "config_sha256": file_sha256(Path(CONFIG)),
                "git_commit": _git("rev-parse", "HEAD"), "role": role,
                "seeds": [min(seeds), max(seeds)]}  # fmt: skip
        if not a.debug:
            (out / "LOCK.json").write_text(json.dumps(meta, indent=2) + "\n")
        jobs = [(c, root, k, cell, str(out), seeds) for k in range(3) for cell in j1_cells()]
        started = time.time()
        with Pool(a.processes) as pool:
            parts = pool.map(run_task, jobs)
        e = pd.DataFrame([r for part in parts for r in part[0]])
        integrity = [r for part in parts for r in part[1]]
        ok = all(r["identical"] for r in integrity)
        write_raw(out, {RAW[0]: e}, {**meta, "episodes_rows": len(e), "integrity": integrity,
                                     "integrity_passed": ok,
                                     "seconds": round(time.time() - started, 1)})  # fmt: skip
        print(f"{role}: {len(e)} episodes; integrity {'PASSED' if ok else 'FAILED'} "
              f"({len(integrity)} checks)")  # fmt: skip
        if not ok:
            raise SystemExit("INTEGRITY CHECK FAILED: analysis stopped (protocol)")
        if a.debug:
            print(e.groupby("configuration").size().to_string())
            return
        publish_metadata(out, results)
    _, summary = analyze_existing(out, lambda e: derive_j1(e), results, DERIVED, RAW)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
