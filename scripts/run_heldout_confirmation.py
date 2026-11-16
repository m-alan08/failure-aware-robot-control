"""One-shot held-out confirmation of Phases E and F1 (docs/HELDOUT_CONFIRMATION_PLAN.md).

- ``--debug``: development seeds 3,655,000-009 into a separate directory. Code checks only;
  outcomes are not analysed.
- Default: the approved held-out slice 4,000,000-4,000,299, run once. The runner refuses if the
  plan changed, the approval is missing, the git tree is dirty, or the output directory
  exists. It writes LOCK.json before the first episode.
- ``--restart-after-crash REASON``: allowed only if LOCK.json exists and no raw outcome file
  was written. The crash is appended to the lock record.
- ``--analyze-only``: re-derives tables from the raw files, which are never modified.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from farc.contact.heldout import DERIVED_TABLES, derive_heldout, load_heldout, run_policy
from farc.counterfactual.artifacts import METADATA, analyze_existing, publish_metadata, write_raw
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256

RAW = ("episodes.csv.gz", "steps.csv.gz", METADATA)
CONFIG = "configs/contact/heldout_confirmation.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--restart-after-crash", metavar="REASON")
    args = parser.parse_args()
    spec = load_heldout(args.config, debug=args.debug)
    if args.debug:
        spec = replace(spec, contact=replace(spec.contact, name=spec.contact.name + "_debug",
                                             output=spec.contact.output + "_debug"))  # fmt: skip
    out = resolve(spec.contact.output)
    results = Path("results") / spec.contact.name
    lock = out / "LOCK.json"

    if not args.analyze_only:
        if not args.debug:
            dirty = _git("status", "--porcelain")
            if dirty:
                raise SystemExit(f"refusing the held-out run with a dirty tree:\n{dirty}")
            if args.restart_after_crash:
                if not lock.exists() or (out / RAW[0]).exists():
                    raise SystemExit("a restart needs a lock record and no written outcomes")
                record = json.loads(lock.read_text())
                record.setdefault("crash_restarts", []).append(
                    {"reason": args.restart_after_crash, "commit": _git("rev-parse", "HEAD"),
                     "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})  # fmt: skip
            elif out.exists():
                raise SystemExit(f"{out} exists: the held-out run is one-shot")
            else:
                record = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                          "commit": _git("rev-parse", "HEAD"),
                          "plan_sha256": spec.contact.plan_sha256,
                          "approval": spec.heldout_approved,
                          "seeds": [min(spec.contact.seeds), max(spec.contact.seeds)]}  # fmt: skip
            out.mkdir(parents=True, exist_ok=True)
            lock.write_text(json.dumps(record, indent=2) + "\n")
        elif (out / RAW[0]).exists():
            raise SystemExit(f"{out} already holds a debug run")
        started = time.time()
        with Pool(len(spec.contact.checkpoints)) as pool:
            parts = pool.map(run_policy, [(spec, c) for c in spec.contact.checkpoints])
        episodes = pd.DataFrame([r for part in parts for r in part[0]])
        steps = pd.DataFrame([r for part in parts for r in part[1]])
        write_raw(out, {RAW[0]: episodes, RAW[1]: steps}, {
            "config": asdict(spec), "config_file": args.config,
            "config_sha256": file_sha256(Path(args.config)),
            "plan_sha256": spec.contact.plan_sha256, "model_sha256": spec.contact.model_sha256,
            "approval": spec.heldout_approved, "git_commit": _git("rev-parse", "HEAD"),
            "checkpoint_sha256": {c: file_sha256(resolve(c)) for c in spec.contact.checkpoints},
            "seeds": spec.contact.seeds, "debug_run": args.debug,
            "episodes_rows": len(episodes), "step_rows": len(steps),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds": round(time.time() - started, 1),
        })  # fmt: skip
        if args.debug:
            print(f"debug run written to {out}; outcomes are not analysed")
            return
        publish_metadata(out, results)

    tables, summary = analyze_existing(
        out, lambda e, s: derive_heldout(e, s, spec), results, DERIVED_TABLES, RAW
    )
    print(json.dumps(summary, indent=2, default=str))
    print(tables["gatekeeping"].round(4).to_string(index=False))
    print(tables["safety_gates"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
