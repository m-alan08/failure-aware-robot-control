"""Phase E2: pre-registered evaluation of online delay compensation.

``--debug`` runs the debugging seeds into a separate output directory (code checks only; the
pre-registration forbids analysing those outcomes). The evaluation run writes its raw files
once; ``--analyze-only`` rewrites only derived tables (``farc.counterfactual.artifacts``).
"""

from __future__ import annotations

import argparse
import subprocess
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from farc.adaptation.delay_eval import (
    DERIVED_TABLES,
    derive_tables,
    load_delay_evaluation,
    run_policy,
)
from farc.counterfactual.artifacts import METADATA, analyze_existing, publish_metadata, write_raw
from farc.paths import resolve
from farc.prediction.evaluation import file_sha256

RAW = ("episodes.csv.gz", "steps.csv.gz", METADATA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adaptation/phase_e2_delay.json")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--debug", action="store_true", help="debugging seeds, separate output")
    args = parser.parse_args()
    config = load_delay_evaluation(args.config)  # refuses a changed pre-registration
    if args.debug:
        config = replace(config, debug_run=True, name=config.name + "_debug",
                         output=config.output + "_debug")  # fmt: skip
    out = resolve(config.output)
    results = Path("results") / config.name

    if not args.analyze_only:
        if (out / RAW[0]).exists():
            raise SystemExit(f"{out} already holds raw results; use --analyze-only")
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts",
                                "configs", config.plan], capture_output=True, text=True,
                               check=False).stdout.strip()  # fmt: skip
        if dirty and not args.debug:
            raise SystemExit(f"uncommitted changes to code, config or plan:\n{dirty}")
        started = time.time()
        with Pool(len(config.checkpoints)) as pool:
            parts = pool.map(run_policy, [(config, c) for c in config.checkpoints])
        episodes = pd.DataFrame([r for part in parts for r in part[0]])
        steps = pd.DataFrame([r for part in parts for r in part[1]])
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                check=False).stdout.strip()  # fmt: skip
        write_raw(out, {"episodes.csv.gz": episodes, "steps.csv.gz": steps}, {
            "config": asdict(config), "config_file": args.config,
            "config_sha256": file_sha256(Path(args.config)),
            "plan_sha256": config.plan_sha256, "git_commit": commit,
            "checkpoint_sha256": {c: file_sha256(resolve(c)) for c in config.checkpoints},
            "seeds": config.seeds, "debug_run": config.debug_run,
            "episodes_rows": len(episodes), "step_rows": len(steps),
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "seconds": round(time.time() - started, 1),
        })  # fmt: skip
        if args.debug:
            print(f"debug run written to {out}; outcomes are not analysed (pre-registration)")
            return
        publish_metadata(out, results)

    tables, summary = analyze_existing(
        out, lambda e, s: derive_tables(e, s, config), results, DERIVED_TABLES, RAW
    )
    print(summary)
    print(tables["hypotheses"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
