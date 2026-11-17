"""Phase H training (docs/PHASE_H_RETRAINED_BASELINE_PROPOSAL.md, section 3.1).

    uv run python scripts/run_phase_h_training.py --arm R2 --seed 0

Trains one new SAC+HER FetchPush policy with the ported upstream trainer, under the Phase H
delay distribution. Refuses a dirty tree or an existing run directory. ``--steps`` and
``--output`` are for Stage-0 code checks only (the run is then labelled ``stage0``).
The frozen base policies are never read or written by this script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from farc.paths import resolve
from farc.prediction.evaluation import file_sha256
from farc.retrain.phase_h import ARMS, env_wrapper, load_phase_h
from farc.retrain.qrl.config import config_from_dict
from farc.retrain.qrl.training import train

CONFIG = "configs/retrain/phase_h.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--nominal",
        action="store_true",
        help="no delay, no history: the optional bit-reproducibility retrain",
    )
    parser.add_argument("--steps", type=int, default=None, help="Stage-0 only")
    parser.add_argument("--output", default=None, help="Stage-0 only")
    args = parser.parse_args()
    c = load_phase_h(CONFIG)
    stage0 = args.steps is not None or args.output is not None
    if not stage0:
        if _git("status", "--porcelain"):
            raise SystemExit("refusing a training run with a dirty tree")
        if args.seed not in c.training_seeds:
            raise SystemExit(f"training seed must be one of {c.training_seeds}")
    values = json.loads(json.dumps(c.training))
    tag = "nominal" if args.nominal else args.arm
    values["name"] = f"phase_h_{tag}" + ("_stage0" if stage0 else "")
    values["seed"] = args.seed
    values["output_dir"] = str(resolve(args.output or c.output))
    if args.steps is not None:
        values["train"]["total_steps"] = args.steps
    config = config_from_dict(values)
    if config.run_dir.exists():
        raise SystemExit(f"{config.run_dir} exists: training runs are write-once")
    wrapper = None if args.nominal else env_wrapper(args.arm, args.seed)
    summary = train(config, env_wrapper=wrapper)
    record = {"arm": tag, "seed": args.seed, "stage0": stage0,
              "checkpoint_sha256": file_sha256(config.run_dir / "checkpoint.pt"),
              "git_commit": _git("rev-parse", "HEAD"), "config_sha256": file_sha256(Path(CONFIG)),
              "plan_sha256": c.plan_sha256, "summary": summary}  # fmt: skip
    (config.run_dir / "phase_h_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("arm", "seed", "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
