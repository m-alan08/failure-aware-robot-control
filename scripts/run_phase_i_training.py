"""Phase I training (docs/PHASE_I_STRONG_BASELINE_PROPOSAL.md, sections 4-5).

    uv run python scripts/run_phase_i_training.py --arm IA --seed 0

Fine-tunes a NEW policy warm-started from a copy of the frozen ``fetch_push_fp32_seed{k}``
(the frozen file is hash-verified and only read), with the Phase I features and training delay
distribution. Refuses a dirty tree or an existing run directory. ``--steps``/``--output`` are
for Stage-0 code checks only (the run is labelled ``stage0``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from farc.paths import resolve
from farc.prediction.evaluation import file_sha256
from farc.retrain.phase_i import ARMS, env_wrapper, init_hook, load_config
from farc.retrain.qrl.config import config_from_dict
from farc.retrain.qrl.training import train

CONFIG = "configs/retrain/phase_i.json"


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--steps", type=int, default=None, help="Stage-0 only")
    p.add_argument("--output", default=None, help="Stage-0 only")
    a = p.parse_args()
    c = load_config(CONFIG)
    stage0 = a.steps is not None or a.output is not None
    if a.seed not in c.training_seeds:
        raise SystemExit(f"training seed must be one of {c.training_seeds}")
    if not stage0 and _git("status", "--porcelain"):
        raise SystemExit("refusing a training run with a dirty tree")
    values = json.loads(json.dumps(c.training))
    values["name"] = f"phase_i_{a.arm}" + ("_stage0" if stage0 else "")
    values["seed"] = a.seed
    values["output_dir"] = str(resolve(a.output or c.output))
    if a.steps is not None:
        values["train"]["total_steps"] = a.steps
        values["train"]["eval_interval"] = a.steps
    config = config_from_dict(values)
    if config.run_dir.exists():
        raise SystemExit(f"{config.run_dir} exists: training runs are write-once")
    summary = train(config, env_wrapper=env_wrapper(a.arm, a.seed, c.estimator),
                    init_hook=init_hook(a.seed, ARMS[a.arm]))  # fmt: skip
    record = {"arm": a.arm, "seed": a.seed, "stage0": stage0,
              "checkpoint_sha256": file_sha256(config.run_dir / "checkpoint.pt"),
              "git_commit_at_end": _git("rev-parse", "HEAD"),
              "config_sha256": file_sha256(Path(CONFIG)), "plan_sha256": c.plan_sha256,
              "summary": summary}  # fmt: skip
    (config.run_dir / "phase_i_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("arm", "seed", "checkpoint_sha256")}, indent=2))


if __name__ == "__main__":
    main()
