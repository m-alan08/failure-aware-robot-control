"""Summarise a residual-RL training log by step window (SAC-5 development check and
registered-run monitoring). Reads OUT/log.jsonl only.

    PYTHONPATH=src python scripts/residual_dev_report.py results/residual_dev_check/R_a0.2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from farc.residual.training_env import SHARES, draw_training


def summarise(run: Path, window: int = 20_000, generator_seed: int = 20261900) -> dict:
    rows = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    ep = pd.DataFrame([r for r in rows if r["kind"] == "episode"])
    up = pd.DataFrame([r for r in rows if r["kind"] == "update"])
    cfg = next(r for r in rows if r["kind"] == "config")
    shares = cfg.get("shares") or SHARES
    ep["family"] = [draw_training(int(s), shares, generator_seed).family for s in ep.seed]
    ep["w"] = (ep.step - 1) // window * window + window
    up["w"] = (up.step - 1) // window * window + window
    e = ep.groupby("w").agg(success=("success", "mean"), ret=("ret", "mean"),
                            delta_abs=("delta_abs", "mean"), delta_sat=("delta_sat", "mean"),
                            command_clip=("command_clip", "mean"),
                            correction_abs=("correction_abs", "mean"),
                            alpha=("alpha", "last"))  # fmt: skip
    u = up.groupby("w").agg(entropy_coef=("entropy_coef", "last"), entropy=("entropy", "mean"),
                            critic_loss=("critic_loss", "mean"),
                            critic_loss_max=("critic_loss", "max"), q_mean=("q_mean", "mean"),
                            actor_loss=("actor_loss", "mean"))  # fmt: skip
    fam = ep.pivot_table(index="w", columns="family", values="success", aggfunc="mean")
    evals = [r for r in rows if r["kind"] in ("nominal_eval", "rollback", "resume", "end")]
    return {"config": {k: cfg[k] for k in ("arm", "alpha_target", "target_entropy",
                                           "initial_entropy_coef", "hidden", "batch_size")},
            "windows": e.join(u).round(4).reset_index().to_dict(orient="records"),
            "family_success": fam.round(3).reset_index().to_dict(orient="records"),
            "events": evals}  # fmt: skip


if __name__ == "__main__":
    run = Path(sys.argv[1])
    out = summarise(run)
    (run / "report.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(out["windows"]).to_string(index=False))
    print(pd.DataFrame(out["family_success"]).to_string(index=False))
    for x in out["events"]:
        print(x)
