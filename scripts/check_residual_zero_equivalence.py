"""Real-simulator zero-residual equivalence (docs/RESIDUAL_RL_PROPOSAL.md, section 6.1).

Code check only, on the G4 DEBUGGING seeds 3,690,000-009 (``load_g4(role="debugging")``):
for every G4 cell x frozen SAC policy x debugging seed, closed-loop Candidate C
(``mechanism.run_episode`` with ``sequential``, exactly G4's ``C``) is compared with the
residual controller in two zero modes:

- ``zero``: ``ZeroResidual`` at alpha = 0;
- ``actor``: an untrained, zero-initialised ``ResidualActor`` at alpha = 0.2.

Pass = issued commands bitwise identical, and the same delay used and alarms at every step.
Outcomes (success) are recorded but not analysed. No training, no held-out seed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from multiprocessing import Pool
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.estimator import ActuatorEstimator
from farc.delaychange.heldout import _session, _setup, g4_cells, load_g4
from farc.delaychange.mechanism import run_episode, sequential
from farc.residual.controller import ResidualController, run_residual_episode
from farc.residual.core import CandidateCCore
from farc.residual.observation import ObservationSpec
from farc.residual.policy import ResidualActor, ZeroResidual

CONFIG = "configs/delaychange/g4_heldout.json"


def check(args) -> list[dict[str, Any]]:
    config, checkpoint, cell = args
    loaded, policy, normalizer, env, est_cfg, preds = _setup(config, checkpoint)
    actor_state = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    torch.manual_seed(0)
    modes = {"zero": (ZeroResidual(), 0.0),
             "actor": (ResidualActor(ObservationSpec().dim), 0.2)}  # fmt: skip
    out = []
    for seed in config.seeds:
        s = _session(env, policy, normalizer, cell, seed, config)
        ref = run_episode(s, ActuatorEstimator(est_cfg), sequential(config.constants), preds)
        ref_cmd = np.array(s.commands)
        row: dict[str, Any] = {"arm": cell.arm, "condition": cell.name, "family": cell.family,
                               "policy_seed": loaded.training_seed, "episode_seed": seed,
                               "c_active_steps": int(sum(r["active"] for r in ref["rows"])),
                               "c_alarms": int(sum(r["alarm"] for r in ref["rows"])),
                               "c_success": bool(ref["success"])}  # fmt: skip
        for name, (pol, alpha) in modes.items():
            core = CandidateCCore(policy, normalizer, preds["learned"], config.estimator,
                                  config.constants)  # fmt: skip
            s = _session(env, policy, normalizer, cell, seed, config)
            res = run_residual_episode(s, ResidualController(core, pol, alpha))
            row[f"{name}_commands_identical"] = bool(np.array_equal(np.array(s.commands),
                                                                    ref_cmd))  # fmt: skip
            row[f"{name}_decisions_identical"] = all(
                a["delay_used"] == b["delay_used"] and a["alarm"] == b["alarm"]
                and a["active"] == b["active"] and a["suspended"] == b["suspended"]
                for a, b in zip(res["rows"], ref["rows"], strict=True))  # fmt: skip
            row[f"{name}_max_residual"] = max(r["residual_norm"] for r in res["rows"])
            row[f"{name}_success"] = bool(res["success"])
        out.append(row)
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor_state[k]):
            raise RuntimeError("the frozen actor changed")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processes", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--out", default="results/residual_zero_equivalence")
    args = parser.parse_args()
    config = load_g4(CONFIG, "debugging")
    assert config.seeds == list(range(3_690_000, 3_690_010))
    jobs = [(config, c, cell) for c in config.checkpoints for cell in g4_cells()]
    with Pool(args.processes) as pool:
        df = pd.DataFrame([r for p in pool.map(check, jobs) for r in p])
    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "equivalence.csv"), index=False)
    flags = [c for c in df.columns if c.endswith("_identical")]
    summary = {
        "role": "debugging", "seeds": [3_690_000, 3_690_009],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip(),
        "checkpoints": list(config.checkpoints),
        "cells": int(df["condition"].nunique()), "episodes_compared": len(df),
        "episodes_with_c_active": int((df["c_active_steps"] > 0).sum()),
        "episodes_with_c_alarm": int((df["c_alarms"] > 0).sum()),
        **{f: int(df[f].sum()) for f in flags},
        "max_residual_norm": float(df[["zero_max_residual", "actor_max_residual"]].max().max()),
        "all_identical": bool(df[flags].all().all()),
        "note": "code check on debugging seeds; success outcomes not analysed",
    }  # fmt: skip
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if not summary["all_identical"]:
        raise SystemExit("ZERO-RESIDUAL EQUIVALENCE FAILED")


if __name__ == "__main__":
    main()
