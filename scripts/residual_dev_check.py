"""SAC-5 bounded development check (amendment R-A2): one residual-RL run at the REGISTERED
network size, batch, warm-up, critic-only updates, alpha ramp, eval interval and SAC settings,
on unregistered SMOKE seeds only. Software and stability check; not a result.

    PYTHONPATH=scripts python scripts/residual_dev_check.py --alpha 0.2 --steps 200000 \\
        --out results/residual_dev_check/R_a0.2
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import residual_train as rt

from farc.residual.training import TrainConfig, load_plan_config, train

TRAIN_SEEDS = (10_000, 20_000)  # unregistered (SMOKE_SEEDS = [0, 100,000))
MONITOR_SEEDS = (200, 300)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--network-seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    plan = load_plan_config(rt.CONFIG_PATH)
    reg = TrainConfig(arm="R", alpha_target=a.alpha, network_seed=a.network_seed,
                      policy_index=a.network_seed % 3)  # registered defaults  # fmt: skip
    cfg = replace(reg, mode="smoke", env_steps=a.steps, target_entropy=plan["target_entropy"],
                  rollback_drop=plan["rollback_drop"],
                  episode_seeds=tuple(range(*TRAIN_SEEDS)),
                  nominal_eval_seeds=tuple(range(*MONITOR_SEEDS)))  # fmt: skip
    rt.check_run(cfg, plan)
    renv, nominal_eval, c_nominal, _f, cfg = rt.build(cfg, plan)
    c = c_nominal()
    t0 = time.perf_counter()
    r = train(cfg, renv, nominal_eval, c, a.out, plan, resume=a.resume)
    dt = time.perf_counter() - t0
    out = {
        "status": r.status,
        "steps": r.steps,
        "rollbacks": r.rollbacks,
        "seconds": dt,
        "c_nominal_success": c,
        "note": "smoke seeds only; software/stability check",
    }
    Path(a.out, "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
