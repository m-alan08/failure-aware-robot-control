"""Throughput benchmark for residual-RL training at the registered network and batch sizes.

Runs a short smoke-mode job (unregistered SMOKE_SEEDS only) with the §4 sizes: 256x256
networks, batch 256, one update per environment step, and the critic-only warm-up shortened
so the timed window covers full actor+critic updates. The numbers are compute estimates,
not results.

    PYTHONPATH=scripts python scripts/residual_benchmark.py --steps 5000 \
        --out results/residual_stage_a/bench_R
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import residual_train as rt  # run from the repo root with PYTHONPATH=scripts

from farc.residual.training import load_plan_config, smoke_config, train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--arm", default="R")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plan = load_plan_config(rt.CONFIG_PATH)
    episodes = -(-a.steps // 50) + 2
    cfg = smoke_config(arm=a.arm, env_steps=a.steps, hidden=(256, 256), batch_size=256,
                       replay_size=100_000, zero_data_steps=1000, critic_only_updates=500,
                       alpha_ramp_steps=max(1, a.steps // 2), eval_interval=a.steps,
                       log_interval=500, episode_seeds=tuple(range(1000, 1000 + episodes)),
                       nominal_eval_seeds=tuple(range(100, 120)))  # fmt: skip
    rt.check_run(cfg, plan)
    renv, nominal_eval, c_nominal, _factory, cfg = rt.build(cfg, plan)
    c = c_nominal()
    t0 = time.perf_counter()
    result = train(cfg, renv, nominal_eval, c, a.out, plan)
    dt = time.perf_counter() - t0
    sps = cfg.env_steps / dt
    est_h = 1_000_000 / sps / 3600
    out = {
        "arm": a.arm,
        "env_steps": cfg.env_steps,
        "seconds": dt,
        "steps_per_second": sps,
        "hours_per_1M_step_run": est_h,
        "status": result.status,
        "note": "includes 20 nominal-eval episodes and shortened warm-up; smoke seeds only",
    }
    Path(a.out, "benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
