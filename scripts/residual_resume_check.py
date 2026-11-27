"""Real-simulator resume identity (amendment R-A2): an interrupted-and-resumed run must match an
uninterrupted run bitwise (weights, optimiser-visible state, replay RNG, seed cursor, log).
Registered network sizes, short schedule, unregistered smoke seeds. Software check only.

    PYTHONPATH=src:scripts python scripts/residual_resume_check.py \
        --out results/residual_resume_check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import residual_train as rt
import torch

from farc.residual.training import load_plan_config, smoke_config, train


class Interrupt(Exception):
    pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plan = load_plan_config(rt.CONFIG_PATH)
    cfg = smoke_config(env_steps=1500, hidden=(256, 256), batch_size=256, replay_size=100_000,
                       zero_data_steps=300, critic_only_updates=100, alpha_ramp_steps=1000,
                       eval_interval=500, log_interval=50, episode_seeds=tuple(range(500, 531)),
                       nominal_eval_seeds=tuple(range(100, 110)))  # fmt: skip
    out = Path(a.out)
    results = {}
    for name in ("full", "cut"):
        renv, nominal_eval, c_nominal, _f, run_cfg = rt.build(cfg, plan)
        c = c_nominal()
        calls = {"n": 0}

        def ev(agent, alpha, name=name, nominal_eval=nominal_eval, calls=calls):
            calls["n"] += 1
            if name == "cut" and calls["n"] == 2:
                raise Interrupt  # simulated crash during the step-1000 check
            return nominal_eval(agent, alpha)

        try:
            results[name] = train(run_cfg, renv, ev, c, out / name, plan)
        except Interrupt:
            renv, nominal_eval, c_nominal, _f, run_cfg = rt.build(cfg, plan)  # fresh process state
            results[name] = train(run_cfg, renv, nominal_eval, c_nominal(), out / name, plan,
                                  resume=True)  # fmt: skip
    fa = torch.load(results["full"].final_checkpoint, weights_only=False)
    fb = torch.load(results["cut"].final_checkpoint, weights_only=False)
    same = {}
    for k in ("actor", "critic", "critic_target"):
        same[k] = all(
            torch.equal(x, y)
            for x, y in zip(fa["agent"][k].values(), fb["agent"][k].values(), strict=True)
        )
    same["log_entropy_coef"] = torch.equal(fa["agent"]["log_entropy_coef"],
                                           fb["agent"]["log_entropy_coef"])  # fmt: skip
    same["replay_rng"] = fa["replay_rng"] == fb["replay_rng"]
    same["seeds_used"] = fa["seeds_used"] == fb["seeds_used"]
    la = Path(results["full"].log).read_text().splitlines()
    lb = [x for x in Path(results["cut"].log).read_text().splitlines() if '"resume"' not in x]
    same["log"] = la == lb
    summary = {"identical": same, "all_identical": all(same.values()),
               "resumed_from_step": 500, "steps": cfg.env_steps,
               "note": "smoke seeds 500-530, 100-109; software check only"}  # fmt: skip
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["all_identical"]:
        raise SystemExit("RESUME IDENTITY FAILED")


if __name__ == "__main__":
    main()
