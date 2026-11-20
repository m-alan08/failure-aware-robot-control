"""Train one residual-RL run (pre-registration §4). REFUSED until the plan is approved.

    python scripts/residual_train.py --arm R --alpha 0.1 --network-seed 0 --out DIR
    python scripts/residual_train.py --smoke --steps 300 --out DIR   # unregistered seeds only

A registered run reads every pinned value from ``configs/residual/residual_rl.json`` and
``farc.residual.training.check_run`` refuses it while ``approved`` is false. ``--smoke`` runs
a tiny job on ``SMOKE_SEEDS`` (unregistered) to exercise the simulator wiring; its numbers are
not results.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from farc.delaychange.heldout import _setup, load_g4
from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
from farc.disturbances.config import ConditionSpec
from farc.residual.controller import ResidualController, ResidualEnv, run_residual_episode
from farc.residual.core import CandidateCCore
from farc.residual.observation import ObservationSpec
from farc.residual.policy import ResidualActor
from farc.residual.training import (
    CONFIG_PATH,
    TrainConfig,
    check_run,
    load_plan_config,
    smoke_config,
    train,
)
from farc.residual.training_env import SHARES, TrainingSessionFactory, check_shares, without
from farc.supervision.seeds import R_SEED_BLOCKS

G4_CONFIG = "configs/delaychange/g4_heldout.json"
ARM_SHARES = {"R": SHARES, "A1": SHARES, "B2": SHARES, "A2": without(SHARES, "T-D"),
              "A3": check_shares({"T-N": 0.5, "T-D": 0.5})}  # fmt: skip


def _spec(arm: str) -> ObservationSpec:
    # A1 and B2: task + actions + validity only (41 dims); others: full (65 dims)
    if arm in ("A1", "B2"):
        return ObservationSpec(delay=False, supervisor=False, prediction=False)
    return ObservationSpec()


def build(cfg: TrainConfig, plan: dict):
    if cfg.arm == "B2":
        raise SystemExit("B2 (residual on frozen SAC) needs a base-only core: not wired yet")
    g4 = load_g4(G4_CONFIG, "debugging")  # constants/estimator only; no G4 seed is used
    loaded, policy, normalizer, env, _, preds = _setup(g4, g4.checkpoints[cfg.policy_index])
    pseed = plan["generator_seed"]
    # Evaluation gets its own simulator instance: sharing the training env would clobber the
    # in-progress training episode (caught by the real-simulator smoke run).
    eval_env = _setup(g4, g4.checkpoints[cfg.policy_index])[3]

    def base(seed: int, condition: ConditionSpec, schedule: DelaySchedule, sim=None):
        return ScheduledDelaySession(sim or env, policy, normalizer, condition, seed, pseed,
                                     schedule, "replay_drop")  # fmt: skip

    def core():
        return CandidateCCore(policy, normalizer, preds["learned"], g4.estimator, g4.constants)

    from farc.residual.observation import ResidualObservationBuilder

    spec = _spec(cfg.arm)
    factory = TrainingSessionFactory(base, cfg.shares or ARM_SHARES[cfg.arm], pseed)
    controller = ResidualController(core(), alpha=cfg.alpha_start,
                                    builder=ResidualObservationBuilder(spec))  # fmt: skip
    renv = ResidualEnv(factory, controller)

    def nominal_eval(agent, alpha: float) -> float:
        wins = []
        for seed in cfg.nominal_eval_seeds:
            s = base(seed, ConditionSpec("nominal", ()), DelaySchedule(()), eval_env)
            ctl = ResidualController(core(), lambda x: agent.act(x, True), alpha,
                                     ResidualObservationBuilder(spec))  # fmt: skip
            wins.append(run_residual_episode(s, ctl)["success"])
        return float(np.mean(wins))

    def c_nominal() -> float:
        zero = ResidualActor(spec.dim, cfg.hidden)
        wins = []
        for seed in cfg.nominal_eval_seeds:
            s = base(seed, ConditionSpec("nominal", ()), DelaySchedule(()), eval_env)
            ctl = ResidualController(core(), zero, 0.0, ResidualObservationBuilder(spec))
            wins.append(run_residual_episode(s, ctl)["success"])
        return float(np.mean(wins))

    return renv, nominal_eval, c_nominal, factory, replace(cfg, observation_dim=spec.dim)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="R")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--network-seed", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    plan = load_plan_config(CONFIG_PATH)
    if a.smoke:
        cfg = smoke_config(arm=a.arm, alpha_target=a.alpha, network_seed=a.network_seed,
                           policy_index=a.network_seed % 3, env_steps=a.steps,
                           eval_interval=max(1, a.steps // 2))  # fmt: skip
    else:
        lo, hi = R_SEED_BLOCKS["training"]
        slo, _ = R_SEED_BLOCKS["selection"]
        cfg = TrainConfig(arm=a.arm, alpha_target=a.alpha, network_seed=a.network_seed,
                          policy_index=a.network_seed % 3, env_steps=plan["env_steps"],
                          target_entropy=plan["target_entropy"],
                          rollback_drop=plan["rollback_drop"],
                          episode_seeds=tuple(range(lo, hi)),
                          nominal_eval_seeds=tuple(range(slo, slo + 100)))  # fmt: skip
    check_run(cfg, plan)  # refuses before any simulator work
    renv, nominal_eval, c_nominal, factory, cfg = build(cfg, plan)
    c = c_nominal()
    result = train(cfg, renv, nominal_eval, c, a.out, plan)
    with open(f"{a.out}/episodes.jsonl", "w") as f:
        for d in factory.draws:
            f.write(json.dumps(d.to_dict()) + "\n")
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
