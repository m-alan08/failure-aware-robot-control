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
import torch

from farc.delaychange.heldout import _setup, load_g4
from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
from farc.disturbances.config import ConditionSpec
from farc.residual.base_core import SACBaseCore
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


def preds_state(preds) -> dict:
    """Every tensor reachable from the predictor closures (the F1 model's parameters)."""
    out = {}
    for name, fn in preds.items():
        for i, cell in enumerate(getattr(fn, "__closure__", None) or ()):
            obj = cell.cell_contents
            if isinstance(obj, torch.nn.Module):
                out.update({f"{name}.{i}.{k}": v for k, v in obj.state_dict().items()})
            for field in getattr(obj, "__dataclass_fields__", {}):  # e.g. DynamicsModel
                v = getattr(obj, field)
                if isinstance(v, np.ndarray):
                    out[f"{name}.{i}.{field}"] = torch.from_numpy(np.array(v, copy=True))
                elif isinstance(v, torch.Tensor):
                    out[f"{name}.{i}.{field}"] = v
                elif isinstance(v, torch.nn.Module):  # the F1 network itself
                    out.update({f"{name}.{i}.{field}.{k}": t for k, t in v.state_dict().items()})
                elif isinstance(v, (list, tuple)):
                    for j, x in enumerate(v):
                        if isinstance(x, np.ndarray):
                            out[f"{name}.{i}.{field}.{j}"] = torch.from_numpy(np.array(x))
    return out


def build(cfg: TrainConfig, plan: dict):
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
        if cfg.arm == "B2":  # residual on frozen SAC: no Candidate C at all
            return SACBaseCore(policy, normalizer)
        return CandidateCCore(policy, normalizer, preds["learned"], g4.estimator, g4.constants)

    from farc.residual.observation import ResidualObservationBuilder

    spec = _spec(cfg.arm)
    # The draft's literal T-M friction semantics unless an approved amendment sets the key
    # (docs/RESIDUAL_RL_PROTOCOL_REVIEW.md, T-M-2).
    friction_mode = plan.get("tm_friction_mode", "object_geom")
    factory = TrainingSessionFactory(base, cfg.shares or ARM_SHARES[cfg.arm], pseed,
                                     friction_mode)  # fmt: skip
    controller = ResidualController(core(), alpha=cfg.alpha_start,
                                    builder=ResidualObservationBuilder(spec))  # fmt: skip
    renv = ResidualEnv(factory, controller)
    frozen = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    model_params = {k: v.clone() for k, v in preds_state(preds).items()}

    def frozen_check() -> None:
        """The frozen SAC actor and the F1 predictor must be bitwise unchanged."""
        for k, v in loaded.actor.state_dict().items():
            if not torch.equal(v, frozen[k]):
                raise RuntimeError(f"the frozen SAC actor changed ({k})")
        for k, v in preds_state(preds).items():
            if not torch.equal(v, model_params[k]):
                raise RuntimeError(f"the frozen F1 predictor changed ({k})")

    renv.frozen_check = frozen_check  # type: ignore[attr-defined]

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
    p.add_argument("--resume", action="store_true", help="continue from OUT/resume.pt")
    a = p.parse_args()
    plan = load_plan_config(CONFIG_PATH)
    if a.smoke:
        cfg = smoke_config(arm=a.arm, alpha_target=a.alpha, network_seed=a.network_seed,
                           policy_index=a.network_seed % 3, env_steps=a.steps,
                           eval_interval=max(1, a.steps // 2))  # fmt: skip
    else:
        from farc.residual.training import registered_training_seeds

        mlo, mhi = R_SEED_BLOCKS["training_monitor"]  # amendment R-A1, SEED-2
        cfg = TrainConfig(arm=a.arm, alpha_target=a.alpha, network_seed=a.network_seed,
                          policy_index=a.network_seed % 3, env_steps=plan["env_steps"],
                          target_entropy=plan["target_entropy"],
                          rollback_drop=plan["rollback_drop"],
                          episode_seeds=tuple(registered_training_seeds()),
                          nominal_eval_seeds=tuple(range(mlo, mhi)))  # fmt: skip
    check_run(cfg, plan)  # refuses before any simulator work
    renv, nominal_eval, c_nominal, factory, cfg = build(cfg, plan)
    c = c_nominal()
    result = train(cfg, renv, nominal_eval, c, a.out, plan, resume=a.resume)
    renv.frozen_check()  # raises if the frozen SAC actor or F1 predictor changed
    # Draws are pure functions of (seed, shares, generator seed): regenerate them so the log
    # is complete for resumed runs too.
    from farc.residual.training_env import draw_training

    shares = cfg.shares or ARM_SHARES[cfg.arm]
    with open(f"{a.out}/episodes.jsonl", "w") as f:
        for seed in cfg.episode_seeds[: result.episodes]:
            d = draw_training(seed, shares, plan["generator_seed"])
            f.write(json.dumps(d.to_dict()) + "\n")
    del factory
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
