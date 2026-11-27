"""Selection (§5) and development-evaluation (§6) rollouts for residual RL.

    python scripts/residual_rollouts.py --role selection --runs results/residual_rl \
        --out results/residual_rl/selection --workers 8
    python scripts/residual_rollouts.py --role evaluation --alpha-star A \
        --runs results/residual_rl --out results/residual_rl/evaluation --workers 8
    python scripts/residual_rollouts.py --role smoke ...   # unregistered seeds, software check

Every residual network is loaded from its run's ``final.pt`` (§5: the final checkpoint, no
other selection). A run that stopped by the rollback rule is listed in ``failed.json`` and its
rows enter analysis as C (``analysis.substitute_failed``). Seeds: ``episodes.check_role_seeds``
(selection: exactly the 100 selection seeds; evaluation: exactly the 600 seeds of amendment
R-A2). Registered roles need the approved, integrity-checked plan. Held-out seeds are refused
everywhere.

Configurations:
- selection: C, and R at each registered alpha (3 network seeds).
- evaluation: B0 (frozen SAC), C, R-alpha*, A1, A2, A3, B2 (3 network seeds each) on all §6
  cells. P (PRIVILEGED C: true delay and gain, amendment R-A2 / GEN-3) on the D cells only,
  written to a separate table. It is diagnostic only and never enters the analysis.
"""

from __future__ import annotations

import argparse
import json
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.estimator import ActuatorEstimator
from farc.delaychange.heldout import _setup, load_g4
from farc.delaychange.mechanism import run_episode, sequential
from farc.delaychange.schedule import ScheduledDelaySession
from farc.residual.base_core import SACBaseCore
from farc.residual.controller import ResidualController, run_residual_episode
from farc.residual.core import CandidateCCore
from farc.residual.episodes import (
    check_role_seeds,
    development_cells,
    evaluation_seeds,
    generate,
    selection_cells,
)
from farc.residual.observation import ObservationSpec, ResidualObservationBuilder
from farc.residual.policy import ResidualActor, ZeroResidual
from farc.residual.training import CONFIG_PATH, load_plan_config
from farc.supervision.seeds import R_SEED_BLOCKS

G4_CONFIG = "configs/delaychange/g4_heldout.json"
EVAL_ARMS = ("R", "A1", "A2", "A3", "B2")


def spec_for(arm: str) -> ObservationSpec:
    if arm in ("A1", "B2"):
        return ObservationSpec(delay=False, supervisor=False, prediction=False)
    return ObservationSpec()


def run_dir(runs: Path, arm: str, alpha: float, ns: int) -> Path:
    return runs / (f"R_a{alpha}_n{ns}" if arm == "R" else f"{arm}_n{ns}")


def load_actor(path: Path, arm: str) -> ResidualActor:
    state = torch.load(path / "final.pt", weights_only=False)
    cfg = state["config"]
    if cfg["arm"] != arm:
        raise RuntimeError(f"{path}: arm {cfg['arm']} != {arm}")
    actor = ResidualActor(spec_for(arm).dim, tuple(cfg["hidden"]))
    actor.load_state_dict(state["agent"]["actor"])
    actor.eval()
    return actor


def _task(args: tuple) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    (role, cell_names, policy_index, seeds, configs, alpha, runs, plan, include_p) = args
    torch.set_num_threads(1)
    g4 = load_g4(G4_CONFIG, "debugging")  # constants/estimator/checkpoints only; no G4 seed
    loaded, policy, normalizer, env, est_cfg, preds = _setup(g4, g4.checkpoints[policy_index])
    frozen = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    pseed = plan["generator_seed"]
    cells_all = selection_cells() if role == "selection" else development_cells()
    cells = [c for c in cells_all if c.name in cell_names]
    actors: dict[tuple[str, float, int], ResidualActor] = {}
    for cfg_name, nss in configs.items():
        arm, a = cfg_name.split("@") if "@" in cfg_name else (cfg_name, "nan")
        if arm in EVAL_ARMS:
            for ns in nss:
                actors[(arm, float(a), ns)] = load_actor(run_dir(runs, arm, float(a), ns), arm)

    def build(seed, condition, schedule, semantics, _policy):
        return ScheduledDelaySession(env, policy, normalizer, condition, seed, pseed, schedule,
                                     semantics)  # fmt: skip

    def c_core():
        return CandidateCCore(policy, normalizer, preds["learned"], g4.estimator, g4.constants)

    zero = ZeroResidual()

    def run(cfg_name: str, ns: int, _policy, session) -> dict[str, Any]:
        arm, a = cfg_name.split("@") if "@" in cfg_name else (cfg_name, "nan")
        if arm == "C":
            ctl = ResidualController(
                c_core(), zero, 0.0, ResidualObservationBuilder(ObservationSpec())
            )
        elif arm == "B0":  # frozen SAC: the B2 base with a zero residual
            sac: Any = SACBaseCore(policy, normalizer)
            ctl = ResidualController(sac, zero, 0.0,
                                     ResidualObservationBuilder(spec_for("B2")))  # fmt: skip
        else:
            actor = actors[(arm, float(a), ns)]
            core: Any = SACBaseCore(policy, normalizer) if arm == "B2" else c_core()

            def act(x, actor=actor):
                with torch.no_grad():
                    t = torch.as_tensor(np.asarray(x, np.float32)).reshape(1, -1)
                    return actor(t, deterministic=True)[0].double().numpy()

            ctl = ResidualController(core, act, float(a), ResidualObservationBuilder(spec_for(arm)))
        r = run_residual_episode(session, ctl)
        rows = r["rows"]
        return {"success": r["success"], "final_distance": r["final_distance"],
                "alarms": int(sum(x["alarm"] for x in rows)),
                "residual_norm_mean": float(np.mean([x["residual_norm"] for x in rows])),
                "fallbacks": int(sum(bool(x["fallback"]) for x in rows))}  # fmt: skip

    out: list[dict[str, Any]] = []
    for cfg_name, nss in configs.items():
        arm, a = cfg_name.split("@") if "@" in cfg_name else (cfg_name, "nan")
        rows = generate(cells, seeds, [policy_index], {cfg_name: nss}, build, run, role, plan,
                        alpha=float(a), friction_mode=plan["tm_friction_mode"])  # fmt: skip
        for row in rows:
            row["config"] = arm
        out.extend(rows)
    p_rows: list[dict[str, Any]] = []
    if include_p:  # GEN-3: PRIVILEGED C (true delay and gain) on D cells; diagnostic only

        def run_p(_cfg, _ns, _policy, session):
            r = run_episode(session, ActuatorEstimator(est_cfg), sequential(g4.constants), preds,
                            mode="true")  # fmt: skip
            rows = r["rows"]
            return {"success": r["success"], "final_distance": r["final_distance"],
                    "true_delay_used": bool(all(x["delay_used"] == x["delay_true"]
                                                for x in rows if x["active"]))}  # fmt: skip

        d_cells = [c for c in cells if c.group == "D"]
        if d_cells:
            p_rows = generate(d_cells, seeds, [policy_index], {"P": [-1]}, build, run_p, role,
                              plan, friction_mode=plan["tm_friction_mode"])  # fmt: skip
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, frozen[k]):
            raise RuntimeError("the frozen SAC actor changed during rollouts")
    return out, p_rows


def _indexed(item: tuple[int, tuple]) -> tuple[int, list, list]:
    k, args = item
    r, pr = _task(args)
    return k, r, pr


def plan_tasks(role: str, alpha_star: float | None, runs: Path, plan: dict,
               smoke_seeds: list[int] | None) -> tuple[list[tuple], list[int], list]:  # fmt: skip
    if role == "selection":
        a, b = R_SEED_BLOCKS["selection"]
        seeds = list(range(a, b))
        cells = selection_cells()
        configs: dict[str, list[int]] = {"C": [-1]}
        configs.update({f"R@{al}": list(plan["network_seeds"]) for al in plan["alphas"]})
        include_p = False
    elif role == "evaluation":
        if alpha_star is None:
            raise SystemExit("evaluation needs --alpha-star from the selection output")
        seeds = evaluation_seeds()
        cells = development_cells()
        nss = list(plan["network_seeds"])
        configs = {
            "B0": [-1],
            "C": [-1],
            f"R@{alpha_star}": nss,
            **{f"{arm}@{alpha_star}": nss for arm in ("A1", "A2", "A3", "B2")},
        }
        include_p = True
    else:  # smoke
        seeds = smoke_seeds or [0, 1]
        cells = development_cells()
        configs = {"B0": [-1], "C": [-1]}
        include_p = True
    check_role_seeds("smoke" if role == "smoke" else role, seeds, plan)
    names = [c.name for c in cells]
    chunks = [names[i::8] for i in range(8)]
    tasks = [(role, chunk, pi, seeds, configs, alpha_star, runs, plan, include_p)
             for pi in range(3) for chunk in chunks]  # fmt: skip
    return tasks, seeds, cells


def failed_runs(runs: Path, configs: dict[str, list[int]]) -> list[list[Any]]:
    out = []
    for cfg_name, nss in configs.items():
        arm, a = cfg_name.split("@") if "@" in cfg_name else (cfg_name, "nan")
        if arm not in EVAL_ARMS:
            continue
        for ns in nss:
            s = torch.load(run_dir(runs, arm, float(a), ns) / "final.pt", weights_only=False)
            if s.get("status") != "completed":
                out.append([arm, float(a), ns])
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--role", choices=("selection", "evaluation", "smoke"), required=True)
    p.add_argument("--runs", default="results/residual_rl")
    p.add_argument("--alpha-star", type=float, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--smoke-seeds", type=int, nargs="*", default=None)
    a = p.parse_args()
    plan = load_plan_config(CONFIG_PATH)
    runs = Path(a.runs)
    tasks, seeds, _cells = plan_tasks(a.role, a.alpha_star, runs, plan, a.smoke_seeds)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # Resumable: each finished task is cached under parts/; a restart reruns only the rest.
    parts = out / "parts"
    parts.mkdir(exist_ok=True)

    def part(k: int) -> Path:
        return parts / f"task_{k:03d}.pt"

    todo = [k for k in range(len(tasks)) if not part(k).exists()]
    with get_context("spawn").Pool(a.workers) as pool:
        done = pool.imap_unordered(_indexed, [(k, tasks[k]) for k in todo])
        for i, (k, r, pr) in enumerate(done, start=1):
            torch.save({"task": k, "rows": r, "p_rows": pr}, part(k))
            print(f"[{time.perf_counter() - t0:7.0f}s] task {k} done ({i}/{len(todo)})",
                  flush=True)  # fmt: skip
    rows, p_rows = [], []
    for k in range(len(tasks)):
        d = torch.load(part(k), weights_only=False)
        rows.extend(d["rows"])
        p_rows.extend(d["p_rows"])
    df = pd.DataFrame(rows).sort_values(["config", "alpha", "network_seed", "policy", "cell",
                                         "episode_seed"]).reset_index(drop=True)  # fmt: skip
    df.to_csv(out / f"{a.role}_episodes.csv.gz", index=False)
    if p_rows:
        pd.DataFrame(p_rows).to_csv(out / f"{a.role}_privileged_P.csv.gz", index=False)
    if a.role != "smoke":
        configs = tasks[0][4]
        (out / "failed.json").write_text(json.dumps(failed_runs(runs, configs)))
    meta = {"role": a.role, "seeds": [seeds[0], seeds[-1], len(seeds)], "episodes": len(df),
            "privileged_episodes": len(p_rows), "seconds": time.perf_counter() - t0,
            "alpha_star": a.alpha_star}  # fmt: skip
    (out / f"{a.role}_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
