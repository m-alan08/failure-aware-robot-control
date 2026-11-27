"""Representative trajectories from the development evaluation (descriptive, after analysis).

    PYTHONPATH=src:scripts python scripts/residual_trajectories.py results/residual_rl --per 3

Picks episodes from ``evaluation/evaluation_episodes.csv.gz`` (seed-ordered, no outcome-based
tuning: the first ``--per`` of each category in seed order) and re-simulates them
deterministically with per-step records for C, R (each network) and B0. Categories:
rescued (C fails, R-majority succeeds), harmed (C succeeds, R-majority fails), and
both-succeed nominal. Every re-simulated outcome must match the evaluation table; a mismatch
is an error. Writes ``report/trajectories.jsonl`` and ``report/trajectories.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from residual_rollouts import G4_CONFIG, load_actor, run_dir, spec_for

from farc.delaychange.heldout import _setup, load_g4
from farc.delaychange.schedule import ScheduledDelaySession
from farc.disturbances.config import ConditionSpec
from farc.residual.base_core import SACBaseCore
from farc.residual.controller import ResidualController
from farc.residual.core import CandidateCCore
from farc.residual.episodes import development_cells
from farc.residual.observation import ObservationSpec, ResidualObservationBuilder
from farc.residual.policy import ZeroResidual
from farc.residual.training_env import disturb
from farc.supervision.seeds import check_r_seeds

KEY = ["policy", "episode_seed", "cell"]


def pick(df: pd.DataFrame, per: int) -> pd.DataFrame:
    c = df[df.config == "C"].set_index(KEY).success.astype(bool)
    r = df[df.config == "R"].groupby(KEY).success.mean()
    w = pd.DataFrame({"C": c, "R": r.reindex(c.index)}).reset_index()
    w["group"] = df[df.config == "C"].set_index(KEY).group.reindex(w.set_index(KEY).index).values
    cats = {
        "rescued": w[~w.C & (w.R > 0.5)],
        "harmed": w[w.C & (w.R < 0.5)],
        "nominal_both_succeed": w[(w.group == "N") & w.C & (w.R == 1.0)],
    }
    out = []
    for name, sub in cats.items():
        for _grp, g in sub.groupby("group"):
            g = g.sort_values(["episode_seed", "policy", "cell"]).head(per)
            out.append(g.assign(category=name))
    return pd.concat(out) if out else pd.DataFrame()


def simulate(cell, seed, policy_index, alpha, runs, plan, configs):
    g4 = load_g4(G4_CONFIG, "debugging")
    loaded, policy, normalizer, env, _e, preds = _setup(g4, g4.checkpoints[policy_index])
    spec = cell.make(seed)
    res = {}
    for name, ns in configs:
        base = ScheduledDelaySession(env, policy, normalizer, ConditionSpec(cell.name, spec.extra),
                                     seed, plan["generator_seed"], spec.schedule,
                                     spec.semantics)  # fmt: skip
        s = disturb(base, spec, spec.dynamics, spec.obs_sigma, seed, plan["generator_seed"],
                    plan["tm_friction_mode"])  # fmt: skip
        core: Any
        act: Any
        if name == "C":
            core, act, a, sp = (CandidateCCore(policy, normalizer, preds["learned"],
                                               g4.estimator, g4.constants),
                                ZeroResidual(), 0.0, ObservationSpec())  # fmt: skip
        elif name == "B0":
            core, act, a, sp = SACBaseCore(policy, normalizer), ZeroResidual(), 0.0, spec_for("B2")
        else:
            actor = load_actor(run_dir(runs, "R", alpha, ns), "R")

            def act_fn(x, actor=actor):
                with torch.no_grad():
                    t = torch.as_tensor(np.asarray(x, np.float32)).reshape(1, -1)
                    return actor(t, deterministic=True)[0].double().numpy()

            act = act_fn
            core = CandidateCCore(policy, normalizer, preds["learned"], g4.estimator,
                                  g4.constants)  # fmt: skip
            a, sp = alpha, ObservationSpec()
        ctl = ResidualController(core, act, a, ResidualObservationBuilder(sp))
        ctl.reset()
        steps = []
        while not s.done:
            view = s.observe()
            st = ctl.propose(view.observation, view.goal, view.base_action)
            s.act(st.command)
            ctl.commit(st)
            o = np.asarray(view.observation)
            steps.append({"t": view.step, "a_c": np.round(st.a_c, 4).tolist(),
                          "a_final": np.round(st.command, 4).tolist(),
                          "delta": np.round(st.delta, 4).tolist(),
                          "grip": np.round(o[:3], 4).tolist(), "obj": np.round(o[3:6], 4).tolist(),
                          "goal": np.round(np.asarray(view.goal), 4).tolist(),
                          "delay_used": int(st.proposal.delay), "active": bool(st.proposal.active),
                          "alarm": bool(st.proposal.decision.alarm)})  # fmt: skip
        s.close()
        res[f"{name}{'' if ns < 0 else ns}"] = {"success": bool(s.final_success),
                                                "final_distance": float(s.final_distance),
                                                "steps": steps}  # fmt: skip
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root")
    p.add_argument("--per", type=int, default=3)
    a = p.parse_args()
    root = Path(a.root)
    plan = json.loads(Path("configs/residual/residual_rl.json").read_text())
    alpha = json.loads((root / "selection" / "selection.json").read_text())["alpha_star"]
    df = pd.read_csv(root / "evaluation" / "evaluation_episodes.csv.gz")
    picks = pick(df, a.per)
    cells = {c.name: c for c in development_cells()}
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Representative trajectories (descriptive)", ""]
    with (out / "trajectories.jsonl").open("w") as f:
        for rec_ in picks.to_dict("records"):
            row: Any = argparse.Namespace(**{str(k): v for k, v in rec_.items()})
            seed = int(row.episode_seed)
            block = "evaluation" if seed < 3_740_000 else "evaluation_extension"
            check_r_seeds([seed], block)
            configs = [("C", -1), ("B0", -1), ("R", 0), ("R", 1), ("R", 2)]
            res = simulate(cells[row.cell], seed, int(row.policy), alpha, root, plan, configs)
            for k, v in res.items():  # outcomes must reproduce the evaluation table
                cfg, ns = ("R", int(k[1:])) if k.startswith("R") else (k, -1)
                m = ((df.config == cfg) & (df.network_seed == ns) & (df.policy == row.policy)
                     & (df.episode_seed == seed) & (df.cell == row.cell))  # fmt: skip
                if bool(df[m].success.iloc[0]) != v["success"]:
                    raise RuntimeError(f"re-simulation mismatch {k} {row}")
            rec = {"category": row.category, "group": row.group, "cell": row.cell,
                   "episode_seed": seed, "policy": int(row.policy), "alpha": alpha,
                   "results": res}  # fmt: skip
            f.write(json.dumps(rec) + "\n")
            summ = {k: (v["success"], round(v["final_distance"], 3),
                        round(float(np.mean([np.abs(np.subtract(s["a_final"], s["a_c"])).mean()
                                             for s in v["steps"]])), 4))
                    for k, v in res.items()}  # fmt: skip
            md.append(
                f"- **{row.category}** {row.group}/{row.cell} seed {seed} policy "
                f"{row.policy}: (success, final distance, mean |a_final-a_C|) {summ}"
            )
    (out / "trajectories.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
