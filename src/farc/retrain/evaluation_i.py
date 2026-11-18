"""Phase I Stage-2 evaluation and its pre-written analysis (proposal sections 4.1 and 6).

Configurations on the 50 Phase H cells, paired by (policy seed k, condition, episode seed):
frozen SAC, contact-aware and Candidate C (frozen seed k, as in G4/H), Phase H R2 (registered),
and Phase I I-A and I-C (warm-started from frozen seed k).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.delay_eval import holm
from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.dynamics import load_dynamics
from farc.contact.episodes import learned_predictor
from farc.delaychange.heldout import G4Cell
from farc.delaychange.mechanism import run_episode, sequential
from farc.delaychange.mechanism_analysis import two
from farc.delaychange.schedule import ScheduledDelaySession
from farc.delaychange.stage3 import Latched
from farc.disturbances.config import ConditionSpec
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.retrain.evaluation import h_cells, retrained_episode
from farc.retrain.phase_h import load_retrained
from farc.retrain.phase_i import Featurizer, load_phase_i

CONFIGURATIONS = ("base_only", "contact_learned", "C", "retrained_R2", "phase_i_IA",
                  "phase_i_IC")  # fmt: skip
I_ARMS = ("IA", "IC")


def cells() -> list[G4Cell]:
    return h_cells()


@dataclass(frozen=True)
class ContextI:
    frozen_checkpoints: tuple[str, ...]
    r2_paths: dict[int, str]
    phase_i_paths: dict[str, str]  # "IA_seed0" -> path
    verify: bool
    model: str
    model_sha256: str
    estimator: dict[str, Any]
    constants: dict[str, Any]
    seeds: tuple[int, ...]
    perturbation_seed: int


def phase_i_episode(session: ScheduledDelaySession, policy, normalizer, arm: str,
                    estimator: dict[str, Any]) -> dict[str, Any]:  # fmt: skip
    feat = Featurizer(arm, estimator)
    rows = []
    while not session.done:
        view = session.observe()
        assert view.observation is not None
        feat.observe(view.observation)
        obs = np.concatenate([np.asarray(view.observation), feat.features()])
        command = np.clip(np.asarray(policy(normalizer(obs[None], view.goal[None]))[0],
                                     np.float64), -1.0, 1.0)  # fmt: skip
        rows.append({"t": view.step, "delay_true": session.schedule.delay_at(view.step)})
        feat.issued(command)
        session.act(command)
    return {"success": session.final_success, "rows": rows}


def run_task(args: tuple[ContextI, int, G4Cell]):
    ctx, k, cell = args
    torch.set_num_threads(1)
    frozen = load_policy(resolve(ctx.frozen_checkpoints[k]))
    actor = {n: v.clone() for n, v in frozen.actor.state_dict().items()}
    f_policy = deterministic_policy_function(frozen.actor)
    f_norm = FrozenNormalizer(frozen.normalizer)
    env = make_env(frozen.env_config)
    est_cfg = EstimatorConfig(**ctx.estimator)
    preds = {"learned": learned_predictor(load_dynamics(resolve(ctx.model), ctx.model_sha256),
                                          move_object=True)}  # fmt: skip
    r2 = load_retrained(resolve(ctx.r2_paths[k]), f"phase_h_R2_seed{k}", verify=ctx.verify)
    r2_fn = (deterministic_policy_function(r2.actor), FrozenNormalizer(r2.normalizer))
    arms = {}
    for arm in I_ARMS:
        pol, hyb = load_phase_i(resolve(ctx.phase_i_paths[f"{arm}_seed{k}"]),
                                f"phase_i_{arm}_seed{k}", verify=ctx.verify)  # fmt: skip
        if pol.training_seed != k:
            raise RuntimeError(
                f"{arm} checkpoint for seed {k} has training seed {pol.training_seed}"
            )
        arms[arm] = (deterministic_policy_function(pol.actor), hyb)
    episodes: list[dict[str, Any]] = []
    for seed in ctx.seeds:
        for name in CONFIGURATIONS:
            session = ScheduledDelaySession(env, f_policy, f_norm,
                                            ConditionSpec(cell.name, cell.extra), seed,
                                            ctx.perturbation_seed, cell.schedule(seed),
                                            "replay_drop")  # fmt: skip
            if name == "retrained_R2":
                result = retrained_episode(session, r2_fn[0], r2_fn[1], 4)
            elif name.startswith("phase_i_"):
                arm = name.removeprefix("phase_i_")
                result = phase_i_episode(session, arms[arm][0], arms[arm][1], arm, ctx.estimator)
            else:
                ctrl: Any = sequential(ctx.constants) if name == "C" else Latched()
                mode = "base" if name == "base_only" else "normal"
                result = run_episode(session, ActuatorEstimator(est_cfg), ctrl, preds, mode=mode)
            episodes.append({"arm": cell.arm, "configuration": name, "policy_seed": k,
                             "condition": cell.name, "family": cell.family,
                             "episode_seed": seed, "success": result["success"]})  # fmt: skip
    for n, v in frozen.actor.state_dict().items():
        if not torch.equal(v, actor[n]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes


# --- pre-written analysis (proposal section 6) -------------------------------------------------

TI = ("RT", "RI")
PERSISTENT = ("RP", "F-P", "E-P")
NOMINAL = ("H_undisturbed", "E_C0_undisturbed")
NOISE = ("H_noise0.01", "E_C8_noise0.01")


def _fam(e: pd.DataFrame, fams: tuple[str, ...]) -> pd.DataFrame:
    return e[e["family"].isin(fams)]


def _row(name, frame, a, b, bound, n, s, scope) -> dict[str, Any]:
    return {"test": name, "scope": scope, "configuration": a, "reference": b,
            **two(frame, a, b, bound, n, s)}  # fmt: skip


def gates(e: pd.DataFrame, n: int, s: int) -> pd.DataFrame:
    rows = []
    for arm in ("phase_i_IA", "phase_i_IC"):
        nom = _row("G-nominal", e[e["condition"].isin(NOMINAL)], arm, "base_only", -0.02, n, s,
                   "H_undisturbed + E_C0")  # fmt: skip
        per_seed = (e[e["condition"].isin(NOMINAL) & (e["configuration"] == arm)]
                    .groupby("policy_seed")["success"].mean())  # fmt: skip
        nom["per_seed_success"] = per_seed.round(4).to_dict()
        nom["passed"] = bool(nom["ci_low"] > -0.02 and (per_seed >= 0.95).all())
        noise = _row("G-noise", e[e["condition"].isin(NOISE)], arm, "base_only", -0.02, n, s,
                     "H_noise0.01 + E_C8")  # fmt: skip
        noise["passed"] = bool(noise["ci_low"] > -0.02)
        rows += [nom, noise]
    return pd.DataFrame(rows)


def derive_i(episodes: pd.DataFrame, n: int = 10_000, s: int = 20262000,
             alpha: float = 0.025) -> tuple[dict, dict]:  # fmt: skip
    g = gates(episodes, n, s)
    passed = {arm: bool(g[g["configuration"] == arm]["passed"].all())
              for arm in ("phase_i_IA", "phase_i_IC")}  # fmt: skip

    def family(arm: str, labels: tuple[str, ...]) -> pd.DataFrame:
        rows = [
            _row(labels[0], _fam(episodes, PERSISTENT), arm, "C", -0.02, n, s, "persistent"),
            _row(labels[1], _fam(episodes, TI), arm, "base_only", -0.02, n, s, "RT+RI"),
            _row(labels[2], _fam(episodes, TI), arm, "C", 0.0, n, s, "RT+RI"),
            _row(labels[3], _fam(episodes, PERSISTENT), arm, "retrained_R2", 0.0, n, s,
                 "persistent"),
        ]  # fmt: skip
        return pd.DataFrame(rows)

    prim = family("phase_i_IA", ("P1", "P2", "P3", "P4"))
    prim["p_holm"] = holm(prim["p"].tolist())
    prim["result"] = np.where(prim["p_holm"] <= alpha, "supported", "not supported")
    prim["gate"] = "passed" if passed["phase_i_IA"] else "baseline not nominally competent"
    sec = pd.concat([
        family("phase_i_IC", ("S1", "S2", "S3", "S4")),
        pd.DataFrame([
            _row("S5", _fam(episodes, PERSISTENT), "phase_i_IC", "phase_i_IA", 0.0, n, s,
                 "persistent"),
            _row("S6", _fam(episodes, TI), "phase_i_IA", "contact_learned", 0.0, n, s, "RT+RI"),
        ]),
    ], ignore_index=True)  # fmt: skip
    sec["p_holm"] = holm(sec["p"].tolist())
    sec["result"] = np.where(sec["p_holm"] <= alpha, "supported", "not supported")
    sec["gate_IC"] = "passed" if passed["phase_i_IC"] else "baseline not nominally competent"
    wide = episodes.pivot_table(index="condition", columns="configuration", values="success",
                                aggfunc="mean")  # fmt: skip
    by_condition = (wide * 100).reset_index()
    by_family = (episodes.groupby(["family", "configuration"])["success"].mean().unstack()
                 * 100).reset_index()  # fmt: skip
    per_seed = []
    for k, frame in episodes.groupby("policy_seed"):
        for arm in ("phase_i_IA", "phase_i_IC"):
            for t, b, fams in (("vs_C_persistent", "C", PERSISTENT),
                               ("vs_SAC_short", "base_only", TI),
                               ("vs_C_short", "C", TI)):  # fmt: skip
                per_seed.append(_row(t, _fam(frame, fams), arm, b, 0.0, n, s, f"seed {k}"))
    ng = pd.DataFrame([_row("NG_descriptive", _fam(episodes, ("NG",)), arm, "C", 0.0, n, s, "NG")
                       for arm in ("phase_i_IA", "phase_i_IC")])  # fmt: skip
    tables = {"gates": g, "primary": prim, "secondary": sec, "by_condition": by_condition,
              "by_family": by_family, "per_seed": pd.DataFrame(per_seed), "ng": ng}  # fmt: skip
    summary = {"gates_passed": passed,
               "primary": {r["test"]: {"estimate": round(r["estimate"], 4),
                                       "ci": [round(r["ci_low"], 4), round(r["ci_high"], 4)],
                                       "p_holm": r["p_holm"], "result": r["result"]}
                           for r in prim.to_dict("records")},
               "episodes": len(episodes)}  # fmt: skip
    return tables, summary


DERIVED = ("gates", "primary", "secondary", "by_condition", "by_family", "per_seed", "ng")


def phase_i_paths(output: str, suffix: str = "") -> dict[str, str]:
    return {f"{arm}_seed{k}": str(Path(output) / f"phase_i_{arm}{suffix}_seed{k}" / "checkpoint.pt")
            for arm in I_ARMS for k in range(3)}  # fmt: skip


def r2_paths(output: str) -> dict[int, str]:
    return {k: str(Path(output) / f"phase_h_R2_seed{k}" / "checkpoint.pt") for k in range(3)}
