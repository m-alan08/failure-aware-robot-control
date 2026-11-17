"""Phase H Stage-2 evaluation and its pre-written analysis (proposal sections 3.4-4).

Configurations on every cell, paired by (policy seed k, condition, episode seed):
- ``base_only``, ``contact_learned``, ``C``: frozen SAC seed k, exactly as in G4;
- ``retrained_R1``, ``retrained_R2``: the Phase H policy of training seed k, acting on the
  (noisy) observation, plus for R2 the last 4 issued commands, through the same
  ``ScheduledDelaySession``.
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
from farc.delaychange.heldout import G4Cell, g4_cells
from farc.delaychange.mechanism import run_episode, sequential
from farc.delaychange.mechanism_analysis import step_metrics, two
from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
from farc.delaychange.stage3 import Latched
from farc.disturbances.config import ConditionSpec, DisturbanceSpec
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.retrain.phase_h import ARMS, augment, load_retrained

CONFIGURATIONS = ("base_only", "contact_learned", "C", "retrained_R1", "retrained_R2")
PERTURBATION_SEED = 20261950
NOISE_5 = DisturbanceSpec(type="observation_noise", severity=0.005, mask="positions")
NOISE_10 = DisturbanceSpec(type="observation_noise", severity=0.01, mask="positions")
SCALE = DisturbanceSpec(type="action_scale", severity=0.5)


def e_cells() -> list[G4Cell]:
    """The E/F1 conditions C0-C8, as fixed schedules (replay/drop = the original
    ``action_delay`` semantics)."""
    spec = [("E_C0_undisturbed", "E-0", (), ()),
            ("E_C1_delay1", "E-P", ((0, None, 1),), ()),
            ("E_C2_delay2", "E-P", ((0, None, 2),), ()),
            ("E_C3_delay3", "E-P", ((0, None, 3),), ()),
            ("E_C4_delay4", "E-P", ((0, None, 4),), ()),
            ("E_C5_delay3_onset15", "E-P", ((15, None, 3),), ()),
            ("E_C6_delay3_noise0.005", "E-P", ((0, None, 3),), (NOISE_5,)),
            ("E_C7_delay3_scale0.5", "E-P", ((0, None, 3),), (SCALE,)),
            ("E_C8_noise0.01", "E-H", (), (NOISE_10,))]  # fmt: skip
    return [G4Cell("E", n, f, 100 + i, extra, DelaySchedule(p))
            for i, (n, f, p, extra) in enumerate(spec)]  # fmt: skip


def h_cells() -> list[G4Cell]:
    return g4_cells() + e_cells()


@dataclass(frozen=True)
class Context:
    frozen_checkpoints: tuple[str, ...]  # index k = training seed k
    retrained: dict[str, str]  # "R1_seed0" -> checkpoint path
    verify_retrained: bool
    model: str
    model_sha256: str
    estimator: dict[str, Any]
    constants: dict[str, Any]
    seeds: tuple[int, ...]


def retrained_episode(session: ScheduledDelaySession, policy, normalizer,
                      history: int) -> dict[str, Any]:  # fmt: skip
    rows = []
    while not session.done:
        view = session.observe()
        assert view.observation is not None
        obs = augment(np.asarray(view.observation), list(session.commands), history)
        command = np.clip(np.asarray(policy(normalizer(obs, view.goal)[None])[0], np.float64),
                          -1.0, 1.0)  # fmt: skip
        rows.append({"t": view.step, "delay_true": session.schedule.delay_at(view.step),
                     "delay_used": 0, "active": False, "alarm": False, "suspended": False,
                     "ctrl_ns": 0, "learned_calls": 0})  # fmt: skip
        session.act(command)
    return {"success": session.final_success, "rows": rows}


def run_task(args: tuple[Context, int, G4Cell]):
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
    retrained = {}
    for arm in ARMS:
        r = load_retrained(resolve(ctx.retrained[f"{arm}_seed{k}"]), f"phase_h_{arm}_seed{k}",
                           verify=ctx.verify_retrained)  # fmt: skip
        if int(r.training_seed) != k:
            raise RuntimeError(f"{arm} checkpoint for seed {k} has training seed {r.training_seed}")
        retrained[arm] = (deterministic_policy_function(r.actor), FrozenNormalizer(r.normalizer),
                          r.sha256)  # fmt: skip
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for seed in ctx.seeds:
        for name in CONFIGURATIONS:
            session = ScheduledDelaySession(env, f_policy, f_norm,
                                            ConditionSpec(cell.name, cell.extra), seed,
                                            PERTURBATION_SEED, cell.schedule(seed),
                                            "replay_drop")  # fmt: skip
            if name.startswith("retrained_"):
                arm = name.split("_")[1]
                policy, norm, _ = retrained[arm]
                result = retrained_episode(session, policy, norm, ARMS[arm])
            else:
                ctrl: Any = sequential(ctx.constants) if name == "C" else Latched()
                mode = "base" if name == "base_only" else "normal"
                result = run_episode(session, ActuatorEstimator(est_cfg), ctrl, preds, mode=mode)
            key = {"arm": cell.arm, "configuration": name, "policy_seed": k,
                   "condition": cell.name, "family": cell.family, "episode_seed": seed}  # fmt: skip
            episodes.append({**key, "success": result["success"]})
            steps.extend({**key, **{c: r[c] for c in ("t", "delay_true", "delay_used", "active",
                                                      "alarm", "suspended", "ctrl_ns",
                                                      "learned_calls")}}
                         for r in result["rows"])  # fmt: skip
    for n, v in frozen.actor.state_dict().items():
        if not torch.equal(v, actor[n]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes, steps


# --- pre-written analysis (section 4) ---------------------------------------------------------

TI_R = ("RT", "RI")
PERSISTENT = ("RP", "F-P", "E-P")


def _fam(e: pd.DataFrame, fams: tuple[str, ...]) -> pd.DataFrame:
    return e[e["family"].isin(fams)]


def _row(name, frame, a, b, bound, n, s, scope) -> dict[str, Any]:
    return {"test": name, "scope": scope, "configuration": a, "reference": b,
            **two(frame, a, b, bound, n, s)}  # fmt: skip


def derive_h(episodes: pd.DataFrame, steps: pd.DataFrame, n: int = 10_000,
             s: int = 20261950, alpha: float = 0.025) -> tuple[dict, dict]:  # fmt: skip
    prim = pd.DataFrame([
        _row("H1", _fam(episodes, TI_R), "retrained_R2", "base_only", -0.02, n, s, "RT+RI"),
        _row("H2", _fam(episodes, TI_R), "retrained_R2", "C", 0.0, n, s, "RT+RI"),
        _row("H3", _fam(episodes, PERSISTENT), "C", "retrained_R2", -0.02, n, s,
             "persistent: RP + F-P + E C1-C7"),
    ])  # fmt: skip
    prim["p_holm"] = holm(prim["p"].tolist())
    prim["result"] = np.where(prim["p_holm"] <= alpha, "supported", "not supported")
    sec = pd.DataFrame([
        _row("S1", _fam(episodes, TI_R), "retrained_R2", "retrained_R1", 0.0, n, s, "RT+RI"),
        _row("S1", _fam(episodes, PERSISTENT), "retrained_R2", "retrained_R1", 0.0, n, s,
             "persistent"),
        _row("S2", _fam(episodes, TI_R), "retrained_R1", "base_only", -0.02, n, s, "RT+RI"),
        _row("S3", _fam(episodes, PERSISTENT), "C", "retrained_R1", -0.02, n, s, "persistent"),
        _row("S4", _fam(episodes, TI_R), "retrained_R2", "contact_learned", 0.0, n, s, "RT+RI"),
    ])  # fmt: skip
    sec["p_holm"] = holm(sec["p"].tolist())
    sec["result"] = np.where(sec["p_holm"] <= alpha, "supported", "not supported")
    desc = [_row("S5_descriptive", _fam(episodes, ("NG",)), "retrained_R2", "C", 0.0, n, s, "NG")]
    gates = []
    for arm in ("retrained_R1", "retrained_R2"):
        for cond in ("H_undisturbed", "E_C0_undisturbed"):
            r = _row("nominal", episodes[episodes["condition"] == cond], arm, "base_only", -0.02,
                     n, s, cond)  # fmt: skip
            r["passed"] = r["ci_low"] > -0.02
            gates.append(r)
    nominal = episodes[episodes["condition"].isin(["H_undisturbed", "E_C0_undisturbed"])]
    competence = (nominal.groupby(["configuration", "policy_seed"])["success"].mean()
                  .rename("undisturbed_success").reset_index())  # fmt: skip
    competence["training_failed_flag"] = (competence["configuration"].str.startswith("retrained")
                                          & (competence["undisturbed_success"] < 0.90))  # fmt: skip
    per_family = (episodes.groupby(["family", "configuration"])["success"].mean().unstack()
                  * 100).reset_index()  # fmt: skip
    per_seed = []
    for k, frame in episodes.groupby("policy_seed"):
        for t, a, b, fams in (("H1", "retrained_R2", "base_only", TI_R),
                              ("H2", "retrained_R2", "C", TI_R),
                              ("H3", "C", "retrained_R2", PERSISTENT)):  # fmt: skip
            per_seed.append({**_row(t, _fam(frame, fams), a, b, 0.0, n, s, f"seed {k}")})
    tables = {"primary": prim, "secondary": sec, "descriptive": pd.DataFrame(desc),
              "gates": pd.DataFrame(gates), "competence": competence,
              "success_by_family": per_family, "per_seed": pd.DataFrame(per_seed),
              **{k: v for k, v in step_metrics(steps).items() if k != "overhead"}}  # fmt: skip
    summary = {"primary": {r["test"]: {"estimate": round(r["estimate"], 4),
                                       "ci": [round(r["ci_low"], 4), round(r["ci_high"], 4)],
                                       "p_holm": r["p_holm"], "result": r["result"]}
                           for r in prim.to_dict("records")},
               "nominal_gates_passed": bool(pd.DataFrame(gates)["passed"].all()),
               "training_failed": competence[competence["training_failed_flag"]][
                   ["configuration", "policy_seed"]].to_dict("records"),
               "episodes": len(episodes)}  # fmt: skip
    return tables, summary


DERIVED = ("primary", "secondary", "descriptive", "gates", "competence", "success_by_family",
           "per_seed", "latency", "rates")  # fmt: skip


def retrained_paths(output: str, suffix: str = "") -> dict[str, str]:
    """``suffix="_stage0"`` selects the Stage-0 code-check checkpoints (debug run only)."""
    return {f"{arm}_seed{k}": str(Path(output) / f"phase_h_{arm}{suffix}_seed{k}" / "checkpoint.pt")
            for arm in ARMS for k in range(3)}  # fmt: skip
