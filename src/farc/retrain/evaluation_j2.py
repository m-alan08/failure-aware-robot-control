"""Phase J-2 Stage-2 evaluation and pre-written analysis (docs/PHASE_J2_PROPOSAL.md, 3-5)."""

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
from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
from farc.delaychange.stage3 import Latched
from farc.disturbances.config import ConditionSpec, DisturbanceSpec
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.retrain.evaluation import h_cells
from farc.retrain.evaluation_i import phase_i_episode
from farc.retrain.phase_i import load_phase_i

DUPLICATES = ("E_C0_undisturbed", "E_C8_noise0.01", "E_C1_delay1", "E_C2_delay2", "E_C3_delay3",
              "E_C4_delay4")  # fmt: skip
CONFIGURATIONS = ("base_only", "contact_learned", "C", "IA", "IC", "J2A", "J2C")
ARM_FEATURES = {"IA": "IA", "IC": "IC", "J2A": "IA", "J2C": "IC"}


def j2_cells() -> list[G4Cell]:
    none = DelaySchedule(())
    cells = [c for c in h_cells() if c.name not in DUPLICATES]
    for i, sev in enumerate((0.005, 0.015)):
        spec = DisturbanceSpec(type="observation_noise", severity=sev, mask="positions")
        cells.append(G4Cell("J", f"J_noise{sev}", "noise", 200 + i, (spec,), none))
    return cells


@dataclass(frozen=True)
class ContextJ2:
    frozen_checkpoints: tuple[str, ...]
    paths: dict[str, str]  # "IA_seed0" etc
    ids: dict[str, str]
    verify: bool
    model: str
    model_sha256: str
    estimator: dict[str, Any]
    constants: dict[str, Any]
    seeds: tuple[int, ...]
    perturbation_seed: int


def run_task(args: tuple[ContextJ2, int, G4Cell]):
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
    nets = {}
    for arm in ARM_FEATURES:
        key = f"{arm}_seed{k}"
        pol, hyb = load_phase_i(resolve(ctx.paths[key]), ctx.ids[key], verify=ctx.verify,
                                manifest_path=_manifest(arm))  # fmt: skip
        if pol.training_seed != k:
            raise RuntimeError(f"{key} has training seed {pol.training_seed}")
        nets[arm] = (deterministic_policy_function(pol.actor), hyb)
    episodes: list[dict[str, Any]] = []
    for seed in ctx.seeds:
        for name in CONFIGURATIONS:
            session = ScheduledDelaySession(env, f_policy, f_norm,
                                            ConditionSpec(cell.name, cell.extra), seed,
                                            ctx.perturbation_seed, cell.schedule(seed),
                                            "replay_drop")  # fmt: skip
            if name in nets:
                success = phase_i_episode(session, nets[name][0], nets[name][1],
                                          ARM_FEATURES[name], ctx.estimator)["success"]  # fmt: skip
            else:
                ctrl: Any = sequential(ctx.constants) if name == "C" else Latched()
                mode = "base" if name == "base_only" else "normal"
                success = run_episode(session, ActuatorEstimator(est_cfg), ctrl, preds,
                                      mode=mode)["success"]  # fmt: skip
            episodes.append(
                {
                    "configuration": name,
                    "policy_seed": k,
                    "condition": cell.name,
                    "family": cell.family,
                    "episode_seed": seed,
                    "success": success,
                }
            )
    for n, v in frozen.actor.state_dict().items():
        if not torch.equal(v, actor[n]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes


def _manifest(arm: str) -> Path:
    here = Path(__file__).parent
    return here / (
        "checkpoints_phase_j2.json" if arm.startswith("J2") else "checkpoints_phase_i.json"
    )


# --- pre-written analysis -----------------------------------------------------------------------

TI = ("RT", "RI")
PERSISTENT_CONDS_EXTRA = ("E_C5_delay3_onset15", "E_C6_delay3_noise0.005", "E_C7_delay3_scale0.5")


def _persistent(e: pd.DataFrame) -> pd.DataFrame:
    return e[e["family"].isin(("RP", "F-P")) | e["condition"].isin(PERSISTENT_CONDS_EXTRA)]


def _ti(e: pd.DataFrame) -> pd.DataFrame:
    return e[e["family"].isin(TI)]


def _cond(e: pd.DataFrame, c: str) -> pd.DataFrame:
    return e[e["condition"] == c]


def _row(t, frame, a, b, bound, n, s, scope):
    return {"test": t, "scope": scope, "configuration": a, "reference": b,
            **two(frame, a, b, bound, n, s)}  # fmt: skip


def derive_j2(e: pd.DataFrame, n: int = 10_000, s: int = 20262200) -> tuple[dict, dict]:
    if e["condition"].isin(DUPLICATES).any():
        raise RuntimeError("duplicate cells present")
    gates, passed = [], {}
    for arm in ("J2A", "J2C"):
        nom = _row("G-nominal", _cond(e, "H_undisturbed"), arm, "base_only", -0.02, n, s,
                   "H_undisturbed")  # fmt: skip
        und = _cond(e, "H_undisturbed")
        ps = und[und["configuration"] == arm].groupby("policy_seed")["success"].mean()
        nom["per_seed_success"] = ps.round(4).to_dict()
        nom["passed"] = bool(nom["ci_low"] > -0.02 and (ps >= 0.95).all())
        noi = _row("G-noise", _cond(e, "H_noise0.01"), arm, "base_only", -0.02, n, s,
                   "H_noise0.01")  # fmt: skip
        noi["passed"] = bool(noi["ci_low"] > -0.02)
        gates += [nom, noi]
        passed[arm] = nom["passed"] and noi["passed"]
    noise = _cond(e, "H_noise0.01")
    prim = pd.DataFrame([
        _row("N1", noise, "J2A", "IA", 0.0, n, s, "1 cm noise"),
        _row("N2", noise, "J2C", "IC", 0.0, n, s, "1 cm noise"),
        _row("D1", _persistent(e), "J2A", "IA", -0.02, n, s, "persistent"),
        _row("D2", _persistent(e), "J2C", "IC", -0.02, n, s, "persistent"),
    ])  # fmt: skip
    prim["p_holm"] = holm(prim["p"].tolist())
    prim["result"] = np.where(prim["p_holm"] <= 0.025, "supported", "not supported")
    sec_rows = []
    for arm, tags in (("J2A", ("S1", "S3", "S5", "S7")), ("J2C", ("S2", "S4", "S6", "S8"))):
        sec_rows += [
            _row(tags[0], _persistent(e), arm, "C", -0.02, n, s, "persistent"),
            _row(tags[1], _ti(e), arm, "base_only", -0.02, n, s, "RT+RI"),
            _row(tags[2], _ti(e), arm, "C", 0.0, n, s, "RT+RI"),
            _row(tags[3], _cond(e, "J_noise0.015"), arm, "base_only", -0.02, n, s, "1.5 cm"),
        ]
    sec = pd.DataFrame(sec_rows).sort_values("test").reset_index(drop=True)
    sec["p_holm"] = holm(sec["p"].tolist())
    sec["result"] = np.where(sec["p_holm"] <= 0.025, "supported", "not supported")
    sec["gate"] = [("passed" if passed[c] else "not competent") for c in sec["configuration"]]
    desc = []
    for cond in ("J_noise0.005", "J_noise0.015", "H_noise0.01", "H_undisturbed"):
        for a in ("IA", "IC", "J2A", "J2C"):
            desc.append(_row("vs_SAC", _cond(e, cond), a, "base_only", 0.0, n, s, cond))
    for k, frame in e.groupby("policy_seed"):
        for a, b, sc, f in (
            ("J2A", "IA", "noise", _cond(frame, "H_noise0.01")),
            ("J2C", "IC", "noise", _cond(frame, "H_noise0.01")),
            ("J2A", "base_only", "noise", _cond(frame, "H_noise0.01")),
            ("J2C", "base_only", "noise", _cond(frame, "H_noise0.01")),
        ):
            desc.append(_row("per_seed", f, a, b, 0.0, n, s, f"seed {k} {sc}"))
    by_cond = (e.pivot_table(index="condition", columns="configuration", values="success",
                             aggfunc="mean") * 100).reset_index()  # fmt: skip
    tables = {"gates": pd.DataFrame(gates), "primary": prim, "secondary": sec,
              "descriptive": pd.DataFrame(desc), "by_condition": by_cond}  # fmt: skip
    summary = {"gates_passed": passed,
               "primary": {r["test"]: {"estimate": round(r["estimate"], 4),
                                       "ci": [round(r["ci_low"], 4), round(r["ci_high"], 4)],
                                       "p_holm": r["p_holm"], "result": r["result"]}
                           for r in prim.to_dict("records")},
               "episodes": len(e)}  # fmt: skip
    return tables, summary


DERIVED = ("gates", "primary", "secondary", "descriptive", "by_condition")
