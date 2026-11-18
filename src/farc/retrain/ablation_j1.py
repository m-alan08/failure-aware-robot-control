"""Phase J-1 privileged-input ablation (docs/PHASE_J1_PROTOCOL.md). Evaluation only.

The PRIVILEGED variants (``IA_clean_features``, ``IC_clean_estimate``) compute the extra features
from the simulator's noise-free observation (``session.observation``) while the policy's base
observation stays the noisy view. They are diagnostic only, never deployable.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.delay_eval import holm
from farc.adaptation.experiment import AdaptationConfigError
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.delaychange.heldout import G4Cell
from farc.delaychange.mechanism_analysis import two
from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
from farc.disturbances.config import ConditionSpec, DisturbanceSpec
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import checkpoint_path
from farc.retrain.phase_i import Featurizer, load_phase_i
from farc.supervision.seeds import check_j_seeds

CONFIGURATIONS = ("base_only", "IA", "IA_clean_features", "IC", "IC_clean_estimate")
PRIVILEGED = ("IA_clean_features", "IC_clean_estimate")
NOISE_10 = DisturbanceSpec(type="observation_noise", severity=0.01, mask="positions")
NOISE_5 = DisturbanceSpec(type="observation_noise", severity=0.005, mask="positions")
EST = {"nominal_gain": 0.62, "window": 10, "max_lag": 4, "min_transitions": 4,
       "min_excitation": 0.000225, "min_r2": 0.8, "min_margin": 2.0}  # fmt: skip
NOISE_CELLS = ("H_noise0.01", "E_C8_noise0.01")
CLEAN_CELLS = ("H_undisturbed", "E_C0_undisturbed")


def j1_cells() -> list[G4Cell]:
    none = DelaySchedule(())
    return [G4Cell("J", "H_noise0.01", "noise", 0, (NOISE_10,), none),
            G4Cell("J", "E_C8_noise0.01", "noise", 1, (NOISE_10,), none),
            G4Cell("J", "J_noise0.005", "noise_half", 2, (NOISE_5,), none),
            G4Cell("J", "H_undisturbed", "clean", 3, (), none),
            G4Cell("J", "E_C0_undisturbed", "clean", 4, (), none)]  # fmt: skip


@dataclass(frozen=True)
class J1Config:
    raw: dict[str, Any]

    def seeds(self, role: str) -> list[int]:
        off, n = {"debugging": (self.raw["debug_seed_offset"], self.raw["debug_episodes"]),
                  "evaluation": (self.raw["evaluation_seed_offset"],
                                 self.raw["evaluation_episodes"])}[role]  # fmt: skip
        return list(range(off, off + n))


def load_j1(path: str | Path) -> J1Config:
    v = json.loads(Path(path).read_text())
    c = J1Config(v)
    for role in ("debugging", "evaluation"):
        check_j_seeds(c.seeds(role), role)
    for doc, key in (("protocol", "protocol_sha256"), ("recommendation", "recommendation_sha256")):
        if hashlib.sha256(Path(v[doc]).read_bytes()).hexdigest() != v[key]:
            raise AdaptationConfigError(f"{v[doc]} changed since the config pinned it")
    return c


def phase_i_episode(session: ScheduledDelaySession, policy, normalizer, arm: str,
                    clean_features: bool, log: list[dict[str, Any]] | None) -> bool:  # fmt: skip
    feat = Featurizer(arm, EST)
    while not session.done:
        view = session.observe()
        assert view.observation is not None
        clean_obs = np.asarray(session.observation["observation"], np.float64)
        feat.observe(clean_obs if clean_features else view.observation)
        fx = feat.features()
        obs = np.concatenate([np.asarray(view.observation), fx])
        command = np.clip(np.asarray(policy(normalizer(obs[None], view.goal[None]))[0],
                                     np.float64), -1.0, 1.0)  # fmt: skip
        if log is not None:
            log.append(
                {
                    "t": view.step,
                    "clean": clean_obs,
                    "view": np.asarray(view.observation),
                    "goal": np.asarray(view.goal),
                    "features": fx,
                    "action": command,
                }
            )
        feat.issued(command)
        session.act(command)
    return bool(session.final_success)


def run_task(args: tuple[J1Config, str, int, G4Cell, str, tuple[int, ...]]):
    config, root, k, cell, raw_dir, seeds = args
    torch.set_num_threads(1)
    frozen = load_policy(checkpoint_path(f"fetch_push_fp32_seed{k}"))
    actor = {n: v.clone() for n, v in frozen.actor.state_dict().items()}
    f_policy = deterministic_policy_function(frozen.actor)
    f_norm = FrozenNormalizer(frozen.normalizer)
    env = make_env(frozen.env_config)
    nets = {}
    for arm in ("IA", "IC"):
        pol, hyb = load_phase_i(Path(root) / f"phase_i_{arm}_seed{k}" / "checkpoint.pt",
                                f"phase_i_{arm}_seed{k}")  # fmt: skip
        nets[arm] = (deterministic_policy_function(pol.actor), hyb)
    episodes, logs = [], {}
    for seed in seeds:
        for name in CONFIGURATIONS:
            pseed = int(config.raw["perturbation_seed"])
            session = ScheduledDelaySession(env, f_policy, f_norm,
                                            ConditionSpec(cell.name, cell.extra), seed, pseed,
                                            cell.schedule(seed), "replay_drop")  # fmt: skip
            if name == "base_only":
                while not session.done:
                    view = session.observe()
                    session.act(np.clip(view.base_action, -1.0, 1.0))
                success = bool(session.final_success)
            else:
                arm = name[:2]
                log: list[dict[str, Any]] = []
                success = phase_i_episode(session, nets[arm][0], nets[arm][1], arm,
                                          name in PRIVILEGED, log)  # fmt: skip
                logs[(name, seed)] = log
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
    _write_logs(Path(raw_dir) / "steps", k, cell.name, logs)
    return episodes, _integrity(logs, cell)


def _write_logs(folder: Path, k: int, cell: str, logs: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    for (name, seed), rows in logs.items():
        for field in ("clean", "view", "goal", "features", "action"):
            arrays[f"{name}|{seed}|{field}"] = np.array([r[field] for r in rows], np.float32)
    path = folder / f"seed{k}_{cell}.npz"
    if path.exists():
        raise RuntimeError(f"{path} exists: raw logs are write-once")
    save: Any = np.savez_compressed
    save(path, **arrays)


def _integrity(logs: dict, cell: G4Cell) -> list[dict[str, Any]]:
    """On noise-free cells, privileged and deployed variants must act identically."""
    if cell.family != "clean":
        return []
    out = []
    for priv, dep in (("IA_clean_features", "IA"), ("IC_clean_estimate", "IC")):
        seeds = sorted(s for (n, s) in logs if n == priv)
        same = all(np.array_equal(np.array([r["action"] for r in logs[(priv, s)]]),
                                  np.array([r["action"] for r in logs[(dep, s)]]))
                   for s in seeds)  # fmt: skip
        out.append({"cell": cell.name, "pair": f"{priv} vs {dep}", "identical": bool(same)})
    return out


# --- pre-registered analysis --------------------------------------------------------------------


def derive_j1(e: pd.DataFrame, n: int = 10_000, s: int = 20262100) -> tuple[dict, dict]:
    noise = e[e["condition"].isin(NOISE_CELLS)]
    prim = pd.DataFrame([
        {"test": "J1", "configuration": "IA_clean_features", "reference": "IA",
         **two(noise, "IA_clean_features", "IA", 0.0, n, s)},
        {"test": "J2", "configuration": "IC_clean_estimate", "reference": "IC",
         **two(noise, "IC_clean_estimate", "IC", 0.0, n, s)},
    ])  # fmt: skip
    prim["p_holm"] = holm(prim["p"].tolist())
    prim["result"] = np.where(prim["p_holm"] <= 0.025, "supported", "not supported")
    gates = []
    for name in CONFIGURATIONS[1:]:
        r = two(noise, name, "base_only", -0.02, n, s)
        gates.append({"configuration": name, **r, "passes_noise_gate": r["ci_low"] > -0.02})
    desc = []
    for k, frame in list(noise.groupby("policy_seed")) + [("pooled", noise)]:
        for a, b in (
            ("IA_clean_features", "IA"),
            ("IC_clean_estimate", "IC"),
            ("IA", "base_only"),
            ("IC", "base_only"),
            ("IA_clean_features", "base_only"),
            ("IC_clean_estimate", "base_only"),
        ):
            desc.append({"scope": f"1 cm noise, seed {k}", "configuration": a, "reference": b,
                         **two(frame, a, b, 0.0, n, s)})  # fmt: skip
    half = e[e["condition"] == "J_noise0.005"]
    for a, b in (("IA_clean_features", "IA"), ("IC_clean_estimate", "IC"), ("IA", "base_only"),
                 ("IC", "base_only")):  # fmt: skip
        desc.append({"scope": "0.5 cm noise, pooled", "configuration": a, "reference": b,
                     **two(half, a, b, 0.0, n, s)})  # fmt: skip
    success = e.groupby(["condition", "configuration"])["success"].mean().unstack() * 100
    tables = {"primary": prim, "gate_checks": pd.DataFrame(gates),
              "descriptive": pd.DataFrame(desc), "success": success.reset_index()}  # fmt: skip
    summary = {"primary": {r["test"]: {"estimate": round(r["estimate"], 4),
                                       "ci": [round(r["ci_low"], 4), round(r["ci_high"], 4)],
                                       "p_holm": r["p_holm"], "result": r["result"]}
                           for r in prim.to_dict("records")},
               "gate_checks": {r["configuration"]: bool(r["passes_noise_gate"]) for r in gates},
               "episodes": len(e)}  # fmt: skip
    return tables, summary


DERIVED = ("primary", "gate_checks", "descriptive", "success")
