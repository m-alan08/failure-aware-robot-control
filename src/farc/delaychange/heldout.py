"""Phase G4: held-out confirmation of Candidate C (docs/PHASE_G4_HELDOUT_PROPOSAL.md).

Two arms on the same held-out episode seeds:
- R: G3's random per-episode schedules (primary claims);
- F: G2's fixed Stage-3 grid (G2's claims), its P, T, I and NG cells.

Every controller is an existing one: Candidate C (`Sequential`, pinned constants), the latch
(`contact_learned`), and the G2/G3 shams built from them. Nothing here changes C.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.adaptation.experiment import AdaptationConfigError
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.dynamics import load_dynamics
from farc.contact.episodes import learned_predictor, phase_e_predictor
from farc.delaychange.mechanism import (
    events_of,
    replay,
    rotate,
    run_episode,
    sequential,
    suspended_of,
    yoked_events,
)
from farc.delaychange.random_schedules import g3_cells, schedule_for
from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
from farc.delaychange.stage3 import Latched, evaluation_cells, run_g2_episode
from farc.disturbances.config import ConditionSpec, DisturbanceSpec
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.supervision.seeds import check_g4_seeds

KEYS = ["policy_seed", "condition", "episode_seed"]
ARM_R = ("base_only", "contact_learned", "C", "sham_C_cross_yoked", "sham_C_random_pause")
ARM_F = ("base_only", "contact_learned", "C", "sham_C_same_condition")
F_FAMILY = {"P": "F-P", "T": "F-T", "I": "F-I", "NG": "NG"}


@dataclass(frozen=True)
class G4Cell:
    arm: str  # R | F
    name: str
    family: str  # RT RI RS RP H | F-P F-T F-I NG
    index: int  # the cell's index in its source list (G3 generator / rotation seed)
    extra: tuple[DisturbanceSpec, ...]
    fixed: DelaySchedule | None  # arm F only

    def schedule(self, seed: int) -> DelaySchedule:
        if self.fixed is not None:
            return self.fixed
        return schedule_for(g3_cells()[self.index], self.index, seed)


def g4_cells() -> list[G4Cell]:
    cells = [G4Cell("R", c.name, c.family, i, c.extra, None) for i, c in enumerate(g3_cells())]
    for i, c in enumerate(evaluation_cells()):
        if c.family in F_FAMILY:
            cells.append(G4Cell("F", c.name, F_FAMILY[c.family], i, c.extra, c.schedule))
    return cells


@dataclass(frozen=True)
class G4Config:
    name: str
    plan: str
    plan_sha256: str
    model: str
    model_sha256: str
    checkpoints: tuple[str, ...]
    debug_seed_offset: int
    debug_episodes: int
    confirmation_seed_offset: int
    confirmation_episodes: int
    perturbation_seed: int
    estimator: dict[str, Any]
    constants: dict[str, Any]
    alpha_one_sided: float
    practical_threshold: float
    specificity_minimum: float
    persistent_noninferiority: float
    sac_noninferiority: float
    robustness_noninferiority: float
    family_harm_floor: float
    harm_ci_floor: float
    harm_max_fraction: float
    bootstrap_resamples: int
    bootstrap_seed: int
    output: str
    role: str = "confirmation"

    @property
    def seeds(self) -> list[int]:
        offset, count = {"debugging": (self.debug_seed_offset, self.debug_episodes),
                         "confirmation": (self.confirmation_seed_offset,
                                          self.confirmation_episodes)}[self.role]  # fmt: skip
        return list(range(offset, offset + count))


def load_g4(path: str | Path, role: str = "confirmation") -> G4Config:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    values["checkpoints"] = tuple(values["checkpoints"])
    config = G4Config(**values, role=role)
    for r in ("debugging", "confirmation"):
        check_g4_seeds(G4Config(**values, role=r).seeds, r)
    if hashlib.sha256(Path(config.plan).read_bytes()).hexdigest() != config.plan_sha256:
        raise AdaptationConfigError("the G4 plan changed since the config pinned it")
    return config


def _setup(config: G4Config, checkpoint: str):
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    policy = deterministic_policy_function(loaded.actor)
    normalizer, env = FrozenNormalizer(loaded.normalizer), make_env(loaded.env_config)
    est_cfg = EstimatorConfig(**config.estimator)
    model = load_dynamics(resolve(config.model), config.model_sha256)
    preds = {"learned": learned_predictor(model, move_object=True),
             "phase_e": phase_e_predictor(est_cfg.nominal_gain)}  # fmt: skip
    return loaded, policy, normalizer, env, est_cfg, preds


def _session(env, policy, normalizer, cell: G4Cell, seed: int, config: G4Config):
    return ScheduledDelaySession(env, policy, normalizer, ConditionSpec(cell.name, cell.extra),
                                 seed, config.perturbation_seed, cell.schedule(seed),
                                 "replay_drop")  # fmt: skip


def run_task(args: tuple[G4Config, str, G4Cell]):
    config, checkpoint, cell = args
    loaded, policy, normalizer, env, est_cfg, preds = _setup(config, checkpoint)
    actor = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    seeds, k, ps = config.seeds, config.constants, loaded.training_seed
    names = ARM_R if cell.arm == "R" else ARM_F
    c_events: dict[int, list[tuple[int, int]]] = {}
    c_mask: dict[int, np.ndarray] = {}
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for sham_phase in (False, True):
        for seed in seeds:
            for name in names:
                if name.startswith("sham_") != sham_phase:
                    continue
                nxt = seeds[(seeds.index(seed) + 1) % len(seeds)]
                kw: dict[str, Any] = {}
                ctrl: Any = Latched()
                if name == "base_only":
                    kw["mode"] = "base"
                elif name == "C":
                    ctrl = sequential(k)
                elif name == "sham_C_cross_yoked":  # G3 sham_full_yoked
                    ctrl = replay(k, yoked_events(c_events[nxt]), True, True)
                elif name == "sham_C_random_pause":  # G3 sham_pause_random
                    kw["force_off"] = rotate(c_mask[seed], ps, cell.index, seed)
                elif name == "sham_C_same_condition":  # G2 sham_C_yoked
                    kw["force_off"] = c_mask[nxt]
                session = _session(env, policy, normalizer, cell, seed, config)
                result = run_episode(session, ActuatorEstimator(est_cfg), ctrl, preds, **kw)
                rows = result["rows"]
                if name == "C":
                    c_events[seed], c_mask[seed] = events_of(rows), suspended_of(rows)
                key = {
                    "arm": cell.arm,
                    "configuration": name,
                    "policy_seed": ps,
                    "condition": cell.name,
                    "family": cell.family,
                    "episode_seed": seed,
                }
                episodes.append({**key, "success": result["success"],
                                 "final_distance": result["final_distance"],
                                 "alarms": int(sum(r["alarm"] for r in rows))})  # fmt: skip
                steps.extend({**key, **{c: r[c] for c in ("t", "delay_true", "delay_used",
                                                          "active", "alarm", "suspended",
                                                          "ctrl_ns", "learned_calls")}}
                             for r in rows)  # fmt: skip
    for name, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor[name]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes, steps


def equivalence_check(args: tuple[G4Config, str, G4Cell]) -> list[dict[str, Any]]:
    """Code check (debugging seeds only): C, contact-aware and the same-condition sham through
    the G4 runner reproduce the G2 Stage-3 runner's issued commands bitwise."""
    config, checkpoint, cell = args
    _, policy, normalizer, env, est_cfg, preds = _setup(config, checkpoint)
    g2_constants = {"A_window": 7, "B_fast_window": 4, **config.constants}
    out = []
    for seed in config.seeds:
        cmds: dict[str, np.ndarray] = {}
        mask = None
        for label, g2_name in (("C", "cand_C_sequential"), ("contact", "contact_learned")):
            s = _session(env, policy, normalizer, cell, seed, config)
            ctrl: Any = sequential(config.constants) if label == "C" else Latched()
            r = run_episode(s, ActuatorEstimator(est_cfg), ctrl, preds)
            cmds[f"g4_{label}"] = np.array(s.commands)
            if label == "C":
                mask = suspended_of(r["rows"])
            s = _session(env, policy, normalizer, cell, seed, config)
            run_g2_episode(s, ActuatorEstimator(est_cfg), g2_name, preds, g2_constants)
            cmds[f"g2_{label}"] = np.array(s.commands)
        assert mask is not None
        s = _session(env, policy, normalizer, cell, seed, config)
        run_episode(s, ActuatorEstimator(est_cfg), Latched(), preds, force_off=mask)
        cmds["g4_sham"] = np.array(s.commands)
        s = _session(env, policy, normalizer, cell, seed, config)
        run_g2_episode(s, ActuatorEstimator(est_cfg), "sham_C_yoked", preds, g2_constants, mask)
        cmds["g2_sham"] = np.array(s.commands)
        out.append({"arm": cell.arm, "condition": cell.name, "episode_seed": seed,
                    "c_paused": bool(mask.any()),
                    **{f"{x}_identical": bool(np.array_equal(cmds[f"g4_{x}"], cmds[f"g2_{x}"]))
                       for x in ("C", "contact", "sham")}})  # fmt: skip
    return out
