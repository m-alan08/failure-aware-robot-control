"""G2 Stages 2-3: calibration and the pre-registered development evaluation
(docs/PHASE_G2_PROPOSAL.md, section 5)."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.compensation import LatchedEstimate
from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig, lag_residuals
from farc.adaptation.experiment import AdaptationConfigError
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.dynamics import load_dynamics
from farc.contact.episodes import Predictor, learned_predictor, phase_e_predictor, queued_commands
from farc.delaychange.candidates import (
    Decision,
    Sequential,
    ShortWindow,
    TwoTimescale,
)
from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
from farc.disturbances.config import ConditionSpec, DisturbanceSpec
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.supervision.seeds import check_g2_seeds
from farc.supervision.supervisors import ReleaseSupervisor, SupervisorConfig

KEYS = ["policy_seed", "condition", "episode_seed"]
NOISE = DisturbanceSpec(type="observation_noise", severity=0.005, mask="positions")
NOISE_H = DisturbanceSpec(type="observation_noise", severity=0.01, mask="positions")
SCALE = DisturbanceSpec(type="action_scale", severity=0.5)


@dataclass(frozen=True)
class Cell:
    name: str
    family: str
    pulses: tuple[tuple[int, int | None, int], ...]
    extra: tuple[DisturbanceSpec, ...] = ()

    @property
    def schedule(self) -> DelaySchedule:
        return DelaySchedule(self.pulses)


def evaluation_cells() -> list[Cell]:
    """The pre-registered Stage-3 grid (section 5.1): 38 conditions."""
    cells = [Cell(f"P_d{d}", "P", ((0, None, d),)) for d in (1, 2, 3, 4)]
    cells += [Cell(f"P_d{d}_on18", "P", ((18, None, d),)) for d in (2, 4)]
    cells += [Cell(f"T_d{d}_on{o}_dur{L}", "T", ((o, L, d),))
              for d in (2, 4) for o in (6, 18, 30) for L in (4, 8, 16)]  # fmt: skip
    for d in (2, 4):
        for p in (4, 8):
            cells.append(Cell(f"I_d{d}_p{p}", "I",
                              tuple((o, p, d) for o in range(6, 50, 2 * p))))  # fmt: skip
    cells += [Cell(f"S_{a}to{b}", "S", ((0, 20, a), (20, None, b)))
              for a, b in ((1, 4), (4, 1), (2, 4), (4, 2))]  # fmt: skip
    cells += [
        Cell("NG_d4_noise", "NG", ((0, None, 4),), (NOISE,)),
        Cell("NG_d4_scale", "NG", ((0, None, 4),), (SCALE,)),
        Cell("NG_T_d4_on18_dur8_noise", "NG", ((18, 8, 4),), (NOISE,)),
        Cell("NG_T_d4_on18_dur8_scale", "NG", ((18, 8, 4),), (SCALE,)),
    ]
    cells += [Cell("H_undisturbed", "H", ()), Cell("H_noise0.01", "H", (), (NOISE_H,))]
    return cells


def calibration_cells() -> list[Cell]:
    """Section 5.3: persistent d in {1..4}, both H conditions, two generic transient cells."""
    cells = [Cell(f"K_d{d}", "P", ((0, None, d),)) for d in (1, 2, 3, 4)]
    cells += [Cell("KH_undisturbed", "H", ()), Cell("KH_noise0.01", "H", (), (NOISE_H,))]
    cells += [Cell(f"KT_d3_on12_dur{L}", "T", ((12, L, 3),)) for L in (6, 12)]
    return cells


EVALUATION = ("base_only", "contact_learned", "phase_e_predictor", "sup_A_release",
              "cand_A_short_window", "cand_B_two_timescale", "cand_C_sequential",
              "PRIV_true_schedule", "PRIV_release_oracle",
              "sham_B_yoked", "sham_C_yoked")  # fmt: skip
YOKED = {"sham_B_yoked": "cand_B_two_timescale", "sham_C_yoked": "cand_C_sequential"}
CANDIDATES = ("cand_A_short_window", "cand_B_two_timescale", "cand_C_sequential")
PRIVILEGED = ("PRIV_true_schedule", "PRIV_release_oracle")


def calibration_configurations() -> tuple[str, ...]:
    names = ["contact_learned"]
    names += [f"cand_A_w{w}" for w in (4, 5, 6, 7)]
    names += [f"cand_B_wf{w}" for w in (4, 5)]  # w_f = 3 can never be reliable (A7)
    names += [f"cand_C_{v}_h{h:g}_s{s}" for v in ("gauss", "scalefree")
              for h in (1, 2, 3, 4, 6, 8, 12) for s in (1, 2, 4)]  # fmt: skip
    return tuple(names)


# --- controllers --------------------------------------------------------------------------------


@dataclass
class Latched:
    latch: LatchedEstimate = field(default_factory=LatchedEstimate)

    def step(self, est: ActuatorEstimator) -> Decision:
        e = est.estimate()
        d = int(self.latch.parameters(e).delay or 0)
        return Decision(d > 0, d, e)

    @property
    def gain(self) -> float:
        return float(self.latch.gain) if self.latch.gain is not None else 1.0


@dataclass
class G1Release:
    """G1 supervisor A, unchanged (h = 0.5, dwell 5)."""

    supervisor: ReleaseSupervisor
    latch: LatchedEstimate = field(default_factory=LatchedEstimate)

    def step(self, est: ActuatorEstimator) -> Decision:
        e = est.estimate()
        d = int(self.latch.parameters(e).delay or 0)
        transitions, sse = lag_residuals(est)
        active = self.supervisor.allow(e.step, d, e, transitions, sse)
        return Decision(active, d, e, suspended=self.supervisor.released)

    @property
    def gain(self) -> float:
        return float(self.latch.gain) if self.latch.gain is not None else 1.0


def make_controller(name: str, constants: dict[str, Any]):
    if name.startswith("cand_A_w"):
        return ShortWindow(int(name.rsplit("w", 1)[1]))
    if name.startswith("cand_B_wf"):
        return TwoTimescale(int(name.rsplit("wf", 1)[1]))
    if name.startswith("cand_C_") and name != "cand_C_sequential":
        _, _, variant, h, s = name.split("_")
        return Sequential(float(h[1:]), float(s[1:]) / 1000.0, variant)
    if name == "cand_A_short_window":
        return ShortWindow(int(constants["A_window"]))
    if name == "cand_B_two_timescale":
        return TwoTimescale(int(constants["B_fast_window"]))
    if name == "cand_C_sequential":
        return Sequential(float(constants["C_threshold"]), float(constants["C_sigma0_mm"]) / 1000,
                          str(constants["C_variant"]))  # fmt: skip
    if name == "sup_A_release":
        return G1Release(ReleaseSupervisor(SupervisorConfig(cusum_threshold=0.5), "cusum"))
    return Latched()


def run_g2_episode(session: ScheduledDelaySession, est: ActuatorEstimator, name: str,
                   preds: dict[str, Predictor], constants: dict[str, Any],
                   yoked: np.ndarray | None = None) -> dict[str, Any]:  # fmt: skip
    if (name in YOKED) != (yoked is not None):
        raise ValueError("yoked shams need a source mask, and only they take one")
    ctrl = make_controller(name, constants)
    schedule = session.schedule
    last_end: int | None = None
    predictor = "phase_e" if name == "phase_e_predictor" else "learned"
    rows = []
    while not session.done:
        view = session.observe()
        assert view.observation is not None
        obs, t = view.observation, view.step
        d_true = schedule.delay_at(t)
        if t > 0 and schedule.delay_at(t - 1) > 0 and d_true == 0:
            last_end = t
        t0 = time.perf_counter_ns()
        est.observe(view.gripper)
        dec = ctrl.step(est)
        d, active = dec.delay, dec.active
        if name == "base_only":
            active = False
        elif name == "PRIV_true_schedule":  # PRIVILEGED
            d, active = d_true, d_true > 0
        elif name == "PRIV_release_oracle" and last_end is not None and d_true == 0:
            active = False  # PRIVILEGED: off exactly from every true end to the next onset
        elif yoked is not None and t < len(yoked) and yoked[t]:
            active = False
        calls = 0
        gain = ctrl.gain
        if name == "PRIV_true_schedule":  # PRIVILEGED: the true gain as well (A10)
            gain = float(np.prod([1.0 - x.severity for x in session.condition.disturbances
                                  if x.type == "action_scale" and x.active(t)]))  # fmt: skip
        if active:
            queued = queued_commands(list(session.commands), d)
            predicted = preds[predictor](obs.copy(), queued, gain)
            calls = int(predictor == "learned")
            command = np.asarray(session.policy(session.normalizer(predicted, view.goal)[None])[0],
                                 np.float64)  # fmt: skip
        else:
            command = view.base_action
        command = np.clip(command, -1.0, 1.0)
        ns = time.perf_counter_ns() - t0
        session.act(command)
        est.issued(command)
        rows.append({"t": t, "delay_true": d_true, "delay_used": d, "active": bool(active),
                     "alarm": dec.alarm, "suspended": dec.suspended,
                     "est_delay": dec.estimate.delay, "est_reliable": dec.estimate.delay_reliable,
                     "ctrl_ns": ns, "learned_calls": calls})  # fmt: skip
    return {"success": session.final_success, "final_distance": session.final_distance,
            "rows": rows}  # fmt: skip


# --- config and tasks -------------------------------------------------------------------------


@dataclass(frozen=True)
class G2Config:
    name: str
    plan: str
    plan_sha256: str
    model: str
    model_sha256: str
    checkpoints: tuple[str, ...]
    debug_seed_offset: int
    debug_episodes: int
    calibration_seed_offset: int
    calibration_episodes: int
    evaluation_seed_offset: int
    evaluation_episodes: int
    perturbation_seed: int
    estimator: dict[str, Any]
    constants: dict[str, Any]
    calibration_record: str | None
    calibration_record_sha256: str | None
    alpha_one_sided: float
    transient_margin_practical: float
    persistent_noninferiority: float
    sac_noninferiority: float
    robustness_noninferiority: float
    harm_ci_floor: float
    harm_max_fraction: float
    bootstrap_resamples: int
    bootstrap_seed: int
    output: str
    role: str = "evaluation"

    @property
    def seeds(self) -> list[int]:
        offset, count = {"debugging": (self.debug_seed_offset, self.debug_episodes),
                         "calibration": (self.calibration_seed_offset, self.calibration_episodes),
                         "evaluation": (self.evaluation_seed_offset,
                                        self.evaluation_episodes)}[self.role]  # fmt: skip
        return list(range(offset, offset + count))


def load_g2(path: str | Path, role: str = "evaluation") -> G2Config:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    values["checkpoints"] = tuple(values["checkpoints"])
    config = G2Config(**values, role=role)
    for r in ("debugging", "calibration", "evaluation"):
        check_g2_seeds(G2Config(**values, role=r).seeds, r)
    if hashlib.sha256(Path(config.plan).read_bytes()).hexdigest() != config.plan_sha256:
        raise AdaptationConfigError("the G2 plan changed since the config pinned it")
    if role == "evaluation":
        if not config.calibration_record or not config.calibration_record_sha256:
            raise AdaptationConfigError("the evaluation needs the pinned calibration record")
        digest = hashlib.sha256(Path(config.calibration_record).read_bytes()).hexdigest()
        if digest != config.calibration_record_sha256:
            raise AdaptationConfigError("the calibration record changed since it was pinned")
        missing = {"A_window", "B_fast_window", "C_threshold", "C_sigma0_mm", "C_variant"} - set(
            config.constants
        )  # None = failed calibration (A8)
        if missing:
            raise AdaptationConfigError(f"constants missing from the config: {sorted(missing)}")
    return config


def run_task(args: tuple[G2Config, str, Cell, tuple[str, ...]]):
    config, checkpoint, cell, configurations = args
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    actor = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    policy = deterministic_policy_function(loaded.actor)
    normalizer, env = FrozenNormalizer(loaded.normalizer), make_env(loaded.env_config)
    est_cfg = EstimatorConfig(**config.estimator)
    model = load_dynamics(resolve(config.model), config.model_sha256)
    preds = {"learned": learned_predictor(model, move_object=True),
             "phase_e": phase_e_predictor(est_cfg.nominal_gain)}  # fmt: skip
    seeds = config.seeds
    ordered = [c for c in configurations if c not in YOKED] + [
        c for c in configurations if c in YOKED]  # fmt: skip
    masks: dict[tuple[str, int], np.ndarray] = {}
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for yoked_phase in (False, True):
        for seed in seeds:
            for name in ordered:
                if (name in YOKED) != yoked_phase:
                    continue
                mask = None
                if name in YOKED:
                    source = seeds[(seeds.index(seed) + 1) % len(seeds)]
                    mask = masks[(YOKED[name], source)]
                session = ScheduledDelaySession(
                    env, policy, normalizer, ConditionSpec(cell.name, cell.extra), seed,
                    config.perturbation_seed, cell.schedule, "replay_drop")  # fmt: skip
                result = run_g2_episode(session, ActuatorEstimator(est_cfg), name, preds,
                                        config.constants, mask)  # fmt: skip
                if name in YOKED.values():
                    masks[(name, seed)] = np.array([r["suspended"] for r in result["rows"]])
                key = {
                    "configuration": name,
                    "policy_seed": loaded.training_seed,
                    "condition": cell.name,
                    "family": cell.family,
                    "episode_seed": seed,
                }
                episodes.append({**key, "success": result["success"],
                                 "final_distance": result["final_distance"]})  # fmt: skip
                steps.extend({**key, **r} for r in result["rows"])
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor[k]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes, steps


# --- calibration (detection criteria only; success is never written or read) ------------------


def detection_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    """Per configuration, from calibration logs:
    - wrong_reliable_rate: steps with a reliable estimate whose lag differs from the truth,
      over all steps of the persistent cells;
    - suspended_rate: off steps after first engagement while the true delay is > 0, over
      such steps (persistent cells);
    - false_activation_rate: active steps over all steps (H cells);
    - release_latency_median: median steps from each true end to the first off step on the
      transient cells, an end never followed by an off step counting as infinite."""
    out = []
    for name, sub in steps.groupby("configuration"):
        p = sub[sub["condition"].str.startswith("K_d")]
        wrong = (p["est_reliable"] & (p["est_delay"].fillna(-1) != p["delay_true"])).mean()
        engaged = (p.sort_values("t").groupby(KEYS)["active"].cummax()) & (p["delay_true"] > 0)
        suspended = float((engaged & ~p["active"]).sum() / max(engaged.sum(), 1))
        h = sub[sub["condition"].str.startswith("KH_")]
        latencies = []
        for _, ep in sub[sub["condition"].str.startswith("KT_")].groupby(KEYS):
            ep = ep.sort_values("t")
            dt, act = ep["delay_true"].to_numpy(), ep["active"].to_numpy(bool)
            for t in range(1, len(dt)):
                if dt[t - 1] > 0 and dt[t] == 0:
                    off = np.flatnonzero(~act[t:])
                    latencies.append(float(off[0]) if len(off) else np.inf)
        out.append({"configuration": name, "wrong_reliable_rate": float(wrong),
                    "suspended_rate": suspended, "false_activation_rate": float(h["active"].mean()),
                    "release_latency_median": float(np.median(latencies)) if latencies else np.nan,
                    "release_events": len(latencies)})  # fmt: skip
    return pd.DataFrame(out)


def choose_constants(metrics: pd.DataFrame) -> dict[str, Any]:
    """Section 5.3 (with A7-A10 in the pre-calibration record)."""
    m = metrics.set_index("configuration")

    def v(name: str, column: str) -> float:
        return float(m.at[name, column])  # type: ignore[arg-type]

    ref = v("contact_learned", "wrong_reliable_rate")
    a_ok = [w for w in (4, 5, 6, 7)
            if v(f"cand_A_w{w}", "wrong_reliable_rate") <= ref + 0.005
            and v(f"cand_A_w{w}", "false_activation_rate") <= 0.005]  # fmt: skip
    b_ok = [w for w in (4, 5)
            if v(f"cand_B_wf{w}", "suspended_rate") <= 0.01
            and v(f"cand_B_wf{w}", "false_activation_rate") <= 0.005]  # fmt: skip
    c = m[m.index.str.startswith("cand_C_")].copy()
    c = c[(c["suspended_rate"] <= 0.01) & (c["false_activation_rate"] <= 0.005)]
    chosen_c = None
    if len(c):
        parts = c.index.to_series().str.split("_", expand=True)
        c = c.assign(variant=parts[2], h=parts[3].str[1:].astype(float),
                     s=parts[4].str[1:].astype(float))  # fmt: skip
        c = c.sort_values(
            ["release_latency_median", "h", "s", "variant"], ascending=[True, False, False, True]
        )  # gauss sorts before scalefree
        best = c.iloc[0]
        chosen_c = {"C_threshold": float(best["h"]), "C_sigma0_mm": float(best["s"]),
                    "C_variant": str(best["variant"])}  # fmt: skip
    return {"A_window": min(a_ok) if a_ok else None,
            "B_fast_window": min(b_ok) if b_ok else None,
            **(chosen_c or {"C_threshold": None, "C_sigma0_mm": None, "C_variant": None}),
            "reference_wrong_reliable_rate": ref}  # fmt: skip
