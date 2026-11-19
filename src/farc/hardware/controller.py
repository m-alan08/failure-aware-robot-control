"""Offline controller integration for the four frozen configurations (proposal section 4.4).

- F: ``stage3.Latched`` (computed, never used); no predictor; reproduces
  ``mechanism.run_episode(mode="base")``.
- K: ``stage3.Latched``; ``phase_e_predictor`` (kinematic); reproduces
  ``stage3.run_g2_episode("phase_e_predictor")``.
- CA: ``stage3.Latched``; ``learned_predictor`` (F1, move_object); reproduces
  ``mechanism.run_episode`` (G4 ``contact_learned``).
- C: ``candidates.Sequential`` via ``mechanism.sequential`` (h = 12, sigma0 = 4 mm, gauss);
  F1; reproduces ``mechanism.run_episode`` (G4 ``C``).

``HardwareController.step`` performs the reference loop body in the same order: base policy on
the observation, ``estimator.observe(gripper)``, ``ctrl.step(estimator)``, the predictor on the
queued commands with the controller's gain, the frozen policy at the predicted observation,
clip to [-1, 1], ``estimator.issued(command)``. No constant, weight or rule is re-implemented:
every scientific object is imported from its original module.

Each step is timed per component with ``time.perf_counter_ns`` and returns a log record.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.contact.episodes import Predictor, queued_commands
from farc.delaychange.candidates import Sequential
from farc.delaychange.mechanism import sequential
from farc.delaychange.stage3 import Latched
from farc.envs.observation_layout import OBJECT

S = OBJECT.slices
CONFIGURATIONS = ("F", "K", "CA", "C")
PREDICTOR = {"F": None, "K": "phase_e", "CA": "learned", "C": "learned"}
# Frozen constants, as pinned in configs/delaychange/g4_heldout.json (checked by a test).
ESTIMATOR: dict[str, Any] = {"nominal_gain": 0.62, "window": 10, "max_lag": 4, "min_transitions": 4,
             "min_excitation": 0.000225, "min_r2": 0.8, "min_margin": 2.0}  # fmt: skip
C_CONSTANTS: dict[str, Any] = {"C_threshold": 12.0, "C_sigma0_mm": 4.0, "C_variant": "gauss"}


@dataclass
class StepResult:
    command: np.ndarray
    record: dict[str, Any]


class HardwareController:
    def __init__(self, name: str, policy, normalizer, preds: dict[str, Predictor],
                 estimator: dict[str, Any] | None = None,
                 constants: dict[str, Any] | None = None):  # fmt: skip
        if name not in CONFIGURATIONS:
            raise ValueError(f"configuration must be one of {CONFIGURATIONS}")
        self.name, self.policy, self.normalizer, self.preds = name, policy, normalizer, preds
        self.est_cfg = EstimatorConfig(**(estimator or ESTIMATOR))
        self.constants = constants or C_CONSTANTS
        self.reset()

    def reset(self) -> None:
        self.est = ActuatorEstimator(self.est_cfg)
        self.ctrl: Any = sequential(self.constants) if self.name == "C" else Latched()
        self.issued: list[np.ndarray] = []
        self.t = 0

    def _act(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.asarray(self.policy(self.normalizer(obs, goal)[None])[0], np.float64)

    def step(self, obs: np.ndarray, goal: np.ndarray) -> StepResult:
        ns = time.perf_counter_ns
        t_start = ns()
        obs = np.asarray(obs, np.float64)
        base = self._act(obs, goal)
        t_base = ns()
        self.est.observe(obs[S["gripper_position"]].copy())
        t_obs = ns()
        dec = self.ctrl.step(self.est)
        t_ctrl = ns()
        d, active = dec.delay, dec.active
        gain = self.ctrl.gain
        if self.name == "F":
            active = False
        predicted = None
        t_pred = t_pol = t_ctrl
        if active:
            queued = queued_commands(self.issued, d)
            key = PREDICTOR[self.name]
            assert key is not None
            predicted = self.preds[key](obs.copy(), queued, gain)
            t_pred = ns()
            command = self._act(predicted, goal)
            t_pol = ns()
        else:
            command = base
        command = np.clip(command, -1.0, 1.0)
        self.est.issued(command)
        self.issued.append(np.asarray(command, np.float64).copy())
        t_end = ns()
        rec = self._record(obs, base, command, dec, d, active, gain, predicted)
        rec.update({
            "lat_base_policy_ns": t_base - t_start, "lat_estimator_observe_ns": t_obs - t_base,
            "lat_identification_supervision_ns": t_ctrl - t_obs,
            "lat_predictor_ns": t_pred - t_ctrl, "lat_compensated_policy_ns": t_pol - t_pred,
            "lat_controller_total_ns": t_end - t_start,
            "learned_calls": int(active and PREDICTOR[self.name] == "learned"),
        })  # fmt: skip
        self.t += 1
        return StepResult(command, rec)

    def _record(self, obs, base, command, dec, d, active, gain, predicted) -> dict[str, Any]:
        e = dec.estimate
        rec: dict[str, Any] = {
            "t": self.t, "configuration": self.name,
            "est_delay": None if e.delay is None else int(e.delay),
            "est_delay_reliable": bool(e.delay_reliable),
            "est_gain": None if not np.isfinite(e.gain) else float(e.gain),
            "est_gain_reliable": bool(e.gain_reliable),
            "delay_used": int(d), "gain_used": float(gain), "active": bool(active),
            "alarm": bool(dec.alarm), "suspended": bool(dec.suspended),
            "base_action": base.tolist(), "command": np.asarray(command).tolist(),
            "pred_gripper_position": None, "pred_object_position": None,
            "pred_object_motion_m": None,
        }  # fmt: skip
        if predicted is not None:
            rec["pred_gripper_position"] = predicted[S["gripper_position"]].tolist()
            rec["pred_object_position"] = predicted[S["object_position"]].tolist()
            rec["pred_object_motion_m"] = float(np.linalg.norm(
                predicted[S["object_position"]] - obs[S["object_position"]]))  # fmt: skip
        if isinstance(self.ctrl, Sequential):
            c, n = self.ctrl, self.est_cfg.max_lag + 1
            rec.update({
                "supervisor_mode": ("suspended" if c.suspended else "pending" if c.pending
                                    else "compensating" if c.trusted > 0 else "monitoring"),
                "cusum_stats": [float(c.stats.get(k, 0.0)) for k in range(n)],
                "cusum_trusted": int(c.trusted), "cusum_reset_start": int(c.reset_start),
                "cusum_sigma2": None if c.sigma2 is None else float(c.sigma2),
            })  # fmt: skip
        else:
            rec.update({"supervisor_mode": "latch_active" if active else "latch_idle",
                        "cusum_stats": None, "cusum_trusted": None, "cusum_reset_start": None,
                        "cusum_sigma2": None})  # fmt: skip
        return rec
