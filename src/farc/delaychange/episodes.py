"""Stage-1 diagnostic episodes (docs/PHASE_G2_PROPOSAL.md, section 3).

A configuration is (delay source, predictor, optional privileged override):

- delay source: ``latched`` (deployable Phase E estimator + latch), ``never_release``
  (deployable "always-on" latch), ``true`` (PRIVILEGED: the true current delay d(t)),
  ``schedule`` (PRIVILEGED, simulator predictor only: steps until the command issued now
  executes under the true future schedule), or ``none`` (frozen SAC).
- predictor: ``phase_e``, ``learned`` (F1), or ``sim`` (PRIVILEGED simulator look-ahead).
- overrides (PRIVILEGED unless noted): ``release_after_end`` = k (compensation forced off from
  k steps after every true end until the next onset; M1), ``forced_off`` and ``forced_on``
  windows (M4), ``switch_to_true_at`` = t (latched learned until t, then true learned; the M1
  point-of-no-return branches).

Every PRIVILEGED configuration is a diagnostic. None is a deployable controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from farc.adaptation.compensation import LatchedEstimate
from farc.adaptation.estimator import ActuatorEstimator
from farc.contact.episodes import Predictor, oracle_prediction, queued_commands
from farc.delaychange.schedule import ScheduledDelaySession
from farc.envs.observation_layout import OBJECT
from farc.supervision.supervisors import NeverReleaseLatch

S = OBJECT.slices


@dataclass(frozen=True)
class Diagnostic:
    name: str
    source: str  # none | latched | never_release | true | schedule
    predictor: str = "none"  # none | phase_e | learned | sim
    release_after_end: int | None = None
    forced_off: tuple[int, int] | None = None  # (start, length)
    forced_on: tuple[int, int, int] | None = None  # (start, length, delay)
    switch_to_true_at: int | None = None

    @property
    def privileged(self) -> bool:
        return (self.source in ("true", "schedule") or self.predictor == "sim"
                or self.release_after_end is not None or self.forced_off is not None
                or self.forced_on is not None or self.switch_to_true_at is not None)  # fmt: skip


def _diagnostics() -> dict[str, Diagnostic]:
    items = [
        Diagnostic("base_only", "none"),
        Diagnostic("deployed_phase_e", "latched", "phase_e"),
        Diagnostic("deployed_learned", "latched", "learned"),
        Diagnostic("never_release", "never_release", "learned"),
        *[Diagnostic(f"release_k{k}", "latched", "learned", release_after_end=k)
          for k in (0, 1, 2, 3, 5, 7)],
        Diagnostic("true_phase_e", "true", "phase_e"),
        Diagnostic("true_learned", "true", "learned"),
        Diagnostic("true_sim", "true", "sim"),
        Diagnostic("latched_sim", "latched", "sim"),
        Diagnostic("schedule_sim", "schedule", "sim"),
        *[Diagnostic(f"forced_off_L{L}", "true", "learned", forced_off=(18, L)) for L in (3, 6)],
        *[Diagnostic(f"forced_on_L{L}", "none", "learned", forced_on=(18, L, 3))
          for L in (3, 6, 9)],
    ]  # fmt: skip
    return {d.name: d for d in items}


DIAGNOSTICS = _diagnostics()


def branch(t: int) -> Diagnostic:
    return Diagnostic(f"branch_t{t}", "latched", "learned", switch_to_true_at=t)


def run_diagnostic_episode(
    session: ScheduledDelaySession,
    estimator: ActuatorEstimator,
    diag: Diagnostic,
    predictors: dict[str, Predictor],
) -> dict[str, Any]:
    latch = NeverReleaseLatch() if diag.source == "never_release" else LatchedEstimate()
    schedule, horizon = session.schedule, session.horizon
    last_end: int | None = None
    rows: list[dict[str, Any]] = []
    while not session.done:
        view = session.observe()
        assert view.observation is not None
        obs, t = view.observation, view.step
        true_obs = np.asarray(session.observation["observation"], np.float64).copy()
        estimator.observe(view.gripper)
        estimate = estimator.estimate()
        params = latch.parameters(estimate)
        latched = int(params.delay or 0)
        d_true = schedule.delay_at(t)
        if t > 0 and schedule.delay_at(t - 1) > 0 and d_true == 0:
            last_end = t
        source, predictor = diag.source, diag.predictor
        if diag.switch_to_true_at is not None and t >= diag.switch_to_true_at:
            source = "true"
        if source in ("latched", "never_release"):
            d = latched
        elif source == "true":
            d = d_true
        elif source == "schedule":
            d = schedule.command_horizon(t, horizon) or 0
        else:
            d = 0
        if (diag.release_after_end is not None and last_end is not None and d_true == 0
                and t >= last_end + diag.release_after_end):  # fmt: skip
            d = 0
        if diag.forced_off is not None and diag.forced_off[0] <= t < sum(diag.forced_off):
            d = 0
        if diag.forced_on is not None:
            start, length, delay = diag.forced_on
            if start <= t < start + length:
                d = delay
        active = bool(d) and predictor != "none"
        predicted = None
        placeholders = 0
        if active:
            if predictor == "sim":
                predicted = oracle_prediction(session, d)
                predicted[S["finger_position"]] = obs[S["finger_position"]]
                placeholders = schedule.placeholders(t, min(d, horizon - t))
            else:
                gain = float(params.gain) if (source == "latched" and params.gain) else 1.0
                queued = queued_commands(list(session.commands), d)
                predicted = predictors[predictor](obs.copy(), queued, gain)
            normalized = session.normalizer(predicted, view.goal)[None]
            command = np.asarray(session.policy(normalized)[0], np.float64)
        else:
            command = view.base_action
        command = np.clip(command, -1.0, 1.0)
        session.act(command)
        estimator.issued(command)
        row = {"t": t, "delay_true": d_true, "delay_used": d, "latched": latched,
               "active": active, "est_delay": estimate.delay,
               "est_reliable": estimate.delay_reliable, "placeholders": placeholders}  # fmt: skip
        for key in ("gripper_position", "object_position"):
            for i, v in enumerate(true_obs[S[key]]):
                row[f"true_{key}{i}"] = v
            if predicted is not None:
                for i, v in enumerate(predicted[S[key]]):
                    row[f"pred_{key}{i}"] = v
        rows.append(row)
    return {"success": session.final_success, "final_distance": session.final_distance,
            "rows": rows}  # fmt: skip
