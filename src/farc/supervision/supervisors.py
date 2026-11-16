"""Phase G1 supervisors (docs/PHASE_G1_PROPOSAL.md, section 2.2).

Each supervisor wraps the *unchanged* Phase E estimator and latch and the unchanged F1/Phase E
predictors. It sees only what the controller sees at decision time: the latched parameters,
the current estimate, the estimator's own window residuals, the controller's (possibly noisy)
observation, and its own earlier forecasts. No simulator, session or ground truth reaches it.

Common rules:
- minimum dwell: ``min_dwell`` steps after any mode change the supervisor itself makes;
- a released compensator re-engages only through a fresh reliable estimate with d > 0;
- the existing latch path is untouched: a reliable d = 0 still switches compensation off.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from farc.adaptation.compensation import LatchedEstimate, Parameters
from farc.adaptation.estimator import Estimate
from farc.envs.observation_layout import OBJECT

S = OBJECT.slices
NEVER = -(10**9)


@dataclass(frozen=True)
class SupervisorConfig:
    min_dwell: int = 5
    cusum_kappa: float = math.log(2.0)
    cusum_threshold: float | None = None  # None: never release (calibration traces)
    timeout_steps: int = 10
    ewma_span: int = 5
    selection_margin: float = 0.8
    contact_open_m: float = 0.05
    contact_close_m: float = 0.06
    min_transitions: int = 4  # the estimator's own minimum, for the CUSUM increment


class NeverReleaseLatch(LatchedEstimate):
    """The "always-on" control: like the latch, but once a reliable d > 0 has engaged
    compensation, a later reliable d = 0 is ignored. A later reliable d > 0 still updates d."""

    def __init__(self) -> None:
        super().__init__()
        self.engaged = False

    def parameters(self, estimate: Estimate) -> Parameters:
        if estimate.gain_reliable:
            self.gain = estimate.gain
        if estimate.delay_reliable and estimate.delay is not None:
            if estimate.delay > 0:
                self.delay, self.engaged = estimate.delay, True
            elif not self.engaged:
                self.delay = estimate.delay
        return Parameters(self.gain, self.delay)


def cusum_increment(transitions: int, sse: np.ndarray, delay: int, cfg: SupervisorConfig):
    """log(SSE(d) / SSE(0)) - kappa on the current window, or None when it is undefined
    (too few transitions, or a lag with no commanded displacement)."""
    if transitions < cfg.min_transitions or not (np.isfinite(sse[delay]) and np.isfinite(sse[0])):
        return None
    return math.log(max(float(sse[delay]), 1e-15) / max(float(sse[0]), 1e-15)) - cfg.cusum_kappa


@dataclass
class ReleaseSupervisor:
    """Supervisor A (``cusum``), its timeout ablation (``timeout``), and the yoked sham
    (``yoked``: released exactly at the steps given by ``mask``)."""

    cfg: SupervisorConfig
    kind: Literal["cusum", "timeout", "yoked"]
    mask: np.ndarray | None = None
    released: bool = False
    statistic: float = 0.0
    last_switch: int = NEVER
    last_confirm: int = NEVER
    delay: int = 0
    increment: float | None = field(default=None, init=False)

    def allow(self, t: int, latched: int, estimate: Estimate, transitions: int,
              sse: np.ndarray | None) -> bool:  # fmt: skip
        """Whether compensation may run at step t, given the latch's delay ``latched``."""
        self.increment = None
        if latched <= 0:  # the latch itself is off: reset
            self.released, self.statistic, self.delay = False, 0.0, 0
            return False
        if self.kind == "yoked":
            assert self.mask is not None
            self.released = bool(self.mask[t]) if t < len(self.mask) else False
            self.delay = latched
            return not self.released
        fresh = bool(estimate.delay_reliable and estimate.delay is not None
                     and estimate.delay > 0)  # fmt: skip
        if self.delay == 0:  # the latch has just engaged: a mode change
            self.last_switch, self.last_confirm, self.statistic = t, t, 0.0
        elif latched != self.delay:  # new d from a fresh reliable estimate
            self.statistic, self.last_confirm = 0.0, t
        self.delay = latched
        if fresh and estimate.delay == latched:
            self.last_confirm = t
        dwell_over = t - self.last_switch >= self.cfg.min_dwell
        if self.released:
            if dwell_over and fresh:
                self.released, self.last_switch = False, t
                self.statistic, self.last_confirm = 0.0, t
            return not self.released
        if self.kind == "cusum":
            assert sse is not None
            self.increment = cusum_increment(transitions, sse, latched, self.cfg)
            if self.increment is not None:
                self.statistic = max(0.0, self.statistic + self.increment)
            h = self.cfg.cusum_threshold
            if h is not None and self.statistic > h and dwell_over:
                self.released, self.last_switch, self.statistic = True, t, 0.0
        elif self.kind == "timeout":
            if t - self.last_confirm >= self.cfg.timeout_steps and dwell_over:
                self.released, self.last_switch = True, t
        return not self.released


def position_error(forecast: tuple[np.ndarray, np.ndarray], observation: np.ndarray) -> float:
    gripper, cube = forecast
    return float(np.linalg.norm(gripper - observation[S["gripper_position"]])
                 + np.linalg.norm(cube - observation[S["object_position"]]))  # fmt: skip


@dataclass
class PredictorSelector:
    """Supervisor B: keep both predictors' forecasts of the observation at t + d, score them
    when step t + d is observed, and use the predictor with the lower recent error (EWMA),
    switching only when the other is better by the margin and the dwell has passed."""

    cfg: SupervisorConfig
    choice: str = "learned"
    last_switch: int = NEVER
    ewma: dict[str, float | None] = field(
        default_factory=lambda: {"learned": None, "phase_e": None}
    )
    forecasts: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = field(default_factory=dict)
    last_errors: dict[str, float] = field(default_factory=dict)

    @property
    def alpha(self) -> float:
        return 2.0 / (self.cfg.ewma_span + 1.0)

    def score(self, t: int, observation: np.ndarray) -> None:
        """Score every forecast that targeted step t against the observation at t."""
        self.last_errors = {}
        for target in [k for k in self.forecasts if k <= t]:
            made = self.forecasts.pop(target)
            if target != t:
                continue
            for name, forecast in made.items():
                e = position_error(forecast, observation)
                self.last_errors[name] = e
                old = self.ewma[name]
                self.ewma[name] = e if old is None else old + self.alpha * (e - old)

    def choose(self, t: int) -> str:
        learned, phase_e = self.ewma["learned"], self.ewma["phase_e"]
        if learned is None or phase_e is None or t - self.last_switch < self.cfg.min_dwell:
            return self.choice
        m = self.cfg.selection_margin
        if self.choice == "learned" and phase_e < m * learned:
            self.choice, self.last_switch = "phase_e", t
        elif self.choice == "phase_e" and learned < m * phase_e:
            self.choice, self.last_switch = "learned", t
        return self.choice

    def record(self, t: int, delay: int, predictions: dict[str, np.ndarray]) -> None:
        self.forecasts[t + delay] = {
            name: (p[S["gripper_position"]].copy(), p[S["object_position"]].copy())
            for name, p in predictions.items()
        }


@dataclass
class ContactGate:
    """Supervisor C: the learned rollout only while the observed horizontal gripper-cube
    distance says the cube is close (open below ``contact_open_m``, close above
    ``contact_close_m``), with the minimum dwell between changes."""

    cfg: SupervisorConfig
    open: bool | None = None
    last_switch: int = NEVER

    def choose(self, t: int, observation: np.ndarray) -> str:
        distance = float(np.linalg.norm(observation[S["object_relative_position"]][:2]))
        if self.open is None:
            self.open, self.last_switch = distance < self.cfg.contact_open_m, t
        elif t - self.last_switch >= self.cfg.min_dwell:
            if not self.open and distance < self.cfg.contact_open_m:
                self.open, self.last_switch = True, t
            elif self.open and distance > self.cfg.contact_close_m:
                self.open, self.last_switch = False, t
        return "learned" if self.open else "phase_e"
