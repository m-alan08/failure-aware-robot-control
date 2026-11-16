"""G2 candidates (docs/PHASE_G2_PROPOSAL.md, section 2).

All three sit on the *unchanged* Phase E estimator history (positions and issued commands) and
its gates. ``window_fit`` reproduces ``ActuatorEstimator.estimate()`` exactly for the default
window and start (a test checks this). It only adds a window length and a window start, so
the frozen ``estimate()`` itself is never modified.

- A (short window): the latch on ``window_fit`` with window w_s.
- B (two-timescale): the slow latch (window 10) plus a fast ``window_fit`` (window w_f).
  When the fast fit reliably disagrees with the latched delay on m consecutive steps, the
  controller suspends compensation (frozen SAC). It re-engages only on a reliable slow
  estimate d' > 0 at least ``dwell`` steps after the alarm. The fast lag is never applied.
- C (sequential): a multi-alternative CUSUM on the *newest* transition's residuals. On an
  alarm it suspends compensation and restarts the identification window at the estimated
  change point. It re-engages only when the reset window gives a reliable d' > 0 through
  every existing gate. It also detects onsets: with no delay trusted, an alarm resets the
  window so identification uses post-onset data only. After any alarm, detection pauses
  until the reset window gives a reliable estimate (otherwise repeated alarms would keep
  resetting the window to a single transition).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from farc.adaptation.compensation import LatchedEstimate
from farc.adaptation.estimator import STEP, ActuatorEstimator, Estimate


def window_fit(est: ActuatorEstimator, window: int, start: int = 0) -> tuple[Estimate, np.ndarray]:
    """``estimate()`` over transitions max(start, t - window) .. t-1, plus the per-lag SSE."""
    cfg, t = est.config, est.step
    taus = list(range(max(0, t - window, start), t))
    nan = float("nan")
    sse_all = np.full(cfg.max_lag + 1, np.inf)
    if len(taus) < 1:
        return Estimate(t, 0, None, nan, nan, nan, 0.0, False, False), sse_all
    pos = est.positions
    y = np.concatenate([pos[tau + 1] - pos[tau] for tau in taus])
    fits = []
    for k in range(cfg.max_lag + 1):
        x = STEP * np.concatenate([est._command(tau - k) for tau in taus])
        sxx = float(x @ x)
        if sxx <= 1e-12:
            fits.append((np.inf, nan, 0.0))
            continue
        slope = float(x @ y) / sxx
        residual = y - slope * x
        fits.append((float(residual @ residual), slope, sxx))
    sse = np.array([f[0] for f in fits])
    k = int(np.argmin(sse))
    if not np.isfinite(sse[k]):
        return Estimate(t, len(taus), None, nan, nan, nan, 0.0, False, False), sse
    ordered = np.sort(sse)
    margin = float(ordered[1] / max(ordered[0], 1e-15)) if len(ordered) > 1 else np.inf
    sst = float(y @ y)
    r2 = 1.0 - sse[k] / sst if sst > 1e-15 else nan
    excitation = fits[k][2] / len(taus)
    base_ok = (len(taus) >= cfg.min_transitions and excitation >= cfg.min_excitation
               and np.isfinite(r2) and r2 >= cfg.min_r2)  # fmt: skip
    gain = fits[k][1] / cfg.nominal_gain
    return (Estimate(t, len(taus), k, float(gain), float(r2), margin, float(excitation),
                     bool(base_ok and gain > 0), bool(base_ok and margin >= cfg.min_margin)),
            sse)  # fmt: skip


def newest_residuals(est: ActuatorEstimator, gain: float) -> np.ndarray | None:
    """||r(k)||^2 of the newest transition (t-1 -> t) under each lag k, with the nominal
    plant gain K0 times the trusted gain estimate. None before the first transition."""
    t = est.step
    if t < 1:
        return None
    cfg = est.config
    y = est.positions[t] - est.positions[t - 1]
    scale = cfg.nominal_gain * gain * STEP
    return np.array([float(np.sum((y - scale * est._command(t - 1 - k)) ** 2))
                     for k in range(cfg.max_lag + 1)])  # fmt: skip


@dataclass
class Decision:
    active: bool
    delay: int
    estimate: Estimate
    alarm: bool = False
    suspended: bool = False


@dataclass
class ShortWindow:
    """Candidate A."""

    window: int
    latch: LatchedEstimate = field(default_factory=LatchedEstimate)

    def step(self, est: ActuatorEstimator) -> Decision:
        e, _ = window_fit(est, self.window)
        d = int(self.latch.parameters(e).delay or 0)
        return Decision(d > 0, d, e)

    @property
    def gain(self) -> float:
        return float(self.latch.gain) if self.latch.gain is not None else 1.0


@dataclass
class TwoTimescale:
    """Candidate B."""

    fast_window: int
    consecutive: int = 2
    dwell: int = 5
    latch: LatchedEstimate = field(default_factory=LatchedEstimate)
    suspended: bool = False
    alarm_at: int = -(10**9)
    count: int = 0

    def step(self, est: ActuatorEstimator) -> Decision:
        t = est.step
        slow = est.estimate()  # the frozen slow estimator, unchanged
        d = int(self.latch.parameters(slow).delay or 0)
        fast, _ = window_fit(est, self.fast_window)
        alarm = False
        if d == 0:
            self.suspended, self.count = False, 0
        elif self.suspended:
            if (t - self.alarm_at >= self.dwell and slow.delay_reliable and slow.delay
                    and slow.delay > 0):  # fmt: skip
                self.suspended = False
        else:
            disagrees = bool(fast.delay_reliable and fast.delay is not None and fast.delay != d)
            self.count = self.count + 1 if disagrees else 0
            if self.count >= self.consecutive:
                alarm, self.suspended, self.alarm_at, self.count = True, True, t, 0
        return Decision(d > 0 and not self.suspended, d, slow, alarm, self.suspended)

    @property
    def gain(self) -> float:
        return float(self.latch.gain) if self.latch.gain is not None else 1.0


@dataclass
class Sequential:
    """Candidate C."""

    threshold: float
    sigma0: float  # metres
    variant: str = "gauss"  # gauss | scalefree
    kappa: float = 0.5
    window: int = 10
    trusted: int = 0
    suspended: bool = False
    reset_start: int = 0
    sigma2: float | None = None
    gain_estimate: float = 1.0
    stats: dict[int, float] = field(default_factory=dict)
    starts: dict[int, int] = field(default_factory=dict)
    origin: int = 0  # first transition the current statistics may use
    pending: bool = False  # after an alarm, detection pauses until the reset window is reliable

    def _clear(self, next_transition: int) -> None:
        self.stats, self.starts, self.origin = {}, {}, next_transition

    def _llr(self, r2: np.ndarray, d: int, k: int) -> float:
        s0 = self.sigma0**2
        if self.variant == "gauss":
            sigma2 = max(self.sigma2 if self.sigma2 is not None else s0, s0)
            return float((r2[d] - r2[k]) / (2.0 * sigma2))
        return 0.5 * math.log((r2[d] + s0) / (r2[k] + s0))

    def step(self, est: ActuatorEstimator) -> Decision:
        t = est.step
        alarm = False
        r2 = newest_residuals(est, self.gain_estimate)
        tau = t - 1
        if (r2 is not None and not self.suspended and not self.pending
                and tau >= max(self.reset_start, self.origin)):  # fmt: skip
            d = self.trusted
            for k in range(len(r2)):
                if k == d:
                    continue
                s = max(0.0, self.stats.get(k, 0.0) + self._llr(r2, d, k) - self.kappa)
                if s == 0.0 or k not in self.starts:
                    self.starts[k] = tau + 1 if s == 0.0 else tau
                self.stats[k] = s
            if self.stats and max(self.stats.values()) > self.threshold:
                winner = max(self.stats, key=lambda k: self.stats[k])
                alarm = True
                self.reset_start = min(self.starts[winner], tau)
                if self.trusted > 0:
                    self.suspended = True
                self.pending = True
                self._clear(t)
        e, sse = window_fit(est, self.window, self.reset_start)
        if e.gain_reliable:
            self.gain_estimate = e.gain
        if e.delay_reliable and e.delay is not None:
            self.sigma2 = max(self.sigma0**2, float(sse[e.delay]) / (2.0 * max(e.transitions, 1)))
            if e.delay != self.trusted or self.suspended:
                self._clear(t)
            self.trusted, self.suspended, self.pending = e.delay, False, False
        return Decision(self.trusted > 0 and not self.suspended, self.trusted, e, alarm,
                        self.suspended)  # fmt: skip

    @property
    def gain(self) -> float:
        return self.gain_estimate
