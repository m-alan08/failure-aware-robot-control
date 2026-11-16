"""Causal online identification of actuator gain and action delay.

Model (horizontal plane, decision step tau):

    g[tau+1] - g[tau]  ~=  K0 * gain * STEP * c[tau - delay]

``g`` is the gripper position the controller observes (possibly noisy), ``c`` the command the
controller issued (zero before the episode starts, matching how ``action_delay`` replays
commands), ``STEP`` = 0.05 m the environment's per-step scaling, and ``K0`` the nominal plant
gain: the mocap weld does not move the gripper the full commanded distance within one control
period, so even undisturbed the observed displacement is only about 0.6 of the command. K0 is a
fixed calibration constant measured on debugging seeds, never on evaluation seeds.

At decision step t the estimator fits the last ``window`` transitions ending at t (all already
observed) by least squares through the origin, once for every candidate lag k = 0..max_lag,
and keeps the lag with the smallest residual. Only past commands and past and current
observations enter; there is no access to the true disturbance or to the future.

Reliability (abstention) checks, each recorded separately:

- enough transitions (``min_transitions``);
- excitation: the mean squared commanded displacement in the window must exceed
  ``min_excitation`` (a stationary or barely moving arm says nothing about gain or delay);
- fit quality: R^2 >= ``min_r2`` (observation noise and contact lower it);
- for the delay only: the second-best lag's residual must be at least ``min_margin`` times the
  best one (smooth, autocorrelated commands make neighbouring lags look alike).

Gain uses the first three checks; delay uses all four.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STEP = 0.05


@dataclass(frozen=True)
class EstimatorConfig:
    nominal_gain: float  # K0, calibrated on debugging seeds
    window: int = 10
    max_lag: int = 4
    min_transitions: int = 4
    min_excitation: float = (STEP * 0.3) ** 2  # per-transition mean of |0.05 c_xy|^2
    min_r2: float = 0.8
    min_margin: float = 2.0


@dataclass(frozen=True)
class Estimate:
    step: int
    transitions: int
    delay: int | None  # best lag (None if no fit was possible)
    gain: float  # relative to nominal; nan if no fit
    r2: float
    margin: float
    excitation: float
    gain_reliable: bool
    delay_reliable: bool


@dataclass
class ActuatorEstimator:
    """Streaming estimator: call ``observe(g)`` at every decision step before issuing, then
    ``issued(c)`` with the command actually sent. ``estimate()`` uses only data seen so far."""

    config: EstimatorConfig
    positions: list[np.ndarray] = field(default_factory=list)
    commands: list[np.ndarray] = field(default_factory=list)

    def observe(self, gripper: np.ndarray) -> None:
        if len(self.positions) != len(self.commands):
            raise RuntimeError("observe() twice without issued()")
        self.positions.append(np.asarray(gripper, float)[:2].copy())

    def issued(self, command: np.ndarray) -> None:
        if len(self.positions) != len(self.commands) + 1:
            raise RuntimeError("issued() without a preceding observe()")
        self.commands.append(np.asarray(command, float)[:2].copy())

    @property
    def step(self) -> int:
        return len(self.positions) - 1

    def _command(self, index: int) -> np.ndarray:
        return self.commands[index] if index >= 0 else np.zeros(2)

    def estimate(self) -> Estimate:
        cfg, t = self.config, self.step
        taus = list(range(max(0, t - cfg.window), t))  # transitions tau -> tau+1, tau+1 <= t
        nan = float("nan")
        if len(taus) < 1:
            return Estimate(t, 0, None, nan, nan, nan, 0.0, False, False)
        pos = self.positions
        y = np.concatenate([pos[tau + 1] - pos[tau] for tau in taus])
        fits = []
        for k in range(cfg.max_lag + 1):
            x = STEP * np.concatenate([self._command(tau - k) for tau in taus])
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
            return Estimate(t, len(taus), None, nan, nan, nan, 0.0, False, False)
        ordered = np.sort(sse)
        margin = float(ordered[1] / max(ordered[0], 1e-15)) if len(ordered) > 1 else np.inf
        sst = float(y @ y)
        r2 = 1.0 - sse[k] / sst if sst > 1e-15 else nan
        excitation = fits[k][2] / len(taus)
        base_ok = (len(taus) >= cfg.min_transitions and excitation >= cfg.min_excitation
                   and np.isfinite(r2) and r2 >= cfg.min_r2)  # fmt: skip
        gain = fits[k][1] / cfg.nominal_gain
        return Estimate(t, len(taus), k, float(gain), float(r2), margin, float(excitation),
                        bool(base_ok and gain > 0),
                        bool(base_ok and margin >= cfg.min_margin))  # fmt: skip


def lag_residuals(estimator: ActuatorEstimator) -> tuple[int, np.ndarray]:
    """READ-ONLY (Phase G1): the residual sum of squares of the window fit for every lag
    0..max_lag, exactly as ``estimate()`` computes them, with ``inf`` where a lag has no
    commanded displacement. Returns (transitions, sse). It reads the estimator's history and
    changes nothing; ``estimate()`` is untouched and stays the frozen Phase E estimator."""
    cfg, t = estimator.config, estimator.step
    taus = list(range(max(0, t - cfg.window), t))
    sse = np.full(cfg.max_lag + 1, np.inf)
    if not taus:
        return 0, sse
    pos = estimator.positions
    y = np.concatenate([pos[tau + 1] - pos[tau] for tau in taus])
    for k in range(cfg.max_lag + 1):
        x = STEP * np.concatenate([estimator._command(tau - k) for tau in taus])
        sxx = float(x @ x)
        if sxx <= 1e-12:
            continue
        residual = y - (float(x @ y) / sxx) * x
        sse[k] = float(residual @ residual)
    return len(taus), sse
