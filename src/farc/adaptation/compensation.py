"""Conservative compensation on top of the frozen base policy.

Every controller issues the base policy's own command unless its parameters say otherwise:

- gain compensation: ``c = clip(a / max(gain, gain_floor), -1, 1)`` on the three position
  components (the finger component is blocked in Push and left alone). It is applied only while
  the gain is below ``activate_below``; a gain near nominal means nothing to compensate.
- delay compensation (predictor): with delay d, the commands executed at steps t..t+d-1 are
  already in the queue (c[t-d] .. c[t-1]). The gripper position at t+d is predicted from them
  with the nominal plant gain (and the gain, if known), the observation is shifted to that
  predicted gripper position (gripper position +delta, object-relative position -delta; the
  object is assumed not to move and velocities are left unchanged), and the base policy is
  evaluated there. This is a Smith-predictor-style guess, not an exact model: under contact the
  object moves and the prediction is wrong.

Parameters come from a ``ParameterSource``: the online estimator (latched to its last reliable
estimate), the true injected values (ORACLE, privileged), or fixed values (SHAM controls).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from farc.adaptation.estimator import STEP, ActuatorEstimator, Estimate
from farc.envs.observation_layout import FetchObservationLayout


@dataclass(frozen=True)
class Parameters:
    gain: float | None  # relative actuator gain to compensate; None = do not compensate gain
    delay: int | None  # delay to predict over; None or 0 = do not compensate delay


class ParameterSource(Protocol):
    def parameters(self, estimate: Estimate) -> Parameters: ...


class LatchedEstimate:
    """Deployable: the most recent reliable estimate of each parameter (abstains until one)."""

    def __init__(self) -> None:
        self.gain: float | None = None
        self.delay: int | None = None

    def parameters(self, estimate: Estimate) -> Parameters:
        if estimate.gain_reliable:
            self.gain = estimate.gain
        if estimate.delay_reliable:
            self.delay = estimate.delay
        return Parameters(self.gain, self.delay)


@dataclass
class Fixed:
    """ORACLE (true values) or SHAM (arbitrary values), optionally gated on the estimator's
    reliability so a sham activates exactly when the deployable controller could."""

    gain: float | None
    delay: int | None
    gate_on_reliability: bool = False
    _gain_open: bool = False
    _delay_open: bool = False

    def parameters(self, estimate: Estimate) -> Parameters:
        if not self.gate_on_reliability:
            return Parameters(self.gain, self.delay)
        self._gain_open |= estimate.gain_reliable
        self._delay_open |= estimate.delay_reliable
        return Parameters(self.gain if self._gain_open else None,
                          self.delay if self._delay_open else None)  # fmt: skip


@dataclass(frozen=True)
class CompensationConfig:
    use_gain: bool
    use_delay: bool
    gain_floor: float = 0.25
    activate_below: float = 0.85


def predicted_observation(
    observation: np.ndarray,
    layout: FetchObservationLayout,
    queued: list[np.ndarray],
    nominal_gain: float,
    gain: float,
) -> np.ndarray:
    """Shift the observation by the gripper displacement the queued commands should produce."""
    delta = np.zeros(3)
    for command in queued:
        delta += nominal_gain * gain * STEP * np.clip(np.asarray(command, float)[:3], -1, 1)
    out = np.asarray(observation, float).copy()
    out[layout.slices["gripper_position"]] += delta
    out[layout.slices["object_relative_position"]] -= delta
    return out


@dataclass
class Decision:
    command: np.ndarray
    gain_active: bool
    delay_active: bool
    saturated: bool  # the compensated command hit the action bound where the base one did not


def compensate(
    config: CompensationConfig,
    params: Parameters,
    base_action: np.ndarray,
    policy_at,  # callable(observation) -> action, the frozen base policy on a given view
    observation: np.ndarray,
    layout: FetchObservationLayout,
    history: list[np.ndarray],  # commands issued so far (c[0] .. c[t-1])
    nominal_gain: float,
) -> Decision:
    action = np.asarray(base_action, float).copy()
    gain = params.gain if params.gain is not None else 1.0
    delay_active = bool(config.use_delay and params.delay)
    if delay_active:
        d = int(params.delay or 0)
        queued = [history[j] if j >= 0 else np.zeros(4) for j in range(len(history) - d,
                                                                         len(history))]  # fmt: skip
        action = np.asarray(policy_at(predicted_observation(
            observation, layout, queued, nominal_gain, min(gain, 1.0))), float).copy()  # fmt: skip
    gain_active = bool(config.use_gain and params.gain is not None
                       and params.gain < config.activate_below)  # fmt: skip
    command = action.copy()
    if gain_active:
        command[:3] = action[:3] / max(float(params.gain), config.gain_floor)  # type: ignore[arg-type]
    clipped = np.clip(command, -1.0, 1.0)
    saturated = bool(np.any(np.abs(command[:3]) > 1.0) and not np.any(np.abs(action[:3]) > 1.0))
    return Decision(clipped, gain_active, delay_active, saturated)


__all__ = [
    "ActuatorEstimator",
    "CompensationConfig",
    "Decision",
    "Fixed",
    "LatchedEstimate",
    "Parameters",
    "compensate",
    "predicted_observation",
]
