"""Candidate C split into ``propose`` and ``commit`` so a residual can be inserted between them.

``HardwareController.step`` (configuration "C") and ``mechanism.run_episode`` (G4 ``C``) run
one loop body:

    base = pi(obs)                       frozen SAC on the observation
    est.observe(gripper)                 Phase E estimator history
    dec = Sequential.step(est)           Candidate C identification + CUSUM supervision
    if dec.active: predicted = F1(obs, queued(d), gain); command = pi(predicted)
    else:          command = base
    command = clip(command, -1, 1)       <-- a_C
    est.issued(command)                  <-- the only write-back into C's state

The residual is inserted at the arrow between ``clip`` and ``est.issued``. ``propose`` runs
everything up to and including the clip and returns a_C; ``commit`` performs the write-back
with the command that is *actually sent to the actuator* (a_final). Nothing else of
Candidate C is touched: the policy, normaliser, estimator, ``Sequential`` and F1 predictor are
imported unchanged and called in the same order, and no constant is re-implemented.

Why the estimator must see a_final, not a_C: the estimator identifies delay from observed
positions against *issued* commands. If the plant executes a_final while the estimator records
a_C, every non-zero residual becomes unmodelled input and biases the delay fit and the CUSUM
residuals. Recording a_final keeps C's model of the plant physically correct; it is also what
the hardware FIFO and the simulator delay queue replay. With alpha = 0, a_final == a_C
bitwise, so C's state trajectory is unchanged (tests check this against both reference loops).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.contact.episodes import Predictor, queued_commands
from farc.delaychange.candidates import Decision, Sequential
from farc.delaychange.mechanism import sequential
from farc.envs.observation_layout import OBJECT
from farc.hardware.controller import C_CONSTANTS, ESTIMATOR

S = OBJECT.slices


@dataclass(frozen=True)
class Proposal:
    """Candidate C's output at one step (all arrays are private copies)."""

    t: int
    a_c: np.ndarray  # clipped Candidate C command
    base: np.ndarray  # frozen SAC on the raw observation
    decision: Decision
    delay: int
    active: bool
    gain: float
    predicted: np.ndarray | None  # F1 predicted observation (None when not compensating)


class CandidateCCore:
    """Frozen Candidate C with the commit step exposed. Not a new controller."""

    def __init__(self, policy, normalizer, predictor: Predictor,
                 estimator: dict[str, Any] | None = None,
                 constants: dict[str, Any] | None = None):  # fmt: skip
        self.policy, self.normalizer, self.predictor = policy, normalizer, predictor
        self.est_cfg = EstimatorConfig(**(estimator or ESTIMATOR))
        self.constants = dict(constants or C_CONSTANTS)
        self.reset()

    def reset(self) -> None:
        self.est = ActuatorEstimator(self.est_cfg)
        self.ctrl: Sequential = sequential(self.constants)
        self.issued: list[np.ndarray] = []
        self.t = 0
        self._open: Proposal | None = None

    def _act(self, obs: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.asarray(self.policy(self.normalizer(obs, goal)[None])[0], np.float64)

    def propose(self, obs: np.ndarray, goal: np.ndarray,
                base_action: np.ndarray | None = None) -> Proposal:  # fmt: skip
        """Steps 1-6 of the loop body. ``base_action`` lets a simulator session pass the base
        output it already computed on the same view (as ``mechanism.run_episode`` does)."""
        if self._open is not None:
            raise RuntimeError("propose() twice without commit()")
        obs = np.asarray(obs, np.float64)
        base = self._act(obs, goal) if base_action is None else np.asarray(base_action, np.float64)
        self.est.observe(obs[S["gripper_position"]].copy())
        dec = self.ctrl.step(self.est)
        d, active, gain = dec.delay, dec.active, self.ctrl.gain
        predicted = None
        if active:
            predicted = self.predictor(obs.copy(), queued_commands(self.issued, d), gain)
            command = self._act(predicted, goal)
        else:
            command = base
        a_c = np.clip(command, -1.0, 1.0)
        self._open = Proposal(self.t, a_c.copy(), base.copy(), dec, int(d), bool(active),
                              float(gain), None if predicted is None else
                              np.asarray(predicted, np.float64).copy())  # fmt: skip
        return self._open

    def commit(self, command: np.ndarray) -> None:
        """Step 7: record the command actually sent. The only write into C's state."""
        if self._open is None:
            raise RuntimeError("commit() without propose()")
        c = np.asarray(command, np.float64)
        if c.shape != (4,) or not np.all(np.isfinite(c)) or np.any(np.abs(c) > 1.0):
            raise ValueError("the committed command must be a finite 4-vector in [-1, 1]")
        self.est.issued(c)
        self.issued.append(c.copy())
        self.t += 1
        self._open = None
