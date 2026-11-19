"""Candidate C in logging-only shadow mode (docs/HARDWARE_H1_PREREGISTRATION.md, section 3.4).

The shadow owns an independent ``ActuatorEstimator`` and a ``Sequential`` supervisor built by
``mechanism.sequential`` with the frozen constants. After the active controller has decided,
the shadow is fed *copies* of the observed gripper position and of the command the active
controller actually issued, in the estimator's required order (observe, step, issued). It
returns only a log dictionary. It holds no reference to the active controller, its estimator,
the FIFO or any command path, and nothing it returns is used to form a command.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.delaychange.mechanism import sequential
from farc.hardware.controller import C_CONSTANTS, ESTIMATOR


class ShadowSupervisor:
    def __init__(self, estimator: dict[str, Any] | None = None,
                 constants: dict[str, Any] | None = None):  # fmt: skip
        self._est_cfg = EstimatorConfig(**(estimator or ESTIMATOR))
        self._constants = dict(constants or C_CONSTANTS)
        self.reset()

    def reset(self) -> None:
        self._est = ActuatorEstimator(self._est_cfg)
        self._sup = sequential(self._constants)

    def step(self, gripper: np.ndarray, issued_command: np.ndarray) -> dict[str, Any]:
        """Observe the gripper, update the supervisor, then record the issued command."""
        self._est.observe(np.array(gripper, np.float64, copy=True))
        dec = self._sup.step(self._est)
        self._est.issued(np.array(issued_command, np.float64, copy=True))
        n = self._est_cfg.max_lag + 1
        return {
            "shadow_alarm": bool(dec.alarm),
            "shadow_suspended": bool(dec.suspended),
            "shadow_active": bool(dec.active),
            "shadow_trusted": int(self._sup.trusted),
            "shadow_cusum_stats": [float(self._sup.stats.get(k, 0.0)) for k in range(n)],
        }
