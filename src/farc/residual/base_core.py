"""Frozen-SAC base for arm B2 (residual on frozen SAC, no Candidate C) and for B0.

Same ``propose``/``commit`` interface as ``CandidateCCore``, so ``ResidualController`` and
``ResidualEnv`` work unchanged. ``a_c`` here is the clipped frozen-SAC command (what
``mechanism.run_episode(mode="base")`` issues), so B2 with a zero residual is frozen SAC
exactly, and B0 is B2 with ``ZeroResidual``. There is no estimator, supervisor or predictor,
so only the 41-dim observation (task + actions + validity) is allowed.
"""

from __future__ import annotations

import math

import numpy as np

from farc.adaptation.estimator import Estimate
from farc.delaychange.candidates import Decision
from farc.residual.core import Proposal
from farc.residual.observation import ObservationSpec

_NO_ESTIMATE = Estimate(0, 0, None, math.nan, math.nan, math.nan, 0.0, False, False)
B2_SPEC = ObservationSpec(delay=False, supervisor=False, prediction=False)


def check_b2_spec(spec: ObservationSpec) -> ObservationSpec:
    if spec.delay or spec.supervisor or spec.prediction:
        raise ValueError("B2 has no Candidate C signals: use task + actions + validity only")
    return spec


class SACBaseCore:
    """Frozen SAC as the base controller. Never compensates, never alarms."""

    ctrl = None  # no supervisor (the 41-dim builder does not read it)

    def __init__(self, policy, normalizer):
        self.policy, self.normalizer = policy, normalizer
        self.reset()

    def reset(self) -> None:
        self.issued: list[np.ndarray] = []
        self.t = 0
        self._open: Proposal | None = None

    def propose(self, obs, goal, base_action=None) -> Proposal:
        if self._open is not None:
            raise RuntimeError("propose() twice without commit()")
        obs = np.asarray(obs, np.float64)
        if base_action is None:
            base = np.asarray(self.policy(self.normalizer(obs, goal)[None])[0], np.float64)
        else:
            base = np.asarray(base_action, np.float64)
        a = np.clip(base, -1.0, 1.0)
        dec = Decision(False, 0, _NO_ESTIMATE)
        self._open = Proposal(self.t, a.copy(), base.copy(), dec, 0, False, 1.0, None)
        return self._open

    def commit(self, command) -> None:
        if self._open is None:
            raise RuntimeError("commit() without propose()")
        c = np.asarray(command, np.float64)
        if c.shape != (4,) or not np.all(np.isfinite(c)) or np.any(np.abs(c) > 1.0):
            raise ValueError("the committed command must be a finite 4-vector in [-1, 1]")
        self.issued.append(c.copy())
        self.t += 1
        self._open = None
