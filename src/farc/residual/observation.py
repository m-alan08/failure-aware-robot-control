"""Residual observation space.

Every component is available to the controller at inference time on the robot: it is either
the controller's own (possibly noisy) observation and goal, a command the controller computed
or issued, or a read-only summary of Candidate C's internal state. Explicitly EXCLUDED (and
never passed to the builder): true delay or schedule, true gain, simulator state, success or
distance-to-goal, oracle predictions, the episode seed and the condition label.

Layout (in order; optional groups are configurable and all scaled to roughly [-1, 1]):

| group | dims | content |
|---|---|---|
| task (always) | task_dim (28) | frozen normaliser(obs, goal), i.e. exactly the SAC input |
| actions (always) | 12 | a_C, base SAC action, previous residual delta |
| delay (opt.) | max_lag + 5 | one-hot delay used by C, active, delay/gain reliable, gain - 1 |
| supervisor (opt.) | max_lag + 4 | suspended, pending, alarm, CUSUM statistics / threshold |
| prediction (opt.) | 4 | newest-transition residual at trusted lag, F1 gripper and object
                          prediction errors for this step, error-valid flag |
| time (opt.) | 1 | t / horizon |
| validity | 1 + #optional | 1 if the group's raw values were finite, else 0 (and zeroed) |

The prediction error compares the observation now with the F1 prediction C made ``d`` steps
ago for this step (prediction made at s with delay d targets s + d). It is causal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from farc.adaptation.estimator import STEP
from farc.delaychange.candidates import newest_residuals
from farc.envs.observation_layout import OBJECT
from farc.residual.core import CandidateCCore, Proposal

S = OBJECT.slices
OPTIONAL = ("delay", "supervisor", "prediction", "time")


class ResidualObservationError(ValueError):
    """The controller's own input (observation or goal) is missing, mis-shaped or non-finite."""


@dataclass(frozen=True)
class ObservationSpec:
    task_dim: int = 28
    max_lag: int = 4
    horizon: int = 50
    delay: bool = True
    supervisor: bool = True
    prediction: bool = True
    time: bool = False

    def groups(self) -> list[tuple[str, int]]:
        n = self.max_lag + 1
        sizes = {"delay": n + 4, "supervisor": n + 3, "prediction": 4, "time": 1}
        out = [("task", self.task_dim), ("actions", 12)]
        out += [(g, sizes[g]) for g in OPTIONAL if getattr(self, g)]
        out.append(("validity", 1 + sum(getattr(self, g) for g in OPTIONAL)))
        return out

    @property
    def task_validity_index(self) -> int:
        return self.dim - self.groups()[-1][1]

    @property
    def dim(self) -> int:
        return sum(n for _, n in self.groups())


def _clip(x, lo=-1.0, hi=1.0) -> np.ndarray:
    return np.clip(np.asarray(x, np.float64), lo, hi)


class ResidualObservationBuilder:
    """Stateful (previous residual, pending F1 predictions); reset with the episode."""

    def __init__(self, spec: ObservationSpec | None = None):
        self.spec = spec or ObservationSpec()
        self.reset()

    def reset(self) -> None:
        self.prev_delta = np.zeros(4)
        self._targets: dict[int, np.ndarray] = {}

    def build(self, core: CandidateCCore, p: Proposal, obs: np.ndarray,
              goal: np.ndarray) -> np.ndarray:  # fmt: skip
        """Read-only with respect to ``core``: only attribute reads and the pure
        ``newest_residuals``; a test checks C's state is unchanged by this call."""
        obs, goal = np.asarray(obs, np.float64), np.asarray(goal, np.float64)
        if obs.shape != (OBJECT.size,) or goal.shape != (3,):
            raise ResidualObservationError("observation must be (25,) and goal (3,)")
        if not (np.all(np.isfinite(obs)) and np.all(np.isfinite(goal))):
            raise ResidualObservationError("non-finite controller observation or goal")
        spec, ctrl = self.spec, core.ctrl
        task = np.asarray(core.normalizer(obs, goal), np.float64).reshape(-1)
        if task.shape != (spec.task_dim,):
            raise ResidualObservationError(f"normaliser output {task.shape} != {spec.task_dim}")
        parts: dict[str, np.ndarray] = {
            "task": task,
            "actions": np.concatenate([p.a_c, _clip(p.base), self.prev_delta]),
        }
        n = spec.max_lag + 1
        if spec.delay:
            e = p.decision.estimate
            onehot = np.zeros(n)
            onehot[min(max(p.delay, 0), n - 1)] = 1.0
            parts["delay"] = np.concatenate(
                [
                    onehot,
                    [float(p.active), float(e.delay_reliable), float(e.gain_reliable)],
                    _clip([p.gain - 1.0]),
                ]
            )
        if spec.supervisor:
            thr = float(ctrl.threshold)
            stats = [ctrl.stats.get(k, 0.0) / thr for k in range(n)]
            parts["supervisor"] = np.concatenate(
                [
                    [float(ctrl.suspended), float(ctrl.pending), float(p.decision.alarm)],
                    _clip(stats, 0, 2),
                ]
            )
        if spec.prediction:
            r2 = newest_residuals(core.est, ctrl.gain_estimate)
            resid = 0.0 if r2 is None else math.log1p(float(r2[ctrl.trusted]) / ctrl.sigma0**2)
            target = self._targets.pop(p.t, None)
            g_err = o_err = valid = 0.0
            if target is not None:
                g_err = (
                    float(
                        np.linalg.norm(obs[S["gripper_position"]] - target[S["gripper_position"]])
                    )
                    / STEP
                )
                o_err = (
                    float(np.linalg.norm(obs[S["object_position"]] - target[S["object_position"]]))
                    / STEP
                )
                valid = 1.0
            parts["prediction"] = np.array(
                [min(resid / 10.0, 1.0), min(g_err, 1.0), min(o_err, 1.0), valid]
            )
            if p.predicted is not None and p.delay > 0:
                self._targets[p.t + p.delay] = p.predicted.copy()
        if spec.time:
            parts["time"] = np.array([p.t / spec.horizon])
        validity = []
        for name, size in spec.groups():
            if name == "validity":
                continue
            v = parts[name]
            if v.shape != (size,):
                raise AssertionError(f"group {name} has {v.shape}, expected {size}")
            ok = bool(np.all(np.isfinite(v)))
            if not ok:
                parts[name] = np.where(np.isfinite(v), v, 0.0)
            if name != "actions":
                validity.append(float(ok))
        parts["validity"] = np.array(validity)
        out = np.concatenate([parts[name] for name, _ in spec.groups()])
        assert out.shape == (spec.dim,)
        return out

    def record(self, delta: np.ndarray) -> None:
        self.prev_delta = np.asarray(delta, np.float64).copy()
