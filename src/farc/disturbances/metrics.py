# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/metrics.py
# (introduced in 7e7f67a, last changed in 7e7f67a). See PROVENANCE.md.
"""Episode metrics, defined on per-step success against the goal active at that step.

Indexing: state 0 is the observation after reset, state k (1..T) the observation after the
k-th ``env.step``. ``successes[k - 1]`` is ``info["is_success"]`` after step k. A disturbance
with onset ``o`` first affects the action chosen at state o, i.e. env step o + 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def first_success_step(successes: np.ndarray) -> int | None:
    hits = np.flatnonzero(np.asarray(successes, dtype=bool))
    return int(hits[0]) + 1 if hits.size else None


@dataclass(frozen=True)
class RecoveryOutcome:
    recovered: bool
    recovery_time: int | None  # env steps after onset until the first success; None if censored
    censored: bool
    censor_time: int  # steps observed after onset (T - onset)


def recovery_outcome(successes: np.ndarray, onset: int, horizon: int) -> RecoveryOutcome:
    """Recovery = first success at a state strictly after the onset state, judged against the
    goal active at that state (for a goal shift, the new goal). Episodes with no such success
    are right-censored at T - onset steps; they get no recovery time."""
    successes = np.asarray(successes, dtype=bool)
    if successes.shape != (horizon,):
        raise ValueError(f"successes has shape {successes.shape}, expected ({horizon},)")
    if not 0 <= onset < horizon:
        raise ValueError(f"onset {onset} is outside [0, {horizon})")
    window = successes[onset:]  # states onset+1 .. T
    hits = np.flatnonzero(window)
    if hits.size:
        return RecoveryOutcome(True, int(hits[0]) + 1, False, horizon - onset)
    return RecoveryOutcome(False, None, True, horizon - onset)


def kaplan_meier_median(times: np.ndarray, observed: np.ndarray) -> float | None:
    """Median of a right-censored time-to-event sample (Kaplan-Meier); None if the survival
    curve never drops to 0.5 (more than half the episodes censored before the median)."""
    times = np.asarray(times, dtype=float)
    observed = np.asarray(observed, dtype=bool)
    if times.size == 0:
        return None
    survival = 1.0
    for time in np.unique(times[observed]):
        at_risk = np.sum(times >= time)
        events = np.sum((times == time) & observed)
        survival *= 1.0 - events / at_risk
        if survival <= 0.5:
            return float(time)
    return None
