"""G3 randomised delay schedules (docs/PHASE_G3_PROPOSAL.md, section 3).

A schedule depends only on (generator seed, cell index, episode seed): every controller and
every frozen policy of a pair sees the identical schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from farc.delaychange.schedule import DelaySchedule
from farc.disturbances.config import DisturbanceSpec

GENERATOR_SEED = 20261700
HORIZON = 50
NOISE_H = DisturbanceSpec(type="observation_noise", severity=0.01, mask="positions")
S_PAIRS = ((1, 4), (4, 1), (2, 4), (4, 2))


@dataclass(frozen=True)
class RandomCell:
    name: str
    family: str  # RT | RI | RS | RP | H
    delay: int = 0
    extra: tuple[DisturbanceSpec, ...] = ()


def g3_cells() -> list[RandomCell]:
    return [
        RandomCell("RT_d2", "RT", 2),
        RandomCell("RT_d4", "RT", 4),
        RandomCell("RI_d2", "RI", 2),
        RandomCell("RI_d4", "RI", 4),
        RandomCell("RS", "RS"),
        RandomCell("RP_d2", "RP", 2),
        RandomCell("RP_d4", "RP", 4),
        RandomCell("H_undisturbed", "H"),
        RandomCell("H_noise0.01", "H", 0, (NOISE_H,)),
    ]


def _u(rng: np.random.Generator, low: int, high: int) -> int:
    """Uniform on the integers low .. high inclusive."""
    return int(rng.integers(low, high + 1))


def draw_pulses(
    cell: RandomCell, cell_index: int, episode_seed: int, generator_seed: int = GENERATOR_SEED
) -> tuple[tuple[int, int | None, int], ...]:
    """``generator_seed`` defaults to the G3/G4 value; Phase H training passes its own."""
    rng = np.random.default_rng([generator_seed, cell_index, episode_seed])
    if cell.family == "RT":
        onset = _u(rng, 4, 34)
        return ((onset, _u(rng, 3, min(16, 46 - onset)), cell.delay),)
    if cell.family == "RI":
        pulses = []
        onset = _u(rng, 4, 12)
        while onset <= 44:
            on = _u(rng, 3, 10)
            duration = min(on, HORIZON - onset)
            pulses.append((onset, duration, cell.delay))
            onset += on + _u(rng, 3, 10)
        return tuple(pulses)
    if cell.family == "RS":
        a, b = S_PAIRS[_u(rng, 0, 3)]
        change = _u(rng, 12, 32)
        return ((0, change, a), (change, None, b))
    if cell.family == "RP":
        return ((_u(rng, 0, 20), None, cell.delay),)
    return ()


def schedule_for(cell: RandomCell, cell_index: int, episode_seed: int) -> DelaySchedule:
    return DelaySchedule(draw_pulses(cell, cell_index, episode_seed))


def transitions(schedule: DelaySchedule, horizon: int = HORIZON) -> list[tuple[int, str]]:
    """Steps t >= 1 at which the true delay changes, with the kind of change."""
    out = []
    for t in range(1, horizon):
        a, b = schedule.delay_at(t - 1), schedule.delay_at(t)
        if a != b:
            out.append((t, "end" if b == 0 else "onset" if a == 0 else "change"))
    return out
