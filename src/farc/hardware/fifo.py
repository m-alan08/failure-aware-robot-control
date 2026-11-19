"""Deterministic command FIFO for injected delay (proposal section 4.3).

Semantics are exactly ``farc.retrain.phase_h.DelayHistoryWrapper.step`` (itself verified
bitwise against ``ScheduledDelaySession``'s ``replay_drop`` mode): at control step t the
executed command is the command issued at step ``t - d(t)``, or zeros if that step precedes
the episode, clipped to the action bounds. A delay change therefore *replays* a command (on an
increase) or *drops* commands (on a decrease) exactly as in simulation. ``d(t)`` comes from a
``DelaySchedule`` or from ``set_delay`` (effective from the next ``push``).

Not connected to any robot: ``push`` only returns what would be executed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from farc.delaychange.schedule import DelaySchedule

MAX_DELAY = 4


@dataclass(frozen=True)
class FifoOutput:
    step: int
    delay: int
    source_step: int  # step whose command executes now (negative: zero command)
    executed: np.ndarray
    issued_ns: int  # monotonic time the command was issued at this step
    source_issued_ns: int | None  # when the executed command was originally issued


class CommandFIFO:
    def __init__(self, action_dim: int = 4, low: float = -1.0, high: float = 1.0,
                 schedule: DelaySchedule | None = None):  # fmt: skip
        self.action_dim, self.low, self.high = action_dim, low, high
        self.schedule = schedule
        self.reset()

    def reset(self, schedule: DelaySchedule | None = None) -> None:
        if schedule is not None:
            self.schedule = schedule
        self.t = 0
        self._manual: int | None = None
        self._buf: deque[tuple[int, np.ndarray, int]] = deque(maxlen=MAX_DELAY + 1)
        self._last_ns: int | None = None

    def set_delay(self, delay: int) -> None:
        if self.schedule is not None:
            raise RuntimeError("delay comes from the schedule; set_delay is for manual mode")
        self._manual = _check(delay)

    def delay_at(self, t: int) -> int:
        if self.schedule is not None:
            return _check(self.schedule.delay_at(t))
        return self._manual or 0

    def push(self, command: np.ndarray, issued_ns: int) -> FifoOutput:
        c = np.asarray(command, np.float64)
        if c.shape != (self.action_dim,) or not np.all(np.isfinite(c)):
            raise ValueError("command must be a finite vector of the action dimension")
        if self._last_ns is not None and issued_ns <= self._last_ns:
            raise ValueError("issue timestamps must increase strictly")
        self._last_ns = issued_ns
        t = self.t
        self._buf.append((t, c.copy(), issued_ns))
        d = self.delay_at(t)
        source = t - d
        if source >= 0:
            match = [(cmd, ns) for (s, cmd, ns) in self._buf if s == source]
            executed, source_ns = match[0][0].copy(), match[0][1]
        else:
            executed, source_ns = np.zeros(self.action_dim), None
        executed = np.clip(executed, self.low, self.high)
        self.t += 1
        return FifoOutput(t, d, source, executed, issued_ns, source_ns)


def _check(delay: int) -> int:
    if int(delay) != delay or not 0 <= delay <= MAX_DELAY:
        raise ValueError(f"injected delay must be an integer in 0..{MAX_DELAY}")
    return int(delay)
