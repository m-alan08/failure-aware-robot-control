"""Delay schedules and transition semantics for Phase G2 (docs/PHASE_G2_PROPOSAL.md, section 3).

The existing pipeline (``farc.disturbances.perturbations.ActionDelay``) is left untouched.
``ScheduledDelaySession`` reproduces it exactly under ``replay_drop`` (a test checks the
executed actions are identical) and adds the M5 diagnostic variant ``drain_hold``:

- ``replay_drop`` (existing): the executed command is the issued command from d(t) steps
  earlier. When a delay starts or grows, already-executed commands are replayed; when it ends
  or shrinks, queued commands are dropped.
- ``drain_hold`` (M5): a FIFO of issued commands. When the delay grows, the actuator *holds*
  the last executed command while the queue fills (nothing is replayed). When it shrinks, the
  queued commands that would otherwise be dropped are *merged* into the step's command (their
  sum, clipped later by the action bounds), so no commanded displacement is discarded.
  This "merge" is how a one-action-per-step actuator can shorten its queue without dropping
  commands (see the Stage-1 amendment in the results document).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from farc.counterfactual.session import EpisodeSession
from farc.disturbances.perturbations import ActionPerturbation
from farc.disturbances.rollout import RolloutContractError

Semantics = Literal["replay_drop", "drain_hold"]


@dataclass(frozen=True)
class DelaySchedule:
    """Non-overlapping pulses (onset, duration or None for the rest of the episode, delay)."""

    pulses: tuple[tuple[int, int | None, int], ...]

    def __post_init__(self) -> None:
        spans = sorted((o, o + (L if L is not None else 10**9)) for o, L, _ in self.pulses)
        for (_a0, a1), (b0, _) in zip(spans[:-1], spans[1:], strict=True):
            if b0 < a1:
                raise ValueError("delay pulses overlap; overlapping delays would compose")
        if any(d < 1 or o < 0 or (L is not None and L < 1) for o, L, d in self.pulses):
            raise ValueError("pulses need onset >= 0, duration >= 1 and delay >= 1")

    def delay_at(self, step: int) -> int:
        for onset, duration, delay in self.pulses:
            if onset <= step and (duration is None or step < onset + duration):
                return delay
        return 0

    def ends(self, horizon: int) -> list[int]:
        """Steps at which a delay drops to zero."""
        return [t for t in range(1, horizon) if self.delay_at(t - 1) > 0 and self.delay_at(t) == 0]

    def command_horizon(self, step: int, horizon: int) -> int | None:
        """Under replay_drop: steps until the command issued at ``step`` executes (None if it
        is dropped or the episode ends first). PRIVILEGED: it reads the future schedule."""
        for later in range(step, horizon):
            if later - self.delay_at(later) == step:
                return later - step
        return None

    def placeholders(self, step: int, lookahead: int) -> int:
        """Under replay_drop: how many of the look-ahead's placeholder commands (issued at
        step .. step+lookahead-1) actually execute inside the look-ahead."""
        return sum(1 for j in range(lookahead) if step + j - self.delay_at(step + j) >= step)


class ScheduledDelaySession(EpisodeSession):
    """An episode whose action delay follows a ``DelaySchedule`` with the given semantics.
    The condition must not contain ``action_delay``; its other perturbations (scale, noise,
    observation noise) apply as usual, after the delay."""

    PYTHON_STATE = (*EpisodeSession.PYTHON_STATE, "queue", "last_executed")  # type: ignore[assignment]

    def __init__(self, env, policy, normalizer, condition, episode_seed: int,
                 perturbation_seed: int, schedule: DelaySchedule,
                 semantics: Semantics = "replay_drop"):  # fmt: skip
        if any(d.type == "action_delay" for d in condition.disturbances):
            raise ValueError("schedule the delay, not an action_delay disturbance")
        super().__init__(env, policy, normalizer, condition, episode_seed, perturbation_seed)
        self.schedule, self.semantics = schedule, semantics
        self.queue: list[np.ndarray] = []
        self.last_executed = np.zeros(int(self.low.shape[0]))

    def delayed(self, step: int, command: np.ndarray) -> np.ndarray:
        d = self.schedule.delay_at(step)
        if self.semantics == "replay_drop":
            source = step - d
            return self.commands[source].copy() if source >= 0 else np.zeros_like(command)
        self.queue.append(command.copy())
        extra = len(self.queue) - 1 - d
        if extra == 0:
            out = self.queue.pop(0)
        elif extra > 0:  # shrink: merge what would be dropped
            out = np.sum([self.queue.pop(0) for _ in range(extra + 1)], axis=0)
        else:  # grow: hold the last executed command while the queue fills
            out = self.last_executed.copy()
        self.last_executed = np.asarray(out, np.float64).copy()
        return out

    def act(self, command: np.ndarray) -> np.ndarray:
        if self.pending_view is None:
            raise RolloutContractError("call observe() before act()")
        step = self.step
        command = np.asarray(command, np.float64).copy()
        self.commands.append(command)
        action = self.delayed(step, command)
        for perturbation in self._ordered(ActionPerturbation):
            action = perturbation.apply(step, action, self.commands)
        executed = np.clip(action, self.low, self.high)
        self.observation, _, terminated, truncated, info = self.env.step(
            executed.astype(np.float32)
        )
        last = step == self.horizon - 1
        if terminated or (truncated and not last) or (last and not truncated):
            raise RolloutContractError(f"episode ended unexpectedly at step {step + 1}")
        self.successes[step] = bool(info["is_success"])
        self.final_distance = float(
            np.linalg.norm(self.observation["achieved_goal"] - self.raw.goal)
        )
        if self.successes[step] != (self.final_distance < self.threshold):
            raise RolloutContractError("info['is_success'] disagrees with the ground truth")
        self.step += 1
        self.pending_view = None
        return executed
