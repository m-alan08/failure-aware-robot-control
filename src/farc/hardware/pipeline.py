"""One control cycle: raw sensor state -> adapter -> controller -> FIFO -> log record.

Safety rule (proposal section 4.5): any invalid, missing or stale observation ends the trial.
The cycle returns a zero command with ``safety_event`` set, the trial is marked aborted, and
every later call refuses to act. Nothing is retried or interpolated. The FIFO output is what a
robot *would* execute; there is no robot connection.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from farc.hardware.controller import HardwareController
from farc.hardware.fifo import CommandFIFO
from farc.hardware.logging import empty_record
from farc.hardware.observation import ObservationAdapter, ObservationError, RawState
from farc.hardware.shadow import ShadowSupervisor


@dataclass
class CycleOutput:
    command: np.ndarray  # issued by the controller (zeros on a safety stop)
    executed: np.ndarray | None  # what the FIFO releases this cycle (None after a stop)
    record: dict[str, Any]


class HardwarePipeline:
    def __init__(self, adapter: ObservationAdapter, controller: HardwareController,
                 fifo: CommandFIFO, shadow: ShadowSupervisor | None = None):  # fmt: skip
        self.adapter, self.controller, self.fifo = adapter, controller, fifo
        self.shadow = shadow  # logging only; never read when forming a command
        self.reset()

    def reset(self, schedule=None) -> None:
        self.adapter.reset()
        self.controller.reset()
        self.fifo.reset(schedule)
        if self.shadow is not None:
            self.shadow.reset()
        self.aborted: str | None = None

    def cycle(self, state: RawState | None, goal_world: np.ndarray) -> CycleOutput:
        rec = empty_record()
        rec.update({"t": self.controller.t, "configuration": self.controller.name,
                    "mono_ns": time.monotonic_ns()})  # fmt: skip
        if self.aborted is not None:
            rec.update({"error": "trial already aborted", "safety_event": self.aborted})
            return CycleOutput(np.zeros(4), None, rec)
        t0 = time.perf_counter_ns()
        try:
            if state is None:
                raise ObservationError("missing", "no sensor sample this cycle")
            rec["obs_gripper_stamp_ns"] = int(state.gripper_stamp_ns)
            rec["obs_object_stamp_ns"] = int(state.object_stamp_ns)
            obs = self.adapter.observation(state)
            goal = self.adapter.goal(goal_world)
        except ObservationError as err:
            self.aborted = f"observation_{err.kind}"
            rec.update({"error": str(err), "safety_event": self.aborted})
            return CycleOutput(np.zeros(4), None, rec)
        t1 = time.perf_counter_ns()
        out = self.controller.step(obs, goal)
        fifo = self.fifo.push(out.command, time.monotonic_ns())
        if self.shadow is not None:  # after the command is final; fed copies only
            ts = time.perf_counter_ns()
            rec.update(self.shadow.step(obs[0:3], out.command))
            rec["lat_shadow_ns"] = time.perf_counter_ns() - ts
        t2 = time.perf_counter_ns()
        rec.update(out.record)
        rec.update({"lat_adapter_ns": t1 - t0, "lat_pipeline_total_ns": t2 - t0,
                    "injected_delay": fifo.delay, "executed_source_step": fifo.source_step,
                    "executed_command": fifo.executed.tolist()})  # fmt: skip
        return CycleOutput(out.command, fifo.executed, rec)
