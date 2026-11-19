"""Hardware configuration with explicit calibration placeholders.

Every value that needs a real measurement is ``None`` in the committed template
(``configs/hardware/h0_template.json``). ``HardwareConfig.require_calibrated()`` refuses to
proceed while any placeholder is unset, so a live run cannot silently use an invented value.
The offline replay configuration (``replay_config``) is the *simulation* frame by definition
(identity transform), not a calibration of any robot.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

CONTROL_PERIOD_S = 0.04  # FetchPush: n_substeps (20) * timestep (0.002); 25 Hz
ACTION_SCALE_M = 0.05  # FetchPush: pos_ctrl *= 0.05 (metres per unit action per step)
PLATFORMS = ("fetch", "franka", "sim_replay")


class CalibrationMissing(RuntimeError):
    """A value that needs a real measurement is still a placeholder."""


@dataclass(frozen=True)
class Calibration:
    """Values that can only come from the physical setup (all placeholders until measured).

    - ``world_from_base_rotation`` / ``_translation``: robot base frame -> FetchPush world frame
      (3x3 rotation, metres). FetchPush's world frame is the simulator's; the table top is at
      z = 0.4 m there. On hardware this is a chosen convention fixed by a calibration procedure.
    - ``object_marker_offset_m``: marker centre -> object centre, in the object frame.
    - ``camera_latency_s``, ``proprio_latency_s``: measured sensor latencies (reported, Q5).
    - ``max_observation_age_s``: oldest acceptable sample at decision time.
    - ``max_sensor_skew_s``: largest allowed gripper/object timestamp difference.
    - ``workspace_min_m`` / ``workspace_max_m``: Cartesian box in the world frame enforced by
      the robot-side safety layer (documented here; enforcement lives below the policy).
    - ``finger_position_m``: the blocked gripper's finger slide positions (Push keeps them
      closed).
    """

    world_from_base_rotation: list[list[float]] | None = None
    world_from_base_translation: list[float] | None = None
    object_marker_offset_m: list[float] | None = None
    camera_latency_s: float | None = None
    proprio_latency_s: float | None = None
    max_observation_age_s: float | None = None
    max_sensor_skew_s: float | None = None
    workspace_min_m: list[float] | None = None
    workspace_max_m: list[float] | None = None
    finger_position_m: list[float] | None = None

    def missing(self) -> list[str]:
        return [f.name for f in fields(self) if getattr(self, f.name) is None]


@dataclass(frozen=True)
class HardwareConfig:
    platform: str
    control_period_s: float = CONTROL_PERIOD_S
    action_scale_m: float = ACTION_SCALE_M
    calibration: Calibration = field(default_factory=Calibration)
    p99_budget_s: float = 0.010  # proposal section 4.4

    def __post_init__(self) -> None:
        if self.platform not in PLATFORMS:
            raise ValueError(f"platform must be one of {PLATFORMS}")
        if self.control_period_s != CONTROL_PERIOD_S:
            raise ValueError("the frozen policy was trained at 25 Hz; the period is fixed")

    def require_calibrated(self) -> None:
        missing = self.calibration.missing()
        if missing:
            raise CalibrationMissing(f"{self.platform}: uncalibrated placeholders: {missing}")

    @property
    def rotation(self) -> np.ndarray:
        r = self.calibration.world_from_base_rotation
        if r is None:
            raise CalibrationMissing("world_from_base_rotation")
        return np.asarray(r, np.float64)

    @property
    def translation(self) -> np.ndarray:
        p = self.calibration.world_from_base_translation
        if p is None:
            raise CalibrationMissing("world_from_base_translation")
        return np.asarray(p, np.float64)


def load_config(path: str | Path) -> HardwareConfig:
    v: dict[str, Any] = json.loads(Path(path).read_text())
    v.pop("description", None)
    cal = Calibration(**v.pop("calibration", {}))
    return HardwareConfig(calibration=cal, **v)


def replay_config() -> HardwareConfig:
    """The simulator's own frame: identity transform, no latency, no age limit beyond one
    period. Used only to replay recorded simulation trajectories."""
    return HardwareConfig(
        platform="sim_replay",
        calibration=Calibration(
            world_from_base_rotation=np.eye(3).tolist(),
            world_from_base_translation=[0.0, 0.0, 0.0],
            object_marker_offset_m=[0.0, 0.0, 0.0],
            camera_latency_s=0.0,
            proprio_latency_s=0.0,
            max_observation_age_s=CONTROL_PERIOD_S,
            max_sensor_skew_s=0.0,
            workspace_min_m=[-np.inf, -np.inf, -np.inf],
            workspace_max_m=[np.inf, np.inf, np.inf],
            finger_position_m=None,  # replay passes the recorded finger state through
        ),
    )
