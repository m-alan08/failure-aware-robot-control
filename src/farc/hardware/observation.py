"""Hardware observation adapter: robot/sensor state -> the frozen policy's 25-dim FetchPush
observation (layout in ``farc.envs.observation_layout``; source ``fetch_env._get_obs``).

Input (``RawState``), all in the **robot base frame**, SI units, with sensor timestamps:
- gripper (grip site) position [m] and linear velocity [m/s], from proprioception;
- finger slide positions [m] and velocities [m/s], or ``None`` (then the calibrated blocked
  finger position and zero velocity are used, as FetchPush keeps the gripper closed);
- object marker position [m], orientation quaternion (w, x, y, z), linear [m/s] and angular
  [rad/s] velocity, from the object sensor.

Output, in the **FetchPush world frame** (``world = R @ base + t`` from the calibration):

| entries | quantity | computed as (same operation order as the simulator) |
| --- | --- | --- |
| 0:3 | gripper position | R p_g + t |
| 3:6 | object position | R (p_m + R_o offset) + t |
| 6:9 | object - gripper | (3:6) - (0:3) |
| 9:11 | finger positions | measured, or calibrated constant |
| 11:14 | object Euler angles | ``rotations.mat2euler(R R_o)`` (the simulator's function) |
| 14:17 | relative linear velocity per step | (R v_o) dt - (R v_g) dt |
| 17:20 | object angular velocity per step | (R w_o) dt |
| 20:23 | gripper linear velocity per step | (R v_g) dt |
| 23:25 | finger velocities per step | v_f dt (zero if not measured) |

The goal is a world-frame point (3,). Normalisation is *not* done here: the controller applies
the frozen policy normaliser, unchanged.

Validation (``ObservationError``; the controller turns any error into a safety stop): wrong
shapes, non-finite values, a non-unit quaternion, a sample older than
``max_observation_age_s`` at decision time, gripper/object timestamps further apart than
``max_sensor_skew_s``, or a timestamp that does not increase (a repeated, stale sample). No
interpolation or extrapolation is done: H0 uses the latest valid sample of each sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium_robotics.utils import rotations

from farc.envs.observation_layout import OBJECT
from farc.hardware.config import HardwareConfig

S = OBJECT.slices
OBS_DIM = 25
QUAT_TOL = 1e-3


class ObservationError(RuntimeError):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"{kind}: {detail}" if detail else kind)
        self.kind = kind


@dataclass(frozen=True)
class RawState:
    decision_ns: int  # monotonic clock at the decision
    gripper_stamp_ns: int
    object_stamp_ns: int
    gripper_position: np.ndarray
    gripper_velocity: np.ndarray
    object_position: np.ndarray  # marker position
    object_quaternion: np.ndarray  # (w, x, y, z)
    object_velocity: np.ndarray
    object_angular_velocity: np.ndarray
    finger_position: np.ndarray | None = None
    finger_velocity: np.ndarray | None = None


def _vec(name: str, v, n: int) -> np.ndarray:
    a = np.asarray(v, np.float64)
    if a.shape != (n,):
        raise ObservationError("shape", f"{name} has shape {a.shape}, expected ({n},)")
    if not np.all(np.isfinite(a)):
        raise ObservationError("non_finite", name)
    return a


class ObservationAdapter:
    def __init__(self, config: HardwareConfig):
        self.config = config
        self.R, self.t = config.rotation, config.translation
        if not np.allclose(self.R @ self.R.T, np.eye(3), atol=1e-9) or np.linalg.det(self.R) < 0:
            raise ValueError("world_from_base_rotation is not a proper rotation")
        cal = config.calibration
        self.offset = np.asarray(cal.object_marker_offset_m or [0.0, 0.0, 0.0], np.float64)
        self.fingers = (None if cal.finger_position_m is None
                        else np.asarray(cal.finger_position_m, np.float64))  # fmt: skip
        self.dt = config.control_period_s
        self.reset()

    def reset(self) -> None:
        self._last: tuple[int, int] | None = None

    def _check_time(self, s: RawState) -> None:
        cal = self.config.calibration
        if cal.max_observation_age_s is None or cal.max_sensor_skew_s is None:
            raise ObservationError("uncalibrated", "age/skew limits are placeholders")
        age = max(s.decision_ns - s.gripper_stamp_ns, s.decision_ns - s.object_stamp_ns) * 1e-9
        if min(s.decision_ns - s.gripper_stamp_ns, s.decision_ns - s.object_stamp_ns) < 0:
            raise ObservationError("future_stamp", "a sample is newer than the decision time")
        if age > cal.max_observation_age_s + 1e-12:
            raise ObservationError("stale", f"sample age {age:.4f} s")
        if abs(s.gripper_stamp_ns - s.object_stamp_ns) * 1e-9 > cal.max_sensor_skew_s + 1e-12:
            raise ObservationError("skew", "gripper and object samples too far apart")
        if self._last is not None and (s.gripper_stamp_ns <= self._last[0]
                                       or s.object_stamp_ns <= self._last[1]):  # fmt: skip
            raise ObservationError("repeated", "a sensor timestamp did not increase")

    def observation(self, s: RawState) -> np.ndarray:
        self._check_time(s)
        R, t, dt = self.R, self.t, self.dt
        pg = _vec("gripper_position", s.gripper_position, 3)
        vg = _vec("gripper_velocity", s.gripper_velocity, 3)
        pm = _vec("object_position", s.object_position, 3)
        q = _vec("object_quaternion", s.object_quaternion, 4)
        vo = _vec("object_velocity", s.object_velocity, 3)
        wo = _vec("object_angular_velocity", s.object_angular_velocity, 3)
        if abs(float(np.linalg.norm(q)) - 1.0) > QUAT_TOL:
            raise ObservationError("quaternion", "object quaternion is not unit length")
        if s.finger_position is not None:
            pf = _vec("finger_position", s.finger_position, 2)
            vf = (_vec("finger_velocity", s.finger_velocity, 2) if s.finger_velocity is not None
                  else np.zeros(2))  # fmt: skip
        elif self.fingers is not None:
            pf, vf = self.fingers.copy(), np.zeros(2)
        else:
            raise ObservationError("uncalibrated", "no finger measurement and no constant")
        ro = rotations.quat2mat(q)
        obs = np.zeros(OBS_DIM, np.float64)
        obs[S["gripper_position"]] = R @ pg + t
        obs[S["object_position"]] = R @ (pm + ro @ self.offset) + t
        obs[S["object_relative_position"]] = obs[S["object_position"]] - obs[S["gripper_position"]]
        obs[S["finger_position"]] = pf
        obs[S["object_rotation"]] = rotations.mat2euler(R @ ro)
        grip_velp = (R @ vg) * dt
        object_velp = (R @ vo) * dt
        object_velp -= grip_velp
        obs[S["object_relative_velocity"]] = object_velp
        obs[S["object_angular_velocity"]] = (R @ wo) * dt
        obs[S["gripper_velocity"]] = grip_velp
        obs[S["finger_velocity"]] = vf * dt
        self._last = (s.gripper_stamp_ns, s.object_stamp_ns)
        return obs

    def goal(self, goal_world) -> np.ndarray:
        return _vec("goal", goal_world, 3).copy()

    def goal_from_base(self, goal_base) -> np.ndarray:
        return self.R @ _vec("goal", goal_base, 3) + self.t


def raw_from_sim_observation(obs: np.ndarray, decision_ns: int, period_ns: int = 40_000_000
                             ) -> RawState:  # fmt: skip
    """Inverse map for offline replay: a recorded simulator observation expressed as the raw
    sensor state an identity-calibrated robot would report. Velocities are divided by dt;
    Euler angles become a quaternion (``rotations.euler2quat``)."""
    o = np.asarray(obs, np.float64)
    dt = period_ns * 1e-9
    vg = o[S["gripper_velocity"]] / dt
    vo = (o[S["object_relative_velocity"]] + o[S["gripper_velocity"]]) / dt
    return RawState(
        decision_ns=decision_ns, gripper_stamp_ns=decision_ns, object_stamp_ns=decision_ns,
        gripper_position=o[S["gripper_position"]].copy(), gripper_velocity=vg,
        object_position=o[S["object_position"]].copy(),
        object_quaternion=rotations.euler2quat(o[S["object_rotation"]]),
        object_velocity=vo, object_angular_velocity=o[S["object_angular_velocity"]] / dt,
        finger_position=o[S["finger_position"]].copy(),
        finger_velocity=o[S["finger_velocity"]] / dt,
    )  # fmt: skip
