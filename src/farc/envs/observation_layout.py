# Migrated from m-alan08/quantrl-fetch src/quantrl/disturbances/observation_layout.py
# (introduced in 3b3f495, last changed in 3b3f495). See PROVENANCE.md.
"""Layout of the Fetch ``observation`` vector (gymnasium-robotics 1.4.2, MuJoCo bindings).

Source: ``generate_mujoco_observations`` and ``_get_obs`` in
``gymnasium_robotics/envs/fetch/fetch_env.py``. Without an object (FetchReach) the object
entries have length 0. Velocities are multiplied by the control period
``dt = n_substeps * opt.timestep`` (0.04 s), so they are displacements per control step.

| Component | Reach | Push | Quantity | Unit in the vector |
| --- | --- | --- | --- | --- |
| gripper_position | 0:3 | 0:3 | grip site world position | m |
| object_position | - | 3:6 | object site world position | m |
| object_relative_position | - | 6:9 | object_position - gripper_position | m |
| finger_position | 3:5 | 9:11 | left/right finger slide joints | m |
| object_rotation | - | 11:14 | object Euler angles | rad |
| object_relative_velocity | - | 14:17 | (object - gripper) linear velocity * dt | m per step |
| object_angular_velocity | - | 17:20 | object angular velocity * dt | rad per step |
| gripper_velocity | 5:8 | 20:23 | grip site linear velocity * dt | m per step |
| finger_velocity | 8:10 | 23:25 | finger joint velocities * dt | m per step |

``validate_layout`` checks these indices against the simulator at the start of every episode.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class ObservationLayoutError(RuntimeError):
    pass


@dataclass(frozen=True)
class FetchObservationLayout:
    size: int
    has_object: bool
    slices: dict[str, slice]

    def indices(self, component: str) -> np.ndarray:
        part = self.slices[component]
        return np.arange(part.start, part.stop)


REACH = FetchObservationLayout(
    10,
    False,
    {
        "gripper_position": slice(0, 3),
        "finger_position": slice(3, 5),
        "gripper_velocity": slice(5, 8),
        "finger_velocity": slice(8, 10),
    },
)
OBJECT = FetchObservationLayout(
    25,
    True,
    {
        "gripper_position": slice(0, 3),
        "object_position": slice(3, 6),
        "object_relative_position": slice(6, 9),
        "finger_position": slice(9, 11),
        "object_rotation": slice(11, 14),
        "object_relative_velocity": slice(14, 17),
        "object_angular_velocity": slice(17, 20),
        "gripper_velocity": slice(20, 23),
        "finger_velocity": slice(23, 25),
    },
)

# Cartesian positions that a position sensor would measure. Finger positions are excluded:
# the gripper is blocked in Reach and Push, so they are constant. The object's relative
# position is not given independent noise; it is recomputed from the noisy positions.
POSITION_COMPONENTS = ("gripper_position", "object_position")
# Linear velocities, severity in m/s before the per-step conversion. Angular velocity (rad/s)
# is excluded so that one severity has one unit. The object's relative velocity gets the
# object's own noise minus the gripper's, like the relative position.
VELOCITY_COMPONENTS = ("gripper_velocity", "object_relative_velocity", "finger_velocity")


def layout_for(unwrapped: Any, observation_size: int) -> FetchObservationLayout:
    has_object = bool(getattr(unwrapped, "has_object", False))
    layout = OBJECT if has_object else REACH
    if observation_size != layout.size:
        raise ObservationLayoutError(
            f"observation has {observation_size} entries; the Fetch layout with "
            f"has_object={has_object} has {layout.size}"
        )
    return layout


def control_period(unwrapped: Any) -> float:
    return float(unwrapped.n_substeps * unwrapped.model.opt.timestep)


def validate_layout(unwrapped: Any, observation: np.ndarray, layout: FetchObservationLayout):
    """Check the layout against the simulator's state for this observation."""
    utils, model, data = unwrapped._utils, unwrapped.model, unwrapped.data
    observation = np.asarray(observation, dtype=np.float64)
    dt = control_period(unwrapped)
    checks = {
        "gripper_position": utils.get_site_xpos(model, data, "robot0:grip"),
        "gripper_velocity": utils.get_site_xvelp(model, data, "robot0:grip") * dt,
    }
    if layout.has_object:
        object_position = utils.get_site_xpos(model, data, "object0")
        checks["object_position"] = object_position
        checks["object_relative_position"] = object_position - checks["gripper_position"]
        checks["object_relative_velocity"] = (
            utils.get_site_xvelp(model, data, "object0") * dt - checks["gripper_velocity"]
        )
    for component, expected in checks.items():
        actual = observation[layout.slices[component]]
        if not np.allclose(actual, expected, atol=1e-9):
            raise ObservationLayoutError(
                f"observation[{layout.slices[component]}] is {actual}, but the simulator's "
                f"{component} is {expected}"
            )


def masked_noise(
    layout: FetchObservationLayout,
    mask: str,
    severity: float,
    standard_draw: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Noise to add to the observation vector for one step.

    ``standard_draw`` is one standard normal per observation entry, drawn whatever the mask,
    so a given step uses the same draws under every mask and severity.
    """
    noise = np.zeros(layout.size)
    if mask == "full":
        return severity * standard_draw
    if mask == "positions":
        for component in POSITION_COMPONENTS:
            if component in layout.slices:
                part = layout.slices[component]
                noise[part] = severity * standard_draw[part]
        if layout.has_object:
            noise[layout.slices["object_relative_position"]] = (
                noise[layout.slices["object_position"]] - noise[layout.slices["gripper_position"]]
            )
        return noise
    if mask == "velocities":
        for component in ("gripper_velocity", "finger_velocity"):
            part = layout.slices[component]
            noise[part] = severity * dt * standard_draw[part]
        if layout.has_object:
            # the entry is (object - gripper) velocity: independent noise on the object's own
            # velocity, minus the gripper's velocity noise, keeps the two entries consistent
            part = layout.slices["object_relative_velocity"]
            noise[part] = (
                severity * dt * standard_draw[part] - noise[layout.slices["gripper_velocity"]]
            )
        return noise
    raise ValueError(f"unknown observation mask {mask!r}")
