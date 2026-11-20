"""Training disturbance families T-M and T-O of the residual RL pre-registration (§3).

``docs/RESIDUAL_RL_PREREGISTRATION.md`` is a DRAFT and is NOT approved. This module only
implements the two families the plan marks "to implement", exactly as defined there, so that
the §3 implementation checks can be run once the plan is approved. Nothing here trains, and
nothing here chooses episode seeds.

- **T-M (object dynamics):** object mass x U[0.5, 2.0] and sliding friction x U[0.5, 1.5],
  fixed for the whole episode.
- **T-O (correlated observation noise):** AR(1) noise on the gripper and object Cartesian
  positions (mask ``positions``) with rho = 0.8 and stationary sigma ~ U[0, 1 cm]. The object's
  relative position is recomputed from the noisy positions, as in the existing
  ``observation_noise`` mask.

Both have a severity at which they are an exact no-op (scales 1.0, sigma 0.0), which is what
the pre-registered nominal-identity check uses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from farc.envs.observation_layout import FetchObservationLayout, masked_noise

MASS_RANGE = (0.5, 2.0)
FRICTION_RANGE = (0.5, 1.5)
AR_RHO = 0.8
SIGMA_RANGE = (0.0, 0.01)  # metres
OBJECT_NAME = "object0"


# ----------------------------------------------------------------------------------- T-M ---


@dataclass(frozen=True)
class ObjectDynamics:
    """Multiplicative scales on the object's mass and sliding friction (T-M)."""

    mass_scale: float = 1.0
    friction_scale: float = 1.0

    def __post_init__(self):
        for name, value in (("mass_scale", self.mass_scale),
                            ("friction_scale", self.friction_scale)):
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive finite number, got {value!r}")

    @classmethod
    def sample(cls, rng: np.random.Generator) -> ObjectDynamics:
        return cls(float(rng.uniform(*MASS_RANGE)), float(rng.uniform(*FRICTION_RANGE)))

    @property
    def is_nominal(self) -> bool:
        return self.mass_scale == 1.0 and self.friction_scale == 1.0


class ObjectDynamicsPatch:
    """Applies an ``ObjectDynamics`` to a MuJoCo model and restores the original values.

    Use as a context manager around one episode. The nominal setting writes nothing, so it is
    bitwise identical to an unpatched episode. Restoring writes back the exact saved values.
    """

    def __init__(self, model: Any, dynamics: ObjectDynamics, object_name: str = OBJECT_NAME):
        import mujoco

        self.model = model
        self.dynamics = dynamics
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        self.geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_name)
        if self.body_id < 0 or self.geom_id < 0:
            raise ValueError(f"model has no body and geom named {object_name!r}")
        self._saved: tuple[float, np.ndarray, float] | None = None

    def __enter__(self) -> ObjectDynamicsPatch:
        if self._saved is not None:
            raise RuntimeError("patch is already applied")
        m = self.model
        self._saved = (float(m.body_mass[self.body_id]),
                       m.body_inertia[self.body_id].copy(),
                       float(m.geom_friction[self.geom_id, 0]))
        if not self.dynamics.is_nominal:
            s = self.dynamics.mass_scale
            m.body_mass[self.body_id] = self._saved[0] * s
            # Same shape and density change: the inertia scales with the mass.
            m.body_inertia[self.body_id] = self._saved[1] * s
            m.geom_friction[self.geom_id, 0] = self._saved[2] * self.dynamics.friction_scale
        return self

    def __exit__(self, *exc) -> None:
        if self._saved is None:
            return
        mass, inertia, friction = self._saved
        self.model.body_mass[self.body_id] = mass
        self.model.body_inertia[self.body_id] = inertia
        self.model.geom_friction[self.geom_id, 0] = friction
        self._saved = None


# ----------------------------------------------------------------------------------- T-O ---


class CorrelatedObservationNoise:
    """AR(1) position noise with stationary standard deviation ``sigma`` (T-O).

    A unit-variance AR(1) state z is kept per observation entry:
    z_0 ~ N(0, 1), z_t = rho * z_{t-1} + sqrt(1 - rho^2) * e_t. The added noise is
    ``masked_noise(layout, "positions", sigma, z_t, dt)``, so every perturbed position entry is
    stationary with standard deviation sigma and lag-1 autocorrelation rho. Draws happen at
    every step whatever sigma is (common random numbers across severities).
    """

    def __init__(self, layout: FetchObservationLayout, sigma: float,
                 rng: np.random.Generator, rho: float = AR_RHO, dt: float = 0.04):
        if not (math.isfinite(sigma) and sigma >= 0):
            raise ValueError(f"sigma must be a non-negative finite number, got {sigma!r}")
        if not 0 <= rho < 1:
            raise ValueError(f"rho must be in [0, 1), got {rho!r}")
        self.layout, self.sigma, self.rng, self.rho, self.dt = layout, sigma, rng, rho, dt
        self._innovation = math.sqrt(1.0 - rho * rho)
        self._state: np.ndarray | None = None

    @classmethod
    def sample(cls, layout: FetchObservationLayout, rng: np.random.Generator,
               dt: float = 0.04) -> CorrelatedObservationNoise:
        return cls(layout, float(rng.uniform(*SIGMA_RANGE)), rng, dt=dt)

    def reset(self) -> None:
        self._state = None

    def noise(self) -> np.ndarray:
        draw = self.rng.standard_normal(self.layout.size)
        if self._state is None:
            self._state = draw
        else:
            self._state = self.rho * self._state + self._innovation * draw
        return masked_noise(self.layout, "positions", self.sigma, self._state, self.dt)

    def apply(self, observation: np.ndarray) -> np.ndarray:
        """Return the controller's noisy copy; the input is never modified."""
        noise = self.noise()
        if self.sigma == 0.0:
            return observation.copy()
        return observation + noise
