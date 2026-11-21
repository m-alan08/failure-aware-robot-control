"""Training-disturbance mixture for ``ResidualEnv`` (residual RL pre-registration §3).

NOT APPROVED, NOT RUN. This wires the five training families into per-episode sessions:

| id  | share | built by |
|-----|-------|----------|
| T-N | 0.25  | the base session with no disturbance |
| T-D | 0.25  | a ``replay_drop`` G3-generator schedule (RT/RI/RS/RP, delay 1..4), seed 20261900 |
| T-M | 0.20  | ``ObjectDynamicsPatch`` for the episode, restored when it ends |
| T-O | 0.15  | an AR(1) ``positions`` observation perturbation injected into the session |
| T-A | 0.15  | ``action_noise`` sigma ~ U[0, 0.2] plus ``action_scale`` s ~ U[0, 0.4] |

Everything an episode sees is a pure function of (generator seed, episode seed): the family,
its parameters, the delay schedule and the noise streams. No draw depends on training state,
so the training curriculum is reproducible and identical across arms that share an episode
seed.

The disturbances act exactly as their existing counterparts do: observation noise reaches the
controller's copy of the observation (the frozen base policy and Candidate C both see it),
action disturbances apply after the delay, and the simulator state, success and distance used
for reward stay ground truth.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from farc.delaychange.random_schedules import RandomCell, draw_pulses
from farc.delaychange.schedule import DelaySchedule
from farc.disturbances.config import ConditionSpec, DisturbanceSpec
from farc.disturbances.perturbations import ObservationNoise
from farc.residual.disturbances import (
    AR_RHO,
    SIGMA_RANGE,
    ObjectDynamics,
    ObjectDynamicsPatch,
)

TRAINING_GENERATOR_SEED = 20261900
FAMILIES = ("T-N", "T-D", "T-M", "T-O", "T-A")
SHARES = {"T-N": 0.25, "T-D": 0.25, "T-M": 0.20, "T-O": 0.15, "T-A": 0.15}
DELAY_FAMILIES = ("RT", "RI", "RS", "RP")
# G3's draw_pulses takes a cell index; training cells get indices disjoint from G3/G4's 0-8 so
# that no training schedule coincides with an evaluation schedule by construction.
DELAY_CELL_INDEX_BASE = 1000
ACTION_NOISE_RANGE = (0.0, 0.2)
ACTION_SCALE_RANGE = (0.0, 0.4)
# Stream ids for the training RNGs (spawn keys); never renumber.
_STREAM_DRAW, _STREAM_TO = 1, 2


def check_shares(shares: Mapping[str, float]) -> dict[str, float]:
    """Shares must cover a subset of FAMILIES, be non-negative and sum to 1 (A2/A3 arms pass
    renormalised or reduced shares)."""
    unknown = set(shares) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown training families {sorted(unknown)}")
    if any(not np.isfinite(v) or v < 0 for v in shares.values()):
        raise ValueError("shares must be finite and non-negative")
    if abs(sum(shares.values()) - 1.0) > 1e-9:
        raise ValueError(f"shares must sum to 1, got {sum(shares.values())}")
    return {f: float(shares[f]) for f in FAMILIES if shares.get(f, 0.0) > 0}


def without(shares: Mapping[str, float], family: str) -> dict[str, float]:
    """Drop one family and renormalise (arm A2: T without T-D)."""
    rest = {f: v for f, v in shares.items() if f != family}
    total = sum(rest.values())
    return check_shares({f: v / total for f, v in rest.items()})


@dataclass(frozen=True)
class TrainingDraw:
    family: str
    episode_seed: int
    dynamics: ObjectDynamics = field(default_factory=ObjectDynamics)
    obs_sigma: float = 0.0
    action_noise: float = 0.0
    action_scale: float = 0.0
    delay_family: str | None = None
    delay: int = 0
    schedule: DelaySchedule = field(default_factory=lambda: DelaySchedule(()))

    @property
    def extra(self) -> tuple[DisturbanceSpec, ...]:
        """Existing disturbance specs for T-A; zero-severity parts are omitted (a no-op)."""
        specs = []
        if self.action_noise > 0:
            specs.append(DisturbanceSpec(type="action_noise", severity=self.action_noise))
        if self.action_scale > 0:
            specs.append(DisturbanceSpec(type="action_scale", severity=self.action_scale))
        return tuple(specs)

    @property
    def is_nominal(self) -> bool:
        return (self.dynamics.is_nominal and self.obs_sigma == 0.0 and not self.extra
                and not self.schedule.pulses)  # fmt: skip

    def to_dict(self) -> dict[str, Any]:
        return {"family": self.family, "episode_seed": self.episode_seed,
                "mass_scale": self.dynamics.mass_scale,
                "friction_scale": self.dynamics.friction_scale, "obs_sigma": self.obs_sigma,
                "action_noise": self.action_noise, "action_scale": self.action_scale,
                "delay_family": self.delay_family, "delay": self.delay,
                "pulses": [list(p) for p in self.schedule.pulses]}  # fmt: skip


def _rng(generator_seed: int, episode_seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence(generator_seed, spawn_key=(episode_seed, stream))
    )


def draw_training(episode_seed: int, shares: Mapping[str, float] = SHARES,
                  generator_seed: int = TRAINING_GENERATOR_SEED) -> TrainingDraw:  # fmt: skip
    """The episode's family and parameters, a pure function of the two seeds."""
    shares = check_shares(shares)
    rng = _rng(generator_seed, episode_seed, _STREAM_DRAW)
    names = list(shares)
    family = names[int(rng.choice(len(names), p=[shares[n] for n in names]))]
    if family == "T-N":
        return TrainingDraw(family, episode_seed)
    if family == "T-M":
        return TrainingDraw(family, episode_seed, dynamics=ObjectDynamics.sample(rng))
    if family == "T-O":
        return TrainingDraw(family, episode_seed, obs_sigma=float(rng.uniform(*SIGMA_RANGE)))
    if family == "T-A":
        return TrainingDraw(family, episode_seed,
                            action_noise=float(rng.uniform(*ACTION_NOISE_RANGE)),
                            action_scale=float(rng.uniform(*ACTION_SCALE_RANGE)))  # fmt: skip
    k = int(rng.integers(len(DELAY_FAMILIES)))
    delay = int(rng.integers(1, 5))
    cell = RandomCell(f"train_{DELAY_FAMILIES[k]}_d{delay}", DELAY_FAMILIES[k], delay)
    pulses = draw_pulses(cell, DELAY_CELL_INDEX_BASE + 10 * k + delay, episode_seed,
                         generator_seed)  # fmt: skip
    return TrainingDraw(family, episode_seed, delay_family=DELAY_FAMILIES[k], delay=delay,
                        schedule=DelaySchedule(pulses))  # fmt: skip


class AR1PositionNoise(ObservationNoise):
    """T-O as a session observation perturbation: AR(1) unit state per entry (rho 0.8),
    scaled by sigma through the ``positions`` mask. Subclasses ``ObservationNoise`` so the
    session applies it to the controller's view only, in its usual place."""

    def __init__(self, sigma: float, rng: np.random.Generator, horizon: int,
                 rho: float = AR_RHO):  # fmt: skip
        # The spec documents the family; its severity is informational (sigma may be 0).
        spec = DisturbanceSpec(type="observation_noise", severity=max(sigma, 1e-300),
                               mask="positions")  # fmt: skip
        super().__init__(spec, rng, horizon)
        self.sigma, self.rho = float(sigma), float(rho)
        self._innovation = float(np.sqrt(1.0 - rho * rho))
        self._state: np.ndarray | None = None

    def apply(self, step: int, view: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        from farc.envs.observation_layout import masked_noise

        if self.layout is None:
            raise RuntimeError("AR1PositionNoise needs configure(layout, dt)")
        draw = self.rng.standard_normal(view["observation"].shape)
        self._state = draw if self._state is None else (
            self.rho * self._state + self._innovation * draw)  # fmt: skip
        noisy = {key: value.copy() for key, value in view.items()}
        if self.sigma > 0:
            noisy["observation"] = noisy["observation"] + masked_noise(
                self.layout, "positions", self.sigma, self._state, self.dt)  # fmt: skip
        self.active_steps[step] = True
        self.changed_steps[step] = not np.array_equal(view["observation"], noisy["observation"])
        return noisy


class DisturbedSession:
    """Wraps one episode session and owns its T-M patch: the model is patched when the
    session is built and restored as soon as the episode ends (or on ``close``), so the next
    episode always starts from the unpatched model."""

    def __init__(self, session: Any, patch: ObjectDynamicsPatch | None, draw: Any):
        self._session, self._patch, self.draw = session, patch, draw
        if patch is not None:
            patch.__enter__()

    def act(self, command):
        try:
            return self._session.act(command)
        finally:
            if self._session.done:
                self.close()

    def close(self) -> None:
        if self._patch is not None:
            self._patch.__exit__(None, None, None)
            self._patch = None

    @property
    def patched(self) -> bool:
        return self._patch is not None

    def __getattr__(self, name):
        return getattr(self._session, name)


SessionBuilder = Callable[[int, ConditionSpec, DelaySchedule], Any]


class TrainingSessionFactory:
    """``factory(seed)`` for ``ResidualEnv``. ``build(seed, condition, schedule)`` makes the
    base session (in the simulator: a ``ScheduledDelaySession`` with ``replay_drop`` on the
    frozen SAC policy); this class adds the drawn family. Seed validation stays in
    ``ResidualEnv`` / the runner. ``draws`` keeps the episode log for the trainer."""

    def __init__(self, build: SessionBuilder, shares: Mapping[str, float] = SHARES,
                 generator_seed: int = TRAINING_GENERATOR_SEED,
                 friction_mode: str = "object_geom"):  # fmt: skip
        self.build, self.shares = build, check_shares(shares)
        self.generator_seed, self.friction_mode = generator_seed, friction_mode
        self.current: DisturbedSession | None = None
        self.draws: list[TrainingDraw] = []

    def __call__(self, seed: int) -> DisturbedSession:
        if self.current is not None:  # an episode abandoned before its end
            self.current.close()
        draw = draw_training(seed, self.shares, self.generator_seed)
        session = self.build(seed, ConditionSpec(f"train_{draw.family}", draw.extra),
                             draw.schedule)  # fmt: skip
        self.current = disturb(session, draw, draw.dynamics, draw.obs_sigma, seed,
                               self.generator_seed, self.friction_mode)  # fmt: skip
        self.draws.append(draw)
        return self.current


def disturb(session: Any, record: Any, dynamics: ObjectDynamics, obs_sigma: float, seed: int,
            generator_seed: int = TRAINING_GENERATOR_SEED,
            friction_mode: str = "object_geom") -> DisturbedSession:  # fmt: skip
    """Add T-O (AR(1) controller-view noise) and T-M (model patch for the episode) to a fresh
    session. Zero sigma and nominal dynamics add nothing at all. ``record`` is kept on the
    returned session for logging (a ``TrainingDraw`` or an evaluation episode spec)."""
    if session.step != 0 or session.pending_view is not None:
        raise RuntimeError("disturbances must be added before the first observation")
    if obs_sigma > 0:
        from farc.envs.observation_layout import control_period

        noise = AR1PositionNoise(obs_sigma, _rng(generator_seed, seed, _STREAM_TO),
                                 session.horizon)  # fmt: skip
        noise.configure(session.layout, control_period(session.raw))
        session.perturbations.append(noise)
    patch = None
    if not dynamics.is_nominal:
        patch = ObjectDynamicsPatch(session.raw.model, dynamics, friction_mode=friction_mode)
    return DisturbedSession(session, patch, record)
