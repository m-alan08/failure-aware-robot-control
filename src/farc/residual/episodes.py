"""Selection (§5) and development-evaluation (§6) episode generation. NOT RUN.

The cell grids follow the DRAFT pre-registration. This module turns (cell, episode seed) into a
disturbed session and loops over configurations, policies, network seeds and episode seeds,
writing the long episode table that ``farc.residual.analysis`` reads. Seed rules:

- ``role="selection"`` / ``"evaluation"``: every seed must be in that registered R block, and
  the residual plan must be approved (``check_approved``). Refused today.
- ``role="smoke"``: unregistered ``SMOKE_SEEDS`` only. This is the only role usable now.

Every disturbance is the existing one or the T-M/T-O implementation; nothing new is defined
here. Interpretations of the draft grid are marked ``# INTERPRETATION`` and listed in
``docs/RESIDUAL_RL_PROTOCOL_REVIEW.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import numpy as np

from farc.delaychange.heldout import G4Cell, g4_cells
from farc.delaychange.random_schedules import RandomCell, draw_pulses, g3_cells, schedule_for
from farc.delaychange.schedule import DelaySchedule
from farc.disturbances.config import ConditionSpec, DisturbanceSpec
from farc.residual.disturbances import ObjectDynamics
from farc.residual.training import RunRefused, check_approved, check_smoke_seeds
from farc.residual.training_env import TRAINING_GENERATOR_SEED, disturb, draw_training
from farc.supervision.seeds import R_SEED_BLOCKS, check_r_seeds


@dataclass(frozen=True)
class EpisodeSpec:
    cell: str
    group: str  # N | D | M | X (development); "" at selection
    family: str
    extra: tuple[DisturbanceSpec, ...] = ()
    schedule: DelaySchedule = field(default_factory=lambda: DelaySchedule(()))
    semantics: str = "replay_drop"
    dynamics: ObjectDynamics = field(default_factory=ObjectDynamics)
    obs_sigma: float = 0.0


@dataclass(frozen=True)
class Cell:
    name: str
    group: str
    family: str
    make: Callable[[int], EpisodeSpec]


def _fixed(name, group, family, **kw) -> Cell:
    spec = EpisodeSpec(name, group, family, **kw)
    return Cell(name, group, family, lambda seed: spec)


def _ta(noise: float, scale: float) -> tuple[DisturbanceSpec, ...]:
    return (
        DisturbanceSpec(type="action_noise", severity=noise),
        DisturbanceSpec(type="action_scale", severity=scale),
    )


def _drawn(name: str, family: str, shares: Mapping[str, float], delay_family: str | None = None):
    """Selection cells draw parameters per episode from the training distribution of their
    family (INTERPRETATION: §5 says "one cell per T family" without fixing severities)."""

    def make(seed: int) -> EpisodeSpec:
        if delay_family is not None:
            # INTERPRETATION: T-D split into RT and RP; delay uniform on 1..4 per episode.
            rng = np.random.default_rng([TRAINING_GENERATOR_SEED, 77, seed])
            delay = int(rng.integers(1, 5))
            # cell indices 2000+ are disjoint from G3/G4 (0-8) and training (1000+)
            index = 2000 + 10 * ("RT", "RP").index(delay_family) + delay
            pulses = draw_pulses(RandomCell(name, delay_family, delay), index, seed,
                                 TRAINING_GENERATOR_SEED)  # fmt: skip
            return EpisodeSpec(name, "", family, schedule=DelaySchedule(pulses))
        d = draw_training(seed, shares)
        return EpisodeSpec(name, "", family, d.extra, d.schedule, "replay_drop", d.dynamics,
                           d.obs_sigma)  # fmt: skip

    return Cell(name, "", family, make)


def selection_cells() -> list[Cell]:
    """§5: one cell per T family (T-D split into RT and RP) plus nominal. No X cells."""
    return [
        _fixed("nominal", "", "nominal"),
        _drawn("T-M", "T-M", {"T-M": 1.0}),
        _drawn("T-O", "T-O", {"T-O": 1.0}),
        _drawn("T-A", "T-A", {"T-A": 1.0}),
        _drawn("T-D_RT", "RT", {}, "RT"),
        _drawn("T-D_RP", "RP", {}, "RP"),
    ]


def _g4_spec(c: G4Cell, seed: int) -> EpisodeSpec:
    return EpisodeSpec(c.name, "D", c.family, c.extra, c.schedule(seed), "replay_drop")


def _g4_spec_group(c: G4Cell, group: str, seed: int) -> EpisodeSpec:
    return EpisodeSpec(c.name, group, c.family, c.extra, c.schedule(seed), "replay_drop")


def _xdh_spec(name: str, index: int, cell: RandomCell, seed: int) -> EpisodeSpec:
    # the G3 schedule of the same cell, with drain_hold semantics (never seen in training)
    return EpisodeSpec(name, "X", "X-DH", (), schedule_for(cell, index, seed), "drain_hold")


# Amendment R-A2, GEN-2: the two G4 Arm R cells without any delay leave group D and are
# reported descriptively in their own groups (never part of a primary or secondary contrast).
H_REPORTING_GROUPS = {"H_undisturbed": "N_rep", "H_noise0.01": "O_rep"}


def development_cells(include_h_in_d: bool = False) -> list[Cell]:
    """§6 cells. D = G4 Arm R + Arm F delay cells (39). Under amendment R-A2 (GEN-2) the
    no-delay cells H_undisturbed and H_noise0.01 are excluded from D and kept as the
    reporting groups ``N_rep`` (nominal) and ``O_rep`` (observation noise).
    ``include_h_in_d=True`` reproduces the draft's literal 41-cell D, for audits only."""
    cells = [_fixed("nominal", "N", "nominal")]
    for c in g4_cells():
        if c.family == "H" and not include_h_in_d:
            cells.append(Cell(c.name, H_REPORTING_GROUPS[c.name], c.family,
                              partial(_g4_spec_group, c, H_REPORTING_GROUPS[c.name])))  # fmt: skip
            continue
        cells.append(Cell(c.name, "D", c.family, partial(_g4_spec, c)))
    cells += [
        _fixed("TM_mass0.5", "M", "T-M", dynamics=ObjectDynamics(0.5, 1.0)),
        _fixed("TM_mass2", "M", "T-M", dynamics=ObjectDynamics(2.0, 1.0)),
        _fixed("TM_friction0.5", "M", "T-M", dynamics=ObjectDynamics(1.0, 0.5)),
        _fixed("TO_0.5cm", "M", "T-O", obs_sigma=0.005),
        _fixed("TO_1cm", "M", "T-O", obs_sigma=0.01),
        _fixed("TA_0.1_0.2", "M", "T-A", extra=_ta(0.1, 0.2)),
        _fixed("TA_0.2_0.4", "M", "T-A", extra=_ta(0.2, 0.4)),
    ]
    g3 = {r.name: (i, r) for i, r in enumerate(g3_cells())}
    for name in ("RT_d2", "RT_d4", "RP_d4"):
        i, r = g3[name]
        cells.append(Cell(f"XDH_{name}", "X", "X-DH", partial(_xdh_spec, f"XDH_{name}", i, r)))
    shift = DisturbanceSpec(type="goal_shift", severity=0.05, onset=20)
    cells.append(_fixed("XG_goal_shift", "X", "X-G", extra=(shift,)))
    return cells


def check_role_seeds(role: str, seeds: list[int], plan_config: dict | None) -> None:
    if role == "smoke":
        check_smoke_seeds(seeds)
        return
    if role not in ("selection", "evaluation"):
        raise RunRefused(f"unknown role {role!r}")
    if plan_config is None:
        raise RunRefused("registered episode generation needs the plan config")
    check_approved(plan_config)
    if role == "evaluation":
        # Amendment R-A2: fixed 600 evaluation seeds = evaluation + evaluation_extension
        if list(seeds) != evaluation_seeds():
            raise RunRefused("evaluation uses exactly the 600 registered evaluation seeds")
        check_r_seeds(seeds[:300], "evaluation")
        check_r_seeds(seeds[300:], "evaluation_extension")
        return
    a, b = R_SEED_BLOCKS["selection"]
    if list(seeds) != list(range(a, b)):
        raise RunRefused("selection uses exactly the 100 registered selection seeds")
    check_r_seeds(seeds, role)


def evaluation_seeds() -> list[int]:
    """Amendment R-A2 (fixed budget): 3,722,000-299 then 3,740,100-399, 600 seeds."""
    a, b = R_SEED_BLOCKS["evaluation"]
    c, d = R_SEED_BLOCKS["evaluation_extension"]
    return [*range(a, b), *range(c, d)]


# build(seed, condition, schedule, semantics, policy_id) -> base session
Builder = Callable[[int, ConditionSpec, DelaySchedule, str, Any], Any]
# run(config, network_seed, policy_id, session) -> dict with "success" (and optional extras)
Runner = Callable[[str, int, Any, Any], Mapping[str, Any]]


def generate(cells: Iterable[Cell], seeds: list[int], policies: list[Any],
             configs: Mapping[str, Iterable[int]], build: Builder, run: Runner, role: str,
             plan_config: dict | None = None, alpha: float = float("nan"),
             friction_mode: str = "object_geom",
             generator_seed: int = TRAINING_GENERATOR_SEED) -> list[dict[str, Any]]:  # fmt: skip
    """``configs`` maps a configuration name to its network seeds (``[-1]`` for C and B0).
    Every (configuration, network seed, policy, cell, episode seed) gets a fresh session."""
    check_role_seeds(role, list(seeds), plan_config)
    rows = []
    for cell in cells:
        for seed in seeds:
            spec = cell.make(seed)
            for policy in policies:
                for config, network_seeds in configs.items():
                    for ns in network_seeds:
                        base = build(seed, ConditionSpec(cell.name, spec.extra), spec.schedule,
                                     spec.semantics, policy)  # fmt: skip
                        s = disturb(base, spec, spec.dynamics, spec.obs_sigma, seed,
                                    generator_seed, friction_mode)  # fmt: skip
                        try:
                            out = run(config, ns, policy, s)
                        finally:
                            s.close()
                        rows.append({"config": config,
                                     "alpha": alpha if config != "C" else float("nan"),
                                     "network_seed": ns, "policy": policy,
                                     "episode_seed": seed, "cell": cell.name,
                                     "group": cell.group, "family": cell.family,
                                     "success": bool(out["success"]),
                                     **{k: v for k, v in out.items()
                                        if k != "success"}})  # fmt: skip
    return rows
