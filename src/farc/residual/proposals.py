"""PROPOSED, NOT REGISTERED seed blocks for the residual-RL amendments.

Nothing in the repository reads these as usable seeds: ``check_r_seeds`` only knows
``farc.supervision.seeds.R_SEED_BLOCKS``. These blocks become usable only if the amendment
in ``docs/RESIDUAL_RL_PROTOCOL_REVIEW.md`` is approved and they are copied into
``R_SEED_BLOCKS`` *before* any of them is used. ``proposal_conflicts`` checks them against
every block the repository registers or reserves.
"""

from __future__ import annotations

import farc.supervision.seeds as _seeds
from farc.disturbances.config import SEED_RANGES

EPISODE_STEPS = 50
TRAINING_STEPS = 1_000_000  # §4 budget per run

PROPOSED_R_SEED_BLOCKS = {
    # with the registered 3,710,000-3,719,999: 20,000 episodes = 1M steps / 50, exactly,
    # given that a rollback does not rewind the environment-step counter
    "training_2": (3_730_000, 3_740_000),
    # 100 nominal episodes for the rollback rule, so that §5 selection seeds are never seen
    # during training (the draft uses the selection block for both)
    "training_monitor": (3_740_000, 3_740_100),
    # STAT-4: a 300-seed extension of the evaluation block (3,722,000-299), used only if the
    # pre-specified selection-block discordance rule calls for 600 evaluation seeds
    "evaluation_extension": (3_740_100, 3_740_400),
}


def registered_blocks() -> dict[str, tuple[int, int]]:
    """Every (low, high) block the repository registers, reserves or lists as used."""
    out = {f"SEED_RANGES.{k}": v for k, v in SEED_RANGES.items() if k != "validation"}
    for name in dir(_seeds):
        value = getattr(_seeds, name)
        if isinstance(value, dict):
            for k, v in value.items():
                is_block = (isinstance(v, tuple) and len(v) == 2
                            and all(isinstance(x, int) for x in v))  # fmt: skip
                if is_block and v != SEED_RANGES["validation"]:  # umbrella range, see below
                    out[f"{name}.{k}"] = v
    return out


def proposal_conflicts() -> list[str]:
    """Overlaps between proposed blocks and registered ones, or between proposed blocks.
    (``SEED_RANGES.validation`` = [3.5M, 4M) is the umbrella range every R block lives in,
    so it is excluded from the overlap check and checked for containment instead.)"""
    problems = []
    blocks = registered_blocks()
    items = list(PROPOSED_R_SEED_BLOCKS.items())
    vlo, vhi = SEED_RANGES["validation"]
    for name, (a, b) in items:
        if not (vlo <= a < b <= vhi):
            problems.append(f"{name} is not inside the validation range")
        for other, (c, d) in blocks.items():
            if a < d and c < b:
                problems.append(f"{name} overlaps {other}")
    for i, (n1, (a, b)) in enumerate(items):
        for n2, (c, d) in items[i + 1 :]:
            if a < d and c < b:
                problems.append(f"{n1} overlaps {n2}")
    return problems


def training_episodes_needed(steps: int = TRAINING_STEPS) -> int:
    return -(-steps // EPISODE_STEPS)


def proposed_training_seeds() -> list[int]:
    """The in-order training seed list under the amendment (registered block, then
    training_2). For validation and tests only until approved."""
    a, b = _seeds.R_SEED_BLOCKS["training"]
    c, d = PROPOSED_R_SEED_BLOCKS["training_2"]
    return [*range(a, b), *range(c, d)]
