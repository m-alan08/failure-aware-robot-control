"""Seed blocks reserved for Phase G1 (docs/PHASE_G1_PROPOSAL.md, section 2.5).

Every block lies in the validation range, below the held-out range, and is disjoint from every
seed block any earlier phase used. No G1 seed has been used yet.
"""

from __future__ import annotations

from farc.adaptation.experiment import AdaptationConfigError
from farc.contact.data import F1_SEED_BLOCKS, F1_USED_SEED_RANGES
from farc.disturbances.config import SEED_RANGES

G1_USED_SEED_RANGES = {
    **F1_USED_SEED_RANGES,
    **{f"phase_f1_{role}": block for role, block in F1_SEED_BLOCKS.items()},
}
G1_SEED_BLOCKS = {
    "debugging": (3_660_000, 3_660_010),
    "calibration": (3_661_000, 3_661_050),
    "evaluation": (3_662_000, 3_662_100),
    "exploratory": (3_663_000, 3_663_100),  # post-hoc only, labelled as such
}


def check_g1_seeds(seeds: list[int], role: str) -> None:
    """Refuse held-out seeds, seeds outside the validation range, seeds any earlier phase
    used, and seeds outside the block reserved for ``role``."""
    if role not in G1_SEED_BLOCKS:
        raise AdaptationConfigError(f"unknown G1 seed role {role!r}")
    if not seeds:
        raise AdaptationConfigError(f"{role} needs at least one seed")
    held_low, held_high = SEED_RANGES["heldout"]
    if any(s >= held_low for s in seeds):
        raise AdaptationConfigError("held-out seeds (4,000,000+) are reserved")
    low, high = SEED_RANGES["validation"]
    if not all(low <= s < high for s in seeds):
        raise AdaptationConfigError(f"{role} seeds must lie in the validation range")
    for name, (a, b) in G1_USED_SEED_RANGES.items():
        if any(a <= s < b for s in seeds):
            raise AdaptationConfigError(f"{role} seeds overlap {name} [{a}, {b})")
    a, b = G1_SEED_BLOCKS[role]
    if not all(a <= s < b for s in seeds):
        raise AdaptationConfigError(f"{role} seeds must lie in [{a}, {b})")
