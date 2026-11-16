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


# --- Phase G2 (docs/PHASE_G2_PROPOSAL.md, section 4); no G2 seed has been used yet ---------

G2_USED_SEED_RANGES = {
    **G1_USED_SEED_RANGES,
    **{f"phase_g1_{role}": block for role, block in G1_SEED_BLOCKS.items()},
}
G2_SEED_BLOCKS = {
    "debugging": (3_670_000, 3_670_010),
    "mechanism": (3_671_000, 3_671_050),
    "calibration": (3_672_000, 3_672_100),
    "evaluation": (3_673_000, 3_673_100),
    "exploratory": (3_674_000, 3_674_100),  # post-hoc only, labelled as such
}
# Reserved for a future, separately approved held-out confirmation. No loader accepts it.
G2_FUTURE_HELDOUT = (4_000_300, 4_000_600)


def check_g2_seeds(seeds: list[int], role: str) -> None:
    """As ``check_g1_seeds``, for the G2 blocks; every seed >= 4,000,000 is refused."""
    if role not in G2_SEED_BLOCKS:
        raise AdaptationConfigError(f"unknown G2 seed role {role!r}")
    if not seeds:
        raise AdaptationConfigError(f"{role} needs at least one seed")
    held_low, _ = SEED_RANGES["heldout"]
    if any(s >= held_low for s in seeds):
        raise AdaptationConfigError("held-out seeds (4,000,000+) are reserved")
    low, high = SEED_RANGES["validation"]
    if not all(low <= s < high for s in seeds):
        raise AdaptationConfigError(f"{role} seeds must lie in the validation range")
    for name, (a, b) in G2_USED_SEED_RANGES.items():
        if any(a <= s < b for s in seeds):
            raise AdaptationConfigError(f"{role} seeds overlap {name} [{a}, {b})")
    a, b = G2_SEED_BLOCKS[role]
    if not all(a <= s < b for s in seeds):
        raise AdaptationConfigError(f"{role} seeds must lie in [{a}, {b})")
