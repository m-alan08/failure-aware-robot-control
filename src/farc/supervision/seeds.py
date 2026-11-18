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


# --- Phase G3 (docs/PHASE_G3_PROPOSAL.md, section 4.1); every G2 block is now used ------------

G3_USED_SEED_RANGES = {
    **G2_USED_SEED_RANGES,
    **{f"phase_g2_{role}": block for role, block in G2_SEED_BLOCKS.items()},
}
G3_SEED_BLOCKS = {
    "debugging": (3_680_000, 3_680_010),
    "evaluation": (3_681_000, 3_682_000),
    "exploratory": (3_682_000, 3_682_100),  # post-hoc only, labelled as such
}


def check_g3_seeds(seeds: list[int], role: str) -> None:
    """As ``check_g2_seeds``, for the G3 blocks; every seed >= 4,000,000 is refused."""
    if role not in G3_SEED_BLOCKS:
        raise AdaptationConfigError(f"unknown G3 seed role {role!r}")
    if not seeds:
        raise AdaptationConfigError(f"{role} needs at least one seed")
    held_low, _ = SEED_RANGES["heldout"]
    if any(s >= held_low for s in seeds):
        raise AdaptationConfigError("held-out seeds (4,000,000+) are reserved")
    low, high = SEED_RANGES["validation"]
    if not all(low <= s < high for s in seeds):
        raise AdaptationConfigError(f"{role} seeds must lie in the validation range")
    for name, (a, b) in G3_USED_SEED_RANGES.items():
        if any(a <= s < b for s in seeds):
            raise AdaptationConfigError(f"{role} seeds overlap {name} [{a}, {b})")
    a, b = G3_SEED_BLOCKS[role]
    if not all(a <= s < b for s in seeds):
        raise AdaptationConfigError(f"{role} seeds must lie in [{a}, {b})")


# --- Phase G4 (docs/PHASE_G4_HELDOUT_PROPOSAL.md, section 2) -----------------------------------

G4_USED_SEED_RANGES = {
    **G3_USED_SEED_RANGES,
    **{f"phase_g3_{role}": block for role, block in G3_SEED_BLOCKS.items()},
}
G4_DEBUGGING = (3_690_000, 3_690_010)
# The only held-out slice G4 may use. 4,000,000-4,000,299 were used by the contact-model
# confirmation; nothing at or above 4,000,600 is approved.
G4_CONFIRMATION = (4_000_300, 4_000_600)


def check_g4_seeds(seeds: list[int], role: str) -> None:
    """``debugging``: the fresh validation block only. ``confirmation``: exactly the approved
    held-out slice [4,000,300, 4,000,600). This is the only G-phase guard that admits held-out
    seeds."""
    if not seeds:
        raise AdaptationConfigError(f"{role} needs at least one seed")
    if role == "debugging":
        held_low, _ = SEED_RANGES["heldout"]
        if any(s >= held_low for s in seeds):
            raise AdaptationConfigError("held-out seeds (4,000,000+) are reserved")
        for name, (a, b) in G4_USED_SEED_RANGES.items():
            if any(a <= s < b for s in seeds):
                raise AdaptationConfigError(f"debugging seeds overlap {name} [{a}, {b})")
        a, b = G4_DEBUGGING
    elif role == "confirmation":
        a, b = G4_CONFIRMATION
    else:
        raise AdaptationConfigError(f"unknown G4 seed role {role!r}")
    if not all(a <= s < b for s in seeds):
        raise AdaptationConfigError(f"{role} seeds must lie in [{a}, {b})")


# --- Phase H (docs/PHASE_H_RETRAINED_BASELINE_PROPOSAL.md, section 3.3) -----------------------

H_USED_SEED_RANGES = {
    **G4_USED_SEED_RANGES,
    "phase_g4_debugging": G4_DEBUGGING,
}
H_SEED_BLOCKS = {
    "debugging": (3_700_000, 3_700_010),
    "evaluation": (3_701_000, 3_701_300),
}


def check_h_seeds(seeds: list[int], role: str) -> None:
    """Development blocks only; every seed >= 4,000,000 is refused (a Phase H held-out stage
    would need its own approved guard)."""
    if role not in H_SEED_BLOCKS:
        raise AdaptationConfigError(f"unknown Phase H seed role {role!r}")
    if not seeds:
        raise AdaptationConfigError(f"{role} needs at least one seed")
    held_low, _ = SEED_RANGES["heldout"]
    if any(s >= held_low for s in seeds):
        raise AdaptationConfigError("held-out seeds (4,000,000+) are reserved")
    for name, (a, b) in H_USED_SEED_RANGES.items():
        if any(a <= s < b for s in seeds):
            raise AdaptationConfigError(f"{role} seeds overlap {name} [{a}, {b})")
    a, b = H_SEED_BLOCKS[role]
    if not all(a <= s < b for s in seeds):
        raise AdaptationConfigError(f"{role} seeds must lie in [{a}, {b})")


# --- Phase I (docs/PHASE_I_STRONG_BASELINE_PROPOSAL.md, section 8) -----------------------------

I_USED_SEED_RANGES = {
    **H_USED_SEED_RANGES,
    **{f"phase_h_{role}": block for role, block in H_SEED_BLOCKS.items()},
}
I_SEED_BLOCKS = {
    "debugging": (3_700_100, 3_700_110),
    "evaluation": (3_702_000, 3_702_300),
}


def check_i_seeds(seeds: list[int], role: str) -> None:
    """Development blocks only; every seed >= 4,000,000 is refused."""
    if role not in I_SEED_BLOCKS:
        raise AdaptationConfigError(f"unknown Phase I seed role {role!r}")
    if not seeds:
        raise AdaptationConfigError(f"{role} needs at least one seed")
    held_low, _ = SEED_RANGES["heldout"]
    if any(s >= held_low for s in seeds):
        raise AdaptationConfigError("held-out seeds (4,000,000+) are reserved")
    for name, (a, b) in I_USED_SEED_RANGES.items():
        if any(a <= s < b for s in seeds):
            raise AdaptationConfigError(f"{role} seeds overlap {name} [{a}, {b})")
    a, b = I_SEED_BLOCKS[role]
    if not all(a <= s < b for s in seeds):
        raise AdaptationConfigError(f"{role} seeds must lie in [{a}, {b})")
