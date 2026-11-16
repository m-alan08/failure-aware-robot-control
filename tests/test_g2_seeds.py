"""Phase G2 draft: seed blocks are fresh, role-bound and never held-out."""

from __future__ import annotations

import itertools

import pytest

from farc.adaptation.experiment import AdaptationConfigError
from farc.disturbances.config import SEED_RANGES
from farc.supervision.seeds import (
    G1_SEED_BLOCKS,
    G2_FUTURE_HELDOUT,
    G2_SEED_BLOCKS,
    G2_USED_SEED_RANGES,
    check_g2_seeds,
)


def test_g2_blocks_are_fresh_disjoint_and_below_heldout():
    used = list(G2_USED_SEED_RANGES.values())
    assert set(G1_SEED_BLOCKS.values()) <= set(used)  # every G1 block counts as used
    low, high = SEED_RANGES["validation"]
    for a, b in G2_SEED_BLOCKS.values():
        assert low <= a < b <= high <= SEED_RANGES["heldout"][0]
        assert all(b <= ua or a >= ub for ua, ub in used)
    for (a, b), (c, d) in itertools.combinations(G2_SEED_BLOCKS.values(), 2):
        assert b <= c or d <= a


@pytest.mark.parametrize("role", list(G2_SEED_BLOCKS))
def test_each_block_only_for_its_role(role):
    a, b = G2_SEED_BLOCKS[role]
    check_g2_seeds(list(range(a, b)), role)
    for other in set(G2_SEED_BLOCKS) - {role}:
        with pytest.raises(AdaptationConfigError):
            check_g2_seeds([a], other)


@pytest.mark.parametrize("seed", [4_000_000, *G2_FUTURE_HELDOUT, 4_000_599, 4_999_999])
def test_heldout_and_future_confirmation_seeds_are_refused(seed):
    for role in G2_SEED_BLOCKS:
        with pytest.raises(AdaptationConfigError, match="held-out"):
            check_g2_seeds([G2_SEED_BLOCKS[role][0], seed], role)


@pytest.mark.parametrize("seed", [3_662_050, 3_663_000, 3_661_010, 3_653_000, 3_641_000])
def test_earlier_phase_seeds_are_refused(seed):
    with pytest.raises(AdaptationConfigError):
        check_g2_seeds([seed], "evaluation")
