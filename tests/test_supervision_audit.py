"""Phase G1 draft: seed guards, and the existing gate behaviour the G1 audit relies on."""

from __future__ import annotations

import itertools

import pytest

from farc.adaptation.compensation import LatchedEstimate
from farc.adaptation.estimator import Estimate
from farc.adaptation.experiment import AdaptationConfigError
from farc.contact.data import F1_SEED_BLOCKS
from farc.disturbances.config import SEED_RANGES
from farc.supervision.seeds import G1_SEED_BLOCKS, G1_USED_SEED_RANGES, check_g1_seeds


def test_g1_blocks_are_fresh_and_below_heldout():
    low, high = SEED_RANGES["validation"]
    used = list(G1_USED_SEED_RANGES.values())
    assert set(F1_SEED_BLOCKS.values()) <= set(used)  # every F1 block counts as used
    for a, b in G1_SEED_BLOCKS.values():
        assert low <= a < b <= high <= SEED_RANGES["heldout"][0]
        assert all(b <= ua or a >= ub for ua, ub in used)
    for (a, b), (c, d) in itertools.combinations(G1_SEED_BLOCKS.values(), 2):
        assert b <= c or d <= a


@pytest.mark.parametrize("role", list(G1_SEED_BLOCKS))
def test_each_block_is_accepted_only_for_its_role(role):
    a, b = G1_SEED_BLOCKS[role]
    check_g1_seeds(list(range(a, b)), role)
    for other in set(G1_SEED_BLOCKS) - {role}:
        with pytest.raises(AdaptationConfigError):
            check_g1_seeds([a], other)


@pytest.mark.parametrize("seed", [4_000_000, 4_000_299, 4_000_300, 4_999_999, 5_000_000])
def test_heldout_seeds_are_refused(seed):
    for role in G1_SEED_BLOCKS:
        with pytest.raises(AdaptationConfigError, match="held-out"):
            check_g1_seeds([G1_SEED_BLOCKS[role][0], seed], role)


@pytest.mark.parametrize("seed", [3_653_000, 3_654_050, 3_641_000, 3_610_000, 3_499_999])
def test_earlier_phase_seeds_are_refused(seed):
    with pytest.raises(AdaptationConfigError):
        check_g1_seeds([seed], "evaluation")


def test_empty_and_unknown_roles_are_refused():
    with pytest.raises(AdaptationConfigError):
        check_g1_seeds([], "evaluation")
    with pytest.raises(AdaptationConfigError):
        check_g1_seeds([3_662_000], "heldout")


def _estimate(delay, reliable):
    return Estimate(step=0, transitions=10, delay=delay, gain=1.0, r2=0.9, margin=3.0,
                    excitation=1.0, gain_reliable=False, delay_reliable=reliable)  # fmt: skip


def test_existing_latch_releases_only_on_a_reliable_zero():
    """Audit claim (proposal 1.2): the current compensator is already gated. It engages at the
    first reliable d > 0, holds through any number of abstentions, and releases only when a
    reliable d = 0 arrives. Unreliable evidence for 'no delay' never releases it."""
    latch = LatchedEstimate()
    assert latch.parameters(_estimate(3, False)).delay is None  # abstains until reliable
    assert latch.parameters(_estimate(3, True)).delay == 3  # engages
    for _ in range(40):  # unreliable zeros (e.g. low excitation late in an episode)
        assert latch.parameters(_estimate(0, False)).delay == 3
    assert latch.parameters(_estimate(None, False)).delay == 3
    assert latch.parameters(_estimate(0, True)).delay == 0  # releases on a reliable zero
    assert latch.parameters(_estimate(2, True)).delay == 2  # re-engages on fresh evidence
