"""Phase J-1: seed guard, cells and the pinned protocol."""

from __future__ import annotations

import pytest

from farc.adaptation.experiment import AdaptationConfigError
from farc.retrain.ablation_j1 import CONFIGURATIONS, PRIVILEGED, j1_cells, load_j1
from farc.supervision.seeds import check_j_seeds


def test_config_pins_protocol_and_fresh_seed_blocks():
    c = load_j1("configs/retrain/phase_j1.json")
    assert c.seeds("debugging") == list(range(3_700_200, 3_700_210))
    assert c.seeds("evaluation") == list(range(3_703_000, 3_703_300))
    for bad in ([4_000_600], [3_702_000], [3_700_100], [3_703_300], [4_000_000]):
        with pytest.raises(AdaptationConfigError):
            check_j_seeds(bad, "evaluation")


def test_cells_and_configurations_match_the_protocol():
    names = [c.name for c in j1_cells()]
    assert names == ["H_noise0.01", "E_C8_noise0.01", "J_noise0.005", "H_undisturbed",
                     "E_C0_undisturbed"]  # fmt: skip
    assert set(PRIVILEGED) <= set(CONFIGURATIONS) and "C" not in CONFIGURATIONS
    assert [c.extra[0].severity for c in j1_cells()[:3]] == [0.01, 0.01, 0.005]
