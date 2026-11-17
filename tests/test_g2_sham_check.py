"""G2 exploratory sham check (docs/PHASE_G2_SHAM_CHECK.md)."""

import numpy as np
import pandas as pd

from farc.delaychange.sham_check import (
    check_cells,
    derive_check,
    exploratory_seeds,
    random_rotation,
)
from farc.delaychange.stage3 import load_g2


def test_seeds_are_the_exploratory_block_and_below_heldout():
    seeds = exploratory_seeds()
    assert seeds[0] == 3_674_000 and seeds[-1] == 3_674_099 and len(seeds) == 100
    assert max(seeds) < 4_000_000


def test_cells_are_the_stage3_t_and_i_cells():
    cells = check_cells()
    assert len(cells) == 22 and {c.family for c in cells} == {"T", "I"}


def test_rotation_keeps_count_and_runs_and_moves_the_mask():
    mask = np.zeros(50, bool)
    mask[10:14] = mask[30:32] = True
    moved = 0
    for seed in range(200):
        r = random_rotation(mask, 0, 3, seed)
        assert r.sum() == mask.sum()
        assert any(np.array_equal(np.roll(r, k), mask) for k in range(len(mask)))  # a rotation
        moved += not np.array_equal(r, mask)
    assert moved == 200  # an offset in 1..n-1 never returns the same non-periodic mask
    a = random_rotation(mask, 1, 3, 7)
    assert np.array_equal(a, random_rotation(mask, 1, 3, 7))  # deterministic


def test_derive_check_reports_e1_to_e5():
    rng = np.random.default_rng(0)
    rows = []
    for name in ("contact_learned", "cand_C_sequential", "sham_C_yoked", "sham_C_random"):
        for fam, cond in (("T", "T_x"), ("I", "I_x")):
            for seed in range(40):
                rows.append(
                    {
                        "configuration": name,
                        "policy_seed": 0,
                        "condition": cond,
                        "family": fam,
                        "episode_seed": seed,
                        "success": int(
                            rng.random() < (0.9 if name == "cand_C_sequential" else 0.6)
                        ),
                        "final_distance": 0.0,
                        "suspended_steps": 2,
                    }
                )
    episodes = pd.DataFrame(rows)
    steps = episodes[
        ["configuration", "policy_seed", "condition", "family", "episode_seed"]
    ].assign(t=0, delay_true=2, delay_used=2, active=True, suspended=False)
    config = load_g2("configs/delaychange/g2_stage3.json")
    tables, summary = derive_check(episodes, steps, config)
    assert set(tables["comparisons"]["test"]) == {"E1", "E2", "E3", "E4", "E5"}
    assert summary["E1_estimate"] > 0.2 and summary["E1_result"] == "timing matters"
