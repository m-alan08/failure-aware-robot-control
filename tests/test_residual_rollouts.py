"""GEN-3 (amendment R-A2): the PRIVILEGED P configuration runs only on D cells, uses the true
delay, and is written apart from the analysis table. Real simulator, smoke seeds only."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import residual_rollouts as rr  # noqa: E402

from farc.residual.training import RunRefused, load_plan_config  # noqa: E402

PLAN = load_plan_config()


def test_privileged_p_only_on_delay_cells_and_uses_true_delay():
    d_cell = next(c.name for c in rr.development_cells() if c.group == "D")
    args = ("smoke", ["nominal", d_cell, "TO_1cm"], 0, [0], {"C": [-1], "B0": [-1]}, None,
            Path("unused"), PLAN, True)  # fmt: skip
    rows, p_rows = rr._task(args)
    assert {r["cell"] for r in rows} == {"nominal", d_cell, "TO_1cm"}
    assert {r["config"] for r in rows} == {"C", "B0"}
    assert p_rows and {r["cell"] for r in p_rows} == {d_cell}
    assert all(r["config"] == "P" and r["group"] == "D" for r in p_rows)
    assert all(r["true_delay_used"] for r in p_rows)


def test_registered_roles_refused_before_approval_and_seed_sets_fixed():
    if PLAN.get("approved") is not True:
        with pytest.raises(RunRefused):
            rr.plan_tasks("selection", None, Path("x"), PLAN, None)
    seeds = rr.evaluation_seeds()
    assert len(seeds) == 600 and seeds[0] == 3_722_000 and seeds[-1] == 3_740_399
    assert len(set(seeds)) == 600 and max(seeds) < 4_000_000


def test_evaluation_refuses_anything_but_the_fixed_600(monkeypatch):
    from farc.residual import episodes

    approved = dict(PLAN, approved=True)
    monkeypatch.setattr(episodes, "check_approved", lambda plan: None)
    seeds = rr.evaluation_seeds()
    episodes.check_role_seeds("evaluation", seeds, approved)
    for bad in (seeds[:300], seeds[300:], seeds[:-1], [*seeds[:-1], 4_000_600]):
        with pytest.raises((RunRefused, ValueError)):
            episodes.check_role_seeds("evaluation", bad, approved)
    a = list(range(3_721_000, 3_721_100))
    episodes.check_role_seeds("selection", a, approved)
    with pytest.raises(RunRefused):
        episodes.check_role_seeds("selection", a[:50], approved)
