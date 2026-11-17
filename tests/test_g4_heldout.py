"""Phase G4: the held-out guard, cells, runner equivalence and verdict logic."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from farc.adaptation.experiment import AdaptationConfigError
from farc.delaychange.heldout import ARM_F, ARM_R, g4_cells, load_g4
from farc.delaychange.heldout_analysis import derive_g4, primary
from farc.supervision.seeds import check_g4_seeds

CONFIG = "configs/delaychange/g4_heldout.json"


def test_guard_admits_only_the_approved_heldout_slice():
    config = load_g4(CONFIG)
    assert config.seeds == list(range(4_000_300, 4_000_600))
    assert load_g4(CONFIG, "debugging").seeds == list(range(3_690_000, 3_690_010))
    check_g4_seeds([4_000_300, 4_000_599], "confirmation")
    for bad in ([4_000_299], [4_000_000], [4_000_600], [4_100_000], [3_681_000]):
        with pytest.raises(AdaptationConfigError):
            check_g4_seeds(bad, "confirmation")
    for bad in ([4_000_300], [3_681_500], [3_682_000], [3_674_000], [3_690_010]):
        with pytest.raises(AdaptationConfigError):
            check_g4_seeds(bad, "debugging")
    with pytest.raises(AdaptationConfigError):
        check_g4_seeds([3_690_000], "evaluation")


def test_cells_and_arms():
    cells = g4_cells()
    r = [c for c in cells if c.arm == "R"]
    f = [c for c in cells if c.arm == "F"]
    assert len(r) == 9 and len(f) == 32
    assert {c.family for c in f} == {"F-P", "F-T", "F-I", "NG"}
    assert len({c.name for c in cells}) == len(cells)
    assert sum(len(ARM_R) for _ in r) + sum(len(ARM_F) for _ in f) == 173  # x 900 = 155,700
    rt = next(c for c in r if c.name == "RT_d2")
    assert rt.schedule(4_000_300) != rt.schedule(4_000_301)  # random per episode seed
    ft = next(c for c in f if c.family == "F-T")
    assert ft.schedule(4_000_300) == ft.schedule(4_000_301)  # fixed grid


def _episodes(eff: dict[tuple[str, str], float], n=150, seed=0):
    rng = np.random.default_rng(seed)
    fams = {"RT": "R", "RI": "R", "RS": "R", "RP": "R", "F-P": "F", "F-T": "F", "F-I": "F",
            "NG": "F"}  # fmt: skip
    rows = []
    for f, arm in fams.items():
        names = ARM_R if arm == "R" else ARM_F
        for s in range(n):
            u = rng.random()
            for name in names:
                rows.append({"arm": arm, "configuration": name, "policy_seed": 0,
                             "condition": f + "_x", "family": f, "episode_seed": s,
                             "success": int(u < 0.5 + eff.get((f, name), 0.0))})  # fmt: skip
    for cond in ("H_undisturbed", "H_noise0.01"):
        for s in range(n):
            for name in ARM_R:
                rows.append({"arm": "R", "configuration": name, "policy_seed": 0,
                             "condition": cond, "family": "H", "episode_seed": s,
                             "success": 1})  # fmt: skip
    return pd.DataFrame(rows)


def test_verdicts():
    c = load_g4(CONFIG)
    good = {(f, "C"): 0.3 for f in ("RT", "RI")}
    tab, s = primary(_episodes(good), c)
    assert s["step1_passed"] and s["P1_label"] == "confirmed and practically meaningful"
    assert s["step2_passed"] and s["P3_label"] == "meaningful"
    weak = {**good, ("RP", "C"): -0.3, ("F-P", "C"): -0.3}  # persistent cost: step 1 fails
    _, s = primary(_episodes(weak), c)
    assert not s["step1_passed"] and s["P3_label"] == "not tested (step 1 failed)"
    yoked = {**good, **{(f, "sham_C_cross_yoked"): 0.3 for f in ("RT", "RI")}}
    _, s = primary(_episodes(yoked), c)
    assert s["step1_passed"] and not s["step2_passed"]


def test_gate_failure_blocks_confirmation():
    c = load_g4(CONFIG)
    ep = _episodes({(f, "C"): 0.3 for f in ("RT", "RI")})
    ep.loc[(ep["condition"] == "H_noise0.01") & (ep["configuration"] == "C")
           & (ep["episode_seed"] < 20), "success"] = 0  # fmt: skip
    steps = ep[["arm", "configuration", "policy_seed", "condition", "family", "episode_seed"]]
    steps = pd.concat([steps.assign(t=t, delay_true=d) for t, d in ((0, 0), (1, 2))])
    steps = steps.assign(delay_used=0, active=False, alarm=False, suspended=False, ctrl_ns=1,
                         learned_calls=0).sort_values([*steps.columns[:6], "t"])  # fmt: skip
    tables, summary = derive_g4(ep, steps, c)
    assert summary["step1_passed"] and not summary["gates_passed"]
    assert summary["verdict"] == "C not confirmed"


# --- simulator --------------------------------------------------------------------------------

from farc.paths import checkpoint_path, resolve  # noqa: E402

CHECKPOINT = checkpoint_path("fetch_push_fp32_seed1")
MODEL = resolve("$FARC_DATA_DIR/models/contact_predictor/model.pt")
needs_sim = pytest.mark.skipif(not (CHECKPOINT.exists() and MODEL.exists()),
                               reason="checkpoint or contact model absent")  # fmt: skip


@needs_sim
def test_g4_runner_matches_the_g2_runner():
    from dataclasses import replace

    from farc.delaychange.heldout import equivalence_check

    c = replace(load_g4(CONFIG, "debugging"), debug_episodes=2)
    for cell in [g4_cells()[2], next(x for x in g4_cells() if x.family == "F-I")]:
        for r in equivalence_check((c, str(CHECKPOINT), cell)):
            assert r["C_identical"] and r["contact_identical"] and r["sham_identical"], r
