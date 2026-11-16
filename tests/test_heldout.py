"""Held-out confirmation runner: access control and pre-registered gatekeeping."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from farc.adaptation.experiment import AdaptationConfigError
from farc.contact import heldout
from farc.contact.data import check_seeds
from farc.contact.heldout import check_heldout_slice, derive_heldout, load_heldout

CONFIG = "configs/contact/heldout_confirmation.json"


def _write(tmp_path, **changes):
    values = json.loads(Path(CONFIG).read_text())
    values.update(changes)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(values))
    return path


def test_config_loads_with_the_approved_slice_only():
    spec = load_heldout(CONFIG)
    assert spec.contact.seeds == list(range(4_000_000, 4_000_300))
    assert load_heldout(CONFIG, debug=True).contact.seeds == list(range(3_655_000, 3_655_010))


def test_changed_plan_and_missing_approval_are_refused(tmp_path):
    plan = tmp_path / "plan.md"
    plan.write_text(Path("docs/HELDOUT_CONFIRMATION_PLAN.md").read_text() + "\nedit\n")
    with pytest.raises(AdaptationConfigError, match="plan changed"):
        load_heldout(_write(tmp_path, plan=str(plan)))
    with pytest.raises(AdaptationConfigError, match="approval"):
        load_heldout(_write(tmp_path, heldout_approved=" "))


@pytest.mark.parametrize("offset,episodes", [(4_000_001, 300), (3_999_999, 2), (4_000_300, 1),
                                             (4_500_000, 10)])  # fmt: skip
def test_only_the_approved_slice_is_accepted(tmp_path, offset, episodes):
    path = _write(tmp_path, evaluation_seed_offset=offset, evaluation_episodes=episodes)
    with pytest.raises(AdaptationConfigError):
        load_heldout(path)


def test_other_loaders_still_refuse_held_out_seeds():
    for role in ("evaluation", "exploratory", "heldout_debugging"):
        with pytest.raises(AdaptationConfigError):
            check_seeds([4_000_000], role)
    check_heldout_slice([4_000_000, 4_000_299])


def _episodes(effects: dict[str, float], seeds=60, base_rate=0.3):
    """Synthetic paired episodes: each configuration succeeds where the base does, plus a
    fixed extra share of seeds."""
    rng = np.random.default_rng(1)
    spec = load_heldout(CONFIG)
    rows = []
    for cond in [c["name"] for c in spec.contact.conditions]:
        delay = cond in spec.contact.delay_conditions
        for seed in range(seeds):
            base = bool(rng.random() < (base_rate if delay else 0.95))
            for name in heldout.CONFIGURATIONS:
                extra = effects.get(name, 0.0) if delay else 0.0
                rows.append({"configuration": name, "policy_seed": 0, "condition": cond,
                             "episode_seed": seed,
                             "success": base or seed < extra * seeds})  # fmt: skip
    return pd.DataFrame(rows), spec


@pytest.fixture
def no_accuracy(monkeypatch):
    acc = pd.DataFrame({"contact": [True, False], "horizon": [3, 3],
                        "static_object_position_error": [0.03, 0.001]})  # fmt: skip
    monkeypatch.setattr(heldout, "offline_accuracy", lambda steps, config: acc)
    monkeypatch.setattr(heldout, "closed_loop_prediction", lambda steps: pd.DataFrame())


def test_gate_closes_after_the_first_failure(no_accuracy):
    episodes, spec = _episodes({"phase_e_predictor": 0.0, "contact_learned": 0.5,
                                "oracle_sim_predictor": 0.5})  # fmt: skip
    tables, summary = derive_heldout(episodes, pd.DataFrame(), replace(
        spec, contact=replace(spec.contact, bootstrap_resamples=500)))  # fmt: skip
    status = summary["hypotheses"]
    assert status == {"H1": "not supported", "H2": "not tested (gate closed)",
                      "H3": "not tested (gate closed)"}  # fmt: skip


def test_all_supported_and_shrinkage_flag(no_accuracy):
    episodes, spec = _episodes({"phase_e_predictor": 0.5, "contact_learned": 0.7,
                                "learned_gripper_consistent": 0.5,
                                "oracle_sim_predictor": 0.7})  # fmt: skip
    tables, summary = derive_heldout(episodes, pd.DataFrame(), replace(
        spec, contact=replace(spec.contact, bootstrap_resamples=500)))  # fmt: skip
    assert set(summary["hypotheses"].values()) == {"supported"}
    gk = tables["gatekeeping"].set_index("hypothesis")
    assert not gk.loc["H1", "shrinkage_flag"]  # large effect
    assert summary["safety_gates_passed"]
