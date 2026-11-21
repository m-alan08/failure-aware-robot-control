"""T-M amendment check (protocol review, not an experiment).

On UNREGISTERED smoke seeds only (default 0-19), with the frozen ``fetch_push_fp32_seed0``
policy:
1. record the policy's closed-loop command sequence on the nominal model;
2. replay that exact sequence open loop under each T-M setting, for both friction modes;
3. report the object-trajectory change vs nominal, whether the nominal replay is bitwise
   identical, and which geoms the object touches (contact-pair tally).

Outputs ``results/residual_protocol_review/tm_effect_check.json``. Software/protocol evidence
only: no residual, no training, no registered seed.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from farc.delaychange.heldout import _setup, load_g4
from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
from farc.disturbances.config import ConditionSpec
from farc.residual.disturbances import ObjectDynamics, ObjectDynamicsPatch
from farc.residual.training import check_smoke_seeds

SETTINGS = {
    "mass x0.5": ObjectDynamics(0.5, 1.0),
    "mass x2.0": ObjectDynamics(2.0, 1.0),
    "friction x0.5": ObjectDynamics(1.0, 0.5),
    "friction x1.5": ObjectDynamics(1.0, 1.5),
    "nominal (1, 1)": ObjectDynamics(1.0, 1.0),
}


def _session(env, policy, normalizer, seed):
    return ScheduledDelaySession(
        env, policy, normalizer, ConditionSpec("n", ()), seed, 0, DelaySchedule(()), "replay_drop"
    )


def _rollout(env, policy, normalizer, seed, commands=None, dyn=None, mode="object_geom"):
    s = _session(env, policy, normalizer, seed)
    raw = s.raw
    obj = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_GEOM, "object0")
    patch = ObjectDynamicsPatch(raw.model, dyn, friction_mode=mode) if dyn else None
    if patch:
        patch.__enter__()
    block: list[np.ndarray] = []
    cmds: list[np.ndarray] = []
    contacts: Counter[str] = Counter()
    try:
        while not s.done:
            v = s.observe()
            c = v.base_action if commands is None else commands[s.step]
            cmds.append(np.asarray(c).copy())
            s.act(c)
            block.append(np.asarray(s.observation["observation"][3:6]).copy())  # true state
            for i in range(raw.data.ncon):
                g1, g2 = raw.data.contact[i].geom1, raw.data.contact[i].geom2
                if obj in (g1, g2):
                    other = g2 if g1 == obj else g1
                    body = raw.model.geom_bodyid[other]
                    contacts[mujoco.mj_id2name(raw.model, mujoco.mjtObj.mjOBJ_BODY, body)] += 1
    finally:
        if patch:
            patch.__exit__(None, None, None)
    return np.array(block), np.array(cmds), s.final_distance, contacts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs=2, default=(0, 20))
    ap.add_argument("--out", default="results/residual_protocol_review")
    a = ap.parse_args()
    seeds = list(range(*a.seeds))
    check_smoke_seeds(seeds)
    g4 = load_g4("configs/delaychange/g4_heldout.json", "debugging")  # loads constants only
    _, policy, normalizer, env, _, _ = _setup(g4, g4.checkpoints[0])
    contact_tally: Counter = Counter()
    rows: dict[tuple[str, str], list[dict[str, float]]] = {
        (m, k): [] for m in ("object_geom", "object_table") for k in SETTINGS
    }
    identical = {(m, k): 0 for m in ("object_geom", "object_table") for k in SETTINGS}
    for seed in seeds:
        ref, cmds, ref_dist, contacts = _rollout(env, policy, normalizer, seed)
        contact_tally.update(contacts)
        for mode in ("object_geom", "object_table"):
            for name, dyn in SETTINGS.items():
                got, _, dist, _ = _rollout(env, policy, normalizer, seed, cmds, dyn, mode)
                identical[(mode, name)] += int(np.array_equal(got, ref))
                rows[(mode, name)].append(
                    {
                        "max_block_dev_m": float(np.abs(got - ref).max()),
                        "final_dist_change_m": float(dist - ref_dist),
                    }
                )
    settings: dict[str, dict[str, float]] = {}
    summary = {
        "seeds": [seeds[0], seeds[-1]],
        "policy": "fetch_push_fp32_seed0 (frozen), nominal commands replayed open loop",
        "object_contact_steps_by_body": dict(contact_tally.most_common()),
        "settings": settings,
        "note": "protocol check on unregistered seeds; not a result",
    }
    for (mode, name), vals in rows.items():
        dev = np.array([v["max_block_dev_m"] for v in vals])
        settings[f"{mode} | {name}"] = {
            "episodes_bitwise_identical_to_nominal": identical[(mode, name)],
            "episodes": len(vals),
            "median_max_block_dev_m": float(np.median(dev)),
            "share_episodes_dev_gt_1mm": float(np.mean(dev > 1e-3)),
        }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tm_effect_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
