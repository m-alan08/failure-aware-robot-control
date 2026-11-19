"""Stage H0 offline replay equivalence (docs/HARDWARE_H0_RESULTS.md). No simulator stepping.

Input: archived Phase F1 evaluation step records (development seeds 3,653,000-099, 3 frozen SAC
policies, 9 conditions, configurations base_only / phase_e_predictor / contact_learned). The
desired goal is reconstructed by ``env.reset(seed)`` and checked against the recorded t = 0
observation (``farc.hardware.replay``).

Part A (closed-loop consistency): the hardware controller replays each recorded episode of the
configuration that produced it (base_only -> F, phase_e_predictor -> K, contact_learned -> CA)
and must reproduce the recorded issued command, latched delay and gain, activity and predicted
positions at every step. Two input paths:
- ``direct``: the recorded 25-dim view goes straight to the controller;
- ``adapter``: the view is converted to a raw sensor state (``raw_from_sim_observation``) and
  back through the observation adapter, then through the full pipeline (adapter, controller,
  FIFO).

Part B (open-loop, against the original loops): every recorded base_only trajectory is served
by ``ReplaySession`` to the original simulation loops and, in parallel, to the hardware
controller, for all four configurations (F, CA, C through ``mechanism.run_episode``; K through
``stage3.run_g2_episode``). Commands, delay used, activity, alarms, suspension, estimator
outputs, CUSUM statistics, reset start and predictions must match.

    uv run python scripts/hardware_h0_replay.py --out results/hardware_h0 [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.dynamics import load_dynamics
from farc.contact.episodes import learned_predictor, phase_e_predictor
from farc.delaychange.mechanism import run_episode, sequential
from farc.delaychange.schedule import ScheduledDelaySession
from farc.delaychange.stage3 import Latched, run_g2_episode
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.hardware.config import replay_config
from farc.hardware.controller import C_CONSTANTS, ESTIMATOR, HardwareController
from farc.hardware.fifo import CommandFIFO
from farc.hardware.observation import ObservationAdapter, raw_from_sim_observation
from farc.hardware.pipeline import HardwarePipeline
from farc.hardware.replay import Episode, ReplaySession, load_f1, reconstruct_goal
from farc.paths import resolve

MODEL = "$FARC_DATA_DIR/models/contact_predictor/model.pt"
MODEL_SHA = "8a960329371c7b31d81ae2e2dc16998fde75d4b4feeeb59b624900127befe363"
RECORDED = {"base_only": "F", "phase_e_predictor": "K", "contact_learned": "CA"}
PERIOD_NS = 40_000_000


def setup(seed: int):
    torch.set_num_threads(1)
    loaded = load_policy(
        resolve(f"$FARC_DATA_DIR/checkpoints/fetch_push_fp32_seed{seed}/checkpoint.pt")
    )
    policy = deterministic_policy_function(loaded.actor)
    normalizer = FrozenNormalizer(loaded.normalizer)
    env = make_env(loaded.env_config)
    model = load_dynamics(resolve(MODEL), MODEL_SHA)
    preds = {"learned": learned_predictor(model, move_object=True),
             "phase_e": phase_e_predictor(ESTIMATOR["nominal_gain"])}  # fmt: skip
    return loaded, policy, normalizer, env, preds


def _diff(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


# --- Part A ------------------------------------------------------------------------------------


def closed_loop(ep: Episode, name: str, policy, normalizer, preds, path: str) -> dict[str, Any]:
    ctrl = HardwareController(name, policy, normalizer, preds)
    pipe = None
    if path == "adapter":
        pipe = HardwarePipeline(ObservationAdapter(replay_config()), ctrl, CommandFIFO())
        pipe.reset()
    assert ep.goal is not None
    out: dict[str, Any] = {"cmd_max_abs": 0.0, "cmd_exact_steps": 0, "pred_max_abs": 0.0,
           "first_divergence": -1,
           "delay_mismatch": 0, "gain_max_abs": 0.0, "active_mismatch": 0,
           "adapter_obs_max_abs": 0.0, "steps": len(ep.view)}  # fmt: skip
    for t in range(len(ep.view)):
        if pipe is None:
            rec = ctrl.step(ep.view[t], ep.goal).record
        else:
            state = raw_from_sim_observation(ep.view[t], (t + 1) * PERIOD_NS, PERIOD_NS)
            obs = ObservationAdapter(replay_config()).observation(state)
            out["adapter_obs_max_abs"] = max(out["adapter_obs_max_abs"], _diff(obs, ep.view[t]))
            rec = pipe.cycle(state, ep.goal).record
            if rec["safety_event"] is not None:
                raise RuntimeError(f"unexpected safety event {rec['safety_event']}")
        d = _diff(rec["command"][:3], ep.cmd[t])
        out["cmd_max_abs"] = max(out["cmd_max_abs"], d)
        out["cmd_exact_steps"] += int(d == 0.0)
        if d != 0.0 and out["first_divergence"] < 0:
            out["first_divergence"] = t
        out["delay_mismatch"] += int(rec["delay_used"] != ep.delay_param[t])
        out["gain_max_abs"] = max(out["gain_max_abs"], abs(rec["gain_used"] - ep.gain_param[t]))
        active = rec["active"] if name != "F" else False
        out["active_mismatch"] += int(bool(active) != bool(ep.active[t]))
        if ep.active[t] and rec["pred_gripper_position"] is not None:
            pred = rec["pred_gripper_position"] + rec["pred_object_position"]
            out["pred_max_abs"] = max(out["pred_max_abs"], _diff(pred, ep.pred[t]))
    return out


# --- Part B ------------------------------------------------------------------------------------


class Recorder:
    """Wraps an original controller object; delegates everything, records each decision."""

    def __init__(self, inner):
        self.inner, self.log = inner, []

    def step(self, est):
        dec = self.inner.step(est)
        stats = getattr(self.inner, "stats", None)
        self.log.append((dec.estimate.delay, bool(dec.estimate.delay_reliable),
                         None if stats is None else [float(stats.get(k, 0.0)) for k in range(5)],
                         getattr(self.inner, "sigma2", None)))  # fmt: skip
        return dec

    @property
    def gain(self):
        return self.inner.gain

    def __getattr__(self, item):
        return getattr(self.inner, item)


def reference(ep: Episode, name: str, policy, normalizer, preds) -> dict[str, Any]:
    calls: list[np.ndarray] = []

    def recording(fn):
        def wrapped(o, q, g):
            r = fn(o, q, g)
            calls.append(np.asarray(r, np.float64).copy())
            return r

        return wrapped

    rpreds = {k: recording(v) for k, v in preds.items()}
    assert ep.goal is not None
    replay = ReplaySession(ep.view, ep.goal, policy, normalizer)
    # ReplaySession provides the members the original loops use (observe, act, commands,
    # schedule, policy, normalizer, done, condition); the loops are typed for the simulator
    # session, so the stub is presented under that type at this boundary only.
    session = cast(ScheduledDelaySession, replay)
    est = ActuatorEstimator(EstimatorConfig(**ESTIMATOR))
    rec_ctrl = None
    if name == "K":
        rows = run_g2_episode(session, est, "phase_e_predictor", rpreds, C_CONSTANTS)["rows"]
        dec = [(r["est_delay"], bool(r["est_reliable"]), None, None) for r in rows]
    else:
        rec_ctrl = Recorder(sequential(C_CONSTANTS) if name == "C" else Latched())
        mode = "base" if name == "F" else "normal"  # "normal" is run_episode's default
        rows = run_episode(session, est, rec_ctrl, rpreds, mode=mode)["rows"]
        dec = rec_ctrl.log
    return {"commands": np.array(replay.commands), "rows": rows, "dec": dec, "preds": calls}


def open_loop(ep: Episode, name: str, policy, normalizer, preds) -> dict[str, Any]:
    ref = reference(ep, name, policy, normalizer, preds)
    ctrl = HardwareController(name, policy, normalizer, preds)
    assert ep.goal is not None
    hw = [ctrl.step(ep.view[t], ep.goal).record for t in range(len(ep.view))]
    cmds = np.array([r["command"] for r in hw])
    hw_preds = [np.array(r["pred_gripper_position"] + r["pred_object_position"]) for r in hw
                if r["pred_gripper_position"] is not None]  # fmt: skip
    ref_preds = [np.r_[p[0:3], p[3:6]] for p in ref["preds"]]
    out: dict[str, Any] = {"cmd_max_abs": _diff(cmds, ref["commands"]),
           "cmd_bitwise": bool(np.array_equal(cmds, ref["commands"])), "mismatch": []}  # fmt: skip
    for t, (h, r, d) in enumerate(zip(hw, ref["rows"], ref["dec"], strict=True)):
        for key, rv in (("delay_used", r["delay_used"]), ("active", r["active"]),
                        ("alarm", r["alarm"]), ("suspended", r["suspended"]),
                        ("est_delay", d[0]), ("est_delay_reliable", d[1])):  # fmt: skip
            if h[key] != rv:
                out["mismatch"].append((t, key))
        if name == "C":
            if h["cusum_reset_start"] != r["reset_start"]:
                out["mismatch"].append((t, "reset_start"))
            if h["cusum_stats"] != d[2]:
                out["mismatch"].append((t, "cusum_stats"))
            if h["cusum_sigma2"] != d[3]:
                out["mismatch"].append((t, "cusum_sigma2"))
    out["pred_count_equal"] = len(hw_preds) == len(ref_preds)
    out["pred_bitwise"] = out["pred_count_equal"] and all(
        np.array_equal(a, b) for a, b in zip(hw_preds, ref_preds, strict=True)
    )
    out["pred_max_abs"] = max((_diff(a, b) for a, b in zip(hw_preds, ref_preds, strict=False)),
                              default=0.0)  # fmt: skip
    out["alarms"] = int(sum(r["alarm"] for r in ref["rows"]))
    out["active_steps"] = int(sum(r["active"] for r in ref["rows"]))
    return out


# --- driver --------------------------------------------------------------------------------------


def task(args: tuple[int, str, list[Episode]]) -> tuple[list[dict], list[dict], list[dict]]:
    seed, condition, eps = args
    _, policy, normalizer, env, preds = setup(seed)
    a_rows, b_rows, goals = [], [], []
    for ep in eps:
        try:
            ep.goal = reconstruct_goal(env, ep)
            goals.append({"episode_seed": ep.episode_seed, "policy_seed": seed,
                          "configuration": ep.configuration, "ok": True})  # fmt: skip
        except RuntimeError as err:
            goals.append({"episode_seed": ep.episode_seed, "policy_seed": seed,
                          "configuration": ep.configuration, "ok": False,
                          "error": str(err)})  # fmt: skip
            continue
        key = {"policy_seed": seed, "condition": condition, "episode_seed": ep.episode_seed}
        name = RECORDED[ep.configuration]
        for path in ("direct", "adapter"):
            a_rows.append({**key, "configuration": name, "path": path,
                           **closed_loop(ep, name, policy, normalizer, preds, path)})  # fmt: skip
        if ep.configuration == "base_only":
            for cfg in ("F", "K", "CA", "C"):
                r = open_loop(ep, cfg, policy, normalizer, preds)
                b_rows.append({**key, "configuration": cfg, "cmd_max_abs": r["cmd_max_abs"],
                               "cmd_bitwise": r["cmd_bitwise"],
                               "decision_mismatches": len(r["mismatch"]),
                               "mismatch_detail": json.dumps(r["mismatch"][:10]),
                               "pred_bitwise": r["pred_bitwise"],
                               "pred_max_abs": r["pred_max_abs"], "alarms": r["alarms"],
                               "active_steps": r["active_steps"]})  # fmt: skip
    return a_rows, b_rows, goals


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/hardware_h0")
    p.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    p.add_argument("--limit", type=int, default=None, help="episodes per task (smoke runs)")
    a = p.parse_args()
    eps = load_f1()
    groups: dict[tuple[int, str], list[Episode]] = {}
    for ep in eps:
        groups.setdefault((ep.policy_seed, ep.condition), []).append(ep)
    tasks = [(s, c, v[: a.limit] if a.limit else v) for (s, c), v in sorted(groups.items())]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.map(task, tasks)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A = pd.DataFrame([r for x in results for r in x[0]])
    B = pd.DataFrame([r for x in results for r in x[1]])
    G = pd.DataFrame([r for x in results for r in x[2]])
    A.to_csv(out / "closed_loop_episodes.csv.gz", index=False)
    B.to_csv(out / "open_loop_episodes.csv.gz", index=False)
    summary = {
        "goal_reconstruction": {"episodes": len(G), "ok": int(G["ok"].sum())},
        "closed_loop": A.groupby(["configuration", "path"]).agg(
            episodes=("steps", "size"), steps=("steps", "sum"),
            cmd_exact_steps=("cmd_exact_steps", "sum"), cmd_max_abs=("cmd_max_abs", "max"),
            diverged_episodes=("first_divergence", lambda x: int((x >= 0).sum())),
            delay_mismatch=("delay_mismatch", "sum"), gain_max_abs=("gain_max_abs", "max"),
            active_mismatch=("active_mismatch", "sum"), pred_max_abs=("pred_max_abs", "max"),
            adapter_obs_max_abs=("adapter_obs_max_abs", "max"),
        ).reset_index().to_dict("records"),
        "open_loop": B.groupby("configuration").agg(
            episodes=("cmd_bitwise", "size"), cmd_bitwise=("cmd_bitwise", "sum"),
            cmd_max_abs=("cmd_max_abs", "max"),
            decision_mismatches=("decision_mismatches", "sum"),
            pred_bitwise=("pred_bitwise", "sum"), pred_max_abs=("pred_max_abs", "max"),
            alarms=("alarms", "sum"), active_steps=("active_steps", "sum"),
        ).reset_index().to_dict("records"),
    }  # fmt: skip
    (out / "equivalence_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
