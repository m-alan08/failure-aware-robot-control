"""Stage H0 offline latency benchmark (docs/HARDWARE_H0_RESULTS.md).

Measures, on THIS machine only, the per-call compute time of each component and of the full
cycle (adapter -> controller -> FIFO), using recorded F1 development views as inputs. No
simulator stepping, no robot, no GPU (torch on CPU, one thread, as in the G4 runner). The
numbers say nothing about the target lab computer until they are re-measured there.

    uv run python scripts/hardware_h0_benchmark.py --out results/hardware_h0
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.contact.episodes import queued_commands
from farc.delaychange.mechanism import sequential
from farc.hardware.config import replay_config
from farc.hardware.controller import C_CONSTANTS, ESTIMATOR, HardwareController
from farc.hardware.fifo import CommandFIFO
from farc.hardware.observation import ObservationAdapter, raw_from_sim_observation
from farc.hardware.pipeline import HardwarePipeline
from farc.hardware.replay import load_f1, reconstruct_goal

sys.path.insert(0, str(Path(__file__).parent))
from hardware_h0_replay import setup  # noqa: E402

PERIOD_NS = 40_000_000
POLICY_SEED = 1  # the proposal's pre-selected policy (section 9, item 5)
CONDITIONS = ("C0_undisturbed", "C3_delay3", "C4_delay4", "C5_delay3_onset15")


def stats(ns: list[int]) -> dict[str, float]:
    a = np.asarray(ns, np.float64) / 1e3  # microseconds
    return {"n": int(a.size), "median_us": float(np.median(a)),
            "p95_us": float(np.percentile(a, 95)), "p99_us": float(np.percentile(a, 99)),
            "max_us": float(a.max())}  # fmt: skip


def machine() -> dict[str, str]:
    cpu = platform.processor()
    with contextlib.suppress(OSError, subprocess.CalledProcessError):
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                             text=True, check=True).stdout.strip() or cpu  # fmt: skip
    return {"role": "LOCAL development machine (NOT the target lab computer)",
            "platform": platform.platform(), "cpu": cpu, "python": platform.python_version(),
            "torch": torch.__version__, "torch_threads": str(torch.get_num_threads())}  # fmt: skip


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/hardware_h0")
    p.add_argument("--episodes", type=int, default=100, help="per condition")
    a = p.parse_args()
    _, policy, normalizer, env, preds = setup(POLICY_SEED)
    eps = [e for e in load_f1(("base_only",)) if e.policy_seed == POLICY_SEED
           and e.condition in CONDITIONS]  # fmt: skip
    eps = [e for c in CONDITIONS for e in [x for x in eps if x.condition == c][: a.episodes]]
    goal_of = {id(e): reconstruct_goal(env, e) for e in eps}
    for e in eps:
        e.goal = goal_of[id(e)]
    views = np.concatenate([e.view for e in eps])
    goals = np.concatenate([np.repeat(goal_of[id(e)][None], len(e.view), 0) for e in eps])
    cmds = np.concatenate([np.c_[e.cmd, np.zeros(len(e.cmd))] for e in eps])
    ns = time.perf_counter_ns
    out: dict[str, dict] = {}
    gc.collect()

    # warm-up (allocations, torch kernels)
    for i in range(200):
        policy(normalizer(views[i], goals[i])[None])

    adapter = ObservationAdapter(replay_config())
    raws = [raw_from_sim_observation(v, (i + 1) * PERIOD_NS) for i, v in enumerate(views)]
    lat = []
    for r in raws:
        t0 = ns()
        adapter.observation(r)
        lat.append(ns() - t0)
    out["observation_adapter"] = stats(lat)

    lat = []
    for v, g in zip(views, goals, strict=True):
        t0 = ns()
        policy(normalizer(v, g)[None])
        lat.append(ns() - t0)
    out["frozen_sac_inference"] = stats(lat)

    est_lat, c_lat, kin_lat, f1_lat = [], [], [], []
    for e in eps:
        est = ActuatorEstimator(EstimatorConfig(**ESTIMATOR))
        est_c = ActuatorEstimator(EstimatorConfig(**ESTIMATOR))
        sup = sequential(C_CONSTANTS)
        hist: list[np.ndarray] = []
        for t in range(len(e.view)):
            t0 = ns()
            est.observe(e.view[t][0:3])
            est.estimate()
            est_lat.append(ns() - t0)
            est_c.observe(e.view[t][0:3])
            t0 = ns()
            sup.step(est_c)
            c_lat.append(ns() - t0)
            q = queued_commands(hist, 4)  # worst case: d = 4
            t0 = ns()
            preds["phase_e"](e.view[t].copy(), q, 1.0)
            kin_lat.append(ns() - t0)
            t0 = ns()
            preds["learned"](e.view[t].copy(), q, 1.0)
            f1_lat.append(ns() - t0)
            c = np.r_[e.cmd[t], 0.0]
            est.issued(c)
            est_c.issued(c)
            hist.append(c)
    out["phase_e_estimator_observe_estimate"] = stats(est_lat)
    out["candidate_c_supervisor_step"] = stats(c_lat)
    out["kinematic_predictor_d4"] = stats(kin_lat)
    out["f1_contact_model_rollout_d4"] = stats(f1_lat)
    del cmds

    for name in ("F", "K", "CA", "C"):
        pipe = HardwarePipeline(ObservationAdapter(replay_config()),
                                HardwareController(name, policy, normalizer, preds),
                                CommandFIFO())  # fmt: skip
        total, ctrl = [], []
        for e in eps:
            pipe.reset()
            for t in range(len(e.view)):
                r = raw_from_sim_observation(e.view[t], (t + 1) * PERIOD_NS)
                rec = pipe.cycle(r, goal_of[id(e)]).record
                total.append(rec["lat_pipeline_total_ns"])
                ctrl.append(rec["lat_controller_total_ns"])
        out[f"full_pipeline_{name}"] = stats(total)
        out[f"controller_only_{name}"] = stats(ctrl)

    budget_us = 10_000.0
    result = {"machine": machine(), "inputs": {"policy_seed": POLICY_SEED,
              "conditions": list(CONDITIONS), "episodes": len(eps), "steps": int(len(views))},
              "budget_p99_us": budget_us, "latency": out,
              "p99_within_budget_on_this_machine": {k: v["p99_us"] <= budget_us
                                                    for k, v in out.items()
                                                    if k.startswith("full_pipeline")}}  # fmt: skip
    path = Path(a.out)
    path.mkdir(parents=True, exist_ok=True)
    (path / "latency_local.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(1)
    main()
