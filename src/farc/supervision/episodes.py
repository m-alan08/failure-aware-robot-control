"""Closed-loop FetchPush episodes for every Phase G1 configuration.

Every configuration runs the frozen Phase E estimator on every step (for logging), and all of
them share one code path. ``base_only``, ``phase_e_predictor``, ``contact_learned`` and
``oracle_sim_predictor`` reproduce ``farc.contact.episodes.run_episode`` exactly (a test checks
the issued commands are identical).

Timing (``perf_counter_ns``, descriptive only):
- ``est_ns``: ``estimator.observe`` + ``estimate`` (+ the latch);
- ``ctrl_ns``: supervisor, predictor(s) and the frozen policy on the predicted observation.

The base policy's forward pass on the raw observation happens inside ``session.observe()``
for every configuration and is excluded from both. ``base_only``'s controller overhead is
zero by definition; its estimator runs only to log what it would have done.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np

from farc.adaptation.compensation import LatchedEstimate
from farc.adaptation.estimator import ActuatorEstimator, lag_residuals
from farc.adaptation.experiment import true_parameters
from farc.contact.episodes import oracle_prediction, queued_commands
from farc.counterfactual.session import EpisodeSession
from farc.disturbances.config import ConditionSpec
from farc.supervision.supervisors import (
    ContactGate,
    NeverReleaseLatch,
    PredictorSelector,
    ReleaseSupervisor,
    S,
    SupervisorConfig,
)

Predictor = Callable[[np.ndarray, list[np.ndarray], float], np.ndarray]

CONFIGURATIONS = (
    "base_only",
    "phase_e_predictor",
    "contact_learned",
    "never_release",
    "sup_A_release",
    "sup_A_timeout",
    "sup_B_select",
    "sup_C_contact_gate",
    "oracle_sim_predictor",
    "sup_A_yoked",  # yoked shams run after their sources
    "sup_B_yoked",
)
YOKED = {"sup_A_yoked": "sup_A_release", "sup_B_yoked": "sup_B_select"}
SUPERVISORS = ("sup_A_release", "sup_A_timeout", "sup_B_select", "sup_C_contact_gate")


def yoke_source(seeds: list[int], seed: int) -> int:
    """Episode i takes its switching sequence from episode (i + 1) mod n."""
    return seeds[(seeds.index(seed) + 1) % len(seeds)]


def run_supervised_episode(
    session: EpisodeSession,
    estimator: ActuatorEstimator,
    name: str,
    predictors: dict[str, Predictor],  # "learned" and "phase_e", deployable only
    condition: ConditionSpec,
    cfg: SupervisorConfig,
    yoked: np.ndarray | None = None,  # released flags (A) or predictor choices (B) by step
) -> dict[str, Any]:
    if name not in CONFIGURATIONS:
        raise ValueError(f"unknown configuration {name}")
    if (name in YOKED) != (yoked is not None):
        raise ValueError("yoked shams need a source sequence, and only they take one")
    latch = NeverReleaseLatch() if name == "never_release" else LatchedEstimate()
    release = None
    if name == "sup_A_release":
        release = ReleaseSupervisor(cfg, "cusum")
    elif name == "sup_A_timeout":
        release = ReleaseSupervisor(cfg, "timeout")
    elif name == "sup_A_yoked":
        release = ReleaseSupervisor(cfg, "yoked", mask=np.asarray(yoked, bool))
    selector = PredictorSelector(cfg) if name == "sup_B_select" else None
    gate = ContactGate(cfg) if name == "sup_C_contact_gate" else None
    rows: list[dict[str, Any]] = []
    while not session.done:
        view = session.observe()
        assert view.observation is not None
        obs = view.observation
        true_obs = np.asarray(session.observation["observation"], np.float64)
        near_true = bool(np.linalg.norm(true_obs[S["object_relative_position"]][:2]) < 0.05)
        t0 = time.perf_counter_ns()
        estimator.observe(view.gripper)
        estimate = estimator.estimate()
        params = latch.parameters(estimate)
        t1 = time.perf_counter_ns()
        delay = int(params.delay or 0)
        gain = float(params.gain) if params.gain is not None else 1.0
        active = bool(delay) and name != "base_only"
        released, increment, statistic = False, None, None
        if release is not None:
            transitions, sse = lag_residuals(estimator) if release.kind == "cusum" else (0, None)
            active = release.allow(view.step, delay, estimate, transitions, sse)
            released, increment, statistic = release.released, release.increment, release.statistic
        errors: dict[str, float] = {}
        if selector is not None:
            selector.score(view.step, obs)
            errors = dict(selector.last_errors)
        used, learned_calls = "none", 0
        if active:
            queued = queued_commands(list(session.commands), delay)
            if name == "oracle_sim_predictor":
                used = "oracle"
                predicted = oracle_prediction(session, delay)
                predicted[S["finger_position"]] = obs[S["finger_position"]]
            else:
                if name in ("phase_e_predictor",):
                    used = "phase_e"
                elif selector is not None:
                    used = selector.choose(view.step)
                elif name == "sup_B_yoked":
                    assert yoked is not None
                    choice = yoked[view.step] if view.step < len(yoked) else "none"
                    used = "phase_e" if choice == "phase_e" else "learned"
                elif gate is not None:
                    used = gate.choose(view.step, obs)
                else:
                    used = "learned"
                if selector is not None:  # B scores both predictors on every active step
                    both = {k: predictors[k](obs.copy(), queued, gain)
                            for k in ("learned", "phase_e")}  # fmt: skip
                    selector.record(view.step, delay, both)
                    predicted, learned_calls = both[used], 1
                else:
                    predicted = predictors[used](obs.copy(), queued, gain)
                    learned_calls = int(used == "learned")
            normalized = session.normalizer(predicted, view.goal)[None]
            command = np.asarray(session.policy(normalized)[0], np.float64)
        else:
            command = view.base_action
        command = np.clip(command, -1.0, 1.0)
        t2 = time.perf_counter_ns()
        session.act(command)
        estimator.issued(command)
        gain_true, delay_true = true_parameters(condition, view.step)
        rows.append({
            "t": view.step, "delay_true": delay_true, "gain_true": gain_true,
            "delay_param": delay, "active": active, "predictor": used,
            "released": released, "cusum_increment": increment, "cusum_statistic": statistic,
            "est_delay": estimate.delay, "est_reliable": estimate.delay_reliable,
            "transitions": estimate.transitions, "learned_calls": learned_calls,
            "mlp_steps": learned_calls * delay, "est_ns": t1 - t0, "ctrl_ns": t2 - t1,
            "near_true": near_true,
            "err_learned": errors.get("learned"), "err_phase_e": errors.get("phase_e"),
        })  # fmt: skip
    return {"success": session.final_success, "final_distance": session.final_distance,
            "rows": rows}  # fmt: skip
