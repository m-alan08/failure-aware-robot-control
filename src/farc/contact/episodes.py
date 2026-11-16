"""Closed-loop FetchPush episodes with delay compensation through a pluggable predictor.

Every configuration uses Phase E's estimator and latched delay and gain estimates, unchanged.
Only the step that maps (observation, queued commands) to a predicted observation differs:

- deployable predictors are plain functions ``(observation, queued, gain) -> observation``.
  They receive the controller's (possibly noisy) observation vector and the commands it has
  already issued, and nothing else. No simulator, session or ground truth reaches them; a
  test checks this.
- the PRIVILEGED oracle needs the live session. It snapshots the simulator, executes the
  next d̂ steps through the real disturbance pipeline, reads the true state, and restores
  the snapshot exactly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from farc.adaptation.compensation import LatchedEstimate, predicted_observation
from farc.adaptation.estimator import ActuatorEstimator
from farc.adaptation.experiment import true_parameters
from farc.contact.dynamics import DynamicsModel, rollout
from farc.counterfactual.session import EpisodeSession
from farc.disturbances.config import ConditionSpec
from farc.envs.observation_layout import OBJECT

Predictor = Callable[[np.ndarray, list[np.ndarray], float], np.ndarray]
S = OBJECT.slices


def phase_e_predictor(nominal_gain: float) -> Predictor:
    def predict(observation, queued, gain):
        return predicted_observation(observation, OBJECT, queued, nominal_gain, min(gain, 1.0))

    return predict


def learned_predictor(model: DynamicsModel, move_object: bool = True) -> Predictor:
    def predict(observation, queued, gain):
        return rollout(model, observation, queued, min(gain, 1.0), move_object)

    return predict


def oracle_prediction(session: EpisodeSession, steps: int) -> np.ndarray:
    """PRIVILEGED: the true observation after the next ``steps`` decisions, whose executed
    commands are already queued. The commands issued during the look-ahead are placeholders
    and only execute if the true delay is shorter than ``steps``. The session is restored
    exactly afterwards."""
    snapshot = session.snapshot()
    pending = session.pending_view
    try:
        horizon = min(steps, session.horizon - session.step)
        for j in range(horizon):
            view = pending if j == 0 else session.observe()
            assert view is not None
            session.act(view.base_action)
        return np.asarray(session.observation["observation"], np.float64).copy()
    finally:
        session.restore(snapshot)


def queued_commands(history: list[np.ndarray], delay: int) -> list[np.ndarray]:
    """Commands issued at t-d .. t-1 (zero before the episode), in execution order."""
    t = len(history)
    return [history[j] if j >= 0 else np.zeros(4) for j in range(t - delay, t)]


def run_episode(
    session: EpisodeSession,
    estimator: ActuatorEstimator,
    predictor: Predictor | None,
    condition: ConditionSpec,
    use_oracle: bool = False,
    log: bool = False,
) -> dict[str, Any]:
    if predictor is not None and use_oracle:
        raise ValueError("choose a deployable predictor or the oracle, not both")
    source = LatchedEstimate()
    rows = []
    while not session.done:
        view = session.observe()
        assert view.observation is not None
        estimator.observe(view.gripper)
        estimate = estimator.estimate()
        params = source.parameters(estimate)
        delay = int(params.delay or 0)
        gain = float(params.gain) if params.gain is not None else 1.0
        true_obs = np.asarray(session.observation["observation"], np.float64).copy()
        predicted = None
        if delay and (predictor is not None or use_oracle):
            if use_oracle:
                predicted = oracle_prediction(session, delay)
                predicted[S["finger_position"]] = view.observation[S["finger_position"]]
            else:
                assert predictor is not None
                queued = queued_commands(list(session.commands), delay)
                predicted = predictor(view.observation.copy(), queued, gain)
            normalized = session.normalizer(predicted, view.goal)[None]
            command = np.asarray(session.policy(normalized)[0], np.float64)
        else:
            command = view.base_action
        command = np.clip(command, -1.0, 1.0)
        executed = session.act(command)
        estimator.issued(command)
        if log:
            gain_true, delay_true = true_parameters(condition, view.step)
            row = {"t": view.step, "delay_param": delay, "gain_param": gain,
                   "delay_true": delay_true, "gain_true": gain_true,
                   "active": predicted is not None}  # fmt: skip
            row.update({f"true{i}": v for i, v in enumerate(true_obs)})
            row.update({f"view{i}": v for i, v in enumerate(view.observation)})
            row.update({f"exec{i}": v for i, v in enumerate(executed[:3])})
            row.update({f"cmd{i}": v for i, v in enumerate(command[:3])})
            if predicted is not None:
                for key in ("gripper_position", "object_position"):
                    for i, v in zip(range(3), predicted[S[key]], strict=True):
                        row[f"pred_{key}{i}"] = v
            rows.append(row)
    return {"success": session.final_success, "final_distance": session.final_distance,
            "rows": rows}  # fmt: skip
