"""G3 mechanism isolation for Candidate C (docs/PHASE_G3_PROPOSAL.md).

``EventReplay`` is Candidate C with its CUSUM detector replaced by injected events, and with
the pause and the window reset as separate switches. ``Sequential`` itself is not modified:
the post-detection code below repeats it, and tests check that replaying C's own events with
both switches on reproduces closed-loop C's issued commands bitwise.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.adaptation.experiment import AdaptationConfigError
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.dynamics import load_dynamics
from farc.contact.episodes import Predictor, learned_predictor, queued_commands
from farc.delaychange.candidates import Decision, Sequential, window_fit
from farc.delaychange.random_schedules import (
    GENERATOR_SEED,
    RandomCell,
    g3_cells,
    schedule_for,
    transitions,
)
from farc.delaychange.schedule import ScheduledDelaySession
from farc.delaychange.stage3 import Latched
from farc.disturbances.config import ConditionSpec
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.supervision.seeds import check_g3_seeds

KEYS = ["policy_seed", "condition", "episode_seed"]
SOURCES = ("base_only", "contact_learned", "C_full", "ORACLE_full", "PRIV_true_schedule")
DEPENDANTS = ("C_pause", "C_reset", "C_pause_own", "sham_pause_random", "sham_full_yoked",
              "ORACLE_pause", "ORACLE_reset")  # fmt: skip
CONFIGURATIONS = SOURCES + DEPENDANTS
PRIVILEGED = ("ORACLE_full", "ORACLE_pause", "ORACLE_reset", "PRIV_true_schedule")
# Runs whose shadow detector must reproduce C's first alarm exactly (section 4.4).
FIRST_ALARM_IDENTICAL = ("C_pause", "C_reset", "C_pause_own")
NO_SHADOW = ("base_only", "PRIV_true_schedule")


@dataclass
class EventReplay(Sequential):
    """Candidate C driven by injected events {step: change point} instead of its CUSUM."""

    events: dict[int, int] = field(default_factory=dict)
    pause: bool = True
    reset: bool = True

    def step(self, est: ActuatorEstimator) -> Decision:
        t = est.step
        alarm = t in self.events
        if alarm:
            if self.reset:
                self.reset_start = self.events[t]
            if self.pause and self.trusted > 0:
                self.suspended = True
            self.pending = True
            self._clear(t)
        # --- from here on, identical to Sequential.step -----------------------------------
        e, sse = window_fit(est, self.window, self.reset_start)
        if e.gain_reliable:
            self.gain_estimate = e.gain
        if e.delay_reliable and e.delay is not None:
            self.sigma2 = max(self.sigma0**2, float(sse[e.delay]) / (2.0 * max(e.transitions, 1)))
            if e.delay != self.trusted or self.suspended:
                self._clear(t)
            self.trusted, self.suspended, self.pending = e.delay, False, False
        return Decision(self.trusted > 0 and not self.suspended, self.trusted, e, alarm,
                        self.suspended)  # fmt: skip


def sequential(constants: dict[str, Any]) -> Sequential:
    return Sequential(float(constants["C_threshold"]), float(constants["C_sigma0_mm"]) / 1000,
                      str(constants["C_variant"]))  # fmt: skip


def replay(constants: dict[str, Any], events: dict[int, int], pause: bool,
           reset: bool) -> EventReplay:  # fmt: skip
    return EventReplay(float(constants["C_threshold"]), float(constants["C_sigma0_mm"]) / 1000,
                       str(constants["C_variant"]), events=dict(events), pause=pause,
                       reset=reset)  # fmt: skip


def oracle_events(schedule, horizon: int = 50) -> dict[int, int]:
    """PRIVILEGED: an event at every true transition step t, with change point t (the command
    executed at t is the first one the new delay affects; transition t -> t+1 is the first
    affected position transition)."""
    return {t: t for t, _ in transitions(schedule, horizon)}


def yoked_events(source: list[tuple[int, int]]) -> dict[int, int]:
    """Another episode's events (alarm step, change point), keeping each alarm-to-change-point
    offset."""
    return {t: cp for t, cp in source}


def rotate(mask: np.ndarray, policy_seed: int, cell_index: int, episode_seed: int) -> np.ndarray:
    n = len(mask)
    rng = np.random.default_rng([GENERATOR_SEED, policy_seed, cell_index, episode_seed])
    return np.roll(mask, int(rng.integers(1, n))) if n > 1 else mask.copy()


def run_episode(session: ScheduledDelaySession, est: ActuatorEstimator, ctrl,
                preds: dict[str, Predictor], *, mode: str = "normal",
                force_off: np.ndarray | None = None,
                shadow: Sequential | None = None) -> dict[str, Any]:  # fmt: skip
    """mode: normal | base (never compensate) | true (PRIVILEGED: true delay and gain)."""
    schedule = session.schedule
    rows = []
    while not session.done:
        view = session.observe()
        assert view.observation is not None
        obs, t = view.observation, view.step
        d_true = schedule.delay_at(t)
        t0 = time.perf_counter_ns()
        est.observe(view.gripper)
        dec = ctrl.step(est)
        d, active = dec.delay, dec.active
        gain = ctrl.gain
        if mode == "base":
            active = False
        elif mode == "true":  # PRIVILEGED
            d, active = d_true, d_true > 0
            gain = float(np.prod([1.0 - x.severity for x in session.condition.disturbances
                                  if x.type == "action_scale" and x.active(t)]))  # fmt: skip
        elif force_off is not None and t < len(force_off) and force_off[t]:
            active = False
        calls = 0
        if active:
            queued = queued_commands(list(session.commands), d)
            predicted = preds["learned"](obs.copy(), queued, gain)
            calls = 1
            command = np.asarray(session.policy(session.normalizer(predicted, view.goal)[None])[0],
                                 np.float64)  # fmt: skip
        else:
            command = view.base_action
        command = np.clip(command, -1.0, 1.0)
        ns = time.perf_counter_ns() - t0
        shadow_alarm = bool(shadow.step(est).alarm) if shadow is not None else False
        session.act(command)
        est.issued(command)
        rows.append({"t": t, "delay_true": d_true, "delay_used": d, "active": bool(active),
                     "alarm": bool(dec.alarm), "suspended": bool(dec.suspended),
                     "shadow_alarm": shadow_alarm,
                     "reset_start": int(getattr(ctrl, "reset_start", -1)),
                     "ctrl_ns": ns, "learned_calls": calls})  # fmt: skip
    return {"success": session.final_success, "final_distance": session.final_distance,
            "rows": rows}  # fmt: skip


def events_of(rows: list[dict[str, Any]]) -> list[tuple[int, int]]:
    return [(r["t"], r["reset_start"]) for r in rows if r["alarm"]]


def suspended_of(rows: list[dict[str, Any]]) -> np.ndarray:
    return np.array([r["suspended"] for r in rows], bool)


# --- config and tasks -------------------------------------------------------------------------


@dataclass(frozen=True)
class G3Config:
    name: str
    plan: str
    plan_sha256: str
    model: str
    model_sha256: str
    checkpoints: tuple[str, ...]
    debug_seed_offset: int
    debug_episodes: int
    evaluation_seed_offset: int
    evaluation_episodes: int
    perturbation_seed: int
    estimator: dict[str, Any]
    constants: dict[str, Any]
    alpha_one_sided: float
    minimum_meaningful: float
    persistent_noninferiority: float
    sac_noninferiority: float
    harm_ci_floor: float
    harm_max_fraction: float
    bootstrap_resamples: int
    bootstrap_seed: int
    output: str
    role: str = "evaluation"

    @property
    def seeds(self) -> list[int]:
        offset, count = {"debugging": (self.debug_seed_offset, self.debug_episodes),
                         "evaluation": (self.evaluation_seed_offset,
                                        self.evaluation_episodes)}[self.role]  # fmt: skip
        return list(range(offset, offset + count))


def load_g3(path: str | Path, role: str = "evaluation") -> G3Config:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    values["checkpoints"] = tuple(values["checkpoints"])
    config = G3Config(**values, role=role)
    for r in ("debugging", "evaluation"):
        check_g3_seeds(G3Config(**values, role=r).seeds, r)
    if hashlib.sha256(Path(config.plan).read_bytes()).hexdigest() != config.plan_sha256:
        raise AdaptationConfigError("the G3 plan changed since the config pinned it")
    return config


def _session(env, policy, normalizer, cell: RandomCell, index: int, seed: int,
             config: G3Config) -> ScheduledDelaySession:  # fmt: skip
    return ScheduledDelaySession(env, policy, normalizer, ConditionSpec(cell.name, cell.extra),
                                 seed, config.perturbation_seed, schedule_for(cell, index, seed),
                                 "replay_drop")  # fmt: skip


def run_task(args: tuple[G3Config, str, RandomCell]):
    config, checkpoint, cell = args
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    actor = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    policy = deterministic_policy_function(loaded.actor)
    normalizer, env = FrozenNormalizer(loaded.normalizer), make_env(loaded.env_config)
    est_cfg = EstimatorConfig(**config.estimator)
    model = load_dynamics(resolve(config.model), config.model_sha256)
    preds = {"learned": learned_predictor(model, move_object=True)}
    index = [c.name for c in g3_cells()].index(cell.name)
    seeds, k = config.seeds, config.constants
    ps = loaded.training_seed
    c_events: dict[int, list[tuple[int, int]]] = {}
    c_mask: dict[int, np.ndarray] = {}
    o_mask: dict[int, np.ndarray] = {}
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    def plan(name: str, seed: int, schedule) -> tuple[Any, dict[str, Any]]:
        kw: dict[str, Any] = {}
        if name == "base_only":
            return Latched(), {"mode": "base"}
        if name == "PRIV_true_schedule":
            return Latched(), {"mode": "true"}
        if name == "contact_learned":
            ctrl: Any = Latched()
        elif name == "C_full":
            ctrl = sequential(k)
        elif name == "ORACLE_full":
            ctrl = replay(k, oracle_events(schedule), True, True)
        elif name == "C_pause":
            ctrl, kw = Latched(), {"force_off": c_mask[seed]}
        elif name == "C_reset":
            ctrl = replay(k, yoked_events(c_events[seed]), False, True)
        elif name == "C_pause_own":
            ctrl = replay(k, yoked_events(c_events[seed]), True, False)
        elif name == "sham_pause_random":
            ctrl, kw = Latched(), {"force_off": rotate(c_mask[seed], ps, index, seed)}
        elif name == "sham_full_yoked":  # the next seed of the same cell: another schedule
            source = seeds[(seeds.index(seed) + 1) % len(seeds)]
            ctrl = replay(k, yoked_events(c_events[source]), True, True)
        elif name == "ORACLE_pause":
            ctrl, kw = Latched(), {"force_off": o_mask[seed]}
        elif name == "ORACLE_reset":
            ctrl = replay(k, oracle_events(schedule), False, True)
        else:
            raise ValueError(name)
        return ctrl, kw

    for group in (SOURCES, DEPENDANTS):
        for seed in seeds:
            for name in group:
                session = _session(env, policy, normalizer, cell, index, seed, config)
                ctrl, kw = plan(name, seed, session.schedule)
                shadow = None if name in NO_SHADOW else sequential(k)
                result = run_episode(session, ActuatorEstimator(est_cfg), ctrl, preds,
                                     shadow=shadow, **kw)  # fmt: skip
                rows = result["rows"]
                if name == "C_full":
                    c_events[seed], c_mask[seed] = events_of(rows), suspended_of(rows)
                elif name == "ORACLE_full":
                    o_mask[seed] = suspended_of(rows)
                key = {"configuration": name, "policy_seed": ps, "condition": cell.name,
                       "family": cell.family, "episode_seed": seed}  # fmt: skip
                episodes.append(
                    {
                        **key,
                        "success": result["success"],
                        "final_distance": result["final_distance"],
                        "events": int(sum(r["alarm"] for r in rows)),
                        "paused_steps": int(sum(r["suspended"] for r in rows))
                        + int(kw["force_off"].sum() if "force_off" in kw else 0),
                    }
                )
                steps.extend({**key, **r} for r in rows)
    for name, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor[name]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes, steps


def replay_check(args: tuple[G3Config, str, RandomCell]) -> list[dict[str, Any]]:
    """Code check (debugging seeds only): replaying C's own events with pause and reset
    reproduces closed-loop C's issued commands bitwise, and an event-free replay reproduces
    contact_learned."""
    config, checkpoint, cell = args
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    policy = deterministic_policy_function(loaded.actor)
    normalizer, env = FrozenNormalizer(loaded.normalizer), make_env(loaded.env_config)
    est_cfg = EstimatorConfig(**config.estimator)
    model = load_dynamics(resolve(config.model), config.model_sha256)
    preds = {"learned": learned_predictor(model, move_object=True)}
    index = [c.name for c in g3_cells()].index(cell.name)
    out = []
    for seed in config.seeds:
        runs: dict[str, tuple[np.ndarray, list[tuple[int, int]]]] = {}
        for name in ("C_full", "replay_full", "contact_learned", "replay_none"):
            session = _session(env, policy, normalizer, cell, index, seed, config)
            if name == "C_full":
                ctrl: Any = sequential(config.constants)
            elif name == "replay_full":
                ctrl = replay(config.constants, yoked_events(runs["C_full"][1]), True, True)
            elif name == "contact_learned":
                ctrl = Latched()
            else:
                ctrl = replay(config.constants, {}, True, True)
            result = run_episode(session, ActuatorEstimator(est_cfg), ctrl, preds)
            runs[name] = (np.array(session.commands), events_of(result["rows"]))
        out.append(
            {
                "condition": cell.name,
                "policy_seed": loaded.training_seed,
                "episode_seed": seed,
                "c_events": len(runs["C_full"][1]),
                "replay_identical": bool(np.array_equal(runs["C_full"][0], runs["replay_full"][0])),
                "empty_replay_is_contact": bool(
                    np.array_equal(runs["contact_learned"][0], runs["replay_none"][0])
                ),
            }
        )
    return out
