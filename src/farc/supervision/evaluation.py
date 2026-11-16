"""Phase G1 runs and pre-registered analysis (docs/PHASE_G1_PROPOSAL.md, Part 2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.delay_eval import holm, paired_difference
from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.adaptation.experiment import AdaptationConfigError
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.data import build_conditions
from farc.contact.dynamics import load_dynamics
from farc.contact.episodes import learned_predictor, phase_e_predictor
from farc.counterfactual.session import EpisodeSession
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.supervision.episodes import (
    CONFIGURATIONS,
    SUPERVISORS,
    YOKED,
    run_supervised_episode,
    yoke_source,
)
from farc.supervision.seeds import check_g1_seeds
from farc.supervision.supervisors import SupervisorConfig

KEYS = ["policy_seed", "condition", "episode_seed"]
EPISODE = ["configuration", *KEYS]
CALIBRATION_CONFIGURATION = "sup_A_release"  # run with cusum_threshold = None (never releases)


@dataclass(frozen=True)
class G1Config:
    name: str
    plan: str
    plan_sha256: str
    model: str
    model_sha256: str
    conditions_from: str
    conditions: tuple[dict[str, Any], ...]
    constant_delay_conditions: tuple[str, ...]
    transient_conditions: tuple[str, ...]
    harm_conditions: tuple[str, ...]
    calibration_conditions: tuple[str, ...]
    checkpoints: tuple[str, ...]
    debug_seed_offset: int
    debug_episodes: int
    calibration_seed_offset: int
    calibration_episodes: int
    evaluation_seed_offset: int
    evaluation_episodes: int
    perturbation_seed: int
    estimator: dict[str, Any]
    supervisor: dict[str, Any]
    noninferiority_margin: float
    call_reduction_min: float
    alpha_one_sided: float
    harm_ci_floor: float
    harm_max_fraction: float
    bootstrap_resamples: int
    bootstrap_seed: int
    output: str
    role: str = field(default="evaluation", compare=False)

    def seeds_for(self, role: str) -> list[int]:
        offset, count = {
            "debugging": (self.debug_seed_offset, self.debug_episodes),
            "calibration": (self.calibration_seed_offset, self.calibration_episodes),
            "evaluation": (self.evaluation_seed_offset, self.evaluation_episodes),
        }[role]
        return list(range(offset, offset + count))

    @property
    def seeds(self) -> list[int]:
        return self.seeds_for(self.role)

    def supervisor_config(self, threshold_required: bool = True) -> SupervisorConfig:
        values = {k: v for k, v in self.supervisor.items()
                  if k not in ("cusum_threshold_grid", "false_release_max")}  # fmt: skip
        if threshold_required and values.get("cusum_threshold") is None:
            raise AdaptationConfigError("cusum_threshold is unset: calibrate first")
        values["min_transitions"] = int(self.estimator["min_transitions"])
        return SupervisorConfig(**values)


def load_g1(path: str | Path, role: str = "evaluation", check_plan: bool = True) -> G1Config:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    for key in ("conditions", "constant_delay_conditions", "transient_conditions",
                "harm_conditions", "calibration_conditions", "checkpoints"):  # fmt: skip
        values[key] = tuple(values[key])
    config = G1Config(**values, role=role)
    for r in ("debugging", "calibration", "evaluation"):
        check_g1_seeds(config.seeds_for(r), r)
    names = {c["name"] for c in config.conditions}
    groups = (set(config.constant_delay_conditions) | set(config.transient_conditions)
              | set(config.harm_conditions) | set(config.calibration_conditions))  # fmt: skip
    if groups - names:
        raise AdaptationConfigError(f"unknown conditions in groups: {sorted(groups - names)}")
    digest = hashlib.sha256(Path(config.plan).read_bytes()).hexdigest()
    if check_plan and digest != config.plan_sha256:
        raise AdaptationConfigError("the G1 pre-registration changed; record an amendment")
    return config


# --- running --------------------------------------------------------------------------------


def run_task(args: tuple[G1Config, str, str, tuple[str, ...], bool]):
    """Every configuration and seed for one (policy, condition). Yoked shams run after their
    sources, inside the same task, so their source sequences come from this policy and
    condition only."""
    config, checkpoint, condition_name, configurations, threshold_required = args
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    actor_state = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    policy = deterministic_policy_function(loaded.actor)
    normalizer = FrozenNormalizer(loaded.normalizer)
    env = make_env(loaded.env_config)
    model = load_dynamics(resolve(config.model), config.model_sha256)
    estimator_cfg = EstimatorConfig(**config.estimator)
    predictors = {"learned": learned_predictor(model, move_object=True),
                  "phase_e": phase_e_predictor(estimator_cfg.nominal_gain)}  # fmt: skip
    cfg = config.supervisor_config(threshold_required)
    library = build_conditions(config.conditions_from, config.conditions)
    condition = {c.name: c for c in library}[condition_name]
    seeds = config.seeds
    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    sources: dict[tuple[str, int], np.ndarray] = {}
    ordered = [c for c in configurations if c not in YOKED] + [
        c for c in configurations if c in YOKED]  # fmt: skip
    for name in ordered:
        if name in YOKED and YOKED[name] not in configurations:
            raise ValueError(f"{name} needs {YOKED[name]} in the same run")
    for phase in (False, True):  # sources first, then yoked shams
        for seed in seeds:
            for name in ordered:
                if (name in YOKED) != phase:
                    continue
                yoked = None
                if name in YOKED:
                    yoked = sources[(YOKED[name], yoke_source(seeds, seed))]
                session = EpisodeSession(env, policy, normalizer, condition, seed,
                                         config.perturbation_seed)  # fmt: skip
                result = run_supervised_episode(session, ActuatorEstimator(estimator_cfg), name,
                                                predictors, condition, cfg, yoked)  # fmt: skip
                rows = pd.DataFrame(result["rows"])
                if name == "sup_A_release":
                    sources[(name, seed)] = rows["released"].to_numpy(bool)
                elif name == "sup_B_select":
                    sources[(name, seed)] = rows["predictor"].to_numpy(object)
                key = {"configuration": name, "policy_seed": loaded.training_seed,
                       "condition": condition.name, "episode_seed": seed}  # fmt: skip
                episodes.append({**key, "success": result["success"],
                                 "final_distance": result["final_distance"]})  # fmt: skip
                steps.extend({**key, **r} for r in result["rows"])
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor_state[k]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes, steps


def tasks(config: G1Config, conditions: tuple[str, ...], configurations: tuple[str, ...],
          threshold_required: bool) -> list[tuple]:  # fmt: skip
    return [(config, c, cond, configurations, threshold_required)
            for c in config.checkpoints for cond in conditions]  # fmt: skip


# --- calibration (detection criterion only; success is never read) ---------------------------


def simulate_release(ep: pd.DataFrame, h: float, dwell: int) -> tuple[int, int]:
    """Off-steps a CUSUM with threshold h would have caused on a never-releasing trace, and
    the trace's active steps. After a release the compensator stays off for the dwell and
    until the trace shows a fresh reliable d > 0, as the supervisor requires. The trace
    continues with compensation on, so this is an approximation used only to set h."""
    active = ep["active"].to_numpy(bool)
    inc = ep["cusum_increment"].to_numpy(float)
    fresh = (ep["est_reliable"].to_numpy(bool)
             & (ep["est_delay"].fillna(0).to_numpy(float) > 0))  # fmt: skip
    off, s, released, last_switch, engaged = 0, 0.0, False, -(10**9), False
    for t in range(len(ep)):
        if not active[t]:
            s, released, engaged = 0.0, False, False
            continue
        if not engaged:
            engaged, last_switch, s = True, t, 0.0
        if released:
            if t - last_switch >= dwell and fresh[t]:
                released, last_switch, s = False, t, 0.0
            else:
                off += 1
                continue
        if np.isfinite(inc[t]):
            s = max(0.0, s + inc[t])
        if s > h and t - last_switch >= dwell:
            released, last_switch, s = True, t, 0.0
            off += 1
    return off, int(active.sum())


def calibrate_threshold(steps: pd.DataFrame, grid: list[float], dwell: int, max_rate: float):
    """The smallest h on the grid whose false-off share of active steps is <= max_rate under
    constant delay. Returns (h or None, curve)."""
    trace = steps[steps["configuration"] == CALIBRATION_CONFIGURATION].sort_values(EPISODE + ["t"])
    groups = [ep for _, ep in trace.groupby(EPISODE, sort=False)]
    curve = []
    for h in grid:
        totals = np.array([simulate_release(ep, h, dwell) for ep in groups])
        off, active = totals.sum(axis=0)
        curve.append({"threshold": h, "false_off_steps": int(off), "active_steps": int(active),
                      "false_off_rate": off / max(active, 1),
                      "episodes_with_release": int((totals[:, 0] > 0).sum()),
                      "episodes": len(groups)})  # fmt: skip
    frame = pd.DataFrame(curve)
    ok = frame[frame["false_off_rate"] <= max_rate]
    return (float(ok["threshold"].min()) if len(ok) else None), frame


# --- pre-registered analysis -----------------------------------------------------------------

DERIVED_TABLES = ("hypotheses", "specificity", "safety_gates", "by_condition", "latency",
                  "false_activations", "false_releases", "switching", "oscillation_cases",
                  "computation", "prediction_errors")  # fmt: skip


def one_sided(episodes: pd.DataFrame, a: str, b: str, bound: float, n: int, seed: int):
    """Paired difference a - b (episode-seed cluster bootstrap, as every earlier phase) plus
    the one-sided bootstrap p-value for 'difference > bound'."""
    est, low, high, p2, count = paired_difference(episodes, a, b, n, seed)
    wide = episodes.pivot_table(index=KEYS, columns="configuration", values="success",
                                aggfunc="first").astype(float)  # fmt: skip
    diff = (wide[a] - wide[b]).dropna()
    groups = diff.groupby(level="episode_seed")
    sums, counts = groups.sum().to_numpy(), groups.count().to_numpy()
    picks = np.random.default_rng(seed).integers(0, len(sums), size=(n, len(sums)))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return {"estimate": est, "ci_low": low, "ci_high": high, "p_two_sided": p2,
            "p_one_sided": float((draws <= bound).mean()), "bound": bound,
            "episodes": count}  # fmt: skip


def _outcomes(frame: pd.DataFrame, a: str, b: str) -> dict[str, Any]:
    w = frame.pivot_table(index=KEYS, columns="configuration", values="success",
                          aggfunc="first").astype(bool)  # fmt: skip
    ref, new = w[b], w[a]
    return {"success_rate": float(new.mean()), "reference_success_rate": float(ref.mean()),
            "rescued": int((~ref & new).sum()), "harmed": int((ref & ~new).sum()),
            "reference_successes": int(ref.sum()),
            "harmed_fraction": float((ref & ~new).sum() / max(ref.sum(), 1))}  # fmt: skip


def learned_calls(steps: pd.DataFrame, conditions: tuple[str, ...], name: str) -> pd.Series:
    sub = steps[(steps["configuration"] == name) & steps["condition"].isin(conditions)]
    return sub.groupby(KEYS)["learned_calls"].sum()


def _changes(episode_steps: pd.DataFrame) -> tuple[int, list[int]]:
    mode = (episode_steps["active"].astype(str) + "/" + episode_steps["predictor"]).to_numpy()
    changes = np.flatnonzero(mode[1:] != mode[:-1]) + 1
    bounds = [0, *changes.tolist(), len(mode)]
    return len(changes), [b - a for a, b in zip(bounds[:-1], bounds[1:], strict=True)]


def derive_g1(episodes: pd.DataFrame, steps: pd.DataFrame, config: G1Config):
    n, s = config.bootstrap_resamples, config.bootstrap_seed
    const = episodes[episodes["condition"].isin(config.constant_delay_conditions)]
    trans = episodes[episodes["condition"].isin(config.transient_conditions)]
    both = episodes[episodes["condition"].isin(config.constant_delay_conditions
                                               + config.transient_conditions)]  # fmt: skip
    margin = -config.noninferiority_margin
    family = [
        ("G1", "sup_A_release", "contact_learned", trans, 0.0, "T1-T4"),
        ("G2", "sup_A_release", "contact_learned", const, margin, "C1-C7"),
        ("G3", "sup_B_select", "contact_learned", both, 0.0, "C1-C7 + T1-T4"),
        ("G4", "sup_C_contact_gate", "contact_learned", const, margin, "C1-C7"),
    ]
    rows = [{"hypothesis": h, "configuration": a, "reference": b, "conditions": label,
             **one_sided(frame, a, b, bound, n, s)}
            for h, a, b, frame, bound, label in family]  # fmt: skip
    hyp = pd.DataFrame(rows)
    hyp["p_holm"] = holm(hyp["p_one_sided"].tolist())
    base_calls = learned_calls(steps, config.constant_delay_conditions, "contact_learned")
    gate_calls = learned_calls(steps, config.constant_delay_conditions, "sup_C_contact_gate")
    reduction = 1.0 - gate_calls.sum() / max(base_calls.sum(), 1)
    hyp["call_reduction"] = np.where(hyp["hypothesis"] == "G4", reduction, np.nan)
    passed = hyp["p_holm"] <= config.alpha_one_sided
    passed &= (hyp["hypothesis"] != "G4") | (reduction >= config.call_reduction_min)
    hyp["status"] = np.where(passed, "supported", "not supported")

    spec_rows = [
        ("S-A1", "sup_A_release", "sup_A_yoked", trans, "T1-T4"),
        ("S-A2_transient", "sup_A_release", "sup_A_timeout", trans, "T1-T4"),
        ("S-A2_constant", "sup_A_release", "sup_A_timeout", const, "C1-C7"),
        ("S-B", "sup_B_select", "sup_B_yoked", both, "C1-C7 + T1-T4"),
        ("S-BC", "sup_B_select", "sup_C_contact_gate", both, "C1-C7 + T1-T4"),
        ("AO", "contact_learned", "never_release", trans, "T1-T4"),
    ]
    specificity = pd.DataFrame([
        {"check": c, "configuration": a, "reference": b, "conditions": label,
         **one_sided(frame, a, b, 0.0, n, s), **_outcomes(frame, a, b)}
        for c, a, b, frame, label in spec_rows])  # fmt: skip

    gates = []
    for name in (*SUPERVISORS, "never_release", "sup_A_yoked", "sup_B_yoked"):
        for cond in config.harm_conditions:
            sub = episodes[episodes["condition"] == cond]
            r, o = one_sided(sub, name, "base_only", config.harm_ci_floor, n, s), _outcomes(
                sub, name, "base_only")  # fmt: skip
            gates.append({"configuration": name, "condition": cond, **r, **o,
                          "binding": name in SUPERVISORS,
                          "passed": r["ci_low"] > config.harm_ci_floor
                          and o["harmed_fraction"] <= config.harm_max_fraction})  # fmt: skip
    safety = pd.DataFrame(gates)

    by_condition = []
    for cond_key, sub in episodes.groupby("condition"):
        for config_name in CONFIGURATIONS[1:]:
            for ref in ("base_only", "contact_learned"):
                if config_name == ref:
                    continue
                by_condition.append({"condition": cond_key, "configuration": config_name,
                                     "reference": ref,
                                     **one_sided(sub, config_name, ref, 0.0, n, s),
                                     **_outcomes(sub, config_name, ref)})  # fmt: skip

    latency, false_act, false_rel, switching, oscillation, compute = [], [], [], [], [], []
    for group, ep in steps.sort_values(EPISODE + ["t"]).groupby(EPISODE):
        name, pol, cond, seed = str(group[0]), group[1], str(group[2]), group[3]
        ep = ep.reset_index(drop=True)
        dt = ep["delay_true"].to_numpy(int)
        act = ep["active"].to_numpy(bool)
        right = act & (ep["delay_param"].to_numpy(int) == dt)
        for t in range(len(ep)):
            prev = dt[t - 1] if t else 0
            if dt[t] != prev and dt[t] > 0:  # onset or change of value
                hit = np.flatnonzero(right[t:])
                latency.append({"configuration": name, "condition": cond, "kind": "engage",
                                "change_step": t, "to": int(dt[t]),
                                "latency": int(hit[0]) if len(hit) else np.nan,
                                "never": not len(hit)})  # fmt: skip
            if dt[t] == 0 and prev > 0:  # end of a delay
                hit = np.flatnonzero(~act[t:])
                latency.append({"configuration": name, "condition": cond, "kind": "release",
                                "change_step": t, "to": 0,
                                "latency": int(hit[0]) if len(hit) else np.nan,
                                "never": not len(hit)})  # fmt: skip
        no_delay = dt == 0
        false_act.append({"configuration": name, "condition": cond,
                          "steps_without_delay": int(no_delay.sum()),
                          "false_active_steps": int((act & no_delay).sum()),
                          "episode_has_false_activation":
                              bool((act & no_delay).any())})  # fmt: skip
        engaged = np.maximum.accumulate(act) & (dt > 0)
        false_rel.append({"configuration": name, "condition": cond,
                          "steps_after_engagement": int(engaged.sum()),
                          "off_after_engagement": int((engaged & ~act).sum())})  # fmt: skip
        count, dwells = _changes(ep)
        switching.append({"configuration": name, "condition": cond, "switches": count,
                          "min_dwell": min(dwells), "active_share": float(act.mean()),
                          "learned_share": float((ep["predictor"] == "learned").mean()),
                          "phase_e_share": float((ep["predictor"] == "phase_e").mean()),
                          "released_share": float(ep["released"].mean())})  # fmt: skip
        if count >= 6:
            oscillation.append({"configuration": name, "policy_seed": pol, "condition": cond,
                                "episode_seed": seed, "switches": count})  # fmt: skip
        overhead = (ep["est_ns"] + ep["ctrl_ns"]) if name != "base_only" else ep["ctrl_ns"] * 0
        compute.append({"configuration": name, "condition": cond,
                        "overhead_us_mean": float(overhead.mean() / 1e3),
                        "learned_calls": int(ep["learned_calls"].sum()),
                        "mlp_steps": int(ep["mlp_steps"].sum())})  # fmt: skip

    lat = pd.DataFrame(latency)
    latency_table = (lat.groupby(["configuration", "condition", "kind"])
                     .agg(events=("never", "size"), never=("never", "sum"),
                          median=("latency", "median"),
                          p90=("latency", lambda x: x.quantile(0.9)))
                     .reset_index()) if len(lat) else pd.DataFrame()  # fmt: skip
    fa = (pd.DataFrame(false_act).groupby(["configuration", "condition"])
          .agg(steps_without_delay=("steps_without_delay", "sum"),
               false_active_steps=("false_active_steps", "sum"),
               episodes_with_false_activation=("episode_has_false_activation", "sum"))
          .reset_index())  # fmt: skip
    fa["false_active_rate"] = fa["false_active_steps"] / fa["steps_without_delay"].clip(lower=1)
    fr = (pd.DataFrame(false_rel).groupby(["configuration", "condition"])
          .sum(numeric_only=True).reset_index())  # fmt: skip
    fr["false_release_rate"] = fr["off_after_engagement"] / fr["steps_after_engagement"].clip(
        lower=1)  # fmt: skip
    sw = (pd.DataFrame(switching).groupby(["configuration", "condition"])
          .agg(switches_mean=("switches", "mean"), switches_max=("switches", "max"),
               episodes_ge4=("switches", lambda x: int((x >= 4).sum())),
               episodes_ge6=("switches", lambda x: int((x >= 6).sum())),
               min_dwell=("min_dwell", "min"), active_share=("active_share", "mean"),
               learned_share=("learned_share", "mean"), phase_e_share=("phase_e_share", "mean"),
               released_share=("released_share", "mean"))
          .reset_index())  # fmt: skip
    per_step = steps.assign(
        overhead_us=np.where(
            steps["configuration"] == "base_only", 0.0, (steps["est_ns"] + steps["ctrl_ns"]) / 1e3
        )
    )
    comp = (
        per_step.groupby("configuration")["overhead_us"]
        .agg(median="median", p99=lambda x: x.quantile(0.99), mean="mean")
        .reset_index()
    )
    calls = (pd.DataFrame(compute).groupby("configuration")
             .agg(learned_calls_per_episode=("learned_calls", "mean"),
                  mlp_steps_per_episode=("mlp_steps", "mean"),
                  overhead_us_per_step_episode_mean=("overhead_us_mean", "mean"))
             .reset_index())  # fmt: skip
    computation = comp.merge(calls, on="configuration")

    b = steps[(steps["configuration"] == "sup_B_select") & steps["err_learned"].notna()]
    pred = (b.groupby(["condition", "near_true"])
            .agg(scored_steps=("err_learned", "size"), learned_error=("err_learned", "mean"),
                 phase_e_error=("err_phase_e", "mean"))
            .reset_index()) if len(b) else pd.DataFrame()  # fmt: skip

    tables = {"hypotheses": hyp, "specificity": specificity, "safety_gates": safety,
              "by_condition": pd.DataFrame(by_condition), "latency": latency_table,
              "false_activations": fa, "false_releases": fr, "switching": sw,
              "oscillation_cases": pd.DataFrame(oscillation, columns=[
                  "configuration", "policy_seed", "condition", "episode_seed", "switches"]),
              "computation": computation, "prediction_errors": pred}  # fmt: skip
    binding = safety[safety["binding"]]
    summary = {
        "hypotheses": dict(zip(hyp["hypothesis"], hyp["status"], strict=True)),
        "safety_gates_passed": {k: bool(v) for k, v in
                                binding.groupby("configuration")["passed"].all().items()},
        "call_reduction_G4": float(reduction),
        "oscillation_episodes": len(oscillation),
        "plan_sha256": config.plan_sha256, "model_sha256": config.model_sha256,
        "cusum_threshold": config.supervisor.get("cusum_threshold"),
        "episodes": len(episodes), "step_rows": len(steps),
        "seeds": [int(min(config.seeds)), int(max(config.seeds))],
    }  # fmt: skip
    return tables, summary
