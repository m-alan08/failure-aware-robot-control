"""Phase E2: pre-registered evaluation of online delay compensation.

The plan is ``docs/PHASE_E_DELAY_PREREGISTRATION.md``. The config pins its SHA-256, and loading
fails if the plan has changed. The estimator and compensator are the Phase E pilot's,
unchanged (amendment A1).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.compensation import CompensationConfig, Fixed, LatchedEstimate, Parameters
from farc.adaptation.estimator import ActuatorEstimator, Estimate, EstimatorConfig
from farc.adaptation.experiment import (
    USED_SEED_RANGES,
    AdaptationConfigError,
    Oracle,
    _estimation_summary,
    run_episode,
)
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.counterfactual.session import EpisodeSession
from farc.disturbances.config import SEED_RANGES, ConditionSpec, DisturbanceSpec, load_experiment
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve

E2_USED_SEED_RANGES = {
    **USED_SEED_RANGES,
    "phase_e_pilot_selection": (3_631_000, 3_631_020),
    "phase_e_pilot_evaluation": (3_632_000, 3_632_030),
}


@dataclass(frozen=True)
class DelayEvaluation:
    name: str
    plan: str
    plan_sha256: str
    conditions_from: str
    conditions: tuple[dict[str, Any], ...]
    delay_conditions: tuple[str, ...]
    harm_conditions: tuple[str, ...]
    no_cue_condition: str
    checkpoints: tuple[str, ...]
    debug_seed_offset: int
    debug_episodes: int
    evaluation_seed_offset: int
    evaluation_episodes: int
    perturbation_seed: int
    estimator: dict[str, Any]
    gain_floor: float
    activate_below: float
    sham_delay: int
    shuffle_seed: int
    min_effect: float
    harm_ci_floor: float
    harm_max_fraction: float
    bootstrap_resamples: int
    bootstrap_seed: int
    output: str
    debug_run: bool = field(default=False, compare=False)

    @property
    def debug_seeds(self) -> list[int]:
        return list(range(self.debug_seed_offset, self.debug_seed_offset + self.debug_episodes))

    @property
    def evaluation_seeds(self) -> list[int]:
        o = self.evaluation_seed_offset
        return list(range(o, o + self.evaluation_episodes))

    @property
    def seeds(self) -> list[int]:
        return self.debug_seeds if self.debug_run else self.evaluation_seeds


def _check_seeds(seeds: list[int], label: str) -> None:
    low, high = SEED_RANGES["validation"]
    if not seeds or not (low <= min(seeds) and max(seeds) < high):
        raise AdaptationConfigError(f"{label} seeds must lie in the validation range")
    held_low, held_high = SEED_RANGES["heldout"]
    if any(held_low <= s < held_high for s in seeds):
        raise AdaptationConfigError("held-out seeds are reserved")
    for name, (a, b) in E2_USED_SEED_RANGES.items():
        if any(a <= s < b for s in seeds):
            raise AdaptationConfigError(f"{label} seeds overlap {name} [{a}, {b})")


def validate_delay_evaluation(config: DelayEvaluation, check_plan: bool = True):
    _check_seeds(config.debug_seeds, "debugging")
    _check_seeds(config.evaluation_seeds, "evaluation")
    if set(config.debug_seeds) & set(config.evaluation_seeds):
        raise AdaptationConfigError("debugging and evaluation seeds overlap")
    names = [c["name"] for c in config.conditions]
    if len(set(names)) != len(names):
        raise AdaptationConfigError("duplicate condition names")
    for group in (config.delay_conditions, config.harm_conditions, (config.no_cue_condition,)):
        if set(group) - set(names):
            raise AdaptationConfigError(f"unknown conditions {sorted(set(group) - set(names))}")
    if check_plan:
        digest = hashlib.sha256(Path(config.plan).read_bytes()).hexdigest()
        if digest != config.plan_sha256:
            raise AdaptationConfigError(
                "the pre-registration changed since the config pinned it; record an amendment"
            )
    return config


def load_delay_evaluation(path: str | Path, check_plan: bool = True) -> DelayEvaluation:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    for key in ("conditions", "delay_conditions", "harm_conditions", "checkpoints"):
        values[key] = tuple(values[key])
    return validate_delay_evaluation(DelayEvaluation(**values), check_plan)


def build_conditions(config: DelayEvaluation) -> list[ConditionSpec]:
    """Named Phase 1 conditions (``from``) or explicit disturbance lists (combined, A2)."""
    library = {c.name: c for c in load_experiment(resolve(config.conditions_from)).conditions}
    out = []
    for entry in config.conditions:
        if "from" in entry:
            out.append(ConditionSpec(entry["name"], library[entry["from"]].disturbances))
        else:
            specs = tuple(DisturbanceSpec(**{**d, "fields": tuple(d.get("fields",
                                                                    ("observation",)))})
                          for d in entry["disturbances"])  # fmt: skip
            out.append(ConditionSpec(entry["name"], specs))
    return out


# --- parameter sources ---------------------------------------------------------------------


class ShuffledDelay:
    """SHAM: opens when this episode's own delay estimate first becomes reliable (same gating
    as the deployable controller), then predicts with the delay that a *different* episode's
    estimator had latched at the same step (amendment A3)."""

    def __init__(self, partner_latched: list[int | None]):
        self.partner = partner_latched
        self.open = False

    def parameters(self, estimate: Estimate) -> Parameters:
        self.open |= estimate.delay_reliable
        step = min(estimate.step, len(self.partner) - 1)
        return Parameters(None, self.partner[step] if self.open else None)


def latched_delays(steps: pd.DataFrame) -> list[int | None]:
    """The delay a LatchedEstimate would hold at each step, from one episode's step log."""
    current: int | None = None
    out: list[int | None] = []
    for reliable, delay in zip(steps["delay_reliable"], steps["delay_est"], strict=True):
        if reliable:
            current = int(delay)
        out.append(current)
    return out


def derangement(n: int, seed: int) -> np.ndarray:
    """A seeded permutation with no fixed points."""
    if n < 2:
        raise ValueError("a derangement needs at least two items")
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            return perm


CONFIGURATIONS = ("base_only", "estimated_delay", "estimated_combined_f0.5", "oracle_delay",
                  "sham_delay_fixed2", "sham_delay_shuffled")  # fmt: skip
LOGGED = ("base_only", "estimated_delay")


def compensation_for(name: str, config: DelayEvaluation) -> CompensationConfig:
    floor, below = config.gain_floor, config.activate_below
    return {
        "base_only": CompensationConfig(False, False, floor, below),
        "estimated_delay": CompensationConfig(False, True, floor, below),
        "estimated_combined_f0.5": CompensationConfig(True, True, floor, below),
        "oracle_delay": CompensationConfig(False, True, floor, below),
        "sham_delay_fixed2": CompensationConfig(False, True, floor, below),
        "sham_delay_shuffled": CompensationConfig(False, True, floor, below),
    }[name]


def run_policy(args: tuple[DelayEvaluation, str]) -> tuple[list[dict], list[dict]]:
    config, checkpoint = args
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    actor_state = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    policy = deterministic_policy_function(loaded.actor)
    normalizer = FrozenNormalizer(loaded.normalizer)
    env = make_env(loaded.env_config)
    est_config = EstimatorConfig(**config.estimator)
    seeds = config.seeds
    episodes: list[dict[str, Any]] = []
    logged: list[dict[str, Any]] = []
    for c_index, condition in enumerate(build_conditions(config)):
        partner = derangement(len(seeds), config.shuffle_seed + c_index)
        base_latched: dict[int, list[int | None]] = {}
        for name in CONFIGURATIONS:  # base_only first: the shuffled sham needs its estimates
            for i, seed in enumerate(seeds):
                source: Any
                if name == "oracle_delay":
                    source = Oracle(condition)
                elif name == "sham_delay_fixed2":
                    source = Fixed(None, config.sham_delay, gate_on_reliability=True)
                elif name == "sham_delay_shuffled":
                    source = ShuffledDelay(base_latched[seeds[partner[i]]])
                elif name == "base_only":
                    source = Fixed(None, None)
                else:
                    source = LatchedEstimate()
                session = EpisodeSession(env, policy, normalizer, condition, seed,
                                         config.perturbation_seed)  # fmt: skip
                result = run_episode(session, ActuatorEstimator(est_config),
                                     compensation_for(name, config), source, condition)  # fmt: skip
                steps = result["steps"]
                if name == "base_only":
                    base_latched[seed] = latched_delays(steps)
                key = {"configuration": name, "policy_seed": loaded.training_seed,
                       "condition": condition.name, "episode_seed": seed}  # fmt: skip
                episodes.append({**key, "success": result["success"],
                                 "final_distance": result["final_distance"],
                                 "command_digest": result["command_digest"],
                                 "near_object_rate": steps["near_object"].mean(),
                                 "delay_mismatch_rate": float((steps["delay_param"].clip(lower=0)
                                                               != steps["delay_true"]).mean()),
                                 **({"partner_seed": seeds[partner[i]]}
                                    if name == "sham_delay_shuffled" else {}),
                                 **_estimation_summary(steps)})  # fmt: skip
                if name in LOGGED:
                    logged.extend({**key, **row} for row in steps.to_dict("records"))
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor_state[k]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes, logged


# --- pre-registered analysis ----------------------------------------------------------------

KEYS = ["policy_seed", "condition", "episode_seed"]


def paired_difference(episodes: pd.DataFrame, a: str, b: str, resamples: int, seed: int):
    """Success difference a - b on the same episodes, episode-seed cluster bootstrap.
    Returns estimate, 95% CI, two-sided bootstrap p-value, episodes."""
    wide = episodes.pivot_table(index=KEYS, columns="configuration", values="success",
                                aggfunc="first").astype(float)  # fmt: skip
    diff = (wide[a] - wide[b]).dropna()
    groups = diff.groupby(level="episode_seed")
    sums, counts = groups.sum().to_numpy(), groups.count().to_numpy()
    estimate = sums.sum() / counts.sum()
    picks = np.random.default_rng(seed).integers(0, len(sums), size=(resamples, len(sums)))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    p = min(1.0, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return float(estimate), float(low), float(high), float(p), int(len(diff))


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted.tolist()


def _outcomes(episodes: pd.DataFrame, name: str) -> dict[str, Any]:
    base = episodes[episodes["configuration"] == "base_only"].set_index(KEYS)["success"]
    this = episodes[episodes["configuration"] == name].set_index(KEYS)["success"]
    a, b = base.loc[this.index].to_numpy(bool), this.to_numpy(bool)
    return {"success_rate": b.mean(), "base_success_rate": a.mean(),
            "rescued": int((~a & b).sum()), "base_failures": int((~a).sum()),
            "harmed": int((a & ~b).sum()), "base_successes": int(a.sum()),
            "harmed_fraction_of_base_successes": (a & ~b).sum() / max(a.sum(), 1)}  # fmt: skip


DERIVED_TABLES = ("hypotheses", "by_condition", "by_policy_seed", "identification",
                  "onset_latency", "contact_harm")  # fmt: skip


def derive_tables(episodes: pd.DataFrame, steps: pd.DataFrame, config: DelayEvaluation):
    n, s = config.bootstrap_resamples, config.bootstrap_seed
    primary = "estimated_delay"
    delay = episodes[episodes["condition"].isin(config.delay_conditions)]

    def test(frame, a, b):
        return paired_difference(frame, a, b, n, s)

    h1 = test(delay, primary, "base_only")
    cue = test(episodes[episodes["condition"] == config.no_cue_condition], primary, "base_only")
    spec = test(delay, primary, "sham_delay_shuffled")
    rows = [
        {"test": "H1_pooled_delay_vs_base", "criterion": f"ci_low > {config.min_effect}",
         **dict(zip(["estimate", "ci_low", "ci_high", "p", "episodes"], h1, strict=True)),
         "passed": h1[1] > config.min_effect},
        {"test": f"no_start_cue_{config.no_cue_condition}_vs_base", "criterion": "ci_low > 0",
         **dict(zip(["estimate", "ci_low", "ci_high", "p", "episodes"], cue, strict=True)),
         "passed": cue[1] > 0},
        {"test": "specificity_vs_shuffled_sham", "criterion": "ci_low > 0",
         **dict(zip(["estimate", "ci_low", "ci_high", "p", "episodes"], spec, strict=True)),
         "passed": spec[1] > 0},
    ]  # fmt: skip
    for cond in config.harm_conditions:
        sub = episodes[episodes["condition"] == cond]
        r = test(sub, primary, "base_only")
        harmed = _outcomes(sub, primary)["harmed_fraction_of_base_successes"]
        rows.append({"test": f"harm_{cond}",
                     "criterion": f"ci_low > {config.harm_ci_floor} and harmed <= "
                                  f"{config.harm_max_fraction}",
                     **dict(zip(["estimate", "ci_low", "ci_high", "p", "episodes"], r,
                                strict=True)),
                     "harmed_fraction": harmed,
                     "passed": r[1] > config.harm_ci_floor
                     and harmed <= config.harm_max_fraction})  # fmt: skip
    hypotheses = pd.DataFrame(rows)
    supported = bool(hypotheses["passed"].all())

    by_condition = []
    for cond_key, sub in episodes.groupby("condition"):
        for name in CONFIGURATIONS[1:]:
            est, low, high, p, count = test(sub, name, "base_only")
            part = sub[sub["configuration"] == name]
            by_condition.append({"condition": cond_key, "configuration": name, "difference": est,
                                 "ci_low": low, "ci_high": high, "p": p, "episodes": count,
                                 **_outcomes(sub, name),
                                 "delay_activation_rate": part["delay_activation_rate"].mean(),
                                 "gain_activation_rate": part["gain_activation_rate"].mean(),
                                 "delay_mismatch_rate": part["delay_mismatch_rate"].mean(),
                                 "compensation_saturation_rate":
                                     part["compensation_saturation_rate"].mean()})  # fmt: skip
    table = pd.DataFrame(by_condition)
    mask = (table["configuration"] == primary) & table["condition"].isin(config.delay_conditions)
    table["p_holm_primary"] = np.nan
    table.loc[mask, "p_holm_primary"] = holm(table.loc[mask, "p"].tolist())
    oracle = table[table["configuration"] == "oracle_delay"].set_index("condition")["difference"]
    table["share_of_oracle"] = [r.difference / oracle[r.condition] if abs(oracle[r.condition])
                                > 1e-12 else np.nan for r in table.itertuples()]  # fmt: skip

    by_policy = []
    for (pol,), sub in delay.groupby(["policy_seed"]):
        for name in CONFIGURATIONS[1:]:
            est, low, high, p, count = test(sub, name, "base_only")
            by_policy.append({"policy_seed": pol, "configuration": name, "difference": est,
                              "ci_low": low, "ci_high": high, "episodes": count})  # fmt: skip

    base_steps = steps[steps["configuration"] == "base_only"]
    ident = []
    for ident_key, sub in base_steps.groupby("condition"):
        rel = sub[sub["delay_reliable"]]
        ident.append({"condition": ident_key, "steps": len(sub),
                      "delay_reliable_rate": sub["delay_reliable"].mean(),
                      "delay_accuracy_when_reliable": (rel["delay_est"] == rel["delay_true"])
                      .mean() if len(rel) else np.nan,
                      "delay_accuracy_unconditional": (sub["delay_est"] == sub["delay_true"])
                      .mean(),
                      "wrong_lag_reliable_steps": int((rel["delay_est"] != rel["delay_true"])
                                                      .sum()),
                      "gain_reliable_rate": sub["gain_reliable"].mean()})  # fmt: skip

    latency = []
    applied = steps[steps["configuration"] == primary]
    for (lat_cond, pol, seed), sub in applied.groupby(["condition", "policy_seed", "episode_seed"]):
        sub = sub.sort_values("t")
        true = sub["delay_true"].to_numpy()
        used = sub["delay_param"].clip(lower=0).to_numpy()
        for change in np.flatnonzero(np.diff(true)) + 1:  # steps where the true delay changes
            after = np.flatnonzero(used[change:] == true[change])
            latency.append({"condition": lat_cond, "policy_seed": pol, "episode_seed": seed,
                            "change_step": int(change), "from": int(true[change - 1]),
                            "to": int(true[change]),
                            "latency": int(after[0]) if len(after) else np.nan,
                            "matched_by_end": bool(len(after))})  # fmt: skip
    onset = pd.DataFrame(latency, columns=["condition", "policy_seed", "episode_seed",
                                           "change_step", "from", "to", "latency",
                                           "matched_by_end"])  # fmt: skip
    if len(onset):
        onset = onset.groupby(["condition", "change_step", "from", "to"]).agg(
            episodes=("latency", "size"), matched_rate=("matched_by_end", "mean"),
            latency_median=("latency", "median"),
            latency_p90=("latency", lambda x: x.quantile(0.9))).reset_index()  # fmt: skip

    contact = []
    sub = episodes[episodes["condition"].isin(config.delay_conditions)]
    wide = sub.pivot_table(index=KEYS, columns="configuration", values="success",
                           aggfunc="first")  # fmt: skip
    near = sub[sub["configuration"] == primary].set_index(KEYS)["near_object_rate"]
    for label, mask_ in (
        ("harmed", wide["base_only"].astype(bool) & ~wide[primary].astype(bool)),
        ("kept", wide["base_only"].astype(bool) & wide[primary].astype(bool)),
    ):
        contact.append({"group": f"base_success_{label}", "episodes": int(mask_.sum()),
                        "near_object_rate_mean": float(near[mask_].mean())
                        if mask_.any() else np.nan})  # fmt: skip

    tables = {"hypotheses": hypotheses, "by_condition": table,
              "by_policy_seed": pd.DataFrame(by_policy), "identification": pd.DataFrame(ident),
              "onset_latency": onset, "contact_harm": pd.DataFrame(contact)}  # fmt: skip
    summary = {"supported": supported, "primary": primary,
               "plan_sha256": config.plan_sha256, "episodes": len(episodes),
               "step_rows": len(steps)}  # fmt: skip
    return tables, summary
