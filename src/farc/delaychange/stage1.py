"""G2 Stage 1: mechanism isolation (docs/PHASE_G2_PROPOSAL.md, section 3) and its pre-registered
decision rules (section 3.4)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.adaptation.experiment import AdaptationConfigError
from farc.base_policy.checkpoint import deterministic_policy_function, load_policy
from farc.contact.dynamics import load_dynamics
from farc.contact.episodes import learned_predictor, phase_e_predictor
from farc.delaychange.episodes import DIAGNOSTICS, Diagnostic, branch, run_diagnostic_episode
from farc.delaychange.schedule import DelaySchedule, ScheduledDelaySession
from farc.disturbances.config import ConditionSpec
from farc.disturbances.rollout import FrozenNormalizer
from farc.envs.factory import make_env
from farc.paths import resolve
from farc.supervision.seeds import check_g2_seeds

KEYS = ["policy_seed", "condition", "episode_seed"]
RELEASE_KS = (0, 1, 2, 3, 5, 7)


@dataclass(frozen=True)
class Cell:
    name: str
    pulses: tuple[tuple[int, int | None, int], ...]
    semantics: str
    group: str
    configurations: tuple[str, ...]

    @property
    def schedule(self) -> DelaySchedule:
        return DelaySchedule(self.pulses)


TRANSIENT_CONFIGS = ("base_only", "deployed_phase_e", "deployed_learned", "never_release",
                     *[f"release_k{k}" for k in RELEASE_KS], "true_phase_e", "true_learned",
                     "true_sim", "latched_sim", "schedule_sim")  # fmt: skip


def stage1_cells() -> list[Cell]:
    """The pre-registered Stage-1 matrix (section 3.1-3.2)."""
    cells = []
    for onset in (6, 18):
        for duration in (4, 8, 16):
            pulses = ((onset, duration, 3),)
            cells.append(Cell(f"P3_on{onset}_dur{duration}", pulses, "replay_drop", "transient",
                              TRANSIENT_CONFIGS))  # fmt: skip
            cells.append(Cell(f"P3_on{onset}_dur{duration}_drain", pulses, "drain_hold",
                              "transient_drain",
                              ("base_only", "deployed_learned", "true_learned")))  # fmt: skip
    cells.append(Cell("D3_on6", ((6, None, 3),), "replay_drop", "persistent",
                      ("base_only", "deployed_phase_e", "deployed_learned", "never_release",
                       "true_phase_e", "true_learned", "true_sim", "latched_sim",
                       "forced_off_L3", "forced_off_L6")))  # fmt: skip
    cells.append(Cell("U", (), "replay_drop", "undisturbed",
                      ("base_only", "deployed_phase_e", "deployed_learned", "forced_on_L3",
                       "forced_on_L6", "forced_on_L9")))  # fmt: skip
    cells.append(Cell("I3_6to15_26to35", ((6, 10, 3), (26, 10, 3)), "replay_drop",
                      "intermittent", ("base_only", "deployed_learned", "true_learned",
                                       "latched_sim", "true_sim", "schedule_sim")))  # fmt: skip
    return cells


@dataclass(frozen=True)
class Stage1Config:
    name: str
    plan: str
    plan_sha256: str
    model: str
    model_sha256: str
    checkpoints: tuple[str, ...]
    debug_seed_offset: int
    debug_episodes: int
    mechanism_seed_offset: int
    mechanism_episodes: int
    perturbation_seed: int
    estimator: dict[str, Any]
    branch_span: int
    bootstrap_resamples: int
    bootstrap_seed: int
    output: str
    debug_run: bool = field(default=False, compare=False)

    @property
    def seeds(self) -> list[int]:
        if self.debug_run:
            return list(range(self.debug_seed_offset, self.debug_seed_offset + self.debug_episodes))
        o = self.mechanism_seed_offset
        return list(range(o, o + self.mechanism_episodes))


def load_stage1(path: str | Path, debug: bool = False) -> Stage1Config:
    values = json.loads(Path(path).read_text())
    values.pop("description", None)
    values["checkpoints"] = tuple(values["checkpoints"])
    config = Stage1Config(**values, debug_run=debug)
    check_g2_seeds(Stage1Config(**values, debug_run=True).seeds, "debugging")
    check_g2_seeds(Stage1Config(**values, debug_run=False).seeds, "mechanism")
    if hashlib.sha256(Path(config.plan).read_bytes()).hexdigest() != config.plan_sha256:
        raise AdaptationConfigError("the G2 plan changed since the config pinned it")
    return config


def _setup(config: Stage1Config, checkpoint: str):
    torch.set_num_threads(1)
    loaded = load_policy(resolve(checkpoint))
    model = load_dynamics(resolve(config.model), config.model_sha256)
    est = EstimatorConfig(**config.estimator)
    preds = {"learned": learned_predictor(model, move_object=True),
             "phase_e": phase_e_predictor(est.nominal_gain)}  # fmt: skip
    return (
        loaded,
        deterministic_policy_function(loaded.actor),
        FrozenNormalizer(loaded.normalizer),
        make_env(loaded.env_config),
        est,
        preds,
    )


def _episode(config, env, policy, normalizer, est, preds, cell: Cell, seed: int,
             diag: Diagnostic, policy_seed: int):  # fmt: skip
    session = ScheduledDelaySession(env, policy, normalizer, ConditionSpec(cell.name, ()), seed,
                                    config.perturbation_seed, cell.schedule,
                                    cell.semantics)  # type: ignore[arg-type]  # fmt: skip
    result = run_diagnostic_episode(session, ActuatorEstimator(est), diag, preds)
    key = {"configuration": diag.name, "policy_seed": policy_seed, "condition": cell.name,
           "group": cell.group, "semantics": cell.semantics, "episode_seed": seed,
           "privileged": diag.privileged}  # fmt: skip
    return ({**key, "success": result["success"], "final_distance": result["final_distance"]},
            [{**key, **r} for r in result["rows"]])  # fmt: skip


def run_task(args: tuple[Stage1Config, str, str]):
    config, checkpoint, cell_name = args
    loaded, policy, normalizer, env, est, preds = _setup(config, checkpoint)
    actor = {k: v.clone() for k, v in loaded.actor.state_dict().items()}
    cell = {c.name: c for c in stage1_cells()}[cell_name]
    episodes, steps = [], []
    for seed in config.seeds:
        for name in cell.configurations:
            e, s = _episode(config, env, policy, normalizer, est, preds, cell, seed,
                            DIAGNOSTICS[name], loaded.training_seed)  # fmt: skip
            episodes.append(e)
            steps.extend(s)
    for k, v in loaded.actor.state_dict().items():
        if not torch.equal(v, actor[k]):
            raise RuntimeError("the frozen actor changed during the experiment")
    return episodes, steps


def branch_targets(episodes: pd.DataFrame) -> pd.DataFrame:
    """Episodes for the point-of-no-return branches: transient (replay_drop) episodes where
    frozen SAC succeeds and the deployed learned compensator fails."""
    sub = episodes[episodes["group"] == "transient"]
    w = sub.pivot_table(index=KEYS, columns="configuration", values="success",
                        aggfunc="first").astype(bool)  # fmt: skip
    harmed = w[w["base_only"] & ~w["deployed_learned"]].reset_index()
    return harmed[KEYS]


def run_branch_task(args: tuple[Stage1Config, str, list[tuple[str, int]]]):
    """For each target episode, rerun the deployed controller and switch to PRIVILEGED
    true-delay learned compensation from step t, for t = end .. end + branch_span."""
    config, checkpoint, targets = args
    loaded, policy, normalizer, env, est, preds = _setup(config, checkpoint)
    cells = {c.name: c for c in stage1_cells()}
    out = []
    for cell_name, seed in targets:
        cell = cells[cell_name]
        end = cell.schedule.ends(50)[0]
        for t in range(end, end + config.branch_span + 1):
            e, _ = _episode(config, env, policy, normalizer, est, preds, cell, seed, branch(t),
                            loaded.training_seed)  # fmt: skip
            out.append({**e, "switch_step": t, "end_step": end})
    return out


# --- pre-registered analysis (section 3.4) ---------------------------------------------------


def _wide(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.pivot_table(index=KEYS, columns="configuration", values="success",
                             aggfunc="first").astype(float)  # fmt: skip


def _boot(per_seed: pd.DataFrame, statistic, n: int, seed: int):
    """Cluster bootstrap over episode seeds. ``per_seed`` holds per-seed sums (and counts) of
    every column; ``statistic`` maps column means to a number."""
    groups = per_seed.groupby(level="episode_seed")
    sums, counts = groups.sum(), groups.count()
    est = statistic(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sums), size=(n, len(sums)))
    s, c = sums.to_numpy(), counts.to_numpy()
    draws = np.array([statistic(pd.Series(s[i].sum(0) / c[i].sum(0), index=sums.columns))
                      for i in idx])  # fmt: skip
    low, high = np.nanquantile(draws, [0.025, 0.975])
    return float(est), float(low), float(high), draws


def diff(frame: pd.DataFrame, a: str, b: str, n: int, seed: int) -> dict[str, float]:
    """Paired success difference a - b, episode-seed cluster bootstrap (vectorised)."""
    w = _wide(frame)[[a, b]].dropna()
    d = (w[a] - w[b]).groupby(level="episode_seed")
    sums, counts = d.sum().to_numpy(), d.count().to_numpy()
    est = sums.sum() / counts.sum()
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n, len(sums)))
    draws = sums[idx].sum(1) / counts[idx].sum(1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {"estimate": est, "ci_low": low, "ci_high": high, "episodes": len(w),
            "rescued": int(((w[b] == 0) & (w[a] == 1)).sum()),
            "harmed": int(((w[b] == 1) & (w[a] == 0)).sum())}  # fmt: skip


def decision_rules(episodes: pd.DataFrame, n: int, seed: int) -> dict[str, Any]:
    tr = episodes[episodes["group"] == "transient"]
    w = _wide(tr)
    cols = ["base_only", "deployed_learned", *[f"release_k{k}" for k in RELEASE_KS]]
    w = w[cols].dropna()

    def share(m):
        gap = m["base_only"] - m["deployed_learned"]
        return (m["release_k0"] - m["deployed_learned"]) / gap if abs(gap) > 1e-12 else np.nan

    est, low, high, _ = _boot(w, share, n, seed)
    ks = np.array(RELEASE_KS, float)

    def slope(m):
        y = np.array([m[f"release_k{k}"] for k in RELEASE_KS])
        return float(np.polyfit(ks, y, 1)[0])

    s_est, s_low, s_high, _ = _boot(w, slope, n, seed)
    stale_main = bool(low > 0.5 and s_high < 0)

    drain = episodes[episodes["group"] == "transient_drain"].assign(
        condition=lambda f: f["condition"].str.replace("_drain", "", regex=False))  # fmt: skip
    a = _wide(tr)[["base_only", "deployed_learned"]].add_suffix("_replay")
    b = _wide(drain)[["base_only", "deployed_learned"]].add_suffix("_drain")
    both = a.join(b, how="inner").dropna()

    def harm_change(m):
        return ((m["deployed_learned_drain"] - m["base_only_drain"])
                - (m["deployed_learned_replay"] - m["base_only_replay"]))  # fmt: skip

    p_est, p_low, p_high, _ = _boot(both, harm_change, n, seed)
    pipeline = bool(abs(p_est) > 0.02 and (p_low > 0 or p_high < 0))
    return {
        "recovered_share_k0": {"estimate": est, "ci_low": low, "ci_high": high},
        "release_slope_per_step": {"estimate": s_est, "ci_low": s_low, "ci_high": s_high},
        "stale_compensation_is_main_cause": stale_main,
        "harm_change_drain_minus_replay": {"estimate": p_est, "ci_low": p_low,
                                           "ci_high": p_high},
        "pipeline_dependence_flagged": pipeline,
        "gap_base_minus_deployed": float(w["base_only"].mean() - w["deployed_learned"].mean()),
    }  # fmt: skip


def mismatch_metrics(steps: pd.DataFrame) -> pd.DataFrame:
    """Per configuration and transition: first mismatched decision and stale duration."""
    rows = []
    for (cfg, _pol, cond, _seed), ep in steps.sort_values("t").groupby(
            ["configuration", "policy_seed", "condition", "episode_seed"]):  # fmt: skip
        dt = ep["delay_true"].to_numpy(int)
        du = np.where(ep["active"].to_numpy(bool), ep["delay_used"].to_numpy(int), 0)
        for t in range(1, len(dt)):
            if dt[t] == dt[t - 1]:
                continue
            nxt = next((u for u in range(t + 1, len(dt)) if dt[u] != dt[t]), len(dt))
            wrong = du[t:nxt] != dt[t:nxt]
            first = int(np.argmax(wrong)) if wrong.any() else None
            rows.append({"configuration": cfg, "condition": cond,
                         "kind": "end" if dt[t] == 0 else "onset" if dt[t - 1] == 0 else "change",
                         "first_mismatch": first, "mismatched_steps": int(wrong.sum()),
                         "segment_steps": int(nxt - t)})  # fmt: skip
    return pd.DataFrame(rows)


def prediction_errors(steps: pd.DataFrame) -> pd.DataFrame:
    """M3: at every active step, the predicted gripper/cube position against the true state at
    t (the right target when there is no delay) and at t + d_used."""
    out = []
    act = steps[steps["active"]].sort_values("t")
    full = steps.set_index(["configuration", "policy_seed", "condition", "episode_seed", "t"])
    for key in ("gripper_position", "object_position"):
        tcols = [f"true_{key}{i}" for i in range(3)]
        pcols = [f"pred_{key}{i}" for i in range(3)]
        now = np.linalg.norm(act[pcols].to_numpy() - act[tcols].to_numpy(), axis=1)
        idx = pd.MultiIndex.from_arrays([act["configuration"], act["policy_seed"],
                                         act["condition"], act["episode_seed"],
                                         act["t"] + act["delay_used"]])  # fmt: skip
        future = full.reindex(idx)[tcols].to_numpy()
        ahead = np.linalg.norm(act[pcols].to_numpy() - future, axis=1)
        out.append(pd.DataFrame({"configuration": act["configuration"].to_numpy(),
                                 "group": act["group"].to_numpy(),
                                 "stale": (act["delay_true"] == 0).to_numpy(),
                                 "quantity": key, "error_vs_now": now,
                                 "error_vs_t_plus_d": ahead}))  # fmt: skip
    frame = pd.concat(out, ignore_index=True)
    return (frame.groupby(["configuration", "group", "stale", "quantity"])
            .agg(steps=("error_vs_now", "size"), error_vs_now_mm=("error_vs_now",
                 lambda x: 1000 * x.mean()),
                 error_vs_t_plus_d_mm=("error_vs_t_plus_d", lambda x: 1000 * np.nanmean(x)))
            .reset_index())  # fmt: skip


DERIVED = ("decision_rules", "success", "comparisons", "mismatch", "prediction_errors",
           "placeholders")  # fmt: skip


def derive_stage1(episodes: pd.DataFrame, steps: pd.DataFrame, config: Stage1Config):
    n, s = config.bootstrap_resamples, config.bootstrap_seed
    rules = decision_rules(episodes, n, s)
    success = (episodes.groupby(["group", "condition", "configuration"])["success"]
               .agg(["mean", "size"]).reset_index())  # fmt: skip
    pooled = (episodes.groupby(["group", "configuration"])["success"].agg(["mean", "size"])
              .reset_index().assign(condition="POOLED"))  # fmt: skip
    success = pd.concat([success, pooled], ignore_index=True)
    comps = []
    for group, frame in episodes.groupby("group"):
        names = set(frame["configuration"])
        for a in sorted(names - {"base_only"}):
            for b in ("base_only", "deployed_learned"):
                if a != b and b in names:
                    comps.append({"group": group, "configuration": a, "reference": b,
                                  **diff(frame, a, b, n, s)})  # fmt: skip
    mism = mismatch_metrics(steps)
    mismatch = (mism.groupby(["configuration", "kind"])
                .agg(transitions=("segment_steps", "size"),
                     first_mismatch_median=("first_mismatch", "median"),
                     never_mismatched=("first_mismatch", lambda x: int(x.isna().sum())),
                     mismatched_steps_mean=("mismatched_steps", "mean"))
                .reset_index())  # fmt: skip
    sims = steps[steps["active"] & steps["configuration"].str.endswith("_sim")]
    placeholders = (sims.groupby(["configuration", "group"])
                    .agg(active_steps=("placeholders", "size"),
                         steps_with_placeholders=("placeholders", lambda x: int((x > 0).sum())),
                         placeholders_mean=("placeholders", "mean"))
                    .reset_index())  # fmt: skip
    tables = {
        "decision_rules": pd.json_normalize(rules),
        "success": success,
        "comparisons": pd.DataFrame(comps),
        "mismatch": mismatch,
        "prediction_errors": prediction_errors(steps),
        "placeholders": placeholders,
    }
    summary = {"decision_rules": rules, "plan_sha256": config.plan_sha256,
               "episodes": len(episodes), "step_rows": len(steps),
               "seeds": [min(config.seeds), max(config.seeds)]}  # fmt: skip
    return tables, summary


def derive_branches(branches: pd.DataFrame) -> pd.DataFrame:
    """Point of no return: per target episode, the last switch step that still rescues it,
    relative to the end of the delay (None if no branch rescues it)."""
    rows = []
    for key, ep in branches.groupby(KEYS):
        ep = ep.sort_values("switch_step")
        ok = ep[ep["success"]]
        rows.append({**dict(zip(KEYS, key, strict=True)), "end_step": int(ep["end_step"].iloc[0]),
                     "rescued_at_end": bool(ep["success"].iloc[0]),
                     "last_rescuing_offset": (int(ok["switch_step"].max() - ep["end_step"].iloc[0])
                                              if len(ok) else None),
                     "branches_rescuing": int(len(ok)), "branches": len(ep)})  # fmt: skip
    return pd.DataFrame(rows)
