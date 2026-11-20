"""SAC training runner for the residual on Candidate C (pre-registration §4). NOT RUN.

The pre-registration is a draft and is NOT approved, so ``check_run`` refuses every
``"registered"`` run until ``configs/residual/residual_rl.json`` says ``"approved": true`` and
its pinned SHA-256 matches the plan on disk. The only runs this module allows today are
``"smoke"`` runs whose every seed lies in ``SMOKE_SEEDS`` (unregistered, disjoint from every
registered block; a test checks this). Smoke runs test the software, never the hypothesis.

Pre-registered settings implemented here (§4): ``ResidualActor`` [256, 256] with zero-
initialised output, twin-Q critic [256, 256], learning rates 3e-4 / 1e-3 / 1e-3, gamma 0.98,
tau 0.005, batch 256, replay 1M, HER off, fixed target entropy -8, 20k steps of delta = 0
data then 20k critic-only updates before the first actor update, residual-scale (``alpha``)
linear ramp 0.05 -> target over the first 200k steps, 1M environment steps, the nominal
protection rule (every 50k steps; drop > 2 points -> roll back and halve the actor learning
rate; second rollback -> stop, run counted as failed and analysed as C).

Naming: ``alpha`` (no prefix) is always the residual scale of §4; SAC's entropy coefficient is
``entropy_coef``.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from farc.adaptation.experiment import AdaptationConfigError
from farc.residual.policy import ResidualActor
from farc.retrain.qrl.algos.networks import TwinQCritic
from farc.supervision.seeds import R_SEED_BLOCKS, check_r_seeds

CONFIG_PATH = "configs/residual/residual_rl.json"
# Unregistered seeds for software smoke tests only. Every registered block in this repository
# starts at 1,000,000 or above (training-time evaluation, calibration, pilots, validation,
# R blocks, held-out), so [0, 100,000) is free; tests/test_residual_training.py checks this.
SMOKE_SEEDS = (0, 100_000)
ARMS = ("R", "A1", "A2", "A3", "B2")


class RunRefused(AdaptationConfigError):
    """A run that the approvals or the seed rules do not allow."""


class SeedsExhausted(RuntimeError):
    """The run needs more training episodes than its seed list holds (see the infrastructure
    note: 1M steps / 50-step episodes = 20,000 episodes, the training block holds 10,000)."""


@dataclass(frozen=True)
class TrainConfig:
    arm: str
    alpha_target: float
    network_seed: int
    policy_index: int  # which frozen SAC policy (seed0/1/2); cycles with network_seed
    mode: str = "registered"  # "registered" | "smoke"
    env_steps: int = 1_000_000
    hidden: tuple[int, ...] = (256, 256)
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_lr: float = 1e-3
    initial_entropy_coef: float = 1.0
    gamma: float = 0.98
    tau: float = 0.005
    batch_size: int = 256
    replay_size: int = 1_000_000
    target_entropy: float = -8.0
    zero_data_steps: int = 20_000
    critic_only_updates: int = 20_000
    alpha_start: float = 0.05
    alpha_ramp_steps: int = 200_000
    eval_interval: int = 50_000
    rollback_drop: float = 0.02  # 2 points, as a fraction
    max_rollbacks: int = 2
    log_interval: int = 1_000
    episode_seeds: tuple[int, ...] = ()
    nominal_eval_seeds: tuple[int, ...] = ()
    observation_dim: int = 65
    shares: dict[str, float] | None = None  # None: SHARES (T families, §3)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        e = self.episode_seeds
        d["episode_seeds"] = [e[0], e[-1], len(e)] if e else []
        d["nominal_eval_seeds"] = list(self.nominal_eval_seeds)
        return d


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_plan_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def check_approved(plan_config: dict[str, Any], root: str | Path = ".") -> None:
    """A registered run needs the approved flag and an unchanged plan."""
    if plan_config.get("approved") is not True:
        raise RunRefused("the residual RL pre-registration is not approved; registered runs "
                         "are refused (smoke runs on SMOKE_SEEDS only)")  # fmt: skip
    if _sha256(Path(root) / plan_config["plan"]) != plan_config["plan_sha256"]:
        raise RunRefused("the residual RL plan changed since the config pinned it")


def check_smoke_seeds(seeds: Sequence[int]) -> None:
    lo, hi = SMOKE_SEEDS
    if not seeds or not all(int(s) == s and lo <= s < hi for s in seeds):
        raise RunRefused(f"smoke runs accept only seeds in [{lo}, {hi})")


def check_run(cfg: TrainConfig, plan_config: dict[str, Any], root: str | Path = ".") -> None:
    """Validate everything about a run before any environment step. Raises ``RunRefused``."""
    problems = []
    if cfg.arm not in ARMS:
        problems.append(f"arm must be one of {ARMS}")
    if cfg.mode not in ("registered", "smoke"):
        problems.append("mode must be 'registered' or 'smoke'")
    if not (math.isfinite(cfg.alpha_target) and 0 < cfg.alpha_target <= 1):
        problems.append("alpha_target must be in (0, 1]")
    if not 0 < cfg.alpha_start <= cfg.alpha_target:
        problems.append("alpha_start must be in (0, alpha_target]")
    if cfg.policy_index not in (0, 1, 2):
        problems.append("policy_index must be 0, 1 or 2")
    for name in ("env_steps", "batch_size", "replay_size", "eval_interval", "log_interval",
                 "alpha_ramp_steps"):  # fmt: skip
        if int(getattr(cfg, name)) < 1:
            problems.append(f"{name} must be positive")
    if cfg.zero_data_steps < cfg.batch_size:
        problems.append("zero_data_steps must be at least one batch")
    if problems:
        raise RunRefused("invalid run:\n" + "\n".join(f"  - {p}" for p in problems))
    if cfg.mode == "smoke":
        check_smoke_seeds([*cfg.episode_seeds, *cfg.nominal_eval_seeds])
        return
    check_approved(plan_config, root)
    pinned = {
        "env_steps": plan_config["env_steps"],
        "target_entropy": plan_config["target_entropy"],
        "rollback_drop": plan_config["rollback_drop"],
    }
    for k, v in pinned.items():
        if getattr(cfg, k) != v:
            raise RunRefused(f"{k}={getattr(cfg, k)} differs from the pinned {v}")
    if cfg.arm == "R" and cfg.alpha_target not in plan_config["alphas"]:
        raise RunRefused("R arms use only the pre-registered alphas")
    if cfg.network_seed not in plan_config["network_seeds"]:
        raise RunRefused("network seed not pre-registered")
    if cfg.policy_index != cfg.network_seed % 3:
        raise RunRefused("each run's frozen policy cycles with its network seed (§4)")
    check_r_seeds(list(cfg.episode_seeds), "training")
    check_r_seeds(list(cfg.nominal_eval_seeds), "selection")
    lo, hi = R_SEED_BLOCKS["training"]
    if list(cfg.episode_seeds) != list(range(lo, lo + len(cfg.episode_seeds))):
        raise RunRefused("training episodes use the training block in order (§4)")


def alpha_schedule(step: int, cfg: TrainConfig) -> float:
    """Linear ramp alpha_start -> alpha_target over the first alpha_ramp_steps env steps."""
    frac = min(1.0, step / cfg.alpha_ramp_steps)
    return cfg.alpha_start + frac * (cfg.alpha_target - cfg.alpha_start)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int = 4):
        self.capacity, self.size, self.pos = capacity, 0, 0
        self.obs = np.zeros((capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), np.float32)
        self.act = np.zeros((capacity, act_dim), np.float32)
        self.rew = np.zeros(capacity, np.float32)
        self.done = np.zeros(capacity, np.float32)  # true termination only (never here)

    def add(self, o, a, r, o2, terminal: bool) -> None:
        i = self.pos
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i] = o, a, r, o2
        self.done[i] = float(terminal)
        self.pos, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)

    def sample(self, rng: np.random.Generator, n: int) -> dict[str, torch.Tensor]:
        idx = rng.integers(0, self.size, n)
        return {k: torch.as_tensor(getattr(self, k)[idx])
                for k in ("obs", "act", "rew", "next_obs", "done")}  # fmt: skip


class ResidualSAC:
    """SAC around ``ResidualActor``. The actor's deterministic output starts at exactly 0."""

    def __init__(self, cfg: TrainConfig):
        torch.manual_seed(cfg.network_seed)
        self.cfg = cfg
        self.actor = ResidualActor(cfg.observation_dim, cfg.hidden)
        self.critic = TwinQCritic(cfg.observation_dim, 4, cfg.hidden)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_entropy_coef = torch.tensor(math.log(cfg.initial_entropy_coef),
                                             requires_grad=True)  # fmt: skip
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.entropy_opt = torch.optim.Adam([self.log_entropy_coef], lr=cfg.entropy_lr)
        self.generator = torch.Generator().manual_seed(cfg.network_seed)

    def _sample(self, obs: torch.Tensor, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.actor.trunk(obs)
        mu = self.actor.mean(h)
        log_std = self.actor.log_std(h).clamp(-5.0, 2.0)  # as ResidualActor.forward
        eps = torch.randn(mu.shape, generator=generator, dtype=mu.dtype)
        pre = mu + log_std.exp() * eps
        a = torch.tanh(pre)
        logp = (-0.5 * eps.pow(2) - log_std - 0.5 * math.log(2 * math.pi)).sum(-1)
        logp = logp - (2 * (math.log(2) - pre - F.softplus(-2 * pre))).sum(-1)
        return a, logp

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs, np.float32)).reshape(1, -1)
        if deterministic:
            return self.actor(x, deterministic=True)[0].double().numpy()
        return self._sample(x, self.generator)[0][0].double().numpy()

    def set_actor_lr(self, lr: float) -> None:
        for g in self.actor_opt.param_groups:
            g["lr"] = lr

    def update(self, b: dict[str, torch.Tensor], actor: bool) -> dict[str, float]:
        coef = self.log_entropy_coef.exp().detach()
        with torch.no_grad():
            a2, logp2 = self._sample(b["next_obs"], self.generator)
            q1t, q2t = self.critic_target(b["next_obs"], a2)
            target = b["rew"] + self.cfg.gamma * (1 - b["done"]) * (torch.min(q1t, q2t)
                                                                    - coef * logp2)  # fmt: skip
        q1, q2 = self.critic(b["obs"], b["act"])
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        out = {"critic_loss": critic_loss.item(), "q_mean": q1.mean().item()}
        if actor:
            self.critic.requires_grad_(False)
            a, logp = self._sample(b["obs"], self.generator)
            qa1, qa2 = self.critic(b["obs"], a)
            actor_loss = (coef * logp - torch.min(qa1, qa2)).mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()
            self.critic.requires_grad_(True)
            ent_loss = -(self.log_entropy_coef * (logp.detach() + self.cfg.target_entropy)).mean()
            self.entropy_opt.zero_grad(set_to_none=True)
            ent_loss.backward()
            self.entropy_opt.step()
            out.update(actor_loss=actor_loss.item(), entropy=-logp.mean().item(),
                       entropy_coef=self.log_entropy_coef.exp().item())  # fmt: skip
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters(),
                             strict=True):  # fmt: skip
                pt.mul_(1 - self.cfg.tau).add_(p, alpha=self.cfg.tau)
        return out

    def state_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.state_dict(), "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "log_entropy_coef": self.log_entropy_coef.detach().clone(),
                "actor_opt": self.actor_opt.state_dict(),
                "critic_opt": self.critic_opt.state_dict(),
                "entropy_opt": self.entropy_opt.state_dict(),
                "generator": self.generator.get_state()}  # fmt: skip

    def load_state_dict(self, s: dict[str, Any]) -> None:
        self.actor.load_state_dict(s["actor"])
        self.critic.load_state_dict(s["critic"])
        self.critic_target.load_state_dict(s["critic_target"])
        with torch.no_grad():
            self.log_entropy_coef.copy_(s["log_entropy_coef"])
        self.actor_opt.load_state_dict(s["actor_opt"])
        self.critic_opt.load_state_dict(s["critic_opt"])
        self.entropy_opt.load_state_dict(s["entropy_opt"])
        self.generator.set_state(s["generator"])


NominalEval = Callable[[ResidualSAC, float], float]  # (agent, alpha) -> nominal success rate


@dataclass
class TrainResult:
    status: str  # "completed" | "failed_rollback"
    steps: int
    rollbacks: int
    episodes: int
    final_checkpoint: str
    log: str


def _seeds(seeds: Sequence[int]) -> Iterator[int]:
    yield from seeds
    raise SeedsExhausted(f"all {len(seeds)} training episode seeds are used")


def train(cfg: TrainConfig, env, nominal_eval: NominalEval, c_nominal_success: float,
          out_dir: str | Path, plan_config: dict[str, Any] | None = None,
          root: str | Path = ".") -> TrainResult:  # fmt: skip
    """Run one arm x network seed. ``env`` is a ``ResidualEnv`` (or the same interface);
    ``nominal_eval(agent, alpha)`` returns deterministic success on ``nominal_eval_seeds``;
    ``c_nominal_success`` is C's success on those seeds, computed once before training."""
    check_run(cfg, plan_config if plan_config is not None else load_plan_config(
        Path(root) / CONFIG_PATH), root)  # fmt: skip
    torch.set_num_threads(1)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"
    log = log_path.open("w")

    def emit(kind: str, **values) -> None:
        log.write(json.dumps({"kind": kind, **values}, sort_keys=True) + "\n")
        log.flush()

    emit("config", **cfg.to_dict(), c_nominal_success=c_nominal_success)
    agent = ResidualSAC(cfg)
    buffer = ReplayBuffer(cfg.replay_size, cfg.observation_dim)
    rng = np.random.default_rng([cfg.network_seed, 20261900])
    seeds = _seeds(cfg.episode_seeds)

    def save(name: str, step: int, extra: dict[str, Any] | None = None) -> Path:
        path = out / name
        torch.save(
            {
                "agent": agent.state_dict(),
                "step": step,
                "config": cfg.to_dict(),
                "actor_lr": agent.actor_opt.param_groups[0]["lr"],
                **(extra or {}),
            },
            path,
        )
        return path

    good = save("checkpoint_0.pt", 0)  # = Candidate C exactly (zero-initialised output)
    rollbacks, episodes, ep_return, status = 0, 0, 0.0, "completed"
    env.controller.alpha = alpha_schedule(0, cfg)
    obs, _ = env.reset(seed=next(seeds))
    step = 0
    for step in range(1, cfg.env_steps + 1):
        alpha = alpha_schedule(step - 1, cfg)
        env.controller.alpha = alpha
        delta = np.zeros(4) if step <= cfg.zero_data_steps else agent.act(obs, False)
        obs2, r, terminated, truncated, info = env.step(delta)
        buffer.add(obs, delta, r, obs2, terminated)
        ep_return += r
        if terminated or truncated:
            episodes += 1
            emit("episode", step=step, episode=episodes, ret=ep_return,
                 success=bool(info["success"]), alpha=alpha)  # fmt: skip
            ep_return = 0.0
            obs, _ = env.reset(seed=next(seeds))
        else:
            obs = obs2
        if step == cfg.zero_data_steps:
            for _ in range(cfg.critic_only_updates):
                agent.update(buffer.sample(rng, cfg.batch_size), actor=False)
            emit("critic_warmup_done", step=step, updates=cfg.critic_only_updates)
        elif step > cfg.zero_data_steps:
            m = agent.update(buffer.sample(rng, cfg.batch_size), actor=True)
            if step % cfg.log_interval == 0:
                emit("update", step=step, alpha=alpha, **m)
        if step % cfg.eval_interval == 0:
            success = float(nominal_eval(agent, alpha))
            ok = success >= c_nominal_success - cfg.rollback_drop
            emit("nominal_eval", step=step, success=success, c_success=c_nominal_success,
                 passed=ok)  # fmt: skip
            if ok:
                good = save(f"checkpoint_{step}.pt", step)
            else:
                rollbacks += 1
                state = torch.load(good, weights_only=False)
                lr = agent.actor_opt.param_groups[0]["lr"]
                agent.load_state_dict(state["agent"])
                agent.set_actor_lr(lr / 2)
                emit("rollback", step=step, to=good.name, rollbacks=rollbacks, actor_lr=lr / 2)
                if rollbacks >= cfg.max_rollbacks:
                    status = "failed_rollback"  # analysed as C (delta = 0), not excluded
                    break
    final = save("final.pt", step, {"status": status, "rollbacks": rollbacks})
    emit("end", status=status, step=step, rollbacks=rollbacks, episodes=episodes)
    log.close()
    return TrainResult(status, step, rollbacks, episodes, str(final), str(log_path))


def smoke_config(**overrides) -> TrainConfig:
    """A tiny run on unregistered seeds for software tests (not a result)."""
    base = TrainConfig(arm="R", alpha_target=0.1, network_seed=0, policy_index=0, mode="smoke",
                       env_steps=400, hidden=(16, 16), batch_size=16, replay_size=1000,
                       zero_data_steps=60, critic_only_updates=10, alpha_ramp_steps=100,
                       eval_interval=100, log_interval=50, episode_seeds=tuple(range(0, 50)),
                       nominal_eval_seeds=tuple(range(100, 105)))  # fmt: skip
    return replace(base, **overrides)
