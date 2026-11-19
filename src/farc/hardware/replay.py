"""Offline replay of recorded simulation trajectories (no simulator stepping).

Source: the archived Phase F1 evaluation step records (development seeds 3,653,000-099; no
held-out data). Each record has, per step, the observation the controller saw (``view0..24``,
noisy where the condition adds noise), the true observation (``true0..24``), the issued command
(``cmd0..2``), the latched delay and gain, and predicted positions.

**Goal reconstruction.** The records do not store the desired goal. FetchPush samples it in
``env.reset(seed=episode_seed)``, so ``reconstruct_goal`` calls ``reset`` with the recorded,
already-used episode seed and reads ``desired_goal``. No ``env.step`` is executed, no episode
is generated and no new seed is used. The reconstruction is checked: the reset observation must
equal the recorded true observation at t = 0 exactly, or the episode is rejected.

``ReplaySession`` serves recorded views to the *original* simulation episode loops
(``mechanism.run_episode``, ``stage3.run_g2_episode``) open-loop: the next view is the
recorded one whatever command was issued. Both the reference loop and the hardware path see
identical inputs, so their outputs must agree; the trajectories are not closed-loop for any
configuration other than the one that recorded them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from farc.counterfactual.session import Observed
from farc.delaychange.schedule import DelaySchedule
from farc.disturbances.config import ConditionSpec
from farc.envs.observation_layout import OBJECT
from farc.paths import resolve

S = OBJECT.slices
F1_STEPS = "$FARC_DATA_DIR/experiments/phase_f1_eval/steps.csv.gz"
VIEW = [f"view{i}" for i in range(25)]
TRUE = [f"true{i}" for i in range(25)]
CMD = [f"cmd{i}" for i in range(3)]
PRED = [f"pred_{k}{i}" for k in ("gripper_position", "object_position") for i in range(3)]
F1_SEEDS = (3_653_000, 3_653_100)  # [start, stop)
HELDOUT = (4_000_000, 5_000_000)


@dataclass
class Episode:
    configuration: str
    policy_seed: int
    condition: str
    episode_seed: int
    view: np.ndarray  # (T, 25)
    true: np.ndarray  # (T, 25)
    cmd: np.ndarray  # (T, 3)
    delay_param: np.ndarray
    gain_param: np.ndarray
    active: np.ndarray
    pred: np.ndarray  # (T, 6), NaN where inactive
    goal: np.ndarray | None = None


def load_f1(configurations: tuple[str, ...] | None = None) -> list[Episode]:
    cols = (["configuration", "policy_seed", "condition", "episode_seed", "t", "delay_param",
             "gain_param", "active"] + TRUE + VIEW + CMD + PRED)  # fmt: skip
    s = pd.read_csv(resolve(F1_STEPS), usecols=cols, float_precision="round_trip")
    if configurations is not None:
        s = s[s["configuration"].isin(configurations)]
    lo, hi = s["episode_seed"].min(), s["episode_seed"].max()
    if lo < F1_SEEDS[0] or hi >= F1_SEEDS[1] or (s["episode_seed"] >= HELDOUT[0]).any():
        raise RuntimeError("replay data outside the F1 development block")
    out = []
    for key, ep in s.sort_values("t").groupby(["configuration", "policy_seed", "condition",
                                                "episode_seed"], sort=True):  # fmt: skip
        if not np.array_equal(ep["t"].to_numpy(), np.arange(len(ep))):
            raise RuntimeError(f"{key}: steps not contiguous")
        out.append(Episode(*key, ep[VIEW].to_numpy(np.float64), ep[TRUE].to_numpy(np.float64),
                           ep[CMD].to_numpy(np.float64), ep["delay_param"].to_numpy(int),
                           ep["gain_param"].to_numpy(np.float64),
                           ep["active"].to_numpy(bool), ep[PRED].to_numpy(np.float64)))  # fmt: skip
    return out


def reconstruct_goal(env, episode: Episode) -> np.ndarray:
    obs, _ = env.reset(seed=int(episode.episode_seed))
    if not np.array_equal(np.asarray(obs["observation"], np.float64), episode.true[0]):
        raise RuntimeError(f"{episode.episode_seed}: reset does not reproduce the record")
    return np.asarray(obs["desired_goal"], np.float64).copy()


@dataclass
class ReplaySession:
    views: np.ndarray
    goal: np.ndarray
    policy: Any
    normalizer: Any
    schedule: DelaySchedule = field(default_factory=lambda: DelaySchedule(()))
    step: int = 0
    commands: list[np.ndarray] = field(default_factory=list)
    condition: ConditionSpec = field(default_factory=lambda: ConditionSpec("replay", ()))
    pending_view: Observed | None = None

    @property
    def horizon(self) -> int:
        return len(self.views)

    @property
    def done(self) -> bool:
        return self.step >= self.horizon

    def observe(self) -> Observed:
        v = self.views[self.step].copy()
        base = np.asarray(self.policy(self.normalizer(v, self.goal)[None])[0], np.float64)
        self.pending_view = Observed(self.step, v[S["gripper_position"]].copy(),
                                     v[S["object_position"]].copy(), self.goal.copy(), base,
                                     v.copy())  # fmt: skip
        return self.pending_view

    def act(self, command: np.ndarray) -> np.ndarray:
        if self.pending_view is None:
            raise RuntimeError("observe() before act()")
        c = np.asarray(command, np.float64).copy()
        self.commands.append(c)
        self.step += 1
        self.pending_view = None
        return c

    @property
    def final_success(self) -> bool:
        return False  # not meaningful open-loop

    @property
    def final_distance(self) -> float:
        return float("nan")
