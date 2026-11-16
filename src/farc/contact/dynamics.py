"""A small learned one-step dynamics model of the FetchPush observation, and its rollout.

State entries that change from step to step, in observation units (positions in m;
velocities are per control step, as the environment reports them):

    gripper_position (3), object_position (3), object_rotation (3),
    gripper_velocity (3), object_relative_velocity (3), object_angular_velocity (3)

The model predicts the one-step change of those 18 entries. Its inputs are
translation-invariant (no absolute positions):

    gripper velocity, object position relative to the gripper, object rotation,
    object relative velocity, object angular velocity, the executed position command (3)

The gripper's height above the object is already part of the relative position.

``rollout`` steps the model through the queued commands and rebuilds a *consistent*
observation: the relative position is recomputed from the predicted positions, and finger
entries are left unchanged. With ``move_object=False`` (ablation), the object keeps its
observed position, rotation and velocities, and only the gripper entries are predicted.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from farc.envs.observation_layout import OBJECT

S = OBJECT.slices
STATE = ("gripper_position", "object_position", "object_rotation", "gripper_velocity",
         "object_relative_velocity", "object_angular_velocity")  # fmt: skip
INPUT = ("gripper_velocity", "object_relative_position", "object_rotation",
         "object_relative_velocity", "object_angular_velocity")  # fmt: skip
GRIPPER_ENTRIES = ("gripper_position", "gripper_velocity")


def state_of(obs: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, np.float64)
    return np.concatenate([obs[..., S[k]] for k in STATE], axis=-1)


def features(obs: np.ndarray, command: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, np.float64)
    command = np.clip(np.asarray(command, np.float64)[..., :3], -1.0, 1.0)
    return np.concatenate([*(obs[..., S[k]] for k in INPUT), command], axis=-1)


def with_state(obs: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Write a predicted state back into an observation, keeping it self-consistent."""
    out = np.asarray(obs, np.float64).copy()
    offset = 0
    for key in STATE:
        width = S[key].stop - S[key].start
        out[..., S[key]] = state[..., offset : offset + width]
        offset += width
    out[..., S["object_relative_position"]] = (
        out[..., S["object_position"]] - out[..., S["gripper_position"]]
    )
    return out


class _MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, width: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, width), nn.Tanh(), nn.Linear(width, width),
                                 nn.Tanh(), nn.Linear(width, n_out))  # fmt: skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DynamicsModel:
    width: int
    weight_decay: float
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    net: _MLP

    def step(self, obs: np.ndarray, command: np.ndarray) -> np.ndarray:
        """Predicted next observation(s); works on a single observation or a batch."""
        x = (features(obs, command) - self.x_mean) / self.x_std
        with torch.no_grad():
            y = self.net(torch.as_tensor(np.atleast_2d(x), dtype=torch.float32)).numpy()
        delta = y.astype(np.float64) * self.y_std + self.y_mean
        delta = delta.reshape(*np.shape(obs)[:-1], -1)
        return with_state(obs, state_of(obs) + delta)

    def to_bytes(self) -> bytes:
        buffer = io.BytesIO()
        torch.save({"width": self.width, "weight_decay": self.weight_decay,
                    "x_mean": self.x_mean, "x_std": self.x_std, "y_mean": self.y_mean,
                    "y_std": self.y_std, "state_dict": self.net.state_dict()}, buffer)  # fmt: skip
        return buffer.getvalue()

    def save(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.to_bytes()
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()


def load_dynamics(path: str | Path, expected_sha256: str | None = None) -> DynamicsModel:
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"dynamics model hash {digest} != pinned {expected_sha256}")
    blob = torch.load(io.BytesIO(data), weights_only=False)
    net = _MLP(blob["x_mean"].shape[0], blob["y_mean"].shape[0], blob["width"])
    net.load_state_dict(blob["state_dict"])
    net.eval()
    return DynamicsModel(blob["width"], blob["weight_decay"], blob["x_mean"], blob["x_std"],
                         blob["y_mean"], blob["y_std"], net)  # fmt: skip


def train_dynamics(obs: np.ndarray, command: np.ndarray, next_obs: np.ndarray, width: int,
                   weight_decay: float, epochs: int = 60, seed: int = 0,
                   batch: int = 1024) -> DynamicsModel:  # fmt: skip
    torch.manual_seed(seed)
    x = features(obs, command)
    y = state_of(next_obs) - state_of(obs)
    x_mean, x_std = x.mean(0), x.std(0) + 1e-8
    y_mean, y_std = y.mean(0), y.std(0) + 1e-8
    xt = torch.as_tensor((x - x_mean) / x_std, dtype=torch.float32)
    yt = torch.as_tensor((y - y_mean) / y_std, dtype=torch.float32)
    net = _MLP(x.shape[1], y.shape[1], width)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=weight_decay)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), batch):
            idx = order[i : i + batch]
            opt.zero_grad()
            loss = ((net(xt[idx]) - yt[idx]) ** 2).mean()
            loss.backward()
            opt.step()
    net.eval()
    return DynamicsModel(width, weight_decay, x_mean, x_std, y_mean, y_std, net)


def rollout(model: DynamicsModel, obs: np.ndarray, commands: list[np.ndarray],
            gain: float = 1.0, move_object: bool = True) -> np.ndarray:  # fmt: skip
    """Predict the observation after executing ``commands`` (scaled by ``gain``) in order."""
    current = np.asarray(obs, np.float64).copy()
    for command in commands:
        executed = gain * np.clip(np.asarray(command, np.float64)[:3], -1.0, 1.0)
        nxt = model.step(current, executed)
        if not move_object:
            keep = current.copy()
            for key in GRIPPER_ENTRIES:
                keep[S[key]] = nxt[S[key]]
            # ablation: keep the observed object entries; recompute the relative position
            keep[S["object_relative_position"]] = keep[S["object_position"]] - keep[
                S["gripper_position"]]  # fmt: skip
            nxt = keep
        current = nxt
    return current


def rollout_batch(model: DynamicsModel, obs: np.ndarray, executed: np.ndarray,
                  move_object: bool = True) -> np.ndarray:  # fmt: skip
    """Batched rollout from observations (N, 25) through executed commands (N, h, 3)."""
    current = np.asarray(obs, np.float64).copy()
    for j in range(executed.shape[1]):
        nxt = model.step(current, executed[:, j])
        if not move_object:
            keep = current.copy()
            for key in GRIPPER_ENTRIES:
                keep[:, S[key]] = nxt[:, S[key]]
            keep[:, S["object_relative_position"]] = (
                keep[:, S["object_position"]] - keep[:, S["gripper_position"]])  # fmt: skip
            nxt = keep
        current = nxt
    return current


def static_object_batch(obs: np.ndarray, executed: np.ndarray, nominal_gain: float) -> np.ndarray:
    """Phase E's prediction: gripper shifted by K0 * 0.05 * sum(executed), object static."""
    out = np.asarray(obs, np.float64).copy()
    delta = nominal_gain * 0.05 * np.clip(executed, -1, 1).sum(axis=1)
    out[:, S["gripper_position"]] += delta
    out[:, S["object_relative_position"]] -= delta
    return out
