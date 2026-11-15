"""Read-only loading of the frozen SAC+HER base policies.

Checkpoints are the files written by quantrl-fetch's training loop (see docs/DATA.md for how
to obtain them). They contain only tensors, NumPy arrays and plain containers, so they are read
with ``torch.load(weights_only=True)`` and a small allow-list of NumPy reconstruction
functions; nothing from quantrl-fetch needs to be importable. Every load verifies the file's
SHA-256 against ``checkpoints.json`` (keyed by the checkpoint's directory name), the actor is
put in eval mode with gradients disabled, and only full-precision checkpoints are accepted.
"""

from __future__ import annotations

import codecs
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, fields
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from farc.base_policy.networks import GaussianActor
from farc.base_policy.normalizer import GoalObservationNormalizer
from farc.envs.factory import EnvConfig

MANIFEST_PATH = Path(__file__).with_name("checkpoints.json")
FULL_PRECISION_BITS = 32
SAFE_GLOBALS: list[Any] = [
    np._core.multiarray._reconstruct,  # type: ignore[attr-defined]
    np.ndarray,
    np.dtype,
    codecs.encode,
    type(np.dtype(np.float64)),
    type(np.dtype(np.int64)),
]


class CheckpointError(RuntimeError):
    pass


@cache
def manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path, checkpoint_id: str | None = None) -> str:
    """Return the file's SHA-256 after checking it against the manifest entry for
    ``checkpoint_id`` (default: the name of the directory holding the file)."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"{path} does not exist; see docs/DATA.md to obtain checkpoints")
    checkpoint_id = checkpoint_id or path.parent.name
    entry = manifest()["checkpoints"].get(checkpoint_id)
    if entry is None:
        raise CheckpointError(f"{checkpoint_id!r} is not in {MANIFEST_PATH.name}")
    sha = file_sha256(path)
    if sha != entry["sha256"]:
        raise CheckpointError(
            f"{path} has SHA-256 {sha}, but the manifest expects {entry['sha256']} for "
            f"{checkpoint_id!r}; the file is not the frozen base policy"
        )
    return sha


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    with torch.serialization.safe_globals(SAFE_GLOBALS):
        return torch.load(Path(path), map_location="cpu", weights_only=True)


@dataclass
class FrozenPolicy:
    path: Path
    checkpoint_id: str
    sha256: str
    env_config: EnvConfig
    training_seed: int
    total_steps: int
    actor: GaussianActor
    normalizer: GoalObservationNormalizer
    dimensions: dict[str, int]
    metadata: dict[str, Any]
    training_config: dict[str, Any]


def load_policy(path: str | Path, checkpoint_id: str | None = None) -> FrozenPolicy:
    path = Path(path)
    checkpoint_id = checkpoint_id or path.parent.name
    sha = verify_checkpoint(path, checkpoint_id)
    raw = read_checkpoint(path)
    config = raw["config"]
    quant = config.get("quant", {})
    bits = (quant.get("weight_bits", 32), quant.get("activation_bits", 32))
    if bits != (FULL_PRECISION_BITS, FULL_PRECISION_BITS):
        raise CheckpointError(f"{path} is a quantized policy ({bits}); only fp32 is supported")
    known = {f.name for f in fields(EnvConfig)}
    env_config = EnvConfig(**{k: v for k, v in config["env"].items() if k in known})
    dimensions = dict(raw["dimensions"])
    sac = config["sac"]
    actor = GaussianActor(
        dimensions["observation_dim"] + dimensions["goal_dim"],
        dimensions["action_dim"],
        sac["hidden_sizes"],
        sac["log_std_min"],
        sac["log_std_max"],
    )
    actor.load_state_dict(raw["agent"]["actor"])
    actor.eval()
    actor.requires_grad_(False)
    normalizer = GoalObservationNormalizer(
        dimensions["observation_dim"],
        dimensions["goal_dim"],
        env_config.obs_clip,
        env_config.normalized_clip,
        env_config.normalizer_eps,
    )
    normalizer.load_state_dict(raw["normalizer"])
    return FrozenPolicy(
        path=path,
        checkpoint_id=checkpoint_id,
        sha256=sha,
        env_config=env_config,
        training_seed=int(config["seed"]),
        total_steps=int(raw["total_steps"]),
        actor=actor,
        normalizer=normalizer,
        dimensions=dimensions,
        metadata=raw["metadata"],
        training_config=config,
    )


def deterministic_policy_function(actor: GaussianActor) -> Callable[[np.ndarray], np.ndarray]:
    reference_parameter = next(actor.parameters())

    @torch.no_grad()
    def policy(inputs: np.ndarray) -> np.ndarray:
        tensor_inputs = torch.as_tensor(
            inputs, dtype=reference_parameter.dtype, device=reference_parameter.device
        )
        return actor.deterministic_action(tensor_inputs).float().cpu().numpy()

    return policy
