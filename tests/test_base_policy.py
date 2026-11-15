from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from farc.base_policy.checkpoint import (
    CheckpointError,
    deterministic_policy_function,
    load_policy,
    manifest,
    read_checkpoint,
    verify_checkpoint,
)
from farc.base_policy.networks import GaussianActor
from farc.paths import checkpoint_path, data_dir, resolve

PUSH = "fetch_push_fp32_seed0"
needs_checkpoints = pytest.mark.skipif(
    not checkpoint_path(PUSH).exists(), reason="checkpoints not imported (see docs/DATA.md)"
)


def test_resolve_expands_the_data_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("FARC_DATA_DIR", str(tmp_path))
    assert data_dir() == tmp_path
    assert resolve("$FARC_DATA_DIR/experiments") == tmp_path / "experiments"
    assert resolve("${FARC_DATA_DIR}/x") == tmp_path / "x"
    assert checkpoint_path("abc") == tmp_path / "checkpoints/abc/checkpoint.pt"
    monkeypatch.delenv("FARC_DATA_DIR")
    assert data_dir() == Path("~/farc-data").expanduser()
    assert resolve("~/a") == Path.home() / "a"


def test_manifest_lists_the_six_frozen_policies():
    entries = manifest()["checkpoints"]
    assert len(entries) == 6
    assert all(len(e["sha256"]) == 64 and e["precision"] == "fp32" for e in entries.values())
    assert {e["env_id"] for e in entries.values()} == {"FetchReach-v4", "FetchPush-v4"}


def test_actor_matches_a_manual_forward_pass():
    torch.manual_seed(0)
    actor = GaussianActor(28, 4, [16, 16])
    inputs = torch.randn(5, 28)
    layers = [m for m in actor.network if isinstance(m, torch.nn.Linear)]
    hidden = inputs
    for layer in layers[:-1]:
        hidden = torch.relu(hidden @ layer.weight.T + layer.bias)
    mean = (hidden @ layers[-1].weight.T + layers[-1].bias)[:, :4]
    assert torch.allclose(actor.deterministic_action(inputs), torch.tanh(mean))


@needs_checkpoints
def test_load_verifies_hash_and_freezes_the_actor():
    policy = load_policy(checkpoint_path(PUSH))
    entry = manifest()["checkpoints"][PUSH]
    assert policy.sha256 == entry["sha256"] and policy.training_seed == 0
    assert policy.env_config.env_id == "FetchPush-v4" and policy.total_steps == 1_000_000
    assert not policy.actor.training
    assert not any(p.requires_grad for p in policy.actor.parameters())
    raw = read_checkpoint(checkpoint_path(PUSH))
    for name, value in policy.actor.state_dict().items():
        assert torch.equal(value, raw["agent"]["actor"][name])
    assert np.array_equal(
        policy.normalizer.observation.mean, raw["normalizer"]["observation"]["mean"]
    )
    action = deterministic_policy_function(policy.actor)(np.zeros((1, 28), np.float32))
    assert action.shape == (1, 4) and np.all(np.abs(action) <= 1)


@needs_checkpoints
def test_tampered_or_unlisted_checkpoints_are_rejected(tmp_path):
    unlisted = tmp_path / "some_other_policy" / "checkpoint.pt"
    unlisted.parent.mkdir()
    shutil.copy(checkpoint_path(PUSH), unlisted)
    with pytest.raises(CheckpointError, match="not in"):
        verify_checkpoint(unlisted)
    tampered = tmp_path / PUSH / "checkpoint.pt"
    tampered.parent.mkdir()
    shutil.copy(checkpoint_path(PUSH), tampered)
    tampered.chmod(0o644)
    with tampered.open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(CheckpointError, match="SHA-256"):
        load_policy(tampered)
    with pytest.raises(CheckpointError, match="does not exist"):
        verify_checkpoint(tmp_path / "missing" / "checkpoint.pt")


@needs_checkpoints
def test_imported_copies_are_read_only():
    assert not checkpoint_path(PUSH).stat().st_mode & 0o222
