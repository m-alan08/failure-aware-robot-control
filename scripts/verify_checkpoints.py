"""Verify every checkpoint in the manifest under $FARC_DATA_DIR/checkpoints and load it."""

from __future__ import annotations

import sys

from farc.base_policy.checkpoint import CheckpointError, load_policy, manifest
from farc.paths import checkpoint_path


def main() -> int:
    problems = 0
    for checkpoint_id, entry in sorted(manifest()["checkpoints"].items()):
        path = checkpoint_path(checkpoint_id)
        try:
            policy = load_policy(path, checkpoint_id)
        except CheckpointError as error:
            problems += 1
            print(f"FAIL {checkpoint_id}: {error}")
            continue
        assert policy.env_config.env_id == entry["env_id"]
        assert policy.training_seed == entry["training_seed"]
        print(f"ok   {checkpoint_id}: {policy.sha256[:12]} {policy.env_config.env_id} "
              f"seed {policy.training_seed}, {policy.total_steps} steps")  # fmt: skip
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
