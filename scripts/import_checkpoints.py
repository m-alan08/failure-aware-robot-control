"""Copy the frozen base-policy checkpoints into the data directory and verify their hashes.

The checkpoints are produced by m-alan08/quantrl-fetch (``runs/<id>/checkpoint.pt``; see
docs/DATA.md). This script never modifies the source files: it copies them to
``$FARC_DATA_DIR/checkpoints/<id>/checkpoint.pt``, checks each copy's SHA-256 against
src/farc/base_policy/checkpoints.json, and marks the copy read-only.
"""

from __future__ import annotations

import argparse
import shutil
import stat
from pathlib import Path

from farc.base_policy.checkpoint import file_sha256, manifest, verify_checkpoint
from farc.paths import checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--source-runs",
        required=True,
        help="quantrl-fetch runs directory holding <id>/checkpoint.pt",
    )
    parser.add_argument("--ids", nargs="*", default=None, help="Default: every manifest entry")
    arguments = parser.parse_args()
    source_runs = Path(arguments.source_runs).expanduser()
    ids = arguments.ids or sorted(manifest()["checkpoints"])
    for checkpoint_id in ids:
        source = source_runs / checkpoint_id / "checkpoint.pt"
        source_sha = verify_checkpoint(source, checkpoint_id)
        target = checkpoint_path(checkpoint_id)
        if target.exists():
            verify_checkpoint(target, checkpoint_id)
            print(f"{checkpoint_id}: already present and verified ({source_sha[:12]})")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        target.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        if file_sha256(target) != source_sha:
            raise RuntimeError(f"copy of {source} does not match its source")
        print(f"{checkpoint_id}: copied and verified ({source_sha[:12]}) -> {target}")


if __name__ == "__main__":
    main()
