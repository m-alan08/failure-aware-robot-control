"""External data locations.

Checkpoints, trajectories and experiment outputs are kept outside the repository under the
data directory: ``$FARC_DATA_DIR`` if set, otherwise ``~/farc-data``. Config files refer to it
as ``$FARC_DATA_DIR`` so they stay portable.
"""

from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = "~/farc-data"
DATA_DIR_VARIABLES = ("${FARC_DATA_DIR}", "$FARC_DATA_DIR")


def data_dir() -> Path:
    return Path(os.environ.get("FARC_DATA_DIR") or DEFAULT_DATA_DIR).expanduser()


def resolve(path: str | Path) -> Path:
    """Expand ``$FARC_DATA_DIR`` (or ``${FARC_DATA_DIR}``) and ``~`` in a configured path."""
    text = str(path)
    for variable in DATA_DIR_VARIABLES:
        if text.startswith(variable):
            text = str(data_dir()) + text[len(variable) :]
            break
    return Path(text).expanduser()


def checkpoint_path(checkpoint_id: str) -> Path:
    return data_dir() / "checkpoints" / checkpoint_id / "checkpoint.pt"
