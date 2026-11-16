"""Phase D experiment artifacts: what a run writes, and what analysis may rewrite.

An output directory holds two kinds of files:

- RAW (written once, by the rollout run): ``branches.csv.gz`` (every branch outcome) and
  ``metadata.json`` (the run's configuration, seeds, predictor hash and timing). Analysis never
  opens these for writing; a rollout run refuses to replace them unless told to explicitly.
- DERIVED (recomputed by every analysis): the tables in ``DERIVED_TABLES`` and
  ``analysis.json``. Only these names are ever written by ``analyze_existing``.

Analysis computes every derived output in memory before writing any of them, then writes each
file to a temporary name and renames it into place, so a failure part-way leaves earlier
derived files intact and never touches raw files. The SHA-256 of the raw files is checked
before and after and recorded in ``analysis.json``.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import pandas as pd

BRANCHES = "branches.csv.gz"
METADATA = "metadata.json"
RAW_FILES = (BRANCHES, METADATA)
ANALYSIS = "analysis.json"
DERIVED_TABLES = (
    "selection_candidates",
    "evaluation_candidates",
    "evaluation_by_disturbance_type",
    "evaluation_by_step",
    "evaluation_by_policy_seed",
    "evaluation_by_warning",
    "evaluation_by_base_outcome",
    "oracle_evaluation",
)
DERIVED_FILES = tuple(f"{n}.csv" for n in DERIVED_TABLES) + (ANALYSIS,)
assert not set(DERIVED_FILES) & set(RAW_FILES)


class ArtifactError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, default=str) + "\n"


def _csv_writer(
    table: pd.DataFrame, compression: Literal["gzip"] | None = None
) -> Callable[[Path], object]:
    return lambda path: table.to_csv(path, index=False, compression=compression)


def _atomic_write(path: Path, write: Callable[[Path], object]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        write(Path(tmp))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def write_raw(out: Path, tables: pd.DataFrame | dict[str, pd.DataFrame],
              metadata: dict[str, Any], overwrite: bool = False) -> None:  # fmt: skip
    """Write the rollout's raw files; refuses to replace existing ones unless ``overwrite``.

    ``tables`` is the branch table (Phase D) or a mapping from raw file name to table; every
    table is written gzip-compressed."""
    out.mkdir(parents=True, exist_ok=True)
    frames = {BRANCHES: tables} if isinstance(tables, pd.DataFrame) else dict(tables)
    names = [*frames, METADATA]
    existing = [n for n in names if (out / n).exists()]
    if existing and not overwrite:
        raise ArtifactError(f"raw files already exist in {out}: {existing}")
    for name, frame in frames.items():
        _atomic_write(out / name, _csv_writer(frame, compression="gzip"))
    _atomic_write(out / METADATA, lambda p: p.write_text(_json(metadata)))


def analyze_existing(
    out: Path,
    derive: Callable[..., tuple[dict[str, pd.DataFrame], dict[str, Any]]],
    results: Path | None = None,
    derived_tables: tuple[str, ...] = DERIVED_TABLES,
    raw_files: tuple[str, ...] = RAW_FILES,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Recompute the derived outputs from the raw files in ``out`` (read-only).

    ``derive`` receives every raw table (all raw files except ``metadata.json``, in order) and
    maps them to (tables named as in DERIVED_TABLES, a JSON summary).
    Derived files are also copied to ``results`` (the committed copy) when given; the raw
    ``metadata.json`` is never copied or rewritten here."""
    if METADATA not in raw_files:
        raise ArtifactError("metadata.json is always a raw file")
    missing = [n for n in raw_files if not (out / n).exists()]
    if missing:
        raise ArtifactError(f"analysis needs the raw files {missing} in {out}")
    before = {n: sha256(out / n) for n in raw_files}
    tables, summary = derive(*[pd.read_csv(out / n) for n in raw_files if n != METADATA])
    if set(tables) != set(derived_tables) or {f"{n}.csv" for n in tables} & {*raw_files, ANALYSIS}:
        raise ArtifactError(f"derive returned {sorted(tables)}, expected {list(derived_tables)}")
    if {n: sha256(out / n) for n in raw_files} != before:
        raise ArtifactError("raw files changed during analysis")
    payload = {**summary, "raw_sha256": before}
    for target in [out] + ([results] if results is not None else []):
        target.mkdir(parents=True, exist_ok=True)
        for name in derived_tables:
            _atomic_write(target / f"{name}.csv", _csv_writer(tables[name]))
        _atomic_write(target / ANALYSIS, lambda p: p.write_text(_json(payload)))
    return tables, payload


def publish_metadata(out: Path, results: Path) -> None:
    """Copy the raw run metadata to the committed results directory (after a rollout run)."""
    results.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out / METADATA, results / METADATA)
