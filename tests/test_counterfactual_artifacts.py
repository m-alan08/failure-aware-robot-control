"""Analysis-only must never modify raw Phase D artifacts."""

from __future__ import annotations

import hashlib

import pandas as pd
import pytest

from farc.counterfactual import artifacts
from farc.counterfactual.artifacts import (
    BRANCHES,
    DERIVED_FILES,
    DERIVED_TABLES,
    METADATA,
    RAW_FILES,
    ArtifactError,
    analyze_existing,
    write_raw,
)


def _digests(directory):
    return {p.name: hashlib.sha256(p.read_bytes()).digest() for p in directory.iterdir()}


def _derive(frame):
    table = pd.DataFrame({"rows": [len(frame)], "mean_t": [frame["t"].mean()]})
    return {name: table for name in DERIVED_TABLES}, {"branch_rows": len(frame)}


@pytest.fixture
def run_dir(tmp_path):
    out = tmp_path / "run"
    frame = pd.DataFrame({"episode_seed": [1, 1, 2], "t": [0, 5, 0], "success": [1, 0, 1]})
    write_raw(out, frame, {"seconds": 12.5, "config": {"name": "x"}})
    return out


def test_analysis_leaves_raw_bytes_unchanged_and_writes_only_derived(run_dir, tmp_path):
    raw = {n: (run_dir / n).read_bytes() for n in RAW_FILES}
    results = tmp_path / "results"
    _, payload = analyze_existing(run_dir, _derive, results)
    _, again = analyze_existing(run_dir, _derive, results)  # idempotent
    assert {n: (run_dir / n).read_bytes() for n in RAW_FILES} == raw
    assert {p.name for p in run_dir.iterdir()} == set(RAW_FILES) | set(DERIVED_FILES)
    assert {p.name for p in results.iterdir()} == set(DERIVED_FILES)  # metadata not copied
    assert payload == again
    assert payload["raw_sha256"][BRANCHES] == hashlib.sha256(raw[BRANCHES]).hexdigest()


def test_failure_inside_derive_writes_nothing(run_dir):
    analyze_existing(run_dir, _derive)
    before = _digests(run_dir)

    def broken(frame):
        raise ValueError("analysis bug")

    with pytest.raises(ValueError):
        analyze_existing(run_dir, broken)
    assert _digests(run_dir) == before


def test_failure_while_writing_keeps_raw_and_leaves_no_partial_files(run_dir, monkeypatch):
    analyze_existing(run_dir, _derive)
    raw_before = {n: (run_dir / n).read_bytes() for n in RAW_FILES}
    original = artifacts._csv_writer
    calls = {"n": 0}

    def flaky(table):
        write = original(table)

        def inner(path):
            calls["n"] += 1
            if calls["n"] == 3:
                path.write_text("partial")
                raise OSError("disk full")
            return write(path)

        return inner

    monkeypatch.setattr(artifacts, "_csv_writer", flaky)
    with pytest.raises(OSError):
        analyze_existing(run_dir, lambda f: ({n: pd.DataFrame({"v": [9]})
                                              for n in DERIVED_TABLES}, {}))  # fmt: skip
    assert {n: (run_dir / n).read_bytes() for n in RAW_FILES} == raw_before
    assert {p.name for p in run_dir.iterdir()} == set(RAW_FILES) | set(DERIVED_FILES)
    third = run_dir / f"{DERIVED_TABLES[2]}.csv"
    assert "partial" not in third.read_text()  # old complete file kept, no truncation


def test_analysis_detects_raw_modification(run_dir):
    def tampering(frame):
        (run_dir / METADATA).write_text("{}")
        return _derive(frame)

    with pytest.raises(ArtifactError, match="raw files changed"):
        analyze_existing(run_dir, tampering)
    assert not (run_dir / "analysis.json").exists()


def test_analysis_rejects_undesignated_outputs(run_dir):
    before = _digests(run_dir)
    with pytest.raises(ArtifactError):
        analyze_existing(run_dir, lambda f: ({"branches.csv": pd.DataFrame()}, {}))
    assert _digests(run_dir) == before


def test_analysis_requires_raw_files(tmp_path):
    with pytest.raises(ArtifactError, match="raw files"):
        analyze_existing(tmp_path, _derive)
    assert list(tmp_path.iterdir()) == []


def test_rollout_refuses_to_replace_raw(run_dir):
    before = _digests(run_dir)
    with pytest.raises(ArtifactError, match="already exist"):
        write_raw(run_dir, pd.DataFrame({"t": [1]}), {})
    assert _digests(run_dir) == before
