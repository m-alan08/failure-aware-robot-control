"""The non-destructive reproduction checker (scripts/verify_reproduction.py)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location("verify_reproduction",
                                              Path("scripts/verify_reproduction.py"))  # fmt: skip
assert SPEC is not None and SPEC.loader is not None
vr = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vr)


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _release(tmp: Path, members: dict[str, bytes], extra: tarfile.TarInfo | None = None):
    rel = tmp / "release"
    rel.mkdir()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
        if extra is not None:
            t.addfile(extra)
    (rel / "a.tar").write_bytes(buf.getvalue())
    manifest = {"assets": {"a.tar": {"sha256": _sha(buf.getvalue()), "bytes": len(buf.getvalue()),
                                     "files": {n: {"sha256": _sha(d), "bytes": len(d)}
                                               for n, d in members.items()}}}}  # fmt: skip
    (rel / "MANIFEST.json").write_text(json.dumps(manifest))
    return rel


def test_clean_archive_extracts_and_verifies(tmp_path):
    rel = _release(tmp_path, {"experiments/x/episodes.csv.gz": b"abc", "models/m.pt": b"w"})
    report = vr.Report()
    vr.verify_archives(rel, tmp_path / "data", True, report)
    assert not report.failed
    assert (tmp_path / "data/models/m.pt").read_bytes() == b"w"


def test_tampered_archive_and_missing_asset_are_rejected(tmp_path):
    rel = _release(tmp_path, {"models/m.pt": b"w"})
    manifest = json.loads((rel / "MANIFEST.json").read_text())
    manifest["assets"]["a.tar"]["sha256"] = "0" * 64
    manifest["assets"]["b.tar"] = {"sha256": "1" * 64, "bytes": 1, "files": {}}
    (rel / "MANIFEST.json").write_text(json.dumps(manifest))
    report = vr.Report()
    vr.verify_archives(rel, tmp_path / "data", True, report)
    names = {c["name"] for c in report.failed}
    assert "a.tar hash" in names and "release assets are exactly the manifest's" in names
    assert not (tmp_path / "data/models/m.pt").exists()  # nothing extracted from a bad archive


@pytest.mark.parametrize("name,kind", [("../escape", tarfile.REGTYPE), ("/abs", tarfile.REGTYPE),
                                       ("link", tarfile.SYMTYPE)])  # fmt: skip
def test_unsafe_members_are_rejected(tmp_path, name, kind):
    info = tarfile.TarInfo(name)
    info.type = kind
    if kind == tarfile.SYMTYPE:
        info.linkname = "/etc/passwd"
    assert vr.unsafe_member(info) is not None
    rel = _release(tmp_path, {"ok.txt": b"x"}, extra=info)
    report = vr.Report()
    vr.verify_archives(rel, tmp_path / "data", True, report)
    assert any("paths are safe" in c["name"] for c in report.failed)
    assert not (tmp_path / "escape").exists()


def test_extraction_refuses_a_non_empty_data_directory(tmp_path):
    rel = _release(tmp_path, {"m.pt": b"w"})
    (tmp_path / "data").mkdir()
    (tmp_path / "data/old").write_text("x")
    report = vr.Report()
    vr.verify_archives(rel, tmp_path / "data", True, report)
    assert any(c["name"] == "data directory is empty before extraction" for c in report.failed)


def test_table_comparison_levels(tmp_path):
    ref = tmp_path / "t.csv"
    df = pd.DataFrame({"a": [0.1, 0.2], "b": ["x", "y"]})
    df.to_csv(ref, index=False)
    assert vr.compare_table(df, ref)[0] == vr.EXACT
    near = df.assign(a=df["a"] * (1 + 1e-13))
    assert vr.compare_table(near, ref)[0] in (vr.EXACT, vr.EQUAL)
    assert vr.compare_table(df.assign(a=[0.1, 0.3]), ref)[0] == vr.MISMATCH
    assert vr.compare_table(df.assign(b=["x", "z"]), ref)[0] == vr.MISMATCH
    assert vr.compare_table(df, tmp_path / "missing.csv")[0] == vr.MISMATCH


def test_lockfile_parser_keeps_every_locked_build(tmp_path):
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "torch"\nversion = "2.0"\n\n[[package]]\n'
                    'name = "torch"\nversion = "2.0+cpu"\n')  # fmt: skip
    assert vr.locked_versions(lock) == {"torch": {"2.0", "2.0+cpu"}}


def test_summary_closeness():
    assert vr._close({"a": [1.0, 2.0]}, {"a": [1.0, 2.0 + 1e-12]})
    assert not vr._close({"a": 1.0}, {"a": 1.1})
    assert vr._close(float("nan"), float("nan"))


def test_paired_design_completeness():
    rows = [{"configuration": c, "policy_seed": 0, "condition": "x", "episode_seed": s}
            for c in ("a", "b") for s in (1, 2)]  # fmt: skip
    assert vr.paired_design_complete(pd.DataFrame(rows))[0]
    assert not vr.paired_design_complete(pd.DataFrame(rows + rows[:1]))[0]  # duplicate
    assert not vr.paired_design_complete(pd.DataFrame(rows[:-1]))[0]  # missing pair
