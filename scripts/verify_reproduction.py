"""Non-destructive reproduction check for the archived results (docs/REPRODUCIBILITY_REPORT.md).

Run from a clean checkout of the commit tagged ``artifacts-v1``, after downloading the release:

    gh release download artifacts-v1 --dir <RELEASE_DIR>
    uv sync --frozen
    uv run --frozen python scripts/verify_reproduction.py \
        --release-dir <RELEASE_DIR> --data-dir <EMPTY_DIR> --extract --report <REPORT_JSON>

The script never runs the simulator, trains anything or writes into the repository or the
release directory. It:

1. records the environment and checks that the git tree is clean, HEAD is the tagged commit
   and installed package versions match ``uv.lock``;
2. verifies every release archive against ``MANIFEST.json``, extracts it safely into an empty
   data directory (no absolute or parent paths, links or special files, and nothing outside
   the manifest), and verifies every extracted file;
3. checks the checkpoints against ``src/farc/base_policy/checkpoints.json``, the F1 model
   against its pinned hash and every raw file against the committed ``analysis.json``;
4. recomputes each supported analysis in memory from the restored raw files with the
   repository's own code, and compares every derived table and the summary with ``results/``:
   byte-identical, numerically equal (relative 1e-9) or a mismatch.

It writes a JSON report with paths shown as ``<RELEASE_DIR>``, ``<DATA_DIR>`` and ``<REPO>``,
and exits 1 if any required check fails. ``--repo`` (default: the current directory) lets the
script run from outside a pristine checkout without adding files to it.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

TAG = "artifacts-v1"
PACKAGES = ("torch", "numpy", "pandas", "mujoco", "gymnasium", "gymnasium-robotics", "scipy")
EXACT, EQUAL, MISMATCH = "byte-identical", "numerically equal", "MISMATCH"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, group: str, name: str, ok: bool, detail: Any = None,
            required: bool = True) -> bool:  # fmt: skip
        self.checks.append({"group": group, "name": name, "passed": bool(ok),
                            "required": required, "detail": detail})  # fmt: skip
        return ok

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [c for c in self.checks if c["required"] and not c["passed"]]


# --- 1. environment ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=False).stdout.strip()  # fmt: skip


def locked_versions(lock: Path) -> dict[str, set[str]]:
    """Package versions from uv.lock ([[package]] name = ..., version = ...). A package can be
    locked more than once (e.g. a CUDA and a CPU build of torch for different platforms)."""
    out: dict[str, set[str]] = {}
    name = None
    for line in lock.read_text().splitlines():
        m = re.match(r'^name = "([^"]+)"$', line)
        if m:
            name = m.group(1)
            continue
        m = re.match(r'^version = "([^"]+)"$', line)
        if m and name:
            out.setdefault(name, set()).add(m.group(1))
            name = None
    return out


def check_environment(repo: Path, report: Report, expected_commit: str | None) -> dict[str, Any]:
    head = _git(repo, "rev-parse", "HEAD")
    tagged = _git(repo, "rev-parse", f"{TAG}^{{commit}}")
    dirty = _git(repo, "status", "--porcelain")
    target = expected_commit or tagged
    report.add("environment", "HEAD is the commit tagged artifacts-v1", bool(target) and
               head == target, {"head": head, "tagged": tagged or "tag not found"})  # fmt: skip
    report.add("environment", "git tree is clean", dirty == "",
               "clean" if not dirty else f"{len(dirty.splitlines())} changed paths")  # fmt: skip
    lock = locked_versions(repo / "uv.lock")
    versions: dict[str, str | None] = {}
    for p in PACKAGES:
        try:
            versions[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            versions[p] = None
    mismatched = {p: (versions[p], sorted(lock[p])) for p in PACKAGES
                  if p in lock and versions[p] not in lock[p]}  # fmt: skip
    report.add("environment", "installed versions match uv.lock", not mismatched,
               mismatched or "all match")  # fmt: skip
    py = (repo / ".python-version").read_text().strip()
    report.add(
        "environment",
        "python version matches .python-version",
        platform.python_version().startswith(py + "."),
        platform.python_version(),
    )
    return {"commit": head, "uv_lock_sha256": sha256(repo / "uv.lock"),
            "python": platform.python_version(), "platform": platform.platform(),
            "machine": platform.machine(), "packages": versions}  # fmt: skip


# --- 2. archives ------------------------------------------------------------------------------


def unsafe_member(member: tarfile.TarInfo) -> str | None:
    name = member.name
    if name.startswith("/") or name.startswith("\\") or re.match(r"^[A-Za-z]:", name):
        return "absolute path"
    if ".." in Path(name).parts:
        return "parent reference"
    if not (member.isfile() or member.isdir()):
        return "link or special file"
    return None


def verify_archives(release: Path, data: Path, extract: bool, report: Report) -> dict:
    manifest = json.loads((release / "MANIFEST.json").read_text())
    assets = manifest["assets"]
    present = sorted(p.name for p in release.iterdir() if p.is_file())
    expected = sorted([*assets, "MANIFEST.json"])
    report.add("archives", "release assets are exactly the manifest's", present == expected,
               {"missing": sorted(set(expected) - set(present)),
                "unexpected": sorted(set(present) - set(expected))})  # fmt: skip
    if extract:
        if data.exists() and any(data.iterdir()):
            report.add("archives", "data directory is empty before extraction", False)
            return manifest
        data.mkdir(parents=True, exist_ok=True)
    archive_ok = file_ok = 0
    for asset, info in sorted(assets.items()):
        path = release / asset
        if not path.exists():
            continue
        good = sha256(path) == info["sha256"] and path.stat().st_size == info["bytes"]
        archive_ok += good
        report.add("archives", f"{asset} hash", good)
        with tarfile.open(path) as tar:
            members = tar.getmembers()
            bad = {m.name: why for m in members if (why := unsafe_member(m))}
            names = sorted(m.name for m in members if m.isfile())
            report.add("archives", f"{asset} paths are safe and match the manifest",
                       not bad and names == sorted(info["files"]),
                       {"unsafe": bad} if bad else None)  # fmt: skip
            if extract and good and not bad and names == sorted(info["files"]):
                tar.extractall(data, filter="data")
        for rel, f in info["files"].items():
            target = data / rel
            ok = target.is_file() and sha256(target) == f["sha256"]
            file_ok += ok
            if not ok:
                report.add("files", rel, False, "missing or hash mismatch")
    n_files = sum(len(a["files"]) for a in assets.values())
    report.add("archives", "all archives verified", archive_ok == len(assets),
               f"{archive_ok}/{len(assets)}")  # fmt: skip
    report.add("files", "all manifest files verified", file_ok == n_files,
               f"{file_ok}/{n_files}")  # fmt: skip
    listed = {rel for a in assets.values() for rel in a["files"]}
    extra = sorted(str(p.relative_to(data)) for p in data.rglob("*")
                   if p.is_file() and str(p.relative_to(data)) not in listed)  # fmt: skip
    report.add("files", "no files outside the manifest in the data directory", not extra,
               extra[:20] or None)  # fmt: skip
    return manifest


def verify_models(repo: Path, data: Path, report: Report) -> None:
    ck = json.loads((repo / "src/farc/base_policy/checkpoints.json").read_text())
    items = ck.get("checkpoints", ck)
    pairs = items.items() if isinstance(items, dict) else [(i["id"], i) for i in items]
    ok = 0
    for cid, info in pairs:
        p = data / "checkpoints" / cid / "checkpoint.pt"
        ok += p.is_file() and sha256(p) == info["sha256"]
    report.add("models", "checkpoints match checkpoints.json", ok == len(list(pairs)) == 6,
               f"{ok}/6")  # fmt: skip
    pinned = json.loads((repo / "configs/delaychange/g4_heldout.json").read_text())
    model = data / "models/contact_predictor/model.pt"
    report.add("models", "F1 contact model matches its pinned hash",
               model.is_file() and sha256(model) == pinned["model_sha256"],
               pinned["model_sha256"][:12])  # fmt: skip


# --- 3/4. analyses ----------------------------------------------------------------------------


def _analyses() -> dict[str, Callable[[], dict[str, Any]]]:
    """Each entry returns: config-derived raw directory, results directory, raw file names,
    derived table names and the derive function, exactly as the original script calls them."""
    from farc.paths import resolve

    def e_pilot():
        m = importlib.import_module("farc.adaptation.experiment")
        c = m.load_adaptation_experiment("configs/adaptation/phase_e_pilot.json")
        return dict(out=resolve(c.output), results=Path("results") / c.name,
                    raw=("episodes.csv.gz", "steps.csv.gz"), tables=m.DERIVED_TABLES,
                    derive=lambda e, s: m.derive_tables(e, s, c))  # fmt: skip

    def e2():
        m = importlib.import_module("farc.adaptation.delay_eval")
        c = m.load_delay_evaluation("configs/adaptation/phase_e2_delay.json")
        return dict(out=resolve(c.output), results=Path("results") / c.name,
                    raw=("episodes.csv.gz", "steps.csv.gz"), tables=m.DERIVED_TABLES,
                    derive=lambda e, s: m.derive_tables(e, s, c))  # fmt: skip

    def f1():
        m = importlib.import_module("farc.contact.evaluation")
        c = m.load_contact_evaluation("configs/contact/phase_f1_eval.json")
        return dict(out=resolve(c.output), results=Path("results") / c.name,
                    raw=("episodes.csv.gz", "steps.csv.gz"), tables=m.DERIVED_TABLES,
                    derive=lambda e, s: m.derive_tables(e, s, c))  # fmt: skip

    def heldout_e_f1():
        m = importlib.import_module("farc.contact.heldout")
        spec = m.load_heldout("configs/contact/heldout_confirmation.json")
        return dict(out=resolve(spec.contact.output), results=Path("results") / spec.contact.name,
                    raw=("episodes.csv.gz", "steps.csv.gz"), tables=m.DERIVED_TABLES,
                    derive=lambda e, s: m.derive_heldout(e, s, spec))  # fmt: skip

    def g1():
        m = importlib.import_module("farc.supervision.evaluation")
        c = m.load_g1("configs/supervision/phase_g1.json", role="evaluation")
        return dict(out=resolve(c.output), results=Path("results") / c.name,
                    raw=("episodes.csv.gz", "steps.csv.gz"), tables=m.DERIVED_TABLES,
                    derive=lambda e, s: m.derive_g1(e, s, c))  # fmt: skip

    def g2_stage1():
        import pandas as pd

        m = importlib.import_module("farc.delaychange.stage1")
        c = m.load_stage1("configs/delaychange/g2_stage1.json")

        def derive(e, s, b):
            tables, summary = m.derive_stage1(e, s, c)
            tables["branches"] = m.derive_branches(b) if len(b) else pd.DataFrame()
            return tables, summary

        return dict(out=resolve(c.output), results=Path("results") / c.name,
                    raw=("episodes.csv.gz", "steps.csv.gz", "branches.csv.gz"),
                    tables=(*m.DERIVED, "branches"), derive=derive)  # fmt: skip

    def g2_stage3():
        s3 = importlib.import_module("farc.delaychange.stage3")
        a = importlib.import_module("farc.delaychange.stage3_analysis")
        c = s3.load_g2("configs/delaychange/g2_stage3.json", "evaluation")
        return dict(out=resolve(c.output), results=Path("results") / c.name,
                    raw=("episodes.csv.gz", "steps.csv.gz"), tables=a.DERIVED,
                    derive=lambda e, s: a.derive_g2(e, s, c))  # fmt: skip

    def g2_sham_check():
        s3 = importlib.import_module("farc.delaychange.stage3")
        m = importlib.import_module("farc.delaychange.sham_check")
        c = m.check_config(s3.load_g2("configs/delaychange/g2_stage3.json", "evaluation"))
        return dict(out=resolve(c.output), results=Path("results/phase_g2_sham_check"),
                    raw=("episodes.csv.gz", "steps.csv.gz"), tables=m.DERIVED,
                    derive=lambda e, s: m.derive_check(e, s, c))  # fmt: skip

    def g3():
        m = importlib.import_module("farc.delaychange.mechanism")
        a = importlib.import_module("farc.delaychange.mechanism_analysis")
        c = m.load_g3("configs/delaychange/g3_mechanism.json", "evaluation")
        return dict(out=resolve(c.output), results=Path("results") / c.name,
                    raw=("episodes.csv.gz", "steps.csv.gz"), tables=a.DERIVED,
                    derive=lambda e, s: a.derive_g3(e, s, c))  # fmt: skip

    def g4():
        m = importlib.import_module("farc.delaychange.heldout")
        a = importlib.import_module("farc.delaychange.heldout_analysis")
        c = m.load_g4("configs/delaychange/g4_heldout.json")
        return dict(out=resolve(c.output), results=Path("results") / c.name,
                    raw=("episodes.csv.gz", "steps.csv.gz"), tables=a.DERIVED,
                    derive=lambda e, s: a.derive_g4(e, s, c))  # fmt: skip

    return {"e_pilot": e_pilot, "e2": e2, "f1_eval": f1, "heldout_e_f1": heldout_e_f1,
            "g1": g1, "g2_stage1": g2_stage1, "g2_stage3": g2_stage3,
            "g2_sham_check": g2_sham_check, "g3": g3, "g4": g4}  # fmt: skip


ANALYSES = ("e_pilot", "e2", "f1_eval", "heldout_e_f1", "g1", "g2_stage1", "g2_stage3",
            "g2_sham_check", "g3", "g4")  # fmt: skip


def compare_table(table, reference: Path) -> tuple[str, str | None]:
    """byte-identical | numerically equal (relative 1e-9, NaN-aware) | MISMATCH, with detail."""
    import numpy as np
    import pandas as pd

    if not reference.exists():
        return MISMATCH, "committed table missing"
    buf = io.StringIO()
    table.to_csv(buf, index=False)
    if buf.getvalue() == reference.read_text():
        return EXACT, None
    new, ref = pd.read_csv(io.StringIO(buf.getvalue())), pd.read_csv(reference)
    if list(new.columns) != list(ref.columns) or new.shape != ref.shape:
        return MISMATCH, f"shape/columns {new.shape} vs {ref.shape}"
    for col in ref.columns:
        a, b = new[col], ref[col]
        if a.dtype.kind in "fiub" and b.dtype.kind in "fiub":
            if not np.allclose(a.astype(float), b.astype(float), rtol=1e-9, atol=1e-12,
                               equal_nan=True):  # fmt: skip
                diff = float(np.nanmax(np.abs(a.astype(float) - b.astype(float))))
                return MISMATCH, f"column {col}: max abs difference {diff:.3g}"
        elif not (a.astype(str) == b.astype(str)).all():
            return MISMATCH, f"column {col}: text differs"
    return EQUAL, None


def paired_design_complete(episodes) -> tuple[bool, str]:
    """Within each condition, every configuration has exactly one row per (policy seed,
    episode seed), and all configurations cover the same pairs. The analyses pivot with
    ``aggfunc="first"``, which would hide duplicates, and some code paths cast missing cells to
    bool, so an incomplete design would bias results silently."""
    keys = ["configuration", "policy_seed", "condition", "episode_seed"]
    dup = int(episodes.duplicated(keys).sum())
    gaps = 0
    for _, sub in episodes.groupby("condition"):
        pairs = sub.groupby("configuration")[["policy_seed", "episode_seed"]].apply(
            lambda f: frozenset(map(tuple, f.to_numpy()))
        )
        gaps += int(pairs.nunique() != 1)
    return (
        dup == 0 and gaps == 0,
        f"{len(episodes)} rows, {dup} duplicates, {gaps} incomplete conditions",
    )


def run_analysis(name: str, data: Path, report: Report) -> dict[str, Any]:
    import pandas as pd

    art = importlib.import_module("farc.counterfactual.artifacts")
    spec = _analyses()[name]()
    out, results = Path(spec["out"]), Path(spec["results"])
    if not str(out.resolve()).startswith(str(data.resolve())):
        report.add(name, "raw directory resolves inside <DATA_DIR>", False)
        return {}
    committed = json.loads((results / "analysis.json").read_text())
    recorded = committed.get("raw_sha256", {})
    files = (*spec["raw"], "metadata.json")
    actual = {f: sha256(out / f) for f in files if (out / f).exists()}
    report.add(name, "raw files match the committed analysis.json",
               bool(recorded) and all(actual.get(f) == h for f, h in recorded.items()),
               {f: "match" if actual.get(f) == h else "MISMATCH or missing"
                for f, h in recorded.items()})  # fmt: skip
    raw = [pd.read_csv(out / f) for f in spec["raw"]]
    ok, detail = paired_design_complete(raw[0])
    report.add(name, "paired design complete (one row per pair, same pairs per condition)",
               ok, detail)  # fmt: skip
    started = time.time()
    tables, summary = spec["derive"](*raw)
    seconds = round(time.time() - started, 1)
    results_detail = {}
    for t in spec["tables"]:
        status, why = compare_table(tables[t], results / f"{t}.csv")
        results_detail[t] = status if why is None else f"{status}: {why}"
        report.add(name, f"table {t}", status != MISMATCH, results_detail[t])
    payload = {**summary, "raw_sha256": {f: actual.get(f) for f in recorded}}
    summary_exact = art._json(payload) == (results / "analysis.json").read_text()
    if summary_exact:
        summary_status = EXACT
    else:
        a = json.loads(art._json(payload))
        summary_status = EQUAL if _close(a, committed) else MISMATCH
    report.add(name, "summary (analysis.json)", summary_status != MISMATCH, summary_status)
    return {"tables": results_detail, "summary": summary_status, "seconds": seconds}


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, float) and isinstance(b, float):
        return (a != a and b != b) or abs(a - b) <= 1e-9 * max(1.0, abs(b))
    return a == b


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--extract",
        action="store_true",
        help="extract the archives into the (empty) data directory first",
    )
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--report", type=Path, required=True, help="JSON report path; keep it outside the checkout"
    )
    parser.add_argument("--analyses", default=",".join(ANALYSES))
    parser.add_argument(
        "--expected-commit",
        default=None,
        help="commit to require instead of the local artifacts-v1 tag",
    )
    args = parser.parse_args(argv)
    repo, data, release = args.repo.resolve(), args.data_dir.resolve(), args.release_dir.resolve()
    report = Report()
    env = check_environment(repo, report, args.expected_commit)
    verify_archives(release, data, args.extract, report)
    verify_models(repo, data, report)
    os.environ["FARC_DATA_DIR"] = str(data)  # never fall back to a default data directory
    os.chdir(repo)
    sys.path.insert(0, str(repo / "src"))
    analyses = {}
    if not report.failed:
        for name in [a for a in args.analyses.split(",") if a]:
            if name not in ANALYSES:
                report.add("analyses", f"unknown analysis {name}", False)
                continue
            print(f"[verify] {name} ...", flush=True)
            try:
                analyses[name] = run_analysis(name, data, report)
            except Exception as exc:  # report, do not hide
                report.add(name, "analysis ran", False, f"{type(exc).__name__}: {exc}"[:400])
    else:
        report.add("analyses", "skipped because artifact or environment checks failed", False)
    text = json.dumps({"tag": TAG, "environment": env, "analyses": analyses,
                       "checks": report.checks, "failed": len(report.failed),
                       "passed": len(report.failed) == 0}, indent=2, default=str)  # fmt: skip
    for p, label in ((str(data), "<DATA_DIR>"), (str(release), "<RELEASE_DIR>"),
                     (str(repo), "<REPO>"), (str(Path.home()), "<HOME>")):  # fmt: skip
        text = text.replace(p, label)
    args.report.write_text(text + "\n")
    print(f"[verify] {len(report.checks)} checks, {len(report.failed)} failed -> {args.report}")
    for c in report.failed:
        print(f"  FAILED {c['group']}: {c['name']} {c['detail'] or ''}")
    return 0 if not report.failed else 1


if __name__ == "__main__":
    sys.exit(main())
