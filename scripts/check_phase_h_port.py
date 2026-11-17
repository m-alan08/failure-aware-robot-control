"""Phase H Stage-0 port check (code check only; no evaluation seed is used).

Trains the same short run twice, with the upstream quantrl-fetch trainer (``--upstream-src``,
a checkout at 95e9611) and with the ported ``farc.retrain.qrl`` trainer (no wrapper), in
separate processes, and compares every tensor of the two checkpoints and the normaliser
state bitwise. Also reports the wall-clock throughput, for the Stage-1 compute estimate.

    uv run python scripts/check_phase_h_port.py --upstream-src <QUANTRL>/src --workdir <TMP>
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

SNIPPET = """
import json, sys
from {pkg}.config import config_from_dict
from {pkg}.training import train
values = json.loads(sys.argv[1])
train(config_from_dict(values), verbose=False)
"""


def run(pkg: str, values: dict, pythonpath: str) -> float:
    env = {**os.environ, "PYTHONPATH": pythonpath}
    started = time.time()
    subprocess.run([sys.executable, "-c", SNIPPET.format(pkg=pkg), json.dumps(values)],
                   check=True, env=env)  # fmt: skip
    return time.time() - started


def tensors(obj, prefix=""):
    if isinstance(obj, torch.Tensor):
        yield prefix, obj
    elif isinstance(obj, np.ndarray):
        yield prefix, torch.as_tensor(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from tensors(v, f"{prefix}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from tensors(v, f"{prefix}[{i}]")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--upstream-src", required=True)
    p.add_argument("--workdir", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--report", default=None)
    a = p.parse_args()
    base = json.loads(Path("configs/retrain/phase_h.json").read_text())["training"]
    out = {}
    sources = (
        ("upstream", "quantrl", a.upstream_src),
        ("ported", "farc.retrain.qrl", str(Path("src").resolve())),
    )
    for label, pkg, path in sources:
        values = json.loads(json.dumps(base))
        values.update(name=f"stage0_{label}", seed=0, output_dir=str(Path(a.workdir) / label))
        values["train"]["total_steps"] = a.steps
        values["train"]["eval_interval"] = a.steps
        out[label] = run(pkg, values, path)

    def load(lab: str):
        path = Path(a.workdir) / lab / f"stage0_{lab}_seed0" / "checkpoint.pt"
        return torch.load(path, map_location="cpu", weights_only=False)

    u, q = load("upstream"), load("ported")
    tu = dict(tensors({"agent": u["agent"], "normalizer": u["normalizer"]}))
    tq = dict(tensors({"agent": q["agent"], "normalizer": q["normalizer"]}))
    same_keys = tu.keys() == tq.keys()
    diffs = [k for k in tu if k in tq and not torch.equal(tu[k], tq[k])]
    report = {"steps": a.steps, "tensors": len(tu), "same_keys": same_keys,
              "differing_tensors": diffs, "bitwise_identical": same_keys and not diffs,
              "seconds": {k: round(v, 1) for k, v in out.items()},
              "seconds_per_1M_steps_estimate": round(out["ported"] / a.steps * 1e6)}  # fmt: skip
    text = json.dumps(report, indent=2)
    print(text)
    if a.report:
        Path(a.report).write_text(text + "\n")
    sys.exit(0 if report["bitwise_identical"] else 1)


if __name__ == "__main__":
    main()
