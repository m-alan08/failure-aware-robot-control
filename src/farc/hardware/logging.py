"""Structured, tamper-evident step log (JSON Lines).

Each line is one control step. ``seq`` increases by one; ``prev_sha256`` is the SHA-256 of the
previous line's bytes (the first line chains to the header). ``verify_log`` re-checks the chain,
the schema and the sequence. Raw logs are write-once: ``StepLogger`` refuses to open an
existing file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "h1-2"  # h0-1 + shadow fields (h1-1) + receipt time (H1-A1)
# field -> allowed JSON types (None allowed where listed)
SCHEMA: dict[str, tuple[type, ...]] = {
    "seq": (int,), "prev_sha256": (str,),
    "t": (int,), "configuration": (str,),
    "mono_ns": (int,), "obs_gripper_stamp_ns": (int, type(None)),
    "obs_object_stamp_ns": (int, type(None)),
    "obs_gripper_receipt_ns": (int, type(None)),
    "injected_delay": (int, type(None)), "executed_source_step": (int, type(None)),
    "executed_command": (list, type(None)), "command": (list, type(None)),
    "base_action": (list, type(None)),
    "est_delay": (int, type(None)), "est_delay_reliable": (bool, type(None)),
    "est_gain": (float, int, type(None)), "est_gain_reliable": (bool, type(None)),
    "delay_used": (int, type(None)), "gain_used": (float, int, type(None)),
    "active": (bool, type(None)), "alarm": (bool, type(None)), "suspended": (bool, type(None)),
    "supervisor_mode": (str, type(None)),
    "cusum_stats": (list, type(None)), "cusum_trusted": (int, type(None)),
    "cusum_reset_start": (int, type(None)), "cusum_sigma2": (float, type(None)),
    "pred_gripper_position": (list, type(None)), "pred_object_position": (list, type(None)),
    "pred_object_motion_m": (float, type(None)),
    "contact_measured": (bool, type(None)),  # from a force/torque sensor; None offline
    "learned_calls": (int, type(None)),
    "lat_adapter_ns": (int, type(None)), "lat_base_policy_ns": (int, type(None)),
    "lat_estimator_observe_ns": (int, type(None)),
    "lat_identification_supervision_ns": (int, type(None)),
    "lat_predictor_ns": (int, type(None)), "lat_compensated_policy_ns": (int, type(None)),
    "lat_controller_total_ns": (int, type(None)), "lat_pipeline_total_ns": (int, type(None)),
    "error": (str, type(None)), "safety_event": (str, type(None)),
    # Candidate C shadow (logging only; None when the shadow is disabled)
    "shadow_alarm": (bool, type(None)), "shadow_suspended": (bool, type(None)),
    "shadow_active": (bool, type(None)), "shadow_trusted": (int, type(None)),
    "shadow_cusum_stats": (list, type(None)), "lat_shadow_ns": (int, type(None)),
}  # fmt: skip


def _line(obj: dict[str, Any]) -> bytes:
    return (json.dumps(obj, sort_keys=True, allow_nan=False) + "\n").encode()


def validate_record(rec: dict[str, Any]) -> None:
    missing = set(SCHEMA) - set(rec)
    extra = set(rec) - set(SCHEMA)
    if missing or extra:
        raise ValueError(f"schema mismatch: missing {sorted(missing)}, extra {sorted(extra)}")
    for k, types in SCHEMA.items():
        v = rec[k]
        if isinstance(v, bool) and bool not in types:
            raise ValueError(f"{k}: bool not allowed")
        if not isinstance(v, types):
            raise ValueError(f"{k}: {type(v).__name__} not in {[t.__name__ for t in types]}")


def empty_record() -> dict[str, Any]:
    return {k: None for k in SCHEMA if k not in ("seq", "prev_sha256")}


class StepLogger:
    def __init__(self, path: str | Path, header: dict[str, Any]):
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"{self.path} exists; raw logs are write-once")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("xb")
        head = _line({"schema_version": SCHEMA_VERSION, "header": header})
        self._fh.write(head)
        self._prev = hashlib.sha256(head).hexdigest()
        self.seq = 0

    def write(self, rec: dict[str, Any]) -> None:
        full = {**rec, "seq": self.seq, "prev_sha256": self._prev}
        validate_record(full)
        line = _line(full)
        self._fh.write(line)
        self._prev = hashlib.sha256(line).hexdigest()
        self.seq += 1

    def close(self) -> str:
        self._fh.close()
        return self._prev

    def __enter__(self) -> StepLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def verify_log(path: str | Path) -> int:
    """Returns the number of step records; raises on any break in schema, chain or sequence."""
    lines = Path(path).read_bytes().splitlines(keepends=True)
    head = json.loads(lines[0])
    if head.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown schema version")
    prev = hashlib.sha256(lines[0]).hexdigest()
    for i, line in enumerate(lines[1:]):
        rec = json.loads(line)
        if rec["seq"] != i or rec["prev_sha256"] != prev:
            raise ValueError(f"chain broken at record {i}")
        validate_record(rec)
        prev = hashlib.sha256(line).hexdigest()
    return len(lines) - 1
