"""P1 amendment check (docs/HARDWARE_H1_AMENDMENT_OPTIONS.md). SYNTHETIC only.

A deterministic synthetic kinematic plant, not the simulator and not a measurement: the gripper
velocity is the zero-order-hold command (scaled by K0 * 0.05 m per 40 ms step) delayed by a
transport lag L; positions reach the controller S ms after they are measured. The unchanged
Phase E estimator runs on the positions the controller would receive. For each (L, S) the
script reports:

- the estimator's discrete lag (mode over episodes) - what Q5's d0 would be;
- d0_ts under each candidate latency definition:
  - ``prereg``: section 4.2 as written (linear interpolation, measurement timestamps);
  - ``A``: zero-order-hold command resampling, measurement timestamps;
  - ``B``: commands stamped at their hold-interval centre (linear), measurement timestamps;
  - ``C``: prereg with a -20 ms correction;
  - ``A+S``: zero-order hold, with velocity samples stamped when the controller receives them;
- the share of (L, S) points where d0_ts equals the estimator's d0 (criterion 5.1 (d)).

    uv run python scripts/h1_p1_amendment_check.py --out results/hardware_h1_amendments
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from farc.adaptation.estimator import ActuatorEstimator, EstimatorConfig
from farc.hardware.controller import ESTIMATOR
from farc.hardware.h1_analysis import LAG_RANGE_MS, Latency, _ncc

STEP_NS = 40_000_000
K0 = 0.62
SEED = 20264200  # synthetic fixture seed (not an episode seed)


# PRE-AMENDMENT copy of the section 4.2 method (linear interpolation, as first written), kept
# so the committed p1_check.json reproduces exactly. The locked analysis now uses H1-A1.
def episode_latency(cmd_t_ns: np.ndarray, cmd_xy: np.ndarray, vel_t_ns: np.ndarray,
                    vel_xy: np.ndarray) -> Latency:  # fmt: skip
    """L_ep: the lag in [0, 300] ms maximising the normalised cross-correlation between
    c(t - lag) (driver-send timestamps) and v(t) (proprioception timestamps), both linearly
    interpolated onto a common 1 ms grid. Sensor latency is not subtracted (4.2 step 3)."""
    cmd_t, vel_t = np.asarray(cmd_t_ns, np.int64), np.asarray(vel_t_ns, np.int64)
    if np.any(np.diff(cmd_t) <= 0) or np.any(np.diff(vel_t) <= 0):
        raise ValueError("timestamps must increase strictly")
    t0 = int(max(cmd_t[0], vel_t[0]))
    t1 = int(min(cmd_t[-1], vel_t[-1]))
    span = (t1 - t0) // 1_000_000
    if span <= LAG_RANGE_MS[1] + 100:
        raise ValueError("overlap too short for the 0-300 ms lag search")
    grid = np.arange(0, span + 1, dtype=np.float64)
    off_c, off_v = (t0 - cmd_t[0]) / 1e6, (t0 - vel_t[0]) / 1e6
    lags = range(LAG_RANGE_MS[0], LAG_RANGE_MS[1] + 1)
    scores = np.zeros((2, len(lags)))
    for axis in range(2):
        c = np.interp(grid + off_c, (cmd_t - cmd_t[0]) / 1e6, np.asarray(cmd_xy)[:, axis])
        v = np.interp(grid + off_v, (vel_t - vel_t[0]) / 1e6, np.asarray(vel_xy)[:, axis])
        for j, lag in enumerate(lags):
            scores[axis, j] = _ncc(c[: len(c) - lag], v[lag:])
    per_axis = (int(np.nanargmax(scores[0])), int(np.nanargmax(scores[1])))
    pooled = np.nanmean(scores, axis=0)
    j = int(np.nanargmax(pooled))
    return Latency(lags[j], (lags[per_axis[0]], lags[per_axis[1]]), float(pooled[j]))


def plant(cmd: np.ndarray, lag_ms: float, rate_hz: int = 1000):
    """Velocity (m/s) on a fine grid from ZOH commands delayed by lag_ms."""
    n = len(cmd)
    dt_ns = 1_000_000_000 // rate_hz
    t = np.arange(0, (n + 12) * STEP_NS, dt_ns, dtype=np.int64)
    j = np.floor((t - lag_ms * 1e6) / STEP_NS).astype(int)
    v = np.where(((j >= 0) & (j < n))[:, None], cmd[np.clip(j, 0, n - 1)], 0.0)
    v = v * (K0 * 0.05 / 0.04)
    p = np.cumsum(v, 0) * (dt_ns / 1e9)
    return t, v, p


def run(lag_ms: float, sensor_ms: float, rng: np.random.Generator) -> dict:
    cmd = rng.uniform(-1, 1, (50, 2))
    t, v, p = plant(cmd, lag_ms)
    cmd_t = np.arange(50, dtype=np.int64) * STEP_NS
    # estimator: at decision k (time k*40 ms) it receives the position measured S ms earlier
    est = ActuatorEstimator(EstimatorConfig(**ESTIMATOR))
    for k in range(50):
        when = k * STEP_NS - int(sensor_ms * 1e6)
        idx = max(0, int(np.searchsorted(t, when, side="right")) - 1) if when >= 0 else 0
        g = np.r_[p[idx], 0.0] if when >= 0 else np.zeros(3)
        est.observe(g)
        est.issued(np.r_[cmd[k], 0.0, 0.0])
    e = est.estimate()
    # velocity samples at 500 Hz, measurement stamps vs arrival stamps (+S)
    sl = slice(0, None, 2)
    vt, vv = t[sl], v[sl]
    out: dict[str, Any] = {"est_lag": e.delay if e.delay_reliable else None}
    fine = np.arange(cmd_t[0], cmd_t[-1] + STEP_NS, 1_000_000, dtype=np.int64)
    zoh = cmd[np.clip((fine // STEP_NS).astype(int), 0, 49)]
    out["prereg"] = episode_latency(cmd_t, cmd, vt, vv).pooled_ms
    out["A"] = episode_latency(fine, zoh, vt, vv).pooled_ms
    out["B"] = episode_latency(cmd_t + STEP_NS // 2, cmd, vt, vv).pooled_ms
    out["C"] = out["prereg"] - 20
    out["A+S"] = episode_latency(fine, zoh, vt + int(sensor_ms * 1e6), vv).pooled_ms
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/hardware_h1_amendments")
    ap.add_argument("--episodes", type=int, default=20)
    a = ap.parse_args()
    rng = np.random.default_rng(SEED)
    rows = []
    for sensor in (0, 15, 30):
        for lag in range(5, 150, 10):
            eps = [run(lag, sensor, rng) for _ in range(a.episodes)]
            lags = [x["est_lag"] for x in eps if x["est_lag"] is not None]
            d0 = Counter(lags).most_common(1)[0][0] if lags else None
            row = {"lag_ms": lag, "sensor_ms": sensor, "reliable": len(lags), "est_d0": d0,
                   "model_round_L_plus_S": int(np.floor((lag + sensor) / 40 + 0.5))}  # fmt: skip
            for m in ("prereg", "A", "B", "C", "A+S"):
                med = float(np.median([x[m] for x in eps]))
                row[f"{m}_median_ms"] = med
                row[f"{m}_d0_ts"] = int(np.floor(med / 40 + 0.5))
                row[f"{m}_ambiguous"] = bool(abs((med % 40) - 20) <= 8)
                row[f"{m}_agrees"] = d0 is not None and row[f"{m}_d0_ts"] == d0
            rows.append(row)
    summary: dict[str, Any] = {m: {"agree_share": float(np.mean([r[f"{m}_agrees"] for r in rows])),
                   "ambiguous_share": float(np.mean([r[f"{m}_ambiguous"] for r in rows]))}
               for m in ("prereg", "A", "B", "C", "A+S")}  # fmt: skip
    summary["est_matches_round_L_plus_S"] = float(np.mean(
        [r["est_d0"] == r["model_round_L_plus_S"] for r in rows]))  # fmt: skip
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"label": "SYNTHETIC kinematic model", "summary": summary, "rows": rows}
    (out / "p1_check.json").write_text(json.dumps(payload, indent=1))
    print(json.dumps(summary, indent=1))
    for r in rows:
        print(
            r["sensor_ms"],
            r["lag_ms"],
            r["est_d0"],
            r["model_round_L_plus_S"],
            {m: r[f"{m}_d0_ts"] for m in ("prereg", "A", "B", "C", "A+S")},
        )


if __name__ == "__main__":
    main()
