"""Locked H1 analysis (docs/HARDWARE_H1_PREREGISTRATION.md, sections 4-5).

Every threshold below is copied from the pre-registration and must not be changed without a
new, separately approved protocol. Section references are to that document.

This module only analyses records it is given. It never reads or fabricates hardware data;
the tests use clearly labelled SYNTHETIC fixtures.

Implementation choices where the protocol leaves a detail open are listed in
``IMPLEMENTATION_NOTES`` and in docs/HARDWARE_H1_SOFTWARE_READINESS.md (section 3); they are
flagged for confirmation, not silently decided.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --- pre-registered constants (do not edit) ----------------------------------------------------
STEP_MS = 40.0
LAG_RANGE_MS = (0, 300)  # section 4.2
Q5_EPISODES = 20
Q5_MIN_RELIABLE = 15  # 5.1 (a)
Q5_MIN_AGREEMENT = 0.80  # 5.1 (b)
HALF_STEP_BAND_MS = 8.0  # 5.1: ambiguous if (median mod 40) within +-8 ms of 20 ms
MAX_LAG = 4  # estimator maximum lag: eligibility d* <= 4 (5.2)
Q1_EPISODES_PER_DELAY = 20
FEASIBILITY_MIN_CORRECT = 18  # 5.2
WRONG_LATCH_MAX = 1  # 5.2
ADDITIVITY_POINT_MS = 10.0  # 5.3 (ii)
ADDITIVITY_CI_MS = 20.0  # 5.3 (ii)
BOOTSTRAP_RESAMPLES = 2_000  # 5.3
BOOTSTRAP_SEED = 20264101  # 5.3
CONFIDENCE = 0.95
GAIN_RANGE = (0.7, 1.3)  # 5.2
NOISE_FLAG_M = 0.0025  # 5.4
SAFETY_STOP_LIMIT = 3  # section 6

IMPLEMENTATION_NOTES = {
    "pooled_latency": "L_ep pooled over x and y = the lag maximising the mean of the two per-axis "
    "normalised cross-correlations (section 4.2 says 'separately and then pooled').",
    "aborted_episodes": "an aborted or invalid-log episode counts as not correct and not a wrong "
    "latch; it stays in the 20-episode denominator (safety stop = failure, proposal 4.5).",
    "bootstrap_pairing": "the paired bootstrap pairs Q5 and Q1 episodes by excitation-sequence id; "
    "if no ids are shared the additivity test is reported as not computable (= not passed).",
    "half_period_offset": "FINDING P1 (not resolved here): with zero-order-hold 25 Hz commands, "
    "the pre-registered linear interpolation makes L_ep read about +20 ms (half a period) above "
    "the transport lag. It cancels in the additivity difference but can shift d0_ts by one step "
    "relative to the estimator's d0 (criterion 5.1 (d)). Implemented as written; needs a "
    "decision before execution.",
    "rounding": "d0_ts = floor(median/40 + 0.5); exact half-steps are excluded by the ambiguity "
    "band anyway.",
}


# --- exact binomial bounds -----------------------------------------------------------------------


def _binom_cdf(k: int, n: int, p: float) -> float:
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k + 1))


def _solve(f, lo: float = 0.0, hi: float = 1.0) -> float:
    for _ in range(200):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def clopper_pearson_lower(k: int, n: int, alpha: float = 1 - CONFIDENCE) -> float:
    """One-sided exact lower bound: the p with P(X >= k | p) = alpha."""
    if k == 0:
        return 0.0
    return _solve(lambda p: 1 - _binom_cdf(k - 1, n, p) >= alpha)


def clopper_pearson_interval(k: int, n: int, alpha: float = 1 - CONFIDENCE) -> tuple[float, float]:
    lo = clopper_pearson_lower(k, n, alpha / 2)
    hi = 1.0 if k == n else _solve(lambda p: _binom_cdf(k, n, p) <= alpha / 2)
    return lo, hi


# --- section 4.2: timestamp latency ------------------------------------------------------------


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a @ a) * (b @ b)))
    return float(a @ b) / den if den > 0 else float("nan")


@dataclass(frozen=True)
class Latency:
    pooled_ms: int
    per_axis_ms: tuple[int, int]
    peak_correlation: float


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


# --- episode records -------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeResult:
    """One H1 episode as the analysis consumes it (built from a verified step log)."""

    block: str  # "Q5" | "Q1" | "Q5-repeat"
    sequence_id: int
    injected_delay: int
    latched_series: tuple[int, ...]  # the latch's delay at every step (0 = none)
    final_reliable: bool  # estimator delay_reliable at step 49
    final_lag: int | None  # estimator least-squares lag at step 49
    gain_estimate: float | None  # final gain estimate g-hat
    latency_ms: float | None  # L_ep (None if not computable)
    fifo_exact: bool  # executed_source_step == t - d at every step
    aborted: bool = False
    abort_reason: str | None = None
    log_valid: bool = True

    @property
    def usable(self) -> bool:
        return self.log_valid and not self.aborted

    @property
    def final_latched(self) -> int:
        return self.latched_series[-1] if self.latched_series else 0


# --- 5.1 Q5 ---------------------------------------------------------------------------------------


@dataclass
class Q5Result:
    reliable: int
    episodes: int
    mode: int | None
    agreement: float
    median_latency_ms: float | None
    d0_ts: int | None
    ambiguous: bool
    d0: int | None
    resolved: bool
    reasons: list[str] = field(default_factory=list)
    expected: dict[int, int] = field(default_factory=dict)  # d -> d*


def resolve_q5(episodes: list[EpisodeResult]) -> Q5Result:
    eps = [e for e in episodes if e.block == "Q5"]
    reasons = []
    if len(eps) != Q5_EPISODES:
        reasons.append(f"expected {Q5_EPISODES} Q5 episodes, got {len(eps)}")
    lags = [e.final_lag for e in eps if e.usable and e.final_reliable and e.final_lag is not None]
    reliable = len(lags)
    counts = Counter(lags)
    top = counts.most_common()
    mode = top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else None
    agreement = (top[0][1] / reliable) if reliable else 0.0
    if reliable < Q5_MIN_RELIABLE:
        reasons.append(f"(a) only {reliable} reliable Q5 episodes (< {Q5_MIN_RELIABLE})")
    if mode is None:
        reasons.append("(b) no unique mode")
    elif agreement < Q5_MIN_AGREEMENT:
        reasons.append(f"(b) mode agreement {agreement:.2f} < {Q5_MIN_AGREEMENT}")
    lat = [e.latency_ms for e in eps if e.usable and e.latency_ms is not None]
    med = float(np.median(lat)) if lat else None
    d0_ts = ambiguous = None
    if med is None:
        reasons.append("(c) no timestamp latency available")
        ambiguous = True
    else:
        d0_ts = int(math.floor(med / STEP_MS + 0.5))
        ambiguous = abs((med % STEP_MS) - STEP_MS / 2) <= HALF_STEP_BAND_MS
        if ambiguous:
            reasons.append(f"(c) median latency {med:.1f} ms is within the half-step band")
    if mode is not None and d0_ts is not None and mode != d0_ts:
        reasons.append(f"(d) estimator d0 = {mode} differs from timestamp d0_ts = {d0_ts}")
    resolved = not reasons
    res = Q5Result(reliable, len(eps), mode, agreement, med, d0_ts, bool(ambiguous),
                   mode if resolved else None, resolved, reasons)  # fmt: skip
    if resolved:
        assert mode is not None
        res.expected = {d: d + mode for d in range(MAX_LAG + 1)}
    return res


# --- 5.3 additivity --------------------------------------------------------------------------


@dataclass
class Additivity:
    delay: int
    fifo_exact: bool
    delta_ms: float | None
    ci_ms: tuple[float, float] | None
    passed: bool
    reason: str | None


def additivity(q5: list[EpisodeResult], q1_d: list[EpisodeResult], d: int) -> Additivity:
    fifo = all(e.fifo_exact for e in q1_d if e.log_valid)
    a = {e.sequence_id: e.latency_ms for e in q5 if e.usable and e.latency_ms is not None}
    b = {e.sequence_id: e.latency_ms for e in q1_d if e.usable and e.latency_ms is not None}
    if not a or not b:
        return Additivity(d, fifo, None, None, False, "latency not computable")
    delta = float(np.median(list(b.values())) - np.median(list(a.values())) - d * STEP_MS)
    shared = sorted(set(a) & set(b))
    if not shared:
        return Additivity(d, fifo, delta, None, False, "no shared sequence ids for pairing")
    rng = np.random.default_rng([BOOTSTRAP_SEED, d])
    av, bv = np.array([a[s] for s in shared]), np.array([b[s] for s in shared])
    idx = rng.integers(0, len(shared), (BOOTSTRAP_RESAMPLES, len(shared)))
    boot = np.median(bv[idx], axis=1) - np.median(av[idx], axis=1) - d * STEP_MS
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    reason = None
    if not fifo:
        reason = "FIFO not exact"
    elif abs(delta) > ADDITIVITY_POINT_MS:
        reason = f"|delta| {abs(delta):.1f} ms > {ADDITIVITY_POINT_MS} ms"
    elif ci[0] < -ADDITIVITY_CI_MS or ci[1] > ADDITIVITY_CI_MS:
        reason = f"CI {ci} not within +-{ADDITIVITY_CI_MS} ms"
    return Additivity(d, fifo, delta, ci, reason is None, reason)


# --- 5.2 Q1 ------------------------------------------------------------------------------------


@dataclass
class DelayResult:
    delay: int
    expected: int
    eligible: bool
    episodes: int
    correct: int
    wrong_latch: int
    transient_false_latch: int | None  # only defined when d* = 0
    abstained: int  # final latched 0 while d* > 0 (incorrect, not a wrong latch)
    aborted: int
    invalid_logs: int
    one_sided_lower: float
    exact_ci: tuple[float, float]
    additivity: Additivity
    status: str  # passed | failed | not_interpretable | ineligible
    reasons: list[str] = field(default_factory=list)


def analyse_delay(d: int, expected: int, q5: list[EpisodeResult],
                  q1_d: list[EpisodeResult]) -> DelayResult:  # fmt: skip
    n = len(q1_d)
    reasons = []
    if n != Q1_EPISODES_PER_DELAY:
        reasons.append(f"expected {Q1_EPISODES_PER_DELAY} episodes, got {n}")
    correct = wrong = abstained = 0
    transient = 0 if expected == 0 else None
    for e in q1_d:
        if not e.usable:
            continue
        f = e.final_latched
        if f == expected:
            correct += 1
        if expected > 0:
            if f > 0 and f != expected:
                wrong += 1
            if f == 0:
                abstained += 1
        else:
            if f > 0:
                wrong += 1  # false positive
            if any(x > 0 for x in e.latched_series):
                assert transient is not None
                transient += 1
    add = additivity(q5, q1_d, d)
    eligible = expected <= MAX_LAG
    if not eligible:
        status = "ineligible"
    elif not add.passed:
        status = "not_interpretable"
        reasons.append(f"additivity violated: {add.reason}")
    else:
        if correct < FEASIBILITY_MIN_CORRECT:
            reasons.append(f"{correct}/{n} correct < {FEASIBILITY_MIN_CORRECT}")
        if wrong > WRONG_LATCH_MAX:
            reasons.append(f"{wrong} wrong latches > {WRONG_LATCH_MAX}")
        status = "failed" if reasons else "passed"
    return DelayResult(d, expected, eligible, n, correct, wrong, transient, abstained,
                       sum(e.aborted for e in q1_d), sum(not e.log_valid for e in q1_d),
                       clopper_pearson_lower(correct, n) if n else 0.0,
                       clopper_pearson_interval(correct, n) if n else (0.0, 1.0),
                       add, status, reasons)  # fmt: skip


# --- 5.4 sensor noise, gain, safety accounting ------------------------------------------------


def sensor_noise(t_ns: np.ndarray, positions: np.ndarray, label: str) -> dict[str, Any]:
    """Per-axis std and lag-1 autocorrelation after removing a linear trend; achieved rate and
    timestamp jitter (5.4)."""
    t = (np.asarray(t_ns, np.int64) - int(t_ns[0])) / 1e9
    p = np.asarray(positions, np.float64)
    out: dict[str, Any] = {"label": label, "samples": len(t)}
    dt = np.diff(t)
    out["rate_hz"] = float(1.0 / dt.mean())
    out["jitter_ms"] = float(dt.std() * 1e3)
    stds, acs = [], []
    for axis in range(p.shape[1]):
        resid = p[:, axis] - np.polyval(np.polyfit(t, p[:, axis], 1), t)
        stds.append(float(resid.std()))
        acs.append(float(np.corrcoef(resid[:-1], resid[1:])[0, 1]))
    out["std_m"], out["lag1_autocorr"] = stds, acs
    out["flag_over_2p5mm"] = label == "gripper" and max(stds) > NOISE_FLAG_M
    return out


def gain_summary(q1: list[EpisodeResult]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for d in sorted({e.injected_delay for e in q1}):
        g = [e.gain_estimate for e in q1 if e.injected_delay == d and e.usable
             and e.gain_estimate is not None]  # fmt: skip
        if not g:
            out[d] = {"n": 0, "median": None, "iqr": None, "outside_range": None}
            continue
        med = float(np.median(g))
        out[d] = {"n": len(g), "median": med,
                  "iqr": (float(np.percentile(g, 25)), float(np.percentile(g, 75))),
                  "outside_range": not (GAIN_RANGE[0] <= med <= GAIN_RANGE[1])}  # fmt: skip
    return out


def safety_accounting(episodes: list[EpisodeResult]) -> dict[str, Any]:
    stops = [e for e in episodes if e.aborted]
    reasons = Counter(e.abort_reason or "unspecified" for e in stops)
    return {"safety_stops": len(stops), "by_reason": dict(reasons),
            "by_block": dict(Counter(e.block for e in stops)),
            "session_must_stop": len(stops) >= SAFETY_STOP_LIMIT,
            "invalid_logs": sum(not e.log_valid for e in episodes)}  # fmt: skip


# --- top level ------------------------------------------------------------------------------


def analyse(episodes: list[EpisodeResult]) -> dict[str, Any]:
    q5_eps = [e for e in episodes if e.block == "Q5"]
    q5 = resolve_q5(episodes)
    report: dict[str, Any] = {"q5": q5, "safety": safety_accounting(episodes),
                              "implementation_notes": IMPLEMENTATION_NOTES}  # fmt: skip
    rep = [e for e in episodes if e.block == "Q5-repeat" and e.usable and e.latency_ms is not None]
    report["drift"] = {"q5_repeat_median_ms": float(np.median([e.latency_ms for e in rep]))
                       if rep else None, "q5_median_ms": q5.median_latency_ms}  # fmt: skip
    if not q5.resolved:
        report["q1"] = {"status": "not_run", "reason": "Q5 unresolved (section 5.1)"}
        return report
    q1 = [e for e in episodes if e.block == "Q1"]
    delays = [analyse_delay(d, q5.expected[d], q5_eps, [e for e in q1 if e.injected_delay == d])
              for d in range(MAX_LAG + 1)]  # fmt: skip
    eligible = [r for r in delays if r.eligible]
    report["q1"] = {"status": "passed" if eligible and all(r.status == "passed" for r in eligible)
                    else "not_passed", "delays": delays,
                    "gain": gain_summary(q1)}  # fmt: skip
    return report
