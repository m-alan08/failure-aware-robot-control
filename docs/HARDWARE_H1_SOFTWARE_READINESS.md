# H1 pre-execution software: status and readiness checklist

**Status:**
- **H1 execution is NOT approved.** This work is software only.
- No robot was connected or moved, there was no commissioning, nothing was purchased or booked,
  and no simulation episodes were run.
- No held-out seeds were used, and no frozen model, constant or scientific logic changed.

**Authoritative protocol:** `docs/HARDWARE_H1_PREREGISTRATION.md` (PR #25).

## 1. Implemented

| item | where | protocol section |
|---|---|---|
| Candidate C shadow mode | `src/farc/hardware/shadow.py`; `HardwarePipeline(..., shadow=...)` | §3.4 |
| Timestamp-based latency L_ep | `h1_analysis.episode_latency` | §4.2 |
| Q5 natural-delay resolution, criteria (a)–(d), frozen expected answers | `h1_analysis.resolve_q5` | §5.1 |
| Per-delay counts, exact Clopper–Pearson bounds, wrong latches, transient false latches, abstentions, eligibility | `h1_analysis.analyse_delay` | §5.2 |
| Additivity test (FIFO exactness, \|Δ_d\| ≤ 10 ms, paired bootstrap CI within ±20 ms, 2,000 resamples, seed 20264101) | `h1_analysis.additivity` | §5.3 |
| Safety-stop accounting, including the 3-stop session limit | `h1_analysis.safety_accounting` | §6 |
| Sensor noise (detrended std, lag-1 autocorrelation, rate, jitter, 2.5 mm flag) and gain summary (median, IQR, [0.7, 1.3] flag) | `h1_analysis.sensor_noise`, `gain_summary` | §5.4, §5.2 |
| Top-level locked analysis: Q1 is "not_run" if Q5 is unresolved | `h1_analysis.analyse` | §5 |
| Log schema `h1-1`: the H0 fields plus the shadow fields | `src/farc/hardware/logging.py` | §5.5 |

**Shadow design:**
- The shadow owns an independent `ActuatorEstimator` and a `Sequential` supervisor built by
  `mechanism.sequential` with the frozen constants.
- It is stepped only after the active command is final, on *copies* of the observed gripper
  position and the issued command.
- It returns a log dictionary only, and holds no reference to the controller, the FIFO or any
  command path.
- After a safety stop it is not stepped.

## 2. Verification

**Shadow isolation:** active commands and FIFO outputs are **bitwise identical** with the shadow
on and off.

| check | cases |
|---|---|
| **SYNTHETIC** inputs (runs in CI) | <ul><li>F, K, CA and C × 4 delay schedules, including increases, decreases and multi-pulse changes</li><li>a missing observation and a stale observation mid-episode: same safety stop, no executed command, shadow not stepped</li></ul> |
| **Recorded** F1 development trajectories with the real frozen policy (skipped where data are absent) | all four configurations, under delay changes, with a missing sample |

When the active controller is C, the shadow's alarms, suspension, trusted lag and CUSUM
statistics equal the active controller's at every step.

**Analysis tests (SYNTHETIC fixtures only, labelled in code):**
- **Latency recovery (as amended by H1-A1):** a zero-order-hold plant gives back its transport lag (0–290 ms, ±2 ms); receipt-time stamps give lag + sensor latency; smooth signals agree within the hold quantisation.
- **Exact bounds:** 18/20 → 0.717; 20/20 → 0.861; two-sided bounds for 10/20.
- **Q5:** resolution, and each unresolved condition (a)–(d), including ties, the half-step band
  edges (8.0 ms in, 8.1 ms out), and aborted episodes.
- **Q1 per delay:**
  - pass at 18/20 with abstentions;
  - fail at 17/20;
  - fail on 2 wrong latches;
  - aborted episodes count as not correct;
  - additivity violation and an inexact FIFO make a delay "not_interpretable" even at 20/20;
  - d\* > 4 is ineligible.
- **Zero total delay:** false positives and transient false latches.
- **Other checks:**
  - bootstrap determinism;
  - unpaired sequence ids;
  - the 3-stop safety limit;
  - the noise and gain flags.
- **Thresholds:** a test asserts that every pre-registered threshold equals the document's text.

**H0 still holds:**
- The H0 tests pass.
- The **full H0 replay was re-run** after the pipeline change. The summary and all
  per-episode outputs are **byte-identical** to the committed H0 results (8,100 episodes,
  bitwise controller equivalence).

**Suite:** 492 tests pass (436 before, plus 56). Ruff, the formatter and mypy are clean.

## 3. Protocol questions found (**resolved**: amendments H1-A1–A4 adopted 2026-11-18)

**Resolution:** P1 (with P5, sensor latency) → H1-A1; P2 → H1-A2; P3 → H1-A3; P4 → H1-A4. See
the pre-registration §10 and `docs/HARDWARE_H1_AMENDMENT_OPTIONS.md`. The analysis implements
them, and the tests check the amended behaviour. The table below records the findings as
originally raised.

| # | finding | effect | options (for a numbered amendment, your call) |
|---|---|---|---|
| **P1** (→ H1-A1) | **Half-period offset in the pre-registered latency method.** Commands are held for 40 ms (zero-order hold). §4.2 resamples c(t) by *linear interpolation*, which places each command at its tick rather than over its hold interval. On SYNTHETIC data, L_ep then reads **the true transport lag plus about 20 ms**. | It cancels in the additivity difference Δ_d (§5.3). It **can shift d₀_ts by one step** relative to the estimator's d₀, so criterion 5.1 (d) could fail, or pass, for a reason unrelated to the plant. | (a) resample commands as a zero-order hold; (b) timestamp each command at its hold-interval centre; (c) keep the method and define d₀_ts with a −20 ms correction. Implemented as written until you decide; a test pins the current behaviour. |
| P2 | **"Pooled over x and y" is not defined precisely** (§4.2) | the choice of pooled lag | implemented as the lag maximising the mean of the per-axis normalised correlations; please confirm |
| P3 | **Aborted episodes in the 20-episode denominator** | 18/20 accounting | implemented as not correct and not a wrong latch, kept in the denominator (safety stop = failure); please confirm |
| P4 | **Sequence assignment.** §3.1 selects 60 sequences, while §3.3 runs each of 5 delays × 20 sequences in 3 blocks and Q5 uses sequences 1–20. The paired bootstrap (§5.3) pairs Q5 and Q1 episodes by sequence id. | if Q1 delays do not reuse Q5's sequence ids, pairing is impossible and additivity is "not computable" (not passed) | state explicitly which sequence ids each delay uses (for example, sequences 1–20 for every d, randomised across 3 blocks) |

## 4. Hardware-specific blockers (unchanged; none can be resolved in software)

| # | blocker |
|---|---|
| B1 | robot access (Fetch or Franka) through a host lab |
| B2 | host-lab safety approval; authorisation and supervision for commissioning (§1.2) |
| B3 | platform choice, which sets the driver SDK and command interface |
| B4 | robot driver, Cartesian command mapping, 40 ms watchdog and middleware bridge, including **the receipt timestamp of every proprioception sample (H1-A1)**. They need the platform SDK, and testing them needs the robot or vendor tools. |
| B5 | every **[CAL]** value (frames, extrinsics, marker offset, latencies, age/skew limits, workspace, fingers) |
| B6 | the target-computer timing gate G0-T1, with the shadow enabled, on the lab computer |
| B7 | a common monotonic clock across robot, sensors and controller (G0-clock) |

**Note:**
- **Local estimate only:** the shadow adds one estimator update and one C step per cycle. That
  is about 0.05–0.07 ms p99 on the local machine, extrapolated from the H0 component benchmark
  rather than measured in the pipeline with the shadow on.
- **Not evidence for G0-T1:** this says nothing about the target computer.

## 5. Readiness checklist for an H1 execution request

| item | state |
|---|---|
| Pre-registration revised with the approved design decisions | done (PR #25) |
| H0 software path, bitwise-equivalent to the confirmed controllers | done (PR #24); re-verified here |
| Shadow mode implemented, with isolation tests | **done** |
| Locked Q5/Q1 analysis implemented and tested on SYNTHETIC fixtures | **done** |
| Protocol questions P1–P4 (and P5) decided by numbered amendment | **done** (H1-A1–A4) |
| B1 robot access | open |
| B2 safety approval and commissioning authorisation | open |
| B3 platform chosen | open |
| B4 driver, watchdog and middleware, implemented and reviewed | open (needs B3) |
| B5 calibration values measured | open (needs B1, B2) |
| Commissioning completed and signed off by the host lab | open (needs B1, B2, B4) |
| B6 target timing gate passed (with the shadow) | open (needs B1, B4) |
| B7 clock synchronisation documented | open (needs B1) |
| Analysis code frozen at a commit before the session | ready (amendments implemented); freeze at the execution commit |
| **Your explicit approval of H1 execution** | **not given** |

**H1 is not ready for execution approval.** The software preparation and the protocol
amendments are complete. B1–B7 need hardware access.
