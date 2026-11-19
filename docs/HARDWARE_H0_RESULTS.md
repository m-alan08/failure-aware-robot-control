# Stage H0: results (offline only)

**Scope:**
- Software and offline validation on recorded **development** simulation data: Phase F1, seeds
  3,653,000–099.
- No hardware, no robot motion, no simulation episodes, no new seeds; held-out
  4,000,600–899 untouched.

**Method:** `docs/HARDWARE_H0_IMPLEMENTATION.md`.

**Reproduce:**
- `uv run python scripts/hardware_h0_replay.py --out results/hardware_h0`
- `uv run python scripts/hardware_h0_benchmark.py --out results/hardware_h0`

## 1. Verdict

**Offline controller equivalence passed with exact (bitwise) agreement everywhere it was
tested. No tolerance was needed for any controller output.**

## 2. Goal reconstruction

**8,100 / 8,100** episodes: the reset observation matched the recorded t = 0 observation
bitwise.

## 3. Part A: closed-loop consistency with the recorded episodes

| configuration | input path | episodes | steps | steps with issued command bitwise equal | latched delay mismatches | gain max abs diff | activity mismatches | prediction max abs diff |
|---|---|---|---|---|---|---|---|---|
| F | direct | 2,700 | 135,000 | **135,000** | 0 | 0 | 0 | – |
| F | adapter + full pipeline | 2,700 | 135,000 | **135,000** | 0 | 0 | 0 | – |
| K | direct | 2,700 | 135,000 | **135,000** | 0 | 0 | 0 | 0 |
| K | adapter + full pipeline | 2,700 | 135,000 | **135,000** | 0 | 0 | 0 | 0 |
| CA | direct | 2,700 | 135,000 | **135,000** | 0 | 0 | 0 | 0 |
| CA | adapter + full pipeline | 2,700 | 135,000 | **135,000** | 0 | 0 | 0 | 0 |

**Coverage:** all 9 conditions, including:
- persistent delays 1–4;
- a delay onset at step 15;
- 0.5 cm and 1 cm observation noise;
- action-gain scaling (`C7_delay3_scale0.5`), which exercises the latched gain.

No episode diverged.

## 4. Part B: against the original simulation loops (open-loop, all recorded base_only trajectories)

| configuration | episodes | commands bitwise equal | decision mismatches | predictions bitwise equal | compensation-active steps | C alarms |
|---|---|---|---|---|---|---|
| F | 2,700 | 2,700 | 0 | 2,700 | 0 | – |
| K | 2,700 | 2,700 | 0 | 2,700 | 89,507 | – |
| CA | 2,700 | 2,700 | 0 | 2,700 | 89,475 | – |
| C | 2,700 | 2,700 | 0 | 2,700 | 42,015 | **7,614** |

**What "decision mismatches" counts, at every step:**
- delay used, activity, alarm and suspension;
- estimated delay and its reliability flag;
- for C, also the CUSUM statistics for every lag, σ² and the window reset start.

**The supervisor's switching path is exercised heavily** (7,614 alarms), and matches exactly.

**Caveat:** Part B is open-loop for K, CA and C, because the trajectories are frozen-SAC's. It
shows that the hardware path computes exactly what the original code computes on the same
inputs. It is not a new closed-loop evaluation, and nothing in it is a performance claim.

## 5. Observation adapter round trip

Recorded view → raw state → adapter, compared with the recorded view. Maximum absolute
difference per component over 135,000 base_only views:

| component | max abs diff |
|---|---|
| positions, finger state, velocities | 0 – 2.8 × 10⁻¹⁷ |
| relative position | 2.2 × 10⁻¹⁶ (one rounding step) |
| object Euler angles | **1.6 × 10⁻¹² rad** |

**Interpretation:**
- **The Euler difference is a replay artefact.** The recorded Euler angles must be converted to
  a quaternion and back (euler → quat → matrix → euler). On hardware the quaternion is the
  measurement, so the round trip does not occur.
- **It never changed an action.** All 405,000 adapter-path commands were bitwise equal to the
  recorded commands (Part A). Differences this small vanish when the frozen normaliser casts to
  float32.
- **The unit test therefore uses** atol = 10⁻¹³ on random vectors away from gimbal lock, and
  documents this source.

## 6. FIFO

Against `DelayHistoryWrapper` driven by a stub environment with no physics, the FIFO's executed
commands are identical at every step for 8 schedules:
- no delay;
- constant delays 1–4;
- pulses;
- on-off delays;
- step changes in both directions.

**Commands included out-of-bound values,** to test clipping.

**Comparison detail:** the wrapper casts to float32 at `env.step` (the simulator boundary), while
the FIFO keeps float64 for the robot interface. The test compares after the same cast. A first
version of the test compared without the cast and failed at the 10⁻⁸ level. The FIFO was not
changed; the test now states the boundary explicitly.

## 7. Latency (LOCAL machine only)

**Machine:** an Apple M5 laptop, macOS 26.5 (arm64), Python 3.12, torch 2.14 on CPU with
1 thread. This is **not** the target lab computer.

**Inputs:**
- 400 recorded episodes, 20,000 steps;
- policy seed 1 (the proposal's pre-selected policy);
- conditions: no delay, delay 3, delay 4, and delay-3 onset.

All values are microseconds.

| component | median | p95 | p99 | max |
|---|---|---|---|---|
| observation adapter | 24.3 | 29.1 | 31.1 | 70.2 |
| frozen SAC inference (normalise + actor) | 27.3 | 31.8 | 33.9 | 134.0 |
| Phase E estimator (observe + estimate) | 22.2 | 26.9 | 29.7 | 156.4 |
| Candidate C supervisor step | 30.1 | 35.0 | 38.2 | 277.6 |
| kinematic predictor, d = 4 (worst case) | 7.9 | 8.5 | 13.0 | 33.5 |
| F1 contact model rollout, d = 4 (worst case) | 109.8 | 115.5 | 124.8 | 502.8 |
| **full pipeline F** | 86.0 | 93.0 | **99.1** | 140.5 |
| **full pipeline K** | 124.6 | 134.2 | **140.0** | 518.5 |
| **full pipeline CA** | 207.7 | 245.0 | **252.6** | 636.8 |
| **full pipeline C** | 100.9 | 256.8 | **265.5** | 302.7 |

**Against the budget:**
- **Locally, every configuration's p99 is 0.10–0.27 ms,** against the 10 ms budget: about 37×
  headroom.
- **The worst single cycle was 0.64 ms,** against the 40 ms period.

**Limits of these numbers:**
- They do **not** show that the target lab computer meets the budget.
- They exclude sensor acquisition, middleware (ROS 2 / libfranka) transport, the robot's own
  control loop, OS scheduling jitter under a real-time load, and Python garbage-collection
  pauses outside this run.
- They must be re-measured on the target computer (H1 readiness, item T1).

## 8. Tests

| suite | result |
|---|---|
| new H0 tests | **27 passed** |
| full suite | **436 passed** (409 before H0, plus 27) |

**What the new tests cover:**
- observation dimensions and ordering;
- frame transformation;
- the marker offset;
- the adapter round trip;
- 5 invalid-input kinds, plus repeated and skewed samples;
- placeholder refusal and the fixed control period;
- FIFO equivalence on 8 schedules;
- FIFO ordering, initial zeros, delay increase and decrease, reset, invalid delays and
  timestamps;
- log hash chain, tampering, gaps, write-once and schema;
- frozen constants against the G4 configuration;
- bitwise controller equivalence, both closed-loop and open-loop for F, K, CA and C (on a
  subset);
- deterministic replay with logging;
- timing-field consistency;
- the safety stop on a missing observation.

The tests that need checkpoints or F1 records skip when those are absent.

## 9. Discrepancies and issues

| # | item | status |
|---|---|---|
| D1 | The pandas CSV parser altered the last bit of recorded floats | **found and fixed in the replay loader** (`float_precision="round_trip"`); the archive is unchanged |
| D2 | The FIFO test needed the simulator's float32 boundary cast | test made explicit; FIFO unchanged |
| D3 | 1.6 × 10⁻¹² rad Euler round trip | replay-only artefact; no effect on any action |
| D4 | The goal is reconstructed by `env.reset` (no stepping) | disclosed; bitwise-verified on all episodes |
| D5 | Part B is open-loop for K, CA and C | stated; equivalence only, not performance |

**No controller discrepancy was found.** Nothing in Candidate C, the estimator, F1 or the
policies was changed.

## 10. Artefact integrity

**Re-verified after the work:**
- the frozen SAC seeds 0–2, against their manifest;
- the F1 model's SHA-256 (`8a960329…`).

**Files changed:** no file outside `src/farc/hardware/`, `configs/hardware/`,
`scripts/hardware_h0_*`, `tests/test_hardware_h0.py`, `results/hardware_h0/` and these documents
changed.
