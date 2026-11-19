# Stage H0: implementation (software and offline validation only)

**Authority:** Stage H0 of `docs/HARDWARE_VALIDATION_PROPOSAL.md`, approved 2026-11-18. H0 covers
software development, offline validation, documentation and tests only.

**Not done:**
- No hardware was bought, borrowed, booked or operated, and no robot motion was executed.
- Nothing was deployed, and there is no robot, camera or network connection anywhere in the
  code.
- Stages H1–H5 have not started.

**Unchanged:**
- Candidate C, the Phase E estimator, the F1 model, the frozen SAC policies, their constants and
  all earlier results. Hashes were re-verified after the work.
- No simulation episode was run, and no new seed was used.

## 1. Layout

| path | role |
|---|---|
| `src/farc/hardware/config.py` | `HardwareConfig` and `Calibration`. Every calibration value is an explicit placeholder; `require_calibrated()` refuses to run with any left unset. |
| `configs/hardware/h0_template.json` | the per-platform template: every calibration value is `null` |
| `src/farc/hardware/observation.py` | `RawState` → 25-dim FetchPush observation, with validation |
| `src/farc/hardware/fifo.py` | deterministic command FIFO for injected delay |
| `src/farc/hardware/controller.py` | F, K, CA and C built from the original objects |
| `src/farc/hardware/pipeline.py` | one cycle: adapter → controller → FIFO → log record, plus the safety-stop rule |
| `src/farc/hardware/logging.py` | JSON Lines step log: schema, write-once files, SHA-256 hash chain |
| `src/farc/hardware/replay.py` | replay of archived F1 trajectories, goal reconstruction, a session stub for the original loops |
| `scripts/hardware_h0_replay.py` | offline equivalence (Parts A and B) |
| `scripts/hardware_h0_benchmark.py` | offline latency on the local machine |
| `tests/test_hardware_h0.py` | 27 tests |

## 2. Observation mapping

The frozen policy expects FetchPush's 25-dim observation (`farc.envs.observation_layout`; source
`fetch_env._get_obs`).

### 2.1 Frames

- **Robot base frame (B):** where proprioception and, after camera-to-base calibration, the object
  sensor report.
- **FetchPush world frame (W):** the frame the policy was trained in. The table top is at
  z = 0.4 m, and the gripper starts near (1.34, 0.75, 0.53).
- **The map:** `p_W = R p_B + t`, where `R` and `t` are **calibration placeholders**
  (`world_from_base_rotation`, `world_from_base_translation`).
- **Fetch and Franka are not assumed to share a frame:** each platform needs its own measured
  `R` and `t`.
- **Rotation checks:** the adapter rejects a matrix that is not orthonormal or has
  determinant < 0.
- **Not addressed by the transform:** the policy's workspace is FetchPush's. Matching the table
  height, reachable area and object scale is a setup and calibration task for H1; it cannot
  be solved in software.

### 2.2 Entries

| entries | quantity | computation (same operation order as the simulator) |
|---|---|---|
| 0:3 | gripper position | `R p_g + t` |
| 3:6 | object position | `R (p_marker + R_o · offset) + t`. The marker-to-centre offset is a placeholder. |
| 6:9 | object − gripper | (3:6) − (0:3) |
| 9:11 | finger positions | measured, or the calibrated blocked value (placeholder); Push keeps the gripper closed |
| 11:14 | object Euler angles | `rotations.mat2euler(R · R_o)`, the simulator's own function, from the measured quaternion (w, x, y, z) |
| 14:17 | relative linear velocity per step | `(R v_o)·dt − (R v_g)·dt`, in the simulator's order of operations |
| 17:20 | object angular velocity per step | `(R ω_o)·dt` |
| 20:23 | gripper linear velocity per step | `(R v_g)·dt` |
| 23:25 | finger velocities per step | `v_f·dt`, or zero if not measured |

- **dt** = 0.04 s, fixed because the policy was trained at 25 Hz. The config rejects any other
  period.
- **Velocities** enter as m/s and rad/s from the robot or sensor. H0 does not choose between a
  measured velocity and differencing positions; that is an H1 sensor decision, and either feeds
  the same entry.
- **Goal:** a world-frame point. `goal_from_base` maps a base-frame goal through the same
  transform.
- **Normalisation:** the adapter outputs raw values. The controller applies the policy's frozen
  normaliser, unchanged.

### 2.3 Timestamps and missing data

Each `RawState` carries the decision time and the gripper and object sample times, all on a
monotonic clock. The adapter raises `ObservationError` for:
- wrong shape, a non-finite value, or a non-unit quaternion (|‖q‖ − 1| > 10⁻³);
- a sample older than `max_observation_age_s` (placeholder);
- a sample newer than the decision time;
- gripper and object times further apart than `max_sensor_skew_s` (placeholder);
- a sensor timestamp that did not increase, meaning a repeated, stale sample.

**No interpolation or extrapolation** is done. **The pipeline turns any error into a safety
stop:**
- the command is zero;
- `safety_event` is set to `observation_<kind>`;
- the trial is aborted, and every later cycle refuses to act.

This implements the proposal's rule that a safety stop ends the trial as a failure.

## 3. Command FIFO

`CommandFIFO.push(command, issued_ns)` returns the command that *would* execute at this step:
the command issued at step `t − d(t)`, or zeros before the episode, clipped to [−1, 1]. This is
exactly `DelayHistoryWrapper.step`, which is itself verified bitwise against
`ScheduledDelaySession` in `replay_drop` mode:
- **on a delay increase,** an older command is replayed;
- **on a decrease,** queued commands are dropped.

| property | detail |
|---|---|
| delay source | a `DelaySchedule`, or `set_delay(d)` in manual mode |
| delay range | integers 0–4 only; anything else raises |
| reset | clears the buffer and the step counter; no command survives a reset |
| issue timestamps | must increase strictly |
| output | the delay used, the source step, and the source command's original issue time, for logging |
| not done | not connected to a robot. The real command interface (action × 0.05 m mapped to a Cartesian target in the base frame via `Rᵀ`) is listed as pending in `docs/HARDWARE_H1_READINESS.md`. |

## 4. Controller integration

`HardwareController(name)` composes the **original** objects; nothing scientific is
re-implemented:

| id | decision object | predictor | reference loop reproduced |
|---|---|---|---|
| F | `stage3.Latched` (computed, unused) | none | `mechanism.run_episode(mode="base")` (G4 `base_only`) |
| K | `stage3.Latched` | `contact.episodes.phase_e_predictor` | `stage3.run_g2_episode("phase_e_predictor")` |
| CA | `stage3.Latched` | `contact.episodes.learned_predictor(F1, move_object=True)` | `mechanism.run_episode` (G4 `contact_learned`) |
| C | `candidates.Sequential` via `mechanism.sequential` | F1 | `mechanism.run_episode` (G4 `C`) |

**The step order is that of the reference loops:**
1. The base policy on the observation.
2. `estimator.observe(gripper)`.
3. `ctrl.step(estimator)`.
4. If active, the predictor on `queued_commands(issued, d)` with the controller's gain, then
   the frozen policy at the predicted observation.
5. Clip to [−1, 1].
6. `estimator.issued(command)`.

**Constants** are the G4 pins: estimator window 10, maximum lag 4, R² ≥ 0.8, margin ≥ 2,
excitation ≥ 2.25 × 10⁻⁴; C with h = 12, σ₀ = 4 mm, Gaussian. A test asserts they equal
`configs/delaychange/g4_heldout.json`. The F1 model is loaded with its SHA-256 check.

**Each step returns:**
- per-component timings from `perf_counter_ns`;
- the estimator outputs, delay and gain used, activity and alarm/suspension;
- for C: supervisor mode, per-lag CUSUM statistics, trusted lag, reset start and σ²;
- the predicted gripper and object positions, and the predicted object motion. This is the
  offline proxy for contact state; measured contact needs a force/torque sensor and is `None`
  offline.

## 5. Logging

`StepLogger` writes one JSON line per step after a header carrying the schema version and run
metadata. Each line holds:

| group | fields |
|---|---|
| sequence and timing | `seq`; `mono_ns`; observation timestamps |
| commands | issued and executed commands; injected delay; source step |
| estimation | estimated delay and reliability; gain |
| supervisor | supervisor mode; CUSUM state |
| predictions | predictor outputs |
| contact | `contact_measured` |
| actions | base action |
| latency | per-component latency |
| errors | `error`; `safety_event` |

**Integrity:**
- Each record carries `prev_sha256`, the hash of the previous line. `verify_log` re-checks the
  hash chain, sequence numbers and schema types.
- Files open in exclusive-create mode (write-once).

**Deterministic replay:** a repeated replay produces identical records apart from the latency and
monotonic-clock fields. A test checks this.

## 6. Offline replay method

**Source:** the archived Phase F1 evaluation step records:
- development seeds 3,653,000–099, 3 frozen policies, 9 conditions, 3 recorded configurations;
- 8,100 episodes and 405,000 steps.

It contains the controller's view (noisy where the condition adds noise), the issued commands,
the latched delay and gain, and the predictions.

**Goal reconstruction (disclosed deviation from "recorded data only"):**
- The records lack the desired goal. It is recovered with `env.reset(seed=recorded_seed)`.
- **No `env.step` is executed, no episode is generated, and no new seed is used.**
- Each reconstruction is accepted only if the reset observation equals the recorded t = 0 true
  observation **bitwise**.
- CSV values are parsed with `float_precision="round_trip"`. The pandas default parser altered
  the last bit of some values, which a first smoke run caught.

**Part A, closed-loop consistency:** each recorded episode is replayed by the hardware
controller of the configuration that produced it (base_only → F, phase_e_predictor → K,
contact_learned → CA), through two input paths:
- **direct:** the recorded view goes straight to the controller;
- **adapter:** the view is converted to a raw state and back through the full pipeline (adapter,
  controller, FIFO).

**Part B, against the original code:** every recorded base_only trajectory is served by
`ReplaySession` to the unmodified simulation loops and, in parallel, to the hardware controller,
for all four configurations.
- This is **open-loop:** the next view is the recorded one.
- Both sides see identical inputs, so they must agree, but the trajectories are not the
  configurations' own closed-loop trajectories.
- The original controller objects are wrapped by a delegating recorder so that their estimator
  outputs, CUSUM statistics and σ² can be compared.
- No original module was modified.
