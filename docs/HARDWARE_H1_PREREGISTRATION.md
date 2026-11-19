# Stage H1 pre-registration: free-space system identification

**Status:**
- **Design approved** (2026-11-18), with the decisions recorded in §0. **H1 execution is NOT
  approved.**
- No hardware has been arranged, no code deployed, no robot moved, and nothing here was run.
- Values that need a measurement are marked **[CAL]**. They are filled during setup, before any
  trial, under the rules stated here.

**Parent documents:**
- `docs/HARDWARE_VALIDATION_PROPOSAL.md` (§5, H1);
- `docs/HARDWARE_H0_RESULTS.md`;
- `docs/HARDWARE_H1_READINESS.md`, whose blockers B1–B5 must be closed before H1 starts.

**Unchanged and frozen:**
- the Phase E estimator and its constants: window 10, maximum lag 4, R² ≥ 0.8, margin ≥ 2,
  excitation ≥ 2.25 × 10⁻⁴, nominal gain 0.62;
- Candidate C (h = 12, σ₀ = 4 mm, Gaussian), F1 and the frozen SAC policies;
- all earlier results;
- **simulation held-out seeds 4,000,600–899**, which H1 does not touch.

---

## 0. Approved design decisions (recorded verbatim in intent)

| # | decision | where |
|---|---|---|
| 1 | 18/20 per delay is a **feasibility threshold**, not an accuracy estimate | §5.2 |
| 2 | wrong latches ≤ 1/20, with false positives defined for a true total delay of zero | §5.2 |
| 3 | Q5 first. Injected discrete delay is kept distinct from natural latency, and the additivity of d + d₀ is tested by independent timestamps. Expected answers are frozen before Q1. | §4, §5.1, §5.3 |
| 4 | target-computer timing gate: p99 ≤ 10 ms, max ≤ 40 ms, ≥ 10,000 cycles | §2 |
| 5 | order seed 20264100 and the pre-registered sequence selection | §3 |
| 6 | Candidate C runs in **shadow mode only** | §3.4 |
| 7 | robot-side limits are authoritative; offline screening is not a safety argument | §1.1 |
| 8 | safety commissioning is separate from trials and needs host-lab authorisation | §1.2 |
| 9 | fixed protocol, stopping rules, no selective re-runs, write-once logs | §6 |
| 10 | models, constants, results and held-out seeds unchanged | header |

## 1. Safety framework

### 1.1 Authority

- **The robot-side safety layer is authoritative:** the workspace box and the force, velocity
  and acceleration limits enforced in the robot controller, plus the hardware e-stop.
- **Nothing in this software can override, relax or replace those limits.** The software stops
  (H0 observation checks, watchdog) are *additional*, not substitutes.
- **The offline sequence screening (§3.1) is not a safety argument.** It integrates a
  simulation-derived nominal gain, and it exists only to avoid planned paths that hit
  the limits and abort the trial. Physical safety rests entirely on the robot-side limits,
  host-lab procedures and supervision.

### 1.2 Commissioning (separate from experimental trials)

- **Safety commissioning** is any robot motion used to verify limits, the e-stop, the watchdog,
  the command mapping or calibration. That includes the low-speed test at injected delay 4 with
  no object.
- **Commissioning is not an H1 trial,** and its data are not used in any H1 analysis.
- **Before execution, commissioning requires** documented host-lab authorisation and
  supervision by the lab's designated responsible person. The authorisation is filed (without
  personal data) in the amendments log.
- Commissioning logs are kept separately, labelled `commissioning`, and are never pooled with
  trial logs.
- **No H1 trial starts** until commissioning is complete and signed off by the host lab (gate
  G0-safety).

## 2. Pre-trial gates (all must pass; recorded in the amendments log before any trial)

| gate | requirement |
|---|---|
| **G0-access** | robot access is confirmed, and the platform (Fetch or Franka) is recorded |
| **G0-safety** | the host lab's written safety approval; a trained operator; an e-stop within reach. Commissioning (§1.2) is complete and signed off, including the d = 4 low-speed test. |
| **G0-cal** | every **[CAL]** value in `configs/hardware/<platform>.json` is measured; `HardwareConfig.require_calibrated()` passes |
| **G0-sw** | the robot driver, command mapping and a 40 ms missed-cycle watchdog are implemented, unit-tested and reviewed. The H0 equivalence suite passes bitwise on the deployment commit. The shadow-mode isolation test passes (§3.4). |
| **G0-T1** | target-computer timing: the full H0 pipeline, including the shadow C supervisor, in the deployment process (middleware running, robot idle) reaches **p99 ≤ 10 ms and max ≤ 40 ms over ≥ 10,000 consecutive cycles**. If it fails, H1 does not start. |
| **G0-clock** | robot, sensors and controller timestamps share one monotonic time base, or have a measured, documented offset. The method is recorded. |

## 3. Design

### 3.1 Excitation (no policy) and the selection procedure

**Excitation:**
- H1 drives the arm with **pre-generated command sequences,** not the policy.
- **Source:** issued-command sequences (`cmd0..2`) from archived Phase F1 frozen-SAC
  **no-delay** episodes (`C0_undisturbed`, development seeds 3,653,000–099, policy seed 1).
  These seeds are already consumed; they serve only as excitation signals.
- The finger component is 0, and z is set to 0 (planar motion). The estimator uses x and y only.

**Selection procedure (fixed now; applied offline after G0-cal, before any trial):**
1. Integrate each sequence's planned displacement, K0 × 0.05 m × command per step, from the
   calibrated start pose **[CAL]**.
2. Keep a sequence only if the planned path stays inside the robot-side workspace box shrunk by
   5 cm per side **[CAL]**. Delay shifts only the timing, not the path.
3. Take the first **60** admissible sequences in episode-seed order. They are never chosen by any
   hardware outcome.

If fewer than 60 are admissible, sequences are reused in the same order, and this is
disclosed. As §1.1 says, this screening does not establish physical safety.

### 3.2 Episode

- **Start:** move to the calibrated start pose at low speed (outside the experiment) and hold
  for 2 s.
- **Steps:** 50 control steps at 25 Hz.
- **Active path:** adapter → estimator + `stage3.Latched` (unchanged) → FIFO with injected delay
  d → robot Cartesian command. No predictor, no policy.
- **Logging:** every step goes to the H0 hash-chained, write-once log, with command-issue,
  FIFO-release, driver-send and sensor timestamps (§4.1).

### 3.3 Conditions and order

| block | content | episodes |
|---|---|---|
| **S** (sensor) | robot holding still, puck static, 30 s | 3 recordings |
| **Q5** (latency) | d = 0, sequences 1–20 | 20 |
| **Q1** | d ∈ {0, 1, 2, 3, 4} × 20 sequences, run in 3 blocks. Within each block the (sequence, d) order is randomised by a generator with **seed 20264100**. | 100 |
| **Q5-repeat** | d = 0, sequences 1–5, at the end of the session (drift check) | 5 |

### 3.4 Candidate C in shadow mode only

**What the shadow does:**
- The C supervisor (`mechanism.sequential`, frozen constants) runs on its **own**
  `ActuatorEstimator` instance.
- That instance is fed the same observations and the commands **actually issued by the active
  controller**.
- Its decisions (alarm, suspension, trusted lag, CUSUM statistics) are **logged only.**

**Isolation:** no shadow output reaches the command path, the active estimator, the latch, the
FIFO or the robot.

**Enforcement:**
- The shadow has no reference to the command path.
- G0-sw requires a test showing that the issued and executed commands are **bitwise identical**
  with the shadow enabled and disabled, on recorded H0 replay data.

**Analysis:** shadow results are descriptive only. There is no criterion and no decision depends
on them.

## 4. Definitions: injected delay vs natural latency

### 4.1 Quantities

| symbol | meaning | how it is known |
|---|---|---|
| **d** | **injected discrete delay** (0–4 control steps) applied by the FIFO | exact by construction; verified from the log (`executed_source_step = t − d` at every step) |
| **L** | **natural command-to-motion latency** (continuous, ms): time from the command leaving the controller (driver-send timestamp) to the measured onset of the corresponding gripper motion | independent timestamp measurement (§4.2) |
| **d₀** | natural latency expressed as the estimator's discrete lag at d = 0 | the estimator's mode in Q5 (§5.1) |
| **d\*** | the **expected total discrete delay** the estimator should report in a Q1 episode with injected d | fixed before Q1: d\* = d + d₀, valid only if the additivity test passes (§5.3) |

### 4.2 Independent timestamp-based latency measurement (fixed now)

Per episode, on x and y separately and then pooled:
1. Let c(t) be the issued command at driver-send timestamps, and v(t) the measured gripper
   velocity at proprioception timestamps, both resampled onto a common 1 ms grid by linear
   interpolation. Sample times are not altered.
2. **L_ep** = the lag in [0, 300] ms that maximises the normalised cross-correlation between
   c(t − lag) and v(t).
3. Sensor latency **[CAL]** (`proprio_latency_s`) is reported separately. It is not subtracted
   from L_ep: the estimator sees it too.

**L_ep is computed independently of the estimator.** It never uses estimator outputs.

## 5. Analysis (fixed now)

### 5.1 Q5: natural latency (before any Q1 trial)

**Estimator d₀:** the unchanged estimator's least-squares lag at step 49 of each Q5 episode,
kept only if it is delay-reliable. **d₀ is the mode** over reliable episodes.

**Timestamp d₀_ts:**
- d₀_ts = round(median L_ep over Q5 / 40 ms).
- **Half-step ambiguity:** if (median L_ep mod 40 ms) lies within ±8 ms of 20 ms, the latency is
  close to a step boundary and d₀_ts is flagged as **ambiguous**.

**Q5 is resolved only if all of these hold:**
- (a) at least 15 / 20 Q5 episodes are delay-reliable;
- (b) at least 80% of reliable episodes agree on the mode;
- (c) d₀_ts is not ambiguous;
- (d) **d₀ = d₀_ts.**

**Otherwise Q5 is "unresolved",** and Q1 is not run (§7).

**Freezing:** at the end of Q5, d₀ and the expected answers d\* = d + d₀ for every d are written
to a numbered amendment and committed **before the first Q1 trial**. They are never changed after
any Q1 data are seen.

**Drift check:** Q5-repeat is reported against Q5, both median L_ep and the estimator lag.

### 5.2 Q1: delay identification

**Per episode:**

| quantity | definition |
|---|---|
| **final latched estimate** | the latch's delay at step 49 (primary) |
| **correct** | final latched estimate = d\* |
| reliable share | share of steps with `delay_reliable` |
| correct-when-reliable | share of reliable steps whose estimate = d\* |
| time to first reliable | first step at which the estimate is reliable |

**Wrong latch, per episode:**
- **If d\* > 0:** the final latched estimate is > 0 and ≠ d\*. The controller would compensate
  for the wrong delay.
- **If d\* = 0** (no injected delay and d₀ = 0): **a false positive** is any final latched
  estimate > 0. The controller would compensate when no delay exists.
  - In addition, a **transient false latch** is reported: the latch was > 0 at any step, even if
    it returned to 0. This is descriptive and has no criterion.
- A final latched estimate of 0 (abstention) when d\* > 0 is **incorrect but not a wrong latch.**
  It is the estimator's conservative failure mode, and it is counted against the 18/20 criterion,
  not the wrong-latch limit.

**Eligibility:** d is eligible if d\* ≤ 4, the estimator's maximum lag. Ineligible d are reported
descriptively and are expected to fail.

**Criteria, per eligible d** (20 episodes each):
- **Feasibility: ≥ 18 / 20 correct.**
- **Wrong latches: ≤ 1 / 20.**
- **Q1 passes** if every eligible d meets both and passes the additivity test (§5.3).

**What 18/20 does and does not mean:**
- It is a **feasibility threshold**: a pre-set bar for whether identification works well enough
  on this robot to justify H2.
- **Observing 18/20 (90%) does not establish 90% population accuracy.** With 20 episodes the
  one-sided exact (Clopper-Pearson) 95% lower bound for 18/20 is **0.72**, and for 20/20 it is
  0.86.
- The result is reported as the observed count with that exact interval. It is not claimed as
  an accuracy estimate beyond what the interval supports.
- It applies to one robot, one session, these excitation sequences and injected delays only.

**Reported, never tuned:**
- the gain estimate ĝ per d (median, interquartile range);
- if the median ĝ lies outside [0.7, 1.3], that is a plant-model mismatch, to be resolved only by
  a new proposal.

### 5.3 Additivity test: is d\* = d + d₀ valid? (fixed now)

**Purpose:** d\* assumes that injected discrete delay and natural latency simply add. That is
checked by timestamps, independently of the estimator.

**For each Q1 delay d:**
- **(i) FIFO exactness:** in every episode, `executed_source_step = t − d` at every step. This
  holds by construction and is verified from the log.
- **(ii) Timestamp additivity:**
  - Δ_d = median L_ep(Q1, d) − median L_ep(Q5) − d × 40 ms.
  - **Additivity holds for d if |Δ_d| ≤ 10 ms** (a quarter of a step), with a paired-bootstrap
    95% CI on Δ_d (2,000 resamples over episodes, seed 20264101) inside ±20 ms.

**If additivity fails for some d:**
- d\* for that d is **not revised.**
- That d is reported as "**additivity violated: not interpretable**" and **counts as not
  passed.**
- The expected answers are never changed after inspecting Q1 outcomes.

### 5.4 S: sensor noise

For gripper and object position, per axis:
- the standard deviation and lag-1 autocorrelation over each 30 s still recording, after
  removing a linear trend;
- the sampling rate achieved, and timestamp jitter.

**A gripper-position std above 2.5 mm** is flagged as a risk to Q1, but does not stop the
session. Reference points: J-3 found increasing estimator abstention above about 0.5 cm of
independent noise.

### 5.5 Logs and integrity

- Every episode log must pass `verify_log`. A broken chain makes the episode invalid; it is
  reported, never selectively re-run.
- The analysis code (§4.2, §5) is written, tested on H0 replay data and committed **before** the
  session. It is run once on the locked logs and never changed afterwards.
- Raw logs are write-once. A SHA-256 manifest is committed after the session.

## 6. Stopping and deviation rules

| event | rule |
|---|---|
| a safety stop (observation error, watchdog, robot-side limit, e-stop) | the episode is recorded as aborted, with its reason, and the session pauses for an operator check. After 3 safety stops in one session, H1 stops; resumption needs your approval and host-lab clearance. |
| an aborted episode | **no selective re-run.** One re-run at the end of its block is allowed only for a documented external interruption, decided before seeing the episode's estimator output. |
| hardware fault | stop and report. Resumption is a new session with the same order. |
| changes | no change to frozen constants, the selection procedure, the order seed, the expected answers d\* after Q1 starts, the additivity rule, or the criteria. Any change needs a new, separately approved protocol. |
| amendments | numbered, committed before the trials they affect, never after seeing the affected data |

## 7. Outcomes and what they decide (no automatic progression)

| outcome | reading | next step (needs your approval) |
|---|---|---|
| Q5 resolved, additivity holds, Q1 passes | identification is feasible on this arm under injected delay | propose H2 (nominal gate) |
| Q5 unresolved | the natural latency is ambiguous or variable, or the estimator is unreliable at d = 0 | stop and report; investigate timing or the plant first |
| additivity violated for some d | the injected and natural delays do not combine simply on this system | report; those d are not interpretable. A redesign is a new protocol. |
| Q1 fails the feasibility threshold or the wrong-latch limit for some d | identification is not shown to be feasible at those delays | report as a negative result; any estimator change is a new protocol |
| ĝ outside [0.7, 1.3] | plant gain mismatch | report; resolve in a new proposal before H2 |
| gripper noise > 2.5 mm | sensing too noisy | report; consider other sensing in a new proposal |

**Claims allowed:** identification feasibility on *one* robot under *injected* delay, with the
measured natural latency. They come as counts with exact intervals. No task-performance claim and
no population-accuracy claim.

## 8. Not in H1

- No policy, no object contact, no pushing task.
- No active predictor or Candidate C. C runs as a logging-only shadow (§3.4).
- No retuning, no learned-model fitting, no use of simulation held-out seeds.

## 9. Still required before execution can be approved

- Blockers B1–B5 in `docs/HARDWARE_H1_READINESS.md` closed: access, safety approval, platform
  choice, driver and watchdog, calibration.
- Commissioning completed under host-lab authorisation (§1.2).
- The shadow-mode isolation test and the §4.2/§5 analysis code implemented and committed. This is
  software only, but it needs your instruction.
- **Your explicit approval of H1 execution.** This document does not grant it.
