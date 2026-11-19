# Stage H1 pre-registration: free-space system identification (draft)

**Status: a draft for approval. H1 is NOT approved.**
- No hardware has been arranged, and nothing here was run.
- The design, endpoints, decision rules and analysis are fixed *now*. Values that need a
  measurement are marked **[CAL]** and are filled at the H1 setup step, before any trial, under
  rules stated here.

**Parent documents:**
- `docs/HARDWARE_VALIDATION_PROPOSAL.md` (§5, H1);
- `docs/HARDWARE_H0_RESULTS.md`;
- `docs/HARDWARE_H1_READINESS.md`, whose blockers B1–B5 must be closed before H1 starts.

**Unchanged:**
- The Phase E estimator, its constants (window 10, maximum lag 4, R² ≥ 0.8, margin ≥ 2,
  excitation ≥ 2.25 × 10⁻⁴, nominal gain 0.62), Candidate C, F1 and the frozen SAC policies.
- **H1 runs no policy and moves no object.**
- **Simulation held-out seeds 4,000,600–899 are untouched.**

---

## 1. Questions and endpoints

| | question | primary endpoint |
|---|---|---|
| **Q5** | What is the natural command-to-motion latency of the real command path, and how variable is it? | natural delay d₀ in control steps (40 ms), plus the latency distribution in ms |
| **Q1** | Does the unchanged Phase E estimator identify injected command delays on the real arm? | per injected d, the share of episodes whose **final latched estimate** equals d + d₀ |
| Q1-gain | Does the real per-step tracking gain match the simulator's K0 = 0.62? | the estimator's gain estimate ĝ = K̂ / K0, reported (not tuned) |
| S | Sensor noise floor relevant to the estimator | per-axis std and lag-1 autocorrelation of gripper and object position |

**Q5 is measured first,** because Q1's correct answer is d + d₀: the estimator sees the *total*
command delay, natural plus injected. d₀ is fixed from the Q5 data before any Q1 trial is run.

## 2. Pre-trial gates (all must pass; recorded in an amendments log before Q5 starts)

| gate | requirement |
|---|---|
| **G0-access** | robot access is confirmed, and the platform (Fetch or Franka) is recorded |
| **G0-safety** | the host lab's written safety approval; a trained operator; an e-stop. The robot-side workspace box and velocity, acceleration and force limits are active **and tested at d = 4 at low speed** with no object. |
| **G0-cal** | every **[CAL]** value in `configs/hardware/<platform>.json` is measured and `HardwareConfig.require_calibrated()` passes |
| **G0-sw** | the robot driver, the command mapping and a 40 ms missed-cycle watchdog are implemented, unit-tested, and reviewed. The H0 equivalence suite still passes bitwise on the deployment commit. |
| **G0-T1** | target-computer timing: the full H0 pipeline under the deployment process (middleware running, robot idle) reaches **p99 ≤ 10 ms and max ≤ 40 ms over ≥ 10,000 cycles**. If it fails, H1 does not start. |

## 3. Design

### 3.1 Excitation: no policy

H1 drives the arm with **pre-generated command sequences**, not the policy. The estimator needs
command variation (excitation), and running the policy without an object would be off-task.

**Source:**
- Issued-command sequences (`cmd0..2`) from the archived Phase F1 **frozen-SAC, no-delay**
  episodes (`C0_undisturbed`, development seeds 3,653,000–099, policy seed 1). That gives
  realistic magnitudes and spectra.
- These seeds are already consumed. They are used only as excitation signals, not as a
  test-set.
- The finger component is 0. The z component is set to 0, so motion stays planar above the
  table; this matches the estimator, which uses x and y only.

**Selection rule (fixed now, applied offline after G0-cal, before any trial):**
1. Integrate each sequence's commanded displacement, K0 × 0.05 m × command per step, from the
   calibrated start pose **[CAL]**.
2. Keep a sequence only if the whole path stays inside the workspace box shrunk by 5 cm on each
   side **[CAL]**, under every injected d. Delay does not change the commanded path, only its
   timing.
3. Take the first **60** admissible sequences in episode-seed order. Never choose them by
   hardware outcome.

**The admissible set:** 20 sequences per Q1 block × 3 blocks (§3.3). If fewer than 60 are
admissible, sequences are reused in the same order, and this is disclosed.

### 3.2 Episode

- **Start:** the arm moves to the calibrated start pose at low speed (outside the experiment)
  and holds for 2 s.
- **Steps:** 50 control steps at 25 Hz, the same horizon as FetchPush.
- **Pipeline:** each step runs the H0 software path: adapter → estimator and latch only (no
  predictor, no policy) → FIFO with injected delay d → the robot's Cartesian command.
- **Logging:** every step is written to the H0 hash-chained, write-once log.
- **The estimator** is the unchanged `ActuatorEstimator` with `stage3.Latched`.

### 3.3 Conditions and order

| block | content | episodes |
|---|---|---|
| **S** (sensor) | robot holding still, puck static on the table, 30 s recording | 3 recordings |
| **Q5** (latency) | d = 0; excitation sequences 1–20 | 20 |
| **Q1** | d ∈ {0, 1, 2, 3, 4} × 20 sequences = 100 episodes, run in 3 blocks: within each block, the (sequence, d) order is randomised by a generator with seed **20264100** (fixed now) | 100 |
| **Q5-repeat** | d = 0; sequences 1–5, at the end of the session (drift check) | 5 |

**Total:** about 125 episodes and about 1 lab day, matching the proposal.

## 4. Measurements and analysis (fixed now)

### 4.1 Q5: natural latency

**Primary estimator of d₀:**
- For each Q5 episode, run the unchanged `ActuatorEstimator.estimate()` at the end of the
  episode. Its least-squares lag at step 49 is the episode's d₀ estimate, kept only if it was
  delay-reliable.
- **d₀ is the mode** over reliable Q5 episodes.
- It is **valid only if** at least 15 of 20 Q5 episodes are reliable and at least 80% of those
  agree on the mode. Otherwise Q5 is "unresolved" and **Q1 is not run** (§6).

**Secondary, descriptive:**
- Latency in ms from cross-correlating the commanded and measured gripper velocity. Reported as
  median, p95 and max per episode, with the quantisation relative to 40 ms.
- Q5-repeat compared with Q5, to check drift.

### 4.2 Q1: delay identification

For each episode:

| quantity | definition |
|---|---|
| **final latched estimate** | the latch's delay at step 49 (the primary quantity) |
| **correct** | final latched estimate = d + d₀ |
| reliable share | share of steps with `delay_reliable` |
| correct-when-reliable | share of reliable steps where the estimate is d + d₀ |
| time to first reliable | first step at which the estimate is reliable |
| wrong latch | final latched estimate ≠ d + d₀ and > 0; a false compensation risk |

**Primary success criterion, per injected d:**
- **≥ 18 / 20 episodes correct.**
- For d + d₀ > 4, the true delay exceeds the estimator's maximum lag. Those d are reported
  descriptively, are expected to fail, and are excluded from the criterion.
- **Q1 passes** if every eligible d meets the criterion.
- **Statistical meaning:** with 18/20 the one-sided exact 95% lower bound on accuracy is 0.72.
  The simulation reference (Phase J-3, noise-free) is 0.90–0.99.

**Safety-relevant secondary criterion:** wrong latches ≤ 1 / 20 per d. A wrong latch would make
compensation act on the wrong delay.

**Reported, not a criterion:** the gain estimate ĝ per d (median, interquartile range).
- The estimator's nominal gain stays 0.62; **nothing is retuned.**
- If the median ĝ lies outside [0.7, 1.3], that is reported as a plant-model mismatch to be
  resolved by a *new* proposal before H2. It is not changed in H1.

### 4.3 S: sensor noise

For gripper and object position, per axis:
- the standard deviation and lag-1 autocorrelation over each 30 s still recording, after
  removing a linear trend;
- the sampling rate actually achieved, and timestamp jitter.

**Reference points (simulation, J-3):**
- the estimator abstains increasingly above about 0.5 cm of independent gripper-position noise,
  and mostly abstains at 1 cm;
- the H2 design assumes gripper noise ≤ 1 mm.

**A gripper-position std above 2.5 mm** is flagged as a risk for Q1, but does not stop the
session.

### 4.4 Logs and integrity

- Every episode log must pass `verify_log`. An episode with a broken chain is invalid, and is
  reported, not re-run selectively.
- Analysis code is written and committed **before** the session, run once on the locked logs,
  and not changed afterwards.
- Raw logs are write-once, and a manifest of SHA-256 hashes is committed after the session.

## 5. Stopping and deviation rules

| event | rule |
|---|---|
| a safety stop (observation error, watchdog, robot-side limit, e-stop) | the episode is recorded as aborted, with its reason. The session pauses for an operator check. After 3 safety stops in one session, H1 stops, the cause is investigated, and a resumption needs your approval. |
| an aborted episode | **no selective re-run.** One re-run at the end of its block is allowed only for a stop caused by a documented external interruption (for example a person entering the cell), decided before seeing the episode's estimator output |
| hardware fault | stop; report; any resumption is a new session with the same order |
| any change to a frozen constant, the selection rule, d₀ after Q1 has started, or the criteria | **not allowed.** Any change is a new, separately approved protocol. |
| amendments | numbered, committed before the trials they affect, never after seeing the affected data |

## 6. Outcomes and what they decide (no automatic progression)

| outcome | reading | next step (needs your approval) |
|---|---|---|
| Q5 resolved, Q1 passes, wrong latches within limit | identification transfers to the real arm under injected delay | propose H2 (nominal gate) |
| Q5 unresolved | the natural delay is too variable or the estimator is unreliable at d = 0 on this plant | stop; report; investigate timing or the plant before anything else |
| Q1 fails for some d | identification does not transfer at those delays | report as a negative result. Any estimator change is a new protocol. |
| ĝ outside [0.7, 1.3] | the plant gain differs from the simulator | report; resolve in a new proposal before H2 |
| gripper noise > 2.5 mm | sensing is too noisy for the estimator | report; consider other sensing in a new proposal |

**Paper claims allowed from H1:** identification accuracy on *one* robot, under *injected* delay,
with the measured natural latency. No task-performance claim.

## 7. What H1 does not do

- No policy, no object contact, no pushing task.
- No predictor or Candidate C evaluation. The C supervisor may run in *shadow* mode for logging
  only; this is descriptive and has no criterion.
- No retuning, no learned-model fitting, no use of simulation held-out seeds.

## 8. Decisions requiring your approval

1. The design above: excitation from recorded commands, 20 episodes per d, 18/20 per-d
   criterion, wrong latches ≤ 1/20.
2. Measuring d₀ first and scoring Q1 against d + d₀.
3. The pre-trial gates G0, including the target-computer timing requirement.
4. The order seed 20264100 and the selection rule.
5. Whether C may run in shadow mode for descriptive logging (default: yes, with no criterion).
6. **Approval to proceed** only once blockers B1–B5 in `docs/HARDWARE_H1_READINESS.md` are
   closed. Until then this document stays a draft.
