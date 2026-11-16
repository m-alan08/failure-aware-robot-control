# Phase G1: supervisory switching for the contact-aware delay compensator

**Status: DRAFT for approval. Nothing here has been run.**
- No supervisor is implemented or tuned.
- No G1 seed has been used.
- The pre-registration (Part 2) becomes binding only after it is approved and its SHA-256 is
  pinned in a config, as in F1 and the held-out run.

**Question:** can a causal supervisor that decides *when* (and *which*) compensation to run
improve robustness, safety or computation over (a) frozen SAC and (b) the validated
contact-aware compensator as it is now?

**Out of scope:** residual RL, any learned failure predictor, and any change to the frozen SAC
policies, the Phase E estimator or the F1 dynamics model.

---

## Part 1: architecture audit

### 1.1 The controller as it runs today (`farc.contact.episodes.run_episode`)

At each decision step t (FetchPush: 50 steps, a 40 ms control period):

1. **Observe.** `ActuatorEstimator.observe(gripper)`.
2. **Estimate.** `estimate()` fits Δg[τ] ≈ K0·gain·0.05·c[τ−k] by least squares through the
   origin, over the last 10 transitions, for lags k = 0..4, and keeps the lag with the lowest
   residual.
3. **Latch.** `LatchedEstimate.parameters()` keeps the last *reliable* delay and gain. Before
   the first reliable estimate it returns `None`.
4. **Compensate.** If the latched delay d̂ > 0:
   - take the d̂ already-queued commands;
   - a predictor maps (observation, queued commands, min(gain, 1)) to a predicted observation:
     the Phase E static-cube shift, or the F1 learned rollout of d̂ MLP steps;
   - frozen SAC is evaluated on that predicted observation.

   Otherwise frozen SAC sees the raw observation.
5. **Act.** The command is clipped to [−1, 1], executed through the disturbance pipeline, and
   passed to `estimator.issued(command)`.

F1 and the held-out run use **no gain compensation**. The gain estimate only scales the rollout.

### 1.2 Is the compensator already selectively activated? Yes, in two layers

| layer | rule | what it decides |
|---|---|---|
| **Abstention gate** (estimator) | delay is *reliable* only if: ≥ 4 transitions, excitation ≥ 2.25e-4, R² ≥ 0.8, and second-best/best residual ratio ≥ 2 | whether this step's estimate may update the latch |
| **Latch** (`LatchedEstimate`) | switches on at the first reliable d̂ > 0; switches off **only** at a later reliable d̂ = 0; otherwise holds indefinitely | whether compensation runs, and with which d̂ |

**Measured behaviour of this gate.** This is a read-only analysis of the existing F1
development logs (`phase_f1_eval/steps.csv.gz`, `contact_learned`, 300 episodes per
condition); no new runs were made.

| condition | active steps | mode toggles per episode (mean / max) | episodes with ≥ 2 toggles | onset → active, median (p90) | active while true delay = 0 |
|---|---|---|---|---|---|
| C0 undisturbed | 0.6% | 0.02 / 2 | 0.3% | n/a | 93 steps (≈ 3 episodes) |
| C1–C4 constant delay 1–4 | 90–92% | 1.00 / 1–3 | 0–0.3% | 4–5 (4–5) | 0 |
| C5 delay 3 from step 15 | 48% | 0.95 / 1 | 0% | 9 (11) | 0 |
| C6 delay 3 + noise 0.005 m | 89% | 1.00 / 1 | 0% | 4 (8) | 0 |
| C7 delay 3 + scale 0.5 | 92% | 1.00 / 1 | 0% | 4 (4) | 0 |
| C8 noise 0.01 m | 0.2% | 0.01 / 2 | 0.7% | n/a | 33 steps |

From E2 (`results/phase_e2_delay/onset_latency.csv`), where a delay starts, ends or changes
mid-episode:
- the applied delay takes a median of 8 steps (p90 9) to match the new value;
- after a delay *ends* (E2 C6, steps 15–29), 23% of episodes **never release** the stale
  estimate before the episode ends, because low late-episode excitation keeps the estimator
  abstaining.

**Consequences for G1:**
1. **There is no "always-on" compensator to beat.** The current controller is already
   estimator-gated. An "always-on" baseline therefore has to be defined explicitly: here it is
   `never_release`, which latches on at the first reliable d̂ > 0 and ignores later d̂ = 0.
   The literal always-on alternative, compensating from step 0 with a fixed d, was already
   tested as the E2 fixed-d sham. It harmed undisturbed episodes by 16 points, so it is not
   worth re-running.
2. **Oscillation is not a current problem.** The gate switches about once per episode. Any
   new supervisor must not make this worse (§2.6).
3. **False activations are already near zero, with no harm:** C0 and C8 show 0 harmed out of
   890 and 860 base successes held-out, and 0 in F1 development.
   - A supervisor cannot improve *safety on these conditions* measurably.
   - Proposing it as a safety device there would be claiming an effect that has no room to
     exist.
4. **Remaining headroom on the F1 conditions is small** (F1 development, pooled C1–C7, 2,100
   paired episodes):
   - contact-aware harms 17 base successes (0.8 points);
   - it loses 59 episodes that Phase E solved (2.8 points).

   2.8 points is the *upper bound* for any per-episode choice between the two predictors, and
   a causal per-step supervisor will capture less. Any predictor-selection effect will be
   small, about at the resolution of the design (§2.8).
5. **The one structural gap is release.** The latch has no way to drop a stale estimate when
   evidence for "no delay" never becomes *reliable*.
   - Compensating with a wrong d̂ is harmful: the E2 fixed-d sham lost 16 points
     undisturbed.
   - So an episode in which a delay *ends early* should expose the latch.
   - None of the F1 or held-out conditions contain such an episode. E2's C6 ended its delay
     at step 29, too late to matter (base success 96.7%, Δ −0.7).

### 1.3 Confidence signals available at decision time (all causal)

| signal | already computed? | notes |
|---|---|---|
| `transitions`, `excitation`, `r2`, `margin`, `gain`, `*_reliable` | yes (`Estimate`) | the existing gate |
| per-lag residuals SSE(k) | computed inside `estimate()`, **not exposed** | needed for evidence-based release (lag 0 vs the latched lag). Exposing them is read-only and leaves `Estimate` unchanged. |
| latch age (steps since the last reliable confirmation) | no; trivial | timeout-style release |
| **realised prediction error** | no | the prediction made at t−d̂ for time t can be checked against the observation at t. This is the only signal about *model validity*, as opposed to delay identification. Noisy under observation noise. |
| disagreement between learned and Phase E predictions | no; ~10 µs extra | an unverified proxy, weaker than the realised error |
| predicted or observed gripper–cube distance | trivial | the F1 contact label uses < 5 cm horizontally |
| model uncertainty (ensemble, dropout) | **not available** | there is a single frozen MLP; adding one would mean retraining, which is out of scope |

### 1.4 Computation (measured, synthetic inputs, 1 CPU thread, no environment)

| component | µs per call |
|---|---|
| SAC forward (with normalisation) | 51 |
| `estimator.estimate()` | 37 |
| Phase E predictor (d = 3) | 10 |
| learned rollout | 48 · d (d = 3: 142; d = 4: 187) |

- **Per step:** contact-aware at d = 3 costs about 230 µs, against 51 µs for frozen SAC. That
  is 0.6% of the 40 ms control period, and well below a MuJoCo step.
- **Consequence:** computational efficiency is **not a meaningful bottleneck on this
  platform**. G1 can measure overhead and report reductions, but cannot claim a practically
  important saving. A saving would matter only on a much smaller compute budget, which this
  study does not model.

### 1.5 Would a supervisor be a genuinely different intervention?

Only if it answers a question the existing gate does not. The existing gate asks: "is there a
reliably identified delay of d steps right now?" Three questions it cannot answer, each
justified by §1.2–1.4:

| supervisor | question it asks | genuinely new? |
|---|---|---|
| **A, evidence-based release** | "Has the evidence turned against the latched delay, even though no *reliable* d̂ = 0 has arrived?" | Yes. It changes when compensation *switches off*, which the latch never does without a reliable zero. |
| **B, realised-error predictor selection** | "Which predictor has actually been right in the last few steps?" | Yes. It uses the verified outcome of earlier predictions, a signal the estimator never sees. |
| **C, contact-gated learned model** | "Is the cube close enough that predicting its motion matters?" | Partly. It changes the *form* of compensation, not activation, and its main value is computation and the free-space drift, both small. |
| stricter engagement (k consecutive reliable estimates) | "Is the onset real?" | **Not proposed.** False activations are already harmless and about 0.5% of steps, so there is nothing to gain and onset latency to lose. |

---

## Part 2: experimental proposal (draft pre-registration)

### 2.1 Configurations (frozen SAC policies `fetch_push_fp32_seed{0,1,2}` throughout)

| configuration | description | role |
|---|---|---|
| `base_only` | frozen SAC | baseline |
| `phase_e_predictor` | Phase E estimator + latch + static-cube predictor | reference |
| `contact_learned` | **current validated compensator** (estimator gate + latch + learned rollout) | primary comparator |
| `never_release` | as `contact_learned`, but ignores later reliable d̂ = 0 | "always-on" control |
| `sup_A_release` | `contact_learned` + supervisor A | **strategy 1** |
| `sup_A_timeout` | `contact_learned`, releasing when the latch has gone unconfirmed for L steps | ablation: does A need *evidence*, or does any release do? |
| `sup_A_yoked` | releases at the *same steps* as `sup_A_release` did in a different episode of the same condition and policy | sham: matched release rate and timing, wrong episode |
| `sup_B_select` | `contact_learned` + supervisor B | **strategy 2** |
| `sup_B_yoked` | predictor-choice sequence copied from a different episode's `sup_B_select` run | sham |
| `sup_C_contact_gate` | learned rollout only when the gripper is within 5 cm of the cube; otherwise Phase E | **strategy 3** (efficiency) |
| `oracle_sim_predictor` | PRIVILEGED simulator look-ahead with the latched d̂ | descriptive upper bound only |

**Yoking:** episode i takes its sequence from episode (i + 1) mod 100 of the same condition and
policy. The yoked shams therefore run after their source configuration. Yoking controls for
*how often* and *when in an episode* a supervisor switches, but not for *why*.

### 2.2 The strategies (all causal, using only information available at step t)

**Common rules for every supervisor:**
- **Minimum dwell:** 5 steps (= max_lag + 1) after any mode change. Commands issued under the
  previous mode keep executing for up to d̂ steps after a switch, so switching back sooner
  cannot be evaluated on its own consequences.
- **No re-engagement without fresh evidence:** a released supervisor can re-engage only
  through a *new* reliable d̂ > 0 from the existing estimator, which already needs ≥ 4 fresh
  transitions.
- **Recorded every step:** mode, reason for the switch, and the signal values.

**A. Evidence-based release (one-sided CUSUM).**
- While the latch holds d̂ > 0, compute ℓ_t = log(SSE(d̂) / SSE(0)) on the current window,
  whenever the window has at least the minimum number of transitions.
- Accumulate S_t = max(0, S_{t−1} + ℓ_t − κ), and release (back to raw SAC) when S_t > h.
- **The reliable-estimate path is unchanged:** a reliable d̂ = 0 still releases immediately.
- **Constants:**
  - κ = log 2, mirroring the estimator's margin of 2.
  - h is set on the *calibration* seed block (§2.5) using a detection criterion only: the
    smallest h whose false-release rate is ≤ 1% of active steps under constant delay
    (C1–C4). Task success is **never** used to set h.
- **Hysteresis:** release needs accumulated evidence (h), re-engagement needs a fresh reliable
  estimate, and the minimum dwell applies.

**A-timeout (ablation).**
- Release when the latch has not been re-confirmed by a reliable estimate for L = 10 steps
  (one window), fixed in advance.
- Expected to release wrongly under *constant* delay late in episodes, where excitation drops
  (E: reliable only 33% of steps at t ≥ 30). That is exactly why it is the right ablation for
  A's evidence requirement.

**B. Realised-error predictor selection.**
- At decision t−d̂, store both predictors' forecasts of gripper and cube position at t. At t,
  score each against the observed positions:
  e = ‖Δgripper‖ + ‖Δcube‖, over horizontal + vertical position, in metres.
- Keep an exponentially weighted mean of each predictor's error with a span of 5 steps, fixed
  in advance.
- Use the learned rollout by default. Switch to Phase E when its error is below
  0.8 × the learned model's error, and switch back when the learned error is below
  0.8 × Phase E's.
  - The 20% margin is fixed in advance; it is the hysteresis.
  - The minimum dwell also applies.
- **No calibration constants are fit.**
- **Noise:** observation noise inflates both errors alike, and the selection only compares
  them, so it is insensitive to the noise level by construction. A test checks this on
  synthetic data.
- **Expected behaviour (stated now, before any run):** B will mostly pick Phase E in free
  space (learned cube drift 4.8 mm vs 1.3 mm) and the learned model in contact. So B is a
  *data-driven* version of C, and comparing B with C tests whether verified error adds
  anything beyond a geometric rule.

**C. Contact-gated learned model.**
- Run the learned rollout only when the observed horizontal gripper–cube distance is < 5 cm
  (F1's pre-registered contact radius); otherwise use the Phase E prediction.
- No constants are fit.
- Hysteresis: the gate opens at < 5 cm and closes at > 6 cm, plus the minimum dwell.

### 2.3 Conditions

The F1 conditions are kept so the comparator replicates. New transient conditions are added
**because of the mechanism in §1.2(5)**, before any outcome is seen.

| id | condition | purpose |
|---|---|---|
| C0 | undisturbed | harm |
| C1–C4 | constant delay 1, 2, 3, 4 | non-inferiority of the supervisors |
| C5 | delay 3 from step 15 | mid-episode onset |
| C6 | delay 3 + position noise 0.005 m | noise (stresses B) |
| C7 | delay 3 + action scale 0.5 | gain loss |
| C8 | position noise 0.01 m | harm (stresses B and the A false-release rate) |
| **T1** | delay 3 on steps 0–14, then none | **early end: about 35 steps of possible stale compensation** |
| **T2** | delay 3 on steps 15–29 | E2 C6 replication (late end; little expected) |
| **T3** | delay 3 on steps 0–9 and 20–29 | intermittent: oscillation and re-engagement stress |
| **T4** | delay 1, changing to 3 at step 20 | E2 C9 replication (change of value) |

**Transient conditions:** T1–T4. **Constant-delay conditions:** C1–C7. **Harm conditions:**
C0 and C8. Whether the disturbance pipeline handles a delay that ends (queue flush) correctly
is checked on debugging seeds before anything else. If it does not, the run stops and the
issue is reported to you; it is not patched silently.

### 2.4 Metrics (pre-specified)

- **Success:** final success rate per configuration and condition.
- **Paired differences:** with rescued and harmed counts, vs `base_only` and vs
  `contact_learned`.
- **Detection latency:** steps from a true delay onset to compensation running with the
  correct d̂.
- **Release latency:** steps from a true delay end to compensation off. Episodes that never
  release are counted separately.
- **False activations:** steps and episodes with compensation on while the true delay is 0
  (C0, C8, and the off-phases of T1–T3).
- **False releases:** steps with compensation off while the true delay is > 0, after first
  engagement (C1–C7).
- **Switching:** mode changes per episode (mean, max, share of episodes with ≥ 4), dwell-time
  distribution, and occupancy of each mode.
  - **Oscillation flag:** any episode with ≥ 6 switches is listed individually.
  - Maximum possible switches: 50 / 5 = 10.
- **Computation:**
  - per-step controller wall time, excluding the environment step (`perf_counter_ns`
    around the decision): median, p99, and mean per episode;
  - learned-model calls per episode;
  - single-thread, timings interleaved across configurations.

  Wall time is descriptive only, because it depends on the machine.
- **Prediction:** realised gripper and cube errors for each predictor, split by contact and
  free space.

### 2.5 Seeds (development only; nothing at or above 4,000,000)

Fresh validation-range blocks, disjoint from every earlier phase (enforced by
`farc.supervision.seeds.check_g1_seeds`, added with this draft and tested):

| role | seeds | used for |
|---|---|---|
| debugging | 3,660,000–009 | code checks, table shapes, delay-ending pipeline check; outcomes not analysed |
| calibration | 3,661,000–049 | setting h for supervisor A by its false-release criterion; success not computed |
| evaluation | 3,662,000–099 | the pre-registered run: 100 seeds × 3 policies = 300 paired episodes per cell |
| exploratory | 3,663,000–099 | reserved for post-hoc follow-ups, labelled as such |

**Size:** 13 conditions × 11 configurations × 300 = **42,900 episodes**. At about 1,350
episodes per minute (held-out run rate) that is roughly 35–45 minutes on 3 processes.

### 2.6 Hypotheses

**Statistics:** paired episode-seed cluster bootstrap, 10,000 resamples. **Multiplicity:**
Holm across the primary family (G1–G4) at one-sided α = 0.025. Non-inferiority margins are
fixed now.

| id | hypothesis | comparison | conditions | supported if |
|---|---|---|---|---|
| **G1** | A rescues stale-latch episodes | `sup_A_release` − `contact_learned` | T1–T4 pooled (1,200) | CI lower bound > 0 |
| **G2** | A costs nothing under constant delay | `sup_A_release` − `contact_learned` | C1–C7 pooled (2,100) | CI lower bound > −1.0 point |
| **G3** | B improves on the current compensator | `sup_B_select` − `contact_learned` | C1–C7 + T1–T4 pooled (3,300) | CI lower bound > 0 |
| **G4** | C keeps performance while cutting learned-model calls | `sup_C_contact_gate` − `contact_learned` | C1–C7 pooled | CI lower bound > −1.0 point **and** learned-model calls reduced by ≥ 30% |

**Specificity checks**, secondary and reported with CIs: these decide whether a supported
effect is *specific*, or just activity.

| check | comparison | conditions | specific if |
|---|---|---|---|
| S-A1 | `sup_A_release` − `sup_A_yoked` | T1–T4 | CI lower bound > 0 |
| S-A2 | `sup_A_release` − `sup_A_timeout` | C1–C7 + T1–T4 | reported both ways: A should match the timeout on T1 and beat it on C1–C7 |
| S-B | `sup_B_select` − `sup_B_yoked` | C1–C7 + T1–T4 | CI lower bound > 0 |
| S-BC | `sup_B_select` − `sup_C_contact_gate` | C1–C7 + T1–T4 | descriptive: does verified error add anything beyond geometry? |
| AO | `contact_learned` − `never_release` | T1–T4 | descriptive: how much the existing release already helps |

**Safety gates**, binding for any recommendation: every supervisor vs `base_only` on each of
C0 and C8 must have a CI lower bound > −2 points and ≤ 2% of base successes harmed. These are
the same thresholds as the held-out run.

**Prior expectations, stated before any run** (so the outcome can be judged against them):
- **G1:** the effect is likely only on T1, where stale compensation lasts long; T2 is likely
  null, as in E2.
- **G2:** should pass if h is calibrated as specified.
- **G3:** the upper bound is about 2.8 points, so a small positive effect is plausible but
  may not reach significance.
- **G4:** very likely to pass the success criterion, but the computation saving it buys is
  practically negligible (§1.4).

### 2.7 Failure criteria and what each outcome means

| outcome | conclusion |
|---|---|
| G1 fails | Evidence-based release does not rescue episodes. The latch's existing release is adequate for these transients, and no switching layer is recommended for activation. |
| G1 passes but S-A1 fails | The benefit is from releasing at *some* time, not from the evidence. Prefer the simpler timeout, if G2 holds for it (descriptively). |
| G2 fails | Release is too eager under constant delay. A is **not recommended** even if G1 passes. |
| G3 fails | Verified prediction error does not improve control beyond the current compensator. Report the measured switching behaviour; no recommendation. |
| G3 passes but S-B fails | The benefit comes from switching patterns, not the error signal. Not recommended. |
| G4 fails | The contact gate harms success. The always-learned compensator stays. |
| any safety gate fails for a supervisor | That supervisor is not recommended, whatever its other results. |
| any oscillation flag | Reported per episode. The dwell or hysteresis rule is a candidate for revision **only as a new pre-registered variant on the exploratory block**. |

### 2.8 Precision

- **Constant-delay comparisons (2,100 pairs):** F1's paired CI half-widths for contact-aware
  vs Phase E were about ±1.7 points, from discordant pairs at 3–6%. A supervisor that agrees
  with `contact_learned` most of the time will have far fewer discordant pairs and tighter
  intervals. That is why the −1-point non-inferiority margins are testable.
- **Transient comparisons (1,200 pairs):** CI half-widths of roughly ±2 points. An effect
  under about 3 points on T1–T4 pooled will probably not reach significance. That is accepted
  and stated here.

### 2.9 Procedure after approval

1. Commit the approved plan and pin its SHA-256 in the G1 config.
2. Implement:
   - expose per-lag residuals read-only;
   - supervisors A, B and C as wrappers around the *unchanged* estimator, latch and
     predictors;
   - the yoked shams;
   - per-step timing;
   - unit tests:
     - causality (no session or ground truth reaches a supervisor);
     - minimum dwell and hysteresis;
     - CUSUM arithmetic;
     - noise invariance of B's comparison;
     - the never-release semantics;
     - seed guards.
3. Debugging seeds: check that the code runs and the tables build, and check the delay-ending
   pipeline.
4. Calibration seeds: set h by the false-release criterion. Record the curve and the chosen
   h, then commit before the evaluation.
5. Run the evaluation once, with write-once raw files, a clean tree and the plan hash checked.
6. Analysis: the pre-written code, run once. Results are reported in full, failures included.
7. A separate PR, not merged without your approval.

### 2.10 Limitations (known now)

- **Development-only.** This is a pre-registered development test, not a held-out
  confirmation.
  - Supervisor constants are fixed or calibrated on separate seeds, but the conditions
    T1–T4 are of my own design.
  - Any positive result would need a separate, approved held-out run on 4,000,300+.
- **Small headroom on established conditions:** ≤ 2.8 points for B on C1–C7, and ≈ 0 for
  safety on C0/C8. G1 deliberately targets the one structural gap; a null there is
  informative, not a failure of the study.
- **Efficiency is not a bottleneck here** (§1.4), so G4 can show a reduction in calls, not a
  practical saving.
- **No epistemic uncertainty.** A single frozen MLP gives no uncertainty estimate. B's
  realised error is retrospective by d̂ steps, so it reacts to model failure rather than
  anticipating it.
- **Low excitation limits A.** When the arm barely moves late in an episode, the window holds
  little evidence either way, so A may still fail to release (the 23% stale case in E2). G1
  measures how often.
- **Simulation only:** a pure transport delay of issued commands, MuJoCo contact, and a nearly
  linear mocap plant.
- **Frozen components.** The Phase E estimator and the F1 model stay frozen, so a better
  detector (for example a shorter window) is outside G1 and would be its own phase.
