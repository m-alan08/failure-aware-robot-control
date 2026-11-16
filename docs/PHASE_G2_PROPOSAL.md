# Phase G2: fast detection of nonstationary actuator delays

**Status: DRAFT for approval. Nothing has been implemented, calibrated or run.**
- No G2 seed has been used, no frozen model was modified, and no held-out seed was touched.
- The only new numbers here come from *read-only* analyses of existing development logs: the
  G1 evaluation and the F1 evaluation. They are labelled as such.
- G1's transient conditions motivated G2. They are development evidence, not independent
  confirmation.

**Question:** can the controller recognise a change in actuator delay quickly enough to avoid
harmful stale compensation, while keeping the large gains under persistent delay?

**Intended contribution:** robust adaptation to *nonstationary* actuator delay, tested on a
pre-specified grid of onsets, durations, magnitudes and directions. It is not meant as a fix
for G1's three conditions.

**Repository state:**
- PRs #6 (Phase E) and #7 (F1) were merged earlier with your permission. PR #8 (G1) is open
  and unmerged.
- This branch, `claude/phase-g2-delay-change-detection`, is stacked on the G1 branch because
  it builds on the G1 runner. It is committed locally only.

---

## 1. Architecture audit

### 1.1 What runs at each decision step t (`run_supervised_episode`)

1. `session.observe()`: applies observation noise and evaluates frozen SAC on the raw view
   (the *base action*).
2. `ActuatorEstimator.observe(g_t)`: appends the controller's possibly noisy gripper position.
3. `estimate()`, over the window of transitions τ ∈ [t−10, t−1]: for each lag k ∈ {0..4},
   - fit Δg_xy[τ] ≈ β_k · 0.05 · c_xy[τ−k] by least squares through the origin;
   - SSE(k) is the residual sum of squares, and d̂ = argmin_k SSE(k).
4. **Gates.** The delay is *reliable* only if all of these hold:
   - at least 4 transitions;
   - excitation, the mean of ‖0.05 c‖², is at least 2.25e-4;
   - R² ≥ 0.8;
   - margin, the second-best SSE over the best SSE, is at least 2.
5. `LatchedEstimate`: switches on at the first reliable d̂ > 0, holds through any number of
   abstentions, and switches off only at a reliable d̂ = 0.
6. G1 supervisor A, if enabled: a CUSUM on log(SSE(d̂)/SSE(0)) − log 2 over the *same
   window*, with a 5-step dwell. Re-engagement needs a fresh reliable estimate.
7. **Compensation, when the latched d̂ > 0:**
   - the predictor takes (view, queued = c[t−d̂..t−1], min(ĝ, 1));
   - **Phase E** shifts the gripper by K0·ĝ·0.05·Σ queued and keeps the cube static;
   - **F1** rolls the learned MLP forward d̂ steps, moving the cube too;
   - SAC is then evaluated on the predicted observation.
8. `session.act(c_t)`: the disturbance pipeline turns the issued command into the executed
   one, applying delay, then scale, then noise.

### 1.2 Disturbance-pipeline semantics (`ActionDelay.apply`)

While a delay spec is active at step t, the executed command is the *issued* command from
d steps earlier, c[t−d]; before step 0 it is zero. Otherwise the executed command is c[t].

**Consequences at transitions** (checked against the code; the G1 debug run verified them
step by step):

| transition | what the actuator executes | artifact |
|---|---|---|
| onset 0 → d at step o > 0 | c[o−d] … c[o−1], **each a second time** | already-executed commands are replayed |
| onset at step 0 | zeros for d steps | a *start-up cue*: the arm visibly stalls. Onset detection is 4 steps instead of 9–10 mid-episode. |
| end d → 0 at step e | c[e] immediately; **c[e−d] … c[e−1] are never executed** | queued commands are dropped |
| change d₁ → d₂ > d₁ | d₂ − d₁ commands replayed | as onset |
| change d₁ → d₂ < d₁ | d₁ − d₂ commands dropped | as end |

These are the semantics of a "replay the command from d steps ago" model. A real transport
buffer that shrinks might instead *drain* its queue, and one that grows might *hold* the last
command. **Every transient result so far is conditional on drop/replay semantics.** §3 tests
whether they matter.

### 1.3 Information causally available at step t

| available | not available |
|---|---|
| current and past *views* (noisy when noise is injected): gripper, cube, velocities, goal | executed commands |
| every issued command c[0..t−1] | the true delay, or when it changes |
| the base action on the current view | the simulator state, future delays, noise level |
| the estimator, latch and supervisor state | |
| the latest transition: Δg[t−1] = g_t − g_{t−1} | |

**The earliest possible evidence of a change at step e is at decision e + 1**, from one
transition. Decision e itself has no evidence.

### 1.4 Why release takes about 9 steps

This is a read-only analysis of the G1 development logs. `est_delay` (the argmin, logged even
when the estimator abstains) and `est_reliable` were recorded on every step. k is the number
of decision steps since the true transition, which is also the number of post-change
transitions in the window. The table uses `base_only` episodes, so the estimator runs with no
compensation feedback (1,200 end-of-delay events).

| k | argmin = new lag | argmin = old lag | reliable (any) | reliable and new |
|---|---|---|---|---|
| 0 | 0.00 | 0.99 | 0.94 | 0.00 |
| 2 | 0.06 | 0.77 | 0.38 | 0.01 |
| 4 | 0.18 | 0.54 | 0.17 | 0.03 |
| 6 | 0.39 | 0.32 | 0.12 | 0.08 |
| 7 | 0.54 | 0.20 | 0.18 | 0.16 |
| 8 | 0.70 | 0.08 | 0.30 | 0.29 |
| 9 | 0.83 | 0.02 | 0.47 | 0.46 |
| 10 | 0.97 | 0.00 | 0.79 | 0.79 |

- **The same pattern holds for 1 → 3 increases** and for mid-episode onsets.
- **Under contact-aware compensation it is nearly identical.** "Compensation off" reaches
  50% at k = 9.
- **G1 A's window statistic** log(SSE(3)/SSE(0)) has a median of −2.3 at k = 1, −0.6 at
  k = 4, and crosses 0 only at k = 6.

**Answer to questions 1 and 2: it is mainly the window, then the gates; the latch only sets
what happens in the meantime.**
1. **Window (dominant).** With k post-change and 10 − k pre-change transitions, SSE(old)
   collects large residuals from the post-change part and SSE(new) from the pre-change part.
   The *best-fit lag itself* stays on the old value until roughly 7 of 10 transitions are
   post-change. A batch least-squares fit over a fixed window cannot react faster than the
   point where the new regime dominates the window. That is structural, not a tuning
   accident.
2. **Gates (adds about 1–2 steps).** A mixed window fits *no* single lag well. R² and the
   margin collapse (reliability 12–18% at k = 4–6), so the estimator abstains. Even once the
   argmin is right, reliability lags by about 1–2 steps (k = 8: 70% against 29%). The gates
   behave correctly: they refuse mixed windows. Wrong lags reported as reliable are rare
   (≤ 2% at any k).
3. **Latch (no latency; it sets what happens meanwhile).** The latch adds no delay of its own;
   it updates on the first reliable estimate. But while the estimator abstains, it holds the
   *old* d̂. Stale compensation is therefore the latch doing exactly what it was designed to
   do.
4. **Excitation (the tail).** Late transitions carry smaller commands, so more episodes never
   release before the episode ends: 49 of 300 when the delay ends at step 30, against 10 of
   300 at step 15 (G1).

### 1.5 Do the residuals contain enough evidence to detect a change faster? (Question 4)

This is a read-only analysis of F1 development logs (`base_only`; views and issued commands
were logged). For each transition τ it computes e_k = ‖Δg_xy[τ] − K0·0.05·c_xy[τ−k]‖².

| condition (true lag) | wrong/true residual, *closest* wrong lag (median) | lag-0 vs lag-3 ratio (median) | share with 0-vs-3 ratio > 4 | median true-lag residual |
|---|---|---|---|---|
| C0 undisturbed (0) | 6.3 | 101 | 0.76 | 1.4 mm |
| C3 delay 3 (3) | **1.2** | 81 | 0.76 | 2.9 mm |
| C6 delay 3 + noise 0.005 (3) | 1.0 | 11 | 0.64 | 10.0 mm |
| C8 noise 0.01 (0) | 1.2 | **3.9** | 0.49 | 16.9 mm |

- **Yes, for the question "is d̂ still right, or is there no delay?".** A single transition
  separates lag 0 from lag 3 by a factor of about 100 without noise and about 11 at 0.5 cm
  noise. One or two post-change transitions should usually suffice there.
  - At 1 cm noise the evidence per transition is weak (median ratio 3.9), and several
    transitions are needed.
  - Evidence also weakens late in an episode: the C0 median falls from 281 at t ≤ 10 to 10
    at t > 40.
- **No, for the question "which lag now?".** Adjacent lags are nearly indistinguishable from
  one transition (median 1.2 for the closest wrong lag in C3), because SAC commands are
  smooth.
- **Design consequence:** *invalidation* (is the current estimate wrong?) can be fast;
  *re-identification* (what is the new value?) needs several transitions. This argues for
  separating the two (§2).
- **The window statistic dilutes this evidence.** G1 A uses the window SSE ratio, so the
  strong single-transition evidence is averaged with up to 9 old transitions. That explains
  why A, built on the same residuals, could not release faster than about 6 steps.

### 1.6 Alternative explanations for the transient harm (Question 5)

The stale-estimate hypothesis is **plausible but not established**, and the logs contain a
warning sign against it.

**What supports it:**
- **Condition-level dose-response.** On T1, `never_release` (35 stale steps) loses 22 points
  against frozen SAC; contact-aware (about 9 stale steps) loses 6.7.
- **Persistent delays show no harm.** Where a delay never ends (T4, C1–C7), the same
  compensator gains 20–80 points over SAC.

**What weakens it (read-only, G1 logs):**
- **No episode-level association.** *Within* each transient condition, episodes harmed by
  contact-aware do **not** have more stale steps than episodes it kept:
  - T1: 8.3 against 10.0;
  - T2: 9.1 against 10.0;
  - T3: 19.6 against 18.5.
- **This is a weak test.** Stale duration varies little between episodes, and the logs do not
  record *when* an episode was lost. But it means the logs cannot establish the mechanism.

**Competing explanations, each tested in §3:**

| id | mechanism | why it is plausible |
|---|---|---|
| E1 | **stale latching:** compensating with d̂ = 3 after the delay ends | the latch holds through abstention (§1.4) |
| E2 | **double-counting by the predictor:** after an end, `queued_commands` = c[t−3..t−1] have *already executed*, so the predictor adds their displacement a second time and SAC acts on a state about 3 steps ahead of reality | follows from the code. The learned model, which also moves the cube, may err more than Phase E (T1: −6.7 against −2.3). |
| E3 | **dropped commands at the end:** c[e−3..e−1], planned by the compensator for steps e..e+2, are never executed | follows from the pipeline (§1.2); it would hurt even with perfect identification |
| E4 | **replayed commands at onset or increase:** already-executed commands run again | follows from the pipeline |
| E5 | **switching discontinuity:** SAC's input jumps between the predicted and raw observation when compensation turns on or off | it happens in every compensated transient episode |
| E6 | **compensation *during* a short delay** sends the episode on a different path that SAC handles worse after the end | frozen SAC tolerates 10–15-step delays well (97–98%); compensation may not be needed |
| E7 | **misidentification:** a reliable *wrong* lag during the mixed window | rare in the logs (≤ 2% of steps) |

**The G1 oracle anomaly (T3: 59%).** `oracle_sim_predictor` is privileged about *dynamics*
only; it uses the *estimated* latched d̂.
- When d̂ = 3 is stale (true delay 0), its look-ahead executes `session.act(base_action)`
  placeholders for 3 steps. With no delay these placeholders *actually move the arm* in the
  look-ahead, so it predicts the state after 3 steps of base-policy motion that will not
  happen. That is a stronger version of E2.
- In T3 the off-phase (steps 10–19) is about as long as the release latency, so the oracle
  spends most of it in this regime.
- **Hypothesis (to test in M6):** the anomaly is stale d̂ combined with look-ahead semantics,
  not a simulator-oracle bug.

### 1.7 Other implementation notes

- **Overlapping delays compose.** Two `action_delay` specs active at the same step would
  compose (a double delay). G2 configs must forbid overlaps, and a test will check this.
- **Onset at step 0 is a different, easier event** (the zero-padding start-up cue). G2's
  grid therefore uses onsets > 0.
- **G1's "release latency"** counts steps to the first *off* step. After an end, "correct"
  means off, so for an end this is the same as stale-compensation duration. For changes
  between nonzero delays, G2 uses the more general measure in §6.2.
- **Estimator and pipeline are consistent.** The estimator regresses on *issued* commands,
  and the pipeline replays issued commands. Gain loss enters as β_k; noise enters through Δg.

---

## 2. Candidates

**Notation:**
- r_τ(k) = Δg_xy[τ] − K0·ĝ·0.05·c_xy[τ−k], the per-transition residual under lag k;
- d̂ is the latched delay, W = 10, and K = {0..4}.

### Candidate A: short-window estimation

- **Formulation:** the existing estimator with window w_s < 10. Everything else is
  unchanged: the gates (re-calibrated, see below), the latch and the predictor.
- **Inputs:** the same as now.
- **Overhead:** below the existing estimator's, which scales roughly with the window.
  Measured in the G1 audit: 37 µs per call at w = 10, on synthetic inputs.
- **Expected latency (analytic only):**
  - the mixed-window argument of §1.4 scales with the window, so the best-fit lag should
    switch after roughly the same *fraction* of w post-change transitions;
  - lower bound: `min_transitions` = 4.
  - No empirical estimate is claimed.
- **False-alarm risk (main weakness):**
  - Fewer transitions give less power to separate *adjacent* lags (§1.5: ratio 1.2 per
    transition), so under persistent delay d̂ may flip 3 ↔ 2/4 with the same gates. That
    would erode exactly the persistent gains G2 must keep.
  - Excitation per window is lower, so the estimator abstains more and the latch holds longer.
- **Behaviour:**
  - persistent: risk of lag flicker;
  - onset, end, change: all faster by the same mechanism, onset included, which also raises
    false-activation risk on C0/C8;
  - no change point is estimated.
- **What changes:** the estimator configuration. The class itself is unchanged and a new
  config instance is used.
- **Calibration** on calibration seeds, using detection criteria only:
  - w_s ∈ {4, 5, 6, 7};
  - choose the smallest w_s whose *wrong reliable-lag rate* under persistent delay is at
    most the w = 10 rate + 0.5 percentage points;
  - and whose false-activation rate on C0/C8 is at most 0.5% of steps.
  - The gates stay at their Phase E values. Re-tuning them is not allowed: that would be a
    second search.

### Candidate B: two-timescale estimation (fast window as invalidation detector)

- **Formulation:**
  - **Slow:** the existing estimator and latch, unchanged (w = 10). They supply the
    *trusted* d̂.
  - **Fast:** a second estimator instance with window w_f, using the same gates.
  - **Alarm:** while d̂ > 0, alarm if the fast estimator reports a reliable lag ≠ d̂ on
    m = 2 consecutive steps.
  - **Response:** *suspend* compensation; the controller acts as frozen SAC.
  - **The fast lag is never applied.** Re-engagement happens only when the slow estimator
    gives a reliable d̂′ > 0 on a step at least 5 steps after the alarm (dwell). If the slow
    estimate becomes a reliable 0, compensation stays off, as the latch already does.
- **Inputs:** the same as now.
- **Overhead:** one extra estimator call per step.
- **Expected latency:** alarm latency is the fast window's mixed-window latency plus m − 1.
  Re-engagement after an *increase* waits for the slow estimator, which is no faster than
  now.
- **False-alarm risk:**
  - the fast window's adjacent-lag flicker under persistent delay (as in A), filtered by
    m = 2;
  - every false alarm costs at least 5 uncompensated steps.
- **Behaviour:**
  - persistent: suspensions on false alarms;
  - end: faster switch-off;
  - change between nonzero delays: suspends, then re-engages with the slow estimate (the gap
    is uncompensated);
  - **onset: unchanged** (the slow path).
- **What changes:** a supervisor layer only; the estimator, latch and predictors are
  unchanged.
- **Calibration:** w_f ∈ {3, 4, 5}, with m and the dwell fixed. Choose the smallest w_f whose
  alarm rate under persistent delay suspends at most 1% of active steps (the G1 criterion).

### Candidate C: sequential change detection on fresh per-transition residuals (recommended)

- **Formulation:** a multi-alternative CUSUM, equivalent to a window-limited GLR on Gaussian
  residuals.
  - **Per-transition log-likelihood ratio** for an alternative lag k ≠ d̂:
    ℓ_τ(k) = ( ‖r_τ(d̂)‖² − ‖r_τ(k)‖² ) / (2 σ̂²).
  - **Recursion:** S_k ← max(0, S_k + ℓ_τ(k) − κ), for each k ∈ K \ {d̂}.
  - **Alarm:** max_k S_k > h.
  - **Change-point estimate:** τ̂ = the step after the last time the winning S_k was 0.
  - **Noise scale:** σ̂² is the per-axis residual variance of the slow window under d̂ at the
    time the latch last confirmed d̂, floored at σ₀². σ₀ is a calibration constant.
    Alternatively a scale-free variant replaces ℓ with ½·log((‖r(d̂)‖² + σ₀²) / (‖r(k)‖² +
    σ₀²)). The choice between the two is made on calibration seeds by the same detection
    criterion (§5.3), before evaluation.
- **Response** (the same idea as B, but with a reset instead of waiting for the slow window):
  1. Suspend compensation (frozen SAC).
  2. **Restart the identification window at τ̂.** The estimator fit uses only transitions
     τ ≥ τ̂. This is a new window-start argument; the frozen fit, gates and latch are
     otherwise unchanged.
  3. Re-engage when the *reset* window gives a reliable d̂′ > 0. That needs at least
     `min_transitions` = 4 post-change transitions and every existing gate.
  4. A reliable 0 keeps compensation off.

  An unverified alternative lag is never applied: the CUSUM's winning k is used only to
  locate τ̂.
- **Onset** (latched d̂ = 0 or none): the same test with d̂ = 0 detects "some delay
  appeared", and resets the window so that identification uses only post-onset data. This
  shortens onset identification as well, again through the existing gates.
- **Inputs:** the same as now (views and issued commands), plus the slow estimator's
  residual variance.
- **Overhead:** 5 two-dimensional residuals and 4 scalar recursions per step (negligible),
  plus at most one extra windowed fit while a reset window is filling.
- **Expected latency (analytic only):**
  - the alarm can fire from decision e + 1, one transition after the change;
  - the expected delay grows roughly as h / KL, where KL is the per-transition divergence
    between the d̂ and k residual models;
  - §1.5 shows that divergence is large for 3 vs 0 without noise and small at 1 cm noise.
  - Re-engagement after an increase needs at least 4 post-change transitions.
- **False-alarm risk:**
  - model mismatch (contact, gain error, the mocap lag) inflates every residual. Comparing
    *lags* rather than absolute fit partly cancels this, because contact affects all lags
    alike;
  - adjacent lags give near-zero ℓ (little drift either way, so few alarms from them);
  - heavy observation noise lowers power and raises variance, so h and κ are set on
    calibration seeds.
- **Behaviour:**
  - persistent: rare alarms, each costing at least 4 suspended steps;
  - end: alarm, then suspend, which is the correct action;
  - increase or decrease between nonzero delays: alarm, suspend, then re-identification
    from post-change data only;
  - onset: faster engagement, with verification.
- **What changes:** a supervisor layer, plus a window-start argument to a *new* estimator
  wrapper built on `lag_residuals`. The frozen `ActuatorEstimator.estimate()` and the latch
  are not modified. The predictors are unchanged.

**How C differs from G1 supervisor A.** The raw information is the same: positions and issued
commands. C does **not** add a new information source. What differs is the statistic and the
response:

| | G1 A | C |
|---|---|---|
| evidence per step | window SSE ratio: 10 overlapping transitions, dominated by pre-change data | one *fresh* transition, so increments are nearly independent |
| alternatives | d̂ vs 0 only | d̂ vs every other lag, so increases and decreases are detected too |
| noise model | none (scale-free ratio) | explicit σ̂² with a floor |
| change time | not estimated | τ̂, used to restart identification |
| response | release; re-engage through the ordinary 10-step window | suspend; re-identify from post-change data only |

§1.4 shows the window statistic cannot cross zero before about k = 6. §1.5 shows the fresh
per-transition evidence is about 100:1 for 3 vs 0 without noise. The "repackaging" concern
is therefore real for A, and it is what C is designed to avoid.

### Summary

| | A short window | B two-timescale | C sequential + reset |
|---|---|---|---|
| speeds up the end of a delay | yes | yes | yes (fastest in principle) |
| speeds up onset | yes (no verification beyond the gates) | no | yes (with verification) |
| speeds up a change of value | yes | no (waits for the slow window) | yes (reset window) |
| main risk under persistent delay | lag flicker | false suspensions | false suspensions |
| trusts an unverified new lag | yes (a short-window reliable estimate) | no | no |
| new parameters | w_s | w_f (m, dwell fixed) | h, κ, σ₀ (dwell fixed) |
| changes to frozen code | none (new config) | none (layer) | none (layer + wrapper) |

---

## 3. Mechanism isolation (Stage 1: small, privileged diagnostics first)

**Purpose:** decide *whether* faster detection can help, before building and comparing
detectors. If stale compensation is not the main cause of transient harm, faster detection
cannot fix it, and the main evaluation should not run as designed.

**Every configuration marked PRIVILEGED** reads the true delay schedule or the simulator. It
is a diagnostic, never a candidate controller, and is excluded from every deployable claim.

### 3.1 Conditions

- **Transient grid (Stage 1 only):** delay 3 with onset ∈ {6, 18} × duration ∈ {4, 8, 16}
  (6 conditions).
- **Controls:** persistent delay 3 from onset 6; undisturbed.
- **Size:** 50 seeds × 3 policies = 150 paired episodes per cell, on the **mechanism** seed
  block.

### 3.2 Experiments

| id | design | what it isolates |
|---|---|---|
| **M1 stale-duration dose-response** (PRIVILEGED release) | contact-aware, but compensation is forced off exactly k steps after the true end, k ∈ {0, 1, 2, 3, 5, 7}, compared with deployed (about 9) and never-release | **E1.** If success at k = 0 is close to frozen SAC, stale latching explains the harm, and success should fall as k grows. If k = 0 is still clearly below SAC, the harm lies elsewhere. |
| **M2 identification × predictor** (2 × 3 factorial) | delay source {estimated latch, **PRIVILEGED** true schedule} × predictor {Phase E, learned, **PRIVILEGED** simulator look-ahead} | the true-schedule row removes E1 and E7. The remaining harm against frozen SAC comes from E2–E6. Comparing predictors isolates model error under each source (E2). |
| **M3 learned-model error under a stale delay** (offline, on M2's logged steps) | at every stale step, compare the learned and Phase E predicted states with the realised state at t + d̂ and at t | quantifies the double-count (E2) for each predictor and checks whether the learned model's cube motion amplifies it |
| **M4 switching without a transition** | (a) persistent delay 3: true-schedule compensation with a forced *off* window of L ∈ {3, 6} steps at step 18. (b) undisturbed: forced *on* with d = 3 for L ∈ {3, 6, 9} steps at step 18. | (a) harm from switching off and back on with no delay change (E5); (b) harm from stale-like compensation with no pipeline transition (E1/E2 without E3/E4) |
| **M5 pipeline semantics** (new variant, existing pipeline untouched) | rerun the transient grid with **drain** semantics at an end (the queue empties one command per step, so nothing is dropped) and **hold** semantics at an onset (the last executed command is held instead of replaying old ones). Frozen SAC, contact-aware and true-schedule compensation are each run under both semantics. | E3/E4. If transient harm largely disappears under drain/hold semantics, the drop/replay artifact explains part of it, and G2's conclusions must state their dependence on pipeline semantics. |
| **M6 the G1 oracle anomaly** | simulator look-ahead with {latched d̂, true current d, true schedule}, on an intermittent condition (delay 3, 10 on / 10 off ×2). The number of look-ahead placeholder actions that actually execute is logged. | tests whether the T3 collapse is stale d̂ combined with placeholder execution |

### 3.3 Time from a true transition to the first harmful compensation decision

Two measures, defined in advance:
1. **First mismatched decision** (parameter level, every configuration): the first step
   t ≥ the transition at which the compensation state differs from truth. That means active
   after an end, or active with d̂ ≠ d after a change. The **stale-compensation duration** is
   the number of such steps until the state is correct again.
2. **Point of no return** (outcome level, branching; M1 only, on episodes where frozen SAC
   succeeds and contact-aware fails):
   - for each step t from the transition to transition + 15, snapshot the session (exact
     restore already exists);
   - continue with **PRIVILEGED** correct compensation from t onwards, and record success.
   - The point of no return is the earliest t from which correct compensation no longer
     rescues the episode.
   - The first *harmful* decision is bounded by the last rescuing branch point.
   - This gives the window in which a detector would have had to act.

### 3.4 Stage-1 decision rules (fixed now)

- **"Stale compensation is the main cause"** if, pooled over the transient grid:
  - M1 at k = 0 recovers at least half of the gap between deployed contact-aware and frozen
    SAC (CI on the recovered share above 0.5);
  - and success decreases as k grows (one-sided trend test: a paired regression slope on k
    below 0).
- **Pipeline dependence** is flagged if M5's drain/hold semantics change contact-aware's
  transient harm by more than 2 points (CI excluding 0).
- **After Stage 1, I stop and report to you** whatever the result. G2's evaluation (§5)
  proceeds only with your approval:
  - if stale compensation is *not* the main cause, I will propose a revised Stage 3, or
    none;
  - if pipeline dependence is flagged, the §5 grid is run under both semantics.

**Stage-1 size:** about 17 configurations and variants × 8 conditions × 150 episodes ≈ 20,000
episodes, plus the M1 branches (a few thousand short rollouts) and M5's drain/hold variants.
G1 ran 42,900 episodes in about 8 minutes, so Stage 1 should take minutes.

---

## 4. Integrity, seeds and freezing

| role | seeds | use |
|---|---|---|
| debugging | 3,670,000–009 | code and table-shape checks only; outcomes not analysed |
| mechanism (Stage 1) | 3,671,000–049 | §3, its own pre-registered decision rules |
| calibration (Stage 2) | 3,672,000–099 | detection criteria only (§5.3); task success never read |
| development evaluation (Stage 3) | 3,673,000–099 | the single pre-registered comparison (§5) |
| exploratory | 3,674,000–099 | post-hoc only, labelled as such |
| **future held-out confirmation** | **4,000,300–4,000,599** | **untouched.** Usable only under a separate written plan that you approve. The G2 guard refuses every seed ≥ 4,000,000. |

All G2 blocks are disjoint from every earlier phase; a guard and tests are added with this
draft.

**Freezing procedure:**
1. On approval:
   - commit this plan;
   - pin its SHA-256 in the G2 configs;
   - runners refuse a changed plan or a dirty tree.
2. **Stage 1** (mechanism): run once, analysed with the §3.4 rules, then a checkpoint with you.
3. **Stage 2** (calibration): constants are chosen by the detection criteria on calibration
   seeds only. They are recorded in an **amendment file** (`docs/PHASE_G2_CALIBRATION.md`),
   whose hash is pinned in the evaluation config, so this plan itself is never edited.
4. **Stage 3** (evaluation): one run, with write-once raw files, a lock record and the
   pre-written analysis run once.
5. **Results:** reported in full, including failures. A separate PR, never merged without
   your approval.
6. **Held-out confirmation** would need its own plan, naming only the candidate(s) that pass
   Stage 3, and a separate approval.

**Preserved:** every existing model (SAC checkpoints and the F1 model, hash-checked), every
raw and derived result of Phases A–G1, and the held-out artifacts. Nothing existing is
modified. New code is additive.

---

## 5. Pre-registered development evaluation (Stage 3)

### 5.1 Conditions: a pre-specified grid, not G1's failures

Delay magnitudes for the grid are {2, 4} rather than G1's 3, and onsets are > 0, to avoid
reusing G1's exact cells and its start-up cue.

| family | cells | count |
|---|---|---|
| **P** persistent | constant d ∈ {1, 2, 3, 4}; onset at step 18 with d ∈ {2, 4} | 6 |
| **T** transient pulse | d ∈ {2, 4} × onset ∈ {6, 18, 30} × duration ∈ {4, 8, 16} (onset 30 + 16 < 50) | 18 |
| **I** intermittent | d ∈ {2, 4} × on/off period ∈ {4/4, 8/8}, from step 6 to the end | 4 |
| **S** steps between nonzero delays at step 20 | 1→4, 4→1, 2→4, 4→2 | 4 |
| **N/G** noise and gain | d = 4 persistent + position noise 0.005; d = 4 persistent + scale 0.5; transient d = 4 (onset 18, duration 8) + noise 0.005; the same + scale 0.5 | 4 |
| **H** harm | undisturbed; position noise 0.01 | 2 |

**Total:** 38 conditions. **Paired episodes:** 100 seeds × 3 frozen SAC policies = 300 per
cell.

### 5.2 Configurations

| configuration | role |
|---|---|
| `base_only` | frozen SAC |
| `contact_learned` | existing F1 compensator (primary comparator) |
| `phase_e_predictor` | reference (less transient harm in G1) |
| `sup_A_release` | G1 supervisor A (h = 0.5, unchanged) |
| `cand_A_short_window` | candidate A |
| `cand_B_two_timescale` | candidate B |
| `cand_C_sequential` | candidate C |
| `sham_C_yoked` | C's suspension mask copied from another episode of the same condition and policy: matched count and timing, wrong episode |
| `sham_B_yoked` | the same for B |
| **PRIV_true_schedule** | PRIVILEGED: learned predictor with the *true* current delay. A ceiling on what perfect identification buys. |
| **PRIV_release_oracle** | PRIVILEGED: deployed estimator, but compensation off exactly at every true end. A ceiling on what perfect *invalidation* buys. |

**Size:** 11 × 38 × 300 = 125,400 episodes, roughly 25 minutes at G1's measured rate.

### 5.3 Calibration criteria (Stage 2; detection only, success never read)

Calibration conditions: persistent d ∈ {1, 2, 3, 4} and the two H conditions, plus two
generic transient cells (d = 3, onset 12, durations 6 and 12; not in the §5.1 grid).

| candidate | criterion |
|---|---|
| A | as in §2 |
| B | as in §2 |
| C | choose h (grid {1, 2, 3, 4, 6, 8, 12}), κ (fixed at 0.5 nats), σ₀ (grid {1, 2, 4} mm) and the ℓ variant so that suspended steps are ≤ 1% of active steps under persistent delay, and false activations are ≤ 0.5% of steps on H. Among the settings that meet this, pick the one with the smallest **median release latency** on the two transient calibration cells. |

That last criterion uses a detection quantity from the calibration cells, not success.

### 5.4 Hypotheses

**Statistics:** paired episode-seed cluster bootstrap, 10,000 resamples, one-sided
α = 0.025. Differences are in success-rate points.

**Primary (co-primary for each candidate X ∈ {A, B, C}):** an intersection-union test.
Candidate X is **supported** only if both hold:
- **H1_X (transient harm reduced):** X − contact-aware, pooled over families T + I (22 cells,
  6,600 pairs), CI lower bound > 0.
- **H2_X (persistent gains kept):** X − contact-aware, pooled over family P (1,800 pairs),
  CI lower bound > −1.0 (non-inferiority).

The intersection-union test needs no adjustment within X; its p-value is the larger of the
two one-sided p-values. **Across the three candidates, Holm** is applied to those
intersection-union p-values at 0.025.

**Practical significance** (reported, not a pass criterion): a pooled H1 estimate of at
least **+2 points** counts as a meaningful reduction. G1's compensators lost 6–16 points to
frozen SAC on transients.

**Secondary (Holm within this family):**
- **S1 (change between nonzero delays):** X − contact-aware on family S, for each candidate.
- **S2 (specificity):** C − yoked C and B − yoked B on T + I.
- **S3 (beyond G1):** C − G1 A on T + I.
- **S4 (closing the gap to frozen SAC):** X − frozen SAC on T + I, non-inferiority margin
  −2.
- **S5 (robustness):** X − contact-aware on the N/G family, non-inferiority margin −2.

**Safety gates** (binding for any recommendation; the same as G1): for each candidate against
frozen SAC on each H condition, the CI lower bound is above −2 and at most 2% of base
successes are harmed.

**Descriptive:** each PRIVILEGED ceiling against contact-aware; Phase E against contact-aware.

### 5.5 What each outcome means (fixed now)

| outcome | conclusion |
|---|---|
| X supported (H1 and H2) and gates pass | X reduces transient harm without costing persistent gains: a candidate for held-out confirmation (separate approval) |
| H1 passes, H2 fails | faster detection costs persistent performance; not recommended |
| H1 fails for every X, but PRIV_release_oracle beats contact-aware | perfect invalidation would help, but no causal detector achieved it with the available evidence |
| PRIV_release_oracle also fails | stale compensation is not the bottleneck; consistent with a negative Stage 1 |
| S4 fails even for a supported X | harm is reduced, but compensation is still worse than frozen SAC on short delays; report as such |
| C supported but S2 fails | the timing of suspension is not specific to the detected change; C's benefit is non-specific |

---

## 6. Metrics and analysis

### 6.1 Outcomes
- Success rate per configuration and condition.
- **Rescued and harmed counts reported separately** against frozen SAC and against
  contact-aware, never only the net difference.

### 6.2 Detection and estimation (every configuration, every true transition)

- **Detection latency:** steps from a true transition until the compensation state is
  correct (off after an end; on with the correct d after an onset or change). Reported as
  median and p90, with "never" counted separately.
- **Alarm latency** (B, C): steps from the transition to the alarm.
- **Stale-compensation duration:** steps active with a mismatched d after each transition
  (§3.3).
- **False alarms and false releases:** alarms or suspended steps while the true delay is
  constant and nonzero, per active step.
- **False activations:** active steps while the true delay is 0 (H, and the off-phases).
- **Estimator accuracy:** share of steps with the latched d̂ equal to the true d, by family
  and by phase (before, during, after).
- **Switching:** mode changes per episode, dwell distribution, and episodes with at least 6
  switches listed individually.
- **Control overhead:** per-step `perf_counter_ns` of estimator plus detector plus
  predictor, median and p99; descriptive, with the number of concurrent processes recorded.

### 6.3 Statistical plan
- Same estimator throughout: paired cluster bootstrap over episode seeds (each seed carries
  its 3 policy episodes).
- Per-cell estimates are descriptive. Family pooling is pre-specified (§5.4).
- No subgroup is promoted to a claim after the fact.

---

## 7. Risks, limitations and expected contribution

**Risks:**
- **Stage 1 may show that stale compensation is not the main cause.** Then G2's detectors are
  aimed at the wrong target. The checkpoint after Stage 1 exists for this reason.
- **The harm may be partly a property of drop/replay pipeline semantics** (M5). If so,
  conclusions must be stated per semantics.
- **The upside on short transients is bounded.** Frozen SAC already reaches 97–98% there, so
  the best achievable result is "do no harm" (closing the 6–16 point gap), not beating SAC.
- **Noise:** at 1 cm observation noise the per-transition evidence is weak (§1.5). Detectors
  will be slower or noisier there, and calibration must not over-fit noise-free cells.
- **Calibration carries some researcher latitude** (grids for w and h). It is limited by
  fixed grids, detection-only criteria and separate seeds.

**Limitations:**
- Simulation only, with a pure transport delay of issued commands.
- Frozen SAC policies and a frozen F1 model.
- Development evidence only until a separately approved held-out run.
- The read-only analyses above use G1/F1 development logs; they motivated the design and are
  not evidence for it.

**Expected contribution:**
- A mechanism-tested account of why identification-based delay compensation can hurt under
  nonstationary delay.
- A comparison of three ways to detect change (window length, two timescales, sequential
  detection with window reset) on a pre-specified grid.
- Evidence on whether invalidation can be fast while re-identification stays conservative.

---

## 8. Recommendation

**Candidate C (sequential change detection on fresh per-transition residuals, with suspension
and window reset).**

1. **It targets the identified bottleneck directly.** The 10-step batch fit cannot move
   until about 7 transitions are post-change (§1.4). C's evidence comes from fresh
   transitions only, and the logs show one transition already carries about 100:1 evidence
   for "lag 3 is no longer valid" without noise (§1.5).
2. **It separates the easy question from the hard one.** *Invalidation* (fast, strong
   evidence) triggers suspension, which is the right action after an end. *Re-identification*
   of a new value (slow, adjacent lags are ambiguous) still passes every existing gate, but on
   post-change data only. It never trusts an unverified lag.
3. **It handles every transition type.** Ends, increases, decreases and onsets are all
   covered. B does not speed up onsets or changes of value, and A trusts short-window
   estimates directly.
4. **It is not a repackaged G1 A.** It uses the same raw information, but avoids the window
   dilution that limited A, tests every alternative lag, and estimates the change time.
5. **It leaves the frozen estimator, latch and predictors untouched,** and its overhead is a
   few vector operations per step.

**Its main risks** are false suspensions under persistent delay (the main threat to H2) and
lower power at high observation noise. Calibration criteria and the H2 non-inferiority test
address both.

**I recommend Stage 1 before any of this.** If stale compensation turns out not to be the
main cause, faster detection, including C, will not close the gap.

**Approval needed** to:
1. implement the Stage-1 diagnostics and candidates A–C;
2. run Stage 1 on the mechanism seeds;
3. checkpoint with you before calibration and evaluation.
