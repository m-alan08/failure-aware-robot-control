# G2 Stage 1: mechanism isolation, results

## Run record

| item | value |
|---|---|
| Plan | `docs/PHASE_G2_PROPOSAL.md` §3 (Stage 1 approved 2026-11-16), SHA-256 `799fb487…` |
| Recorded before the run | `docs/PHASE_G2_STAGE1_AMENDMENTS.md` (A1–A6), commit `c58d6dc` |
| Seeds | mechanism block **3,671,000–049** × 3 frozen SAC policies = 150 paired episodes per cell. Debugging seeds 3,670,000–009 were used for code checks only. |
| Size | 19,500 episodes + 432 branch rollouts (27 target episodes × 16 switch points); one run, 211 s |
| Raw files | `$FARC_DATA_DIR/experiments/phase_g2_stage1/` (write-once); tables in `results/phase_g2_stage1/` |
| Held-out | **not used.** Nothing at or above 4,000,000 was touched. |
| Frozen | SAC `fetch_push_fp32_seed{0,1,2}`, the Phase E estimator, the F1 model `8a960329…` |

Configurations marked PRIVILEGED read the true delay or the simulator. They are diagnostics,
not controllers.

## Pre-registered decision rules (§3.4)

| rule | result | verdict |
|---|---|---|
| Release exactly at the true end (k = 0) recovers more than half of the gap between deployed contact-aware and frozen SAC | recovered share **1.39** [1.00, 2.31] | met |
| Success falls as the release offset k grows | slope **−0.21 points per step** [−0.36, −0.06] | met |
| **"Stale compensation is the main cause"** | both conditions met | **yes** |
| Drain/hold semantics change contact-aware's harm against SAC by more than 2 points | change **0.0** [−2.1, +2.0] | **not flagged**: conclusions do not depend on drop/replay |

**Important qualification:** on this pre-registered grid (delay 3, onsets 6 and 18, durations
4, 8 and 16), deployed contact-aware is only **2.0 points** below frozen SAC (−2.0 [−3.0, −1.0];
9 rescued, 27 harmed). That is much smaller than G1's T1–T3 (6–16 points).
- **Why:** G1's conditions started at step 0, where the start-up cue makes the delay
  engage within 4 steps, and lasted 15 steps. Here, mid-episode onsets take longer to
  identify, so short pulses often end before compensation ever engaged. In 965 of 2,100 ends
  there was never a mismatch.
- **Consequence:** the "share" above is a ratio of small differences. The mechanism is clear,
  but the absolute harm on single pulses of this grid is modest.

## M1: stale duration, dose-response (PRIVILEGED release)

Transient grid, pooled (900 paired episodes). Success, and the difference against frozen SAC
and against deployed contact-aware:

| compensation forced off at | success | vs frozen SAC | vs deployed |
|---|---|---|---|
| true end (k = 0) | **97.4%** | +0.8 [0.0, +1.7] (10 / 3) | **+2.8** [+1.8, +3.8] (29 / 4) |
| k = 1 | 96.4% | −0.2 | +1.8 [+0.9, +2.8] |
| k = 2 | 96.4% | −0.2 | +1.8 [+0.8, +2.8] |
| k = 3 | 96.2% | −0.4 | +1.6 [+0.8, +2.3] |
| k = 5 | 96.0% | −0.7 | +1.3 [+0.4, +2.2] |
| k = 7 | 95.7% | −1.0 | +1.0 [+0.3, +1.8] |
| deployed (estimator, about 5–9 stale steps) | 94.7% | **−2.0** [−3.0, −1.0] (9 / 27) | – |
| never release | 84.0% | **−12.7** [−15.2, −10.1] (6 / 120) | −10.7 |
| frozen SAC | 96.7% | – | – |

- **Perfect release removes all of the harm** (k = 0 is +0.8 against SAC).
- **Harm grows with every stale step,** and is large when compensation never releases.
- **The first stale step alone accounts for about a third of the recoverable gain** (k = 0
  against k = 1: 2.8 against 1.8). At the end step itself, the 3 queued commands the
  predictor relies on are dropped by the pipeline.

**Point of no return (branching).** 27 episodes were successes for frozen SAC and failures for
deployed contact-aware. Each was replayed with PRIVILEGED true-delay compensation switched in
from step end + j, for j = 0 … 15.
- **25 of 27 are rescued by switching exactly at the end.**
- **The last rescuing offset** has a median of **7 steps** (IQR 3–8; range 0–9). One episode
  is not rescued by any branch.
- **Required speed:** a detector acting within 2 steps would have rescued 25 of the 26
  rescuable episodes; within 3 steps, 21; within 7, 14. The deployed estimator takes about
  9.

## M2: identification vs predictor

| | transient grid vs frozen SAC | persistent delay 3 from step 6: success |
|---|---|---|
| deployed: estimated delay + learned | −2.0 [−3.0, −1.0] | 86.7% |
| deployed: estimated delay + Phase E | −0.8 [−2.0, +0.3] | 76.0% |
| PRIVILEGED: true delay + learned | **+0.6** [−1.0, +2.1] | **95.3%** |
| PRIVILEGED: true delay + Phase E | +0.3 [−0.9, +1.7] | 80.0% |
| PRIVILEGED: true delay + simulator | +1.3 [+0.2, +2.6] | 98.7% |
| PRIVILEGED: estimated delay + simulator (G1's oracle) | −4.9 [−6.6, −3.3] | 92.7% |
| frozen SAC | – | 24.7% |

**Transients:**
- With the true delay, no predictor harms transient episodes, so the transient harm is an
  *identification* problem, not a model problem.
- With the estimated delay, Phase E harms less than the learned model (+1.2 [+0.1, +2.3]
  between them).

**Persistent delay with a mid-episode onset — a second, larger prize that was not
anticipated:**
- True-delay learned compensation beats deployed by **+8.7** [+3.3, +14.7] (16 rescued,
  3 harmed).
- The deployed controller runs uncompensated for about 7.8 steps after each onset (mean
  mismatched steps per onset).
- So onset latency costs as much as stale release on these conditions.

## M3: prediction error under a stale delay

At stale steps (true delay 0, compensation still running with d̂ = 3), the correct prediction
is the *current* state. Instead, the deployed predictors shift it:
- **gripper:** by 44.5 mm (learned) and 51.3 mm (Phase E);
- **cube:** by 16.8 mm (learned). Phase E keeps the cube static.

This is the double-counting of already-executed commands (E2 in the proposal), about 4–5 cm
of fictitious lead. On these conditions the learned model's extra cube motion was not shown
to be more harmful: Phase E's stale gripper error is slightly larger, and it harmed less
overall.

## M4: switching with no delay transition

- **Persistent delay 3, true-delay learned compensation, forced off for L steps at step 18:**
  - L = 3: −4.0 [−8.7, 0.0] (3 / 9);
  - L = 6: −1.3 [−4.7, +2.0] against uninterrupted true-delay compensation.
  - Weak evidence that briefly dropping compensation mid-delay costs a little; there is no
    sign of a large switching penalty.
- **Undisturbed, compensation forced on with d = 3 for L steps** (stale-like compensation with
  no pipeline event):
  - L = 3: −0.7;
  - L = 6: −2.0 [−4.7, 0.0];
  - L = 9: −2.0 [−4.7, 0.0] against frozen SAC (0–1 rescued, 2–3 harmed).
  - Small harm that grows with L. It is consistent with M1, without any dropped or replayed
    commands.

## M5: pipeline semantics

Under drain/hold, contact-aware's harm against SAC is identical (−2.0 [−3.6, −0.7]), and
true-delay compensation is again harmless (+0.8). Frozen SAC itself does a little worse under
drain/hold (95.0% against 96.7%). The mechanism does **not** depend on drop/replay semantics.

## M6: the G1 oracle anomaly, explained

Intermittent delay 3 (on 6–15, off 16–25, on 26–35, off after):

| configuration | success | vs frozen SAC |
|---|---|---|
| frozen SAC | 92.7% | – |
| deployed contact-aware | **58.0%** | **−34.7** [−44.0, −25.3] (5 / 57) |
| PRIVILEGED simulator, latched d̂ (G1's oracle) | **27.3%** | −65.3 |
| PRIVILEGED learned, true delay | 97.3% | +4.7 [+0.7, +9.3] |
| PRIVILEGED simulator, true delay | 98.7% | +6.0 |
| PRIVILEGED simulator, schedule horizon | 98.7% | +6.0 |

- **Placeholder execution:** with the latched d̂, the simulator look-ahead executes its
  placeholder actions on **92% of active steps** in the intermittent condition and 81% on
  the transient grid (2.2–2.5 per look-ahead). With the true delay it does so on 20–21% of
  active steps, all at transitions. With the schedule horizon it never does.
- **Conclusion:** G1's oracle collapse is a stale or wrong delay combined with look-ahead
  semantics, not a simulator-oracle bug. With the right delay, the simulator predictor is the
  best configuration everywhere.
- **Intermittent delays are where the deployed system fails badly** (−35 points). With the
  true delay it would *gain* about 5 points over SAC. The about 8-step identification latency
  in *both* directions is the cause.

## What Stage 1 establishes (for the G2 decision)

1. **Stale compensation after a delay ends is a cause of the transient harm.**
   - Dose-response holds in k; perfect release removes it.
   - True-delay compensation is harmless on every transient condition.
   - It does not depend on pipeline semantics.
   - On single pulses with mid-episode onsets the harm is modest (2 points). On intermittent
     delays it is large (35 points).
2. **Onset latency matters as much as release latency.** On a mid-episode onset of a
   persistent delay, perfect identification adds 8.7 points over the deployed controller. A
   detector that only speeds up the *end* of a delay (candidate B) leaves this on the table.
3. **A detector must act within about 2–3 steps of a transition** to rescue most harmed
   episodes. The current estimator takes about 8–9 steps.
4. **The G1 oracle anomaly is explained**, and it was not a code bug.

## Limits

- These are mechanism-isolation results on development seeds, with privileged diagnostics.
  They establish mechanisms, not deployable gains.
- One delay magnitude (3) and a small grid were used. The point-of-no-return analysis covers
  27 episodes.
- The M4 and M5 effects are small, with wide CIs (150 episodes per cell).
