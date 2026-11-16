# Phase G1: supervisory switching, results

## Run record

| item | value |
|---|---|
| Plan | `docs/PHASE_G1_PROPOSAL.md`, approved 2026-11-16 (commit `6d328fb`), SHA-256 `cf152f04…` |
| Code | `5191c16`, clean tree |
| Calibration | seeds 3,661,000–049, C1–C4, never-releasing trace only; success not written. **h = 0.5** (committed before the evaluation) |
| Evaluation | seeds **3,662,000–099** × 3 frozen SAC policies = 300 paired episodes per cell; 13 conditions × 11 configurations = **42,900 episodes**; one run, 489 s |
| Frozen | SAC `fetch_push_fp32_seed{0,1,2}`; Phase E estimator constants; F1 model `8a960329…` |
| Raw files | `$FARC_DATA_DIR/experiments/phase_g1_supervision/` (write-once); tables in `results/phase_g1_supervision/` |
| Held-out | **not used.** Nothing at or above 4,000,000 was touched. |

The analysis is the pre-written `derive_g1`, run once. Nothing was changed after the run.

## Pre-registered hypotheses

One-sided α = 0.025 with Holm adjustment across G1–G4; paired episode-seed cluster bootstrap,
10,000 resamples. Differences are in success-rate points.

| | hypothesis | comparison | conditions | estimate [95% CI] | Holm p | result |
|---|---|---|---|---|---|---|
| G1 | A rescues stale-latch episodes | A − contact-aware | T1–T4 (1,200) | **+1.4** [+0.7, +2.2] | 0.001 | **supported** |
| G2 | A costs nothing under constant delay | A − contact-aware | C1–C7 (2,100) | **0.0** [0.0, 0.0] | < 0.001 | **supported** (margin −1) |
| G3 | B improves on contact-aware | B − contact-aware | C1–C7 + T1–T4 (3,300) | −0.4 [−1.1, +0.4] | 1.0 | **not supported** |
| G4 | C keeps success, cuts learned-model calls ≥ 30% | C − contact-aware | C1–C7 | **−3.4** [−4.8, −2.0]; calls −57% | 1.0 | **not supported** (harms) |

**Safety gates:** every supervisor passed on C0 and C8. No base success was harmed in either
condition, and CI lower bounds were 0.

### Specificity checks (secondary)

| check | comparison | conditions | estimate [95% CI] | rescued / harmed |
|---|---|---|---|---|
| S-A1 | A − yoked A | T1–T4 | **+1.9** [+0.9, +3.0] | 30 / 7 |
| S-A2 | A − timeout | T1–T4 | +1.2 [+0.3, +2.0] | 21 / 7 |
| S-A2 | A − timeout | C1–C7 | +0.3 [0.0, +0.7] | 8 / 1 |
| S-B | B − yoked B | C1–C7 + T1–T4 | +0.7 [−0.2, +1.6] | 113 / 91 |
| S-BC | B − C | C1–C7 + T1–T4 | +1.1 [+0.1, +2.2] | 203 / 167 |
| AO | contact-aware − never-release | T1–T4 | **+9.2** [+7.0, +11.3] | 140 / 30 |

## Success rates

| condition | frozen SAC | Phase E | contact-aware | never-release | **A** | A-timeout | **B** | **C** | oracle |
|---|---|---|---|---|---|---|---|---|---|
| C0 undisturbed | 99.7 | 99.7 | 99.7 | 99.7 | 99.7 | 99.7 | 99.7 | 99.7 | 99.7 |
| C1 delay 1 | 83.0 | 97.7 | 99.3 | 99.3 | 99.3 | 99.3 | 99.3 | 99.3 | 100.0 |
| C2 delay 2 | 31.3 | 89.7 | 98.3 | 98.3 | 98.3 | 98.0 | 97.7 | 96.0 | 99.3 |
| C3 delay 3 | 16.7 | 90.3 | 97.0 | 97.0 | 97.0 | 95.7 | 95.7 | 90.3 | 99.3 |
| C4 delay 4 | 10.0 | 80.0 | 94.0 | 94.0 | 94.0 | 94.0 | 92.7 | 86.7 | 99.0 |
| C5 delay 3 from 15 | 73.0 | 88.3 | 94.3 | 94.3 | 94.3 | 94.3 | 93.7 | 93.7 | 98.0 |
| C6 delay 3 + noise 0.005 | 13.7 | 82.0 | 94.7 | 94.7 | 94.7 | 93.7 | 95.0 | 87.7 | 99.7 |
| C7 delay 3 + scale 0.5 | 59.7 | 81.7 | 92.7 | 92.7 | 92.7 | 93.0 | 91.0 | 93.0 | 91.7 |
| C8 noise 0.01 | 95.3 | 95.7 | 95.7 | 95.3 | 95.7 | 95.7 | 95.7 | 95.7 | 95.7 |
| T1 delay 3, steps 0–14 | **98.3** | 96.0 | 91.7 | 69.3 | 92.3 | 91.7 | 92.7 | 95.3 | 92.7 |
| T2 delay 3, steps 15–29 | **97.7** | 95.3 | 91.3 | 83.0 | 92.3 | 91.3 | 90.7 | 95.0 | 91.7 |
| T3 delay 3, steps 0–9 and 20–29 | **96.7** | 87.0 | 80.3 | 74.3 | 84.3 | 81.0 | 82.3 | 82.3 | 59.3 |
| T4 delay 1 → 3 at 20 | 48.0 | 88.0 | 97.3 | 97.3 | 97.3 | 97.7 | 96.3 | 95.7 | 99.0 |

**Supervisor A vs contact-aware, by condition:** the same outcome in every one of the 2,700
episodes of C0–C8. A released rarely there (§ Detection), and never changed an outcome. On the transients: T1 +0.7 (3/1), T2 +1.0 (4/1), **T3 +4.0
[+1.7, +6.7] (14/2)**, T4 0.0.

## The main negative finding: on transient delays, frozen SAC beats every compensator

Compensator vs frozen SAC (points [95% CI], rescued / harmed):

| condition | Phase E | contact-aware | A | C |
|---|---|---|---|---|
| T1 | −2.3 [−5.0, 0.0] | −6.7 [−9.7, −3.7] | −6.0 [−9.0, −3.3] | −3.0 [−6.3, 0.0] |
| T2 | −2.3 [−4.7, 0.0] | −6.3 [−9.7, −3.0] | −5.3 [−8.3, −2.3] | −2.7 [−5.0, −0.3] |
| T3 | −9.7 [−14.0, −5.7] | −16.3 [−21.0, −11.7] | −12.3 [−16.7, −7.7] | −14.3 [−19.0, −9.7] |
| T4 (delay never ends) | +40.0 | +49.3 | +49.3 | +47.7 |

- **Frozen SAC tolerates a delay that lasts 10–15 steps:** 97–98% success on T1–T3.
- **Every compensator loses episodes on those conditions.** The likely mechanism is the
  stale estimate after the delay ends:
  - release takes a median of 9 steps (p90 10) for every configuration, which is the
    10-step window filling with the new regime;
  - during those steps, compensation runs with d̂ = 3 while the true delay is 0. That is
    25–29% of delay-free steps on T1–T2 and about 60% on T3.
  - `never_release`, which never drops the stale estimate, loses 22 points on T1.
- **Supervisor A narrows the gap but does not close it.** A does **not** release faster in
  the median (9 steps). What it changes is the share of episodes that never release:
  - T1: 10 → 2;
  - T2: 49 → 17;
  - T3: 24 → 15 events.
  - False-active steps after a delay ends fall by 7–17% (T1 2,965 → 2,595; T2
    3,019 → 2,520; T3 5,624 → 5,249).
- **Mechanism is suggested, not tested.** The learned predictor is hurt more than Phase E by a
  stale d̂ (T1: −6.7 vs −2.3). A plausible but untested mechanism is that it predicts a
  confident 3-step cube motion that never happens.
- **The privileged oracle is not an upper bound here** (T3: 59.3%). It simulates d̂ steps
  ahead with the *latched* d̂, executing placeholder actions. When d̂ is stale and the true
  delay is 0, those placeholders actually take effect in the look-ahead, so it predicts a
  future that will not happen. It is privileged about dynamics, not about the delay.

This was not a hypothesis. It is reported because it is the most decision-relevant result.
If transient delays occur in practice, the compensator should not be deployed as-is, with or
without A. What would be needed is faster end-of-delay detection (a shorter window or a
residual-based detector), which changes the frozen estimator and so belongs to a new
pre-registered phase.

## Detection, switching and oscillation

- **Detection latency** (onset to the correct d̂) is unchanged by any supervisor:
  - 4 steps at a start-of-episode onset;
  - 9–10 steps mid-episode (C5, T2), with 13 of 300 C5 episodes never engaging.
- **False activations** (compensation on while the true delay is 0) on C0 and C8: 1–25 steps
  per 15,000 for every configuration except `never_release` (70 and 46).
- **False releases under constant delay:**
  - A switched compensation off on 35 of 89,332 post-engagement steps in C1–C7 (0.04%).
    Contact-aware's own latch did so on 14. None of A's releases changed an episode's
    outcome. The calibration margin was thin: the largest statistic on the calibration
    traces was 0.32, against h = 0.5.
  - The timeout ablation switched compensation off on 8.0% of those steps, and lost episodes
    on C3 (−1.3) and C6 (−1.0). So A's evidence requirement is what keeps it nearly free
    under constant delay.

**Switching per episode, and oscillation flags (≥ 6 switches):**

| configuration | mean switches | max | flagged episodes |
|---|---|---|---|
| contact-aware | 1.2 | 5 | 0 |
| A | 1.2 | 6 | 1 |
| A-timeout | 1.5 | 5 | 0 |
| B | 2.1 | 7 | 70 |
| yoked B | 2.2 | 8 | 103 |
| C | 3.3 | 9 | **651** (of 3,900) |
| yoked A | 1.3 | 8 | 44 |

- **The minimum dwell held:** no supervisor-initiated switch came sooner than 5 steps.
- **C chattered up to 9 times per episode**, which is at the dwell's bound of 10. It switched
  most on long constant delays and on T3, as the gripper repeatedly crossed the 5–6 cm band.
- **C's loss is concentrated in the conditions where it switched most:** C3 −6.7, C4 −7.3,
  C6 −7.0. Switching predictors mid-push plausibly hands the policy inconsistent predicted
  states. That is suggested by the pattern, not tested.

## Prediction errors seen by supervisor B (closed loop, realised)

Mean of gripper + cube position error, per scored step:

| | learned | Phase E |
|---|---|---|
| near the cube (114,914 scored steps in total) | 18.0 mm | 24.3 mm |
| free space | 21.8 mm | 33.3 mm |

- **My prior expectation was wrong.** I expected B to pick Phase E in free space because of
  the learned model's cube drift. The score includes gripper error, where the learned model
  is much better, so the learned model wins in both regimes.
- **B still used Phase E on 17% of steps**, mostly from noisy single-episode swings. Those
  swings neither helped nor hurt beyond its yoked sham (S-B +0.7, CI includes 0).

## Computation (descriptive)

Per-step controller overhead, excluding the environment step and the base SAC pass. These are
wall times with 9 processes running at once, so they are noisy.

| configuration | median µs | p99 µs | learned rollouts per episode |
|---|---|---|---|
| frozen SAC | 0 | 0 | 0 |
| Phase E | 139 | 306 | 0 |
| contact-aware | 261 | 606 | 31.2 |
| A | 306 | 688 | 30.9 |
| B | 291 | 658 | 31.2 |
| C | 149 | 532 | 12.5 (−57%) |

- **Practically negligible:** everything is under 1% of the 40 ms control period, as §1.4
  of the plan predicted.
- **A adds about 45 µs** for recomputing the window residuals.
- **C's 57% call reduction** is real but buys nothing practical, and it costs success (G4).

## Notes on how the plan was implemented (no change to any criterion)

1. **Holm** was applied to one-sided bootstrap p-values, P(draw ≤ bound), from the same
   cluster bootstrap as the CIs. For every hypothesis the Holm decision agrees with the
   unadjusted "CI lower bound > bound" rule.
2. **"False-release rate ≤ 1% of active steps"** was implemented as the share of active steps
   that a release would have switched off, simulated on the never-releasing calibration
   traces, including the dwell and the wait for a fresh estimate. This is the more
   conservative of the two readings: it counts every off step rather than only the release
   events. Both readings give 0 at every grid value, so h = 0.5 under either.
3. **One extra check during debugging.** The debug run's oscillation table had 72 flagged
   episodes, so I counted flags per configuration to rule out a bug in the switch counter.
   The flags came from B and C, and none from configurations that cannot oscillate.
   - This was a code check on debugging seeds only; nothing was changed as a result.
   - It did show that C switches a lot before the evaluation ran.
4. **The switching table's `min_dwell` column** reports the shortest mode run, including runs
   truncated by the episode end and switches made by the unchanged latch. Values of 1 are
   therefore not dwell violations.

## Conclusions

- **A (evidence-based release) is supported and specific, but small.**
  - +1.4 points on transient delays: it beats its yoked sham (+1.9) and the timeout (+1.2).
  - It costs nothing elsewhere: the same outcome as contact-aware on every constant-delay,
    undisturbed and noise-only episode, with releases on 0.04% of steps under constant
    delay.
  - It is safe to add, but it does not fix the real problem on transients.
- **B (realised-error selection) is not supported.** The error signal mostly confirms that
  the learned model is the better predictor, so switching away from it does not help.
- **C (contact gate) is not supported, and it harms.**
  - −3.4 points under constant delay, with frequent mode chatter.
  - Its 57% compute saving is irrelevant at 0.6% of the control period.
- **Safety and false activations:** no supervisor changed anything on undisturbed or
  noise-only episodes, where the existing estimator gate already does the job.
- **The decision-relevant finding is negative and unplanned:** on delays that end after 10–15
  steps, *every* compensator, supervised or not, is 2–16 points worse than frozen SAC. The
  existing release (vs never-release, +9.2) and A (+1.4) only reduce that loss.

## Limits

- **Development only:** fresh development seeds, but conditions T1–T4 were designed by me. A
  claim about A would need a separately approved held-out run on 4,000,300+.
- **Simulation only:** a pure transport delay of issued commands, MuJoCo contact, frozen
  components.
- **Mechanisms are not tested:** stale prediction for the transient loss, and mid-push
  predictor switching for C's loss are both inferred from patterns.
- **Timing is from a loaded machine** and is descriptive only.

## Recommendation

- **Do not adopt B or C.**
- **A could be kept as a zero-cost safeguard,** but only after held-out confirmation, which
  needs your approval.
- **Before any deployment claim covering transient delays,** a new pre-registered phase should
  test faster end-of-delay detection against frozen SAC as the baseline to beat. For example:
  - a shorter estimator window;
  - a residual-based detector;
  - not compensating until a delay has persisted for a minimum time.

  The estimator would no longer be frozen in that phase, so it must not be tuned on these
  results.
