# Phase G2: fast detection of nonstationary actuator delays, results

## Run record

| item | value |
|---|---|
| Plan | `docs/PHASE_G2_PROPOSAL.md` (SHA-256 `799fb487…`). Stage 1 approved 2026-11-16; Stages 2–3 approved at the Stage-1 checkpoint ("proceed with all three"). |
| Recorded before use | Stage 1: A1–A6 (`docs/PHASE_G2_STAGE1_AMENDMENTS.md`). Stages 2–3: A7–A15 (`docs/PHASE_G2_STAGE3_DEFINITIONS.md`). Calibration record pinned (`docs/PHASE_G2_CALIBRATION.md`, SHA-256 `96a0fb4e…`). |
| Stage 1 | mechanism seeds 3,671,000–049; see `docs/PHASE_G2_STAGE1_RESULTS.md` |
| Stage 2 | calibration seeds 3,672,000–099; detection criteria only, success not read. A: window 7. B: fast window 4. C: Gaussian, h = 12, σ₀ = 4 mm. |
| Stage 3 | evaluation seeds **3,673,000–099** × 3 frozen SAC policies = 300 paired episodes per cell; 38 conditions × 11 configurations = **125,400 episodes**; one run |
| Frozen | SAC `fetch_push_fp32_seed{0,1,2}`, the Phase E estimator (`estimate()` unchanged), the F1 model `8a960329…` |
| Raw files | `$FARC_DATA_DIR/experiments/phase_g2_stage3/` (write-once); tables in `results/phase_g2_stage3/` |
| Held-out | **not used.** Nothing at or above 4,000,000 was touched; 4,000,300–4,000,599 remain reserved. |

The analysis is the pre-written `derive_g2`, run once. Nothing was changed after the run.

## Primary hypotheses

Each candidate X passes only if both H1 and H2 hold (an intersection-union test), with Holm
across the three candidates at one-sided α = 0.025. Differences are in success-rate points
against the existing contact-aware compensator; counts are rescued / harmed episodes.

| candidate | H1: short-lived and on-off delays (T + I, 6,600 pairs) | H2: persistent delays (P, 1,800 pairs; margin −1) | Holm p | result |
|---|---|---|---|---|
| **C** sequential detection + window reset | **+1.7** [+1.1, +2.4] (303 / 188) | +0.1 [−0.3, +0.4] (8 / 7) | < 0.001 | **supported** |
| **B** two-timescale | **+0.9** [+0.5, +1.2] (134 / 76) | 0.0 [0.0, 0.0] (0 / 0) | < 0.001 | **supported** |
| **A** short window (w = 7) | **−1.6** [−2.3, −0.9] (214 / 321) | +0.8 [−0.3, +1.8] | 1.0 | **not supported** (it harms) |

- **Practical significance** (pre-registered threshold +2 points): **neither supported effect
  reaches it.** C's +1.7 is below the threshold, though its CI includes it; B's +0.9 is
  well below.
- **Safety gates:** all three candidates passed on undisturbed and 1 cm-noise episodes. They
  harmed none of the 300 and 286 base successes there.

## Secondary hypotheses (Holm across 12 tests)

| test | comparison | family | estimate [95% CI] | result |
|---|---|---|---|---|
| S1 changes between nonzero delays | C − contact-aware | S | **+3.1** [+1.4, +4.8] (74 / 37) | supported |
| | A − contact-aware | S | +3.0 [+1.4, +4.6] (74 / 38) | supported |
| | B − contact-aware | S | +0.2 [−1.2, +1.8] | not supported |
| S2 specificity | C − yoked C | T + I | +0.5 [−0.0, +1.1] (244 / 209) | **not supported** |
| | B − yoked B | T + I | +0.2 [−0.3, +0.6] | not supported |
| S3 beyond G1 | C − G1 supervisor A | T + I | **+1.7** [+1.0, +2.3] | supported |
| S4 against frozen SAC (margin −2) | C − frozen SAC | T + I | −1.2 [−1.7, −0.6] | supported (non-inferior within 2) |
| | B − frozen SAC | T + I | −2.0 [−2.6, −1.5] | not supported |
| | A − frozen SAC | T + I | −4.5 [−5.3, −3.8] | not supported |
| S5 noise and gain loss (margin −2) | C − contact-aware | NG | +0.6 [−0.3, +1.5] | supported |
| | B − contact-aware | NG | +0.4 [−0.4, +1.2] | supported |
| | A − contact-aware | NG | −0.2 [−1.7, +1.3] | not supported |

### Specificity: what the failed check does and does not show

Per the pre-registered reading (§5.5), "C supported but S2 fails" means **C's benefit is not
shown to be specific** to the change it detects in each episode. One property of the sham
limits how far that conclusion goes:
- The yoked sham copies C's suspension steps from *another episode of the same condition*.
- Every episode of a condition has the same true transition times.
- So the sham inherits correctly *timed* suspensions; only the episode-specific detection is
  removed.
- The sham itself gains +1.2 [+0.7, +1.7] over contact-aware on T + I.

**What the data therefore support:**
- Suspending compensation around the times delays change explains most of C's gain.
- C's episode-specific detection adds at most about half a point beyond schedule-aligned
  timing (CI includes 0).

A deployed controller cannot copy timing from another episode, so C's gain over
contact-aware is real. But the claim that its *per-episode detection* is what matters is not
supported. This is the same design weakness as E2's shuffled sham. A sham with correct
*counts* but *random* timing would have been the stronger control.

## How far the candidates get toward the privileged ceilings (T + I)

| configuration | vs contact-aware | vs frozen SAC |
|---|---|---|
| PRIVILEGED true delay and gain | **+5.0** [+4.3, +5.8] | **+2.1** [+1.4, +2.8] |
| PRIVILEGED release exactly at every true end | +3.5 [+3.0, +4.0] | +0.6 [+0.3, +0.8] |
| **C** | **+1.7** [+1.1, +2.4] | −1.2 [−1.7, −0.6] |
| B | +0.9 | −2.0 |
| Phase E predictor (reference) | +0.7 | −2.2 |
| G1 supervisor A | +0.1 [−0.2, +0.3] | −2.9 |
| contact-aware | – | −2.9 [−3.5, −2.4] |
| A | −1.6 | −4.5 |

- **C captures about a third** of what perfect identification would gain (1.7 of 5.0), and
  about half of what perfect *release* would gain (1.7 of 3.5).
- **Perfect identification would beat frozen SAC** even on short-lived delays (+2.1). No
  causal candidate gets there.

## Success by family

| family | frozen SAC | contact-aware | A | B | **C** | true delay (PRIV) |
|---|---|---|---|---|---|---|
| P persistent (6 cells) | 50.6 | 96.1 | 96.8 | 96.1 | 96.1 | 96.6 |
| T short-lived pulse (18) | 97.7 | 96.8 | 96.8 | 97.4 | **97.9** | 98.9 |
| I on-off (4) | 86.5 | 74.2 | 65.2 | 76.2 | **79.1** | 92.7 |
| S change of value (4) | 36.4 | 90.2 | 93.2 | 90.5 | **93.3** | 96.9 |
| NG noise / gain loss (4) | 57.6 | 90.9 | 90.8 | 91.3 | 91.5 | 92.2 |
| H no delay (2) | 97.7 | 97.7 | 97.7 | 97.7 | 97.7 | 97.7 |

- **On single short-lived pulses** C is at frozen SAC's level (97.9 against 97.7).
- **On-off delays remain the unsolved case.** C recovers 5 of the 12 points contact-aware
  loses against SAC, but stays 7 points below SAC and 14 below the true-delay ceiling.

## Detection and switching

Averages over all transitions:

| configuration | onset: steps to correct | end: mismatched steps | change: steps to correct | median alarm latency |
|---|---|---|---|---|
| contact-aware | 5.3 (8,072 of 12,000 never correct before the next transition) | 3.0 | 9.0 | – |
| G1 supervisor A | 5.3 | 2.7 | 9.0 | – |
| A | **3.8** | **3.9** (more stale or false activity) | 7.0 | – |
| B | 5.8 | 1.8 | 10.0 | about 4–5 |
| **C** | **4.0** | 2.2 | **4.0** | **about 1** |
| PRIVILEGED true delay | 0 | 0 | 0 | – |

"Never correct" counts mostly reflect short pulses that end before any estimator can
identify them.

- **C is fastest at detection:** alarms about 1 step after a transition. It re-identifies a
  changed value in 4 steps instead of 9, and engages after onsets in 4 instead of 5.3.
- **A reacts faster at onsets, but at a cost:**
  - false activations of 48% of no-delay steps in the on-off cells (contact-aware 33%,
    C 26%) and 10.5% in the T cells (contact-aware 8.9%);
  - more wrong-lag activity, which is what loses its H1.
- **False releases under persistent delay:** 0.5% of engaged steps for B and C, 0.1% for A,
  0% for contact-aware. This is consistent with H2 holding.
- **False activations with no delay** (H family): A was active on 69 of 30,000 steps (0.2%);
  B and C on 5 each.
  - C raised 123 alarms in 116 of the 600 no-delay episodes. They only reset the
    identification window and, through the gates, led to no extra activation.
  - So C's onset path fires spuriously but harmlessly.
- **Switching:** mean mode changes per episode were 1.05 (contact-aware), 1.31 (B), 1.43 (A)
  and 1.97 (C). The true-delay ceiling has 2.14, since the delays themselves switch.
  - Episodes with at least 6 changes: 574 for C and 1,200 for the ceiling, all in the on-off
    cells.
  - C switches more because it tracks the schedule more closely, not because it chatters.
- **Overhead** (per-step controller time including the estimator, 9 concurrent processes,
  descriptive): median 82 µs for contact-aware, 116 µs for C, 136 µs for B and 137 µs for G1
  A. All are far below the 40 ms control period.

## Conclusions

1. **Candidate C is the best of the three, and it is supported,** but its effect is modest.
   - **Short-lived and on-off delays:** +1.7 points over contact-aware without cost under
     persistent delay (+0.1), below the +2 practical-significance mark.
   - **Changes of value:** +3.1.
   - **Against G1's supervisor A:** +1.7.
   - **Against frozen SAC on short-lived and on-off delays:** within 2 points (−1.2).
2. **B is supported but smaller** (+0.9): it only speeds up the end of a delay, as Stage 1
   predicted.
3. **A is not supported.** A shorter window speeds up onsets but produces more false and
   wrong-lag compensation, a net loss on short-lived and on-off delays (−1.6).
4. **The benefit is not shown to come from per-episode detection** (S2). A sham with
   schedule-aligned suspension times gets most of it, though that sham is weaker than ideal.
5. **The remaining gap is identification speed, not the model.** Perfect identification adds
   +5.0 over contact-aware and beats frozen SAC by +2.1; C reaches a third of that. On-off
   delays are where most of the remaining loss is.

## Limits

- **Development seeds only.** Every G2 claim needs a separately approved held-out
  confirmation (4,000,300+ reserved). The grid's conditions were designed by me, after G1
  and Stage 1 informed the design.
- **Calibration put C at the edge of its grid** (the largest h and σ₀). A wider grid might
  have found a better setting; that would now be tuning on evaluation results and is not
  done.
- **The yoked sham was weaker than intended** (see Specificity).
- **Simulation only:** a pure transport delay with drop/replay semantics (Stage 1 found no
  dependence on that choice), and frozen SAC and F1 components.

## Recommendation

- **C** is a supported, deployable improvement, with no measured cost under persistent delay,
  noise, gain loss or no delay. Its effect is below the practical-significance threshold on
  short-lived delays, and its mechanism-specificity is unproven.
- **Before a held-out confirmation** (which would need your approval), the specificity
  question deserves a cheap development check: a random-timing sham with C's suspension
  counts.
- **On-off delays are the open problem.** Closing that gap needs faster identification than
  any residual test on this estimator gives, or a policy that does not need compensation for
  short delays at all.

## Addendum: exploratory random-timing sham (after the run)

Added after the Stage-3 analysis, on the user's request; nothing above was changed. On the
exploratory seeds 3,674,000–099, C beat a sham with the same suspension counts at random times
by +1.6 [+0.8, +2.4] on T + I. Random-timed suspension gave nothing over contact-aware (+0.2
[−0.2, +0.6]). So the timing of C's suspensions matters; how much of it comes from per-episode
detection rather than schedule-aligned timing is still open. See
`docs/PHASE_G2_SHAM_CHECK_RESULTS.md`.
