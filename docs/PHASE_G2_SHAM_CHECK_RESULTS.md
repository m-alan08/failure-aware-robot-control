# G2 exploratory sham check: results

**Exploratory and post hoc.** Plan: `docs/PHASE_G2_SHAM_CHECK.md`, committed (38c22aa) before
any exploratory seed was used. Seeds 3,674,000–099 × 3 frozen SAC policies; the 22 T + I cells;
26,400 episodes, one run. Raw files: `$FARC_DATA_DIR/experiments/phase_g2_stage3_sham_check/`
(write-once); tables in `results/phase_g2_sham_check/`. No held-out seed was used. Stage-3
results and conclusions are unchanged.

## Results (T + I, success-rate points, 6,600 paired episodes; counts are rescued / harmed)

| | comparison | estimate [95% CI] | counts |
|---|---|---|---|
| **E1** | **C − random-timing sham** | **+1.6 [+0.8, +2.4]**, one-sided p < 0.001 | 309 / 204 |
| | on single pulses (T) | +0.9 [+0.4, +1.5] | 109 / 58 |
| | on on-off delays (I) | +4.5 [+1.3, +7.7] | 200 / 146 |
| E2 | yoked sham − random-timing sham | +0.8 [+0.4, +1.3] | 157 / 102 |
| E3 | random-timing sham − contact-aware | +0.2 [−0.2, +0.6] | 93 / 81 |
| E4 | C − yoked sham | +0.8 [+0.1, +1.5] | 268 / 218 |
| E5 | C − contact-aware | +1.8 [+1.0, +2.5] | 326 / 209 |

All three shams suspend exactly as many steps as C (20,844 in total, 3.2 per episode).

| success (%) | contact-aware | random sham | yoked sham | C |
|---|---|---|---|---|
| T single pulses | 96.4 | 96.3 | 96.6 | 97.3 |
| I on-off | 72.8 | 74.0 | 77.3 | 78.5 |

## Reading (as pre-recorded)

- **E1 passes: the timing of C's suspensions matters.** The same amount of suspension at a random
  time gives nothing measurable over contact-aware (E3, +0.2, CI includes 0). C's gain is not a
  generic effect of switching compensation off more often.
- **E5 replicates Stage 3's H1** on fresh seeds (+1.8 against +1.7).
- **How the gain splits** (descriptive, additive by construction):
  1.8 = 0.2 (untimed suspension) + 0.8 (schedule-aligned timing, E2) + 0.8 (C beyond the yoked
  sham, E4).

## What it does not show

- **E4 is descriptive**, not a test. Its CI excludes 0 on these seeds, while Stage 3's S2
  (+0.5 [−0.0, +1.1]) did not. Both point the same way; neither is a confirmed claim.
- **E4 is not purely per-episode detection.** The yoked sham only switches compensation off; C
  also resets its identification window at each alarm. C compensates on 6.8% of delayed steps
  in the T cells against 3.4% for contact-aware and the yoked sham, so faster re-identification
  is part of what E4 measures.
- **Schedule-aligned timing versus detection** still cannot be separated cleanly: every episode
  of a condition has the same transition times. That needs conditions whose onset times vary
  between episodes.

## Updated summary of G2 on development seeds

C's improvement is real, and it depends on suspending at the right times. Untimed suspension of
the same amount does nothing. The remaining open question for a held-out confirmation is how
much comes from per-episode detection and window reset, as opposed to timing a fixed schedule
would supply. A grid with randomized onset times would answer it.
