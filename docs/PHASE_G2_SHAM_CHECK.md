# G2 exploratory check: does the timing of C's suspensions matter?

**Status:** written and committed before any exploratory seed (3,674,000–099) was used.
Approved by the user on 2026-11-16 ("Run sham check") after the Stage-3 results. This check is
**exploratory and post hoc**. It does not change any Stage-3 result or conclusion, and it is
not a held-out confirmation. No seed at or above 4,000,000 is used.

## Why

Stage 3's specificity test (S2) compared C with a sham that copies C's suspension steps from
another episode of the same condition. Every episode of a condition has the same transition
times, so that sham inherited correctly timed suspensions, and S2 could not tell whether
*when* C suspends matters. This check adds the control S2 should have had: the same number and
pattern of suspended steps, at a random time.

## Design

| item | value |
|---|---|
| Seeds | exploratory block 3,674,000–099 × 3 frozen SAC policies = 300 paired episodes per cell |
| Cells | the 22 Stage-3 T + I cells (18 single pulses, 4 on-off schedules), unchanged |
| Configurations | `contact_learned`, `cand_C_sequential`, `sham_C_yoked` (as in Stage 3), `sham_C_random` (new) |
| Constants | the pinned Stage-3 constants (C: Gaussian, h = 12, σ₀ = 4 mm), unchanged |
| Size | 22 × 4 × 300 = 26,400 episodes, one run, write-once raw files |

**`sham_C_random`:**
- Take C's suspension mask from the *same* episode (same policy, condition and episode seed).
- Rotate it circularly by a random offset, uniform on 1 … n − 1 for an n-step mask. The RNG is
  seeded from (20261604, policy seed, condition index, episode seed).
- Apply it as the yoked sham does: the `contact_learned` controller, with compensation forced off
  on masked steps. There is no window reset.
- This keeps the number of suspended steps exactly and moves them to a random time. Run lengths
  are kept too, except that a run rotated across the episode end is split into two.
- In the on-off cells a rotation by a multiple of the schedule period realigns the mask with
  the schedule by chance. That is part of random timing and is kept.

## Comparisons (all on T + I, paired episode-seed cluster bootstrap, 10,000 resamples)

| | comparison | role |
|---|---|---|
| **E1** | C − `sham_C_random` | **the one test**: one-sided at α = 0.025, H0 Δ ≤ 0 |
| E2 | `sham_C_yoked` − `sham_C_random` | descriptive: does schedule-aligned timing matter on its own? |
| E3 | `sham_C_random` − `contact_learned` | descriptive: what untimed suspension does |
| E4 | C − `sham_C_yoked` | descriptive: S2 again on fresh seeds |
| E5 | C − `contact_learned` | descriptive: H1 again on fresh seeds |

Rescued and harmed counts are reported for each. E1 is also reported per family (T and I),
descriptively.

## How it will be read

- **E1 CI low > 0:** C's gain depends on *when* it suspends, not only on how much. That is the
  specificity S2 meant to test, against untimed suspension.
  - It does **not** separate C's per-episode detection from schedule-aligned timing; that needs
    conditions whose transition times vary between episodes, which this grid does not have.
    E2 and E4 are reported for that reason.
- **E1 CI includes 0 or lies below it:** suspension of this amount explains C's gain whatever its
  timing; specificity is not shown.
- If E5 differs materially from Stage 3's +1.7, that is reported as seed-block variation.
