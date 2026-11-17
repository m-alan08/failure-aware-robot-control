# Phase H: delay-randomised retraining vs compensating the frozen policy (Stage 2 results)

## Run record

| item | value |
|---|---|
| Plan | `docs/PHASE_H_RETRAINED_BASELINE_PROPOSAL.md` (SHA-256 `f7989f90…`), approved as written 2026-11-17 |
| Recorded before use | J1–J8 before training (commit `07e4e72`); J9–J10 after training, before evaluation (`fea242f`). The evaluation ran at `fea242f` on a clean tree. |
| Training | 6 new policies, R1 and R2 × seeds 0–2, 1,000,000 steps each, upstream hyperparameters, final checkpoints, no tuning. About 6,600–6,720 s per run, 6 in parallel (≈ 11 CPU-hours). |
| Evaluation | development seeds **3,701,000–299** × 3 policy seeds = 900 pairs per cell; 50 cells (G4's 41 + E/F1 C0–C8) × 5 configurations = **225,000 episodes**, one run (26 minutes) |
| Frozen | SAC `fetch_push_fp32_seed{0,1,2}`, the F1 model, and Candidate C's code and constants. All unchanged. |
| Held-out | **not used.** Stage 3 needs your decision. |
| Raw | `$FARC_DATA_DIR/experiments/phase_h_evaluation/` (write-once); tables in `results/phase_h_evaluation/` |

Numbers are success-rate points, with paired 95% episode-seed cluster-bootstrap CIs; counts are
rescued / harmed. **All results are development-seed results.**

## Stage 0 and Stage 1 (code and training checks)

- **The ported trainer is bitwise identical to upstream** quantrl-fetch @ `95e9611`, in all
  124 tensors, on a 20,000-step run.
- **Nominal seed-0 retrain = frozen `fetch_push_fp32_seed0`, bitwise, in all 124 tensors.**
  The frozen FetchPush policy is exactly regenerable from upstream code on this machine (J10).
  This resolves the open "bitwise re-trainability" item of the audit, for one machine.
- **The delay wrapper** executes exactly what `ScheduledDelaySession` executes, across all six
  training families.

## Gates for the retrained policies (checked first)

| | undisturbed success (pooled H + E C0) | vs frozen SAC | gate (CI low > −2) |
|---|---|---|---|
| frozen SAC | 98.9 | – | – |
| R1 | 96.3 (seeds: 93.3, 96.3, 99.3) | −2.7 [−3.9, −1.4] | **failed** |
| R2 | 96.0 (seeds: 99.0, 92.7, 96.3) | −3.0 [−4.4, −1.6] | **failed** |

- **No run crossed the 90% training-failure flag.** The lowest policy reached 92.7%.
- **Delay randomisation at this budget costs about 3 points of nominal performance.** By the
  pre-registered reading (proposal §5), the gate failure means:
  - the retrained baseline is **weak at this budget**;
  - its comparisons are reported below;
  - **no conclusion about retraining in general is drawn.**

## Primary hypotheses (Holm across H1–H3)

| | question | contrast | estimate [95% CI] | counts | Holm p | result |
|---|---|---|---|---|---|---|
| **H1** | Does retraining avoid the short-delay failure? | R2 − frozen SAC, RT + RI (margin −2) | −0.2 [−1.4, +1.1] | 218 / 225 | 0.006 | **supported** |
| **H2** | Does retraining beat the compensator there? | R2 − C, RT + RI | +0.9 [−0.4, +2.2] | 251 / 217 | 0.079 | not supported |
| **H3** | Is the compensator competitive under persistent delay? | C − R2, persistent (RP + F-P + E C1–C7; margin −2) | **+28.6 [+27.0, +30.3]** | 4,160 / 295 | < 0.001 | **supported** (far better, not just non-inferior) |

**Per training seed:**

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| H1 | +1.9 | −3.1 | +0.6 |
| H2 | +2.7 | −1.4 | +1.6 |
| H3 | +22.4 | +32.2 | +31.3 |

The short-delay contrasts vary across seeds; the persistent-delay contrast is large for every
seed.

## Secondary (Holm across S1–S4) and descriptive

| | contrast | scope | estimate [95% CI] | result |
|---|---|---|---|---|
| S1 | R2 − R1 | RT + RI | +0.1 [−1.3, +1.4] | not supported |
| S1 | R2 − R1 | persistent | **+12.1 [+10.7, +13.4]** | supported |
| S2 | R1 − frozen SAC (margin −2) | RT + RI | −0.3 [−1.5, +1.0] | supported |
| S3 | C − R1 (margin −2) | persistent | +40.7 [+39.0, +42.4] | supported |
| S4 | R2 − contact-aware | RT + RI | **+5.5 [+4.0, +7.0]** | supported |
| S5 (descriptive) | R2 − C | noise / gain loss (out of distribution) | −29.1 [−31.0, −27.2] | – |

## Success by family (%)

| family | frozen SAC | contact-aware | **C** | R1 | **R2** |
|---|---|---|---|---|---|
| RT random single pulses | **98.0** | 96.1 | 97.8 | 95.1 | 94.8 |
| RI random on-off | 87.7 | 78.3 | 85.6 | 90.1 | **90.5** |
| F-T fixed single pulses | 97.7 | 96.9 | **97.9** | 95.3 | 94.4 |
| F-I fixed on-off | 86.0 | 73.7 | 78.2 | **87.8** | 86.1 |
| RS random change of value | 36.6 | 90.2 | **93.6** | 52.2 | 60.0 |
| RP random-onset persistent | 56.3 | 93.1 | **94.9** | 62.7 | 71.1 |
| F-P fixed persistent | 50.2 | 96.5 | **96.6** | 59.8 | 69.5 |
| E C1–C7 constant and mid-onset | 40.0 | 94.8 | **94.9** | 48.5 | 63.6 |
| NG noise / gain loss | 58.3 | 89.4 | **90.7** | 55.4 | 61.6 |
| H / E C0 / E C8 no delay | **97–99** | 95–99 | 95–99 | 93–96 | 93–96 |

## Reading

1. **Under persistent delay, compensating the frozen policy is far better than delay-randomised
   retraining at the same budget.**
   - C beats R2 by +28.6 points on persistent cells.
   - R2 recovers only about a third of the gap to C: 63.6% vs 40.0% frozen SAC vs 94.9% C on
     E/F1 constant delays.
   - The command history helps a lot there (S1, +12.1), consistent with delay-aware RL, but not
     nearly enough.
2. **On short and intermittent delays, retraining avoids the stale-compensation failure.**
   - R2 is within the margin of frozen SAC (H1, −0.2).
   - R2 beats contact-aware by +5.5 (S4).
   - It does not significantly beat Candidate C (H2, +0.9 [−0.4, +2.2]).
   - By family, R2 is best on on-off delays (RI 90.5 vs C 85.6), but worse than C and frozen SAC
     on single pulses (RT 94.8 vs 97.8 and 98.0).
3. **Retraining has a nominal cost:** about −3 points undisturbed, which fails the
   pre-registered gate.
4. **Combined, no single approach wins everywhere at this budget:**
   - Candidate C dominates under persistent and changing delays.
   - Frozen SAC or R2 are best on on-off delays.
   - Frozen SAC is best with no delay and on single pulses.

## What this does and does not show

- **Shown, on development seeds:** under this project's delay distribution, budget and
  hyperparameters, delay-randomised SAC+HER retraining:
  - is not competitive with frozen-policy compensation under persistent delay;
  - does not suffer the short-delay failure;
  - costs nominal performance.
- **Not shown:**
  - **Retraining in general.** The gate failed, and only one budget (1M steps) and one set of
    hyperparameters were tried, with no tuning by design. A larger budget, a curriculum, a
    recurrent policy (as Peng et al. 2018 used), or a delay-aware algorithm (BPQL, VDPO, ACDA)
    could do better.
  - **Held-out confirmation:** none of this is held-out. Stage 3 (4,000,600–899) needs your
    decision.
  - **Real-robot or other-task results.**
- **Implication for the negative finding** (`docs/RESEARCH_POSITIONING.md`):
  - The short-delay failure is a property of *compensating with a stale estimate*, not of delay
    as such: an agent trained with delays tolerates short delays (H1) at the cost of persistent
    performance.
  - This sharpens the "when to compensate" thesis. Compensation is strongly preferable under
    persistent delay. For short, frequent delays, not compensating (frozen SAC) or a
    delay-trained policy is as good or better.

## Budget comparison (descriptive)

| approach | extra learning | data / compute |
|---|---|---|
| Candidate C on the frozen policy | F1 contact model (2 × 256 MLP), plus the estimator and detector (no learning) | 352,800 model-training transitions; no policy training |
| R1 / R2 retraining | full SAC+HER policy per seed | 1,000,000 environment steps and about 1.85 CPU-hours per seed |

## Stage 3 status

- **Decision:** on 2026-11-17 the user chose **not yet** to run the Stage 3 held-out
  confirmation.
- **Seeds:** 4,000,600–899 remain unused and reserved. No Phase H guard admits any held-out
  seed.
- **Status of these results:** they remain development-only.
