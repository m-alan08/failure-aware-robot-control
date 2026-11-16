# G2 Stage 2: calibration record

**Status:** set on calibration seeds 3,672,000–099 before any evaluation seed was used. This
file's SHA-256 is pinned in `configs/delaychange/g2_stage3.json`, and the evaluation refuses
to run if it changes.

| item | value |
|---|---|
| Rules | `docs/PHASE_G2_PROPOSAL.md` §5.3, with A7–A15 in `docs/PHASE_G2_STAGE3_DEFINITIONS.md` |
| Cells | persistent delay 1, 2, 3, 4; undisturbed; position noise 0.01 m; delay 3 from step 12 for 6 and for 12 steps |
| Size | 49 settings × 8 cells × 100 seeds × 3 frozen SAC policies = 117,600 episodes |
| Read | detection metrics only. Task success was not written and not read. |
| Raw | `$FARC_DATA_DIR/experiments/phase_g2_stage3_calibration/` (write-once); metrics in `results/phase_g2_stage3_calibration/` |

## Chosen constants

| candidate | constant | admissible settings | why this one |
|---|---|---|---|
| **A** short window | **w_s = 7** | only w = 7 | w = 4, 5, 6 exceeded the 0.5% false-activation limit on the harm cells (2.01%, 0.74%, 0.55%). w = 7: 0.42%, with a wrong-reliable rate of 0.06% (limit 0.015% + 0.5%). |
| **B** two-timescale | **w_f = 4** | w_f = 4 and 5 | the smallest admissible; suspended 0.74% (limit 1%) |
| **C** sequential | **Gaussian, h = 12, σ₀ = 4 mm** | Gaussian h12/s4 and h8/s4; scale-free h1/s2, h1/s4, and every scale-free setting with h ≥ 2 | the lowest median release latency (1 step) is shared by Gaussian h12/s4, Gaussian h8/s4, scale-free h1/s2 and scale-free h1/s4. The pre-registered tie-break (larger h first) picks Gaussian h12/s4: suspended 0.45%, false activation 0.07%. |

Reference, `contact_learned`: wrong-reliable rate 0.015%, suspended 0.05%, false activation
0.28%.

## Notes recorded now, before the evaluation

- **The release-latency metric counts an end at which compensation was already off as 0.**
  This is the definition in A9, the first off step at or after the end. On the shorter
  calibration pulse (6 steps), many ends come before compensation ever engaged. That is why
  `contact_learned` shows a median of 4 and B a median of 0. The metric was applied the same
  way to every C setting, which is the only place it decided anything.
- **The Gaussian variant is much more alarm-prone than the scale-free one** at small h
  (suspended up to 11%). Only h = 8 and 12 with σ₀ = 4 mm were admissible. The chosen
  setting is at the edge of the grid (the largest h and σ₀), which suggests the Gaussian noise
  model is tight for this plant. This is noted, not acted on; the rule was applied as
  written.
- **A's admissible window is 7.** A will therefore be only modestly faster than the 10-step
  estimator.
