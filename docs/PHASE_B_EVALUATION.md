# Phase B evaluation: learned failure predictors vs reactive detectors

Code: `src/farc/prediction/evaluation.py`; script: `scripts/evaluate_failure_predictors.py`;
config: `configs/prediction/phase_b_evaluation.json` (fixed before any evaluation numbers were
seen); tests: `tests/test_phase_b_evaluation.py`; outputs: `results/phase_b_evaluation/`.

```bash
uv run python scripts/build_failure_dataset.py        # optional: regenerates the dataset report
uv run python scripts/evaluate_failure_predictors.py  # about 20 s
```

## What is evaluated

- **Learned:** `$FARC_DATA_DIR/models/failure_predictor/{unweighted,weighted}.pt`, the
  24 → 128 → 64 → 1 ReLU MLP, sigmoid output. Loaded `weights_only`, strictly, after checking
  every parameter name and shape; the SHA-256 of each file is recorded in `metadata.json`.
  The checkpoints hold no normalizer: the training-set normalizer is recomputed with
  `build_datasets(split_seed=0, validation_fraction=0.2, progress_window=5)` and must match
  `results/failure_dataset/report.json` exactly, or the script stops. Missing files also stop
  it. Nothing is retrained or tuned.
- **Reactive** (unchanged code, departure monitor off): distance-only k ∈ {5, 10} × p ∈ {3, 5},
  phase-aware the same with h = 3. Primary comparators, fixed in advance from Phase 1:
  `distance_k10_p5` and `phase_k5_p5_h3`.

## Splits (by episode seed; every condition and policy seed of a seed stays together)

| Part | Episode seeds | Episodes | Disturbed failures / successes |
| --- | --- | --- | --- |
| Training (predictors) | 80 seeds | 5,280 | not used here |
| Calibration (thresholds) | 3500009, 3500010, 3500011, 3500016, 3500027, 3500037, 3500052, 3500075, 3500083, 3500094 | 660 | 149 / 481 |
| Evaluation (all reported warning metrics) | 3500005, 3500008, 3500013, 3500020, 3500036, 3500072, 3500081, 3500082, 3500090, 3500093 | 660 | 133 / 497 |

Calibration/evaluation is a seeded permutation (`calibration_seed = 0`) of the 20 validation
seeds. No seed of 4,000,000+ is read (the loader refuses them).

## Definitions

- **Score per decision step** t = 0..49 (T = 50; T − t ≥ 1 steps remain): learned = predicted
  failure probability; reactive = r_t, minimum deficit over the last p states. Every detector
  warns at the **first** t with score > θ; no persistence or smoothing is added to the learned
  predictors.
- **Episode-level warning metrics** (disturbed episodes pooled, as in Phase 1):
  - FPR: successful episodes with any warning.
  - Detection, any lead: failed episodes with a first warning at t ≤ 49.
  - Detection, lead > 5 / > 10 (strict): first warning at t with T − t > 5 (t ≤ 44) /
    T − t > 10 (t ≤ 39). A first warning at or after step 45 / 40 does not count, and a later
    warning cannot replace it.
  - First-warning timestep: median and mean t of the first warning on detected failures.
- **Thresholds:** for target FPR α ∈ {2, 5, 10%}, the smallest θ whose FPR on *calibration*
  disturbed successes is ≤ α. The realized evaluation FPR is reported and can exceed α.
- **Timestep-level metrics** (learned only): every (episode, step) of the evaluation part,
  all 22 conditions, label = episode's final failure. AUROC, PR-AUC (average precision),
  Brier, 10-bin reliability and ECE. These rank and calibrate per-step probabilities; they are
  *not* warning metrics.
- **Intervals:** 95% percentile, 2,000 resamples (500 for timestep metrics). Calibration seeds
  and evaluation seeds are resampled with replacement separately; a drawn seed brings all its
  66 episodes, so repeated conditions and policies sharing a seed stay together. Thresholds
  are re-chosen per resample; differences are paired on the same resample.

## Results (evaluation part)

Pooled warning metrics, % [95% interval]:

| Target FPR | Detector | Realized FPR | Detection any | Lead > 5 | Lead > 10 | Median first-warning t |
| --- | --- | --- | --- | --- | --- | --- |
| 2% | learned unweighted | 2.4 [0.8, 5.9] | 78.9 [73.0, 88.6] | 75.2 [67.1, 84.1] | 67.7 [58.6, 80.7] | 22 |
| 2% | learned weighted | 3.0 [0.8, 5.8] | 78.9 [68.2, 86.4] | 74.4 [60.3, 84.1] | 68.4 [48.6, 80.4] | 19 |
| 2% | distance k10 p5 | 1.2 [0.2, 3.1] | 57.1 [44.8, 75.7] | 35.3 [25.0, 56.4] | 21.8 [11.5, 38.2] | 43 |
| 2% | phase k5 p5 h3 | 2.4 [0.8, 4.4] | 72.2 [64.1, 82.3] | 48.9 [42.7, 67.0] | 39.1 [31.8, 53.7] | 38.5 |
| 5% | learned unweighted | 5.8 [2.9, 8.5] | 87.2 [81.0, 93.3] | 84.2 [79.1, 91.5] | 79.7 [74.1, 85.3] | 5 |
| 5% | learned weighted | 6.6 [3.5, 9.7] | 86.5 [80.6, 91.4] | 83.5 [79.2, 87.4] | 78.9 [72.9, 84.7] | 5 |
| 5% | distance k10 p5 | 5.8 [3.2, 10.3] | 80.5 [74.0, 86.9] | 69.2 [54.5, 80.9] | 57.1 [40.6, 71.8] | 32 |
| 5% | phase k5 p5 h3 | 4.2 [2.0, 6.3] | 84.2 [77.5, 89.5] | 75.2 [62.0, 81.4] | 68.4 [49.6, 76.3] | 19 |
| 10% | learned unweighted | 10.3 [6.3, 14.9] | 91.7 [85.5, 96.9] | 88.7 [83.3, 93.9] | 84.2 [79.1, 89.1] | 4 |
| 10% | learned weighted | 11.3 [7.3, 15.6] | 90.2 [84.7, 95.1] | 88.0 [83.1, 93.2] | 84.2 [79.2, 90.5] | 4 |
| 10% | distance k10 p5 | 11.1 [7.0, 15.8] | 85.0 [76.8, 91.5] | 74.4 [61.3, 85.3] | 64.7 [49.6, 79.3] | 26 |
| 10% | phase k5 p5 h3 | 6.4 [3.9, 9.1] | 88.0 [83.1, 92.9] | 81.2 [74.8, 87.3] | 77.4 [70.6, 84.6] | 14 |

Paired differences in detection, percentage points [95% interval]:

| Comparison | Target FPR | Any | Lead > 5 | Lead > 10 |
| --- | --- | --- | --- | --- |
| unweighted − distance k10 p5 | 2% | +21.8 [+6.1, +34.8] | +39.8 [+22.0, +51.6] | +45.9 [+30.8, +61.6] |
| | 5% | +6.8 [+2.2, +10.9] | +15.0 [+3.8, +28.5] | +22.6 [+10.5, +36.3] |
| | 10% | +6.8 [+2.4, +13.2] | +14.3 [+3.4, +26.6] | +19.5 [+6.8, +33.1] |
| unweighted − phase k5 p5 h3 | 2% | +6.8 [−2.7, +20.3] | +26.3 [+9.0, +37.5] | +28.6 [+13.7, +44.2] |
| | 5% | +3.0 [−3.3, +10.1] | +9.0 [+3.2, +21.4] | +11.3 [+4.8, +28.7] |
| | 10% | +3.8 [−3.0, +9.1] | +7.5 [+0.7, +13.2] | +6.8 [+0.7, +11.9] |
| weighted − distance k10 p5 | 2% | +21.8 [+3.9, +32.9] | +39.1 [+19.0, +51.2] | +46.6 [+26.7, +60.8] |
| | 5% | +6.0 [+0.9, +10.4] | +14.3 [+3.4, +26.8] | +21.8 [+8.8, +37.1] |
| | 10% | +5.3 [+1.3, +11.7] | +13.5 [+2.3, +27.4] | +19.5 [+6.5, +33.9] |
| weighted − phase k5 p5 h3 | 2% | +6.8 [−7.7, +18.3] | +25.6 [+4.0, +37.5] | +29.3 [+6.4, +43.8] |
| | 5% | +2.3 [−4.3, +8.8] | +8.3 [+2.3, +20.7] | +10.5 [+4.3, +28.7] |
| | 10% | +2.3 [−4.5, +7.9] | +6.8 [+0.7, +12.4] | +6.8 [+0.7, +12.5] |
| weighted − unweighted | 2% | 0.0 [−10.0, +4.3] | −0.8 [−11.4, +6.5] | +0.8 [−15.3, +8.8] |
| | 5% | −0.8 [−4.0, +1.6] | −0.8 [−5.9, +2.5] | −0.8 [−4.2, +2.7] |
| | 10% | −1.5 [−4.0, +0.7] | −0.8 [−3.5, +3.0] | 0.0 [−2.2, +3.8] |

FPR differences and all other reactive settings: `warning_differences.csv`,
`warning_intervals.csv`, `warning_pooled.csv`.

Timestep-level (evaluation part, 33,000 examples, failure fraction 20.2%):

| Predictor | AUROC | PR-AUC | Brier (constant baseline 0.161) | ECE (10 bins) |
| --- | --- | --- | --- | --- |
| unweighted | 0.920 [0.903, 0.933] | 0.826 [0.799, 0.850] | 0.073 [0.066, 0.084] | 0.012 |
| weighted | 0.920 [0.904, 0.932] | 0.825 [0.800, 0.851] | 0.074 [0.066, 0.084] | 0.010 |

Reliability is close to the diagonal, with slight over-prediction between 0.2 and 0.8 (e.g. bin
0.7–0.8: predicted 0.75, observed 0.69) (`reliability.csv`). AUROC falls with the time left:
0.99 in the last 10 steps, 0.83 with 40–50 steps left (`timestep_by_remaining.csv`).

## Breakdowns

`warning_by_condition.csv` (all 22 conditions, including undisturbed) and
`warning_by_policy_seed.csv`, at the pooled calibration thresholds. At 5% FPR with lead > 10,
the learned predictors are ahead of both primary reactive detectors for all three policy seeds
(unweighted 77 / 80 / 83% vs phase-aware 67 / 59 / 78%). Each condition has only 30 evaluation
episodes, so per-condition rates are descriptive. Patterns at 5% FPR:

- The gain comes from persistent slow-progress disturbances (action scale 0.5–0.85, 3-step
  delay), which the learned predictors detect from the first few steps.
- False positives concentrate in the same hard conditions (3-step delay 40%, full-vector
  observation noise 75%, action scale 0.75 23%): the predictors in effect recognize a
  hard condition early and warn on its successes too. The undisturbed condition has no warnings.
- No detector warns early for the goal shift at step 40 (0/11).
- On observation_noise_positions 0.02 the learned predictors detect fewer failures early than
  distance-only (43 vs 71%, 7 failures).

## Caveats

- **The validation split is not untouched for the learned predictors.** The training script
  saves the epoch with the lowest validation loss, which uses all 20 validation seeds,
  including the evaluation half. Epoch selection is a small amount of selection, but these
  numbers are not held-out estimates.
- The reactive detectors and their settings were designed on these same seeds in Phase 1.
- 10 evaluation seeds give wide intervals; seed-cluster resampling treats seeds as the
  independent units. The three policy seeds are not resampled.
- Features use ground-truth state; observation noise reaches them only through its effect on
  the motion.
- Final evidence needs the reserved held-out range, which was not touched.
