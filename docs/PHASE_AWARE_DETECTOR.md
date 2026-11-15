# Phase-aware failure detector for FetchPush (exploratory, offline)

> Migrated from m-alan08/quantrl-fetch `docs/PHASE_DETECTOR.md` (branch `phase1-disturbance-eval`, last changed in `633d953`). Results and commit hashes below refer to runs made in quantrl-fetch; the same numbers are regenerated here by `scripts/reproduce_phase1.sh` and checked against `results/`. Paths were updated to this repository.

> **Superseded conclusion.** The "not shown" verdict below compares the two detectors at threshold 0, where they sit at different false-positive rates. The matched-FPR threshold sweep in [THRESHOLD_SWEEP.md](THRESHOLD_SWEEP.md) finds that phase awareness improves actionable detection at 2–10% FPR on these validation episodes, but not at 20–30%. That comparison is also exploratory.

This compares an interpretable phase-aware detector with the distance-only progress-deficit detector ([PROGRESS_DEFICIT_DETECTOR.md](PROGRESS_DEFICIT_DETECTOR.md)) on exactly the same FetchPush episodes. Nothing is trained, and no policy or checkpoint is changed.
- Code: `src/farc/detection/phase_aware.py`, `phase_aware_eval.py`, `phase_aware_plots.py`.
- Scripts: `scripts/evaluate_phase_detector.py`, `scripts/compare_detectors.py`.
- Config: `configs/detectors/phase_aware_push.json`. Tests: `tests/test_phase_aware.py`.

## Instrumentation

**Gripper positions were not recoverable from the saved logs.** The Phase 1 trajectories stored the object (the achieved goal), the goal, and the actions, but not the observation vector. Replaying the logged actions would have reconstructed them, but logging the true position directly was simpler and verifiable.
- `run_disturbed_episode` now records the ground-truth grip-site position for every state, taken from `observation[0:3]` before any observation noise. The observation layout is checked against the simulator in every episode. The trajectory CSVs gain `gripper_0..2`.
- In quantrl-fetch, FetchPush was rerun with the frozen seed-0/1/2 policies on the same validation seeds (3,500,000 to 3,500,099), conditions, and perturbation seed, into `phase1_extended_push_gripper/` (48 MB, 358 s, commit `09ea4f4`, clean tree), leaving the original `phase1_extended/` untouched.
- Match checks done there:
  - all 6,600 episode rows were identical to the original run;
  - the evaluation verified, episode by episode, that object, goal, and action trajectories were bit-identical to `phase1_extended` (`matched_reference` in the metadata);
  - the original-evaluator check passed for all three checkpoints.
- **In this repository** the runner logs gripper positions for every task from the start, so a single `phase1_extended` run (`configs/disturbances/phase1_extended.json`) serves both the distance-only and the phase-aware evaluations; `configs/detectors/phase_aware_push.json` reads it directly. `scripts/check_reproduction.py` confirms the phase-aware results are unchanged.
- The reserved held-out range (4,000,000+) was not used.

## Phase definitions

Inputs are ground truth: gripper g, object o, and the goal G active for the decision at each state. A goal shift takes effect at its decision state, as in the distance-only detector. eps = 0.05 m.

| Phase | Entered when | Progress test |
| --- | --- | --- |
| **Maintain** | ‖o − G‖ < eps | none; the departure monitor watches for exits |
| **Manipulate** | ‖g − o‖ ≤ 6 cm for h consecutive states (from approach) | remaining path L = max(‖o − G‖ − eps, 0): object progress against the deadline |
| **Approach** | otherwise; returns from manipulate when ‖g − o‖ > 9 cm for h consecutive states | L = ‖g − p*‖ + max(‖o − G‖ − eps, 0), where p* is the pre-push point 5 cm behind the object on the side away from the goal (horizontal direction) |

- **Deficit test.** In both non-maintain phases: observed = L(t − k) − L(t), with the goal and phase of t; required = k · L(t) / (50 − t); flagged when required − observed > 0. A deficit warning needs p consecutive flags.
- **Resets.** The window and the persistence count reset at every phase change and every goal change.
- **Contact tracking.** The contact tracker (approach vs manipulate) keeps running inside maintain, so leaving the success radius resumes in the phase the geometry indicates.
- **Gripper motion counts as progress.** An unmoved object is not a failure by itself, because gripper travel toward p* reduces L. But approach is not exempt: an approach too slow for the deadline still warns, and the warning can fire before the object moves.
- **Departure monitor (m).** After the object has been inside the radius of the *current* goal, a warning fires once it has been outside for m consecutive states (m = off disables it). Success against a shifted-away goal is forgotten at the shift.
- **Fixed before running:** the geometry constants (6/9 cm hysteresis band, 5 cm pre-push offset) were set from the block's 5 cm size, and the grid was k ∈ {5, 10}, p ∈ {3, 5}, h ∈ {1, 3}, m ∈ {3, off}, threshold 0. Baselines are the distance-only detector at the same k and p.

## Evaluation quantities

- **Labels:** failure = not successful at step 50 (from the source run). Rates are over the 6,300 disturbed episodes; undisturbed FPR is over 300 episodes that all succeed.
- **Timing:** the first object movement (≥ 1 mm from its start) and the first success state, for every episode.
- **Phase at warning:** for both detectors, the phase at the warning state comes from one reference tracker (h = 3, same radii), so "false alarms during approach" means the same thing for both.
- **Late warnings, two ways:**
  - lead ≤ 5 steps;
  - the object would need to move faster than the 95th percentile of the undisturbed base policy's per-step object speed (4.4 cm per step). This second bound is generous.
- **Paired comparisons:** differences on the same episodes, with seed-then-episode bootstrap intervals. These describe uncertainty; they are not significance tests.

## Results

Full tables:
- `results/phase_aware_push/phase_detector_overall.csv`;
- `phase_detector_paired.csv` (each phase setting against the distance-only detector with the same k and p);
- `detector_by_condition.csv` (lead time and rates by disturbance type and severity);
- `detector_per_seed.csv`;
- `post_hoc_comparisons.csv`.

Plots are in `docs/figures/phase_aware_tradeoff.png` (others in the output directory).

| Detector | Detection | FPR, disturbed successes | FPR, undisturbed | Median lead | False alarms in approach / before object moved |
| --- | --- | --- | --- | --- | --- |
| distance k5 p3 | 98.3% | 51.0% | 17.0% | 43 | 2,339 / 1,383 of 2,469 |
| distance k10 p3 | 91.2% | 26.3% | 2.0% | 38 | 1,057 / 451 of 1,273 |
| distance k10 p5 | 89.8% | 20.6% | 1.0% | 36 | 812 / 310 of 997 |
| phase k5 p3 h3 m3 | 97.3% | 23.6% | 5.3% | 39 | 861 / 177 of 1,142 |
| phase k5 p3 h3 m-off | 93.5% | 14.0% | 0.7% | 39 | 537 / 177 of 679 |
| phase k10 p3 h3 m-off | 82.1% | 7.8% | 0.7% | 37 | 314 / 113 of 380 |
| phase k10 p5 h3 m-off | 78.5% | 5.8% | 0.3% | 34 | 237 / 83 of 281 |

**Same k and p (paired, all 16 settings).**
- Every phase-aware setting has a lower FPR than the distance-only detector with the same k and p, by 2.4 to 42.7 points; every interval excludes zero.
- Every setting also detects fewer failures, by 0.1 to 21.6 points. The two settings with h = 3, m = 3 at k = 10 (−0.2 and −0.1 points) have intervals that include zero.
- Hysteresis h = 3 detects more failures than h = 1 at every k, p and m (by 3.2 to 10.3 points), at a higher FPR (by 2.0 to 5.8 points). With h = 1 the phase flickers at the contact boundary, and each flicker resets the window and the persistence count.

**Across settings (post hoc: pairs chosen after seeing the table above).** Without a lead-time requirement, phase k5 p3 h3 m-off detects 3.6 points more than distance k10 p5 [0.2, 7.9] at a 6.6-point lower FPR [−8.1, −5.0]. That looks like a better point on both axes, but the extra detections are almost all goal shifts at step 40: phase k5 p3 detects 73% of those failures, 86 of them with 5 or fewer steps left; distance k10 p5 detects 1%.

Counting only warnings with more than 5 steps left, every phase-aware setting compared detects fewer failures than its distance-only comparator:

| First vs second (min lead > 5) | Detection difference | FPR difference |
| --- | --- | --- |
| phase k5 p3 h3 m-off vs distance k10 p5 | −3.8 pts [−5.7, −2.0] | −6.8 pts |
| phase k5 p3 h3 m-off vs distance k10 p3 | −5.2 pts [−7.4, −3.3] | −12.5 pts |
| phase k5 p3 h3 m3 vs distance k5 p5 | −1.1 pts [−2.1, −0.2] | −10.1 pts |
| phase k10 p3 h3 m-off vs distance k10 p5 | −10.6 pts [−13.8, −6.9] | −12.7 pts |
| phase k10 p5 h3 m-off vs distance k10 p5 | −15.0 pts [−19.0, −11.0] | −14.8 pts |

## What changed, and what did not

- **False alarms during legitimate repositioning dropped but did not disappear.**
  - At k10 p5, false alarms issued while the object had not yet moved fell from 310 to 83, and those issued in the approach phase from 812 to 237.
  - Most remaining phase-aware false alarms are still in the approach phase. Under action scaling or delay, the gripper approaches slower than a constant-rate deadline requires and the episode still succeeds.
  - The constant-rate requirement does not know that pushing goes faster than the approach budget assumes, or that the policy front-loads approach time.
- **The departure monitor (m = 3) mostly adds false alarms.**
  - It adds 5 points of undisturbed FPR: the block often leaves the radius for 3 or more states and comes back on its own.
  - It produces 576 to 655 departure warnings on successful episodes against 216 to 323 on failures.
  - It is the only component that warns after first success (median +3 to +5 states for false alarms).
- **Warnings fire later.** The median lead time is 30 to 39 steps for the phase-aware settings, against 36 to 43 for distance-only. For the h = 3 settings, false alarms come a median of +7 to +11 states after the object starts moving, against −1 to +5 for distance-only.
- **Late warnings remain.** Goal shifts at step 40 cannot be caught with useful lead time by any setting (k = 5: median lead 3; k = 10: essentially never). The 4.4 cm-per-step speed bound flags 2 to 68 detections per setting as infeasible. Both criteria are rough. Whether a warning leaves enough time can only be judged against a measured recovery controller.

## Does phase awareness improve the detection trade-off?

**Not shown.** At matched k and p it trades detection for a lower FPR: it moves along the trade-off curve, mainly by removing approach-phase alarms. It does not move the curve. Once warnings must leave more than 5 steps, no phase-aware setting tested matches the detection rate of a distance-only setting with a higher FPR, and the one apparent improvement without that requirement comes from too-late goal-shift detections.

What the phase-aware detector does provide is operating points the distance-only grid could not reach (FPR 6 to 14% on disturbed successes, under 1% undisturbed). Whether the distance-only detector reaches those points with a positive deficit threshold has not been tested. A fair test needs both detectors swept over the deficit threshold and compared at matched FPR.

## Limitations

- **Fixed geometry constants.** The contact radii (6/9 cm) and pre-push offset (5 cm) were set a priori and are not validated against contact forces. Pushing from the side or corner, and sliding contact, are classified by distance alone.
- **Ground-truth inputs.** The detector sees ground-truth positions, not the controller's noisy observation, so observation-noise conditions are easier for it than they would be online.
- **Binary labels.** Labels are final-step success: an episode that recovers on its own makes an early correct warning a false positive.
- **Validation data, post hoc comparisons.** Everything is on the validation range that was also used to design the disturbance grid and to view the earlier detector results. The cross-setting comparisons were chosen after seeing results. Nothing here is a held-out estimate. Three training seeds make the seed-level intervals coarse.

## Configurations worth further study

- **phase k5 p3 h3 m-off:** 85% of failures warned with more than 5 steps left, at 14% FPR on disturbed successes and 0.7% undisturbed.
- **phase k10 p5 h3 m-off:** the low-false-alarm option (5.8% / 0.3%), at 74% detection with more than 5 steps left.
- **Next steps:**
  - sweep the deficit threshold for both detector families and compare them at matched FPR;
  - replace the constant-rate requirement in approach with a separate approach budget;
  - drop or lengthen the departure monitor;
  - measure recovery time with an actual recovery controller before claiming any warning is actionable;
  - then test the chosen settings once on the held-out range.
