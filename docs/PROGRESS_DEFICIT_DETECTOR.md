# Offline progress-deficit detector (exploratory)

> Migrated from m-alan08/quantrl-fetch `docs/DETECTOR.md` (branch `phase1-disturbance-eval`, last changed in `d6052bf`). Results and commit hashes below refer to runs made in quantrl-fetch; the same numbers are regenerated here by `scripts/reproduce_phase1.sh` and checked against `results/`. Paths were updated to this repository.

This evaluates a reactive, deadline-aware failure detector offline on the logged Phase 1 extended trajectories (validation seed range, 3 training seeds × 100 episodes × 20 or 22 conditions per task, 12,600 episodes). The detector never acts, and no policy is trained or changed.
- Code: `src/farc/detection/progress_deficit.py` (detector), `progress_deficit_eval.py` (evaluation), `progress_deficit_plots.py`.
- Runner: `scripts/evaluate_detector.py`. Config: `configs/detectors/progress_deficit_grid.json`. Tests: `tests/test_progress_deficit.py`.

```bash
uv run python scripts/run_disturbance_eval.py --config configs/disturbances/phase1_extended.json   # source data, about 6 min
uv run python scripts/evaluate_detector.py --config configs/detectors/progress_deficit_grid.json    # about 2 min
```

## Is the logged data sufficient?

Yes. No new logging and no reruns were needed. Every part's `trajectories.csv.gz` holds one row per state t = 0..50 with:
- the ground-truth achieved goal (the gripper for FetchReach, the **object** for FetchPush);
- the goal active in the simulator at that state;
- success and the executed actions.

`episodes.csv` holds the outcome, the disturbance type, severity, onset and duration, and the goal-shift step. Not logged: the gripper position in FetchPush. The detector does not need it. The repositioning analysis below uses two logged quantities as stand-ins: whether the object has left its start position, and whether actions were commanded.

## Definition

At decision state t (0 ≤ t < 50), with eps = 0.05 m:
- d_t = ‖achieved_t − g_t‖, where g_t is the goal active for the decision at t. For a goal shift decided at state o, the shift is logged from state o + 1, and the detector uses the new goal from state o on.
- Observed progress over k steps: P_obs = ‖achieved_{t−k} − g_t‖ − d_t.
- Required progress over k steps: P_req = k · max(d_t − eps, 0) / (50 − t).
- Deficit = P_req − P_obs. The state is flagged if deficit > 0.
- Suppression: no flag while d_t < eps, and the persistence count resets.
- Reset: the window may not reach back across a goal change, so the first k states after a shift are never flagged and the count resets.
- Warning: the first state where the flag has held for p consecutive states. The earliest possible warning is state k + p − 1.

Episode labels come from the source run: **failure = not successful at step 50.**
- Detection rate = failures warned / failures.
- False-positive rate (FPR) = successful episodes warned / successful episodes.
- Missed = failures not warned.
- Lead time = 50 − warning state (steps left when the warning fires).
- `infeasible_at_warning`: the remaining distance exceeds what the end effector can cover at full diagonal speed (√3 × 0.05 m per step). This is a loose necessary bound: a pushed block moves slower than the gripper.

The grid (k ∈ {3, 5, 10}, p ∈ {2, 3, 5}, threshold 0) was fixed before running. Outputs:
- full: `$FARC_DATA_DIR/experiments/phase1_extended/detectors/progress_deficit_grid/`;
- compact: `results/progress_deficit_grid/`;
- plots: `docs/figures/progress_deficit_tradeoff.png` (others in the output directory).
- Provenance: detector code at commit `934063d` on a clean tree; source data from `fed638e` (clean).

## Results over all disturbed conditions

Brackets are 95% seed-then-episode bootstrap intervals. "Undisturbed FPR" covers the 300 undisturbed episodes, which all succeed.

| Task | k | p | Detection | FPR (disturbed successes) | Undisturbed FPR | Median lead (steps) | Detected with lead ≤ 5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Push | 3 | 2 | 99.8% [99.5, 100] | 90.4% [88.7, 92.1] | 88.3% | 46 | 3 / 1452 |
| Push | 5 | 3 | 98.3% [97.5, 99.0] | 51.0% [46.7, 54.3] | 17.0% | 43 | 90 / 1430 |
| Push | 5 | 5 | 97.0% [95.9, 98.1] | 33.5% [31.8, 35.2] | 1.7% | 41 | 92 / 1411 |
| Push | 10 | 3 | 91.2% [87.9, 93.9] | 26.3% [24.9, 27.6] | 2.0% | 38 | 14 / 1327 |
| Push | 10 | 5 | 89.8% [86.7, 92.4] | 20.6% [19.4, 21.7] | 1.0% | 36 | 15 / 1307 |
| Reach | 3 | 2 | 89.4% [85.7, 92.4] | 21.8% [17.5, 26.7] | 0% | 40 | 10 / 572 |
| Reach | 5 | 3 | 65.6% [59.3, 73.4] | 9.2% [6.8, 11.4] | 0% | 38 | 15 / 420 |
| Reach | 5 | 5 | 45.2% [39.4, 51.5] | 2.4% [1.5, 3.3] | 0% | 28 | 33 / 289 |
| Reach | 10 | 5 | 46.9% [39.8, 53.9] | 2.3% [1.6, 3.0] | 0% | 20 | 38 / 300 |

All nine settings are in `results/progress_deficit_grid/detector_overall.csv`. The per-condition breakdown (`detector_by_condition.csv`, the output directory's `detector_by_condition.png`) and per-seed breakdown (`detector_per_seed.csv`) are also there.

## What these numbers do and do not show

1. **On FetchPush, high detection rates mostly come from warnings in the approach phase, not from discrimination.**
   - Successful and failed episodes warn at the same early states. With k = 3, p = 2 the median first warning is state 4 for both, and the lead-time quartiles of detected failures are all 46.
   - While the gripper travels to the block, the block does not move, so observed progress is zero and any required progress produces a deficit.
   - Among first warnings on successful Push episodes, 4,103 of 4,380 (k3 p2) and 310 of 997 (k10 p5) fire while the object has not yet moved more than 1 mm from its start. This is legitimate gripper repositioning flagged as a failure.
   - Disturbances that slow the approach (action scaling, delays, noise) lengthen this phase. That is why disturbed successes have far higher FPR than undisturbed ones (20.6% vs 1.0% at k10 p5).
2. **The detector can only fire at state k + p − 1 or later, and k states after a goal shift.**
   - A goal shift at step 40 leaves 10 steps. At k = 10 it is essentially undetectable (1 of 120 failures).
   - At k = 5, p = 3, 100 of 120 are detected, but 86 of those with 5 or fewer steps left. These are detections, but they come too late for any recovery to finish.
3. **FetchReach failures are mostly "hold" failures.**
   - The arm reaches the goal and then leaves the 5 cm radius by a small margin (noise, delay).
   - Suppression inside the radius plus a small required progress near the radius makes these hard to separate from successful episodes that hover near the boundary: k3 p2 warns on 21.8% of disturbed successes.
   - The 1-step delay (78 failures) is almost never detected at p ≥ 3.
   - The 3-step delay is detected at every setting but also warns on nearly all of its 24 successes.
4. **The kinematic-infeasibility bound almost never triggers** (0 to 49 detections out of about 1,300 to 1,450 per Push setting, 0 on Reach). It is too loose to say whether a warning is early enough. Lead time is not evidence that recovery is possible: a residual or recovery controller would need its own measured time to recover.
5. **Labels are binary final-step success.** A warning on an episode that recovers on its own counts as a false positive even if the warning was correct at that moment. Conversely, warnings are counted at the episode level, so a warning early in an episode that later fails for a different reason still counts as a detection.
6. **Limits of the data:**
   - These are the validation seeds that were also used to design the disturbance grid. The detector grid was fixed beforehand, but nothing here is a held-out estimate.
   - Three training seeds give a coarse seed-level uncertainty.
   - The analysis is offline: the detector sees ground-truth achieved and goal positions, not the controller's (possibly noisy) observation.

## Configurations worth further investigation

These are not claims of predictive or recovery capability.
- **k = 10, p = 5 and k = 10, p = 3** have the lowest false-positive rates on FetchPush (1.0–2.0% undisturbed, 20.6–26.3% on disturbed successes) and still warn on 90–91% of failures, with a median of 36–38 steps left. Their remaining false positives still include the approach phase.
- **k = 5, p = 5** is the low-FPR option on FetchReach (2.4%), at 45% detection.
- Before tuning the threshold, test a variant that does not judge progress until the object has started moving (or that measures progress for the gripper–object distance during the approach). Approach-phase warnings dominate every short-window setting on FetchPush.
- Evaluate detection timing against a recovery controller's measured time-to-recover, and only then on the reserved held-out seed range.
