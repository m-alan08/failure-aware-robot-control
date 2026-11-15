# Threshold sweep: distance-only vs phase-aware detector (exploratory)

> Migrated from m-alan08/quantrl-fetch `docs/THRESHOLD_SWEEP.md` (branch `phase1-disturbance-eval`, last changed in `88bb9c0`). Results and commit hashes below refer to runs made in quantrl-fetch; the same numbers are regenerated here by `scripts/reproduce_phase1.sh` and checked against `results/`. Paths were updated to this repository.

The earlier comparison ([PHASE_AWARE_DETECTOR.md](PHASE_AWARE_DETECTOR.md)) held the deficit threshold at 0. That left the two detectors at different false-positive rates, so it could not say whether phase awareness moves the trade-off curve or only moves along it. This sweep compares them at **matched false-positive rates** over all thresholds, with and without a requirement that the warning leave time to act.

**Status: exploratory.** These FetchPush validation-range trajectories (seeds 3,500,000 to 3,500,099, training seeds 0 to 2) have already informed both detectors' design, the earlier comparisons, and the choice of settings below. Nothing here is a held-out estimate, and the reserved range (4,000,000+) was not used. Nothing was trained, and no policy was changed.

- Code: `src/farc/detection/threshold_sweep.py`, `threshold_sweep_plots.py`.
- Config: `configs/detectors/threshold_sweep_push.json`. Tests: `tests/test_threshold_sweep.py`.

```bash
uv run python scripts/run_disturbance_eval.py --config configs/disturbances/phase1_extended.json  # source, ~6 min
uv run python scripts/sweep_detector_thresholds.py --config configs/detectors/threshold_sweep_push.json        # ~2 min
```

- Provenance: commit `3815180` on a clean tree; source data from `09ea4f4` (identical to `phase1_extended`).
- Full outputs: `$FARC_DATA_DIR/experiments/phase1_extended/detectors/threshold_sweep_push/`.
- Compact copies: `results/threshold_sweep_push/` and `docs/figures/threshold_sweep_*.png`.

## Method

**Detectors.**
- Distance-only: k ∈ {5, 10}, p ∈ {3, 5}.
- Phase-aware: the same k and p with hysteresis h = 3, and the departure monitor off (m = off, the main family) or on (m = 3).

**An exact sweep, not a grid.** Both detectors warn at the first state where the deficit has exceeded θ at p consecutive states, and every reset is followed by an unavailable state. So for each episode one number, the largest θ at which the detector still warns early enough, decides the outcome at every threshold:
- with r_t = the minimum deficit over the last p states (−∞ if any is unavailable), the detector warns with more than L steps left iff max over t < 50 − L of r_t > θ;
- departure warnings do not depend on θ, so they enter as +∞.

This reproduces both original detectors exactly at every threshold and lead requirement, on synthetic and on logged episodes (tested), and at θ = 0 it reproduces the earlier results to the episode.

**Actionable trade-off.**
- *Detection* counts a failed episode only if the first warning leaves more than L steps (L = 0, 5, 10).
- *FPR* counts every warning on a successful episode, whenever it fires.
- Both are over the 6,300 disturbed episodes: 1,455 failures and 4,845 successes. Failure = not successful at step 50.

**Matched FPR.** For each detector and target FPR (2, 5, 10, 20, 30%), θ is the smallest threshold whose FPR does not exceed the target, so the realized FPR is at or just under the target. A detector with the departure monitor cannot go below its departure-only FPR (about 14%), so those targets are reported as unreachable (NaN).

**Envelopes.** "Best of family" takes, at each target and lead, the highest detection over the family's settings. That is a selection step, and it is redone inside every bootstrap resample.

**Uncertainty.**
- 2,000 stratified episode bootstrap resamples (episodes resampled with replacement within each training seed × condition). Thresholds and envelope choices are re-made in every resample, and differences are paired on the same resampled episodes.
- Training seeds are not resampled. As a separate consistency check, the matched comparison is repeated within each training seed (`results/threshold_sweep_push/sweep_per_seed.csv`).
- Intervals are percentile intervals, not significance tests.

## Results

Detection at matched FPR, best of each family (95% bootstrap intervals in `results/threshold_sweep_push/sweep_envelopes.csv`):

| Lead requirement | FPR target | Best distance-only | Best phase-aware (m off) | Difference [95% interval] |
| --- | --- | --- | --- | --- |
| none | 2% | 64.1% | 64.7% | +0.5 pts [−2.7, +4.6] |
| none | 5% | 76.7% | 82.1% | +5.4 pts [+2.4, +8.2] |
| none | 10% | 84.9% | 90.2% | +5.2 pts [+3.2, +7.5] |
| none | 20% | 92.0% | 94.7% | +2.7 pts [+1.2, +4.4] |
| none | 30% | 96.5% | 96.2% | −0.3 pts [−1.2, +0.7] |
| > 5 steps | 2% | 41.4% | 47.5% | +6.0 pts [+1.9, +11.3] |
| > 5 steps | 5% | 59.8% | 71.1% | +11.3 pts [+7.8, +14.4] |
| > 5 steps | 10% | 75.2% | 81.8% | +6.6 pts [+3.9, +8.8] |
| > 5 steps | 20% | 88.1% | 87.1% | −1.0 pts [−2.4, +0.3] |
| > 5 steps | 30% | 90.6% | 89.2% | −1.4 pts [−2.2, −0.6] |
| > 10 steps | 2% | 27.4% | 37.3% | +9.9 pts [+5.7, +16.1] |
| > 10 steps | 5% | 51.2% | 65.0% | +13.8 pts [+9.8, +17.6] |
| > 10 steps | 10% | 68.3% | 78.6% | +10.2 pts [+7.4, +12.8] |
| > 10 steps | 20% | 85.6% | 85.0% | −0.6 pts [−2.3, +0.8] |
| > 10 steps | 30% | 89.4% | 88.1% | −1.3 pts [−2.2, −0.5] |

- **Per training seed** (threshold re-chosen within each seed): at 2–10% FPR with more than 5 or more than 10 steps left, the phase-aware envelope is ahead for all three trained policies, by 4.6 to 16.5 points. At 20–30% FPR it is behind by 0.5 to 1.8 points for all three, except one tie (+0.2) at 20%.
- **The same pattern holds for matched k and p** (`results/threshold_sweep_push/sweep_paired.csv`). For example, phase k5 p5 vs distance k5 p5: +17.8 points at 5% FPR with more than 5 steps left; −1.3 points at 30%.
- **The advantage grows with the lead requirement.** At low FPR it is +5 points with no lead requirement and +10 to +14 points with more than 10 steps. Phase awareness mainly buys *earlier* warnings at a given false-alarm budget, not more warnings overall.
- **The departure monitor (m = 3)** cannot operate below about 14% FPR. At 20–30% it matches distance-only and adds at most 1.7 points over m = off (`phase_aware_all` rows). It does not help the actionable trade-off.
- **Detection ceilings differ.** Some failures never produce a valid warning at any threshold: with more than 5 steps left, 58 of 1,455 (phase k5 p3), 220 (phase k10 p5), and 122 (distance k10 p5). Phase resets and the maintain phase remove deficits that the distance-only detector keeps. This is why the phase-aware curves flatten and fall slightly behind at high FPR.

## Per-disturbance breakdown

`results/threshold_sweep_push/sweep_by_condition.csv` and `docs/figures/threshold_sweep_by_condition.png`. Each detector is run at the threshold that gives it 10% pooled FPR. Detection with more than 10 steps left:

| Condition (failures) | distance k10 p5 | distance k10 p3 | phase k5 p5 h3 | phase k10 p3 h3 |
| --- | --- | --- | --- | --- |
| action scale 0.5 (33) | 15% | 27% | 91% | 88% |
| action scale 0.65 (89) | 60% | 48% | 82% | 90% |
| action scale 0.75 (149) | 67% | 56% | 93% | 96% |
| action scale 0.85 (241) | 83% | 79% | 100% | 100% |
| action delay 3 (260) | 88% | 84% | 98% | 97% |
| full-vector obs noise 0.01 (238) | 85% | 84% | 93% | 92% |
| action noise 0.6 (40) | 72% | 75% | 40% | 35% |
| action noise 1.0 (104) | 64% | 68% | 69% | 70% |
| action delay 1 (46) | 59% | 61% | 41% | 43% |
| position noise 0.02 m (81) | 62% | 72% | 56% | 48% |
| velocity noise 0.25 m/s (15) | 67% | 67% | 53% | 33% |
| goal shift at step 25 (10) | 20% | 40% | 10% | 10% |
| goal shift at step 40 (120) | 0% | 1% | 2% | 2% |

- **Where the gain comes from.** It is concentrated in persistent slow motion: action scaling, the 3-step delay, and full-vector observation noise. There the failure is a whole episode of insufficient progress, and the phase-specific remaining path shows it early.
- **Where it loses.** Phase awareness is worse on action noise σ 0.6, the 1-step delay, position and velocity noise, and the goal shift at step 25: failures that develop late or near the goal, where phase resets and the maintain phase delay the warning.
- **Where false alarms land.** At the same pooled 10% FPR the phase-aware detectors put more of their false alarms on the hardest successful episodes (action scale 0.85: 53–58% vs 0–5%; delay 3: 57–65% vs 25–35%) and fewer on action noise and transient disturbances.
- **Goal shift at step 40** stays undetectable with useful lead for every detector.

**The pooled result depends on the disturbance mix.** Action-scaling and 3-step-delay failures make up 772 of the 1,455 validation failures, which is where phase awareness wins. A mix dominated by noise or late goal shifts could reverse the pooled difference.

## Conclusion

Within this exploratory validation set, phase awareness **improves the actionable detection trade-off in the low-false-alarm regime, not only the raw detection rate**:
- at 2–10% FPR it warns on 6 to 14 more failures out of 100 with more than 5 or 10 steps left;
- the episode-bootstrap intervals exclude zero;
- the sign is the same for all three trained policies.

The improvement does not extend to 20–30% FPR, where the distance-only detector is equal or slightly better. It is not uniform across disturbances: it comes from persistent slow-progress conditions and reverses on noise-driven and late failures.

This revises the earlier fixed-threshold conclusion ("not shown"). That comparison was confounded by the two detectors sitting at different FPRs.

**What this does not show:**
- that the detector is predictive on unseen episodes;
- that a warning with more than 10 steps left gives a controller enough time to recover;
- that the effect survives a different disturbance mix.

The settings, envelopes and target FPRs were all examined on the same data used to design the detector.

## Recommended next steps

1. **Freeze one setting per family before any further look at results**, for example phase k5 p5 h3 m-off and distance k10 p5, each at the threshold giving 5% or 10% pooled validation FPR. Then evaluate them once on the reserved held-out seeds.
2. **Report per-condition results and a condition-balanced summary** next to the pooled number, so the mix dependence is visible.
3. **Pair any warning with a measured recovery controller.** Only then do lead-time thresholds like 5 or 10 steps mean "actionable".
