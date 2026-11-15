# Phase C: failure-triggered deterministic recovery (FetchPush-v4, development seeds)

**Result: with this recovery controller, triggering it on the learned predictor's or the
phase-aware detector's warnings does not improve task success; both triggered configurations
are about 1 point *worse* than the frozen base policy, and always-on recovery is 11 points
worse.** An oracle that runs recovery only on the episodes the base policy fails would gain
+5.5 points, so the recovery controller can rescue some failures; the triggers do not select
them well enough to outweigh the harm on episodes the base policy would have solved.

Code: `src/farc/recovery/` (`controller.py`, `rollout.py`, `experiment.py`); script:
`scripts/run_recovery_experiment.py`; config: `configs/recovery/phase_c_development.json`;
tests: `tests/test_recovery.py`; tables: `results/phase_c_development/`.

```bash
uv run python scripts/run_recovery_experiment.py                  # ~9 min, 3 processes
uv run python scripts/run_recovery_experiment.py --analyze-only   # re-run the analysis
```

## Setup

| Item | Value |
| --- | --- |
| Base policies | frozen `fetch_push_fp32_seed{0,1,2}` (SHA-256-verified; actor weights checked unchanged after the run) |
| Conditions | the 21 FetchPush disturbance conditions of `phase1_extended` + undisturbed |
| Episode seeds | **3,610,000 – 3,610,099** (validation stage; disjoint from Phase 1/B seeds 3,500,000–099 and from 3,600,000–029, used to debug the controller). No seed ≥ 4,000,000 is read; the config loader refuses those and the used ranges. |
| Perturbation seed | 20261201 |
| Episodes | 100 seeds × 22 conditions × 3 policies × 4 configurations = 26,400, fully paired |
| Learned trigger | `models/failure_predictor/unweighted.pt` (shape-checked), training normalizer recomputed and verified against `results/failure_dataset/report.json`; threshold **0.871566** (Phase B 5% target FPR) |
| Reactive trigger | phase-aware k = 5, p = 5, h = 3, departure monitor off; threshold **0.010676** (Phase B 5% target FPR) |

The run checks both thresholds against `results/phase_b_evaluation/metadata.json` and stops
if they differ.

Pairing: every configuration replays the same reset seed (initial state and goal), the same
perturbation generators (noise draws, goal-shift targets), the same policy and condition.
With recovery never active the closed-loop rollout reproduces Phase 1's `run_disturbed_episode`
action for action (tested on delay, goal shift and observation noise).

## Configurations

1. **base_only**: the frozen SAC+HER policy.
2. **always_on**: the recovery controller acts at every step.
3. **reactive_triggered** and 4. **learned_triggered**: the base policy acts until the trigger
   fires; then recovery acts under the switching protocol below.

**Recovery controller** (`RecoveryController`). Inputs are the gripper position obs[0:3], the
object position obs[3:6] and the desired goal, read from the controller's observation (after
any observation noise): the same information the base policy has, no simulator internals.
With u the horizontal unit vector from object to goal:
- **HOLD** if the object is within 1.5 cm of the goal (command 0).
- **PUSH** if the gripper is behind the object (more than 3 cm behind along u, within 1.5 cm of
  the object→goal line, within 2 cm in height; looser once pushing). The gripper steers onto
  the line and advances by at most 3 cm or the remaining distance, so it stops when the object
  arrives.
- **REPOSITION** otherwise: lift 7 cm if the path to the pre-push point (6 cm behind the object)
  would cross it, move above that point, descend.
- Command: clip((target − g) / 0.05, −1, 1); fingers 0 (blocked in Push).

**Switching** (`InterventionSwitch`, `max_duration = 20`, `exit_radius = 0.03`,
`exit_steps = 2`, `cooldown = 5`, `min_steps_left = 1`):
- *Activation*: at decision t, if recovery is not active, not in cooldown, the trigger score is
  above threshold, and more than 1 step is left, recovery acts from step t.
- *Termination*: the object is within 3 cm of the goal for 2 consecutive decisions (success
  exit), or 20 recovery steps have passed (timeout), or the episode ends.
- *Return*: the base policy acts from the next step; the trigger is ignored for 5 steps and
  then may fire again (a new intervention).
- Triggering does not guarantee success: success is measured on the simulator at step 50.

**Disturbances act on whichever controller commands** (action scale, delay and noise are
properties of the actuator; delay replays the issued command history). Triggers and recovery
read the noisy controller view.

**Online features without lookahead.** At decision t the history holds the controller's view
of states 0..t, the goal used at decision t (after a goal shift at step s the trigger sees the
new goal from decision s, exactly as the offline features use `goal[t+1]`), and base-policy
outputs up to t − 1. The base policy is evaluated at every step, also during recovery, because
the `previous_action` feature is the base policy's output. Future rows are filled with the
latest values and the unchanged offline functions are called. Tests check that online features
and phase-aware scores equal the offline ones at every tested step (including around a goal
shift) and do not depend on later rows.

## Results (pooled over 6,600 paired episodes per configuration)

| Configuration | Success | Δ vs base [95% CI] | Activated | Recovered (base fail → success) | Harmed (base success → fail) | Unnecessary interventions (of 5,063 base successes) | Interventions per activated episode | Recovery steps per activated episode | Median first activation step |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base only | 76.7% | – | – | – | – | – | – | – | – |
| always on | 66.1% | −10.6 [−12.0, −9.1] | 100% | 359 | 1,060 | 100% | 1 | 50 | 0 |
| reactive triggered | 75.9% | −0.8 [−1.4, −0.4] | 22.3% | 115 | 171 | 310 (6.1%) | 1.41 | 21.3 | 17 |
| learned triggered | 75.8% | −0.9 [−1.6, −0.3] | 27.9% | 187 | 249 | 492 (9.7%) | 1.54 | 26.5 | 8 |
| oracle (privileged, post hoc: always-on only on base failures) | 82.2% | +5.5 | – | 359 | 0 | 0 | – | – | – |

- Learned − reactive: −0.1 points [−0.7, +0.5]. Learned − always-on: +9.7 [+8.3, +11.0].
- Of the 1,537 base failures, learned triggering recovers 187 (12%), reactive 115 (7%), always-on 359 (23%).
- Intervention exits: learned 91 success, 1,830 timeout, 901 episode end; reactive 82 / 1,020 / 980. Recovery rarely brings the object to within 3 cm of the goal inside 20 steps.

Intervals: 95% percentile, 2,000 resamples. Episode seeds are the clusters: a resampled
seed brings every (policy, condition) episode under both configurations, so the paired
difference respects the shared initial states.

**By policy seed** (Δ success vs base, points): learned −0.7 [−1.8, +0.3], −1.3 [−2.4, −0.2],
−0.8 [−1.8, +0.2]; reactive −0.7 [−1.5, 0.0], −1.0 [−1.9, −0.2], −0.8 [−1.7, 0.0]. The pattern is
the same for all three policies.

**By disturbance type** (Δ vs base, learned / reactive, points): action delay +0.5 / 0.0;
action noise −2.9 / −2.1; action scale 0.0 / −1.1; goal shift −0.3 / −0.5; observation noise
−1.1 / −0.1; undisturbed 0 / 0 (1% activation). Full tables: `summary_by_*.csv`,
`differences_by_*.csv`.

Where recovery helps or hurts (success rate base → learned):
- **Helps:** full-vector observation noise 0.01 (20 → 34%, always-on 50%), action scale 0.5
  (93 → 96%), action scale 0.65 (69 → 72%), 3-step delay (12 → 14%).
- **Hurts:** action noise 1.0 (62 → 51%), position noise 0.02 (68 → 47%, with 80% activation),
  action scale 0.75–0.85 (46 → 43%, 19 → 15%). Always-on recovery is much worse than the base
  policy under action noise and late goal shifts (55 → 14% at step 40).

## Interpretation and limitations

- **The controller is the bottleneck, not only the trigger.** Under strong actuator weakness
  its reposition-then-push route is longer than the base policy's motion, and noisy commands
  break its behind-the-object alignment test, so taking over often *removes* the base policy's
  remaining chance. The +5.5-point oracle shows recovery is useful on a minority of failures;
  neither trigger can tell which.
- **Activation under observation noise is out of distribution for the learned trigger.**
  Online features come from the noisy view; the predictor was trained on ground-truth
  features. With 2 cm position noise it fires on 80% of episodes and lowers success by 21
  points. Using the ground-truth state would be an oracle baseline.
- **Timeouts dominate.** With `max_duration = 20` most interventions end by timeout, then hand
  a disrupted state back to the base policy.
- **Design choices made without these seeds but not tuned.** Controller constants were set by
  hand. One bug (pushing past the goal) was fixed after looking at 30 undisturbed
  episodes on seeds 3,600,000–029, which are therefore excluded. Further probing used pilot
  seeds. A 3-seed smoke run on the development seeds checked the pipeline; nothing was changed
  after it. The switching constants and thresholds were fixed before the run.
- These are development results on one seed range; the held-out evaluation was not run.

## Research decisions needed

1. Whether to improve the recovery skill (e.g. a recovery that continues the base policy's
   push instead of re-approaching, or a learned residual, as in the roadmap) before revisiting
   triggering; with the current skill no trigger can beat the base policy by much.
2. Operating point: 5% target FPR still gives 6–10% unnecessary interventions on base
   successes, each of which risks harm. A lower-FPR point, or a cost-aware rule that weighs
   lead time against harm, may be needed.
3. Whether the learned trigger should be retrained or evaluated on features from the
   noisy controller view, since that is what it receives online.
4. Recovery duration and return policy: timeout length, and whether to hand back at all.
