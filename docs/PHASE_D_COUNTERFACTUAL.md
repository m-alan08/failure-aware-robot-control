# Phase D: counterfactual residual-recovery feasibility (FetchPush-v4)

**Result: none of the 36 interpretable residual corrections improves success over continuing
the frozen base policy from the same state, and a placebo of random-direction nudges of the
same size rescues and harms exactly as often as the designed corrections.** Whatever rescues
small residuals produce here look like sensitivity of the outcome to *any* perturbation, not
corrective signal. This does not justify residual RL training with the current state
representation and intervention design (see the recommendation).

Code: `src/farc/counterfactual/` (`session.py`, `residuals.py`, `experiment.py`); script:
`scripts/run_counterfactual_experiment.py`; configs: `configs/counterfactual/`; tests:
`tests/test_counterfactual.py`; tables: `results/phase_d_{pilot,full,placebo}/`.

```bash
uv run python scripts/run_counterfactual_experiment.py --config configs/counterfactual/phase_d_pilot.json    # ~5 min
uv run python scripts/run_counterfactual_experiment.py --config configs/counterfactual/phase_d_full.json     # ~4.2 h, 3 processes
uv run python scripts/run_counterfactual_experiment.py --config configs/counterfactual/phase_d_placebo.json  # ~30 min
uv run python scripts/run_counterfactual_experiment.py --config <config> --analyze-only
```

Raw files (`branches.csv.gz`, `metadata.json`) are written once by a rollout run, which refuses
to replace them without `--overwrite-raw`. `--analyze-only` reads them and rewrites only the
derived tables and `analysis.json` (selected candidate, seed split, raw-file SHA-256), each via a
temporary file and rename, after computing everything in memory
(`src/farc/counterfactual/artifacts.py`). Rerunning the analysis on the existing raw files of
all three runs reproduces the committed tables byte for byte. The committed `metadata.json`
files predate this split and also hold the analysis summary.

## Exact counterfactual branching

`EpisodeSession` runs the Phase 1 step loop in two halves: `observe()` applies the goal shift
and observation noise for step t and evaluates the base policy; `act(command)` applies action
perturbations, clipping and `env.step`. A snapshot is taken between them and contains:

- the whole MuJoCo `MjData` (`mujoco.mj_copyData`: qpos, qvel, actuator state, mocap pose,
  warm start, time, …);
- `unwrapped.goal` and the `TimeLimit` step counter;
- deep copies of every perturbation object (random generator mid-stream, active and changed
  records, the goal-shift event), the issued-command history that `action_delay` replays, the
  last observation and the pending controller view, the online feature history, the step.

Restoring sets all of it back; nothing is re-derived from the episode seed. Verification:

- On 22 conditions × 2 pilot-range seeds, sessions reproduce `run_disturbed_episode` action for
  action, and restoring a snapshot (even after running a deliberately different branch)
  reproduces the base continuation bit for bit, including a SHA-256 of simulator state, goal,
  step counters, generator states and command history.
- Tests repeat this for 5 conditions (3-step delay, action noise, goal shift at 25 with
  branches at 24/25/26, full observation noise, transient action scaling) at 5 branch steps;
  check that noise generators are in identical states at every later step of two branches with
  different commands; and that the delay queue replays the branch's own commands.
- Every run re-checks branch-A replay on the first seed of every condition and policy and aborts
  on any difference. Branch A from a state on the base trajectory is the base episode itself.

Because both branches keep the same generators, they get the same future disturbance draws
(noise, goal-shift target); they differ only in what they command, and under `action_delay`
the queue replays each branch's own commands, which is the physical behaviour.

## Residual candidates

`a_t = clip(a_base,t + alpha * delta_t, -1, 1)` on the three position components; the finger
command is the base policy's. `delta_t` (components in [−1, 1]) uses only the controller's view
at step t (gripper, object, goal, base action), the same information as the base policy. With u
the horizontal unit vector from object to goal:

| Strategy | delta |
| --- | --- |
| goal_directed | u |
| alignment | steer the gripper onto the object→goal line at object height, clip((p − g)/0.05) |
| reduced | −a_base (the base action scaled by 1 − alpha) |
| preserve_push | min(max(a_base,xy · u, 0), 1) u (amplify the base's own push component) |
| random_direction (placebo, separate run) | a deterministic pseudo-random horizontal unit vector seeded by the observation |

alpha ∈ {0.25, 0.5, 1.0} × duration ∈ {3, 6, 12} steps = 36 candidates (placebo: alpha ∈
{0.25, 0.5} × duration ∈ {3, 6}). After the correction the base policy acts to step 50.

## Design

| Item | Value |
| --- | --- |
| Policies, conditions | `fetch_push_fp32_seed{0,1,2}`, the 21 Phase 1 FetchPush disturbances + undisturbed |
| Pilot | seeds 3,620,000–004 (1,750 branch states, 5 min); used only to check the pipeline and timing |
| Full study | seeds **3,621,000–059**, disjoint from Phase 1/B (3,500,000–099), Phase C (3,600,000–029, 3,610,000–099) and the pilot; config loader refuses those and ≥ 4,000,000 |
| Branch states | every episode at t ∈ {5, 15, 25, 35, 42} (early to late), plus the learned predictor's first warning step (Phase B unweighted, 5% threshold 0.8716) when it differs. Warning status is recorded at every state, so grid states give matched non-warning states at the same steps. States are not selected on outcome. |
| Size | 20,768 branch states, 747,648 branches (36 per state) |
| Selection / evaluation | episode seeds split 30 / 30 (`split_seed = 0`). The "selected" candidate is the best on the selection seeds; every reported number is on the 30 evaluation seeds (10,382 states, 2,610 base failures, 7,772 base successes) |
| Intervals | 95% percentile, 2,000 resamples, clustered by episode seed (a seed brings all its policies, conditions and states) |

## Results (evaluation seeds)

**Candidate-level feasibility** (counterfactual success difference B − A, points [95% CI]):

| Candidate | Δ success | Rescued (of base failures) | Harmed (of base successes) | Mean final-distance gain (m) | Mean ‖a − a_base‖ |
| --- | --- | --- | --- | --- | --- |
| goal_directed α 0.25, 3 steps (best on evaluation) | +0.3 [−0.4, +0.8] | 15.7% | 4.9% | +0.0004 | 0.22 |
| preserve_push α 0.5, 3 | +0.1 [−0.4, +0.6] | 9.1% | 3.0% | 0.0000 | 0.07 |
| reduced α 0.25, 3 | −0.0 [−0.7, +0.7] | 15.3% | 5.2% | +0.0004 | 0.26 |
| alignment α 0.25, 3 | −0.1 [−0.8, +0.5] | 15.2% | 5.3% | −0.0007 | 0.23 |
| **preserve_push α 0.25, 6 (selected on selection seeds)** | **−0.4 [−0.9, +0.1]** | 10.5% | 4.1% | +0.0001 | 0.04 |
| alignment α 1.0, 6 | −2.7 [−3.7, −1.6] | 18.3% | 9.7% | −0.006 | 0.77 |
| goal_directed α 1.0, 12 | −3.0 [−3.9, −2.1] | 16.3% | 9.5% | −0.004 | 0.86 |
| reduced α 1.0, 12 | −5.0 [−6.0, −3.9] | 12.3% | 10.8% | −0.005 | 1.08 |
| alignment α 1.0, 12 (worst) | −6.0 [−7.2, −4.8] | 17.4% | 13.9% | −0.015 | 0.77 |
| *placebo* random α 0.25, 3 | +0.0 [−0.6, +0.6] | 14.4% | 4.8% | +0.0006 | 0.22 |
| *placebo* random α 0.25, 6 | +0.0 [−0.5, +0.6] | 15.5% | 5.2% | +0.0003 | 0.22 |
| *placebo* random α 0.5, 3 | −0.2 [−0.7, +0.4] | 16.1% | 5.6% | −0.0001 | 0.42 |

All 36: `results/phase_d_full/evaluation_candidates.csv`. Only 4 candidates are positive
(at most +0.3 points), none with an interval above 0; 10 are significantly negative, mainly
α = 1 or 12 steps. On the selection seeds *every* candidate was negative; the least negative,
preserve_push α 0.25 / 6 steps (−0.7 [−1.3, −0.0]), is the "selected" candidate, and it is −0.4
on the evaluation seeds. The best evaluation-seed candidate was −0.8 on the selection seeds:
rankings among small candidates do not replicate across seed halves.

**Placebo.** The random-direction placebo, at the same magnitudes, rescues 14–16% of base
failures and harms 5–6% of base successes, indistinguishable from goal_directed at the same α
and duration (15.3–17.8% / 4.9–5.9%). The designed directions carry no measurable information
beyond "perturb the trajectory".

**By disturbance type, timestep, policy seed, warning status** (`evaluation_by_*.csv`): every
small candidate is within about ±2 points of zero in every disturbance type and policy seed,
with intervals covering 0 (e.g. goal_directed α 0.25, 3: action scale +0.9 [−0.1, +2.0], goal
shift +1.5 [−2.2, +5.5]; policy seeds −0.0, −0.5, +1.3). At the grid steps (1,980 states each)
small candidates are within ±1.3 points; goal_directed α 1, 12 steps is negative at every step.
At predictor-warning states (1,462) vs non-warning states (8,920): goal_directed α 0.25, 3
+0.5 [−1.0, +2.1] vs +0.2 [−0.4, +0.8].

## Which failures look recoverable (oracle analyses, privileged, not deployable)

These use counterfactual outcomes to pick, per state, whether *any* candidate succeeds.

| Group | Base success | Any of 36 succeeds | Base failures | Rescued by ≥ 1 candidate | Rescued by ≥ 18 of 36 |
| --- | --- | --- | --- | --- | --- |
| pooled | 74.9% | 87.1% | 2,610 | 1,274 (49%) | 326 (12%) |
| action delay | 39.3% | 69.4% | 619 | 307 | 55 |
| action noise | 90.5% | 97.9% | 219 | 171 | 71 |
| action scale | 68.4% | 78.0% | 898 | 273 | 45 |
| goal shift | 71.1% | 89.4% | 268 | 170 | 70 |
| observation noise | 78.9% | 91.1% | 600 | 348 | 81 |

- The 87% "any candidate" oracle is not evidence of recoverability: by the same count, 2,719 of
  7,772 base *successes* (35%) are broken by at least one candidate, and the placebo rescues as
  often as the designed candidates. Picking the best of 36 perturbations from a sensitive state
  selects luck.
- A stricter signal, a majority of candidates rescuing, holds for 12% of failed states: 24% at
  t = 5, 18% at 15, 14% at 25, 7% at 35, 3% at 42. These are mostly action-noise, goal-shift and
  observation-noise failures, where the base policy is close to succeeding.
- Persistent actuator failures (strong action scaling, 3-step delay) are rarely rescued by a
  majority: the base policy's commands are already weak or late, and a bounded additive
  correction suffers the same scaling or delay.
- Failed states at the predictor's warning steps are the least recoverable: 3% majority-rescued
  vs 21% at non-warning failed states. By the time the predictor warns at 5% FPR, the failure
  is usually structural.

## Candidate-level feasibility vs deployable performance

Everything above is candidate-level: the same fixed correction applied at a given state,
compared with continuing the base policy from that exact state. A deployable controller also
needs an online trigger (Phase C showed that triggering costs success through interventions on
episodes the base would solve); its performance would be at most what a candidate achieves at
the states it triggers on, minus harm on false triggers. No oracle-selected candidate is
reported as a controller.

## Recommendation on residual RL

**Not justified yet in this form.** Bounded, short residuals around the frozen policy, applied
at these states, show no exploitable signal: designed corrections equal a random placebo, and
the per-state "best residual" is dominated by outcome sensitivity. An RL residual trained on
this setup would mostly learn to exploit that sensitivity (high-variance credit assignment)
rather than a recovery behaviour. Before investing in RL:

1. Identify a recoverable failure class with a positive *majority* signal and a mechanism, e.g.
   action-noise and late goal-shift states near success; test residuals targeted at those.
2. Give the residual information the base policy lacks: the dominant failures are actuator
   weakness and delay, which no correction based on the current state can undo. Estimating the
   actuator gain or delay online (from commanded vs observed motion) and compensating, e.g.
   rescaling by 1/(1 − estimated weakness), is a concrete, testable hypothesis this
   infrastructure can evaluate counterfactually.
3. Measure outcome sensitivity directly (placebo rescue/harm rates per state) and use it as the
   baseline for any future residual result.

## Limitations

- One state representation (the base policy's observation), one family of interpretable
  residuals, fixed α and durations; a learned residual could in principle find structure these
  miss, but the placebo result gives no evidence that it exists here.
- The placebo was added after the main results were seen, as a control; its design (same
  magnitudes and durations as goal_directed) was fixed before any placebo outcome was looked
  at.
- 60 development seeds; no held-out evaluation was run.
