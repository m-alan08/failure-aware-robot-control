# Disturbance evaluation (research Phase 1)

> Migrated from m-alan08/quantrl-fetch `docs/DISTURBANCES.md` (branch `phase1-disturbance-eval`, last changed in `2613e3e`). Results and commit hashes below refer to runs made in quantrl-fetch; the same numbers are regenerated here by `scripts/reproduce_phase1.sh` and checked against `results/`. Paths were updated to this repository.

This framework evaluates the frozen fp32 SAC + HER policies under controlled disturbances. It does not train anything. The code is in `src/farc/disturbances/`, the runner is `scripts/run_disturbance_eval.py`, and the configs are in `configs/disturbances/`. Training, the existing evaluation, HER, the reward, the Q-target clipping, and the checkpoints are unchanged.

## What the controller sees and what the metrics use

| Signal | Controller (policy input) | Metrics |
| --- | --- | --- |
| `observation` | Simulator value, plus observation noise when active | Not used |
| `desired_goal` | The goal active in the simulator (the new goal from the shift step on); noise only if `fields` includes `desired_goal` | `env.unwrapped.goal` at each state, ground truth |
| `achieved_goal` | Not used by the policy | Simulator value returned by `env.step`, ground truth |
| Success | Not used | `info["is_success"]` after each step, checked at every step against `‖achieved − active goal‖ < 0.05` |
| Reward | Not used | Sparse −1/0 from the environment, checked to equal −(not success) |

The policy is the deterministic actor output `tanh(mean)`, the same as the existing evaluation. Its normalizer is a deep copy wrapped in `FrozenNormalizer`: `update` raises, and the statistics, the actor parameters, and the checkpoint file's SHA-256 are checked as unchanged after every run. A test confirms that an undisturbed rollout reproduces `farc.base_policy.reference_eval.run_episode` exactly (success, return, final distance).

## Indexing

- State 0 is the observation after `reset`; state k (1 to 50) is the observation after env step k.
- A disturbance with `onset = o` first affects the decision taken at state o, which is env step o + 1.
- `duration = d` affects decisions o to o + d − 1. `null` means until the episode ends.
- The 50-step horizon is kept. The rollout raises if the episode terminates or truncates at any other step.

## Perturbations

| Type | Severity | Effect while active |
| --- | --- | --- |
| `action_scale` | s in (0, 1] | executed = (1 − s) · command (all 4 dims; the gripper dim is ignored because Fetch Reach and Push block the gripper) |
| `action_noise` | σ > 0 (action units; 1.0 = full range, 5 cm of mocap motion) | command + σ·ε |
| `action_delay` | k ≥ 1 steps | executed = base command from decision t − k (zeros before the episode starts) |
| `observation_noise`, `mask: "full"` | σ > 0, raw units on every entry | controller's copy of `fields` + σ·ε (the original behaviour, kept for comparison) |
| `observation_noise`, `mask: "positions"` | σ in metres | gripper and object Cartesian positions + σ·ε; the object's relative position is recomputed from the noisy positions |
| `observation_noise`, `mask: "velocities"` | σ in m/s | linear velocities (gripper, object relative to gripper, finger joints) + σ·dt·ε, because the vector stores velocity × dt |
| `goal_shift` | minimum displacement in metres | At decision `onset`, the goal is replaced by a draw from the environment's own `_sample_goal()`, redrawn until it lies at least σ from the current goal (rejection sampling, at most `max_attempts` draws). Persistent; `duration` must be null. |

- Masked observation noise only touches the `observation` field; the validator rejects a mask combined with `desired_goal` noise.
- Action perturbations are applied in a fixed order, delay → scale → noise, and then clipped to [−1, 1]. The commanded (base), pre-clip, and executed actions are all logged.
- The environment would also clip, so clipping here first makes the logged executed action the one actually applied.

## Observation layout

Indices come from `generate_mujoco_observations` and `_get_obs` in the installed gymnasium-robotics 1.4.2 (`gymnasium_robotics/envs/fetch/fetch_env.py`). `farc.envs.observation_layout.validate_layout` checks them against the simulator's site positions and velocities at the start and end of every episode, and a test checks them while the arm moves. The control period is dt = 20 substeps × 0.002 s = 0.04 s.

| Component | FetchReach | FetchPush | Quantity | Unit in the vector | Position mask | Velocity mask |
| --- | --- | --- | --- | --- | --- | --- |
| gripper_position | 0:3 | 0:3 | grip site world position | m | yes | |
| object_position | | 3:6 | object site world position | m | yes | |
| object_relative_position | | 6:9 | object − gripper position | m | derived | |
| finger_position | 3:5 | 9:11 | finger slide joints (constant: gripper blocked) | m | | |
| object_rotation | | 11:14 | object Euler angles | rad | | |
| object_relative_velocity | | 14:17 | (object − gripper) linear velocity × dt | m per step | | object noise − gripper noise |
| object_angular_velocity | | 17:20 | angular velocity × dt | rad per step | | (excluded: different unit) |
| gripper_velocity | 5:8 | 20:23 | grip site linear velocity × dt | m per step | | yes |
| finger_velocity | 8:10 | 23:25 | finger joint velocities × dt | m per step | | yes |

Every mask uses the same standard-normal draw per entry and step, so the position entries get identical noise under `full` and `positions` at the same σ (tested).

## Randomness

- Episode seeds are shared by every condition (paired design).
- Each perturbation has its own generator from `SeedSequence(perturbation_seed, spawn_key=(episode_seed, type id, position))`. Noise is drawn at every step, active or not, so:
  - the draws do not depend on onset or duration, or on which other perturbations are present (tested);
  - the same standardized noise is reused across severities (common random numbers), so severity comparisons are paired.
- A goal shift samples with its own generator by temporarily swapping the environment's `_np_random` and restoring it. The environment's random stream is not advanced (tested).

Seed ranges:

| Range | Use |
| --- | --- |
| 1,000,000+ | existing training-time and final evaluation |
| 2,000,000+ | PTQ/QAT calibration |
| 3,000,000 to 3,499,999 | disturbance pilots (`stage: "pilot"`; the pilot used 3,000,000 to 3,000,019) |
| 3,500,000 to 3,999,999 | validation: baseline characterization and design decisions (`stage: "validation"`) |
| 4,000,000 to 4,999,999 | reserved for the final held-out test (`stage: "heldout"`); never used so far |

The validator rejects seeds outside the stage's range.

## Effect verification

Every perturbation records, per decision, whether it was active and whether it changed its signal. The rollout raises `PerturbationEffectError` in three cases:
- a perturbation changed its signal outside its window;
- an active perturbation never changed its signal;
- a goal shift moved the goal less than its severity.

## Episode metrics (definitions)

- **Final success:** success after step 50. This is the existing evaluation's definition and is unchanged.
- **First success step:** the first k in 1 to 50 with success against the goal active at state k.
- **Regime.** A condition is *permanent* if any of its disturbances lasts to the end of the episode (every goal shift is permanent) and *transient* if all of them have a duration.
- **Recovery** uses a reference state r:
  - permanent: r = onset. The outcome is *success reached under the disturbance*.
  - transient: r = onset + duration, the state produced by the last disturbed decision. The outcome is *recovery after the disturbance has ended*.
- **Eligibility.** Only episodes that are failing at r are eligible (`recovery_eligible`). Success is judged against the goal active right after r, which for a goal-shift onset is the new goal. Episodes already successful at r are not counted as recoveries; their `recovered` is empty.
- **Recovered:** there is a first state k > r with success against the goal active at k. The recovery time is k − r env steps.
- **Censoring.** Eligible episodes with no such state are right-censored at 50 − r steps and get no recovery time. The summary's time-to-recovery median is the Kaplan–Meier median, which is empty when the survival curve never drops to one half.
- **`success_at_onset`:** whether the robot was already within 5 cm of the active goal at onset.
- **Transient-only counts:** `lost_success_in_window` (successful at onset, unsuccessful at some state inside the window) and `retained_success_through_window` (successful at onset and at every state in the window).
- The pilot (before this definition) counted every episode from onset. Its committed summary was produced with that earlier definition and is kept unchanged.
- **Goal shifts:** success against the old goal never counts. `success_at_old_goal_after_shift` records it for diagnosis only, and `distance_to_original_goal` is logged at every state.

**Alternatives not chosen** (reversible, documented so they can be revisited):
- Recovery requiring sustained success to the end of the episode. This is available as final success.
- Observation noise in normalized units (σ × the normalizer's std per dimension). The masks use physical units instead.
- Independent noise on `object_relative_position`. It is derived from the noisy positions instead, as a position sensor would produce it.

## Outputs (written outside the repository, default `$FARC_DATA_DIR/experiments/<name>/`)

| File | Contents |
| --- | --- |
| `manifest.json` | the configuration and its fingerprint; a resumed run must match it |
| `parts/<env>_seed<k>/<condition>/` | `episodes.csv`, `trajectories.csv.gz` (one row per state), `distance_curve.csv`, and `done.json`, which is written last. `--resume` skips parts that have `done.json`. |
| `episodes.csv` | every episode (training seed, condition, final and ever success, distances, recovery fields, goal-shift details) |
| `summary_per_seed.csv` | per training seed × condition: success with an episode-bootstrap CI, distances, recovery among eligible episodes with CI and KM median, and paired lost or gained successes against the undisturbed run on the same seeds |
| `summary.csv` | pooled over training seeds: the same, plus per-seed min/max, hierarchical (seed, then episode) bootstrap CIs, and the paired success difference against undisturbed with a hierarchical CI |
| `distance_curves.csv` | per seed × condition: mean, median, and quartiles of the distance to the active goal at every state |
| `metadata.json` | git commit and dirty flag, packages, checkpoint SHA-256, normalizer hash, the original-evaluator check, timings |
| `success_overview.png`, `recovery_overview.png`, `distance_trajectories.png` | plots |

Every run re-runs the repository's original evaluator (`evaluate_policy`) on the undisturbed episode seeds of every checkpoint and stops unless per-episode success, return, and final distance are identical.

## Pilot (exploratory, not evidence)

```bash
uv run python scripts/run_disturbance_eval.py --config configs/disturbances/pilot.json
```

Setup:
- 20 episodes per condition (seeds 3,000,000 to 3,000,019), seed-0 fp32 policy per task, perturbation seed 20261114.
- 25 conditions per task, with severity grids fixed before running.
- Runs on CPU in about 30 s. Output is about 6 MB, mostly the compressed trajectories.

The compact summary is copied to `results/phase1_pilot/summary.csv`.

Caveats:
- With 20 episodes, a 95% interval on a success rate is about ±20 points. Only one training seed per task was used.
- **Observation noise in raw units is dominated by near-constant dimensions.** Several observation entries (gripper finger state and velocities) have a normalizer std at the 0.01 floor, so 5 mm of raw noise is 0.5 normalized std there, far outside anything seen in training. A one-off diagnostic (not part of the framework) on the same seeds:
  - noise on gripper and object positions only: Reach 100% at 0.005 m and 90% at 0.02 m; Push 100% and 95%;
  - noise on the other dimensions only: Reach 80% and 35%; Push 80% and 5%.
  - The pilot's observation-noise rows therefore measure sensitivity to out-of-distribution noise on non-position channels more than to position-sensing error.
- **FetchReach goal shifts give identical numbers for every onset.** The new goal is drawn per episode seed (common random numbers), and the arm is already parked at the goal by step 10, so the post-shift trajectory does not depend on when the shift happens.

## Extended baseline characterization (validation range)

```bash
uv run python scripts/run_disturbance_eval.py --config configs/disturbances/phase1_extended.json
# after an interruption, the same command with --resume continues at the condition level
```

Setup:
- Training seeds 0, 1, 2 for both tasks, 100 episodes per condition per training seed (episode seeds 3,500,000 to 3,500,099, shared by every condition and training seed), perturbation seed 20261115.
- 20 conditions for FetchReach and 22 for FetchPush; 12,600 episodes in total. The grid in `configs/disturbances/phase1_extended.json` was fixed before the run and not changed afterwards.
- The masked-noise comparison rows (full 0.01 raw; velocities 0.05 and 0.25 m/s) are supplementary.
- Runtime: 322 s of single-threaded CPU compute on an Apple M5 (about 5.5 min wall). Output: 77 MB in `$FARC_DATA_DIR/experiments/phase1_extended/`.
- Provenance: the episode data came from commit `fad4381`. The summaries were re-aggregated from the saved parts at `fed638e`, which only changed row order; both runs were on a clean tree.
- Checks: every checkpoint's undisturbed episodes were identical to the original evaluator on all 100 seeds. Every goal shift was drawn from the training goal range, moved the goal by at least 0.101 m, and needed at most 4 draws.

Compact copies: `results/phase1_extended/summary.csv` (pooled), `results/phase1_extended/summary_per_seed.csv`, `results/phase1_extended/distance_curves.csv`, `results/phase1_extended/metadata.json`, and plots in `docs/figures/phase1_*.png`.

Final-step success, in percent. The first number is pooled over 300 episodes; the bracket is the 95% seed-then-episode bootstrap interval; the per-seed values for seeds 0, 1 and 2 follow. Undisturbed is 100 for every seed of both tasks.

| Condition | FetchReach | FetchPush |
| --- | --- | --- |
| action scale 0.5 (whole episode) | | 89.0 [80, 96]; 92 / 95 / 80 |
| action scale 0.65 | | 70.3 [64, 77]; 71 / 74 / 66 |
| action scale 0.75 | 100; 100 / 100 / 100 | 50.3 [41, 59]; 53 / 56 / 42 |
| action scale 0.85 | | 19.7 [15, 24]; 20 / 20 / 19 |
| action scale 0.9 | 100; 100 / 100 / 100 | |
| action noise 0.3 | 98.0 [96, 100]; 98 / 99 / 97 | 98.7 [97, 100]; 99 / 98 / 99 |
| action noise 0.6 | 86.7 [82, 91]; 84 / 89 / 87 | 86.7 [83, 91]; 88 / 86 / 86 |
| action noise 1.0 | 65.3 [58, 72]; 61 / 70 / 65 | 65.3 [59, 71]; 68 / 66 / 62 |
| action delay 1 step | 74.0 [62, 87]; 70 / 64 / 88 | 84.7 [75, 92]; 88 / 91 / 75 |
| action delay 3 steps | 8.0 [5, 12]; 10 / 6 / 8 | 13.3 [9, 18]; 12 / 16 / 12 |
| goal shift ≥ 0.10 m at step 25 | 100; 100 / 100 / 100 | 96.7 [94, 99]; 96 / 97 / 97 |
| goal shift ≥ 0.10 m at step 40 | 100; 100 / 100 / 100 | 60.0 [41, 74]; 39 / 71 / 70 |
| position noise 0.005 m | 99.3 [98, 100]; 98 / 100 / 100 | 99.0 [97, 100]; 98 / 99 / 100 |
| position noise 0.01 m | 97.0 [93, 100]; 93 / 99 / 99 | 97.3 [95, 99]; 98 / 98 / 96 |
| position noise 0.02 m | 87.0 [79, 94]; 79 / 89 / 93 | 73.0 [58, 84]; 83 / 79 / 57 |
| action scale 0.5, steps 15 to 24 | 100 | 99.0 |
| action scale 0.85, steps 15 to 24 | 100 | 99.3 |
| action noise 0.6, steps 15 to 24 | 100 | 99.3 |
| action noise 1.0, steps 15 to 24 | 100 | 98.3 |
| (suppl.) full-vector noise 0.01 raw | 74.3 [63, 90]; 67 / 90 / 66 | 20.7 [8, 37]; 16 / 38 / 8 |
| (suppl.) velocity noise 0.05 m/s | 100 | 99.3 |
| (suppl.) velocity noise 0.25 m/s | 97.0 | 95.0 |

Recovery among eligible episodes, pooled (`summary.csv` has the intervals, KM medians and per-seed values):
- Permanent disturbances on FetchReach: every eligible episode reaches the goal under every disturbance except a 3-step delay (84%). The final-step failures above are therefore failures to stay inside 5 cm, not failures to reach it.
- Permanent disturbances on FetchPush, success reached under the disturbance:
  - action noise 1.0: 88%;
  - 1-step delay: 99%;
  - action scale 0.75: 49%; action scale 0.85: 14%;
  - 3-step delay: 23%;
  - goal shift at step 40: 63% (KM median recovery time not reached: too little time remains).
  - The 21 FetchPush episodes whose block starts within 5 cm of the goal are excluded at onset 0.
- Transient disturbances (steps 15 to 24): undisturbed FetchPush reaches the goal at a median of step 10 (90th percentile 14), so at most 39 of 300 episodes were failing when the window ended.
  - Of those, 88 to 90% recover after action noise, and all 6 to 14 recover after action scaling.
  - On FetchReach, noise knocks 223 (σ 0.6) and 278 (σ 1.0) of 300 episodes out of the 5 cm radius during the window. Only 43 and 106 are still outside when it ends, and all of them recover within a median of 1 step.
  - Action scaling at steps 15 to 24 cannot move a FetchReach arm that is already resting at the goal, so no episode is eligible.

Observations, not tests of significance:
- **The full-vector noise artefact is confirmed.** 0.01 raw units on every entry gives 74% on Reach and 21% on Push, against 97% and 97% for the same σ on positions only.
- **Training seeds differ most under goal shift at step 40 and 2 cm position noise on FetchPush** (39 to 71% and 57 to 83%), and under a 1-step delay on both tasks. Pooled intervals there are wide.
- **FetchReach under a 3-step delay oscillates around the goal** without settling (see `distance_trajectories.png`).
