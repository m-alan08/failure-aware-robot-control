# Failure-prediction dataset (FetchPush)

Dataset infrastructure only; no model is defined here. Code: `src/farc/prediction/`;
report: `results/failure_dataset/report.json` (`scripts/build_failure_dataset.py`).

## Source and schema

Source: `$FARC_DATA_DIR/experiments/phase1_extended/parts/FetchPush-v4_seed<k>/<condition>/`
(validation seeds 3,500,000–3,500,099; 22 conditions × 3 policy seeds × 100 episodes). No new
rollouts were run; held-out seeds (4,000,000+) are refused by the loader.

FetchPush observation (25-d, validated against the simulator each episode, see
`farc/envs/observation_layout.py`): gripper pos 0:3, object pos 3:6, object−gripper 6:9,
fingers 9:11, object rotation 11:14, object−gripper velocity 14:17, object angular velocity
17:20, gripper velocity 20:23, finger velocity 23:25. `achieved_goal` = object position,
`desired_goal` = goal (3-d). Action (4-d): gripper dx, dy, dz, finger.

The full observation is not logged. `trajectories.csv.gz` has rows t = 0..50 with
ground-truth `achieved_0..2` (object), `gripper_0..2`, `goal_0..2` (goal active when state t
was observed), and from t ≥ 1 `base_action_*`/`executed_action_*` of step t−1, and `success`.

**Goal timing.** A shift for step s is applied before the decision at s (the policy sees it),
but the log shows it from row s+1. Features at decision t use `goal[t+1]`, which is the goal
the controller already knew at t.

## Features (24, decision steps t = 0..49)

| Index | Name | Definition |
| --- | --- | --- |
| 0–2 | object_to_goal_xyz | goal(t) − object(t) |
| 3–5 | gripper_to_object_xyz | object(t) − gripper(t) |
| 6–8 | object_velocity_xyz | object(t) − object(t−1), m/step |
| 9–11 | gripper_velocity_xyz | gripper(t) − gripper(t−1), m/step |
| 12–15 | previous_action_0..3 | base (commanded) action of step t−1 |
| 16 | object_to_goal_distance | ‖0–2‖ |
| 17 | gripper_to_object_distance | ‖3–5‖ |
| 18 | remaining_time | (50 − t)/50 |
| 19 | object_goal_progress | (d(t−k) − d(t))/k, d = object-goal distance, k = window |
| 20 | approach_progress | same for gripper-object distance |
| 21 | progress_window | k/K (K = 5) |
| 22 | time_since_goal_change | (t − t_goal)/50 |
| 23 | has_previous | 0 at t = 0, else 1 |

k = min(K, t − t_goal): the window is reset when the goal changes (both progress features by
default) and truncated at the episode start; with k = 0 progress is 0. At t = 0 velocities
and the previous action are 0.

Excluded on purpose: executed actions (encode the disturbance, e.g. 0.5 × base under action
scaling), `active_*` flags, condition, observation error, rewards, success. Features use
ground-truth state; under observation noise the controller saw a noisier view that was not
logged, so those failures are visible only through their effects.

## Labels, splits, normalization

`label` = 1 if the episode fails at the final step (`final_success` False, cross-checked
with the last log row), repeated for all 50 steps. Metadata per example: `episode_id`, `t`,
`condition_id`, `policy_seed`, `episode_seed`.

Splits group by `episode_seed` (same initial state across all 66 condition/policy episodes):
80 train / 20 validation seeds, `split_seed = 0`. Normalization (mean/std, std ≥ 1e-6) is fit
on training rows only.

## Numbers (split_seed 0)

| | Episodes | Successful | Failed | Examples | Failure fraction |
| --- | --- | --- | --- | --- | --- |
| All | 6,600 | 5,145 | 1,455 | 330,000 | 22.0% |
| Train | 5,280 | 4,107 | 1,173 | 264,000 | 22.2% |
| Validation | 1,320 | 1,038 | 282 | 66,000 | 21.4% |

Duplicate audit: 6,600 unique trajectories (no exact duplicates). 55,896 examples share an
identical feature row with another episode (prefixes before a disturbance onset, or
disturbances with no effect yet); none of these span episode seeds, so none cross splits.
462 episodes (7 seeds) start inside the success region. The undisturbed condition has no
failures; failures concentrate in action_delay_sev3, action_scale_sev0.85, full-vector
observation noise and goal_shift at step 40, so report per-condition metrics.
