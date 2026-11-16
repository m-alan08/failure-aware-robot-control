# Phase F1: contact-aware motion prediction for delay compensation (pre-registration)

**Status:** pre-registered on 2026-11-16, before any F1 data collection, training or
evaluation episode.
- After training and model selection, the frozen predictor's SHA-256 and the final config
  are committed (the "design freeze"). Evaluation runs only from that commit.
- Any deviation is listed under "Amendments", with its date and reason, before evaluation.

## Question

Phase E's delay compensator predicts the state the frozen policy will face d steps ahead. It
shifts the gripper by K0 × ĝ × 0.05 × Σ(queued commands) and assumes the cube stays still and
all velocities are unchanged.

**Does predicting gripper *and* cube motion over the delay horizon improve closed-loop
success beyond that gripper-only predictor?** Prediction accuracy and control success are
measured and reported separately.

## Predictors compared

All of them share Phase E's latched online delay estimate, unchanged, and apply no gain
compensation. Only the prediction step differs.

| configuration | prediction | deployable? |
|---|---|---|
| `base_only` | none (frozen SAC) | yes |
| `phase_e_predictor` | Phase E: gripper shift, cube static, velocities unchanged | yes |
| `contact_learned` | learned one-step dynamics model rolled out over the queued commands: gripper and cube positions, cube rotation, all velocities | yes |
| `learned_gripper_only` | **ablation**: the same learned rollout, but the cube is held at its observed position and velocity | yes |
| `oracle_sim_predictor` | **PRIVILEGED, not deployable**: snapshots the simulator, executes the actual queued commands for d̂ steps through the real disturbance pipeline, reads the true state, then restores the snapshot | no |

- **Same horizon for everyone:** the oracle uses the *estimated* delay d̂ like the others, so
  its advantage is the prediction alone.
- **The oracle carries two privileges:** it starts from the true, noise-free state, and it
  uses the true dynamics, including contact.

## Learned predictor (lightweight; no world model or video model)

- **Model:** an MLP that maps the current observation's translation-invariant entries plus the
  executed command (3 position components) to the one-step change of:
  - gripper position, cube position, cube rotation;
  - gripper velocity, cube-relative velocity, cube angular velocity.

  Inputs: gripper velocity, cube position relative to the gripper, cube rotation, cube relative
  velocity, cube angular velocity, the gripper's height above the cube, and the command.
- **Rollout:** step it d̂ times with the queued commands scaled by the latched gain estimate
  (1 if none). Then rebuild a *consistent* observation: the cube's relative position is
  recomputed from the predicted positions, and finger entries are left unchanged.
- **Training data:** frozen-policy rollouts on **training seeds 3,650,000–199**.
  - Policies: the 3 frozen policies.
  - Conditions: undisturbed; delays 1–4; action scale 0.5.
  - Each condition is run with both `base_only` and the Phase E compensator, to cover the
    states the compensators visit.
  - Targets come from the true (noise-free) observations. Every transition is logged with
    the action actually executed.
- **Model selection:** **seeds 3,651,000–049**, same generation procedure.
  - Candidates: hidden width {64, 256} × weight decay {0, 1e-4}. Two hidden layers, Adam,
    a fixed 60 epochs, a fixed training seed.
  - Criterion: mean open-loop **cube + gripper position error over horizons 1–4** on the
    selection trajectories, contact states weighted ×2.
  - Prediction error only. No control outcome is used in selection.
- **Freeze:** the selected model is saved under `$FARC_DATA_DIR/models/contact_predictor/`,
  together with its normaliser and its SHA-256. The evaluation config pins that hash.

## Conditions (evaluation)

| # | condition | role |
|---|---|---|
| C0 | undisturbed | harm check |
| C1–C4 | constant delay 1, 2, 3, 4 | main |
| C5 | delay 3 from step 15 | no start-up cue |
| C6 | delay 3 + position noise 0.005 m | noisy predictor input |
| C7 | delay 3 + action scale 0.5 | gain loss, using the latched gain |
| C8 | position noise 0.01 m | harm check |

## Seeds

- **Training:** 3,650,000–199.
- **Model selection:** 3,651,000–049.
- **Debugging smoke test:** 3,652,000–009. Code checks only; outcomes are not analysed.
- **Evaluation:** 3,653,000–099. That is 100 seeds × 3 policies = 300 paired episodes per
  condition per configuration, for 13,500 episodes in total.
- All of these are disjoint from every earlier range (Phases 1–E2) and below 4,000,000. The
  loader enforces both.

## Contact states (for stratification; evaluation-only, from the true state)

A step t, with horizon h = the true delay, is a **contact state** if the gripper is within
5 cm of the cube horizontally at t **or** the true cube displacement over [t, t+h] exceeds
2 mm. Contact is never an input to any deployable predictor.

## Hypotheses and decision rules

Statistics throughout: a paired episode-seed cluster bootstrap with 10,000 resamples and
two-sided 95% CIs.

- **HP (prediction accuracy):** on contact states, `contact_learned`'s cube-position error at
  the true horizon, given the true executed queue, is lower than the static-cube assumption's.
  - Measured offline from logged `base_only` trajectories in C1–C7.
  - Unit: per-seed mean error. Statistic: paired difference.
  - Supported if the CI of (static − learned) lies above 0.
- **HC (control, PRIMARY):** `contact_learned` − `phase_e_predictor` success, pooled over the
  delay conditions C1–C7, is greater than 0.
  - **Improves** if the CI lower bound is above 0.
  - **Non-inferior** if the CI lower bound is above −3 points.
  - **Worse** if the CI upper bound is below 0.
- **Harm:** `contact_learned` − `base_only` on C0 and on C8 must each have a CI lower bound
  above −2 points, with at most 2% of base successes harmed.
- **HA (attribution, secondary):** `contact_learned` − `learned_gripper_only`, pooled over
  C1–C7. If HC holds but HA's CI includes 0, the gain is attributed to better *gripper*
  prediction, not to cube prediction.
- **HO (headroom, secondary):** `oracle_sim_predictor` − `phase_e_predictor`, pooled over
  C1–C7. If even the oracle does not improve control, better prediction is not the
  bottleneck.

## Secondary analyses

- Per-condition differences with Holm adjustment across C1–C7.
- Rescued and harmed counts; share of the oracle's improvement.
- Prediction error of gripper and cube by horizon, contact vs non-contact, and condition.
  This includes C6, where the input is noisy.
- Success differences in episodes with high vs low contact share, split at the median of the
  `base_only` contact-state share. Exploratory.

## Confounders and how they are handled

1. **Better gripper prediction, not cube prediction:** the `learned_gripper_only` ablation
   (HA).
2. **Velocity entries:** the Phase E predictor leaves velocities stale, while the learned one
   updates them. That is part of "better prediction"; the ablation updates gripper velocity
   but not cube velocity.
3. **Delay-estimation error** is shared by every configuration through the same latched
   estimator, so prediction is the only difference. The oracle uses d̂ as well.
4. **Distribution shift:** the model is trained on base and Phase E trajectories, then used in
   its own closed loop. This is reported through the closed-loop prediction error logged in
   `contact_learned` episodes.
5. **Observation noise:** deployable predictors receive the noisy view; the oracle does not.
   C6 isolates this.
6. **Ceiling:** in Phase E2 the compensator already reached 93–102% of the delay oracle,
   which also used the gripper-only predictor. The remaining failures may not be prediction
   failures at all. HO tests this.
7. **Policy sensitivity:** a more accurate predicted observation can still lead to a worse
   action, if the frozen policy relies on cues that the static-cube approximation happened
   to preserve. HP and HC are therefore reported separately.

## Failure modes reported regardless of outcome

- Rollout divergence: error that grows with horizon, or physically implausible cube jumps.
- Episodes harmed by `contact_learned` that `phase_e_predictor` kept, with their contact share.
- Episodes where the predictor acts in C0 or C8.

## Integrity

- **Frozen artifacts:** SAC actors are verified unchanged. The predictor is frozen before any
  evaluation episode and pinned by hash.
- **Raw files are write-once;** `--analyze-only` cannot touch them.
- **Run guards:** the runner refuses to start with uncommitted code, config or plan changes,
  or if this plan's hash differs from the one in the config.
- **Smoke test:** run on the debugging seeds before the evaluation. Only the logs and table
  shapes are inspected.
- **Held-out seeds (4,000,000+)** are not used.

## Amendments

(none)
