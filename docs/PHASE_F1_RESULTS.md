# Phase F1 results: contact-aware motion prediction for delay compensation

## Provenance

- **Pre-registration:** `docs/PHASE_F1_PREREGISTRATION.md`, commit `eeda964`, written before
  any F1 data existed. SHA-256 `d17f8860…`.
- **Design freeze and evaluation run:** commit `4e1b574` (clean tree, enforced by the runner).
- **Frozen predictor:** SHA-256 `8a960329…`, an MLP with 2 × 256 tanh layers and no weight
  decay.
  - Trained on 352,800 transitions from training seeds 3,650,000–199.
  - Selected on seeds 3,651,000–049 by open-loop prediction error only
    (`results/phase_f1_dynamics/model_selection.csv`).
- **Evaluation:** seeds 3,653,000–099; 3 frozen SAC policies; 9 conditions × 5 configurations
  × 300 paired episodes = 13,500 episodes.
  - Tables: `results/phase_f1_eval/`. Raw files: `$FARC_DATA_DIR/experiments/phase_f1_eval/`,
    write-once.
- **Smoke test:** debugging seeds 3,652,000–009, checking table shapes only.
- **Not used:** Phase E results, unchanged; held-out seeds 4,000,000+.

```bash
uv run python scripts/train_contact_predictor.py                 # data + model selection + freeze (~8 min)
uv run python scripts/run_contact_evaluation.py --debug          # smoke test (debugging seeds)
uv run python scripts/run_contact_evaluation.py                  # pre-registered evaluation (~9 min)
uv run python scripts/run_contact_evaluation.py --analyze-only   # re-derive tables, raw files untouched
uv run python scripts/explore_f1_ablations.py                    # EXPLORATORY post-hoc ablations
```

## Pre-registered results

Statistics: paired episode-seed cluster bootstrap, 10,000 resamples, 95% CIs.

| test | estimate [95% CI] | pre-registered outcome |
|---|---|---|
| **HP**: cube-position error in contact states, static − learned | **−20.5 mm** less error [19.6, 21.4] (33.0 → 12.4 mm) | supported |
| **HC (primary)**: contact-aware − Phase E success, pooled C1–C7 | **+10.4 points** [+8.5, +12.3] | **improves** |
| Harm, C0 undisturbed (vs base) | 0.0, 0 harmed | passed |
| Harm, C8 noise 0.01 m (vs base) | 0.0, 0 harmed | passed |
| HA: contact-aware − learned gripper-only | +53.3 [+51.1, +55.5] | **not interpretable** (see below) |
| HO: oracle − Phase E | +12.7 [+10.9, +14.4] | secondary |

### Success rate by condition

| condition | base SAC | Phase E | **contact-aware** | ORACLE (privileged) | learned gripper-only (flawed) |
|---|---|---|---|---|---|
| C0 undisturbed | 99.7% | 99.7% | 99.7% | 99.7% | 99.7% |
| C1 delay 1 | 81.7% | 96.3% | 98.3% | 99.3% | 38.0% |
| C2 delay 2 | 25.3% | 87.0% | **97.0%** | 99.3% | 35.0% |
| C3 delay 3 | 15.3% | 84.7% | **96.3%** | 99.3% | 25.7% |
| C4 delay 4 | 10.7% | 74.0% | **93.3%** | 98.3% | 24.3% |
| C5 delay 3 from step 15 | 78.7% | 92.0% | 94.3% | 97.3% | 56.3% |
| C6 delay 3 + noise 0.005 m | 14.0% | 80.7% | **94.3%** | 98.7% | 25.7% |
| C7 delay 3 + scale 0.5 | 60.7% | 77.7% | **91.7%** | 88.7% | 87.3% |
| C8 noise 0.01 m | 95.7% | 95.7% | 95.7% | 95.7% | 95.7% |

**Contact-aware vs Phase E, by condition** (rescued / harmed; Holm-adjusted p):

| condition | difference [95% CI] | rescued / harmed | Holm p |
|---|---|---|---|
| C1 | +2.0 [−0.7, +4.7] | 10 / 4 | 0.33 |
| C2 | +10.0 [+5.7, +14.3] | 38 / 8 | < 0.001 |
| C3 | +11.7 [+7.3, +16.0] | 43 / 8 | < 0.001 |
| C4 | +19.3 [+13.7, +25.0] | 70 / 12 | < 0.001 |
| C5 | +2.3 [−0.7, +5.3] | 14 / 7 | 0.33 |
| C6 | +13.7 [+8.7, +19.0] | 53 / 12 | < 0.001 |
| C7 | +14.0 [+9.0, +19.3] | 50 / 8 | < 0.001 |

- **Pattern:** the gain grows with the delay horizon, from +2 at delay 1 to +19 at delay 4.
  That is where a static cube is most wrong.
- **Share of the oracle's gain:** contact-aware captures 82% of the oracle's improvement over
  Phase E (+10.4 of +12.7). In C7 (delay plus gain loss) it slightly exceeds the oracle, 91.7%
  vs 88.7%. I have not explained why the oracle does worse there.

### Prediction accuracy (separate from control)

**Offline, base-policy trajectories, at the true horizon with the true executed queue,
C1–C7:**

| | contact states (35,731) | free states (59,069) |
|---|---|---|
| cube error, static (Phase E) | 33.0 mm | **1.3 mm** |
| cube error, learned | **12.4 mm** | 4.8 mm |
| gripper error, Phase E (K0 shift) | 15.5 mm | 20.1 mm |
| gripper error, learned | **7.2 mm** | **7.5 mm** |

- **Contact:** the learned model is 2.7× more accurate for the cube in contact.
- **Free space:** it is *less* accurate for the cube than "the cube does not move" (4.8 vs
  1.3 mm). It learned a small spurious drift, and I did not correct it after the fact.
- **Closed loop:** inside each predictor's own closed loop, cube error in contact states at
  delay 3 is 11.2 mm (contact-aware) vs 36.1 mm (Phase E). At delay 4 it is 16.5 vs 46.2 mm.
- **Noisy input:** under 0.5 cm position noise (C6), learned cube error in contact rises to
  16.0 mm, still well below Phase E's 39.5 mm.

### Contact-heavy vs contact-light episodes (exploratory, pre-registered split)

The gain is the same in episodes above and below the median contact share (0.39): +10.4
[+7.8, +13.0] vs +10.5 [+7.9, +13.2]. The oracle shows the same pattern. So the benefit is not
confined to contact-heavy *episodes*. Per-step contact errors matter in almost every delayed
episode, because nearly all of them involve pushing.

## The pre-registered ablation (HA) failed as a design

- **What went wrong:** `learned_gripper_only` updated the gripper position *and velocity* from
  the model, but left the cube-relative velocity (cube velocity − gripper velocity) at its
  observed value. The policy therefore received an observation no physical state could
  produce.
- **Result:** success collapsed (C3: 25.7%) even though the variant predicts the gripper
  *better* than Phase E. That is confounder #7 from the plan (policy sensitivity to
  inconsistent observations) showing up in my own ablation.
- **Consequence:** HA cannot attribute the gain, and I do not report it as evidence either
  way.

### Exploratory follow-up (post hoc, fresh seeds 3,654,000–099; not confirmatory)

I ran two corrected ablations next to Phase E and contact-aware, on C1–C7 (2,100 paired
episodes):
- `learned_gripper_position_only`: Phase E with the gripper *position* taken from the learned
  rollout; cube static; all velocities as observed.
- `learned_gripper_consistent`: learned gripper position and velocity; cube static, with
  velocities consistent with a static cube.

| comparison (pooled C1–C7) | difference [95% CI] |
|---|---|
| contact-aware − Phase E (**replication**) | **+9.6** [+7.9, +11.3] |
| gripper-position-only − Phase E | +0.4 [−1.3, +2.2] |
| gripper-consistent − Phase E | +1.6 [−0.1, +3.3] |
| gripper-position-only − contact-aware | −9.1 [−11.0, −7.4] |
| gripper-consistent − contact-aware | −8.0 [−9.7, −6.4] |

**Exploratory reading:**
- The gain replicates on fresh seeds.
- Better gripper prediction alone does not improve control. Predicting the *cube's* motion is
  what helps.
- This was designed after seeing the confirmatory results, so it is supporting evidence only.

## Failure cases and limits

- **Not significant at short horizons:** delay 1 (C1) and delay from step 15 (C5), with
  +2.0 and +2.3 points.
- **Harm:** 59 episodes that Phase E solved were lost by contact-aware. Their contact share
  (median 0.40) is no different from the rest, so there is no evidence that contact
  prediction errors specifically cause them.
- **Free-space drift:** the learned model drifts the cube by about 5 mm in free space. It does
  not matter for control here, but a model used for planning would need to fix it.
- **Remaining gap to the oracle:** −2.3 points pooled, largest at delay 4 (93.3% vs 98.3%).
  Rollout error grows with horizon.
- **Scope:** everything is in simulation, with a pure transport delay and a nearly linear
  mocap plant.
  - The model was trained on base-policy and Phase E trajectories from the *same simulator*,
    and its contact dynamics are MuJoCo's.
  - Nothing here says it would transfer to hardware.
- **Not part of F1:** no RL, no new SAC, and no world or video model. A 2-layer MLP on 21
  numbers was enough.

## Recommendation

- **Contact-aware prediction** is a supported improvement over the gripper-only predictor:
  pre-registered, with a fresh-seed replication, no harm, and 82% of the oracle's gain.
  Use it as the model-based compensator in the next phase.
- **Before a controller-selection agent:**
  1. A pre-registered held-out confirmation, combined with E2's (only with your approval).
  2. A pre-registered attribution test using the corrected ablation design.
  3. Fixing the free-space drift, for example by gating cube motion on predicted contact. That
     would be a new variant needing its own evaluation, not tuning on these results.
- **Residual RL:** the remaining 2–5 points to the oracle at long delays are small. A learned
  residual would have to beat this controller, not frozen SAC.
