# Held-out confirmation of Phases E and F1: results

## Run record

| item | value |
|---|---|
| Plan | `docs/HELDOUT_CONFIRMATION_PLAN.md`, approved 2026-11-16 (commit `f35ad5e`), SHA-256 `44f38eba…` |
| Code | `e97a3e9`, clean tree |
| Lock record | written 2026-11-16 10:35:38 UTC, before the first episode; no crash restarts |
| Seeds | held-out **4,000,000–4,000,299**, the only held-out seeds ever used. 4,000,300+ remain reserved. |
| Frozen | SAC `fetch_push_fp32_seed{0,1,2}`; Phase E estimator constants; F1 predictor `8a960329…` |
| Size | 9 conditions × 5 configurations × 900 paired episodes = 40,500 episodes |
| Raw files | `$FARC_DATA_DIR/experiments/heldout_confirmation_e_f1/` (write-once); tables in `results/heldout_confirmation_e_f1/` |
| Debug run | development seeds 3,655,000–009; only table shapes were checked |

The analysis is the pre-written code, run once. Nothing was changed after the run.

## Pre-registered gatekeeping (one-sided α = 0.025, fixed sequence)

Pooled over the delay conditions C1–C7 (6,300 paired episodes); 95% CIs from an episode-seed
cluster bootstrap.

| | hypothesis | estimate [95% CI] | bound | result | development estimate | shrinkage |
|---|---|---|---|---|---|---|
| H1 | Phase E compensation − frozen SAC | **+44.9** [+43.1, +46.7] | > +5 | **supported** | +43.7 | no |
| H2 | contact-aware − Phase E | **+9.3** [+8.3, +10.4] | > 0 | **supported** | +10.4 | no |
| H3 | contact-aware − gripper-only (corrected ablation) | **+8.7** [+7.7, +9.7] | > 0 | **supported** | +8.0 | no |

**Safety gates** (contact-aware vs frozen SAC; binding):

| condition | difference [95% CI] | harmed / base successes | passed |
|---|---|---|---|
| C0 undisturbed | 0.0 [0.0, 0.0] | 0 / 890 | yes |
| C8 noise 0.01 m | +0.1 [0.0, +0.3] | 0 / 860 | yes |

Phase E's own gates also pass (C8: 1 of 860 harmed).

## Descriptive results

**Success rates:**

| condition | frozen SAC | Phase E | **contact-aware** | gripper-only ablation | ORACLE (privileged) |
|---|---|---|---|---|---|
| C0 undisturbed | 98.9% | 98.9% | 98.9% | 98.9% | 98.9% |
| C1 delay 1 | 80.7% | 97.7% | 98.9% | 94.3% | 99.4% |
| C2 delay 2 | 29.2% | 89.3% | 98.1% | 89.9% | 99.6% |
| C3 delay 3 | 13.4% | 81.1% | 95.0% | 87.4% | 99.2% |
| C4 delay 4 | 10.1% | 77.6% | 93.6% | 80.0% | 99.1% |
| C5 delay 3 from step 15 | 74.9% | 91.7% | 95.2% | 90.2% | 97.9% |
| C6 delay 3 + noise 0.005 m | 13.1% | 82.0% | 93.4% | 84.8% | 98.8% |
| C7 delay 3 + scale 0.5 | 62.0% | 78.6% | 89.0% | 75.8% | 91.3% |
| C8 noise 0.01 m | 95.6% | 95.6% | 95.7% | 95.7% | 95.7% |

**Contact-aware vs Phase E, by condition** (rescued / harmed):

| condition | difference [95% CI] | rescued / harmed |
|---|---|---|
| C1 | +1.2 [+0.1, +2.4] | 20 / 9 |
| C2 | +8.8 [+6.7, +10.9] | 92 / 13 |
| C3 | +13.9 [+11.1, +16.8] | 156 / 31 |
| C4 | +16.0 [+12.8, +19.3] | 187 / 43 |
| C5 | +3.6 [+1.7, +5.4] | 54 / 22 |
| C6 | +11.4 [+8.4, +14.6] | 147 / 44 |
| C7 | +10.4 [+7.3, +13.7] | 154 / 60 |

- **By policy seed:**
  - H2: +8.8, +8.0 and +11.3, each with CI above 0.
  - H3: +9.3, +7.0 and +9.8, each with CI above 0.
  - H1: +44.7 to +45.3.
- **Share of the oracle's gain:** contact-aware captures **75%** of the privileged oracle's
  gain over Phase E (development: 82%). The oracle remains 3–6 points higher at delays of 3
  and 4.
- **Prediction accuracy** (offline, base-policy trajectories, true horizon):

  | | contact states (107,437) | free states (176,963) |
  |---|---|---|
  | cube error, learned | 12.6 mm | 4.8 mm (the drift seen in F1) |
  | cube error, static | 33.8 mm | 1.3 mm |
  | gripper error, learned | 7.2 mm | 7.4 mm |
  | gripper error, Phase E | 15.3 mm | 19.7 mm |

  All of these match development within 0.5 mm.

## Interpretation

On seeds no phase had seen, with everything frozen beforehand, all three claims hold:
1. **Online delay identification with model-based compensation** recovers most of the success
   that action delay destroys: +45 points over frozen SAC.
2. **Predicting the cube's motion** over the delay horizon adds about 9 points beyond the
   gripper-only predictor. The gain grows with delay length, from +1 at delay 1 to +16 at
   delay 4.
3. **That gain comes from cube prediction, not gripper prediction:** a consistent gripper-only
   variant using the same learned model is 8.7 points worse.

None of the effects shrank relative to development, and nothing harmed undisturbed or
noise-only episodes.

## Limits that still apply

- **Simulation only:** a pure transport delay of issued commands, MuJoCo contact, and a model
  trained in the same simulator. Nothing here establishes transfer to hardware.
- **Not perfect:**
  - Contact-aware still loses episodes that Phase E solves: 9–60 per condition, most under
    delay plus gain loss (C7: 60).
  - It leaves 3–6 points to the oracle at long delays.
  - The free-space cube drift (4.8 mm) is still in the frozen model.
- **Untested here:** transient delays (E2 showed no benefit there), a fast change detector,
  and residual RL. The held-out slice used here cannot be reused to tune any of them.
