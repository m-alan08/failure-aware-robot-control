# Phase E2: pre-registered evaluation of online delay compensation (PLAN, not yet run)

Status: **draft for approval.**
- Nothing in this document has been run.
- No E2 seed has been touched.
- Once approved, this file is committed before any E2 code is used on selection or evaluation
  seeds, and the commit hash is recorded in the run metadata.
- Any later change to the plan is listed under "Amendments", with its date and reason.

## Question

The Phase E pilot (PR #6) found that online delay identification plus a Smith-style predictor
raised success under a constant, zero-initialised 3-step delay from 10% to 79%. That case
gives the estimator an unusually easy start-up cue: the arm does not move for the first d
steps.

E2 asks whether the same frozen estimator and compensator, with no retuning, still improve
success when the delay:
- starts mid-episode;
- has an untested length;
- is combined with noise or gain loss;
- changes during the episode.

## What is frozen (from the pilot, unchanged)

- **Estimator constants** (`configs/adaptation/phase_e_pilot.json`):
  - K0 = 0.62;
  - window 10, lags 0–4 (raised to 0–5 only for the 4-step condition; see "Planned code
    changes");
  - minimum 4 transitions, excitation threshold 2.25e-4, minimum R² 0.8, lag margin 2.0.
- **Compensator:** latched reliable estimate. The primary configuration is
  `estimated_delay`, with no gain compensation.
- **Frozen policies:** `fetch_push_fp32_seed{0,1,2}`, verified against their manifest
  SHA-256.
- **No selection stage:** nothing is tuned. Every evaluation seed counts.

## Conditions

| # | condition | role | new code? |
|---|---|---|---|
| C0 | undisturbed | harm check | no |
| C1 | delay 1, from step 0 | replication of the pilot | no |
| C2 | delay 3, from step 0 | replication of the pilot | no |
| C3 | delay 2, from step 0 | delay length the pilot did not test | no |
| C4 | delay 4, from step 0 | delay length the pilot did not test (needs lag range 0–5) | config only |
| C5 | delay 3, starting at step 15 | **no start-up cue**: the replayed commands are real ones | no |
| C6 | delay 3, steps 15–29 only | transient delay; tests the latched estimate after it ends | no |
| C7 | delay 3 + position noise 0.005 m | identification under noise | combined condition |
| C8 | delay 3 + action scale 0.5 | delay combined with gain loss | combined condition |
| C9 | delay 1 → 3 at step 20 | delay that changes mid-episode | new perturbation type |
| C10 | position noise 0.01 m | harm check under noise | no |

## Configurations

| configuration | role |
|---|---|
| `base_only` | reference |
| `estimated_delay` | **primary** |
| `estimated_combined_f0.5` | secondary |
| `oracle_delay` | privileged: true delay at each step (true gain only in C8) |
| `sham_delay_fixed2` | sham: predicts a fixed 2-step delay, gated like the primary |
| `sham_delay_shuffled` | **new sham**: same gating; uses a delay estimate taken from a different, randomly paired episode in the same condition |

`sham_delay_shuffled` keeps the distribution of delay values but breaks their link to the
current episode. Its pairing is fixed by a seeded permutation that is decided before the run.

## Seeds and size

- **Debugging:** 3,640,000–009. Used only to make the new code work. Outcomes are not
  analysed.
- **Evaluation:** 3,641,000–099, i.e. 100 seeds × 3 policies = 300 paired episodes per
  condition per configuration.
- Both ranges are in the validation stage and disjoint from every earlier range, Phase E
  pilot included. Nothing ≥ 4,000,000 is used.
- **Expected precision:** the pilot's cluster-bootstrap CI half-width was about 8 points on
  90 episodes. With 300 episodes it should be about 4–5 points, enough to resolve the
  primary effect of 10 points below.
- **Run size:** about 11 × 6 × 300 = 19,800 episodes, roughly 10 minutes on 3 processes.

## Primary hypothesis and decision rule

- **H1:** `estimated_delay` − `base_only`, pooled over the delay conditions C1–C9, is greater
  than 0.
- **Test:** a paired episode-seed cluster bootstrap (10,000 resamples), one-sided α = 0.025
  (equivalently, the lower end of the two-sided 95% CI is above 0).
- **Supported** only if all of the following hold:
  1. The lower bound of H1's CI is above **+5 points**, a minimal practically useful effect.
  2. **C5 (no start-up cue)** on its own has a CI lower bound above 0. If C5 fails, the pilot
     result is attributed to the start-up cue, whatever H1 shows.
  3. **No harm:** on C0 and C10, the CI lower bound is above −2 points and no more than 2% of
     base successes are harmed.
  4. **Specificity:** `estimated_delay` beats `sham_delay_shuffled` pooled over C1–C9, with
     the CI of the paired difference above 0.

## Secondary analyses (reported, not used to declare success)

- Per-condition effects with 95% CIs (Holm-adjusted across C1–C9), rescued vs harmed, and
  the share of the oracle's improvement captured.
- Identification:
  - delay accuracy when the estimator reports confidence, and unconditionally;
  - how many steps it takes to detect an onset (C5, C6, C9) and to recover after the delay
    ends (C6) or changes (C9);
  - how often it abstains;
  - wrong-lag activations, and their outcomes.
- By policy seed; activation rate; saturation rate.
- Exploratory, and labelled so: a breakdown by base policy outcome, and
  `estimated_combined` vs `estimated_delay`.

## Failure modes to report regardless of outcome

- Latched delay estimates that stay active after the delay has ended (C6, C9).
- Wrong predictions under contact: when the gripper is within 5 cm of the object, compare
  outcomes on episodes the base policy would solve.
- Episodes where the compensator acts in C0 or C10.

## Planned code changes (before the debugging seeds)

1. A `max_lag` override for C4. This is the only estimator change, chosen in advance to fit
   the delay range, not tuned.
2. Combined conditions (C7, C8), built as multi-disturbance `ConditionSpec`s in the E2 config
   loader. Disturbance streams stay independent, as they already are.
3. A `delay_schedule` perturbation for C9. It replays issued commands with a step-dependent
   delay, has unit tests, and is checked with the existing perturbation-effect checks.
4. An `oracle_delay` source and the `sham_delay_shuffled` source.
5. Results go to `results/phase_e2_delay/` through the write-once raw-file mechanism.
   `--analyze-only` cannot touch raw files.
6. A test that pins this document's SHA-256 in the E2 config. The run refuses to start if the
   plan changes without a recorded amendment.

## Out of scope

- RL training.
- Gain compensation as a primary question.
- Held-out seeds 4,000,000+: these stay reserved for a final, separately approved evaluation.

## Amendments

(none)
