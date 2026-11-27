# Residual RL: protocol amendments

Amendments to `docs/RESIDUAL_RL_PREREGISTRATION.md`. That file is **not edited**: its SHA-256
(`32330bd5…`) is still the one pinned in `configs/residual/residual_rl.json`. Each amendment is
recorded here **before** the step it affects runs.

## R-A1 (2026-11-27): critical pre-training corrections

**Approved by:** the project owner, in chat on 2026-11-27 ("Approve all critical"), for the
items marked critical in `docs/RESIDUAL_RL_PROTOCOL_REVIEW.md` §1.
**State when approved:** no residual-RL training had run, and no seed in any R block (registered
or proposed) had been used. The only simulator runs used unregistered seeds below 100,000.
**Plan approval:** R-A1 does **not** approve the pre-registration as a whole.
`approved` stays `false`, so registered training is still refused (see "Still open" below).

| id | change | where it is implemented |
|---|---|---|
| T-M-2 | T-M friction scales the sliding friction of the object **and the table** by f (`friction_mode="object_table"`). As written, object-only scaling below 1 does nothing at all: 20/20 episodes are bitwise identical to nominal, because MuJoCo takes the max of the two geoms' friction. | config key `tm_friction_mode = "object_table"`. `scripts/residual_train.py` reads it. Every evaluation call to `episodes.generate` must pass it. |
| SEED-1 | Training uses the registered block 3,710,000–3,719,999 and then **`training_2` = 3,730,000–3,739,999**, in order. That gives 20,000 episodes = 1M steps / 50. | `seeds.R_AMENDMENT_1_SEED_BLOCKS`, `training.registered_training_seeds`, `check_run` |
| SEED-2 | The nominal rollback rule (prereg §4) evaluates on **`training_monitor` = 3,740,000–3,740,099**, not on the §5 selection block. | `check_run` refuses selection seeds for monitoring. `scripts/residual_train.py` |
| STAT-1 | If step 1 or 2 fails and the upper 97.5% bound is ≥ 0, the claim is "non-inferiority not shown (inconclusive)". "Degrades C" needs an upper bound < 0. | `analysis.development_analysis` returns `claim`. The draft wording stays as `claim_draft_wording`, superseded. |
| STAT-2 | Inference is stated as **conditional on the 3 trained networks**. Step 3 (efficacy on M) also requires **each of the 3 per-network estimates to be > 0**. (The review's chosen option: 3 networks plus sign consistency. 5 networks were not chosen.) | `analysis.per_network_estimates`, `require_sign_consistency=True` (default), config `per_network_sign_consistency` |
| STAT-4 | The evaluation size follows a rule fixed now. The R-α\* vs C discordance on the §5 selection block (fraction of paired episodes with different outcomes) is measured **before** any evaluation seed is used. If it is ≤ 8%, evaluation uses 300 seeds (3,722,000–299). Otherwise it uses 600 (adding **`evaluation_extension` = 3,740,100–3,740,399**). | `analysis.discordance`, `analysis.evaluation_seed_count`, config `evaluation_rule` |

All three new blocks were checked against every block the repository registers or reserves
(`proposals.proposal_conflicts() == []`). They are below 4,000,000. The held-out seeds
4,000,000+ are untouched.

**Regression tests:**
- `test_amendment_r_a1_registers_the_proposed_blocks_unchanged`
- `test_registered_validation_logic_on_a_hypothetical_approved_copy`: selection seeds are refused for monitoring, and `training_2` is refused before `training`.
- `test_efficacy_needs_every_network_to_agree_in_sign`
- `test_evaluation_seed_count_rule`

### Still open after R-A1 (all resolved by R-A2 below)

Registered training stays refused until these are settled and the plan is approved:

- **Wording-level corrections** (review: "yes, wording"): T-M-1 (mass and inertia), T-M-3 (finger contacts keep max(f, 1)), STAT-5 ("primary fails" = step 3 not passed), SAC-1 (time-limit transitions stored as terminal; the code already does this), SAC-2 (rollback does not restore the noise RNG; the code already does this).
- **To confirm:**
  - SAC-3: critic warm-up targets come from the stochastic actor.
  - SAC-4: Q-target clipping off.
  - SAC-5: 1 update per step, initial entropy coefficient 1.0, log-std clamp [−5, 2].
  - B2-1: B2 rollback is compared with frozen SAC.
  - GEN-1: selection parameters are drawn per episode.
- **To decide:** GEN-2, whether `H_undisturbed` and `H_noise0.01` stay in group D.
- **Before evaluation:** GEN-3 (P config wiring).
- **The plan itself:** set `approved: true` with the SHA-256 unchanged.

Observation from the Stage A smoke (software only, not a result): at the start of training the
entropy bonus (coefficient ≈ 0.9 × entropy ≈ 2.7 per step) is much larger than the per-step task
reward (about −0.3 to −0.7). Soft-Q estimates are therefore positive (q_mean ≈ +4 after 950 steps)
while episode returns are negative. This is standard SAC behaviour, not a bug, but it bears on
SAC-5 (initial entropy coefficient 1.0, target entropy −8) and should be considered when that
item is confirmed.

## R-A2 (2026-11-27): remaining decisions and final pre-training validation

**Approved by:** the project owner, in chat on 2026-11-27. The decisions below are recorded as
they were given.
**State when recorded:** no residual-RL training run, selection or evaluation had happened. The
only simulator runs used unregistered smoke seeds (< 100,000) and the R debugging block
3,720,000–009. No evaluation outcome existed, so none could inform these decisions.

| id | decision | implementation |
|---|---|---|
| T-M-1 | accepted: T-M scales mass **and rotational inertia** by m (a uniform density change) | already the behaviour of `ObjectDynamicsPatch` |
| T-M-3 | accepted: finger–object contacts keep friction max(f, 1) under `object_table` | documented consequence of MuJoCo's max rule |
| STAT-5 | accepted: "primary fails" = step 3 not passed | `analysis.development_analysis` (`descriptive_only = passed < 3`) |
| SAC-1 | accepted: time-limit transitions are stored as terminal (no bootstrap) | `training.train` |
| SAC-2 | accepted: a rollback restores networks, optimisers and the entropy coefficient, not the noise generator, and does not rewind steps | `training.train` |
| SAC-3 | accepted: critic-only warm-up TD targets sample a′ from the current stochastic actor | `ResidualSAC.td_target` |
| SAC-4 | accepted: Q-target clipping off | `ResidualSAC` (no clip) |
| SAC-5 | accepted: 1 update per environment step; log-std clamp [−5, 2]. The initial entropy coefficient 1.0 was accepted provisionally and is **retained** after the development check below. | `TrainConfig` defaults |
| B2-1 | accepted: B2's rollback reference is frozen SAC's nominal success | `build().c_nominal` uses the arm's own core |
| GEN-1 | accepted: selection cells draw parameters per episode from the family's training distribution; RT/RP delay uniform on 1..4 | `episodes.selection_cells` |
| **GEN-2** | **H_undisturbed and H_noise0.01 are excluded from D.** D = the 39 G4 delay cells (Arm R: RT 2, RP 2, RI 2, RS 1; Arm F: F-T 18, F-P 6, F-I 4, NG 4). The two excluded cells are kept as **reporting-only** groups: `N_rep` (H_undisturbed, nominal) and `O_rep` (H_noise0.01, observation noise). They enter no primary or secondary contrast. Their R − C estimates are reported descriptively. N (the §6 `nominal` cell) is unchanged. **Weighting:** each D cell has equal weight within an episode-seed cluster (unchanged rule), now over 39 cells instead of 41. | `episodes.development_cells` (default `include_h_in_d=False`), `H_REPORTING_GROUPS`, `analysis.development_analysis` → `reporting` |
| **STAT-4 (revised)** | **Fixed evaluation budget of 600 seeds**: 3,722,000–299 followed by `evaluation_extension` 3,740,100–399. This replaces R-A1's adaptive 300/600 rule, which had not been applied: no selection discordance had been measured. | `episodes.evaluation_seeds`, `check_role_seeds("evaluation")` accepts exactly these 600 seeds |
| **GEN-3** | P (PRIVILEGED C: `mechanism.run_episode(mode="true")`, true delay and gain) runs on the **D cells only**, for diagnosis only. It is written to a separate table (`evaluation_privileged_P.csv.gz`) and never enters the analysis. | `scripts/residual_rollouts.py`, `tests/test_residual_rollouts.py` |

`analysis.evaluation_seed_count` (the R-A1 adaptive rule) remains in the code for the record. No
entry point calls it.

### Feasibility of 600 seeds (cost only; no outcome was used)

Rollout throughput was measured on a smoke rollout (2 smoke seeds, 53 cells, C and B0, plus P on
D): about 950 environment steps/s per core.

The §6 evaluation has:
- 53 cells (N 1, D 39, M 7, X 4, reporting 2);
- 17 rollouts per (cell, policy, seed): B0, C, and 3 networks each for R, A1, A2, A3 and B2;
- 3 policies;
- P on the 39 D cells.

At 600 seeds that is ≈ 1.69M episodes, or ≈ 85M steps, ≈ 25 CPU-hours. That is ≈ 3 h on 8
worker processes, against ≈ 1.5 h at 300 seeds. The extra cost is minor, so 600 seeds is adopted.

### Power audit under R-A2

`scripts/residual_power_audit.py --out results/residual_amendment_r_a2`: synthetic simulation
calibrated on the existing G4 confirmation table. That table is read-only, already analysed and
archived. No simulator episode was run and no new held-out seed was used. Settings: D = 39 cells,
C success 0.942 (G4 delay families, H excluded); M = 7 cells, 0.85; N = 1 cell, 0.997.

Rejection rate of H0: Δ ≤ −1 point (margin 0.01):

| group | true Δ | discordance | 300 seeds | 600 seeds |
|---|---|---|---|---|
| D | 0 | 0.05 | 1.000 | 1.000 |
| D | 0 | 0.10 | 0.700 | **0.935** |
| D | −1 point (type-I) | 0.03 / 0.06 / 0.11 | 0.015 / 0.025 / 0.020 | 0.035 / 0.028 / 0.010 |
| M | 0 | 0.10 | 0.620 | **0.928** |
| M | −1 point (type-I) | 0.03 / 0.06 / 0.11 | 0.018 / 0.018 / 0.018 | 0.030 / 0.038 / 0.020 |
| N | 0 | 0.0054 | 0.998 | 1.000 |

Run-level variance (STAT-2): when the 3 networks differ in true effect, the coverage of the
method-level effect is 0.94 / 0.33 / 0.13 at 600 seeds (net sd 0 / 0.5 / 1.0). At 300 seeds it
is 0.945 / 0.435 / 0.19. More seeds narrow the within-network interval without addressing
between-network variance. Hence the conditional-inference statement and the per-network
sign-consistency requirement (R-A1, STAT-2) are kept unchanged.

### SAC-5 bounded development check

`scripts/residual_dev_check.py --alpha 0.2 --steps 200000`:
- one R run at the **registered** sizes and schedule: 256×256, batch 256, 20k zero-data steps, 20k critic-only updates, α ramp over 200k, nominal check every 50k, target entropy −8, initial coefficient 1.0;
- α = 0.2, the largest registered α, which carries the most saturation risk;
- unregistered smoke seeds: training 10,000–13,999, monitor 200–299;
- network seed 0, policy 0.

This is a stability check, **not a result**. Report: `results/residual_dev_check/R_a0.2/report.json`.

| window (steps) | success | δ mean abs | δ saturated (≥ 0.99) | command clipped | entropy coef | entropy (target −8) | critic loss (mean / max) | q_mean |
|---|---|---|---|---|---|---|---|---|
| 0–20k (zero data) | 0.968 | 0 | 0 | 0 | – | – | – | – |
| 20–40k | 0.980 | 0.41 | 0.001 | 0.071 | 0.0007 | −5.8 | 2.64 / 16.5 (warm-up) | +4.3 → −2 |
| 40–60k | 0.980 | 0.39 | 0.004 | 0.074 | 0.0007 | −8.01 | 0.52 / 0.81 | −2.24 |
| 100k | 0.983 | 0.42 | 0.005 | 0.080 | 0.0006 | −8.08 | 0.45 / 0.58 | −2.42 |
| 160k | 0.970 | 0.43 | 0.006 | 0.096 | 0.0006 | −7.91 | 0.62 / 1.25 | −2.78 |
| 200k | 0.970 | 0.46 | 0.010 | 0.109 | 0.0006 | −8.00 | 0.77 / 2.07 | −2.97 |

Nominal checks (100 monitor seeds) at 50k, 100k, 150k and 200k: 0.98, 0.98, 0.98, 0.99, against
C's 0.98. All passed, with 0 rollbacks. Throughput was 383 steps/s (522 s for 200k).

**Assessment: no material instability.**
- **Entropy coefficient:** it falls from 1.0 to about 7e-4 within the first ~20k actor-update steps, and the policy entropy then tracks the −8 target to within ±0.1.
- **Soft-Q:** the early positive soft-Q values (a large entropy bonus) disappear once the coefficient falls.
- **Critic loss:** bounded, rising slowly with α (0.45 → 0.77).
- **Residual δ:** it uses about 40–46% of its range, with saturation at most 1%. The applied correction |a_final − a_C| grows with α (0.03 → 0.08).
- **Command clipping:** 7–11% of components, mostly where a_C is already near ±1.
- **Nominal success:** not degraded.

The initial entropy coefficient 1.0 is therefore retained. No other hyperparameter was changed
or searched. This check gives no evidence about efficacy, and the training-episode success
(≈ 0.97–0.98, flat) is not a result.

### Final pre-training validation (all at commit `9e78103`, unless noted)

| check | result |
|---|---|
| test suite | 654 passed |
| ruff check, ruff format, mypy | clean (204 files) |
| original pre-registration SHA-256 | `32330bd5…604a`, unchanged and pinned |
| effective protocol integrity | `check_approved` now also requires this file's SHA-256 (`amendments_sha256`). A changed or unpinned amendments file is refused (`test_changed_amendments_are_refused`). |
| entry points | `residual_train.py` (`check_run` → `check_approved`, training/training_2/training_monitor seeds); `residual_rollouts.py` (`check_role_seeds`: exactly the 100 selection or 600 evaluation seeds); `residual_select.py` and `residual_analysis.py` (`check_approved`; the analysis reads the margins, the practical threshold and sign consistency from the config). All use `tm_friction_mode = object_table` from the config. |
| zero-residual equivalence, R debugging seeds 3,720,000–009 | 1230/1230 episodes bitwise identical (commands and decisions) in both zero modes: `ZeroResidual` at α = 0, and the untrained actor at α = 0.2. Output: `results/residual_zero_equivalence_r_debugging/` |
| resume identity, real simulator | interrupted at the step-1000 check, resumed from step 500. Weights, critic target, entropy coefficient, replay RNG, seed cursor and log all identical. Output: `results/residual_resume_check/`. Synthetic regression tests cover resuming with and without a rollback, and refuse a changed configuration. |
| frozen components | `residual_train.py` raises if the frozen SAC actor or the F1 predictor (network and normalisation arrays) changed during training. `residual_rollouts.py` checks the SAC actor per task. Candidate C's code and the archived artefacts are not modified. |
| residual observation | 65 dims (task + action history + validity + C's delay, supervisor and prediction signals). A1/B2 use 41 dims. No disturbance label, schedule or simulator ground truth: the session schedule is read only by P, which is diagnostic. |
| held-out seeds | Every simulator seed in this phase is < 100,000 (smoke) or in 3,720,000–009 (R debugging). No seed ≥ 4,000,000 was used. The power audit reads the archived G4 confirmation table (episodes already collected and analysed in Phase G4), with no rerun. |
| GEN-3 | implemented and tested (`tests/test_residual_rollouts.py`): P rows only on D cells, true delay used, separate table |

### Approval

All the conditions set on 2026-11-27 are satisfied:
- the remaining amendments recorded;
- the pre-registration hash intact and the amendments integrity-checked;
- entry points consistent;
- full-state resume verified;
- identity, equivalence and regression checks passed;
- D and the analysis verified;
- no held-out access;
- GEN-3 implemented;
- historical artefacts and Candidate C unchanged.

The plan is therefore **conditionally approved for registered residual-RL training** (R-A2,
2026-11-27). `configs/residual/residual_rl.json` is set to `approved: true` with
`amendments_sha256` pinning this file.

This approval does **not** cover the held-out evaluation (prereg §7), which needs a separate
approval.

### Registered campaign order

1. Train the 9 R-α runs (α ∈ {0.05, 0.1, 0.2} × network seeds {0, 1, 2}).
2. Selection rollouts on 3,721,000–099, then `residual_select.py`, which fixes α\*.
3. Train the 12 A1, A2, A3 and B2 runs at α\*.
4. Evaluation rollouts on the fixed 600 seeds, then `residual_analysis.py`.

## Stage A and compute (2026-11-27; software validation only, not evidence)

Hardware: Apple M5, 10 cores, 16 GB, CPU PyTorch. No GPU, no paid compute.

| run | seeds | outcome |
|---|---|---|
| `residual_train.py --smoke --steps 2500` (16×16 nets) | smoke 0–49, monitor 100–104 | completed: 50 episodes, 3 checkpoints + final, 2 nominal checks passed (1.0 vs C 1.0), 0 rollbacks, 50 training draws over all 5 families (T-A 11, T-M 11, T-D 10, T-N 9, T-O 9) |
| `--smoke --steps 3000` | smoke 0–49 | stopped with `SeedsExhausted`, as designed: 3,000 steps need 60 episodes and the smoke list has 50. No seed was reused. |
| `residual_benchmark.py --steps 5000` (256×256, batch 256) | smoke 1000–1101, monitor 100–119 | 369 env steps/s including updates, so **≈ 0.75 h per 1M-step run** |

The registered design has 21 runs (prereg §4: 9 R-α, then 12 A1–A3/B2 runs after α\* is fixed).
That is ≈ 16–18 h sequentially on this machine, plus selection and evaluation rollouts. It is
feasible locally, with no reduction of the registered budget. The replay buffer at 1M
transitions × 65-dim observations is about 0.6 GB.

Logs are in `results/residual_stage_a/` (checkpoints are git-ignored).

Commands: see `scripts/residual_campaign.sh` (resumable driver, R-A2 campaign order).
