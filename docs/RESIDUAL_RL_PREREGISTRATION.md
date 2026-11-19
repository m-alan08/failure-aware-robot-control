# Residual RL on Candidate C: pre-registration (DRAFT, NOT APPROVED)

**Status:** this is a draft for review. **Nothing in it is approved, and nothing is trained.** No
seed below has been used. The design comes from `docs/RESIDUAL_RL_PROPOSAL.md`. Architecture:
`src/farc/residual/` (zero-residual equivalence: 1,230/1,230 bitwise on G4 debugging seeds,
proposal §1.5).

Once approved:
- `configs/residual/residual_rl.json` pins this file's SHA-256, and the runner refuses a changed
  plan.
- Any later change is an amendment, recorded **before** the affected step runs, in
  `docs/RESIDUAL_RL_AMENDMENTS.md`.

## 1. Questions

**Q1 (H-R).** Does a bounded residual on frozen Candidate C raise success under disturbances C
does not model, without nominal or delay losses?

**Q2 (H-R0).** Is any gain *generalisable* robustness, or only fitting of the training
disturbances, in particular the synthetic `replay_drop` delay?

**Q3 (attribution).** Does the residual need Candidate C (vs a residual on frozen SAC) and C's
internal signals (vs a residual without them)?

## 2. Frozen components (unchanged; verified by SHA-256 at start)

| component | identity |
|---|---|
| base policies | `fetch_push_fp32_seed{0,1,2}` (as in G4) |
| Candidate C | `candidates.Sequential`, Gaussian, h = 12, σ₀ = 4 mm, Phase E estimator constants as in `configs/delaychange/g4_heldout.json` |
| F1 predictor | `model.pt`, SHA-256 `8a960329…` |
| residual architecture | `src/farc/residual/` at the commit that approves this plan |

## 3. Disturbance families

**Training families.** The draw per training episode is uniform over families with the shares
below, with parameters drawn per episode:

| id | family | share | definition | status |
|---|---|---|---|---|
| T-N | nominal | 0.25 | none | exists |
| T-D | delay, `replay_drop` | 0.25 | G3 generator families RT / RI / RS / RP with delay ∈ {1..4}, **generator seed 20261900** (new) | exists |
| T-M | object dynamics | 0.20 | object mass × U[0.5, 2.0] and sliding friction × U[0.5, 1.5], fixed for the episode | **to implement** |
| T-O | correlated observation noise | 0.15 | AR(1) on gripper and object positions (mask `positions`): ρ = 0.8, stationary σ ~ U[0, 1 cm] | **to implement** |
| T-A | actuator | 0.15 | `action_noise` σ ~ U[0, 0.2] plus `action_scale` s ~ U[0, 0.4], whole episode | exists |

**Never used in training** (these test generalisation, Q2):

| id | family | definition |
|---|---|---|
| X-DH | delay with `drain_hold` semantics | G3 RT/RI/RP cells under `ScheduledDelaySession(..., "drain_hold")` |
| X-G | goal shift | the existing `goal_shift` disturbance, severity 0.05 m, onset step 20 |

**Implementation conditions for T-M and T-O** (amendment-free, but they must pass before
training starts):
1. **Nominal-identity check:** with zero severity they reproduce undisturbed episodes bitwise,
   on R debugging seeds.
2. **Effect check:** T-M changes the object trajectory under a fixed command sequence, and T-O
   noise has a lag-1 autocorrelation of 0.8 ± 0.05 on synthetic draws.

## 4. Arms and training

**Algorithm:** SAC (repository `retrain/qrl/algos/sac.py` hyperparameters unless listed), driving
`ResidualEnv`:

| setting | value |
|---|---|
| actor | `ResidualActor`, [256, 256], output layer zero-initialised |
| critic | twin Q, [256, 256] |
| learning rates | actor 3e-4, critic 1e-3, entropy-coefficient 1e-3 |
| gamma, tau, batch | 0.98, 0.005, 256 |
| replay | 1M transitions; HER off |
| target entropy | −8 (fixed, pre-registered) |
| critic warm-up | 20k steps of delta = 0 data, then 20k critic-only updates, before the first actor update |
| alpha schedule | linear ramp from 0.05 to the arm's target over the first 200k steps |
| reward | `RewardConfig()` defaults: sparse task reward, w_l2 = 0.1, w_rate = 0.05, no shaping |
| budget | **1M environment steps per run.** Not extended. |
| training seeds | episodes from the R `training` block 3,710,000–3,719,999 in order. Network seeds {0, 1, 2}. Each run uses one frozen SAC policy, cycling seed0/1/2 by network seed. |

**Arms** (3 network seeds each):

| arm | base | residual observation | trained on | alpha target |
|---|---|---|---|---|
| **R-α0.05, R-α0.1, R-α0.2** | C | full (65 dims) | all T families | 0.05 / 0.1 / 0.2 |
| A1 (no C signals) | C | task + actions + validity (41 dims) | all T | α* |
| A2 (no delay in training) | C | full | T without T-D (shares renormalised) | α* |
| A3 (delay only) | C | full | T-N 0.5, T-D 0.5 | α* |
| B2 (residual on SAC) | frozen SAC | task + actions + validity | all T | α* |

α* is the alpha selected in §5. **A1–A3 and B2 are trained only after α* is fixed**, so 9 + 12 =
21 runs.

**Nominal protection rule (applied during training, logged, not discretionary):**
- Every 50k steps, the actor is evaluated deterministically on 100 nominal episodes from the
  `selection` block.
- If success is more than 2 points below C's success on those seeds (computed once, before
  training), the run rolls back to the previous checkpoint and halves the actor learning rate.
- After a second rollback the run stops. It counts as **failed** and enters analysis as C (delta
  = 0), not as excluded.

## 5. Selection (selection block 3,721,000–099 only)

**Checkpoint:** the final checkpoint of each run. There is no early stopping beyond the rollback
rule.

**α\* rule (fixed in advance):**
1. Among the R-α arms that pass nominal non-inferiority on selection seeds (pooled across the 3
   network seeds, C+R − C ≥ −1 point), pick the one with the largest mean C+R − C on T-M ∪ T-O ∪
   T-A.
2. Ties within 0.5 points go to the **smaller** alpha.
3. If no arm passes, the experiment stops with a null result (§8).

**Selection cells:** one cell per T family (with T-D split into RT and RP), plus X-free nominal.
100 seeds × 3 SAC policies. X families are **not** evaluated at selection.

## 6. Development evaluation (evaluation block 3,722,000–299)

**Configurations:** B0 frozen SAC, B1 C, R-α\* (3 network seeds), A1, A2, A3, B2. Also
PRIVILEGED P (C with true delay and gain) on delay cells, for diagnosis only.

**Cells:**
- **D (delay):** G4 Arm R + Arm F cells, `replay_drop`;
- **M (not modelled):** T-M at 3 fixed severities (mass × 0.5 / × 2, friction × 0.5), T-O at σ =
  0.5 / 1 cm, T-A at (σ 0.1, s 0.2) and (σ 0.2, s 0.4);
- **X (never trained):** X-DH (RT_d2, RT_d4, RP_d4), X-G;
- **N (nominal):** undisturbed.

**Unit:**
- 300 episode seeds × 3 SAC policies;
- the residual effect is averaged over the 3 network seeds within each (policy, episode seed)
  pair before analysis;
- inference uses an episode-seed cluster bootstrap with 10,000 resamples and bootstrap seed
  20261900.

**Primary hypotheses** (fixed-sequence gatekeeping, one-sided α = 0.025; a step is tested only if
every earlier step passes):

| step | claim | contrast | criterion |
|---|---|---|---|
| 1 | nominal protection | R − C on N | non-inferior, margin −1 point |
| 2 | delay protection | R − C on D | non-inferior, margin −1 point |
| 3 | **efficacy (Q1)** | R − C on M | CI lower bound > 0 **and** estimate ≥ +2.0 (practically meaningful) |
| 4 | **generalisation (Q2)** | R − C on X | CI lower bound > 0 |

**Claim wording, fixed now:**

| steps passed | claim |
|---|---|
| 1–4 | "residual RL adds robustness beyond C, including to disturbance families not seen in training" |
| 1–3 only | "residual RL improves performance on the disturbance families it was trained on; generalisation not shown" |
| 1–2 only | null for Q1 |
| step 1 or 2 fails | "the residual degrades C"; reported as a negative result |

**Secondary (Holm across 5; descriptive if the primary fails):**
- S1: R − A1 on M (C signals used; Q3).
- S2: R − B2 on M (C needed; Q3).
- S3: R − A2 on D. A *non-positive* value with S4 positive supports H-R0 not being the source.
- S4: A3 − C on D, the delay-only upper bound.
- S5: R − C on D restricted to random on/off cells (RI), where frozen SAC beats C (G3).

**Safety and diagnostics, always reported:**
- harmed vs rescued episode counts per family;
- C's alarm and false-alarm rates under R vs C alone (closed-loop coupling, proposal §8.5);
- residual magnitude distribution;
- fallback counts;
- rollback events.

## 7. Held-out

**Not part of this plan.** If steps 1–3 pass, a separate one-shot held-out plan on 4,000,600–899
is written with R-α\* frozen, in the G4 style, and approved separately. Every loader refuses
seeds ≥ 4,000,000 (`check_r_seeds`, `ResidualEnv`).

## 8. Stopping rules

The experiment stops, and the result is reported as it stands, if any of these happens:
1. T-M / T-O implementation checks (§3) fail and cannot be fixed without changing their
   definitions. Changing a definition needs an amendment.
2. No R-α arm passes selection (§5.3).
3. More than one third of the R-α\* runs fail by the rollback rule.
4. The real-simulator zero-residual check fails at the approved commit. It is re-run on R
   debugging seeds 3,720,000–009 before training.

## 9. Compute (estimate)

- **Environment:** about 1 ms per step for the environment plus C and F1, measured (proposal
  §1.5), so 1M steps is about 17 CPU-minutes.
- **SAC updates:** 1M updates at batch 256 on CPU is about 2–3 h per run.
- **Total training:** 21 runs, about 50–65 CPU-hours, about 8 h wall-clock at 8 processes.
- **Selection:** about 9 runs × 7 cells × 300 episodes, under 10 minutes.
- **Development evaluation:**
  - 8 configurations × about 55 cells × 900 pairs × 3 network seeds for residual arms, so about
    0.7M episodes;
  - at the measured rate, roughly 1–2 h wall-clock.

## 10. Seeds (registered in `farc.supervision.seeds.R_SEED_BLOCKS`, unused)

| role | block |
|---|---|
| training episodes | 3,710,000–3,719,999 |
| debugging | 3,720,000–009 |
| selection | 3,721,000–099 |
| development evaluation | 3,722,000–299 |
| new G3-style generator seed (delay schedules) | 20261900 |
| bootstrap seed | 20261900 |

These blocks are disjoint from every registered block, from the G4 debugging block used for the
equivalence check, and from 3,704,300–599 (left free for a possible J arm). A test checks this.

## 11. Before execution (checklist)

1. Approve this plan; pin its SHA-256 in `configs/residual/residual_rl.json`.
2. Implement T-M and T-O; pass §3's checks on debugging seeds.
3. Implement the training runner (SAC on `ResidualEnv`), the selection script and the
   pre-written analysis, with unit tests, **before** any training seed is used.
4. Re-run zero-residual equivalence on R debugging seeds at that commit.
5. Train.
