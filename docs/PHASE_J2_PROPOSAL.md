# Phase J-2: noise-augmented fine-tuning (proposal)

**Status:** proposal only, for approval. Nothing has been implemented, trained or run, and no
seed in the blocks below has been used.

**Unchanged:**
- Candidate C, the frozen SAC policies, and the Phase H, Phase I and J-1 checkpoints and results.
- **Held-out 4,000,600–899** stays untouched.
- PRs #8–#19 stay unmerged.

This trains **new** policies, so it needs your explicit approval (§9).

## 1. Motivation (from Phase J and J-1)

| finding | source |
|---|---|
| I-C fails the 1 cm noise gate, and the failure **replicates** on fresh seeds: −3.2, then −2.8 [−5.1, −0.4] | Phase I, J-1 |
| I-A's noise-gate failure did **not** replicate (+0.9 [−1.1, +3.0] on J-1 seeds). Its gate status is borderline. | J-1 |
| Removing feature noise did not help I-A (J1 −2.0). A noise-free estimator input helped I-C by +2.3 [+0.2, +4.4], below the pre-registered threshold. | J-1 |
| The best-supported explanation is H4/H5: fine-tuned policies are fragile to perturbations absent from training. Neither Phase I's nor Phase H's training included observation noise. | Phase J, J-1 |

**The question:** does adding observation noise to fine-tuning, *and changing nothing else*, make
the Phase I baselines robust to observation noise without costing nominal or delay performance?

## 2. Design principle: one change from Phase I

Everything is identical to Phase I except that training observations receive position noise.
This makes the comparison with I-A and I-C a clean test of noise augmentation.

| item | Phase I | J-2 |
|---|---|---|
| arms | I-A, I-C | **J2-A** (I-A features), **J2-C** (I-C features) |
| initialisation | warm start from a copy of frozen `fetch_push_fp32_seed{k}` | **the same** (not from the Phase I checkpoints, to avoid building on evaluated models) |
| algorithm, hyperparameters, budget | upstream SAC+HER, 1M steps, `random_steps = 0` | **the same** |
| delay distribution | 1/3 nominal, persistent, transient (generator seed 20262000) | **the same distribution,** with a new generator seed 20262200 |
| normaliser | frozen original statistics, extras clipped | **the same** |
| **training observation noise** | none | **per episode σ ~ U[0, 1.0 cm], positions only**, through the existing `ObservationNoise` model. Gripper and object positions are noised; the relative position is recomputed; velocities are not noised; the noise is independent per step. |

**How the noise enters training:**
- It is applied to the observation the agent sees, before the features and the policy.
- Desired and achieved goals, and therefore HER relabelling and rewards, are untouched, as in
  evaluation.
- The noise random stream is seeded from (20262300, run seed, episode).

## 3. Evaluation design

**De-duplicated cells (fixes the Phase H/I erratum):**
- The 44 distinct Phase H/I cells: the 50 cells minus the duplicates `E_C0`, `E_C8` and
  `E_C1`–`E_C4`. G2's `P_d1`–`P_d4`, `H_undisturbed` and `H_noise0.01` are kept.
- Plus two new noise cells, with no delay:
  - `J_noise0.005`, inside the training range;
  - **`J_noise0.015`**, **outside** it, so the result is not simply fitting the training noise.
- A test will assert that no two cells define the same episodes.

**Configurations (7):**
- references: frozen SAC, contact-aware, Candidate C;
- Phase I I-A and I-C (registered, read-only);
- J2-A and J2-C.

**Seeds and size:**
- Seeds: new development block **3,704,000–3,704,299** × 3 policy seeds; debugging
  3,700,300–309.
- Size: 46 cells × 7 configurations × 900 = **289,800 episodes**, one run.

**Statistics, as before:**
- paired 95% episode-seed cluster bootstrap, 10,000 resamples, seed 20262200;
- one-sided α = 0.025; Holm within each family;
- minimum meaningful difference 2 points.

**Scopes:**
- **1 cm noise:** `H_noise0.01`.
- **Persistent:** RP ∪ F-P ∪ `E_C5`–`E_C7`. De-duplicated; `E_C1`–`E_C4` are dropped as copies
  of `P_d1`–`P_d4`.
- **Short and on-off:** RT ∪ RI.

## 4. Gates (evaluated first, per arm; fixed now)

These are the same rules as Phase I, on de-duplicated cells:

- **G-nominal:** arm − frozen SAC on `H_undisturbed` has CI low > −2, **and** every seed's
  undisturbed success is ≥ 95%.
- **G-noise:** arm − frozen SAC on `H_noise0.01` has CI low > −2.

**An arm that fails:**
- is reported as "baseline not competent" for the failed component;
- its delay comparisons are descriptive only;
- it is not eligible for Stage 3.

## 5. Hypotheses

### 5.1 Primary (Holm across N1, N2, D1, D2)

| | question | contrast | scope | test |
|---|---|---|---|---|
| **N1** | Does noise augmentation improve I-A under noise? | J2-A − I-A | 1 cm noise | > 0 |
| **N2** | Does it improve I-C under noise? | J2-C − I-C | 1 cm noise | > 0 |
| **D1** | Does it preserve I-A's persistent-delay performance? | J2-A − I-A | persistent | non-inferiority, margin −2 |
| **D2** | The same for I-C | J2-C − I-C | persistent | non-inferiority, margin −2 |

**Power note:** N1 has low power, because I-A's noise deficit was small and did not replicate. A
non-significant N1 would therefore be **inconclusive, not evidence of no effect.** N2 targets
the replicated I-C deficit.

### 5.2 Secondary (Holm across S1–S8; per arm, labelled with that arm's gate status)

| | contrast | scope |
|---|---|---|
| S1 / S2 | J2-A / J2-C − C | persistent (margin −2) |
| S3 / S4 | J2-A / J2-C − frozen SAC | short and on-off (margin −2) |
| S5 / S6 | J2-A / J2-C − C | short and on-off (> 0) |
| S7 / S8 | J2-A / J2-C − frozen SAC | `J_noise0.015`, out of training range (margin −2) |

### 5.3 Descriptive

- `J_noise0.005`;
- noise and gain loss (NG);
- per delay length;
- per seed;
- training curves;
- compute;
- J2 vs Phase I on undisturbed episodes.

### 5.4 Reading

| outcome | reading |
|---|---|
| N2 holds and J2-C passes both gates | noise exposure in training fixes I-C's noise fragility; the Phase I failure was H4 |
| N2 fails | noise augmentation alone does not fix I-C. H2 (estimator abstention) or broader fragility remains. A recurrent or estimator-robust design would be next. |
| D1 or D2 fails | robustness is bought at a delay-performance cost; this is reported as a trade-off |
| An arm passes both gates | its S1–S6 contrasts against C and frozen SAC become interpretable. **They describe this baseline only,** not delay-aware RL in general. |
| S7 or S8 fails | robustness does not extend beyond the training noise range |

## 6. Stopping rules

- **Full budget, final checkpoint;** no early stopping or selection.
- **Crashes:** a crashed run restarts once from scratch with an identical config. If it crashes
  again, it is reported missing, never replaced.
- **No tuning after the run:** no retuning, budget extension, or change to noise range,
  thresholds or cells after Stage 1 or 2.
- **Stage 3** (held-out) only for arms passing both gates, with your separate approval. A Stage 3
  guard for 4,000,600–899 would be added only then.

## 7. Implementation outline (after approval)

| item | change |
|---|---|
| wrapper | the Phase I `PhaseIWrapper` plus an `ObservationNoise` step on the observation before the featuriser, in the training role only. It reuses `masked_noise`, so training noise matches the evaluation noise model exactly. |
| Stage-0 tests | <ul><li>with σ = 0 the wrapper is bitwise identical to Phase I's;</li><li>the training noise statistics match the evaluation model (std, independence, relative-position consistency), checked on debugging seeds;</li><li>the warm start is bitwise equal to the frozen policy (as Phase I);</li><li>the evaluation cell list contains no duplicates.</li></ul> |
| records | numbered amendments committed before training and before evaluation; a separate checkpoint manifest; write-once raw files |

## 8. Compute

Estimates use Phase I's measured throughput.

| item | wall-clock |
|---|---|
| Stage 0 (tests, 20k-step smoke runs) | about 5 min |
| Stage 1: J2-A + J2-C, 3 seeds each (6 parallel runs, about 1.5 h each) | **about 1.5–2 h** |
| Stage 2: 289,800 episodes | **about 25–30 min** |
| **Total** | **about 2–2.5 h** (about 10 CPU-hours) |

## 9. Decisions requiring your approval

1. **Train new policies** (J2-A and J2-C), warm-started from copies of the frozen weights.
2. **Arms:** both (recommended), or J2-C only. J2-C targets the replicated failure; dropping J2-A
   halves training.
3. **Training noise:** σ ~ U[0, 1.0 cm] per episode, positions only, plus the out-of-range
   evaluation level of 1.5 cm.
4. **Unchanged from Phase I:** gates, margins and the 2-point minimum.
5. **Seed blocks:** debugging 3,700,300–309, evaluation 3,704,000–299, plus new generator and
   noise seeds.
6. **The de-duplicated cell set** (§3).
7. **Stage 3:** decided separately after Stage 2.
