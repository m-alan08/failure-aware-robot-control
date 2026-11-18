# Phase J-1: privileged-input ablation, pre-registered protocol

**Status:** approved by the user on 2026-11-18 ("Approve J-1"), as specified in
`docs/PHASE_J_NEXT_EXPERIMENT_RECOMMENDATION.md` (SHA-256 `01b2c695…`). This document fixes the
operational details. It is committed, and its SHA-256 is pinned in
`configs/retrain/phase_j1.json`, **before** any J-1 episode runs.

## Rules

- **Evaluation only:** no training, fine-tuning or model selection.
- **Unchanged:** the existing I-A and I-C checkpoints are loaded read-only, as are frozen SAC
  and Candidate C. No checkpoint, raw file or earlier result changes.
- **Seeds:** debugging **3,700,200–209** (code checks only; outcomes not read); evaluation
  **3,703,000–3,703,299**, both new. Everything at or above 4,000,000 is refused;
  4,000,600–899 is untouched.
- **Privileged variants:** they read noise-free simulator positions. They are diagnostic only:
  **never deployable, never a baseline, never compared against Candidate C.**

## Configurations

All run through `ScheduledDelaySession`, so each pair shares the initial state, goal and noise
draws.

| name | policy input | extra features computed from |
|---|---|---|
| `base_only` | frozen SAC on the noisy observation | – |
| `IA` | I-A on the noisy observation | noisy positions (as deployed) |
| `IA_clean_features` | I-A on the **noisy** observation | **noise-free** positions (PRIVILEGED) |
| `IC` | I-C on the noisy observation | noisy positions (as deployed) |
| `IC_clean_estimate` | I-C on the **noisy** observation | **noise-free** positions feed the estimator (PRIVILEGED) |

I-C's command-history entries are noise-free in both I-C variants. Only the estimator's position
input differs.

## Cells

| cell | noise σ | delay |
|---|---|---|
| `H_noise0.01` | 1 cm, positions | none |
| `E_C8_noise0.01` | 1 cm, positions | none |
| `J_noise0.005` | 0.5 cm, positions | none |
| `H_undisturbed` | none | none |
| `E_C0_undisturbed` | none | none |

- Perturbation seed 20262100.
- Size: 5 cells × 5 configurations × 300 episode seeds × 3 policy seeds = **22,500 episodes**.

## Contrasts (paired, 95% episode-seed cluster bootstrap, 10,000 resamples, seed 20262100)

Pooled over the two 1 cm noise cells (`H_noise0.01` + `E_C8_noise0.01`). One-sided α = 0.025,
Holm across J1–J2.

| | contrast | test |
|---|---|---|
| **J1** | `IA_clean_features` − `IA` | > 0 |
| **J2** | `IC_clean_estimate` − `IC` | > 0 |

**Gate checks** (descriptive, the same rule as the Phase I noise gate): each variant − `base_only`
on the 1 cm cells, CI low > −2.

**Descriptive:** the same contrasts per seed and on the 0.5 cm cell; rescued and harmed counts.

## Integrity check (must hold, or the analysis stops)

On the two noise-free cells, each privileged variant gets identical inputs to its deployed
counterpart. Their **actions must be bitwise identical at every step**, and their outcomes
identical.

## Reading (fixed now)

| outcome | reading |
|---|---|
| J1 supported **and** `IA_clean_features` passes the gate check | feature noise is **sufficient** to explain I-A's noise-gate failure: H1 causal for I-A |
| J1 supported, gate check not passed | feature noise **contributes**, but the noise in the base observation (or general fragility) also matters |
| J1 not supported | no evidence at this sensitivity that feature noise causes I-A's degradation: favours H4/H5. **This is not proof of no effect.** |
| J2, read the same way | H2 for I-C |

## Logging (for H3, analysed later; no decision rests on it here)

**What is logged:** for the `IA`, `IA_clean_features`, `IC` and `IC_clean_estimate` episodes, every
step's:
- noise-free and noisy observation;
- goal;
- the extra features the policy received;
- the issued action.

**Where:** compressed `npz` files per task, in the write-once raw directory.
