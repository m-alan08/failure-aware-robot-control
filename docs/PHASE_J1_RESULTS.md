# Phase J-1: privileged-input ablation results (development, evaluation only)

## Run record

| item | value |
|---|---|
| Protocol | `docs/PHASE_J1_PROTOCOL.md` (SHA-256 `b9e7a81a…`), pinned in `configs/retrain/phase_j1.json`; approved 2026-11-18 |
| Code | `782d258`, clean tree |
| Seeds | development **3,703,000–299** × 3 policy seeds. Debugging 3,700,200–209 was used for code checks only. No held-out seed was used. |
| Size | 5 cells × 5 configurations × 900 = 22,500 episodes, one run (2 minutes). Per-step inputs were logged for the four Phase I configurations (write-once `npz`). |
| Integrity | **passed**: on both noise-free cells, the privileged variants' actions were bitwise identical to the deployed ones at every step (12 of 12 checks) |
| Unchanged | the I-A and I-C checkpoints (read-only), frozen SAC, Candidate C |

`IA_clean_features` and `IC_clean_estimate` are **PRIVILEGED diagnostic variants**. They are
never deployable and never baselines.

**Duplicate cells** (see the erratum below): `H_noise0.01` and `E_C8_noise0.01` turned out to be
identical episodes, as do `H_undisturbed` and `E_C0`. The pooled 1 cm estimates and CIs below
equal the single-cell values, because the cluster bootstrap resamples whole episode seeds.
**Counts are halved here,** to count each distinct episode once.

## Pre-registered contrasts (1 cm noise; Holm across J1–J2)

| | contrast | estimate [95% CI] | counts (distinct episodes) | Holm p | result |
|---|---|---|---|---|---|
| **J1** | `IA_clean_features` − `IA` | **−2.0 [−3.8, −0.2]** | 30 / 48 | 0.99 | **not supported** |
| **J2** | `IC_clean_estimate` − `IC` | +2.3 [+0.2, +4.4] | 60 / 39 | 0.033 (raw one-sided 0.017) | **not supported** at the pre-registered level |

**Gate checks** (each vs frozen SAC, 1 cm noise, CI low > −2):

| | estimate [95% CI] | passes the noise gate |
|---|---|---|
| I-A | **+0.9 [−1.1, +3.0]** | yes |
| I-A, clean features | −1.1 [−3.3, +1.1] | no |
| I-C | **−2.8 [−5.1, −0.4]** | no |
| I-C, clean estimate | −0.4 [−2.4, +1.7] | no (narrowly) |

**Per seed** (descriptive; per-seed intervals are wide, about ±3.5):

| seed | I-A − SAC | I-C − SAC | clean − deployed, I-A | clean − deployed, I-C |
|---|---|---|---|---|
| 0 | −0.7 | −3.3 | 0.0 | 0.0 |
| 1 | −1.0 | −4.0 | −4.0 [−7.3, −0.7] | +0.7 |
| 2 | **+4.3** | −1.0 | −2.0 | **+6.3 [+2.7, +10.0]** |

**At 0.5 cm noise** (pooled): I-A − SAC −1.0 [−2.0, −0.1]; I-C − SAC −0.7 [−1.4, +0.1].
Clean inputs made no measurable difference there.

## Reading, as pre-registered

- **J1 not supported.** There is no evidence that I-A's displacement-feature noise causes its
  degradation; removing it made I-A *worse* (−2.0).
  - Per the protocol, this **favours H4/H5 over H1** for I-A. It is not proof of no effect.
  - **Caveat I did not anticipate in the protocol:** `IA_clean_features` pairs noise-free history
    features with a *noisy* base observation. That combination is inconsistent and was never
    seen in training, so it is itself out of distribution, which may explain why it does worse.
    The ablation therefore cannot cleanly isolate H1.
- **J2 not supported** at α = 0.025 after Holm.
  - The direction (+2.3) and seed 2 (+6.3) are *consistent with* the estimator's noise
    sensitivity contributing to I-C's degradation (H2).
  - But it is not established, and even with a clean estimate I-C does not pass the gate.

## New facts this run adds

1. **I-A's Phase I noise-gate failure did not replicate.** On fresh development seeds, I-A is
   **+0.9 [−1.1, +3.0]** against frozen SAC and passes.
   - In Phase I it was −2.0 [−4.1, +0.1], failing by a margin of about 0.1 points.
   - Together, the I-A noise gate is **borderline and seed-block-sensitive.**
   - **The Phase I result stands as recorded:** it was pre-registered and run once. It should not
     be read as a robust I-A defect.
2. **I-C's noise fragility replicated:** −2.8 [−5.1, −0.4], against −3.2 in Phase I.
3. **Under these interventions, neither tested feature mechanism is causal.** H1 for I-A is
   contradicted as a cause. H2 for I-C is suggestive, but below the pre-registered threshold.

## Erratum: duplicate cells in Phase H and Phase I (found in this run)

**What happened:** the Phase H and Phase I evaluations each contain six pairs of conditions that
define identical episodes, with the same disturbance, seeds and perturbation seed:
- `H_undisturbed` = `E_C0_undisturbed`;
- `H_noise0.01` = `E_C8_noise0.01`;
- `P_d1`–`P_d4` = `E_C1`–`E_C4`.

This was verified on the raw files: every outcome is identical.

**Impact:**
- **Gates** (no-delay and noise cells): estimates and cluster-bootstrap CIs are unchanged,
  because duplicating within an episode-seed cluster leaves the ratio estimate and the bootstrap
  draws unchanged. Verified: I-A −2.0 [−4.11, +0.11] pooled and single-cell. **Rescued, harmed
  and base-success counts were doubled.**
- **Persistent scope** (RP ∪ F-P ∪ E C1–C7): the four duplicated constant-delay cells were
  double-weighted. Descriptive recomputation with duplicates removed:
  - Phase H H3, C − R2: **+28.6 → +25.8 [+24.2, +27.4]**;
  - Phase I P1, I-A − C: −17.6 → −17.2 [−18.5, −15.9];
  - Phase I P4, I-A − R2: **+11.8 → +9.2 [+7.9, +10.5]**.
- **No pre-registered decision changes.**
- **Not affected:** the G2/G3/G4 analyses and the E/F1 held-out confirmation contain no such
  duplicates.
- **Records:** the original Phase H and Phase I tables and documents are kept as recorded, and a
  dated erratum note has been added to each.

## What requires your approval next

- **J-2, noise-augmented fine-tuning:** a separate pre-registered plan, and new training and
  evaluation seed blocks.
  - **Rationale:** J1 favours H4/H5, so the recommendation's precondition for J-2 is met.
  - **Weaker motivation for I-A:** its gate failure did not replicate, so the case is mainly
    I-C's replicated fragility and a robustness margin for I-A.
- **Any analysis of the logged per-step inputs (H3):** read-only, no approval needed for the
  analysis itself. It has not been done here.
- **Any held-out stage:** separate approval. 4,000,600–899 stays untouched.
