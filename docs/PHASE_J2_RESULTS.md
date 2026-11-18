# Phase J-2: noise-augmented fine-tuning, Stage 2 results (development only)

## Run record

| item | value |
|---|---|
| Plan | `docs/PHASE_J2_PROPOSAL.md` (SHA-256 `0b0f0046…`), approved as written 2026-11-18 |
| Recorded before use | L1–L3 before training (`cc0a5d4`); L4 after training, before evaluation (`4296781`). The evaluation ran at `4296781` on a clean tree. |
| Training | J2-A and J2-C × seeds 0–2, warm-started from copies of frozen SAC. 1M steps, identical to Phase I except training observation noise σ ~ U[0, 1 cm] per episode. About 7,900–8,200 s per run. |
| Evaluation | development seeds **3,704,000–299**; **46 de-duplicated cells** × 7 configurations × 900 = **289,800 episodes**, one run |
| Unchanged | frozen SAC, C, and the Phase H/I checkpoints and results |
| Held-out | **not used.** 4,000,600–899 untouched. |

## Gates (evaluated first): both arms FAIL both components

| arm | G-nominal (undisturbed vs frozen SAC) | per-seed undisturbed | G-noise (1 cm vs frozen SAC) | gate |
|---|---|---|---|---|
| J2-A | **−7.1 [−9.0, −5.3]** | 91.0 / 89.7 / 97.3% | **−8.7 [−11.2, −6.1]** | **failed** |
| J2-C | **−4.7 [−6.1, −3.3]** | 95.3 / 95.0 / 95.0% | **−8.1 [−10.8, −5.4]** | **failed** |

**For reference, in the same run:**
- Phase I I-A: −1.2 undisturbed and −1.7 [−3.7, +0.2] under 1 cm noise.
- Phase I I-C: −0.3 undisturbed and −3.6 [−5.8, −1.3] under 1 cm noise.

**Consequence (pre-registered):** both J-2 arms are "not competent". Every comparison is
descriptive only, and neither arm is eligible for Stage 3.

## Pre-registered hypotheses (Holm across N1, N2, D1, D2)

| | contrast | scope | estimate [95% CI] | counts | result |
|---|---|---|---|---|---|
| N1 | J2-A − I-A | 1 cm noise | **−7.0 [−9.9, −4.1]** | 47 / 110 | not supported |
| N2 | J2-C − I-C | 1 cm noise | **−4.6 [−7.4, −1.7]** | 65 / 106 | not supported |
| D1 | J2-A − I-A (margin −2) | persistent | **−15.9 [−17.4, −14.4]** | 872 / 2,444 | not supported |
| D2 | J2-C − I-C (margin −2) | persistent | **−17.1 [−18.5, −15.8]** | 741 / 2,438 | not supported |

- **The direction reversed:** noise augmentation made both arms **worse under noise**, not better.
- **It also made them much worse under delay:**
  - persistent delays: −16 to −17 against the Phase I arms;
  - no delay: −5 to −7 against frozen SAC;
  - 1.5 cm noise, outside the training range: −4.7 and −6.6.

**Secondary** (Holm across S1–S8, all "not competent"):

| contrast | J2-A | J2-C |
|---|---|---|
| − C, persistent | −30.7 | −32.0 |
| − frozen SAC, short and on-off | −12.2 | −9.1 |

**Per seed:** both arms are worse than their Phase I counterparts under noise on every seed
(J2-A −6.7 to −7.7, J2-C −3.7 to −5.7).

## Success rates (%), selected cells

| | frozen SAC | C | I-A | I-C | **J2-A** | **J2-C** |
|---|---|---|---|---|---|---|
| no delay | 99.8 | 99.8 | 98.6 | 99.4 | 92.7 | 95.1 |
| 0.5 cm noise | 99.0 | 99.0 | 97.9 | 97.7 | 88.7 | 93.4 |
| 1 cm noise | 95.7 | 95.7 | 94.0 | 92.1 | 87.0 | 87.6 |
| 1.5 cm noise | 84.8 | 85.0 | 80.4 | 80.1 | 80.1 | 78.2 |
| persistent d = 2 (P_d2) | 28.9 | 98.0 | 85.8 | 84.7 | 61.4 | 66.0 |
| persistent d = 4 (P_d4) | 9.6 | 90.7 | 61.1 | 68.1 | 35.0 | 37.8 |
| on-off d = 4 (RI_d4) | 86.1 | 81.8 | 76.4 | 80.8 | 71.0 | 75.3 |

## Reading

1. **Hypothesis H4, as implemented here, is contradicted.** "Training without observation noise
   caused the fragility, and noise augmentation fixes it" does not hold. Augmenting with
   σ ~ U[0, 1 cm] at the Phase I budget degraded nominal, noise and delay performance alike.
2. **Plausible but untested explanations:**
   - **Every training episode was noisy,** with mean σ = 0.5 cm. That made delay identification
     from noisy position differences much harder during learning, so the policies may have learned
     to under-use the history inputs. They then lose delay robustness without gaining noise
     robustness.
   - **Noisy positions reached the critic's input too.** That raises value-estimate variance at a
     fixed budget.
   - **The budget** (1M steps) may be insufficient for the harder, noisier task.

   None of these is tested. Distinguishing them would need a new, separately approved experiment,
   for example a lower or mixed noise schedule, or noise on the actor only.
3. **The Phase I arms remain the better RL baselines,** and both still fail the noise gate in
   this run (I-A −1.7 [−3.7, +0.2]; I-C −3.6). I-C's noise fragility replicates a third time.
   I-A's failure is marginal, as in J-1.
4. **For the project's comparison:** none of the delay-aware RL baselines built so far (R1, R2,
   I-A, I-C, J2-A, J2-C) is competent under 1 cm observation noise *and* competitive with
   Candidate C under persistent delay. This applies to these baselines only, not to delay-aware
   RL in general.

## Integrity

- Pre-registered gates and hypotheses were applied exactly. Nothing was retuned, and no
  checkpoint was selected.
- The Phase I, H and J-1 results are unchanged.
- The cells are de-duplicated, and a test asserts it.
- Checkpoints are registered in `src/farc/retrain/checkpoints_phase_j2.json`, training records
  are in `results/phase_j2_training/`, and the tables are in `results/phase_j2_evaluation/`.
