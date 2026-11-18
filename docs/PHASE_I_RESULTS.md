# Phase I: stronger delay-aware RL baseline, Stage 2 results (development only)

## Run record

| item | value |
|---|---|
| Plan | `docs/PHASE_I_STRONG_BASELINE_PROPOSAL.md` (SHA-256 `409b2f8e…`), approved as written 2026-11-17 |
| Recorded before use | K1–K8 before training (`7164d68`); K9 after training, before evaluation (`064e18d`). The evaluation ran at `064e18d` on a clean tree. |
| Training | I-A and I-C × seeds 0–2, each warm-started from a copy of frozen `fetch_push_fp32_seed{k}`. 1,000,000 fine-tuning steps, upstream hyperparameters (`random_steps = 0`), final checkpoints, no tuning. About 5,400–5,500 s per run, 6 in parallel. |
| Evaluation | development seeds **3,702,000–299** × 3 policy seeds; 50 cells × 6 configurations × 900 = **270,000 episodes**, one run (23 minutes) |
| Unchanged | frozen SAC, the F1 model, Candidate C, the Phase H checkpoints and results |
| Held-out | **not used.** 4,000,600–899 untouched. |

Numbers are success-rate points with paired 95% episode-seed cluster-bootstrap CIs; counts are
rescued / harmed.

## Stage 0 (code checks)

- **Warm start:** warm-started actions were **bitwise identical** to the frozen policy, for all
  seeds and both arms.
- **Features:** training and evaluation features were **bitwise identical**.
- **Port:** the upstream port stayed bitwise identical with the new hook.
- **Smoke-run transient (K7):** nominal success dipped early (0.90 I-A, 0.82 I-C at 20k steps).
  It recovered in every full run (K9).

## Gate (evaluated first): both arms FAIL, on the noise component

| arm | G-nominal: undisturbed (H + E C0) vs frozen SAC | per-seed undisturbed | G-noise: 1 cm noise (H + E C8) vs frozen SAC | gate |
|---|---|---|---|---|
| **I-A** | −0.6 [−1.6, +0.4], passed | 97.7 / 98.3 / 99.3%, passed (≥ 95%) | **−2.0 [−4.1, +0.1]** (104 of 1,722 harmed), **failed** (CI low ≤ −2) | **failed** |
| **I-C** | +0.1 [−0.6, +0.8], passed | 99.0 / 98.7 / 99.7%, passed | **−3.2 [−5.3, −1.1]** (124 of 1,722 harmed), **failed** | **failed** |

- **The undisturbed part improved:** warm-starting fixed the undisturbed loss that sank Phase H.
  R2 lost 3.0 points; I-A and I-C are within about half a point.
- **Under 1 cm position noise, both arms lose competence.**
- **Pre-registered consequence (proposal §6.1):** both arms are "baseline not nominally
  competent":
  - their comparisons below are **descriptive only**;
  - **no claim about delay-aware RL is drawn**;
  - **neither arm is eligible for Stage 3**.

## Pre-registered contrasts (descriptive because the gate failed)

**Primary, I-A (Holm across P1–P4):**

| | contrast | scope | estimate [95% CI] | counts | Holm p | nominal result |
|---|---|---|---|---|---|---|
| P1 | I-A − C (margin −2) | persistent | **−17.6 [−18.9, −16.3]** | 360 / 2,739 | 1.0 | not supported |
| P2 | I-A − frozen SAC (margin −2) | RT + RI | **−4.8 [−6.0, −3.5]** | 151 / 323 | 1.0 | not supported |
| P3 | I-A − C | RT + RI | −2.6 [−3.9, −1.3] | 213 / 308 | 1.0 | not supported |
| P4 | I-A − Phase H R2 | persistent | **+11.8 [+10.4, +13.2]** | 3,040 / 1,445 | < 0.001 | supported |

**Secondary (Holm across S1–S6):**

| | contrast | scope | estimate [95% CI] | nominal result |
|---|---|---|---|---|
| S1 | I-C − C (margin −2) | persistent | −17.0 [−18.2, −15.8] | not supported |
| S2 | I-C − frozen SAC (margin −2) | RT + RI | −4.6 [−5.9, −3.4] | not supported |
| S3 | I-C − C | RT + RI | −2.5 [−3.8, −1.2] | not supported |
| S4 | I-C − R2 | persistent | +12.4 [+11.1, +13.8] | supported |
| S5 | I-C − I-A | persistent | +0.6 [−0.5, +1.7] | not supported |
| S6 | I-A − contact-aware | RT + RI | +1.0 [−0.4, +2.5] | not supported |

**Per seed, I-A − C persistent / I-A − SAC short:**

| seed | I-A − C persistent | I-A − SAC short |
|---|---|---|
| 0 | −23.8 | −9.3 |
| 1 | −17.6 | −5.7 |
| 2 | −11.4 | +0.6 |

Short-delay behaviour varies strongly by seed. Every seed is far below C under persistent delay.

## Success rates (%)

| | frozen SAC | contact-aware | **C** | Phase H R2 | I-A | I-C |
|---|---|---|---|---|---|---|
| delay 1 (E C1) | 82.2 | 99.0 | **99.0** | 89.2 | 92.7 | 93.6 |
| delay 2 (E C2) | 27.8 | 97.6 | **97.6** | 74.1 | 84.1 | 83.3 |
| delay 3 (E C3) | 11.2 | 96.2 | **96.1** | 47.4 | 74.9 | 77.7 |
| delay 4 (E C4) | 9.7 | 93.2 | **93.2** | 23.9 | 58.9 | 62.3 |
| delay 3 + 50% gain loss (E C7) | 57.4 | 88.8 | **88.8** | 73.1 | 52.2 | 38.1 |
| random-onset persistent (RP) | 54.2 | 93.4 | **95.2** | 70.9 | 80.3 | 82.1 |
| change of value (RS) | 31.6 | 90.2 | **93.6** | 57.9 | 77.2 | 75.6 |
| single pulses (RT) | **97.5** | 96.4 | **97.7** | 94.9 | 96.7 | 96.8 |
| on-off (RI) | **91.3** | 80.8 | 86.8 | 89.7 | 82.6 | 82.8 |
| no delay (E C0) | 99.0 | 99.0 | 99.0 | 97.1 | 98.4 | 99.1 |
| 1 cm noise (E C8) | **95.7** | 95.7 | 95.7 | 94.2 | 93.7 | 92.4 |
| noise / gain loss (NG; out of distribution) | 57.6 | 90.3 | **91.9** | 63.0 | 63.6 | 72.3 |

## Reading (descriptive; the gate failed)

1. **The Phase I changes addressed what they targeted, but only in part.**
   - The undisturbed loss is gone: within about half a point of frozen SAC, against R2's −3.0.
   - Persistent-delay performance improved substantially over R2 (+11.8 to +12.4). At delay 4,
     I-A reaches 58.9% against R2's 23.9%, consistent with Hy1, the identifiability hypothesis.
   - Persistent performance still falls well short of compensation (−17.6), and still declines
     with delay length.
2. **Short and on-off delays got worse** than frozen SAC (−4.8) and than R2 (on-off: 82.6
   against 89.7), with large variation across seeds.
   - **Hypothesis (untested):** the warm-started policy learned to rely on the history features
     in a way that hurts when delays switch on and off frequently.
3. **Observation noise breaks the gate. Hypothesis (untested):**
   - training had no observation noise;
   - I-A's displacement features divide noisy position differences by K0·0.05 ≈ 3.1 cm, so
     1 cm noise is a large out-of-distribution perturbation of those inputs;
   - I-C's latched estimate is likewise noise-sensitive.

   R2, without such features, held up better (94.2%).
4. **Gain loss.** I-C collapses on delay plus gain loss (E C7: 38.1%). Gain loss was not in
   training, and the estimator's gain mismatch may confuse the policy. This is not analysed
   further.
5. **The explicit estimate does not help** beyond the raw window (S5, +0.6, n.s.).

## What this does and does not show

- **Shown (development only, descriptive):** two delay-aware SAC+HER baselines, warm-started and
  information-matched:
  - fix the undisturbed nominal loss;
  - improve on Phase H R2 by about 12 points under persistent delay;
  - remain **about 17–18 points below Candidate C** under persistent delay;
  - are **worse than frozen SAC on short and on-off delays**;
  - **fail the pre-registered noise gate**.
- **Not shown:**
  - any claim about delay-aware RL in general (the gate failed, and only one algorithm family
    and one budget were tried);
  - held-out confirmation;
  - any real-robot result.
- **Not done, by rule:**
  - no retuning;
  - no added noise to training;
  - no budget extension;
  - no Stage 3 for these arms.
- **What a next baseline would need:** observation noise in training, and/or a recurrent
  policy (arm B). That would be a new, separately pre-registered phase.

## Integrity

- Phase H results are unchanged. Candidate C and the frozen checkpoints are unchanged.
- All six Phase I checkpoints are registered with their warm-start sources.
- Training records are in `results/phase_i_training/`, and the analysis tables in
  `results/phase_i_evaluation/`.
