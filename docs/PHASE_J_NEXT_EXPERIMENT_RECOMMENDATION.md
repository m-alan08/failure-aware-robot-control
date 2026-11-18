# Phase J: recommended next experiment (recommendation only; nothing implemented)

This is based on `docs/PHASE_J_DIAGNOSTIC_REPORT.md`. Nothing here has been implemented, trained
or run. Every option below needs its own pre-registered plan, new development seed blocks and
your approval. Held-out 4,000,600–899 stays untouched.

## What the diagnosis leaves open

**Established mechanisms (analytic or measured):**
- I-A's displacement features carry noise with standard deviation 0.456, 5–12× the base inputs'
  normalised noise.
- I-C's estimator abstains under noise, so its reliability flag shifts.

**Not established:** that either mechanism *causes* the gate failures.
- **Cross-arm evidence points the other way:** I-C and Phase H's R2 degrade without any
  differenced features, and one I-A seed does not degrade.
- **The data fit best:** fragility of fine-tuned policies to perturbations absent from training
  (H4/H5).

## Options

### 1. Noise-augmented fine-tuning of I-A

- **What:** re-fine-tune I-A (warm-started from frozen SAC, as before), adding position noise to
  the *training* observations. For example, σ ~ U[0, 1 cm] per episode, through the same
  `ObservationNoise` model.
- **Targets:** H4 directly; H1 indirectly, since the policy learns to discount noisy features.

### 2. Robust history representation

- **What:** filtered or wider-difference features.
- **Against it:** the diagnostics show a noise-versus-lag trade-off with no clear winner (§3 of the
  report). It does not address I-C's or R2's degradation.

### 3. Recurrent delay-aware RL

- **What:** a GRU/LSTM actor-critic with sequence replay.
- **Targets:** identifiability (Phase I's persistent gap) more than noise. It is no guarantee of
  noise robustness unless trained with noise.

### 4. Hybrid

- **What:** combine (1) with (2) or (3).
- **Against it:** more arms, and attribution becomes harder.

### Ranking (1 = best)

| criterion | 1 noise-augmented I-A | 2 robust representation | 3 recurrent | 4 hybrid |
|---|---|---|---|---|
| Scientific value (tests the best-supported explanation) | **1** | 3 | 2 | 2 |
| Implementation risk | **1**: one wrapper change, reusing the verified noise model | 2 | 4: new replay and trainer, outside the verified port | 3 |
| Compute cost | **1**: about 3 × 1.5–2 h runs in parallel | 1 | 4 | 3 |
| Ability to address the identified failure | **1** | 3 | 3 | 2 |
| Fairness against Candidate C | 2: RL sees the evaluation noise *type* in training, while C's constants never did | 1 | 2 | 3 |

**Note on fairness for option 1.** The training noise distribution must be specified *before*
evaluation, as a range rather than the exact evaluation value. The evaluation should then also
include a noise level outside the training range, so the result isn't simply fitting the gate
condition.

## Recommendation

**Run first: an evaluation-only privileged-input ablation (no training).** Then, conditionally,
option 1.

### Step J-1: privileged-input ablation (no training)

This is the cheapest experiment that can turn the analytic mechanisms into causal evidence.

**Configurations,** on the existing, unmodified I-A and I-C checkpoints:
- **I-A-clean-features:** I-A with its displacement features computed from *noise-free*
  positions, while the base observation stays noisy.
- **I-C-clean-estimate:** I-C with its estimator fed *noise-free* positions.
- **References:** I-A, I-C and frozen SAC, unchanged.

These "clean" inputs are **PRIVILEGED diagnostic configurations only**, never deployable and
never compared as baselines.

**Cells:** H_noise0.01 and E_C8 (the gate cells), H_undisturbed and E_C0 (nominal controls), and
a 0.5 cm noise cell.

**Seeds:** a new development block, for example 3,703,000–299. Never a held-out seed.

**Size:** about 5 cells × 5 configurations × 900 ≈ 22,500 episodes, **about 3–5 minutes**. It
also logs per-step inputs (observation, goal, features, action), so H3 becomes analysable
afterwards.

**Reading (pre-registered before running):**

| outcome | meaning |
|---|---|
| I-A recovers to within the gate with clean features | H1 is causal for I-A |
| I-C recovers with a clean estimate | H2 is causal for I-C |
| neither recovers | H4/H5: the policy's response to noise in the *base* observation, or general fragility, dominates. Feature redesign alone would not help. |

### Step J-2: noise-augmented fine-tuning of I-A (only if J-1 supports H1 or H4/H5)

**Design:**
- I-A only; 3 seeds; warm start; 1M steps.
- Training observation noise σ ~ U[0, 1 cm] per episode, on positions only, as in the evaluation
  model.
- Otherwise identical to Phase I.

**Evaluation:**
- the Phase I cells plus a 1.5 cm noise cell outside the training range;
- new development seeds;
- the same gates, fixed now: −2 pooled, a 95% per-seed floor, and the noise gate.

**Compute:** about 2 h of training (3 runs in parallel) plus about 25–35 minutes of evaluation.

**Why not recurrence (option 3) yet:** it addresses a different gap, persistent-delay
identification, at much higher risk and cost. It should only follow if a noise-robust I-A still
trails C by a wide margin under persistent delay.

## What requires your approval before any further training

1. **J-1 (no training, but new simulator episodes):**
   - approval to run the privileged-input ablation;
   - its new development seed block (proposed 3,703,000–299);
   - its pre-registered reading.
2. **J-2 (training):**
   - a separate, pre-registered plan, written after J-1;
   - the training noise distribution, the out-of-range evaluation level, the gates, and new
     training and evaluation seed blocks.

   No training happens without that approval.
3. **Any held-out (Stage 3) evaluation:** a separate approval, and only for arms that pass every
   gate. 4,000,600–899 stays reserved.

**Explicitly not proposed:**
- retuning I-A or I-C;
- selecting a checkpoint from this diagnosis;
- feeding filtered features into the existing policies and reporting control results;
- any use of held-out data.
