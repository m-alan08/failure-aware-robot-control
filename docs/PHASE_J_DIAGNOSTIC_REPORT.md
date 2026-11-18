# Phase J diagnostic audit: why I-A and I-C failed the observation-noise gate

**Scope:** analysis of existing records only.
- **Not done:** no training, no simulator episodes, no new evaluation or held-out seeds.
- **Unchanged:** Candidate C, the frozen policies, the Phase H/I checkpoints, and every raw file
  and reported result.
- **Data:** all data read is development data. The held-out E/F1 step records contain the same
  kind of observation histories, but were deliberately **not** used, because using them to
  inform the next design would amount to selecting on held-out data.

**Reproduce** (read-only, about 10 s):

```bash
uv run python scripts/phase_j_diagnostics.py --out results/phase_j_diagnostics
```

| evidence class | meaning in this report |
|---|---|
| **measured** | computed from archived records |
| **analytic** | derived from the verified noise model and feature definition |
| **causal** | a demonstrated effect of an intervention. **No causal claim in this report meets that bar.** |

## 1. Phase I evidence re-checked

**Gate results confirmed** from the raw Phase I episodes, seeds 3,702,000–299, recomputed with the
committed analysis code:

| arm | 1 cm noise vs frozen SAC (H_noise0.01 + E_C8) | counts |
|---|---|---|
| I-A | −2.0 [−4.1, +0.1] | 68 rescued / 104 harmed |
| I-C | −3.2 [−5.3, −1.1] | 66 rescued / 124 harmed |

**The noise model,** from code and verified on data:
- **Code:** `ObservationNoise` with mask `positions` adds independent N(0, σ²) noise per step to the
  gripper and object *positions* only. The object-relative position is recomputed from the
  noisy positions. **Velocities are not noised.**
- **Data:** in the archived F1 development records (C8, σ = 0.01 m, 15,000 frozen-SAC steps):
  - measured gripper noise standard deviation 0.00998, 0.0100 and 0.0099 m per axis;
  - lag-1 autocorrelation about 0 (−0.026 to 0.013), so independent over time;
  - relative-position noise exactly object minus gripper;
  - gripper-velocity noise exactly 0.

**Training vs evaluation consistency** (code and Stage-0 tests):

| item | finding |
|---|---|
| Feature definitions | the same `Featurizer` code in training and evaluation; bitwise-identical features in Stage 0 |
| History initialisation | zero padding in both |
| Action timing | `observe(o_t)` → `features` → act → `issued(c_t)` in both |
| Normalisation | the frozen statistics plus clipping in both |
| **The only training–evaluation difference relevant here** | **training had no observation noise.** The Phase I training wrapper adds none, so under noise every position-derived input is out of distribution. |

**Not verified:** the Stage-0 feature-identity test ran without observation noise. That is
consistent with the design, since noise is an evaluation perturbation, but it means no test
covers the noisy path specifically.

## 2. Per-seed failure analysis (measured)

Noise cells, 600 paired episodes per seed:

| seed | I-A − SAC | I-A counts | I-C − SAC | I-C counts | R2 (Phase H) − SAC |
|---|---|---|---|---|---|
| 0 | **−4.3 [−8.0, −0.7]** | 20 / 46 | **−4.0 [−7.7, −0.3]** | 20 / 44 | −0.7 [−4.0, +2.3] |
| 1 | −2.0 [−5.7, +1.7] | 26 / 38 | −2.0 [−5.7, +1.7] | 26 / 38 | +0.7 [−2.7, +4.0] |
| 2 | +0.3 [−2.7, +3.3] | 22 / 20 | −3.7 [−7.3, 0.0] | 20 / 42 | **−4.3 [−8.3, −0.3]** |
| pooled | −2.0 [−4.1, +0.1] | 68 / 104 | −3.2 [−5.3, −1.1] | 66 / 124 | −1.4 [−3.3, +0.4] |

Frozen SAC's noise-cell success is 95.3–96.0% per seed.

- **I-A's failure is not shared by all seeds.** It is driven by seed 0 (and seed 1 less
  clearly). Seed 2 shows no measurable degradation.
  - The seed 1 and seed 2 intervals include 0. **That does not show equivalence**, because a
    per-seed interval of about ±3.5 points cannot exclude a 2-point loss.
- **I-C's degradation is broadly shared:** every seed's point estimate is −2 to −4, and seed 0's
  interval excludes 0.
- **Phase H's R2 would also fail this gate definition** (pooled CI low −3.3 ≤ −2), mainly
  through seed 2. **R2 has no displacement or estimator features.**
- **Undisturbed, for contrast:** I-A and I-C are within about 1 point on every seed. R2 loses
  about 3.3 points on seeds 1–2.
- **Episode stage: not analysable.** The Phase I evaluation logged episode outcomes only, with no
  per-step records, so *when* the extra failures occur cannot be determined.

## 3. Feature-noise sensitivity (I-A)

**Analytic.**
- Each displacement entry is $(\tilde g_{t-k+1}-\tilde g_{t-k})/(K_0\cdot0.05)$, with
  $K_0\cdot0.05 = 0.031$ m.
- With independent noise of σ = 0.01 m per position, the noise on a difference has standard
  deviation $\sqrt2\sigma$.
- That gives **0.456** normalised units per coordinate.
- Adjacent entries share one noise term, so their noise correlation is **−0.5**: differencing
  introduces negative MA(1) structure.

**Measured,** on archived noisy and noise-free frozen-SAC trajectories (C8), using the deployed
`Featurizer`:

| quantity | value |
|---|---|
| noise standard deviation, all 10 lags | **0.451–0.452** (analytic 0.456) |
| adjacent-lag noise correlation | **−0.495 to −0.497** (analytic −0.5) |
| clean-feature standard deviation | 0.65–0.67 |
| clean-feature median magnitude | 0.56–0.60 |
| ratio of clean-feature std to noise std | **about 1.45**; the noise std is about 70% of the clean feature's spread |
| clipping at ±5 | **none** (0% of noisy or clean entries) |
| by episode stage, mean magnitude of clean signal vs noise | 0.69 vs 0.33 (steps 0–9), falling to 0.45 vs 0.36 (steps 40–49) |

Late in an episode, commands are small and the features are close to noise level.

**Noise relative to the base policy's own inputs (measured from the frozen normalisers):**
- The same 1 cm noise is 0.04–0.09 normalised units on the base observation's gripper-position
  entries (normaliser std 0.11–0.26 m).
- On I-A's displacement entries it is 0.456. **That is 5–12× more noise on the features than
  on the base position inputs.**
- The base observation's gripper *velocity*, which carries similar information, is not noised
  at all.

**Sensitivity to the noise assumption (analytic).**
- Feature noise scales linearly with σ: 0.23 at 5 mm, 0.46 at 10 mm.
- It depends on noise being independent across steps. Temporally correlated sensor noise
  would be partly cancelled by differencing; independent noise is the worst case for it.

### Representation diagnostics (measured; no policy uses these)

| representation | noise std | correlation with the true per-step displacement, from noisy input | same, from noise-free input |
|---|---|---|---|
| A: per-step difference (deployed) | 0.452 | **0.836** | 1.000 |
| B: 3-step moving average of differences | 0.153 | 0.608 | 0.655 |
| B: exponential filter (α 0.3) | 0.135 | 0.678 | 0.718 |
| C: 2-step difference ÷ 2 | 0.227 | 0.716 | 0.801 |
| C: 3-step difference ÷ 3 | 0.153 | 0.608 | 0.655 |
| C: 5-step difference ÷ 5 | 0.091 | 0.445 | 0.463 |

- **Filtering or wider differences cut the noise 2–5×,** but they also lose the step-by-step
  information. Even noise-free, they correlate only 0.46–0.80 with the true per-step
  displacement, which is what delay identification needs.
- **On noisy data the deployed per-step difference has the highest correlation** with the true
  displacement (0.84).
- **No representation here is clearly better.** This is a noise-versus-lag trade-off, and which
  side matters for control is not knowable from these data.

## 4. Estimator sensitivity (I-C)

**Measured in the simulator** (Phase E2 logged estimator outputs, frozen-SAC episodes,
development seeds):

| condition | delay reliably reported (share of steps) | correct when reported |
|---|---|---|
| no delay | 57.4% | 99.9% |
| **1 cm noise, no delay** | **21.9%** | 99.9% |
| delay 3 | 83.2% | 100.0% |
| delay 3 + 0.5 cm noise | 74.4% | 100.0% |

**Measured offline,** reconstructing I-C's features with the deployed `Featurizer` on archived
F1 trajectories:

| condition | reliability-flag share, from noise-free positions | from the noisy positions the policy sees | latched d̂ > 0 (noisy) |
|---|---|---|---|
| 1 cm noise, no delay | 90.6% | **18.8%** | 0.17% |
| delay 3 + 0.5 cm noise | 84.2% | 73.0% | 89.3% |

**Reading:**
- **Under noise, the estimator mostly abstains rather than errs.** False delay latches are rare
  (0.17%), and reported estimates stay about 99.9% correct.
- I-C's latched-delay one-hot therefore stays mostly at "0", **which is in-distribution**.
- **Its reliability flag shifts strongly,** from about 57% "reliable" (no delay, training-like) to
  about 19–22%. That is a measured input-distribution shift for I-C.
- Whether the shift *causes* I-C's degradation is not shown.

## 5. Competing explanations

| | hypothesis | supporting | contradicting | unverified | cheapest distinguishing test |
|---|---|---|---|---|---|
| **H1** | Differenced position features amplify observation noise | **Analytic and measured:** 0.456 noise std, 5–12× the base inputs' normalised noise, about 70% of the clean feature spread, MA(1) correlation | **I-C fails more than I-A without any displacement features. R2, without them, also degrades (pooled −1.4, seed 2 −4.3).** I-A seed 2, with the same feature noise, shows no degradation. | whether I-A's actions respond to the feature noise; no causal test | evaluation-only **privileged-input ablation**: I-A with displacement features computed from noise-free positions, base observation still noisy, on fresh development seeds |
| **H2** | The delay estimator becomes unreliable under noise | **Measured:** reliability 57% → 22% (simulator) and 91% → 19% (offline); I-C's flag distribution shifts | estimates are not *wrong*: about 99.9% correct when reported, false latches 0.17% | whether the flag shift drives I-C's failure | the same ablation for I-C: flag and d̂ computed from noise-free positions |
| **H3** | The policy over-relies on history features after fine-tuning | indirect only: large seed variation on short delays and under noise | – | **unresolved.** An open-loop sensitivity probe was not possible, because the archived step records lack the desired goal and the policy's input cannot be rebuilt without inventing it. | a logging evaluation recording full inputs (observation, goal, features) on a few cells, then an offline sensitivity probe |
| **H4** | Training lacked observation noise | **verified in the code:** no noise in the Phase I (or Phase H) training wrappers. All three retrained families degrade under noise to some degree (I-A −2.0, I-C −3.2, R2 −1.4). | **frozen SAC was also trained without noise and is robust** (95.7%). Absent noise alone does not cause fragility; fine-tuning, the new inputs, or both are implicated. | – | noise-augmented fine-tuning, a training experiment |
| **H5** | A broader out-of-distribution fragility, not one defective feature | **measured:** gain loss and noise (NG) I-A 63.6 / I-C 72.3 vs C 91.9; I-C collapses on delay plus gain loss (38.1); R2 also degrades under noise; failures are seed-dependent | I-A and I-C are nominally competent on every seed, so not globally fragile | which perturbations matter | the privileged-input ablation (if I-A or I-C *do not* recover with clean features, that points to H5 over H1/H2) |

**Overall:**
- **Analytic mechanisms are established for H1 and H2:** feature noise amplification, and
  estimator abstention with a flag shift.
- **Neither is shown to be the cause.** The cross-arm pattern contradicts H1 as a sole
  explanation: I-C and R2 lack the differenced features yet degrade, and I-A seed 2 does not.
- **The data fit H4 and H5 best:** the fine-tuned policies, which never saw noise or gain loss,
  are fragile to perturbations absent from training.
- **H3 is unresolved.**

## 6. Is the existing data sufficient for a causal diagnosis?

**No.**
- **The Phase I evaluation logged only episode outcomes.** There are no per-step observations,
  features, actions or goals, so failure timing and policy responses cannot be analysed.
- **The archived observation histories** (F1 development) come from *frozen-SAC* trajectories,
  not from I-A or I-C closed loops, and they lack goals.
- **A causal diagnosis needs an intervention:** the privileged-input ablation of §5, which is
  evaluation-only and has no training. That requires new simulator episodes on a new development
  seed block, and therefore **your approval**. See `docs/PHASE_J_NEXT_EXPERIMENT_RECOMMENDATION.md`.

## Outputs

`results/phase_j_diagnostics/`:
- `per_seed_gate.csv`;
- `noise_model_check.json`;
- `ia_feature_noise.csv` and `ia_feature_noise_by_stage.csv`;
- `filtered_alternatives.csv`;
- `estimator_reliability_e2.csv`;
- `ic_features.csv`.
