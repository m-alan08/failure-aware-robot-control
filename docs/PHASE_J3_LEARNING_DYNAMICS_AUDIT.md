# Phase J-3: learning-dynamics and failure-mechanism audit (diagnostic only)

**Scope:**
- No training, no fine-tuning, no simulator episodes, and no new seeds.
- Inputs are the committed J-2 and Phase I training logs, the saved final checkpoints (read
  only, SHA-256 checked against the manifests), and archived Phase F1 development step records.
- Held-out 4,000,600–899 is untouched.
- All J-2 and Phase I numbers quoted are **development** results.

**Reproduce:** `uv run python scripts/phase_j3_audit.py --out results/phase_j3_audit`.

**Evidence labels:**
- **[code]**: read from the implementation.
- **[log]**: from the training `progress.csv` files.
- **[weights]**: from the checkpoints.
- **[offline]**: computed on archived recorded trajectories with *synthetic* noise added after
  the fact. This is not closed-loop, and the trajectories are frozen-SAC's, not J-2's.
- **[Stage 2]**: from the J-2 results.

## 0. Summary

| hypothesis | verdict | confidence |
|---|---|---|
| H1 critic destabilised by noise | **not demonstrated.** No divergence. Critic loss is about 10% higher and mean Q about 1.9 lower, offset in parallel throughout. That fits a harder task, not instability. | low that it is the main cause |
| H2 delay information corrupted | **supported for J2-C** [offline]; for J2-A, **plausible but not shown**. The information survives in the input [offline], and the displacement weights were not suppressed [weights]. | moderate (C), low (A) |
| H3 fine-tuning interference with nominal behaviour | **present in both Phase I and J-2:** weight drift of about 100% relative norm and the same early nominal dip. Phase I recovered and J-2 did not. Drift alone does not separate them. | moderate that nominal loss is a learning outcome, not a drift-size effect |
| H4 budget insufficient | **weak:** J2-A had plateaued at about 0.89 clean success; J2-C was still rising slowly | low–moderate |
| H5 implementation problem | **no bug found.** One structural property matters: the critic sees noisy observations. One record-keeping note. | high that there is no bug |

**Most useful new fact:** noise augmentation did **not reduce noise sensitivity.** The drop from
undisturbed to 1 cm noise was the same as Phase I's [Stage 2]:

| arm | drop, undisturbed → 1 cm noise |
|---|---|
| frozen SAC | 4.1 |
| I-A | 4.6 |
| J2-A | 5.7 |
| I-C | 7.3 |
| J2-C | 7.5 |

J-2's noise deficit is therefore mostly a **uniform loss of competence:**
- J2-A is 5.9 points below I-A undisturbed and 7.0 below it under noise.
- J2-C is 4.3 below I-C undisturbed and 4.6 below it under noise.

There is also an **additional, delay-specific loss** of 16–17 points under persistent delay.

## 1. Implementation trace [code]

| stage | clean or noisy | detail |
|---|---|---|
| 1. env wrapper | **noisy** `observation` | `NoisyPhaseIWrapper._obs` noises the base 25-dim observation before the Phase I featuriser, at reset and every step. σ is drawn once per episode from U[0, 1 cm], and the per-step draws are independent. `masked_noise("positions")` noises gripper and object positions; the relative position is recomputed; velocities and fingers are not noised. |
| 1. env wrapper | clean `achieved_goal` and `desired_goal` | The dict is passed through unchanged. |
| 1. env wrapper | clean physics | Delay acts on executed actions (`DelayHistoryWrapper.step`), independently of the noise. |
| 2. history features | **noisy** | I-A displacements are differences of noisy gripper positions. The I-C estimator observes noisy gripper positions. Commands are clean (they are the policy's own actions). |
| 3. actor input | **noisy** observation + features, clean desired goal | `collect_episode` acts on `normalizer(noisy obs, desired_goal)`. |
| 4. critic input | **noisy** observation + features, clean goal, action | `build_update_batch` normalises the stored noisy observations for both `inputs` and `next_inputs`. |
| 5. replay storage | noisy observations; clean achieved and desired goals | `store_episode(observations, achieved, desired, actions)`. `next_obs[t]` is the same stored row as `obs[t+1]`, so the two are consistent. |
| 6. HER relabelling | **clean** | Relabelled goals are clean future `achieved_goals`. |
| 7. reward | **clean** | `compute_reward(next clean achieved_goal, goal)`. |
| 8. target networks | noisy `next_inputs` | Polyak averaging is unchanged; the target-actor sample a′ = π(noisy o′). |
| normaliser | frozen; `update` is a no-op | Noise does not shift its statistics. |
| periodic eval during training | **clean, no delay** | `role="eval"` uses `PhaseIWrapper` without a schedule. The logged eval curve is therefore clean nominal success. |

**Consistency findings:**

- **No bug.**
  - Observation and next observation are consistent.
  - Reward, achieved goal and desired goal are mutually consistent; all three are clean.
  - The noise matches the evaluation model (Stage-0 test), and evaluation applies the same
    `masked_noise` to the policy input only.
- **Structural property (not a bug):** within a stored transition, `observation[3:6]` (the noisy
  object position) differs from `achieved_goal` (the clean object position) by the noise. The
  critic is therefore trained on a **partially observed problem**: its targets come from clean
  rewards, but its inputs are noisy. Irreducible TD variance rises with σ. A privileged (clean)
  critic was not used; the proposal did not specify one.
- **Train/eval mismatch:** none beyond what was designed. Training noise was U[0, 1 cm]. The
  1 cm evaluation level sits at the top edge of that range, and 1.5 cm lies outside it, as
  pre-registered.
- **Record-keeping note:** `phase_j2_record.json` has `git_commit_at_end = 179c06b`, while
  training started at `cc0a5d4`. Between them, `179c06b` committed evaluation code and a
  backwards-compatible `load_phase_i(manifest_path=…)` argument in `phase_i.py`. The training
  processes had imported their modules at launch, and the training path does not call
  `load_phase_i`, so training is unaffected. The manifest records the training commit correctly.

## 2. Critic and learning dynamics [log]

**Logged:** critic loss, actor loss, alpha loss, α, entropy, mean Q(s,a) over the update batch,
training success (noisy and delayed) and clean periodic evaluation success. All are averaged per
20k steps over 50 rows.

**Not logged:** TD-error distributions, gradient norms, per-Q-head statistics and intermediate
checkpoints (only final checkpoints exist). **These cannot be reconstructed** without
re-running training or replaying stored batches, and the batches were not stored.

Mean over the three seeds in 200k-step blocks:

| | 0–200k | 200–400k | 400–600k | 600–800k | 800k–1M |
|---|---|---|---|---|---|
| clean eval success: I-A | 0.898 | 0.966 | 0.970 | 0.979 | 0.986 |
| J2-A | 0.841 | 0.884 | 0.887 | 0.887 | **0.895** |
| I-C | 0.905 | 0.961 | 0.973 | 0.980 | 0.985 |
| J2-C | 0.851 | 0.901 | 0.881 | 0.898 | **0.925** |
| critic loss: I-A / J2-A | 6.28 / 6.46 | 7.40 / 7.80 | 6.62 / 7.08 | 6.03 / 6.70 | 5.65 / 6.35 |
| I-C / J2-C | 5.60 / 5.57 | 6.77 / 7.10 | 6.45 / 7.01 | 6.08 / 6.63 | 5.75 / 6.29 |
| mean Q: I-A / J2-A | −9.8 / −11.0 | −9.7 / −11.9 | −8.4 / −10.4 | −7.6 / −9.5 | −7.0 / −8.9 |
| I-C / J2-C | −8.7 / −10.0 | −9.4 / −11.6 | −8.3 / −10.8 | −7.7 / −10.0 | −7.2 / −9.4 |

**Per seed, last 10 log rows:** see `results/phase_j3_audit/training_log_summary.csv`.
- α and entropy are identical across phases (α ≈ 0.003–0.005, entropy −4.0, at target).
- The maximum critic loss is similar: 6.4–9.1 for J-2 vs 6.5–8.3 for Phase I.
- J-2 training success (noisy and delayed) was 0.70–0.77, vs 0.84–0.90 for Phase I.

**Reading:**
- **Both phases share the same warm-start shock.** Clean success falls from about 0.998 to
  0.82–0.90 within 20–140k steps, and critic loss peaks at 200–400k. Phase I then recovers to
  about 0.99; **J-2 does not**, plateauing at about 0.89–0.93.
- **The critic is not unstable.** Loss and Q follow the same trajectory with a near-constant
  offset (+0.3 to +0.7 loss; −1.3 to −2.2 Q), with no divergence, oscillation or entropy
  collapse.
- The lower Q matches J-2's lower training success. Returns are −1 per unsuccessful step, so a
  harder task gives lower values. The slightly higher loss matches the larger irreducible TD
  variance expected from noisy critic inputs (§1).
- **H1, as "instability", is not demonstrated.** A weaker form, "noisier value targets slowed
  learning", cannot be separated from H4 with these logs.

## 3. Delay-information corruption

### 3.1 I-A displacement features [code, offline]

**Analytic.** Noise on one entry is √2·σ / (0.62·0.05 m):

| σ | noise std (normalised units) | P(clip at ±5) |
|---|---|---|
| 0.25 cm | 0.114 | 0 |
| 0.5 cm | 0.228 | 0 |
| 1 cm | 0.456 | ≈ 0 |
| J-2 training mixture, σ ~ U[0, 1 cm] (rms σ 0.58 cm) | **0.263** | ≈ 0 |

- Adjacent entries share one noise term, so their noise correlation is −0.5 (MA(1)).
- Phase J measured a clean-feature std of about 0.66 on frozen-SAC trajectories, so the
  training-mixture SNR is about 2.5 in std terms.

**Information content [offline].** On 15,000 recorded frozen-SAC steps per delay (C0–C4,
constant delay 0–4), synthetic position noise was added and an oracle-structure lag identifier
applied: a least-squares fit of the 10 most recent displacements on commands at lag L = 0..5,
choosing the minimum residual. This is a best-case linear test, not what the network does.

| σ | displacement SNR | lag accuracy, d = 0 | d = 1 | d = 2 | d = 3 | d = 4 |
|---|---|---|---|---|---|---|
| 0 | ∞ | 0.84 | 0.97 | 0.99 | 0.99 | 0.99 |
| 0.5 cm | 3.1 | 0.69 | 0.95 | 0.99 | 0.99 | 0.98 |
| 1 cm | 1.6 | 0.62 | 0.91 | 0.97 | 0.96 | 0.95 |

**Delay-identifying information largely survives** in a 10-step window at the noise levels used
in training, except for detecting *no* delay, which is hard even when clean.

**Weights [weights].** J2-A's first-layer displacement columns have the **same** mean column norm
as I-A's: actor 4.94 vs 4.97, critic 6.45 vs 6.59. The history pathway was **not suppressed**.
This contradicts the "learned to ignore history" explanation offered in the J-2 results.

**Verdict for I-A:** the noise makes delay inference *harder to learn* (SNR about 2.5 on average),
but the information is present and the pathway is used. H2 is **plausible but not shown** for
J2-A.

### 3.2 I-C estimator features [offline, deployed `Featurizer`]

| σ | reliability-flag share, d = 0 | d = 1–4 | latched d̂ correct, d = 1–4 | latched d̂ = 0 under delay |
|---|---|---|---|---|
| 0 | 0.57 | 0.80–0.86 | 0.90–0.92 | 0.08–0.10 |
| 0.5 cm | 0.34 | 0.65–0.76 | 0.86–0.91 | 0.09–0.14 |
| 0.75 cm | 0.22 | 0.38–0.50 | 0.75–0.88 | 0.12–0.25 |
| 1 cm | 0.09 | 0.14–0.20 | **0.53–0.70** | **0.30–0.47** |

- **Under noise the estimator abstains** rather than mis-latching. With no delay, latched d̂ is
  still 0 in 99.6–100% of steps.
- **Under delay with σ ≥ 0.75 cm,** a third to a half of steps show the delay **missed** (d̂ = 0)
  and the reliability flag is rarely set.
- **A quarter of J-2C training episodes** had σ ≥ 0.75 cm (σ ~ U[0, 1 cm]). In those episodes the
  estimate features were weakly informative or misleading about the delay, while the delay was
  still present in the dynamics.

**Verdict for I-C:** **supported** (offline measurement). J-2 training **degraded the delay
signal** that J2-C was meant to exploit. This is a credible mechanism for the delay-specific
loss: J2-C is −17.1 vs I-C under persistent delay, against −4.3 undisturbed.

**Caveat:** the trajectories are frozen-SAC closed loops, and noise was added after the fact.

## 4. Fine-tuning interference [weights]

Relative change ‖W − W₀‖ / ‖W₀‖ from the expanded frozen initialisation, averaged over layers
and seeds:

| | actor | critic | first-layer original-obs columns, actor | new columns / original columns, actor |
|---|---|---|---|---|
| I-A | 0.97 | 1.11 | 1.64 | 0.34 |
| J2-A | 0.97 | 1.16 | 1.69 | 0.34 |
| I-C | 0.85 | 1.01 | 1.32 | 0.63 |
| J2-C | 0.87 | 1.07 | 1.31 | 0.60 |

- **The warm start was not a small perturbation in either phase.** Weights moved by about their
  own norm, and the original-observation input pathway changed by 130–170%.
- **J-2 and Phase I drift is almost identical,** layer by layer: actor layers within 0.05 of
  each other; see `results/phase_j3_audit/parameter_drift.csv`.
- The new input columns grew from zero to 34% (A) or 60% (C) of the original columns' norm. They
  were **learned, not suppressed**, in both phases.
- J2-C's reliability-flag columns are about 15% smaller than I-C's (actor 8.8 vs 10.4), and the
  one-hot columns about 7% smaller. That is consistent with the flag being less informative in
  training (§3.2), but it is descriptive only.

**Reading:**
- Large drift means the "warm-started" policies are effectively re-learned networks. Phase I's
  nominal recovery shows such drift is *compatible* with near-frozen nominal performance.
- Drift magnitude therefore **does not explain** J-2's nominal loss. What differs is *what* was
  learned, not *how far* the weights moved.
- H3, as "interference", is real in both phases: the early dip is shared. J-2's failure is the
  **failure to recover** from it.

## 5. Competing hypotheses

| | supporting | contradicting | missing | confidence | cheapest discriminating experiment |
|---|---|---|---|---|---|
| **H1** critic destabilised | critic loss +0.3–0.7; noisy critic inputs (§1) raise irreducible TD variance | no divergence; Q and loss trajectories parallel; α and entropy unchanged | TD-error distribution, gradient norms, Q calibration against Monte Carlo returns | **low** as instability; **moderate** as "noisier targets slow learning" | **A:** an asymmetric (clean, privileged) critic with a noisy actor; the only change is critic inputs |
| **H2** delay information corrupted | I-C estimator collapses at σ ≥ 0.75 cm (§3.2); delay-specific loss (−16 to −17) much larger than nominal loss (−4 to −6) | I-A: information survives (§3.1) and displacement weights are not suppressed | closed-loop feature traces of J-2 policies; input-sensitivity probe | **moderate (C), low (A)** | **C:** mixed clean and noisy episodes. Clean episodes keep the delay signal; if the delay loss shrinks, H2 is implicated. |
| **H3** fine-tuning interference | shared early nominal dip; large drift | Phase I recovers with the same drift, so drift alone is not the cause | intermediate checkpoints | **moderate** as "J-2 fails to recover", not as a distinct mechanism | **C** also tests this: clean episodes restore the nominal training signal |
| **H4** budget | J2-C still rising (+0.03 to +0.17 per 1M in the second half, on noisy 50-episode evals) | J2-A plateaued (+0.01 over the last 800k); Phase I recovered by 400k | longer runs | **low–moderate** | extend a run (but extending J-2 is barred by its own stopping rule; it would be a new protocol) |
| **H5** implementation | none | trace (§1) clean; Stage-0 tests; consistent replay | a test of the noisy training path end to end | **high: no bug** | – |

**Not resolved:** whether the uniform competence loss comes from noisy critic targets (H1 weak
form) or from the noisier training distribution as a whole (H3/H4). Both remain plausible.

## 6. Integrity

- Nothing was trained or simulated, and no seed was consumed.
- Checkpoints were read only and their hashes verified.
- Phase I, J-1 and J-2 raw data and results are unchanged.
- The synthetic noise uses a fixed offline seed (20263300); it is not an episode seed.
- These findings describe *these* baselines. They say nothing about delay-aware RL in general,
  and Candidate C's comparison status is unchanged.
