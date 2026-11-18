# Phase I: a stronger delay-aware RL baseline (proposal)

**Status:** proposal and design only, for approval.
- **Nothing implemented or run:** nothing has been implemented, trained or simulated, and no
  seed in the blocks below has been used.
- **Unchanged:** Phase H results, Candidate C, the frozen SAC checkpoints, the F1 model and
  `artifacts-v1`.
- **Held-out:** 4,000,600–899 stays untouched.
- **PRs:** #8–#15 stay unmerged.

**What the audit used.** The Phase H audit in §1 reads only committed records and the
write-once Phase H raw files:
- `results/phase_h_training/*/progress.csv`;
- `results/phase_h_evaluation/`;
- `$FARC_DATA_DIR/experiments/phase_h_evaluation/episodes.csv.gz`.

No episode was run for it.

## 1. Motivation and Phase H diagnosis

### 1.1 What Phase H found (development only; `docs/PHASE_H_RESULTS.md`)

| result | estimate [95% CI] |
|---|---|
| C − R2, persistent delays | +28.6 [+27.0, +30.3] |
| R2 − frozen SAC, short and on-off delays (margin −2) | −0.2 [−1.4, +1.1] |
| R2 − C, short and on-off delays | +0.9 [−0.4, +2.2] |
| nominal-competence gate | **failed**: R1 −2.7, R2 −3.0 vs frozen SAC undisturbed |

Because the gate failed, Phase H could not say whether *a competent* delay-aware RL policy
would match compensation under persistent delay. That is the question Phase I targets.

### 1.2 Observed (measured in the Phase H records)

**O1. Success falls steeply with delay length** under delays present from step 0, in
conditions that were in R2's training distribution:

| E/F1 condition | frozen SAC | C | R1 | R2 | R2 by seed (0 / 1 / 2) |
|---|---|---|---|---|---|
| delay 1 | 78.3 | 99.0 | 90.9 | 91.2 | 93 / 91 / 89 |
| delay 2 | 31.9 | 97.6 | 50.8 | 74.4 | 81 / 78 / 64 |
| delay 3 | 12.8 | 96.3 | 23.6 | 46.9 | 55 / 44 / 42 |
| delay 4 | 9.1 | 93.3 | 14.7 | 28.2 | 39 / 24 / 22 |

Random-onset persistent delays show the same pattern: R2 reaches 85.9 at d = 2 and 56.3 at
d = 4.

**O2. R2 had not converged at 1M steps; R1 had plateaued.** Rolling training success on the
delayed training mix (noisy, since it comes from the stochastic policy):

| run | 500–600k | last 200k |
|---|---|---|
| R1 seed 0 | 0.61 | 0.75 |
| R1 seed 1 | 0.70 | 0.73 |
| R1 seed 2 | 0.71 | 0.73 |
| R2 seed 0 | 0.62 | 0.82 |
| R2 seed 1 | 0.14 | 0.70 |
| R2 seed 2 | 0.11 | 0.69 |

- R1's undisturbed evaluation curve passed 0.9 by about 400–600k.
- Two R2 seeds took off only after about 700–800k steps.
- The nominal (no-delay) retrain took off at about 400–600k.

**O3. R2 started learning late; R1 did not.** On the in-training undisturbed evaluation at 400k
steps:
- R2 seeds 1 and 2 were at 0.08, and R2 seed 0 at 0.28;
- R1 seeds 1 and 2 were already at 0.92–0.94, *ahead of* the nominal run's 0.30.

So delay randomisation alone does not explain the late start. R2's larger input and seed
variance are candidate causes, and they are not distinguished here.

**O4. Nominal cost about 3 points** (the gate). In-training undisturbed evaluation over the last
200k steps was 0.89–0.99 for R1 and R2, against 0.996 for the nominal run.

**O5. Short and on-off delays were not the problem.** R1 and R2 match or beat frozen SAC there,
for example RI_d4 87.1 for R2 against 83.9 for frozen SAC.

### 1.3 Structural facts (verified in the code; not measured outcomes)

**S1. The delay is not identifiable from a single R2 input at d = 4.**
- The observation's gripper velocity (entries 20–22) reflects the command executed at
  t − 1, which under delay d is $c_{t-1-d}$.
- R2's history holds $c_{t-1},\dots,c_{t-4}$.
- For d ≤ 3, the matching command is in the input. For d = 4 it is not ($c_{t-5}$).
- Matching a single noisy step is also much weaker evidence than the 10-transition fit
  Candidate C's estimator uses.

**S2. Given d, R2's input is Markov for d ≤ 4.**
- The augmented state of a constant-delay MDP needs the last d issued commands (Katsikopoulos
  & Engelbrecht 2003).
- R2 has the last 4. FetchPush blocks the gripper (`block_gripper=True`), so the 3 position
  components suffice.
- **The missing information is d itself, not the pending commands.** This holds for constant
  delays. Under replay/drop at an onset, already-executed commands are replayed, which R2 also
  cannot see.

**S3. R1 has neither** the pending commands nor d.

**S4. At t = 0, every training episode looks the same whatever its delay.** A policy that cannot
identify the delay must hedge across the mixture. Only 1/6 of training episodes were
delay-free.

**S5. HER relabelling is valid under delay,** because reward depends only on the achieved
goal. HER does not, however, repair a non-Markov observation: the critic $Q(o,g,a)$ is learned
on an input that omits d.

**S6. The normaliser** keeps running statistics over the delayed rollouts, including the history
entries. No evidence links it to the failure.

### 1.4 Hypotheses for why R1/R2 underperformed (not yet tested)

| | hypothesis | support | how Phase I tests it |
|---|---|---|---|
| **Hy1** | **Delay unidentifiability is the main cause of the persistent-delay failure** | O1 (worst at d = 3–4), S1, S3; R2 > R1 by +12 on persistent cells | Give the policy enough history to identify d (arm I-A), or give it d̂ directly (arm I-C) |
| **Hy2** | **The 1M-step budget from scratch was insufficient for R2** | O2 (still rising), O3 (R2's late take-off; not seen in R1) | Warm-start from the competent frozen policy, removing the take-off phase rather than adding steps blindly |
| **Hy3** | **Nominal cost comes from hedging plus a small nominal share** | O4, S4 | More nominal episodes (1/3), warm start, and identifiability (less need to hedge) |
| Hy4 | Long persistent delays were under-sampled (each (family, d) cell about 4% of episodes) | O1 | A rebalanced distribution (§4.4); confounded with Hy1 and not separately tested |
| Hy5 | SAC capacity or architecture is inadequate | none specific (3 × 256 handled the nominal task) | Not targeted; same architecture |
| Hy6 | Normaliser or HER confounds | none (S5, S6) | Not targeted; the frozen normaliser is kept for the original entries (§4.3) |

**More steps alone are unlikely to fix the persistent-delay failure.** R1 plateaued (O2), and S1
is an information limit, not an optimisation limit. Phase I therefore changes *what the policy
can know* (Hy1) and *where training starts* (Hy2). It does not simply extend Phase H's budget.

## 2. Related-work justification

The sources and access levels are in `docs/RELATED_WORK.md`.

| idea | source (access) | use here |
|---|---|---|
| Augmented state: the last d actions make a constant-delay MDP Markov | Katsikopoulos & Engelbrecht 2003 (META); Walsh et al. 2009 (META) | the basis of S2; command history in all arms |
| Delay-aware RL with augmented or belief states | BPQL (Kim et al. 2023, META); VDPO (Wu et al. 2024, META) | assume constant, known delay, so they are not directly applicable to unknown, time-varying delay |
| Random, unobservable delays | Wikman et al. 2026, ACDA (FT) | the closest method for unknown time-varying delay, but it needs an "interaction layer" in which the agent sends matrices of future actions, which changes the actuator interface (§3, arm C′) |
| Random action and observation delays | Bouteiller/Ramstedt et al. 2021, DCAC (ABS) | **unverified** whether DCAC needs the delay values to be observed. Not proposed. |
| Recurrent policies with delay or timing randomisation | Peng et al. 2018 (FT): an LSTM policy trained with HER, randomised action timestep, Fetch pushing, real robot | motivates arm B; Peng et al. randomise timing, not transport delay, and are not reproduced here |
| Delay given as an input | Sandha et al. 2021, TSRL (ABS) | arm I-C conditions on an *estimate*, not a measured delay |
| Online system identification feeding a policy | Yu et al. 2017, UP-OSI (ABS) | arm I-C is **UP-OSI-style** (identify, then condition), using our causal estimator instead of a learned OSI network. **It is not a reproduction of UP-OSI.** |
| Imitating an undelayed expert | Liotet et al. 2022, DIDA (ABS) | an alternative (§3, D); not first |
| Change-point detection for non-stationary delay | Wang et al. 2025, RA-L (META) | full text unavailable; cannot be reproduced faithfully |

## 3. Candidate architectures

All candidates keep the upstream SAC+HER algorithm and hyperparameters, through the port
verified bitwise in Phase H. Only the input, the initialisation and the training distribution
change.

| arm | input | memory | initialisation | targets | complexity | compatibility |
|---|---|---|---|---|---|---|
| **I-A** (recommended first) | observation + last **10** issued position commands + last **10** gripper displacements $\Delta g$ (25 + 30 + 30 = 85) | finite window, feedforward | **warm start** from frozen SAC seed k, with zero-initialised weights for the new inputs | Hy1 (the 10-transition window is the same information C's estimator uses), Hy2, Hy3 | low: wrapper and first-layer extension | full: Phase H wrapper, runner and analysis |
| **I-C** | observation + last 4 commands + one-hot of latched d̂ ∈ {0, …, 4} + reliability flag from the **Phase E estimator** (the same causal estimator C uses) | the estimator supplies memory | warm start, as I-A | Hy1 with *identical* identification to C; isolates "learned policy conditioned on d̂" vs "model-based compensation given d̂" | low to medium: run the estimator inside the training wrapper | full |
| B | observation + the issued command at each step, through a GRU/LSTM actor and critic | recurrent | from scratch (a recurrent model cannot inherit the feedforward weights) | Hy1 without a fixed window | **high**: sequence replay with burn-in, recurrent HER relabelling, slower updates | needs a new replay and trainer, so the Phase H bitwise port no longer covers it |
| C′ | ACDA interaction layer (Wikman et al. 2026) | model-based | from scratch | Hy1 for stochastic delays | **high**, and it changes the actuator interface (action matrices; arrival-time feedback); code availability unverified | low without changing the simulator's actuator model |
| D | DIDA-style imitation of frozen SAC under delay | augmented state | imitation | Hy2 | medium | medium |

**Is recurrence necessary? Not for the first study.**
- Identifying d needs a bounded window. Our own estimator does it from 10 transitions, and the
  maximum delay is 4. A feedforward policy on a sufficient window (I-A) is therefore a fair test
  of Hy1.
- Recurrence (B) adds sequence replay and HER complications, and would leave the verified port.
- **B is the planned follow-up only if** I-A passes the competence gate yet still fails under
  persistent delay. That would point to the policy failing to *use* the window rather than to
  missing information.

## 4. Recommended experimental design

### 4.1 Arms

| arm | role |
|---|---|
| **I-A**, 3 seeds | **primary** stronger baseline |
| **I-C**, 3 seeds | **secondary**, recommended (decision 2) |
| frozen SAC, contact-aware, Candidate C | references, unchanged |
| Phase H R2 | continuity reference: is I-A actually stronger? |

Seed k is warm-started from frozen `fetch_push_fp32_seed{k}` and paired with it in evaluation.

### 4.2 Why warm-start

- **The training curves:** Phase H's from-scratch R2 spent 700–800k of its 1M steps before
  take-off on two seeds (O2, O3).
- **Starting from the competent frozen policy** removes that phase. It also gives nominal
  competence from the first step (Hy2, Hy3).
- **Zero-initialised new input weights** make the initial policy *exactly* the frozen policy.
  Stage 0 tests this bitwise.
- **The question changes:** it becomes "can delay-aware RL fine-tuning of the base policy beat
  compensating it?". That is the comparison a practitioner faces when the base policy exists.
- **Scratch training remains possible** as an optional extra arm (decision 3), to separate the
  effect of initialisation.

### 4.3 Input representation and normalisation

- **Original 25 entries:** use the *frozen* normaliser statistics, kept frozen throughout.
  Updating them would move the warm-started policy away from the frozen one.
- **Command history** (positions, already in [−1, 1]): used unscaled.
- **Gripper displacements** $\Delta g$: divided by $K_0 \cdot 0.05$ (≈ 0.031 m), so they are
  roughly in [−1, 1]. They use the same noisy observed positions as C's estimator. No running
  statistics are kept.
- **First layers:** the actor's and both critics' first layers get the new input columns
  initialised to zero. Every other weight, and the optimiser state, is copied from the frozen
  checkpoint.
- **Temperature α:** copied from the frozen checkpoint.

### 4.4 Training delay distribution (fixed now)

Families are drawn per episode with **probabilities 1/3, 1/3 and 1/3**:

| group | content |
|---|---|
| **nominal** | no delay |
| **persistent** | constant from step 0 with d ~ U{1, …, 4} (half); random-onset persistent, the RP generator with d ~ U{1, …, 4} (half) |
| **transient** | pulse, on-off and change, the G3 generators (a third each) |

- **Generator seed:** 20262000, new, and distinct from G3/G4 (20261700) and Phase H
  (20261900).
- **This is informed by Phase H's development results:** it raises the nominal share (O4) and
  the long persistent delays (O1). That is stated openly. Evaluation uses fresh seeds, and any
  confirmatory claim needs Stage 3.

### 4.5 Protecting nominal competence

1. Warm start (§4.2).
2. A nominal share of 1/3 (§4.4).
3. Frozen original-input normalisation (§4.3).
4. **The gate** (§6.1), evaluated *before* any robustness comparison is interpreted.

### 4.6 Preventing overfitting to development conditions

- **Fixed in advance:**
  - hyperparameters are the upstream ones (with the single change of §5), and nothing is tuned;
  - the **final** checkpoint is used, with no early stopping and no selection on any evaluation;
  - the budget is fixed, with no extension after seeing results.
- **Seeds:**
  - training schedules come from a separate generator seed;
  - evaluation uses a **fresh** development block that no phase has used;
  - the evaluation includes out-of-distribution conditions (noise and gain loss, NG) that are
    not in training, reported separately.
- **Between stages:** nothing learned in Stage 2 may change the training, the arms or the
  thresholds before Stage 3, apart from documented bug fixes recorded as amendments.

## 5. Training budgets and compute costs

**Budget: 1,000,000 fine-tuning steps per run,** with the upstream configuration except
`random_steps = 0`. A warm-started policy should collect its first data with its own competent
policy, not with uniform random actions. `update_after` stays at 10,000.

**Why 1M is expected to suffice under warm start, and why this isn't "more steps":**
- Phase H's from-scratch R2 improved its delayed-mix success by about 0.2–0.6 between 500–600k
  and the last 200k steps (O2), after a long take-off.
- Warm start removes the take-off, so the full 1M goes to delay adaptation.
- The budget is unchanged. What changes is the information (window, or d̂) and the starting
  point.

**Compute** (estimates from Phase H's measured throughput, about 6,650 s per 1M-step run with 6
in parallel on this 10-core machine; the 85-entry input changes only the first layer):

| item | runs | per run | wall-clock (parallel) |
|---|---|---|---|
| Stage 0 checks (equivalence tests, a 20k-step smoke run per arm) | 2 | about 1 min | about 5 min |
| Stage 1: I-A | 3 | about 1.9 h | – |
| Stage 1: I-C (recommended) | 3 | about 2.0 h (estimator in the loop) | **about 2–2.5 h** for I-A and I-C (6 in parallel) |
| Stage 1: optional scratch I-A (decision 3) | 3 | about 1.9 h | adds about 2 h, or 9 parallel at about 3 h total |
| Stage 2 evaluation: 50 cells × 6 configurations × 900 = 270,000 episodes | 1 | – | about 30–35 min |
| Stage 3 (optional, separate approval): held-out 4,000,600–899 | 1 | – | about 30 min |
| **Total (I-A + I-C, Stages 0–2)** | | | **about 3–3.5 h**, about 12 CPU-hours |

## 6. Pre-registered hypotheses and gates

**Statistics, as in Phases G–H:**
- paired (policy seed k, condition, episode seed);
- 95% episode-seed cluster-bootstrap CIs, 10,000 resamples, bootstrap seed 20262000;
- one-sided α = 0.025;
- Holm within each family;
- minimum practically meaningful difference: 2 points.

**Scopes:**
- **Persistent:** RP ∪ F-P ∪ E C1–C7.
- **Short and on-off:** RT ∪ RI.

### 6.1 Gate (evaluated first, per arm)

An arm passes if both hold:
- **G-nominal:** arm − frozen SAC on the pooled undisturbed cells (H_undisturbed, E_C0) has
  **CI low > −2**, *and* every seed's undisturbed success is **≥ 95%**;
- **G-noise:** arm − frozen SAC on the 1 cm-noise cells (H_noise0.01, E_C8) has CI low > −2.

**If an arm fails:**
- its robustness comparisons are reported, labelled "baseline not nominally competent";
- no claim about delay-aware RL is drawn from it;
- it is excluded from Stage 3.

These thresholds are fixed now. Phase H's −2 margin is kept for comparability, and the per-seed
floor is raised from 90% to 95%.

### 6.2 Primary hypotheses: arm I-A (Holm across P1–P4)

| | question | contrast | scope | test |
|---|---|---|---|---|
| **P1** | Is a competent delay-aware RL baseline non-inferior to compensation under persistent delay? | I-A − C | persistent | non-inferiority, margin −2 |
| **P2** | Does it avoid the short-delay failure? | I-A − frozen SAC | short and on-off | non-inferiority, margin −2 |
| **P3** | Does it beat C on short and on-off delays? | I-A − C | short and on-off | superiority (> 0) |
| **P4** | Is it stronger than Phase H's R2? | I-A − R2 | persistent | superiority (> 0) |

### 6.3 Secondary (Holm across S1–S6)

| | contrast | scope |
|---|---|---|
| S1–S4 | the P1–P4 contrasts for I-C | as P1–P4 |
| S5 | I-C − I-A: does the explicit estimate help beyond the raw window? | persistent |
| S6 | I-A − contact-aware | short and on-off |

### 6.4 Descriptive

- **Per delay length:** E C1–C4, RP_d2 and RP_d4. This tests O1 directly.
- **Per seed.**
- **NG** (out of distribution).
- **Training curves.**
- **Compute and data budgets.**

### 6.5 Reading the outcomes

| outcome | reading |
|---|---|
| Gate passes; P1 holds | A competent delay-aware fine-tuned policy is non-inferior to compensation under persistent delay. **Compensation's main advantage reduces to not needing training.** This would be reported as such. |
| Gate passes; P1 fails; P4 holds | The stronger baseline improves on R2 but still trails compensation. Compensation's advantage under persistent delay survives a competent, information-matched RL baseline. **This applies to this baseline only.** |
| Gate passes; P2 holds; P3 holds | Fine-tuned RL beats C on short and on-off delays, which sharpens the "when to compensate" thesis |
| Gate fails | Reported as a failed baseline. Arm B or a different approach is the next step. No conclusion about delay-aware RL. |

### 6.6 Stopping rules

- **No early stopping:** each run trains its full budget, and its final checkpoint is used.
- **Crashes:** a run that crashes is restarted once from scratch with the identical config and
  seed. A second crash is reported, and that seed is reported missing. It is not replaced by
  another seed.
- **No extension:** there is no budget extension or retuning after Stage 1 or Stage 2.
- **Stage 3** runs only on your separate approval, only for arms that passed the gate, and once.

## 7. Risks and alternative explanations

1. **Warm-start bias.** Fine-tuning may stay near the delay-free behaviour, under-exploring
   delay-specific strategies. That would *understate* RL. The optional scratch arm (decision 3)
   measures this.
2. **Information matching is approximate.** I-A sees the same window as C's estimator but must
   *learn* to use it. Failure under P1 could reflect learning difficulty rather than an inherent
   limit; arm B would be the next test.
3. **The distribution is informed by Phase H dev results** (§4.4). Fresh evaluation seeds and an
   optional held-out Stage 3 mitigate this, but a reader should know.
4. **Replay/drop delay semantics and the mocap plant are simulator-specific,** as in all phases.
   Real-robot validity is untested.
5. **The R2 comparator was under-trained (O2).** P4 could be "easy". It is included for
   continuity, and is not evidence of strength by itself. P1–P3 against C and frozen SAC carry
   the claims.
6. **Freezing the original-input normaliser** departs from upstream's continual update, a
   deliberate change for warm-start equivalence. It is documented and tested.
7. **Single algorithm family.** Even a positive P1 would show that *one* delay-aware RL
   baseline matches compensation, not delay-aware RL in general. The reverse also holds.

## 8. Research-integrity rules (binding)

- **Prohibited:**
  - retuning anything against held-out results;
  - reusing any consumed held-out seed (4,000,000–599) for training, selection or evaluation;
  - changing Candidate C, its constants, the frozen SAC checkpoints, the F1 model or Phase H
    artifacts;
  - hiding or omitting negative results, failed gates or crashed runs;
  - reporting Stage 2 development results as confirmation;
  - claiming general superiority over, or parity with, delay-aware RL from this baseline (§7.7).
- **Required:**
  - this plan's SHA-256 pinned in the Phase I config;
  - numbered amendments committed before each stage they govern;
  - clean-tree runs, write-once raw files and pre-written analysis;
  - new checkpoints in a separate manifest;
  - Phase H results preserved byte-for-byte.
- **Seeds:**
  - Phase I debugging: **3,700,100–109**;
  - Phase I development evaluation: **3,702,000–3,702,299**;
  - both new, in the validation range, with guards refusing everything at or above 4,000,000.
  - A Stage 3 guard for 4,000,600–899 would be added only after approval.

## 9. Decisions requiring your approval

1. **Proceed with Phase I.** This trains *new* policies initialised from copies of the frozen
   weights; the frozen files are only read.
2. **Arms:** I-A + I-C (recommended), or I-A only (halves Stage 1).
3. **Optional scratch I-A arm** (3 more runs, about 2 h), to separate the warm-start effect.
   Recommended *against* for the first study, to keep it small. Approve it if attributing the
   gain to initialisation matters to you.
4. **Budget:** 1M fine-tuning steps per run, with `random_steps = 0` (§5).
5. **Training distribution:** 1/3 nominal, persistent and transient, as in §4.4.
6. **Gate thresholds:** −2 pooled and a 95% per-seed floor (§6.1).
7. **The frozen original-input normaliser** (§4.3).
8. **Seed blocks** 3,700,100–109 and 3,702,000–299.
9. **Stage 3:** decided separately after Stage 2.
