# Phase H: does retraining with delay randomisation beat compensating a frozen policy? (proposal)

**Status:** proposal only, for approval.
- Nothing has been implemented, trained or run, and no seed in the blocks below has been used.
- The frozen SAC policies, the F1 model, Candidate C and every earlier result stay unchanged.
- PRs #8–#14 stay unmerged.

**This phase departs from a standing constraint.** Every phase so far followed "never retrain
the base SAC policies". Phase H trains **new** policies:
- in new files, under new identifiers;
- with the frozen checkpoints only read and still hash-verified.

It must not start without your explicit approval of that departure. Its held-out stage (§6)
needs a separate approval for a new held-out slice.

## 1. Why this phase

The related-work audit (`docs/RESEARCH_POSITIONING.md` §1.6) names one comparison as the most
direct challenge to the project's central claims: a policy **retrained with delay
randomisation**, with and without the recent commands in its input. Peng et al. (2018) did
exactly this kind of retraining for Fetch puck pushing, with randomised action timing, on a
real robot.

Two claims depend on it:

| claim (current evidence) | what the comparison decides |
|---|---|
| Compensation recovers a frozen policy under persistent delay (E/F1 HO +44.9; G4 HO P2) | Is compensation competitive with simply retraining? If not, "no retraining" is only a cost argument. |
| Under short, frequent delays, stale compensation does worse than the uncompensated policy (G4 HO: C − SAC −2.2; contact-aware −9.5 on on-off delays, descriptive) | Is this a property of *compensators*, or of *delay in general* that retraining also suffers? |

## 2. Feasibility, from reading the code

**SAC+HER training is not in this repository.** Only the frozen actor was ported
(`src/farc/base_policy/`). Training lives upstream in `m-alan08/quantrl-fetch`:
- `src/quantrl/training.py`;
- `algos/sac.py`, `algos/replay.py` and `envs/factory.py`;
- `scripts/train.py`.

The training loop:
- seeds the environment once per run (`env.reset(seed=config.seed)`) and then draws episodes
  from that stream, exactly as the frozen policies were trained;
- has evaluations during training (`eval_interval`) on quantrl-fetch's own evaluation seeds.

**Training configuration** (`configs/fetch_push.json` upstream):
- SAC with 3 × 256 hidden layers, batch 512, γ 0.98, τ 0.005;
- HER with replay_k 4 and a 20,000-episode buffer;
- 1,000,000 environment steps, 10,000 random steps, 0.5 updates per step.

**Provenance discrepancy found while inspecting this (reported, not fixed):**
- `src/farc/base_policy/checkpoints.json` and `PROVENANCE.md` record the FetchPush policies as
  trained at quantrl-fetch `fdb6266` with `configs/fetch_push.json` on a clean tree.
- That file does **not exist** at `fdb6266`. It was first added two commits later, in
  `95e9611` ("Phase 2: FetchPush at fp32 (#6)", 2026-10-14). The GitHub compare API shows
  `95e9611` 2 commits ahead of `fdb6266`.
- Possible explanations:
  - training ran with the config as an uncommitted, untracked file, which a "clean tree"
    check that ignores untracked files would miss;
  - or the recorded commit is imprecise.
- Either way, `docs/DATA.md`'s retraining instruction ("at `fdb6266` with
  `configs/fetch_push.json`") does not work as written. It has been corrected to point to
  `95e9611` with this caveat.
- Bit-exact retraining of the frozen checkpoints remains undemonstrated. Stage 0 below would
  test it.

**What is needed:**
1. **Port the training code** into `src/farc/retrain/`, with a provenance header and unchanged
   apart from imports and data paths.
   - It will not be imported by any existing module, so earlier code paths are untouched.
2. **An environment wrapper** that applies a `DelaySchedule` to executed actions with the
   replay/drop semantics of `ScheduledDelaySession`. Its executed-action stream must be bitwise
   identical to the session's on debugging seeds.
3. **Optional observation augmentation** with the last 4 issued commands (12 position
   components), appended to the observation. The normaliser handles the larger dimension
   unchanged.
4. **Evaluation:** a loader for the new checkpoints, under new identifiers, with their own hash
   manifest. The G4 runner is reused as is: these policies replace `base_only` in their own
   configuration.

**Compute:** not measured. The upstream logs are not in git. Stage 0 measures throughput on a
20,000-step run, and the full budget is reported to you before Stage 1 starts.

## 3. Design

### 3.1 Arms (all new policies; frozen checkpoints untouched)

| arm | training | policy input | role |
|---|---|---|---|
| **R1: randomised** | SAC+HER from scratch, delay schedule drawn per episode (§3.2) | the standard observation | delay-randomised retraining |
| **R2: randomised + command history** | as R1 | observation + last 4 issued commands | delay-aware state augmentation, a Markov-restoring input in the spirit of constant-delay augmentation |

- **Seeds:** 3 training seeds per arm (0, 1, 2, matching the frozen policies' seeds).
- **Budget:** the original budget of 1,000,000 steps.
- **Hyperparameters:** the upstream FetchPush configuration, unchanged, with **no tuning**.
- **Checkpoint:** the **final** one, with no early stopping and no selection on any evaluation.

### 3.2 Training delay distribution (fixed now)

Per training episode, one family is drawn uniformly. The schedule generator is the G3/G4
generator, unchanged, but with a **separate generator seed (20261900)** and a per-run episode
counter, so no evaluation schedule is reused.

| family | share | schedule |
|---|---|---|
| none | 1/6 | – |
| constant from step 0 | 1/6 | d ~ U{1, …, 4} |
| persistent, random onset | 1/6 | the RP generator, d ~ U{1, …, 4} |
| single pulse | 1/6 | the RT generator, d ∈ {2, 4} |
| on-off | 1/6 | the RI generator, d ∈ {2, 4} |
| change of value | 1/6 | the RS generator |

**This favours the retrained policies.** Their training distribution covers the evaluation
families, while C's constants were set on G2's calibration cells. That is the realistic
comparator, a practitioner who knows the delay distribution, and it is reported as such.
Gain loss and observation noise are not in the training distribution, so the NG cells are
out-of-distribution for R1 and R2. They are evaluated descriptively.

### 3.3 Stages and seeds

| stage | seeds | use |
|---|---|---|
| 0: port checks | debugging block 3,700,000–009 | code checks only, outcomes not read |
| 1: training | environment streams from training seeds 0–2 (upstream mechanism); schedules from generator seed 20261900 | 6 training runs |
| 2: development evaluation | **3,701,000–3,701,299** (300 episode seeds × 3 policies per arm) | pre-registered, run once |
| 3: held-out confirmation | **4,000,600–4,000,899**, *requires separate explicit approval* | the same hypotheses, run once |

- **Seed guards:** all new blocks get guards. Debugging and evaluation blocks sit in the
  validation range and are disjoint from every used block (F1, E2, G1–G4).
- **Held-out:** the held-out slice would get a new single-purpose guard, like G4's. Nothing
  at or above 4,000,900 would be touched.

**Stage 0 checks** (debugging seeds only):
- The wrapper's executed actions are bitwise identical to `ScheduledDelaySession`, on every
  family.
- The ported trainer is bitwise identical to upstream on a 20,000-step run (parameters and
  replay contents), comparing ported against upstream code at `95e9611`.
- **Optional:** a full 1,000,000-step nominal (no-delay) retrain of seed 0, compared by hash
  with the frozen `fetch_push_fp32_seed0`. This would settle whether the frozen checkpoints
  are bit-reproducible (see §2). It costs one extra training run.

### 3.4 Evaluation configurations (stages 2 and 3)

| configuration | source |
|---|---|
| `base_only` (frozen SAC) | as G4 |
| `contact_learned` | as G4 |
| `C` | Candidate C, as G4 (pinned constants) |
| `retrained_R1` | new, frozen after training |
| `retrained_R2` | new, frozen after training |

**Cells:**
- G4's Arm R (RT, RI, RS, RP, H) and Arm F (fixed P, T, I, NG);
- the E/F1 constant-delay conditions C0–C8, for continuity with H1–H3.

**Pairing:**
- Episodes pair on (policy seed, condition, episode seed): the same initial state, goal and
  delay schedule for every configuration.
- Retrained seed k pairs with frozen seed k by index only. They are different policies, so
  per-seed results are always reported.
- The bootstrap clusters on episode seed, as before.

**Size** (stage 2): 50 cells (G4's 41 plus E/F1's 9) × 5 configurations × 900 = 225,000
episodes. Using G4's throughput (155,700 episodes in 14 minutes), that is about 20–25 minutes.

## 4. Hypotheses (identical in stages 2 and 3; stage 3 is confirmatory)

Differences are success-rate points, paired, with 95% cluster-bootstrap CIs. One-sided
α = 0.025.

### 4.1 Primary family (Holm across H1–H3)

| | question | contrast | scope | test |
|---|---|---|---|---|
| **H1** | Does retraining avoid the short-delay failure? | R2 − frozen SAC | random RT + RI | non-inferiority, margin −2 |
| **H2** | Does retraining beat the compensator where compensation struggles? | R2 − C | random RT + RI | superiority (> 0) |
| **H3** | Is the compensator competitive with retraining under persistent delay? | C − R2 | persistent: RP + fixed P + constant C1–C7 | non-inferiority, margin −2 |

R2 is the primary retrained arm because it is the stronger, delay-aware variant. R1 is
secondary.

### 4.2 Secondary (Holm across S1–S5)

- **S1:** R2 − R1 on RT + RI and on persistent cells. Does the command history help?
- **S2:** R1 − frozen SAC on RT + RI (margin −2).
- **S3:** C − R1 on persistent cells (margin −2).
- **S4:** R2 − contact-aware on RT + RI.
- **S5:** R2 − C on noise and gain loss (NG), out-of-distribution for R2. Descriptive only.

### 4.3 Gates for the retrained policies

These are checked before any comparison is interpreted:
- **Nominal competence:** R1 and R2 success on undisturbed episodes (H_undisturbed, C0) has CI
  low > −2 vs frozen SAC, per arm. This tells us whether delay randomisation cost nominal
  performance.
- **Training success:** a run whose final policy has undisturbed success below 90% is
  **reported as failed**.
  - It is kept in the analysis.
  - It is not retrained, re-seeded or tuned.
  - Results are also shown excluding it, labelled post hoc.

### 4.4 Reported alongside (descriptive)

- **Per-family success** for all five configurations.
- **Per training seed,** for R1 and R2.
- **Budgets:**
  - environment steps and wall-clock for R1 and R2;
  - for the compensator: the F1 model's 352,800 training transitions plus the estimator and
    detector, with no policy training.
- **Per-step controller cost.**

## 5. How the outcomes would be read

| outcome | meaning for the paper (`docs/RESEARCH_POSITIONING.md`) |
|---|---|
| H1 holds, H2 holds | Retraining avoids the short-delay failure and beats the compensator there. The negative finding is about *compensators*; the paper must say retraining is preferable for short, frequent delays. |
| H1 fails | Retraining also loses to the frozen policy on short delays. The failure is about *delay in general* for this task, which strengthens the negative finding as a general result. |
| H3 holds | Compensation without retraining is competitive under persistent delay. "No retraining" is supported on performance, not only on cost. |
| H3 fails | Retraining is better under persistent delay too. Compensation's case reduces to cost and to settings where retraining is impossible. |
| S1 positive | The command history matters, consistent with delay-aware RL theory. |
| Gates fail | The retrained baseline is weak at this budget. Its comparisons are reported, but no conclusion about retraining *in general* is drawn. |

Every outcome is publishable as part of the primary thesis. None changes earlier results.

## 6. Integrity

- **Before any seed or training run:**
  - this plan is approved and committed, and its SHA-256 is pinned in the Phase H config;
  - the departure from the "no retraining" constraint is approved explicitly.
- **Training:**
  - every run requires a clean tree and records its commit, config SHA-256 and seed;
  - each final checkpoint gets a SHA-256 in a new manifest (`src/farc/retrain/checkpoints.json`);
  - frozen checkpoints are only read and are hash-verified.
- **No selection or tuning:** hyperparameters are fixed (§3.1), and the checkpoint is the
  final one. Stage 2 results cannot change anything before Stage 3, apart from documented bug
  fixes recorded as numbered amendments.
- **Evaluation:** write-once raw files, a LOCK, a pre-written analysis, a single run, and the
  `scripts/verify_reproduction.py` pattern extended to cover Phase H.
- **Archive:** the new checkpoints and raw data would join a new release (`artifacts-v2`).
  `artifacts-v1` stays unchanged.

## 7. Risks

1. **Port fidelity.** A subtle difference from upstream would weaken the baseline. *Mitigation:*
   the bitwise trainer check (Stage 0), and optionally the hash comparison with a frozen
   checkpoint.
2. **The training budget may not suffice.** Delay randomisation makes the task harder, and
   1M steps may under-train R1 and R2. *Mitigation:* the competence gate (§4.3) and honest
   reporting. A pre-specified 2M-step variant is a possible extension (decision 3 below), but
   no post-hoc extension.
3. **The non-delay-aware arm faces a POMDP.** R1 cannot see the delay, so HER's relabelled
   transitions are non-Markov for it. That is why R2 is primary.
4. **Favourable training distribution** (§3.2). This is reported explicitly. An
   out-of-distribution delay test, such as a delay of 5, is not possible, because the
   compensator's estimator stops at lag 4.
5. **Compute** is unknown until Stage 0. You see the estimate before Stage 1.
6. **Policy-level variance.** Retrained policies vary more across seeds than frozen-policy
   compensation. Three seeds give limited power for policy-level claims, so per-seed results are
   always shown.
7. **Held-out consumption.** Stage 3 uses a fresh 300-seed held-out slice and needs your
   approval.

## 8. Decisions for you

The defaults are my recommendation. Approving the plan approves them.

1. **The departure from "never retrain"** for these *new* policies: required.
2. **Arms:** R1 and R2 (default). Alternatives: R2 only, which halves the training; or add
   **R3**, fine-tuning a *copy* of each frozen policy with delay randomisation for 200,000
   steps. R3 is a cheaper "light retraining" comparator.
3. **Budget:** 1M steps (default, matching the frozen policies), or a pre-specified 2M.
4. **Optional nominal retrain** of seed 0, to test bit-reproducibility of the frozen checkpoint
   (default: include, if Stage 0 timing allows).
5. **Stage 3 held-out confirmation** on 4,000,600–899: approve now, or decide after Stage 2
   (default: after Stage 2).
