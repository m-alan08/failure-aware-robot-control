# Research and code audit (Phases 1 – G4)

**Scope:** the full pipeline on `main` (Phases 1, B, C, D, E, E2, F1 and the E/F1 held-out
confirmation) and the four open, stacked PRs:
- [#8 G1](https://github.com/m-alan08/failure-aware-robot-control/pull/8);
- [#9 G2](https://github.com/m-alan08/failure-aware-robot-control/pull/9);
- [#10 G3](https://github.com/m-alan08/failure-aware-robot-control/pull/10);
- [#11 G4](https://github.com/m-alan08/failure-aware-robot-control/pull/11).

**Method:** reading code, configurations, pre-registrations and result records; tracing each
result directory to its recorded commit; checking raw-file hashes; re-deriving one held-out
analysis in memory; running the static checks and test suite.

**Not done:** no simulation episode was run, and nothing in `results/`, `$FARC_DATA_DIR` or
any model was changed.

**Author's caveat:** I (Claude) designed and ran Phases G1–G4 in this same project. This audit
is therefore not independent in the sense a third-party review would be. The places where my
own designs were weak are listed as findings below. An external reviewer should repeat the
checks in §7.

**Severity scale:**
- **Critical:** threatens the validity of a reported conclusion, or prevents independent
  reproduction outright.
- **Moderate:** limits how far a conclusion generalises or how strongly it can be stated.
- **Documentation:** records that are wrong, stale or incomplete, without affecting results.

## 1. Summary

| severity | count | headline |
|---|---|---|
| Critical | **1** | Checkpoints, learned models and raw data exist only on one local machine. Nothing outside it can reproduce Phases B–G4. |
| Moderate | 13 | One task, one simulator, one policy family; synthetic delay semantics; no retraining or delay-aware-RL baselines; policies treated as fixed in inference; mechanism results development-only; G-phase results only on unmerged branches |
| Documentation | 11 | Stale README and DATA.md; missing commit records for Phases B–D; Phase 1 source commits not public; "p = 0" reporting |

**No critical flaw in internal validity was found.** The held-out confirmations used
pre-registered, SHA-pinned plans; write-once raw files; clean-tree runs; seed guards; and
pre-written analyses. The one G4 analysis re-derived in this audit matched the committed tables
exactly.

## 2. Provenance: which code and models produced each result

| result directory | recorded code commit | on `main`? | models used | notes |
|---|---|---|---|---|
| `phase1_extended`, `phase1_pilot`, detector dirs | quantrl-fetch `cd8ccf5` / `110591d` (README, PROVENANCE) | n/a | SAC `fetch_push_fp32_seed{0,1,2}`, `fetch_reach_*` | Bit-identical regeneration was documented at migration. **The source branch commits (e.g. `633d953`) are not on GitHub** (checked through the API). |
| `phase_b_evaluation`, `phase_c_development`, `phase_d_*` | **not recorded** (config and model SHA recorded) | merged PRs #2–#4 | failure predictor (`$FARC_DATA_DIR/models/failure_predictor/*.pt`, SHA in metadata) | commit must be inferred from PR history |
| `phase_e_pilot` | `1166d31` | yes | SAC ×3 | |
| `phase_e2_delay` | `cf2bd00` | yes | SAC ×3 | |
| `phase_f1_dynamics` (model) | `612a90d` (in `model.json`, outside git) | yes | F1 MLP `8a960329…` | `model.json` stores an absolute local path |
| `phase_f1_eval` | `4e1b574` | yes | SAC ×3, F1 | |
| `phase_f1_exploratory_ablations` | **no metadata.json** | yes | SAC ×3, F1 | exploratory |
| `heldout_confirmation_e_f1` | `e97a3e9` | yes | SAC ×3, F1 | held-out 4,000,000–299 |
| `phase_g1_supervision` (+ calibration) | `5191c16` | **no** (#8) | SAC ×3, F1 | |
| `phase_g2_stage1` | `c58d6dc` | **no** (#9) | SAC ×3, F1 | |
| `phase_g2_stage3` (+ calibration) | `4675335` | **no** (#9) | SAC ×3, F1 | |
| `phase_g2_sham_check` | `38c22aa` | **no** (#9) | SAC ×3, F1 | exploratory |
| `phase_g3_mechanism` | `8a825c9` | **no** (#10) | SAC ×3, F1 | |
| `phase_g4_heldout` | `dd7e691` | **no** (#11) | SAC ×3, F1 | held-out 4,000,300–599 |

- Every G-phase and E/F1 run records its config SHA-256, plan SHA-256, checkpoint SHA-256s, seed
  range and git commit.
- Every loader verifies checkpoint hashes against `src/farc/base_policy/checkpoints.json`, and
  every run checks that the frozen actor's weights are unchanged afterwards.

## 3. Reproducibility

### What was verified in this audit

- **G4 re-derivation:** the analysis was re-derived in memory from the raw files
  (`derive_g4`). All eight derived tables matched the committed CSVs exactly. Both raw-file
  SHA-256 values matched those in `results/phase_g4_heldout/analysis.json`.
- **Static checks and tests:** `ruff check`, `ruff format --check`, `mypy` and the full
  `pytest` suite pass (368 tests) on the consolidation branch.
- **Bitwise checks recorded during the phases** (debugging seeds only):
  - G4's runner reproduced the G2 Stage-3 runner for C, contact-aware and the same-condition
    sham;
  - G3's `EventReplay` reproduced closed-loop C;
  - Stage 3 reproduced the Stage-1 diagnostics.

### Findings

**[CRITICAL → resolved 2026-11-17] R1. No archived artifacts.**
- **Status update, added after this audit:** the release
  [`artifacts-v1`](https://github.com/m-alan08/failure-aware-robot-control/releases/tag/artifacts-v1)
  now holds the checkpoints, models and all raw experiment directories, with a per-file
  SHA-256 manifest, and was verified before and after upload (see `docs/DATA.md`).
- **What remains:**
  - the release is private, like the repository, so outside reproduction still needs access
    or a public archive with a DOI;
  - bitwise re-trainability of the models is still undemonstrated.
- The original finding follows unchanged.
- The following exist only in `~/farc-data` on one machine:
  - the six SAC checkpoints;
  - the F1 contact model (`8a960329…`);
  - the Phase B failure-predictor models;
  - about 1 GB of write-once raw experiment files.
- There is no GitHub release (`gh release list` is empty), and `docs/DATA.md` lists the
  archival copy as "recommended, not yet done".
- **Consequences:**
  - An independent party cannot re-run any analysis from raw data, nor reproduce any result
    from Phase B onward without retraining.
  - The F1 training script is seeded and records its data hashes, but bitwise
    re-trainability has not been demonstrated.
  - The FetchPush SAC policies are retrainable from public quantrl-fetch commit `fdb6266`. That
    commit was verified to exist, but a retrained file is only accepted if its hash matches,
    which has not been shown.
- **Remedy:** a versioned archive (release assets or a DOI-backed store) of the checkpoints,
  models and raw directories, with the manifest hashes.

**[MODERATE] R2. G1–G4 live only on unmerged stacked branches.** `main` contains neither the code
nor the results of G1–G4, so a reader of `main` sees an incomplete project.

**[MODERATE] R3. Single-machine determinism.** Every run used one Apple-silicon CPU with locked
packages. Cross-machine reproducibility has not been tested; MuJoCo and PyTorch can differ in
the last bits across platforms, which can flip closed-loop outcomes. The statistical
conclusions should be robust to this, but bitwise reproduction is not guaranteed elsewhere.

**[DOC] R4. Phase 1 provenance is not public.** quantrl-fetch's `phase1-disturbance-eval`
commits are not on GitHub, as PROVENANCE.md says; this was confirmed for `633d953`. The FetchReach
checkpoints come from a dirty tree, but they are not used after Phase 1.

**[DOC] R5. Missing commit records** for Phases B–D. Several result directories have no
`metadata.json`: `failure_dataset`, `phase_aware_push`, `phase_f1_dynamics`,
`phase_f1_exploratory_ablations` and the two calibration directories. The calibration
directories record their provenance in `calibration.json`.

## 4. Statistical analysis

### 4.1 Does the code implement the pre-registered hypotheses?

Checked against each plan:

| phase | check | finding |
|---|---|---|
| E2 | H1 threshold (CI low > +5), C5 cue test, shuffled-sham specificity, harm gates | implemented as written |
| F1 | HP, HC, gates, HA, HO | implemented. **HA was mis-designed** (an inconsistent observation), which is disclosed, and the attribution was later confirmed held-out with a corrected ablation (H3). |
| E/F1 held-out | fixed-sequence H1 → H2 → H3, gates | implemented |
| G1 | Holm across G1–G4, margins, gates, yoked and timeout shams | implemented |
| G2 Stage 3 | IUT (H1 ∧ H2) per candidate, Holm across candidates, 12-test secondary Holm, gates | implemented. The proposal's "+2 practical threshold" is reported but never used as a pass criterion, as planned. |
| G3 | gatekeeper G, Holm M1–M4, two-sided M3 via a four-way contrast, labels, sensitivity | implemented. **Label-rule quirk:** M2's CI lies wholly below 0 (−1.3 [−2.0, −0.7]), yet the pre-registered rule labels it "inconclusive". The results document states the harm explicitly. |
| G4 | fixed sequence: IUT(P1, P2), then P3; thresholds; Holm across S1–S7; gates; verdict | implemented and re-derived in this audit |

### 4.2 Is the design appropriate?

**Strengths:**
- **Pairing:** every configuration in a cell runs the same episode seed, perturbation seed,
  policy and (from G3 on) delay schedule. Paired differences remove most between-episode
  variance.
- **Cluster bootstrap on episode seed:** this accounts for correlation across conditions and
  policies that share a seed. It is the right unit, because the same seed sets the initial
  state in every condition.
- **Gatekeeping and IUT:** fixed-sequence testing (E/F1 held-out, G3, G4) and
  intersection-union tests (G2, G4) control the familywise error without splitting α.
- **Holm** is correctly implemented (`farc.adaptation.delay_eval.holm`: step-down with a
  monotone maximum).

**Findings:**

**[MODERATE] S1. Policies are fixed effects.**
- Only 3 SAC policies are used, and the bootstrap does not resample them. CIs describe
  uncertainty over episodes for *these three policies*, not over the population of SAC
  policies.
- Per-policy held-out results are consistent but vary:
  - G4 P1 is +3.0, +2.1 and +5.3;
  - E/F1 H2 is +8.8, +8.0 and +11.3.
- With 3 policies, a policy-level random-effects interval would be very wide. Claims should be
  stated as conditional on these policies.

**[MODERATE] S2. Bootstrap p-values.**
- One-sided p-values are the share of bootstrap draws at or beyond the bound, which is a
  percentile-inversion p-value. It is consistent with the CIs reported, but it is not a test
  under a null-centred distribution.
- "p = 0.0" in the tables means p < 1/10,000 (see D6).
- For the effects that matter (z ≈ 4–20) this does not change any decision. It could matter
  near thresholds:
  - G3's secondary M1 on RS: Holm p = 0.066 against 0.025;
  - G3's `yoked_timing`: Holm p = 0.050.

**[MODERATE] S3. Many development hypotheses.**
- Each phase controls its own family, but across Phases E–G3 dozens of development-seed
  contrasts were run, with designs adapted between phases.
- Development findings should therefore be read as exploratory unless they were confirmed on
  held-out seeds. The confirmed set is E/F1 (H1–H3) and G4 (P1–P3, S1–S7).

**[MODERATE] S4. Weak shams in E2 and G2.** Disclosed in both phases:
- E2's shuffled sham and G2's same-condition yoked sham drew from episodes that shared the
  true delay or the transition times.
- They could not test specificity. G3 and G4 fixed this with random per-episode schedules.

**[MINOR] S5. Degenerate gate CIs.** With 0 harmed and 0 rescued, the CI is [0, 0] and the
"CI low > −2" part of the gate is vacuous. The harmed-fraction criterion carries the gate in
those cells.

**[MINOR] S6. Pooling across the G4 arms.**
- P2 pooled random-onset persistent cells (Arm R) with the fixed persistent cells (Arm F).
  This was pre-registered, and both components are reported:
  - RP +1.1 [−0.2, +2.3];
  - F-P +0.3 [+0.04, +0.63].
- Each component alone also clears the −1 margin. The consolidated results below report the
  components separately.

## 5. Leakage, privileged information, simulator shortcuts, invalid comparisons

### Checked and found sound

**No seed leakage:**
- Every phase draws from disjoint, guarded seed blocks (`farc.supervision.seeds`,
  `farc.contact.data`, `farc.disturbances.config`).
- The F1 model's training seeds (3,650,000–199) and selection seeds (3,651,000–049) are
  disjoint from every evaluation block.
- The two held-out slices (4,000,000–299 and 4,000,300–599) were each used once by a single
  guarded code path.

**No observation leakage:**
- `EpisodeSession.observe` applies observation noise *before* building the controller view.
- The estimator (`est.observe(view.gripper)`) and the predictors read that noisy view, not
  simulator ground truth.
- The ground-truth gripper log is separate, and `test_gripper_positions_are_ground_truth_and_ignore_observation_noise`
  covers it.

**Privileged configurations are fenced:**
- ORACLE, `PRIV_*` and `ORACLE_*` are named, used only as references, and never in a
  deployable claim.
- Phase E/F1 tests assert that only oracles read the true parameters
  (`test_only_the_oracle_reads_true_parameters`, `test_oracle_and_predictor_are_exclusive`).

**F1 training targets** come from noise-free observations. That is standard supervised
labelling, and the deployed inputs are the noisy observations.

### Findings

**[MODERATE] L1. No automated privileged-access test in G2–G4.**
- In the G2–G4 runners (`run_g2_episode`, `farc.delaychange.mechanism.run_episode`), the true
  delay is computed every step for logging and for the PRIV modes.
- Deployable controllers receive only the estimator object, so they cannot read it by
  construction.
- Unlike Phases E and F1, though, no test asserts this.

**[MODERATE] L2. Simulator-specific delay semantics.**
- Delay is a pure transport delay of *issued* commands under replay/drop semantics:
  - at an onset, already-executed commands are replayed;
  - at an end, queued commands are dropped.
- G2 Stage 1 compared a drain/hold alternative only on its own grid, and found no
  dependence (0.0 [−2.1, +2.0]). G3 and G4 used replay/drop only.
- Real actuator latency, jitter and buffering behave differently.

**[MODERATE] L3. Mocap position control with full state.**
- FetchPush drives a mocap weld with position-displacement commands. The plant is nearly
  linear (R² 0.94–1.00), which is what makes least-squares lag identification work so well.
- The policy, the estimator and the F1 model all use simulator-provided object pose and
  velocity. A real system would need perception and state estimation.

**[MODERATE] L4. F1 model trained in-simulator, in-distribution.**
- Its training conditions (delays 1–4, scale 0.5, undisturbed; base and Phase E rollouts)
  overlap the E/F1 evaluation conditions.
- Its use under G2–G4 schedules is out of its training distribution. That worked, but it is
  not characterised.

**[DOC] L5. "Oracle" is not always an upper bound.**
- G1's `oracle_sim_predictor` (T3: 59.3%) and G2 Stage 1's oracle anomaly come from executing
  placeholder actions with a stale delay estimate.
- Both are disclosed. The term should not be read as "ceiling" without qualification.

### Comparisons that need care

- **Different phases are not comparable:** each has different seeds, conditions and
  condition mixes. Frozen SAC's success on "delay 3" ranges from 13.4% to 16.7% across
  blocks, as expected. Effects must be compared within a phase.
- **Fixed vs random schedules:**
  - G2's fixed grid and G3/G4's random schedules give different effect sizes for the same
    controller: C − contact-aware is +1.4 to +1.7 on the fixed grid and +3.0 to +3.5 on random
    schedules.
  - They must not be pooled. The difference is a property of the schedule distribution,
    which the experimenter chose.

## 5b. Scope and baselines

**[MODERATE] B1. One task, one simulator, one policy family.**
- Every compensation result is on FetchPush-v4 (MuJoCo, Gymnasium-Robotics) with three SAC+HER
  policies trained by one codebase.
- FetchReach appears only in Phase 1. No other manipulation task, simulator or policy class
  was tested.

**[MODERATE] B2. Missing strong baselines.**
- The baselines are frozen SAC and this project's own simpler compensators.
- Not compared:
  - a policy retrained or fine-tuned with delay randomisation;
  - delay-aware RL that augments the state with the action buffer;
  - an analytic or physics-based object predictor, such as a MuJoCo look-ahead used causally;
  - MPC.
- "Without retraining" is shown to work. Whether it beats retraining, and at what cost, is not
  known.

**[MODERATE] B3. Mechanism claims are development-only.**
- G3's decomposition (pause timing, reset, interaction) used development seeds and replayed
  alarms.
- About a third of later replayed alarms are not causal for the variant.
- The interaction disappears (+0.1) in the agreeing subset. That subset is unrepresentative,
  so this is not decisive either way.
- G4 deliberately did not test the mechanism.

## 6. Claims vs evidence

**Supported as stated, on held-out seeds in simulation:**
- Online delay identification with model-based compensation, without retraining the policy:
  +44.9 over frozen SAC on constant and mid-episode delays.
- Contact-aware prediction beats the static-object predictor (+9.3) and the gripper-only
  learned predictor (+8.7).
- Candidate C beats the contact-aware compensator:
  - on random short-lived and on-off delays, +3.5;
  - under persistent delay, +0.5, non-inferior.
- C beats the same pause-and-reset mechanism driven by another episode's events (+3.2).

**Stated more strongly than the evidence supports:**

| claim (where) | issue | suggested wording |
|---|---|---|
| "C's benefit depends on detecting each episode's own transitions" (G4 results, conclusion 2) | The control was the same mechanism driven by *mis-timed* events. A controller that exploits the schedule *distribution* optimally (e.g. a timer tuned to the generator) was not tested. | "C outperforms the same pause-and-reset mechanism driven by another episode's alarms" |
| "deployable" (G2–G4 documents) | It means "causal; uses no privileged information in simulation", not hardware-deployable. | "causal (non-privileged)" |
| "Two mechanisms, each doing a different job" (G3) | Development-only. The interaction estimate is sensitive to alarm divergence: +2.4 on all pairs, +0.1 on fully agreeing pairs. | already qualified in G3; keep that qualification wherever it is cited |
| "The effect generalises" (E2) | It generalises to more delay lengths and onsets in the same simulator and task. | "across the tested delay conditions" |
| README "Findings so far" | Superseded and stale (see D1) | update |

**Not shown anywhere:**
- real-robot validity;
- other tasks, simulators or policy classes;
- superiority to a policy retrained with delay randomisation or to delay-aware RL;
- novelty relative to the literature.

## 7. Documentation issues

| id | issue |
|---|---|
| D1 | `README.md` says the held-out range was never used and that residual RL, recovery and the learned predictor were not started. Its Findings cover Phase 1 only. |
| D2 | `docs/DATA.md` seed table says 4,000,000+ was "never used so far". The archival copy is listed as not done (see R1). |
| D3 | Phases B–D metadata lack a git commit (see R5). |
| D4 | `model.json` (F1) records an absolute local path. |
| D5 | Phase 1 source commits are not public (see R4). |
| D6 | p-values are printed as `0.0` instead of `< 1e-4` throughout the G-phase tables. |
| D7 | G3's M2 label quirk (see §4.1). |
| D8 | G2 Stage 3's first draft reported 24 no-delay alarms instead of 123; corrected in the same PR. |
| D9 | G1's specificity sham, G2's S2 and E2's shuffled sham share one design weakness; the reasoning is spread over three documents. |
| D10 | Overhead timings were measured with 9 concurrent processes on a laptop; they are labelled descriptive, but no single-process or real-time measurement exists. |
| D11 | `results/README.md` describes only the Phase 1 tables. |

## 8. Checks for an external reviewer

1. Restore the archived artifacts (once R1 is fixed) and verify the checkpoint and model hashes.
2. Re-run `--analyze-only` for `heldout_confirmation_e_f1`, `phase_g3_mechanism` and
   `phase_g4_heldout`, and diff against `results/`.
3. Re-run the G4 debug run on 3,690,000–009 on a different machine. It must reproduce the
   bitwise equivalence checks.
4. Add and run a test that G2–G4 deployable controllers never read the schedule (L1).
5. Read the six pre-registration documents next to the results documents and confirm that no
   analysis choice was made after the data. The amendment files list every implementation
   choice and when it was committed.
