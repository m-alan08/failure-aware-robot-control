# Residual RL: pre-training protocol review and proposed amendments

**Status: proposals for review. Nothing here is approved, and the draft pre-registration was
not edited.**
- `docs/RESIDUAL_RL_PREREGISTRATION.md` still matches the SHA-256 pinned in
  `configs/residual/residual_rl.json`, which a test checks.
- No SAC training ran, and no registered seed was used. The proposed blocks below are **not
  registered**.
- No frozen result and no part of Candidate C changed.
- Every number here comes from software checks on unregistered seeds, synthetic simulation,
  or the already-collected G4 table. None is an experimental result.

**Execution approval is not requested.** The protocol-critical items in §1 have to be decided
first.

## 1. Decision table

| id | draft protocol | proposed revision | rationale | before training? |
|---|---|---|---|---|
| T-M-1 | object "mass × U[0.5, 2]" (inertia unspecified) | mass **and rotational inertia** × m (uniform density change) | MuJoCo derives the cube's inertia from its geom, so this equals recompiling with the scaled mass. Mass alone would also move mass within the block. | **yes** (wording) |
| T-M-2 | "sliding friction × U[0.5, 1.5]" of the object | scale the sliding friction of the object **and the table** by f (`friction_mode="object_table"`) | MuJoCo uses the max of the two geoms' friction. Object-only scaling below 1 is a **bitwise no-op** (20/20 episodes), and the §6 cell "friction × 0.5" would duplicate nominal. | **yes (critical)** |
| T-M-3 | (not stated) | finger–object contacts keep friction max(f, 1): unchanged for f < 1, × f for f > 1 | they are about 6% of object contact steps; stated so the asymmetry is explicit | yes (wording) |
| SEED-1 | 1M steps from training block 3,710,000–3,719,999 | add **3,730,000–3,739,999** (`training_2`), used in order after the first block | 1M / 50 = 20,000 episodes, but the block holds 10,000. The runner stops rather than reuse seeds. | **yes (critical)** |
| SEED-2 | the rollback rule evaluates on the §5 **selection** block | separate **`training_monitor` 3,740,000–3,740,099** | otherwise §5 selects on seeds already used to accept or reject checkpoints | **yes (critical)** |
| STAT-1 | step 1 or 2 fails → "the residual degrades C" | fails, upper bound ≥ 0 → "non-inferiority not shown (inconclusive)"; upper 97.5% bound < 0 → "degrades C" | failure to show non-inferiority is not evidence of harm | **yes (critical)** |
| STAT-2 | CI over episode seeds, averaged over 3 networks | state that inference is **conditional on the 3 trained networks**; also require all three per-network estimates to have the claimed sign on M (or train 5 networks) | the bootstrap ignores run-to-run variance; coverage of the method-level effect falls to 0.47 (net sd 0.5) and 0.22 (net sd 1.0) | **yes** |
| STAT-3 | margins −1 point (steps 1–2), efficacy estimate ≥ +2 points | **keep** both; document that the ≥ +2-point rule gives at most about 50% power at a true effect of exactly +2 | the margins are substantive; changing them for power would be choosing them for significance | no (document) |
| STAT-4 | 300 evaluation seeds | evaluation size set by a **pre-specified rule** from selection-block discordance: ≤ 8% → 300 seeds, otherwise 600 (extension block registered beforehand); or fix 600 now | at 10% discordance, power is 0.72 (D) and 0.65 (M) with 300 seeds, and 0.95 / 0.90 with 600 | **yes** |
| STAT-5 | secondaries "descriptive if the primary fails" | define "primary fails" as "step 3 not passed" (implemented) | the draft does not say which step | yes (wording) |
| SAC-1 | time-limit handling not stated | the time-limit transition is stored **terminal** (no bootstrap) | `ResidualEnv` returns an all-zero placeholder at the limit, and the runner was bootstrapping from it (**bug, fixed**). Option: add the observation's time feature. | **yes** |
| SAC-2 | "roll back to the previous checkpoint" | restore networks, optimisers and entropy coefficient, but **not** the exploration-noise generator; environment steps are not rewound | restoring the generator replays identical noise; not rewinding is what makes SEED-1 exactly sufficient | yes |
| SAC-3 | "20k critic-only updates" on delta = 0 data | keep. TD targets sample a′ from the current stochastic actor (standard SAC), so warm-up learns Q of the initial stochastic policy, not of C | the alternative (a′ = 0) is a different design; to confirm | confirm |
| SAC-4 | repository SAC hyperparameters unless listed | **Q-target clipping off** | penalties push rewards below −1, so the base SAC's [−1/(1−γ), 0] clip would be wrong | confirm |
| SAC-5 | (not stated) | 1 update per environment step; initial entropy coefficient 1.0; ResidualActor log-std clamp [−5, 2] (the repository SAC uses [−20, 2]) | inferred from §9 and existing code | confirm |
| B2-1 | rollback compares against "C's success" | for B2, compare against **frozen SAC** (its own base) | B2 never runs C, so C's success is the wrong reference for its protection | confirm |
| GEN-1 | selection: "one cell per T family, T-D split RT/RP" | parameters drawn per episode from the training distribution; RT/RP delay uniform on 1..4 | severities are not stated | confirm |
| GEN-2 | D = "G4 Arm R + Arm F cells" | literal: 41 cells, which includes `H_undisturbed` and `H_noise0.01` (no delay) | they dilute D; excluding them is a change | decide |
| GEN-3 | P (privileged C) on delay cells, diagnostic only | not yet wired into the generator | diagnostic only | before evaluation |
| BOUND-1 | (software) | no reset after the final training step | the extra reset consumed one more seed and raised `SeedsExhausted` when the budget fit exactly (**bug, fixed**) | fixed |

## 2. T-M: evidence for T-M-1 to T-M-3

`scripts/residual_tm_effect_check.py` ran on unregistered seeds 0–19 with the frozen
`fetch_push_fp32_seed0` policy. Each nominal command sequence was replayed open loop under
each setting. Output: `results/residual_protocol_review/tm_effect_check.json`.

| setting | object_geom (draft, literal): episodes changed | object_table (proposed): episodes changed |
|---|---|---|
| nominal (1, 1) | 0/20 (bitwise identical) | 0/20 (bitwise identical) |
| mass × 0.5 | 20/20 | 20/20 |
| mass × 2 | 20/20 | 20/20 |
| friction × 0.5 | **0/20 (bitwise identical: no-op)** | 20/20 (median max block deviation 7.5 cm) |
| friction × 1.5 | 20/20 | 20/20 (8.9 cm) |

- **Contacts:** the block's contact steps were 3,390 with the table (94%), 221 with the finger
  links (6%) and 2 with the gripper link. Under `object_table`, a block–finger contact has
  friction max(f, 1). That is unchanged for f < 1 and × f for f > 1, so the finger part of the
  perturbation is one-directional.
- **Alternative:** scaling the fingers as well would make every object contact exactly × f,
  but it also changes the finger–table contacts, which are rare. Not proposed.
- **Restoration and identity:** the model is restored bitwise after each episode, including
  abandoned ones. Nominal settings write nothing. Both are tested.
- **Inertia:** the block is `object0`, a 5 cm box with `mass="2"` and no explicit inertia, so
  MuJoCo computes I = m(a² + b²)/12 = 8.333e-4 (this checks).

**Amendment wording (proposed):**

> T-M: the object's mass and rotational inertia × m ~ U[0.5, 2.0] (uniform density change)
> and the sliding friction of the object and the table × f ~ U[0.5, 1.5], fixed for the
> episode. Finger–object contacts use max(f, 1).

The §6 cells then become mass × 0.5, mass × 2 and **object–table** friction × 0.5. Software
switch: `"tm_friction_mode": "object_table"` in `configs/residual/residual_rl.json`. Until that
key is approved and set, the draft's literal behaviour runs.

## 3. Seed blocks: evidence for SEED-1 and SEED-2

`src/farc/residual/proposals.py` holds the proposed blocks. Nothing reads them as usable
seeds.

| block | range | size | purpose |
|---|---|---|---|
| training (registered) | 3,710,000–3,719,999 | 10,000 | episodes 1–10,000 |
| **training_2 (proposed)** | 3,730,000–3,739,999 | 10,000 | episodes 10,001–20,000 |
| **training_monitor (proposed)** | 3,740,000–3,740,099 | 100 | rollback-rule nominal evaluations |
| **evaluation_extension (proposed)** | 3,740,100–3,740,399 | 300 | used only if the STAT-4 rule calls for 600 evaluation seeds |

Checks (`tests/test_residual_protocol.py`):
- **Disjoint from every registered block:** 298 blocks across `farc.supervision.seeds` and
  `SEED_RANGES`, including held-out 4,000,000+.
- **Inside the validation range** [3.5M, 4M).
- **Disjoint from each other.**
- **Unused:** no repository config or doc, and no `metadata.json` or `LOCK.json` under the
  experiment data directory, mentions 3,730,000–3,749,999.
- **Not usable today:** `check_r_seeds` refuses them, as do the smoke checks.

Exact sufficiency (20,000 = 1M / 50) relies on SAC-2: rollbacks do not rewind the
environment-step counter. If rewinding is wanted instead, add 2,000 seeds (two rollbacks ×
50k steps).

## 4. Statistical design audit (STAT-1 to STAT-5)

`scripts/residual_power_audit.py` produces `results/residual_protocol_review/power_audit.json`
and `power_audit_grid.csv`.

### 4.1 Inputs from existing G4 data

The table has 155,700 episodes, held-out 4,000,300–599, already collected and analysed in G4.

| family | C success | discordance C vs contact_learned | discordance C vs base_only |
|---|---|---|---|
| H (nominal + noise) | 0.977 (H_undisturbed: 0.997) | 0.0006 | 0.0011 |
| RT / RI / RP / RS | 0.97 / 0.86 / 0.94 / 0.94 | 0.036 / 0.194 / 0.066 / 0.089 | 0.019 / 0.137 / 0.401 / 0.667 |
| F-T / F-P / F-I / NG | 0.98 / 0.97 / 0.78 / 0.91 | 0.029 / 0.013 / 0.294 / 0.042 | 0.021 / 0.482 / 0.243 / 0.367 |

Close variants of C disagree on 0–7% of episodes in most families, and up to 19–29% on
RI and F-I. C's 99.7% nominal success caps any nominal discordance: rescue ≤ 0.3%. Success
correlates across policies within a seed (median r ≈ 0.003, up to 0.37 in some cells), so
clustering by seed matters in some cells.

### 4.2 Simulation (synthetic)

The simulation reproduces the §6 design:
- paired binary outcomes per (seed, policy, cell);
- R averaged over 3 network seeds, with the discordance draw shared across networks with
  probability 0.5;
- a logit-normal seed effect shared across policies and cells;
- an episode-seed cluster percentile bootstrap (1,000 resamples), one-sided 0.025;
- 400 simulated studies per row.

Groups: N is 1 cell (C 99.7%), D is 41 cells (C 93%), M is 7 cells (C 85%, assumed).

**The simulator itself needed three fixes before its results could be trusted.** All three
made the *bootstrap* look invalid when the fault was in the simulation:
1. it rescaled success by each run's realised mean, which coupled the clusters;
2. it set rates from the unclipped mean, which moved the true effect off the boundary;
3. it estimated E[success] from too few seed draws.

After the fixes, spot checks at the boundary give type-I error 0.020–0.035 and coverage
0.94–0.96 across N, D and M.

Final grid: rejection rate of H0 (δ ≤ −margin). The **δ = −margin** rows are type-I error
(target 0.025); the **δ = 0** rows are power.

| group | margin | discordance at δ = 0 | power, 300 / 600 / 1200 seeds | type-I range at δ = −margin |
|---|---|---|---|---|
| N | 1 pt | 0 / 0.2% / 0.54% | 1.00 / 1.00 / 1.00 (0.998 at 0.54%, 300 seeds) | 0.022–0.050 |
| D | 1 pt | 2% / 5% | 1.00 at every n | 0.015–0.035 |
| D | 1 pt | **10%** | **0.72 / 0.95 / 1.00** | (as above) |
| M | 1 pt | 2% / 5% | 1.00; 0.97 at 5% with 300 seeds | 0.012–0.035 |
| M | 1 pt | **10%** | **0.65 / 0.90 / 0.99** | (as above) |
| any | 2 pt | ≤ 10% | 1.00 | 0.012–0.035 |

Coverage of the 95% interval averaged 0.942–0.950 per group and scenario (minimum 0.88, at N
with very rare events). Type-I error was at or below 0.035 everywhere except one cell, N at
300 seeds with a 1-point margin, at 0.050. With 400 studies per row (Monte Carlo SE about
0.008), that is a mild excess, consistent with the discreteness of 0.3% event rates.

### 4.3 Conclusions

1. **The cluster percentile bootstrap is valid** for the conditional question: type-I error is
   close to 0.025 at the boundary, and coverage is close to 0.95.
2. **The −1-point margins are well powered with 300 seeds when discordance is ≤ 5%.**
   - At N the ceiling forces discordance ≤ 0.6% under no true difference, so N is always well
     powered.
   - At **10%** discordance, power at 300 seeds falls to **0.72 (D) and 0.65 (M)**. 600 seeds
     give 0.95 and 0.90.
   - G4's close C variants show ≤ 7% discordance in most families but 19–29% in RI and F-I,
     so a residual could plausibly land near 10% on part of D.
   - The margins themselves are **not** changed. They are substantive choices (1 point of
     success), and changing them to gain power would be choosing them for significance.
3. **Recommendation (STAT-4):** pre-specify the evaluation size from the selection block's
   observed discordance. Evaluate 300 seeds if pooled discordance on D and M is ≤ 8%,
   otherwise 600. Register a 300-seed extension block before any selection data exist.
   - Alternatively, fix 600 seeds now; evaluation compute is minor (§9 of the draft).
   - The efficacy step's point-estimate threshold (≥ +2 points) gives at most about 50% power
     when the true effect is exactly +2. That is a property of the claim rule, documented
     (STAT-3), not changed.
4. **Run-level variance is not covered (STAT-2).** With 3 trained networks whose true effects
   differ (log-sd 0.5 / 1.0 on harm), the episode-seed bootstrap covers the *mean-network*
   effect only 47% / 22% of the time (95% with identical networks). The CI is a statement
   about these 3 trained networks, not about "residual RL" in general.
   - **Recommendation:** word the claims as conditional, and add a per-network sign-consistency
     requirement for step 3.
   - **Alternative:** 5 networks, about +67% training compute.
   - A two-level bootstrap over only 3 networks is not reliable and is not recommended.
5. **The §5 nominal filter** (point estimate ≥ −1 point on 100 seeds × 3 policies) passes a
   harmless arm 100% of the time, an arm with 0.5 points of harm 97%, 1 point 56%, and 2
   points 1%. It is a reasonable screen. Selection and evaluation use different blocks, so the
   §6 estimates are free of the winner's curse. SEED-2 is still needed: as drafted, the
   rollback rule reads the selection seeds.
6. **STAT-1 is implemented.** `development_analysis` returns `claim_draft_wording`, the draft
   text reproduced literally, and `claim`, the corrected wording, plus per-step
   `degradation_shown` (upper bound < 0) and `degradation_beyond_margin`. Tests cover an
   inconclusive case (CI straddles −1 point and 0) and a clear-harm case.

## 5. SAC implementation audit (SAC-1 to SAC-5)

| item | finding | status |
|---|---|---|
| stochastic action sampling | pre-tanh Gaussian with the reparameterisation trick; log-prob uses the numerically stable tanh correction 2(log 2 − x − softplus(−2x)). It matches the closed form wherever the naive form is stable, and stays finite elsewhere. | correct, tested |
| target calculation | r + γ(1 − done)(min target Q − coef · log π) with fresh a′ ~ π. Bootstrapping is fully suppressed when done = 1. | correct, tested |
| time-limit transitions | **bug:** the runner bootstrapped from `ResidualEnv`'s all-zero placeholder observation | **fixed** (SAC-1) |
| entropy tuning | loss on log coef with detached log-prob; the coefficient falls when entropy is above the −8 target | correct, tested |
| update order | critic update, then actor against the updated critic (critic frozen during the actor step), then entropy, then Polyak | standard |
| critic-only warm-up | actor and entropy untouched; critic and target updated | correct, tested |
| replay buffer | ring buffer, capacity 1M ≥ budget (no overwrite in a run); float32 storage; uniform sampling from a seeded numpy generator | correct |
| rollback | networks, optimisers and entropy restored, then the actor LR is halved (Adam state restored from the checkpoint). Noise stream **no longer** restored (SAC-2). Buffer kept. | fixed, tested |
| checkpoint reproducibility | two identical runs give bitwise-identical checkpoints and logs. Checkpoints now also record the replay RNG state, buffer size and seeds used. Runs are **not resumable** mid-run (the 1M-transition buffer is not saved). | tested; documented |
| final-step reset | **bug:** an extra reset after the last step | **fixed** (BOUND-1) |

## 6. Implicit settings compared with the draft

Checked against the draft and in agreement with it: arms and observation sizes (65 / 41),
learning rates, γ, τ, batch, replay, HER off, target entropy, warm-up, α ramp, budget, policy
cycling, the rollback rule's threshold and stop, §5 selection rule and tie rule, §6 pairing
and bootstrap seed, Holm across 5, and the §8.3 stop rule (now implemented as
`rollback_stop`).

Deviations or interpretations, each listed in §1: SAC-1 to SAC-5, B2-1, GEN-1, GEN-2, STAT-5,
and the T-A/T-D perturbation seed (20261900; G4 used 20261800). One more point: the reward's
residual penalty uses the current α, which changes during the ramp, so rewards are
non-stationary for the first 200k steps. This is inherent to the draft design and noted only.

## 7. Software added in this pass

- **B2 / B0:** `SACBaseCore` (`src/farc/residual/base_core.py`).
  - With a zero residual it is frozen SAC bitwise, across 4 delay schedules × 3 seeds.
  - Only the 41-dim observation is accepted.
  - It is wired into `scripts/residual_train.py`, and a real-simulator smoke run on
    unregistered seeds completed.
- **Episode generation:** `src/farc/residual/episodes.py`.
  - The §5 and §6 cell grids (N 1, D 41, M 7, X 4; the X-DH cells use `drain_hold`).
  - `generate()` writes the table that `analysis` reads.
  - Seed roles: `selection` and `evaluation` need approval and registered seeds; `smoke` is
    unregistered only.
  - Tested with synthetic sessions and on FetchPush, where T-M is applied and restored.
  - **Real-simulator wiring smoke** (frozen `fetch_push_fp32_seed0`, unregistered seeds 0–2,
    `object_table`): 15 cells (N, all 7 M, all 4 X, 3 D), with C, R (untrained actor at
    α = 0.2) and B0.
    - R's commands were identical to C's in 45/45 episodes, so the zero-initialised residual
      is exactly C under every disturbance family.
    - B0 differed from C in 6/9 delay episodes.
    - This is a wiring check, not a result.
- **Fixes:** SAC-1, SAC-2, BOUND-1; the secondaries no longer crash when a family is absent;
  margins are now parameters of `development_analysis`.

## 8. Decisions needed (none is taken here)

1. **Before any training:** T-M-1 to T-M-3, SEED-1, SEED-2, STAT-1, STAT-2, STAT-4, SAC-1, SAC-2 (§1,
   rows marked yes).
2. **Confirm:** SAC-3, SAC-4, SAC-5, B2-1, GEN-1, STAT-5.
3. **Decide:** GEN-2 (H cells in D).
4. **Then:**
   - write these as amendments into a revised pre-registration;
   - register the new blocks in `R_SEED_BLOCKS` before use;
   - set `tm_friction_mode`;
   - re-pin the plan's SHA-256;
   - re-run the §3 identity checks and the §8.4 zero-residual check on debugging seeds
     3,720,000–009 at the approving commit.

   Only then would execution approval be asked for.
