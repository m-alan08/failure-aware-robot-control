# Phase G4: held-out confirmation of Candidate C (proposal)

**Status:** proposal only, for approval. Nothing below has been implemented or run. No seed at or
above 4,000,300 has been touched. Candidate C's code and constants, the frozen SAC policies, the
Phase E estimator and the F1 model stay unchanged. PRs #8, #9 and #10 stay unmerged.

## 1. What is being confirmed, and what is not

Only **Candidate C** (`farc.delaychange.candidates.Sequential`, Gaussian variant, h = 12,
σ₀ = 4 mm, pinned in `docs/PHASE_G2_CALIBRATION.md`) is confirmed. No other candidate, sham or
mechanism variant is a confirmation target. Shams appear only as controls for C's claims.

The development evidence comes from two different designs, and their claims are kept apart:

| | G2 (fixed schedule grid) | G3 (random per-episode schedules) |
|---|---|---|
| Schedules | the same transition times in every episode of a condition | onset, end and on-off phase drawn per episode seed |
| C − contact-aware, short-lived and on-off delays | +1.7 [+1.1, +2.4] (Stage 3); +1.8 (exploratory) | +3.0 [+2.4, +3.7] |
| Persistent delays | +0.1, non-inferior at −1 (fixed onsets) | +1.5 (random onsets) |
| C − frozen SAC, short-lived and on-off | −1.2, non-inferior at −2 | −2.0, **not** non-inferior at −2 |
| Specificity | same-condition yoked sham: +0.5, not shown, because that sham inherited correct timing; random-timing sham check (exploratory): +1.6 | cross-episode yoked sham: +2.4, meaningful |
| Mechanism | – | pause timing (M1), reset (M2), pause × reset interaction (M3), all on development seeds only |

G4 therefore has two arms on the same held-out seeds:
- **Arm R** (random schedules, G3's generator) carries the primary claims.
- **Arm F** (G2's fixed grid) re-tests G2's claims under their original design.

Each claim is reported against the arm it came from.

### Not confirmed here: the mechanism decomposition

G3's pause-only and reset-only variants replay C's alarms. Only the first replayed alarm is
exactly causal. On a pause-only trajectory about a third of C's later alarms would not have
occurred (66% matched within ±1 step).

The replay interaction estimate is **sensitive to that alarm divergence**:
- M3 was +2.4 [+1.7, +3.1] on all pairs.
- It was +0.1 [−0.5, +0.6] on the pairs where every replay agreed with C. That subset was 77%
  single pulses and is not representative.
- The oracle 2 × 2, which has no replay, supported an interaction (+1.9 [+1.2, +2.6]), but the
  oracle is privileged.

So G3's mechanism results (M1–M3), and the size of the interaction in particular, remain
**development-only findings**. G4 does not replay alarms and does not test M1–M3. Confirming a
mechanism would need a design whose variants each run their own detector. That would be a
separate proposal.

## 2. Seeds

| block | seeds | use |
|---|---|---|
| debugging | 3,690,000–009 (fresh, validation range) | code checks only; outcomes not read |
| **confirmation** | **4,000,300–4,000,599** (300 seeds; the reserved G2/G3 held-out block) | the one run |

- 4,000,000–4,000,299 were used by the earlier contact-model confirmation and are refused.
- Nothing at or above 4,000,600 is used.
- A new guard is the *only* code path that accepts held-out seeds, and it accepts exactly
  [4,000,300, 4,000,600).
- The G3 exploratory block 3,682,000–099 stays unused.

Each episode seed runs with all 3 frozen SAC policies, giving 900 pairs per cell.

## 3. Design

### 3.1 Arm R: random per-episode schedules (primary)

- **Generator:** G3's generator unchanged (`farc.delaychange.random_schedules`, generator seed
  20261700, the ranges of `docs/PHASE_G3_PROPOSAL.md` §3). A schedule depends only on (cell,
  episode seed), so every configuration and policy of a pair sees the same schedule. The new
  episode seeds give new schedules.
- **Cells:** `RT_d2`, `RT_d4`, `RI_d2`, `RI_d4`, `RS`, `RP_d2`, `RP_d4`, `H_undisturbed`,
  `H_noise0.01`.
- **Configurations:**

| name | role |
|---|---|
| `base_only` | frozen SAC |
| `contact_learned` | the existing contact-aware compensator, the primary reference |
| `C` | Candidate C, closed loop |
| `sham_C_cross_yoked` | C's pause-and-reset mechanism driven by C's events from the next episode seed of the same cell, which has a different random schedule (G3's `sham_full_yoked`) |
| `sham_C_random_pause` | C's paused steps from the same episode, rotated to a random time, with no reset (the G2 sham-check control, as in G3) |

### 3.2 Arm F: G2's fixed grid (secondary, for G2's claims)

- **Cells:** from `evaluation_cells()` in `farc.delaychange.stage3`, unchanged:
  - 6 P cells (persistent);
  - 18 T cells (short pulses);
  - 4 I cells (on-off);
  - 4 NG cells (noise and gain loss).
- **Configurations:** `base_only`, `contact_learned`, `C`, and `sham_C_same_condition`. The last
  is G2's yoked sham, for reporting the G2/G3 contrast; it is descriptive only.

### 3.3 Size and run

| arm | cells | configurations | episodes |
|---|---|---|---|
| R | 9 | 5 | 40,500 |
| F | 32 | 4 | 115,200 |
| **total** | | | **155,700** |

From G3's throughput, this takes about 30 minutes. It is one run with write-once raw files and
a LOCK, the plan SHA pinned in the config, and a clean tree required.

- **Perturbation seed:** 20261800, new for this phase. It only affects the noise cells.
- **Bootstrap:** seed 20261800, with 10,000 episode-seed cluster resamples.

## 4. Pre-registered hypotheses

Differences are paired success-rate differences in points, with 95% percentile CIs from the
episode-seed cluster bootstrap. One-sided α = 0.025 throughout.

### 4.1 Primary: fixed-sequence gatekeeping (familywise α = 0.025, no splitting)

**Step 1, efficacy without persistent cost.** This is an intersection-union test: it passes
only if both parts hold.

- **P1, efficacy:** `C` − `contact_learned` on Arm R single pulses and on-off delays (RT + RI,
  3,600 pairs). H0: Δ ≤ 0.
- **P2, persistent non-inferiority:** `C` − `contact_learned` on all persistent cells (Arm R's
  RP, random onsets, pooled with Arm F's 6 P cells; 7,200 pairs). H0: Δ ≤ −1.0. The two arms
  are also reported separately.

**Step 2, episode-specific detection.** Tested only if Step 1 passes.

- **P3, specificity:** `C` − `sham_C_cross_yoked` on Arm R, RT + RI. H0: Δ ≤ 0.

### 4.2 Effect-size thresholds

| claim | threshold | label |
|---|---|---|
| P1 efficacy | **+2.0** (G2's pre-registered practical threshold) | **confirmed and practically meaningful** if Step 1 passes and the estimate ≥ +2.0; **confirmed, below practical threshold** if Step 1 passes and the estimate < +2.0 |
| P3 specificity | **+1.0** (G3's minimum meaningful effect) | **meaningful** if Step 2 passes and the estimate ≥ +1.0; **significant but small** if it passes below that; **negligible** if its CI lies inside (−1, +1); otherwise **inconclusive** |
| P2 persistent | margin **−1.0** | passes or fails with Step 1 |

### 4.3 Secondary (Holm across 7, α = 0.025; interpreted only as secondary)

| | contrast | arm and scope | test | source claim |
|---|---|---|---|---|
| S1 | `C` − frozen SAC | R, RT + RI | non-inferiority, margin −2 | G3 found −2.0 (not shown) |
| S2 | `C` − frozen SAC | F, T + I | non-inferiority, margin −2 | G2 S4 (−1.2, shown) |
| S3 | `C` − frozen SAC | R + F, persistent cells | superiority, > 0 | SAC fails under persistent delay |
| S4 | `C` − `contact_learned` | F, T + I | > 0 | G2 H1 (+1.7) |
| S5 | `C` − `contact_learned` | R, RS | > 0 | G3 changes of value (+2.3) |
| S6 | `C` − `sham_C_random_pause` | R, RT + RI | > 0 | G2 sham check (+1.6), now on random schedules |
| S7 | `C` − `contact_learned` | F, NG | non-inferiority, margin −2 | G2 S5 |

**Descriptive only (no test):**
- `C` − `sham_C_same_condition` on Arm F, which reproduces G2's weak specificity control,
  alongside P3;
- C's alarm latency, recall and false-alarm rate;
- incorrect and stale compensation;
- compensation activity on delayed and on delay-free steps;
- results per cell and per policy.

### 4.4 Safety gates (all must pass for a "confirmed" verdict)

1. **No delay:** `C` − `base_only` on `H_undisturbed` and `H_noise0.01`. Each needs CI low > −2
   and at most 2% of base successes harmed (as G2 and G3).
2. **No large harm in any family:** `C` − `contact_learned` with CI low > −2 in each of RT, RI,
   RS, RP, F-T, F-I, F-P and NG.
3. **Frozen components unchanged:** checked in every task, as in G2 and G3.

A failed gate means C is **not confirmed**, whatever Steps 1–2 show, and the failure is
reported.

### 4.5 Verdicts

| outcome | verdict |
|---|---|
| Step 1 passes, all gates pass | **C confirmed** on held-out seeds, with the P1 label of §4.2 |
| … and Step 2 passes | **… with episode-specific detection confirmed**, with the P3 label |
| Step 1 fails, or any gate fails | **C not confirmed**; reported as is, with no re-analysis or re-run |

Whatever the verdict, the secondary results are reported. S1 is likely to be decisive for any
statement about frozen SAC. The development evidence on it conflicts (G2 non-inferior, G3 not),
and G4 does not resolve that in C's favour by assumption.

## 5. Power (planning assumptions from development data, not predictions)

Standard errors are approximated from discordant-pair rates in G2 and G3 at G4's pair counts.

- **P1:** G3's discordance on RT + RI was 11.4%. At 3,600 pairs, SE ≈ 0.56 points.
  - If the true effect were G3's +3.0: power > 0.99.
  - If it were G2's +1.7: power ≈ 0.88.
  - Even at +1.5: power ≈ 0.77.
- **P2:** discordance was 7.3% on G3's RP cells and under 1% on G2's P cells. Pooled over 7,200
  pairs, SE ≈ 0.25–0.35 points. If the true Δ is 0, the −1 margin is about 3 SEs away: power
  ≈ 0.85–0.99.
- **P3:** G3's discordance was 10.7%, so SE ≈ 0.55. If the true effect were G3's +2.4, power is
  about 0.99. At the minimum meaningful +1.0, power is about 0.40. So a "significant but small"
  or "inconclusive" P3 would not by itself argue against a 1-point effect.
- **S1:** with G3's −2.0, non-inferiority at −2 is expected to fail. It is pre-registered
  anyway, because it is a claim the project has made before.

The 300-seed block is fixed. If you want more power for P3, the only route is a further
held-out block (for example 4,000,600–4,000,899). That would need your explicit approval of the
new range; I have not assumed it.

## 6. Integrity

- **Before any held-out seed:** this plan is approved and committed, and its SHA-256 is pinned
  in `configs/delaychange/g4_heldout.json`. The runner refuses a changed plan, a dirty tree, an
  existing output directory, or any seed outside [4,000,300, 4,000,600).
- **No tuning:** C's constants are the G2 calibration's; the generator is G3's; every threshold
  and margin is fixed above.
- **Recorded before the confirmation seeds:** implementation decisions go in a numbered
  amendments file, committed before the confirmation run.
- **Code checks only on 3,690,000–009:**
  - C via the G4 runner reproduces C via the G2 Stage-3 runner bitwise (fixed cells);
  - C via the G4 runner reproduces C via the G3 runner bitwise (random cells);
  - both shams reproduce their G2 and G3 definitions;
  - the guard refuses 4,000,000–4,000,299, 4,000,600+ and every development block.
- **The analysis is written and tested** on synthetic data and the debugging run before the
  held-out run. It is then run once, and every result is reported.
- **Preserved:** all earlier results, raw directories and models. The only edits to existing
  files are additive (seed guard, new modules).

## 7. Implementation outline (after approval)

| file | content |
|---|---|
| `src/farc/supervision/seeds.py` | `G4_CONFIRMATION = (4_000_300, 4_000_600)`, the G4 debugging block and `check_g4_seeds` (additive) |
| `src/farc/delaychange/heldout.py` | the two-arm runner, reusing `run_episode`, `sequential`, `replay`, `rotate` and the G3 generator, and the Stage-3 fixed cells |
| `src/farc/delaychange/heldout_analysis.py` | §4: gatekeeping, the IUT, thresholds, Holm for the secondary family, gates and verdicts |
| `scripts/run_g4_heldout.py`, `configs/delaychange/g4_heldout.json` | `--debug`, the one confirmation run, `--analyze-only` |
| `tests/test_g4_heldout.py` | the guard, bitwise equivalences, the verdict logic on synthetic data |

## 8. Risks

1. **One shot.** A bug found after the run cannot be fixed by re-running on the same seeds.
   *Mitigation:* bitwise equivalence with the G2 and G3 runners, the debug run, and pre-tested
   analysis code.
2. **P3 power at the minimum meaningful effect** is modest (§5). A small true specificity
   effect could come out "inconclusive".
3. **The frozen-SAC comparison may fail** (S1), as it did in G3. That would be a real limitation
   of C, not a design flaw.
4. **Arm F re-uses G2's grid,** whose same-condition sham is a known weak control. It is
   descriptive only and is not used for any specificity claim.
5. **Simulation only:** a pure transport delay with drop/replay semantics, and frozen SAC and
   F1 components. A held-out confirmation does not extend the claims beyond that setting.
