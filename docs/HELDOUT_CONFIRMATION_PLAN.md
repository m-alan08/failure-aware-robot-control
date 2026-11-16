# Held-out confirmation of Phases E and F1 (plan, DRAFT for approval)

**Status: draft.**
- Nothing here has been run, and no held-out seed has been touched. Every existing loader
  still refuses seeds 4,000,000+.
- Held-out seeds are a one-time resource. This plan runs only after you explicitly approve
  it, and once it has run its results are final whatever they show.

## Purpose

Phases E2 and F1 made their claims on development seeds, so the held-out set is still clean.
Those claims are:
- online delay compensation beats frozen SAC (E2);
- contact-aware prediction beats the gripper-only predictor (F1);
- cube prediction, not gripper prediction, carries the F1 gain (exploratory only so far).

This run tests all three once, on seeds no phase has seen, with everything frozen beforehand.

## Frozen before the run (nothing changes after approval)

| item | value |
|---|---|
| SAC policies | `fetch_push_fp32_seed{0,1,2}`, verified against their manifest SHA-256 |
| delay estimator | Phase E constants (K0 = 0.62, window 10, lags 0–4, gating as in the E pilot) |
| contact predictor | the F1 model, SHA-256 `8a960329…`, unchanged. The free-space drift is **not** fixed. |
| code | one commit, recorded in the run metadata; the runner refuses uncommitted changes |
| this plan | SHA-256 pinned in the config; the runner refuses a changed plan |

## Seeds

- **Run:** held-out seeds **4,000,000–4,000,299** (300 seeds × 3 policies = 900 paired
  episodes per condition and configuration).
- **Kept in reserve:** 4,000,300–4,999,999 stay untouched for any future final test.

## Conditions (identical to the F1 evaluation, so this is a direct replication)

- **C0** undisturbed.
- **C1–C4** constant delay 1, 2, 3, 4.
- **C5** delay 3 from step 15.
- **C6** delay 3 + position noise 0.005 m.
- **C7** delay 3 + action scale 0.5.
- **C8** position noise 0.01 m.

**Delay conditions:** C1–C7. **Harm conditions:** C0 and C8.

## Configurations

| configuration | role |
|---|---|
| `base_only` | frozen SAC |
| `phase_e_predictor` | Phase E delay compensation |
| `contact_learned` | F1 contact-aware compensation |
| `learned_gripper_consistent` | corrected gripper-only ablation, from the F1 exploratory follow-up |
| `oracle_sim_predictor` | PRIVILEGED, descriptive only; not part of any hypothesis |

Total: 9 × 5 × 900 = 40,500 episodes, about 30 minutes on 3 processes.

## Hypotheses (fixed-sequence gatekeeping)

Each hypothesis is tested one-sided at α = 0.025, which is equivalent to requiring the
two-sided 95% CI to lie above the stated bound. A hypothesis is tested only if every earlier
one passed, so the family-wise error stays at 0.025 with no further correction.

Statistics: paired episode-seed cluster bootstrap with 10,000 resamples, pooled over C1–C7
(6,300 paired episodes).

| order | hypothesis | comparison | passes if | development estimate |
|---|---|---|---|---|
| H1 | delay compensation beats frozen SAC | `phase_e_predictor` − `base_only` | CI lower bound > **+5 points** | about +39 (E2), similar in F1 |
| H2 | contact-aware beats Phase E | `contact_learned` − `phase_e_predictor` | CI lower bound > 0 | +10.4 (F1), +9.6 (replication) |
| H3 | the gain comes from cube prediction | `contact_learned` − `learned_gripper_consistent` | CI lower bound > 0 | +8.0 (exploratory) |

- **Precision:** with 6,300 paired episodes, the expected CI half-width is about 1.2 points.
  The development effects for H2 and H3 are 6–8 times larger, so a true effect of half their
  size would still be detected.
- **If a hypothesis fails,** testing stops there. The later rows are reported as descriptive
  only and labelled "not tested (gate closed)".

## Safety gates (must pass for any deployment recommendation; not part of the gatekeeping family)

For `contact_learned` vs `base_only`, on **each** of C0 and C8:
- the CI lower bound is above −2 points; and
- at most 2% of base successes are harmed.

The same check is reported for `phase_e_predictor`.

## Reported regardless (descriptive)

- Per-condition differences with CIs.
- Rescued and harmed counts.
- Share of the oracle's improvement.
- Per-policy-seed effects.
- Prediction accuracy: offline cube and gripper error in contact and free states, and
  closed-loop prediction error, computed as in F1.
- How each held-out estimate compares with its development estimate. A held-out effect below
  half of its development effect is flagged as **shrinkage**, even when the test passes.

## One-shot procedure

1. **After approval, before any held-out seed is used:**
   - commit the approved plan;
   - write a dedicated held-out runner and config (pinning the plan, model and code hashes);
   - debug both on *development* seeds (3,655,000–009), checking only that the code runs and
     the tables build.
2. **The runner refuses to start unless:**
   - the config carries `"heldout_approved": "<date> by user"`;
   - the plan hash matches;
   - the git tree is clean;
   - the output directory does not exist.

   It writes a lock record (start time, commit) *before* the first episode.
3. **Single execution.** If the run crashes for an infrastructure reason (process killed, out
   of disk), I may restart it from scratch on the same seeds only if:
   - no outcome was written or looked at;
   - the crash is recorded in the lock record.

   Any other restart is forbidden.
4. **Analysis** is the pre-written analysis code, run once. Every table is committed as
   produced. Raw files are write-once, and `--analyze-only` cannot alter them.
5. **Results are reported in full,** including failures and gate closures. No configuration,
   threshold, model or seed range changes afterwards.

## What this run does not do

- No new methods: no drift fix, no faster change detector, no residual RL.
- No tuning or model selection. Nothing from this run feeds back into any method.
- No use of seeds beyond 4,000,299.

## Amendments

(none)
