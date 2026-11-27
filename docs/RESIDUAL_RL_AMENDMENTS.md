# Residual RL: protocol amendments

Amendments to `docs/RESIDUAL_RL_PREREGISTRATION.md`. That file is **not edited**: its SHA-256
(`32330bd5…`) is still the one pinned in `configs/residual/residual_rl.json`. Each amendment is
recorded here **before** the step it affects runs.

## R-A1 (2026-11-27): critical pre-training corrections

**Approved by:** the project owner, in chat on 2026-11-27 ("Approve all critical"), for the
items marked critical in `docs/RESIDUAL_RL_PROTOCOL_REVIEW.md` §1.
**State when approved:** no residual-RL training had run, and no seed in any R block (registered
or proposed) had been used. The only simulator runs used unregistered seeds below 100,000.
**Plan approval:** R-A1 does **not** approve the pre-registration as a whole.
`approved` stays `false`, so registered training is still refused (see "Still open" below).

| id | change | where it is implemented |
|---|---|---|
| T-M-2 | T-M friction scales the sliding friction of the object **and the table** by f (`friction_mode="object_table"`). As written, object-only scaling below 1 does nothing at all: 20/20 episodes are bitwise identical to nominal, because MuJoCo takes the max of the two geoms' friction. | config key `tm_friction_mode = "object_table"`. `scripts/residual_train.py` reads it. Every evaluation call to `episodes.generate` must pass it. |
| SEED-1 | Training uses the registered block 3,710,000–3,719,999 and then **`training_2` = 3,730,000–3,739,999**, in order. That gives 20,000 episodes = 1M steps / 50. | `seeds.R_AMENDMENT_1_SEED_BLOCKS`, `training.registered_training_seeds`, `check_run` |
| SEED-2 | The nominal rollback rule (prereg §4) evaluates on **`training_monitor` = 3,740,000–3,740,099**, not on the §5 selection block. | `check_run` refuses selection seeds for monitoring. `scripts/residual_train.py` |
| STAT-1 | If step 1 or 2 fails and the upper 97.5% bound is ≥ 0, the claim is "non-inferiority not shown (inconclusive)". "Degrades C" needs an upper bound < 0. | `analysis.development_analysis` returns `claim`. The draft wording stays as `claim_draft_wording`, superseded. |
| STAT-2 | Inference is stated as **conditional on the 3 trained networks**. Step 3 (efficacy on M) also requires **each of the 3 per-network estimates to be > 0**. (The review's chosen option: 3 networks plus sign consistency. 5 networks were not chosen.) | `analysis.per_network_estimates`, `require_sign_consistency=True` (default), config `per_network_sign_consistency` |
| STAT-4 | The evaluation size follows a rule fixed now. The R-α\* vs C discordance on the §5 selection block (fraction of paired episodes with different outcomes) is measured **before** any evaluation seed is used. If it is ≤ 8%, evaluation uses 300 seeds (3,722,000–299). Otherwise it uses 600 (adding **`evaluation_extension` = 3,740,100–3,740,399**). | `analysis.discordance`, `analysis.evaluation_seed_count`, config `evaluation_rule` |

All three new blocks were checked against every block the repository registers or reserves
(`proposals.proposal_conflicts() == []`). They are below 4,000,000. The held-out seeds
4,000,000+ are untouched.

**Regression tests:**
- `test_amendment_r_a1_registers_the_proposed_blocks_unchanged`
- `test_registered_validation_logic_on_a_hypothetical_approved_copy`: selection seeds are refused for monitoring, and `training_2` is refused before `training`.
- `test_efficacy_needs_every_network_to_agree_in_sign`
- `test_evaluation_seed_count_rule`

### Still open (not approved by R-A1)

Registered training stays refused until these are settled and the plan is approved:

- **Wording-level corrections** (review: "yes, wording"): T-M-1 (mass and inertia), T-M-3 (finger contacts keep max(f, 1)), STAT-5 ("primary fails" = step 3 not passed), SAC-1 (time-limit transitions stored as terminal; the code already does this), SAC-2 (rollback does not restore the noise RNG; the code already does this).
- **To confirm:**
  - SAC-3: critic warm-up targets come from the stochastic actor.
  - SAC-4: Q-target clipping off.
  - SAC-5: 1 update per step, initial entropy coefficient 1.0, log-std clamp [−5, 2].
  - B2-1: B2 rollback is compared with frozen SAC.
  - GEN-1: selection parameters are drawn per episode.
- **To decide:** GEN-2, whether `H_undisturbed` and `H_noise0.01` stay in group D.
- **Before evaluation:** GEN-3 (P config wiring).
- **The plan itself:** set `approved: true` with the SHA-256 unchanged.

Observation from the Stage A smoke (software only, not a result): at the start of training the
entropy bonus (coefficient ≈ 0.9 × entropy ≈ 2.7 per step) is much larger than the per-step task
reward (about −0.3 to −0.7). Soft-Q estimates are therefore positive (q_mean ≈ +4 after 950 steps)
while episode returns are negative. This is standard SAC behaviour, not a bug, but it bears on
SAC-5 (initial entropy coefficient 1.0, target entropy −8) and should be considered when that
item is confirmed.

## Stage A and compute (2026-11-27; software validation only, not evidence)

Hardware: Apple M5, 10 cores, 16 GB, CPU PyTorch. No GPU, no paid compute.

| run | seeds | outcome |
|---|---|---|
| `residual_train.py --smoke --steps 2500` (16×16 nets) | smoke 0–49, monitor 100–104 | completed: 50 episodes, 3 checkpoints + final, 2 nominal checks passed (1.0 vs C 1.0), 0 rollbacks, 50 training draws over all 5 families (T-A 11, T-M 11, T-D 10, T-N 9, T-O 9) |
| `--smoke --steps 3000` | smoke 0–49 | stopped with `SeedsExhausted`, as designed: 3,000 steps need 60 episodes and the smoke list has 50. No seed was reused. |
| `residual_benchmark.py --steps 5000` (256×256, batch 256) | smoke 1000–1101, monitor 100–119 | 369 env steps/s including updates, so **≈ 0.75 h per 1M-step run** |

The registered design has 21 runs (prereg §4: 9 R-α, then 12 A1–A3/B2 runs after α\* is fixed).
That is ≈ 16–18 h sequentially on this machine, plus selection and evaluation rollouts. It is
feasible locally, with no reduction of the registered budget. The replay buffer at 1M
transitions × 65-dim observations is about 0.6 GB.

Logs are in `results/residual_stage_a/` (checkpoints are git-ignored).

**Commands, once the plan is approved** (do not run before then: `check_run` refuses them):

```bash
for a in 0.05 0.1 0.2; do for n in 0 1 2; do
  uv run python scripts/residual_train.py --arm R --alpha $a --network-seed $n \
    --out results/residual_rl/R_a${a}_n${n}
done; done
# then §5 selection (scripts/residual_select.py) fixes alpha*; then A1, A2, A3, B2 x 3 seeds
```
