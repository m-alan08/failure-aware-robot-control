# Residual RL registered campaign: execution log

This is an operational record. The protocol is `docs/RESIDUAL_RL_PREREGISTRATION.md` as amended by
`docs/RESIDUAL_RL_AMENDMENTS.md` (hash-pinned). Nothing here amends it.

## 2026-11-27 01:07: nominal-check count discrepancy (resolved, no violation)

During monitoring, `grep -c nominal_eval log.jsonl` reported 3 nominal checks at step ~112k,
where 2 were expected (50k and 100k).

**Cause:** that pattern also matches the first log line (`kind: config`), which contains the
key `nominal_eval_seeds`. Counting only `"kind": "nominal_eval"` events gives exactly one check
per 50k steps. For example, all three α = 0.05 runs had 11 checks at step ~550k (50k, 100k, …,
550k), at 0 rollbacks.

**Classification:** an error in the ad-hoc monitoring command, not an implementation or protocol
defect. `training.train` evaluates when `step % eval_interval == 0`, which the regression test
`test_smoke_run_is_deterministic_and_logs_everything` pins at 4 checks in 400 steps. No run was
stopped or changed.

## 2026-11-27 03:28: R-α runs and selection complete

- All 9 R-α runs completed (1M steps each). Rollbacks: 0 in 8 runs; 1 in `R_a0.2_n1`, which then
  completed (one rollback is permitted; a second would have failed the run). `failed.json` is empty.
- Selection (3,721,000–099, 100 seeds, 18,000 episodes, final checkpoints only, §5):

  | α | nominal R − C | efficacy-family R − C | passes nominal (≥ −1 pt) |
  |---|---|---|---|
  | 0.05 | +1.22 pt | −0.37 pt | yes |
  | 0.1 | +0.78 pt | −0.48 pt | yes |
  | 0.2 | +1.11 pt | −0.15 pt | yes |

- Rule: best efficacy among passing α is −0.15 (α 0.2); α 0.05 and 0.1 are within the registered
  0.5-point tie band, and ties go to the smaller α, so **α\* = 0.05**. Checked by hand against
  pre-registration §5 item 2; no stop under §8.2 (an α passed) or §8.3 (0 failed R-α\* runs).
- Descriptive note, not a test: every α has a slightly negative efficacy-family difference on the
  selection block. The confirmatory question is answered only by the development evaluation.
- A1, A2, A3 and B2 runs at α\* = 0.05 started 03:28.
