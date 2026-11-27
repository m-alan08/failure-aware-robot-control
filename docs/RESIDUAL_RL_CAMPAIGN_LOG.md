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
