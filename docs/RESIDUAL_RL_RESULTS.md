# Residual RL: registered campaign results (development evaluation)

Protocol: `docs/RESIDUAL_RL_PREREGISTRATION.md` (SHA-256 `32330bd5…`) as amended by
`docs/RESIDUAL_RL_AMENDMENTS.md` (R-A1, R-A2; SHA-256 `68462908…0234`). Nothing in the protocol,
training configuration or evaluation criteria changed during the campaign. Operational events are
in `docs/RESIDUAL_RL_CAMPAIGN_LOG.md`. **No held-out seed (≥ 4,000,000) was used.**

## Verdict

**Pre-registered claim: "null for Q1".** The residual policy does not detectably improve on
Candidate C on the not-modelled failures (M).

- **Step 1, nominal protection: passed.**
- **Step 2, delay protection: passed.**
- **Step 3, efficacy: failed.** The 95% interval upper bound, +0.70 points, is below the
  registered +2-point practical threshold. So the data rule out a practically meaningful efficacy
  effect, not just fail to show one. This is conditional on the three trained networks.
- **Step 4, generalisation (Q2): not tested.** Gatekeeping stops at step 3.

Recommendation: **NO-GO**. Details are at the end.

## Campaign

| Stage | Outcome |
|---|---|
| 9 R-α runs (α ∈ {0.05, 0.1, 0.2} × 3 networks, 1M steps) | all completed; 1 rollback (`R_a0.2_n1`), 0 failures |
| Selection (3,721,000–099, 18,000 episodes, final checkpoints) | all α pass nominal; efficacy-family R − C = −0.37 / −0.48 / −0.15 points; tie rule → **α\* = 0.05** |
| 12 A1/A2/A3/B2 runs at α\* | all completed; 1 rollback (`A3_n0`), 0 failures |
| Development evaluation (600 seeds, 53 cells, 3 policies, 17 configurations) | 1,621,800 episodes and 70,200 privileged-P episodes; `failed.json` empty |

The evaluation was paused for 45 minutes by a low-battery hibernation. This changed no outcome:
the episodes are deterministic, and all 33 re-simulated trajectories (5 configurations each) reproduced the evaluation
table exactly.

## Confirmatory analysis (points; 95% episode-seed cluster bootstrap, 10,000 resamples)

| Step | Group | R − C | 95% CI | Criterion | Result |
|---|---|---|---|---|---|
| 1 nominal protection | N | +0.46 | [−0.00, +0.94] | lower bound > −1 | **passed** |
| 2 delay protection | D (39 cells) | +0.29 | [+0.01, +0.57] | lower bound > −1 | **passed** |
| 3 efficacy (Q1) | M | +0.30 | [−0.09, +0.70] | lower bound > 0, estimate ≥ +2, all 3 networks > 0 | **failed** (all three conditions) |
| 4 generalisation (Q2) | X | (+0.22) | [−0.31, +0.76] | lower bound > 0 | not tested |

Per-network efficacy estimates are −0.04, +0.60 and +0.35 points. Network 0 is negative.

Secondary contrasts are marked descriptive-only, because the analysis requires three passed
steps. Holm-adjusted p is 1.00 for every contrast except S4 (0.93).

| Contrast | Estimate | 95% CI |
|---|---|---|
| S1 R − A1 on M (value of C's signals) | +0.05 | [−0.24, +0.33] |
| S2 R − B2 on M (C base vs SAC base) | +0.01 | [−0.26, +0.28] |
| S3 R − A2 on D (value of T-D training) | −0.04 | [−0.21, +0.14] |
| S4 A3 − C on D (delay-only residual) | +0.12 | [−0.14, +0.39] |
| S5 R − C on D, RI cells | −0.64 | [−1.84, +0.59] |

## Descriptive findings

The numbers below are from `results/residual_rl/report/report.md`. None of them is a test.

1. **The residual gives a small, uniform gain that is not failure-aware.**
   - R improves on C by roughly 0.2–0.5 points in every group, including nominal.
   - Every ablation does about as well on M: A1 (no C signals) +0.25, A2 (no T-D training) +0.51,
     A3 (delay only) +0.31, B2 (on the SAC base) +0.29.
   - The C signals and the delay-aware training add nothing measurable.
2. **There is real headroom on D that the residual does not capture.**
   - Privileged C (P, true delay) reaches 96.68% on D, against C's 94.22%.
   - R reaches 94.50%, so it recovers about 0.3 of the 2.5 available points.
   - The largest gaps are F-I (P 91.8, C 79.4, R 79.8) and RI (P 95.5, C 86.6, R 86.0). On RI,
     R is slightly *below* C.
3. **The residual cannot replace delay compensation.**
   - On D, B2 (residual on SAC) reaches 81.42%, about the same as B0 (81.20%) and 12.8 points
     below C.
   - X-DH and the F-P, NG, RP and RS families follow the same pattern.
4. **Outcomes churn while the net change stays tiny.**
   - On D and M, R disagrees with C on about 8% of episode-network pairs, more than C's failure
     rate of about 6%.
   - The three R networks disagree with each other on about 12% of episodes. In F-I, R loses 2,847
     of C's successes and adds 2,939 new ones.
   - The trajectories show why. The per-step correction is about 0.02–0.03 in action units, yet
     the same episode often ends in success under one network and failure under another. Many of
     these episodes end near the 5 cm success threshold.
   - The net effect of the residual is a near-zero sum of many outcome flips, not a systematic
     correction.
5. **Safety and coupling.**
   - C's mean alarms per episode are essentially unchanged with the residual (D: 1.80 → 1.79).
   - There were no fallbacks.
   - Training-time nominal checks stayed at 0.95–1.00 for all 21 runs.
   - Mean |δ| was about 0.42–0.66, and saturation was ≤ 3% in every run except `R_a0.05_n1`
     (11%).

## Representative trajectories

`results/residual_rl/report/trajectories.md` lists 33 episodes, with per-step records in
`trajectories.jsonl` (regenerate with the command below). Each category takes the first 3 per group
in seed order, with no outcome-based choice.

- **Recoveries (C fails, most R networks succeed).** For example, M/TM_friction0.5 seed 3722003:
  C ends 9.1 cm from the goal; the three R networks end at 0.5, 2.4 and 0.9 cm. Several N
  recoveries look the same (C at 43 cm, R at 1–3 cm). These are cases where a small push changes
  which contact sequence happens, not cases of detected and corrected failure.
- **Failures (C succeeds, most R networks fail).** For example, M/TM_mass0.5 seed 3722005: C
  succeeds at 1.9 cm; R networks 1 and 2 end at 86 and 53 cm, which is a lost object. In D/I_d4_p8
  seed 3722001, all three networks fail where C succeeds.
- Network disagreement within one episode is common in both categories. The same corrective
  mechanism produces as many failures as recoveries.

## Limitations

- **Inference is conditional on the three trained networks (STAT-2).** It does not generalise to
  a new training seed. The per-network spread (−0.04 to +0.60 points on M) is small next to the
  +2-point threshold.
- **α\* = 0.05 caps the correction at 0.05 per action dimension.**
  - Selection compared all three registered α values on final checkpoints, and none showed a
    positive efficacy-family difference.
  - A larger α or a different reward is a new design, not part of this plan.
- One architecture, one training budget (1M steps) and one environment (FetchPush) were tested.

## Recommendation: NO-GO

- **Held-out evaluation: do not run it.** Pre-registration §7 allows a held-out plan only if
  steps 1–3 pass, and step 3 failed. Seeds 4,000,600–899 remain unused.
- **Residual RL as an efficacy improvement over C: do not pursue it in this form.**
  - The effect is pinned below +0.7 points, against a +2-point practical threshold.
  - The ablations show no contribution from the failure-aware signals.
- **What is positive and reportable:** residual training on a frozen C is safe at these settings.
  - Nominal and delay performance are non-inferior, with small positive estimates.
  - No run failed and no fallbacks occurred.
  - The privileged-P gap on F-I and RI marks where the remaining delay-handling headroom is.

## Reproduce

```bash
caffeinate -i bash scripts/residual_campaign.sh 2>&1 | tee -a results/residual_rl/campaign.log
PYTHONPATH=src uv run python scripts/residual_report.py results/residual_rl
PYTHONPATH=src:scripts uv run python scripts/residual_trajectories.py results/residual_rl --per 3
```

The driver is resumable and skips finished stages. Checkpoints, episode tables and per-run logs
are under `results/residual_rl/` and are not committed: `*.pt` and `*.csv.gz` are ignored, and
the logs are about 100 MB. The committed summaries are the report, trajectories.md, effects,
analysis.json, harmed_rescued.csv, the selection outputs and the campaign log.
