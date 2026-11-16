# Phase E2 results: pre-registered evaluation of online delay compensation

The plan was `docs/PHASE_E_DELAY_PREREGISTRATION.md`: approved 2026-11-16, with amendments
A1–A3 recorded before any E2 seed was used, and its SHA-256 `a8883569…` pinned in the config.

**Run provenance:**
- Code commit `cf2bd00`, with a clean working tree, which the runner enforces.
- Evaluation seeds 3,641,000–099; 3 frozen policies.
- 19,800 episodes = 11 conditions × 6 configurations × 300 paired episodes.
- Tables are in `results/phase_e2_delay/`. Raw files are under
  `$FARC_DATA_DIR/experiments/phase_e2_delay/` and are write-once.
- Debugging seeds 3,640,000–009 were used only to check that the code runs. Their outcomes
  were not analysed; I only checked table shapes.

## Pre-registered decision: **supported** (all five criteria pass)

| criterion | estimate | 95% CI | passed |
|---|---|---|---|
| H1: estimated delay − base, pooled C1–C9 (CI low > +5) | **+39.2** | [+36.4, +41.9] | yes |
| No start-up cue: C5 (delay 3 from step 15) (CI low > 0) | **+17.0** | [+12.7, +21.7] | yes |
| Specificity: estimated delay − shuffled sham, pooled C1–C9 (CI low > 0) | +3.6 | [+2.0, +5.2] | yes (weak; see below) |
| Harm, C0 undisturbed (CI low > −2, ≤ 2% harmed) | 0.0 | [0, 0] | yes, 0 harmed |
| Harm, C10 position noise 0.01 m | 0.0 | [0, 0] | yes, 0 harmed |

## By condition (secondary; Δ success vs base in points [95% CI], rescued/harmed)

| condition | base | **estimated delay** | combined (exploratory) | ORACLE delay | sham fixed d = 2 | sham shuffled |
|---|---|---|---|---|---|---|
| C0 undisturbed | 100% | 0 (0/0) | −0.7 (0/2) | 0 | −16.0 (0/48) | 0 |
| C1 delay 1 | 82.0% | **+16.0** [+11.0, +21.0] (53/5) | +15.7 | +15.7 | −8.0 | +15.3 |
| C2 delay 3 | 14.0% | **+68.0** [+61.7, +74.0] (208/4) | +67.0 | +72.3 | +51.7 | +67.3 |
| C3 delay 2 | 30.3% | **+59.0** [+52.0, +65.7] (187/10) | +59.0 | +58.3 | +54.3 | +54.7 |
| C4 delay 4 | 10.3% | **+63.3** [+55.7, +70.7] (200/10) | +63.0 | +67.7 | +20.7 | +70.0 |
| C5 delay 3 from step 15 | 76.3% | **+17.0** [+12.7, +21.7] (55/4) | +16.0 | +17.3 | −3.3 | +13.3 |
| C6 delay 3, steps 15–29 | 96.7% | −0.7 [−2.7, +1.3] (5/7) | −1.0 | +1.3 | −17.3 | −3.0 |
| C7 delay 3 + noise 0.005 m | 12.7% | **+64.7** [+58.7, +70.3] (199/5) | +64.7 | +69.3 | +47.7 | +63.7 |
| C8 delay 3 + scale 0.5 | 54.0% | **+20.7** [+13.3, +28.0] (89/27) | +32.0 | +22.0 | +7.0 | −8.0 |
| C9 delay 1 → 3 at step 20 | 43.7% | **+44.3** [+38.3, +50.3] (144/11) | +44.0 | +46.3 | +18.3 | +46.7 |
| C10 noise 0.01 m | 94.7% | 0 (0/0) | 0 | 0 | −19.0 (10/67) | +0.3 |

- **Multiple-comparison correction:** after Holm adjustment over C1–C9, every delay condition
  except C6 has p < 0.001. C6 has p = 0.63.
- **Share of the oracle's improvement:** 93–102% in every condition except C6.
- **By policy seed (pooled C1–C9):** +39.4, +38.6 and +39.4, against the oracle's +42.2,
  +39.7 and +41.6.

## Identification (base-policy episodes)

- **Delay accuracy when the estimator reports confidence:**
  - ≥ 99.9% for constant delays, including under 0.5 cm noise and 50% gain loss;
  - 94.6% (C5), 87.8% (C6) and 95.1% (C9) where the delay changes.
- **Abstention:** the estimator reports confidence on 80–86% of steps under constant delay,
  74% with noise, 57% undisturbed (low excitation late in episodes) and 22% at 1 cm noise.
- **Latency:** after a delay starts, ends or changes, the applied delay takes a median of 8
  steps (90th percentile 9) to match the new value. That is the 10-step window filling with
  the new regime.
- **Stale estimates:** 23% of C6 episodes never switch back to "no delay" before the episode
  ends, because the window has too little excitation late in the episode.
- **Contact:** base successes that compensation harmed spent more time with the gripper within
  5 cm of the object (35% of steps vs 29% for those it kept). This fits the predictor's
  static-object assumption breaking under contact.

## Interpretation

- **The pilot's start-up cue does not explain the effect.** With the delay starting mid-episode
  (C5), where the replayed commands are real ones, compensation still rescues 55 and harms 4,
  matching the oracle (+17.0 vs +17.3).
- **The effect generalises** to delay lengths the pilot never saw (2 and 4 steps), to delay
  under position noise, to delay combined with gain loss, and to a delay that changes
  mid-episode. Each time it is within a few points of the privileged oracle.
- **The right delay value matters.** The fixed d = 2 sham is good only when 2 is close to the
  truth (C3: +54, C2: +52). It fails when it is not (C1 −8, C4 +21 vs +63), and it harms
  undisturbed and noise-only episodes (−16, −19). The deployable controller never acts there
  (0.1–0.2% of steps).

## Limits and honest caveats

- **The pre-registered specificity test was weak, which is a flaw in my design.**
  - The shuffled sham borrows a delay estimate from another episode *in the same condition*.
    Under a constant delay, that estimate is usually the right value. So the sham nearly
    matches the real controller in C1–C4 and C7, and the +3.6 pooled margin comes mainly from
    C5, C6 and C8, where episode-specific timing matters.
  - It shows that timing specific to the episode adds a little. It does not show that a
    correct delay value matters; the fixed-delay sham and the harm conditions show that.
- **C8 is unexplained.** The shuffled sham is −8.0 there, while the real controller gains
  +20.7 with the same mismatch rate. I have not worked out why.
- **Transient delays are not helped (C6).** About 8 steps to detect onset, about 8 to detect
  the end, and stale estimates afterwards together cancel the benefit of a 15-step delay.
  Short disturbances need a faster detector, which this pilot-frozen estimator is not.
- **Gain compensation (combined configuration) is exploratory.** It added +11 points in C8
  (+32.0 vs +20.7), but cost 2 harmed episodes undisturbed. It was not a pre-registered
  comparison.
- **Scope:** all of this is in simulation, with a pure transport delay of *issued* commands
  and a nearly linear mocap plant. Delays on real hardware also include plant dynamics that
  this probe found negligible here.
- **Held-out seeds were not used.** Seeds 4,000,000+ remain reserved.

## Recommendation

- **Delay:** the online delay estimate plus predictor is a supported, deployable component.
  Next steps:
  1. A final, separately approved held-out confirmation of the pre-registered primary on
     seeds 4,000,000+ (only with your approval).
  2. If transient delays matter, a faster change detector (a shorter window, or a CUSUM-style
     test on the residuals) as a *new* pre-registered variant. The current estimator should
     not be tuned on these results.
  3. Only after that, consider learning (for example a residual policy conditioned on the
     identified delay), using this controller as the baseline to beat.
- **Gain:** the C8 gain-compensation gain (+11) is exploratory. It would need its own
  pre-registered test.
