# Phase G4: held-out confirmation of Candidate C, results

## Run record

| item | value |
|---|---|
| Plan | `docs/PHASE_G4_HELDOUT_PROPOSAL.md` (SHA-256 `2382ea38…`), approved as written 2026-11-16 |
| Recorded before use | D1–D7 in `docs/PHASE_G4_AMENDMENTS.md`; the run used commit dd7e691 |
| Seeds | **held-out 4,000,300–4,000,599** × 3 frozen SAC policies = 900 pairs per cell. No other held-out seed was touched. |
| Size | Arm R (random schedules): 9 cells × 5 configurations; Arm F (G2's fixed grid): 32 cells × 4 configurations; **155,700 episodes**, one run (14 minutes) |
| Code checks | debugging seeds 3,690,000–009 only. The G4 runner reproduced the G2 runner bitwise for C, contact-aware and the same-condition sham. |
| Frozen | Candidate C's code and constants (Gaussian, h = 12, σ₀ = 4 mm), SAC `fetch_push_fp32_seed{0,1,2}`, the Phase E estimator, the F1 model `8a960329…` |
| Raw | `$FARC_DATA_DIR/experiments/phase_g4_heldout/` (write-once); tables in `results/phase_g4_heldout/` |

The analysis is the pre-written `derive_g4`, run once. Nothing was changed after the run.

## Verdict

**Candidate C is confirmed on held-out seeds, with episode-specific detection confirmed.** All
safety gates passed.

Numbers are success-rate points; brackets are 95% episode-seed cluster-bootstrap CIs; counts
are rescued / harmed episodes.

## Primary (fixed-sequence gatekeeping, α = 0.025)

| step | claim | contrast and scope | estimate [95% CI] | result |
|---|---|---|---|---|
| 1 | **P1 efficacy** | C − contact-aware, random single pulses and on-off (RT + RI, 3,600 pairs) | **+3.5 [+2.4, +4.6]** (270 / 145) | **confirmed and practically meaningful** (≥ +2.0) |
| 1 | **P2 persistent non-inferiority** (margin −1) | C − contact-aware, all persistent cells (RP + fixed P, 7,200 pairs) | **+0.5 [+0.2, +0.9]** (113 / 76) | **passed**; C is better, not just non-inferior |
| 2 | **P3 episode-specific detection** | C − cross-episode yoked sham (RT + RI) | **+3.2 [+2.2, +4.2]** (258 / 144) | **meaningful** (≥ +1.0) |

**Step 1** (the intersection-union of P1 and P2) passed with p < 0.001, so **Step 2** was tested,
and it passed.

**Across the three frozen SAC policies, P1 was** +3.0, +2.1 and +5.3. All three CIs exclude 0.

## Secondary (Holm across 7)

| | contrast | scope | estimate [95% CI] | result |
|---|---|---|---|---|
| S1 | C − frozen SAC, margin −2 | R: RT + RI | −2.2 [−3.2, −1.3] | **not supported** |
| S2 | C − frozen SAC, margin −2 | F: T + I | −1.6 [−1.9, −1.2] | supported |
| S3 | C − frozen SAC | R + F: persistent | +45.0 [+43.4, +46.6] | supported |
| S4 | C − contact-aware (G2's H1) | F: T + I | +1.4 [+1.0, +1.8] | supported |
| S5 | C − contact-aware | R: changes of value | +4.0 [+2.2, +5.8] | supported |
| S6 | C − random-timing pause sham | R: RT + RI | +2.9 [+1.8, +3.9] | supported |
| S7 | C − contact-aware, margin −2 | F: noise and gain loss | +1.6 [+0.9, +2.3] | supported |

## Safety gates (all passed)

| gate | result |
|---|---|
| No delay, undisturbed: C − frozen SAC | 0.0 [0.0, 0.0]; 0 of 897 harmed |
| No delay, 1 cm noise: C − frozen SAC | 0.0 [−0.3, +0.3]; 1 of 861 harmed |
| C − contact-aware per family, CI low > −2 | RT +1.6, RI +5.3, RS +4.0, RP +1.1 [−0.2, +2.3], F-T +0.8, F-I +4.3, F-P +0.3, NG +1.6 |

## Success rates (%)

| family | frozen SAC | contact-aware | **C** | cross-episode yoked sham | random-timing pause sham | G2 same-condition sham |
|---|---|---|---|---|---|---|
| RT random single pulses | 97.4 | 95.5 | **97.1** | 95.7 | 95.5 | – |
| RI random on-off | 90.2 | 80.7 | **86.0** | 81.1 | 81.9 | – |
| RS random change of value | 30.1 | 90.3 | **94.3** | 89.6 | 87.7 | – |
| RP random-onset persistent | 54.7 | 93.2 | **94.2** | 92.6 | 93.1 | – |
| F-T fixed single pulses | 98.3 | 97.3 | **98.0** | – | – | 97.6 |
| F-I fixed on-off | 85.8 | 74.0 | **78.3** | – | – | 77.8 |
| F-P fixed persistent | 49.7 | 96.2 | **96.6** | – | – | 96.2 |
| NG noise / gain loss | 58.9 | 89.9 | **91.5** | – | – | 90.5 |
| H no delay | 97.7 | 97.7 | 97.7 | 97.7 | 97.7 | – |

## How this lines up with G2 and G3

| claim | development | held-out (G4) |
|---|---|---|
| C − contact-aware, fixed short-lived and on-off (G2 H1) | +1.7 (G2 Stage 3) | **+1.4** (S4); below the +2 practical threshold, as in G2 |
| C − contact-aware, random short-lived and on-off | +3.0 (G3) | **+3.5** (P1); above the +2 practical threshold |
| Persistent non-inferiority | +0.1 (G2, fixed); +1.5 (G3, random onset) | **+0.5** pooled (P2) |
| Specificity, cross-episode yoked sham (random schedules) | +2.4 (G3 M4) | **+3.2** (P3) |
| Specificity, G2 same-condition sham (fixed grid) | +0.5 [−0.0, +1.1] (G2 S2, not shown) | +0.5 [+0.05, +0.9] (descriptive). That sham inherits correct timing, so it remains a weak control. |
| Timing, random-timing pause sham | +1.6 (G2 exploratory check) | **+2.9** (S6) |
| C − frozen SAC, margin −2, fixed grid | −1.2, shown (G2 S4) | −1.6, **shown** (S2) |
| C − frozen SAC, margin −2, random schedules | −2.0, not shown (G3) | −2.2, **not shown** (S1) |

All held-out results point the same way as their development counterparts. The two designs
again give different sizes:

- **On the fixed grid,** C's gain over contact-aware is about 1.4–1.7 points. That is real, but
  below the practical threshold.
- **On random schedules,** it is about 3.0–3.5 points.

## Conclusions

1. **Candidate C is confirmed** as an improvement over the contact-aware compensator:
   - on random short-lived and on-off delays, +3.5 (practically meaningful);
   - under persistent delays it is better rather than just non-inferior (+0.5);
   - on changes of value, +4.0;
   - under noise and gain loss, +1.6;
   - with no measurable harm when there is no delay.
2. **C's benefit depends on detecting each episode's own transitions.** The same pause-and-reset
   mechanism driven by another episode's alarms gives +0.2 to +0.4 over contact-aware. C gives
   +3.5. The same pauses at random times are also clearly worse (+2.9 for C).
3. **C does not match frozen SAC everywhere.**
   - When delays are short and frequent, frozen SAC still does better: by 2.2 points on random
     single pulses and on-off delays, and by 4.2 on random on-off delays alone (90.2 vs 86.0).
     Non-inferiority within 2 points is shown on G2's fixed grid but not on random schedules.
   - When delays persist or change value, C is far better than frozen SAC (+45 on persistent
     delays).

## What G4 does not confirm

- **The G3 mechanism decomposition** (pause timing, reset, and their interaction). G4 did not
  test it.
  - G3's replay-based interaction estimate (+2.4) is sensitive to alarm divergence: it was +0.1
    on the pairs where all replays agreed with C.
  - Its supporting oracle check used privileged transition times.
  - Confirming a mechanism would need variants that each run their own detector.
- **Anything outside this setting:** simulation, a pure transport delay with drop/replay
  semantics, frozen SAC and F1 components, and the generator's schedule distributions.

## Held-out seed accounting

- 4,000,300–4,000,599 are now used by G4.
- 4,000,000–4,000,299 were used earlier by the contact-model confirmation.
- Nothing at or above 4,000,600 has been touched. No held-out seed remains approved for any
  further run.
