# Phase G3: what makes Candidate C work? Results

## Run record

| item | value |
|---|---|
| Plan | `docs/PHASE_G3_PROPOSAL.md` (SHA-256 `ec2c7623…`), approved as written 2026-11-16 |
| Recorded before use | B1–B7 in `docs/PHASE_G3_AMENDMENTS.md` (commit 8a825c9; the run used that commit) |
| Seeds | evaluation **3,681,000–999** × 3 frozen SAC policies = 3,000 pairs per cell; random schedules per episode seed |
| Size | 9 cells × 12 configurations × 3,000 = **324,000 episodes**, one run (48 minutes) |
| Code checks | debugging seeds 3,680,000–009 only: replaying C's own alarms reproduced C bitwise in all 270 check episodes |
| Frozen | SAC `fetch_push_fp32_seed{0,1,2}`, the Phase E estimator, the F1 model `8a960329…`, Candidate C's code and constants |
| Raw | `$FARC_DATA_DIR/experiments/phase_g3_mechanism/` (write-once); tables in `results/phase_g3_mechanism/` |
| Held-out | **not used.** Nothing at or above 4,000,000 was touched. |

The analysis is the pre-written `derive_g3`, run once. The integrity check passed: in every
`C_pause`, `C_reset` and `C_pause_own` run, the shadow detector's first alarm equalled C's.

All numbers are success-rate points against `contact_learned` unless stated. Brackets are 95%
cluster-bootstrap CIs; counts are rescued / harmed episodes.

## Gatekeeper: C's gain survives unpredictable schedules

**G: C − contact-aware on random single pulses and on-off delays (RT + RI): +3.0 [+2.4, +3.7]
(862 / 501). Passed.** The mechanism tests are therefore confirmatory.

The gain is larger than G2's +1.7 on the fixed grid. Every random-schedule episode has its own
transition times, so C's gain cannot come from a predictable schedule.

## Primary mechanism hypotheses (RT + RI, 12,000 pairs, Holm across M1–M4)

| | question | contrast | estimate [95% CI] | Holm p | label |
|---|---|---|---|---|---|
| **M1** | does pausing at the right time help? | C pause-only − random-timing pause | **+1.3** [+0.9, +1.8] (516 / 357) | < 0.001 | **meaningful** |
| **M2** | does resetting stale history help on its own? | C reset-only − contact-aware | **−1.3** [−2.0, −0.7] (744 / 905) | 1.0 | inconclusive by the pre-registered rule; *it harms* |
| **M3** | do pause and reset interact? | (C − reset-only) − (pause-only − contact) | **+2.4** [+1.7, +3.1] | < 0.001 | **meaningful** |
| **M4** | does C need this episode's transitions? | C − full sham yoked from another episode | **+2.4** [+1.8, +3.0] (784 / 497) | < 0.001 | **meaningful** |

On M2, the label rule assigns "meaningful" or "small" only to a significant *positive* effect,
and "negligible" only to a CI inside ±1. A CI wholly below zero is therefore labelled
"inconclusive". Descriptively, reset without pause makes things worse on these cells.

### How C's +3.0 decomposes (RT + RI)

| | vs contact-aware |
|---|---|
| pause only, at C's times | +2.0 [+1.5, +2.5] |
| reset only, at C's times | −1.3 [−2.0, −0.7] |
| interaction | +2.4 [+1.7, +3.1] |
| **full C** | **+3.0** |

By family the picture differs sharply:

| family | C | pause-only | reset-only | pause, own release | yoked full sham | random pause |
|---|---|---|---|---|---|---|
| RT single pulses | +1.6 | +0.5 | **+1.0** | +0.8 | +0.3 | +0.1 |
| RI on-off | +4.4 | **+3.4** | **−3.7** | +3.3 | +0.9 | +1.2 |
| RS change of value | +2.3 | −0.2 | **+2.4** | −0.1 | −0.9 | −1.4 |
| RP persistent | +1.5 | 0.0 | **+1.6** | 0.0 | −0.2 | −0.1 |

## What the step-level measures show (descriptive)

Incorrect compensation is steps compensating with a wrong lag, per 100 steps. Stale
compensation is the share of those using the previous segment's delay.

| on-off delays (RI) | contact-aware | C | pause-only | reset-only | oracle (full) |
|---|---|---|---|---|---|
| incorrect compensation per 100 steps | 18.8 | **11.7** | 13.3 | **23.0** | 0.1 |
| compensation on during delay-free steps | 32% | 20% | 22% | **40%** | 0.1% |
| compensation on during delayed steps | 28% | 33% | 24% | **41%** | 30% |
| onsets never correctly compensated | 13,263 | 8,760 | 14,007 | 7,205 | 8,693 (of 20,052) |
| median release latency after an end | 0 | 1 | 0 | **4** | 0 |

Read together:

- **Reset makes identification faster.** With a short post-change window, reset-only and C
  identify far more onsets than contact-aware: 7,205 and 8,760 left unidentified, against
  13,263. Reset is also what drives C's gain on changes of value (RS) and on persistent onsets
  (RP).
- **Reset alone leaves stale compensation running after a delay ends.**
  - Without the pause, compensation continues with the old lag until the short reset window is
    reliable again (median 4 steps).
  - In on-off schedules this happens at every end, so reset-only compensates on 40% of
    delay-free steps.
- **The pause is what cuts that stale compensation.** That is the interaction: reset brings
  compensation in sooner, and the pause takes it out at the right moment.
- **On changes between nonzero delays** the pause does not help (−0.2), and reset alone gives
  all of C's gain.

## Episode-specific detection (M4)

- **C's events help only in the episode they came from.** C beats the same pause-and-reset
  mechanism driven by another episode's events by +2.4. That sham gains only +0.6 [+0.1,
  +1.2] over contact-aware.
- **Its events are misaligned:** false alarms per 1,000 quiet steps were 60 on on-off delays
  against 6 for C, and the median alarm offset from the nearest earlier transition was 3–8
  steps against 1–2.

With random schedules, **C's gain depends on detecting this episode's transitions.** G2's
same-condition sham could not show this, because it inherited the right timing.

## Pause timing (M1)

C's pauses at C's times beat the same pauses rotated to random times by +1.3. This replicates
the G2 sham check (+1.6) under random schedules.

- The random-timing pause itself gains +0.6 [+0.3, +1.0] over contact-aware, all of it on
  on-off delays. There, a random pause often lands on an off phase by chance.
- `C_pause_own` pauses at C's alarms but releases as soon as the un-reset estimator is reliable.
  It performs like `C_pause` (+2.0 vs +2.0). So M1 is not inflated by `C_pause` inheriting C's
  reset-informed pause lengths, which was risk 2 in the proposal.

## Alarm divergence and the sensitivity analysis

| replayed run | first alarm = C's | C's later alarms matched within ±1 | episodes in full agreement |
|---|---|---|---|
| `C_pause` | 100% | 66% | 77% |
| `C_pause_own` | 100% | 67% | 77% |
| `C_reset` | 100% | 80% | 85% |
| `contact_learned` (reference) | 100% | 67% | 77% |

- **About a third of C's later alarms would not occur on a pause-only trajectory.** For
  `C_pause` that is just above the proposal's 30% caution level. M1–M3 should therefore be
  read partly as "C's events applied to another trajectory".
- **The divergence is not special to the replays:** C's own detector on the plain contact-aware
  trajectory matches C's later alarms at the same rate (67%).

**Sensitivity** (pre-registered, descriptive and selection-biased): the 5,427 RT + RI pairs in
which all three replays fully agree.

| | full data | agreeing pairs only |
|---|---|---|
| M1 | +1.3 | +0.4 [+0.0, +0.9] |
| M2 | −1.3 | **+0.7 [+0.2, +1.2]** |
| M3 | +2.4 | **+0.1 [−0.5, +0.6]** |
| M4 | +2.4 | +1.3 [+0.8, +1.8] |

**This subset is not representative.** A post-hoc description, not part of the plan:
- it holds 77% of the single-pulse pairs but only 14% of the on-off pairs;
- its episodes have 1.2 alarms on average, against 3.9 outside it.

The interaction and reset's harm are concentrated in on-off delays, which the subset mostly
excludes. So the sensitivity result **cannot distinguish** two readings: the interaction being
an artefact of replaying C's alarms on diverged trajectories, or the interaction living in the
episodes that diverge.

**The oracle 2 × 2 settles most of that,** because its events are the true transitions in every
run, with no replay and no divergence:

| RT + RI, oracle events | vs contact-aware |
|---|---|
| pause only | +3.6 [+3.0, +4.1] |
| reset only | +0.5 [−0.2, +1.1] |
| interaction | **+1.9 [+1.2, +2.6]** |
| full | +5.9 [+5.4, +6.5] |

The same pattern appears with perfect events: the pause dominates, reset alone does nothing on
RT + RI, and the two interact positively. On changes of value (RS) the oracle again shows reset
as the mechanism (+2.8) and the pause as useless (−0.8).

**Conclusion on M3:** the pause-reset interaction is real. Its size with C's own events
(+2.4) is uncertain, because a third of the later replayed alarms are not causal for the
variant.

## Headroom (PRIVILEGED, diagnosis only)

| RT + RI | vs contact-aware | vs C |
|---|---|---|
| oracle: pause + reset at the true transition times | +5.9 | **+2.9 [+2.4, +3.4]** |
| true delay and gain | +8.1 | +5.1 |

- **Perfect transition timing would double C's gain.** C's alarms come a median 1 step after a
  transition, but C catches only 52–62% of transitions within 5 steps, and its false alarms
  cost more.
- **On changes of value C already matches the oracle** (−0.1 [−1.1, +0.8]).
- **Even with perfect timing, identification still takes about 4 steps after an onset.** The
  true-delay ceiling is 2.2 points above the oracle for that reason.

## Safety and non-inferiority

- **All gates passed.**
- **Persistent delays with random onset:** C +1.5 [+0.8, +2.2] and reset-only +1.6 over
  contact-aware. Both are improvements, not just non-inferior.
- **No delay:** at most 2 harmed episodes of about 2,900 base successes, every CI above −2.
- **Against frozen SAC on RT + RI:** C −2.0 [−2.5, −1.5]. Non-inferiority within 2 points is
  **not shown** (p = 0.49), unlike G2's S4.
  - On random on-off delays, frozen SAC beats contact-aware by +8.2 and C by +3.8.
  - Only the privileged configurations beat SAC there.

## Secondary tests (Holm across 6)

| test | estimate [95% CI] | result |
|---|---|---|
| M1 on RS | +1.2 [−0.1, +2.5] | inconclusive |
| M2 on RS | **+2.4 [+1.4, +3.4]** | meaningful |
| M4 on RS | **+3.2 [+2.0, +4.4]** | meaningful |
| pause main effect, RT + RI | **+2.0 [+1.5, +2.5]** | meaningful |
| C − frozen SAC, RT + RI (margin −2) | −2.0 [−2.5, −1.5] | not non-inferior |
| yoked full sham − contact-aware, RT + RI | +0.6 [+0.1, +1.2] | inconclusive (Holm p = 0.050) |

## Conclusions

1. **C's gain does not depend on a predictable schedule.** With random per-episode schedules it
   is +3.0 over contact-aware.
2. **It depends on detecting this episode's transitions (M4, +2.4).** The same mechanism driven
   by another episode's events gives +0.6.
3. **Two mechanisms, each doing a different job:**
   - *Reset* speeds up identification after a change. It alone carries C's gain on changes of
     value and persistent onsets.
   - *Pause* removes stale compensation after a delay ends. It alone carries most of the gain on
     on-off delays.
   - *Reset without pause is harmful* when delays switch off often, because it keeps stale
     compensation running.
   - So the two interact (M3). The perfect-timing oracle confirms the pattern; the size with C's
     own events is uncertain because of alarm divergence.
4. **Pausing at the right time matters (M1, +1.3),** and not only through pause length.
5. **Detection is the remaining bottleneck.** The same mechanism at the true transition times
   would add another +2.9. On-off delays remain the weak case: frozen SAC still beats C there.

## Limits

- **Development seeds only.** Any held-out claim needs a separate, approved plan
  (4,000,300+ remain reserved).
- **Later replayed alarms are not causal** for the variants (about a third of them). The oracle
  2 × 2 is the clean check of the interaction.
- **The generator's ranges are my choices,** fixed before the run.
- **Simulation only;** drop/replay transport delay; frozen SAC and F1 components.
