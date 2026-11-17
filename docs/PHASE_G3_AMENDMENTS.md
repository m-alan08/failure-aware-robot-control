# Phase G3: implementation decisions recorded before the evaluation

**Status:** written and committed before any evaluation seed (3,681,000–999) was used. Only
unit tests, a simulator test on test seeds and the debugging run (3,680,000–009, code checks)
had been run. The approved proposal (`docs/PHASE_G3_PROPOSAL.md`, SHA-256 `ec2c7623…`) is
unchanged and pinned in `configs/delaychange/g3_mechanism.json`.

**B1. Perturbation seed 20261700.** It only affects the observation noise in `H_noise0.01`.
G2 used 20261604; a new phase gets a new value.

**B2. Shadow detectors run in every configuration except `base_only` and
`PRIV_true_schedule`.** The proposal required them in the replayed runs. Running them more
widely costs nothing and gives a reference: `C_full`'s shadow must equal C exactly, and
`contact_learned`'s shows how often C's later alarms would differ on the contact-aware
trajectory. Shadow time is excluded from the overhead measure.

**B3. Episodes with a delay from step 0** (RS's first delay; RP when the onset is 0) have no
transition at step 0. The oracle raises no event there, consistent with C, which has no change
to detect.

**B4. Measure definitions** (section 5, made exact):
- **Alarm recall:** an alarm at offsets 0–5 after a transition, before the next transition.
- **False alarms:** alarms on steps more than 5 steps after the latest transition, per 1,000
  such steps.
- **Stale compensation:** compensating with the lag equal to the true delay of the previous
  segment.
- **Change latency:** first step with compensation on at the new lag. Release latency (ends):
  first step with compensation off. Onset latency: first step on at the true lag.
- **`paused_steps`:** for masked configurations, the number of masked steps, whether or not
  compensation would have been on.

**B5. Sensitivity set.** "Full agreement" for a pair needs, for each of `C_pause`, `C_reset`
and `C_pause_own`, that:
- its shadow's first alarm equals C's;
- every C alarm is matched by a shadow alarm within ±1 step;
- no shadow alarm is unmatched.

The sensitivity analysis keeps the pairs where all three agree.

**B6. Secondary labels.** The six secondary contrasts carry the same four labels as the primary
ones. For the frozen-SAC non-inferiority contrast, the result is its p-value against the −2
margin; the label is descriptive only.

**B7. What the debugging run showed (code checks only; no outcome was read):**
- `EventReplay` with C's own events reproduced closed-loop C's issued commands bitwise in all
  270 check episodes, which contained 440 alarms.
- An event-free `EventReplay` reproduced `contact_learned` bitwise in all of them.
- The shadow's first alarm equalled C's in 100% of the `C_pause`, `C_reset` and `C_pause_own`
  debug runs.

The divergence summary printed by the debug run reports alarm agreement, not outcomes. On the
270 debug pairs, 67–81% of C's later alarms were matched by the replayed variants' shadows. This
is close to the proposal's 30% caution level for risk 1, so the divergence table will be read
alongside M1–M3 as planned. The design is unchanged.
