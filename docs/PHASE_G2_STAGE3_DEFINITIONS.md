# G2 Stages 2–3: definitions recorded before calibration

**Status:** written and committed before any calibration seed (3,672,000–099) or evaluation
seed (3,673,000–099) was used. Only synthetic unit tests and debugging seeds (3,670,000–009,
code checks) had been run.
- The approved plan (`docs/PHASE_G2_PROPOSAL.md`, SHA-256 `799fb487…`) is unchanged.
- On 2026-11-16 the user approved proceeding with all three candidates and declined the
  optional extra hypotheses I had floated. The Stage-3 hypotheses are therefore exactly
  §5.4.
- Numbering continues from the Stage-1 record (A1–A6).

**A7. Candidate B's fast-window grid is {4, 5}, not {3, 4, 5}.**
- With the unchanged gate `min_transitions` = 4, a 3-transition window can never be
  reliable, so w_f = 3 would never raise an alarm.
- It would then trivially meet the false-alarm criterion and be chosen as "smallest".
- Excluding it is a mechanical consequence of the gates, not an outcome-based choice.

**A8. A candidate with no admissible setting is not run.** If no setting of a candidate meets
its calibration criterion, the candidate is reported as failing calibration and is not run in
Stage 3 (nor is its sham). Holm then covers the remaining candidates.

**A9. Calibration metrics (detection only; success is never written or read).** All from the
candidate's own closed-loop logs on the calibration cells:
- **wrong reliable-lag rate:** steps whose estimate is reliable with a lag different from the
  true one, over all steps of the persistent cells. Candidate A's criterion compares it with
  `contact_learned` (window 10) from the same calibration run.
- **suspended rate:** steps with compensation off after first engagement while the true delay
  is above 0, over such steps (persistent cells).
- **false-activation rate:** active steps over all steps (undisturbed and 1 cm noise cells).
- **release latency:** for each true end on the two transient calibration cells, steps until
  compensation is first off. An end never followed by an off step counts as infinite. The
  median is used.
- **C tie-breaks** (equal median latency): larger threshold h, then larger σ₀, then the
  Gaussian variant before the scale-free one. Each choice is the more conservative.

**A10. PRIV_true_schedule uses the true delay *and* the true gain.** This is the
perfect-identification ceiling. With no gain loss it equals Stage 1's `true_learned`, and a
test checks the issued commands are identical.

**A11. Candidate C as implemented** (`farc.delaychange.candidates.Sequential`):
- **Residual:** r(k) = Δg_xy − K0·ĝ·0.05·c_xy[τ−k] on the newest transition. ĝ is C's own
  latched gain estimate (1.0 until a reliable one).
- **Noise scale:** σ̂² = max(σ₀², SSE(d)/(2n)), updated at every reliable estimate. The
  scale-free variant uses σ₀² only as a floor.
- **Change point:** the transition after the winning statistic was last zero.
- **After any alarm,** detection pauses until the reset window gives a reliable estimate.
  - Without this, repeated alarms during an onset would reset the window to a single
    transition forever.
  - Synthetic unit tests caught this before any seed was used.
- **With no alarm,** C is identical to `contact_learned`; a test checks this in the
  simulator.

**A12. Candidate B as implemented** (`TwoTimescale`):
- **Alarm:** the fast fit is reliable with a lag different from the latched d > 0, on 2
  consecutive steps.
- **Response:** suspend. Re-engage only on a reliable *slow* estimate d′ > 0 at least 5 steps
  after the alarm.
- **No onset path:** onsets go through the unchanged slow latch, as planned.

**A13. Intermittent cells:** pulses of length p at onsets 6, 6 + 2p, 6 + 4p, … until the
episode ends, with p ∈ {4, 8}. "T + I" pools the 18 T cells and the 4 I cells.

**A14. Drop/replay semantics only.** Stage 1 did not flag pipeline dependence (change 0.0
[−2.1, +2.0]), so under §3.4 Stage 3 runs only under the existing drop/replay semantics.

**A15. Debug-run constants are placeholders.** The debug run on debugging seeds uses fixed
placeholder constants so every code path executes. Its outcomes are not analysed.
