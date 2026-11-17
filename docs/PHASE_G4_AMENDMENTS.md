# Phase G4: implementation decisions recorded before the confirmation

**Status:** written and committed before any held-out seed (4,000,300–599) was used. Only unit
tests, a simulator test on test seeds and the debugging run (3,690,000–009, code checks) had
been run. The approved plan (`docs/PHASE_G4_HELDOUT_PROPOSAL.md`, SHA-256 `2382ea38…`) is
unchanged and pinned in `configs/delaychange/g4_heldout.json`.

**D1. Controllers are reused, not re-implemented.**
- `C` is `Sequential` with the pinned constants.
- `contact_learned` and `base_only` use the latch; `base_only` never compensates.
- `sham_C_cross_yoked` is G3's `EventReplay` with both switches on, driven by C's events
  from the next episode seed of the same cell.
- `sham_C_random_pause` is G3's rotation of C's paused steps from the same episode.
- `sham_C_same_condition` is G2's yoked mask from the next episode seed of the same fixed cell.
- Every configuration runs through G3's `run_episode`, with no shadow detector.

**D2. Bitwise equivalence with the G2 runner (debugging run, code check):**
- On all 1,230 check episodes over both arms, the G4 runner reproduced the G2 Stage-3 runner's
  issued commands exactly for:
  - `C` (`cand_C_sequential`);
  - `contact_learned`;
  - the same-condition sham (`sham_C_yoked` with the same mask).
- C paused in 516 of those episodes, so the sham path was exercised.
- G3's equivalence of `EventReplay` with C was established in G3 and is unchanged.
- The analysis ran end to end on the debugging data. Only table shapes were printed; no outcome
  was read.

**D3. Family labels.**
- Arm R keeps G3's families: RT, RI, RS, RP and H.
- Arm F relabels G2's families as F-P, F-T and F-I; NG keeps its name.
- The "persistent" scope of P2 and S3 is RP ∪ F-P. "RT + RI" is Arm R only; "F: T + I" is
  F-T ∪ F-I.

**D4. Cell indices.**
- A random cell's index is its position in `g3_cells()`. This keeps G3's generator and
  rotation seeds.
- A fixed cell's index is its position in G2's `evaluation_cells()`. It is used only as the
  rotation seed, which Arm F does not use.

**D5. Bootstrap pairing.** Pairs are (policy seed, condition, episode seed). Condition names are
unique across the two arms. The bootstrap clusters on episode seed across both arms, so the
pooled persistent scope (P2, S3) resamples each seed's R and F episodes together.

**D6. The P2 label** is "passed" or "failed", against its own one-sided p at α = 0.025. Step 1
uses max(p_P1, p_P2), the intersection-union test.

**D7. Secondary results** are "supported" or "not supported" after Holm across S1–S7.
