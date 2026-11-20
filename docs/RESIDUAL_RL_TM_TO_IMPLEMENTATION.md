# T-M and T-O implementation (residual RL pre-registration §3, checklist item 2)

**Status:** software only. The pre-registration (`docs/RESIDUAL_RL_PREREGISTRATION.md`) is still
a **draft and not approved**; it was not edited. No training, no residual or policy episodes, no R
seed block and no held-out seed were used.

`src/farc/residual/disturbances.py` implements the two families exactly as §3 defines them:

| family | implementation | no-op setting |
|---|---|---|
| T-M | `ObjectDynamics` (mass × U[0.5, 2.0], sliding friction × U[0.5, 1.5]) applied by `ObjectDynamicsPatch` to `object0`'s body mass, inertia (scaled with mass) and geom sliding friction; restored on exit | scales 1.0 write nothing |
| T-O | `CorrelatedObservationNoise`: unit AR(1) state per entry (ρ = 0.8, stationary start) through the existing `positions` mask, σ ~ U[0, 1 cm]; relative position recomputed | σ = 0 returns an unchanged copy |

One design choice not stated in §3: inertia is scaled together with mass (same shape, denser
block). Flag it at approval if a mass-only change was intended.

## Unit-level evidence (seeds 0–2 only; NOT the pre-registered checks)

`tests/test_residual_disturbances.py` (6 tests):
- AR(1) lag-1 autocorrelation within 0.8 ± 0.05 and stationary SD ≈ σ on synthetic draws (the T-O
  effect check, which §3 defines on synthetic draws);
- σ = 0 identity; common random numbers across σ;
- T-M nominal patch bitwise identical to an unpatched FetchPush rollout, model values restored;
- T-M changes the object trajectory under one fixed, replayed command sequence (with a guard that
  the sequence actually moves the block).

## Still required before training (unchanged)

1. Approve the pre-registration and pin its SHA-256.
2. Run the §3 nominal-identity check on R debugging seeds 3,720,000–009 at the approved commit,
   plus the §8.4 real-simulator zero-residual re-check.
3. Wire T-M/T-O into the `ResidualEnv` session factory and write the training runner, selection
   script and pre-written analysis, with tests, before any training seed is used.
