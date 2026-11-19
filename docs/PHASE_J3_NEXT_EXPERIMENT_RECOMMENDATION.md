# Phase J-3: next-experiment recommendation

**Status:** recommendation only. **Nothing is approved, implemented or run.** Any option below
would need its own pre-registered protocol and seed allocation, approved separately. The
evidence comes from `docs/PHASE_J3_LEARNING_DYNAMICS_AUDIT.md`.

## Options compared

| | isolates a mechanism | scientific relevance | complexity | compute (Phase I/J-2 scale) | risk of inconclusive | paper contribution |
|---|---|---|---|---|---|---|
| **A** actor-only noise with a clean (privileged) critic | **H1:** one change, critic inputs | moderate; asymmetric actor-critic is a standard, citable design | **medium-high** (see below) | ≈ 2.5 h wall, about 15 CPU-h, + 30 min eval | moderate; H1 is only weakly supported, so a null is likely | moderate: "even a privileged critic does not fix it" |
| **B** noise curriculum | weak; it changes both the schedule and the amount of exposure | low | low | same | high; many free parameters | low |
| **C** mixed clean and noisy episodes (for example 50% at σ = 0) | **H2 (C arm) and H3/H4:** keeps the clean delay signal and nominal signal in half the data | moderate | **low:** one wrapper parameter | same | moderate | moderate |
| **D** recurrent delay-aware policy | none; new architecture, many changes | high in general, low for diagnosis | high | several times larger | high | a new baseline, not an explanation |
| **E** stop retraining; real-robot validation and paper | – | **high for the project's core claim** | – | 0 simulation | – | **highest** |

## Is actor-only noise well-defined here?

**Not as a one-line change.** The replay stores one observation vector per step, and it is
already noisy. A correct asymmetric version would need four changes:

1. **Store both** clean and noisy observation *and feature* vectors for every step.
   - The I-A displacements and I-C estimator state depend on the whole noisy history, so they
     cannot be regenerated at sampling time from clean rows.
   - Collection must therefore run two featurisers in parallel: noisy for acting, clean for the
     critic.
2. **Critic:** Q(clean s, a) and target Q(clean s′, a′).
3. **Actor:** the loss is Q(clean s, π(noisy o)) and the target action is a′ = π(noisy o′). This
   is consistent, because the actor is evaluated at the true state paired with its own noisy
   observation of that state.
4. The **normaliser** is frozen, so it is unaffected.

**Remaining subtlety:** the critic then values states under a policy that acts on noisy inputs.
That is the standard asymmetric actor-critic setting, so it is valid but no longer "identical to
Phase I except one thing": the critic input distribution also changes, becoming clean where
J-2's was noisy.

**Status:** with these changes, A is well-defined. Without them, a naive "add noise only in the
actor forward pass" would pair stored noisy rows with a supposedly clean critic, which is
**not** valid.

## Recommendation

**Primary: E.** Stop retraining RL baselines and move to physical-robot validation and the
paper. The reasons:

- **Six RL baselines** (R1, R2, I-A, I-C, J2-A, J2-C) have now been built under pre-registered
  rules. None is both noise-competent and competitive with C under persistent delay, and the
  audit finds no implementation bug.
- **Further baseline tuning has a real cost.** Each additional variant spends development seeds,
  adds a forking path, and invites a "baseline tuned until it lost / won" critique. The project's
  claim should be scoped to these baselines in any case.
- **The audit's open questions concern why one baseline variant failed,** not the method. They
  are secondary for the paper.
- **The physical system tests** what simulation cannot: real latency structure, real sensor
  noise (often temporally correlated, which §3.1 shows would *help* the differencing features),
  and contact. That is where C's value is decided.

**If one more simulation experiment is wanted: C (mixed clean/noisy), not A.**

- **It targets the best-supported mechanism:** H2, demonstrated for J2-C, plus the nominal
  non-recovery (H3/H4).
- **It is the smallest valid change:** one wrapper probability. A needs the replay and featuriser
  changes above.
- **It has a clear pre-registrable prediction:** C-variant J2-C recovers most of the persistent-
  delay deficit vs I-C (D2-style non-inferiority, margin −2) and passes G-nominal.
- **Compute:**
  - training: 6 runs × about 2.3 h in parallel ≈ **2.5 h wall-clock, about 15 CPU-hours;**
  - Stage 2: about 30 min, about 290k episodes;
  - **total about 3 h.**
  - J2-C only would halve training, and is recommended if run.

A would follow only if C left the nominal loss unexplained.

## Seed implications

- **A new development evaluation block is required** (for example 3,704,300–599), plus new
  generator and noise seeds.
- **Reusing 3,704,000–299** would make the new arm's comparison with J-2 and Phase I conditional
  on a block already used to diagnose J-2's failure, and that diagnosis motivated the design.
  Results on that block would be exploratory, not a test.
- **Held-out 4,000,600–899 stays reserved.** No baseline currently qualifies for Stage 3.
