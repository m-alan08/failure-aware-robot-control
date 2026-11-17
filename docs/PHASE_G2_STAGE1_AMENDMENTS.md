# G2 Stage 1: implementation decisions recorded before the run

**Status:** written and committed after the debug run (code checks only, no outcomes looked
at) and **before** any mechanism seed (3,671,000–049) was used. The approved plan
(`docs/PHASE_G2_PROPOSAL.md`, SHA-256 `799fb487…`) is not edited.

**A1. "Drain" at the end of a delay is implemented as *merge*.**
- **The problem:** the plan describes drain as "the queue empties one command per step, so
  nothing is dropped". An actuator that executes one command per step while new commands keep
  arriving cannot shorten its queue that way: the delay would never end.
- **The implementation:** when the delay shrinks, the queued commands that `replay_drop`
  would drop are summed into that step's command, then clipped by the action bounds. No
  commanded displacement is discarded, except by clipping.
- **"Hold" at onset** is as planned: the last executed command is held while the queue
  fills.
- Code: `farc.delaychange.schedule.ScheduledDelaySession` with `semantics="drain_hold"`.

**A2. What "true schedule" means in M2.**
- **Phase E and learned predictors:** the PRIVILEGED row uses the **true current delay**
  d(t), which is what a perfect identifier of the *current* delay would provide. The queued
  commands c[t−d(t)..t−1] are then exactly those `replay_drop` executes next.
- **Simulator predictor:** it is run with d(t) (`true_sim`), with the latched estimate
  (`latched_sim`, G1's oracle), and with the **future-schedule horizon** (`schedule_sim`:
  the steps until the command issued now executes; M6).

**A3. Point-of-no-return branches (§3.3).**
- **Targets:** every transient (`replay_drop`) episode in which frozen SAC succeeds and
  `deployed_learned` fails.
- **Branches:** the deployed controller runs until step t, then PRIVILEGED true-delay
  learned compensation takes over, for t = end … end + 15.
- **Reported:**
  - whether switching at the end rescues the episode;
  - the last rescuing offset;
  - the number of rescuing branches.

  Episodes that no branch rescues are counted separately.

**A4. Gain for the true-delay configurations is 1.0.** Stage 1 has no gain disturbance.
Latched configurations use the latched gain, as F1 does.

**A5. The pipeline-dependence rule is applied as written:** a change in contact-aware's harm
against frozen SAC (drain/hold minus replay/drop) larger than 2 points in magnitude, with a
95% CI excluding 0.

**A6. Scope of this run.** Only the Stage-1 diagnostics are implemented. Candidates A–C are
**not** implemented yet; they wait for the Stage-1 checkpoint, because Stage 1 may change what
is worth building.
