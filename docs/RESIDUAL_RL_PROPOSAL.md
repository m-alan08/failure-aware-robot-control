# Residual RL on top of Candidate C: architecture and research proposal

**Status:** architecture and unit tests only. **Nothing has been trained. No simulator
evaluation episode has been run. No held-out seed has been used.** Candidate C, the frozen SAC
checkpoints, the Phase E estimator and the F1 predictor are unchanged. H0/H1 protocols are
untouched. Any training run below needs separate approval, a pre-registration and a registered
seed allocation.

Code: `src/farc/residual/` · tests: `tests/test_residual.py`

## 1. Architecture

### 1.1 Where the residual goes (audit)

Candidate C's loop body is identical in `HardwareController.step` (configuration "C") and
`delaychange.mechanism.run_episode` (G4 `C`):

| # | step | Candidate C state touched |
|---|---|---|
| 1 | `base = pi(normaliser(obs, goal))` | none (frozen SAC, frozen normaliser) |
| 2 | `est.observe(gripper)` | estimator position history |
| 3 | `dec = Sequential.step(est)` | CUSUM statistics, trusted delay, reset window, sigma², gain |
| 4 | if `dec.active`: `predicted = F1(obs, queued(d), gain)`; `command = pi(predicted)` | none (reads issued history) |
| 5 | else `command = base` | none |
| 6 | `a_C = clip(command, -1, 1)` | none |
| **→** | **residual inserted here** | |
| 7 | `est.issued(command)`; `issued.append(command)` | estimator command history; queued-command history |

Steps 1–6 never read anything a residual could produce in the current step. Step 7 is the only
write-back. The residual is therefore inserted **between 6 and 7**, and step 7 records
**a_final**:

    a_final = clip(a_C + alpha * clip(delta_RL, -1, 1), -1, 1)

**Why step 7 must record a_final, not a_C.** The estimator fits observed displacements against
*issued* commands, and the F1 predictor rolls out the *queued* commands. The plant executes
a_final. Recording a_C would make every residual an unmodelled input: it would bias the delay
fit and the CUSUM residuals, and F1 would predict with commands that were never sent. Recording
a_final keeps C's plant model physically correct. It is the same convention the hardware FIFO
and the simulator delay queue already use.

So the residual **can influence C's future decisions**, but only through the physical channel:
the command that was actually sent. It has no other path. A test checks this: an independent
reference Candidate C, fed the same positions and the same issued a_final, stays in identical
state step by step.

### 1.2 Modules

| module | role |
|---|---|
| `core.CandidateCCore` | C's loop split into `propose()` (steps 1–6, returns a_C and a read-only `Proposal`) and `commit(a_final)` (step 7). Imports the policy, normaliser, estimator, `Sequential` (via `mechanism.sequential`) and F1 unchanged. Frozen constants come from `hardware.controller`. |
| `observation.ResidualObservationBuilder` | builds the residual observation (§1.3). Read-only with respect to C (tested). |
| `policy` | `ResidualPolicy` protocol, `ZeroResidual`, and `ResidualActor`: a tanh-Gaussian SAC actor with a **zero-initialised output layer**, so its untrained deterministic output is exactly 0. |
| `controller.ResidualController` | composes core, builder and policy. alpha in [0, 1]; **alpha = 0 returns a_C itself, with no arithmetic on it**. |
| `controller.run_residual_episode` | closed loop on an `EpisodeSession`-like object (`ScheduledDelaySession` in simulation). Mirrors `mechanism.run_episode`. |
| `controller.ResidualEnv` | Gymnasium env for a future SAC run: the action is delta. It **refuses seeds ≥ 4,000,000** and refuses a missing seed. |
| `reward` | the reward of §4, as a pure function. |

### 1.3 Observation space (61 + 4 validity = 65 dims by default)

| group | dims | content | why it is deployable |
|---|---|---|---|
| task | 28 | `normaliser(obs, goal)`, exactly the SAC input | controller's own observation |
| actions | 12 | a_C, base SAC action, previous delta | computed by the controller |
| delay (opt.) | 9 | one-hot delay used by C, active, delay-reliable, gain-reliable, gain − 1 | C's own estimate |
| supervisor (opt.) | 8 | suspended, pending, alarm, CUSUM statistics / threshold | C's own state |
| prediction (opt.) | 4 | newest-transition residual at C's trusted lag; ‖obs − F1 prediction made d steps ago for now‖ for gripper and object; valid flag | causal: compares a past prediction with the present |
| time (opt., off) | 1 | t / 50 | known horizon |
| validity | 1 + #opt | 1 if the group was finite, else 0 (and the group is zeroed) | |

**Excluded on purpose:** true delay or schedule, true gain, simulator state, success, distance
to goal, oracle predictions, episode seed and condition label. The builder's signature accepts
only `(core, proposal, obs, goal)` (tested). The reward uses success and distance, which the
**trainer** sees and the policy does not.

**Failure handling:**
- A non-finite or mis-shaped observation or goal raises `ResidualObservationError`. This matches
  the hardware rule that such a sample ends the trial.
- A non-finite or mis-shaped residual output falls back to delta = 0 (a_final = a_C), flagged
  `invalid_residual`.
- Non-finite task features also force delta = 0, flagged `invalid_task_observation`.

### 1.4 Zero-residual equivalence: results

All on **synthetic** inputs (a linear stub plant with `replay_drop` delay semantics, and a fixed
random stand-in policy). Nothing here is a simulator episode.

| test | comparison | result |
|---|---|---|
| open loop, 5 random-walk trajectories × 60 steps | vs `HardwareController("C")`: commands, trusted delay, CUSUM statistics, reset start | **bitwise identical** |
| closed loop, 4 delay schedules × 3 seeds × 4 zero modes | vs `mechanism.run_episode` with `Sequential`: issued commands, delay used, alarms | **bitwise identical** |
| zero modes | `ZeroResidual` at alpha 0 and 0.5; random N(0, 3²) residual at alpha 0; **untrained `ResidualActor` at alpha 1** | all identical |
| non-vacuity guard | the fixture must make C compensate *and* alarm | passes |
| `ResidualEnv` at alpha 0 with delta = 1 every step | vs `mechanism.run_episode` | bitwise identical |

The full suite gives **562 passed**; ruff and mypy are clean. The new file has 70 tests, which
cover action bounds, all 16 observation-group combinations, read-only building, isolation, the
delay queue carrying a_final, causal prediction errors, deterministic resets, invalid inputs,
protocol misuse and held-out seed refusal.

### 1.5 Real-simulator zero-residual equivalence (§6.1.1, approved 2026-11-18)

Run with `uv run python scripts/check_residual_zero_equivalence.py`; outputs are in
`results/residual_zero_equivalence/`.

**Inputs:**
- G4 **debugging** seeds 3,690,000–009 only;
- all 41 G4 cells × the 3 frozen SAC policies;
- the real F1 model `8a960329…`;
- C's frozen constants.

**Comparison:** each residual mode against G4's `C` loop (`mechanism.run_episode` with
`sequential`).

| mode | episodes | commands bitwise identical | delay used, active, alarm, suspended identical |
|---|---|---|---|
| `ZeroResidual`, alpha 0 | 1,230 | **1,230** | **1,230** |
| untrained `ResidualActor`, alpha 0.2 | 1,230 | **1,230** | **1,230** |

**The check is not vacuous:**
- C compensated in 928 of the 1,230 episodes and raised an alarm in 963.
- The largest residual applied was 0.0.

**Frozen actor:** its weights were checked unchanged after every task.

Success outcomes were recorded, as a code check only, and are **not analysed**.

**Timing (§6.1.2, rough):** 3,690 episodes took 188 CPU-seconds, about 1 ms per step for the
environment plus C and F1. One million steps of environment interaction is therefore about 17
CPU-minutes. SAC updates, not the environment, will dominate the per-run cost, so the ~3 h/run
estimate in §7 is conservative.

## 2. Hypothesis and expected contribution

**H-R (primary).** A bounded residual (alpha ≤ 0.2) trained on top of frozen Candidate C raises
success under disturbances that **C does not model**, such as contact and object-dynamics
mismatch, temporally correlated observation noise and actuator noise. It does this without
losing C's delay gains or nominal performance.

**H-R0 (null to rule out).** Any gain comes from relearning **synthetic delay compensation**: for
example sharper transition timing on the exact `replay_drop` schedules, or memorising when
frozen SAC beats C on on/off delays (G3: SAC +3.8 over C there). This would not survive a change
of delay semantics or real hardware.

**Contribution if H-R holds:** a *layered* controller. Model-based identification and supervision
handle the structured failure (delay), and a small learned correction handles unstructured
residual error. It is also an ablation showing which of C's internal signals the correction
uses.

## 3. Algorithm: SAC, with qualifications

**Recommended: SAC with the zero-initialised actor, HER off, n-step 1, and the critic warm-started
on C-only rollouts.**

**For SAC:**
- **Continuity:** the base policies and Phase H/I/J retraining used SAC, and the repository's
  `retrain/qrl/algos/sac.py` is audited (J-3: no bug).
- **Off-policy reuse:** each episode costs about 50 F1 rollouts, so sample efficiency matters.
- **Entropy regularisation** helps exploration around a strong base.
- **Residual-on-base results** (Silver et al. 2018; Johannink et al. 2019) used off-policy
  actor-critic methods.

**Against, and how each is handled:**

| concern | why it matters | handling |
|---|---|---|
| Early high-entropy exploration damages a strong base | the policy starts at C, and noise only hurts | alpha bounds the damage. Schedule alpha from 0.05 to its target. Use a target entropy well below the −4 default (for example −8). Warm-start the critic for N steps on delta = 0 data before any actor update. |
| Sparse reward with near-ceiling success gives a weak signal | | a dense residual-cost term (§4), and training conditions weighted towards cells where C fails (§5) |
| Non-Markovian state: the delay queue and C's internal state | | the observation includes C's state summary and the previous delta. A recurrent critic is a later ablation, not the default. |
| HER relabelling changes the goal, but C's decisions do not depend on the goal | relabelled transitions would be off-model | **HER off** |

**Alternatives:**
- **TD3:** deterministic and simpler. Kept as a robustness check.
- **PPO:** on-policy, about 10× more samples. Rejected.
- **Offline RL from logged C rollouts:** cannot discover corrections that C never tried. Rejected
  as the primary method.
- **Behaviour cloning of an oracle residual (a privileged teacher):** strong candidate for a
  second stage, but it presumes the oracle signal is the one worth learning.

## 4. Reward

    r_t = r_task(success_t) + w_p * (dist_{t-1} - dist_t)
          - w_l2 * ||alpha * delta_t||^2 - w_rate * ||delta_t - delta_{t-1}||^2

| term | default | role |
|---|---|---|
| r_task | 0 on success, −1 otherwise | the same sparse task reward as the base SAC |
| w_p (progress shaping) | 0 (off) | ablation only: potential-style shaping on object-goal distance |
| w_l2 (residual magnitude) | 0.1 | prefer C. A correction must pay for itself. |
| w_rate (residual smoothness) | 0.05 | no chattering; protects hardware later |

The reward uses no delay information, so the residual is never paid directly for matching the
synthetic delay.

## 5. Training conditions

**Goal:** expose the residual to error that C cannot explain. Delay alone should **not** dominate
training.

| family | share | instances | C's model of it |
|---|---|---|---|
| nominal (no disturbance) | 25% | – | exact |
| delay (random pulses, on/off, persistent; `replay_drop`) | 25% | G2/G3 generator, *new* seeds | modelled |
| object dynamics: mass ×[0.5, 2], friction ×[0.5, 1.5] | 20% | per-episode draw | **not modelled** (F1 was trained nominal) |
| observation noise, AR(1)-correlated, σ ≤ 1 cm | 15% | | partly (CUSUM sigma) |
| actuator noise and gain drift | 15% | | gain only |
| **held out from training entirely** (for H-R0) | – | `drain_hold` delay semantics; object-size change; a new disturbance family chosen before training | |

The mass/friction and size disturbances need small additions to the environment factory. These
are **not implemented yet**, because they belong to the experiment and not the architecture.

**Proposed seed blocks (not registered; must be checked disjoint from every used block before
registration):**
- training episodes: 3,710,000–3,719,999
- debugging: 3,720,000–009
- checkpoint selection: 3,721,000–099
- development evaluation: 3,722,000–299

**Held-out 4,000,600–899** stays reserved for a separately approved confirmation.

## 6. Protocol

### 6.1 Before any training (debugging seeds only)

1. **Real-simulator zero-residual equivalence:** `run_residual_episode` at alpha 0 vs G4 `C` on
   debugging seeds. Bitwise commands are required. This is the only simulator use before
   approval.
2. **Budget check:** a sampled-step timing of the env, to confirm the compute estimate.

### 6.2 Nominal-performance protection

1. **Structural:** alpha ≤ 0.2 bounds per-step deviation (tested), and the zero-initialised actor
   starts training exactly at C.
2. **Training-time gate:** every 50k steps, evaluate on 100 nominal selection seeds. If nominal
   success drops by more than 2 points vs C, roll back to the previous checkpoint and halve the
   actor learning rate. After two rollbacks, the run stops and is reported as such.
3. **Deployment-time option, an ablation:** a gated residual, with delta multiplied by 0 when
   C's supervisor is "monitoring" with no alarm and no disturbance evidence. If H-R holds only
   with the gate, the residual is a disturbance-specific tool, not a general improvement.
4. **Evaluation gate (pre-registered):** C+R − C on nominal must be non-inferior with margin −1.
   The same margin applies to every no-delay cell.

### 6.3 Baselines and ablations (development evaluation)

| id | configuration | question |
|---|---|---|
| B0 | frozen SAC | reference |
| B1 | Candidate C | **primary comparator** |
| B2 | residual on frozen SAC (no C), same observation minus C's groups, same budget | does C add anything under a learned residual? |
| B3 | best prior retrained baseline (I-C) | context only |
| A1 | C+R with delay, supervisor and prediction groups removed | does the residual use C's signals? |
| A2 | C+R trained **without** delay conditions | is the gain delay-specific (H-R0)? |
| A3 | C+R trained on delay only | the H-R0 upper bound |
| A4 | alpha ∈ {0.05, 0.1, 0.2} | the size of correction needed |
| A5 | gated residual (§6.2.3) | nominal protection vs reach |
| P | PRIVILEGED: C with true delay and gain | headroom ceiling, diagnosis only |

### 6.4 Development evaluation

- **Scope:** G4's Arm R and Arm F cell structure plus the new families. 300 development seeds ×
  3 SAC policies × 3 residual training seeds.
- **Same statistics as G4:**
  - episode-seed cluster bootstrap;
  - fixed-sequence gatekeeping:
    1. nominal non-inferiority;
    2. C+R − C on the not-modelled families (practical minimum +2.0);
    3. C+R − C on delay families, non-inferiority with margin −1;
    4. the **H-R0 test:** C+R − C on the families held out from training.
- **Claim discipline:** H-R is claimed only if step 4 is also positive. A gain confined to
  training families is reported as "fits the training disturbances", not as robustness.

### 6.5 Held-out confirmation

Held-out confirmation follows only if development passes every step, with a frozen checkpoint
and a pre-registered plan on 4,000,600–899. It is one shot, in the G4 style.

## 7. Stopping criteria and compute

**Stop and report a null if any of these happens:**
1. After the full budget, the selection-seed C+R − C on not-modelled families is below +1.0 for
   every alpha.
2. The nominal gate trips twice in a run (§6.2.2).
3. A2 and A1 both reach about the same as C+R. The gain is then not attributable to the layered
   design, and B2 becomes the more honest result.
4. The real-simulator zero-residual check (§6.1) fails. Training does not start until it is
   bitwise.

**Budget:** 1M env steps per run (20k episodes). No extension after looking at development
results.

**Compute (to be confirmed by §6.1.2):**
- **One run:** C's per-step cost (windowed fit + F1 rollout) plus SAC updates. Comparable to
  Phase I runs (about 2.3 h for a similar step count), so about 3 h CPU per run.
- **Primary arm:** 3 seeds × 3 alpha = 9 runs.
- **A1–A3 and B2:** 3 seeds each = 12 runs.
- **Total:** about 21 runs, about 65 CPU-hours, about 8 h wall-clock at 8-way parallelism.
- **Development evaluation:** a few hours, at G4's rate of 155,700 episodes in 14 min multiplied
  by the F1 cost of the arms.

## 8. Critical assessment: will this add genuine robustness?

**Honest prior: modest at best on delay, uncertain elsewhere.**

**1. On delay, the headroom is small and it is mostly about timing.**
- G3 headroom: true delay and gain is only +5.1 over C on transient cells; perfect transition
  timing is +2.9.
- On persistent and value-change delays, C already matches the oracle.
- So a delay-trained residual can gain at most a few points, and the obvious way to get them is
  to learn the `replay_drop` transition structure, for example to suppress C's costly false
  alarms or to imitate frozen SAC on on/off cells.
- That is **relearning synthetic delay compensation**, and it is the H-R0 failure mode. The design
  above deliberately weights training away from delay and tests on unseen semantics.

**2. Six RL baselines have already failed to beat C under delay (J-3).**
- A residual is a different setting: it starts at C rather than at SAC, and alpha bounds it, so
  the J-2 failure modes (nominal drift, corrupted delay signal) are structurally limited.
- But J-3's main lesson still applies: noise augmentation produced a **uniform loss of
  competence**, not robustness. A residual with a small alpha probably avoids the loss mostly by
  doing little.

**3. The plausible source of real value is the error C cannot model.**
- Examples are F1's nominal-dynamics assumption under object mass or friction change, and
  correlated noise.
- The prediction-error features give the residual a direct, deployable signal of exactly that
  mismatch.
- This is the only part of the proposal with a credible novelty claim. Even there, a learned
  residual on a model-based controller is well established (residual policy learning; residual
  RL for contact tasks).
- The novel element would be **conditioning the residual on a supervisory delay estimator's
  internal state**, and showing (via A1) that the conditioning matters.

**4. Sim-to-real tension with the project's current priority.**
- The memory of 2026-11-18 records option E: stop retraining and focus on robot validation and
  the paper.
- A residual trained on simulator disturbances is exactly the component least likely to transfer.
  Its value would have to be re-established on hardware, which H1 does not yet cover.
- **Recommendation:** treat this as an exploratory extension that does not block H1 or the paper.
  Run §6.1 and the A2/A3 contrast first, because it is the cheapest decisive test of H-R0.

**5. The isolation guarantee has a deliberate limit.**
- C's internal state is not modified by the residual except through a_final in the estimator
  history, which is physically required.
- A residual can still change *when* C alarms, by shaping the commands C later identifies from.
  This is real closed-loop coupling, not a bug.
- It means "C's decisions are unchanged" holds only at alpha = 0. The evaluation must report C's
  alarm and false-alarm rates under C+R, not just success.

## 9. Not done (by design)

- No training. The only simulator episodes are the approved §1.5 code check on debugging seeds.
- No new environment disturbances (mass, friction, size, correlated noise).
- No seed registration.
- No changes to H0/H1, frozen checkpoints or prior results.
