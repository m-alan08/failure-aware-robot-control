# Hardware validation of online delay identification and compensation (proposal)

**Status:** a proposal for discussion and approval only.

**Not done:**
- No hardware has been bought, borrowed or booked.
- No controller has been deployed, and no experiment run.
- Nothing in this document has been implemented.

**Approval:** every stage below needs your separate approval before it starts, and so do any
purchase, any lab booking and any safety sign-off. Stages are approved one at a time, never as
a batch.

**Unchanged:**
- Candidate C, the frozen SAC policies, the Phase E estimator, the F1 contact model, and all
  simulation results.
- **Held-out simulation seeds 4,000,600–899** stay reserved. Hardware trials use no simulation
  seeds.

**Context:** the user chose option E after Phase J-3: stop retraining RL baselines and move to
physical validation. `docs/PAPER_READINESS.md` §4.1 lists real-robot validation as the
highest-value missing evidence.

---

## 1. Questions

The simulation claims, by how much each is expected to survive on hardware:

| sim claim (held-out) | hardware question | why it may not transfer |
|---|---|---|
| Online lag identification is accurate (Phase E) | **Q1:** does the least-squares estimator identify injected command delays on a real arm? | real tracking dynamics, sensing noise (J-3 shows the estimator abstains above about 0.75 cm i.i.d. noise), jitter, gain mismatch with the assumed K0 = 0.62 |
| Delay identification + kinematic compensation: **+44.9** over the frozen policy | **Q2:** does compensation recover task success under injected persistent delay? | it needs a competent frozen policy on hardware first (§3) |
| Contact-aware prediction: **+9.3** over kinematic | **Q3:** does the sim-trained F1 contact model predict real object motion? | contact, friction and object mass differ, so it may need re-fitting |
| Candidate C: **+3.5** on short and on-off delays | **Q4:** is change detection's benefit visible on hardware? | **unlikely to be measurable:** the effect is too small for feasible trial counts (§6) |
| (new) | **Q5:** how large and how variable is the *natural* latency of the real command path? | – |

**What hardware can and cannot establish:**
- **Can:** whether the *mechanism* (identify, then predict) works on a real plant, and whether
  the large effects survive.
- **Cannot:** re-confirm small simulation effects with comparable precision.

The paper will say this plainly.

## 2. Platform options (not chosen; no purchase proposed)

| option | match to simulation | main gap | notes |
|---|---|---|---|
| **Fetch mobile manipulator** | **exact** robot model as FetchPush-v4 | availability; mobile base is irrelevant | best match if a lab already has one |
| **Franka FR3 / Panda** | Cartesian control at 25 Hz is straightforward | different kinematics, workspace and dynamics | common in labs; well supported (libfranka, ROS 2) |
| **UR5e / UR3e** | servoing via RTDE | the same gaps; lower bandwidth for servo control | common |
| **Low-cost arm** (for example a hobby 5–6 DoF arm) | weakest | backlash and low-precision tracking dominate | cheapest; useful for a Q1/Q5-only pilot, not for Q2–Q3 |

**Recommendation:** use **existing lab access** to a Fetch or Franka through a collaborator or
host lab before considering any purchase. If no access exists, a reduced Q1/Q5 plan on a
low-cost arm is possible. It would be reported as component validation only.

**Common requirements:**
- an end-effector Cartesian position interface at 25 Hz (dt = 0.04 s, as in the simulation);
- a flat table and a puck of known size, matching FetchPush's 5 cm block as far as practical;
- object position sensing (§4.2);
- a deterministic software delay buffer in the command path (§4.3).

## 3. The central risk: is the frozen policy competent on hardware?

The sim-trained SAC policy was never trained for a real robot. If it cannot push the puck
without any delay, every comparison of compensation against the frozen policy is meaningless.

**Gate G-HW (pre-registered, checked first, in a small pilot):**
- The frozen policy (no delay, no compensation) reaches **≥ 60% success** on ≥ 30 nominal trials
  using the deployed observation pipeline.
- It also needs no manual intervention on ≥ 90% of those trials.
- The 60% threshold is lower than simulation's 99.8%. It is high enough that an absolute drop
  of 20–40 points under delay is measurable.

**If G-HW fails, three options, chosen with your approval and none pre-selected:**
1. **Component validation only:** run Q1, Q5 and the Q3 open-loop prediction test, and report
   no closed-loop task results.
2. **Policy adaptation:** a sim-to-real-adapted base policy, for example via domain
   randomisation or a short real-world fine-tune. This changes "the frozen policy", so it would
   be a new protocol, with the adapted policy then frozen *before* any delay trials.
3. **A substitute base policy:** a scripted or behaviour-cloned pusher, frozen, then delayed.
   It tests the compensation mechanism with a different policy class. That supports breadth
   claims but departs from the SAC results.

## 4. Pipeline design (proposed, not built)

### 4.1 Observation mapping

The policy expects the 25-dim FetchPush observation:
- gripper, object and relative positions;
- object rotation;
- velocities as per-step displacements;
- fingers.

Each entry gets a documented mapping. The design choices to fix in advance:
- **Frame alignment:** the table frame mapped to the simulation's world frame, with an offset
  and a workspace scale check.
- **Velocities:** computed by differencing positions at 25 Hz, or from the robot's joint
  state for the gripper.
- **Finger state:** fixed closed, as in FetchPush, where the gripper is blocked.
- **Object rotation and angular velocity:** from tag pose, or set to constants if pose is
  unreliable. Either choice is recorded and tested in the pilot.

### 4.2 Object sensing

| option | expected noise | latency | notes |
|---|---|---|---|
| motion capture | sub-mm | about 5 ms | best; available in some labs |
| overhead RGB camera + fiducial tag | about 1–3 mm, correlated in time | 30–60 ms | cheap; the latency must be measured |
| depth camera segmentation | 3–10 mm | 30–100 ms | weakest |

**This matters because of J-3.** The estimator uses *gripper* positions, which come from robot
proprioception and are typically sub-mm, so it should be unaffected. The policy's object input
is affected by perception noise.

**Pre-registered measurements:** the per-axis noise std and lag-1 autocorrelation of object and
gripper positions, on a static object and on a robot holding still.

### 4.3 Delay injection

- A software FIFO in the command path reproduces the simulation's replay/drop semantics
  (`DelayHistoryWrapper`): executed command = the command issued d steps earlier, with zeros
  before the first.
- Delays are integer multiples of 40 ms, using the same schedule families as simulation:
  - persistent constant delays d = 1–4;
  - single pulses and on-off delays, as descriptive only (§6).
- The natural latency (Q5) is measured separately and reported, and is *not* subtracted.
  "d = 0" means "no injected delay".

### 4.4 Controller configurations (code unchanged from G4)

| id | configuration |
|---|---|
| F | frozen SAC (seed fixed in advance, for example `fetch_push_fp32_seed1`; one policy only, see §6) |
| K | estimator + kinematic predictor |
| CA | estimator + F1 contact-aware predictor (sim weights, **not** fine-tuned) |
| C | Candidate C (CA + CUSUM change detector), same constants (h = 12, σ₀ = 4 mm) |

**Deployment assumption:** the controllers run in a single process at 25 Hz on the lab
computer. Before any robot motion, the timing per step is measured offline (Stage H0). The
budget is ≤ 10 ms for the 99th-percentile compute time.

### 4.5 Safety (required before any motion)

- The host lab's safety approval and a trained operator present at all times. No unattended
  runs.
- Cartesian workspace box, velocity and acceleration limits, and force/torque thresholds that
  stop motion, all enforced *below* the policy in the robot's controller.
- Hardware e-stop within reach.
- Delay injection increases overshoot risk. The workspace box is tested at d = 4 with no object
  before any closed-loop trial.
- Any safety stop ends the trial as a failure. It is logged, never re-run selectively.

## 5. Staged plan

Each stage has its own pre-registration document and amendments log, and needs your approval.

| stage | content | robot motion | needs |
|---|---|---|---|
| **H0** desk work | <ul><li>observation mapping</li><li>FIFO</li><li>logging schema</li><li>offline timing of estimator, F1 and CUSUM on the target CPU</li><li>replay of recorded *simulation* trajectories through the hardware code path to check bitwise equivalence with simulation features</li></ul> | none | no hardware; software only |
| **H1** system identification | <ul><li>free-space motions (no object) with injected d = 0–4</li><li>estimator accuracy and reliability per d (Q1)</li><li>natural latency and jitter (Q5)</li><li>gain K estimate vs 0.62</li><li>sensor noise characterisation</li></ul> | yes, no contact | lab access, safety sign-off |
| **H2** nominal gate | frozen policy, no delay, ≥ 30 trials: **G-HW** (§3) | yes | H1 complete |
| **H3** open-loop contact model | push data from the frozen policy (nominal, from H2 logs); F1 one-step prediction error on hardware vs simulation (Q3) | none extra | H2 logs |
| **H4** main closed-loop comparison | F vs K vs CA (vs C) under persistent injected delay (Q2) | yes | G-HW passed |
| **H5** (optional) | short and on-off delays, C vs CA, descriptive (Q4) | yes | H4 complete |

## 6. Hypotheses and power (to be fixed in each stage's pre-registration)

**Pairing:**
- Hardware has no episode seeds, so each trial uses a **pre-generated list of initial object
  and goal positions**, drawn once from a new generator seed. The draw is from the FetchPush
  distribution, clipped to the safe workspace.
- Every configuration runs on every start state. Configuration order is randomised within each
  start state, and resets are by fixture or template.

**Analysis:**
- The paired cluster bootstrap by start state, as in simulation.
- Exact McNemar tests as a check.

**Feasibility.** Approximate trial counts for 80% power at one-sided α = 0.025, paired binary
outcomes, from a normal approximation for the paired difference. Each must be recomputed from
H2 pilot rates before H4.

| claim | sim effect | assumed discordant share | pairs needed | trials (×configs) | status |
|---|---|---|---|---|---|
| **HP1:** K − F under persistent d ∈ {2, 3, 4} | +45 | about 50% | **≈ 20–30** | ≈ 60 | **primary, feasible** |
| **HP2:** CA − K, persistent | +9 | about 15–20% | **≈ 150–250** | ≈ 400 | **secondary,** feasible only with automated resets |
| **HP3:** C − CA, short and on-off | +3.5 | about 10% | **≈ 650+** | ≈ 1,300 | **descriptive only; not powered** |
| **Q1:** lag-identification accuracy per d | – | – | ≈ 20 free-space episodes per d | ≈ 100 | **primary, feasible** |

**Effort estimate:**
- About 1–2 min per trial with manual reset.
- H1 + H2 + H4 (HP1): about **2–3 lab days**.
- HP2: about **1–2 additional weeks**, unless resets are automated.

**Policy seeds:** one frozen policy (pre-selected, not chosen by hardware performance). Three
would triple the cost. Results are conditional on that one policy, and the paper will say so.

## 7. What the paper can and cannot claim

| outcome | claim |
|---|---|
| Q1 + HP1 pass | "Online lag identification and kinematic compensation transfer to a physical arm under injected delay." This is mechanism-level, for one policy, one robot and injected delay. |
| HP2 passes | the contact-aware advantage transfers, with or without re-fitting as reported |
| G-HW fails | component results only: the frozen sim policy did not transfer, and that is reported |
| HP3 | no confirmatory claim in any outcome |

The paper will **not** claim:
- superiority over delay-aware RL in general;
- that natural latency behaves like injected latency, unless Q5 shows it;
- any result from hardware runs that were not pre-registered.

## 8. Integrity rules carried over

- Pre-registration per stage, amendments before each stage, write-once raw logs, run once.
- No retuning after seeing results, and no selective re-runs.
- Nothing changes to Candidate C's code or constants, the estimator, F1 or the SAC weights. Any
  hardware-adapted variant is a separate, named configuration.
- No private paths, lab names or personal data in committed outputs. Videos are kept private
  unless you approve release.
- Simulation held-out seeds 4,000,600–899 are untouched by this work.

## 9. Decisions requiring your approval

1. **Platform route:** seek access to an existing Fetch or Franka (recommended), versus a
   low-cost component-only pilot. **No purchase is proposed.**
2. **Scope:** Q1 + Q5 + HP1 as primary, HP2 secondary, HP3 descriptive.
3. **G-HW threshold** of 60% nominal success, and the fallback if it fails (§3, options 1–3).
4. **Object sensing:** motion capture preferred, fiducial camera acceptable.
5. **One frozen policy** (proposed `fetch_push_fp32_seed1`, fixed now rather than chosen on
   hardware).
6. **Approval to begin Stage H0 only:** software and desk work, with no robot motion. It is
   the only stage that needs no hardware.
