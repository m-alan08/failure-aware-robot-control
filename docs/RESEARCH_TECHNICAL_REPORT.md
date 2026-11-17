# Online Delay Identification, Contact-Aware Prediction and Change Detection for a Frozen Manipulation Policy: A Pre-Registered Simulation Study

*Technical report, consolidating Phases 1–G4 of `m-alan08/failure-aware-robot-control`.
Simulation only. Development results are labelled "dev"; held-out confirmations are labelled
"HO". Methodological caveats are in `docs/RESEARCH_AUDIT.md`.*

---

## 1. Abstract

We study how far a *frozen* goal-conditioned manipulation policy (SAC+HER on FetchPush-v4) can
be protected against actuator command delay without retraining it.

**Method.** A causal least-squares estimator identifies the delay online from the
command–motion history. The policy is then evaluated at a predicted future observation, in the
manner of a Smith predictor. We compare three predictors:
- a static-object kinematic model;
- a learned contact-aware dynamics model;
- the learned model plus a sequential (CUSUM) change detector that pauses compensation and
  resets the identification window when the delay changes.

**Design.** All experiments are paired across controllers. They are analysed with
episode-seed cluster bootstrap intervals, and the confirmatory claims were pre-registered and
tested once on held-out seeds.

**Results on held-out seeds:**
- Delay identification with kinematic compensation raises success on delayed episodes by
  **+44.9 points [+43.1, +46.7]** over the frozen policy.
- Contact-aware prediction adds **+9.3 [+8.3, +10.4]** over the kinematic predictor.
- Its advantage comes from predicting the object rather than the gripper: **+8.7 [+7.7, +9.7]**
  over a gripper-only learned variant.
- On randomly timed, short-lived and on-off delays, the change-detecting controller ("Candidate
  C") adds **+3.5 [+2.4, +4.6]** over the contact-aware compensator. It is non-inferior under
  persistent delay (**+0.5 [+0.2, +0.9]**).
- It beats the same pause-and-reset mechanism driven by another episode's alarms by **+3.2
  [+2.2, +4.2]**.

**Negative findings.**
- When delays are short and frequent, compensating with an outdated delay estimate is worse
  than leaving the frozen policy alone. Even Candidate C remains **2.2 points [−3.2, −1.3]**
  below the frozen policy there.
- Failure-triggered recovery and small residual corrections did not improve success.

All results are in simulation, on one task, with three policies.

## 2. Introduction and motivation

**Why this question.** Learned manipulation policies are usually trained under nominal
actuation. Command latency is common in deployed systems, for example from communication,
computation or buffering. Our baseline characterisation found it to be the most damaging of
the disturbances tested: a constant 3-step delay reduced FetchPush success to 13%. Retraining
for every actuation change is expensive, and a frozen policy is often all that is available.

**The project's arc.** The project began as *failure-aware hierarchical control*, with
detectors, residual RL and recovery orchestration. Three early phases redirected it:
- learned failure predictors give useful warning lead time (Phase B);
- but triggered deterministic recovery did not improve success (Phase C);
- and small residual corrections were indistinguishable from a random-direction placebo
  (Phase D).

The work then moved from *reacting to failure* to *identifying the cause*, actuator delay, and
compensating for it model-based (Phases E–F1). It then turned to *non-stationary* delay, where
a delay starts, ends or changes value (Phases G1–G4).

**Research questions:**
1. Can online delay identification recover performance without retraining the policy?
2. Does predicting contact dynamics improve compensation beyond kinematic prediction?
3. Does sequential change detection improve adaptation when the delay is non-stationary?
4. Does the benefit depend on detecting each episode's own changes?
5. When does compensation do harm?

## 3. Problem formulation

- **Task:** FetchPush-v4. The episode horizon is $T = 50$ steps with a control period of
  40 ms.
- **Observation:** $o_t$, comprising the gripper position $g_t \in \mathbb{R}^3$, the object
  pose and velocities, and the gripper velocity. The goal is $G$.
- **Policy:** a frozen deterministic actor $\pi$ issues a command $c_t = \pi(o_t, G) \in
  [-1, 1]^4$, three position components plus the gripper.

**Actuator delay.** A delay schedule $d(t) \in \{0, \dots, 4\}$ determines which issued
command is executed:

$$c^{\text{exec}}_t = \begin{cases} c_{t-d(t)} & t - d(t) \ge 0 \\ 0 & \text{otherwise} \end{cases} \qquad \text{(replay/drop semantics)}$$

- When $d$ increases, already executed commands are executed again.
- When $d$ decreases, queued commands are dropped.
- Other action perturbations, such as gain loss $c \mapsto (1-s)c$, apply after the delay and
  before clipping.

**Objective.** Choose, causally, an observation transformation $\tilde o_t = \Phi(o_{0:t},
c_{0:t-1})$, so that the unchanged policy $c_t = \pi(\tilde o_t, G)$ maximises task success.
Success is the object within 5 cm of the goal at the final step. Training, fine-tuning or
replacing $\pi$ is out of scope by design.

## 4. Baseline controller and simulated actuator dynamics

**Base policies.**
- Three SAC+HER actors (`fetch_push_fp32_seed{0,1,2}`), trained for $10^6$ steps in the
  upstream repository at a public commit.
- They are loaded read-only and SHA-256 verified; their weights are checked unchanged after
  every run.
- Undisturbed success is 97–100% across seed blocks.

**Actuation.**
- The Fetch arm is driven through a MuJoCo mocap weld. The command's position part is scaled
  by 0.05 m and added to the gripper's current position, and the gripper follows over 20
  physics substeps.
- A probe on debugging seeds found the plant close to linear:

$$\Delta g_t \equiv g_{t+1} - g_t \approx K_0 \cdot 0.05 \cdot c^{\text{exec}}_{t,xy}, \qquad K_0 = 0.62,\; R^2 \in [0.94, 1.00]$$

- The effective gain drifts from about 0.64 to 0.56 over an episode, and about 25% of base
  commands saturate at ±1.

**Disturbance grid, frozen SAC success, development:**

| disturbance | success |
|---|---|
| constant delay 3 | 13% |
| gain loss of 50% (scale 0.5) | 89% |
| position noise up to 1 cm | 97–99% |
| short pulses of delay 3 lasting 10–15 steps (G1) | 97–98% |

## 5. Online delay estimation (Phase E)

**Estimator** (`farc.adaptation.estimator`). At step $t$, it uses only the observed (noisy)
gripper positions and the issued commands. For each candidate lag $k \in \{0, \dots, 4\}$, it
fits a gain through the origin over the window $W_t = \{t-10, \dots, t-1\}$ (horizontal plane):

$$\hat\beta_k = \frac{\sum_{\tau \in W_t} \langle \Delta g_\tau, 0.05\, c_{\tau-k} \rangle}{\sum_{\tau \in W_t} \|0.05\, c_{\tau-k}\|^2}, \qquad \mathrm{SSE}_k = \sum_{\tau \in W_t} \|\Delta g_\tau - \hat\beta_k\, 0.05\, c_{\tau-k}\|^2$$

$$\hat d_t = \arg\min_k \mathrm{SSE}_k, \qquad \hat\gamma_t = \hat\beta_{\hat d}/K_0$$

**Reliability gates.** An estimate is reported only if all of these hold:
- at least 4 transitions;
- excitation $\frac{1}{|W|}\sum\|0.05 c\|^2 \ge 2.25\times10^{-4}$;
- $R^2 \ge 0.8$;
- for the delay, a margin $\mathrm{SSE}_{(2)}/\mathrm{SSE}_{(1)} \ge 2$.

All constants were fixed on debugging seeds.

**Latch.** The deployed controller keeps the last reliable $\hat d$. It engages on a reliable
$\hat d > 0$, and releases only on a reliable $\hat d = 0$.

**Kinematic compensation** (Phase E, a Smith-predictor-style step). With latched $\hat d$, the
next $\hat d$ executed commands are already known: $c_{t-\hat d}, \dots, c_{t-1}$. The predicted
gripper displacement is

$$\hat\Delta = K_0 \hat\gamma \cdot 0.05 \sum_{j=1}^{\hat d} c_{t-j,xy}$$

The observation is shifted accordingly: gripper $+\hat\Delta$, object-relative position
$-\hat\Delta$, object assumed static. The policy is evaluated at the shifted observation.

**Algorithm 1 (Phase E controller, per step $t$):**
1. Observe $o_t$ and update the estimator with $g_t$.
2. Compute $(\hat d, \text{reliable})$. If reliable, update the latch.
3. If latched $\hat d > 0$: $\tilde o_t \leftarrow \text{shift}(o_t, \hat\Delta)$; else
   $\tilde o_t \leftarrow o_t$.
4. Issue $c_t = \pi(\tilde o_t, G)$, and record $c_t$ in the estimator.

## 6. Contact-aware prediction and compensation (Phase F1)

**Model** (`farc.contact.dynamics`).
- Architecture: an MLP $f_\theta$ with 2 × 256 tanh hidden layers.
- Inputs: translation-invariant observation features (gripper velocity, object position
  relative to the gripper, object rotation, relative and angular velocities, gripper height
  above the object) and the executed position command.
- Outputs: the one-step change in gripper and object position, object rotation, and the
  velocities.

**Training.**
- 352,800 transitions from frozen-policy and Phase E rollouts on dedicated training seeds,
  covering undisturbed runs, delays 1–4 and gain loss of 50%.
- Targets are noise-free.
- Selected over {64, 256} width × {0, 10⁻⁴} weight decay, by open-loop position error over
  horizons 1–4 on separate selection seeds, with contact states weighted ×2.
- Frozen model SHA-256 `8a960329…`.

**Compensation.** The model is rolled out $\hat d$ steps from the current (noisy) observation
with the queued commands, each scaled by $\min(\hat\gamma, 1)$. The policy observation is
rebuilt consistently: relative positions are recomputed from predicted absolute positions, and
the finger entries are kept. Contact is never an input to the deployed controller.

This configuration is called **`contact_learned`** (the contact-aware compensator) below.

## 7. Sequential change detection and supervisory control (Phases G1–G2)

**Motivation.** The 10-transition window mixes data from before and after a change, so a new
delay takes a median of about 8–9 steps to identify (E2, G1). During that time, compensation
keeps running with the stale estimate.

### 7.1 G1 supervisors, on top of `contact_learned`

- **A: evidence-based release.**
  - While $\hat d > 0$ is latched, a one-sided CUSUM accumulates the log-ratio
    $\ell_t = \log(\mathrm{SSE}_{\hat d}/\mathrm{SSE}_0)$:
    $S_t = \max(0, S_{t-1} + \ell_t - \log 2)$.
  - Compensation is released when $S_t > h$, with $h = 0.5$ set on calibration seeds from the
    false-release rate only.
  - Every supervisor uses a minimum dwell of 5 steps.
- **B:** per-step selection between the kinematic and learned predictors by their realised
  error.
- **C:** a contact gate (learned model only near the object), for efficiency.

### 7.2 G2 candidates (all causal)

- **A, short window:** the same estimator with window $w_s = 7$.
- **B, two-timescale:**
  - A fast window ($w_f = 4$) raises an alarm when it reliably disagrees with the latched lag
    on 2 consecutive steps.
  - Compensation is then suspended until the slow window reliably identifies $d' > 0$, at
    least 5 steps later.
- **C, sequential multi-alternative CUSUM ("Candidate C")**, `farc.delaychange.candidates.Sequential`:
  - On the newest transition, the per-lag residual is

    $$r_\tau(k) = \big\|\Delta g_\tau - K_0\, \hat\gamma\, 0.05\, c_{\tau-k,xy}\big\|^2$$

  - With trusted lag $d$, the Gaussian log-likelihood ratio for alternative $k \ne d$ is
    $\lambda_\tau(k) = \frac{r_\tau(d) - r_\tau(k)}{2\hat\sigma^2}$. Here
    $\hat\sigma^2 = \max(\sigma_0^2, \mathrm{SSE}_{d}/2n)$, and it is updated at each reliable
    estimate.
  - The CUSUM statistics are $S_k \leftarrow \max(0, S_k + \lambda_\tau(k) - \kappa)$, with
    $\kappa = 0.5$.
  - **Alarm** when $\max_k S_k > h$. The change point $\hat\tau$ is the transition after the
    winning statistic was last zero.

**Algorithm 2 (Candidate C, per step $t$):**
1. If not paused and not pending, update $S_k$ for all $k \ne d$ using $r_{t-1}(\cdot)$.
2. **On alarm:**
   - **reset:** the identification window starts at $\hat\tau$;
   - **pause:** if $d > 0$, suspend compensation, so the frozen policy acts on $o_t$;
   - set pending, which stops detection until the reset window is reliable;
   - clear $S$.
3. Fit the window $[\max(t-10, \hat\tau), t)$ with the Phase E gates.
4. If the fit is reliable: set $d \leftarrow \hat d$, unpause, clear pending, and update
   $\hat\sigma^2$ and $\hat\gamma$.
5. Compensate with `contact_learned`'s rollout if $d > 0$ and not paused.

**Constants:** Gaussian variant, $h = 12$, $\sigma_0 = 4$ mm. They were chosen on calibration
seeds from detection criteria only (false activation, wrong-lag rate, release latency). The
choice sits at the edge of the grid, which was noted and not widened.

**Equivalences checked:** with no alarm, C is bitwise identical to `contact_learned` (tested in
the simulator). `EventReplay` with C's own alarms reproduces C bitwise.

## 8. Experimental methodology

- **Pairing:** within a cell (policy × condition × episode seed), every configuration starts
  from the same initial state and goal, and uses the same perturbation random streams.
- **Delay schedules:** from G3 on, schedules are drawn per episode seed and shared by every
  configuration and policy.
- **Seed hygiene:** disjoint, code-enforced blocks per phase and role:
  - debugging, calibration or selection, evaluation, exploratory;
  - held-out (4,000,000+) is reachable only through a dedicated guard, once per approved
    slice.
- **Pre-registration:** each confirmatory phase had a plan whose SHA-256 is pinned in the
  run config. Implementation decisions were recorded as numbered amendments before the
  evaluation seeds were used.
- **Run integrity:**
  - runs require a clean git tree;
  - raw files are write-once and a LOCK file is written;
  - frozen actor weights are checked after every run;
  - the analysis is pre-written and run once.
- **Statistics:**
  - Differences are paired success-rate differences, in percentage points.
  - 95% CIs come from an episode-seed cluster bootstrap with 10,000 resamples (2,000 in the
    Phase E pilot).
  - One-sided p-values are the share of bootstrap draws at or beyond the bound.
  - Multiplicity is controlled by Holm, intersection-union tests and fixed-sequence
    gatekeeping, as each plan specified.
  - Non-inferiority margins: −1 point vs contact-aware, −2 vs frozen SAC.
  - Harm gates: CI low > −2 and at most 2% of base successes harmed on undisturbed and on
    1 cm-noise episodes.
- **Controls:**
  - privileged oracles (true delay, true transition times), used only as references;
  - fixed-delay, shuffled, yoked and random-timing shams;
  - and, in G3, factorial pause-only and reset-only variants.

## 9. Results

Development (dev) and held-out (HO) results come from different seed blocks and often
different condition sets. **Compare within a row group, not across groups.**

### 9.1 Consolidated table

| phase | seeds | comparison | scope | estimate [95% CI] | rescued / harmed | gates |
|---|---|---|---|---|---|---|
| E pilot | dev, 30 seeds | estimated delay − frozen SAC | 5 actuator conditions | +16.7 [+14.2, +19.1] | 77 / 2 | no harm |
| E2 | dev, pre-reg. | estimated delay − frozen SAC | delay C1–C9 | +39.2 [+36.4, +41.9] | – | passed |
| E2 | dev | same, delay starts at step 15 (no start-up cue) | C5 | +17.0 [+12.7, +21.7] | 55 / 4 | – |
| E2 | dev | same, transient delay (steps 15–29) | C6 | −0.7 [−2.7, +1.3] | 5 / 7 | – |
| F1 | dev, pre-reg. | contact-aware − Phase E | C1–C7 | +10.4 [+8.5, +12.3] | – | passed |
| F1 | dev | object prediction error in contact, static − learned | offline | 33.0 → 12.4 mm | – | – |
| **E/F1** | **HO 4,000,000–299** | **H1:** Phase E − frozen SAC | C1–C7 | **+44.9 [+43.1, +46.7]** | – | passed |
| **E/F1** | **HO** | **H2:** contact-aware − Phase E | C1–C7 | **+9.3 [+8.3, +10.4]** | – | passed |
| **E/F1** | **HO** | **H3:** contact-aware − gripper-only learned | C1–C7 | **+8.7 [+7.7, +9.7]** | – | passed |
| G1 | dev, pre-reg. | supervisor A − contact-aware | transients T1–T4 | +1.4 [+0.7, +2.2] | – | passed |
| G1 | dev | contact-aware − frozen SAC | T1 / T2 / T3 | −6.7 / −6.3 / −16.3 | – | – |
| G1 | dev | contact gate (C) − contact-aware | C1–C7 | −3.4 [−4.8, −2.0] | – | – |
| G2 St. 3 | dev, fixed grid | C − contact-aware | T + I | +1.7 [+1.1, +2.4] | 303 / 188 | passed |
| G2 St. 3 | dev | B − contact-aware | T + I | +0.9 [+0.5, +1.2] | 134 / 76 | passed |
| G2 St. 3 | dev | A − contact-aware | T + I | −1.6 [−2.3, −0.9] | 214 / 321 | passed |
| G2 St. 3 | dev | C − contact-aware | persistent | +0.1 [−0.3, +0.4] | 8 / 7 | – |
| G2 St. 3 | dev | C − same-condition yoked sham | T + I | +0.5 [−0.0, +1.1] | 244 / 209 | – |
| G2 sham | dev, exploratory | C − random-timing pause sham | T + I | +1.6 [+0.8, +2.4] | 309 / 204 | – |
| G3 | dev, random schedules | C − contact-aware | RT + RI | +3.0 [+2.4, +3.7] | 862 / 501 | passed |
| G3 | dev | M1: pause-only − random-timing pause | RT + RI | +1.3 [+0.9, +1.8] | 516 / 357 | – |
| G3 | dev | M2: reset-only − contact-aware | RT + RI | −1.3 [−2.0, −0.7] | 744 / 905 | – |
| G3 | dev | M3: pause × reset interaction | RT + RI | +2.4 [+1.7, +3.1] | – | – |
| G3 | dev | M4: C − cross-episode yoked sham | RT + RI | +2.4 [+1.8, +3.0] | 784 / 497 | – |
| **G4** | **HO 4,000,300–599** | **P1:** C − contact-aware | random RT + RI | **+3.5 [+2.4, +4.6]** | 270 / 145 | passed |
| **G4** | **HO** | **P2:** C − contact-aware, margin −1 | persistent, random onset (RP) | +1.1 [−0.2, +2.3] | 69 / 50 | passed |
| **G4** | **HO** | **P2:** same | persistent, fixed grid (F-P) | +0.3 [+0.04, +0.63] | 44 / 26 | passed |
| **G4** | **HO** | **P3:** C − cross-episode yoked sham | random RT + RI | **+3.2 [+2.2, +4.2]** | 258 / 144 | – |
| **G4** | **HO** | C − contact-aware | fixed grid T + I | +1.4 [+1.0, +1.8] | 904 / 627 | – |
| **G4** | **HO** | C − frozen SAC | random RT + RI | −2.2 [−3.2, −1.3] | 101 / 181 | – |
| **G4** | **HO** | C − frozen SAC | fixed grid T + I | −1.6 [−1.9, −1.2] | 454 / 766 | – |

**Notes on P2.** It was pre-registered as the pooled RP + F-P contrast, which gave +0.5
[+0.2, +0.9] (113 / 76). It is shown here per arm, because the two arms are different designs.

**Safety gates.** All held-out gates passed. On no-delay episodes, harm was 0 of 897
(undisturbed) and 1 of 861 (1 cm noise). In every family, C − contact-aware had CI low > −2.

### 9.2 Held-out success rates (%)

**E/F1 held-out**, constant and mid-episode delays:

| condition | frozen SAC | Phase E | contact-aware | oracle |
|---|---|---|---|---|
| delay 1 | 80.7 | 97.7 | 98.9 | 99.4 |
| delay 2 | 29.2 | 89.3 | 98.1 | 99.6 |
| delay 3 | 13.4 | 81.1 | 95.0 | 99.2 |
| delay 4 | 10.1 | 77.6 | 93.6 | 99.1 |
| delay 3 from step 15 | 74.9 | 91.7 | 95.2 | 97.9 |
| delay 3 + 0.5 cm noise | 13.1 | 82.0 | 93.4 | 98.8 |
| delay 3 + 50% gain loss | 62.0 | 78.6 | 89.0 | 91.3 |
| none | 98.9 | 98.9 | 98.9 | 98.9 |

**G4 held-out**, non-stationary delays:

| family | frozen SAC | contact-aware | **C** |
|---|---|---|---|
| random single pulses (RT) | **97.4** | 95.5 | 97.1 |
| random on-off (RI) | **90.2** | 80.7 | 86.0 |
| random change of value (RS) | 30.1 | 90.3 | **94.3** |
| random-onset persistent (RP) | 54.7 | 93.2 | **94.2** |
| fixed single pulses (F-T) | **98.3** | 97.3 | 98.0 |
| fixed on-off (F-I) | **85.8** | 74.0 | 78.3 |
| fixed persistent (F-P) | 49.7 | 96.2 | **96.6** |
| noise / gain loss (NG) | 58.9 | 89.9 | **91.5** |
| none (H) | 97.7 | 97.7 | 97.7 |

### 9.3 Computation

Per-step controller time includes the estimator and, when active, the learned rollout. It was
measured as median wall-clock on one Apple-silicon laptop core while 9 processes ran
concurrently, and is descriptive only:

| configuration | median | p99 |
|---|---|---|
| frozen SAC | 71 µs | – |
| contact-aware | 81–82 µs | – |
| Candidate C | 112–116 µs | 716 µs |

All are far below the 40 ms control period. No real-time or embedded measurement exists.

## 10. Mechanism analysis

**Why stale compensation hurts** (G2 Stage 1, development).
- Branching experiments restored exact simulator states and varied the step at which
  compensation was released after a delay ended.
- Stale compensation was the main cause of loss: the recovered share was 1.39 [1.00, 2.31].
  Success fell by about 0.21 points per additional stale step.
- Releasing at the true end rescued 25 of 27 tested failure states.
- Onset latency also mattered: under persistent onsets, the true delay gave +8.7 over the
  deployed estimator.

**What Candidate C does** (G3, development, random schedules). A 2 × 2 factorial replayed C's
own alarms with the pause and the reset switched separately:

| on RT + RI vs contact-aware | with C's alarms | with true transition times (PRIVILEGED oracle) |
|---|---|---|
| pause only | +2.0 [+1.5, +2.5] | +3.6 [+3.0, +4.1] |
| reset only | −1.3 [−2.0, −0.7] | +0.5 [−0.2, +1.1] |
| interaction | +2.4 [+1.7, +3.1] | +1.9 [+1.2, +2.6] |
| full | +3.0 [+2.4, +3.7] | +5.9 [+5.4, +6.5] |

**Reading** (development only):
- *Reset* speeds up identification after a change. Fewer onsets go unidentified: 8,760 vs
  13,263 of 20,052 on random on-off delays.
- *Reset* alone carries C's gain on changes of value (+2.4) and on persistent onsets (+1.6).
- *Reset without pause* keeps stale compensation running after a delay ends (median release 4
  steps). On on-off delays it compensates on 40% of delay-free steps and loses −3.7.
- *Pause* removes that stale compensation. Hence the positive interaction.

**Sensitivity.** Replayed alarms are exactly causal only up to the first divergence. About a
third of C's later alarms would not occur on a pause-only trajectory.
- On the 45% of pairs where every replay agreed with C, the interaction was +0.1 [−0.5, +0.6].
- But that subset is 85% single-pulse episodes with about 1 alarm each, so it excludes most
  of the on-off episodes where the interaction arises.
- The oracle factorial, which has no replay, supports the same pattern.
- **The size of the interaction with C's own alarms is therefore uncertain.** The mechanism
  results were not tested on held-out seeds.

**Detection headroom** (G3, development). Pause and reset at the true transition times would
add +2.9 [+2.4, +3.4] over C. C alarms a median 1–2 steps after a transition, but catches only
52–62% of transitions within 5 steps on single pulses and on-off delays.

## 11. Negative findings

1. **Stale compensation is worse than no compensation for short delays.**
   - Dev G1: frozen SAC reached 97–98% on 10–15-step delay pulses, where contact-aware lost
     6–16 points.
   - HO G4, random on-off delays: contact-aware is 9.5 points below frozen SAC, and C is still
     4.2 below (86.0 vs 90.2). C − SAC on random RT + RI is −2.2 [−3.2, −1.3], so it is not
     non-inferior at −2.
2. **A shorter identification window** (G2 candidate A, $w = 7$) increases false and wrong-lag
   compensation, with a net loss of −1.6 [−2.3, −0.9].
3. **Gain compensation by inverse scaling is bounded by the action limits.** Even the oracle
   gains only 3–9 points (Phase E).
4. **Triggered deterministic recovery** (Phase C) was about 1 point worse than the frozen
   policy, and always-on recovery 11 points worse. An oracle trigger would gain +5.5, so the
   triggers, not the recovery, were limiting.
5. **Small residual corrections** (Phase D) were indistinguishable from a random-direction
   placebo of the same size: none of the 36 interpretable corrections beat continuing the
   policy.
6. **Contact gating** of the learned model (G1 strategy C) cut learned-model calls by 57% but
   lost 3.4 points.
7. **Weak specificity controls.** E2's shuffled sham and G2's same-condition yoked sham could
   not test specificity, because the episodes they drew from shared the delay or its timing.
   This was corrected in G3/G4.

## 12. Limitations and threats to validity

- **External validity:**
  - one task (FetchPush), one simulator (MuJoCo), mocap position control, full state;
  - three SAC+HER policies from one training codebase;
  - synthetic transport delay with replay/drop semantics;
  - delay schedules drawn from distributions chosen by the experimenters.
- **No real robot.** Nothing here establishes transfer to hardware, where the delay comes with
  jitter, plant dynamics and perception error.
- **Baselines:** no comparison with a policy retrained under delay randomisation, with
  delay-aware RL (state augmented with the action buffer), or with MPC or an analytic object
  predictor.
- **Statistics:**
  - Policies are fixed effects: CIs do not cover variation over SAC training seeds.
  - Many development contrasts were run across phases. Only the E/F1 and G4 held-out claims
    are confirmatory.
- **Calibration:** C's constants sit at the edge of their calibration grid.
- **Learned model:**
  - trained in the same simulator;
  - its training conditions overlap the E/F1 evaluation conditions;
  - it has a small spurious drift of the object in free space (4.8 mm).
- **Reproducibility:** checkpoints, models and raw data are not yet archived outside one
  machine. Cross-platform determinism is untested (see the audit, R1–R3).
- **Researcher degrees of freedom:** the G3/G4 random-schedule generator was designed after the
  G2 results. G4 confirms performance under that generator on fresh seeds, not under an
  independently specified distribution.

## 13. Related-work research agenda

> **Update (2026-11-17):** a targeted literature check now exists in `docs/RELATED_WORK.md`,
> with the novelty assessment in `docs/NOVELTY_AUDIT.md`.
> - **Closest prior work for compensation:** Wafi (2026, arXiv:2611.00668), a frozen DDPG actor
>   plus a Smith predictor with a fixed nominal delay.
> - **Closest prior work for change detection:** Wang et al. (2025, RA-L,
>   doi:10.1109/LRA.2025.3536291), Bayesian online change-point detection for non-stationary
>   delays in RL.
> - **Consequence:** no algorithmic novelty is claimed. The table below is the original agenda.

No literature search has been done for this report. The relationships below are conceptual.
The two classical references named are the standard originators of their methods; they should
still be checked bibliographically before citation.

| area | relation to this work | questions to verify |
|---|---|---|
| Dead-time compensation (Smith predictor, O. J. M. Smith, 1950s) | Phase E's observation shift is a Smith-predictor-style step applied to a learned policy's *input* rather than to a linear controller | Is predicting the observation for a frozen learned policy with an identified delay already described? How do adaptive or modified Smith predictors handle unknown, time-varying delay? |
| Time-delay identification and adaptive control | Lag selection by least squares over a sliding window; gain identification | What are the standard online delay estimators (e.g. correlation-based, ARX-order selection)? How is excitation handled? |
| Change-point detection (CUSUM, E. S. Page, 1950s; multi-hypothesis and GLR tests) | Candidate C is a multi-alternative CUSUM on model residuals with a window reset | Has residual-based CUSUM been used to re-identify actuator delay online? How do GLR or Bayesian online change-point methods compare on detection delay? |
| Delay-aware RL | The alternative to compensation: train with delays, augment the state with the action buffer, or model-based delay handling | What are the strongest delay-aware RL baselines for manipulation? Do they beat a frozen policy with compensation, and at what training cost? |
| Model-based robotics, learned dynamics with contact | F1's learned contact model used for short-horizon prediction | Prior work on learned push dynamics for state prediction under latency; how do analytic models compare? |
| Residual RL and policy adaptation without retraining | The project's original direction, which Phase D argued against for small residuals | Does residual RL succeed on FetchPush-like tasks under actuator delay when trained, not counterfactually probed? |
| Teleoperation and networked control | Latency compensation through predictive displays and state prediction | Are there direct analogues of observation-level prediction for autonomous learned policies? |
| Supervisory control and switching | G1 and C's pause/release logic | Hysteresis and dwell-time switching results relevant to compensator release |

**Novelty status: unconfirmed.** The individual components (least-squares lag identification,
Smith-style prediction, CUSUM, supervisory switching) are classical. Whether their combination
around a frozen learned policy, with a learned contact model and the reported evaluation, is
new must be established by a literature search. This report does not claim it.

## 14. Conclusion

In simulation, on FetchPush with three frozen SAC policies, held-out pre-registered tests
support three conclusions:
1. Online delay identification with observation-level compensation recovers most of the
   success that constant actuator delay removes, without retraining the policy.
2. Predicting the object's motion with a learned contact-aware model is what makes the
   compensation work well at longer delays.
3. For non-stationary delays, a sequential change detector that pauses compensation and resets
   identification improves on a fixed-window estimator. It beats the same mechanism driven by
   another episode's alarms.

The same evidence also shows a clear limit. When delays are short and frequent, the frozen
policy tolerates them better than any compensator tested, so a deployed system would need to
decide when *not* to compensate. Real-robot validity, generality across tasks and policies,
comparison with retraining-based approaches, and novelty relative to the literature remain open.
