# Related work: delay compensation, delay-aware RL and change detection

**Compiled:** 2026-11-17, by Claude, as a literature check for this project. It is **not a
systematic review**:
- each search was a small number of targeted queries;
- the full texts of several key papers were not accessible.

Every entry below says how much of the source was actually read. Nothing is inferred beyond
what that level of access supports. Companion documents are `docs/NOVELTY_AUDIT.md` and
`docs/RESEARCH_POSITIONING.md`.

## Verification levels

| level | meaning |
|---|---|
| **FT** | full text read (PDF or HTML), at least the method, setup and experiments |
| **ABS** | the official abstract page read (arXiv, PMLR, proceedings); methods and results beyond the abstract are unverified |
| **META** | bibliographic metadata and/or an abstract from an index (OpenAlex, dblp, publisher index, a citing review); content beyond that is unverified |
| **UNVERIFIED** | no primary source located; not used for any claim |

## 1. Comparison matrix

Columns:
- **Delay:** A = action/actuation; O = observation; C = computation/inference.
- **Variation:** F = fixed; S = stochastic per step; N = nonstationary (changes in regime).
- **Known?** Is the delay given to the method, or estimated online?
- **Policy:** R = trained or retrained with the delay; Z = a frozen pre-trained policy is reused.
- **Model:** is a learned dynamics or prediction model used?
- **CPD:** is change-point detection used?
- **Manip.:** manipulation or contact tested?
- **Robot:** physical-robot experiments?

"?" means the accessible source did not say. It is never filled in by inference.

| # | paper | level | delay | variation | known? | policy | model | CPD | manip. | robot |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Wang, Xing, Yang, Wang 2025, RA-L | META | O | N | **estimated online** | ? | per-distribution delayed dynamic models | **yes (BOCPD)** | ? | ? |
| 2 | Chen, Xu, Li, Zhao 2021, Neurocomputing | FT | A | F (several values) | known | R (MBRL/MPC) | yes | no | no | no |
| 3 | Wikman, Proutiere, Broman 2026, ICML | FT | A + O (interaction) | S, temporally correlated, incl. measured traces | unobservable at decision time | R | yes (ACDA) | no | no | no |
| 4 | Wafi 2026, arXiv (ext. of ACC paper) | FT | A (input) | F | fixed nominal; offline ID analysed only | **Z** (DDPG trained delay-free) | no (nominal plant model) | no | no | no |
| 5 | Peng, Andrychowicz, Zaremba, Abbeel 2018, ICRA | FT | timing (random Δt as a latency model) | S per step | not given to policy | R (randomised; LSTM) | no | no | **yes (puck pushing, Fetch)** | **yes (Fetch)** |
| 6 | Bouteiller/Ramstedt et al. 2021, ICLR (DCAC) | ABS | A + O | S | ? | R | ? | no | ? | ? |
| 7 | Ramstedt & Pal 2019, NeurIPS (RTAC) | ABS* | A (1-step) | F | known | R | no | no | ? | ? |
| 8 | Katsikopoulos & Engelbrecht 2003, IEEE TAC | META | A, O, cost | F and random | ? | (theory) | – | no | – | – |
| 9 | Walsh, Nouri, Li, Littman 2009, AAMAS journal | META | O, reward | F | ? | R (model-based simulation) | yes | no | ? | ? |
| 10 | Kim et al. 2023, NeurIPS (BPQL) | META | delayed feedback | ? | ? | R | ? | no | ? | ? |
| 11 | Wu et al. 2024, NeurIPS (VDPO) | META | O | ? | ? | R | ? | no | ? | ? |
| 12 | Wang, Han, Luo, Li 2024, ICLR | FT (partial) | O ("signal delay") | F and random ("unfixed") | ? | R | auxiliary state prediction | no | no | no (MuJoCo) |
| 13 | Liotet, Maran, Bisi, Restelli 2022, ICML (DIDA) | ABS | O / interaction | ? | ? | R (imitates an undelayed expert) | no | no | ? | ? |
| 14 | Karamzade, Kim, Kalsi, Fox 2024, RLJ/RLC | ABS | O | ? | ? | R | yes (world model, Dreamer-V3) | no | ? | ? |
| 15 | Sandha et al. 2021, CoRL 2020 (PMLR 155) | ABS | timing, sensing-to-actuation | S | delay as an observation | R | no | no | ? | yes (1/18-scale car) |
| 16 | Tan et al. 2018, arXiv/RSS (sim-to-real locomotion) | ABS | latency (simulated) | ? | ? | R | actuator model | no | no | yes (quadruped) |
| 17 | Yu, Tan, Liu, Turk 2017, RSS (UP-OSI) | ABS | – (dynamics parameters) | – | online system ID | R (universal policy) | ID network | no | throwing task | ? |
| 18 | Black, Galliker, Levine 2025, NeurIPS (RTC) | ABS | C (inference) | ? | ? | **Z (no retraining)** | no | no | **yes** | **yes (bimanual)** |
| 19 | Wang T. et al. 2026, arXiv (RAPAC-DP) | ABS | C + communication | multiple levels | delay-conditioned | Z + trained adapters | no explicit dynamics model | no | **yes** | **yes (Franka)** |
| 20 | Syed et al. 2026, CoRL (AHEAD) | ABS | observation-to-execution | ? | ? | **Z (frozen VLA) + latent world model** | **yes (latent forecasting)** | no | **yes (dynamic)** | **yes (xArm)** |
| 21 | Zhu et al. 2026, arXiv (ARLI) | ABS | C (inference) | ? | ? | R (RL fine-tuning) | no | no | yes | yes |
| 22 | Deng & Yang 2026, IFAC WC (accepted) | ABS | communication (teleoperation) | S | ? | residual RL + LSTM estimator | LSTM state estimator | no | ? | yes (Franka) |
| 23 | Smith 1957 (Smith predictor) | META | dead time | F | known model | classical | plant model | no | – | – |
| 24 | Normey-Rico & Camacho 2008, CEP (survey) | META | dead time | – | – | classical | – | – | – | – |
| 25 | Lai & Hsu 2010, IEEE TII | META | network | N | estimated online | classical (adaptive SP) | plant model | no | no | ? |
| 26 | Rajkumar, Chakraborty, Dey, Deb 2019, IJCAS | ABS | network | S | estimated online | classical PI + adaptive SP | plant model | no | no | yes (DC servo) |
| 27 | Page 1954; Basseville & Nikiforov 1993; Adams & MacKay 2007 | META / ABS | – | – | – | – | – | **CUSUM; theory; BOCPD** | – | – |
| 28 | Narendra & Balakrishnan 1997, IEEE TAC | META | – | – | – | multiple-model switching | multiple models | switching | – | – |
| 29 | Sha 2001, IEEE Software (Simplex) | META | – | – | – | switching between high-assurance and high-performance controllers | – | – | – | – |
| 30 | Bauza & Rodriguez 2017 (ICRA); Kloss, Schaal, Bohg (IJRR 2020) | ABS | – | – | – | – | **learned pushing models** | – | **yes** | Kloss: real data |
| – | "LagComp" (ResearchGate listing only) | **UNVERIFIED** | | | | | | | | |

\* For Ramstedt & Pal, venue and pages come from the NeurIPS proceedings listing; method
details come from the ar5iv HTML.

## 2. Entries

Each entry records: **what it establishes that this project does not**, and **what this
project's experiments establish that it does not**. Numbers from this project are cited with
their source document and seed status: dev = development seeds; HO = held-out.

### Directly related

**[1] Wang, Z., Xing, D., Yang, Y., Wang, P. (2025). Delayed Dynamic Model Scheduled
Reinforcement Learning With Time-Varying Delays for Robotic Control.** *IEEE Robotics and
Automation Letters* 10(3):2646–2653.
[doi:10.1109/LRA.2025.3536291](https://doi.org/10.1109/LRA.2025.3536291).
- **Access:** META. The full text is closed-access (OpenAlex: `is_oa: false`). The IEEE page
  rendered no content.
- **What the abstract says:**
  - observation delays that vary non-stationarily;
  - multiple truncated delay distributions, each with its own delayed dynamic model;
  - Bayesian online change-point detection to sense delay shifts in real time and select the
    matching model;
  - described by the authors as the first effective approach to non-stationary delay in RL.
- **Unverified:** environments, robot experiments, baselines, and whether the policy is
  retrained per model.
- **It establishes, beyond us:** RL-integrated change-point detection for non-stationary
  delays, published in a robotics venue.
- **We establish, beyond the abstract:**
  - change detection for *actuator* delay around a *frozen* policy;
  - a residual-CUSUM detector with pause-and-reset;
  - held-out evidence on contact manipulation (G4 P1 +3.5 [+2.4, +4.6] HO);
  - a cross-episode yoked sham isolating episode-specific detection (G4 P3 +3.2 HO).
- **This is the closest prior work to our Candidate C.**

**[2] Chen, B., Xu, M., Li, L., Zhao, D. (2021). Delay-Aware Model-Based Reinforcement
Learning for Continuous Control.** *Neurocomputing* 450:119–128.
[arXiv:2005.05440](https://arxiv.org/abs/2005.05440).
- **Access:** FT.
- **Method:**
  - a delay-aware MDP with a known, constant multi-step action delay;
  - the delay part of the dynamics is known exactly, and only the delay-free dynamics
    $p(s_{t+1}|s_t,a_t)$ is learned;
  - planning (MPC) runs through the known action buffer;
  - the learned model transfers across delay values.
- **Setup:** Pendulum, CartPole, Walker2d and Ant, against off-policy model-free baselines; no
  robot.
- **It establishes, beyond us:** learning only delay-free dynamics and composing it with a
  known action buffer is sample-efficient and transfers across delays. This is the same
  decomposition our Phase E/F1 predictor uses.
- **We establish, beyond it:**
  - the delay is *estimated online* rather than given;
  - the controller is a *frozen* model-free policy whose input is shifted, rather than a
    planner;
  - contact manipulation, with object-motion prediction carrying the gain (HO H2 +9.3, H3
    +8.7);
  - non-stationary delay.

**[3] Wikman, J., Proutiere, A., Broman, D. (2026). Adaptive Reinforcement Learning for
Unobservable Random Delays.** *Proc. ICML 2026*, PMLR 306:134946–135011.
[proceedings.mlr.press/v306/wikman26a](https://proceedings.mlr.press/v306/wikman26a.html).
- **Access:** FT.
- **Method:**
  - an "interaction layer": the agent emits a matrix of candidate future actions, one row per
    possible arrival time;
  - ACDA, a model-based actor-critic that adapts to the delay;
  - delays vary stochastically with temporal correlation (Gilbert–Elliott and M/M/1 processes,
    plus measured WiFi TCP delays), and are unobservable at decision time.
- **Setup:**
  - Ant, Humanoid, HalfCheetah, Hopper and Walker2d;
  - baselines SAC, SAC + constant-delay augmentation, Dreamer, BPQL, VDPO and DCAC;
  - policies trained under the delay; no robot, no manipulation.
- **It establishes, beyond us:**
  - handling of *stochastic, unobservable* delays, with measured network traces;
  - comparison with the main delay-RL baselines;
  - evidence that adaptivity beats worst-case constant-delay augmentation.
- **We establish, beyond it:** a frozen policy without retraining, actuator transport delay,
  contact manipulation, and pre-registered held-out evaluation.

**[4] Wafi, M. K. (2026). Deep Reinforcement Learning with Smith-Predictor Delay
Compensation.** arXiv:2611.00668 (extended version of an ACC paper).
[arxiv.org/abs/2611.00668](https://arxiv.org/abs/2611.00668).
- **Access:** FT (arXiv HTML).
- **Method:**
  - a DDPG actor trained delay-free, *frozen* ("actor parameters θ are held fixed after
    training");
  - deployed with a Smith predictor using a *fixed nominal* delay and a nominal plant model;
  - "No online delay estimation is assumed." Offline output-error identification is analysed
    but not used in the experiments.
- **Setup:**
  - first-order stable and unstable plants with valve stiction, constant delays of 5–300 s;
  - baselines PID, LQR and standalone DDPG;
  - robustness theory (small-gain, ν-gap, Razumikhin);
  - tracking results qualitative.
- **It establishes, beyond us:** the frozen-RL-policy-plus-Smith-predictor architecture itself,
  with stability analysis under plant and delay mismatch.
- **We establish, beyond it:**
  - online lag identification;
  - a learned contact-aware predictor;
  - a multi-dimensional manipulation task;
  - time-varying delays and their failure mode;
  - quantitative pre-registered held-out evaluation.

  Our Phase E/E2 fixed-delay sham (d = 2) is effectively a fixed-nominal-delay Smith
  predictor. It helped only when 2 was close to the true delay, and harmed undisturbed and
  noise-only episodes (−16 and −19 points, E2, dev).
- **This is the closest prior work to our Phase E architecture.**

**[5] Peng, X. B., Andrychowicz, M., Zaremba, W., Abbeel, P. (2018). Sim-to-Real Transfer of
Robotic Control with Dynamics Randomization.** *ICRA 2018*.
[arXiv:1710.06537](https://arxiv.org/abs/1710.06537),
[doi:10.1109/ICRA.2018.8460528](https://doi.org/10.1109/ICRA.2018.8460528).
- **Access:** FT.
- **Setup:**
  - Fetch arm puck pushing;
  - 95 randomised parameters, including the timestep between actions, $\Delta t \sim \Delta t_0 + \mathrm{Exp}(\lambda)$
    with $\Delta t_0$ = 0.04 s, described as "a simple model of the latency";
  - an LSTM policy trained with HER, deployed on a real Fetch.
- **Result:** disabling timestep randomisation dropped real-robot success from 0.89 to 0.29.
- **It establishes, beyond us:** that *retraining with latency-like randomisation* works for
  Fetch pushing on hardware.
- **We establish, beyond it:**
  - a frozen policy, with no retraining;
  - explicit transport delay of 1–4 steps, including non-stationary schedules;
  - online identification.
- **Note:** randomised timing is not the same disturbance as transport delay. This is the
  most direct *retraining* comparison for our task family.

### Delay-aware and delay-randomised RL (policy trained with the delay)

- **[6] Ramstedt, S., Bouteiller, Y., Beltrame, G., Pal, C., Binas, J. (2021). Reinforcement
  Learning with Random Delays.** *ICLR 2021*.
  [arXiv:2010.02966](https://arxiv.org/abs/2010.02966).
  - **Access:** ABS. The author order is as listed on arXiv; other indexes list Bouteiller
    first.
  - **Content:** DCAC, for random action and observation delays, on delay-augmented MuJoCo.
- **[7] Ramstedt, S., Pal, C. (2019). Real-Time Reinforcement Learning.** *NeurIPS 2019*.
  [papers.nips.cc/paper/8571](https://papers.nips.cc/paper/8571-real-time-reinforcement-learning).
  - **Content:** RTAC. It shows the real-time setting is equivalent to a 1-step constant
    action delay.
- **[8] Katsikopoulos, K. V., Engelbrecht, S. E. (2003). Markov decision processes with delays
  and asynchronous cost collection.** *IEEE TAC* 48(4):568–574.
  [doi:10.1109/TAC.2003.809799](https://doi.org/10.1109/TAC.2003.809799).
  - **Access:** META.
  - **Content:** reduces delayed MDPs to undelayed MDPs on an augmented state.
- **[9] Walsh, T. J., Nouri, A., Li, L., Littman, M. L. (2009). Learning and planning in
  environments with delayed feedback.** *Autonomous Agents and Multi-Agent Systems* 18:83–105.
  [doi:10.1007/s10458-008-9056-7](https://doi.org/10.1007/s10458-008-9056-7).
  - **Access:** META.
  - **Content:** constant delays; model-based simulation to avoid state augmentation.
- **[10] Kim, J., Kim, H., Kang, J., Baek, J., Han, S. (2023). Belief Projection-Based
  Reinforcement Learning for Environments with Delayed Feedback (BPQL).** *NeurIPS 2023*.
  [neurips.cc/virtual/2023/poster/70249](https://neurips.cc/virtual/2023/poster/70249).
  - **Access:** META.
- **[11] Wu, Q., Zhan, S. S., et al. (2024). Variational Delayed Policy Optimization (VDPO).**
  *NeurIPS 2024*.
  [arXiv:2405.14226](https://arxiv.org/abs/2405.14226).
  - **Access:** META.
- **[12] Wang, W., Han, D., Luo, X., Li, D. (2024). Addressing Signal Delay in Deep
  Reinforcement Learning.** *ICLR 2024*.
  [proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/45cc967d7899616f51993b7b363d35b5-Paper-Conference.pdf).
  - **Access:** FT, partial.
  - **Content:**
    - a delayed-observation MDP formulation;
    - fixed and "unfixed" (random) delays;
    - delay-reconciled critic training, state augmentation and auxiliary state prediction;
    - MuJoCo locomotion.
- **[13] Liotet, P., Maran, D., Bisi, L., Restelli, M. (2022). Delayed Reinforcement Learning by
  Imitation (DIDA).** *ICML 2022*, PMLR 162:13528–13556.
  [proceedings.mlr.press/v162/liotet22a](https://proceedings.mlr.press/v162/liotet22a.html).
  - **Access:** ABS.
  - **Content:** trains a delayed policy by imitating an undelayed expert. It reuses an
    undelayed policy, as we do, but as a teacher for retraining.
- **[14] Karamzade, A., Kim, K., Kalsi, M., Fox, R. (2024). Reinforcement Learning from Delayed
  Observations via World Models.** *Reinforcement Learning Journal* 5 (RLC 2024).
  [arXiv:2403.12309](https://arxiv.org/abs/2403.12309).
  - **Access:** ABS.
  - **Content:** world models reduce delayed POMDPs to delayed MDPs; Dreamer-V3.
- **[15] Sandha, S. S., Garcia, L., Balaji, B., Anwar, F., Srivastava, M. (2021). Sim2Real
  Transfer for Deep RL with Stochastic State Transition Delays.** *CoRL 2020*, PMLR
  155:1066–1083.
  [proceedings.mlr.press/v155/sandha21a](https://proceedings.mlr.press/v155/sandha21a.html).
  - **Access:** ABS.
  - **Content:** domain randomisation of timing was reported as insufficient. TSRL feeds the
    delay to the policy; tested on a real 1/18-scale car.
- **[16] Tan, J., et al. (2018). Sim-to-Real: Learning Agile Locomotion for Quadruped Robots.**
  [arXiv:1804.10332](https://arxiv.org/abs/1804.10332).
  - **Access:** ABS.
  - **Content:** simulated latency and an actuator model during training.
- **[17] Yu, W., Tan, J., Liu, C. K., Turk, G. (2017). Preparing for the Unknown: Learning a
  Universal Policy with Online System Identification.** *RSS 2017*.
  [arXiv:1702.02453](https://arxiv.org/abs/1702.02453).
  - **Access:** ABS.
  - **Content:** online identification feeding a policy *trained* over a parameter range. It is
    the closest RL analogue to our "identify online, then condition"; their policy is
    retrained, ours is frozen.
- **Survey: Alves Neto, A. (2026). Reinforcement Learning for Control Systems with Time Delays:
  A Comprehensive Survey.** [arXiv:2602.00399](https://arxiv.org/abs/2602.00399).
  - **Access:** ABS.
  - **Content:** five families: augmentation, recurrent, predictor/model-aware, robust or
    randomised, and safe RL. The abstract does not mention delay estimation, change detection or
    frozen policies. The full text was not read, so whether it covers them is unverified.

### Frozen pre-trained robot policies under latency (2025–2026)

- **[18] Black, K., Galliker, M. Y., Levine, S. (2025). Real-Time Execution of Action Chunking
  Flow Policies (RTC).** *NeurIPS 2025*.
  [arXiv:2506.07339](https://arxiv.org/abs/2506.07339).
  - **Access:** ABS.
  - **Content:** inference-time handling of *inference* latency for diffusion and flow VLAs
    "with no re-training"; real bimanual manipulation.
- **[19] Wang, T., Wang, W., Wang, J., Ye, Z., Wang, Q., Huang, W., Xu, B. (2026). RAPAC-DP:
  Response-Aligned Pending-Action Compensation for Diffusion Policies under Delayed
  Execution.** [arXiv:2608.15924](https://arxiv.org/abs/2608.15924).
  - **Access:** ABS.
  - **Content:**
    - a frozen base policy, with a trained pending-action encoder and LoRA adapters;
    - communication and inference latency;
    - Franka, 450 trials; high-delay success rose from 28.9% (RTC) to 60.0%.
- **[20] Syed, S. N., Jakobsson, A., Hao, H., Ichnowski, J. (2026). Intercepting the Future:
  Latent-Space Predictive World Model for Dynamic VLA Manipulation (AHEAD).** *CoRL 2026*.
  [arXiv:2606.02486](https://arxiv.org/abs/2606.02486).
  - **Access:** ABS.
  - **Content:**
    - a *predict-then-act wrapper around a frozen policy*: a latent world model forecasts the
      future and the frozen OpenVLA decoder acts on the forecast;
    - observation-to-execution latency with moving objects;
    - real xArm 7.
  - **Relevance:** conceptually the closest robotics analogue to our "frozen policy evaluated
    at a predicted observation".
- **[21] Zhu, B., et al. (2026). Learning to Act While Waiting: RL Finetuning of Generalist
  Robot Policies Under Inference Latency (ARLI).**
  [arXiv:2608.23831](https://arxiv.org/abs/2608.23831).
  - **Access:** ABS.
  - **Content:** latency-aware RL fine-tuning, so the policy is retrained; real manipulation.
- **[22] Deng, K., Yang, Z. (2026). Residual Reinforcement Learning for Robot Teleoperation
  under Stochastic Delays.** IFAC World Congress 2026 (accepted, per arXiv).
  [arXiv:2605.15480](https://arxiv.org/abs/2605.15480).
  - **Access:** ABS.
  - **Content:** an LSTM state estimator plus a residual RL torque policy; Franka.

### Dead-time compensation (classical control)

- **[23] Smith, O. J. M. (1957). Closer control of loops with dead time.** *Chemical
  Engineering Progress* 53:217–219.
  - **Access:** META. Secondary sources only (encyclopedic and trade references). The original
    was not accessed, and the volume and pages are as commonly cited.
- **[24] Normey-Rico, J. E., Camacho, E. F. (2008). Dead-time compensators: A survey.**
  *Control Engineering Practice* 16:407–428.
  [doi:10.1016/j.conengprac.2007.05.006](https://doi.org/10.1016/j.conengprac.2007.05.006).
  - **Access:** META, via the reference list of Normey-Rico et al. (2022), *Front. Control Eng.*
    [doi:10.3389/fcteg.2022.953768](https://doi.org/10.3389/fcteg.2022.953768).
- **[25] Lai, C.-L., Hsu, P.-L. (2010). Design the Remote Control System With the Time-Delay
  Estimator and the Adaptive Smith Predictor.** *IEEE Trans. Industrial Informatics*.
  [doi:10.1109/TII.2009.2037917](https://doi.org/10.1109/TII.2009.2037917).
  - **Access:** META. The repository PDF did not render, and the volume and pages were not
    verified.
  - **Content, per the index:** online delay estimation from round-trip times, feeding an
    adaptive Smith predictor.
- **[26] Rajkumar, S. M., Chakraborty, S., Dey, R., Deb, D. (2019). Online Delay Estimation and
  Adaptive Compensation in Wireless Networked System: An Embedded Control Design.** *Int. J.
  Control, Automation and Systems*.
  [arXiv:1912.00343](https://arxiv.org/abs/1912.00343).
  - **Access:** ABS.
  - **Content:** an empirical delay estimator plus an adaptive digital Smith predictor; DC servo
    hardware.
- **Björklund, S., Ljung, L. (2003). A review of time-delay estimation techniques.** *Proc.
  42nd IEEE CDC*, pp. 2502–2507.
  [doi:10.1109/CDC.2003.1272997](https://doi.org/10.1109/CDC.2003.1272997).
  - **Access:** META.

### Change detection and switching

- **[27a] Page, E. S. (1954). Continuous Inspection Schemes.** *Biometrika* 41(1–2):100–115.
  [doi:10.1093/biomet/41.1-2.100](https://doi.org/10.1093/biomet/41.1-2.100).
  - **Access:** META. This is the origin of CUSUM.
- **[27b] Basseville, M., Nikiforov, I. V. (1993). Detection of Abrupt Changes: Theory and
  Application.** Prentice Hall.
  - **Access:** META. The authors host a free PDF at
    [people.irisa.fr/Michele.Basseville/kniga](https://people.irisa.fr/Michele.Basseville/kniga/kniga.html).
- **[27c] Adams, R. P., MacKay, D. J. C. (2007). Bayesian Online Changepoint Detection.**
  [arXiv:0710.3742](https://arxiv.org/abs/0710.3742).
  - **Access:** ABS. This is the method family Wang et al. (2025) report using.
- **[28] Narendra, K. S., Balakrishnan, J. (1997). Adaptive control using multiple models.**
  *IEEE TAC* 42(2):171–187.
  [doi:10.1109/9.554398](https://doi.org/10.1109/9.554398).
  - **Access:** META.
  - **Content:** switching among fixed and adaptive models. It is the classical analogue of
    G1's supervisors and of C's lag hypotheses.
- **[29] Sha, L. (2001). Using Simplicity to Control Complexity.** *IEEE Software*
  18(4):20–28.
  [doi:10.1109/MS.2001.936213](https://doi.org/10.1109/MS.2001.936213).
  - **Access:** META.
  - **Content:** the Simplex architecture, switching between a high-performance and a verified
    controller. It is the classical frame for "fall back to the nominal controller", here
    frozen SAC.

### Learned contact and pushing prediction

- **[30a] Bauza, M., Rodriguez, A. (2017). A probabilistic data-driven model for planar
  pushing.** *ICRA 2017*. [arXiv:1704.03033](https://arxiv.org/abs/1704.03033).
  - **Access:** ABS.
- **[30b] Kloss, A., Schaal, S., Bohg, J. (2020). Combining learned and analytical models for
  predicting action effects from sensory data.** *IJRR*.
  [arXiv:1710.04102](https://arxiv.org/abs/1710.04102).
  - **Access:** ABS.
  - **Content:** hybrid learned and analytical pushing models; a large real dataset.
- **Silver, T., Allen, K., Tenenbaum, J., Kaelbling, L. (2018). Residual Policy Learning.**
  [arXiv:1812.06298](https://arxiv.org/abs/1812.06298).
  - **Access:** ABS.
  - **Content:** residuals on imperfect controllers for MuJoCo manipulation. Relevant to the
    project's abandoned residual-RL direction (Phase D).

### Unverified

- **"LagComp: Compensating Observation-to-Action Latency Through Latent Feature Forecasting".**
  It appeared only as a ResearchGate listing in one search. No arXiv, venue or authors were
  found, so it is **not used** for any claim.

## 3. Gaps in this review

These need a follow-up search with library access:
1. **Full text of Wang et al. (2025) RA-L:** environments, robot experiments, whether the
   policy is retrained, the detector details and the baselines. *This is the most important
   gap.*
2. **Full text of Wafi's ACC version,** for its exact conference citation.
3. **Adaptive Smith predictors with change detection,** beyond the two delay-estimation papers
   found, and the delay-mismatch robustness literature, i.e. how much a Smith predictor
   degrades when the delay model is stale.
4. **Predictive displays and latency compensation in teleoperation** (not searched).
5. **Learned push or contact dynamics used for latency compensation** specifically (no direct
   hit found).
6. **Whether the 2026 survey [arXiv:2602.00399] discusses frozen-policy compensation or change
   detection** (its full text was not read).
7. **Yoked or sham controls in robot-learning evaluations** (not searched).

## 4. Bibliography (BibTeX)

Entries contain only fields verified at the level stated. Fields not verified are omitted.

```bibtex
@article{wang2025delayed,
  title   = {Delayed Dynamic Model Scheduled Reinforcement Learning With Time-Varying Delays for Robotic Control},
  author  = {Wang, Zechang and Xing, Dengpeng and Yang, Yiming and Wang, Peng},
  journal = {IEEE Robotics and Automation Letters},
  volume  = {10}, number = {3}, pages = {2646--2653}, year = {2025},
  doi     = {10.1109/LRA.2025.3536291}
}
@article{chen2021delayaware,
  title   = {Delay-Aware Model-Based Reinforcement Learning for Continuous Control},
  author  = {Chen, Baiming and Xu, Mengdi and Li, Liang and Zhao, Ding},
  journal = {Neurocomputing}, volume = {450}, pages = {119--128}, year = {2021},
  doi     = {10.1016/j.neucom.2021.04.015}, eprint = {2005.05440}, archivePrefix = {arXiv}
}
@inproceedings{wikman2026adaptive,
  title     = {Adaptive Reinforcement Learning for Unobservable Random Delays},
  author    = {Wikman, John and Proutiere, Alexandre and Broman, David},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  series    = {PMLR}, volume = {306}, pages = {134946--135011}, year = {2026},
  url       = {https://proceedings.mlr.press/v306/wikman26a.html}
}
@misc{wafi2026smith,
  title  = {Deep Reinforcement Learning with Smith-Predictor Delay Compensation},
  author = {Wafi, Moh Kamalul}, year = {2026},
  eprint = {2611.00668}, archivePrefix = {arXiv}, primaryClass = {eess.SY},
  note   = {Extended version of an IEEE ACC paper}
}
@inproceedings{peng2018simtoreal,
  title     = {Sim-to-Real Transfer of Robotic Control with Dynamics Randomization},
  author    = {Peng, Xue Bin and Andrychowicz, Marcin and Zaremba, Wojciech and Abbeel, Pieter},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)}, year = {2018},
  doi       = {10.1109/ICRA.2018.8460528}, eprint = {1710.06537}, archivePrefix = {arXiv}
}
@inproceedings{ramstedt2021random,
  title     = {Reinforcement Learning with Random Delays},
  author    = {Ramstedt, Simon and Bouteiller, Yann and Beltrame, Giovanni and Pal, Christopher and Binas, Jonathan},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2021},
  eprint    = {2010.02966}, archivePrefix = {arXiv}
}
@inproceedings{ramstedt2019realtime,
  title     = {Real-Time Reinforcement Learning},
  author    = {Ramstedt, Simon and Pal, Christopher},
  booktitle = {Advances in Neural Information Processing Systems 32}, year = {2019},
  eprint    = {1911.04448}, archivePrefix = {arXiv}
}
@article{katsikopoulos2003mdp,
  title   = {Markov decision processes with delays and asynchronous cost collection},
  author  = {Katsikopoulos, Konstantinos V. and Engelbrecht, Sascha E.},
  journal = {IEEE Transactions on Automatic Control}, volume = {48}, number = {4},
  pages   = {568--574}, year = {2003}, doi = {10.1109/TAC.2003.809799}
}
@article{walsh2009delayed,
  title   = {Learning and planning in environments with delayed feedback},
  author  = {Walsh, Thomas J. and Nouri, Ali and Li, Lihong and Littman, Michael L.},
  journal = {Autonomous Agents and Multi-Agent Systems}, volume = {18}, pages = {83--105},
  year    = {2009}, doi = {10.1007/s10458-008-9056-7}
}
@inproceedings{kim2023bpql,
  title     = {Belief Projection-Based Reinforcement Learning for Environments with Delayed Feedback},
  author    = {Kim, Jangwon and Kim, Hangyeol and Kang, Jiwook and Baek, Jongchan and Han, Soohee},
  booktitle = {Advances in Neural Information Processing Systems 36}, year = {2023}
}
@inproceedings{wu2024vdpo,
  title     = {Variational Delayed Policy Optimization},
  author    = {Wu, Qingyuan and Zhan, Simon Sinong and others},
  booktitle = {Advances in Neural Information Processing Systems 37}, year = {2024},
  eprint    = {2405.14226}, archivePrefix = {arXiv}
}
@inproceedings{wang2024signal,
  title     = {Addressing Signal Delay in Deep Reinforcement Learning},
  author    = {Wang, Wei and Han, Dongqi and Luo, Xufang and Li, Dongsheng},
  booktitle = {International Conference on Learning Representations (ICLR)}, year = {2024}
}
@inproceedings{liotet2022dida,
  title     = {Delayed Reinforcement Learning by Imitation},
  author    = {Liotet, Pierre and Maran, Davide and Bisi, Lorenzo and Restelli, Marcello},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  series    = {PMLR}, volume = {162}, pages = {13528--13556}, year = {2022}
}
@article{karamzade2024world,
  title   = {Reinforcement Learning from Delayed Observations via World Models},
  author  = {Karamzade, Armin and Kim, Kyungmin and Kalsi, Montek and Fox, Roy},
  journal = {Reinforcement Learning Journal}, volume = {5}, year = {2024},
  eprint  = {2403.12309}, archivePrefix = {arXiv}
}
@inproceedings{sandha2021tsrl,
  title     = {Sim2Real Transfer for Deep Reinforcement Learning with Stochastic State Transition Delays},
  author    = {Sandha, Sandeep Singh and Garcia, Luis and Balaji, Bharathan and Anwar, Fatima and Srivastava, Mani},
  booktitle = {Proceedings of the 2020 Conference on Robot Learning}, series = {PMLR},
  volume    = {155}, pages = {1066--1083}, year = {2021}
}
@misc{tan2018simtoreal,
  title  = {Sim-to-Real: Learning Agile Locomotion For Quadruped Robots},
  author = {Tan, Jie and Zhang, Tingnan and Coumans, Erwin and Iscen, Atil and Bai, Yunfei and Hafner, Danijar and Bohez, Steven and Vanhoucke, Vincent},
  year   = {2018}, eprint = {1804.10332}, archivePrefix = {arXiv}
}
@inproceedings{yu2017uposi,
  title     = {Preparing for the Unknown: Learning a Universal Policy with Online System Identification},
  author    = {Yu, Wenhao and Tan, Jie and Liu, C. Karen and Turk, Greg},
  booktitle = {Robotics: Science and Systems (RSS)}, year = {2017},
  eprint    = {1702.02453}, archivePrefix = {arXiv}
}
@misc{alvesneto2026survey,
  title  = {Reinforcement Learning for Control Systems with Time Delays: A Comprehensive Survey},
  author = {Alves Neto, Armando}, year = {2026}, eprint = {2602.00399}, archivePrefix = {arXiv}
}
@inproceedings{black2025rtc,
  title     = {Real-Time Execution of Action Chunking Flow Policies},
  author    = {Black, Kevin and Galliker, Manuel Y. and Levine, Sergey},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)}, year = {2025},
  eprint    = {2506.07339}, archivePrefix = {arXiv}
}
@misc{wang2026rapacdp,
  title  = {{RAPAC-DP}: Response-Aligned Pending-Action Compensation for Diffusion Policies under Delayed Execution},
  author = {Wang, Tao and Wang, Wei and Wang, Jianhui and Ye, ZhengYang and Wang, Qi and Huang, Weidi and Xu, Bing},
  year   = {2026}, eprint = {2608.15924}, archivePrefix = {arXiv}
}
@inproceedings{syed2026ahead,
  title     = {Intercepting the Future: Latent-Space Predictive World Model for Dynamic {VLA} Manipulation},
  author    = {Syed, Shahram Najam and Jakobsson, Arthur and Hao, Haoran and Ichnowski, Jeffrey},
  booktitle = {Conference on Robot Learning (CoRL)}, year = {2026},
  eprint    = {2606.02486}, archivePrefix = {arXiv}
}
@misc{zhu2026arli,
  title  = {Learning to Act While Waiting: {RL} Finetuning of Generalist Robot Policies Under Inference Latency},
  author = {Zhu, Brian and others}, year = {2026}, eprint = {2608.23831}, archivePrefix = {arXiv}
}
@misc{deng2026residual,
  title  = {Residual Reinforcement Learning for Robot Teleoperation under Stochastic Delays},
  author = {Deng, Kaize and Yang, Zewen}, year = {2026}, eprint = {2605.15480}, archivePrefix = {arXiv},
  note   = {Accepted at the 23rd IFAC World Congress (per arXiv)}
}
@article{smith1957closer,
  title   = {Closer control of loops with dead time},
  author  = {Smith, O. J. M.},
  journal = {Chemical Engineering Progress}, volume = {53}, pages = {217--219}, year = {1957},
  note    = {Bibliographic details from secondary sources; original not accessed}
}
@article{normeyrico2008survey,
  title   = {Dead-time compensators: A survey},
  author  = {Normey-Rico, Julio E. and Camacho, Eduardo F.},
  journal = {Control Engineering Practice}, volume = {16}, pages = {407--428}, year = {2008},
  doi     = {10.1016/j.conengprac.2007.05.006}
}
@article{lai2010remote,
  title   = {Design the Remote Control System With the Time-Delay Estimator and the Adaptive Smith Predictor},
  author  = {Lai, Chih-Lyang and Hsu, Pau-Lo},
  journal = {IEEE Transactions on Industrial Informatics}, year = {2010},
  doi     = {10.1109/TII.2009.2037917}
}
@article{rajkumar2019online,
  title   = {Online Delay Estimation and Adaptive Compensation in Wireless Networked System: An Embedded Control Design},
  author  = {Rajkumar, Santosh Mohan and Chakraborty, Sayan and Dey, Rajeeb and Deb, Dipankar},
  journal = {International Journal of Control, Automation and Systems}, year = {2019},
  eprint  = {1912.00343}, archivePrefix = {arXiv}
}
@inproceedings{bjorklund2003review,
  title     = {A review of time-delay estimation techniques},
  author    = {Bj{\"o}rklund, Svante and Ljung, Lennart},
  booktitle = {42nd IEEE Conference on Decision and Control}, pages = {2502--2507}, year = {2003},
  doi       = {10.1109/CDC.2003.1272997}
}
@article{page1954cusum,
  title   = {Continuous Inspection Schemes},
  author  = {Page, E. S.},
  journal = {Biometrika}, volume = {41}, number = {1-2}, pages = {100--115}, year = {1954},
  doi     = {10.1093/biomet/41.1-2.100}
}
@book{basseville1993detection,
  title     = {Detection of Abrupt Changes: Theory and Application},
  author    = {Basseville, Mich{\`e}le and Nikiforov, Igor V.},
  publisher = {Prentice Hall}, year = {1993}
}
@misc{adams2007bocpd,
  title  = {Bayesian Online Changepoint Detection},
  author = {Adams, Ryan Prescott and MacKay, David J. C.},
  year   = {2007}, eprint = {0710.3742}, archivePrefix = {arXiv}
}
@article{narendra1997multiple,
  title   = {Adaptive control using multiple models},
  author  = {Narendra, Kumpati S. and Balakrishnan, Jeyendran},
  journal = {IEEE Transactions on Automatic Control}, volume = {42}, number = {2},
  pages   = {171--187}, year = {1997}, doi = {10.1109/9.554398}
}
@article{sha2001simplicity,
  title   = {Using Simplicity to Control Complexity},
  author  = {Sha, Lui},
  journal = {IEEE Software}, volume = {18}, number = {4}, pages = {20--28}, year = {2001},
  doi     = {10.1109/MS.2001.936213}
}
@inproceedings{bauza2017pushing,
  title     = {A probabilistic data-driven model for planar pushing},
  author    = {Bauza, Maria and Rodriguez, Alberto},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)}, year = {2017},
  eprint    = {1704.03033}, archivePrefix = {arXiv}
}
@article{kloss2020combining,
  title   = {Combining learned and analytical models for predicting action effects from sensory data},
  author  = {Kloss, Alina and Schaal, Stefan and Bohg, Jeannette},
  journal = {The International Journal of Robotics Research}, year = {2020},
  eprint  = {1710.04102}, archivePrefix = {arXiv}
}
@misc{silver2018residual,
  title  = {Residual Policy Learning},
  author = {Silver, Tom and Allen, Kelsey and Tenenbaum, Josh and Kaelbling, Leslie},
  year   = {2018}, eprint = {1812.06298}, archivePrefix = {arXiv}
}
```
