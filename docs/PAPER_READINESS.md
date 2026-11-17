# Paper-readiness assessment

Companion to `docs/RESEARCH_TECHNICAL_REPORT.md` and `docs/RESEARCH_AUDIT.md`. This is a
deliberately conservative assessment. It is my judgement, not a reviewer's verdict, and no
literature search has been done.

## 1. Strongest defensible thesis

> *For a frozen learned manipulation policy in simulation, actuator command delay can be
> largely compensated at the policy's input, without retraining. An online delay estimate
> drives a learned, contact-aware prediction of the observation the delayed command will meet.
> When the delay is non-stationary, compensating with a stale estimate is the main failure
> mode, and can be worse than not compensating. A residual-based change detector that pauses
> compensation and restarts identification reduces this harm, but does not eliminate it for
> short, frequent delays.*

**Why this thesis:**
- Every clause rests on pre-registered held-out evidence:
  - E/F1 H1–H3: +44.9, +9.3, +8.7;
  - G4 P1–P3: +3.5, +0.5, +3.2;
  - G4's SAC comparison: −2.2 on random short-lived and on-off delays.
- It includes the negative finding, which is the most decision-relevant result and the most
  likely to interest a robotics audience.
- It claims no novelty, no hardware validity and no superiority to retraining.

**Weaker theses to avoid:**
- "Episode-specific detection is the mechanism." The G3 decomposition is development-only and
  sensitive to replay divergence.
- "Better than delay-aware RL." That was never compared.
- "Generalises across tasks." Only FetchPush was tested.

## 2. Assessment by criterion

| criterion | current state | rating |
|---|---|---|
| **Algorithmic novelty** | Components are classical (least-squares lag ID, Smith-style prediction, CUSUM, dwell-time switching). The combination around a frozen learned policy with a learned contact model may be new, but that is unverified. | **weak to unknown** |
| **Baseline strength** | Frozen SAC, kinematic predictor, gripper-only learned predictor, oracles, shams. No retrained or delay-randomised policy, no delay-aware RL, no MPC or analytic object model. | **weak** for a major venue |
| **Breadth** | 1 task, 1 simulator, 1 policy family (3 seeds), mocap position control, synthetic transport delay | **narrow** |
| **Statistical rigour** | Pre-registration with SHA-pinned plans; paired design; cluster bootstrap; IUT, Holm and gatekeeping; two held-out confirmations; write-once data; disclosed negative and failed designs | **strong**, the project's best asset |
| **Real-world transfer** | none | **absent** |
| **Compute and latency** | Per-step medians of 71–116 µs on a laptop core (descriptive, concurrent processes); no embedded or real-time measurement; model inference not profiled separately | **minimal** |
| **Reproducibility** | Code, configs, seeds, hashes and analysis are complete in git, and a held-out analysis re-derives exactly. Artifacts (checkpoints, model, raw data) are not archived, and G1–G4 are on unmerged branches. | **good internally, not yet independent** |
| **Limitations** | Thoroughly documented per phase, and consolidated in the audit | **strong** |

## 3. Readiness by venue

### Workshop paper (e.g. a robot-learning or robustness workshop): **ready after modest work**

- **What it needs:**
  - artifacts archived (audit R1);
  - G1–G4 merged or tagged;
  - the README updated;
  - a short related-work section after an actual literature search;
  - the thesis above framed as an empirical study, with the negative result central.
- **Why it is plausible:** workshops value careful negative and empirical findings, and the
  statistical design is unusually rigorous for the scale.
- **Main risk:** a reviewer who knows the delay-compensation literature may consider the
  method standard. Frame the contribution as the evaluation and the failure mode, not the
  algorithm.

### Undergraduate research publication or symposium: **ready after the same modest work**

- The pre-registration, held-out discipline and honest reporting of failed designs are
  strengths that such venues reward.
- A clear exposition of the Smith-predictor idea, the change detector and the
  stale-compensation finding would be an appropriate scope.

### Major robotics conference (CoRL, RSS, ICRA): **not ready**

The current evidence would most likely be judged:
- **too narrow:** one simulated task;
- **missing the key baseline:** a policy trained with delay randomisation or delay-aware RL;
- **lacking hardware validation:** CoRL and RSS in particular expect physical-robot results
  for a control-robustness claim;
- **of unclear novelty.**

The statistical rigour would be noticed, but it does not offset these gaps.

## 4. Evidence needed for a credible CoRL, RSS or ICRA submission

Listed in order of value. Each would need its own pre-registered plan and approval.

1. **Real-robot validation.** At least one physical arm (or a high-fidelity torque-controlled
   simulation as an intermediate step), with injected and natural command latency. The key
   tests:
   - Does least-squares lag identification survive real plant dynamics, jitter and sensing
     noise?
   - Does the learned contact model transfer, or does it need fine-tuning?
2. **Strong baselines:**
   - SAC (or the same algorithm) retrained with delay randomisation, plus an action-buffer
     state augmentation, matched for training budget;
   - an analytic or physics-based object predictor;
   - optionally, MPC with the same delay estimate.

   The question a reviewer will ask is "why not just retrain?". The answer needs numbers:
   performance, data and compute cost, and robustness to unseen delays.
3. **Breadth:**
   - at least 2–3 more tasks (e.g. FetchPickAndPlace, FetchSlide, or a different robot/task
     suite);
   - at least one more policy class (e.g. a diffusion or behaviour-cloning policy);
   - delay generated from a realistic source (network or compute latency traces) rather than
     only experimenter-designed schedules.
4. **Resolving the short-delay failure mode.** A principled "compensate or not" decision, for
   example a value- or risk-based gate, or a learned estimate of the policy's own delay
   tolerance. It would need to beat frozen SAC on short, frequent delays while keeping the
   persistent-delay gains.
5. **A literature review establishing novelty**, or a reframing as an empirical study if
   novelty is not established. See the questions in §13 of the technical report.
6. **Policy-level uncertainty:** more SAC seeds (e.g. 10), so that conclusions are not
   conditional on three policies.
7. **Systems measurements:** single-process, real-time latency of the estimator, detector and
   learned rollout on the target compute, including worst-case timing.
8. **Archived artifacts** with a DOI, and a one-command reproduction of every held-out table.

## 5. What is publishable now, honestly stated

- **Publishable now, after the artifact and documentation fixes:** a workshop or undergraduate
  paper presenting a rigorous simulation study of delay compensation for frozen policies,
  with the stale-compensation negative result as a central finding.
- **Not publishable now:** a main-track robotics conference paper claiming a new method. It
  would need items 1–3 of §4 at minimum.
- **Contributions that would survive peer review in their current form:**
  - the held-out effect sizes;
  - the documentation that stale compensation is worse than none for short delays;
  - the specificity result against the cross-episode sham.

  The mechanism decomposition would not survive in its current form, and should be presented
  as exploratory if at all.
