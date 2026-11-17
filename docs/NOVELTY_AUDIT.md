# Novelty audit

Our five candidate contributions are audited against the literature verified in
`docs/RELATED_WORK.md`. Bracketed numbers such as [4] refer to its entries.

The audit is limited by that review:
- it is targeted, not systematic;
- the closest detection paper, Wang et al. 2025 [1], was available only at abstract level.

**A classification of "potential" novelty means no contradicting prior work was found in a
limited search. It does not mean novelty is established.** No first-of-its-kind claim is made
anywhere here.

## Categories

| category | meaning |
|---|---|
| **Established prior art** | The idea, or a very close variant, is published. |
| **Known components, potentially new setting** | Each piece is published. The combination or application domain was not found in our search. |
| **Potential methodological novelty** | A design or evaluation method we did not find in this literature |
| **Potential empirical novelty** | A measured finding that we did not find reported. It may still be implied by known theory. |
| **Unsupported novelty claim** | A claim that the evidence or literature does not support |

## What our system actually does

Recapped from the code and `docs/RESEARCH_TECHNICAL_REPORT.md`, so that we compare like with
like:

- **Delay identification (Phase E, `farc.adaptation.estimator`):**
  - least-squares lag selection $\hat d = \arg\min_k \mathrm{SSE}_k$ over a 10-transition
    window of $\Delta g_\tau$ on $0.05\,c_{\tau-k}$;
  - reliability gates (transitions, excitation, $R^2$, margin) and a latch.
- **Compensation (Phase E / F1):** a Smith-predictor-style shift of the frozen policy's
  *observation* to the state predicted after the $\hat d$ queued commands. The predicted state
  comes from either:
  - a kinematic model (static object), or
  - a learned contact-aware MLP rolled out $\hat d$ steps (`farc.contact.dynamics`).

  The frozen SAC+HER policy is evaluated at that predicted observation.
- **Change detection (G2 Candidate C, `farc.delaychange.candidates.Sequential`):**
  - a multi-alternative CUSUM on per-lag residuals of the newest transition;
  - on an alarm: pause compensation (fall back to frozen SAC) and reset the identification
    window to the estimated change point; re-engage through the Phase E gates.
- **Evaluation:**
  - paired designs with pre-registered, SHA-pinned plans;
  - two held-out confirmations (E/F1: 4,000,000–299; G4: 4,000,300–599);
  - sham controls, including a cross-episode yoked sham under per-episode random schedules.

## A. Frozen-policy actuator-delay compensation

| | |
|---|---|
| **Classification** | **Known components, potentially new setting**, with **potential empirical novelty** for the specific held-out results. **Not algorithmically novel.** |
| **Prior art** | **Smith predictor [23]**, the principle of predicting past the dead time with a model. **Adaptive Smith predictors with online delay estimation [25], [26]**, for networked classical controllers. **Wafi 2026 [4]** is the *same architecture* for a frozen RL policy: a DDPG actor trained delay-free is frozen and deployed behind a Smith predictor, with a fixed nominal delay and no online estimation, on first-order plants. **Chen et al. 2021 [2]** learns delay-free dynamics and composes it with a known action buffer. **AHEAD [20]** is a predict-then-act wrapper around a *frozen* robot policy (VLA) using a learned latent forecast, with a real robot. **RTC [18]** and **RAPAC-DP [19]** handle *inference* latency for frozen or near-frozen robot policies. |
| **What differs in ours** | The delay is *identified online* from the command–motion record rather than fixed or given. The predictor feeds a frozen *state-based* RL manipulation policy. The setting is contact-rich pushing. Evaluation is pre-registered and confirmed held-out: +44.9 [+43.1, +46.7] over frozen SAC (E/F1 HO H1). |
| **Assessment** | The architecture is not new: [4] publishes it for a frozen DDPG actor, and Smith-style prediction is classical. Online lag identification plus Smith prediction is also known in classical control [25], [26]. The combination of online identification, frozen *learned* manipulation policy and contact task was not found, but it is a modest combination. |
| **Unsupported** | Any claim to have introduced Smith-predictor compensation for frozen RL policies, or online delay estimation for predictor-based compensation |

## B. Contact-aware learned state prediction

| | |
|---|---|
| **Classification** | **Known components, potentially new setting.** The attribution result is **potential empirical novelty**. |
| **Prior art** | Learned pushing and contact dynamics models [30a], [30b]. Learned dynamics used to predict through a delay [2], [14] (world models for delayed observations). Learned forecasting fed to a frozen robot policy under latency [20]. |
| **What differs in ours** | Held-out evidence that the gain comes from predicting the *object's* motion, not the gripper's. Contact-aware beats the kinematic predictor by +9.3 [+8.3, +10.4] and a consistent gripper-only learned predictor by +8.7 [+7.7, +9.7] (E/F1 HO H2, H3). The gain grows with delay length, from +1.2 at delay 1 to +16.0 at delay 4 (HO, descriptive). |
| **Assessment** | Learning contact dynamics is standard, and using a learned model to bridge delay is standard. The ablation showing *which* predicted quantity matters for a frozen pushing policy under actuator delay was not found. It is a narrow, setting-specific finding. |
| **Unsupported** | "Contact-aware prediction for delay compensation" as a new method |

## C. Online delay-change detection and supervisory switching

| | |
|---|---|
| **Classification** | **Established prior art** for the components and for change detection on non-stationary delays in RL. **Known components, potentially new setting** for the specific form. |
| **Prior art** | **Wang et al. 2025 [1] (RA-L)**: per the abstract, Bayesian online change-point detection detects shifts in non-stationary observation delays and selects among delay-specific dynamic models, and the authors call it the first effective approach to non-stationary delay in RL. CUSUM [27a] and change-detection theory [27b]; BOCPD [27c]. Multiple-model switching adaptive control [28]. Fall-back switching architectures [29]. |
| **What differs in ours** | *Actuator* delay rather than observation delay. A *frozen* policy (whether [1] retrains is unverified). A multi-alternative CUSUM on lag-hypothesis residuals. Two specific responses, *pause compensation* and *reset the identification window to the change point*. Held-out: +3.5 [+2.4, +4.6] over the contact-aware compensator on random short-lived and on-off delays, with persistent non-inferiority +0.5 [+0.2, +0.9] (G4 P1, P2). |
| **Assessment** | Change detection for non-stationary delays in RL has been published [1]. Change detection and model switching in adaptive control are classical [27], [28]. Our detector is a straightforward instance. No first or novelty claim is defensible. **A head-to-head with a BOCPD detector [1], [27c] is missing.** |
| **Unsupported** | "The first change-detection approach to non-stationary delays", which [1] claims for itself. Any claim that the pause-and-reset mechanism is a methodological contribution: its decomposition (G3) is development-only and sensitive to alarm divergence. |

## D. Stale compensation fails under short, intermittent delays

| | |
|---|---|
| **Classification** | **Potential empirical novelty** in this setting. The *mechanism* is consistent with established theory. |
| **Prior art** | The dead-time-compensator literature analyses sensitivity to delay and model mismatch [24]; the survey was verified at metadata level only. Wafi [4] derives closed-loop maps under plant and delay mismatch and notes the sensitivity of standalone DDPG to delay and plant changes. Wang et al. [1] motivate change detection by the cost of mismatched delay models. |
| **What differs in ours** | Measured, with paired held-out evidence, for a frozen learned manipulation policy: *compensating with a stale delay estimate is worse than not compensating* when delays are short and frequent. In G4 (HO), contact-aware was 9.5 points below frozen SAC on random on-off delays (descriptive). Candidate C was 2.2 points below frozen SAC on random short-lived and on-off delays (S1, pre-registered secondary; non-inferiority not shown). Development: G1 T1–T3, contact-aware −6.3 to −16.3 vs SAC. |
| **Assessment** | The direction is predictable from mismatch theory. What we did not find is a quantified, controlled demonstration for frozen learned policies on contact manipulation, nor the observation that the uncompensated policy *tolerates* short delays better than any compensator tested. It is a useful empirical finding. Its generality beyond FetchPush, mocap control and replay/drop delay is untested. |
| **Unsupported** | That this is a new *phenomenon*; it is an instance of known mismatch sensitivity. That it generalises to other tasks or hardware. |

## E. Episode-specific detection vs cross-episode yoked sham controls

| | |
|---|---|
| **Classification** | **Potential methodological novelty in evaluation**, not in algorithm. The finding is **potential empirical novelty**, narrow. |
| **Prior art** | Sham or placebo controls and yoked designs are general experimental-design devices. We did not search for or verify their use in robot-learning or delayed-RL evaluation, so no prior-art claim either way is made. [1] is the work whose specificity claim this design would test. |
| **What differs in ours** | Per-episode random delay schedules make a cross-episode yoked sham *mis-timed* by construction, so it isolates whether the detector uses this episode's transitions. G4 (HO): C beats the same pause-and-reset mechanism driven by another episode's events by +3.2 [+2.2, +4.2] (P3). On the fixed grid, by contrast, the same-condition yoked sham inherits correct timing and cannot test this: G2 +0.5 [−0.0, +1.1]; G4 +0.5 [+0.05, +0.9], descriptive. |
| **Assessment** | This is a sound evaluation design. Our own G2 exposed the weakness of the naive version, and G3/G4 fixed it. It is worth reporting as a methodological lesson. Whether others have used it is unknown. |
| **Unsupported** | "C exploits each episode's transitions better than any schedule-based alternative." The sham uses mis-timed events, not an optimal schedule-aware controller. "First to use yoked shams" was never searched. |

## Summary

| contribution | classification | strongest defensible statement |
|---|---|---|
| A | known components, potentially new setting; not algorithmically novel | Online lag ID plus Smith-style observation prediction recovers a frozen manipulation policy under constant actuator delay (HO +44.9). The architecture is close to [4] and classical [23], [25]. |
| B | known components; potential empirical novelty (attribution) | For a frozen pushing policy, predicting *object* motion is what makes delay compensation work at longer delays (HO +9.3, +8.7). |
| C | established prior art for components and for CPD on non-stationary delay [1]; known components otherwise | A residual CUSUM with pause and reset improves on fixed-window identification (HO +3.5) without persistent cost. No novelty claim. |
| D | potential empirical novelty; mechanism known | With short, frequent delays, stale compensation underperforms the uncompensated frozen policy; change detection narrows but does not close the gap (HO −2.2 vs SAC). |
| E | potential methodological (evaluation) novelty; unverified | Under random per-episode schedules, a cross-episode yoked sham isolates episode-specific detection (HO +3.2). The fixed-schedule version cannot. |

**Overall:**
- **Algorithmic novelty:** none of A–E is established.
- **Most defensible contributions:** empirical and methodological:
  - **D,** the failure mode, with held-out quantification;
  - **B,** the attribution;
  - **E,** the evaluation design;
  - all of it carried by the pre-registered, held-out, reproducible methodology.
- **Two missing comparisons decide how much survives review:**
  - a **retrained, delay-randomised or delay-augmented policy** on the same task and schedules
    ([5] is the direct precedent on Fetch pushing);
  - a **BOCPD-style detector** in the style of [1].
