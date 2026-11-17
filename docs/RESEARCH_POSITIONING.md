# Research positioning

This document builds on `docs/RELATED_WORK.md` (sources and access levels) and
`docs/NOVELTY_AUDIT.md` (classifications). Bracketed numbers refer to the related-work
entries. Development is "dev"; held-out is "HO".

## 1. The questions

**1. What is the closest prior method?**
- **For the compensation architecture (Phases E/F1): Wafi 2026 [4].**
  - A delay-free-trained DDPG actor is *frozen* and deployed behind a Smith predictor.
  - It uses a fixed nominal delay and no online estimation, on first-order plants (full text
    read).
  - Our Phase E is that architecture plus online lag identification and a learned
    contact-aware predictor, on a manipulation task.
- **For non-stationary delay handling (G2–G4): Wang et al. 2025 [1].**
  - Per its abstract, it detects shifts in non-stationary observation delays with Bayesian
    online change-point detection and selects delay-specific dynamic models.
  - The full text was not accessible, so its setting and results are unverified.
- **In robotics, the closest predict-then-act wrapper around a frozen policy: AHEAD [20].**
  It feeds a latent forecast to a frozen VLA, with real robot results.

**2. Is our method algorithmically novel?**
- **No, as far as this search can tell.**
- Each component is published: Smith-predictor compensation [23], [4]; online delay
  estimation [25], [26]; delay-free learned dynamics composed with an action buffer [2];
  change detection for non-stationary delays [1]; CUSUM [27a].
- The specific combination (online lag ID, learned contact predictor and residual CUSUM with
  pause and reset, around a frozen state-based RL manipulation policy) was not found. That is
  a combination, not an algorithmic contribution.

**3. Is our empirical finding genuinely new?**
- **Possibly, but unconfirmed.** We did not find a held-out, quantified demonstration that:
  - (i) object-motion prediction carries delay compensation for a frozen pushing policy;
  - (ii) with short, frequent actuator delays, compensation with a stale estimate underperforms
    the uncompensated frozen policy, and change detection only partly mitigates this.
- Finding (ii) is predictable in direction from Smith-predictor mismatch theory [24], [4].
- A full-text search, especially of [1] and the teleoperation literature, could still turn up
  either result.

**4. What is the strongest result that survives comparison with prior literature?**
- **The negative finding,** quantified on held-out seeds. Compensation helps a lot under
  persistent delay:
  - +44.9 over frozen SAC (E/F1 HO H1);
  - +9.3 more from contact-aware prediction (H2).
- **But where delays are short and frequent, the frozen policy is better left alone:**
  - contact-aware is 9.5 points below frozen SAC on random on-off delays (G4 HO, descriptive);
  - change detection narrows the gap (+3.5 over contact-aware, G4 P1) without closing it (C −
    SAC −2.2 [−3.2, −1.3], G4 S1).
- **Second strongest:** the attribution to object prediction (H3, +8.7).
- **What makes these credible:** pre-registration, two held-out confirmations, sham controls,
  and byte-exact analysis reproduction.

**5. What would CoRL, RSS, ICRA or RA-L reviewers challenge?**
1. **Novelty:**
   - "This is a Smith predictor with online delay identification [23], [25]";
   - "frozen RL plus Smith predictor exists [4]";
   - "change detection for non-stationary delay exists [1]".
2. **Why not retrain?** Delay-randomised training works for exactly this robot and task
   family on hardware [5]. Delay-aware RL is mature: [6], [10], [11], [3].
3. **Simulation only:** a mocap-driven, nearly linear plant (R² 0.94–1.00) makes lag
   identification unusually easy; a synthetic transport delay with replay/drop semantics; full
   state.
4. **One task, one policy family** (3 SAC+HER seeds).
5. **Experimenter-chosen delay schedules** rather than measured latency traces. Wikman et al.
   [3] used measured WiFi delays.
6. **The detector baseline:** no comparison with BOCPD [27c], [1] or with a tuned
   fixed-window or two-timescale baseline beyond our own G2 candidates.

**6. Which baseline would most directly challenge our central claim?**
- **A FetchPush SAC+HER policy retrained with delay randomisation, plus action-buffer state
  augmentation,** following the constant-delay augmentation of [8], [3] or a recurrent policy
  as in [5].
  - Train it on the same delay schedule distribution and evaluate it on the same held-out
    schedules.
  - Report the training compute and data it needs.
- **Why this one:**
  - If it matches or beats compensation under persistent delay, the "no retraining" claim
    reduces to a cost argument.
  - If it *also* avoids the short-delay failure, the negative finding becomes a statement about
    compensators specifically, not about delay in general.
- **Second most direct:** a BOCPD detector [27c] replacing our CUSUM inside the same
  pause-and-reset controller, following [1]'s approach.
- **Third:** a fixed-nominal-delay Smith predictor following [4]. We effectively have it in
  development as the E2 fixed-delay sham. It helped only near the true delay and harmed
  undisturbed episodes (−16, E2 dev), but it has not been confirmed held-out.

**7. Does the negative finding justify a standalone empirical paper?**
- **At a workshop: yes.** The finding is clean, held-out and reproducible, and it is
  decision-relevant for anyone deploying predictor-based compensation around a frozen policy.
- **At a main venue: not on its own.** Reviewers will ask whether the failure is specific to:
  - one task;
  - a near-linear mocap plant;
  - replay/drop delay semantics;
  - our particular compensator.

  The finding becomes main-venue material only with:
  - at least one more task and one more policy class;
  - a retrained or delay-aware baseline evaluated on the same short-delay schedules;
  - ideally, realistic delay traces or a real robot.

**Confirmed vs hypothesis:**

| statement | status |
|---|---|
| Effect sizes cited as HO (H1–H3, P1–P3, S1) | **confirmed** on held-out seeds, in simulation, for these 3 policies |
| G1/G3 numbers and the mechanism decomposition | **development-only** |
| "Not found in the literature" | **search-limited,** not a confirmed absence |
| How a retrained policy, a BOCPD detector or another task would compare | **hypotheses;** untested |

## 2. Recommended paper theses

### Primary (recommended): an empirical study of when compensation helps and when it hurts

> **When to compensate: model-based delay compensation for frozen manipulation policies under
> persistent and intermittent actuator delay.** In simulation, online delay identification
> with contact-aware state prediction recovers a frozen pushing policy under persistent delay
> (pre-registered, held-out). Under short, frequent delays, compensating with a stale estimate
> is worse than not compensating. A residual-based change detector reduces but does not
> remove this gap.

**Why primary:**
- It leads with the findings that survive the literature comparison (audit D and B),
  supported by the strongest methodology (audit E and the held-out design).
- It claims no algorithmic novelty, which keeps it robust to the closest prior work [1], [4].
- It turns the main limitation, that the frozen policy beats compensators on short delays,
  into the paper's central question.

**What it still needs:**
- the retrained, delay-randomised baseline (§1.6);
- one more task or policy class;
- realistic delay traces;
- a full-text check of [1].

**Venue:** a workshop now; RA-L or ICRA after the additions.

### Alternative 1: contact-aware prediction for frozen-policy latency compensation

> *Predicting object motion, not end-effector motion, is what makes observation-level delay
> compensation work for a frozen pushing policy.*

- **Why:** H2 and H3 are clean, held-out, and grow with delay length.
- **Why not primary:**
  - narrower;
  - learned push models [30] and predict-then-act wrappers around frozen policies [20] are
    closely related;
  - its novelty depends most on a literature check of latency compensation with learned
    contact models, which we did not find but did not search exhaustively.
- **Venue:** workshop, or a short paper with a second contact-rich task.

### Alternative 2: a methodology paper on evaluating adaptive components for frozen policies

> *Pre-registration, held-out confirmation and timing-specific sham controls for evaluating
> adaptive add-ons to robot policies: a case study in delay compensation.*

- **Why:** the project's most distinctive asset is its evaluation discipline:
  - SHA-pinned plans;
  - write-once data;
  - two held-out confirmations;
  - byte-exact reproduction;
  - a documented failure of naive sham design (G2) and its fix (G3/G4).
- **Why not primary:**
  - methodology papers need evidence that the practices change conclusions, which we have only
    for the sham design;
  - their audience in robot learning is smaller;
  - the claim of novelty for yoked shams was not searched.
- **Venue:** a reproducibility or evaluation workshop track.

## 3. Highest-value next research steps

1. **The retrained baseline.**
   - Train SAC+HER on FetchPush with delay randomisation over the G3/G4 schedule distribution,
     with and without action-buffer augmentation.
   - Evaluate it on a fresh held-out slice. This needs your approval: 4,000,600+ has not been
     touched.
   - This one comparison decides whether the work is about compensation or about delay.
2. **Detector and predictor baselines from the literature:**
   - BOCPD in place of CUSUM [1], [27c];
   - a fixed-nominal-delay Smith predictor [4].

   Both fit the existing runner without changing Candidate C, and can be pre-registered on
   development seeds first.
3. **External validity:** one more task (for example FetchPickAndPlace or FetchSlide), one more
   policy class, and *measured* delay traces (as in [3]). Real-robot validation is next,
   ideally on a Fetch or Franka given [5], [19] and [22].

**In parallel, at no experimental cost:** obtain the full text of Wang et al. 2025 [1] and
complete the literature gaps listed in `docs/RELATED_WORK.md` §3.
