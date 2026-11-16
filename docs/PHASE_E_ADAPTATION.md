# Phase E pilot: online actuator identification and adaptive compensation (FetchPush-v4)

**Result (pilot, development seeds):**
- **Delay:** identifying action delay online from command–motion history, then evaluating the
  frozen policy at a predicted future state, rescues most delay failures:
  - pooled over the actuator conditions: +16.7 points [+14.2, +19.1], with 77 rescued and 2
    harmed episodes;
  - 3-step delay: +68.9 points;
  - a fixed-delay sham predictor does not reproduce this pooled.
- **Gain:** gain loss is identified accurately but can hardly be compensated, because the
  inverse-gain command runs into the ±1 action bound. The oracle with the true gain gains only
  +3.3 points (scale 0.5) and +8.9 points (scale 0.85).
- **No harm** on undisturbed or noise-only episodes: the estimator abstains under noise, so
  nothing activates there.

These are pilot numbers on 30 evaluation seeds. The hypothesis is supported for delay and
only weakly for gain.

Code: `src/farc/adaptation/` (`estimator.py`, `compensation.py`, `experiment.py`); script:
`scripts/run_adaptation_experiment.py`; config: `configs/adaptation/phase_e_pilot.json`;
tests: `tests/test_adaptation.py`; tables: `results/phase_e_pilot/`.

```bash
uv run python scripts/run_adaptation_experiment.py --config configs/adaptation/phase_e_pilot.json   # ~3 min, 3 processes
uv run python scripts/run_adaptation_experiment.py --config configs/adaptation/phase_e_pilot.json --analyze-only
```

## How commands become motion (inspected before building anything)

- **Command path:** `FetchEnv._set_action` multiplies the three position components by
  0.05 m. `mocap_set_action` first resets the mocap to the gripper body's current position and
  then adds that displacement. The gripper follows through a weld constraint over 20
  substeps.
- **Disturbances:**
  - `action_scale` multiplies the whole issued command by (1 − s).
  - `action_delay` executes the command issued d steps earlier, with zeros before the episode
    starts.
  - Both act after the controller and before clipping.
- **Measured on debugging seeds 3,630,000–009** (3 policies × 6 conditions, base policy only):
  observed gripper displacement ≈ 0.6 × 0.05 × the *executed* command, at lag 0, with
  R² 0.94–1.00.
  - The ratio is about the same in free space (0.55) and near the object (0.62), and when
    commands are saturated (0.63).
  - So the plant is close to linear, but even undisturbed it does not move the full commanded
    distance.
  - The nominal gain K0 = 0.62 is the median reliable raw slope on undisturbed debugging
    seeds. It drifts from 0.64 early in an episode to 0.56 late, so about 10% of plant
    variation is the floor on gain resolution.
- **Saturation:** 25% of base commands sit at the ±1 bound. Command magnitude is largest early
  (|c_xy| ≈ 1.3 at t = 0) and falls to about 0.3 after t = 30.

## Estimator (causal)

- **Inputs, at step t:** only gripper positions g[0..t] as the controller sees them (noisy
  under observation noise) and the commands it issued, c[0..t−1].
- **Fit:**
  - For each lag k = 0..4, a least-squares fit through the origin of Δg[τ] on 0.05·c[τ−k]
    over the last 10 transitions, horizontal plane only.
  - The best lag gives the delay. Its slope divided by K0 gives the gain relative to nominal.
- **Abstention:** each estimate is withheld unless:
  - there are at least 4 transitions;
  - the mean commanded displacement shows enough excitation (|c_xy| ≥ 0.3 equivalent);
  - R² ≥ 0.8;
  - for the delay only, the next-best lag's residual is at least 2× the best one's.
- **Use:** the deployable controller uses the last reliable value of each estimate ("latched").
- **Fixed in advance:** these constants were set on debugging seeds before any selection or
  evaluation episode was run.
- **Ground truth:** the true parameters are used only to score the estimates and in the
  ORACLE configuration. A test checks that no other configuration reads them.

## Compensation (conservative)

- **Gain:** `clip(a / max(ĝ, floor), −1, 1)` on the position components, applied only while
  ĝ < 0.85. The floor (0.25 or 0.5) was selected on the selection seeds and came out at 0.5.
- **Delay:** a Smith-predictor-style step.
  - With estimated delay d, the next d executed commands are already known: the issued
    c[t−d..t−1].
  - The gripper position after them is predicted with K0 (× ĝ).
  - The observation is shifted there: gripper position +Δ, object-relative position −Δ. The
    object is assumed static and velocities are left unchanged.
  - The frozen policy is then evaluated at that predicted state.
- **Combined:** both, with the gain applied after the prediction.
- **ORACLE (privileged):** the same controller fed the true gain and delay at each step.
- **SHAMs**, gated on the same estimator reliability so they activate when the deployable
  controller could:
  - `sham_gain`: a constant 1/0.5 boost whatever the estimate says;
  - `sham_delay`: the predictor with a fixed d = 2.

## Design

- **Conditions:** 8 Phase 1 conditions:
  - undisturbed;
  - action scale 0.5, 0.85, and 0.5 for steps 15–24 only;
  - delay 1 and 3;
  - observation noise on positions at 0.005 and 0.01 m.
- **Policies and configurations:** 3 frozen policies, 9 configurations.
- **Pairing:** every configuration runs the same episode seed with the same perturbation seed,
  so start state, goal and noise draws are identical.
- **Seeds:**
  - debugging 3,630,000–009 (plant probe, K0, thresholds, smoke run);
  - selection 3,631,000–019 (gain floor only);
  - evaluation 3,632,000–029 (reported).
  - All are in the validation stage, disjoint from Phases 1–D, and below 4,000,000. The loader
    enforces all of this.
- **Run integrity:**
  - The run checks that the frozen actors are unchanged.
  - Every configuration was replayed twice on 3 conditions and gave identical commands.
- **Provenance:** recorded in `metadata.json`: config hash, git commit `1166d31`, checkpoint
  SHA-256 values, seeds, timing.
- **Raw files** (`episodes.csv.gz`, `steps.csv.gz`, `metadata.json`) are written once.
  `--analyze-only` rewrites only the derived tables and `analysis.json`, using the same
  mechanism as Phase D.
- **CIs:** 95% episode-seed cluster bootstrap with 2,000 resamples. Each cell has 90 paired
  episodes (3 policies × 30 seeds).

## Results (evaluation seeds)

Pooled over the five actuator conditions (450 paired episodes, base success 60.9%):

| configuration | success | Δ vs base [95% CI] | rescued / harmed |
|---|---|---|---|
| estimated gain (floor 0.5) | 62.0% | +1.1 [−0.9, +2.9] | 12 / 7 |
| **estimated delay** | 77.6% | **+16.7 [+14.2, +19.1]** | 77 / 2 |
| estimated combined (floor 0.5) | 78.4% | +17.6 [+14.9, +20.2] | 85 / 6 |
| ORACLE combined (privileged) | 80.2% | +19.3 [+16.2, +22.4] | 96 / 9 |
| sham gain | 52.9% | −8.0 [−11.3, −4.7] | 20 / 56 |
| sham delay (d = 2) | 56.2% | −4.7 [−8.9, −0.4] | 68 / 89 |

By condition (success rate; Δ [CI] with rescued/harmed):

| condition | base | est. gain | est. delay | est. combined | ORACLE | sham gain | sham delay |
|---|---|---|---|---|---|---|---|
| undisturbed | 98.9% | +1.1 (1/0) | 0 | +1.1 (1/0) | 0 | −3.3 (1/4) | −17.8 (0/16) |
| scale 0.5 | 94.4% | +2.2 [−3.3, +7.8] | 0 | +2.2 | +3.3 [0, +7.8] | +3.3 | −41.1 |
| scale 0.85 | 16.7% | +4.4 [+1.1, +8.9] | 0 | +4.4 | +8.9 [+2.2, +15.6] | +4.4 | −6.7 |
| scale 0.5, steps 15–24 | 100% | 0 | 0 | 0 | 0 | −6.7 | −27.8 |
| delay 1 | 83.3% | 0 | **+14.4 [+6.7, +22.2]** (14/1) | +14.4 | +11.1 [0, +21.1] | −36.7 | −5.6 |
| delay 3 | 10.0% | −1.1 | **+68.9 [+61.1, +76.7]** (63/1) | +66.7 | +73.3 [+63.3, +83.3] | −4.4 | +57.8 |
| obs. noise 0.005 m | 97.8% | 0 | 0 | 0 | 0 | −11.1 | −11.1 |
| obs. noise 0.01 m | 95.6% | 0 | 0 | 0 | 0 | −13.3 | −20.0 |

**By policy seed:** estimated delay gives +16.0, +14.7 and +19.3 points; estimated gain gives
+2.7, +1.3 and −0.7.

**Selection seeds:** these were used only to pick the gain floor, so they are not independent.
They show the same pattern: delay +14.3 [+10.7, +18.0]; gain +2.3 [−1.0, +6.0].

### Estimator accuracy (base-policy episodes, evaluation seeds; `estimator_accuracy.csv`)

- **Gain:**
  - Reliable for 92–100% of steps from t = 5 under actuator disturbances and undisturbed.
  - Median absolute error 0.01–0.04, for example 0.519 vs 0.50 and 0.158 vs 0.15.
  - Late in an episode, small commands reduce excitation. The estimator then abstains more
    (undisturbed, t ≥ 30: reliable only 33% of the time).
- **Delay:**
  - When the estimator reports confidence, the lag is right in ≥ 99.6% of steps in every
    condition.
  - Unconditionally, accuracy is 81–100% after t = 5.
- **Under 1 cm observation noise:** R² drops to 0.6–0.8 and the estimator abstains on 44–93%
  of steps. When it does report, gain is biased +10%: noise in Δg inflates the slope only
  slightly, but enough to show.
- **Transient scale (steps 15–24):** the 10-step window lags the change. Error is 0.32 during
  the window, and the estimate is reliable only 77% of the time there.
- **Activation:**
  - Gain compensation is on for 92% of steps under persistent scaling, and 11–17% of steps
    undisturbed or under delay. Those activations are harmless: 1 rescue, 0 harms.
  - Delay compensation is on for 92% of steps under delay and never elsewhere.

### Oracle and sham comparison

- **Delay:** the deployable estimated-delay controller captures 86% of the oracle's pooled
  gain (+16.7 vs +19.3). On delay 1 it does better than the oracle (+14.4 vs +11.1), which is
  within noise. The early steps before the estimate is reliable cost little.
- **Shams:**
  - A blind 2× boost (sham gain) and a wrong fixed delay (sham delay) are harmful pooled,
    and on undisturbed and noise-only episodes.
  - So the benefit comes from identifying the *right* parameter, not from activity as such.
    This is the opposite of Phase D, where the random-direction placebo matched the
    engineered corrections.
  - The fixed d = 2 sham still helps on 3-step delay (+57.8). A predictor that is off by
    one step is mostly right, so that cell is not a pure placebo.
- **Gain:** even the oracle helps little. The ceiling is physical:
  - With 15% gain, a saturated command still moves the gripper 15% of nominal.
  - 82–97% of compensated steps hit the ±1 bound.
  - At gain 0.5 the base policy already succeeds 94% of the time.

## Limitations and failure cases

- **Pilot scale:** 30 evaluation seeds per cell. The design was conceived after the
  debugging-seed probe, and the gain floor was selected on selection seeds. This is a pilot,
  not a confirmatory result.
- **The delay disturbance is the easy case.** It is:
  - a constant, whole-episode transport delay of the *issued* commands;
  - zero-initialised, so the arm visibly does not move for d steps at the start, a strong
    identification cue;
  - noise-free.

  Delay identification stays accurate later in the episode, so it does not rely only on that
  cue. Still, time-varying delay, delay that starts mid-episode, delay mixed with noise or
  gain loss, and plant lag that resembles delay are untested.
- **The predictor assumes a static object** and the nominal plant gain. Under contact the
  object moves, so the predicted state is wrong. It works here because the policy is fairly
  robust to that error, not because the model is right.
- **Gain compensation is limited by the action bound,** not by identification. Gain loss
  above ~50% cannot be undone by rescaling commands.
- **Abstention under observation noise:** at 0.01 m position noise the estimator rarely
  reports. A delay under heavy noise would go uncompensated.
- **Transient disturbances:** the 10-step window lags, so short disturbances are compensated
  late or not at all.
- **Excluded:** action noise and goal shift. Neither has an actuator parameter to identify.

## Recommendation

- **Delay:** proceed to a larger, pre-registered evaluation of delay compensation on fresh
  seeds, with harder delay conditions:
  - onset mid-episode;
  - delays of 2 and 4 steps;
  - delay combined with observation noise and with gain loss;
  - delay that varies within an episode.

  Keep the oracle and the sham controls. Do not start RL training until that holds up.
- **Gain:** revise or drop inverse-gain compensation as a stand-alone method. Identification
  works, but the action bound limits what rescaling can do. Something that changes the plan
  (for example, allowing more time or a different approach path) would be needed instead.
