# Phase G3: what makes Candidate C work? (proposal)

**Status:** proposal only, for approval. Nothing below has been implemented or run. No seed in
the blocks proposed here has been used. Candidate C, the frozen SAC policies, the Phase E
estimator and the F1 model are unchanged. PRs #8 and #9 stay unmerged.

## 1. The question

G2 found that Candidate C beats the contact-aware compensator on short-lived and on-off delays
(+1.7 in Stage 3, +1.8 on the exploratory seeds). The sham check showed that the *timing* of C's
pauses matters: the same pauses at random times gave nothing (+0.2 [−0.2, +0.6]).

Two things remain tangled:

1. **Two mechanisms act at each alarm.** C *pauses* compensation (frozen SAC acts) and *resets*
   the identification window to the estimated change point, so the old lag's history stops
   contaminating the fit. The sham check only varied the pause.
2. **Episode-specific detection versus a predictable schedule.** In G2 every episode of a
   condition had the same transition times. A sham copying C's pauses from another episode was
   therefore already well timed, and gained +0.8 on its own.

G3 separates pause from reset and the episode's own transitions from the population's timing.
It is a development-only mechanism study, not a new candidate and not a confirmation.

## 2. Feasibility, from reading the code

| need | what exists | what has to be added |
|---|---|---|
| Per-episode random schedules | `DelaySchedule(pulses)` takes any non-overlapping pulses; `ScheduledDelaySession` takes the schedule per episode | A deterministic generator: (generator seed, cell, episode seed) → pulses |
| Fixed schedule within a pair | Every configuration of one (policy, cell, episode seed) already shares one session setup | Build the schedule once per episode seed and share it across controllers and the 3 policies |
| Pause and reset as separate switches | `Sequential.step` does detection, pause and reset in one method | A new `EventReplay` controller in a new module. It repeats `Sequential`'s post-detection code, takes injected events and has `pause` / `reset` flags. **`Sequential` itself is not edited.** |
| Same alarms as C | Stage 3 already ran sources before dependants (`masks`) | Log C's events (alarm step, change point, trusted lag) per episode and replay them |
| Pause mask | `run_g2_episode(..., yoked=mask)` already forces compensation off on masked steps | Nothing |
| Alarm divergence | – | A *shadow* `Sequential` in every replayed run: it observes the same estimator but its decisions are ignored |
| Oracle | `PRIV_true_schedule` and `PRIV_release_oracle` show how privileged code is fenced off | Events at the true transition steps, through `EventReplay` |
| Seeds | `check_g2_seeds`; validation range 3,500,000–3,999,999 | G3 blocks and `check_g3_seeds`; every G2 block, now including the used exploratory block, joins the "used" list |

Two properties make the replay clean:

- **With no events, `EventReplay` equals `contact_learned`.** `Sequential` with no alarm already
  equals it (A11, tested).
- **Up to C's first alarm, every C-derived run is identical to C.** So the *first* replayed
  event is exactly causal in every variant. Only later events can be non-causal, after a
  variant's trajectory departs from C's. §4.4 measures how often.

Throughput is known from Stage 3: 125,400 episodes in 20 minutes on 9 processes, 43 MB of
step logs. The proposed 324,000 episodes are roughly 55 minutes and about 110 MB.

## 3. Schedules

All randomised schedules are drawn by `numpy.random.default_rng([20261700, cell index, episode
seed])` over the 50-step episode. Every parameter is fixed here, before any rollout. A schedule
depends only on the cell and the episode seed: all 12 controllers and all 3 policies of a pair
see the identical schedule.

| cell | delay | randomised | per-episode draw |
|---|---|---|---|
| `RT_d2`, `RT_d4` | 2, 4 | onset and end | onset ~ U{4 … 34}; duration ~ U{3 … min(16, 46 − onset)}, so every pulse has an onset and an end with at least 4 steps after it |
| `RI_d2`, `RI_d4` | 2, 4 | phase, every on and off length | first onset ~ U{4 … 12}; then alternately on ~ U{3 … 10} and off ~ U{3 … 10}, each drawn independently, until the next onset would be past step 44; the last pulse is cut at step 50 |
| `RS` | pair from {1→4, 4→1, 2→4, 4→2}, uniform | pair and change time | first delay from step 0; change ~ U{12 … 32}; second delay to the end |
| `RP_d2`, `RP_d4` | 2, 4 | onset | onset ~ U{0 … 20}, persistent |
| `H_undisturbed`, `H_noise0.01` | none | – | as Stage 3 (safety only) |

Ranges are chosen to cover Stage 3's fixed grid (onsets 6, 18, 30; durations 4, 8, 16;
on-off periods 4 and 8). They are not outcome-based.

## 4. Design

### 4.1 Seeds

| block | seeds | use |
|---|---|---|
| debugging | 3,680,000–009 | code checks only; never analysed |
| **evaluation** | **3,681,000–3,681,999** (1,000) | the one run |
| exploratory | 3,682,000–099 | reserved for post-hoc checks, labelled as such |

All lie in the validation range, are disjoint from every earlier block (F1, E2, G1, G2
including 3,674,000–099), and are far below 4,000,000. The guard refuses anything at or above
4,000,000, so 4,000,300+ cannot be reached.

**Size:** 1,000 episode seeds × 3 frozen SAC policies = 3,000 pairs per cell; the primary scope
(RT + RI, 4 cells) has 12,000 pairs. 9 cells × 12 configurations × 3,000 = **324,000 episodes**.

**Why 1,000 seeds** (a planning assumption from G2 data, not a prediction of any effect): C and
contact-aware disagreed on 7–8% of G2 pairs. At 8% discordance, 12,000 pairs give a standard
error of about 0.26 points. A 1.0-point contrast then has power of roughly 0.9 at the
strictest Holm level (0.025 / 4). With 500 seeds, power would fall to about 0.6.

### 4.2 Configurations

An *event* is (alarm step, change point, lag trusted at the alarm). "C's events" are those C
raised in the same episode.

| name | events from | pause | reset | role |
|---|---|---|---|---|
| `base_only` | – | – | – | frozen SAC; reference for safety |
| `contact_learned` | – | – | – | reference: the "neither" cell |
| `C_full` | C's own CUSUM, closed loop | yes | yes | Candidate C, pinned constants, unchanged |
| `C_pause` | C's events | **yes**: C's exact paused steps | no | diagnostic |
| `C_reset` | C's events | no | **yes**: at C's change points | diagnostic |
| `C_pause_own` | C's events | yes, until the *un-reset* estimator is reliable again | no | diagnostic, descriptive only |
| `sham_pause_random` | C's events | C's paused steps, rotated to a random time (as the G2 sham check) | no | timing control |
| `sham_full_yoked` | C's events from the **next seed of the same cell**, which has a **different random schedule** | yes | yes | specificity control |
| `ORACLE_full` | true transition steps | yes | yes, at the true transition | **PRIVILEGED**, diagnosis only |
| `ORACLE_pause` | `ORACLE_full`'s paused steps | yes | no | **PRIVILEGED**, diagnosis only |
| `ORACLE_reset` | true transition steps | no | yes | **PRIVILEGED**, diagnosis only |
| `PRIV_true_schedule` | – | – | – | **PRIVILEGED** ceiling: true delay and gain (as A10) |

Details fixed now:

- **`C_pause`** is `contact_learned` with C's paused steps forced off, the G2 yoked mechanism
  applied to the same episode. It keeps C's pause durations exactly.
  - Those durations end when C's *reset* window becomes reliable, so `C_pause` inherits some
    reset-derived timing.
  - `C_pause_own` brackets this: it pauses at C's alarms but releases when the ordinary
    10-step window is reliable again, which may be almost at once. It is reported, not tested.
- **`C_reset`** keeps compensating with the old lag until the reset window gives a reliable
  estimate, then switches lag. It is `EventReplay(pause=False, reset=True)`.
- **Onset events** (no delay trusted yet) never pause, in C or in any variant. Pause can only
  matter at ends and changes. Results are broken down by transition kind for that reason.
- **`sham_full_yoked`** keeps each source event's offset between alarm and change point. An
  event pauses only if this run trusts a delay at that step, the same rule as C. Its source
  episode has its own schedule, so the sham carries C's typical event timing but not this
  episode's transitions.
- **`sham_pause_random`** uses the G2 rotation (offset uniform on 1 … 49, seeded from
  (20261700, policy seed, cell index, episode seed)).
- **Oracle event indexing:** when the true delay changes at step t, the command executed at t is
  the first one affected, and the first affected position transition is t → t + 1. The oracle
  therefore pauses from step t and resets the window to transition t. A unit test pins this.
  The oracle reads the future schedule and is never a candidate, never in a hypothesis test,
  and never a deployable controller.

### 4.3 Running order within a task

For each (policy, cell):

1. Sources: `base_only`, `contact_learned`, `C_full` (events and paused steps logged),
   `ORACLE_full` (paused steps logged) and `PRIV_true_schedule`.
2. Dependants: `C_pause`, `C_reset`, `C_pause_own`, `sham_pause_random`, `sham_full_yoked`,
   `ORACLE_pause`, `ORACLE_reset`.

The frozen-actor check from Stages 1–3 is kept.

### 4.4 Alarm divergence (required by the brief)

Replayed events are C's causal alarms. After a variant's trajectory departs from C's, later
replayed events are no longer what C *would* have raised on that trajectory. To document this:

- **Integrity check, must be 100%:** in `C_pause`, `C_reset` and `C_pause_own`, the shadow
  detector's first alarm equals C's first alarm in every episode. Any exception is a bug: the
  analysis stops and I report it.
- **Reported per variant:**
  - share of C's later events matched by a shadow alarm within ±1 step;
  - unmatched shadow alarms per episode;
  - share of episodes with full agreement.
- **Sensitivity (descriptive, selection-biased by construction):** the M1–M3 contrasts in
  episodes with full agreement only.

## 5. Measures

Per configuration, per family and per transition kind (onset, end, change):

- **Success:** paired differences, rescued / harmed counts.
- **Onset latency:** steps from a true onset until compensation is on with the true lag
  (Stage-3 definition).
- **Release latency:** steps from a true end until compensation is first off (A9 definition).
- **Change latency:** steps from a change between nonzero delays until the new lag is applied.
- **Incorrect compensation:** steps compensating with a lag different from the true delay,
  including compensating when the true delay is 0. Split into *stale* (the previous true lag)
  and *other*.
- **Missed compensation:** steps with a true delay above 0 and compensation off.
- **Alarms against true transitions:**
  - the signed offset from each alarm to the nearest preceding transition;
  - recall, meaning transitions followed by an alarm within 5 steps, by kind;
  - false alarms per 1,000 steps with no transition in the previous 5 steps.
- **Activity:** share of delayed steps and of delay-free steps with compensation on.
- **Switching and controller overhead:** descriptive, as Stage 3.

## 6. Hypotheses and statistics

All contrasts are paired over (policy, cell, episode seed), with an episode-seed cluster
bootstrap (10,000 resamples, seed 20261700). The schedule is shared within an episode seed, so
clustering on it also covers the schedule. CIs are 95% percentile intervals. Differences are in
success-rate points.

### 6.1 Gatekeeper (tested first)

**G: `C_full` − `contact_learned` > 0 on RT + RI**, one-sided at α = 0.025.

- If G passes, M1–M4 are tested as below.
- If G fails, C's gain does not survive unpredictable schedules. That is itself the main
  finding, and M1–M4 are reported descriptively only.

### 6.2 Primary mechanism hypotheses (RT + RI; Holm across M1–M4, familywise α = 0.025)

| | question | contrast | test |
|---|---|---|---|
| **M1** | does pausing at the *right time* help? | `C_pause` − `sham_pause_random` | one-sided, > 0 |
| **M2** | does resetting stale history help on its own? | `C_reset` − `contact_learned` | one-sided, > 0 |
| **M3** | do pause and reset interact? | (`C_full` − `C_reset`) − (`C_pause` − `contact_learned`) | two-sided (p = 2 × the smaller one-sided p) |
| **M4** | does C need *this episode's* transitions? | `C_full` − `sham_full_yoked` | one-sided, > 0 |

For M3, the bootstrap resamples the four-way difference within episode seeds, so its CI
accounts for the shared pairs.

**Minimum meaningful effect: 1.0 point** for every contrast, M3 in absolute value. C's total
gain is about 1.7 points, so a component below 1 point is not a meaningful share of it. Each
contrast is labelled:

| label | rule |
|---|---|
| meaningful | Holm-significant and estimate ≥ 1.0 |
| significant but small | Holm-significant and estimate < 1.0 |
| negligible | 95% CI entirely inside (−1.0, +1.0) |
| inconclusive | anything else |

### 6.3 Secondary (descriptive tier; Holm across these 6, reported separately)

- M1, M2 and M4 on `RS` (changes between nonzero delays, where reset should matter most).
- The pause main effect, `C_pause` − `contact_learned` on RT + RI.
- `C_full` − `base_only` on RT + RI, non-inferiority margin −2 (as S4).
- `sham_full_yoked` − `contact_learned` on RT + RI: what population-typical event timing gives.

### 6.4 Oracle diagnostics (no tests)

- `ORACLE_full` − `C_full`: what detection latency and false alarms cost.
- The oracle 2 × 2 (`ORACLE_full`, `ORACLE_pause`, `ORACLE_reset`, `contact_learned`): the pause
  and reset effects under perfect detection.
- `PRIV_true_schedule`: the identification ceiling, for scale.

### 6.5 Safety and non-inferiority (gates, outside both families)

- **Persistent delays:** `C_full` and `C_reset` vs `contact_learned` on RP, margin −1 (CI low
  > −1), as G2's H2.
- **No delay:** every non-privileged configuration vs `base_only` on both H cells: CI low > −2
  and at most 2% of base successes harmed, as G2.

A failed gate is reported next to the variant. It does not change the mechanism tests, but it
rules out any deployability statement about that variant.

## 7. How the outcomes would be read

| pattern | reading |
|---|---|
| G fails | C's G2 gain depended on the fixed grid; nothing is concluded about mechanisms |
| M1 meaningful, M2 negligible | the gain is pausing at the right time; resetting adds nothing on its own |
| M2 meaningful, M1 negligible | the gain is removing stale history; the pause is incidental |
| M1 and M2 meaningful, M3 negligible | two additive mechanisms |
| M3 meaningful, positive | each needs the other, e.g. the pause covers the steps while the reset window fills |
| M4 meaningful | C exploits this episode's transitions, which a predictable schedule cannot supply |
| M4 negligible, G passes | events at C's *typical* times help as much as its actual detections; detection specificity is not shown |
| `ORACLE_full` ≫ `C_full` | detection is the bottleneck; a faster detector is worth pursuing |
| `ORACLE_full` ≈ `C_full` | C is near what its pause-and-reset mechanism can give |

**What G3 still cannot show:**

- The replayed events are only exactly causal up to the first divergence (§4.4).
- `C_pause` keeps durations set by C's reset (§4.2), so M1 slightly favours pause.
  `C_pause_own` shows the other bound.
- The generator's ranges are my choices.
- Simulation only, with drop/replay transport delay. Development seeds only; any held-out claim
  needs a separate plan and approval.

## 8. Integrity

- **Before any seed is used:** this proposal is approved and committed. Its SHA-256 is pinned in
  `configs/delaychange/g3_mechanism.json`; the runner refuses a changed plan or a dirty tree.
- **No calibration:** C's constants are the pinned Stage-3 ones. The generator, sham and oracle
  parameters are fixed in §3–4. Anything decided during implementation is recorded as a
  numbered amendment *before* the evaluation seeds are used.
- **One run:** write-once raw files and `LOCK.json`, as in G2. The analysis is pre-written and
  run once. Code checks use the debugging block only, and their outcomes are not read.
- **Preserved:** all earlier code paths, results and raw directories. The only edit to an
  existing file is the additive seed guard.

## 9. Implementation plan

| file | content |
|---|---|
| `src/farc/supervision/seeds.py` | add G3 blocks, G3 used ranges (all G2 blocks) and `check_g3_seeds` (additive) |
| `src/farc/delaychange/random_schedules.py` | the generator of §3 |
| `src/farc/delaychange/mechanism.py` | `EventReplay`, event extraction (C, oracle, yoked, rotated), the shadow detector, the episode runner and `run_task` |
| `src/farc/delaychange/mechanism_analysis.py` | §5 measures, §6 tests and labels |
| `scripts/run_g3_mechanism.py`, `configs/delaychange/g3_mechanism.json` | runner (`--debug`, default, `--analyze-only`) and pinned config |
| `tests/test_g3_*.py` | see below |

Tests written before any rollout:

- the generator is deterministic, within bounds, has no overlaps, and puts an onset and an end
  in every RT episode;
- one schedule per episode seed, shared by every configuration and policy;
- `EventReplay(pause=True, reset=True)` with C's own events reproduces closed-loop C's issued
  commands bitwise (simulator test, plus all debugging episodes);
- `EventReplay` with no events equals `contact_learned`;
- the shadow detector's first alarm equals C's in every C-derived run;
- oracle event indexing on a known schedule;
- the gatekeeper, Holm, M3 interaction bootstrap and labelling rules on synthetic data.

## 10. Implementation risks

1. **Replay is not fully causal.** After the first divergence, later replayed events are C's,
   not the variant's. *Mitigation:* the exact first-event check, divergence rates, and the
   agreement-only sensitivity analysis. If divergence is common (say over 30% of multi-event
   episodes), M1–M3 should be read as "C's events applied to another trajectory".
2. **Pause duration is not reset-free** (§4.2). *Mitigation:* `C_pause_own` as the other
   bracket. If the two disagree strongly, M1 is reported as bounded rather than point-identified.
3. **`EventReplay` duplicates `Sequential`'s post-detection code.** A silent mismatch would bias
   every contrast. *Mitigation:* the bitwise-equivalence tests. If they fail, nothing runs.
4. **Off-by-one in oracle or change-point indexing.** *Mitigation:* explicit unit tests on known
   schedules.
5. **C's gain may shrink on randomised schedules.** Randomised pulses include shorter and later
   ones than the fixed grid. *Mitigation:* gatekeeping, so a failed G is a reported finding,
   not a reason to change the design.
6. **Power rests on G2's discordance rate.** Contrasts between C variants may disagree less
   often, which helps; extra clustering across cells may hurt. 1,000 seeds leave margin.
7. **Analysis cost.** About 16 million step rows. Stage 3's per-episode metric loop handled 6.3
   million; this will take longer and use several GB of memory. Not a correctness risk.
8. **Design choices that shape conclusions** (generator ranges, the yoked source as the next
   seed, the 1.0-point threshold) are all fixed here, before data.

## 11. Decisions for you

The defaults above are my recommendation. Approving the proposal as written approves them.

- **Sample size:** 1,000 evaluation seeds (about 55 minutes), against 500 (power about 0.6 for a
  1-point contrast).
- **`C_pause_own`:** included as a descriptive bracket (recommended), or dropped to save one
  configuration.
- **Primary scope:** RT + RI (continuity with G2's H1), with RS secondary. Making RS primary
  would favour detecting a reset effect.
