# Residual RL infrastructure: status, protocol findings and decisions needed

**Status:** software only. `docs/RESIDUAL_RL_PREREGISTRATION.md` is a **draft and NOT approved**,
and this work did not edit it: its SHA-256 still matches the pin in
`configs/residual/residual_rl.json`, and a test checks that. No SAC training ran. No registered
seed was used: not training, debugging, selection, evaluation or held-out. No frozen result
and no part of Candidate C changed. **No number in this note or in the tests is an
experimental result.**

## 1. What exists

| piece | file | notes |
|---|---|---|
| T-M, T-O primitives | `src/farc/residual/disturbances.py` | from the earlier commit; T-M gains `friction_mode` (§3.1 below) |
| Training mixture | `src/farc/residual/training_env.py` | `draw_training` gives the family and parameters as a pure function of (generator seed, episode seed). `TrainingSessionFactory` is the `ResidualEnv` session factory. `AR1PositionNoise` injects T-O into the session's controller view. `DisturbedSession` owns and restores the T-M patch. |
| SAC runner | `src/farc/residual/training.py`, `scripts/residual_train.py` | `check_run` refuses registered runs until the plan is approved and its pin matches. Smoke runs are allowed only on `SMOKE_SEEDS` = [0, 100,000). |
| Selection | `analysis.select_alpha`, `scripts/residual_select.py` | the §5 rule, the tie rule, the null stop, and failed runs entering as C |
| Pre-written analysis | `analysis.development_analysis`, `scripts/residual_analysis.py` | §6 gatekeeping, fixed claim wording, Holm-adjusted S1–S5, harmed/rescued counts. The CLIs refuse until the plan is approved. |
| Tests | `tests/test_residual_training.py` (46), `tests/test_residual_disturbances.py` (6) | synthetic plants, plus FetchPush with a scripted stand-in policy on seeds 0–2 |

`ResidualEnv` itself is unchanged: T-M and T-O reach it only through the session factory.

### Runner behaviour

The runner implements the §4 settings listed in `training.py`. The pieces most likely to
matter:
- **Determinism:** two identical smoke runs give bitwise-identical weights and logs (tested).
- **Records:** checkpoints every 50k steps, with `checkpoint_0` equal to C exactly. The log is
  JSONL (config, episodes, updates, nominal evaluations, rollbacks, end), and every episode's
  disturbance draw is saved.
- **Warm-up:** the delta = 0 data phase and critic-only warm-up are tested to leave the actor
  untouched.
- **Rollback rule:** tested to halve the learning rate, then fail the run at the second rollback.
  The threshold is exactly 2 points.
- **Seed exhaustion** raises an error instead of reusing seeds silently (§3.3).

### Real-simulator smoke (wiring check only)

`scripts/residual_train.py --smoke --steps 300` ran with the frozen `fetch_push_fp32_seed0`
policy on smoke seeds 0–6 (episodes) and 100–104 (nominal evaluation). It ran to the end, and
all five families appeared in the episode log.

The smoke run **found and fixed one bug:** the nominal evaluation shared the training simulator
and corrupted the in-progress training episode. Evaluation now gets its own environment. The
rollback during this smoke run comes from a 5-episode evaluation and means nothing.

## 2. Inertia: recommendation

**Recommendation: scale mass and inertia together (as implemented).**
- In FetchPush, `object0` is a free body with one box geom, `size = 0.025` (a 5 cm cube), and
  `mass="2"`. The MJCF gives no explicit inertia, so MuJoCo derives it from the geom. The
  compiled inertia is 8.333e-4 = m·(0.05² + 0.05²)/12 for m = 2 kg, which checks.
- Writing `mass = 2s` in the MJCF would make MuJoCo compute inertia × s exactly. Scaling both
  is therefore what recompiling the model would give: the same cube, uniformly denser.
- Scaling mass alone would keep the old inertia, so the radius of gyration would grow as
  1/√s. That is a cube with its mass moved toward or away from its centre: a second, unstated
  change.
- Minor: `body_subtreemass` is compiled once and is not updated by a runtime patch. Nothing in
  the free-body dynamics used here reads it.

**Proposed pre-registration wording (for approval, not applied):** §3 T-M: "object mass and
rotational inertia × m ~ U[0.5, 2.0] (uniform density change)".

## 3. Protocol findings (need a decision; the draft is unchanged)

### 3.1 Friction × s < 1 is a no-op as written (major)

MuJoCo combines two geoms' friction by element-wise **maximum**. The table and the fingers
have sliding friction 1.0. So scaling only the object's friction by s < 1 changes nothing: it
was verified **bitwise identical** to nominal at s = 0.5 and 0.75, and it is tested.

Consequences:
- About half of T-M's friction range is inert.
- The §6 development cell "friction × 0.5" would be an exact copy of the nominal cell.

Options:
- **(a, recommended)** Also scale the table's sliding friction. The object–table coefficient is
  then exactly × s in both directions; finger contact stays max(s, 1). This is implemented as
  `friction_mode="object_table"`, which is **off by default** and tested to change the dynamics
  and to restore.
- **(b)** Give the object geom priority. This also changes its torsional and rolling friction
  in finger contacts, even at s = 1.
- **(c)** Restrict the range to s ≥ 1. This changes the family.

### 3.2 Training seed budget is half what §4 needs

§4 asks for 1M environment steps per run. With 50-step episodes that is 20,000 episodes, but
the training block 3,710,000–3,719,999 holds 10,000 seeds. It cannot simply be extended,
because 3,720,000 is the debugging block. The runner raises `SeedsExhausted` instead of
choosing.

Options:
- **(a)** Two in-order passes over the block. The second pass repeats the same families,
  parameters and initial states.
- **(b)** Register a second unused block, for example 3,730,000–3,739,999, after a disjointness
  check.
- **(c)** Halve the budget to 500k.

Recommended: (b), registered before use.

### 3.3 Selection seeds are used during training

The rollback rule evaluates the 100 nominal **selection-block** seeds every 50k steps, which is
up to 20 times per run. §5 then selects α* on the same block. Checkpoints that survive are
partly chosen on those seeds, so the §5 nominal non-inferiority test on them is optimistic.

Recommended: a separate registered "training-monitor" block for the rollback rule.

### 3.4 Power for the −1-point non-inferiority margins (steps 1–2)

There is no power analysis. Power depends mostly on how often C and C+R *disagree* on the same
episode:
- In a synthetic fixture with independent draws per network seed, the analysis failed step 1
  with no true effect, so that design was underpowered.
- With coupled draws (the realistic case for a small residual) it passed.

Recommended: before approval, a power calculation from existing G4 C outcomes, plus an assumed
discordance rate. This needs no new seeds.

### 3.5 Settings §4 leaves implicit (implemented as below; confirm or amend)

| setting | implemented |
|---|---|
| updates per environment step | 1 (from the §9 "1M updates" estimate) |
| initial entropy coefficient | 1.0 (the `SACConfig` default) |
| Q-target clipping | **off**: the residual penalties push rewards below −1, so the base SAC's [−1/(1−γ), 0] clip would be wrong |
| α ramp | starts at step 0, so the delta = 0 warm-up data are collected at α = 0.05; this has no effect on commands, since delta = 0 |
| rollback target | the last checkpoint that passed (`checkpoint_0` = C); the replay buffer is kept; network, optimiser and entropy states are restored, then the actor learning rate is halved |
| T-D schedule cell index | `draw_pulses` index 1000 + 10·k + delay, disjoint from G3/G4's 0–8; RS ignores the drawn delay (it uses its own pairs, as in G3) |
| T-A / T-D perturbation seed | the plan's `generator_seed` 20261900 (G4 used 20261800) |
| T-A zero severities | omitted, giving an exact no-op |
| T-O start | stationary (z₀ ~ N(0, 1)), so the noise is stationary from step 0 |
| T-M timing | patch applied after `env.reset`, before the first step; restored at episode end, or when the next episode is built |

### 3.6 Not wired yet

- **B2 (residual on frozen SAC):** it needs a base-only core with the 41-dim observation, and
  `scripts/residual_train.py` refuses it.
- **Selection and development episode generators:** these would run C and C+R over the
  registered cells and write the episode table. Not written, because they would only ever run
  on registered seeds.

## 4. Decisions needed before any residual RL training

1. Choose the friction fix (§3.1) and the inertia wording (§2), recorded as amendments.
2. Choose the seed-budget fix (§3.2), registering any new block before use.
3. Decide whether the rollback rule gets its own seed block (§3.3).
4. Accept or amend the implicit settings (§3.5).
5. Run the power check (§3.4) and decide whether the margins or the sample size change.
6. Approve the pre-registration and pin its new SHA-256 in `configs/residual/residual_rl.json`.
7. At the approving commit, before any training seed, run:
   - the §3 T-M/T-O nominal-identity check on debugging seeds 3,720,000–009;
   - the §8.4 real-simulator zero-residual re-check.
8. Wire B2, and write the selection and evaluation episode generators, with tests.
