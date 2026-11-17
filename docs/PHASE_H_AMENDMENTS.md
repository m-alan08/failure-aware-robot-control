# Phase H: implementation decisions recorded before training and evaluation

**Status:** committed before any Phase H training run (Stage 1) and before any evaluation seed
(3,701,000–299) was used. Only unit tests, the Stage-0 checks and simulator tests on the
Phase H debugging block (3,700,000–009) had run.

- **Approval:** the plan (`docs/PHASE_H_RETRAINED_BASELINE_PROPOSAL.md`, SHA-256
  `f7989f90…`) was approved as written on 2026-11-17, including:
  - training *new* policies;
  - the optional nominal seed-0 retrain;
  - deferring the held-out decision until after Stage 2.
- **Pinned:** its SHA-256 is in `configs/retrain/phase_h.json`.

**J1. The port.**
- Twelve upstream files (quantrl-fetch @ `95e9611`, `src/quantrl/…`) are copied to
  `src/farc/retrain/qrl/`.
- The only changes are:
  - import paths rewritten to `farc.retrain.qrl.`;
  - in `training.py`, **one** marked change: an optional `env_wrapper(env, role)` hook applied
    to the training env (role `"train"`) and the evaluation env (`"eval"`) right after
    `make_env`. With `None`, the code is upstream's.
- A test checks the marked lines. A file-by-file comparison against the upstream checkout
  showed every other file identical.
- The port is excluded from mypy (`ignore_errors`), because it is not re-typed.

**J2. Stage-0 results** (code checks, not outcomes):
- **Trainer:** a 20,000-step seed-0 run with the upstream trainer and with the port (no
  wrapper) gave **bitwise-identical** checkpoints, all 124 tensors: actor, critics, target
  critics, optimisers, temperature and normaliser.
- **Timing:** 35 s per 20,000 steps. The first 10,000 steps have no updates, so the
  steady-state estimate is about 1–1.5 h per 1M-step run on one thread.
- **Wrapper:** `DelayHistoryWrapper` executed exactly the actions `ScheduledDelaySession`
  executes, and produced identical observations, for 8 training-distribution episodes covering
  all six families (simulator test, debugging seeds).

**J3. The wrapper.**
- Executed action = the issued command from d(t) steps earlier, zeros before the start, clipped
  to the action bounds, as `ScheduledDelaySession.delayed` with replay/drop.
- For R2, the observation is followed by the position components of the last 4 issued commands,
  most recent first, zero-padded. That gives 25 + 12 = 37 entries.
- The training env gets the delay distribution, plus history for R2. The upstream evaluation
  env inside `train()` gets history only (R2) or nothing (R1), with no delay.

**J4. Training schedule draws.**
- `training_schedule(run_seed, episode)`: the family comes from
  `default_rng([20261900, run_seed, episode])`. Pulses come from `draw_pulses` with generator
  seed 20261900 and key `run_seed·10⁶ + episode`.
- `draw_pulses` gained an optional `generator_seed` argument. Its default is the G3/G4 value
  (20261700), so G3/G4 schedules are unchanged, and a test checks this.
- `episode` counts the wrapper's resets, including the 10,000 random-action steps at the start.

**J5. Upstream checks that stay as they are, and are not used for any decision:**
- In-training evaluations: 50 episodes every 20,000 steps.
- A final 100-episode evaluation, on quantrl-fetch's own seed range (1,000,000+, no delay).

These outputs are recorded but never used to select or stop anything. The final checkpoint is
used.

**J6. Checkpoint identity.**
- Upstream checkpoints embed run metadata, including a start timestamp. A retrained file can
  therefore never match a frozen checkpoint by file hash.
- The optional nominal seed-0 retrain (no wrapper, `--nominal`) is compared with
  `fetch_push_fp32_seed0` **tensor by tensor**, which is reported descriptively. It is expected
  to differ unless the package versions and code at training time match ours.
- Retrained checkpoints are registered in `src/farc/retrain/checkpoints.json` after training.
  Evaluation refuses unregistered files, except in the debug run.

**J7. Provenance note.**
- Upstream `git_state` runs `git status --porcelain --untracked-files=no`, so an *untracked*
  config file does not mark the tree dirty.
- This plausibly explains the recorded "clean tree at `fdb6266`" while
  `configs/fetch_push.json` was added only in `95e9611` (proposal §2).

**J8. Evaluation details** (Stage 2):
- **Cells:** G4's 41 cells plus the 9 E/F1 conditions as fixed schedules (`E_C0`–`E_C8`; the
  E/F1 noise and scale disturbances unchanged).
- **Perturbation seed:** 20261950. **Bootstrap seed:** 20261950, with 10,000 resamples.
- **Retrained policies:**
  - act through the same `ScheduledDelaySession` on the noisy observation, plus for R2 the last
    4 issued commands;
  - pair with frozen seed k by index.
- **"Persistent" for H3/S3:** RP ∪ F-P ∪ E C1–C7. **"RT + RI":** Arm R only.
- **Training-failure flag:** a retrained policy's mean success below 90% on `H_undisturbed`
  and `E_C0_undisturbed` combined.

## Recorded after Stage 1 (training), before any evaluation seed was used

**J9. Training completed as planned.**
- All 7 runs (R1 and R2 × seeds 0–2, plus the nominal seed-0 retrain) reached 1,000,000
  steps, each taking about 6,600–6,720 s, run in parallel.
- Each checkpoint's metadata records commit `07e4e72` with a clean tree.
- The six R1/R2 checkpoints are registered in `src/farc/retrain/checkpoints.json`.
- Upstream's own end-of-training evaluation (100 episodes, no delay, quantrl seeds) gave
  0.91–1.00. It is recorded in `results/phase_h_training/` and used for nothing (J5).
- **Launch problem:** a first launch failed at import (`ModuleNotFoundError`) before any
  training step. The parallel `uv run` invocations raced to reinstall the editable package. No
  run directory was created. The relaunch used `uv run --no-sync` after a single `uv sync`.

**J10. Nominal retrain = frozen seed 0, bitwise.**
- The nominal seed-0 retrain (ported trainer, no wrapper) is **identical in all 124 tensors** to
  the frozen `fetch_push_fp32_seed0`: actor, critics, targets, optimiser state, temperature and
  normaliser.
- Both checkpoints record the same package versions (torch 2.14.1, numpy 2.5.4, mujoco 3.11.0,
  gymnasium-robotics 1.4.2).
- The frozen FetchPush policy is therefore exactly regenerable from quantrl-fetch `95e9611`'s
  code and `configs/fetch_push.json` **on this machine**. The file hash differs only because of
  the embedded metadata (J6).
- Cross-machine reproducibility remains untested.
