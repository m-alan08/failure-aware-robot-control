# Phase I: implementation decisions recorded before training and evaluation

**Status:** committed before any Phase I training run (Stage 1) and before any Phase I evaluation
seed (3,702,000–299) was used. Only unit and simulator tests and Stage-0 smoke runs had run;
they ran on the debugging block 3,700,100–109 and upstream's own seeds.

- **Approval:** the plan (`docs/PHASE_I_STRONG_BASELINE_PROPOSAL.md`, SHA-256 `409b2f8e…`)
  was approved as written on 2026-11-17. Its SHA-256 is pinned in `configs/retrain/phase_i.json`.
- **Approved choices:**
  - arms I-A and I-C, with no from-scratch arm;
  - 1M fine-tuning steps with `random_steps = 0`;
  - a 1/3 / 1/3 / 1/3 training mix;
  - gates of −2 pooled and a 95% per-seed floor;
  - the frozen original-input normaliser;
  - seed blocks 3,700,100–109 and 3,702,000–299;
  - Stage 3 decided separately.

**K1. A second marked change to the ported trainer.**
- `train()` gains `init_hook(agent, normalizer) -> (agent, normalizer)`, applied right after
  upstream builds the agent and normaliser. With `None` the code is upstream's.
- With both hooks unset, the 20,000-step upstream-vs-port check was re-run and is still
  **bitwise identical** (124/124 tensors).

**K2. The warm start (`farc.retrain.phase_i.warm_start`).**
- Every tensor of the frozen agent state is copied: actor, critics, target critics,
  `log_alpha`, and all three Adam optimiser states.
- Where only the input width differs (the first layers, and their Adam `exp_avg` and
  `exp_avg_sq`), K zero columns are inserted after the 25 original observation entries. The
  normalised input is [observation (25), extra (K), goal (3)], and the critics append the 4
  action entries.
- The frozen checkpoint is SHA-256-verified against the frozen manifest and only read.
- **Stage 0:** the warm-started deterministic actions are **bitwise identical** to the frozen
  policy's, for all three seeds and both arms, on 4,096 random inputs each. Critic outputs agree
  within 1e-5.

**K3. Normaliser.**
- `HybridNormalizer` holds a copy of the frozen normaliser, never updated, for the 25 original
  entries and the goal. Extra entries pass through clipped to ±5.
- `update` is a no-op, so upstream's per-episode `normalizer.update` call does nothing.
- Stage 0: the original entries equal the frozen normaliser's output exactly.

**K4. Features (`Featurizer`), computed identically in training and evaluation.**
- **I-A:**
  - the last 10 issued commands' position components, most recent first, zero-padded;
  - then the last 10 observed gripper displacements $(g_{t-k+1}-g_{t-k})/(K_0\cdot 0.05)$, most
    recent first, zero-padded;
  - 60 entries in total.
- **I-C:**
  - the last 4 issued position commands;
  - a one-hot of the latched delay estimate (Phase E estimator + `LatchedEstimate`, as in
    `contact_learned`; the same estimator constants as G4);
  - the estimator's current `delay_reliable` flag;
  - 18 entries in total.
- Gripper positions come from the observation the policy sees: noise-free in training, noisy
  under observation-noise conditions in evaluation. This is the same input C's estimator uses.
- Stage 0: the training wrapper and the evaluation loop produced **bitwise-identical** features
  on 3 training-distribution episodes per arm (simulator test).

**K5. Training schedule.**
- Group ∈ {nominal, persistent, transient}, uniform, from
  `default_rng([20262000, run_seed, episode])`.
- **Persistent:** half constant from step 0, half random-onset (the RP generator), with
  d ~ U{1, …, 4}.
- **Transient:** a third each pulse (d ∈ {2, 4}), on-off (d ∈ {2, 4}) and change of value.
- Pulses are drawn by `draw_pulses` with generator seed 20262000 and key
  `run_seed·10⁶ + episode`.
- Stage 0: the groups are within ±200 of 3,000 each over 9,000 draws.

**K6. Training-record field.** `phase_i_record.json` stores `git_commit_at_end`. The commit
training *started* from is in the checkpoint's own metadata (`metadata.git`), which is
authoritative.

**K7. Stage-0 observation (not an outcome; the design is unchanged).**
- **20,000-step smoke runs** (seed 0; 10,000 steps of collection, then about 5,000 updates):
  upstream's in-training undisturbed evaluation (50 episodes, its own seeds) gave **0.90
  (I-A) and 0.82 (I-C)**.
- **2,000-step runs with no updates** gave 0.98 and 1.00, matching the frozen policy. This
  confirms the warm start itself.
- **Reading:** the drop after the first updates is a fine-tuning transient on a small fresh
  replay buffer. It is a known risk to the nominal gate (proposal §7).
- **Nothing was changed in response.** It is recorded so that a later gate failure can be read
  in its light.

**K8. Evaluation** (Stage 2; runner committed before Stage 2):
- **Configurations:** frozen SAC, contact-aware, Candidate C, Phase H R2 (registered checkpoints),
  I-A and I-C.
- **Cells:** the same 50 cells as Phase H.
- **Seeds:** perturbation seed 20262050; bootstrap seed 20262000 with 10,000 resamples.
- **Analysis:** as proposal §6, with the gates first.

## Recorded after Stage 1 (training), before any evaluation seed was used

**K9. Training completed as planned.**
- All 6 runs (I-A and I-C × seeds 0–2) reached 1,000,000 steps in about 5,400–5,500 s each,
  run in parallel. Each checkpoint's metadata records commit `7164d68` with a clean tree.
- All 6 are registered in `src/farc/retrain/checkpoints_phase_i.json`, with their warm-start
  sources.
- **The K7 dip recovered in every run.** Upstream's in-training undisturbed evaluation, which is
  descriptive and used for nothing, ended at 0.94–1.00 at 1M steps. The final 100-episode
  evaluation gave 0.97–1.00.
- Rolling training success on the delayed mix ended at 0.85–0.91, against 0.69–0.84 for Phase H
  R2.
- No decision is taken on these numbers. The gates are evaluated in Stage 2.
