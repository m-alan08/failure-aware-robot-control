# Phase J-2: implementation decisions recorded before training and evaluation

**Status:** committed before any J-2 training run and before any J-2 evaluation seed
(3,704,000–299) was used. Plan `docs/PHASE_J2_PROPOSAL.md` (SHA-256 `0b0f0046…`), approved as
written 2026-11-18, pinned in `configs/retrain/phase_j2.json`.

**L1. The single change from Phase I.**
- `NoisyPhaseIWrapper` subclasses the Phase I wrapper. In the training role only, it adds
  `masked_noise(layout, "positions", σ, N(0, 1) draws, dt)` to the observation before the
  features are computed.
- σ ~ U[0, 0.01] m per episode, and the draws are independent per step, both from
  `default_rng([20262300, run_seed, episode])`.
- The evaluation-role environment is the Phase I wrapper, with no noise.
- **Training config:** byte-identical to Phase I's `training` block, which a test checks.
- **Delay distribution:** Phase I's, via `training_schedule(..., generator_seed=20262200)`. The
  new optional argument defaults to Phase I's seed, so the Phase I schedules are unchanged, and a
  test checks this.

**L2. Stage 0.**
- **σ = 0:** the noisy wrapper produced observations bitwise identical to the Phase I wrapper
  over 20 steps.
- **Noise:** it changes gripper and object positions only. The relative position equals
  object noise minus gripper noise, all other entries are untouched, and σ stays within
  [0, 0.01].
- **No-update smoke runs:** 0.98 (J2-A) and 1.00 (J2-C), identical to Phase I's no-update
  runs. This confirms the warm start.

**L3. Evaluation** (runner committed before Stage 2):
- **Cells:** the 44 de-duplicated cells, plus `J_noise0.005` and `J_noise0.015`. A test asserts
  that no duplicate cells remain.
- **Configurations:** frozen SAC, contact-aware, C, Phase I I-A and I-C (registered), and J2-A
  and J2-C (registered after training).
- **Seeds:** perturbation seed 20262250; bootstrap seed 20262200.
