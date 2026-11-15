# Provenance

This repository was split out of [m-alan08/quantrl-fetch](https://github.com/m-alan08/quantrl-fetch) on 2026-11-15. quantrl-fetch was left unchanged: no branch was deleted, rewritten, or merged, and its checkpoints are only read.

## Source

| Item | Value |
| --- | --- |
| Source repository | https://github.com/m-alan08/quantrl-fetch |
| Source branch | `phase1-disturbance-eval` (21 commits, `7e7f67a` to `633d953`) |
| Branch base | `qat-rerun` at `f2591d0` (on GitHub), which descends from `main` at `d9fcb67` |
| State at migration | clean working tree at `633d953` |

**Reachability.** At the time of migration `phase1-disturbance-eval` existed only in the local quantrl-fetch clone; it had not been pushed. Commit hashes below resolve in that clone, and on GitHub only once the branch is pushed to quantrl-fetch.

Commits migrated (oldest first):

| Commit | Subject |
| --- | --- |
| 7e7f67a | Add disturbance framework: validated specs, seeded perturbations, frozen-policy rollout, recovery metrics |
| 0b6b61d | Add disturbance experiment runner, plots, and pilot configs for five disturbance types |
| 7621e7c | Test disturbance determinism, timing, clipping, goal shifts, frozen normalizer, and metrics |
| 916039d | Document disturbance definitions and record the exploratory 20-episode pilot summary |
| 3b3f495 | Add field-masked observation noise with a validated Fetch observation layout, a validation seed range, and goal-range checks for goal shifts |
| fad4381 | Add the extended validation config and a resumable runner with per-seed and pooled summaries, transient and permanent recovery, and an original-evaluator check |
| fed638e | Order summary rows by each condition's first position in the config |
| 2613e3e | Document observation masks and recovery regimes, and record the extended validation-range baseline results and plots |
| 934063d | Add an offline progress-deficit failure detector evaluated on saved disturbance trajectories, with tests |
| 33961fe | Record the offline progress-deficit detector results, plots, and limitations |
| d6052bf | Correct two false-positive counts in the detector notes |
| 09ea4f4 | Log ground-truth gripper positions per state and add a FetchPush rerun config on the same validation seeds |
| 98c903e | Add a phase-aware FetchPush failure detector and a matched comparison with the distance-only detector, with tests |
| 208236c | Narrow an optional warning state in a phase-detector test for mypy |
| c39a4ab | Add an episode-paired comparison between any two saved detectors |
| ff20609 | Let the paired detector comparison count only warnings with enough steps left |
| 558ee2e | Record the phase-aware detector comparison, plots, and limitations |
| 91e44be | Add an exact threshold sweep comparing the distance-only and phase-aware detectors at matched false-positive rates, with tests |
| 3815180 | Report matched-FPR detection within each training seed in the threshold sweep |
| 88bb9c0 | Record the matched-FPR threshold sweep results, per-condition breakdown, and conclusions |
| 633d953 | Point the phase-detector notes to the matched-FPR sweep that supersedes their conclusion |

## File map

**Migrated with content unchanged apart from imports and data paths.** Each file starts with a two-line header naming its source. "First" and "last" are the first and last commits on `phase1-disturbance-eval` that touched the source file.

| quantrl-fetch path | Path here | First | Last | Commits |
| --- | --- | --- | --- | --- |
| `src/quantrl/disturbances/config.py` | `src/farc/disturbances/config.py` | 7e7f67a | 3b3f495 | 2 |
| `src/quantrl/disturbances/perturbations.py` | `src/farc/disturbances/perturbations.py` | 7e7f67a | 3b3f495 | 2 |
| `src/quantrl/disturbances/rollout.py` | `src/farc/disturbances/rollout.py` | 7e7f67a | 09ea4f4 | 3 |
| `src/quantrl/disturbances/metrics.py` | `src/farc/disturbances/metrics.py` | 7e7f67a | 7e7f67a | 1 |
| `src/quantrl/disturbances/experiment.py` | `src/farc/disturbances/experiment.py` | 0b6b61d | 09ea4f4 | 4 |
| `src/quantrl/disturbances/plots.py` | `src/farc/disturbances/plots.py` | 0b6b61d | fad4381 | 2 |
| `src/quantrl/disturbances/observation_layout.py` | `src/farc/envs/observation_layout.py` | 3b3f495 | 3b3f495 | 1 |
| `src/quantrl/disturbances/detector.py` | `src/farc/detection/progress_deficit.py` | 934063d | 934063d | 1 |
| `src/quantrl/disturbances/detector_eval.py` | `src/farc/detection/progress_deficit_eval.py` | 934063d | 934063d | 1 |
| `src/quantrl/disturbances/detector_plots.py` | `src/farc/detection/progress_deficit_plots.py` | 934063d | 934063d | 1 |
| `src/quantrl/disturbances/phase_detector.py` | `src/farc/detection/phase_aware.py` | 98c903e | 98c903e | 1 |
| `src/quantrl/disturbances/phase_eval.py` | `src/farc/detection/phase_aware_eval.py` | 98c903e | ff20609 | 3 |
| `src/quantrl/disturbances/phase_plots.py` | `src/farc/detection/phase_aware_plots.py` | 98c903e | 98c903e | 1 |
| `src/quantrl/disturbances/threshold_sweep.py` | `src/farc/detection/threshold_sweep.py` | 91e44be | 3815180 | 2 |
| `src/quantrl/disturbances/sweep_plots.py` | `src/farc/detection/threshold_sweep_plots.py` | 91e44be | 91e44be | 1 |
| `scripts/run_disturbance_eval.py` | `scripts/run_disturbance_eval.py` | 0b6b61d | fad4381 | 2 |
| `scripts/evaluate_detector.py` | `scripts/evaluate_detector.py` | 934063d | 934063d | 1 |
| `scripts/evaluate_phase_detector.py` | `scripts/evaluate_phase_detector.py` | 98c903e | 98c903e | 1 |
| `scripts/compare_detectors.py` | `scripts/compare_detectors.py` | c39a4ab | ff20609 | 2 |
| `scripts/sweep_detector_thresholds.py` | `scripts/sweep_detector_thresholds.py` | 91e44be | 91e44be | 1 |
| `tests/test_disturbances.py` | `tests/test_disturbances.py` | 7621e7c | 09ea4f4 | 4 |
| `tests/test_detector.py` | `tests/test_progress_deficit.py` | 934063d | 934063d | 1 |
| `tests/test_phase_detector.py` | `tests/test_phase_aware.py` | 98c903e | ff20609 | 4 |
| `tests/test_threshold_sweep.py` | `tests/test_threshold_sweep.py` | 91e44be | 3815180 | 2 |
| `configs/disturbances/*.json` (8 files) | `configs/disturbances/` | 0b6b61d | fad4381 | |
| `configs/detectors/*.json` (3 files) | `configs/detectors/` | 934063d | 91e44be | |
| `docs/DISTURBANCES.md` | `docs/DISTURBANCES.md` | 916039d | 2613e3e | |
| `docs/DETECTOR.md` | `docs/PROGRESS_DEFICIT_DETECTOR.md` | 33961fe | d6052bf | |
| `docs/PHASE_DETECTOR.md` | `docs/PHASE_AWARE_DETECTOR.md` | 558ee2e | 633d953 | |
| `docs/THRESHOLD_SWEEP.md` | `docs/THRESHOLD_SWEEP.md` | 88bb9c0 | 88bb9c0 | |
| `results/*` (compact tables) | `results/<experiment>/` | 916039d | 88bb9c0 | |
| `assets/{phase1_extended,detector,phase_detector,threshold_sweep}/*.png` (8 figures) | `docs/figures/` | 2613e3e | 88bb9c0 | |

**Ported from quantrl-fetch's base code** (`qat-rerun`, `f2591d0`), reduced to what frozen-policy evaluation needs:

| quantrl-fetch path | Path here | Change |
| --- | --- | --- |
| `src/quantrl/envs/factory.py` | `src/farc/envs/factory.py` | `EnvConfig` moved in from `quantrl/config.py` |
| `src/quantrl/envs/normalizer.py` | `src/farc/base_policy/normalizer.py` | unchanged |
| `src/quantrl/algos/networks.py` | `src/farc/base_policy/networks.py` | fp32 actor only (no critic, sampling, or quantized layers) |
| `src/quantrl/checkpoint.py` | `src/farc/base_policy/checkpoint.py` | rewritten: `weights_only` loading, SHA-256 manifest, fp32 only |
| `src/quantrl/eval/evaluate.py` | `src/farc/base_policy/reference_eval.py` | unchanged (used to check undisturbed rollouts) |
| `src/quantrl/eval/stats.py` | `src/farc/statistics.py` | `bootstrap_confidence_interval` only |
| `src/quantrl/disturbances/statistics.py` | `src/farc/statistics.py` | merged |
| `src/quantrl/utils/seeding.py`, `run_info.py` | `src/farc/utils/` | `imageio` dropped from tracked packages |

**Behavioural changes made during migration:**
- **Checkpoint paths.** Checkpoints and outputs are referenced through `$FARC_DATA_DIR` (default `~/farc-data`) instead of quantrl-fetch's `runs/` and `~/quantrl-experiments/`. Episode rows record the portable configured path.
- **Gripper logging for every task.** The runner logs gripper positions for every task, so the FetchPush-only rerun config `phase1_extended_push_gripper.json` is not migrated. The phase-aware and threshold-sweep configs read `phase1_extended` directly, and their `reference_source` check is dropped.
- **Pilot summary format.** The pilot summary in `results/phase1_pilot/` was produced with the first recovery definition (`916039d`). Rerunning `configs/disturbances/pilot.json` with the current code produces the later summary format, so that file is a historical record and is not part of the reproduction check.

## Checkpoints

The six frozen fp32 SAC+HER policies are not stored in either repository. Hashes, sizes, and training provenance are in `src/farc/base_policy/checkpoints.json`; how to obtain them is in [docs/DATA.md](docs/DATA.md).

| Checkpoints | Trained at quantrl-fetch commit | Notes |
| --- | --- | --- |
| `fetch_push_fp32_seed{0,1,2}` | `fdb6266` (on GitHub `main`), clean tree, `configs/fetch_push.json` | 1,000,000 steps each |
| `fetch_reach_fp32_seed{0,1,2}` | `6355bf7`, **dirty tree**, `configs/fetch_reach.json` | 30,000 steps each. `6355bf7` is not on any GitHub branch: it is from history before the rewrite that produced the current `main`. These files cannot be regenerated bit-for-bit from a published commit, so their hashes are the only exact reference. |

## Reproduction status

`scripts/reproduce_phase1.sh` regenerates the extended disturbance run and all three detector evaluations from the checkpoints, and `scripts/check_reproduction.py` compares 17 result tables with `results/`. In the migration run (disturbance run at `cd8ccf5`, detector evaluations at `110591d`, both clean trees, same locked package versions as quantrl-fetch), all 17 tables were bit-identical to the quantrl-fetch results, and every checkpoint's undisturbed episodes matched the original evaluator on all 100 seeds.
