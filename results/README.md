# Reference results

These are compact tables from runs made in m-alan08/quantrl-fetch (branch `phase1-disturbance-eval`; see each `metadata.json` and [PROVENANCE.md](../PROVENANCE.md)). Full outputs (episodes, per-step trajectories) are regenerated under `$FARC_DATA_DIR` and never committed.

| Directory | Produced by | Regenerated and checked by `scripts/check_reproduction.py` |
| --- | --- | --- |
| `phase1_pilot/` | `configs/disturbances/pilot.json`, 20 episodes, seed-0 policies (exploratory) | No. Historical record: produced with the first recovery definition, so rerunning the pilot gives the later summary format. |
| `phase1_extended/` | `configs/disturbances/phase1_extended.json`, validation seeds, 3 training seeds × 100 episodes | Yes: `summary.csv`, `summary_per_seed.csv`, `distance_curves.csv` |
| `progress_deficit_grid/` | `configs/detectors/progress_deficit_grid.json` | Yes: all three tables |
| `phase_aware_push/` | `configs/detectors/phase_aware_push.json` and `scripts/compare_detectors.py` (arguments in `scripts/reproduce_phase1.sh`) | Yes: all five tables |
| `threshold_sweep_push/` | `configs/detectors/threshold_sweep_push.json` | Yes: all six tables |

All of these are exploratory, validation-range results; none used the reserved held-out seeds.
