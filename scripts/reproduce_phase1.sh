#!/usr/bin/env bash
# Regenerate every Phase 1 result from the frozen checkpoints and compare it with results/.
# Needs the checkpoints in $FARC_DATA_DIR/checkpoints (docs/DATA.md). About 15 minutes on
# one CPU core of an Apple M5; writes about 100 MB under $FARC_DATA_DIR/experiments.
# Pass --resume to continue an interrupted disturbance run.
set -euo pipefail
cd "$(dirname "$0")/.."
data="${FARC_DATA_DIR:-$HOME/farc-data}"
detectors="$data/experiments/phase1_extended/detectors"

uv run python scripts/verify_checkpoints.py
uv run python scripts/run_disturbance_eval.py --config configs/disturbances/phase1_extended.json --quiet "$@"
uv run python scripts/evaluate_detector.py --config configs/detectors/progress_deficit_grid.json
uv run python scripts/evaluate_phase_detector.py --config configs/detectors/phase_aware_push.json
uv run python scripts/compare_detectors.py \
  --episodes "$detectors/phase_aware_push/phase_detector_episodes.csv.gz" \
  --out "$detectors/phase_aware_push/post_hoc_comparisons.csv" --min-lead 5 \
  --pair phase_k5_p3_h3_moff_th0 distance_k10_p5 \
  --pair phase_k5_p3_h3_moff_th0 distance_k10_p3 \
  --pair phase_k5_p3_h3_m3_th0 distance_k5_p5 \
  --pair phase_k10_p3_h3_moff_th0 distance_k10_p5 \
  --pair phase_k10_p5_h3_moff_th0 distance_k10_p5
uv run python scripts/sweep_detector_thresholds.py --config configs/detectors/threshold_sweep_push.json
uv run python scripts/check_reproduction.py
