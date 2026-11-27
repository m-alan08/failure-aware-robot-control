#!/usr/bin/env bash
# Registered residual-RL campaign (docs/RESIDUAL_RL_AMENDMENTS.md, R-A2 "campaign order").
# Resumable: re-running it skips finished runs (final.pt), resumes interrupted runs
# (resume.pt), and reuses cached rollout tasks. Local CPU only. Stops before held-out.
#
#   caffeinate -i bash scripts/residual_campaign.sh 2>&1 | tee -a results/residual_rl/campaign.log
#
# PARALLEL (default 3) training runs at a time, one per network seed, each single-threaded.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:scripts
OUT=results/residual_rl
PARALLEL=${PARALLEL:-3}
WORKERS=${WORKERS:-8}
mkdir -p "$OUT"
log() { echo "[$(date '+%F %T')] $*"; }

run_one() {  # arm alpha ns dir
  local arm=$1 alpha=$2 ns=$3 dir=$4
  if [[ -f $dir/final.pt ]]; then log "skip $dir (finished)"; return; fi
  local flag=""
  [[ -f $dir/resume.pt ]] && flag="--resume" && log "resume $dir"
  mkdir -p "$dir"
  log "start $dir"
  uv run python scripts/residual_train.py --arm "$arm" --alpha "$alpha" --network-seed "$ns" \
    --out "$dir" $flag > "$dir/stdout.log" 2>&1
  log "done $dir: $(tail -c 400 "$dir/stdout.log" | tr '\n' ' ' | grep -o '"status": "[a-z_]*"')"
}

train_batch() {  # arm alpha -> 3 network seeds, PARALLEL at a time
  local arm=$1 alpha=$2 pids=()
  for ns in 0 1 2; do
    if [[ $arm == R ]]; then d=$OUT/R_a${alpha}_n${ns}; else d=$OUT/${arm}_n${ns}; fi
    run_one "$arm" "$alpha" "$ns" "$d" &
    pids+=($!)
    if (( ${#pids[@]} >= PARALLEL )); then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
  done
  for p in "${pids[@]}"; do wait "$p"; done
}

# 1. nine R-alpha runs
for a in 0.05 0.1 0.2; do train_batch R "$a"; done
for a in 0.05 0.1 0.2; do for n in 0 1 2; do
  [[ -f $OUT/R_a${a}_n${n}/final.pt ]] || { log "R_a${a}_n${n} missing"; exit 1; }
done; done

# 2. selection (3,721,000-099) -> alpha*
if [[ ! -f $OUT/selection/selection.json ]]; then
  log "selection rollouts"
  uv run python scripts/residual_rollouts.py --role selection --runs "$OUT" \
    --out "$OUT/selection" --workers "$WORKERS"
  uv run python scripts/residual_select.py --episodes "$OUT/selection/selection_episodes.csv.gz" \
    --failed "$OUT/selection/failed.json" --out "$OUT/selection"
fi
ASTAR=$(python3 -c "import json;print(json.load(open('$OUT/selection/selection.json'))['alpha_star'])")
if [[ $ASTAR == None ]]; then log "no alpha passes selection: null result (prereg 8.2); stop"; exit 0; fi
# prereg 8.3: stop if more than one third of the R-alpha* runs failed by the rollback rule
NFAIL=$(python3 -c "import json;print(sum(1 for f in json.load(open('$OUT/selection/failed.json')) if f[0]=='R' and f[1]==$ASTAR))")
if (( NFAIL >= 2 )); then log "$NFAIL/3 R-alpha* runs failed by rollback: stop (prereg 8.3)"; exit 0; fi
log "alpha* = $ASTAR"

# 3. twelve A1-A3 and B2 runs at alpha*
for arm in A1 A2 A3 B2; do train_batch "$arm" "$ASTAR"; done

# 4. development evaluation (fixed 600 seeds) and pre-written analysis
if [[ ! -f $OUT/evaluation/analysis.json ]]; then
  log "evaluation rollouts"
  uv run python scripts/residual_rollouts.py --role evaluation --alpha-star "$ASTAR" \
    --runs "$OUT" --out "$OUT/evaluation" --workers "$WORKERS"
  uv run python scripts/residual_analysis.py \
    --episodes "$OUT/evaluation/evaluation_episodes.csv.gz" \
    --failed "$OUT/evaluation/failed.json" --out "$OUT/evaluation"
fi
log "campaign complete; held-out evaluation needs separate approval"
