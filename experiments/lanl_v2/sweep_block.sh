#!/usr/bin/env bash
# One time-boxed block of the Part 38 mode-B (default) sweep.
#
# The `sweep` subcommand is already resumable, so a block is just a sequence of
# single-unit sweep calls plus a deadline check BEFORE each one: a unit that
# cannot finish inside the budget is never started, so no partial unit is
# thrown away. Predicted unit cost is the max observed in this block, floored
# at 1524 s -- the slowest per-unit training measured in
# results/lanl_v2/sweep_estimate.json.
#
# usage: sweep_block.sh <budget_seconds> <tag> <logfile> <seed>...
set -u
BUDGET=$1; TAG=$2; LOG=$3; shift 3
SEEDS=("$@")
PY=./.venv/Scripts/python.exe
export PYTHONPATH=.
START=$(date +%s)
PREDICT=1524
echo "block_start_epoch=$START budget=$BUDGET" | tee -a "$LOG"
for arm in without_auth with_auth without_flows without_dns; do
  for system in naive prismflow; do
    for seed in "${SEEDS[@]}"; do
      name="${arm}_${system}_seed${seed}"
      if [ -f "results/lanl_v2/units_default/${name}.json" ]; then
        echo "SKIP_DONE $name" | tee -a "$LOG"; continue
      fi
      now=$(date +%s); elapsed=$((now-START))
      if [ $((elapsed+PREDICT)) -gt "$BUDGET" ]; then
        echo "NOT_STARTED $name elapsed=${elapsed}s predict=${PREDICT}s budget=${BUDGET}s" | tee -a "$LOG"
        continue
      fi
      echo "START $name elapsed=${elapsed}s" | tee -a "$LOG"
      $PY experiments/lanl_v2/run_lanl_v2.py sweep --tag "$TAG" \
          --arms "$arm" --systems "$system" --seeds "$seed" --mode default 2>&1 | tee -a "$LOG"
      rc=${PIPESTATUS[0]}
      after=$(date +%s); cost=$((after-now))
      [ $cost -gt $PREDICT ] && PREDICT=$cost
      echo "END $name rc=$rc cost=${cost}s" | tee -a "$LOG"
    done
  done
done
echo "block_stop_epoch=$(date +%s) elapsed=$(( $(date +%s)-START ))s" | tee -a "$LOG"
