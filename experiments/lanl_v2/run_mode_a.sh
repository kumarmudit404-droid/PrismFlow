#!/usr/bin/env bash
# Part 38 execution mode A -- `threads1_concurrent11` (PRE_REGISTRATION.md s6).
#
# DO NOT RUN UNTIL MODE B IS AT 40/40. Pre-flight 3 below enforces that; the
# comment is here because the reason is not obvious. Mode A is a SECOND
# measurement of the same 40 units at a different thread count, and section 6
# forbids pooling the two modes into a single +/-. Running it against a partial
# mode B gives nothing to compare it with, and the box cannot host both: mode A
# wants 11 processes and ~5.4 GB while mode B is still training.
#
# WHAT apply_mode DOES AND DOES NOT DO. `--mode threads1` calls
# torch.set_num_threads(1) and RECORDS `"concurrency": 11` in every unit JSON
# and score npz (run_lanl_v2.py:132-147). It does NOT spawn anything: the
# `sweep` subcommand trains its units sequentially in one process. **The
# concurrency is this script's job.** If you run `sweep --mode threads1` on its
# own you get 40 single-thread units one after another, which is a third mode
# that nobody pre-registered.
#
# OUTPUT. The runner derives its directories from the mode name
# (run_lanl_v2.py: `udir, sdir = OUT / f"units_{mode['mode']}", ...`), so
# `--mode threads1` writes results/lanl_v2/units_threads1/ and
# results/lanl_v2/scores_threads1/ with no extra argument. Mode B's
# units_default/ and scores_default/ are never touched.
#
# RESUMABILITY. `sweep` skips a unit whose JSON already exists, so a worker
# that dies can be relaunched and this whole script can be re-run safely. That
# is also what makes the unit-level slicing below correct: two workers that
# somehow raced on one unit would both write the same deterministic result,
# and the second would skip it.
#
# THE COST OF SLICING AT UNIT LEVEL. Each worker invokes the runner once per
# unit, so each invocation re-reads samples_v2_days16.npz (~3 MB). Against
# ~450-1500 s of training per unit that is noise, and it is what buys exactly
# 11 concurrent processes -- the CLI takes arms x systems x seeds as a cross
# product, and 40 units do not slice into 11 any other way.
#
#   usage: ./experiments/lanl_v2/run_mode_a.sh [--force-preflight]
#
# `--force-preflight` skips ONLY pre-flight 2 (other Python processes), for the
# case where something unrelated and known-harmless is running. It does not
# skip the RAM or 40/40 checks.

set -uo pipefail

TAG=v2_days16
MODE=threads1
WORKERS=11
MIN_FREE_GB=7
EXPECTED_B_UNITS=40
PY=./.venv/Scripts/python.exe
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOGDIR="$ROOT/results/lanl_v2/logs"
UNITS_B="$ROOT/results/lanl_v2/units_default"
UNITS_A="$ROOT/results/lanl_v2/units_threads1"
SAMPLES="$ROOT/results/lanl_v2/samples_$TAG.npz"

FORCE_PREFLIGHT=0
for arg in "$@"; do
  case "$arg" in
    --force-preflight) FORCE_PREFLIGHT=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

cd "$ROOT" || { echo "cannot cd to $ROOT" >&2; exit 2; }

abort() { echo "ABORT: $*" >&2; exit 1; }

echo "=== Part 38 mode A pre-flight, $(date -Is) ==="

# -- 1. free RAM ------------------------------------------------------------
# No `free` on this box (MSYS2 over Windows). CIM is the probe. If the probe
# itself fails we abort: an unknown free-RAM figure is not a pass. 11 processes
# at the ~500-523 MB peak RSS mode B measured per unit is ~5.5-5.8 GB, and the
# step-1 estimate puts mode A at 5,372 MB (results/lanl_v2/sweep_estimate.json),
# so 7 GB is the bar with headroom for the OS and a browser.
free_kb="$(powershell.exe -NoProfile -NonInteractive -Command \
  '(Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory' 2>/dev/null \
  | tr -d '\r' | tr -d '[:space:]')"
case "$free_kb" in
  ''|*[!0-9]*) abort "could not read free physical memory (probe returned '${free_kb}'). Refusing to launch 11 processes without knowing. Check by hand and re-run." ;;
esac
free_gb=$(( free_kb / 1048576 ))
echo "pre-flight 1: free RAM ${free_gb} GB (${free_kb} KB), need >= ${MIN_FREE_GB} GB"
[ "$free_gb" -ge "$MIN_FREE_GB" ] || abort "only ${free_gb} GB free, need ${MIN_FREE_GB} GB. Mode A swapping invalidates its own wall-clock and may invalidate nothing else -- but it will take far longer than the ~2.0 h estimate."

# -- 2. nothing else running ------------------------------------------------
# The bracket in [p]ython keeps the pattern from matching this grep, and $$ is
# excluded so the script's own shell never counts itself. A ps|grep pattern
# that matches its own wrapper reads as a false positive and has cost this
# project a wasted relaunch before.
others="$(ps -ef 2>/dev/null \
  | grep -E '[p]ython|[p]ytest' \
  | awk -v self="$$" '$2 != self' \
  | grep -v "run_mode_a" || true)"
if [ -n "$others" ]; then
  echo "pre-flight 2: FAILED -- Python-ish processes are running:" >&2
  echo "$others" >&2
  if [ "$FORCE_PREFLIGHT" -eq 1 ]; then
    echo "pre-flight 2: overridden by --force-preflight. Continuing." >&2
  else
    abort "another Python process is running. If it is mode B, WAIT -- do not run both. If it is something you know is harmless, re-run with --force-preflight."
  fi
else
  echo "pre-flight 2: no other Python or pytest processes"
fi

# -- 3. mode B complete -----------------------------------------------------
[ -d "$UNITS_B" ] || abort "$UNITS_B does not exist. Mode B has not run."
b_units="$(find "$UNITS_B" -maxdepth 1 -name '*.json' | wc -l | tr -d '[:space:]')"
echo "pre-flight 3: mode B has ${b_units} of ${EXPECTED_B_UNITS} units"
[ "$b_units" -eq "$EXPECTED_B_UNITS" ] \
  || abort "mode B is at ${b_units}/${EXPECTED_B_UNITS}. Mode A is the SECOND measurement of the same cells; running it first leaves nothing to compare against, and both at once will not fit in memory."

# -- 4. inputs present ------------------------------------------------------
[ -f "$SAMPLES" ] || abort "$SAMPLES missing. Mode A must train on the SAME sample table as mode B, not a rebuilt one."
[ -x "$PY" ] || abort "$PY not found or not executable."
echo "pre-flight 4: sample table and interpreter present"
echo "pre-flight 4: sample table sha256 $(sha256sum "$SAMPLES" | cut -d' ' -f1)"

a_done="$(find "$UNITS_A" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d '[:space:]')"
echo "pre-flight 5: mode A already has ${a_done} units; those will be skipped"

mkdir -p "$LOGDIR"
echo "=== pre-flight passed. launching ${WORKERS} workers, mode ${MODE} ==="

# -- launch -----------------------------------------------------------------
# 40 triples distributed round-robin, so no worker gets all the slow prismflow
# units: the mode B measurements run 546 s (without_dns/naive) to 1770 s
# (without_dns/prismflow) per unit, and a contiguous slice would finish hours
# apart.
TRIPLES=()
for arm in without_auth without_flows without_dns with_auth; do
  for system in naive prismflow; do
    for seed in 0 1 2 3 4; do
      TRIPLES+=("${arm}:${system}:${seed}")
    done
  done
done
[ "${#TRIPLES[@]}" -eq 40 ] || abort "built ${#TRIPLES[@]} triples, expected 40"

START=$(date +%s)
PIDS=()
for w in $(seq 0 $((WORKERS - 1))); do
  (
    log="$LOGDIR/mode_a_worker${w}.log"
    echo "worker $w start $(date -Is)" > "$log"
    i=$w
    while [ "$i" -lt "${#TRIPLES[@]}" ]; do
      IFS=: read -r arm system seed <<< "${TRIPLES[$i]}"
      echo "--- worker $w unit ${arm}/${system}/seed${seed} $(date -Is)" >> "$log"
      PYTHONPATH=. "$PY" experiments/lanl_v2/run_lanl_v2.py sweep \
        --tag "$TAG" --arms "$arm" --systems "$system" --seeds "$seed" \
        --mode "$MODE" >> "$log" 2>&1
      echo "--- worker $w rc=$? $(date -Is)" >> "$log"
      i=$(( i + WORKERS ))
    done
    echo "worker $w done $(date -Is)" >> "$log"
  ) &
  PIDS+=($!)
  echo "worker $w pid ${PIDS[-1]}"
done

echo "=== ${#PIDS[@]} workers launched. logs: results/lanl_v2/logs/mode_a_worker*.log ==="
echo "progress: ls results/lanl_v2/units_threads1/ | wc -l"
echo "NOTE: python block-buffers stdout when it is not a tty, so a worker log"
echo "      will look stalled. units_threads1/ is the durable record."

fail=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || { echo "worker pid $pid exited non-zero" >&2; fail=1; }
done

ELAPSED=$(( $(date +%s) - START ))
a_units="$(find "$UNITS_A" -maxdepth 1 -name '*.json' | wc -l | tr -d '[:space:]')"
echo "=== mode A finished in ${ELAPSED}s. units_threads1/ holds ${a_units} of 40 ==="

if [ "$a_units" -ne 40 ]; then
  echo "INCOMPLETE: re-run this script to resume -- completed units are skipped." >&2
  exit 1
fi
[ "$fail" -eq 0 ] || { echo "all 40 units present but a worker reported an error; read the logs before using the results." >&2; exit 1; }

echo
echo "Next, and NOT done here:"
echo "  - mode A's own aggregate is written by the runner to"
echo "    results/lanl_v2/sweep_threads1.json after each unit."
echo "  - H3 compares the two modes. fill_analysis.py reads units_default/"
echo "    ONLY and must not be pointed at units_threads1/: pre-registration"
echo "    section 6 forbids pooling the modes into a single +/-."
echo "  - every ENIV figure carries its thread count. Mode A observed 1,"
echo "    mode B observed 8, and a committed eniv_eigen misses its own value"
echo "    by 0.0107 across that change (Part 34)."
