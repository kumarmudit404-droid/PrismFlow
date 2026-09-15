#!/usr/bin/env bash
# run_headless.sh NN [NN ...]
#
# Runs one or more parts non-interactively and STOPS at the first gate failure.
#
# READ THIS BEFORE USING IT:
#   Do NOT chain the experiment parts (06, 09, 13, 14) unattended. Their entire
#   value is that a human looks at the resulting curve. A pipeline that generates
#   the clone plot and never shows it to you has defeated the point of the
#   project. Use this for scaffolding parts (01-05, 07) and sweeps only.
set -uo pipefail

MODEL="${PF_MODEL:-opus}"
MAXTURNS="${PF_MAX_TURNS:-60}"
TOOLS="Read,Write,Edit,Glob,Grep,Bash(pytest *),Bash(python *),Bash(mkdir *),Bash(ls *)"
mkdir -p logs/agent

for num in "$@"; do
  f=$(ls parts/PART_${num}_*.txt 2>/dev/null | head -1)
  if [ -z "$f" ]; then echo "No prompt file for part $num"; exit 1; fi

  echo "=============================================="
  echo " RUNNING PART $num : $(basename "$f")"
  echo "=============================================="

  claude -p "$(cat "$f")" \
    --model "$MODEL" \
    --max-turns "$MAXTURNS" \
    --allowedTools "$TOOLS" \
    --output-format json \
    | tee "logs/agent/part_${num}.json" \
    | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('result','')); sys.exit(1 if d.get('is_error') else 0)"

  if [ $? -ne 0 ]; then
    echo "PART $num reported an error. Stopping."
    exit 1
  fi

  echo
  echo "--- running gate after part $num ---"
  if ! ./scripts/gate.sh; then
    echo "Gate failed after part $num. Stopping the chain."
    exit 1
  fi
  git add -A && git commit -q -m "part $num" && echo "committed part $num"
done
echo "All requested parts completed and gated."
