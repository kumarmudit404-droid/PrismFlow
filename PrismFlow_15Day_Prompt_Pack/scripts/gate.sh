#!/usr/bin/env bash
# gate.sh - verification gate to run BETWEEN parts.
# This is the thing actually worth automating. Pasting a prompt takes 3 seconds;
# catching a silently broken repo on day 14 costs the project.
set -uo pipefail
FAIL=0
say() { printf '%-42s %s\n' "$1" "$2"; }

echo "=============================================="
echo " PRISMFLOW GATE"
echo "=============================================="

# 1. tests
if pytest -q >/tmp/pf_test.log 2>&1; then say "unit + integration tests" "PASS"
else say "unit + integration tests" "FAIL"; tail -30 /tmp/pf_test.log; FAIL=1; fi

# 2. package imports
if python -c "import prismflow" >/dev/null 2>&1; then say "package imports" "PASS"
else say "package imports" "FAIL"; FAIL=1; fi

# 3. no placeholder / fabricated numbers left in code
if grep -rInE "TODO: *(fill|replace) *(in )?(number|result)|FIXME_RESULT|PLACEHOLDER_RESULT" \
     --include="*.py" --include="*.md" . >/tmp/pf_placeholder.log 2>&1; then
  say "no fabricated placeholders" "FAIL"; cat /tmp/pf_placeholder.log; FAIL=1
else say "no fabricated placeholders" "PASS"; fi

# 4. suspicion detector isolation (Part 10 onward)
if [ -f prismflow/statistics/suspicion.py ]; then
  if grep -inE "def .*\(.*(attack|compromis|adversar|ground_truth|is_attacked)" \
       prismflow/statistics/suspicion.py >/dev/null 2>&1; then
    say "suspicion detector isolation" "FAIL - receives attack metadata"; FAIL=1
  else say "suspicion detector isolation" "PASS"; fi
fi

# 5. encoders must not share weights by default
if [ -f prismflow/models/encoders.py ]; then
  if grep -q "share_weights.*=.*True" prismflow/models/encoders.py; then
    say "encoders not weight-shared" "FAIL - default is True"; FAIL=1
  else say "encoders not weight-shared" "PASS"; fi
fi

# 6. git scope check - what changed since last commit
CHANGED=$(git status --porcelain 2>/dev/null | wc -l)
say "files changed since last commit" "$CHANGED  (review before committing)"
git status --porcelain 2>/dev/null | head -25

echo "----------------------------------------------"
if [ "$FAIL" -eq 0 ]; then
  echo " GATE PASSED - safe to start the next Part."
  echo " Commit now:  git add -A && git commit -m 'part NN'"
else
  echo " GATE FAILED - fix before starting the next Part."
  echo " Do NOT let a failure propagate. A broken Part 05"
  echo " silently corrupts Parts 06-16."
fi
exit $FAIL
