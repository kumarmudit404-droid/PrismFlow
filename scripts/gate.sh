#!/usr/bin/env bash
# gate.sh - verification gate to run BETWEEN parts.
# This is the thing actually worth automating. Pasting a prompt takes 3 seconds;
# catching a silently broken repo on day 14 costs the project.
set -uo pipefail
FAIL=0
say() { printf '%-42s %s\n' "$1" "$2"; }

# Prefer the project venv: a bare `pytest`/`python` resolves to a system
# interpreter without the project's deps, which reports FAIL for a healthy repo.
if [ -x .venv/Scripts/python.exe ]; then PY=.venv/Scripts/python.exe
elif [ -x .venv/bin/python ]; then PY=.venv/bin/python
else PY=python
fi

echo "=============================================="
echo " PRISMFLOW GATE"
echo "=============================================="

# 1. tests
if "$PY" -m pytest -q >/tmp/pf_test.log 2>&1; then say "unit + integration tests" "PASS"
else say "unit + integration tests" "FAIL"; tail -30 /tmp/pf_test.log; FAIL=1; fi

# 2. package imports
if "$PY" -c "import prismflow" >/dev/null 2>&1; then say "package imports" "PASS"
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
  # Match the annotated parameter default only. A looser pattern also matches
  # the docstring text warning against share_weights=True.
  if grep -qE "share_weights[[:space:]]*:[[:space:]]*bool[[:space:]]*=[[:space:]]*True" \
       prismflow/models/encoders.py; then
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
