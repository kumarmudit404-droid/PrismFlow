---
description: PrismFlow Part 39 - AIT breadth (L16, L18, L21, L31)
---

PART 39 - AIT BREADTH
=====================

PRECHECK: Part 34 (prismflow/training/) merged into main; AIT scenarios present in data/raw/ and
passing `unzip -t`. Never download data yourself.

READ FIRST: .claude/commands/limitations-index.md, experiments/ait_clean/README.md,
experiments/ait_attack/README.md, docs/part27-ait-datacard.md, docs/KNOWN_LIMITATIONS.md L16-L21,
L31. Use the prismflow-honest-reporting skill.

STANDING RULES: branch part39-ait-breadth from main; commit per step; do not push. Frozen modules
import-only; prismflow/data/ait_lds.rewindow and Part 29's functions by import. Train with
prismflow/training/. The two directions are never pooled. 5 seeds. Pre-register every analysis
and rule in experiments/part39_ait/config.yaml and commit it BEFORE running anything.

YOU MAY CREATE ONLY
  experiments/part39_ait/** ; results/part39_ait/** ; tests/unit/test_part39_ait.py

A. L16 (cheapest, first): a decision threshold chosen on the TRAINING scenario by a pre-registered
   rule (for example, maximise balanced accuracy on train) and reported as such, per direction,
   beside the argmax-0.5 figures.
B. L31: clean metrics at two more window lengths per direction, chosen from the Part 27 table and
   exact multiples for rewindow. LANL at a second window only if time remains (it needs a rescan).
C. L18: a contaminated-fraction axis, batches with attacked fraction f in {0, 0.1, 0.25, 0.5, 1.0},
   and a within-batch detector AUC that separates cases, not runs.
D. L21: epsilon scaled per view by that view's standard deviation, on a pre-registered subset of
   Part 29's chorus grid, so views are attacked at matched effort.
E. Only if the user has placed more AIT scenarios in data/raw/: add a third scenario as a further
   direction (L16, L17). Otherwise report "no extra scenario present".

BUDGET: time one seed per block; run in the order A, B, C, D; if the projected total exceeds
4 hours, mark the remainder OPEN.

README: numbers, rules, verdicts, proposed status per limitation; proposals to
docs/proposed_limitations_update.md. GATE: pytest -q, check_secrets.py --all, push scan; stop and
report on any match. Do not merge.
End with "PART 39 STOPPED: paste experiments/part39_ait/README.md into the review chat."
