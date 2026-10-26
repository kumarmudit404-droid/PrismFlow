---
description: PrismFlow Part 36 - synthetic estimator and detector checks (L2, L9, L10, L11, L22)
---

PART 36 - SYNTHETIC ESTIMATOR AND DETECTOR
==========================================

READ FIRST: .claude/commands/limitations-index.md, docs/KNOWN_LIMITATIONS.md L2, L9, L10, L11,
L22, experiments/eniv_validation/run_validation.py, experiments/tail/README.md,
experiments/comparison/README.md. Use the prismflow-honest-reporting skill.

STANDING RULES: branch part36-synthetic from main; commit per step; do not push. Frozen modules
import-only; earlier experiments' code is imported, never edited. 5 seeds, mean +/- sample std,
paired within seed. Pre-register each analysis in experiments/part36_synthetic/config.yaml and
commit it BEFORE running anything.

YOU MAY CREATE ONLY
  experiments/part36_synthetic/** (runner, config.yaml, README.md)
  results/part36_synthetic/**
  tests/unit/test_part36_synthetic.py

REPRODUCTION FIRST. Re-run the committed 4-view validation point and the committed Part 10
clone_k2 and Part 13 tail AUC with the original settings. If any differs from its committed value
beyond the reported rounding, STOP and report: nothing after it is interpretable.

A. L9 and L2: the analytic-rho generator from experiments/eniv_validation at 6, 8 and 16 nominal
   views (plus the committed 4). Per view count: ENIV correlation and mean absolute error against
   the analytic effective view count, and mean residual per-pair dependence (L2).
B. L10: batch-size sweep {64, 128, 256, 512, 1024, 2048} at fixed rho. Separate the sample-count
   component (does the bias shrink like 1/batch?) from the structural floor.
C. L11: Part 13's detection arm at 4x the split size. L11's own rule decides: if tail AUC stays
   within 0.4846 to 0.5856, the negative result is substantive, not underpowered.
D. L22: an equal-width clone comparison. Pre-register the design, for example a clean reference
   of 6 independent views against 4 base views plus 2 clones, so both conditions have the same
   view count. Report detector AUC for both at equal width.

BUDGET: time one seed of each analysis first. If an analysis projects past 90 minutes, run its
smallest pre-registered subset, mark the remainder OPEN, and say so.

README: numbers, rule, verdict, proposed status per limitation; proposals to
docs/proposed_limitations_update.md. GATE: pytest -q, check_secrets.py --all, push scan; stop and
report on any match. Do not merge.
End with "PART 36 STOPPED: paste experiments/part36_synthetic/README.md into the review chat."
