---
description: PrismFlow Part 38 - LANL redesign and retrain (L24, L26, L27, L28 all arms, L30 tests C, D, reduced B)
---

PART 38 - LANL REDESIGN AND RETRAIN
===================================

PRECHECK: Part 34 (prismflow/training/, L25) and Part 35 must be merged into main; otherwise STOP
and say which is missing. data/raw/ LANL files present and passing `gzip -t`.

READ FIRST: .claude/commands/limitations-index.md, experiments/lanl/README.md,
configs/data/lanl.yaml, experiments/lanl_reanalysis/README.md, docs/KNOWN_LIMITATIONS.md L23-L30.
Use the prismflow-honest-reporting skill.

STANDING RULES: branch part38-lanl-v2 from main; commit per step; do not push. Raw files read-only
and streamed, never loaded whole. Train with prismflow/training/ (Part 34), not a new mirror.
Hyperparameters, days, split and window are Part 30's, so that the SAMPLE DESIGN is the only
change. 5 seeds. No LANL link or token in any file. Pre-register the design, the split check, every
hypothesis and every decision rule in configs/data/lanl_v2.yaml and experiments/lanl_v2/config.yaml,
and commit them BEFORE any training.

YOU MAY CREATE ONLY
  configs/data/lanl_v2.yaml ; prismflow/data/lanl_cyber1_v2.py (new file; Part 30's loader is not edited)
  experiments/lanl_v2/** ; results/lanl_v2/** ; tests/data/test_lanl_v2.py

DESIGN (addresses L24 and L27):
 - Controls part 1: every benign cell of every attack host, in both blocks (or a pre-registered
   share if that exceeds a stated size cap), so each attack host has a within-host benign contrast.
 - Controls part 2: a hash sample of benign cells drawn from the union of proc, flows and dns
   occupancy ONLY, so control selection does not reference the auth stream.
 - State the case-control weights of each part.

RUNS: four arms x two systems x 5 seeds, in TWO execution modes (12-thread sequential and 1-thread
parallel), so every reported std covers mode as well as seed (L28).

ANALYSES: ROC AUC and paired difference with computer-cluster bootstrap CIs; seen and unseen attack
computers against total_records_across_views, with intervals (L26); within-host AUC on attack hosts,
attack against benign cells of the same host (L27); auth-presence AUC under the new design (L24).

LANL TESTS (L30), importing Part 29's functions, never re-implementing them:
 - Test D: masked-view detector AUC.
 - Test C, naturally dependent half: detector on the most and least dependent pairs.
 - Test B, REDUCED and labelled "reduced grid, not Part 29's full grid": with_auth arm, k in {1, 2},
   epsilon 1.0, beta in {0, 5}. Note L21 (fixed epsilon) beside it.

BUDGET: time one unit first. If the projected total exceeds 5 hours, run in this order and mark
the rest OPEN: (1) new design, with_auth and without_auth, both modes; (2) the other two arms;
(3) test D; (4) test C; (5) reduced test B.

README: numbers, rules, verdicts, proposed status per limitation; proposals to
docs/proposed_limitations_update.md. GATE: pytest -q, check_secrets.py --all, push scan; stop and
report on any match. Do not merge.
End with "PART 38 STOPPED: paste experiments/lanl_v2/README.md into the review chat."
