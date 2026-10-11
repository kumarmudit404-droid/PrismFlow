---
description: PrismFlow Part 30 - LANL cyber1 loader and repeat of Parts 28-29
argument-hint: [day range e.g. 1-7]
---

PART 30 - LANL CYBER1 (SECOND REAL DATASET)
===========================================

Days to use: $ARGUMENTS (if empty, use days 1-7 and say so). Red-team events must fall inside
the chosen range for the test split; check this first and report.

PRECHECK: data/raw/ must hold finished auth, proc, flows, dns, redteam .txt.gz files.
Any file still ending in .fdmdownload or failing `gzip -t` -> STOP and report which.

READ FIRST: docs/CONTRACT.md, docs/KNOWN_LIMITATIONS.md, experiments/ait_clean/README.md,
experiments/ait_attack/README.md. Use the prismflow-honest-reporting skill.

STANDING RULES: frozen modules import-only; 5 seeds; no fabricated numbers; stream the gz files
(about 1.6 billion events total) - never load a whole file into memory; branch part30-lanl;
commit per step; do not push.

YOU MAY CREATE ONLY
  prismflow/data/lanl_cyber1.py
  configs/data/lanl.yaml
  tests/data/test_lanl_cyber1.py
  experiments/lanl/run_lanl.py, config.yaml, README.md
  results/lanl/**

STEPS
1. Views: auth, proc, flows, dns. Sample = (computer, time window); window from config,
   default = the window chosen for AIT.
2. Labels from redteam.txt. LEAKAGE: red-team events are drawn from auth data, so run every
   experiment twice - WITH and WITHOUT the auth view - and report both.
3. Split by TIME with a gap between train and test; confirm both splits contain attack windows,
   otherwise STOP and report.
4. Repeat Part 28's metrics and Part 29's tests A-D with identical settings.
5. Cross-dataset table in the README: synthetic vs AIT vs LANL for the DETECTION claim and the
   honest-duplication false positive. Every cell cites a results/ file or says "not measured".

STOP - REVIEW GATE
End with: "PART 30 STOPPED: paste experiments/lanl/README.md into the review chat."
