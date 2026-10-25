---
description: PrismFlow Part 35 - LANL re-analysis from saved scores (L23, L26, L27, L28 with_auth, L29), no training
---

PART 35 - LANL RE-ANALYSIS FROM SAVED SCORES
============================================

READ FIRST: .claude/commands/limitations-index.md, experiments/lanl/README.md (sections 7 and 11),
docs/KNOWN_LIMITATIONS.md L23, L26-L29, docs/EVALUATION_PROTOCOL.md (Brier is the two-class
form). Use the prismflow-honest-reporting skill.

STANDING RULES: branch part35-lanl-reanalysis from main; commit per step; do not push. NO
TRAINING. results/lanl/ is read-only; verify the sha256 of samples.npz and every scores/*.npz
against results/lanl/results_manifest.json before reading, and stop on a mismatch.

YOU MAY CREATE ONLY
  experiments/lanl_reanalysis/run.py, config.yaml, README.md
  results/lanl_reanalysis/**
  tests/unit/test_lanl_reanalysis.py

THE DATA YOU HAVE: test-block scores only (computer, window_start, label, prob_attack, vacuity)
per arm, system and seed; samples.npz for view masks, blocks and the train block. There are NO
train-block scores. Anything that needs a fit is cross-fitted over test computers (two folds by a
fixed hash of the computer name) and labelled DIAGNOSTIC. It is never a deployable recalibration.

STEP 1 - PRE-REGISTER. Write config.yaml with every analysis, metric, threshold, bootstrap
setting and decision rule below, and commit it BEFORE computing any number.

STEP 2 - L29: why two arms have Brier above the constant baseline (0.0428, two-class form).
Per arm (without_auth, without_flows, and with_auth and without_dns as references), per system,
5 seeds, mean +/- std:
 a. Brier, reliability and resolution on rows with at least one present view for that arm,
    against rows whose view mask is all False for that arm.
 b. Prior-shift correction from train prevalence to test prevalence. ORACLE: it uses the test
    prevalence; say so.
 c. Cross-fitted Platt recalibration by computer fold. DIAGNOSTIC.
Decision rule: a cause "accounts for" the failure only if correcting it alone brings Brier below
0.0428 in 5 of 5 seeds for both systems. Report each cause's verdict, including "none does".

STEP 3 - L23: metrics at the population rate. Inverse-probability weights (attack 1, benign 512,
the declared 1/512 hash) give population-weighted precision, recall and false-positive rate at
the default threshold and at the thresholds that yield 1, 10 and 100 false alarms per test day,
plus weighted Brier. State that the weights reconstruct the occupied-cell population only as far
as the hash is uniform, and show the population cell bounds from results/lanl/population.json.

STEP 4 - L26: intervals. Computer-cluster bootstrap (2000 replicates, seed 12345, pre-registered)
for each arm and system AUC on seen and unseen attack computers, for the scalar
total_records_across_views (recompute it from samples.npz with Part 30's train-fixed orientation
and confirm the test AUC 0.6843 before continuing), and for the paired model-minus-scalar
difference on unseen computers.

STEP 5 - L27: leave-one-host-out for the top 5 attack computers by test count. AUC change per
arm and system.

STEP 6 - L28, with_auth only. with_auth was run in two execution modes (results.run1.json.gz and
results.json). Report the per-seed std within each mode, the spread across modes, and the pooled
spread, per metric. The other arms stay OPEN until Part 38.

README: per analysis, the numbers, the pre-registered rule, the verdict, and the proposed status
of each limitation (RESOLVED / CLOSED / OPEN). Write the proposals to
docs/proposed_limitations_update.md. Do not edit KNOWN_LIMITATIONS.md.

GATE: pytest -q (tests cover the weighting, the cross-fit fold assignment and the bootstrap
helper), check_secrets.py --all, push scan; stop and report on any match. Do not merge.
End with "PART 35 STOPPED: paste experiments/lanl_reanalysis/README.md into the review chat."
