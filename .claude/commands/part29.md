---
description: PrismFlow Part 29 - chorus attack, detector, honest-duplication and missingness tests on AIT
argument-hint: [window-seconds - same as Part 28]
---

PART 29 - ATTACK AND DETECTION ON REAL DATA
===========================================

Window and sample mode: $ARGUMENTS (must equal Part 28's). If results/ait_clean/ is missing, STOP.

READ FIRST: docs/CONTRACT.md, docs/KNOWN_LIMITATIONS.md (L3, L4, L5, L8),
experiments/chorus/README.md, experiments/comparison/README.md, experiments/ait_clean/README.md.
Use the prismflow-honest-reporting skill.

SCOPE NOTE: controlled research simulation against our own trained models on a public dataset.
No live systems, no network traffic, no intrusion tooling.

STANDING RULES: prismflow/ (including attacks/ and statistics/suspicion.py) FROZEN - import only;
5 seeds, mean +/- sample std, paired within-seed; the detector never sees attack labels;
no fabricated numbers; branch part29-ait-attack; commit per step; do not push.

YOU MAY CREATE ONLY
  experiments/ait_attack/run_ait_attack.py
  experiments/ait_attack/config.yaml
  experiments/ait_attack/README.md
  results/ait_attack/**
  tests/unit/test_ait_attack.py

TESTS TO RUN (same trained models and seeds as Part 28)
A. Natural attack windows: ENIV and mean pairwise dependence on real attack windows vs benign
   windows. No synthetic perturbation.
B. Chorus attack: k in {1,2,3}, Part 09's epsilon and beta grid. Features here are on a different
   scale than synthetic - state any epsilon rescaling and why. Report attack success
   (naive vs PrismFlow, paired), ENIV clean vs attacked, dependence among compromised views,
   detector ROC AUC per k.
C. Honest duplication: detector scores on CLEAN data for the same_sensor_group="network" pair
   vs unrelated pairs. Compare with Part 10's clone_k2 false-positive finding (L5).
D. Missingness: 30% and 50% views masked, detector AUC, as in Part 10.

VERDICT SECTION (README)
For each of L3, L4, L5, L8 write exactly one of CONFIRMS / CONTRADICTS / NO EVIDENCE,
followed by the numbers and their results/ paths. Do not edit KNOWN_LIMITATIONS.md;
propose text for the user to approve.

STOP - REVIEW GATE
End with: "PART 29 STOPPED: paste experiments/ait_attack/README.md into the review chat."
