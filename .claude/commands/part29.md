---
description: PrismFlow Part 29 - chorus attack, detector, honest-duplication and missingness tests on AIT
argument-hint: [window-seconds - same as Part 28]
---

PART 29 - ATTACK AND DETECTION ON REAL DATA
===========================================

Window and sample mode: $ARGUMENTS (must equal Part 28's). If results/ait_clean/ is missing, STOP.

READ FIRST: docs/CONTRACT.md, docs/KNOWN_LIMITATIONS.md (L3, L4, L5, L8, L16, L17),
docs/EVALUATION_PROTOCOL.md (the part28_extras exception),
experiments/chorus/README.md, experiments/comparison/README.md, experiments/ait_clean/README.md.
Use the prismflow-honest-reporting skill.

L16 and L17 are not background. L17 says one view solves this label - single_view_dns scores
balanced accuracy 0.9898 on santos and 1.0000 on russellmitchell, no other single view clears
0.71 - so every test below is conditioned on whether dns is involved. L16 says discrimination
and calibration come apart in direction B and the positive class is 3 episodes (santos) and
1 (russellmitchell), so no +/- here is a spread over independent attacks.

SCOPE NOTE: controlled research simulation against our own trained models on a public dataset.
No live systems, no network traffic, no intrusion tooling.

STANDING RULES: prismflow/ (including attacks/ and statistics/suspicion.py) FROZEN - import only;
5 seeds, mean +/- sample std, paired within-seed; the detector never sees attack labels;
no fabricated numbers; branch part29-ait-attack; commit per step; do not push.

METRICS - IMPORT, DO NOT REIMPLEMENT
Balanced accuracy, ROC AUC and per-class recall come from

    from experiments.ait_clean.run_ait_clean import part28_extras

used together with prismflow.evaluation.protocol.compute_metrics, exactly as Part 28 does.
Do NOT write your own balanced accuracy, ROC AUC or recall in experiments/ait_attack/.
docs/EVALUATION_PROTOCOL.md records this as a named exception to the compute_metrics rule and
states that Parts 29 and 30 must share the one function: three experiments each rolling their
own ROC AUC is the divergence that document exists to prevent. If part28_extras is missing or
its signature has changed, STOP and report it - do not reimplement and do not edit Part 28.

YOU MAY CREATE ONLY
  experiments/ait_attack/run_ait_attack.py
  experiments/ait_attack/config.yaml
  experiments/ait_attack/README.md
  results/ait_attack/**
  tests/unit/test_ait_attack.py

TESTS TO RUN (same trained models and seeds as Part 28)

A. Natural attack windows: ENIV and mean pairwise dependence on real attack windows vs benign
   windows. No synthetic perturbation.
   REPORT THE TWO WINDOW POPULATIONS SEPARATELY, NEVER POOLED:
     (i)  dnsteal-only attack windows - the window carries only dnsteal-family steps;
     (ii) multi-step intrusion windows - the handful carrying a step outside that family.
   Take both window sets from results/ait_clean/results.json,
   directions.<name>.test_structure.multi_step_intrusion_windows (the strict definition,
   beyond EXFILTRATION_TOOL_STEPS) and .n_attack_windows_dnsteal_only. Note that
   experiments/ait_clean/config.yaml and the run_ait_clean.py docstring quote a LOOSER count
   for russellmitchell (5, beyond the dnsteal family alone); state which definition you used
   and do not mix the two. Set (ii) is 3 windows on santos and 4 on russellmitchell, so it is a
   handful-of-windows result: DESCRIPTIVE ONLY, no ROC AUC, no significance claim, no
   mean +/- std presented as a finding. PUT THE WINDOW COUNT BESIDE EVERY NUMBER, in the table
   cell itself - "0.41 (n=3)", not "0.41" with the n in a caption.

B. Chorus attack: k in {1,2,3}, Part 09's epsilon and beta grid. Features here are on a different
   scale than synthetic - state any epsilon rescaling and why. Report attack success
   (naive vs PrismFlow, paired), ENIV clean vs attacked, dependence among compromised views,
   detector ROC AUC per k.
   SPLIT EVERY RESULT BY WHETHER dns IS IN THE COMPROMISED SET. dns is the only informative
   view (L17), so "k views compromised" is two different experiments wearing one name:
   compromising dns attacks the only view carrying the label, while compromising a subset of
   apache, auth, audit and suricata perturbs views that score 0.3733 to 0.7061 balanced
   accuracy alone. Report two blocks per k - "dns compromised" and "dns not compromised" -
   each with its own mean +/- std over seeds and its own subset count.
   NEVER POOL ACROSS THE dns SPLIT. No row, mean, table or verdict sentence may average a
   with-dns subset together with a without-dns subset. An attack-success number that does not
   say whether dns was compromised is uninterpretable and must not be written down.
   ENUMERATE ALL SUBSETS. With five views this is 25 subsets over k=1,2,3 - k=1: 1 with dns,
   4 without; k=2: 4 with, 6 without; k=3: 6 with, 4 without. That is affordable, so enumerate
   exhaustively and report the per-subset values, not only the aggregate. If the epsilon/beta
   grid makes 25 subsets x 5 seeds x grid too expensive to finish, you may sample subsets
   within each (k, dns-side) cell - but then SAY SO EXPLICITLY in the README and results JSON:
   which cells were sampled, how many of how many subsets, and by what rule they were chosen.
   Never present a sampled cell as exhaustive. Sampling must be done WITHIN a dns-side cell,
   never across it, and a with-dns cell must never be left empty.

C. Honest duplication: the brief's same_sensor_group="network" pair DOES NOT EXIST on this data.
   sensor_groups() returns {'network': [4]} - one member, suricata - so there is no within-group
   pair, as Part 28's sensor_group_check already records. REPORT "NO EVIDENCE" for strict
   same-sensor pairs and say why; do not substitute another pair and call it a same-sensor
   result.
   ADDITIONALLY, as a different and clearly labelled quantity, score the detector on CLEAN data
   for the two most naturally dependent pairs in each direction against the least dependent
   pairs. Read the pairs from the Part 28 pair table
   (results/ait_clean/results.json, directions.<name>.systems.<system>.pair_table) - do not
   re-derive them. As committed there, the ordering is:
     direction A, test santos:           most  dns-suricata 0.7784 +/- 0.0063,
                                               auth-audit 0.6460 +/- 0.0804
                                         least auth-dns 0.0411 +/- 0.0156,
                                               apache-dns 0.0458 +/- 0.0220
     direction B, test russellmitchell:  most  auth-audit 0.8836 +/- 0.0169,
                                               apache-suricata 0.6575 +/- 0.0353
                                         least auth-dns 0.0112 +/- 0.0237,
                                               auth-suricata 0.0950 +/- 0.0409
   (naive-fusion values; prismflow's ordering agrees - confirm both before relying on it, and if
   the two systems disagree on the ordering, report that instead of picking one.)
   LABEL THIS BLOCK "different sensors, naturally dependent" everywhere it appears - in the
   README heading, the table caption and the results JSON key. These views sit on different
   sensors; the dependence is a property of the scenario, not shared hardware. Calling it a
   same-sensor or honest-duplication result would be a misreport.
   Then compare with Part 10's clone_k2 false-positive finding (L5): clone_k2 scored
   AUC 0.8743 +/- 0.0583, above every chorus condition there (+0.2415 +/- 0.1164 against
   chorus k=2, 5/5 seeds). State whether natural dependence on real data raises the detector
   the way a synthetic clone did, and note that clone_k2 was an exact duplicate while these
   pairs are merely correlated - so a null here does not clear the detector.

D. Missingness: 30% and 50% views masked, detector AUC, as in Part 10.

VERDICT SECTION (README)
For each of L3, L4, L5, L8, L16, L17 write exactly one of CONFIRMS / CONTRADICTS / NO EVIDENCE,
followed by the numbers and their results/ paths. For anything derived from test B, the verdict
must name the dns side it rests on; a verdict that pools the dns split is not a verdict.
Do not edit KNOWN_LIMITATIONS.md; propose text for the user to approve.

STOP - REVIEW GATE
End with: "PART 29 STOPPED: paste experiments/ait_attack/README.md into the review chat."
