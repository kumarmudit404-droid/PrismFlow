---
description: PrismFlow Part 28 - train naive fusion and PrismFlow V1 on AIT real data (clean)
argument-hint: [window-seconds] [sample-mode network|host]  e.g. 300 network
---

PART 28 - PRISMFLOW ON REAL DATA (CLEAN)
========================================

Window: $0 seconds. Sample mode: $1 (default network if omitted).
If the window is empty, STOP: the user chooses it after Part 27.

DIRECTIONS - run BOTH, report them SEPARATELY, never pool:
  A: train russellmitchell -> test santos
  B: train santos -> test russellmitchell
Do not ask; this is decided. Neither direction is "the" result. Two directions are two
measurements on two different test sets with different prevalences, so a mean over them is
meaningless and hides whichever one is worse. Every table, plot, filename and summary sentence
carries its direction label, and no number is averaged, concatenated or otherwise combined
across A and B.

READ FIRST: docs/CONTRACT.md, docs/KNOWN_LIMITATIONS.md, docs/EVALUATION_PROTOCOL.md,
docs/part27-ait-datacard.md, results/ait_window/window_table.md, and the training entry points
already used by experiments/real/ (Part 15). Use the prismflow-honest-reporting skill.

STANDING RULES: prismflow/ and v2/ FROZEN (import only); 5 seeds minimum, mean +/- sample std,
per-seed values saved; paired within-seed comparisons; no fabricated numbers; branch
part28-ait-clean; commit per step; do not push.

YOU MAY CREATE ONLY
  experiments/ait_clean/run_ait_clean.py
  experiments/ait_clean/config.yaml
  experiments/ait_clean/README.md
  results/ait_clean/**
  tests/unit/test_ait_clean.py

WHAT THE LABEL MEANS (state this in every artefact)
- A window is attack=1 if ANY officially labelled line falls in it, and on both scenarios that
  label is dominated by one continuously-running DNS exfiltration tool: 1,026 of 1,031 attack
  windows on russellmitchell and 663 of 667 on santos carry only dnsteal-family steps
  (docs/part26-ait-loader.md sections 0 and 12).
- So the task is "was the exfiltration tool running in this window", NOT "detect an APT" and not
  "detect a multi-step intrusion". Write that sentence in experiments/ait_clean/README.md and in
  every results/ait_clean/*.md. A result described as intrusion, APT or campaign detection is a
  misreport even when every number in it is correct.
- The multi-step intrusion is 5 windows on russellmitchell and 3 on santos. Any claim about
  THOSE is computed on those windows, counted, and labelled as a handful-of-windows result.

SPLIT (most important rule)
- Train and test always come from DIFFERENT scenarios: direction A trains on russellmitchell and
  tests on santos, direction B trains on santos and tests on russellmitchell. Neighbouring
  windows of one scenario leak into each other, so a random split within a scenario is forbidden
  in both directions.
- If only one processed scenario exists: STOP and tell the user to download and run /part26 on a
  second scenario. Do not fall back to a random split and do not run a single direction.
- Feature scaling is fit on the training scenario only - refit per direction, never shared.

RUN
1. Systems: naive fusion and V1 PrismFlow (default config), same encoders/heads as Part 15,
   K = 2 (benign / attack). Seeds 0-4. Both directions, every seed, same code path.
2. Metrics, per seed and summarised (mean +/- sample std over the 5 seeds), per direction:
   - balanced accuracy
   - ROC AUC
   - macro-F1
   - attack-class recall
   - Brier, ECE (15 bins), AURC
   - ENIV and the full [V,V] dependence matrix
   - the MAJORITY-CLASS BASELINE on that direction's test set, for every metric above that
     admits one, computed by the same code as the systems' metrics
   ACCURACY ALONE MUST NEVER APPEAR. Plain accuracy may be printed only inside a table that also
   carries balanced accuracy, ROC AUC and the majority-class baseline; it is never a headline,
   never a sentence on its own, and never the quantity a comparison is made on. The test sets are
   90.0% attack (russellmitchell) and 60.2% attack (santos) among labelled windows, so an
   accuracy figure on its own describes the prevalence and not the system.
   PREVALENCE BESIDE EVERY METRIC: each reported number carries, in the same table row or
   caption, its test set's n, attack count, benign count and attack prevalence. A metric without
   its prevalence beside it is not reportable - not in JSON, not in a README, not in chat.
   Only label_known samples are counted anywhere; labels == -1 is never read as benign.
3. TEST-SET STRUCTURE, reported per direction: the number of INDEPENDENT CONTIGUOUS ATTACK
   BLOCKS in the test set - maximal runs of consecutive attack windows on that scenario's time
   grid - with each block's length in windows and its start time, plus the same for benign runs.
   Measure it at the window length and sample mode actually used; do not carry over Part 26's
   300 s network figures (santos 3 attack blocks of 664/2/1 windows; russellmitchell one long
   block). State the consequence beside the count: the block count, not the attack-window count,
   is the positive class's effective sample size, and every +/- in this Part is a spread over
   seeds, NOT over independent attacks.
4. Structure check: dependence for the same_sensor_group="network" pair versus the mean of
   unrelated pairs, per seed, paired difference reported, per direction.
5. If any view is missing in more than 50% of windows, also run a second configuration without it
   and keep both. Missingness is measured on each direction's TRAIN and TEST set; report both.
6. Write experiments/ait_clean/README.md: what was run, both splits, the label-meaning sentence
   above, every number with its results/ file path, its direction and its prevalence, and a
   "differs from synthetic" list. Do NOT edit KNOWN_LIMITATIONS.md; put a proposed entry in the
   README for the user to approve.

TESTS: each direction's split has no shared scenario; the two directions' test sets are never
concatenated or averaged; scaler fit only on that direction's train scenario; results JSON
carries both directions with 5 seeds per metric per direction, and balanced accuracy, ROC AUC,
macro-F1, attack recall, majority-class baseline and prevalence present for each; attack-block
count present per direction; no metric is computed over labels == -1.

DONE WHEN pytest -q is green and the final message lists files, tests, git diff --stat and ONE
SUMMARY TABLE PER DIRECTION, each carrying its prevalence, its majority-class baseline and its
attack-block count. No pooled table, and no accuracy figure standing on its own.
