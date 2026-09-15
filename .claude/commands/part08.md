---
description: PrismFlow Part 08
---

PART 08 - MISSING AND NOISY VIEW EXPERIMENTS
=============================================

Read docs/CONTRACT.md and prismflow/evaluation/protocol.py.

YOU MAY CREATE ONLY THESE PATHS
  experiments/robustness/run_missing.py
  experiments/robustness/run_noise.py
  experiments/robustness/plot_robustness.py
  experiments/robustness/config.yaml
  experiments/robustness/README.md

MISSING VIEW EXPERIMENT
  Missing rates: 0%, 10%, 30%, 50%, 70%.
  Compare naive fusion vs PrismFlow.
  Metrics: accuracy, ECE, Brier reliability, confidence, uncertainty, ENIV.
  Headline plot: ECE vs missing rate, both systems on one axis with error bars.

NOISE EXPERIMENT
  Conditions: one noisy view, two noisy views, one severely corrupted view.
  Sweep sigma across at least 5 levels.
  Same metric set.

SECONDARY HYPOTHESIS TO TEST HERE
  Does confidence fall APPROPRIATELY as views are lost?
  A system whose confidence does not drop when half its inputs vanish is
  miscalibrated regardless of its accuracy. Plot confidence vs missing rate and
  comment on whether the slope is plausible.

THIRD HYPOTHESIS - OPTIONAL BUT VALUABLE
  If an imputation baseline is cheap to add (reconstruct a missing view as the
  mean of available views), run it. Prediction worth testing: imputation-based
  fusion shows ECE RISING with missing rate even where accuracy improves,
  because a view reconstructed from the others is maximally dependent on them by
  construction and therefore adds belief mass without adding information.
  If this reproduces it is a genuinely novel observation. If it does not, say so.

PROTOCOL
  5 seeds per condition. Mean and standard deviation on every value.
  Never claim superiority unless the error bars do not overlap. State explicitly
  in the README any condition where PrismFlow is not better.

STOP after plots are written to results/robustness/.
