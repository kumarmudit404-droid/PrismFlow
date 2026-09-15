---
description: PrismFlow Part 12
---

PART 12 - PER-SAMPLE ENIV AND TWO-TIMESCALE TRAINING
=====================================================

Read docs/CONTRACT.md, prismflow/eniv/eniv.py, prismflow/eniv/discount.py.
This Part is a V2 addition. Preserve the V1 global-ENIV path as the default
fallback. Do not modify V1 behaviour when the flag is off.

YOU MAY CREATE ONLY THESE PATHS
  prismflow/eniv/per_sample.py
  prismflow/eniv/amortized.py
  prismflow/train_two_timescale.py
  experiments/per_sample/run_per_sample.py
  experiments/per_sample/README.md
  tests/unit/test_per_sample.py

MOTIVATION - RECORD THIS IN THE README
  A global correlation matrix over the dataset is a preprocessing step, not a
  model. But dependence is INPUT-CONDITIONAL. Two cameras with disjoint
  viewpoints are independent in general and become perfectly dependent on the one
  frame where both are washed out by the same glare. A firewall sensor and a
  network sensor are independent in general and become one sensor when both read
  from a tap the attacker controls.

  The object we want is therefore n_eff(x), not n_eff.

AMORTIZED ESTIMATOR (eniv/amortized.py)
  A small network psi mapping the tuple of per-view evidence vectors to a
  sample-conditional [V, V] dependence matrix. Keep it lightweight - two or
  three layers. Do not implement MINE, InfoNCE, CLUB, or PID in this Part.

  NOTE FOR LATER WORK, record as a comment: if a future version estimates mutual
  information to penalise dependence, it must use an UPPER bound such as CLUB.
  MINE and InfoNCE are LOWER bounds on MI. Driving a lower bound to zero says
  nothing about the true quantity, so minimising a lower bound is vacuous. This
  error is common in published work.

TWO-TIMESCALE TRAINING (train_two_timescale.py)
  The problem being solved, to record in the file docstring:
  if the amortized estimator is trained jointly with the encoder and its output
  feeds the discount, gradient descent has two ways to reduce loss. It can learn
  genuinely independent representations, or it can add orthogonal noise to the
  evidence so that MEASURED dependence falls while actual redundancy is
  unchanged. The second is far cheaper, so it wins. The measure stops being a
  measure the moment it becomes a target.

  Implement alternating optimisation, as in actor-critic or GAN training:
     - N_enc steps updating the encoder with the estimator FROZEN
     - N_est steps updating the estimator with the encoder FROZEN
  Configurable ratio, default 5:1. Log both losses separately.

HELD-OUT AUDITOR (validation guard)
  Train a second, independent dependence estimator on held-out data and use it
  ONLY to audit - never in the loss. If audited dependence diverges from trained
  dependence, the model is gaming the estimator. Log the gap every epoch and plot
  it. This plot is your evidence that gaming did or did not occur.

COMPARISON EXPERIMENT
  Global ENIV vs per-sample ENIV on: stability across seeds, calibration,
  Chorus-attack detection rate, wall-clock cost.
  Include a condition with input-conditional dependence: generate synthetic data
  where rho is high for 20% of samples and zero for the rest. Global ENIV cannot
  represent this; per-sample ENIV should. That contrast is the point of the Part.

  If per-sample ENIV is unstable, keep V1 global as default and document why.

STOP.
