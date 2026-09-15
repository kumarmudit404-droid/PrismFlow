---
description: PrismFlow Part 05
---

PART 05 - DEPENDENCE ENGINE, ENIV, DISCOUNT, AND ESTIMATOR VALIDATION
======================================================================

Read docs/CONTRACT.md, prismflow/models/prismflow.py, prismflow/data/synthetic.py.

YOU MAY CREATE ONLY THESE PATHS
  prismflow/statistics/__init__.py
  prismflow/statistics/dependence.py
  prismflow/eniv/__init__.py
  prismflow/eniv/eniv.py
  prismflow/eniv/discount.py
  experiments/eniv_validation/run_validation.py
  experiments/eniv_validation/plot_validation.py
  tests/unit/test_dependence.py
  tests/unit/test_eniv.py
  tests/unit/test_discount.py

You may modify prismflow/models/prismflow.py ONLY to wire the discount path
behind the existing use_discount flag and to populate the dependence_matrix and
eniv fields. Do not change the public return signature.

DEPENDENCE ENGINE (statistics/dependence.py)

  Input:  per-view evidence [B, V, K], view_mask [B, V]
  Method: Pearson correlation between view evidence vectors, computed on
          CLASS-CONDITIONAL RESIDUALS. Subtract the per-predicted-class mean
          before correlating.

  Why conditioning matters - record this in a comment: views SHOULD agree when
  they are all correct, because that is the task working. What we need to detect
  is agreement BEYOND what the label explains. Unconditioned correlation
  confuses a well-functioning system with a redundant one.

  Also implement distance correlation (dCor), selectable by config. Pearson
  detects only linear dependence; dCor is zero if and only if the variables are
  independent.

  Output: symmetric [V, V] matrix, diagonal 1.0, unavailable views excluded.

ENIV (eniv/eniv.py)

  Design effect:   n_eff = n / (1 + (n - 1) * rho_bar)
  rho_bar = mean off-diagonal dependence over AVAILABLE views only.
  Return: nominal_views, effective_views, efficiency_ratio, mean_dependence.
  Constraint 1.0 <= effective_views <= available_views. Clamp and test.

  ENIV IS NOT TRAINABLE. No nn.Module. No parameters. No gradients.

DISCOUNT (eniv/discount.py)

  Shafer discounting with alpha = n_eff / n:
      b_k' = alpha * b_k
      u'   = 1 - alpha * (1 - u)
  Assert the simplex constraint still holds. Apply BEFORE Dempster fusion.

  In V1, compute ENIV under torch.no_grad() and detach it.
  Record this reasoning in a code comment, because it will be questioned:
  if the encoder can reduce the loss by lowering MEASURED dependence, gradient
  descent will learn to add orthogonal noise to the evidence vectors rather than
  learn genuinely independent representations. Measured correlation falls, ENIV
  rises, the discount vanishes, and the model becomes overconfident again while
  displaying an excellent independence score. A measure stops being a measure the
  moment it becomes a target. Part 12 addresses this properly with two-timescale
  training; V1 sidesteps it with a stop-gradient.

ESTIMATOR VALIDATION - THE REASON PART 02 EXISTS

  Generate synthetic data at rho in {0.0, 0.2, 0.4, 0.6, 0.8, 0.95}, 5 seeds each.
  For each, compare estimated ENIV against analytic_n_eff(rho, n_views).

  Write:
      results/eniv_validation/estimator_vs_truth.csv
      results/eniv_validation/estimator_vs_truth.png   (with error bars)
  Report mean absolute error and standard deviation.

  IF THE ESTIMATOR DOES NOT TRACK THE ANALYTIC VALUE:
  report the numbers plainly and STOP. Do not proceed to Part 06.
  Every downstream claim depends on this one plot.

TESTS
  - dependence matrix is symmetric with unit diagonal
  - identical views give dependence near 1.0
  - independent views give dependence near 0.0
  - ENIV of 5 identical views is near 1.0
  - ENIV of 5 independent views is near 5.0
  - discount preserves the simplex constraint
  - discount with alpha=1.0 is an identity operation

STOP after the validation plot is written.
