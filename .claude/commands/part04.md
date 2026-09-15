---
description: PrismFlow Part 04
---

PART 04 - EVIDENTIAL HEAD, DEMPSTER FUSION, NAIVE BASELINE
===========================================================

Read docs/CONTRACT.md and prismflow/models/encoders.py.

YOU MAY CREATE ONLY THESE PATHS
  prismflow/models/evidence.py
  prismflow/models/fusion.py
  prismflow/models/prismflow.py
  prismflow/train.py
  tests/unit/test_evidence.py
  tests/unit/test_fusion.py

EVIDENTIAL HEAD (prismflow/models/evidence.py)

  evidence e = softplus(Linear(features))
  Use softplus, NOT ReLU. ReLU produces dead units with exactly zero gradient,
  and a dead evidence unit silently pins a class to zero belief forever.

  alpha = e + 1
  S     = alpha.sum(-1)
  belief      b_k = e_k / S
  uncertainty u   = K / S
  Assert b.sum() + u == 1 within 1e-5.

  LOSS - Type II maximum likelihood:
      L_edl = sum_k y_k * (log S - log alpha_k)

  KL REGULARISER WITH ANNEALING:
      alpha_tilde = y + (1 - y) * alpha
      L_kl = lambda_t * KL( Dir(alpha_tilde) || Dir(1) )
      lambda_t = min(1.0, epoch / anneal_epochs), anneal_epochs default 10

  CRITICAL - lambda_t MUST start at exactly 0.0.
  If the KL term runs at full strength from epoch zero, the model collapses to
  maximum vacuity - it outputs "I know nothing" for every input - and never
  recovers. This is the single most common EDL implementation failure.
  Add a test asserting lambda_t == 0.0 at epoch 0.

DEMPSTER COMBINATION (prismflow/models/fusion.py)

  Combine opinions (b1, u1) and (b2, u2):
      C   = sum over i != j of  b1_i * b2_j          (conflict mass)
      b_k = (b1_k*b2_k + b1_k*u2 + b2_k*u1) / (1 - C)
      u   = (u1 * u2) / (1 - C)
  Fold sequentially across all AVAILABLE views only.

  NUMERICAL STABILITY - clamp (1 - C) to a minimum of 1e-6.
  As conflict approaches 1 the denominator approaches zero and produces NaN.
  Add a test that feeds two maximally conflicting opinions and asserts the
  output is finite and still satisfies the simplex constraint.

MODEL (prismflow/models/prismflow.py)

  Class PrismFlow with flag use_discount, default False in this Part.
  With use_discount=False this IS the naive baseline:
      encoders -> evidence -> Dempster fusion -> prediction

  Return a dataclass with fields:
      prediction, probs, confidence, uncertainty,
      per_view_evidence, per_view_belief, per_view_uncertainty, view_mask,
      dependence_matrix=None, eniv=None
  Keep the last two fields present but None. Part 05 populates them without
  changing this signature.

TESTS
  - beliefs plus uncertainty sum to 1 for every sample
  - fusing two identical confident opinions reduces uncertainty
  - fusing two maximally conflicting opinions returns finite values
  - after training on synthetic data, mean vacuity is NOT pinned near 1.0

VERIFY
  Train on synthetic data at rho=0.3 with 4 views, 5 seeds.
  Report accuracy and mean confidence with standard deviation.
  Write to results/baseline/.

STOP. Do not implement dependence, ENIV, or discounting.
