---
description: PrismFlow Part 02
---

PART 02 - SYNTHETIC MULTI-VIEW GENERATOR WITH KNOWN DEPENDENCE
==============================================================

Read docs/CONTRACT.md. Do not modify any file outside the scope below.

WHY THIS PART EXISTS - READ THIS BEFORE WRITING CODE
This generator is the calibration standard for the entire project. Because we
control the true dependence between views, we can later check whether the ENIV
estimator recovers a quantity we already know analytically. Without this, every
ENIV number produced for the rest of the project is unfalsifiable: it would be a
value with no way to tell whether it is right. This is not a testing convenience.
It is the measurement instrument's reference standard.

YOU MAY CREATE ONLY THESE PATHS
  prismflow/data/synthetic.py
  prismflow/data/dataset.py
  prismflow/data/loaders.py
  prismflow/data/corruption.py
  tests/unit/test_synthetic.py
  tests/unit/test_dataset.py

GENERATOR SPECIFICATION
  Draw a class-informative latent z_shared of dimension d_latent per sample.
  Draw an independent private latent z_v per view.
  Generate each view as:

      x_v = A_v @ ( sqrt(rho) * z_shared + sqrt(1 - rho) * z_v ) + noise_v

  where A_v is a fixed random projection unique to view v.

  Config controls:
      n_views, n_classes, n_samples, d_latent, d_view
      rho             scalar, or a [V, V] matrix for per-pair control
      noise_std       per-view, allows making a single view unreliable
      signal_strength tune so a naive baseline reaches roughly 80% accuracy.
                      NOT 99%. If the baseline saturates there is no headroom to
                      observe calibration differences and every later experiment
                      becomes uninformative. Add an assertion in the test that
                      baseline accuracy on default config lands in [0.70, 0.88].

  Expose:
      analytic_n_eff(rho, n_views) -> n / (1 + (n - 1) * rho)
  This is the ground truth the ENIV estimator must recover in Part 05.

DATASET LAYER
  Sample: sample_id, views [V, D], label, view_mask [V], metadata dict.
  Batch:  views [B, V, D], view_mask [B, V], labels [B], sample_ids.
  Deterministic train/val/test split. No label leakage across splits.

CORRUPTION UTILITIES (implement now, used by Parts 06, 08, 09)
  duplicate_view(batch, source_idx, k)  -> append k exact copies of a view
  drop_views(batch, rate, seed)         -> set mask False
  add_noise(batch, view_idx, sigma)     -> corrupt one view
  All must be seed-reproducible.

TESTS
  - rho=0.0 produces near-zero empirical cross-view correlation
  - rho=0.9 produces high empirical cross-view correlation
  - empirical correlation increases monotonically with rho over
    {0.0, 0.2, 0.4, 0.6, 0.8}
  - analytic_n_eff(0.0, 5) == 5.0 ; analytic_n_eff(1.0, 5) == 1.0
  - duplicate_view produces exactly identical tensors
  - splits are deterministic and non-overlapping

VERIFY
  Print a toy batch's shapes and the empirical correlation matrix for
  rho in {0.0, 0.5, 0.9}. Confirm empirical correlation tracks rho.

STOP. Do not implement encoders or models.
