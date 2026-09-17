# Calibration under duplication (Part 07 followup)

**Question.** When a view is duplicated, does PrismFlow stay better calibrated
than naive fusion? The prediction was that naive ECE and Brier reliability
worsen as copies are added, while PrismFlow's stay comparatively stable.

**Verdict: the k = 0 null breaks, but not in the way predicted.**

- **Calibration terms: still null.** ECE and Brier reliability do not separate
  PrismFlow from naive fusion at any k. Reliability actually leans the wrong way:
  PrismFlow's own reliability worsens as copies are added (5/5 seeds at k = 4).
- **Discrimination terms: they do separate.** Naive fusion loses resolution,
  accuracy and selective-prediction quality as copies are added. PrismFlow
  mostly does not. The Brier gap and the AURC gap both move in PrismFlow's
  favour in 5/5 seeds at k = 2 and k = 4, and the whole Brier gain comes from
  resolution.

The project's stated calibration claim is "PrismFlow improves reliability, not
resolution." Under duplication this data shows the reverse: the benefit is in
resolution, not reliability. That should be settled before Part 08 builds on
the claim.

All numbers come from `run_calibration_duplicated.py` executed on 2026-09-17:
seeds 0-4, mean +/- sample std, 300 test samples per seed. Every metric was
computed by `prismflow.evaluation.evaluate`. The full tables, including MCE,
vacuity metrics and ENIV, are in `results/calibration_duplicated/summary.md`.
Per-system outputs are in `results/calibration_duplicated/k<k>_<system>/`.

## Design (fixed before the run)

- Base data identical to the existing calibration run: default generator, 4
  views, rho = 0.3. k = 0, 2, 4 bit-identical copies of view 0 are appended by
  `prismflow.data.corruption.duplicate_view`, the same way in training and in
  evaluation batches (the clone experiment's logic).
- Each (k, seed) trains its own (4+k)-view **naive** model
  (`use_discount=False`) and **prismflow** model (`use_discount=True`) from the
  same seed. **naive_weights_discounted** is a copy of the naive model with the
  discount switched on at test time.
- Training and evaluation settings are identical to `config.yaml`: 40 epochs,
  batch 64, lr 1e-3, KL anneal 10, evaluation batch 64, 15 bins.
- **Sanity check passed.** k = 0 reproduces `results/calibration/` exactly, to
  every printed digit, for all three systems.

## ECE (top-label probability)

| k | naive | prismflow | naive weights + discount | prismflow - naive, paired | seeds lower |
|---|---|---|---|---|---|
| 0 | 0.0579 +/- 0.0094 | 0.0637 +/- 0.0137 | 0.0758 +/- 0.0179 | +0.0058 +/- 0.0157 | 2/5 |
| 2 | 0.0608 +/- 0.0146 | 0.0602 +/- 0.0083 | 0.0758 +/- 0.0135 | -0.0005 +/- 0.0198 | 3/5 |
| 4 | 0.0715 +/- 0.0199 | 0.0664 +/- 0.0188 | 0.0889 +/- 0.0081 | -0.0051 +/- 0.0261 | 2/5 |

Within-seed change from k = 0 to k = 4: naive +0.0136 +/- 0.0215 (4/5 up),
prismflow +0.0027 +/- 0.0267 (2/5 up). Naive ECE drifts up and PrismFlow's is
flat, as predicted, but both changes are smaller than their std. The
gap-growth test, (prismflow - naive) at k minus the same at k = 0, gives
-0.0109 +/- 0.0223 at k = 4 (3/5 lower). **ECE is null.**

## Brier decomposition

| k | system | brier | reliability (lower better) | resolution (higher better) |
|---|---|---|---|---|
| 0 | naive | 0.2511 +/- 0.1019 | 0.0310 +/- 0.0067 | 0.4442 +/- 0.0957 |
| 0 | prismflow | 0.2512 +/- 0.0990 | 0.0282 +/- 0.0081 | 0.4410 +/- 0.0957 |
| 0 | naive weights + discount | 0.2602 +/- 0.0981 | 0.0316 +/- 0.0054 | 0.4354 +/- 0.0984 |
| 2 | naive | 0.2697 +/- 0.1037 | 0.0320 +/- 0.0056 | 0.4266 +/- 0.1008 |
| 2 | prismflow | 0.2588 +/- 0.0992 | 0.0341 +/- 0.0048 | 0.4394 +/- 0.1023 |
| 2 | naive weights + discount | 0.2744 +/- 0.1005 | 0.0374 +/- 0.0051 | 0.4272 +/- 0.1028 |
| 4 | naive | 0.2831 +/- 0.1109 | 0.0331 +/- 0.0113 | 0.4135 +/- 0.1039 |
| 4 | prismflow | 0.2633 +/- 0.0977 | 0.0372 +/- 0.0041 | 0.4378 +/- 0.0969 |
| 4 | naive weights + discount | 0.2846 +/- 0.1033 | 0.0402 +/- 0.0076 | 0.4194 +/- 0.1068 |

Uncertainty is 0.6648 +/- 0.0014 in every cell. `brier_within_bin` is at most
0.0012 in magnitude.

Paired, prismflow minus naive:

| k | brier | seeds lower | reliability | seeds lower | resolution | seeds lower |
|---|---|---|---|---|---|---|
| 0 | +0.0002 +/- 0.0043 | 3/5 | -0.0027 +/- 0.0069 | 4/5 | -0.0033 +/- 0.0041 | 4/5 |
| 2 | -0.0110 +/- 0.0091 | 4/5 | +0.0021 +/- 0.0067 | 2/5 | +0.0128 +/- 0.0090 | 0/5 |
| 4 | -0.0198 +/- 0.0154 | 4/5 | +0.0041 +/- 0.0099 | 1/5 | +0.0242 +/- 0.0130 | 0/5 |

Gap growth relative to k = 0, within seed:

| metric | k = 2 | k = 4 |
|---|---|---|
| brier | -0.0111 +/- 0.0060 (5/5 lower) | -0.0200 +/- 0.0118 (5/5 lower) |
| reliability | +0.0048 +/- 0.0092 (2/5 lower) | +0.0069 +/- 0.0116 (1/5 lower) |
| resolution | +0.0160 +/- 0.0124 (0/5 lower) | +0.0275 +/- 0.0166 (0/5 lower) |

Within-seed change from k = 0 to k = 4, per system:

| system | reliability | resolution |
|---|---|---|
| naive | +0.0021 +/- 0.0065 (3/5 up) | **-0.0307 +/- 0.0200 (1/5 up)** |
| prismflow | **+0.0090 +/- 0.0061 (5/5 up)** | -0.0032 +/- 0.0068 (2/5 up) |
| naive weights + discount | +0.0086 +/- 0.0056 (5/5 up) | -0.0160 +/- 0.0133 (1/5 up) |

## AURC and accuracy

| k | naive AURC | prismflow AURC | paired | seeds lower | naive acc | prismflow acc |
|---|---|---|---|---|---|---|
| 0 | 0.0608 +/- 0.0499 | 0.0602 +/- 0.0491 | -0.0006 +/- 0.0016 | 3/5 | 0.830 | 0.830 |
| 2 | 0.0717 +/- 0.0533 | 0.0637 +/- 0.0494 | -0.0080 +/- 0.0049 | 5/5 | 0.810 | 0.826 |
| 4 | 0.0831 +/- 0.0582 | 0.0677 +/- 0.0487 | -0.0155 +/- 0.0104 | 5/5 | 0.807 | 0.823 |

From k = 0 to k = 4 within seed, naive accuracy falls -0.0227 +/- 0.0095
(0/5 up). PrismFlow's falls -0.0073 +/- 0.0128 (2/5 up).

## Reading it

- **Does it reproduce the k = 0 null?** For ECE and reliability, yes, and at
  every k. For Brier, resolution, AURC and accuracy, no. Those separate in
  PrismFlow's favour, consistently in sign (5/5 seeds for the Brier and AURC gap
  growth). The effect sizes are modest: gap growth is about 1.7 std for Brier
  and 1.6 std for AURC at k = 4, with 5 seeds.
- **What duplication does to naive fusion here is lower discrimination, not
  lower calibration.** Its resolution and accuracy fall, and its reliability
  hardly moves. One view counted 1 + k times dominates the fused opinion, which
  plausibly costs accuracy. That is a hypothesis, not tested here.
- **Why naive ECE barely rises (reading of the table means).** Naive
  top-label confidence minus accuracy is 0.807 - 0.830 = -0.023 at k = 0 and
  0.826 - 0.807 = +0.018 at k = 4. Naive starts slightly underconfident and
  moves to slightly overconfident, so its absolute calibration gap stays small
  while crossing zero. The direction is the predicted one (confidence up 5/5,
  accuracy down 5/5). The size is too small for ECE to register at 300 samples
  per seed. PrismFlow stays underconfident throughout (-0.044 at k = 0, -0.028
  at k = 4).
- **Both naive models are also trained on the duplicated data.** The EDL loss
  calibrates the fused output against labels during training. That likely
  limits how much miscalibration naive fusion can accumulate in-distribution.
  This is a hypothesis. A model trained without copies and evaluated with them
  would test it, but that was not run.
- **The discount alone still does not help.** Naive weights + discount has the
  worst ECE and the worst reliability at every k. Its Brier is no better than
  naive's. Its AURC is better than naive's at k = 2 and k = 4 (5/5), but worse
  than PrismFlow's.
- **ENIV rises with k** (prismflow: 3.33, 3.67, 3.84), while alpha falls (0.83,
  0.61, 0.48). Each copy still adds effective views. This is consistent with the
  near-duplicate limit recorded in `docs/KNOWN_LIMITATIONS.md` and is not
  investigated here.

## Limitations

- One base condition (rho = 0.3, 4 views, view 0 duplicated) and k up to 4.
- 300 test samples per seed over 15 bins. MCE is too noisy to read (std up to
  0.16). ECE differences of about 0.01 are below what this setup can resolve.
- 5 seeds. The ECE and reliability nulls are "not detected at this power", not
  "shown to be zero".
- In-distribution only: copies are present in training and in evaluation.

## Reproduce

```
python -m experiments.calibration.run_calibration_duplicated   # ~20 min on CPU
```
