# Part 15: real multi-view data

Seeds: [0, 1, 2, 3, 4]. Mean +/- sample std across seeds.

## How much redundancy does a standard multi-view benchmark contain?

Nominal view count against the effective number of INDEPENDENT views,
measured before any duplication, corruption or attack is applied.

| dataset | nominal views | measured ENIV | efficiency ratio | lambda_U |
|---|---|---|---|---|
| handwritten | 6 | 3.28 +/- 0.07 | 0.547 +/- 0.011 | 0.127 +/- 0.042 |

Mean +/- sample std over seeds. ENIV is the eigenvalue form on encoder-feature dependence; lambda_U is the mean off-diagonal upper tail dependence at q = 0.9 on the per-view evidence for the fused predicted class.

### The control: what the estimator reads when the views ARE independent

ENIV is biased downward at finite sample size, so a reading below the view
count is not by itself evidence of redundancy. Each view's encoder features
are permuted within class, which removes cross-view coupling and leaves the
sample size, the feature dimension, the marginals and the class structure
untouched. The deficit against that null is the claim.

| dataset | n (test) | rho_bar obs | rho_bar null | ENIV obs | ENIV null | deficit |
|---|---|---|---|---|---|---|
| handwritten | 800 +/- 0 | 0.526 +/- 0.015 | 0.051 +/- 0.012 | 3.28 +/- 0.07 | 5.51 +/- 0.08 | -2.23 +/- 0.06 |

`rho_bar null` near zero is what says the permutation did its job: the
estimator's chance correction already removes the mean bias, so what is
left in `ENIV null` below the view count is the eigenvalue form's floor,
and only that part of the shortfall is an artefact.

### handwritten: where the redundancy sits

Test accuracy 0.9808 +/- 0.0034, rho_bar 0.5257 +/- 0.0146, design-effect ENIV 1.65 +/- 0.03, tail ENIV 3.72 +/- 0.45.

Mean cross-view dependence matrix (canonical correlation on encoder
features, class-conditional, permutation-null corrected):

| | fou | fac | kar | pix | zer | mor |
|---|---|---|---|---|---|---|
| fou | 1.000 | 0.374 | 0.304 | 0.374 | 0.391 | 0.466 |
| fac | 0.374 | 1.000 | 0.707 | 0.852 | 0.703 | 0.477 |
| kar | 0.304 | 0.707 | 1.000 | 0.823 | 0.557 | 0.291 |
| pix | 0.374 | 0.852 | 0.823 | 1.000 | 0.706 | 0.402 |
| zer | 0.391 | 0.703 | 0.557 | 0.706 | 1.000 | 0.462 |
| mor | 0.466 | 0.477 | 0.291 | 0.402 | 0.462 | 1.000 |

ENIV with each view removed. A view whose removal barely lowers ENIV
was contributing little independent evidence.

| view removed | ENIV of the remaining 5 | independent views it was carrying |
|---|---|---|
| fou | 2.55 +/- 0.06 | 0.74 |
| fac | 3.05 +/- 0.08 | 0.23 |
| kar | 2.87 +/- 0.08 | 0.41 |
| pix | 3.07 +/- 0.04 | 0.21 |
| zer | 2.90 +/- 0.05 | 0.38 |
| mor | 2.60 +/- 0.04 | 0.68 |

## handwritten: the Part 10 detector matrix, re-run on real data

Thresholds are calibrated per seed on that seed's own CLEAN scores.
The detector is never told which condition it is looking at.

| condition | AUC | detect@5%FPR | detect@1%FPR | mean score | accuracy | ECE |
|---|---|---|---|---|---|---|
| clean | 0.500 +/- 0.000 | 0.050 +/- 0.000 | 0.008 +/- 0.004 | 0.2088 +/- 0.0156 | 0.987 +/- 0.004 | 0.035 +/- 0.004 |
| clone_k2 | 0.648 +/- 0.046 | 0.155 +/- 0.074 | 0.125 +/- 0.102 | 0.2761 +/- 0.0248 | 0.989 +/- 0.002 | 0.039 +/- 0.006 |
| missing_30 | 0.406 +/- 0.026 | 0.074 +/- 0.073 | 0.060 +/- 0.074 | 0.1558 +/- 0.0151 | 0.977 +/- 0.006 | 0.083 +/- 0.008 |
| missing_50 | 0.333 +/- 0.044 | 0.054 +/- 0.034 | 0.046 +/- 0.033 | 0.0972 +/- 0.0296 | 0.940 +/- 0.009 | 0.119 +/- 0.007 |
| noisy_1 | 0.598 +/- 0.028 | 0.343 +/- 0.086 | 0.301 +/- 0.085 | 0.3053 +/- 0.0172 | 0.979 +/- 0.004 | 0.048 +/- 0.004 |
| chorus_k1 | 0.554 +/- 0.028 | 0.071 +/- 0.048 | 0.050 +/- 0.053 | 0.2291 +/- 0.0111 | 0.986 +/- 0.005 | 0.038 +/- 0.004 |
| chorus_k2 | 0.555 +/- 0.031 | 0.096 +/- 0.045 | 0.077 +/- 0.034 | 0.2364 +/- 0.0153 | 0.960 +/- 0.024 | 0.062 +/- 0.008 |
| chorus_k3 | 0.577 +/- 0.047 | 0.139 +/- 0.103 | 0.116 +/- 0.097 | 0.2497 +/- 0.0331 | 0.924 +/- 0.020 | 0.073 +/- 0.011 |
| pgd_k2 | 0.555 +/- 0.024 | 0.073 +/- 0.048 | 0.053 +/- 0.054 | 0.2292 +/- 0.0102 | 0.986 +/- 0.005 | 0.039 +/- 0.004 |

### The same detector, against Part 10 on synthetic data

Part 10's own numbers, read from `results/comparison/detector.json`.

| condition | AUC synthetic (4 views) | AUC real (6 views) | change |
|---|---|---|---|
| clean | 0.500 +/- 0.000 | 0.500 +/- 0.000 | +0.000 |
| clone_k2 | 0.874 +/- 0.058 | 0.648 +/- 0.046 | -0.227 |
| missing_30 | 0.257 +/- 0.062 | 0.406 +/- 0.026 | +0.150 |
| missing_50 | 0.125 +/- 0.032 | 0.333 +/- 0.044 | +0.208 |
| noisy_1 | 0.749 +/- 0.054 | 0.598 +/- 0.028 | -0.150 |
| chorus_k1 | 0.722 +/- 0.051 | 0.554 +/- 0.028 | -0.167 |
| chorus_k2 | 0.633 +/- 0.092 | 0.555 +/- 0.031 | -0.078 |
| chorus_k3 | 0.487 +/- 0.110 | 0.577 +/- 0.047 | +0.090 |
| pgd_k2 | 0.840 +/- 0.097 | 0.555 +/- 0.024 | -0.285 |

Mean change across the 6 conditions that ADD agreement (duplication, noise, chorus, PGD): -0.136.
The two `missing` conditions are excluded from that mean and read below
chance in both runs: dropping views removes agreement rather than adding
it, so the detector is not meant to fire on them.

The detector flags a condition by how far its dependence sits above the
CLEAN baseline's. Those baselines are not alike: ENIV 3.33 +/- 0.08 of 4 on synthetic data against 3.35 +/- 0.09 of 6 here -- 0.83 of the view count against 0.56.
A clean condition that already looks partly collusive leaves an attack
less room to raise the score above a threshold calibrated on it.

(That real figure is the CLEAN cell of this matrix, measured on the
comparison split. The redundancy audit reports the same quantity on the
wider split, and the two are not interchangeable: ENIV's finite-sample
bias is downward, so a smaller test split reads differently.)

That is a plausible mechanism, and it points the same way as 5 of the 6 agreement-adding cells,
but it is NOT established here. The exception is chorus_k3, which moved up;
both runs sit near chance on that cell, so it separates the two runs least
well rather than contradicting them.

The two runs also differ in dataset, view count, class count and clean
accuracy at the same time. In particular this benchmark's near-ceiling clean
accuracy compresses pairwise belief agreement on its own, and this table does
not separate that from redundancy. Treat the comparison as a described
difference, not an explained one.

## handwritten: the Part 13 tail analysis, re-run on real data

Chorus attack, k = 2 compromised views (fou, fac) of 6. Headline quantile q = 0.9.

### The colluding pair

| epsilon | mean L2 of delta | success | rho_bar (comp) | lambda_U (comp) | tail spread |
|---|---|---|---|---|---|
| 0.0 | 0.00 +/- 0.00 | 0.000 +/- 0.000 | 0.1965 +/- 0.0484 | 0.0025 +/- 0.0056 | 0.087 +/- 0.026 |
| 0.2 | 2.63 +/- 0.12 | 0.001 +/- 0.001 | 0.5087 +/- 0.0983 | 0.4800 +/- 0.1491 | 0.106 +/- 0.029 |
| 0.5 | 4.77 +/- 0.39 | 0.012 +/- 0.003 | 0.8284 +/- 0.0328 | 0.7825 +/- 0.0438 | 0.093 +/- 0.030 |
| 1.0 | 7.43 +/- 0.69 | 0.069 +/- 0.014 | 0.8114 +/- 0.0258 | 0.7050 +/- 0.0900 | 0.152 +/- 0.024 |
| 2.0 | 15.38 +/- 0.79 | 0.182 +/- 0.036 | 0.8751 +/- 0.0409 | 0.6500 +/- 0.1920 | 0.075 +/- 0.022 |

### Honest pairs (the control)

| epsilon | rho_bar (honest) | lambda_U (honest) |
|---|---|---|
| 0.0 | 0.3190 +/- 0.0257 | 0.1117 +/- 0.0422 |
| 0.2 | 0.3218 +/- 0.0248 | 0.1117 +/- 0.0422 |
| 0.5 | 0.3460 +/- 0.0237 | 0.1125 +/- 0.0421 |
| 1.0 | 0.4096 +/- 0.0423 | 0.1179 +/- 0.0422 |
| 2.0 | 0.5091 +/- 0.0357 | 0.1346 +/- 0.0537 |

### ENIV, tail ENIV, and the exceedance count behind lambda_U

MIN_TAIL_SAMPLES = 50.

| epsilon | ENIV | tail ENIV | lambda_U (all) | n_tail | reliable |
|---|---|---|---|---|---|
| 0.0 | 3.28 +/- 0.07 | 3.72 +/- 0.45 | 0.1268 +/- 0.0418 | 80 | yes |
| 0.2 | 3.32 +/- 0.08 | 3.35 +/- 0.39 | 0.1630 +/- 0.0473 | 80 | yes |
| 0.5 | 3.30 +/- 0.06 | 3.20 +/- 0.40 | 0.1807 +/- 0.0528 | 80 | yes |
| 1.0 | 2.89 +/- 0.06 | 3.61 +/- 0.30 | 0.1347 +/- 0.0298 | 80 | yes |
| 2.0 | 2.42 +/- 0.08 | 4.05 +/- 0.26 | 0.0972 +/- 0.0188 | 80 | yes |

### The quantile the sample size will not support

Part 13's headline quantile was 0.95. On 2000 real patterns there is no
`n_samples` knob to turn, so 0.95 falls below MIN_TAIL_SAMPLES here and
0.90 is the headline instead. Both are shown rather than only the one
that clears the bar.

| epsilon | lambda_U q=0.9 (n_tail, reliable) | lambda_U q=0.95 (n_tail, reliable) |
|---|---|---|
| 0.0 | 0.1268 +/- 0.0418 (80, yes) | 0.0720 +/- 0.0272 (40, NO) |
| 0.2 | 0.1630 +/- 0.0473 (80, yes) | 0.0923 +/- 0.0290 (40, NO) |
| 0.5 | 0.1807 +/- 0.0528 (80, yes) | 0.1213 +/- 0.0467 (40, NO) |
| 1.0 | 0.1347 +/- 0.0298 (80, yes) | 0.0727 +/- 0.0115 (40, NO) |
| 2.0 | 0.0972 +/- 0.0188 (80, yes) | 0.0473 +/- 0.0139 (40, NO) |

### Detection: tail flag vs the Part 10 rho_bar flag, matched FPR

| epsilon | tail AUC | rho_bar AUC | tail TPR@5%FPR | rho_bar TPR@5%FPR |
|---|---|---|---|---|
| 0.2 | 0.476 +/- 0.014 | 0.534 +/- 0.088 | 0.061 +/- 0.015 | 0.324 +/- 0.198 |
| 0.5 | 0.476 +/- 0.022 | 0.688 +/- 0.052 | 0.068 +/- 0.019 | 0.586 +/- 0.071 |
| 1.0 | 0.496 +/- 0.017 | 0.679 +/- 0.070 | 0.055 +/- 0.016 | 0.616 +/- 0.095 |
| 2.0 | 0.528 +/- 0.020 | 0.433 +/- 0.144 | 0.042 +/- 0.014 | 0.315 +/- 0.224 |

## What this does NOT show

Real views carry no ground-truth dependence, so ENIV is APPLIED here and
not VALIDATED here. The only validation of the estimator is Part 05, on the
synthetic generator, where rho is a parameter that was set. Nothing above
is evidence that ENIV measures what it claims to measure.
See `docs/DATASETS.md` for the full limitation.

Commit: 16ff637d76f33716f58c1b043f2f0db2b78f9beb. torch 2.14.0+cpu, numpy 2.5.3.
Controlled research simulation: own models, public benchmark data.
