# Calibration experiment

Seeds: [0, 1, 2, 3, 4]. Mean +/- sample std across seeds, from each system's `results/calibration/<system>/metrics.json`. `prob_` metrics use the top-label probability; `vacuity_` metrics use 1 - uncertainty.

| metric | naive | prismflow | naive_weights_discounted |
|---|---|---|---|
| accuracy | 0.8300 +/- 0.0791 | 0.8300 +/- 0.0797 | 0.8280 +/- 0.0787 |
| macro_f1 | 0.8265 +/- 0.0854 | 0.8265 +/- 0.0860 | 0.8243 +/- 0.0851 |
| prob_ece | 0.0579 +/- 0.0094 | 0.0637 +/- 0.0137 | 0.0758 +/- 0.0179 |
| prob_mce | 0.2937 +/- 0.1306 | 0.2564 +/- 0.0951 | 0.2990 +/- 0.0904 |
| brier | 0.2511 +/- 0.1019 | 0.2512 +/- 0.0990 | 0.2602 +/- 0.0981 |
| brier_reliability | 0.0310 +/- 0.0067 | 0.0282 +/- 0.0081 | 0.0316 +/- 0.0054 |
| brier_resolution | 0.4442 +/- 0.0957 | 0.4410 +/- 0.0957 | 0.4354 +/- 0.0984 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0004 +/- 0.0008 | -0.0008 +/- 0.0014 | -0.0007 +/- 0.0017 |
| prob_aurc | 0.0608 +/- 0.0499 | 0.0602 +/- 0.0491 | 0.0625 +/- 0.0505 |
| prob_selective_risk@1.00 | 0.1700 +/- 0.0791 | 0.1700 +/- 0.0797 | 0.1720 +/- 0.0787 |
| prob_selective_risk@0.90 | 0.1378 +/- 0.0820 | 0.1333 +/- 0.0771 | 0.1407 +/- 0.0827 |
| prob_selective_risk@0.80 | 0.1042 +/- 0.0774 | 0.1067 +/- 0.0751 | 0.1050 +/- 0.0791 |
| prob_selective_risk@0.50 | 0.0507 +/- 0.0555 | 0.0480 +/- 0.0549 | 0.0520 +/- 0.0563 |
| vacuity_mean_confidence | 0.7790 +/- 0.0829 | 0.7499 +/- 0.0816 | 0.7119 +/- 0.0845 |
| vacuity_ece | 0.0905 +/- 0.0245 | 0.0909 +/- 0.0202 | 0.1220 +/- 0.0245 |
| vacuity_aurc | 0.0697 +/- 0.0563 | 0.0689 +/- 0.0537 | 0.0723 +/- 0.0555 |
| eniv | n/a | 3.3281 +/- 0.0750 | 3.3550 +/- 0.0763 |
| efficiency_ratio | n/a | 0.8320 +/- 0.0188 | 0.8387 +/- 0.0191 |

## Paired within-seed differences (system minus naive)

Same seed means same data and, for prismflow, same initial weights.

| metric | prismflow - naive | seeds lower | naive_weights_discounted - naive | seeds lower |
|---|---|---|---|---|
| prob_ece | +0.0058 +/- 0.0157 | 2/5 | +0.0179 +/- 0.0191 | 1/5 |
| brier | +0.0002 +/- 0.0043 | 3/5 | +0.0091 +/- 0.0040 | 0/5 |
| brier_reliability | -0.0027 +/- 0.0069 | 4/5 | +0.0006 +/- 0.0063 | 3/5 |
| brier_resolution | -0.0033 +/- 0.0041 | 4/5 | -0.0089 +/- 0.0042 | 5/5 |
| prob_aurc | -0.0006 +/- 0.0016 | 3/5 | +0.0018 +/- 0.0008 | 0/5 |
| accuracy | +0.0000 +/- 0.0075 | 2/5 | -0.0020 +/- 0.0051 | 2/5 |
