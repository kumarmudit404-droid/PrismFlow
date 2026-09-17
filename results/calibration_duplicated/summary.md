# Calibration under duplication

Seeds: [0, 1, 2, 3, 4]. Base: 4 views, rho = 0.3; k copies of view 0. Mean +/- sample std across seeds, from `results/calibration_duplicated/k<k>_<system>/metrics.json`. `prob_` = top-label probability; `vacuity_` = 1 - uncertainty.

## k = 0 (4 views)

| metric | naive | prismflow | naive_weights_discounted |
|---|---|---|---|
| accuracy | 0.8300 +/- 0.0791 | 0.8300 +/- 0.0797 | 0.8280 +/- 0.0787 |
| prob_mean_confidence | 0.8072 +/- 0.0557 | 0.7856 +/- 0.0551 | 0.7665 +/- 0.0555 |
| prob_ece | 0.0579 +/- 0.0094 | 0.0637 +/- 0.0137 | 0.0758 +/- 0.0179 |
| prob_mce | 0.2937 +/- 0.1306 | 0.2564 +/- 0.0951 | 0.2990 +/- 0.0904 |
| brier | 0.2511 +/- 0.1019 | 0.2512 +/- 0.0990 | 0.2602 +/- 0.0981 |
| brier_reliability | 0.0310 +/- 0.0067 | 0.0282 +/- 0.0081 | 0.0316 +/- 0.0054 |
| brier_resolution | 0.4442 +/- 0.0957 | 0.4410 +/- 0.0957 | 0.4354 +/- 0.0984 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0004 +/- 0.0008 | -0.0008 +/- 0.0014 | -0.0007 +/- 0.0017 |
| prob_aurc | 0.0608 +/- 0.0499 | 0.0602 +/- 0.0491 | 0.0625 +/- 0.0505 |
| vacuity_mean_confidence | 0.7790 +/- 0.0829 | 0.7499 +/- 0.0816 | 0.7119 +/- 0.0845 |
| vacuity_ece | 0.0905 +/- 0.0245 | 0.0909 +/- 0.0202 | 0.1220 +/- 0.0245 |
| vacuity_aurc | 0.0697 +/- 0.0563 | 0.0689 +/- 0.0537 | 0.0723 +/- 0.0555 |
| eniv | n/a | 3.3281 +/- 0.0750 | 3.3550 +/- 0.0763 |
| efficiency_ratio | n/a | 0.8320 +/- 0.0188 | 0.8387 +/- 0.0191 |

Paired within-seed differences (system minus naive):

| metric | prismflow - naive | seeds lower | naive_weights_discounted - naive | seeds lower |
|---|---|---|---|---|
| prob_ece | +0.0058 +/- 0.0157 | 2/5 | +0.0179 +/- 0.0191 | 1/5 |
| brier | +0.0002 +/- 0.0043 | 3/5 | +0.0091 +/- 0.0040 | 0/5 |
| brier_reliability | -0.0027 +/- 0.0069 | 4/5 | +0.0006 +/- 0.0063 | 3/5 |
| brier_resolution | -0.0033 +/- 0.0041 | 4/5 | -0.0089 +/- 0.0042 | 5/5 |
| prob_aurc | -0.0006 +/- 0.0016 | 3/5 | +0.0018 +/- 0.0008 | 0/5 |
| prob_mean_confidence | -0.0216 +/- 0.0067 | 5/5 | -0.0408 +/- 0.0057 | 5/5 |
| accuracy | +0.0000 +/- 0.0075 | 2/5 | -0.0020 +/- 0.0051 | 2/5 |

## k = 2 (6 views)

| metric | naive | prismflow | naive_weights_discounted |
|---|---|---|---|
| accuracy | 0.8100 +/- 0.0783 | 0.8260 +/- 0.0765 | 0.8153 +/- 0.0792 |
| prob_mean_confidence | 0.8189 +/- 0.0597 | 0.7895 +/- 0.0590 | 0.7493 +/- 0.0659 |
| prob_ece | 0.0608 +/- 0.0146 | 0.0602 +/- 0.0083 | 0.0758 +/- 0.0135 |
| prob_mce | 0.2537 +/- 0.0851 | 0.3214 +/- 0.1589 | 0.2524 +/- 0.0689 |
| brier | 0.2697 +/- 0.1037 | 0.2588 +/- 0.0992 | 0.2744 +/- 0.1005 |
| brier_reliability | 0.0320 +/- 0.0056 | 0.0341 +/- 0.0048 | 0.0374 +/- 0.0051 |
| brier_resolution | 0.4266 +/- 0.1008 | 0.4394 +/- 0.1023 | 0.4272 +/- 0.1028 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0004 +/- 0.0008 | -0.0007 +/- 0.0010 | -0.0006 +/- 0.0004 |
| prob_aurc | 0.0717 +/- 0.0533 | 0.0637 +/- 0.0494 | 0.0689 +/- 0.0543 |
| vacuity_mean_confidence | 0.7879 +/- 0.0849 | 0.7508 +/- 0.0834 | 0.6764 +/- 0.0977 |
| vacuity_ece | 0.0999 +/- 0.0224 | 0.1047 +/- 0.0288 | 0.1434 +/- 0.0304 |
| vacuity_aurc | 0.0871 +/- 0.0582 | 0.0753 +/- 0.0552 | 0.0812 +/- 0.0590 |
| eniv | n/a | 3.6679 +/- 0.1031 | 3.6792 +/- 0.0693 |
| efficiency_ratio | n/a | 0.6113 +/- 0.0172 | 0.6132 +/- 0.0115 |

Paired within-seed differences (system minus naive):

| metric | prismflow - naive | seeds lower | naive_weights_discounted - naive | seeds lower |
|---|---|---|---|---|
| prob_ece | -0.0005 +/- 0.0198 | 3/5 | +0.0150 +/- 0.0277 | 1/5 |
| brier | -0.0110 +/- 0.0091 | 4/5 | +0.0047 +/- 0.0072 | 1/5 |
| brier_reliability | +0.0021 +/- 0.0067 | 2/5 | +0.0053 +/- 0.0083 | 2/5 |
| brier_resolution | +0.0128 +/- 0.0090 | 0/5 | +0.0005 +/- 0.0063 | 4/5 |
| prob_aurc | -0.0080 +/- 0.0049 | 5/5 | -0.0029 +/- 0.0021 | 5/5 |
| prob_mean_confidence | -0.0294 +/- 0.0111 | 5/5 | -0.0696 +/- 0.0101 | 5/5 |
| accuracy | +0.0160 +/- 0.0086 | 0/5 | +0.0053 +/- 0.0096 | 1/5 |

## k = 4 (8 views)

| metric | naive | prismflow | naive_weights_discounted |
|---|---|---|---|
| accuracy | 0.8073 +/- 0.0854 | 0.8227 +/- 0.0789 | 0.8173 +/- 0.0817 |
| prob_mean_confidence | 0.8256 +/- 0.0568 | 0.7943 +/- 0.0576 | 0.7423 +/- 0.0626 |
| prob_ece | 0.0715 +/- 0.0199 | 0.0664 +/- 0.0188 | 0.0889 +/- 0.0081 |
| prob_mce | 0.2083 +/- 0.0560 | 0.3617 +/- 0.1086 | 0.2593 +/- 0.0762 |
| brier | 0.2831 +/- 0.1109 | 0.2633 +/- 0.0977 | 0.2846 +/- 0.1033 |
| brier_reliability | 0.0331 +/- 0.0113 | 0.0372 +/- 0.0041 | 0.0402 +/- 0.0076 |
| brier_resolution | 0.4135 +/- 0.1039 | 0.4378 +/- 0.0969 | 0.4194 +/- 0.1068 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0012 +/- 0.0014 | -0.0009 +/- 0.0013 | -0.0009 +/- 0.0016 |
| prob_aurc | 0.0831 +/- 0.0582 | 0.0677 +/- 0.0487 | 0.0743 +/- 0.0557 |
| vacuity_mean_confidence | 0.7979 +/- 0.0774 | 0.7602 +/- 0.0775 | 0.6642 +/- 0.0902 |
| vacuity_ece | 0.1197 +/- 0.0277 | 0.1019 +/- 0.0304 | 0.1630 +/- 0.0224 |
| vacuity_aurc | 0.0992 +/- 0.0657 | 0.0816 +/- 0.0571 | 0.0879 +/- 0.0604 |
| eniv | n/a | 3.8449 +/- 0.0826 | 3.8816 +/- 0.1125 |
| efficiency_ratio | n/a | 0.4806 +/- 0.0103 | 0.4852 +/- 0.0141 |

Paired within-seed differences (system minus naive):

| metric | prismflow - naive | seeds lower | naive_weights_discounted - naive | seeds lower |
|---|---|---|---|---|
| prob_ece | -0.0051 +/- 0.0261 | 2/5 | +0.0174 +/- 0.0241 | 1/5 |
| brier | -0.0198 +/- 0.0154 | 4/5 | +0.0015 +/- 0.0099 | 3/5 |
| brier_reliability | +0.0041 +/- 0.0099 | 1/5 | +0.0071 +/- 0.0133 | 2/5 |
| brier_resolution | +0.0242 +/- 0.0130 | 0/5 | +0.0059 +/- 0.0090 | 1/5 |
| prob_aurc | -0.0155 +/- 0.0104 | 5/5 | -0.0088 +/- 0.0036 | 5/5 |
| prob_mean_confidence | -0.0313 +/- 0.0099 | 5/5 | -0.0833 +/- 0.0118 | 5/5 |
| accuracy | +0.0153 +/- 0.0156 | 1/5 | +0.0100 +/- 0.0058 | 0/5 |

## Change from k = 0, within seed

Same seed means same base data at every k.

| metric | system | k=2 - k=0 | k=4 - k=0 |
|---|---|---|---|
| prob_ece | naive | +0.0028 +/- 0.0154 (4/5 up) | +0.0136 +/- 0.0215 (4/5 up) |
| prob_ece | prismflow | -0.0035 +/- 0.0161 (2/5 up) | +0.0027 +/- 0.0267 (2/5 up) |
| prob_ece | naive_weights_discounted | -0.0000 +/- 0.0074 (3/5 up) | +0.0131 +/- 0.0175 (4/5 up) |
| brier_reliability | naive | +0.0011 +/- 0.0065 (3/5 up) | +0.0021 +/- 0.0065 (3/5 up) |
| brier_reliability | prismflow | +0.0059 +/- 0.0089 (4/5 up) | +0.0090 +/- 0.0061 (5/5 up) |
| brier_reliability | naive_weights_discounted | +0.0058 +/- 0.0042 (4/5 up) | +0.0086 +/- 0.0056 (5/5 up) |
| brier_resolution | naive | -0.0176 +/- 0.0123 (1/5 up) | -0.0307 +/- 0.0200 (1/5 up) |
| brier_resolution | prismflow | -0.0016 +/- 0.0068 (3/5 up) | -0.0032 +/- 0.0068 (2/5 up) |
| brier_resolution | naive_weights_discounted | -0.0082 +/- 0.0059 (1/5 up) | -0.0160 +/- 0.0133 (1/5 up) |
| prob_mean_confidence | naive | +0.0117 +/- 0.0092 (5/5 up) | +0.0184 +/- 0.0084 (5/5 up) |
| prob_mean_confidence | prismflow | +0.0039 +/- 0.0094 (4/5 up) | +0.0087 +/- 0.0110 (4/5 up) |
| prob_mean_confidence | naive_weights_discounted | -0.0171 +/- 0.0134 (1/5 up) | -0.0241 +/- 0.0095 (0/5 up) |
| accuracy | naive | -0.0200 +/- 0.0133 (0/5 up) | -0.0227 +/- 0.0095 (0/5 up) |
| accuracy | prismflow | -0.0040 +/- 0.0119 (2/5 up) | -0.0073 +/- 0.0128 (2/5 up) |
| accuracy | naive_weights_discounted | -0.0127 +/- 0.0043 (0/5 up) | -0.0107 +/- 0.0043 (0/5 up) |

## Gap growth: (prismflow - naive) at k minus the same at k = 0, within seed

| metric | k=2 | k=4 |
|---|---|---|
| prob_ece | -0.0063 +/- 0.0142 (3/5 lower) | -0.0109 +/- 0.0223 (3/5 lower) |
| brier | -0.0111 +/- 0.0060 (5/5 lower) | -0.0200 +/- 0.0118 (5/5 lower) |
| brier_reliability | +0.0048 +/- 0.0092 (2/5 lower) | +0.0069 +/- 0.0116 (1/5 lower) |
| brier_resolution | +0.0160 +/- 0.0124 (0/5 lower) | +0.0275 +/- 0.0166 (0/5 lower) |
| prob_aurc | -0.0074 +/- 0.0036 (5/5 lower) | -0.0149 +/- 0.0091 (5/5 lower) |
