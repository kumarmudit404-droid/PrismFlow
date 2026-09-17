# Frozen-naive control

Seeds: [0, 1, 2, 3, 4]. Base: 4 views, rho = 0.3; k copies of view 0. `naive_frozen` from `results/calibration_frozen_naive/k<k>_naive_frozen/`; the other systems from `results/calibration_duplicated/k<k>_<system>/`. Mean +/- sample std across seeds.

## k = 0

| metric | naive_frozen | naive | prismflow | naive_weights_discounted |
|---|---|---|---|---|
| accuracy | 0.8300 +/- 0.0791 | 0.8300 +/- 0.0791 | 0.8300 +/- 0.0797 | 0.8280 +/- 0.0787 |
| prob_mean_confidence | 0.8072 +/- 0.0557 | 0.8072 +/- 0.0557 | 0.7856 +/- 0.0551 | 0.7665 +/- 0.0555 |
| prob_ece | 0.0579 +/- 0.0094 | 0.0579 +/- 0.0094 | 0.0637 +/- 0.0137 | 0.0758 +/- 0.0179 |
| brier | 0.2511 +/- 0.1019 | 0.2511 +/- 0.1019 | 0.2512 +/- 0.0990 | 0.2602 +/- 0.0981 |
| brier_reliability | 0.0310 +/- 0.0067 | 0.0310 +/- 0.0067 | 0.0282 +/- 0.0081 | 0.0316 +/- 0.0054 |
| brier_resolution | 0.4442 +/- 0.0957 | 0.4442 +/- 0.0957 | 0.4410 +/- 0.0957 | 0.4354 +/- 0.0984 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0004 +/- 0.0008 | -0.0004 +/- 0.0008 | -0.0008 +/- 0.0014 | -0.0007 +/- 0.0017 |
| prob_aurc | 0.0608 +/- 0.0499 | 0.0608 +/- 0.0499 | 0.0602 +/- 0.0491 | 0.0625 +/- 0.0505 |
| vacuity_mean_confidence | 0.7790 +/- 0.0829 | 0.7790 +/- 0.0829 | 0.7499 +/- 0.0816 | 0.7119 +/- 0.0845 |
| vacuity_ece | 0.0905 +/- 0.0245 | 0.0905 +/- 0.0245 | 0.0909 +/- 0.0202 | 0.1220 +/- 0.0245 |

Paired within-seed differences:

| metric | naive_frozen - naive | seeds lower | naive_frozen - prismflow | seeds lower |
|---|---|---|---|---|
| accuracy | +0.0000 +/- 0.0000 | 0/5 | +0.0000 +/- 0.0075 | 2/5 |
| prob_mean_confidence | +0.0000 +/- 0.0000 | 0/5 | +0.0216 +/- 0.0067 | 0/5 |
| prob_ece | +0.0000 +/- 0.0000 | 0/5 | -0.0058 +/- 0.0157 | 3/5 |
| brier | +0.0000 +/- 0.0000 | 0/5 | -0.0002 +/- 0.0043 | 2/5 |
| brier_reliability | +0.0000 +/- 0.0000 | 0/5 | +0.0027 +/- 0.0069 | 1/5 |
| brier_resolution | +0.0000 +/- 0.0000 | 0/5 | +0.0033 +/- 0.0041 | 1/5 |
| prob_aurc | +0.0000 +/- 0.0000 | 0/5 | +0.0006 +/- 0.0016 | 2/5 |

## k = 2

| metric | naive_frozen | naive | prismflow | naive_weights_discounted |
|---|---|---|---|---|
| accuracy | 0.8167 +/- 0.0735 | 0.8100 +/- 0.0783 | 0.8260 +/- 0.0765 | 0.8153 +/- 0.0792 |
| prob_mean_confidence | 0.8474 +/- 0.0479 | 0.8189 +/- 0.0597 | 0.7895 +/- 0.0590 | 0.7493 +/- 0.0659 |
| prob_ece | 0.0665 +/- 0.0148 | 0.0608 +/- 0.0146 | 0.0602 +/- 0.0083 | 0.0758 +/- 0.0135 |
| brier | 0.2710 +/- 0.1044 | 0.2697 +/- 0.1037 | 0.2588 +/- 0.0992 | 0.2744 +/- 0.1005 |
| brier_reliability | 0.0352 +/- 0.0081 | 0.0320 +/- 0.0056 | 0.0341 +/- 0.0048 | 0.0374 +/- 0.0051 |
| brier_resolution | 0.4287 +/- 0.0963 | 0.4266 +/- 0.1008 | 0.4394 +/- 0.1023 | 0.4272 +/- 0.1028 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0002 +/- 0.0007 | -0.0004 +/- 0.0008 | -0.0007 +/- 0.0010 | -0.0006 +/- 0.0004 |
| prob_aurc | 0.0741 +/- 0.0539 | 0.0717 +/- 0.0533 | 0.0637 +/- 0.0494 | 0.0689 +/- 0.0543 |
| vacuity_mean_confidence | 0.8321 +/- 0.0675 | 0.7879 +/- 0.0849 | 0.7508 +/- 0.0834 | 0.6764 +/- 0.0977 |
| vacuity_ece | 0.1081 +/- 0.0369 | 0.0999 +/- 0.0224 | 0.1047 +/- 0.0288 | 0.1434 +/- 0.0304 |

Paired within-seed differences:

| metric | naive_frozen - naive | seeds lower | naive_frozen - prismflow | seeds lower |
|---|---|---|---|---|
| accuracy | +0.0067 +/- 0.0085 | 0/5 | -0.0093 +/- 0.0119 | 3/5 |
| prob_mean_confidence | +0.0285 +/- 0.0151 | 0/5 | +0.0579 +/- 0.0137 | 0/5 |
| prob_ece | +0.0058 +/- 0.0074 | 1/5 | +0.0063 +/- 0.0193 | 2/5 |
| brier | +0.0013 +/- 0.0095 | 3/5 | +0.0123 +/- 0.0150 | 1/5 |
| brier_reliability | +0.0031 +/- 0.0066 | 2/5 | +0.0011 +/- 0.0118 | 3/5 |
| brier_resolution | +0.0021 +/- 0.0095 | 1/5 | -0.0107 +/- 0.0130 | 4/5 |
| prob_aurc | +0.0024 +/- 0.0045 | 1/5 | +0.0104 +/- 0.0064 | 0/5 |

## k = 4

| metric | naive_frozen | naive | prismflow | naive_weights_discounted |
|---|---|---|---|---|
| accuracy | 0.8087 +/- 0.0678 | 0.8073 +/- 0.0854 | 0.8227 +/- 0.0789 | 0.8173 +/- 0.0817 |
| prob_mean_confidence | 0.8643 +/- 0.0430 | 0.8256 +/- 0.0568 | 0.7943 +/- 0.0576 | 0.7423 +/- 0.0626 |
| prob_ece | 0.0865 +/- 0.0297 | 0.0715 +/- 0.0199 | 0.0664 +/- 0.0188 | 0.0889 +/- 0.0081 |
| brier | 0.2906 +/- 0.1044 | 0.2831 +/- 0.1109 | 0.2633 +/- 0.0977 | 0.2846 +/- 0.1033 |
| brier_reliability | 0.0370 +/- 0.0119 | 0.0331 +/- 0.0113 | 0.0372 +/- 0.0041 | 0.0402 +/- 0.0076 |
| brier_resolution | 0.4103 +/- 0.0929 | 0.4135 +/- 0.1039 | 0.4378 +/- 0.0969 | 0.4194 +/- 0.1068 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0009 +/- 0.0019 | -0.0012 +/- 0.0014 | -0.0009 +/- 0.0013 | -0.0009 +/- 0.0016 |
| prob_aurc | 0.0866 +/- 0.0566 | 0.0831 +/- 0.0582 | 0.0677 +/- 0.0487 | 0.0743 +/- 0.0557 |
| vacuity_mean_confidence | 0.8518 +/- 0.0586 | 0.7979 +/- 0.0774 | 0.7602 +/- 0.0775 | 0.6642 +/- 0.0902 |
| vacuity_ece | 0.1225 +/- 0.0305 | 0.1197 +/- 0.0277 | 0.1019 +/- 0.0304 | 0.1630 +/- 0.0224 |

Paired within-seed differences:

| metric | naive_frozen - naive | seeds lower | naive_frozen - prismflow | seeds lower |
|---|---|---|---|---|
| accuracy | +0.0013 +/- 0.0291 | 3/5 | -0.0140 +/- 0.0205 | 4/5 |
| prob_mean_confidence | +0.0387 +/- 0.0185 | 0/5 | +0.0700 +/- 0.0186 | 0/5 |
| prob_ece | +0.0150 +/- 0.0172 | 1/5 | +0.0201 +/- 0.0234 | 1/5 |
| brier | +0.0075 +/- 0.0210 | 2/5 | +0.0273 +/- 0.0245 | 1/5 |
| brier_reliability | +0.0040 +/- 0.0083 | 1/5 | -0.0002 +/- 0.0116 | 3/5 |
| brier_resolution | -0.0032 +/- 0.0187 | 3/5 | -0.0275 +/- 0.0242 | 5/5 |
| prob_aurc | +0.0034 +/- 0.0074 | 2/5 | +0.0189 +/- 0.0097 | 0/5 |

## Change from k = 0, within seed

At k = 0 all naive-family systems start from the same trained model per seed.

| metric | system | k=2 - k=0 | k=4 - k=0 |
|---|---|---|---|
| accuracy | naive_frozen | -0.0133 +/- 0.0155 (1/5 up) | -0.0213 +/- 0.0257 (1/5 up) |
| accuracy | naive | -0.0200 +/- 0.0133 (0/5 up) | -0.0227 +/- 0.0095 (0/5 up) |
| accuracy | prismflow | -0.0040 +/- 0.0119 (2/5 up) | -0.0073 +/- 0.0128 (2/5 up) |
| accuracy | naive_weights_discounted | -0.0127 +/- 0.0043 (0/5 up) | -0.0107 +/- 0.0043 (0/5 up) |
| prob_mean_confidence | naive_frozen | +0.0402 +/- 0.0084 (5/5 up) | +0.0571 +/- 0.0141 (5/5 up) |
| prob_mean_confidence | naive | +0.0117 +/- 0.0092 (5/5 up) | +0.0184 +/- 0.0084 (5/5 up) |
| prob_mean_confidence | prismflow | +0.0039 +/- 0.0094 (4/5 up) | +0.0087 +/- 0.0110 (4/5 up) |
| prob_mean_confidence | naive_weights_discounted | -0.0171 +/- 0.0134 (1/5 up) | -0.0241 +/- 0.0095 (0/5 up) |
| prob_ece | naive_frozen | +0.0086 +/- 0.0195 (4/5 up) | +0.0285 +/- 0.0319 (4/5 up) |
| prob_ece | naive | +0.0028 +/- 0.0154 (4/5 up) | +0.0136 +/- 0.0215 (4/5 up) |
| prob_ece | prismflow | -0.0035 +/- 0.0161 (2/5 up) | +0.0027 +/- 0.0267 (2/5 up) |
| prob_ece | naive_weights_discounted | -0.0000 +/- 0.0074 (3/5 up) | +0.0131 +/- 0.0175 (4/5 up) |
| brier | naive_frozen | +0.0199 +/- 0.0167 (4/5 up) | +0.0395 +/- 0.0280 (5/5 up) |
| brier | naive | +0.0187 +/- 0.0087 (5/5 up) | +0.0320 +/- 0.0166 (5/5 up) |
| brier | prismflow | +0.0075 +/- 0.0041 (5/5 up) | +0.0121 +/- 0.0105 (5/5 up) |
| brier | naive_weights_discounted | +0.0142 +/- 0.0045 (5/5 up) | +0.0244 +/- 0.0097 (5/5 up) |
| brier_reliability | naive_frozen | +0.0042 +/- 0.0048 (4/5 up) | +0.0061 +/- 0.0069 (3/5 up) |
| brier_reliability | naive | +0.0011 +/- 0.0065 (3/5 up) | +0.0021 +/- 0.0065 (3/5 up) |
| brier_reliability | prismflow | +0.0059 +/- 0.0089 (4/5 up) | +0.0090 +/- 0.0061 (5/5 up) |
| brier_reliability | naive_weights_discounted | +0.0058 +/- 0.0042 (4/5 up) | +0.0086 +/- 0.0056 (5/5 up) |
| brier_resolution | naive_frozen | -0.0155 +/- 0.0145 (1/5 up) | -0.0340 +/- 0.0283 (0/5 up) |
| brier_resolution | naive | -0.0176 +/- 0.0123 (1/5 up) | -0.0307 +/- 0.0200 (1/5 up) |
| brier_resolution | prismflow | -0.0016 +/- 0.0068 (3/5 up) | -0.0032 +/- 0.0068 (2/5 up) |
| brier_resolution | naive_weights_discounted | -0.0082 +/- 0.0059 (1/5 up) | -0.0160 +/- 0.0133 (1/5 up) |
| prob_aurc | naive_frozen | +0.0133 +/- 0.0059 (5/5 up) | +0.0258 +/- 0.0106 (5/5 up) |
| prob_aurc | naive | +0.0110 +/- 0.0047 (5/5 up) | +0.0224 +/- 0.0112 (5/5 up) |
| prob_aurc | prismflow | +0.0036 +/- 0.0024 (5/5 up) | +0.0075 +/- 0.0052 (5/5 up) |
| prob_aurc | naive_weights_discounted | +0.0063 +/- 0.0048 (5/5 up) | +0.0118 +/- 0.0070 (5/5 up) |

## Frozen vs trained-on-duplicates: difference in change from k = 0, within seed

(naive_frozen change) - (naive change). Negative on accuracy/resolution means the frozen model degrades more; positive on ECE/reliability/Brier/AURC means it worsens more.

| metric | k=2 | k=4 |
|---|---|---|
| accuracy | +0.0067 +/- 0.0085 (0/5 lower) | +0.0013 +/- 0.0291 (3/5 lower) |
| prob_mean_confidence | +0.0285 +/- 0.0151 (0/5 lower) | +0.0387 +/- 0.0185 (0/5 lower) |
| prob_ece | +0.0058 +/- 0.0074 (1/5 lower) | +0.0150 +/- 0.0172 (1/5 lower) |
| brier | +0.0013 +/- 0.0095 (3/5 lower) | +0.0075 +/- 0.0210 (2/5 lower) |
| brier_reliability | +0.0031 +/- 0.0066 (2/5 lower) | +0.0040 +/- 0.0083 (1/5 lower) |
| brier_resolution | +0.0021 +/- 0.0095 (1/5 lower) | -0.0032 +/- 0.0187 (3/5 lower) |
| prob_aurc | +0.0024 +/- 0.0045 (1/5 lower) | +0.0034 +/- 0.0074 (2/5 lower) |
