# Mechanism-only control: frozen naive + discount at inference

Seeds: [0, 1, 2, 3, 4]. Base: 4 views, rho = 0.3; k copies of view 0. `naive_frozen_discounted` from `results/calibration_frozen_discount/k<k>_naive_frozen_discounted/`; `naive_frozen` from `results/calibration_frozen_naive/`; the rest from `results/calibration_duplicated/`. Mean +/- sample std across seeds.

## k = 0

| metric | naive_frozen_discounted | naive_frozen | prismflow | naive | naive_weights_discounted |
|---|---|---|---|---|---|
| accuracy | 0.8280 +/- 0.0787 | 0.8300 +/- 0.0791 | 0.8300 +/- 0.0797 | 0.8300 +/- 0.0791 | 0.8280 +/- 0.0787 |
| prob_mean_confidence | 0.7665 +/- 0.0555 | 0.8072 +/- 0.0557 | 0.7856 +/- 0.0551 | 0.8072 +/- 0.0557 | 0.7665 +/- 0.0555 |
| prob_ece | 0.0758 +/- 0.0179 | 0.0579 +/- 0.0094 | 0.0637 +/- 0.0137 | 0.0579 +/- 0.0094 | 0.0758 +/- 0.0179 |
| brier | 0.2602 +/- 0.0981 | 0.2511 +/- 0.1019 | 0.2512 +/- 0.0990 | 0.2511 +/- 0.1019 | 0.2602 +/- 0.0981 |
| brier_reliability | 0.0316 +/- 0.0054 | 0.0310 +/- 0.0067 | 0.0282 +/- 0.0081 | 0.0310 +/- 0.0067 | 0.0316 +/- 0.0054 |
| brier_resolution | 0.4354 +/- 0.0984 | 0.4442 +/- 0.0957 | 0.4410 +/- 0.0957 | 0.4442 +/- 0.0957 | 0.4354 +/- 0.0984 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0007 +/- 0.0017 | -0.0004 +/- 0.0008 | -0.0008 +/- 0.0014 | -0.0004 +/- 0.0008 | -0.0007 +/- 0.0017 |
| prob_aurc | 0.0625 +/- 0.0505 | 0.0608 +/- 0.0499 | 0.0602 +/- 0.0491 | 0.0608 +/- 0.0499 | 0.0625 +/- 0.0505 |
| vacuity_mean_confidence | 0.7119 +/- 0.0845 | 0.7790 +/- 0.0829 | 0.7499 +/- 0.0816 | 0.7790 +/- 0.0829 | 0.7119 +/- 0.0845 |
| vacuity_ece | 0.1220 +/- 0.0245 | 0.0905 +/- 0.0245 | 0.0909 +/- 0.0202 | 0.0905 +/- 0.0245 | 0.1220 +/- 0.0245 |
| eniv | 3.3550 +/- 0.0763 | n/a | 3.3281 +/- 0.0750 | n/a | 3.3550 +/- 0.0763 |
| efficiency_ratio | 0.8387 +/- 0.0191 | n/a | 0.8320 +/- 0.0188 | n/a | 0.8387 +/- 0.0191 |

Paired within-seed differences:

| metric | naive_frozen_discounted - naive_frozen | seeds lower | naive_frozen_discounted - prismflow | seeds lower |
|---|---|---|---|---|
| accuracy | -0.0020 +/- 0.0051 | 2/5 | -0.0020 +/- 0.0112 | 2/5 |
| prob_mean_confidence | -0.0408 +/- 0.0057 | 5/5 | -0.0191 +/- 0.0024 | 5/5 |
| prob_ece | +0.0179 +/- 0.0191 | 1/5 | +0.0121 +/- 0.0088 | 1/5 |
| brier | +0.0091 +/- 0.0040 | 0/5 | +0.0090 +/- 0.0024 | 0/5 |
| brier_reliability | +0.0006 +/- 0.0063 | 3/5 | +0.0033 +/- 0.0043 | 1/5 |
| brier_resolution | -0.0089 +/- 0.0042 | 5/5 | -0.0056 +/- 0.0037 | 4/5 |
| prob_aurc | +0.0018 +/- 0.0008 | 0/5 | +0.0024 +/- 0.0020 | 1/5 |

## k = 2

| metric | naive_frozen_discounted | naive_frozen | prismflow | naive | naive_weights_discounted |
|---|---|---|---|---|---|
| accuracy | 0.8247 +/- 0.0821 | 0.8167 +/- 0.0735 | 0.8260 +/- 0.0765 | 0.8100 +/- 0.0783 | 0.8153 +/- 0.0792 |
| prob_mean_confidence | 0.7814 +/- 0.0550 | 0.8474 +/- 0.0479 | 0.7895 +/- 0.0590 | 0.8189 +/- 0.0597 | 0.7493 +/- 0.0659 |
| prob_ece | 0.0731 +/- 0.0127 | 0.0665 +/- 0.0148 | 0.0602 +/- 0.0083 | 0.0608 +/- 0.0146 | 0.0758 +/- 0.0135 |
| brier | 0.2636 +/- 0.1008 | 0.2710 +/- 0.1044 | 0.2588 +/- 0.0992 | 0.2697 +/- 0.1037 | 0.2744 +/- 0.1005 |
| brier_reliability | 0.0364 +/- 0.0033 | 0.0352 +/- 0.0081 | 0.0341 +/- 0.0048 | 0.0320 +/- 0.0056 | 0.0374 +/- 0.0051 |
| brier_resolution | 0.4362 +/- 0.1003 | 0.4287 +/- 0.0963 | 0.4394 +/- 0.1023 | 0.4266 +/- 0.1008 | 0.4272 +/- 0.1028 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0013 +/- 0.0016 | -0.0002 +/- 0.0007 | -0.0007 +/- 0.0010 | -0.0004 +/- 0.0008 | -0.0006 +/- 0.0004 |
| prob_aurc | 0.0672 +/- 0.0522 | 0.0741 +/- 0.0539 | 0.0637 +/- 0.0494 | 0.0717 +/- 0.0533 | 0.0689 +/- 0.0543 |
| vacuity_mean_confidence | 0.7312 +/- 0.0827 | 0.8321 +/- 0.0675 | 0.7508 +/- 0.0834 | 0.7879 +/- 0.0849 | 0.6764 +/- 0.0977 |
| vacuity_ece | 0.1145 +/- 0.0331 | 0.1081 +/- 0.0369 | 0.1047 +/- 0.0288 | 0.0999 +/- 0.0224 | 0.1434 +/- 0.0304 |
| eniv | 3.5659 +/- 0.0817 | n/a | 3.6679 +/- 0.1031 | n/a | 3.6792 +/- 0.0693 |
| efficiency_ratio | 0.5943 +/- 0.0136 | n/a | 0.6113 +/- 0.0172 | n/a | 0.6132 +/- 0.0115 |

Paired within-seed differences:

| metric | naive_frozen_discounted - naive_frozen | seeds lower | naive_frozen_discounted - prismflow | seeds lower |
|---|---|---|---|---|
| accuracy | +0.0080 +/- 0.0139 | 1/5 | -0.0013 +/- 0.0077 | 3/5 |
| prob_mean_confidence | -0.0660 +/- 0.0096 | 5/5 | -0.0082 +/- 0.0085 | 4/5 |
| prob_ece | +0.0066 +/- 0.0262 | 2/5 | +0.0129 +/- 0.0147 | 2/5 |
| brier | -0.0074 +/- 0.0134 | 4/5 | +0.0048 +/- 0.0031 | 0/5 |
| brier_reliability | +0.0012 +/- 0.0074 | 2/5 | +0.0023 +/- 0.0047 | 1/5 |
| brier_resolution | +0.0075 +/- 0.0123 | 2/5 | -0.0032 +/- 0.0036 | 4/5 |
| prob_aurc | -0.0069 +/- 0.0033 | 5/5 | +0.0035 +/- 0.0038 | 1/5 |

## k = 4

| metric | naive_frozen_discounted | naive_frozen | prismflow | naive | naive_weights_discounted |
|---|---|---|---|---|---|
| accuracy | 0.8193 +/- 0.0797 | 0.8087 +/- 0.0678 | 0.8227 +/- 0.0789 | 0.8073 +/- 0.0854 | 0.8173 +/- 0.0817 |
| prob_mean_confidence | 0.7789 +/- 0.0541 | 0.8643 +/- 0.0430 | 0.7943 +/- 0.0576 | 0.8256 +/- 0.0568 | 0.7423 +/- 0.0626 |
| prob_ece | 0.0753 +/- 0.0146 | 0.0865 +/- 0.0297 | 0.0664 +/- 0.0188 | 0.0715 +/- 0.0199 | 0.0889 +/- 0.0081 |
| brier | 0.2697 +/- 0.1000 | 0.2906 +/- 0.1044 | 0.2633 +/- 0.0977 | 0.2831 +/- 0.1109 | 0.2846 +/- 0.1033 |
| brier_reliability | 0.0375 +/- 0.0064 | 0.0370 +/- 0.0119 | 0.0372 +/- 0.0041 | 0.0331 +/- 0.0113 | 0.0402 +/- 0.0076 |
| brier_resolution | 0.4320 +/- 0.1018 | 0.4103 +/- 0.0929 | 0.4378 +/- 0.0969 | 0.4135 +/- 0.1039 | 0.4194 +/- 0.1068 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0006 +/- 0.0013 | -0.0009 +/- 0.0019 | -0.0009 +/- 0.0013 | -0.0012 +/- 0.0014 | -0.0009 +/- 0.0016 |
| prob_aurc | 0.0709 +/- 0.0524 | 0.0866 +/- 0.0566 | 0.0677 +/- 0.0487 | 0.0831 +/- 0.0582 | 0.0743 +/- 0.0557 |
| vacuity_mean_confidence | 0.7249 +/- 0.0807 | 0.8518 +/- 0.0586 | 0.7602 +/- 0.0775 | 0.7979 +/- 0.0774 | 0.6642 +/- 0.0902 |
| vacuity_ece | 0.1171 +/- 0.0342 | 0.1225 +/- 0.0305 | 0.1019 +/- 0.0304 | 0.1197 +/- 0.0277 | 0.1630 +/- 0.0224 |
| eniv | 3.5657 +/- 0.0739 | n/a | 3.8449 +/- 0.0826 | n/a | 3.8816 +/- 0.1125 |
| efficiency_ratio | 0.4457 +/- 0.0092 | n/a | 0.4806 +/- 0.0103 | n/a | 0.4852 +/- 0.0141 |

Paired within-seed differences:

| metric | naive_frozen_discounted - naive_frozen | seeds lower | naive_frozen_discounted - prismflow | seeds lower |
|---|---|---|---|---|
| accuracy | +0.0107 +/- 0.0202 | 1/5 | -0.0033 +/- 0.0082 | 3/5 |
| prob_mean_confidence | -0.0854 +/- 0.0131 | 5/5 | -0.0154 +/- 0.0106 | 5/5 |
| prob_ece | -0.0111 +/- 0.0386 | 3/5 | +0.0089 +/- 0.0205 | 2/5 |
| brier | -0.0209 +/- 0.0227 | 4/5 | +0.0063 +/- 0.0039 | 0/5 |
| brier_reliability | +0.0005 +/- 0.0138 | 3/5 | +0.0003 +/- 0.0084 | 2/5 |
| brier_resolution | +0.0217 +/- 0.0262 | 0/5 | -0.0058 +/- 0.0098 | 4/5 |
| prob_aurc | -0.0157 +/- 0.0066 | 5/5 | +0.0032 +/- 0.0038 | 1/5 |

## Change from k = 0, within seed

| metric | system | k=2 - k=0 | k=4 - k=0 |
|---|---|---|---|
| accuracy | naive_frozen_discounted | -0.0033 +/- 0.0071 (2/5 up) | -0.0087 +/- 0.0077 (0/5 up) |
| accuracy | naive_frozen | -0.0133 +/- 0.0155 (1/5 up) | -0.0213 +/- 0.0257 (1/5 up) |
| accuracy | prismflow | -0.0040 +/- 0.0119 (2/5 up) | -0.0073 +/- 0.0128 (2/5 up) |
| accuracy | naive | -0.0200 +/- 0.0133 (0/5 up) | -0.0227 +/- 0.0095 (0/5 up) |
| accuracy | naive_weights_discounted | -0.0127 +/- 0.0043 (0/5 up) | -0.0107 +/- 0.0043 (0/5 up) |
| prob_mean_confidence | naive_frozen_discounted | +0.0149 +/- 0.0043 (5/5 up) | +0.0125 +/- 0.0062 (5/5 up) |
| prob_mean_confidence | naive_frozen | +0.0402 +/- 0.0084 (5/5 up) | +0.0571 +/- 0.0141 (5/5 up) |
| prob_mean_confidence | prismflow | +0.0039 +/- 0.0094 (4/5 up) | +0.0087 +/- 0.0110 (4/5 up) |
| prob_mean_confidence | naive | +0.0117 +/- 0.0092 (5/5 up) | +0.0184 +/- 0.0084 (5/5 up) |
| prob_mean_confidence | naive_weights_discounted | -0.0171 +/- 0.0134 (1/5 up) | -0.0241 +/- 0.0095 (0/5 up) |
| prob_ece | naive_frozen_discounted | -0.0027 +/- 0.0133 (2/5 up) | -0.0005 +/- 0.0149 (2/5 up) |
| prob_ece | naive_frozen | +0.0086 +/- 0.0195 (4/5 up) | +0.0285 +/- 0.0319 (4/5 up) |
| prob_ece | prismflow | -0.0035 +/- 0.0161 (2/5 up) | +0.0027 +/- 0.0267 (2/5 up) |
| prob_ece | naive | +0.0028 +/- 0.0154 (4/5 up) | +0.0136 +/- 0.0215 (4/5 up) |
| prob_ece | naive_weights_discounted | -0.0000 +/- 0.0074 (3/5 up) | +0.0131 +/- 0.0175 (4/5 up) |
| brier | naive_frozen_discounted | +0.0034 +/- 0.0061 (3/5 up) | +0.0094 +/- 0.0090 (4/5 up) |
| brier | naive_frozen | +0.0199 +/- 0.0167 (4/5 up) | +0.0395 +/- 0.0280 (5/5 up) |
| brier | prismflow | +0.0075 +/- 0.0041 (5/5 up) | +0.0121 +/- 0.0105 (5/5 up) |
| brier | naive | +0.0187 +/- 0.0087 (5/5 up) | +0.0320 +/- 0.0166 (5/5 up) |
| brier | naive_weights_discounted | +0.0142 +/- 0.0045 (5/5 up) | +0.0244 +/- 0.0097 (5/5 up) |
| brier_reliability | naive_frozen_discounted | +0.0048 +/- 0.0060 (4/5 up) | +0.0060 +/- 0.0035 (5/5 up) |
| brier_reliability | naive_frozen | +0.0042 +/- 0.0048 (4/5 up) | +0.0061 +/- 0.0069 (3/5 up) |
| brier_reliability | prismflow | +0.0059 +/- 0.0089 (4/5 up) | +0.0090 +/- 0.0061 (5/5 up) |
| brier_reliability | naive | +0.0011 +/- 0.0065 (3/5 up) | +0.0021 +/- 0.0065 (3/5 up) |
| brier_reliability | naive_weights_discounted | +0.0058 +/- 0.0042 (4/5 up) | +0.0086 +/- 0.0056 (5/5 up) |
| brier_resolution | naive_frozen_discounted | +0.0008 +/- 0.0029 (3/5 up) | -0.0034 +/- 0.0109 (3/5 up) |
| brier_resolution | naive_frozen | -0.0155 +/- 0.0145 (1/5 up) | -0.0340 +/- 0.0283 (0/5 up) |
| brier_resolution | prismflow | -0.0016 +/- 0.0068 (3/5 up) | -0.0032 +/- 0.0068 (2/5 up) |
| brier_resolution | naive | -0.0176 +/- 0.0123 (1/5 up) | -0.0307 +/- 0.0200 (1/5 up) |
| brier_resolution | naive_weights_discounted | -0.0082 +/- 0.0059 (1/5 up) | -0.0160 +/- 0.0133 (1/5 up) |
| prob_aurc | naive_frozen_discounted | +0.0047 +/- 0.0033 (5/5 up) | +0.0083 +/- 0.0048 (5/5 up) |
| prob_aurc | naive_frozen | +0.0133 +/- 0.0059 (5/5 up) | +0.0258 +/- 0.0106 (5/5 up) |
| prob_aurc | prismflow | +0.0036 +/- 0.0024 (5/5 up) | +0.0075 +/- 0.0052 (5/5 up) |
| prob_aurc | naive | +0.0110 +/- 0.0047 (5/5 up) | +0.0224 +/- 0.0112 (5/5 up) |
| prob_aurc | naive_weights_discounted | +0.0063 +/- 0.0048 (5/5 up) | +0.0118 +/- 0.0070 (5/5 up) |

## How much of PrismFlow's protection does the mechanism alone provide?

Within seed, change from k = 0 minus naive_frozen's change from k = 0. For resolution/accuracy, positive = less degradation than the undiscounted frozen model; for Brier/AURC/ECE/reliability, negative = less degradation.

| metric | system | k=2 | k=4 |
|---|---|---|---|
| accuracy | naive_frozen_discounted | +0.0100 +/- 0.0156 (4/5 up) | +0.0127 +/- 0.0213 (4/5 up) |
| accuracy | prismflow | +0.0093 +/- 0.0086 (4/5 up) | +0.0140 +/- 0.0164 (4/5 up) |
| prob_mean_confidence | naive_frozen_discounted | -0.0253 +/- 0.0070 (0/5 up) | -0.0446 +/- 0.0114 (0/5 up) |
| prob_mean_confidence | prismflow | -0.0363 +/- 0.0145 (0/5 up) | -0.0484 +/- 0.0200 (0/5 up) |
| prob_ece | naive_frozen_discounted | -0.0113 +/- 0.0208 (2/5 up) | -0.0290 +/- 0.0216 (1/5 up) |
| prob_ece | prismflow | -0.0121 +/- 0.0178 (1/5 up) | -0.0258 +/- 0.0145 (0/5 up) |
| brier | naive_frozen_discounted | -0.0166 +/- 0.0132 (0/5 up) | -0.0301 +/- 0.0228 (0/5 up) |
| brier | prismflow | -0.0124 +/- 0.0138 (1/5 up) | -0.0274 +/- 0.0241 (0/5 up) |
| brier_reliability | naive_frozen_discounted | +0.0006 +/- 0.0059 (2/5 up) | -0.0001 +/- 0.0091 (2/5 up) |
| brier_reliability | prismflow | +0.0017 +/- 0.0109 (3/5 up) | +0.0029 +/- 0.0094 (4/5 up) |
| brier_resolution | naive_frozen_discounted | +0.0164 +/- 0.0143 (4/5 up) | +0.0306 +/- 0.0270 (5/5 up) |
| brier_resolution | prismflow | +0.0140 +/- 0.0163 (3/5 up) | +0.0308 +/- 0.0278 (5/5 up) |
| prob_aurc | naive_frozen_discounted | -0.0086 +/- 0.0038 (0/5 up) | -0.0175 +/- 0.0071 (0/5 up) |
| prob_aurc | prismflow | -0.0098 +/- 0.0059 (0/5 up) | -0.0183 +/- 0.0091 (0/5 up) |
