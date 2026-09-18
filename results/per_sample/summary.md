# Per-sample ENIV (Part 12)

Seeds: [0, 1, 2, 3, 4]. Mean +/- sample std across seeds.

| condition | system | ENIV mean | ENIV spread (within run) | separation AUC | seconds | accuracy | ECE |
|---|---|---|---|---|---|---|---|
| homogeneous | global | 3.3310 +/- 0.0769 | 0.1069 +/- 0.0204 | n/a | 0.1549 +/- 0.0166 | 0.8333 +/- 0.0748 | 0.0679 +/- 0.0121 |
| homogeneous | per_sample | 3.4649 +/- 0.0972 | 0.1181 +/- 0.0383 | n/a | 0.0444 +/- 0.0227 | 0.8320 +/- 0.0768 | 0.0661 +/- 0.0140 |
| mixed | global | 3.5082 +/- 0.1281 | 0.1228 +/- 0.0550 | 0.5242 +/- 0.0469 | 0.2389 +/- 0.0157 | 0.7080 +/- 0.0655 | 0.0741 +/- 0.0157 |
| mixed | per_sample | 3.6527 +/- 0.0512 | 0.1315 +/- 0.0201 | 0.6039 +/- 0.0720 | 0.0295 +/- 0.0163 | 0.7073 +/- 0.0682 | 0.0711 +/- 0.0186 |

## ENIV stability across seeds (std of the per-run mean)

| condition | global | per_sample |
|---|---|---|
| homogeneous | 0.0769 | 0.0972 |
| mixed | 0.1281 | 0.0512 |

## Gaming audit (audited - trained dependence)

| condition | seed | final gap | max gap |
|---|---|---|---|
| homogeneous | 0 | +0.0476 | +0.0602 |
| homogeneous | 1 | -0.0345 | +0.0087 |
| homogeneous | 2 | +0.0224 | +0.0647 |
| homogeneous | 3 | +0.0371 | +0.1371 |
| homogeneous | 4 | +0.0393 | +0.0784 |
| mixed | 0 | -0.0109 | +0.0284 |
| mixed | 1 | +0.0308 | +0.0603 |
| mixed | 2 | +0.0661 | +0.1062 |
| mixed | 3 | +0.0571 | +0.0803 |
| mixed | 4 | +0.0645 | +0.0645 |

Audit plot written: True.
