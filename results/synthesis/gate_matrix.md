# Part 14: deployable gate across attack x stress (16 cells)

Gate: rank(dependence) + rank(clique_contrast), signs fixed a priori,
equal weights, NOT refitted per condition. Negatives are clean +
clone_k2 + clone_k3 under the same stress. 5 seeds, mean shown.

## Gate selectivity AUC (defined rows)

| stress | chorus_k1 | chorus_k2 | chorus_k3 | pgd_k2 |
|---|---|---|---|---|
| none | 0.483 | 0.520 | 0.668 | 0.499 |
| missing_30 | 0.469 | 0.534 | 0.637 | 0.490 |
| missing_50 | 0.441 | 0.502 | 0.568 | 0.461 |
| noisy_1 | 0.507 | 0.510 | 0.624 | 0.490 |

## Undefined rate (own column, not folded into dropped rows)

| stress | chorus_k1 | chorus_k2 | chorus_k3 | pgd_k2 |
|---|---|---|---|---|
| none | 0.000 | 0.000 | 0.000 | 0.000 |
| missing_30 | 0.345 | 0.350 | 0.346 | 0.345 |
| missing_50 | 0.706 | 0.709 | 0.708 | 0.706 |
| noisy_1 | 0.006 | 0.006 | 0.006 | 0.006 |

## Four-signal oracle upper bound (label-fitted, NOT deployable)

| stress | chorus_k1 | chorus_k2 | chorus_k3 | pgd_k2 |
|---|---|---|---|---|
| none | 0.766 | 0.734 | 0.716 | 0.827 |
| missing_30 | 0.764 | 0.759 | 0.708 | 0.843 |
| missing_50 | 0.772 | 0.767 | 0.720 | 0.884 |
| noisy_1 | 0.630 | 0.626 | 0.714 | 0.766 |

## Pre-registered missing-views verdict

Band: **SEVERE**. Worst 30%-missing cell: undefined rate 0.3502, gate AUC 0.4686.
Criteria fixed before the run in experiments/synthesis/PREREGISTRATION_missing_views.md: SEVERE above 15% undefined or AUC below 0.60, MATERIAL above 5%.

