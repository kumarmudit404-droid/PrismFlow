# Part 14 Arm B diagnostics

Seeds: [0, 1, 2, 3, 4]. Same pipeline and conditions as Arm B.

## D1: is the tail contribution a between-condition mean shift?

| fit | tail column | selectivity AUC |
|---|---|---|
| dependence + detector | (absent) | 0.6724 +/- 0.0591 |
| + tail | as measured | 0.7115 +/- 0.0593 |
| + tail | condition MEAN only (no per-sample content) | 1.0000 +/- 0.0000 |
| + tail | within-condition CENTRED (content only) | 0.6976 +/- 0.0602 |

Between-condition share of tail variance: 0.0068 +/- 0.0033

tail AUC, clone_k2 vs clone_k3 (SAME label, different condition): 0.5044 +/- 0.0206

## D2: does a rank-based fit recover clique contrast?

| quantity | value |
|---|---|
| clique univariate AUC | 0.6408 +/- 0.0551 |
| clique alone, LINEAR fit | 0.5034 +/- 0.1668 |
| clique alone, RANK fit | 0.6408 +/- 0.0551 |
| clique fitted weight, alone (sign) | 0.0033 +/- 0.2152 |
| all four, LINEAR | 0.7448 +/- 0.0592 |
| all four, RANK | 0.7528 +/- 0.0617 |
| existing three, RANK | 0.6736 +/- 0.0697 |

## D3: where do the dropped rows fall?

Non-finite counts per condition per seed (seeds in config order).

| condition | dependence | detector | tail | clique_contrast |
|---|---|---|---|---|
| chorus_k3 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| clone_k2 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] |
| clone_k3 | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 0, 0] | [0, 0, 0, 1, 1] |

