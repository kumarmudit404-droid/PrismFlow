# Part 35 - LANL re-analysis from the saved scores

No training. Every number below is computed from Part 30's committed test-block scores and sample table, both verified against `results/lanl/results_manifest.json` before being read (42 sha256 checks, all matched). `results/lanl/` is read-only and unchanged.

Pre-registered in `experiments/lanl_reanalysis/config.yaml`, committed before any number here existed (commit `f11bbca`). This file is generated from `results/lanl_reanalysis/results.json` by `experiments/lanl_reanalysis/run.py`, so no figure in it is typed by hand.

**There are no train-block scores in `results/lanl/`.** Anything needing a fit is cross-fitted over test computers and labelled DIAGNOSTIC. None of it is a deployable recalibration.

## 1. L29 - why two arms score worse than a constant

Constant-prevalence baseline: **0.0428** = 2*pi*(1-pi) at test prevalence 0.0219, two-class Brier (`docs/EVALUATION_PROTOCOL.md`: mean_i sum_k (p_ik - y_ik)^2, twice the binary form).

**Pre-registered decision rule.** A cause ACCOUNTS FOR the failure only if correcting that cause ALONE brings Brier below 0.0428 in 5 of 5 seeds for BOTH systems, for the arm in question. 4 of 5, or 5 of 5 in one system only, is NOT accounted for. No combination of causes is tested, and no cause is credited in part. The verdict "none does" is a permitted and reportable outcome.

Saved scores reproduce the committed per-seed Brier of all 40 units to 2.64e-10 (tolerance 1e-09), so these are the scores Part 30's metrics came from.

### 1.1 Brier per cause, 5 seeds, mean +/- std

`present` = rows with at least one view present for that arm (cause a). `prior shift` is an ORACLE: it uses the test prevalence. `Platt` is a DIAGNOSTIC: fitted on test rows, cross-fitted by computer fold.

| arm | role | system | uncorrected | (a) present rows | (a) absent rows | (b) prior shift ORACLE | (c) cross-fit Platt DIAGNOSTIC |
|---|---|---|---|---|---|---|---|
| `without_auth` | failing (L29 names it) | naive | 0.1218 +/- 0.0006 | 0.0318 +/- 0.0007 | 0.5000 +/- 0.0000 | 0.2033 +/- 0.0015 | 0.0432 +/- 0.0000 |
| `without_auth` | failing (L29 names it) | prismflow | 0.1219 +/- 0.0005 | 0.0319 +/- 0.0006 | 0.5000 +/- 0.0000 | 0.2038 +/- 0.0013 | 0.0432 +/- 0.0000 |
| `without_flows` | failing (L29 names it) | naive | 0.0628 +/- 0.0004 | 0.0382 +/- 0.0004 | 0.5000 +/- 0.0000 | 0.0882 +/- 0.0008 | 0.0427 +/- 0.0001 |
| `without_flows` | failing (L29 names it) | prismflow | 0.0630 +/- 0.0006 | 0.0384 +/- 0.0006 | 0.5000 +/- 0.0000 | 0.0891 +/- 0.0011 | 0.0426 +/- 0.0001 |
| `without_dns` | reference | naive | 0.0386 +/- 0.0007 | 0.0368 +/- 0.0007 | 0.5000 +/- 0.0000 | 0.0429 +/- 0.0009 | 0.0396 +/- 0.0003 |
| `without_dns` | reference | prismflow | 0.0386 +/- 0.0002 | 0.0368 +/- 0.0002 | 0.5000 +/- 0.0000 | 0.0435 +/- 0.0005 | 0.0397 +/- 0.0006 |
| `with_auth` | reference | naive | 0.0370 +/- 0.0005 | 0.0370 +/- 0.0005 | not measured | 0.0397 +/- 0.0009 | 0.0396 +/- 0.0005 |
| `with_auth` | reference | prismflow | 0.0366 +/- 0.0002 | 0.0366 +/- 0.0002 | not measured | 0.0403 +/- 0.0005 | 0.0391 +/- 0.0005 |

### 1.2 Reliability and resolution of the same five columns

| arm | system | quantity | uncorrected | (a) present | (a) absent | (b) prior shift | (c) Platt |
|---|---|---|---|---|---|---|---|
| `without_auth` | naive | reliability | 0.0826 +/- 0.0004 | 0.0011 +/- 0.0006 | 0.4258 +/- 0.0000 | 0.1641 +/- 0.0013 | 0.0011 +/- 0.0000 |
| `without_auth` | naive | resolution | 0.0033 +/- 0.0002 | 0.0041 +/- 0.0001 | 0.0000 +/- 0.0000 | 0.0035 +/- 0.0002 | 0.0000 +/- 0.0000 |
| `without_auth` | prismflow | reliability | 0.0826 +/- 0.0004 | 0.0011 +/- 0.0005 | 0.4258 +/- 0.0000 | 0.1645 +/- 0.0012 | 0.0011 +/- 0.0001 |
| `without_auth` | prismflow | resolution | 0.0032 +/- 0.0002 | 0.0040 +/- 0.0001 | 0.0000 +/- 0.0000 | 0.0034 +/- 0.0002 | 0.0000 +/- 0.0000 |
| `without_flows` | naive | reliability | 0.0260 +/- 0.0006 | 0.0005 +/- 0.0001 | 0.5000 +/- 0.0000 | 0.0515 +/- 0.0006 | 0.0013 +/- 0.0002 |
| `without_flows` | naive | resolution | 0.0057 +/- 0.0004 | 0.0071 +/- 0.0005 | 0.0000 +/- 0.0000 | 0.0059 +/- 0.0005 | 0.0009 +/- 0.0001 |
| `without_flows` | prismflow | reliability | 0.0263 +/- 0.0003 | 0.0005 +/- 0.0001 | 0.5000 +/- 0.0000 | 0.0528 +/- 0.0007 | 0.0012 +/- 0.0002 |
| `without_flows` | prismflow | resolution | 0.0057 +/- 0.0007 | 0.0068 +/- 0.0005 | 0.0000 +/- 0.0000 | 0.0063 +/- 0.0006 | 0.0008 +/- 0.0002 |
| `without_dns` | naive | reliability | 0.0021 +/- 0.0003 | 0.0006 +/- 0.0002 | 0.5000 +/- 0.0000 | 0.0065 +/- 0.0006 | 0.0006 +/- 0.0001 |
| `without_dns` | naive | resolution | 0.0060 +/- 0.0007 | 0.0064 +/- 0.0006 | 0.0000 +/- 0.0000 | 0.0063 +/- 0.0007 | 0.0032 +/- 0.0005 |
| `without_dns` | prismflow | reliability | 0.0020 +/- 0.0003 | 0.0006 +/- 0.0003 | 0.5000 +/- 0.0000 | 0.0069 +/- 0.0007 | 0.0009 +/- 0.0002 |
| `without_dns` | prismflow | resolution | 0.0058 +/- 0.0003 | 0.0064 +/- 0.0004 | 0.0000 +/- 0.0000 | 0.0061 +/- 0.0002 | 0.0035 +/- 0.0006 |
| `with_auth` | naive | reliability | 0.0006 +/- 0.0002 | 0.0006 +/- 0.0002 | not measured | 0.0033 +/- 0.0008 | 0.0005 +/- 0.0002 |
| `with_auth` | naive | resolution | 0.0062 +/- 0.0005 | 0.0062 +/- 0.0005 | not measured | 0.0063 +/- 0.0005 | 0.0032 +/- 0.0005 |
| `with_auth` | prismflow | reliability | 0.0007 +/- 0.0002 | 0.0007 +/- 0.0002 | not measured | 0.0045 +/- 0.0006 | 0.0008 +/- 0.0002 |
| `with_auth` | prismflow | resolution | 0.0066 +/- 0.0003 | 0.0066 +/- 0.0003 | not measured | 0.0069 +/- 0.0003 | 0.0039 +/- 0.0007 |

### 1.3 Row counts and the subset's own baseline

| arm | test rows | present rows (attack) | absent rows (attack) | present-subset prevalence | its own 2*pi*(1-pi) |
|---|---:|---|---|---:|---:|
| `without_auth` | 18729 | 15129 (271) | 3600 (139) | 0.0179 | 0.0352 |
| `without_flows` | 18729 | 17731 (410) | 998 (0) | 0.0231 | 0.0452 |
| `without_dns` | 18729 | 18656 (410) | 73 (0) | 0.0220 | 0.0430 |
| `with_auth` | 18729 | 18729 (410) | 0 (0) | 0.0219 | 0.0428 |

The rule is read against the fixed 0.0428. The subset's own baseline is context, not the rule: on the present-row subset the prevalence changes, so a constant predictor there is a different number.

**What the absent rows actually carry.** On every row whose arm views are all absent, in both systems and all 5 seeds, `prob_attack` is exactly 0.5 and the saved `vacuity_confidence` is exactly 0.0 - the least confident value it can take. A constant 0.5 scores the maximum two-class Brier of 0.5 on every such row, whatever its label, which is the 0.5000 +/- 0.0000 column above. Observed in Part 30's committed scores; nothing here changes them.

| arm | absent rows | their share of the test block | their contribution to Brier |
|---|---:|---:|---:|
| `without_auth` | 3600 | 0.1922 | 0.0961 |
| `without_flows` | 998 | 0.0533 | 0.0266 |
| `without_dns` | 73 | 0.0039 | 0.0019 |
| `with_auth` | 0 | 0.0000 | 0.0000 |

Against a whole budget of 0.0428, that column is the arithmetic of L29.

### 1.4 Verdict per cause

| arm | cause | seeds below 0.0428 (naive / prismflow) | accounts for the failure? |
|---|---|---|---|
| `without_auth` | (a) all-False view mask | 5/5 / 5/5 | **YES** |
| `without_auth` | (b) prior shift (ORACLE) | 0/5 / 0/5 | no |
| `without_auth` | (c) cross-fitted Platt (DIAGNOSTIC) | 0/5 / 0/5 | no |
| `without_flows` | (a) all-False view mask | 5/5 / 5/5 | **YES** |
| `without_flows` | (b) prior shift (ORACLE) | 0/5 / 0/5 | no |
| `without_flows` | (c) cross-fitted Platt (DIAGNOSTIC) | 4/5 / 5/5 | no |
| `without_dns` | (a) all-False view mask | 5/5 / 5/5 | **YES** |
| `without_dns` | (b) prior shift (ORACLE) | 2/5 / 0/5 | no |
| `without_dns` | (c) cross-fitted Platt (DIAGNOSTIC) | 5/5 / 5/5 | **YES** |
| `with_auth` | (a) all-False view mask | 5/5 / 5/5 | **YES** |
| `with_auth` | (b) prior shift (ORACLE) | 5/5 / 5/5 | **YES** |
| `with_auth` | (c) cross-fitted Platt (DIAGNOSTIC) | 5/5 / 5/5 | **YES** |

- `without_auth`: accounted for by (a) all-False view mask.
- `without_flows`: accounted for by (a) all-False view mask.

**"Accounts for" is not "fixed".** The rows cause (a) names are unpredictable by construction (Part 30 section 4.2): every model scores them identically whatever the label. Restricting the metric to rows with a present view changes the evaluated population, it does not change a probability - a deployed detector still meets those cells and still emits 0.5. The prior shift makes both failing arms WORSE, and it is an ORACLE. The Platt diagnostic does not meet the rule either. So the operative sentence of L29 is unchanged: the probability outputs of those arms should not be used.

## 2. L23 - metrics at the population rate

Inverse-probability weights: attack **1**, benign **512**. The declared control divisor of configs/data/lanl.yaml: all cases kept, a benign cell kept iff crc32(computer|window) % 512 == 0. The inverse-probability weight of a kept control is therefore 512.

**The weights reconstruct the occupied-cell population only as far as the crc32 hash is uniform over cells. Uniformity is assumed, not measured here, and the weighted counts are estimates of population counts, not counts. The resolution of a weighted false-positive count is 512 cells: no weighted alarm count between 0 and 512 is representable.**

Population cells, days 1-16: between **23,627,216** and **60,634,229** occupied `(computer, window)` cells (`results/lanl/population.json`). Over the test days (13-16, 4 days) alone: between **6,554,950** and **15,730,712**. Summed from population.json population_cells_per_day over days 13-16. The union over views is bounded below by the largest single view and above by the sum, for the same reason population.json gives for the all-day figure.

| view | occupied cells, days 1-16 | occupied cells, test days |
|---|---:|---:|
| `auth` | 22,941,894 | 6,357,376 |
| `proc` | 23,627,216 | 6,554,950 |
| `flows` | 13,484,366 | 2,652,478 |
| `dns` | 580,753 | 165,908 |

### 2.1 Weighted Brier, and what the sample prevalence becomes

| arm | system | sample prevalence | weighted prevalence | weighted Brier |
|---|---|---:|---:|---|
| `without_auth` | naive | 0.021891 | 0.00004371 | 0.097901 +/- 0.000881 |
| `without_auth` | prismflow | 0.021891 | 0.00004371 | 0.098170 +/- 0.000824 |
| `without_flows` | naive | 0.021891 | 0.00004371 | 0.030717 +/- 0.000455 |
| `without_flows` | prismflow | 0.021891 | 0.00004371 | 0.031319 +/- 0.000478 |
| `without_dns` | naive | 0.021891 | 0.00004371 | 0.005472 +/- 0.000437 |
| `without_dns` | prismflow | 0.021891 | 0.00004371 | 0.005936 +/- 0.000623 |
| `with_auth` | naive | 0.021891 | 0.00004371 | 0.003592 +/- 0.000472 |
| `with_auth` | prismflow | 0.021891 | 0.00004371 | 0.004228 +/- 0.000559 |

### 2.2 Population-weighted precision, recall and FPR

Targets are per TEST DAY. `achieved` is what the threshold actually delivers: a weighted false-positive count moves in steps of 512 cells, so over 4 test days no rate between 0 and 128 per day is representable. All three pre-registered targets - 1, 10 and 100 false alarms per day - are below that step, so all three collapse onto the same zero-alarm threshold. **A weighted precision of 1.0 there means no SAMPLED control fired**: it estimates the population false-alarm count as 0 with a resolution of 512 cells, and is not a detector with no false alarms.

| arm | system | operating point | target FA/day | achieved FA/day | threshold | precision | recall | FPR | sampled alarms |
|---|---|---|---:|---|---|---|---|---|---|
| `without_auth` | naive | default | 0.5 prob | 443289.6 +/- 166.9 | 0.500000 +/- 0.000000 | 0.000090 +/- 0.000000 | 0.3912 +/- 0.0013 | 0.18904962 +/- 0.00007117 | 3623.6 +/- 1.8 |
| `without_auth` | naive | fa_1_per_day | 1 | 0.0 +/- 0.0 | 0.871120 +/- 0.070572 | 1.000000 +/- 0.000000 | 0.0468 +/- 0.0040 | 0.00000000 +/- 0.00000000 | 19.2 +/- 1.6 |
| `without_auth` | naive | fa_10_per_day | 10 | 0.0 +/- 0.0 | 0.871120 +/- 0.070572 | 1.000000 +/- 0.000000 | 0.0468 +/- 0.0040 | 0.00000000 +/- 0.00000000 | 19.2 +/- 1.6 |
| `without_auth` | naive | fa_100_per_day | 100 | 0.0 +/- 0.0 | 0.871120 +/- 0.070572 | 1.000000 +/- 0.000000 | 0.0468 +/- 0.0040 | 0.00000000 +/- 0.00000000 | 19.2 +/- 1.6 |
| `without_auth` | prismflow | default | 0.5 prob | 443212.8 +/- 114.5 | 0.500000 +/- 0.000000 | 0.000090 +/- 0.000001 | 0.3907 +/- 0.0032 | 0.18901687 +/- 0.00004883 | 3622.8 +/- 1.9 |
| `without_auth` | prismflow | fa_1_per_day | 1 | 0.0 +/- 0.0 | 0.797833 +/- 0.100550 | 1.000000 +/- 0.000000 | 0.0483 +/- 0.0027 | 0.00000000 +/- 0.00000000 | 19.8 +/- 1.1 |
| `without_auth` | prismflow | fa_10_per_day | 10 | 0.0 +/- 0.0 | 0.797833 +/- 0.100550 | 1.000000 +/- 0.000000 | 0.0483 +/- 0.0027 | 0.00000000 +/- 0.00000000 | 19.8 +/- 1.1 |
| `without_auth` | prismflow | fa_100_per_day | 100 | 0.0 +/- 0.0 | 0.797833 +/- 0.100550 | 1.000000 +/- 0.000000 | 0.0483 +/- 0.0027 | 0.00000000 +/- 0.00000000 | 19.8 +/- 1.1 |
| `without_flows` | naive | default | 0.5 prob | 128742.4 +/- 189.9 | 0.500000 +/- 0.000000 | 0.000089 +/- 0.000008 | 0.1112 +/- 0.0095 | 0.05490474 +/- 0.00008097 | 1051.4 +/- 3.9 |
| `without_flows` | naive | fa_1_per_day | 1 | 0.0 +/- 0.0 | 0.948771 +/- 0.018965 | 1.000000 +/- 0.000000 | 0.0395 +/- 0.0160 | 0.00000000 +/- 0.00000000 | 16.2 +/- 6.6 |
| `without_flows` | naive | fa_10_per_day | 10 | 0.0 +/- 0.0 | 0.948771 +/- 0.018965 | 1.000000 +/- 0.000000 | 0.0395 +/- 0.0160 | 0.00000000 +/- 0.00000000 | 16.2 +/- 6.6 |
| `without_flows` | naive | fa_100_per_day | 100 | 0.0 +/- 0.0 | 0.948771 +/- 0.018965 | 1.000000 +/- 0.000000 | 0.0395 +/- 0.0160 | 0.00000000 +/- 0.00000000 | 16.2 +/- 6.6 |
| `without_flows` | prismflow | default | 0.5 prob | 128947.2 +/- 194.1 | 0.500000 +/- 0.000000 | 0.000085 +/- 0.000011 | 0.1073 +/- 0.0143 | 0.05499208 +/- 0.00008279 | 1051.4 +/- 5.3 |
| `without_flows` | prismflow | fa_1_per_day | 1 | 0.0 +/- 0.0 | 0.956250 +/- 0.020871 | 1.000000 +/- 0.000000 | 0.0185 +/- 0.0188 | 0.00000000 +/- 0.00000000 | 7.6 +/- 7.7 |
| `without_flows` | prismflow | fa_10_per_day | 10 | 0.0 +/- 0.0 | 0.956250 +/- 0.020871 | 1.000000 +/- 0.000000 | 0.0185 +/- 0.0188 | 0.00000000 +/- 0.00000000 | 7.6 +/- 7.7 |
| `without_flows` | prismflow | fa_100_per_day | 100 | 0.0 +/- 0.0 | 0.956250 +/- 0.020871 | 1.000000 +/- 0.000000 | 0.0185 +/- 0.0188 | 0.00000000 +/- 0.00000000 | 7.6 +/- 7.7 |
| `without_dns` | naive | default | 0.5 prob | 10598.4 +/- 826.6 | 0.500000 +/- 0.000000 | 0.001230 +/- 0.000236 | 0.1263 +/- 0.0184 | 0.00451990 +/- 0.00035251 | 134.6 +/- 7.7 |
| `without_dns` | naive | fa_1_per_day | 1 | 0.0 +/- 0.0 | 0.956117 +/- 0.030836 | 1.000000 +/- 0.000000 | 0.0590 +/- 0.0150 | 0.00000000 +/- 0.00000000 | 24.2 +/- 6.1 |
| `without_dns` | naive | fa_10_per_day | 10 | 0.0 +/- 0.0 | 0.956117 +/- 0.030836 | 1.000000 +/- 0.000000 | 0.0590 +/- 0.0150 | 0.00000000 +/- 0.00000000 | 24.2 +/- 6.1 |
| `without_dns` | naive | fa_100_per_day | 100 | 0.0 +/- 0.0 | 0.956117 +/- 0.030836 | 1.000000 +/- 0.000000 | 0.0590 +/- 0.0150 | 0.00000000 +/- 0.00000000 | 24.2 +/- 6.1 |
| `without_dns` | prismflow | default | 0.5 prob | 10342.4 +/- 291.9 | 0.500000 +/- 0.000000 | 0.001232 +/- 0.000152 | 0.1244 +/- 0.0150 | 0.00441072 +/- 0.00012448 | 131.8 +/- 6.7 |
| `without_dns` | prismflow | fa_1_per_day | 1 | 0.0 +/- 0.0 | 0.956757 +/- 0.027364 | 1.000000 +/- 0.000000 | 0.0541 +/- 0.0135 | 0.00000000 +/- 0.00000000 | 22.2 +/- 5.5 |
| `without_dns` | prismflow | fa_10_per_day | 10 | 0.0 +/- 0.0 | 0.956757 +/- 0.027364 | 1.000000 +/- 0.000000 | 0.0541 +/- 0.0135 | 0.00000000 +/- 0.00000000 | 22.2 +/- 5.5 |
| `without_dns` | prismflow | fa_100_per_day | 100 | 0.0 +/- 0.0 | 0.956757 +/- 0.027364 | 1.000000 +/- 0.000000 | 0.0541 +/- 0.0135 | 0.00000000 +/- 0.00000000 | 22.2 +/- 5.5 |
| `with_auth` | naive | default | 0.5 prob | 1459.2 +/- 484.0 | 0.500000 +/- 0.000000 | 0.009390 +/- 0.002252 | 0.1273 +/- 0.0199 | 0.00062230 +/- 0.00020643 | 63.6 +/- 10.8 |
| `with_auth` | naive | fa_1_per_day | 1 | 0.0 +/- 0.0 | 0.956168 +/- 0.026990 | 1.000000 +/- 0.000000 | 0.0600 +/- 0.0051 | 0.00000000 +/- 0.00000000 | 24.6 +/- 2.1 |
| `with_auth` | naive | fa_10_per_day | 10 | 0.0 +/- 0.0 | 0.956168 +/- 0.026990 | 1.000000 +/- 0.000000 | 0.0600 +/- 0.0051 | 0.00000000 +/- 0.00000000 | 24.6 +/- 2.1 |
| `with_auth` | naive | fa_100_per_day | 100 | 0.0 +/- 0.0 | 0.956168 +/- 0.026990 | 1.000000 +/- 0.000000 | 0.0600 +/- 0.0051 | 0.00000000 +/- 0.00000000 | 24.6 +/- 2.1 |
| `with_auth` | prismflow | default | 0.5 prob | 1459.2 +/- 448.9 | 0.500000 +/- 0.000000 | 0.010981 +/- 0.004255 | 0.1444 +/- 0.0108 | 0.00062230 +/- 0.00019145 | 70.6 +/- 4.8 |
| `with_auth` | prismflow | fa_1_per_day | 1 | 0.0 +/- 0.0 | 0.963415 +/- 0.035516 | 1.000000 +/- 0.000000 | 0.0502 +/- 0.0165 | 0.00000000 +/- 0.00000000 | 20.6 +/- 6.8 |
| `with_auth` | prismflow | fa_10_per_day | 10 | 0.0 +/- 0.0 | 0.963415 +/- 0.035516 | 1.000000 +/- 0.000000 | 0.0502 +/- 0.0165 | 0.00000000 +/- 0.00000000 | 20.6 +/- 6.8 |
| `with_auth` | prismflow | fa_100_per_day | 100 | 0.0 +/- 0.0 | 0.963415 +/- 0.035516 | 1.000000 +/- 0.000000 | 0.0502 +/- 0.0165 | 0.00000000 +/- 0.00000000 | 20.6 +/- 6.8 |

## 3. L26 - intervals on the seen and unseen AUCs

**Gate passed before any interval.** `total_records_across_views`, recomputed from `results/lanl/samples.npz` with Part 30's train-fixed orientation (as-is, train AUC 0.8164), gives test AUC **0.6843** against Part 30's committed 0.6843 (results/lanl/results.json :: phase3_diagnostics.trivial_baselines.baselines.total_records_across_views.test_auc).

Computer-cluster bootstrap: **2000** replicates, fixed seed **12345**, clusters = the **8,350** test computers, drawn with replacement. For a model arm and system, the MEAN OVER THE 5 SEEDS of the ROC AUC on the replicate's rows. For the untrained scalar, the ROC AUC itself (no seed). AUC is the Mann-Whitney form with ties credited 0.5, Part 28's roc_auc_binary, imported unedited. Interval: percentile, 2.5 and 97.5.

Groups: **255** seen and **155** unseen attack rows, each against all **18,319** benign test rows.

| arm / system | seen AUC | seen 95% CI | unseen AUC | unseen 95% CI |
|---|---|---|---|---|
| `without_auth/naive` | 0.8153 | [0.7019, 0.8560] | 0.7102 | [0.6787, 0.7384] |
| `without_auth/prismflow` | 0.8168 | [0.7083, 0.8563] | 0.7126 | [0.6819, 0.7400] |
| `without_flows/naive` | 0.8563 | [0.7710, 0.8855] | 0.7435 | [0.7111, 0.7771] |
| `without_flows/prismflow` | 0.8674 | [0.7879, 0.8953] | 0.7508 | [0.7205, 0.7832] |
| `without_dns/naive` | 0.8793 | [0.8139, 0.9011] | 0.7584 | [0.7171, 0.7951] |
| `without_dns/prismflow` | 0.8875 | [0.8244, 0.9080] | 0.7589 | [0.7184, 0.7956] |
| `with_auth/naive` | 0.8686 | [0.7819, 0.8961] | 0.7606 | [0.7208, 0.7967] |
| `with_auth/prismflow` | 0.8837 | [0.8034, 0.9097] | 0.7614 | [0.7213, 0.7973] |
| `total_records_across_views` (untrained) | 0.6405 | [0.5347, 0.8990] | 0.7563 | [0.7030, 0.8078] |

### 3.1 Paired model-minus-scalar on unseen computers

Paired: both AUCs are computed on the same resampled computers in every replicate, so the draw cancels.

| arm / system | observed difference | 95% CI | excludes 0? |
|---|---|---|---|
| `without_auth/naive` | -0.0461 | [-0.1127, 0.0224] | no, includes 0 |
| `without_auth/prismflow` | -0.0436 | [-0.1106, 0.0250] | no, includes 0 |
| `without_flows/naive` | -0.0127 | [-0.0728, 0.0479] | no, includes 0 |
| `without_flows/prismflow` | -0.0055 | [-0.0625, 0.0534] | no, includes 0 |
| `without_dns/naive` | 0.0021 | [-0.0682, 0.0697] | no, includes 0 |
| `without_dns/prismflow` | 0.0026 | [-0.0675, 0.0703] | no, includes 0 |
| `with_auth/naive` | 0.0043 | [-0.0634, 0.0714] | no, includes 0 |
| `with_auth/prismflow` | 0.0052 | [-0.0639, 0.0723] | no, includes 0 |

## 4. L27 - leave one host out

The 5 computers with the most TEST rows of label attack, ranked by that count, ties broken by computer name ascending.

**Interval:** NONE. One host is not a resample, so no interval is attached and the result is descriptive.

Rank is by ATTACK rows, so a host lower in the table can still have more total test rows than the one above it.

| rank | computer | test rows removed | attack | benign |
|---:|---|---:|---:|---:|
| 1 | `C17693` | 138 | 137 | 1 |
| 2 | `C1015` | 13 | 12 | 1 |
| 3 | `C2388` | 13 | 10 | 3 |
| 4 | `C1936` | 7 | 7 | 0 |
| 5 | `C2079` | 10 | 7 | 3 |

| arm / system | all-test AUC | minus `C17693` | minus `C1015` | minus `C2388` | minus `C1936` | minus `C2079` |
|---|---|---|---|---|---|---|
| `without_auth/naive` | 0.7755 +/- 0.0025 | 0.7237 +/- 0.0037 (-0.0519) | 0.7753 +/- 0.0024 (-0.0003) | 0.7764 +/- 0.0024 (0.0008) | 0.7812 +/- 0.0025 (0.0057) | 0.7734 +/- 0.0026 (-0.0022) |
| `without_auth/prismflow` | 0.7775 +/- 0.0030 | 0.7267 +/- 0.0047 (-0.0508) | 0.7772 +/- 0.0031 (-0.0002) | 0.7782 +/- 0.0027 (0.0008) | 0.7832 +/- 0.0030 (0.0058) | 0.7753 +/- 0.0030 (-0.0021) |
| `without_flows/naive` | 0.8137 +/- 0.0022 | 0.7702 +/- 0.0042 (-0.0435) | 0.8101 +/- 0.0023 (-0.0036) | 0.8160 +/- 0.0025 (0.0023) | 0.8172 +/- 0.0022 (0.0035) | 0.8128 +/- 0.0023 (-0.0009) |
| `without_flows/prismflow` | 0.8233 +/- 0.0031 | 0.7800 +/- 0.0042 (-0.0434) | 0.8202 +/- 0.0030 (-0.0032) | 0.8249 +/- 0.0035 (0.0015) | 0.8259 +/- 0.0033 (0.0025) | 0.8225 +/- 0.0032 (-0.0009) |
| `without_dns/naive` | 0.8336 +/- 0.0029 | 0.7976 +/- 0.0036 (-0.0360) | 0.8303 +/- 0.0026 (-0.0032) | 0.8340 +/- 0.0029 (0.0004) | 0.8377 +/- 0.0032 (0.0042) | 0.8317 +/- 0.0032 (-0.0018) |
| `without_dns/prismflow` | 0.8389 +/- 0.0058 | 0.8021 +/- 0.0092 (-0.0368) | 0.8352 +/- 0.0059 (-0.0037) | 0.8390 +/- 0.0060 (0.0001) | 0.8424 +/- 0.0056 (0.0035) | 0.8369 +/- 0.0057 (-0.0019) |
| `with_auth/naive` | 0.8278 +/- 0.0033 | 0.7899 +/- 0.0044 (-0.0378) | 0.8245 +/- 0.0029 (-0.0033) | 0.8287 +/- 0.0036 (0.0010) | 0.8347 +/- 0.0034 (0.0069) | 0.8257 +/- 0.0034 (-0.0021) |
| `with_auth/prismflow` | 0.8375 +/- 0.0043 | 0.7974 +/- 0.0064 (-0.0400) | 0.8340 +/- 0.0041 (-0.0034) | 0.8379 +/- 0.0044 (0.0004) | 0.8434 +/- 0.0040 (0.0059) | 0.8356 +/- 0.0043 (-0.0019) |

The bracketed figure is the change from that unit's own all-test AUC.

## 5. L28 - with_auth across two execution modes

`with_auth` only. run 1: 12 threads, sequential (`results/lanl/results.run1.json.gz`). run 2: 1 thread, concurrency 11 (`results/lanl/results.json`).

**The other three arms were run in ONE execution mode only, so no cross-mode spread exists for them. They stay OPEN until Part 38.** Arms still OPEN: `without_auth`, `without_flows`, `without_dns`.

| system | metric | run 1 mean | run 1 std | run 2 mean | run 2 std | run2 - run1 | max same-seed gap | pooled std (10) | pooled / larger within-mode std |
|---|---|---|---|---|---|---|---|---|---|
| naive | `roc_auc` | 0.8308 | 0.0039 | 0.8278 | 0.0033 | -0.0031 | 0.0140 | 0.0038 | 0.96 |
| naive | `prob_ece` | 0.0085 | 0.0002 | 0.0084 | 0.0009 | -0.0001 | 0.0008 | 0.0006 | 0.69 |
| naive | `benign_recall` | 0.9995 | 0.0001 | 0.9994 | 0.0002 | -0.0000 | 0.0004 | 0.0002 | 0.76 |
| naive | `accuracy` | 0.9802 | 0.0003 | 0.9801 | 0.0003 | -0.0000 | 0.0006 | 0.0003 | 0.87 |
| naive | `balanced_accuracy` | 0.5585 | 0.0081 | 0.5587 | 0.0088 | 0.0002 | 0.0146 | 0.0080 | 0.91 |
| naive | `attack_recall` | 0.1176 | 0.0163 | 0.1180 | 0.0176 | 0.0005 | 0.0293 | 0.0160 | 0.91 |
| naive | `brier` | 0.0369 | 0.0005 | 0.0370 | 0.0005 | 0.0001 | 0.0010 | 0.0005 | 0.94 |
| prismflow | `roc_auc` | 0.8378 | 0.0036 | 0.8375 | 0.0043 | -0.0003 | 0.0042 | 0.0038 | 0.87 |
| prismflow | `prob_ece` | 0.0072 | 0.0007 | 0.0076 | 0.0011 | 0.0004 | 0.0020 | 0.0009 | 0.79 |
| prismflow | `benign_recall` | 0.9996 | 0.0002 | 0.9995 | 0.0002 | -0.0001 | 0.0002 | 0.0002 | 0.96 |
| prismflow | `accuracy` | 0.9802 | 0.0004 | 0.9806 | 0.0003 | 0.0003 | 0.0007 | 0.0004 | 0.96 |
| prismflow | `balanced_accuracy` | 0.5581 | 0.0098 | 0.5676 | 0.0073 | 0.0095 | 0.0170 | 0.0095 | 0.98 |
| prismflow | `attack_recall` | 0.1166 | 0.0196 | 0.1356 | 0.0146 | 0.0190 | 0.0341 | 0.0191 | 0.98 |
| prismflow | `brier` | 0.0369 | 0.0003 | 0.0366 | 0.0002 | -0.0003 | 0.0007 | 0.0003 | 0.88 |

## 6. Proposed status of each limitation

Proposals only. `docs/KNOWN_LIMITATIONS.md` is NOT edited by this Part; the text is in `docs/proposed_limitations_update.md` for the user to accept or reject.

| limitation | proposed status | why |
|---|---|---|
| L29 | **CLOSED** | L29's own "what would change our understanding" is "a test of each candidate cause, and a Brier below the constant baseline once the responsible one is fixed". All three named candidates were tested under a rule fixed before any number existed, one of them is responsible, and removing its rows does put Brier below the baseline. The responsible cause is NOT fixed, because the rows it names are unpredictable by construction - excluding them changes the evaluated population rather than the model - so the operative warning is unchanged. Proposed CLOSED as diagnosed; the user may prefer OPEN on the strength of the unfixed half, and that is their call. |
| L23 | **CLOSED** | L23 asks for "a measurement at, or reweighted to, the population prevalence". That measurement now exists, and it confirms the limitation rather than lifting it: the reweighting is only as good as the uniformity of the control hash, and its false-alarm resolution is 512 cells, so it still gives no operating point that transfers. |
| L26 | **CLOSED** | L26 asks for "an unseen-computer result that clears the scalar by more than 0.01 with an interval attached". The interval is now attached, by the pre-registered 2000-replicate computer-cluster bootstrap, and the result confirms the limitation: NO arm-and-system pair clears that bar, and every paired model-minus-scalar interval on unseen computers includes 0. Closed as confirmed, with the numbers whichever way they fell. |
| L27 | **OPEN** | L27's "what would change our understanding" is a DESIGN change - a sample that keeps all or a stated share of the benign cells of attack hosts. That needs a re-sample and a retrain, which is Part 38. Part 35 only widens the sensitivity measurement from one host to five, so L27 stays OPEN with new evidence attached. |
| L28 | **OPEN** | L28 asks for "seeds run across at least two execution modes, so the reported std covers mode as well as seed". Only `with_auth` has two modes in `results/lanl/`; the quantity is now reported per metric for that arm, and the other three arms have no second mode to compare, so they stay OPEN until Part 38 runs them. |

## 7. What this Part cannot do

- **It cannot recalibrate anything deployably.** `results/lanl/` holds test-block scores only. There are no train-block scores, so every fit here is cross-fitted inside the test block and labelled DIAGNOSTIC.
- **It cannot retrain.** No arm is re-fitted, no view is rescanned and no hyperparameter is changed, so nothing here can fix a cause it diagnoses.
- **It cannot give a population operating point.** The weighted estimates of section 2 inherit the control hash's uniformity assumption and resolve false-alarm counts only in steps of 512 cells.
- **It cannot separate host from behaviour.** Section 4 measures how much each AUC rests on one host; the sample that would separate them does not exist.
- **It cannot speak for the three single-mode arms.** Section 5 covers `with_auth` alone, because only that arm was run twice.
- **The bootstrap interval is test-set sampling only.** It is not the per-seed std and not a spread over attack episodes: the split is fixed and the red-team activity is concentrated in a few days.

## 8. Files

| path | what it holds |
|---|---|
| `experiments/lanl_reanalysis/config.yaml` | the pre-registration, committed first |
| `experiments/lanl_reanalysis/run.py` | this analysis and this README's generator |
| `results/lanl_reanalysis/results.json` | every number above, with its provenance |
| `results/lanl_reanalysis/integrity.json` | the sha256 of every input, as checked |
| `tests/unit/test_lanl_reanalysis.py` | weighting, fold assignment, bootstrap helper |
| `docs/proposed_limitations_update.md` | the proposed limitation text |

Elapsed: 132.5 s, analysis only - no training, nothing rescanned. git HEAD at the time of the run: `eca3cf2fd912b3497e98eb3da2b9663f0ca9d0e6` - `run.py` is committed in the step that follows the run, so HEAD here is the step before it.

