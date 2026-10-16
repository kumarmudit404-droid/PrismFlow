SUPERSEDED by summary.md - kept as run evidence
PARTIAL - 1 of 4 units (A/naive only). Not a result.

# Part 29 - attack and detection on real AIT data

Produced by `experiments/ait_attack/run_ait_attack.py` with `experiments/ait_attack/config.yaml`, commit `8828dba`.

## What the label means

A window is attack=1 if ANY officially labelled line falls in it, and on both scenarios that label is dominated by one continuously-running DNS exfiltration tool (1,026 of 1,031 attack windows on russellmitchell and 663 of 667 on santos carry only dnsteal-family steps, docs/part26-ait-loader.md sections 0 and 12), so the task is 'was the exfiltration tool running in this window', NOT 'detect an APT' and not 'detect a multi-step intrusion'.

## What a seed varies, and what it does not

Train and test are whole scenarios, so a seed cannot redraw or re-partition the data. A seed selects the model's parameter initialisation and the minibatch order of every training epoch. **Every +/- below is initialisation-and-batch-order variance on a fixed pair of scenarios.** It is not dataset variance and it is not a spread over independent attacks: the test sets hold 3 (`santos`) and 1 (`russellmitchell`) independent contiguous attack blocks (`docs/KNOWN_LIMITATIONS.md` L16).

## The dns split, and why nothing is pooled across it

`docs/KNOWN_LIMITATIONS.md` L17: one view solves this label. So every test-B number below is reported separately for compromised sets that contain `dns` and sets that do not. There is no row, mean or verdict anywhere in this file that averages the two, and `assert_no_pooling` raises rather than produce one.

---

# Direction A_rm_to_santos

A: train russellmitchell -> test santos. Train n = 1,145, test n = 1,108 (`label_known` rows only).

## Feature scale, and why epsilon is not rescaled

| view | std | IQR | median abs | max abs | present rows |
|---|---|---|---|---|---|
| `apache` | 16.2177 | 1.3901 | 0.6743 | 1374.93 | 803 |
| `auth` | 1.6059 | 0.3577 | 0.1638 | 71.52 | 586 |
| `dns` | 1.5639 | 2.0759 | 0.9727 | 7.11 | 1,101 |
| `audit` | 0.8423 | 1.1355 | 0.5231 | 7.22 | 898 |
| `suricata` | 1.5284 | 1.8317 | 1.0214 | 13.53 | 1,108 |

Median per-view std **1.5639** against Part 09's synthetic reference **1.3**, range 0.8423 to 16.2177. Epsilon is **not rescaled**; `config.yaml` records why.

## Test A: natural windows, no synthetic perturbation

The two attack-window populations are reported separately and never pooled. `multi_step` uses the STRICT definition (a step outside `EXFILTRATION_TOOL_STEPS`).

benign + dnsteal_only + multi_step + exfiltration_service_only accounts for every label_known row. The last population is the single window the strict and loose multi-step definitions differ by; it is reported separately and pooled with neither.

**3 windows on santos and 4 on russellmitchell. Any claim about the multi-step intrusion is a handful-of-windows result: descriptive only, no ROC AUC, no significance claim, and no mean +/- std over seeds presented as a finding.**

### naive

| population | windows | ENIV (eigen) | ENIV (design effect) | mean dependence | detector mean score |
|---|---|---|---|---|---|
| `benign` | 441 | 3.1332 +/- 0.0888 (5 seeds), n=441 windows | 2.0973 +/- 0.0502 (5 seeds), n=441 windows | 0.3463 +/- 0.0144 (5 seeds), n=441 windows | not measured (n=441 windows) |
| `dnsteal_only` | 663 | 3.2384 +/- 0.0745 (5 seeds), n=663 windows | 2.2735 +/- 0.0552 (5 seeds), n=663 windows | 0.3001 +/- 0.0132 (5 seeds), n=663 windows | not measured (n=663 windows) |
| `multi_step` **DEGENERATE** | 3 | not measured (n=3 windows, degenerate) | not measured (n=3 windows, degenerate) | not measured (n=3 windows, degenerate) | not measured (n=3 windows) |
| `exfiltration_service_only` **DEGENERATE** | 1 | not measured (n=1 windows, degenerate) | not measured (n=1 windows, degenerate) | not measured (n=1 windows, degenerate) | not measured (n=1 windows) |

**DEGENERATE: `multi_step` (n=3), `exfiltration_service_only` (n=1).** At these sample sizes every off-diagonal dependence entry comes back exactly 0, so ENIV pins at the view count (5.0000) and the efficiency ratio at 1.0 in all 5 seeds, with std exactly 0. That is the estimator reporting that it had nothing to work with, NOT a measurement that the five views are independent on these windows. The raw pinned values are in `results.json`; they are not a finding and must not be quoted as one.

## Test B: chorus attack, split by whether dns is compromised

Subsets: **exhaustive**, 25 in total (11 containing `dns`, 14 not). Sampled: **False**. All subsets of every k were run. No cell was sampled, so no cell is presented as exhaustive when it is not.

### chorus, epsilon 0.2, beta 1.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0354 +/- 0.0037 (1x5) | 3.4534 +/- 0.0844 (1x5) | not measured | not measured |
| 1 | no | 4 | 0.0093 +/- 0.0093 (4x5) | 3.4709 +/- 0.0785 (4x5) | not measured | not measured |
| 2 | **yes** | 4 | 0.0306 +/- 0.0066 (4x5) | 3.4552 +/- 0.0758 (4x5) | 0.3734 +/- 0.2972 (4x5) | not measured |
| 2 | no | 6 | 0.0174 +/- 0.0153 (6x5) | 3.4800 +/- 0.0808 (6x5) | 0.2796 +/- 0.1900 (6x5) | not measured |
| 3 | **yes** | 6 | 0.0317 +/- 0.0063 (6x5) | 3.4781 +/- 0.0712 (6x5) | 0.3360 +/- 0.1505 (6x5) | not measured |
| 3 | no | 4 | 0.0140 +/- 0.0037 (4x5) | 3.4865 +/- 0.0801 (4x5) | 0.2618 +/- 0.0916 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 0.5, beta 1.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0523 +/- 0.0046 (1x5) | 3.4633 +/- 0.0793 (1x5) | not measured | not measured |
| 1 | no | 4 | 0.0405 +/- 0.0329 (4x5) | 3.4787 +/- 0.1330 (4x5) | not measured | not measured |
| 2 | **yes** | 4 | 0.1063 +/- 0.0213 (4x5) | 3.4653 +/- 0.0973 (4x5) | 0.4560 +/- 0.2600 (4x5) | not measured |
| 2 | no | 6 | 0.0926 +/- 0.0852 (6x5) | 3.4141 +/- 0.1355 (6x5) | 0.3447 +/- 0.1612 (6x5) | not measured |
| 3 | **yes** | 6 | 0.1135 +/- 0.0194 (6x5) | 3.4697 +/- 0.1018 (6x5) | 0.4144 +/- 0.1696 (6x5) | not measured |
| 3 | no | 4 | 0.0931 +/- 0.0439 (4x5) | 3.3571 +/- 0.1057 (4x5) | 0.2981 +/- 0.0669 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 1.0, beta 0.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.1298 +/- 0.0603 (4x5) | 3.5260 +/- 0.1700 (4x5) | 0.5500 +/- 0.0998 (4x5) | not measured |
| 2 | no | 6 | 0.1594 +/- 0.0553 (6x5) | 3.4185 +/- 0.1860 (6x5) | 0.2465 +/- 0.1340 (6x5) | not measured |
| 3 | **yes** | 6 | 0.1972 +/- 0.0518 (6x5) | 3.4462 +/- 0.1951 (6x5) | 0.5151 +/- 0.0757 (6x5) | not measured |
| 3 | no | 4 | 0.2076 +/- 0.0267 (4x5) | 3.2471 +/- 0.1351 (4x5) | 0.2938 +/- 0.0691 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 1.0, beta 1.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0538 +/- 0.0040 (1x5) | 3.4988 +/- 0.0798 (1x5) | not measured | not measured |
| 1 | no | 4 | 0.0870 +/- 0.0663 (4x5) | 3.4682 +/- 0.1897 (4x5) | not measured | not measured |
| 2 | **yes** | 4 | 0.2559 +/- 0.0741 (4x5) | 3.3615 +/- 0.1729 (4x5) | 0.6579 +/- 0.1432 (4x5) | not measured |
| 2 | no | 6 | 0.1696 +/- 0.0931 (6x5) | 3.3283 +/- 0.1872 (6x5) | 0.4689 +/- 0.2101 (6x5) | not measured |
| 3 | **yes** | 6 | 0.3304 +/- 0.1215 (6x5) | 3.2364 +/- 0.1972 (6x5) | 0.5998 +/- 0.1068 (6x5) | not measured |
| 3 | no | 4 | 0.1814 +/- 0.0575 (4x5) | 3.0506 +/- 0.1551 (4x5) | 0.5267 +/- 0.0885 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

#### Every subset at the primary cell - naive

| compromised views | k | dns | attack success (5 seeds) | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| `dns` | 1 | **yes** | 0.0538 +/- 0.0040 (n=5) | 3.4988 +/- 0.0798 (n=5) | not measured | not measured |
| `apache` | 1 | no | 0.1038 +/- 0.0094 (n=5) | 3.2089 +/- 0.0965 (n=5) | not measured | not measured |
| `audit` | 1 | no | 0.0258 +/- 0.0066 (n=5) | 3.4297 +/- 0.0419 (n=5) | not measured | not measured |
| `auth` | 1 | no | 0.0345 +/- 0.0048 (n=5) | 3.6375 +/- 0.1058 (n=5) | not measured | not measured |
| `suricata` | 1 | no | 0.1838 +/- 0.0228 (n=5) | 3.5965 +/- 0.0806 (n=5) | not measured | not measured |
| `apache+dns` | 2 | **yes** | 0.2273 +/- 0.0092 (n=5) | 3.1051 +/- 0.1294 (n=5) | 0.6287 +/- 0.0855 (n=5) | not measured |
| `auth+dns` | 2 | **yes** | 0.2078 +/- 0.0042 (n=5) | 3.4791 +/- 0.0481 (n=5) | 0.5419 +/- 0.0987 (n=5) | not measured |
| `dns+audit` | 2 | **yes** | 0.2146 +/- 0.0039 (n=5) | 3.4552 +/- 0.0839 (n=5) | 0.6019 +/- 0.0865 (n=5) | not measured |
| `dns+suricata` | 2 | **yes** | 0.3740 +/- 0.0497 (n=5) | 3.4066 +/- 0.0538 (n=5) | 0.8590 +/- 0.0238 (n=5) | not measured |
| `apache+audit` | 2 | no | 0.1211 +/- 0.0110 (n=5) | 3.3436 +/- 0.1071 (n=5) | 0.4195 +/- 0.1241 (n=5) | not measured |
| `apache+auth` | 2 | no | 0.0917 +/- 0.0075 (n=5) | 3.4248 +/- 0.1005 (n=5) | 0.2351 +/- 0.0967 (n=5) | not measured |
| `apache+suricata` | 2 | no | 0.3274 +/- 0.0043 (n=5) | 3.1581 +/- 0.1266 (n=5) | 0.4850 +/- 0.0772 (n=5) | not measured |
| `audit+suricata` | 2 | no | 0.2034 +/- 0.0664 (n=5) | 3.1994 +/- 0.0672 (n=5) | 0.8500 +/- 0.0392 (n=5) | not measured |
| `auth+audit` | 2 | no | 0.0666 +/- 0.0074 (n=5) | 3.2173 +/- 0.0753 (n=5) | 0.4993 +/- 0.0603 (n=5) | not measured |
| `auth+suricata` | 2 | no | 0.2076 +/- 0.0034 (n=5) | 3.6266 +/- 0.1041 (n=5) | 0.3242 +/- 0.0604 (n=5) | not measured |
| `apache+auth+dns` | 3 | **yes** | 0.2088 +/- 0.0061 (n=5) | 3.2351 +/- 0.0974 (n=5) | 0.5111 +/- 0.0844 (n=5) | not measured |
| `apache+dns+audit` | 3 | **yes** | 0.2173 +/- 0.0091 (n=5) | 3.2421 +/- 0.1929 (n=5) | 0.5131 +/- 0.1299 (n=5) | not measured |
| `apache+dns+suricata` | 3 | **yes** | 0.4560 +/- 0.0398 (n=5) | 2.9983 +/- 0.1900 (n=5) | 0.6430 +/- 0.0810 (n=5) | not measured |
| `auth+dns+audit` | 3 | **yes** | 0.2139 +/- 0.0076 (n=5) | 3.1188 +/- 0.0791 (n=5) | 0.5883 +/- 0.0577 (n=5) | not measured |
| `auth+dns+suricata` | 3 | **yes** | 0.4440 +/- 0.0354 (n=5) | 3.4818 +/- 0.1103 (n=5) | 0.6152 +/- 0.0736 (n=5) | not measured |
| `dns+audit+suricata` | 3 | **yes** | 0.4426 +/- 0.0320 (n=5) | 3.3426 +/- 0.0550 (n=5) | 0.7282 +/- 0.0263 (n=5) | not measured |
| `apache+audit+suricata` | 3 | no | 0.2606 +/- 0.0221 (n=5) | 3.0410 +/- 0.1444 (n=5) | 0.5795 +/- 0.0655 (n=5) | not measured |
| `apache+auth+audit` | 3 | no | 0.1058 +/- 0.0079 (n=5) | 2.9774 +/- 0.1345 (n=5) | 0.4445 +/- 0.0663 (n=5) | not measured |
| `apache+auth+suricata` | 3 | no | 0.1838 +/- 0.0039 (n=5) | 3.2177 +/- 0.1183 (n=5) | 0.4914 +/- 0.0587 (n=5) | not measured |
| `auth+audit+suricata` | 3 | no | 0.1755 +/- 0.0100 (n=5) | 2.9664 +/- 0.1037 (n=5) | 0.5913 +/- 0.0797 (n=5) | not measured |

### chorus, epsilon 1.0, beta 5.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.2109 +/- 0.0250 (4x5) | 3.4185 +/- 0.1523 (4x5) | 0.6315 +/- 0.1856 (4x5) | not measured |
| 2 | no | 6 | 0.0591 +/- 0.0225 (6x5) | 3.3182 +/- 0.1080 (6x5) | 0.5218 +/- 0.2131 (6x5) | not measured |
| 3 | **yes** | 6 | 0.2427 +/- 0.0520 (6x5) | 3.3577 +/- 0.1515 (6x5) | 0.5246 +/- 0.1560 (6x5) | not measured |
| 3 | no | 4 | 0.0481 +/- 0.0136 (4x5) | 3.2011 +/- 0.1116 (4x5) | 0.3974 +/- 0.0745 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 2.0, beta 1.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0558 +/- 0.0037 (1x5) | 3.5655 +/- 0.0831 (1x5) | not measured | not measured |
| 1 | no | 4 | 0.1142 +/- 0.0695 (4x5) | 3.4289 +/- 0.2187 (4x5) | not measured | not measured |
| 2 | **yes** | 4 | 0.3610 +/- 0.1188 (4x5) | 3.3239 +/- 0.1952 (4x5) | 0.8525 +/- 0.0750 (4x5) | not measured |
| 2 | no | 6 | 0.2059 +/- 0.0880 (6x5) | 3.2170 +/- 0.2080 (6x5) | 0.6221 +/- 0.1957 (6x5) | not measured |
| 3 | **yes** | 6 | 0.6104 +/- 0.3314 (6x5) | 2.9376 +/- 0.2387 (6x5) | 0.8107 +/- 0.0808 (6x5) | not measured |
| 3 | no | 4 | 0.2093 +/- 0.0482 (4x5) | 2.7439 +/- 0.1904 (4x5) | 0.7508 +/- 0.0703 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### pgd, epsilon 1.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0538 +/- 0.0040 (1x5) | 3.4988 +/- 0.0798 (1x5) | not measured | not measured |
| 1 | no | 4 | 0.0870 +/- 0.0663 (4x5) | 3.4682 +/- 0.1897 (4x5) | not measured | not measured |
| 2 | **yes** | 4 | 0.1298 +/- 0.0603 (4x5) | 3.5255 +/- 0.1703 (4x5) | 0.5503 +/- 0.0995 (4x5) | not measured |
| 2 | no | 6 | 0.1600 +/- 0.0555 (6x5) | 3.4162 +/- 0.1837 (6x5) | 0.2461 +/- 0.1334 (6x5) | not measured |
| 3 | **yes** | 6 | 0.1978 +/- 0.0519 (6x5) | 3.4478 +/- 0.1918 (6x5) | 0.5148 +/- 0.0750 (6x5) | not measured |
| 3 | no | 4 | 0.2076 +/- 0.0268 (4x5) | 3.2398 +/- 0.1255 (4x5) | 0.2947 +/- 0.0679 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### pgd, epsilon 2.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.1569 +/- 0.0650 (4x5) | 3.5973 +/- 0.2040 (4x5) | 0.5013 +/- 0.0808 (4x5) | not measured |
| 2 | no | 6 | 0.1891 +/- 0.0432 (6x5) | 3.3164 +/- 0.2243 (6x5) | 0.3382 +/- 0.1665 (6x5) | not measured |
| 3 | **yes** | 6 | 0.2270 +/- 0.0410 (6x5) | 3.4279 +/- 0.2454 (6x5) | 0.5472 +/- 0.0871 (6x5) | not measured |
| 3 | no | 4 | 0.2323 +/- 0.0104 (4x5) | 2.9492 +/- 0.1594 (4x5) | 0.4360 +/- 0.0713 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### Paired prismflow - naive, per cell, within the dns split

Negative attack success would mean the discount defends. The seed count beside each mean is how many of the seeds favour PrismFlow.

| cell | k | dns compromised | attack success (pf - naive) |
|---|---|---|---|

## Test C: honest duplication

### Strict same-sensor pairs: NO EVIDENCE

sensor_groups() returns {'network': [4]} -- one member, suricata -- so no within-group PAIR exists and the strict same-sensor comparison has no value to report. There is no packet-capture view yet (docs/part27-ait-datacard.md section 9, limitation 9). Nothing was substituted under this name; see natural_dependence for a DIFFERENT and separately labelled quantity.

### Additionally, a different quantity: "different sensors, naturally dependent"

Pairs read from `results/ait_clean/results.json`, not re-derived. Most dependent: `dns-suricata`, `auth-audit`. Least dependent: `auth-dns`, `apache-dns`. Both systems agree on the ordering: **True**.

These views sit on DIFFERENT physical sensors. The dependence is a property of the scenario, not of shared hardware. This is NOT a same-sensor result and must not be read as one.

| system | AUC most-dependent over least-dependent (clean data) |
|---|---|
| naive | not measured |



Per-pair mean unexplained agreement on clean data, PrismFlow, seed-0 row shown with its committed Part 28 dependence beside it:

| pair | committed dependence (Part 28) | mean excess | measurable samples |
|---|---|---|---|

## Test D: extra missingness, on top of the natural rate

`auth` is already absent in 47.1-47.5% of windows and `apache` in 18.0-27.5% (`results/ait_clean/results.json`, `directions.<name>.missingness`), so these rates are ADDITIONAL, not total.

| system | condition | detector ROC AUC | detection @5% | detection @1% | balanced accuracy |
|---|---|---|---|---|---|
| naive | +30% masked | not measured | not measured | not measured | 0.8696 +/- 0.0137 (n=5) |
| naive | +50% masked | not measured | not measured | not measured | 0.7736 +/- 0.0074 (n=5) |

Clean reference, same seeds:

| system | detector ROC AUC | balanced accuracy | ENIV (eigen) |
|---|---|---|---|
| naive | not measured | 0.9444 +/- 0.0056 (n=5) | 3.4507 +/- 0.0790 (n=5) |

