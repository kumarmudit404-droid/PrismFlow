# Part 29 - attack and detection on real AIT data

Produced by `experiments/ait_attack/run_ait_attack.py` with `experiments/ait_attack/config.yaml`, commit `ead178c`.

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

### prismflow

| population | windows | ENIV (eigen) | ENIV (design effect) | mean dependence | detector mean score |
|---|---|---|---|---|---|
| `benign` | 441 | 3.1749 +/- 0.0749 (5 seeds), n=441 windows | 2.1660 +/- 0.0474 (5 seeds), n=441 windows | 0.3273 +/- 0.0124 (5 seeds), n=441 windows | 0.2594 +/- 0.0074 (5 seeds), n=441 windows |
| `dnsteal_only` | 663 | 3.2868 +/- 0.1345 (5 seeds), n=663 windows | 2.3168 +/- 0.1148 (5 seeds), n=663 windows | 0.2906 +/- 0.0274 (5 seeds), n=663 windows | 0.0378 +/- 0.0079 (5 seeds), n=663 windows |
| `multi_step` **DEGENERATE** | 3 | not measured (n=3 windows, degenerate) | not measured (n=3 windows, degenerate) | not measured (n=3 windows, degenerate) | 0.2222 +/- 0.0000 (5 seeds), n=3 windows |
| `exfiltration_service_only` **DEGENERATE** | 1 | not measured (n=1 windows, degenerate) | not measured (n=1 windows, degenerate) | not measured (n=1 windows, degenerate) | 0.0000 +/- 0.0000 (5 seeds), n=1 windows |

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

### chorus, epsilon 0.2, beta 1.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0330 +/- 0.0055 (1x5) | 3.5070 +/- 0.0965 (1x5) | not measured | 0.5921 +/- 0.0093 (1x5) |
| 1 | no | 4 | 0.0051 +/- 0.0040 (4x5) | 3.5088 +/- 0.1102 (4x5) | not measured | 0.4946 +/- 0.0766 (4x5) |
| 2 | **yes** | 4 | 0.0306 +/- 0.0054 (4x5) | 3.5027 +/- 0.1059 (4x5) | 0.4343 +/- 0.2750 (4x5) | 0.4251 +/- 0.0417 (4x5) |
| 2 | no | 6 | 0.0123 +/- 0.0106 (6x5) | 3.4967 +/- 0.1062 (6x5) | 0.2644 +/- 0.1804 (6x5) | 0.4946 +/- 0.0700 (6x5) |
| 3 | **yes** | 6 | 0.0312 +/- 0.0050 (6x5) | 3.5027 +/- 0.1077 (6x5) | 0.3814 +/- 0.1526 (6x5) | 0.4386 +/- 0.0544 (6x5) |
| 3 | no | 4 | 0.0120 +/- 0.0043 (4x5) | 3.4712 +/- 0.1098 (4x5) | 0.2484 +/- 0.0902 (4x5) | 0.4925 +/- 0.0680 (4x5) |

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

### chorus, epsilon 0.5, beta 1.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0504 +/- 0.0088 (1x5) | 3.5078 +/- 0.0979 (1x5) | not measured | 0.5922 +/- 0.0690 (1x5) |
| 1 | no | 4 | 0.0193 +/- 0.0164 (4x5) | 3.5034 +/- 0.1594 (4x5) | not measured | 0.4851 +/- 0.0825 (4x5) |
| 2 | **yes** | 4 | 0.1061 +/- 0.0244 (4x5) | 3.5005 +/- 0.1160 (4x5) | 0.4960 +/- 0.2476 (4x5) | 0.3774 +/- 0.0493 (4x5) |
| 2 | no | 6 | 0.0650 +/- 0.0738 (6x5) | 3.4013 +/- 0.1366 (6x5) | 0.3338 +/- 0.1875 (6x5) | 0.4857 +/- 0.0715 (6x5) |
| 3 | **yes** | 6 | 0.1137 +/- 0.0228 (6x5) | 3.4374 +/- 0.1198 (6x5) | 0.4667 +/- 0.1850 (6x5) | 0.4064 +/- 0.0697 (6x5) |
| 3 | no | 4 | 0.0595 +/- 0.0357 (4x5) | 3.3016 +/- 0.1419 (4x5) | 0.2916 +/- 0.0814 (4x5) | 0.4945 +/- 0.0709 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 1.0, beta 0.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.1298 +/- 0.0603 (4x5) | 3.5260 +/- 0.1700 (4x5) | 0.5500 +/- 0.0998 (4x5) | not measured |
| 2 | no | 6 | 0.1594 +/- 0.0553 (6x5) | 3.4185 +/- 0.1860 (6x5) | 0.2465 +/- 0.1340 (6x5) | not measured |
| 3 | **yes** | 6 | 0.1972 +/- 0.0518 (6x5) | 3.4462 +/- 0.1951 (6x5) | 0.5151 +/- 0.0757 (6x5) | not measured |
| 3 | no | 4 | 0.2076 +/- 0.0267 (4x5) | 3.2471 +/- 0.1351 (4x5) | 0.2938 +/- 0.0691 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 1.0, beta 0.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.0956 +/- 0.0432 (4x5) | 3.5509 +/- 0.1854 (4x5) | 0.5729 +/- 0.1002 (4x5) | 0.5412 +/- 0.0860 (4x5) |
| 2 | no | 6 | 0.1487 +/- 0.0685 (6x5) | 3.4245 +/- 0.2021 (6x5) | 0.2620 +/- 0.1653 (6x5) | 0.4785 +/- 0.0687 (6x5) |
| 3 | **yes** | 6 | 0.1804 +/- 0.0655 (6x5) | 3.4494 +/- 0.1995 (6x5) | 0.5282 +/- 0.0827 (6x5) | 0.5112 +/- 0.0830 (6x5) |
| 3 | no | 4 | 0.2095 +/- 0.0418 (4x5) | 3.2832 +/- 0.1761 (4x5) | 0.2764 +/- 0.0878 (4x5) | 0.4399 +/- 0.0777 (4x5) |

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

### chorus, epsilon 1.0, beta 1.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0518 +/- 0.0098 (1x5) | 3.5369 +/- 0.1000 (1x5) | not measured | 0.5879 +/- 0.0764 (1x5) |
| 1 | no | 4 | 0.0534 +/- 0.0477 (4x5) | 3.4906 +/- 0.1988 (4x5) | not measured | 0.5102 +/- 0.0907 (4x5) |
| 2 | **yes** | 4 | 0.2838 +/- 0.0997 (4x5) | 3.4168 +/- 0.1826 (4x5) | 0.5974 +/- 0.1841 (4x5) | 0.3588 +/- 0.0255 (4x5) |
| 2 | no | 6 | 0.1238 +/- 0.1034 (6x5) | 3.3469 +/- 0.1856 (6x5) | 0.4679 +/- 0.2482 (6x5) | 0.5268 +/- 0.1159 (6x5) |
| 3 | **yes** | 6 | 0.3557 +/- 0.1335 (6x5) | 3.3116 +/- 0.2140 (6x5) | 0.5542 +/- 0.1188 (6x5) | 0.4185 +/- 0.1003 (6x5) |
| 3 | no | 4 | 0.1292 +/- 0.0578 (4x5) | 3.1751 +/- 0.2220 (4x5) | 0.4653 +/- 0.1411 (4x5) | 0.5652 +/- 0.1097 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

#### Every subset at the primary cell - prismflow

| compromised views | k | dns | attack success (5 seeds) | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| `dns` | 1 | **yes** | 0.0518 +/- 0.0098 (n=5) | 3.5369 +/- 0.1000 (n=5) | not measured | 0.5879 +/- 0.0764 (n=5) |
| `apache` | 1 | no | 0.0744 +/- 0.0151 (n=5) | 3.2159 +/- 0.0854 (n=5) | not measured | 0.4500 +/- 0.0373 (n=5) |
| `audit` | 1 | no | 0.0087 +/- 0.0036 (n=5) | 3.4412 +/- 0.0672 (n=5) | not measured | 0.5096 +/- 0.1008 (n=5) |
| `auth` | 1 | no | 0.0155 +/- 0.0084 (n=5) | 3.6587 +/- 0.0744 (n=5) | not measured | 0.4978 +/- 0.0967 (n=5) |
| `suricata` | 1 | no | 0.1150 +/- 0.0300 (n=5) | 3.6464 +/- 0.0855 (n=5) | not measured | 0.5836 +/- 0.0821 (n=5) |
| `apache+dns` | 2 | **yes** | 0.2399 +/- 0.0149 (n=5) | 3.1522 +/- 0.1155 (n=5) | 0.6229 +/- 0.1031 (n=5) | 0.3619 +/- 0.0290 (n=5) |
| `auth+dns` | 2 | **yes** | 0.2318 +/- 0.0289 (n=5) | 3.5707 +/- 0.0576 (n=5) | 0.4006 +/- 0.0905 (n=5) | 0.3548 +/- 0.0163 (n=5) |
| `dns+audit` | 2 | **yes** | 0.2276 +/- 0.0079 (n=5) | 3.4770 +/- 0.0686 (n=5) | 0.5172 +/- 0.0767 (n=5) | 0.3632 +/- 0.0107 (n=5) |
| `dns+suricata` | 2 | **yes** | 0.4359 +/- 0.0863 (n=5) | 3.4674 +/- 0.1107 (n=5) | 0.8489 +/- 0.0083 (n=5) | 0.3554 +/- 0.0425 (n=5) |
| `apache+audit` | 2 | no | 0.0807 +/- 0.0158 (n=5) | 3.3448 +/- 0.0991 (n=5) | 0.3025 +/- 0.1382 (n=5) | 0.4377 +/- 0.0819 (n=5) |
| `apache+auth` | 2 | no | 0.0773 +/- 0.0131 (n=5) | 3.5098 +/- 0.0847 (n=5) | 0.1802 +/- 0.1631 (n=5) | 0.4166 +/- 0.0529 (n=5) |
| `apache+suricata` | 2 | no | 0.3321 +/- 0.0032 (n=5) | 3.2794 +/- 0.1309 (n=5) | 0.3986 +/- 0.0783 (n=5) | 0.5404 +/- 0.0038 (n=5) |
| `audit+suricata` | 2 | no | 0.0556 +/- 0.0156 (n=5) | 3.1845 +/- 0.0668 (n=5) | 0.8753 +/- 0.0353 (n=5) | 0.6601 +/- 0.0395 (n=5) |
| `auth+audit` | 2 | no | 0.0421 +/- 0.0117 (n=5) | 3.1609 +/- 0.1013 (n=5) | 0.5721 +/- 0.1151 (n=5) | 0.4653 +/- 0.0855 (n=5) |
| `auth+suricata` | 2 | no | 0.1549 +/- 0.0460 (n=5) | 3.6019 +/- 0.0667 (n=5) | 0.4791 +/- 0.1237 (n=5) | 0.6409 +/- 0.1021 (n=5) |
| `apache+auth+dns` | 3 | **yes** | 0.2309 +/- 0.0082 (n=5) | 3.3330 +/- 0.1573 (n=5) | 0.4288 +/- 0.1136 (n=5) | 0.3263 +/- 0.0552 (n=5) |
| `apache+dns+audit` | 3 | **yes** | 0.2273 +/- 0.0039 (n=5) | 3.2735 +/- 0.1730 (n=5) | 0.4695 +/- 0.1428 (n=5) | 0.3416 +/- 0.0411 (n=5) |
| `apache+dns+suricata` | 3 | **yes** | 0.4901 +/- 0.0465 (n=5) | 3.1311 +/- 0.1582 (n=5) | 0.6053 +/- 0.0831 (n=5) | 0.4700 +/- 0.0165 (n=5) |
| `auth+dns+audit` | 3 | **yes** | 0.2222 +/- 0.0081 (n=5) | 3.0878 +/- 0.0846 (n=5) | 0.5612 +/- 0.0522 (n=5) | 0.3139 +/- 0.0204 (n=5) |
| `auth+dns+suricata` | 3 | **yes** | 0.4843 +/- 0.0260 (n=5) | 3.6043 +/- 0.0635 (n=5) | 0.5701 +/- 0.0402 (n=5) | 0.5259 +/- 0.0294 (n=5) |
| `dns+audit+suricata` | 3 | **yes** | 0.4798 +/- 0.0361 (n=5) | 3.4397 +/- 0.0887 (n=5) | 0.6902 +/- 0.0488 (n=5) | 0.5331 +/- 0.0272 (n=5) |
| `apache+audit+suricata` | 3 | no | 0.1787 +/- 0.0486 (n=5) | 3.2431 +/- 0.1335 (n=5) | 0.4963 +/- 0.1102 (n=5) | 0.6410 +/- 0.0735 (n=5) |
| `apache+auth+audit` | 3 | no | 0.0773 +/- 0.0129 (n=5) | 3.1014 +/- 0.0832 (n=5) | 0.3107 +/- 0.0576 (n=5) | 0.4179 +/- 0.0756 (n=5) |
| `apache+auth+suricata` | 3 | no | 0.1756 +/- 0.0248 (n=5) | 3.4390 +/- 0.1164 (n=5) | 0.4340 +/- 0.1098 (n=5) | 0.5750 +/- 0.0437 (n=5) |
| `auth+audit+suricata` | 3 | no | 0.0854 +/- 0.0342 (n=5) | 2.9169 +/- 0.1131 (n=5) | 0.6202 +/- 0.0694 (n=5) | 0.6270 +/- 0.0701 (n=5) |

### chorus, epsilon 1.0, beta 5.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.2109 +/- 0.0250 (4x5) | 3.4185 +/- 0.1523 (4x5) | 0.6315 +/- 0.1856 (4x5) | not measured |
| 2 | no | 6 | 0.0591 +/- 0.0225 (6x5) | 3.3182 +/- 0.1080 (6x5) | 0.5218 +/- 0.2131 (6x5) | not measured |
| 3 | **yes** | 6 | 0.2427 +/- 0.0520 (6x5) | 3.3577 +/- 0.1515 (6x5) | 0.5246 +/- 0.1560 (6x5) | not measured |
| 3 | no | 4 | 0.0481 +/- 0.0136 (4x5) | 3.2011 +/- 0.1116 (4x5) | 0.3974 +/- 0.0745 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 1.0, beta 5.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.2269 +/- 0.0389 (4x5) | 3.4412 +/- 0.1894 (4x5) | 0.6054 +/- 0.1967 (4x5) | 0.2960 +/- 0.0378 (4x5) |
| 2 | no | 6 | 0.0497 +/- 0.0359 (6x5) | 3.2810 +/- 0.1572 (6x5) | 0.5306 +/- 0.2711 (6x5) | 0.5462 +/- 0.1034 (6x5) |
| 3 | **yes** | 6 | 0.2856 +/- 0.0869 (6x5) | 3.4049 +/- 0.1799 (6x5) | 0.5113 +/- 0.1646 (6x5) | 0.3333 +/- 0.0972 (6x5) |
| 3 | no | 4 | 0.0431 +/- 0.0194 (4x5) | 3.1636 +/- 0.1516 (4x5) | 0.4044 +/- 0.1009 (4x5) | 0.5766 +/- 0.0995 (4x5) |

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

### chorus, epsilon 2.0, beta 1.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0529 +/- 0.0091 (1x5) | 3.5999 +/- 0.1048 (1x5) | not measured | 0.5857 +/- 0.0730 (1x5) |
| 1 | no | 4 | 0.0992 +/- 0.0832 (4x5) | 3.4657 +/- 0.2310 (4x5) | not measured | 0.4896 +/- 0.0693 (4x5) |
| 2 | **yes** | 4 | 0.3956 +/- 0.1542 (4x5) | 3.4017 +/- 0.1681 (4x5) | 0.8365 +/- 0.0901 (4x5) | 0.3748 +/- 0.0472 (4x5) |
| 2 | no | 6 | 0.1776 +/- 0.0904 (6x5) | 3.2529 +/- 0.2155 (6x5) | 0.6068 +/- 0.2411 (6x5) | 0.5540 +/- 0.1095 (6x5) |
| 3 | **yes** | 6 | 0.6293 +/- 0.3218 (6x5) | 3.0941 +/- 0.2255 (6x5) | 0.7539 +/- 0.1271 (6x5) | 0.3389 +/- 0.0557 (6x5) |
| 3 | no | 4 | 0.1904 +/- 0.0583 (4x5) | 2.9769 +/- 0.2378 (4x5) | 0.6647 +/- 0.1232 (4x5) | 0.5501 +/- 0.1119 (4x5) |

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

### pgd, epsilon 1.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0518 +/- 0.0098 (1x5) | 3.5369 +/- 0.1000 (1x5) | not measured | 0.5879 +/- 0.0764 (1x5) |
| 1 | no | 4 | 0.0534 +/- 0.0477 (4x5) | 3.4906 +/- 0.1988 (4x5) | not measured | 0.5102 +/- 0.0907 (4x5) |
| 2 | **yes** | 4 | 0.0957 +/- 0.0433 (4x5) | 3.5511 +/- 0.1864 (4x5) | 0.5724 +/- 0.1004 (4x5) | 0.5480 +/- 0.0843 (4x5) |
| 2 | no | 6 | 0.1487 +/- 0.0697 (6x5) | 3.4190 +/- 0.2023 (6x5) | 0.2665 +/- 0.1677 (6x5) | 0.4809 +/- 0.0643 (6x5) |
| 3 | **yes** | 6 | 0.1802 +/- 0.0663 (6x5) | 3.4462 +/- 0.2017 (6x5) | 0.5309 +/- 0.0833 (6x5) | 0.5118 +/- 0.0846 (6x5) |
| 3 | no | 4 | 0.2093 +/- 0.0427 (4x5) | 3.2683 +/- 0.1779 (4x5) | 0.2852 +/- 0.0875 (4x5) | 0.4426 +/- 0.0727 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### pgd, epsilon 2.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.1569 +/- 0.0650 (4x5) | 3.5973 +/- 0.2040 (4x5) | 0.5013 +/- 0.0808 (4x5) | not measured |
| 2 | no | 6 | 0.1891 +/- 0.0432 (6x5) | 3.3164 +/- 0.2243 (6x5) | 0.3382 +/- 0.1665 (6x5) | not measured |
| 3 | **yes** | 6 | 0.2270 +/- 0.0410 (6x5) | 3.4279 +/- 0.2454 (6x5) | 0.5472 +/- 0.0871 (6x5) | not measured |
| 3 | no | 4 | 0.2323 +/- 0.0104 (4x5) | 2.9492 +/- 0.1594 (4x5) | 0.4360 +/- 0.0713 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### pgd, epsilon 2.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.1405 +/- 0.0783 (4x5) | 3.6203 +/- 0.2086 (4x5) | 0.5227 +/- 0.0771 (4x5) | 0.5294 +/- 0.0757 (4x5) |
| 2 | no | 6 | 0.1844 +/- 0.0611 (6x5) | 3.3497 +/- 0.2366 (6x5) | 0.3244 +/- 0.1940 (6x5) | 0.4632 +/- 0.0655 (6x5) |
| 3 | **yes** | 6 | 0.2176 +/- 0.0589 (6x5) | 3.4607 +/- 0.2376 (6x5) | 0.5417 +/- 0.0868 (6x5) | 0.4740 +/- 0.0625 (6x5) |
| 3 | no | 4 | 0.2314 +/- 0.0181 (4x5) | 3.0190 +/- 0.1770 (4x5) | 0.4057 +/- 0.0878 (4x5) | 0.4140 +/- 0.0686 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### Paired prismflow - naive, per cell, within the dns split

Negative attack success would mean the discount defends. The seed count beside each mean is how many of the seeds favour PrismFlow.

| cell | k | dns compromised | attack success (pf - naive) |
|---|---|---|---|
| `chorus_k1_eps0.2_beta1.0_[apache]` | 1 | no | -0.0085 +/- 0.0040 (5/5) |
| `chorus_k1_eps0.2_beta1.0_[audit]` | 1 | no | +0.0013 +/- 0.0012 (0/5) |
| `chorus_k1_eps0.2_beta1.0_[auth]` | 1 | no | +0.0004 +/- 0.0010 (1/5) |
| `chorus_k1_eps0.2_beta1.0_[dns]` | 1 | yes | -0.0023 +/- 0.0057 (4/5) |
| `chorus_k1_eps0.2_beta1.0_[suricata]` | 1 | no | -0.0099 +/- 0.0026 (5/5) |
| `chorus_k1_eps0.5_beta1.0_[apache]` | 1 | no | -0.0332 +/- 0.0048 (5/5) |
| `chorus_k1_eps0.5_beta1.0_[audit]` | 1 | no | -0.0034 +/- 0.0031 (4/5) |
| `chorus_k1_eps0.5_beta1.0_[auth]` | 1 | no | -0.0067 +/- 0.0032 (5/5) |
| `chorus_k1_eps0.5_beta1.0_[dns]` | 1 | yes | -0.0020 +/- 0.0068 (3/5) |
| `chorus_k1_eps0.5_beta1.0_[suricata]` | 1 | no | -0.0415 +/- 0.0127 (5/5) |
| `chorus_k1_eps1.0_beta1.0_[apache]` | 1 | no | -0.0294 +/- 0.0079 (5/5) |
| `chorus_k1_eps1.0_beta1.0_[audit]` | 1 | no | -0.0171 +/- 0.0068 (5/5) |
| `chorus_k1_eps1.0_beta1.0_[auth]` | 1 | no | -0.0190 +/- 0.0072 (5/5) |
| `chorus_k1_eps1.0_beta1.0_[dns]` | 1 | yes | -0.0020 +/- 0.0079 (3/5) |
| `chorus_k1_eps1.0_beta1.0_[suricata]` | 1 | no | -0.0688 +/- 0.0179 (5/5) |
| `chorus_k1_eps2.0_beta1.0_[apache]` | 1 | no | -0.0108 +/- 0.0074 (4/5) |
| `chorus_k1_eps2.0_beta1.0_[audit]` | 1 | no | -0.0300 +/- 0.0120 (5/5) |
| `chorus_k1_eps2.0_beta1.0_[auth]` | 1 | no | -0.0236 +/- 0.0073 (5/5) |
| `chorus_k1_eps2.0_beta1.0_[dns]` | 1 | yes | -0.0029 +/- 0.0079 (3/5) |
| `chorus_k1_eps2.0_beta1.0_[suricata]` | 1 | no | +0.0045 +/- 0.0116 (2/5) |
| `chorus_k2_eps0.2_beta1.0_[apache+audit]` | 2 | no | -0.0054 +/- 0.0041 (5/5) |
| `chorus_k2_eps0.2_beta1.0_[apache+auth]` | 2 | no | -0.0049 +/- 0.0024 (5/5) |
| `chorus_k2_eps0.2_beta1.0_[apache+dns]` | 2 | yes | -0.0014 +/- 0.0033 (3/5) |
| `chorus_k2_eps0.2_beta1.0_[apache+suricata]` | 2 | no | -0.0171 +/- 0.0099 (5/5) |
| `chorus_k2_eps0.2_beta1.0_[audit+suricata]` | 2 | no | -0.0023 +/- 0.0008 (5/5) |
| `chorus_k2_eps0.2_beta1.0_[auth+audit]` | 2 | no | -0.0002 +/- 0.0025 (2/5) |
| `chorus_k2_eps0.2_beta1.0_[auth+dns]` | 2 | yes | -0.0004 +/- 0.0019 (2/5) |
| `chorus_k2_eps0.2_beta1.0_[auth+suricata]` | 2 | no | -0.0009 +/- 0.0018 (3/5) |
| `chorus_k2_eps0.2_beta1.0_[dns+audit]` | 2 | yes | +0.0005 +/- 0.0022 (2/5) |
| `chorus_k2_eps0.2_beta1.0_[dns+suricata]` | 2 | yes | +0.0014 +/- 0.0012 (0/5) |
| `chorus_k2_eps0.5_beta1.0_[apache+audit]` | 2 | no | -0.0314 +/- 0.0073 (5/5) |
| `chorus_k2_eps0.5_beta1.0_[apache+auth]` | 2 | no | -0.0116 +/- 0.0100 (4/5) |
| `chorus_k2_eps0.5_beta1.0_[apache+dns]` | 2 | yes | -0.0020 +/- 0.0037 (3/5) |
| `chorus_k2_eps0.5_beta1.0_[apache+suricata]` | 2 | no | -0.0504 +/- 0.0176 (5/5) |
| `chorus_k2_eps0.5_beta1.0_[audit+suricata]` | 2 | no | -0.0181 +/- 0.0075 (5/5) |
| `chorus_k2_eps0.5_beta1.0_[auth+audit]` | 2 | no | -0.0143 +/- 0.0071 (5/5) |
| `chorus_k2_eps0.5_beta1.0_[auth+dns]` | 2 | yes | +0.0007 +/- 0.0059 (1/5) |
| `chorus_k2_eps0.5_beta1.0_[auth+suricata]` | 2 | no | -0.0399 +/- 0.0190 (5/5) |
| `chorus_k2_eps0.5_beta1.0_[dns+audit]` | 2 | yes | -0.0016 +/- 0.0055 (4/5) |
| `chorus_k2_eps0.5_beta1.0_[dns+suricata]` | 2 | yes | +0.0022 +/- 0.0100 (2/5) |
| `chorus_k2_eps1.0_beta0.0_[apache+audit]` | 2 | no | -0.0227 +/- 0.0072 (5/5) |
| `chorus_k2_eps1.0_beta0.0_[apache+auth]` | 2 | no | -0.0195 +/- 0.0112 (5/5) |
| `chorus_k2_eps1.0_beta0.0_[apache+dns]` | 2 | yes | -0.0294 +/- 0.0090 (5/5) |
| `chorus_k2_eps1.0_beta0.0_[apache+suricata]` | 2 | no | +0.0096 +/- 0.0054 (0/5) |
| `chorus_k2_eps1.0_beta0.0_[audit+suricata]` | 2 | no | -0.0031 +/- 0.0088 (4/5) |
| `chorus_k2_eps1.0_beta0.0_[auth+audit]` | 2 | no | -0.0265 +/- 0.0107 (5/5) |
| `chorus_k2_eps1.0_beta0.0_[auth+dns]` | 2 | yes | -0.0222 +/- 0.0073 (5/5) |
| `chorus_k2_eps1.0_beta0.0_[auth+suricata]` | 2 | no | -0.0018 +/- 0.0140 (3/5) |
| `chorus_k2_eps1.0_beta0.0_[dns+audit]` | 2 | yes | -0.0173 +/- 0.0023 (5/5) |
| `chorus_k2_eps1.0_beta0.0_[dns+suricata]` | 2 | yes | -0.0677 +/- 0.0178 (5/5) |
| `chorus_k2_eps1.0_beta1.0_[apache+audit]` | 2 | no | -0.0404 +/- 0.0059 (5/5) |
| `chorus_k2_eps1.0_beta1.0_[apache+auth]` | 2 | no | -0.0144 +/- 0.0099 (5/5) |
| `chorus_k2_eps1.0_beta1.0_[apache+dns]` | 2 | yes | +0.0126 +/- 0.0069 (0/5) |
| `chorus_k2_eps1.0_beta1.0_[apache+suricata]` | 2 | no | +0.0047 +/- 0.0036 (0/5) |
| `chorus_k2_eps1.0_beta1.0_[audit+suricata]` | 2 | no | -0.1478 +/- 0.0566 (5/5) |
| `chorus_k2_eps1.0_beta1.0_[auth+audit]` | 2 | no | -0.0245 +/- 0.0093 (5/5) |
| `chorus_k2_eps1.0_beta1.0_[auth+dns]` | 2 | yes | +0.0240 +/- 0.0257 (0/5) |
| `chorus_k2_eps1.0_beta1.0_[auth+suricata]` | 2 | no | -0.0527 +/- 0.0452 (5/5) |
| `chorus_k2_eps1.0_beta1.0_[dns+audit]` | 2 | yes | +0.0130 +/- 0.0079 (0/5) |
| `chorus_k2_eps1.0_beta1.0_[dns+suricata]` | 2 | yes | +0.0619 +/- 0.0517 (1/5) |
| `chorus_k2_eps1.0_beta5.0_[apache+audit]` | 2 | no | -0.0419 +/- 0.0102 (5/5) |
| `chorus_k2_eps1.0_beta5.0_[apache+auth]` | 2 | no | -0.0206 +/- 0.0098 (5/5) |
| `chorus_k2_eps1.0_beta5.0_[apache+dns]` | 2 | yes | -0.0005 +/- 0.0051 (3/5) |
| `chorus_k2_eps1.0_beta5.0_[apache+suricata]` | 2 | no | +0.0370 +/- 0.0158 (0/5) |
| `chorus_k2_eps1.0_beta5.0_[audit+suricata]` | 2 | no | -0.0013 +/- 0.0046 (4/5) |
| `chorus_k2_eps1.0_beta5.0_[auth+audit]` | 2 | no | -0.0314 +/- 0.0098 (5/5) |
| `chorus_k2_eps1.0_beta5.0_[auth+dns]` | 2 | yes | +0.0141 +/- 0.0093 (0/5) |
| `chorus_k2_eps1.0_beta5.0_[auth+suricata]` | 2 | no | +0.0018 +/- 0.0036 (1/5) |
| `chorus_k2_eps1.0_beta5.0_[dns+audit]` | 2 | yes | +0.0081 +/- 0.0029 (0/5) |
| `chorus_k2_eps1.0_beta5.0_[dns+suricata]` | 2 | yes | +0.0426 +/- 0.0187 (0/5) |
| `chorus_k2_eps2.0_beta1.0_[apache+audit]` | 2 | no | -0.0356 +/- 0.0100 (5/5) |
| `chorus_k2_eps2.0_beta1.0_[apache+auth]` | 2 | no | -0.0072 +/- 0.0124 (3/5) |
| `chorus_k2_eps2.0_beta1.0_[apache+dns]` | 2 | yes | +0.0188 +/- 0.0197 (1/5) |
| `chorus_k2_eps2.0_beta1.0_[apache+suricata]` | 2 | no | +0.0056 +/- 0.0087 (2/5) |
| `chorus_k2_eps2.0_beta1.0_[audit+suricata]` | 2 | no | -0.0993 +/- 0.0639 (5/5) |
| `chorus_k2_eps2.0_beta1.0_[auth+audit]` | 2 | no | -0.0193 +/- 0.0077 (5/5) |
| `chorus_k2_eps2.0_beta1.0_[auth+dns]` | 2 | yes | +0.0245 +/- 0.0087 (0/5) |
| `chorus_k2_eps2.0_beta1.0_[auth+suricata]` | 2 | no | -0.0139 +/- 0.0307 (3/5) |
| `chorus_k2_eps2.0_beta1.0_[dns+audit]` | 2 | yes | +0.0101 +/- 0.0085 (0/5) |
| `chorus_k2_eps2.0_beta1.0_[dns+suricata]` | 2 | yes | +0.0850 +/- 0.0685 (0/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+audit+suricata]` | 3 | no | -0.0004 +/- 0.0032 (2/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+auth+audit]` | 3 | no | -0.0056 +/- 0.0028 (5/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+auth+dns]` | 3 | yes | -0.0025 +/- 0.0033 (4/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+auth+suricata]` | 3 | no | -0.0009 +/- 0.0026 (3/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+dns+audit]` | 3 | yes | -0.0027 +/- 0.0020 (4/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+dns+suricata]` | 3 | yes | -0.0002 +/- 0.0033 (3/5) |
| `chorus_k3_eps0.2_beta1.0_[auth+audit+suricata]` | 3 | no | -0.0013 +/- 0.0018 (3/5) |
| `chorus_k3_eps0.2_beta1.0_[auth+dns+audit]` | 3 | yes | +0.0005 +/- 0.0018 (1/5) |
| `chorus_k3_eps0.2_beta1.0_[auth+dns+suricata]` | 3 | yes | +0.0007 +/- 0.0016 (2/5) |
| `chorus_k3_eps0.2_beta1.0_[dns+audit+suricata]` | 3 | yes | +0.0011 +/- 0.0030 (2/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+audit+suricata]` | 3 | no | -0.0653 +/- 0.0378 (5/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+auth+audit]` | 3 | no | -0.0217 +/- 0.0097 (5/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+auth+dns]` | 3 | yes | -0.0067 +/- 0.0032 (5/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+auth+suricata]` | 3 | no | -0.0262 +/- 0.0483 (4/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+dns+audit]` | 3 | yes | -0.0088 +/- 0.0039 (5/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+dns+suricata]` | 3 | yes | +0.0072 +/- 0.0079 (0/5) |
| `chorus_k3_eps0.5_beta1.0_[auth+audit+suricata]` | 3 | no | -0.0209 +/- 0.0091 (5/5) |
| `chorus_k3_eps0.5_beta1.0_[auth+dns+audit]` | 3 | yes | -0.0013 +/- 0.0032 (3/5) |
| `chorus_k3_eps0.5_beta1.0_[auth+dns+suricata]` | 3 | yes | +0.0060 +/- 0.0052 (1/5) |
| `chorus_k3_eps0.5_beta1.0_[dns+audit+suricata]` | 3 | yes | +0.0049 +/- 0.0082 (1/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+audit+suricata]` | 3 | no | +0.0112 +/- 0.0063 (0/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+auth+audit]` | 3 | no | -0.0220 +/- 0.0134 (5/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+auth+dns]` | 3 | yes | -0.0283 +/- 0.0093 (5/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+auth+suricata]` | 3 | no | +0.0119 +/- 0.0057 (0/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+dns+audit]` | 3 | yes | -0.0296 +/- 0.0085 (5/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+dns+suricata]` | 3 | yes | +0.0079 +/- 0.0053 (0/5) |
| `chorus_k3_eps1.0_beta0.0_[auth+audit+suricata]` | 3 | no | +0.0067 +/- 0.0059 (0/5) |
| `chorus_k3_eps1.0_beta0.0_[auth+dns+audit]` | 3 | yes | -0.0310 +/- 0.0087 (5/5) |
| `chorus_k3_eps1.0_beta0.0_[auth+dns+suricata]` | 3 | yes | -0.0097 +/- 0.0128 (4/5) |
| `chorus_k3_eps1.0_beta0.0_[dns+audit+suricata]` | 3 | yes | -0.0103 +/- 0.0110 (4/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+audit+suricata]` | 3 | no | -0.0819 +/- 0.0389 (5/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+auth+audit]` | 3 | no | -0.0285 +/- 0.0092 (5/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+auth+dns]` | 3 | yes | +0.0220 +/- 0.0047 (0/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+auth+suricata]` | 3 | no | -0.0081 +/- 0.0223 (2/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+dns+audit]` | 3 | yes | +0.0099 +/- 0.0061 (0/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+dns+suricata]` | 3 | yes | +0.0341 +/- 0.0367 (1/5) |
| `chorus_k3_eps1.0_beta1.0_[auth+audit+suricata]` | 3 | no | -0.0901 +/- 0.0324 (5/5) |
| `chorus_k3_eps1.0_beta1.0_[auth+dns+audit]` | 3 | yes | +0.0083 +/- 0.0030 (0/5) |
| `chorus_k3_eps1.0_beta1.0_[auth+dns+suricata]` | 3 | yes | +0.0403 +/- 0.0245 (0/5) |
| `chorus_k3_eps1.0_beta1.0_[dns+audit+suricata]` | 3 | yes | +0.0372 +/- 0.0324 (0/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+audit+suricata]` | 3 | no | -0.0079 +/- 0.0027 (5/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+auth+audit]` | 3 | no | -0.0188 +/- 0.0113 (5/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+auth+dns]` | 3 | yes | +0.0090 +/- 0.0050 (0/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+auth+suricata]` | 3 | no | +0.0060 +/- 0.0260 (2/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+dns+audit]` | 3 | yes | +0.0031 +/- 0.0061 (2/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+dns+suricata]` | 3 | yes | +0.0854 +/- 0.0450 (0/5) |
| `chorus_k3_eps1.0_beta5.0_[auth+audit+suricata]` | 3 | no | +0.0005 +/- 0.0059 (1/5) |
| `chorus_k3_eps1.0_beta5.0_[auth+dns+audit]` | 3 | yes | +0.0139 +/- 0.0073 (0/5) |
| `chorus_k3_eps1.0_beta5.0_[auth+dns+suricata]` | 3 | yes | +0.0574 +/- 0.0248 (0/5) |
| `chorus_k3_eps1.0_beta5.0_[dns+audit+suricata]` | 3 | yes | +0.0884 +/- 0.0135 (0/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+audit+suricata]` | 3 | no | -0.0143 +/- 0.0258 (3/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+auth+audit]` | 3 | no | -0.0401 +/- 0.0151 (5/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+auth+dns]` | 3 | yes | +0.0377 +/- 0.0039 (0/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+auth+suricata]` | 3 | no | +0.0025 +/- 0.0037 (0/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+dns+audit]` | 3 | yes | +0.0289 +/- 0.0065 (0/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+dns+suricata]` | 3 | yes | -0.0083 +/- 0.0300 (2/5) |
| `chorus_k3_eps2.0_beta1.0_[auth+audit+suricata]` | 3 | no | -0.0240 +/- 0.0280 (4/5) |
| `chorus_k3_eps2.0_beta1.0_[auth+dns+audit]` | 3 | yes | +0.0188 +/- 0.0057 (0/5) |
| `chorus_k3_eps2.0_beta1.0_[auth+dns+suricata]` | 3 | yes | +0.0168 +/- 0.0115 (1/5) |
| `chorus_k3_eps2.0_beta1.0_[dns+audit+suricata]` | 3 | yes | +0.0193 +/- 0.0110 (0/5) |
| `pgd_k1_eps1.0_[apache]` | 1 | no | -0.0294 +/- 0.0079 (5/5) |
| `pgd_k1_eps1.0_[audit]` | 1 | no | -0.0171 +/- 0.0068 (5/5) |
| `pgd_k1_eps1.0_[auth]` | 1 | no | -0.0190 +/- 0.0072 (5/5) |
| `pgd_k1_eps1.0_[dns]` | 1 | yes | -0.0020 +/- 0.0079 (3/5) |
| `pgd_k1_eps1.0_[suricata]` | 1 | no | -0.0688 +/- 0.0179 (5/5) |
| `pgd_k2_eps1.0_[apache+audit]` | 2 | no | -0.0233 +/- 0.0073 (5/5) |
| `pgd_k2_eps1.0_[apache+auth]` | 2 | no | -0.0211 +/- 0.0118 (5/5) |
| `pgd_k2_eps1.0_[apache+dns]` | 2 | yes | -0.0296 +/- 0.0087 (5/5) |
| `pgd_k2_eps1.0_[apache+suricata]` | 2 | no | +0.0096 +/- 0.0058 (0/5) |
| `pgd_k2_eps1.0_[audit+suricata]` | 2 | no | -0.0011 +/- 0.0094 (4/5) |
| `pgd_k2_eps1.0_[auth+audit]` | 2 | no | -0.0285 +/- 0.0114 (5/5) |
| `pgd_k2_eps1.0_[auth+dns]` | 2 | yes | -0.0220 +/- 0.0075 (5/5) |
| `pgd_k2_eps1.0_[auth+suricata]` | 2 | no | -0.0032 +/- 0.0142 (3/5) |
| `pgd_k2_eps1.0_[dns+audit]` | 2 | yes | -0.0173 +/- 0.0023 (5/5) |
| `pgd_k2_eps1.0_[dns+suricata]` | 2 | yes | -0.0675 +/- 0.0182 (5/5) |
| `pgd_k2_eps2.0_[apache+audit]` | 2 | no | -0.0240 +/- 0.0104 (5/5) |
| `pgd_k2_eps2.0_[apache+auth]` | 2 | no | -0.0088 +/- 0.0145 (4/5) |
| `pgd_k2_eps2.0_[apache+dns]` | 2 | yes | -0.0119 +/- 0.0094 (4/5) |
| `pgd_k2_eps2.0_[apache+suricata]` | 2 | no | +0.0130 +/- 0.0057 (0/5) |
| `pgd_k2_eps2.0_[audit+suricata]` | 2 | no | +0.0135 +/- 0.0047 (0/5) |
| `pgd_k2_eps2.0_[auth+audit]` | 2 | no | -0.0325 +/- 0.0116 (5/5) |
| `pgd_k2_eps2.0_[auth+dns]` | 2 | yes | -0.0260 +/- 0.0055 (5/5) |
| `pgd_k2_eps2.0_[auth+suricata]` | 2 | no | +0.0103 +/- 0.0065 (0/5) |
| `pgd_k2_eps2.0_[dns+audit]` | 2 | yes | -0.0300 +/- 0.0112 (5/5) |
| `pgd_k2_eps2.0_[dns+suricata]` | 2 | yes | +0.0020 +/- 0.0120 (2/5) |
| `pgd_k3_eps1.0_[apache+audit+suricata]` | 3 | no | +0.0114 +/- 0.0059 (0/5) |
| `pgd_k3_eps1.0_[apache+auth+audit]` | 3 | no | -0.0236 +/- 0.0134 (5/5) |
| `pgd_k3_eps1.0_[apache+auth+dns]` | 3 | yes | -0.0296 +/- 0.0096 (5/5) |
| `pgd_k3_eps1.0_[apache+auth+suricata]` | 3 | no | +0.0117 +/- 0.0063 (0/5) |
| `pgd_k3_eps1.0_[apache+dns+audit]` | 3 | yes | -0.0300 +/- 0.0084 (5/5) |
| `pgd_k3_eps1.0_[apache+dns+suricata]` | 3 | yes | +0.0074 +/- 0.0055 (0/5) |
| `pgd_k3_eps1.0_[auth+audit+suricata]` | 3 | no | +0.0072 +/- 0.0063 (0/5) |
| `pgd_k3_eps1.0_[auth+dns+audit]` | 3 | yes | -0.0329 +/- 0.0107 (5/5) |
| `pgd_k3_eps1.0_[auth+dns+suricata]` | 3 | yes | -0.0114 +/- 0.0147 (3/5) |
| `pgd_k3_eps1.0_[dns+audit+suricata]` | 3 | yes | -0.0090 +/- 0.0100 (4/5) |
| `pgd_k3_eps2.0_[apache+audit+suricata]` | 3 | no | +0.0116 +/- 0.0055 (0/5) |
| `pgd_k3_eps2.0_[apache+auth+audit]` | 3 | no | -0.0287 +/- 0.0206 (5/5) |
| `pgd_k3_eps2.0_[apache+auth+dns]` | 3 | yes | -0.0173 +/- 0.0130 (5/5) |
| `pgd_k3_eps2.0_[apache+auth+suricata]` | 3 | no | +0.0032 +/- 0.0093 (2/5) |
| `pgd_k3_eps2.0_[apache+dns+audit]` | 3 | yes | -0.0319 +/- 0.0104 (5/5) |
| `pgd_k3_eps2.0_[apache+dns+suricata]` | 3 | yes | +0.0101 +/- 0.0060 (0/5) |
| `pgd_k3_eps2.0_[auth+audit+suricata]` | 3 | no | +0.0101 +/- 0.0067 (0/5) |
| `pgd_k3_eps2.0_[auth+dns+audit]` | 3 | yes | -0.0339 +/- 0.0059 (5/5) |
| `pgd_k3_eps2.0_[auth+dns+suricata]` | 3 | yes | +0.0069 +/- 0.0083 (1/5) |
| `pgd_k3_eps2.0_[dns+audit+suricata]` | 3 | yes | +0.0094 +/- 0.0067 (0/5) |

## Test C: honest duplication

### Strict same-sensor pairs: NO EVIDENCE

sensor_groups() returns {'network': [4]} -- one member, suricata -- so no within-group PAIR exists and the strict same-sensor comparison has no value to report. There is no packet-capture view yet (docs/part27-ait-datacard.md section 9, limitation 9). Nothing was substituted under this name; see natural_dependence for a DIFFERENT and separately labelled quantity.

### Additionally, a different quantity: "different sensors, naturally dependent"

Pairs read from `results/ait_clean/results.json`, not re-derived. Most dependent: `dns-suricata`, `auth-audit`. Least dependent: `auth-dns`, `apache-dns`. Both systems agree on the ordering: **True**.

These views sit on DIFFERENT physical sensors. The dependence is a property of the scenario, not of shared hardware. This is NOT a same-sensor result and must not be read as one.

| system | AUC most-dependent over least-dependent (clean data) |
|---|---|
| naive | not measured |
| prismflow | 0.3865 +/- 0.0036 (n=5) |

AUC that a most-dependent pair's per-sample excess exceeds a least-dependent pair's, on CLEAN data. Pair-against-pair, NOT condition-against-clean, so it is not comparable in scale with Part 10's clone_k2 AUC of 0.8743 +/- 0.0583.

Per-pair mean unexplained agreement on clean data, PrismFlow, seed-0 row shown with its committed Part 28 dependence beside it:

| pair | committed dependence (Part 28) | mean excess | measurable samples |
|---|---|---|---|
| `dns-suricata` | 0.7784 +/- 0.0063 | -0.1345 | 1,101 / 1,108 |
| `auth-audit` | 0.6460 +/- 0.0804 | -0.0000 | 586 / 1,108 |
| `auth-dns` | 0.0411 +/- 0.0156 | -0.0257 | 580 / 1,108 |
| `apache-dns` | 0.0458 +/- 0.0220 | -0.0171 | 802 / 1,108 |

## Test D: extra missingness, on top of the natural rate

`auth` is already absent in 47.1-47.5% of windows and `apache` in 18.0-27.5% (`results/ait_clean/results.json`, `directions.<name>.missingness`), so these rates are ADDITIONAL, not total.

| system | condition | detector ROC AUC | detection @5% | detection @1% | balanced accuracy |
|---|---|---|---|---|---|
| naive | +30% masked | not measured | not measured | not measured | 0.8696 +/- 0.0137 (n=5) |
| naive | +50% masked | not measured | not measured | not measured | 0.7736 +/- 0.0074 (n=5) |
| prismflow | +30% masked | 0.3169 +/- 0.0205 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.8671 +/- 0.0144 (n=5) |
| prismflow | +50% masked | 0.2384 +/- 0.0204 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.0000 +/- 0.0000 (n=5) | 0.7731 +/- 0.0085 (n=5) |

Clean reference, same seeds:

| system | detector ROC AUC | balanced accuracy | ENIV (eigen) |
|---|---|---|---|
| naive | not measured | 0.9444 +/- 0.0056 (n=5) | 3.4507 +/- 0.0790 (n=5) |
| prismflow | 0.5000 +/- 0.0000 (n=5) | 0.9489 +/- 0.0069 (n=5) | 3.4958 +/- 0.1015 (n=5) |

---

# Direction B_santos_to_rm

B: train santos -> test russellmitchell. Train n = 1,108, test n = 1,145 (`label_known` rows only).

## Feature scale, and why epsilon is not rescaled

| view | std | IQR | median abs | max abs | present rows |
|---|---|---|---|---|---|
| `apache` | 1.0098 | 0.8789 | 0.4525 | 24.77 | 939 |
| `auth` | 0.9537 | 0.4091 | 0.1913 | 12.16 | 601 |
| `dns` | 0.7285 | 0.9514 | 0.5670 | 3.86 | 1,145 |
| `audit` | 1.5947 | 1.1604 | 0.5691 | 84.47 | 823 |
| `suricata` | 0.9258 | 1.3454 | 0.6044 | 7.68 | 1,145 |

Median per-view std **0.9537** against Part 09's synthetic reference **1.3**, range 0.7285 to 1.5947. Epsilon is **not rescaled**; `config.yaml` records why.

## Test A: natural windows, no synthetic perturbation

The two attack-window populations are reported separately and never pooled. `multi_step` uses the STRICT definition (a step outside `EXFILTRATION_TOOL_STEPS`).

benign + dnsteal_only + multi_step + exfiltration_service_only accounts for every label_known row. The last population is the single window the strict and loose multi-step definitions differ by; it is reported separately and pooled with neither.

**3 windows on santos and 4 on russellmitchell. Any claim about the multi-step intrusion is a handful-of-windows result: descriptive only, no ROC AUC, no significance claim, and no mean +/- std over seeds presented as a finding.**

### naive

| population | windows | ENIV (eigen) | ENIV (design effect) | mean dependence | detector mean score |
|---|---|---|---|---|---|
| `benign` | 114 | 3.0888 +/- 0.0777 (5 seeds), n=114 windows | 2.3920 +/- 0.0952 (5 seeds), n=114 windows | 0.2732 +/- 0.0204 (5 seeds), n=114 windows | not measured (n=114 windows) |
| `dnsteal_only` | 1026 | 2.7853 +/- 0.0810 (5 seeds), n=1026 windows | 1.9750 +/- 0.0645 (5 seeds), n=1026 windows | 0.3835 +/- 0.0203 (5 seeds), n=1026 windows | not measured (n=1026 windows) |
| `multi_step` **DEGENERATE** | 4 | not measured (n=4 windows, degenerate) | not measured (n=4 windows, degenerate) | not measured (n=4 windows, degenerate) | not measured (n=4 windows) |
| `exfiltration_service_only` **DEGENERATE** | 1 | not measured (n=1 windows, degenerate) | not measured (n=1 windows, degenerate) | not measured (n=1 windows, degenerate) | not measured (n=1 windows) |

**DEGENERATE: `multi_step` (n=4), `exfiltration_service_only` (n=1).** At these sample sizes every off-diagonal dependence entry comes back exactly 0, so ENIV pins at the view count (5.0000) and the efficiency ratio at 1.0 in all 5 seeds, with std exactly 0. That is the estimator reporting that it had nothing to work with, NOT a measurement that the five views are independent on these windows. The raw pinned values are in `results.json`; they are not a finding and must not be quoted as one.

### prismflow

| population | windows | ENIV (eigen) | ENIV (design effect) | mean dependence | detector mean score |
|---|---|---|---|---|---|
| `benign` | 114 | 3.1458 +/- 0.0856 (5 seeds), n=114 windows | 2.5139 +/- 0.1229 (5 seeds), n=114 windows | 0.2482 +/- 0.0249 (5 seeds), n=114 windows | 0.1422 +/- 0.1084 (5 seeds), n=114 windows |
| `dnsteal_only` | 1026 | 2.8917 +/- 0.0823 (5 seeds), n=1026 windows | 2.0446 +/- 0.0668 (5 seeds), n=1026 windows | 0.3619 +/- 0.0196 (5 seeds), n=1026 windows | 0.0417 +/- 0.0592 (5 seeds), n=1026 windows |
| `multi_step` **DEGENERATE** | 4 | not measured (n=4 windows, degenerate) | not measured (n=4 windows, degenerate) | not measured (n=4 windows, degenerate) | 0.2554 +/- 0.0079 (5 seeds), n=4 windows |
| `exfiltration_service_only` **DEGENERATE** | 1 | not measured (n=1 windows, degenerate) | not measured (n=1 windows, degenerate) | not measured (n=1 windows, degenerate) | 0.0000 +/- 0.0000 (5 seeds), n=1 windows |

**DEGENERATE: `multi_step` (n=4), `exfiltration_service_only` (n=1).** At these sample sizes every off-diagonal dependence entry comes back exactly 0, so ENIV pins at the view count (5.0000) and the efficiency ratio at 1.0 in all 5 seeds, with std exactly 0. That is the estimator reporting that it had nothing to work with, NOT a measurement that the five views are independent on these windows. The raw pinned values are in `results.json`; they are not a finding and must not be quoted as one.

## Test B: chorus attack, split by whether dns is compromised

Subsets: **exhaustive**, 25 in total (11 containing `dns`, 14 not). Sampled: **False**. All subsets of every k were run. No cell was sampled, so no cell is presented as exhaustive when it is not.

### chorus, epsilon 0.2, beta 1.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0842 +/- 0.0992 (1x5) | 2.9266 +/- 0.0916 (1x5) | not measured | not measured |
| 1 | no | 4 | 0.0462 +/- 0.1035 (4x5) | 2.9810 +/- 0.0714 (4x5) | not measured | not measured |
| 2 | **yes** | 4 | 0.1602 +/- 0.1018 (4x5) | 2.9248 +/- 0.1059 (4x5) | 0.3405 +/- 0.2780 (4x5) | not measured |
| 2 | no | 6 | 0.0431 +/- 0.0877 (6x5) | 2.9267 +/- 0.0676 (6x5) | 0.4451 +/- 0.2755 (6x5) | not measured |
| 3 | **yes** | 6 | 0.1318 +/- 0.1034 (6x5) | 2.9387 +/- 0.0872 (6x5) | 0.3524 +/- 0.1115 (6x5) | not measured |
| 3 | no | 4 | 0.0641 +/- 0.1023 (4x5) | 2.9099 +/- 0.0641 (4x5) | 0.4461 +/- 0.0628 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 0.2, beta 1.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0307 +/- 0.0175 (1x5) | 3.0276 +/- 0.0345 (1x5) | not measured | 0.5590 +/- 0.1129 (1x5) |
| 1 | no | 4 | 0.0146 +/- 0.0443 (4x5) | 3.0635 +/- 0.0706 (4x5) | not measured | 0.4654 +/- 0.0923 (4x5) |
| 2 | **yes** | 4 | 0.0962 +/- 0.0920 (4x5) | 3.0220 +/- 0.1142 (4x5) | 0.3231 +/- 0.2685 (4x5) | 0.4974 +/- 0.1271 (4x5) |
| 2 | no | 6 | 0.0115 +/- 0.0144 (6x5) | 3.0034 +/- 0.0694 (6x5) | 0.4293 +/- 0.2786 (6x5) | 0.4726 +/- 0.1150 (6x5) |
| 3 | **yes** | 6 | 0.0766 +/- 0.0787 (6x5) | 3.0396 +/- 0.0772 (6x5) | 0.3265 +/- 0.1044 (6x5) | 0.5272 +/- 0.1423 (6x5) |
| 3 | no | 4 | 0.0167 +/- 0.0148 (4x5) | 2.9908 +/- 0.0602 (4x5) | 0.4280 +/- 0.0573 (4x5) | 0.4801 +/- 0.1291 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 0.5, beta 1.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0987 +/- 0.1229 (1x5) | 2.9086 +/- 0.1357 (1x5) | not measured | not measured |
| 1 | no | 4 | 0.1152 +/- 0.2152 (4x5) | 2.9978 +/- 0.0684 (4x5) | not measured | not measured |
| 2 | **yes** | 4 | 0.7376 +/- 0.1206 (4x5) | 2.8421 +/- 0.0741 (4x5) | 0.3864 +/- 0.3108 (4x5) | not measured |
| 2 | no | 6 | 0.0481 +/- 0.0901 (6x5) | 2.9759 +/- 0.0765 (6x5) | 0.4386 +/- 0.2902 (6x5) | not measured |
| 3 | **yes** | 6 | 0.7145 +/- 0.0978 (6x5) | 2.8437 +/- 0.0629 (6x5) | 0.3759 +/- 0.1207 (6x5) | not measured |
| 3 | no | 4 | 0.0693 +/- 0.1045 (4x5) | 2.9719 +/- 0.0854 (4x5) | 0.4390 +/- 0.0673 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 0.5, beta 1.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0360 +/- 0.0237 (1x5) | 3.0363 +/- 0.0272 (1x5) | not measured | 0.6178 +/- 0.1092 (1x5) |
| 1 | no | 4 | 0.0651 +/- 0.1393 (4x5) | 3.0474 +/- 0.1101 (4x5) | not measured | 0.4283 +/- 0.0954 (4x5) |
| 2 | **yes** | 4 | 0.7783 +/- 0.0894 (4x5) | 2.9187 +/- 0.0909 (4x5) | 0.3855 +/- 0.3100 (4x5) | 0.5474 +/- 0.1248 (4x5) |
| 2 | no | 6 | 0.0191 +/- 0.0165 (6x5) | 3.0152 +/- 0.0918 (6x5) | 0.4243 +/- 0.2893 (6x5) | 0.3882 +/- 0.0971 (6x5) |
| 3 | **yes** | 6 | 0.7172 +/- 0.1138 (6x5) | 2.9314 +/- 0.0595 (6x5) | 0.3570 +/- 0.1152 (6x5) | 0.5025 +/- 0.1331 (6x5) |
| 3 | no | 4 | 0.0234 +/- 0.0170 (4x5) | 3.0052 +/- 0.0860 (4x5) | 0.4242 +/- 0.0580 (4x5) | 0.3438 +/- 0.0673 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 1.0, beta 0.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.2818 +/- 0.3387 (4x5) | 3.0912 +/- 0.1966 (4x5) | 0.3589 +/- 0.2043 (4x5) | not measured |
| 2 | no | 6 | 0.3779 +/- 0.3741 (6x5) | 3.1854 +/- 0.1744 (6x5) | 0.2923 +/- 0.2170 (6x5) | not measured |
| 3 | **yes** | 6 | 0.4556 +/- 0.3714 (6x5) | 3.1977 +/- 0.2194 (6x5) | 0.3389 +/- 0.1101 (6x5) | not measured |
| 3 | no | 4 | 0.5565 +/- 0.3219 (4x5) | 3.2685 +/- 0.1732 (4x5) | 0.2958 +/- 0.0806 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 1.0, beta 0.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.1873 +/- 0.2678 (4x5) | 3.1800 +/- 0.2034 (4x5) | 0.5255 +/- 0.1664 (4x5) | 0.5076 +/- 0.1272 (4x5) |
| 2 | no | 6 | 0.3891 +/- 0.3524 (6x5) | 3.1403 +/- 0.2001 (6x5) | 0.2929 +/- 0.2196 (6x5) | 0.3470 +/- 0.1004 (6x5) |
| 3 | **yes** | 6 | 0.3923 +/- 0.3385 (6x5) | 3.2185 +/- 0.2139 (6x5) | 0.4552 +/- 0.0946 (6x5) | 0.4242 +/- 0.1316 (6x5) |
| 3 | no | 4 | 0.6100 +/- 0.3173 (4x5) | 3.0613 +/- 0.1795 (4x5) | 0.3288 +/- 0.0569 (4x5) | 0.2680 +/- 0.0881 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 1.0, beta 1.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0996 +/- 0.1248 (1x5) | 2.9507 +/- 0.1610 (1x5) | not measured | not measured |
| 1 | no | 4 | 0.1934 +/- 0.3344 (4x5) | 3.0854 +/- 0.1481 (4x5) | not measured | not measured |
| 2 | **yes** | 4 | 0.8137 +/- 0.1122 (4x5) | 2.8019 +/- 0.0988 (4x5) | 0.4992 +/- 0.3364 (4x5) | not measured |
| 2 | no | 6 | 0.0516 +/- 0.0894 (6x5) | 3.0147 +/- 0.2063 (6x5) | 0.4108 +/- 0.2747 (6x5) | not measured |
| 3 | **yes** | 6 | 0.8825 +/- 0.1332 (6x5) | 2.8282 +/- 0.1266 (6x5) | 0.4120 +/- 0.1463 (6x5) | not measured |
| 3 | no | 4 | 0.0717 +/- 0.1038 (4x5) | 3.0461 +/- 0.1505 (4x5) | 0.4054 +/- 0.0909 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

#### Every subset at the primary cell - naive

| compromised views | k | dns | attack success (5 seeds) | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| `dns` | 1 | **yes** | 0.0996 +/- 0.1248 (n=5) | 2.9507 +/- 0.1610 (n=5) | not measured | not measured |
| `apache` | 1 | no | 0.0049 +/- 0.0100 (n=5) | 2.9402 +/- 0.0857 (n=5) | not measured | not measured |
| `audit` | 1 | no | 0.0094 +/- 0.0172 (n=5) | 3.1381 +/- 0.0705 (n=5) | not measured | not measured |
| `auth` | 1 | no | 0.0051 +/- 0.0089 (n=5) | 3.2279 +/- 0.1549 (n=5) | not measured | not measured |
| `suricata` | 1 | no | 0.7541 +/- 0.0812 (n=5) | 3.0355 +/- 0.0976 (n=5) | not measured | not measured |
| `apache+dns` | 2 | **yes** | 0.7834 +/- 0.1220 (n=5) | 2.7078 +/- 0.0481 (n=5) | 0.6984 +/- 0.1027 (n=5) | not measured |
| `auth+dns` | 2 | **yes** | 0.7822 +/- 0.1228 (n=5) | 2.9068 +/- 0.1043 (n=5) | 0.1931 +/- 0.1198 (n=5) | not measured |
| `dns+audit` | 2 | **yes** | 0.7824 +/- 0.1225 (n=5) | 2.8354 +/- 0.0660 (n=5) | 0.1857 +/- 0.0402 (n=5) | not measured |
| `dns+suricata` | 2 | **yes** | 0.9069 +/- 0.0186 (n=5) | 2.7575 +/- 0.0134 (n=5) | 0.9195 +/- 0.0212 (n=5) | not measured |
| `apache+audit` | 2 | no | 0.0136 +/- 0.0174 (n=5) | 3.0748 +/- 0.0781 (n=5) | 0.1193 +/- 0.0299 (n=5) | not measured |
| `apache+auth` | 2 | no | 0.0096 +/- 0.0112 (n=5) | 3.1354 +/- 0.1731 (n=5) | 0.6686 +/- 0.1570 (n=5) | not measured |
| `apache+suricata` | 2 | no | 0.0917 +/- 0.1223 (n=5) | 2.7851 +/- 0.1135 (n=5) | 0.7426 +/- 0.0408 (n=5) | not measured |
| `audit+suricata` | 2 | no | 0.0893 +/- 0.1236 (n=5) | 2.8914 +/- 0.1012 (n=5) | 0.1726 +/- 0.0572 (n=5) | not measured |
| `auth+audit` | 2 | no | 0.0159 +/- 0.0292 (n=5) | 3.2616 +/- 0.1843 (n=5) | 0.5739 +/- 0.1199 (n=5) | not measured |
| `auth+suricata` | 2 | no | 0.0893 +/- 0.1236 (n=5) | 2.9396 +/- 0.1568 (n=5) | 0.1875 +/- 0.0938 (n=5) | not measured |
| `apache+auth+dns` | 3 | **yes** | 0.7775 +/- 0.1227 (n=5) | 2.9280 +/- 0.0876 (n=5) | 0.4014 +/- 0.0234 (n=5) | not measured |
| `apache+dns+audit` | 3 | **yes** | 0.7782 +/- 0.1226 (n=5) | 2.8473 +/- 0.0638 (n=5) | 0.3283 +/- 0.0280 (n=5) | not measured |
| `apache+dns+suricata` | 3 | **yes** | 0.9748 +/- 0.0287 (n=5) | 2.6523 +/- 0.0586 (n=5) | 0.7196 +/- 0.0382 (n=5) | not measured |
| `auth+dns+audit` | 3 | **yes** | 0.7782 +/- 0.1239 (n=5) | 2.9676 +/- 0.0897 (n=5) | 0.3094 +/- 0.0551 (n=5) | not measured |
| `auth+dns+suricata` | 3 | **yes** | 0.9948 +/- 0.0006 (n=5) | 2.8169 +/- 0.1011 (n=5) | 0.3479 +/- 0.0307 (n=5) | not measured |
| `dns+audit+suricata` | 3 | **yes** | 0.9916 +/- 0.0076 (n=5) | 2.7574 +/- 0.0225 (n=5) | 0.3652 +/- 0.0173 (n=5) | not measured |
| `apache+audit+suricata` | 3 | no | 0.0891 +/- 0.1237 (n=5) | 2.9182 +/- 0.0819 (n=5) | 0.3468 +/- 0.0383 (n=5) | not measured |
| `apache+auth+audit` | 3 | no | 0.0197 +/- 0.0284 (n=5) | 3.1763 +/- 0.1388 (n=5) | 0.4830 +/- 0.0700 (n=5) | not measured |
| `apache+auth+suricata` | 3 | no | 0.0891 +/- 0.1237 (n=5) | 2.9932 +/- 0.1562 (n=5) | 0.4751 +/- 0.0402 (n=5) | not measured |
| `auth+audit+suricata` | 3 | no | 0.0891 +/- 0.1237 (n=5) | 3.0968 +/- 0.0945 (n=5) | 0.3165 +/- 0.0590 (n=5) | not measured |

### chorus, epsilon 1.0, beta 1.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0356 +/- 0.0231 (1x5) | 3.0898 +/- 0.0453 (1x5) | not measured | 0.6220 +/- 0.1049 (1x5) |
| 1 | no | 4 | 0.1636 +/- 0.2719 (4x5) | 3.1067 +/- 0.2018 (4x5) | not measured | 0.4240 +/- 0.0920 (4x5) |
| 2 | **yes** | 4 | 0.8634 +/- 0.0272 (4x5) | 2.8625 +/- 0.1389 (4x5) | 0.6268 +/- 0.2727 (4x5) | 0.2757 +/- 0.1166 (4x5) |
| 2 | no | 6 | 0.0393 +/- 0.0409 (6x5) | 2.9910 +/- 0.1704 (6x5) | 0.4239 +/- 0.2874 (6x5) | 0.3392 +/- 0.1083 (6x5) |
| 3 | **yes** | 6 | 0.9185 +/- 0.0772 (6x5) | 2.7992 +/- 0.1373 (6x5) | 0.4927 +/- 0.1401 (6x5) | 0.2265 +/- 0.0612 (6x5) |
| 3 | no | 4 | 0.0403 +/- 0.0489 (4x5) | 2.9472 +/- 0.1547 (4x5) | 0.4141 +/- 0.0865 (4x5) | 0.3403 +/- 0.1541 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

#### Every subset at the primary cell - prismflow

| compromised views | k | dns | attack success (5 seeds) | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| `dns` | 1 | **yes** | 0.0356 +/- 0.0231 (n=5) | 3.0898 +/- 0.0453 (n=5) | not measured | 0.6220 +/- 0.1049 (n=5) |
| `apache` | 1 | no | 0.0203 +/- 0.0274 (n=5) | 2.8306 +/- 0.0737 (n=5) | not measured | 0.3761 +/- 0.1441 (n=5) |
| `audit` | 1 | no | 0.0262 +/- 0.0406 (n=5) | 3.2132 +/- 0.0498 (n=5) | not measured | 0.4439 +/- 0.0510 (n=5) |
| `auth` | 1 | no | 0.0108 +/- 0.0090 (n=5) | 3.3135 +/- 0.1284 (n=5) | not measured | 0.4279 +/- 0.0943 (n=5) |
| `suricata` | 1 | no | 0.5972 +/- 0.1877 (n=5) | 3.0697 +/- 0.0661 (n=5) | not measured | 0.4479 +/- 0.0622 (n=5) |
| `apache+dns` | 2 | **yes** | 0.8683 +/- 0.0377 (n=5) | 2.7298 +/- 0.0483 (n=5) | 0.8201 +/- 0.1336 (n=5) | 0.2089 +/- 0.0195 (n=5) |
| `auth+dns` | 2 | **yes** | 0.8503 +/- 0.0169 (n=5) | 2.9614 +/- 0.1540 (n=5) | 0.4034 +/- 0.2172 (n=5) | 0.2775 +/- 0.0838 (n=5) |
| `dns+audit` | 2 | **yes** | 0.8491 +/- 0.0172 (n=5) | 2.9249 +/- 0.1131 (n=5) | 0.3934 +/- 0.1463 (n=5) | 0.2868 +/- 0.0927 (n=5) |
| `dns+suricata` | 2 | **yes** | 0.8861 +/- 0.0194 (n=5) | 2.8337 +/- 0.1122 (n=5) | 0.8904 +/- 0.0527 (n=5) | 0.3296 +/- 0.1980 (n=5) |
| `apache+audit` | 2 | no | 0.0562 +/- 0.0743 (n=5) | 3.1201 +/- 0.0867 (n=5) | 0.1233 +/- 0.0537 (n=5) | 0.2783 +/- 0.0633 (n=5) |
| `apache+auth` | 2 | no | 0.0400 +/- 0.0453 (n=5) | 3.0293 +/- 0.1989 (n=5) | 0.5618 +/- 0.2436 (n=5) | 0.3180 +/- 0.1545 (n=5) |
| `apache+suricata` | 2 | no | 0.0398 +/- 0.0178 (n=5) | 2.8136 +/- 0.0796 (n=5) | 0.8098 +/- 0.0866 (n=5) | 0.3113 +/- 0.0858 (n=5) |
| `audit+suricata` | 2 | no | 0.0262 +/- 0.0189 (n=5) | 2.8937 +/- 0.1070 (n=5) | 0.2376 +/- 0.0324 (n=5) | 0.4506 +/- 0.1297 (n=5) |
| `auth+audit` | 2 | no | 0.0470 +/- 0.0516 (n=5) | 3.1415 +/- 0.1160 (n=5) | 0.6265 +/- 0.1691 (n=5) | 0.3560 +/- 0.0951 (n=5) |
| `auth+suricata` | 2 | no | 0.0264 +/- 0.0189 (n=5) | 2.9481 +/- 0.1664 (n=5) | 0.1845 +/- 0.1008 (n=5) | 0.3211 +/- 0.0379 (n=5) |
| `apache+auth+dns` | 3 | **yes** | 0.8433 +/- 0.0213 (n=5) | 2.8817 +/- 0.2140 (n=5) | 0.5271 +/- 0.0753 (n=5) | 0.1842 +/- 0.0151 (n=5) |
| `apache+dns+audit` | 3 | **yes** | 0.8437 +/- 0.0208 (n=5) | 2.8038 +/- 0.1524 (n=5) | 0.5201 +/- 0.0691 (n=5) | 0.2072 +/- 0.0226 (n=5) |
| `apache+dns+suricata` | 3 | **yes** | 0.9914 +/- 0.0071 (n=5) | 2.6609 +/- 0.0814 (n=5) | 0.7277 +/- 0.0342 (n=5) | 0.2170 +/- 0.0557 (n=5) |
| `auth+dns+audit` | 3 | **yes** | 0.8440 +/- 0.0178 (n=5) | 2.8766 +/- 0.0332 (n=5) | 0.4714 +/- 0.0832 (n=5) | 0.2647 +/- 0.0887 (n=5) |
| `auth+dns+suricata` | 3 | **yes** | 0.9939 +/- 0.0020 (n=5) | 2.8150 +/- 0.1240 (n=5) | 0.3204 +/- 0.0309 (n=5) | 0.2377 +/- 0.0597 (n=5) |
| `dns+audit+suricata` | 3 | **yes** | 0.9944 +/- 0.0013 (n=5) | 2.7572 +/- 0.0454 (n=5) | 0.3895 +/- 0.0187 (n=5) | 0.2484 +/- 0.0806 (n=5) |
| `apache+audit+suricata` | 3 | no | 0.0262 +/- 0.0189 (n=5) | 2.9104 +/- 0.1118 (n=5) | 0.3883 +/- 0.0330 (n=5) | 0.3874 +/- 0.1591 (n=5) |
| `apache+auth+audit` | 3 | no | 0.0828 +/- 0.0853 (n=5) | 2.9119 +/- 0.1862 (n=5) | 0.4476 +/- 0.1390 (n=5) | 0.1970 +/- 0.0082 (n=5) |
| `apache+auth+suricata` | 3 | no | 0.0264 +/- 0.0189 (n=5) | 2.9905 +/- 0.1676 (n=5) | 0.4689 +/- 0.0427 (n=5) | 0.2956 +/- 0.0669 (n=5) |
| `auth+audit+suricata` | 3 | no | 0.0260 +/- 0.0189 (n=5) | 2.9759 +/- 0.1785 (n=5) | 0.3517 +/- 0.0499 (n=5) | 0.4810 +/- 0.1651 (n=5) |

### chorus, epsilon 1.0, beta 5.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.7893 +/- 0.1048 (4x5) | 2.7643 +/- 0.1066 (4x5) | 0.4711 +/- 0.3825 (4x5) | not measured |
| 2 | no | 6 | 0.0507 +/- 0.0896 (6x5) | 3.0676 +/- 0.1614 (6x5) | 0.4237 +/- 0.2735 (6x5) | not measured |
| 3 | **yes** | 6 | 0.8233 +/- 0.1447 (6x5) | 2.8172 +/- 0.1159 (6x5) | 0.4001 +/- 0.1509 (6x5) | not measured |
| 3 | no | 4 | 0.0712 +/- 0.1041 (4x5) | 3.1253 +/- 0.1451 (4x5) | 0.4216 +/- 0.0863 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 1.0, beta 5.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.8447 +/- 0.0242 (4x5) | 2.8064 +/- 0.1258 (4x5) | 0.5491 +/- 0.3247 (4x5) | 0.2539 +/- 0.1142 (4x5) |
| 2 | no | 6 | 0.0354 +/- 0.0400 (6x5) | 3.0019 +/- 0.1852 (6x5) | 0.4502 +/- 0.2744 (6x5) | 0.3648 +/- 0.1257 (6x5) |
| 3 | **yes** | 6 | 0.8482 +/- 0.1358 (6x5) | 2.8279 +/- 0.1322 (6x5) | 0.4532 +/- 0.1353 (6x5) | 0.2031 +/- 0.0726 (6x5) |
| 3 | no | 4 | 0.0418 +/- 0.0524 (4x5) | 2.9623 +/- 0.1546 (4x5) | 0.4318 +/- 0.0857 (4x5) | 0.2891 +/- 0.0957 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 2.0, beta 1.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0997 +/- 0.1252 (1x5) | 3.0394 +/- 0.1926 (1x5) | not measured | not measured |
| 1 | no | 4 | 0.2069 +/- 0.3561 (4x5) | 3.1942 +/- 0.2101 (4x5) | not measured | not measured |
| 2 | **yes** | 4 | 0.8167 +/- 0.1152 (4x5) | 2.9672 +/- 0.1766 (4x5) | 0.5796 +/- 0.2967 (4x5) | not measured |
| 2 | no | 6 | 0.0540 +/- 0.0894 (6x5) | 3.1812 +/- 0.2112 (6x5) | 0.4473 +/- 0.2910 (6x5) | not measured |
| 3 | **yes** | 6 | 0.8645 +/- 0.1267 (6x5) | 3.0588 +/- 0.2157 (6x5) | 0.4401 +/- 0.1403 (6x5) | not measured |
| 3 | no | 4 | 0.0728 +/- 0.1037 (4x5) | 3.2447 +/- 0.1817 (4x5) | 0.3876 +/- 0.1160 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### chorus, epsilon 2.0, beta 1.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0344 +/- 0.0213 (1x5) | 3.1755 +/- 0.0942 (1x5) | not measured | 0.5726 +/- 0.1451 (1x5) |
| 1 | no | 4 | 0.2361 +/- 0.3569 (4x5) | 3.1678 +/- 0.2106 (4x5) | not measured | 0.4237 +/- 0.0941 (4x5) |
| 2 | **yes** | 4 | 0.8747 +/- 0.0318 (4x5) | 3.0031 +/- 0.1856 (4x5) | 0.8028 +/- 0.1712 (4x5) | 0.3180 +/- 0.1387 (4x5) |
| 2 | no | 6 | 0.0625 +/- 0.0740 (6x5) | 3.0870 +/- 0.2063 (6x5) | 0.4658 +/- 0.2466 (6x5) | 0.3619 +/- 0.1333 (6x5) |
| 3 | **yes** | 6 | 0.9181 +/- 0.0783 (6x5) | 2.8642 +/- 0.1965 (6x5) | 0.6015 +/- 0.1883 (6x5) | 0.2475 +/- 0.0659 (6x5) |
| 3 | no | 4 | 0.0599 +/- 0.0890 (4x5) | 3.0495 +/- 0.1485 (4x5) | 0.4367 +/- 0.1124 (4x5) | 0.3313 +/- 0.1324 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### pgd, epsilon 1.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0996 +/- 0.1248 (1x5) | 2.9507 +/- 0.1610 (1x5) | not measured | not measured |
| 1 | no | 4 | 0.1934 +/- 0.3344 (4x5) | 3.0854 +/- 0.1481 (4x5) | not measured | not measured |
| 2 | **yes** | 4 | 0.2818 +/- 0.3387 (4x5) | 3.0910 +/- 0.1962 (4x5) | 0.3587 +/- 0.2037 (4x5) | not measured |
| 2 | no | 6 | 0.3819 +/- 0.3782 (6x5) | 3.1844 +/- 0.1701 (6x5) | 0.2953 +/- 0.2177 (6x5) | not measured |
| 3 | **yes** | 6 | 0.4596 +/- 0.3752 (6x5) | 3.1980 +/- 0.2146 (6x5) | 0.3390 +/- 0.1090 (6x5) | not measured |
| 3 | no | 4 | 0.5624 +/- 0.3254 (4x5) | 3.2667 +/- 0.1649 (4x5) | 0.2992 +/- 0.0771 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### pgd, epsilon 1.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 1 | **yes** | 1 | 0.0356 +/- 0.0231 (1x5) | 3.0898 +/- 0.0453 (1x5) | not measured | 0.6220 +/- 0.1049 (1x5) |
| 1 | no | 4 | 0.1636 +/- 0.2719 (4x5) | 3.1067 +/- 0.2018 (4x5) | not measured | 0.4240 +/- 0.0920 (4x5) |
| 2 | **yes** | 4 | 0.1876 +/- 0.2682 (4x5) | 3.1795 +/- 0.2051 (4x5) | 0.5252 +/- 0.1683 (4x5) | 0.5078 +/- 0.1273 (4x5) |
| 2 | no | 6 | 0.3910 +/- 0.3539 (6x5) | 3.1384 +/- 0.1941 (6x5) | 0.2944 +/- 0.2175 (6x5) | 0.3466 +/- 0.1029 (6x5) |
| 3 | **yes** | 6 | 0.3988 +/- 0.3446 (6x5) | 3.2140 +/- 0.2151 (6x5) | 0.4580 +/- 0.0937 (6x5) | 0.4279 +/- 0.1371 (6x5) |
| 3 | no | 4 | 0.6112 +/- 0.3183 (4x5) | 3.0507 +/- 0.1682 (4x5) | 0.3327 +/- 0.0547 (4x5) | 0.2762 +/- 0.0861 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### pgd, epsilon 2.0 - naive

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.2943 +/- 0.3597 (4x5) | 3.2799 +/- 0.2580 (4x5) | 0.4301 +/- 0.2289 (4x5) | not measured |
| 2 | no | 6 | 0.4123 +/- 0.4067 (6x5) | 3.3216 +/- 0.2176 (6x5) | 0.2571 +/- 0.2101 (6x5) | not measured |
| 3 | **yes** | 6 | 0.4879 +/- 0.4028 (6x5) | 3.4041 +/- 0.2616 (6x5) | 0.3821 +/- 0.1232 (6x5) | not measured |
| 3 | no | 4 | 0.6156 +/- 0.3553 (4x5) | 3.3370 +/- 0.1982 (4x5) | 0.2805 +/- 0.0863 (4x5) | not measured |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### pgd, epsilon 2.0 - prismflow

| k | dns in compromised set | subsets | attack success | ENIV (eigen) | dependence among compromised | detector ROC AUC |
|---|---|---|---|---|---|---|
| 2 | **yes** | 4 | 0.2593 +/- 0.3564 (4x5) | 3.3082 +/- 0.2320 (4x5) | 0.6190 +/- 0.2156 (4x5) | 0.5133 +/- 0.1468 (4x5) |
| 2 | no | 6 | 0.4698 +/- 0.3871 (6x5) | 3.2211 +/- 0.2110 (6x5) | 0.3172 +/- 0.1611 (6x5) | 0.3487 +/- 0.1028 (6x5) |
| 3 | **yes** | 6 | 0.4884 +/- 0.3876 (6x5) | 3.3083 +/- 0.2508 (6x5) | 0.5526 +/- 0.1314 (6x5) | 0.4349 +/- 0.1519 (6x5) |
| 3 | no | 4 | 0.6803 +/- 0.3130 (4x5) | 3.0385 +/- 0.1608 (4x5) | 0.3957 +/- 0.1045 (4x5) | 0.2847 +/- 0.1051 (4x5) |

Counts are `subsets x seeds`. **These +/- values pool subset-to-subset variation with seed-to-seed variation**, so a cell spanning more than one subset is not a 5-seed spread; the per-subset tables below carry the 5-seed figures.

### Paired prismflow - naive, per cell, within the dns split

Negative attack success would mean the discount defends. The seed count beside each mean is how many of the seeds favour PrismFlow.

| cell | k | dns compromised | attack success (pf - naive) |
|---|---|---|---|
| `chorus_k1_eps0.2_beta1.0_[apache]` | 1 | no | -0.0005 +/- 0.0019 (2/5) |
| `chorus_k1_eps0.2_beta1.0_[audit]` | 1 | no | -0.0009 +/- 0.0026 (2/5) |
| `chorus_k1_eps0.2_beta1.0_[auth]` | 1 | no | -0.0005 +/- 0.0016 (3/5) |
| `chorus_k1_eps0.2_beta1.0_[dns]` | 1 | yes | -0.0534 +/- 0.0937 (5/5) |
| `chorus_k1_eps0.2_beta1.0_[suricata]` | 1 | no | -0.1244 +/- 0.1237 (5/5) |
| `chorus_k1_eps0.5_beta1.0_[apache]` | 1 | no | +0.0058 +/- 0.0137 (1/5) |
| `chorus_k1_eps0.5_beta1.0_[audit]` | 1 | no | +0.0038 +/- 0.0065 (1/5) |
| `chorus_k1_eps0.5_beta1.0_[auth]` | 1 | no | -0.0017 +/- 0.0046 (3/5) |
| `chorus_k1_eps0.5_beta1.0_[dns]` | 1 | yes | -0.0627 +/- 0.1059 (5/5) |
| `chorus_k1_eps0.5_beta1.0_[suricata]` | 1 | no | -0.2082 +/- 0.1489 (5/5) |
| `chorus_k1_eps1.0_beta1.0_[apache]` | 1 | no | +0.0154 +/- 0.0316 (1/5) |
| `chorus_k1_eps1.0_beta1.0_[audit]` | 1 | no | +0.0168 +/- 0.0235 (0/5) |
| `chorus_k1_eps1.0_beta1.0_[auth]` | 1 | no | +0.0058 +/- 0.0023 (0/5) |
| `chorus_k1_eps1.0_beta1.0_[dns]` | 1 | yes | -0.0639 +/- 0.1086 (5/5) |
| `chorus_k1_eps1.0_beta1.0_[suricata]` | 1 | no | -0.1569 +/- 0.2406 (4/5) |
| `chorus_k1_eps2.0_beta1.0_[apache]` | 1 | no | +0.0409 +/- 0.0704 (1/5) |
| `chorus_k1_eps2.0_beta1.0_[audit]` | 1 | no | +0.0430 +/- 0.0708 (0/5) |
| `chorus_k1_eps2.0_beta1.0_[auth]` | 1 | no | +0.0054 +/- 0.0059 (1/5) |
| `chorus_k1_eps2.0_beta1.0_[dns]` | 1 | yes | -0.0653 +/- 0.1117 (5/5) |
| `chorus_k1_eps2.0_beta1.0_[suricata]` | 1 | no | +0.0276 +/- 0.0941 (4/5) |
| `chorus_k2_eps0.2_beta1.0_[apache+audit]` | 2 | no | -0.0012 +/- 0.0021 (2/5) |
| `chorus_k2_eps0.2_beta1.0_[apache+auth]` | 2 | no | -0.0002 +/- 0.0027 (3/5) |
| `chorus_k2_eps0.2_beta1.0_[apache+dns]` | 2 | yes | -0.0755 +/- 0.1171 (3/5) |
| `chorus_k2_eps0.2_beta1.0_[apache+suricata]` | 2 | no | -0.0624 +/- 0.1066 (5/5) |
| `chorus_k2_eps0.2_beta1.0_[audit+suricata]` | 2 | no | -0.0625 +/- 0.1065 (5/5) |
| `chorus_k2_eps0.2_beta1.0_[auth+audit]` | 2 | no | -0.0012 +/- 0.0034 (2/5) |
| `chorus_k2_eps0.2_beta1.0_[auth+dns]` | 2 | yes | -0.0666 +/- 0.1133 (3/5) |
| `chorus_k2_eps0.2_beta1.0_[auth+suricata]` | 2 | no | -0.0624 +/- 0.1066 (5/5) |
| `chorus_k2_eps0.2_beta1.0_[dns+audit]` | 2 | yes | -0.0723 +/- 0.1220 (3/5) |
| `chorus_k2_eps0.2_beta1.0_[dns+suricata]` | 2 | yes | -0.0416 +/- 0.0951 (4/5) |
| `chorus_k2_eps0.5_beta1.0_[apache+audit]` | 2 | no | +0.0079 +/- 0.0133 (1/5) |
| `chorus_k2_eps0.5_beta1.0_[apache+auth]` | 2 | no | +0.0040 +/- 0.0086 (1/5) |
| `chorus_k2_eps0.5_beta1.0_[apache+dns]` | 2 | yes | +0.0447 +/- 0.1097 (2/5) |
| `chorus_k2_eps0.5_beta1.0_[apache+suricata]` | 2 | no | -0.0625 +/- 0.1075 (5/5) |
| `chorus_k2_eps0.5_beta1.0_[audit+suricata]` | 2 | no | -0.0631 +/- 0.1072 (5/5) |
| `chorus_k2_eps0.5_beta1.0_[auth+audit]` | 2 | no | +0.0024 +/- 0.0033 (2/5) |
| `chorus_k2_eps0.5_beta1.0_[auth+dns]` | 2 | yes | +0.0576 +/- 0.1143 (1/5) |
| `chorus_k2_eps0.5_beta1.0_[auth+suricata]` | 2 | no | -0.0629 +/- 0.1073 (5/5) |
| `chorus_k2_eps0.5_beta1.0_[dns+audit]` | 2 | yes | +0.0548 +/- 0.1131 (1/5) |
| `chorus_k2_eps0.5_beta1.0_[dns+suricata]` | 2 | yes | +0.0054 +/- 0.0509 (1/5) |
| `chorus_k2_eps1.0_beta0.0_[apache+audit]` | 2 | no | +0.0419 +/- 0.0605 (0/5) |
| `chorus_k2_eps1.0_beta0.0_[apache+auth]` | 2 | no | +0.0288 +/- 0.0399 (1/5) |
| `chorus_k2_eps1.0_beta0.0_[apache+dns]` | 2 | yes | -0.0540 +/- 0.0906 (5/5) |
| `chorus_k2_eps1.0_beta0.0_[apache+suricata]` | 2 | no | -0.0115 +/- 0.1168 (4/5) |
| `chorus_k2_eps1.0_beta0.0_[audit+suricata]` | 2 | no | -0.0101 +/- 0.1166 (3/5) |
| `chorus_k2_eps1.0_beta0.0_[auth+audit]` | 2 | no | +0.0325 +/- 0.0295 (0/5) |
| `chorus_k2_eps1.0_beta0.0_[auth+dns]` | 2 | yes | -0.0604 +/- 0.1076 (5/5) |
| `chorus_k2_eps1.0_beta0.0_[auth+suricata]` | 2 | no | -0.0147 +/- 0.1231 (3/5) |
| `chorus_k2_eps1.0_beta0.0_[dns+audit]` | 2 | yes | -0.0417 +/- 0.0698 (5/5) |
| `chorus_k2_eps1.0_beta0.0_[dns+suricata]` | 2 | yes | -0.2217 +/- 0.1943 (5/5) |
| `chorus_k2_eps1.0_beta1.0_[apache+audit]` | 2 | no | +0.0426 +/- 0.0603 (0/5) |
| `chorus_k2_eps1.0_beta1.0_[apache+auth]` | 2 | no | +0.0304 +/- 0.0400 (1/5) |
| `chorus_k2_eps1.0_beta1.0_[apache+dns]` | 2 | yes | +0.0849 +/- 0.1034 (0/5) |
| `chorus_k2_eps1.0_beta1.0_[apache+suricata]` | 2 | no | -0.0519 +/- 0.1116 (4/5) |
| `chorus_k2_eps1.0_beta1.0_[audit+suricata]` | 2 | no | -0.0631 +/- 0.1072 (5/5) |
| `chorus_k2_eps1.0_beta1.0_[auth+audit]` | 2 | no | +0.0311 +/- 0.0289 (0/5) |
| `chorus_k2_eps1.0_beta1.0_[auth+dns]` | 2 | yes | +0.0681 +/- 0.1078 (0/5) |
| `chorus_k2_eps1.0_beta1.0_[auth+suricata]` | 2 | no | -0.0629 +/- 0.1073 (5/5) |
| `chorus_k2_eps1.0_beta1.0_[dns+audit]` | 2 | yes | +0.0667 +/- 0.1071 (0/5) |
| `chorus_k2_eps1.0_beta1.0_[dns+suricata]` | 2 | yes | -0.0208 +/- 0.0208 (4/5) |
| `chorus_k2_eps1.0_beta5.0_[apache+audit]` | 2 | no | +0.0414 +/- 0.0583 (0/5) |
| `chorus_k2_eps1.0_beta5.0_[apache+auth]` | 2 | no | +0.0264 +/- 0.0322 (0/5) |
| `chorus_k2_eps1.0_beta5.0_[apache+dns]` | 2 | yes | +0.0568 +/- 0.1009 (1/5) |
| `chorus_k2_eps1.0_beta5.0_[apache+suricata]` | 2 | no | -0.0627 +/- 0.1074 (5/5) |
| `chorus_k2_eps1.0_beta5.0_[audit+suricata]` | 2 | no | -0.0634 +/- 0.1070 (5/5) |
| `chorus_k2_eps1.0_beta5.0_[auth+audit]` | 2 | no | +0.0302 +/- 0.0295 (0/5) |
| `chorus_k2_eps1.0_beta5.0_[auth+dns]` | 2 | yes | +0.0639 +/- 0.1061 (0/5) |
| `chorus_k2_eps1.0_beta5.0_[auth+suricata]` | 2 | no | -0.0634 +/- 0.1070 (5/5) |
| `chorus_k2_eps1.0_beta5.0_[dns+audit]` | 2 | yes | +0.0652 +/- 0.1078 (0/5) |
| `chorus_k2_eps1.0_beta5.0_[dns+suricata]` | 2 | yes | +0.0358 +/- 0.0384 (0/5) |
| `chorus_k2_eps2.0_beta1.0_[apache+audit]` | 2 | no | +0.0978 +/- 0.1085 (0/5) |
| `chorus_k2_eps2.0_beta1.0_[apache+auth]` | 2 | no | +0.0620 +/- 0.0741 (0/5) |
| `chorus_k2_eps2.0_beta1.0_[apache+dns]` | 2 | yes | +0.1036 +/- 0.1334 (0/5) |
| `chorus_k2_eps2.0_beta1.0_[apache+suricata]` | 2 | no | -0.0442 +/- 0.0983 (3/5) |
| `chorus_k2_eps2.0_beta1.0_[audit+suricata]` | 2 | no | -0.0631 +/- 0.1072 (5/5) |
| `chorus_k2_eps2.0_beta1.0_[auth+audit]` | 2 | no | +0.0611 +/- 0.0623 (0/5) |
| `chorus_k2_eps2.0_beta1.0_[auth+dns]` | 2 | yes | +0.0683 +/- 0.1087 (0/5) |
| `chorus_k2_eps2.0_beta1.0_[auth+suricata]` | 2 | no | -0.0629 +/- 0.1073 (5/5) |
| `chorus_k2_eps2.0_beta1.0_[dns+audit]` | 2 | yes | +0.0672 +/- 0.1097 (0/5) |
| `chorus_k2_eps2.0_beta1.0_[dns+suricata]` | 2 | yes | -0.0072 +/- 0.0136 (3/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+audit+suricata]` | 3 | no | -0.0632 +/- 0.1076 (5/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+auth+audit]` | 3 | no | -0.0007 +/- 0.0033 (3/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+auth+dns]` | 3 | yes | -0.0685 +/- 0.1095 (4/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+auth+suricata]` | 3 | no | -0.0627 +/- 0.1069 (5/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+dns+audit]` | 3 | yes | -0.0723 +/- 0.1168 (4/5) |
| `chorus_k3_eps0.2_beta1.0_[apache+dns+suricata]` | 3 | yes | -0.0417 +/- 0.0945 (4/5) |
| `chorus_k3_eps0.2_beta1.0_[auth+audit+suricata]` | 3 | no | -0.0631 +/- 0.1072 (5/5) |
| `chorus_k3_eps0.2_beta1.0_[auth+dns+audit]` | 3 | yes | -0.0645 +/- 0.1157 (3/5) |
| `chorus_k3_eps0.2_beta1.0_[auth+dns+suricata]` | 3 | yes | -0.0421 +/- 0.0967 (3/5) |
| `chorus_k3_eps0.2_beta1.0_[dns+audit+suricata]` | 3 | yes | -0.0419 +/- 0.0949 (4/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+audit+suricata]` | 3 | no | -0.0632 +/- 0.1071 (5/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+auth+audit]` | 3 | no | +0.0059 +/- 0.0071 (1/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+auth+dns]` | 3 | yes | +0.0477 +/- 0.1080 (3/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+auth+suricata]` | 3 | no | -0.0631 +/- 0.1072 (5/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+dns+audit]` | 3 | yes | +0.0416 +/- 0.1072 (3/5) |
| `chorus_k3_eps0.5_beta1.0_[apache+dns+suricata]` | 3 | yes | -0.0332 +/- 0.0887 (2/5) |
| `chorus_k3_eps0.5_beta1.0_[auth+audit+suricata]` | 3 | no | -0.0634 +/- 0.1070 (5/5) |
| `chorus_k3_eps0.5_beta1.0_[auth+dns+audit]` | 3 | yes | +0.0519 +/- 0.1144 (1/5) |
| `chorus_k3_eps0.5_beta1.0_[auth+dns+suricata]` | 3 | yes | -0.0501 +/- 0.0964 (2/5) |
| `chorus_k3_eps0.5_beta1.0_[dns+audit+suricata]` | 3 | yes | -0.0416 +/- 0.0883 (2/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+audit+suricata]` | 3 | no | +0.0566 +/- 0.0660 (1/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+auth+audit]` | 3 | no | +0.0587 +/- 0.0601 (0/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+auth+dns]` | 3 | yes | -0.0456 +/- 0.0885 (4/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+auth+suricata]` | 3 | no | +0.0508 +/- 0.0684 (1/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+dns+audit]` | 3 | yes | -0.0203 +/- 0.0447 (4/5) |
| `chorus_k3_eps1.0_beta0.0_[apache+dns+suricata]` | 3 | yes | -0.1043 +/- 0.0731 (5/5) |
| `chorus_k3_eps1.0_beta0.0_[auth+audit+suricata]` | 3 | no | +0.0480 +/- 0.0670 (1/5) |
| `chorus_k3_eps1.0_beta0.0_[auth+dns+audit]` | 3 | yes | -0.0293 +/- 0.0735 (4/5) |
| `chorus_k3_eps1.0_beta0.0_[auth+dns+suricata]` | 3 | yes | -0.0980 +/- 0.0965 (5/5) |
| `chorus_k3_eps1.0_beta0.0_[dns+audit+suricata]` | 3 | yes | -0.0826 +/- 0.0831 (5/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+audit+suricata]` | 3 | no | -0.0629 +/- 0.1073 (5/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+auth+audit]` | 3 | no | +0.0631 +/- 0.0648 (0/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+auth+dns]` | 3 | yes | +0.0659 +/- 0.1042 (0/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+auth+suricata]` | 3 | no | -0.0627 +/- 0.1074 (5/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+dns+audit]` | 3 | yes | +0.0655 +/- 0.1048 (0/5) |
| `chorus_k3_eps1.0_beta1.0_[apache+dns+suricata]` | 3 | yes | +0.0166 +/- 0.0319 (1/5) |
| `chorus_k3_eps1.0_beta1.0_[auth+audit+suricata]` | 3 | no | -0.0631 +/- 0.1072 (5/5) |
| `chorus_k3_eps1.0_beta1.0_[auth+dns+audit]` | 3 | yes | +0.0659 +/- 0.1080 (0/5) |
| `chorus_k3_eps1.0_beta1.0_[auth+dns+suricata]` | 3 | yes | -0.0009 +/- 0.0016 (3/5) |
| `chorus_k3_eps1.0_beta1.0_[dns+audit+suricata]` | 3 | yes | +0.0028 +/- 0.0077 (2/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+audit+suricata]` | 3 | no | -0.0629 +/- 0.1073 (5/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+auth+audit]` | 3 | no | +0.0716 +/- 0.0696 (0/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+auth+dns]` | 3 | yes | +0.0592 +/- 0.1043 (1/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+auth+suricata]` | 3 | no | -0.0631 +/- 0.1072 (5/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+dns+audit]` | 3 | yes | +0.0596 +/- 0.1049 (1/5) |
| `chorus_k3_eps1.0_beta5.0_[apache+dns+suricata]` | 3 | yes | -0.1586 +/- 0.2657 (4/5) |
| `chorus_k3_eps1.0_beta5.0_[auth+audit+suricata]` | 3 | no | -0.0632 +/- 0.1071 (5/5) |
| `chorus_k3_eps1.0_beta5.0_[auth+dns+audit]` | 3 | yes | +0.0648 +/- 0.1075 (0/5) |
| `chorus_k3_eps1.0_beta5.0_[auth+dns+suricata]` | 3 | yes | +0.0638 +/- 0.0421 (0/5) |
| `chorus_k3_eps1.0_beta5.0_[dns+audit+suricata]` | 3 | yes | +0.0611 +/- 0.0614 (1/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+audit+suricata]` | 3 | no | -0.0629 +/- 0.1073 (5/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+auth+audit]` | 3 | no | +0.1371 +/- 0.1204 (0/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+auth+dns]` | 3 | yes | +0.0667 +/- 0.1078 (0/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+auth+suricata]` | 3 | no | -0.0627 +/- 0.1074 (5/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+dns+audit]` | 3 | yes | +0.0657 +/- 0.1055 (0/5) |
| `chorus_k3_eps2.0_beta1.0_[apache+dns+suricata]` | 3 | yes | +0.0962 +/- 0.0570 (0/5) |
| `chorus_k3_eps2.0_beta1.0_[auth+audit+suricata]` | 3 | no | -0.0632 +/- 0.1071 (5/5) |
| `chorus_k3_eps2.0_beta1.0_[auth+dns+audit]` | 3 | yes | +0.0666 +/- 0.1086 (0/5) |
| `chorus_k3_eps2.0_beta1.0_[auth+dns+suricata]` | 3 | yes | +0.0147 +/- 0.0343 (2/5) |
| `chorus_k3_eps2.0_beta1.0_[dns+audit+suricata]` | 3 | yes | +0.0121 +/- 0.0289 (1/5) |
| `pgd_k1_eps1.0_[apache]` | 1 | no | +0.0154 +/- 0.0316 (1/5) |
| `pgd_k1_eps1.0_[audit]` | 1 | no | +0.0168 +/- 0.0235 (0/5) |
| `pgd_k1_eps1.0_[auth]` | 1 | no | +0.0058 +/- 0.0023 (0/5) |
| `pgd_k1_eps1.0_[dns]` | 1 | yes | -0.0639 +/- 0.1086 (5/5) |
| `pgd_k1_eps1.0_[suricata]` | 1 | no | -0.1569 +/- 0.2406 (4/5) |
| `pgd_k2_eps1.0_[apache+audit]` | 2 | no | +0.0423 +/- 0.0616 (0/5) |
| `pgd_k2_eps1.0_[apache+auth]` | 2 | no | +0.0276 +/- 0.0368 (1/5) |
| `pgd_k2_eps1.0_[apache+dns]` | 2 | yes | -0.0543 +/- 0.0904 (5/5) |
| `pgd_k2_eps1.0_[apache+suricata]` | 2 | no | -0.0072 +/- 0.1141 (4/5) |
| `pgd_k2_eps1.0_[audit+suricata]` | 2 | no | -0.0182 +/- 0.1183 (4/5) |
| `pgd_k2_eps1.0_[auth+audit]` | 2 | no | +0.0353 +/- 0.0337 (0/5) |
| `pgd_k2_eps1.0_[auth+dns]` | 2 | yes | -0.0606 +/- 0.1075 (5/5) |
| `pgd_k2_eps1.0_[auth+suricata]` | 2 | no | -0.0253 +/- 0.1343 (4/5) |
| `pgd_k2_eps1.0_[dns+audit]` | 2 | yes | -0.0417 +/- 0.0698 (5/5) |
| `pgd_k2_eps1.0_[dns+suricata]` | 2 | yes | -0.2201 +/- 0.1926 (5/5) |
| `pgd_k2_eps2.0_[apache+audit]` | 2 | no | +0.0999 +/- 0.1089 (0/5) |
| `pgd_k2_eps2.0_[apache+auth]` | 2 | no | +0.0603 +/- 0.0628 (0/5) |
| `pgd_k2_eps2.0_[apache+dns]` | 2 | yes | -0.0341 +/- 0.0564 (5/5) |
| `pgd_k2_eps2.0_[apache+suricata]` | 2 | no | +0.0461 +/- 0.0989 (2/5) |
| `pgd_k2_eps2.0_[audit+suricata]` | 2 | no | +0.0283 +/- 0.0633 (2/5) |
| `pgd_k2_eps2.0_[auth+audit]` | 2 | no | +0.0774 +/- 0.0830 (0/5) |
| `pgd_k2_eps2.0_[auth+dns]` | 2 | yes | -0.0606 +/- 0.1119 (5/5) |
| `pgd_k2_eps2.0_[auth+suricata]` | 2 | no | +0.0335 +/- 0.0795 (3/5) |
| `pgd_k2_eps2.0_[dns+audit]` | 2 | yes | -0.0203 +/- 0.0282 (4/5) |
| `pgd_k2_eps2.0_[dns+suricata]` | 2 | yes | -0.0248 +/- 0.0146 (5/5) |
| `pgd_k3_eps1.0_[apache+audit+suricata]` | 3 | no | +0.0541 +/- 0.0663 (1/5) |
| `pgd_k3_eps1.0_[apache+auth+audit]` | 3 | no | +0.0583 +/- 0.0598 (0/5) |
| `pgd_k3_eps1.0_[apache+auth+dns]` | 3 | yes | -0.0454 +/- 0.0872 (4/5) |
| `pgd_k3_eps1.0_[apache+auth+suricata]` | 3 | no | +0.0438 +/- 0.0773 (1/5) |
| `pgd_k3_eps1.0_[apache+dns+audit]` | 3 | yes | -0.0211 +/- 0.0412 (4/5) |
| `pgd_k3_eps1.0_[apache+dns+suricata]` | 3 | yes | -0.0975 +/- 0.0806 (5/5) |
| `pgd_k3_eps1.0_[auth+audit+suricata]` | 3 | no | +0.0388 +/- 0.0755 (1/5) |
| `pgd_k3_eps1.0_[auth+dns+audit]` | 3 | yes | -0.0281 +/- 0.0717 (4/5) |
| `pgd_k3_eps1.0_[auth+dns+suricata]` | 3 | yes | -0.0894 +/- 0.0884 (5/5) |
| `pgd_k3_eps1.0_[dns+audit+suricata]` | 3 | yes | -0.0831 +/- 0.0857 (5/5) |
| `pgd_k3_eps2.0_[apache+audit+suricata]` | 3 | no | +0.0416 +/- 0.0695 (0/5) |
| `pgd_k3_eps2.0_[apache+auth+audit]` | 3 | no | +0.1401 +/- 0.1263 (0/5) |
| `pgd_k3_eps2.0_[apache+auth+dns]` | 3 | yes | -0.0122 +/- 0.0773 (4/5) |
| `pgd_k3_eps2.0_[apache+auth+suricata]` | 3 | no | +0.0468 +/- 0.0862 (0/5) |
| `pgd_k3_eps2.0_[apache+dns+audit]` | 3 | yes | +0.0403 +/- 0.0700 (2/5) |
| `pgd_k3_eps2.0_[apache+dns+suricata]` | 3 | yes | -0.0145 +/- 0.0129 (5/5) |
| `pgd_k3_eps2.0_[auth+audit+suricata]` | 3 | no | +0.0302 +/- 0.0511 (0/5) |
| `pgd_k3_eps2.0_[auth+dns+audit]` | 3 | yes | +0.0148 +/- 0.0710 (4/5) |
| `pgd_k3_eps2.0_[auth+dns+suricata]` | 3 | yes | -0.0129 +/- 0.0139 (4/5) |
| `pgd_k3_eps2.0_[dns+audit+suricata]` | 3 | yes | -0.0124 +/- 0.0063 (5/5) |

## Test C: honest duplication

### Strict same-sensor pairs: NO EVIDENCE

sensor_groups() returns {'network': [4]} -- one member, suricata -- so no within-group PAIR exists and the strict same-sensor comparison has no value to report. There is no packet-capture view yet (docs/part27-ait-datacard.md section 9, limitation 9). Nothing was substituted under this name; see natural_dependence for a DIFFERENT and separately labelled quantity.

### Additionally, a different quantity: "different sensors, naturally dependent"

Pairs read from `results/ait_clean/results.json`, not re-derived. Most dependent: `auth-audit`, `apache-suricata`. Least dependent: `auth-dns`, `auth-suricata`. Both systems agree on the ordering: **True**.

These views sit on DIFFERENT physical sensors. The dependence is a property of the scenario, not of shared hardware. This is NOT a same-sensor result and must not be read as one.

| system | AUC most-dependent over least-dependent (clean data) |
|---|---|
| naive | not measured |
| prismflow | 0.3528 +/- 0.0949 (n=5) |

AUC that a most-dependent pair's per-sample excess exceeds a least-dependent pair's, on CLEAN data. Pair-against-pair, NOT condition-against-clean, so it is not comparable in scale with Part 10's clone_k2 AUC of 0.8743 +/- 0.0583.

Per-pair mean unexplained agreement on clean data, PrismFlow, seed-0 row shown with its committed Part 28 dependence beside it:

| pair | committed dependence (Part 28) | mean excess | measurable samples |
|---|---|---|---|
| `auth-audit` | 0.8836 +/- 0.0169 | -0.0000 | 586 / 1,145 |
| `apache-suricata` | 0.6575 +/- 0.0353 | -0.5556 | 934 / 1,145 |
| `auth-dns` | 0.0112 +/- 0.0237 | 0.0023 | 601 / 1,145 |
| `auth-suricata` | 0.0950 +/- 0.0409 | -0.1482 | 601 / 1,145 |

## Test D: extra missingness, on top of the natural rate

`auth` is already absent in 47.1-47.5% of windows and `apache` in 18.0-27.5% (`results/ait_clean/results.json`, `directions.<name>.missingness`), so these rates are ADDITIONAL, not total.

| system | condition | detector ROC AUC | detection @5% | detection @1% | balanced accuracy |
|---|---|---|---|---|---|
| naive | +30% masked | not measured | not measured | not measured | 0.8337 +/- 0.0383 (n=5) |
| naive | +50% masked | not measured | not measured | not measured | 0.7362 +/- 0.0128 (n=5) |
| prismflow | +30% masked | 0.3166 +/- 0.0689 (n=5) | 0.0289 +/- 0.0176 (n=5) | 0.0056 +/- 0.0077 (n=5) | 0.8463 +/- 0.0203 (n=5) |
| prismflow | +50% masked | 0.2059 +/- 0.0473 (n=5) | 0.0287 +/- 0.0250 (n=5) | 0.0041 +/- 0.0063 (n=5) | 0.7422 +/- 0.0220 (n=5) |

Clean reference, same seeds:

| system | detector ROC AUC | balanced accuracy | ENIV (eigen) |
|---|---|---|---|
| naive | not measured | 0.9500 +/- 0.0685 (n=5) | 2.9579 +/- 0.0638 (n=5) |
| prismflow | 0.5000 +/- 0.0000 (n=5) | 0.9850 +/- 0.0105 (n=5) | 3.0272 +/- 0.0376 (n=5) |

