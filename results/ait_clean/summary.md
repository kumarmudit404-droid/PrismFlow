# Part 28 - PrismFlow on real data (clean)

Produced by `experiments/ait_clean/run_ait_clean.py` with `experiments/ait_clean/config.yaml`, commit `acd4466`.

## What the label means

A window is attack=1 if ANY officially labelled line falls in it, and on both scenarios that label is dominated by one continuously-running DNS exfiltration tool (1,026 of 1,031 attack windows on russellmitchell and 663 of 667 on santos carry only dnsteal-family steps, docs/part26-ait-loader.md sections 0 and 12), so the task is 'was the exfiltration tool running in this window', NOT 'detect an APT' and not 'detect a multi-step intrusion'.

The multi-step intrusion is 3 window(s) on `santos` (4 counting the exfiltration tool's own service record), 4 window(s) on `russellmitchell` (5 counting the exfiltration tool's own service record). No metric in this file is computed over those windows; any claim about them is a handful-of-windows result. The two counts per scenario are the two definitions the source documents use; see `EXFILTRATION_TOOL_STEPS` in `run_ait_clean.py`.

## Two directions, reported separately and never pooled

Neither direction is "the" result. Two directions are two measurements on two different test sets with different prevalences, so a mean over them would be meaningless and would hide whichever one is worse. There is no pooled table in this file, no averaged number across A and B, and no concatenated test set.

## One summary table per direction

### Direction A: train russellmitchell -> test santos

**Train** `russellmitchell`, 1145 label_known windows (1031 attack, 114 benign, prevalence 0.900). **Test** `santos`, 1108 label_known windows (667 attack, 441 benign, prevalence 0.602). Train and test are different scenarios; the scaler is fitted on `russellmitchell` only.

**Independent contiguous attack blocks in this test set: 3** (lengths 664/2/1 windows). The block count, not the attack-window count, is the positive class's effective sample size. Every +/- in this Part is a spread over seeds (model initialisation and batch order), NOT over independent attacks.

**What the label means.** A window is attack=1 if ANY officially labelled line falls in it, and on both scenarios that label is dominated by one continuously-running DNS exfiltration tool (1,026 of 1,031 attack windows on russellmitchell and 663 of 667 on santos carry only dnsteal-family steps, docs/part26-ait-loader.md sections 0 and 12), so the task is 'was the exfiltration tool running in this window', NOT 'detect an APT' and not 'detect a multi-step intrusion'.

| metric | naive fusion | V1 PrismFlow | majority-class baseline |
|---|---|---|---|
| balanced accuracy | 0.9444 +/- 0.0056 | 0.9487 +/- 0.0061 | 0.5000 (1 seed) |
| ROC AUC | 0.9965 +/- 0.0008 | 0.9962 +/- 0.0005 | 0.5000 (1 seed) |
| macro-F1 | 0.9525 +/- 0.0049 | 0.9562 +/- 0.0053 | 0.3758 (1 seed) |
| attack-class recall | 0.9985 +/- 0.0000 | 0.9985 +/- 0.0000 | 1.0000 (1 seed) |
| benign-class recall | 0.8902 +/- 0.0113 | 0.8989 +/- 0.0123 | 0.0000 (1 seed) |
| Brier | 0.0819 +/- 0.0092 | 0.0707 +/- 0.0071 | 0.4792 (1 seed) |
| ECE (15 bins, prob_) | 0.0314 +/- 0.0070 | 0.0194 +/- 0.0042 | 0.0000 (1 seed) |
| AURC (prob_) | 0.0099 +/- 0.0022 | 0.0089 +/- 0.0010 | 0.3980 (1 seed) |
| accuracy (prevalence-bound, never a headline) | 0.9554 +/- 0.0045 | 0.9588 +/- 0.0049 | 0.6020 (1 seed) |
| ENIV (eigen, headline) | 3.4507 +/- 0.0790 | 3.4958 +/- 0.1015 | n/a (no model) |
| ENIV (design effect) | 2.3991 +/- 0.0734 | 2.4811 +/- 0.1234 | n/a (no model) |
| mean dependence | 0.2714 +/- 0.0154 | 0.2548 +/- 0.0244 | n/a (no model) |
| efficiency ratio | 0.6901 +/- 0.0158 | 0.6992 +/- 0.0203 | n/a (no model) |

5 seeds 0-4, mean +/- sample std (ddof = 1). The baseline is deterministic on a fixed test set, so it has one value and no std. Caption: test set `santos` at 300 s / network: n = 1108 label_known windows, 667 attack, 441 benign, attack prevalence 0.602; 3 independent contiguous attack block(s).

### Direction B: train santos -> test russellmitchell

**Train** `santos`, 1108 label_known windows (667 attack, 441 benign, prevalence 0.602). **Test** `russellmitchell`, 1145 label_known windows (1031 attack, 114 benign, prevalence 0.900). Train and test are different scenarios; the scaler is fitted on `santos` only.

**Independent contiguous attack blocks in this test set: 1** (lengths 1031 windows). The block count, not the attack-window count, is the positive class's effective sample size. Every +/- in this Part is a spread over seeds (model initialisation and batch order), NOT over independent attacks.

**What the label means.** A window is attack=1 if ANY officially labelled line falls in it, and on both scenarios that label is dominated by one continuously-running DNS exfiltration tool (1,026 of 1,031 attack windows on russellmitchell and 663 of 667 on santos carry only dnsteal-family steps, docs/part26-ait-loader.md sections 0 and 12), so the task is 'was the exfiltration tool running in this window', NOT 'detect an APT' and not 'detect a multi-step intrusion'.

| metric | naive fusion | V1 PrismFlow | majority-class baseline |
|---|---|---|---|
| balanced accuracy | 0.9500 +/- 0.0685 | 0.9648 +/- 0.0455 | 0.5000 (1 seed) |
| ROC AUC | 0.9999 +/- 0.0002 | 0.9999 +/- 0.0002 | 0.5000 (1 seed) |
| macro-F1 | 0.8511 +/- 0.1460 | 0.8828 +/- 0.1181 | 0.4738 (1 seed) |
| attack-class recall | 0.9001 +/- 0.1369 | 0.9296 +/- 0.0909 | 1.0000 (1 seed) |
| benign-class recall | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 (1 seed) |
| Brier | 0.2237 +/- 0.0768 | 0.1884 +/- 0.0688 | 0.1793 (1 seed) |
| ECE (15 bins, prob_) | 0.2066 +/- 0.0602 | 0.1921 +/- 0.0478 | 0.0000 (1 seed) |
| AURC (prob_) | 0.0363 +/- 0.0499 | 0.0246 +/- 0.0290 | 0.0996 (1 seed) |
| accuracy (prevalence-bound, never a headline) | 0.9100 +/- 0.1233 | 0.9366 +/- 0.0819 | 0.9004 (1 seed) |
| ENIV (eigen, headline) | 2.9579 +/- 0.0638 | 3.0272 +/- 0.0376 | n/a (no model) |
| ENIV (design effect) | 1.9982 +/- 0.0686 | 2.0923 +/- 0.0362 | n/a (no model) |
| mean dependence | 0.3761 +/- 0.0219 | 0.3476 +/- 0.0105 | n/a (no model) |
| efficiency ratio | 0.5916 +/- 0.0128 | 0.6054 +/- 0.0075 | n/a (no model) |

5 seeds 0-4, mean +/- sample std (ddof = 1). The baseline is deterministic on a fixed test set, so it has one value and no std. Caption: test set `russellmitchell` at 300 s / network: n = 1145 label_known windows, 1031 attack, 114 benign, attack prevalence 0.900; 1 independent contiguous attack block(s).

## Paired within-seed comparison, per direction

prismflow - naive computed WITHIN each seed, so the shared initialisation-and-batch-order variance cancels. The seed count on one side of zero is given beside every mean: a mean near zero with 5/5 seeds on one side is a different statement from a mean near zero with 2/5. Not pooled across directions.

A: train russellmitchell -> test santos (test set `santos` at 300 s / network: n = 1108 label_known windows, 667 attack, 441 benign, attack prevalence 0.602; 3 independent contiguous attack block(s)):

| metric | prismflow - naive | seeds favouring prismflow | better when |
|---|---|---|---|
| balanced accuracy | 0.0043 +/- 0.0015 | 5/5 | higher |
| ROC AUC | -0.0003 +/- 0.0005 | 1/5 | higher |
| macro-F1 | 0.0037 +/- 0.0013 | 5/5 | higher |
| attack-class recall | 0.0000 +/- 0.0000 | 0/5 | higher |
| benign-class recall | 0.0086 +/- 0.0030 | 5/5 | higher |
| Brier | -0.0111 +/- 0.0043 | 5/5 | lower |
| ECE (15 bins, prob_) | -0.0120 +/- 0.0066 | 5/5 | lower |
| AURC (prob_) | -0.0010 +/- 0.0015 | 4/5 | lower |
| accuracy (prevalence-bound, never a headline) | 0.0034 +/- 0.0012 | 5/5 | higher |
| ENIV (eigen, headline) | 0.0451 +/- 0.0350 | 5/5 | higher |
| ENIV (design effect) | 0.0821 +/- 0.0709 | 4/5 | higher |
| mean dependence | -0.0166 +/- 0.0144 | 4/5 | lower |
| efficiency ratio | 0.0090 +/- 0.0070 | 5/5 | higher |

B: train santos -> test russellmitchell (test set `russellmitchell` at 300 s / network: n = 1145 label_known windows, 1031 attack, 114 benign, attack prevalence 0.900; 1 independent contiguous attack block(s)):

| metric | prismflow - naive | seeds favouring prismflow | better when |
|---|---|---|---|
| balanced accuracy | 0.0147 +/- 0.0245 | 3/5 | higher |
| ROC AUC | 0.0000 +/- 0.0000 | 3/5 | higher |
| macro-F1 | 0.0317 +/- 0.0428 | 3/5 | higher |
| attack-class recall | 0.0295 +/- 0.0489 | 3/5 | higher |
| benign-class recall | 0.0000 +/- 0.0000 | 0/5 | higher |
| Brier | -0.0353 +/- 0.0348 | 4/5 | lower |
| ECE (15 bins, prob_) | -0.0144 +/- 0.0154 | 4/5 | lower |
| AURC (prob_) | -0.0118 +/- 0.0217 | 4/5 | lower |
| accuracy (prevalence-bound, never a headline) | 0.0266 +/- 0.0440 | 3/5 | higher |
| ENIV (eigen, headline) | 0.0693 +/- 0.0417 | 5/5 | higher |
| ENIV (design effect) | 0.0940 +/- 0.0388 | 5/5 | higher |
| mean dependence | -0.0286 +/- 0.0129 | 5/5 | lower |
| efficiency ratio | 0.0139 +/- 0.0083 | 5/5 | higher |

## Test-set structure, per direction

| direction | test scenario | n (label_known) | attack | benign | prevalence | attack blocks | attack block lengths | benign blocks | benign block lengths |
|---|---|---|---|---|---|---|---|---|---|
| A_rm_to_santos | `santos` | 1108 | 667 | 441 | 0.602 | **3** | 664/2/1 | 37 | 168/7/7/1/1/1/1/3/... |
| B_santos_to_rm | `russellmitchell` | 1145 | 1031 | 114 | 0.900 | **1** | 1031 | 8 | 99/1/1/5/3/3/1/1 |

The block count, not the attack-window count, is the positive class's effective sample size. Every +/- in this Part is a spread over seeds (model initialisation and batch order), NOT over independent attacks.

Attack block start times:

- A_rm_to_santos (`santos`): 2022-01-14T00:00:00+00:00, 664 windows
- A_rm_to_santos (`santos`): 2022-01-17T11:15:00+00:00, 2 windows
- A_rm_to_santos (`santos`): 2022-01-17T11:55:00+00:00, 1 windows
- B_santos_to_rm (`russellmitchell`): 2022-01-21T00:00:00+00:00, 1031 windows

Measured at 300 s / network from each scenario's own `labels.npy` and `window_start.npy`; Part 26's figures are not carried over. The counts reproduce `results/ait_window/window_table.md` section 2 for the same rows.

## Per-view missing fraction, per direction (train and test)

| direction | set | scenario | `apache` | `auth` | `dns` | `audit` | `suricata` |
|---|---|---|---|---|---|---|---|
| A_rm_to_santos | train | `russellmitchell` | 0.1799 | 0.4751 | 0.0000 | 0.2812 | 0.0000 |
| A_rm_to_santos | test | `santos` | 0.2753 | 0.4711 | 0.0063 | 0.1895 | 0.0000 |
| B_santos_to_rm | train | `santos` | 0.2753 | 0.4711 | 0.0063 | 0.1895 | 0.0000 |
| B_santos_to_rm | test | `russellmitchell` | 0.1799 | 0.4751 | 0.0000 | 0.2812 | 0.0000 |

Threshold for the brief's step 5 is 50%. A_rm_to_santos: no view exceeds 50% missing on this direction's train or test set, so the brief's second configuration was not triggered; B_santos_to_rm: no view exceeds 50% missing on this direction's train or test set, so the brief's second configuration was not triggered.

## Dependence pairs, per direction

**Step 4 of the brief is NOT COMPUTABLE.** the 'network' same-sensor group has 1 member(s) (['suricata']), so no within-group PAIR exists. There is no packet-capture view yet (docs/part27-ait-datacard.md section 9, limitation 9), so the paired difference step 4 asks for has no value to report. Substituting a different pair would answer a different question; the full pair table is reported instead.

Full off-diagonal pair table, mean +/- sample std over 5 seeds, measured on each direction's test set with `REPORTING_NULL_PERMUTATIONS = 12`:

A_rm_to_santos / naive fusion (test set `santos` at 300 s / network: n = 1108 label_known windows, 667 attack, 441 benign, attack prevalence 0.602; 3 independent contiguous attack block(s)):

| pair | dependence |
|---|---|
| `apache-auth` | 0.1309 +/- 0.0589 |
| `apache-dns` | 0.0458 +/- 0.0220 |
| `apache-audit` | 0.1056 +/- 0.0616 |
| `apache-suricata` | 0.2077 +/- 0.0827 |
| `auth-dns` | 0.0411 +/- 0.0156 |
| `auth-audit` | 0.6460 +/- 0.0804 |
| `auth-suricata` | 0.1245 +/- 0.0111 |
| `dns-audit` | 0.1790 +/- 0.0635 |
| `dns-suricata` | 0.7784 +/- 0.0063 |
| `audit-suricata` | 0.4551 +/- 0.0084 |

A_rm_to_santos / V1 PrismFlow (test set `santos` at 300 s / network: n = 1108 label_known windows, 667 attack, 441 benign, attack prevalence 0.602; 3 independent contiguous attack block(s)):

| pair | dependence |
|---|---|
| `apache-auth` | 0.0985 +/- 0.0716 |
| `apache-dns` | 0.0451 +/- 0.0488 |
| `apache-audit` | 0.1142 +/- 0.0647 |
| `apache-suricata` | 0.2255 +/- 0.1089 |
| `auth-dns` | 0.0364 +/- 0.0105 |
| `auth-audit` | 0.5996 +/- 0.0806 |
| `auth-suricata` | 0.1060 +/- 0.0097 |
| `dns-audit` | 0.1301 +/- 0.0396 |
| `dns-suricata` | 0.7789 +/- 0.0064 |
| `audit-suricata` | 0.4135 +/- 0.0260 |

B_santos_to_rm / naive fusion (test set `russellmitchell` at 300 s / network: n = 1145 label_known windows, 1031 attack, 114 benign, attack prevalence 0.900; 1 independent contiguous attack block(s)):

| pair | dependence |
|---|---|
| `apache-auth` | 0.4335 +/- 0.0569 |
| `apache-dns` | 0.3606 +/- 0.0862 |
| `apache-audit` | 0.3145 +/- 0.0293 |
| `apache-suricata` | 0.6575 +/- 0.0353 |
| `auth-dns` | 0.0112 +/- 0.0237 |
| `auth-audit` | 0.8836 +/- 0.0169 |
| `auth-suricata` | 0.0950 +/- 0.0409 |
| `dns-audit` | 0.1389 +/- 0.0632 |
| `dns-suricata` | 0.5838 +/- 0.0695 |
| `audit-suricata` | 0.2828 +/- 0.0253 |

B_santos_to_rm / V1 PrismFlow (test set `russellmitchell` at 300 s / network: n = 1145 label_known windows, 1031 attack, 114 benign, attack prevalence 0.900; 1 independent contiguous attack block(s)):

| pair | dependence |
|---|---|
| `apache-auth` | 0.4163 +/- 0.0375 |
| `apache-dns` | 0.2612 +/- 0.0320 |
| `apache-audit` | 0.3099 +/- 0.0088 |
| `apache-suricata` | 0.6575 +/- 0.0189 |
| `auth-dns` | -0.0280 +/- 0.0190 |
| `auth-audit` | 0.8705 +/- 0.0154 |
| `auth-suricata` | 0.0697 +/- 0.0575 |
| `dns-audit` | 0.0849 +/- 0.0221 |
| `dns-suricata` | 0.5516 +/- 0.0674 |
| `audit-suricata` | 0.2822 +/- 0.0234 |

## Files

- `results/ait_clean/results.json` - every number above, both directions, 5 seeds per metric per direction, with each direction's prevalence block.
- `results/ait_clean/summary.md` - this file.
- `results/ait_clean/<direction>/<system>/metrics.json|csv` and `reliability_diagram.png` - the standard `prismflow.evaluation.evaluate()` output, per direction per system.

Every number in this file is a 5-seed mean +/- sample std except the majority-class baseline (deterministic on a fixed test set) and the structural counts (properties of the data, not of a run).

