# Part 28 addendum - non-fusion learned baselines

Produced by `experiments/ait_clean/run_ait_baselines.py`, commit `886dda3`, scikit-learn 1.9.1.

## What the label means

A window is attack=1 if ANY officially labelled line falls in it, and on both scenarios that label is dominated by one continuously-running DNS exfiltration tool (1,026 of 1,031 attack windows on russellmitchell and 663 of 667 on santos carry only dnsteal-family steps, docs/part26-ait-loader.md sections 0 and 12), so the task is 'was the exfiltration tool running in this window', NOT 'detect an APT' and not 'detect a multi-step intrusion'.

This governs these numbers exactly as it governs those in `results/ait_clean/summary.md`. A single view scoring well here says that view sees the exfiltration tool; it does not say that view detects intrusions.

## What was fitted

- `flat_all_views`: one logistic regression on every view's native features concatenated, plus the five `view_mask` indicators. Missing views are zero-filled, and the indicators are what distinguish a zero-filled row from an observed zero.
- `single_view_<name>`: one logistic regression per view, on that view's native features and its own mask indicator.

Solver `lbfgs`, `max_iter=5000`, `C=1.0`, `class_weight=None` - fixed and stated, not tuned. The same windows (`label_known` only), the same scaler policy (fitted on the training scenario alone, refitted per direction) and the same metric functions as `run_ait_clean.py`.

## Determinism

Logistic regression with the lbfgs solver is a deterministic function of its inputs: no initialisation draw, no minibatch order, so a seed has nothing to select and five runs return five identical numbers. Each baseline row is therefore ONE value with no +/-. Reporting +/- 0.0000 over five identical runs would fake the 5-SEED RULE rather than satisfy it. Determinism is checked, not assumed: every model is refitted and its coefficients and predicted probabilities asserted bit-identical. The naive and PrismFlow reference rows DO carry a +/-, because those are 5-seed means over model initialisation and batch order; the two kinds of row are not the same kind of estimate.

## Two directions, reported separately and never pooled

Two directions are two measurements on two test sets with different prevalences, so a mean over them would be meaningless. There is no pooled table here, no averaged number across A and B, and no concatenated test set.

### Direction A: train russellmitchell -> test santos

**Train** `russellmitchell`, 1145 label_known windows, prevalence 0.900. **Test** `santos`, n = 1108 label_known windows, 667 attack, 441 benign, **attack prevalence 0.602**, 3 independent contiguous attack block(s). The scaler is fitted on `russellmitchell` only.

| model | test prevalence | ROC AUC | balanced accuracy | macro-F1 | attack recall | benign recall | Brier | ECE (15 bins) |
|---|---|---|---|---|---|---|---|---|
| `flat_all_views` | 0.602 (n=1108, 667a/441b) | 0.9995 | 0.9804 | 0.9830 | 0.9970 | 0.9637 | 0.0295 | 0.0102 |
| `single_view_apache` | 0.602 (n=1108, 667a/441b) | 0.4258 | 0.5000 | 0.3796 | 0.9955 | 0.0045 | 0.6673 | 0.2996 |
| `single_view_auth` | 0.602 (n=1108, 667a/441b) | 0.5103 | 0.5000 | 0.3758 | 1.0000 | 0.0000 | 0.6562 | 0.2980 |
| `single_view_dns` | 0.602 (n=1108, 667a/441b) | 0.9993 | 0.9898 | 0.9906 | 0.9955 | 0.9841 | 0.0208 | 0.0130 |
| `single_view_audit` | 0.602 (n=1108, 667a/441b) | 0.5584 | 0.5000 | 0.3758 | 1.0000 | 0.0000 | 0.6433 | 0.2899 |
| `single_view_suricata` | 0.602 (n=1108, 667a/441b) | 0.7369 | 0.7061 | 0.7123 | 0.9610 | 0.4512 | 0.3879 | 0.1293 |
| *naive fusion* (5 seeds) | 0.602 (n=1108, 667a/441b) | 0.9965 +/- 0.0008 | 0.9444 +/- 0.0056 | 0.9525 +/- 0.0049 | 0.9985 +/- 0.0000 | 0.8902 +/- 0.0113 | 0.0819 +/- 0.0092 | 0.0314 +/- 0.0070 |
| *V1 PrismFlow* (5 seeds) | 0.602 (n=1108, 667a/441b) | 0.9962 +/- 0.0005 | 0.9487 +/- 0.0061 | 0.9562 +/- 0.0053 | 0.9985 +/- 0.0000 | 0.8989 +/- 0.0123 | 0.0707 +/- 0.0071 | 0.0194 +/- 0.0042 |
| *majority-class baseline* | 0.602 (n=1108, 667a/441b) | 0.5000 | 0.5000 | 0.3758 | 1.0000 | 0.0000 | 0.4792 | 0.0000 |

Rows in `code font` are this addendum's deterministic baselines: one value, no `+/-`. Rows in *italics* are quoted from `results/ait_clean/results.json` for reference and are 5-seed means with a sample std, except the majority-class baseline, which is deterministic on a fixed test set.

### Direction B: train santos -> test russellmitchell

**Train** `santos`, 1108 label_known windows, prevalence 0.602. **Test** `russellmitchell`, n = 1145 label_known windows, 1031 attack, 114 benign, **attack prevalence 0.900**, 1 independent contiguous attack block(s). The scaler is fitted on `santos` only.

| model | test prevalence | ROC AUC | balanced accuracy | macro-F1 | attack recall | benign recall | Brier | ECE (15 bins) |
|---|---|---|---|---|---|---|---|---|
| `flat_all_views` | 0.900 (n=1145, 1031a/114b) | 1.0000 | 0.9588 | 0.8427 | 0.9176 | 1.0000 | 0.1264 | 0.0765 |
| `single_view_apache` | 0.900 (n=1145, 1031a/114b) | 0.3425 | 0.3733 | 0.3879 | 0.6149 | 0.1316 | 0.5281 | 0.2120 |
| `single_view_auth` | 0.900 (n=1145, 1031a/114b) | 0.5008 | 0.4995 | 0.4796 | 0.9903 | 0.0088 | 0.3550 | 0.2860 |
| `single_view_dns` | 0.900 (n=1145, 1031a/114b) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0085 | 0.0376 |
| `single_view_audit` | 0.900 (n=1145, 1031a/114b) | 0.5674 | 0.5088 | 0.4961 | 0.9913 | 0.0263 | 0.3342 | 0.2705 |
| `single_view_suricata` | 0.900 (n=1145, 1031a/114b) | 0.5649 | 0.5115 | 0.1619 | 0.0757 | 0.9474 | 1.4585 | 0.7483 |
| *naive fusion* (5 seeds) | 0.900 (n=1145, 1031a/114b) | 0.9999 +/- 0.0002 | 0.9500 +/- 0.0685 | 0.8511 +/- 0.1460 | 0.9001 +/- 0.1369 | 1.0000 +/- 0.0000 | 0.2237 +/- 0.0768 | 0.2066 +/- 0.0602 |
| *V1 PrismFlow* (5 seeds) | 0.900 (n=1145, 1031a/114b) | 0.9999 +/- 0.0002 | 0.9648 +/- 0.0455 | 0.8828 +/- 0.1181 | 0.9296 +/- 0.0909 | 1.0000 +/- 0.0000 | 0.1884 +/- 0.0688 | 0.1921 +/- 0.0478 |
| *majority-class baseline* | 0.900 (n=1145, 1031a/114b) | 0.5000 | 0.5000 | 0.4738 | 1.0000 | 0.0000 | 0.1793 | 0.0000 |

Rows in `code font` are this addendum's deterministic baselines: one value, no `+/-`. Rows in *italics* are quoted from `results/ait_clean/results.json` for reference and are 5-seed means with a sample std, except the majority-class baseline, which is deterministic on a fixed test set.

## Files

- `results/ait_clean/baselines.json` - every number above, both directions, with each direction's prevalence block and the per-model feature list.
- `results/ait_clean/baselines.md` - this file.

`results/ait_clean/results.json` and `results/ait_clean/summary.md` were not modified by this addendum; the reference rows are read from the former.

