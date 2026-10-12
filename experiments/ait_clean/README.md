# Part 28 - PrismFlow on real data (clean)

Naive fusion and V1 PrismFlow on the AIT Log Data Set v2.0, at **300 s network
windows** (the user's choice at the Part 27 review gate), with train and test
drawn from **different scenarios** in both directions. Clean data only: no
attack, no corruption, no clone.

Code: `experiments/ait_clean/run_ait_clean.py`,
`experiments/ait_clean/config.yaml`, `tests/unit/test_ait_clean.py` (41 tests).
Results: `results/ait_clean/results.json`, `results/ait_clean/summary.md`, and
`results/ait_clean/<direction>/<system>/metrics.{json,csv}` +
`reliability_diagram.png`.

---

## 0. What the label means

> A window is attack=1 if ANY officially labelled line falls in it, and on both
> scenarios that label is dominated by one continuously-running DNS
> exfiltration tool: 1,026 of 1,031 attack windows on `russellmitchell` and 663
> of 667 on `santos` carry only dnsteal-family steps
> (`docs/part26-ait-loader.md` sections 0 and 12). **So the task is "was the
> exfiltration tool running in this window", NOT "detect an APT" and not
> "detect a multi-step intrusion".**

A result from this Part described as intrusion, APT or campaign detection is a
misreport even when every number in it is correct. The sentence is repeated in
`results/ait_clean/summary.md`, in every direction record of
`results/ait_clean/results.json` (`directions.<name>.label_meaning` and
`.test_structure.label_meaning`), and
`tests/unit/test_ait_clean.py::test_label_meaning_is_in_every_direction_record_and_the_summary`
fails if it is dropped from any of them.

**The multi-step intrusion is a handful of windows.** Counted in
`results/ait_clean/results.json`,
`directions.<name>.test_structure.multi_step_intrusion_windows`:

| scenario | windows beyond the dnsteal family | of those, multi-step intrusion steps | window starts (UTC) |
|---|---|---|---|
| `russellmitchell` | 5 | **4** | 2022-01-24T03:00, 03:35, 03:55, 04:35 |
| `santos` | 4 | **3** | 2022-01-17T11:15, 11:20, 11:55 |

The two counts differ by exactly one window per scenario, the one carrying the
exfiltration tool's own `exfiltration-service` SERVICE_START / SERVICE_STOP
record. The brief quotes **5** for `russellmitchell` (the looser count, as in
`docs/part26-ait-loader.md` section 0) and **3** for `santos` (the stricter
count, as in section 12 and the data card). Both definitions are reported so
neither source document is contradicted; `EXFILTRATION_TOOL_STEPS` in
`run_ait_clean.py` records which is which. **No metric in this Part is
computed over these windows.** Any claim about them would be a
handful-of-windows result.

---

## 1. Both splits

Train and test always come from **different scenarios**. A random split
*within* a scenario is forbidden in both directions: neighbouring windows of
one scenario leak into each other, and the labels are a step function in time
(`docs/part27-ait-datacard.md` section 8 - a random split trains and tests on
adjacent windows of the same exfiltration run, and a chronological split on
`santos` is close to a label split).

| direction | train scenario | train n (label_known) | train attack / benign | train prevalence | test scenario | test n (label_known) | test attack / benign | **test prevalence** | **attack blocks** |
|---|---|---|---|---|---|---|---|---|---|
| **A** `A_rm_to_santos` | `russellmitchell` | 1,145 | 1,031 / 114 | 0.900 | `santos` | 1,108 | 667 / 441 | **0.602** | **3** (664/2/1) |
| **B** `B_santos_to_rm` | `santos` | 1,108 | 667 / 441 | 0.602 | `russellmitchell` | 1,145 | 1,031 / 114 | **0.900** | **1** (1,031) |

Source: `results/ait_clean/results.json`, `directions.<name>.n_train`,
`.n_train_attack`, `.n_train_benign`, `.train_attack_prevalence` and
`.test_structure.*`.

- **Only `label_known` rows are used anywhere**, in train and in test.
  `labels == -1` means "no label-covered file contributed an event to this
  window" and is never read as benign. 7 windows on `russellmitchell` and 44
  on `santos` are dropped for this reason
  (`test_structure.n_unlabelled`).
- **Feature scaling is fitted on the training scenario's `label_known` rows
  only**, refitted per direction, never shared
  (`directions.<name>.scaler.fitted_on`). The AIT loader deliberately does not
  z-score, stating that "standardisation has to be fitted on training indices
  only, and the split is Part 28's business"
  (`prismflow/data/ait_lds.py`, `AITMultiViewDataset` docstring), so this is
  the step that module left to here. It is fitted per (view, native feature)
  over rows where that view is **present**, so an absent view's zero row never
  pulls the mean toward zero, and the padded columns beyond a view's native
  width are left at 0.
- **Neither direction is "the" result.** Two directions are two measurements on
  two different test sets with different prevalences, so a mean over them is
  meaningless and would hide whichever one is worse. There is no pooled table,
  no averaged number across A and B, and no concatenated test set anywhere in
  this Part.

### What a seed varies, and what it does not

Train and test are whole scenarios, so a seed **cannot** redraw the data and
**cannot** re-partition it - Part 15's mechanism (`docs/DATASETS.md`, "what a
seed varies") does not apply here. A seed selects exactly two things: the
model's parameter initialisation (`set_seed`) and the minibatch order of every
epoch (`iter_batches(shuffle=True, seed=seed+epoch)`).

**Every `+/-` below is initialisation-and-batch-order variance on a fixed pair
of scenarios.** It is not dataset variance, and it is emphatically not a spread
over independent attacks: the test sets hold **3** and **1** independent
contiguous attack blocks. The block count, not the attack-window count, is the
positive class's effective sample size.

Two consecutive full runs of the committed code produced byte-identical metric
tables, so the numbers below are reproducible rather than merely recorded.

---

## 2. Direction A: train `russellmitchell` -> test `santos`

**Test set `santos` at 300 s / network: n = 1,108 `label_known` windows, 667
attack, 441 benign, attack prevalence 0.602, 3 independent contiguous attack
blocks (664/2/1 windows).** 5 seeds (0-4), mean +/- sample std (ddof = 1).
Source: `results/ait_clean/results.json`,
`directions.A_rm_to_santos.systems.<system>.summary` and
`.diagnostics_summary`; the baseline from
`directions.A_rm_to_santos.majority_class_baseline`.

| metric | naive fusion | V1 PrismFlow | majority-class baseline | paired (prismflow - naive) | seeds favouring |
|---|---|---|---|---|---|
| balanced accuracy | 0.9444 +/- 0.0056 | **0.9487 +/- 0.0061** | 0.5000 | +0.0043 +/- 0.0015 | **5/5** |
| ROC AUC | 0.9965 +/- 0.0008 | 0.9962 +/- 0.0005 | 0.5000 | -0.0003 +/- 0.0005 | 1/5 |
| macro-F1 | 0.9525 +/- 0.0049 | **0.9562 +/- 0.0053** | 0.3758 | +0.0037 +/- 0.0013 | **5/5** |
| attack-class recall | 0.9985 +/- 0.0000 | 0.9985 +/- 0.0000 | 1.0000 | +0.0000 +/- 0.0000 | 0/5 |
| benign-class recall | 0.8902 +/- 0.0113 | **0.8989 +/- 0.0123** | 0.0000 | +0.0086 +/- 0.0030 | **5/5** |
| Brier | 0.0819 +/- 0.0092 | **0.0707 +/- 0.0071** | 0.4792 | -0.0111 +/- 0.0043 | **5/5** |
| ECE (15 bins, `prob_`) | 0.0314 +/- 0.0070 | **0.0194 +/- 0.0042** | 0.0000 | -0.0120 +/- 0.0066 | **5/5** |
| AURC (`prob_`) | 0.0099 +/- 0.0022 | 0.0089 +/- 0.0010 | 0.3980 | -0.0010 +/- 0.0015 | 4/5 |
| accuracy *(prevalence-bound, never a headline)* | 0.9554 +/- 0.0045 | 0.9588 +/- 0.0049 | 0.6020 | +0.0034 +/- 0.0012 | 5/5 |
| ENIV (eigen, headline) | 3.4507 +/- 0.0790 | 3.4958 +/- 0.1015 | n/a (no model) | +0.0451 +/- 0.0350 | 5/5 |
| ENIV (design effect) | 2.3991 +/- 0.0734 | 2.4811 +/- 0.1234 | n/a | +0.0821 +/- 0.0709 | 4/5 |
| mean dependence | 0.2714 +/- 0.0154 | 0.2548 +/- 0.0244 | n/a | -0.0166 +/- 0.0144 | 4/5 |
| efficiency ratio | 0.6901 +/- 0.0158 | 0.6992 +/- 0.0203 | n/a | +0.0090 +/- 0.0070 | 5/5 |

**Reading it.** Both systems clear the majority-class baseline decisively on
every metric that admits one: balanced accuracy 0.94-0.95 against 0.50, ROC AUC
0.996 against 0.50, Brier 0.07-0.08 against 0.4792, AURC 0.009 against 0.3980.
The paired difference favours PrismFlow in 5/5 seeds on balanced accuracy,
macro-F1, benign recall, Brier and ECE, and the effects are small
(+0.0043 balanced accuracy, -0.0111 Brier, -0.0120 ECE). ROC AUC does not
separate (1/5 seeds), and attack recall is identical in all 10 runs
(0.9985 = 666 of 667) - the two systems differ only in how many of the 441
benign windows they get right.

---

## 3. Direction B: train `santos` -> test `russellmitchell`

**Test set `russellmitchell` at 300 s / network: n = 1,145 `label_known`
windows, 1,031 attack, 114 benign, attack prevalence 0.900, 1 independent
contiguous attack block (1,031 windows).** 5 seeds (0-4), mean +/- sample std
(ddof = 1). Source: `results/ait_clean/results.json`,
`directions.B_santos_to_rm.*`.

| metric | naive fusion | V1 PrismFlow | majority-class baseline | paired (prismflow - naive) | seeds favouring |
|---|---|---|---|---|---|
| balanced accuracy | 0.9500 +/- 0.0685 | 0.9648 +/- 0.0455 | 0.5000 | +0.0147 +/- 0.0245 | 3/5 |
| ROC AUC | 0.9999 +/- 0.0002 | 0.9999 +/- 0.0002 | 0.5000 | +0.0000 +/- 0.0000 | 3/5 |
| macro-F1 | 0.8511 +/- 0.1460 | 0.8828 +/- 0.1181 | 0.4738 | +0.0317 +/- 0.0428 | 3/5 |
| attack-class recall | 0.9001 +/- 0.1369 | 0.9296 +/- 0.0909 | 1.0000 | +0.0295 +/- 0.0489 | 3/5 |
| benign-class recall | 1.0000 +/- 0.0000 | 1.0000 +/- 0.0000 | 0.0000 | +0.0000 +/- 0.0000 | 0/5 |
| Brier | 0.2237 +/- 0.0768 | 0.1884 +/- 0.0688 | **0.1793** | -0.0353 +/- 0.0348 | 4/5 |
| ECE (15 bins, `prob_`) | 0.2066 +/- 0.0602 | 0.1921 +/- 0.0478 | **0.0000** | -0.0144 +/- 0.0154 | 4/5 |
| AURC (`prob_`) | 0.0363 +/- 0.0499 | 0.0246 +/- 0.0290 | 0.0996 | -0.0118 +/- 0.0217 | 4/5 |
| accuracy *(prevalence-bound, never a headline)* | 0.9100 +/- 0.1233 | 0.9366 +/- 0.0819 | **0.9004** | +0.0266 +/- 0.0440 | 3/5 |
| ENIV (eigen, headline) | 2.9579 +/- 0.0638 | 3.0272 +/- 0.0376 | n/a (no model) | +0.0693 +/- 0.0417 | **5/5** |
| ENIV (design effect) | 1.9982 +/- 0.0686 | 2.0923 +/- 0.0362 | n/a | +0.0940 +/- 0.0388 | **5/5** |
| mean dependence | 0.3761 +/- 0.0219 | 0.3476 +/- 0.0105 | n/a | -0.0286 +/- 0.0129 | **5/5** |
| efficiency ratio | 0.5916 +/- 0.0128 | 0.6054 +/- 0.0075 | n/a | +0.0139 +/- 0.0083 | **5/5** |

**Reading it. This is the direction that shows why accuracy was banned as a
headline.**

- **Accuracy does not clear the baseline here.** Naive fusion scores
  0.9100 +/- 0.1233 and PrismFlow 0.9366 +/- 0.0819 against a majority-class
  baseline of **0.9004**. Both error bars straddle the baseline. An accuracy
  figure on this test set describes the 90.0% attack prevalence, not the
  system.
- **Balanced accuracy and ROC AUC do clear it, decisively**: 0.95-0.96 and
  0.9999 against 0.50 for both. So the systems are not failing - they rank the
  windows nearly perfectly. What they get wrong is where the 0.5 decision
  threshold falls.
- **The threshold is the whole story.** ROC AUC is 0.9999 +/- 0.0002 while
  attack-class recall ranges from 0.6557 to 0.9942 across the 10 trained models
  (`results/ait_clean/results.json`, `per_seed[].attack_recall`; seed 3 is the
  outlier for both systems, at 0.6557 naive / 0.7692 PrismFlow). Benign recall
  is exactly 1.0000 in all 10. The models never produce a false positive and
  the ranking is essentially perfect; the variance is entirely in how many
  attack windows fall on the wrong side of a fixed probability cut.
- **The baseline beats both systems on Brier and ECE.** Baseline Brier 0.1793
  against 0.2237 (naive) and 0.1884 (PrismFlow); baseline ECE 0.0000 against
  0.2066 and 0.1921. Training at 60.2% prevalence and testing at 90.0% leaves
  both systems badly miscalibrated, and a constant forecast equal to the test
  prevalence is trivially better calibrated while being useless at ranking
  (its ROC AUC is 0.5 and its AURC 0.0996, against 0.025-0.036 for the
  systems). **Calibration and discrimination come apart here**, and reporting
  either alone would misdescribe the result.
- **The spreads are large and they are seed spreads.** `+/- 0.1233` on accuracy
  is initialisation-and-batch-order variance over 5 models on **one** attack
  block of 1,031 windows. It is not a confidence interval over attacks; the
  positive class's effective sample size on this test set is **1**.
- The paired comparison favours PrismFlow on Brier, ECE and AURC in 4/5 seeds
  and on all four ENIV diagnostics in 5/5, but on balanced accuracy, macro-F1
  and attack recall it is 3/5 with a std larger than the mean - **not a
  separation**, and it must not be reported as one.

---

## 4. Test-set structure, per direction

Measured at 300 s / network from each scenario's own `labels.npy` and
`window_start.npy`. Part 26's figures are not carried over; the counts
independently reproduce `results/ait_window/window_table.md` section 2 for the
same rows, which
`tests/unit/test_ait_clean.py::test_block_counts_reproduce_the_part27_window_table`
asserts. Source: `results/ait_clean/results.json`,
`directions.<name>.test_structure`.

| direction | test scenario | attack blocks | attack block lengths | attack block starts (UTC) | benign blocks | benign block lengths |
|---|---|---|---|---|---|---|
| A | `santos` (n = 1,108, prevalence 0.602) | **3** | 664 / 2 / 1 | 2022-01-14T00:00, 2022-01-17T11:15, 2022-01-17T11:55 | 37 | 168/7/7/1/1/1/1/3/... |
| B | `russellmitchell` (n = 1,145, prevalence 0.900) | **1** | 1,031 | 2022-01-21T00:00 | 8 | 99/1/1/5/3/3/1/1 |

**Consequence, which belongs beside every number in this Part:** the block
count, not the attack-window count, is the positive class's effective sample
size. Direction B's 1,031 attack windows are **one** episode of one tool.
Every `+/-` here is a spread over seeds, **not** over independent attacks.

---

## 5. Per-view missingness, per direction (train and test)

Measured on `label_known` rows. Source: `results/ait_clean/results.json`,
`directions.<name>.missingness`.

| direction | set | scenario | `apache` | `auth` | `dns` | `audit` | `suricata` |
|---|---|---|---|---|---|---|---|
| A | train | `russellmitchell` | 0.1799 | 0.4751 | 0.0000 | 0.2812 | 0.0000 |
| A | test | `santos` | 0.2753 | 0.4711 | 0.0063 | 0.1895 | 0.0000 |
| B | train | `santos` | 0.2753 | 0.4711 | 0.0063 | 0.1895 | 0.0000 |
| B | test | `russellmitchell` | 0.1799 | 0.4751 | 0.0000 | 0.2812 | 0.0000 |

**No view exceeds the brief's 50% threshold on either direction's train or test
set, so the second configuration was not run.** `auth` is the closest at
0.4751 (A train / B test) and 0.4711 (A test / B train) - under the threshold,
but close enough that a different window length would cross it (at 60 s it is
0.874, `results/ait_window/window_table.md` section 3). The
`second_configuration_run` field is `false` in both direction records, and
`tests/unit/test_ait_clean.py::test_missingness_is_reported_for_train_and_test_per_direction`
checks the flag against the measured fractions rather than trusting it.

---

## 6. Dependence structure, and the step the brief asked for that has no value

### Step 4 is NOT COMPUTABLE on this data

The brief asks for the dependence of the `same_sensor_group="network"` **pair**
against the mean of unrelated pairs, per seed, as a paired difference.
`prismflow.data.ait_lds.sensor_groups()` returns `{'network': [4]}` - **one
member, `suricata`** - so no within-group pair exists and the paired difference
has no value to report. This is the data card's limitation 9
(`docs/part27-ait-datacard.md`): there is no packet-capture view yet, so "the
near-duplicate view pair that would most directly exercise the project's
redundancy thesis does not exist". Substituting a different pair would answer a
different question, so nothing was substituted. Recorded in
`results/ait_clean/results.json`, `directions.<name>.sensor_group_check`
(`"status": "NOT COMPUTABLE"`), with the full pair table reported in its place.

### The full pair table, reported instead

Mean +/- sample std over 5 seeds, on each direction's test set, with
`REPORTING_NULL_PERMUTATIONS = 12` (the reporting-grade call, not the forward
pass's 4 - the same call Part 15 makes in `dependence_of`). Source:
`results/ait_clean/results.json`,
`directions.<name>.systems.<system>.pair_table`. Four highest pairs per cell:

| direction / system | highest pairs |
|---|---|
| A / naive (test `santos`, prevalence 0.602) | `dns-suricata` 0.7784 +/- 0.0063, `auth-audit` 0.6460 +/- 0.0804, `audit-suricata` 0.4551 +/- 0.0084, `apache-suricata` 0.2077 +/- 0.0827 |
| A / PrismFlow (test `santos`, prevalence 0.602) | `dns-suricata` 0.7789 +/- 0.0064, `auth-audit` 0.5996 +/- 0.0806, `audit-suricata` 0.4135 +/- 0.0260, `apache-suricata` 0.2255 +/- 0.1089 |
| B / naive (test `russellmitchell`, prevalence 0.900) | `auth-audit` 0.8836 +/- 0.0169, `apache-suricata` 0.6575 +/- 0.0353, `dns-suricata` 0.5838 +/- 0.0695, `apache-auth` 0.4335 +/- 0.0569 |
| B / PrismFlow (test `russellmitchell`, prevalence 0.900) | `auth-audit` 0.8705 +/- 0.0154, `apache-suricata` 0.6575 +/- 0.0189, `dns-suricata` 0.5516 +/- 0.0674, `apache-auth` 0.4163 +/- 0.0375 |

**The high pair is not the same on the two test sets.** On `santos` it is
`dns-suricata` (0.78); on `russellmitchell` it is `auth-audit` (0.88), with
`dns-suricata` down at 0.55-0.58. Two scenarios of one dataset, the same five
sensors, the same code - and a different redundancy structure. That is a
measurement bearing on [[L14]]: it says the redundancy pattern is a property of
the deployment, not of the sensor set.

### What this says about L14

`docs/KNOWN_LIMITATIONS.md` L14 asks for the redundancy audit on data whose
views come from genuinely different sensors rather than different transforms of
one image, and notes `handwritten`'s 6 nominal views carrying **3.28** effective
views at efficiency **0.547 +/- 0.011**. Measured here, per direction, on 5
genuinely different log sources:

| test scenario | prevalence | system | ENIV (eigen) of 5 views | efficiency ratio |
|---|---|---|---|---|
| `santos` | 0.602 | naive | 3.4507 +/- 0.0790 | 0.6901 +/- 0.0158 |
| `santos` | 0.602 | PrismFlow | 3.4958 +/- 0.1015 | 0.6992 +/- 0.0203 |
| `russellmitchell` | 0.900 | naive | 2.9579 +/- 0.0638 | 0.5916 +/- 0.0128 |
| `russellmitchell` | 0.900 | PrismFlow | 3.0272 +/- 0.0376 | 0.6054 +/- 0.0075 |

Five different sensors carry 3.0-3.5 effective views, efficiency 0.59-0.70.
That is **higher** than `handwritten`'s 0.547, so different sensors are less
redundant than different transforms of one glyph - but not dramatically, and
the two scenarios differ from each other by about as much as either differs
from `handwritten`. **This does not close L14**: ENIV is still only APPLIED on
this data and never VALIDATED on it, because `rho_matrix` is `None` and real
views have no ground-truth dependence
(`docs/KNOWN_LIMITATIONS.md` L9, `docs/DATASETS.md`). The efficiency ratio must
still never be quoted as a property of multi-view data in general.

---

## 7. How this differs from the synthetic results

1. **The views are different sensors, not transforms of one object.** Apache
   access logs, PAM/sshd auth, dnsmasq, auditd and Suricata. Measured
   efficiency 0.59-0.70 of 5 views, against `handwritten`'s 0.547 of 6 (section
   6 above, and `docs/KNOWN_LIMITATIONS.md` L14).
2. **The redundancy structure is not the same on the two test sets.**
   `dns-suricata` 0.78 on `santos` against `auth-audit` 0.88 on
   `russellmitchell`. On the synthetic generator rho is a knob set uniformly,
   so there is no analogue of "the high pair moved".
3. **There is no ground-truth dependence.** `rho_matrix` is `None`, so ENIV is
   applied and never validated here. Validation exists only on the synthetic
   generator (Part 05), where rho is a parameter we set.
4. **A seed does not redraw the data, and does not even re-partition it.** On
   synthetic data a seed regenerates the dataset; in Part 15 it re-partitions a
   fixed one; here train and test are whole scenarios, so a seed is model init
   and batch order only. The spreads are correspondingly narrower in direction
   A (+/- 0.0056 balanced accuracy) - and, in direction B, much wider
   (+/- 0.0685), because one seed's threshold lands badly.
5. **The classes are not balanced and the two test sets do not share a
   prevalence.** 0.602 on `santos`, 0.900 on `russellmitchell`. The synthetic
   generator is balanced over K classes, so the majority-class baseline is
   1/K there and is not a threat to any reported number. Here it is 0.9004 in
   direction B, and accuracy does not clear it.
6. **Train and test prevalence differ in both directions**: 0.900 -> 0.602 in
   A and 0.602 -> 0.900 in B. Synthetic train and test come from the same
   generator with the same parameters, so there is no prevalence shift to
   absorb. This is the most likely cause of direction B's ECE of 0.19-0.21.
7. **The positive class has 3 and 1 independent episodes, not N i.i.d.
   samples.** Synthetic samples are i.i.d. by construction, so there the
   sample size is the sample count. Here it is the block count.
8. **Missingness is a property of the data, not a knob.** `auth` is absent
   from 47.1-47.5% of windows and `suricata` from 0.0%, because that is how the
   logs fell. Part 10's missing-view conditions set a rate; here there is
   nothing to set.
9. **The label is close to a single-view signal.** The exfiltration is visible
   directly in `dns` as query names up to 250 characters
   (`docs/part27-ait-datacard.md` section 9, limitation 2). The synthetic
   generator spreads class signal across views by construction. This is the
   main reason a window-level result on this task is weak evidence about
   multi-view fusion, however good the numbers look.
10. **Calibration behaves differently from the duplication condition, and the
    two must not be conflated.** `docs/EVALUATION_PROTOCOL.md` ("Current
    claim") reports that under view duplication ECE does **not** separate at
    any k and Brier reliability leans slightly against PrismFlow. Here ECE
    separates in PrismFlow's favour, 5/5 seeds in direction A
    (-0.0120 +/- 0.0066) and 4/5 in direction B (-0.0144 +/- 0.0154). **This
    does not overturn that claim**: there is no view duplication in this Part,
    the condition is a cross-scenario prevalence shift rather than added
    copies, and the effective positive sample size is 3 and 1 blocks. It is a
    new condition, and it should be cited as one.
11. **Nothing is attacked, corrupted or cloned.** This Part is clean data only.
    No statement here bears on the Part 09 robustness result or on the Part 10
    detector matrix.

---

## 8. Two things for the user to resolve

### 8a. The evaluation protocol conflicts with the frozen-module rule

`docs/EVALUATION_PROTOCOL.md` states that an experiment needing a metric that
does not exist must add it to `prismflow.evaluation.protocol.compute_metrics`,
"with a test, so every number in the project comes from the same code". Part
28's standing rules freeze `prismflow/` to import-only. `compute_metrics` has
**no balanced accuracy, no ROC AUC and no per-class recall**, and the brief
requires all three.

**What was done instead, and why it is safe:** the three are computed in
`part28_extras` in `experiments/ait_clean/run_ait_clean.py`, by one function
that every row in every table goes through - both systems and the
majority-class baseline alike, so the comparison is like for like
(`tests/unit/test_ait_clean.py::test_baseline_and_systems_go_through_the_same_metric_code`).
The protocol's own metrics still come from `compute_metrics`; `evaluate()` is
still called and still writes the standard three files per direction per
system; and `run_system` asserts that the row it reports agrees with
`evaluate()`'s own output **seed by seed** on `accuracy`, `macro_f1`, `brier`,
`prob_ece`, `prob_aurc` and `n_samples`, raising if it does not. Balanced
accuracy is additionally pinned to the frozen code: for K = 2 with both classes
present it must equal `macro_precision_recall_f1`'s `macro_recall`, and
`test_balanced_accuracy_equals_macro_recall_when_both_classes_present` checks
it. **Nothing in `prismflow/` was touched.**

**Decision needed:** either `compute_metrics` gains `balanced_accuracy`,
`roc_auc` and per-class recall in a later Part that is allowed to unfreeze it
(after which `part28_extras` should be deleted and this experiment re-run), or
`docs/EVALUATION_PROTOCOL.md` records the exception. Part 29 and Part 30 will
hit the same wall.

### 8b. Proposed `docs/KNOWN_LIMITATIONS.md` entry - NOT added, for approval

`KNOWN_LIMITATIONS.md` was not edited, per the brief. Proposed text:

> ## L16. On real cross-scenario data, discrimination and calibration come
> apart, and the positive class is one or three episodes
>
> **Component:** the real-data evidence base for window-level detection
> (Part 28, `experiments/ait_clean/`).
>
> **Evidence:** `results/ait_clean/results.json`,
> `results/ait_clean/summary.md` (5 seeds, two directions, never pooled).
>
> - **Accuracy does not clear the majority-class baseline in direction B.**
>   Training on `santos` (60.2% attack) and testing on `russellmitchell`
>   (90.0% attack), naive fusion scores accuracy 0.9100 +/- 0.1233 and
>   PrismFlow 0.9366 +/- 0.0819 against a baseline of 0.9004. Both bars
>   straddle it. Balanced accuracy (0.9500 +/- 0.0685 and 0.9648 +/- 0.0455)
>   and ROC AUC (0.9999 +/- 0.0002 for both) clear 0.50 decisively, so the
>   systems rank well and threshold badly.
> - **The majority-class baseline is better calibrated than either system in
>   direction B.** Baseline Brier 0.1793 and ECE 0.0000 against 0.2237 /
>   0.2066 (naive) and 0.1884 / 0.1921 (PrismFlow), while the baseline's ROC
>   AUC is 0.5 and its AURC 0.0996 against 0.0246-0.0363 for the systems.
>   Under a 60.2% -> 90.0% prevalence shift, a constant forecast wins on
>   calibration and loses on everything calibration is supposed to support.
> - **Direction A does not show this**, and the two directions must not be
>   averaged. On `santos` (prevalence 0.602) both systems beat the baseline on
>   every metric, and PrismFlow beats naive fusion in 5/5 seeds on balanced
>   accuracy (+0.0043 +/- 0.0015), Brier (-0.0111 +/- 0.0043) and ECE
>   (-0.0120 +/- 0.0066).
> - **The positive class is 3 episodes (`santos`) and 1 episode
>   (`russellmitchell`).** Every +/- in Part 28 is a spread over 5
>   initialisations on a fixed pair of scenarios, not over independent attacks.
> - **The task is "was the DNS exfiltration tool running".** 1,026 of 1,031
>   and 663 of 667 attack windows carry only dnsteal-family steps, and the
>   multi-step intrusion is 4 and 3 windows respectively. No Part 28 metric is
>   computed over those windows.
>
> **What this means.** Part 28 is evidence that the pipeline runs on real
> multi-sensor log data and that ENIV reads 3.0-3.5 of 5 views there. It is
> not evidence of intrusion detection, and in direction B it is not evidence
> that either system is usable as a detector at a fixed threshold.
>
> **What would change our understanding.** A threshold chosen on the training
> scenario and reported as such, rather than argmax at 0.5; a third and fourth
> scenario so the direction count exceeds the episode count; and a window
> length at which more than one attack episode exists.

---

## 9. Reproducing

```
python -m experiments.ait_clean.run_ait_clean            # 5 seeds, ~14 min CPU
python -m experiments.ait_clean.run_ait_clean --quick    # 2 seeds, 2 epochs, SMOKE TEST
python -m pytest tests/unit/test_ait_clean.py -q
```

`--quick` writes to `results/ait_clean_quick/`, which is gitignored, and marks
its own `summary.md` and `results.json` as **SMOKE TEST, NOT A FINDING**. It
exists to check plumbing and must never be reported.

Requires both scenarios processed at 300 s / network, i.e.
`data/processed/ait_russellmitchell_network_300s/` and
`data/processed/ait_santos_network_300s/` from `/part26`. If either is missing
the runner stops and says so: it does **not** fall back to a random
within-scenario split and does not run a single direction.
