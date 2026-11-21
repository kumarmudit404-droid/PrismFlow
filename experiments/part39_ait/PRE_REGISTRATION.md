# Part 39 item A — PRE-REGISTRATION

**Written 2026-11-21, before anything was run.** No model has been trained, no
threshold has been selected and no metric has been computed for this item.
Nothing below was chosen after seeing a Part 39 number.

**Directory note.** The review brief for this scaffolding said
`experiments/ait_part39/`. This is `experiments/part39_ait/` instead, because
Part 39's own standing rules permit creating only
`experiments/part39_ait/**`, `results/part39_ait/**` and
`tests/unit/test_part39_ait.py`. Scaffolding at a path the Part may not write
to would be orphaned the moment Part 39 runs.

---

## 0. What this item addresses, and what it does not

**L16's written closing condition**, from `docs/KNOWN_LIMITATIONS.md`,
"What would change our understanding": *"A threshold chosen on the training
scenario and reported as such, rather than argmax at 0.5; a third and fourth
scenario so the direction count exceeds the episode count; and a window length
at which more than one attack episode exists."*

This item delivers **only the first clause**. The second is blocked by data —
only `russellmitchell` and `santos` are downloaded (`data/raw/ait/`) — and the
third is Part 39 item B, which is not this item.

**Out of scope here, and not to be smuggled in:** window-length sensitivity
(item B, L31), the contaminated-fraction axis (item C, L18), per-view epsilon
(item D, L21), and any third scenario (item E, blocked).

**What changes from Part 28: the decision threshold, and nothing else.** The
loader, the window length (300 s), the sample mode (`network`), the two
directions, the five seeds, the model (`hidden_dims [64, 64]`, `feature_dim`
32, `dropout` 0.0), the training recipe (40 epochs, batch 64, lr 0.001,
`anneal_epochs` 10, `per_view_loss_weight` 1.0), the z-score scaler fitted on
training `label_known` rows with `present_rows_only`, and the `label_known`
filter are all inherited unchanged from `experiments/ait_clean/config.yaml`.
A retrain is required because **no AIT model checkpoint and no per-row AIT
probability is committed anywhere**: `results/ait_clean/<direction>/<system>/`
holds only `metrics.csv`, `metrics.json` and `reliability_diagram.png`, and
there is no `.npz`, `.npy`, `.pt` or `.ckpt` for AIT in the repository. The
retrain must reproduce Part 28's committed figures before any new threshold is
applied; see §5.

---

## 1. The threshold rule, and why there is no validation split

**The threshold is selected on the WHOLE training scenario. There is no
held-out validation split, and this is forced by the data, not chosen for
convenience.**

`docs/part27-ait-datacard.md` section 8 rules out both ways of carving one:

> - A **random** window split trains and tests on neighbouring windows of one
>   run, which leaks.
> - A **chronological** split on `santos` puts nearly all attack windows before
>   nearly all benign windows, so early/late is close to a label split.

`experiments/ait_clean/config.yaml` states the same prohibition as a rule:
"a random split WITHIN a scenario is forbidden in both directions". The cause
is episode structure: in `network` mode at 300 s, `russellmitchell` has
**1** contiguous attack run (1,031 windows) and `santos` has **3**
(664/2/1) (`window_table.json`, `attack_runs`, quoted in datacard section 8).
A held-out slice of one run is not independent of the rest of that run.

So the rule is:

> **Select the threshold `t` that maximises Youden's J on the TRAINING
> scenario's `label_known` rows, using the same trained model that will be
> scored on the test scenario.** Ties broken by the smaller `t`.

Three things about this rule, stated now:

1. **Youden's J and balanced accuracy have the same argmax.**
   `J = sensitivity + specificity − 1 = 2 × balanced_accuracy − 1`, a strictly
   increasing transform, so maximising J is maximising balanced accuracy. L16's
   condition names "maximise balanced accuracy on train" as its example; the
   brief named Youden's J. They select the same threshold and the chosen `t`
   is reported under both names.
2. **The threshold is fitted on data the model was fitted on, so it is
   optimistic *for the training scenario*.** That is a real cost and it is
   accepted deliberately, because the alternative splits leak or are
   label-adjacent.
3. **The reported number is unaffected by that optimism.** Every reported
   metric is computed on the **test** scenario, which neither the model nor the
   threshold has seen. The threshold is a single scalar carried across a
   scenario boundary. This is exactly what L16 asks for: a threshold chosen on
   the training scenario and *reported as such*.

The grid for `t`: every distinct value of `prob_attack` on the training
scenario's `label_known` rows, plus 0.0 and 1.0. Exhaustive, so there is no
step-size choice to tune.

---

## 2. Scope

| | |
|---|---|
| directions | `A_rm_to_santos` (train `russellmitchell`, test `santos`), `B_santos_to_rm` (train `santos`, test `russellmitchell`) |
| systems | `naive` (no discount), `prismflow` (discount) |
| seeds | 0, 1, 2, 3, 4 |
| units | 2 × 2 × 5 = **20**, a retrain each |

**The two directions are never pooled, averaged or concatenated.** They are
two measurements on two test sets at different prevalences: test `santos`
n = 1,108 at attack prevalence **0.601985559566787**, test `russellmitchell`
n = 1,145 at **0.9004366812227074** (`results/ait_clean/results.json`,
`directions.<name>.majority_class_baseline.n_samples` and the committed
prevalence fields).

**What a seed varies**, so no `+/-` is misread: train and test are whole
scenarios, so a seed cannot redraw or re-partition the data. It selects the
model's parameter initialisation (`set_seed`) and the minibatch order of every
epoch (`iter_batches(shuffle=True, seed=seed+epoch)`). **Every `+/-` is
initialisation-and-batch-order variance on a fixed pair of scenarios.** It is
**not** a spread over independent attacks: the test sets hold 3 (`santos`) and
1 (`russellmitchell`) independent contiguous attack episodes (L16, datacard
section 8).

---

## 3. Hypothesis

**H1.** With the threshold chosen on the training scenario by §1's rule,
balanced accuracy on the test scenario is **within 0.02** of its committed
argmax-at-0.5 value, in **both** directions and for **both** systems.

Committed argmax-at-0.5 references, from
`results/ait_clean/results.json`,
`directions.<name>.systems.<system>.summary.balanced_accuracy`:

| direction | system | committed balanced accuracy |
|---|---|---|
| `A_rm_to_santos` | naive | **0.9443750913658817 ± 0.0056462129696094225** |
| `A_rm_to_santos` | prismflow | **0.9486834813885574 ± 0.00613712861801559** |
| `B_santos_to_rm` | naive | **0.9500484966052376 ± 0.06845841673498461** |
| `B_santos_to_rm` | prismflow | **0.9647914645974782 ± 0.045472647363396554** |

- **H1 is CONFIRMED for a cell** iff `|mean(new) − mean(committed)| < 0.02`.
- **H1 is DISCONFIRMED for a cell** otherwise.
- **H1 is CONFIRMED overall** iff it is confirmed in all four cells.

**The signed difference is reported for every cell, not the absolute value**,
and the direction of a disconfirmation must be stated. This matters because
H1 as written is a *null* hypothesis about threshold choice: a train-chosen
threshold that improves balanced accuracy by more than 0.02 **disconfirms H1**
while being a better outcome for the system. A disconfirmation therefore has
two readings and the sign decides which:

- **disconfirmed upward** — the train-chosen threshold beats 0.5 by more than
  0.02. L16's "rank well and threshold badly" is then *actionable*: the
  ranking was always there and 0.5 was simply the wrong operating point.
- **disconfirmed downward** — the train-chosen threshold is more than 0.02
  worse than 0.5. Then 0.5 was accidentally favourable on this data, and the
  argmax-at-0.5 figures in Part 28 are optimistic as deployment numbers.

Neither reading may be presented as "H1 passed" or "H1 failed" without the
sign.

**Secondary observations, no bar attached:** the selected threshold `t` per
unit (mean ± std over seeds, per direction and system); accuracy, attack
recall, benign recall, macro-F1 and Brier at `t` beside their committed
argmax-at-0.5 values; and the balanced accuracy achieved at `t` on the
*training* scenario, which bounds how optimistic the selection was.

---

## 4. Write-down rules

1. **5 seeds, mean ± sample std (ddof=1), per cell.** A cell with fewer than 5
   completed seeds is **not a finding** and no verdict is issued from it.
2. **Paired within seed.** Every `new − committed` difference is formed seed by
   seed, and the **per-seed sign counts** (`positive`, `negative`,
   `exactly_zero`) are reported **before** the aggregate mean. A cancelling
   population reads as "no effect" if only the mean is shown.
3. **No pooling across directions**, ever — not in a mean, not in a std, not in
   a concatenation.
4. **No pooling across systems.** `naive` and `prismflow` are separate cells.
5. **No result-driven selection.** The threshold rule, the grid, the bar
   (0.02), the metric set and the cell structure are all fixed by this
   document. If the rule turns out to be a poor one, that is reported as a
   finding about the rule; it is not replaced with a better-performing rule
   after the fact.
6. **Every number cites its committed path.** An absent quantity is written
   `not measured`, never 0 and never blank.
7. **No limitation's status changes here.** Proposals go to
   `docs/proposed_limitations_update.md` for the review gate.
8. **Frozen modules are import-only.** `prismflow/**` is not edited;
   `prismflow/data/ait_lds.py`, `prismflow/training/` and
   `prismflow.evaluation` are used as they are. `prismflow.evaluation.
   compute_metrics` carries no balanced accuracy and no ROC AUC and cannot be
   edited, so those are computed in this item's own runner — the same wall
   Parts 29, 30 and 38 hit.

---

## 5. Stopping rules

Checked in this order. Any one stops the item and produces a report rather
than a result.

1. **Reproduction gate.** Before any new threshold is applied, each retrained
   unit's **argmax-at-0.5** balanced accuracy must reproduce Part 28's
   committed per-seed value for that cell. A mismatch beyond `1e-9` stops the
   item and is reported as a defect in the retrain, not worked around. This is
   the gate that makes the comparison in §3 a comparison of thresholds rather
   than of two different training runs.
   - **If the retrain does not reproduce bit-exactly**, the likely cause is
     the thread-count dependence Part 34 measured — a committed `eniv_eigen`
     misses its own value by `0.0107` at `OMP_NUM_THREADS=1`. Record the
     observed thread count with every unit and re-run at the machine default
     before concluding anything.
2. **Degenerate threshold.** If the selected `t` is 0.0 or 1.0 for any unit —
   i.e. the training scenario's optimum is "call everything attack" or "call
   nothing attack" — that unit is reported as **degenerate at selection** and
   its metrics are reported with that label attached. It is not excluded and
   not re-selected.
3. **Label coverage.** Only `label_known` rows are used, in train and in test.
   `labels == -1` means "no label-covered file contributed an event to this
   window" and is **never** read as benign.
4. **Episode count.** The item reports, beside every number, that the positive
   class is **3 episodes** (`santos`) and **1** (`russellmitchell`). If a
   future change makes that count unavailable, stop.

---

## 6. What this item cannot show, stated in advance

- **It cannot make Part 28 evidence about fusion.** L17 stands: `single_view_dns`
  scores balanced accuracy **0.9898** on `santos` and **1.0000** on
  `russellmitchell` (`results/ait_clean/baselines.json`,
  `directions.<name>.baselines.single_view_dns.balanced_accuracy`), and no
  other single view clears 0.7061. The label is "was the exfiltration tool
  running". A better threshold does not change what is being detected.
- **It cannot widen the episode count.** 3 and 1 remain 3 and 1, so no `+/-`
  here is a spread over independent attacks. That is L16's second clause and
  it is blocked by data.
- **It cannot speak to calibration.** A threshold is a decision rule, not a
  probability correction. L16's calibration finding — that in direction B a
  non-oracle `flat_all_views` baseline beats both systems on Brier
  (**0.1264**) and ECE (**0.0765**), outside both systems' one-std ranges —
  is untouched by anything this item measures.
- **It cannot be read as a deployment result.** The threshold transfers from
  one simulated scenario to another simulated scenario in the same testbed.
