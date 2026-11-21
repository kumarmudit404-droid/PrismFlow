# Part 39 item A — a threshold chosen on the training scenario (L16)

**Status: scaffolding only. NOTHING HAS BEEN RUN.** No model trained, no
threshold selected, no metric computed. This directory currently holds a
pre-registration, a config skeleton and this file. Every number here is a
**reference from an earlier committed Part**, cited to its path; there are no
Part 39 results yet.

**Directory name.** The review brief said `experiments/ait_part39/`. This is
`experiments/part39_ait/` because Part 39's standing rules permit creating
only `experiments/part39_ait/**`, `results/part39_ait/**` and
`tests/unit/test_part39_ait.py`. Scaffolding at a path the Part may not write
to would be orphaned the moment Part 39 runs.

---

## 1. Purpose

Close the **first clause** of L16's written closing condition
(`docs/KNOWN_LIMITATIONS.md`): *"A threshold chosen on the training scenario
and reported as such, rather than argmax at 0.5."*

L16's finding is that on AIT the systems **"rank well and threshold badly"**.
In direction B — train `santos` (90.0% attack), test `russellmitchell` —
accuracy does not clear the majority-class baseline of 0.9004: naive fusion
scores **0.9100 ± 0.1233** and PrismFlow **0.9366 ± 0.0819**, both bars
straddling it, while ROC AUC is **0.9999 ± 0.0002** for both. Something that
ranks at 0.9999 and decides at 0.91 has a threshold problem, not a ranking
problem — and 0.5 was never chosen, it was inherited as the argmax of a
two-class softmax.

This item reports balanced accuracy at a threshold **selected on the training
scenario** beside Part 28's argmax-at-0.5 figures, so the two can be compared
on the same trained models.

## 2. Scope

| | |
|---|---|
| directions | `A_rm_to_santos`, `B_santos_to_rm` — **never pooled** |
| systems | `naive`, `prismflow` |
| seeds | 0, 1, 2, 3, 4 |
| units | **20**, each a retrain |
| compute | **~15–20 min** — `experiments/ait_clean/README.md:466` records Part 28's full 5-seed run at **~14 min CPU** |

**A retrain is required**, and this is a measured fact about the repository
rather than a preference: **no AIT model checkpoint and no per-row AIT
probability is committed anywhere.** `results/ait_clean/<direction>/<system>/`
holds only `metrics.csv`, `metrics.json` and `reliability_diagram.png`; there
is no `.npz` or `.npy` under `results/ait_clean/` or `results/ait_attack/`, and
no `.pt` or `.ckpt` in the repository. `run_ait_clean.py` calls
`collect_predictions` in memory and writes aggregates.

So the threshold cannot be applied by re-scoring saved probabilities the way
Part 38's Table 3 re-scores LANL. The models have to be rebuilt first — which
is why §5's reproduction gate exists.

## 3. Pre-registration

`experiments/part39_ait/PRE_REGISTRATION.md`, written before anything was
run, with `experiments/part39_ait/config.yaml` as its machine-readable form.
Read the pre-registration first; this README does not restate its rules, and
where the two differ the pre-registration governs.

The two things worth knowing before reading either:

**There is no validation split, and that is forced by the data.**
`docs/part27-ait-datacard.md` section 8 rules out both ways of making one: a
**random** window split "trains and tests on neighbouring windows of one run,
which leaks", and a **chronological** split on `santos` "puts nearly all attack
windows before nearly all benign windows, so early/late is close to a label
split". At 300 s in `network` mode, `russellmitchell` has **1** contiguous
attack run (1,031 windows) and `santos` has **3** (664/2/1), so a held-out
slice of one run is not independent of the rest of it. The threshold is
therefore selected on the **whole** training scenario.

That makes the threshold optimistic *for the training scenario*, which is
accepted and reported (the pre-registration requires the train-scenario
balanced accuracy at `t` to be published beside everything else). It does
**not** make the reported number optimistic: every reported metric is computed
on the **test** scenario, which neither the model nor the threshold has seen.
A single scalar crosses the scenario boundary. That is what L16 asks for.

**Youden's J and "maximise balanced accuracy on train" are the same rule.**
`J = sensitivity + specificity − 1 = 2 × balanced_accuracy − 1`, a strictly
increasing transform, so the argmax is identical. The brief named J; L16's
condition names balanced accuracy. The selected `t` is reported under both.

## 4. Hypothesis and how to read a disconfirmation

**H1.** Balanced accuracy at the train-chosen threshold is **within 0.02** of
its committed argmax-at-0.5 value, in both directions and both systems.

References, `results/ait_clean/results.json`,
`directions.<name>.systems.<system>.summary.balanced_accuracy`:

| direction | system | argmax at 0.5 |
|---|---|---|
| `A_rm_to_santos` | naive | 0.9443750913658817 ± 0.0056462129696094225 |
| `A_rm_to_santos` | prismflow | 0.9486834813885574 ± 0.00613712861801559 |
| `B_santos_to_rm` | naive | 0.9500484966052376 ± 0.06845841673498461 |
| `B_santos_to_rm` | prismflow | 0.9647914645974782 ± 0.045472647363396554 |

**H1 is a null hypothesis about threshold choice, so "disconfirmed" is not
the same as "bad".** The signed difference is reported for every cell and the
direction decides the reading:

- **disconfirmed upward** — the train-chosen threshold beats 0.5 by more than
  0.02. L16's "rank well and threshold badly" becomes *actionable*: the
  ranking was always there and 0.5 was the wrong operating point.
- **disconfirmed downward** — the train-chosen threshold is more than 0.02
  worse. Then 0.5 was accidentally favourable on this data and Part 28's
  argmax-at-0.5 figures are optimistic as deployment numbers.

Neither may be written as "H1 passed" or "H1 failed" without the sign. Note
also that direction B's committed stds are large (**0.0685** and **0.0455**),
so a 0.02 bar sits well inside one std there: a cell can satisfy H1 while the
underlying per-seed values move a great deal. **Report the per-seed signs
before the mean.**

## 5. Reproduction

Not yet runnable — no runner exists in this directory. When it does:

```sh
# 1. confirm the processed windows Part 28 used are present
ls data/processed/ait_russellmitchell_network_300s data/processed/ait_santos_network_300s

# 2. the item itself (runner not yet written)
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/part39_ait/run_item_a.py
```

**Inputs**, all committed or already on disk:

| input | where |
|---|---|
| processed 300 s network windows | `data/processed/ait_{russellmitchell,santos}_network_300s/` |
| Part 28's recipe | `experiments/ait_clean/config.yaml` |
| the argmax-at-0.5 references | `results/ait_clean/results.json` |
| the single-view context | `results/ait_clean/baselines.json` |

**Expected outputs:**

| path | content |
|---|---|
| `results/part39_ait/units/<direction>_<system>_seed<N>.json` | per unit: selected `t`, metrics at `t`, metrics at 0.5, the reproduction-gate comparison, observed thread count |
| `results/part39_ait/item_a.json` | per cell: mean ± sample std (ddof=1) over 5 seeds, per-seed signs, H1 verdict per cell and overall |
| `experiments/part39_ait/README.md` | updated with the numbers and the verdict |

**The reproduction gate runs before any new threshold is applied.** Each
retrained unit's **argmax-at-0.5** balanced accuracy must reproduce Part 28's
committed per-seed value for that cell to within `1e-9`. Without that gate the
comparison in §4 would be a comparison of two different training runs rather
than of two thresholds. A mismatch **stops the item** and is reported as a
defect in the retrain.

If it does mismatch, suspect the thread count first: Part 34 measured a
committed `eniv_eigen` of **3.4396320614332856** missing its own value by
**0.0107** at `OMP_NUM_THREADS=1` (**3.4289614712229826**). Every unit records
its observed thread count for that reason.

## 6. What this item does NOT do

| | |
|---|---|
| **item B** — window-length sensitivity at two more lengths per direction (L31) | not here; needs `ait_lds.rewindow` and a retrain per length |
| **item C** — contaminated-fraction axis with within-batch detector AUC (L18) | not here; also needs a retrain, since `run_ait_attack.py:860` strips `_scores` before serialisation so no AIT attacked tensor is committed |
| **item D** — per-view epsilon at matched effort (L21) | not here; `ChorusConfig.epsilon` is a scalar and `prismflow/attacks/` is frozen, so it needs a v3-style wrapper |
| **item E** — a third scenario (L16's second clause, L17) | **BLOCKED BY DATA.** Only `russellmitchell` and `santos` are downloaded (`data/raw/ait/`, 2.2 GB each). AIT v2.0 ships eight scenarios. |

## 7. What this item cannot show, whatever it measures

- **It cannot make Part 28 evidence about fusion.** L17 stands:
  `single_view_dns` — a logistic regression on nine `dns` features and that
  view's mask, nothing else — scores balanced accuracy **0.9898** on `santos`
  and **1.0000** on `russellmitchell` (`results/ait_clean/baselines.json`,
  `directions.<name>.baselines.single_view_dns.balanced_accuracy`), and no
  other single view clears **0.7061**. The label is "was the DNS exfiltration
  tool running in this window". A better threshold does not change what is
  being detected.
- **It cannot widen the episode count.** The positive class stays **3**
  episodes on `santos` and **1** on `russellmitchell`, so no `+/-` here is a
  spread over independent attacks. That is L16's second clause and it is
  blocked by data.
- **It cannot speak to calibration.** A threshold is a decision rule, not a
  probability correction. L16's finding that in direction B the non-oracle
  `flat_all_views` baseline beats both systems on Brier (**0.1264**) and ECE
  (**0.0765**), outside both systems' one-std ranges
  (`results/ait_clean/baselines.json`,
  `directions.B_santos_to_rm.baselines.flat_all_views`), is untouched.
- **It cannot be read as a deployment result.** The threshold transfers from
  one simulated scenario to another simulated scenario in the same testbed.

**No limitation's status changes here.** Proposals go to
`docs/proposed_limitations_update.md` for the review gate.
