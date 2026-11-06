# Part 38 step 1 — LANL redesign: PRE-REGISTRATION

**Written and committed before any model was trained, any sample was drawn, or
any metric was computed on the new design.** Nothing in this file may be
changed after a number has been looked at; a change here is a new experiment,
not a revision of this one.

Every figure quoted below as a target or a bar is a **committed Part 30 or
Part 35 number**, cited to its file. No figure in this file is a result of
Part 38.

---

## 0. What this addresses, and what it does not

Part 38 step 1 redesigns the LANL sample so that two limitations can be
separated from the thing they are confounded with. It addresses the written
closing conditions of **L24**, **L27** and **L28**.

It does **not** change any limitation's status. `docs/KNOWN_LIMITATIONS.md` is
not edited by this step. L26 is closed and is out of scope. L29 is closed as
diagnosed; its *naming* defect is avoided here (section 5) but its status is
untouched.

### Frozen-module discipline

`experiments/lanl/run_lanl.py` is **FROZEN** and is not edited, copied, or
imported-and-patched. This runner is new code under `experiments/lanl_v2/`
that calls `prismflow.training` — the dataset-agnostic scaler and trainer that
resolved **L25** — instead of re-mirroring `fit_scaler` / `apply_scaler` /
`train_lanl` for a third time.

`prismflow/` and `app.py` are not modified. The feature parsing and the
per-view accumulators are **reused** from `prismflow.data.lanl_cyber1` by
calling it, not by copying it.

---

## 1. Hypotheses

Pre-specified, directional, each with a numeric bar taken from a committed
number. Each is falsifiable by the sweep this file plans.

**H1 (L24 — selection vs view drop).** With control cells drawn without
reference to the `auth` stream, the *auth-presence-only* trivial baseline on
the test block falls materially below Part 30's committed test AUC of
**0.6637** for that baseline (`results/lanl/results.json`,
`phase3_diagnostics.trivial_baselines`; L24). "Materially" is pre-set at a
drop of at least **0.05** AUC. That baseline is seed-free, so the bar applies
to the single recomputed value.

*Why this is the right test.* Under Part 30's design every attack cell had an
auth record (410/410) while about a third of benign cells did not
(12,323/18,319), so auth presence alone separated the classes. If the new
selection has removed that, the baseline must fall. If it does **not** fall,
H1 is disconfirmed and the selection signal has another source, which is a
reportable finding in its own right.

**H2 (L27 — host vs behaviour).** With all benign cells of attack hosts
retained, dropping every test row of `C17693` changes test ROC AUC by **less
than 0.036** in absolute value, for all four arms and both systems. 0.036 is
the *smallest* of Part 30's committed per-arm drops; the committed band is
**0.036 to 0.052** (`results/lanl_reanalysis/results.json`, `step5_l27`; L27).

*Why.* Under Part 30 that host carried 279 of 849 attack samples with **1**
benign sample in train and **1** in test. If benign contrast for that host is
what the model was missing, restoring it should shrink the leave-host-out
sensitivity. H2 is disconfirmed if any of the eight arm-and-system cells still
moves by 0.036 or more.

**H3 (L28 — mode stability).** The **sign** of the PrismFlow-minus-naive
difference in test ROC AUC agrees across both execution modes, in all four
arms. Part 30 found agreement on `roc_auc` and `prob_ece` but **disagreement**
on `balanced_accuracy`, `attack_recall` and `brier` for `with_auth`
(`results/lanl/execution_mode.json`; L28). H3 is tested on `roc_auc` and
reported, without a bar, for the other five metrics.

---

## 2. The exact sampling rule

The time grid, the split, the feature list, the label definition and the
window length are **unchanged** from `configs/data/lanl.yaml` and are not
restated as choices here: window 300 s, days 1-16, train t in [1, 864000],
dropped gap t in [864001, 1036800], test t in [1036801, 1382400], label
`computer_window`, `attributed_computer: source`.

What changes is **which cells enter the sample**.

### Cases

Every cell with `attack = 1` is kept. Unchanged from Part 30.

### Controls

A cell is kept as a control **iff both**:

**(a) Non-auth occupancy.** The cell is occupied in at least one of
`proc`, `flows`, `dns`. The `auth` stream is **never consulted** to decide
whether a cell enters the sample.

**(b) The host rule.** Either

- the cell's computer has **at least 1 attack cell anywhere in the read
  range** -> the cell is kept, i.e. **all** benign cells of attack hosts are
  kept (share **1/1**); or
- otherwise -> kept iff `crc32(computer | window) % 512 == 0`, which is
  Part 30's rule and its `control_divisor: 512`, unchanged.

`auth` features are attached to the selected cells in a **later pass**, so
auth enters this experiment as *features only* and never as selection.

### How each closing condition is met

| Limitation | Written closing condition | How (a)/(b) meets it |
|---|---|---|
| **L24** | "control cells drawn without reference to the `auth` stream" | (a). Selection reads non-auth occupancy, the cell-identity hash and the attack-host set. No auth record is read until the feature pass, after selection is fixed. |
| **L27** | "keeps all or a stated share of the benign cells of attack hosts" | (b). The stated share for attack hosts is **1/1 — all of them**. |

### Declared size fallback, fixed now

An attack host's benign cells are kept in full, so the sample grows. The bar
and the fallback are set **before any count is seen**, so the rule stays
mechanical:

- If total samples (train + test) exceed **400,000** — about **6.5x** Part
  30's committed 61,641 — the attack-host benign share drops from 1/1 to the
  **largest 1/2^k** that brings the total under 400,000.
- `k` is recorded in the results file and in the README. `k = 0` means the
  full "all" rule held.

### What this design still costs, stated in advance

- Sample prevalence remains an **artefact of the design**, not population
  prevalence, and is now an artefact of *two* rules rather than one. Accuracy
  and every prevalence-dependent figure describes the sample. ROC AUC,
  balanced accuracy and per-class recall are the figures that survive.
- Rule (b) uses **host-level label knowledge** for selection: whether a
  computer has any attack cell. Per-host prevalence therefore differs between
  attack and non-attack hosts *by construction*. This is what L27's closing
  condition asks for, and it is the reason the figure that matters is the
  **within-attack-host** contrast rather than the overall prevalence.
- Rule (a) excludes **auth-only** cells, which Part 30 included. The new
  sample is therefore not a subset or a superset of Part 30's, and the two
  sample tables are **not comparable row-for-row**. Every cross-Part
  comparison in this programme is a comparison of designs, not of runs.
- "benign" still means **no KNOWN red-team event**, not verified clean.

---

## 3. Metrics

Computed in the new runner, because `prismflow.evaluation.compute_metrics`
carries no balanced accuracy and no ROC AUC and **cannot be edited** — it is
frozen, and Parts 29 and 30 hit the same wall.

**Primary (test block):** ROC AUC, balanced accuracy, attack recall, benign
recall, Brier, ECE at 15 bins.

**Diagnostics:**

- auth-presence-only trivial baseline AUC (**H1**);
- leave-`C17693`-out test AUC delta (**H2**);
- eigen ENIV and mean pairwise dependence (eigen is the headline method; the
  design effect is reported beside it, never gated on);
- per-row present-view count, and the count of rows with an **all-False view
  mask** — the population L29 identified as unpredictable by construction;
- sample size and prevalence per split, per day, and per host for the attack
  hosts.

---

## 4. Seed plan

**Seeds 0, 1, 2, 3, 4** — five seeds, mean +/- std reported for every figure.

4 arms x 2 systems x 5 seeds = **40 units per execution mode**. Arms and
systems are unchanged from Part 30: `without_auth`, `without_flows`,
`without_dns`, `with_auth`; `naive` (no discount) and `prismflow` (discount).

**What a seed varies, stated so no +/- is misread.** The split is fixed by
time and the sample is a deterministic function of cell identity, the non-auth
occupancy and the attack-host set. A seed therefore **cannot** redraw or
re-partition the data. It selects exactly two things: the model's parameter
initialisation (`set_seed`) and the minibatch order of every epoch
(`iter_batches(shuffle=True, seed=seed+epoch)`). Every reported +/- is
**initialisation-and-batch-order variance on a fixed sample**. It is not
dataset variance and it is not a spread over independent attacks.

---

## 5. The confidence key (L29's naming defect, avoided not fixed)

The frozen `experiments/lanl/run_lanl.py` writes
`collected["vacuity_confidence"]` into the score npz under the key
**`"vacuity"`**. That name is inverted: the column holds a **confidence**, so
`0.0` on an evidence-free row is the correct value for that column and the
*name* is the defect. The file is frozen and is **not** edited here — this is
a report, not a fix.

This runner writes the same quantity under the key **`confidence`**, and
records in the npz a `confidence_convention` string stating that higher means
more evidence and that an all-False-view-mask row reads 0.0. No file written
by this runner contains a key named `vacuity`.

---

## 6. Execution modes (L28)

Two modes, both run, **mode recorded per run**:

- **Mode A — `threads1_concurrent11`:** `torch.set_num_threads(1)`, 11
  concurrent unit processes. Reproduces the mode recorded in
  `results/lanl/execution_mode.json` for Part 30 run 2.
- **Mode B — `threadsdefault_sequential`:** machine-default thread count,
  units run sequentially. The default is **measured and recorded**, never
  assumed: L28 records this box's default as 8 while Part 30 run 1's record
  says 12.

Every results file and every score npz carries a `mode` block: mode name,
`torch.get_num_threads()` as actually observed, concurrency, and the measured
machine default.

**Aggregation rule, fixed now.** Per-mode mean +/- std over the 5 seeds,
**and** the across-mode spread, reported separately. The two modes are
**never pooled into a single +/-**: L28's finding is that a 5/5 seed count
inside one run shares one execution mode and is not independent evidence
about mode effects.

A reported figure is conditional on a thread count. ENIV reproduces only at
the thread count it was computed at, so every ENIV value is reported with its
mode.

---

## 7. Stopping rules

Checked in this order. Any one of them stops the Part and produces a report
rather than a result.

1. **Reproduction gate.** Before the new design is sampled, the new sampler
   re-implements Part 30's rule and must reproduce Part 30's committed
   per-day sample counts **exactly** (`results/lanl/results.json`,
   `split_report.per_split.*.per_day`). A mismatch of even one cell stops the
   Part and is reported as a defect in the new sampler — not worked around.
2. **Attack-window floor.** At least **10** attack windows per split, train
   and test counted separately, checked after windowing and **before any
   model is built**. Unchanged from `configs/data/lanl.yaml`.
3. **Size bar.** Total samples over 400,000 after the section-2 fallback ->
   stop and report.
4. **Timestamp inversion** in any view -> stop and report.
5. **Gzip stream error** in any view -> stop and report.
6. **No status changes.** This step does not edit
   `docs/KNOWN_LIMITATIONS.md`, and does not mark L24, L27 or L28 closed,
   resolved or narrowed. Status changes require the full sweep and a review
   gate.

---

## 8. What the smoke run is, and is not

The sweep this file plans is 40 units x 2 modes on a freshly built sample
table. Before that is launched, a **2-seed smoke** runs on the `with_auth` arm
only, on a **declared reduced time range and reduced split**.

The smoke exists to show three things and nothing else: the pipeline runs end
to end, the correctly named `confidence` key is written, and seed 0 and seed 1
differ in a way consistent with initialisation-and-batch-order variance.

**The smoke's numbers are pipeline evidence, not findings.** They are 2 seeds,
not 5; they are one arm, not four; they are a reduced range, not days 1-16.
They are labelled as such wherever they appear and may not be quoted as a
result of Part 38, a measurement of any hypothesis in section 1, or evidence
about any limitation.

---

## 9. Pre-registration record

| | |
|---|---|
| Written | 2026-11-06 |
| Branch | `part38-lanl`, off `main` at `805d1fc` |
| Reproduction target | train 42,912 (439 attack, prevalence 0.01023024); test 18,729 (410 attack, prevalence 0.02189118); 849 attack of 61,641 total — `results/lanl/results.json`, `results/lanl/samples.npz` |
| Frozen, not edited | `experiments/lanl/run_lanl.py`, `prismflow/evaluation/`, `prismflow/data/lanl_cyber1.py`, `app.py` |
| Reused, not copied | `prismflow.training` (L25), `prismflow.data.lanl_cyber1` parsing and accumulators |
