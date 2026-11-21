# Part 38 — LANL redesign: ANALYSIS TEMPLATE

**This file contains no results.** Every cell marked `[pending]` is unfilled
and must be filled from a committed file by its exact path, or left as
`not measured`. A `[pending]` that is still present when this document is read
as a report is a defect in the report, not a zero.

Reference numbers quoted below are **measurements from earlier committed
Parts**, each with its JSON path. They are the bars, not this Part's results.

- **Status:** template only. Written 2026-11-21, while mode B (`default`) was
  still running at 10 of 40 units.
- **Not a pre-registration.** `PRE_REGISTRATION.md` and its dated Amendment 1
  are frozen and are not edited by this file. Where the review brief's wording
  of a closing condition differs from the pre-registered hypothesis, the
  **pre-registered** text governs and the difference is recorded in §3.
- **Status changes are out of scope.** Pre-registration §7.6: this programme
  "does not mark L24, L27 or L28 closed, resolved or narrowed. Status changes
  require the full sweep and a review gate."

---

## 1. Methodology

### 1.1 Design, as pre-registered

Time grid, split, feature list, label definition and window length are
unchanged from `configs/data/lanl.yaml`: window 300 s, days 1–16, train
`t ∈ [1, 864000]`, dropped gap `t ∈ [864001, 1036800]`, test
`t ∈ [1036801, 1382400]`, label `computer_window`,
`attributed_computer: source` (PRE_REGISTRATION §2).

What Part 38 changes is **which cells enter the sample**. Every `attack = 1`
cell is kept, unchanged from Part 30. A cell is kept as a control **iff both**:

- **(a) Non-auth occupancy** — the cell is occupied in at least one of `proc`,
  `flows`, `dns`. The `auth` stream is never consulted to decide whether a
  cell enters the sample; auth features are attached in a later pass, so auth
  enters as *features only*.
- **(b) The host rule** — a computer with at least 1 attack cell anywhere in
  the read range has **all** its benign cells kept (share 1/1); otherwise a
  cell is kept iff `crc32(computer | window) % 512 == 0`, which is Part 30's
  rule at its unchanged `control_divisor: 512`.

Size fallback, fixed before any count was seen: if train + test exceed
**400,000** samples, the attack-host benign share drops from 1/1 to the
largest `1/2^k` that brings the total under 400,000. `k` is recorded in the
results file. `k = 0` means the full rule held.

**`k` as built: `k = 2`, share `1/4`. The fallback FIRED.** From
`results/lanl_v2/build_v2_days16.json`, `size_fallback`:
`fallback_fired: true`, `attack_host_benign_share_k: 2`,
`attack_host_benign_share: "1/4"`, `rows_before_fallback: 979122`,
`rows_after_fallback: 281824`, `attack_host_benign_cells_before: 929764`,
`attack_host_benign_cells_dropped: 697298`, against `max_total_bar: 400000`.
Realised sample: `n_samples: 281823`, train 197,363 (prevalence
0.0022243277615358503), test 84,460 (prevalence 0.0048543689320388345);
`attack_hosts: 302`.

**This is a design fact with a direct consequence for L27 (§3.2): the realised
attack-host benign share is 1/4, not 1/1.** 697,298 of 929,764 attack-host
benign cells were dropped by the pre-registered mechanical rule. L27's closing
condition says "all **or a stated share**", so the condition is still met as
written — but every L27 verdict must say "1/4", not "all", and must not be
phrased as though the full rule held.

### 1.2 Arms, systems, seeds

4 arms × 2 systems × 5 seeds = **40 units per execution mode**.

| | |
|---|---|
| Arms | `without_auth` (proc, flows, dns), `without_flows` (auth, proc, dns), `without_dns` (auth, proc, flows), `with_auth` (auth, proc, flows, dns) |
| Systems | `naive` (no discount), `prismflow` (discount) |
| Seeds | 0, 1, 2, 3, 4 |

**What a seed varies** (PRE_REGISTRATION §4), so no ± is misread: the split is
fixed by time and the sample is a deterministic function of cell identity,
non-auth occupancy and the attack-host set. A seed **cannot** redraw or
re-partition the data. It selects the model's parameter initialisation
(`set_seed`) and the minibatch order of every epoch
(`iter_batches(shuffle=True, seed=seed+epoch)`). **Every ± in this document is
initialisation-and-batch-order variance on a fixed sample.** It is not dataset
variance and it is not a spread over independent attacks.

### 1.3 Metrics

Computed in `experiments/lanl_v2/run_lanl_v2.py`, because the frozen
`prismflow.evaluation.compute_metrics` carries no balanced accuracy and no ROC
AUC and cannot be edited (PRE_REGISTRATION §3; Parts 29 and 30 hit the same
wall).

**Primary metric is threshold-free ROC AUC.** Balanced accuracy at the default
0.5 threshold is **secondary** — demoted by dated Amendment 1 §10.1, which
also records that H2's own bar was already written in ROC AUC, so for H2 the
amendment is a confirmation rather than a substitution.

Diagnostics: the auth-presence-only trivial baseline (H1); the
leave-`C17693`-out AUC delta (H2); eigen ENIV and mean pairwise dependence
(eigen is the headline; the design effect is reported beside it and never
gated on); per-row present-view count and the count of all-False-view-mask
rows; size and prevalence per split, per day, and per attack host.

**Declared gap, carried forward:** the design-effect ENIV is **not measured**
by this code path. `prismflow.evaluation.protocol.collect_predictions` returns
only the batch-weighted scalars `eniv`, `mean_dependence` and
`efficiency_ratio`, not the dependence matrix, and it is frozen;
`design_effect_n_eff` needs the full matrix. Every unit JSON records this in
`eniv_design_effect_status`. The headline method is eigen, which is what the
pre-registration gates on, so no hypothesis depends on the gap. Both ENIV
fields are `null` for `naive` by construction — `output.eniv` is `None` when
`use_discount` is false.

### 1.4 Execution modes (L28)

| mode | name | thread count | concurrency |
|---|---|---|---|
| **B** | `threadsdefault_sequential` | machine default, **measured** at 8 on this box | sequential |
| **A** | `threads1_concurrent11` | `torch.set_num_threads(1)` | 11 processes |

Mode B is `--mode default` and is the launched path. Every results file and
score npz carries a `mode` block: mode name, `torch.get_num_threads()` as
observed, concurrency, and the measured machine default.

**Aggregation rule, fixed in PRE_REGISTRATION §6:** per-mode mean ± std over
the 5 seeds, **and** the across-mode spread, reported separately. The two
modes are **never pooled into a single ±**. A reported figure is conditional
on a thread count: ENIV reproduces only at the thread count it was computed
at, so every ENIV value is reported with its mode.

### 1.5 Integrity gate: the Part 38 step-2 sampler audit

The analysis below is conditional on this gate, recorded in
`results/lanl_v2/SWEEP_EXECUTION_LOG.md` §1 as **passed**:

- **Selection pass** against Part 30's committed `population_cells_per_day`
  (`results/lanl/results.json`, `population.per_view.<view>`), summed to the
  same read range — nine of nine exact across days 1–6, 1–7 and 1–16.
- **Records scanned** match Part 30 exactly at 16 days (proc 115,159,740;
  flows 121,284,699; dns 1,055,902).
- **Attack counts** match Part 30 to the row: train 439 over 150 windows, test
  410 over 141.
- **Two independent 11 GB scans** reproduced `per_day`, `selection`,
  `n_samples`, `attack_hosts` and `attack_cells_in_read_range` field for
  field, at peak RSS 2,366.4 MB against 2,367.7 MB.
- **Synthetic audit**, `tests/unit/test_lanl_v2_sampler_synthetic.py`, 8 tests
  passing: `select_cells` compared against a brute-force re-derivation of the
  pre-registered rule **cell by cell**, which is the check a sum cannot give
  because a compensating pair of errors would survive a sum.

Two caveats on the gate, both already recorded and neither resolved:

1. The brief asked for a 10-view toy. The loader's view set is fixed at four
   by `FEATURES`, `SOURCE_COLUMN` and `RECORD_WIDTH` in the frozen
   `prismflow/data/lanl_cyber1.py`, so four real views over 10 hosts was built
   instead.
2. The earlier smoke numbers were **reported as void on account of a sampler
   defect**. Amendment §10.4 records that report as received and not
   confirmed: the §7.1 reproduction gate passes on the committed sampler and
   **no defect has been located**. Nothing in this document rests on those
   smoke numbers.

---

## 2. Tables

### Table 1 — ROC AUC, mean ± std per arm × system (mode `default` / B only)

8 rows. Source: `results/lanl_v2/sweep_default.json`,
`per_cell.<arm>/<system>.roc_auc.{mean,std,n}`. Mode A is a **separate table
and is never pooled into these ±** (PRE_REGISTRATION §6).

| arm | system | n seeds | ROC AUC mean ± std | balanced acc (secondary) | attack recall (secondary) | eigen ENIV | mean dependence |
|---|---|---|---|---|---|---|---|
| `without_auth` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | null by construction | null by construction |
| `without_auth` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_flows` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | null by construction | null by construction |
| `without_flows` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_dns` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | null by construction | null by construction |
| `without_dns` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `with_auth` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | null by construction | null by construction |
| `with_auth` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |

**Fill rule:** a row with `n seeds` below 5 is reported as `n = <k>, NOT A
FINDING` and its ± is withheld. The 5-SEED RULE is not satisfied by a mean
over a partial cell, and `n = 2` std over two values is not a spread.

### Table 2 — Per-seed ROC AUC (40 cells)

Source: `results/lanl_v2/units_default/<arm>_<system>_seed<N>.json`,
field `roc_auc`. The per-cell arrays in `sweep_default.json`
(`per_cell.<cell>.roc_auc.per_seed`) are the same values aggregated.

This table exists for reproducibility, not for averaging: it is what lets a
reader see whether a mean hides a split population (and whether any seed sits
far enough out to drive the ±).

| arm | system | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|---|
| `without_auth` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_auth` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_flows` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_flows` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_dns` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_dns` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `with_auth` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `with_auth` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |

Alongside, the **per-seed sign** of PrismFlow − naive, per arm, from
`sweep_default.json`, `prismflow_minus_naive_roc_auc.<arm>`: report
`seeds_positive`, `seeds_negative` and `seeds_exactly_zero` **before** the
mean. A cancelling population reads as "no effect" if only the mean is shown.
Note `seeds_exactly_zero` is a third category — an "x of 5 agree" gloss that
only counts positives and negatives mislabels it.

### Table 3 — L27 leave-host-out AUC delta (8 rows × 5 hosts = 40 cells)

**Pre-registration status, stated because the two differ.** H2 pre-registers
**one** host, `C17693`, over the eight arm-and-system cells. The four
remaining hosts are **not pre-registered** for Part 38; they are included here
as a descriptive mirror of Part 30's committed table, which has exactly this
40-cell shape (`results/lanl_reanalysis/results.json`, `step5_l27.per_arm`,
4 arms × 2 systems × 5 hosts). **Only the `C17693` column carries H2's bar.**
The other four columns are exploratory and must be labelled as such wherever
they are quoted.

Host set and ranking rule, from Part 30's committed `step5_l27.host_rule`:
"The 5 computers with the most TEST rows of label attack, ranked by that
count, ties broken by computer name ascending." Part 30's rows removed, for
reference:

| host | test rows removed | attack rows removed | benign rows removed |
|---|---|---|---|
| `C17693` | 138 | 137 | 1 |
| `C1015` | 13 | 12 | 1 |
| `C2388` | 13 | 10 | 3 |
| `C1936` | 7 | 7 | 0 |
| `C2079` | 10 | 7 | 3 |

**Part 38's host set must be re-derived on Part 38's own test block, not
copied.** Section 2's rule (b) keeps all benign cells of attack hosts, so the
removed-row counts will differ from the table above by design. Record the
re-derived set and whether it is the same five computers: `[pending]`.

Each cell is `mean(ROC AUC with host removed) − mean(ROC AUC all test)` over
the 5 seeds, matching Part 30's `change_from_all_test`. Report the per-seed
array beside the delta; one host is not a resample, so **no interval is
attached and the result is descriptive** (Part 30's own `interval` field says
exactly this).

| arm | system | all-test AUC | **C17693 Δ** (barred) | C1015 Δ | C2388 Δ | C1936 Δ | C2079 Δ |
|---|---|---|---|---|---|---|---|
| `without_auth` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_auth` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_flows` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_flows` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_dns` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `without_dns` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `with_auth` | naive | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |
| `with_auth` | prismflow | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |

**This table needs no training — confirmed, not assumed.** It is a re-score of
the committed `results/lanl_v2/scores_default/<cell>.npz` arrays with a host
mask applied, and the npz **does** carry the per-row host identifier. Keys
verified on `with_auth_naive_seed0.npz`: `computer` (84460, object),
`window_start` (84460, float64), `label` (84460, int64), `prob_attack`
(84460, float64), `confidence` (84460, float64), plus the scalar
`confidence_convention` and `mode`. So `roc_auc_score(label[~m],
prob_attack[~m])` with `m = (computer == host)` reproduces Part 30's
`change_from_all_test` construction directly, per cell, per seed, at no
training cost.

Note the npz carries `confidence`, **never** `vacuity` — the L29 naming defect
is avoided by this runner and asserted before every write
(`run_lanl_v2.py:444`). Any re-score script must keep that name.

### Table 4 — L24: the auth-presence-only trivial baseline

**Pre-registration status.** The review brief states L24's test as a
"`with_auth` vs `without_auth` AUC gap". That is **not** H1. H1 is a bar on a
**trivial baseline**, not on the fusion arms:

> **H1 (L24 — selection vs view drop).** With control cells drawn without
> reference to the `auth` stream, the *auth-presence-only* trivial baseline on
> the test block falls materially below Part 30's committed test AUC of
> **0.6637** [...]. "Materially" is pre-set at a drop of at least **0.05**
> AUC. That baseline is seed-free, so the bar applies to the single
> recomputed value.

Verified reference: `results/lanl/results.json`,
`phase3_diagnostics.trivial_baselines.baselines.auth_present_indicator.test_auc`
= **0.6636552213548774**. Its sibling baselines, same path, for context:
`total_records_across_views.test_auc` = 0.6842788175411642,
`n_views_present.test_auc` = 0.5624544155807845, all at `n_test` 18,729.
Orientation is fixed on the train block and applied unchanged to test.

| quantity | value | source |
|---|---|---|
| Part 30 `auth_present_indicator` test AUC | **0.6636552213548774** | `results/lanl/results.json` (verified) |
| Pre-registered bar — H1 confirmed if Part 38's value ≤ | **0.6137** (0.6637 − 0.05) | PRE_REGISTRATION §1 H1 |
| Part 38 `auth_present_indicator` test AUC | `[pending]` | `results/lanl_v2/` — **see note below** |
| Drop from Part 30 | `[pending]` | derived |
| H1 verdict | `[pending]` | §3.1 |

**H1 is NOT measured by the sweep as launched. Checked, not assumed.**

- `experiments/lanl_v2/metrics.py:80` defines
  `auth_presence_auc(labels, auth_present)`, which is exactly H1's statistic.
- `run_lanl_v2.py` **never calls it.** Its only metric calls are
  `m.evaluate(labels, prob)` and `m.constant_baseline_brier(labels)` at lines
  448–449. No `auth_present` reference exists anywhere in the runner.
- The baseline is seed-free and is a property of the *sample*, not of a unit,
  so it correctly does not belong in `units_default/*.json` — but no committed
  file carries it either.

**It is cheap to obtain and needs no training and no inference.** Both inputs
are already in the committed sample table
`results/lanl_v2/samples_v2_days16.npz`: `view_mask` (281823, 4) bool,
`labels` (281823,), `block` (281823,) giving train/test. The view order is
fixed by that file's own `meta`, `sources: ["auth", "proc", "flows", "dns"]`,
so **auth presence is `view_mask[:, 0]`**. H1 is therefore
`auth_presence_auc(labels[test], view_mask[test, 0])` — one scalar, seconds of
CPU, using a function that already exists.

**Do not infer H1 from the arm AUCs.** The `with_auth` − `without_auth` gap is
a different quantity (see below) and cannot stand in for the baseline. Until
the scalar pass is run and committed to its own path, H1 is reported as
**not measured**, never as absent evidence either way.

The arm gap the brief asked for is a **different** quantity and is reported
separately, as a descriptive figure with no pre-registered bar:

| quantity | value | source |
|---|---|---|
| `with_auth` − `without_auth` AUC, naive | `[pending]` | Table 1 |
| `with_auth` − `without_auth` AUC, prismflow | `[pending]` | Table 1 |
| per-seed sign counts for each | `[pending]` | Table 2 |
| train/test prevalence, each arm | `[pending]` | `results/lanl_v2/build_v2_days16.json` |

Prevalence is **an artefact of the design**, not population prevalence, and is
now an artefact of two rules rather than one (PRE_REGISTRATION §2). "Prevalence
stability across arms" is therefore a check that the two arms were sampled as
specified — it is not evidence about the network, and it cannot carry a
closing condition.

### Table 5 — L28 cross-mode AUC delta per arm × system

**PENDING MODE A.** Mode A (`threads1_concurrent11`) has not been run. The
sweep execution log records why: it needs an idle box for ~5.4 GB across 11
processes, and the estimate is ~2.0 h. Every cell below is
**pending mode A**, not zero and not missing.

H3 as pre-registered tests the **sign** of the PrismFlow-minus-naive
difference in test ROC AUC, agreeing across both modes, in all four arms.
Part 30 found agreement on `roc_auc` and `prob_ece` but **disagreement** on
`balanced_accuracy`, `attack_recall` and `brier` for `with_auth`
(`results/lanl/execution_mode.json`; L28). H3 is barred on `roc_auc` only and
reported without a bar for the other five metrics.

| arm | system | mode B AUC | mode A AUC | Δ (A − B) | sign of PF − naive, B | sign of PF − naive, A | signs agree |
|---|---|---|---|---|---|---|---|
| `without_auth` | naive | `[pending]` | pending mode A | pending mode A | `[pending]` | pending mode A | pending mode A |
| `without_auth` | prismflow | `[pending]` | pending mode A | pending mode A | `[pending]` | pending mode A | pending mode A |
| `without_flows` | naive | `[pending]` | pending mode A | pending mode A | `[pending]` | pending mode A | pending mode A |
| `without_flows` | prismflow | `[pending]` | pending mode A | pending mode A | `[pending]` | pending mode A | pending mode A |
| `without_dns` | naive | `[pending]` | pending mode A | pending mode A | `[pending]` | pending mode A | pending mode A |
| `without_dns` | prismflow | `[pending]` | pending mode A | pending mode A | `[pending]` | pending mode A | pending mode A |
| `with_auth` | naive | `[pending]` | pending mode A | pending mode A | `[pending]` | pending mode A | pending mode A |
| `with_auth` | prismflow | `[pending]` | pending mode A | pending mode A | `[pending]` | pending mode A | pending mode A |

**Every ENIV value in this table is conditional on its thread count.** Mode B
observed 8 threads; mode A sets 1. A committed ENIV fails a `abs=1e-6` check
when re-run at a different thread count — measured in Part 34 at
`eniv_eigen` 3.4289614712229826 (1 thread) against the committed
3.4396320614332856 (8 threads), a miss of 0.0107. Do not compare ENIV across
the two mode columns without saying which thread count each came from.

### Table 6 — operational record (no hypothesis attached)

Per-unit wall-clock and peak RSS, from `units_default/*.json`
(`unit_seconds`, `peak_rss_mb`), and the degeneracy diagnostic added by
Amendment §10.2 (`rows_at_or_above_threshold`, `max_prob_attack`, and the
per-seed `degenerate_at_threshold_per_seed` flags in `sweep_default.json`).

| arm | system | unit seconds mean ± std | peak RSS MB | rows ≥ 0.5 | max prob_attack | degenerate at 0.5 |
|---|---|---|---|---|---|---|
| (8 rows) | | `[pending]` | `[pending]` | `[pending]` | `[pending]` | `[pending]` |

---

## 3. Verdict templates

Each rule is written here **before** the numbers exist. A result that fails
its rule is reported as a failure, in the same detail as a success.

### 3.1 L24 — closing condition and verdict logic

**Closing condition, verbatim from `docs/KNOWN_LIMITATIONS.md` L24
("On LANL the `auth` view is label-adjacent, and dropping it does not drop the
selection"):**

> **What would change our understanding.** A sample whose control cells are
> drawn without reference to the `auth` stream, which would separate the view
> drop from the selection.

**How the design meets it** (PRE_REGISTRATION §2, "How each closing condition
is met"): rule (a). Selection reads non-auth occupancy, the cell-identity hash
and the attack-host set. No auth record is read until the feature pass, after
selection is fixed.

**Note that the closing condition is a condition on the DESIGN, not on a
number.** It is met by rule (a) being implemented as written, which the
step-2 sampler audit (§1.5) is the evidence for. H1 is a separate,
falsifiable prediction about what that design should produce.

**Verdict logic:**

- **H1 CONFIRMED if** Part 38's `auth_present_indicator` test AUC
  **≤ 0.6137**, i.e. at least 0.05 below Part 30's 0.6636552213548774. The
  baseline is seed-free, so this is a single value and no ± applies.
- **H1 DISCONFIRMED if** the value is > 0.6137. Per PRE_REGISTRATION §1:
  "If it does **not** fall, H1 is disconfirmed and the selection signal has
  another source, which is a reportable finding in its own right." Name the
  recomputed value and state that the source of the remaining separation is
  unidentified.
- **L24 proposed status — NARROWED, not CLOSED, in either branch.** The
  closing condition asks for a design, and the design is delivered; but L24's
  recorded claim is about Part 30's sample, and Part 38's sample "is not a
  subset or a superset of Part 30's, and the two sample tables are **not
  comparable row-for-row**" (PRE_REGISTRATION §2). Any status change goes to
  `docs/proposed_limitations_update.md` for the review gate, never into
  `KNOWN_LIMITATIONS.md` from here (§7.6).

### 3.2 L27 — closing condition and verdict logic

**Closing condition, verbatim from `docs/KNOWN_LIMITATIONS.md` L27
("On LANL one host carries a third of the attack class, with no benign
contrast"):**

> **What would change our understanding.** A design that keeps all or a stated
> share of the benign cells of attack hosts, which would be needed to separate
> host from behaviour.

**How the design meets it:** rule (b), **on its fallback branch**. The §1.1
size fallback fired at `k = 2`, so the realised stated share is **1/4**, not
1/1: 697,298 of 929,764 attack-host benign cells were dropped to bring
979,122 rows under the pre-registered 400,000 bar. The condition says "all
**or a stated share**", so it is met — the share is stated, mechanical, and
was fixed before any count was seen. **Every L27 verdict must read "1/4".**
A verdict that says "all benign cells of attack hosts were retained" is
false on this build.

The 1/4 share also weakens H2's own mechanism in a way the verdict must name:
H2 predicts the leave-host-out sensitivity shrinks because benign contrast for
`C17693` is restored. At `k = 2` three quarters of that restored contrast was
dropped again. So a disconfirmation at `k = 2` does **not** establish that
restoring benign contrast fails to help — it establishes that restoring a
quarter of it did not help. Say which claim the number supports.

**Verdict logic (H2):**

- **H2 CONFIRMED if** all **eight** arm-and-system `C17693` cells move by
  **less than 0.036** in absolute value.
- **H2 DISCONFIRMED if** any one of the eight moves by **0.036 or more**
  (PRE_REGISTRATION §1 H2, verbatim).
- The four non-pre-registered hosts in Table 3 **cannot confirm or disconfirm
  H2** and must not be aggregated with the `C17693` column.

**A sharpness warning that must appear beside the verdict.** Part 30's own
committed `C17693` band is **0.035963 to 0.051875** across the same eight
cells (verified: `results/lanl_reanalysis/results.json`,
`step5_l27.per_arm.<arm>.<system>.per_host.C17693.change_from_all_test`;
all eight are negative). The smallest of those, `without_dns`/`naive` at
−0.035963, is **below the 0.036 bar**. So a Part 38 cell that reproduced
Part 30's weakest cell exactly would be scored as passing. The bar sits on the
edge of the band it was drawn from, and "H2 confirmed" therefore does not by
itself mean the sensitivity shrank. Report the eight Part 38 deltas against
all eight Part 30 deltas pairwise, per arm and system, so the comparison does
not rest on a single threshold.

- **L27 proposed status — NARROWED at most.** Even on a clean confirmation,
  one host is not a resample, no interval attaches (Part 30's own `interval`
  field: "NONE. One host is not a resample, so no interval is attached and the
  result is descriptive"), and rule (b) uses **host-level label knowledge** for
  selection, so per-host prevalence differs between attack and non-attack
  hosts **by construction** (PRE_REGISTRATION §2, "What this design still
  costs"). The figure that matters is the within-attack-host contrast, not the
  overall prevalence.

### 3.3 L28 — closing condition and verdict logic

**Closing condition, verbatim from `docs/KNOWN_LIMITATIONS.md` L28
("On LANL the execution mode changes the fitted weights, and the per-seed std
understates the spread"):**

> **What would change our understanding.** Seeds run across at least two
> execution modes, so the reported std covers mode as well as seed.

**L28 cannot be CLOSED by its own literal text under this pre-registration,
and this must be stated rather than finessed.** The condition asks for a std
that **covers mode as well as seed** — a pooled ±. PRE_REGISTRATION §6
forbids exactly that: "The two modes are **never pooled into a single ±**:
L28's finding is that a 5/5 seed count inside one run shares one execution
mode and is not independent evidence about mode effects." The pre-registration
substitutes H3, a **sign-agreement** test, and that is the only barred L28
analysis Part 38 can produce. Satisfying L28's literal wording would require a
second dated amendment and the review gate's agreement that pooling across two
modes is legitimate; it is not a judgement this document can make.

**Verdict logic (H3), pending mode A:**

- **H3 CONFIRMED if** the sign of the PrismFlow-minus-naive difference in test
  ROC AUC agrees across modes A and B **in all four arms**.
- **H3 DISCONFIRMED if** the sign flips in any arm. Name the arm, both signed
  values, and the per-seed sign counts on each side.
- A sign that rests on a difference smaller than the per-seed std is reported
  as **sign undetermined**, not as agreement. A cell with
  `seeds_exactly_zero > 0` is reported with that count, because an exact zero
  has no sign.
- The other five metrics (`balanced_accuracy`, `attack_recall`,
  `benign_recall`, `brier`, `prob_ece`) are reported **without a bar**, which
  is where Part 30's disagreement lives.

**On the review brief's proposed bar.** The brief proposes "mode A vs mode B
per-seed AUC agreement within 3 stds". That bar is **already failed by the
prior record** and so cannot be a closing condition: L28 records the same seed
differing by up to **0.0140** in AUC across modes (naive seed 0: 0.8375 and
0.8235), which is "about **4** per-seed standard deviations of that same arm
and system (0.0033)". A 3-std bar would score Part 30's own observation as
disconfirming. If a magnitude bar is wanted it must be pre-registered at a
value chosen before mode A runs and justified against that 4-std figure.

- **L28 proposed status — NARROWED at best, and only once mode A has run.**
  With mode A unrun, L28's status is unchanged and Table 5 stays
  `pending mode A`.

---

## 4. Still-OPEN clauses

For each limitation, the exact figure to name if the condition is not met, and
what would be needed. Fill the bracketed figure with the value that failed —
never with a summary adjective.

### L24 still OPEN if

- `auth_present_indicator` test AUC is `[pending]` > 0.6137 → **state the
  value and the drop achieved**, and record that the selection signal has an
  unidentified second source. Needed next: a decomposition of the remaining
  separation — which of `proc`, `flows`, `dns` occupancy, the cell-identity
  hash, or the attack-host rule carries it. That is a new scalar-baseline pass
  over the sample table, not a retrain.
- **The baseline is not written by the sweep at all — this is the confirmed
  state today.** L24 is OPEN for the reason *not measured*, not for a negative
  result. Needed next: a one-off scalar pass,
  `auth_presence_auc(labels[test], view_mask[test, 0])` over
  `results/lanl_v2/samples_v2_days16.npz`, committed to its own path. No
  training, no inference, no dependence on the sweep finishing. Report it as
  missing instrumentation.

### L27 still OPEN if

- Any `C17693` cell moves by `[pending]` ≥ 0.036 → **name the arm, the system,
  the delta and the per-seed array**. H2 is disconfirmed; restoring benign
  contrast for that host did not shrink the leave-host-out sensitivity, and the
  concentration of the attack class in one host remains a live threat to every
  per-arm AUC.
- All eight pass but the smallest Part 30 cell (−0.035963) is not cleared by a
  margin → report H2 as **confirmed at a bar that Part 30's own weakest cell
  also passes**, which is a weak confirmation. Needed next: a bar set from the
  *median* or the *largest* of Part 30's band rather than its minimum, agreed
  at the review gate.
- Part 38's re-derived top-5 host set differs from Part 30's → the two tables
  are not row-comparable; say so and report both host sets.
- The realised share is **1/4** (`k = 2`), so L27's "separate host from
  behaviour" is tested at a quarter of the intended benign contrast whatever
  the deltas say. Needed to test it at 1/1: raising the 400,000 sample bar,
  which is a pre-registered quantity and therefore a review-gate decision and
  a new dated amendment — not a parameter to change quietly. The 979,122-row
  uncapped build is the measured alternative; its cost is a full rebuild plus
  a 3.5x larger sweep.

### L28 still OPEN if

- Mode A has not run → **L28 remains OPEN, unchanged, and this is the expected
  state of this document on first fill.** Needed next: ~2.0 h on an idle box
  with ~5.4 GB free across 11 processes.
- The sign flips in any arm → name the arm and both signed values. H3
  disconfirmed; the mode dependence reaches the primary metric and not only the
  threshold metrics, which is strictly worse than what L28 records.
- Signs agree → L28 is **narrowed, not closed**, because the closing condition
  asks for a pooled std that §6 forbids. Needed next: a review-gate decision on
  whether a mode-pooled ± is legitimate, and if so a second dated amendment.

### Common to all three

Any cell with fewer than 5 completed seeds is **not a finding** and no verdict
may be issued from it. A verdict issued on a partial sweep must name the
missing units.

---

## 5. Range-dependence note

**Recorded here because it bears on how Table 1's secondary columns are read.**
Source: `results/lanl_v2/SWEEP_EXECUTION_LOG.md` §2, which is a dated
operational record and amends nothing.

Amendment §10.3's premise **holds on the reduced ranges and not in the range
the sweep runs on.**

| range | test ROC AUC (seed 0 / seed 1) | balanced accuracy | attack recall |
|---|---|---|---|
| days 1–6 | 0.7319 / 0.7189 | 0.5000 | 0.0000 |
| days 1–7 | 0.7128 / 0.7089 | 0.5000 | 0.0000 |
| **days 1–16** | **0.7201 / 0.7463** | **0.5512 / 0.5549** | **0.1024 / 0.1098** |

Evidence: `results/lanl_v2/smoke.json`, `smoke_days7.json`,
`smoke_days16.json`. All three cells are `with_auth`, `naive`, mode `default`,
seeds 0 and 1 — **two seeds, one arm, one system, and therefore not findings
about any hypothesis.**

Amendment §10.3 states that the model "emits an attack probability below 0.5
on every test row", which pins balanced accuracy at exactly 0.5. That is true
at days 1–6 (max `prob_attack` 0.158440, zero rows at or above 0.5) and at
days 1–7. **At days 1–16 it is false:** balanced accuracy moves off 0.5 and
attack recall is roughly 0.10. **The sweep runs on days 1–16, so the structural
premise does not hold in the target range.**

What this does and does not change:

- **The primary metric is unaffected and stands.** Threshold-free ROC AUC is
  justified by **L16** ("rank well and threshold badly") and **L23**
  ("weighted precision collapses at the default threshold"), both recorded
  before Part 38 and neither contingent on the model being degenerate. What
  the finding removes is one supporting observation, not the reason.
- **The §10.2 degeneracy diagnostic is unaffected.** It is defined as a
  measurement per unit, and on days 1–16 it will simply read
  "not degenerate" — which is a reading, not a failure.
- **The amendment remains dated and frozen.** Correcting §10.3 would mean
  editing a dated amendment after numbers were looked at, which the
  pre-registration forbids. A second dated amendment is the user's call.
- **Consequence for this document:** do not describe Table 1's balanced
  accuracy and attack recall columns as "pinned" or "degenerate by
  construction". On the full range they move, so they carry some information —
  they are secondary because of L16 and L23, not because they are constant.

---

## 6. Fill checklist

- [ ] Every `[pending]` replaced by a value with its JSON path, or by
      `not measured` with the reason.
- [ ] Every cell has `n = 5`; any `n < 5` cell labelled `NOT A FINDING`.
- [ ] Per-seed sign counts reported before every aggregate mean.
- [ ] No ± pools the two execution modes.
- [ ] Every ENIV value carries its thread count.
- [ ] Mode A tables either filled or left `pending mode A` — never blank.
- [ ] Table 3's `C17693` column distinguished from the four exploratory hosts.
- [ ] No number in this file lacks a committed source.
- [ ] `PRE_REGISTRATION.md` and `docs/KNOWN_LIMITATIONS.md` unedited.
- [ ] Status proposals written to `docs/proposed_limitations_update.md` only.
