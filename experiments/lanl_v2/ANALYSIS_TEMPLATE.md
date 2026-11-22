# Part 38 — LANL redesign: ANALYSIS TEMPLATE

**Mode B is complete at 40/40. H1 DISCONFIRMED, H2 DISCONFIRMED, H3 pending
mode A.** Tables 1, 2, 3 and 6 are filled from committed files; Table 4 is the
three scalar passes; **Table 5 is the only `pending mode A` table**, and mode
A has not been run.

Every cell marked `[pending]` is unfilled and must be filled from a committed
file by its exact path, or left as `not measured`. A `[pending]` that is still
present when this document is read as a report is a defect in the report, not
a zero.

Reference numbers quoted below are **measurements from earlier committed
Parts**, each with its JSON path. They are the bars, not this Part's results.

- **Status:** filled 2026-11-22 at 40 of 40 units, mode `default` (B), 8
  threads observed. Source for Tables 1, 2, 3 and 6:
  `results/lanl_v2/filled_tables.json`, written by
  `experiments/lanl_v2/fill_analysis.py` behind its 40-unit gate, over
  `results/lanl_v2/sweep_default.json` and `units_default/*.json`. Every
  arm × system cell is `n = 5`.
- **H1 is DISCONFIRMED** and does not depend on the sweep: it is a seed-free
  scalar over the committed sample table
  (`results/lanl_v2/h1_auth_presence.json`), computed with no training and no
  inference. §3.1.
- **H2 is DISCONFIRMED**, 8 of 8 cells at or above the bar, with Part 38's
  band not overlapping Part 30's and the effect in the **opposite direction**
  to the prediction. §3.2.
- **PrismFlow − naive is negative on one arm and sign-undetermined on three.**
  No arm favours the discount. Table 2.
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

**FILLED 2026-11-22 at 40/40.** Every cell is `n = 5`, so the 5-SEED RULE is
satisfied for every row. All ENIV values were computed at **8 threads**
(`torch_threads_observed`).

| arm | system | n seeds | ROC AUC mean ± std | balanced acc (secondary) | attack recall (secondary) | eigen ENIV (8 threads) | mean dependence |
|---|---|---|---|---|---|---|---|
| `without_auth` | naive | 5 | **0.7900364591 ± 0.0020106638** | 0.6768114218 ± 0.0055826702 | 0.3536585366 ± 0.0111770139 | null by construction | null by construction |
| `without_auth` | prismflow | 5 | **0.7868264419 ± 0.0035939990** | 0.6777763236 ± 0.0045564977 | 0.3556097561 ± 0.0091909481 | 2.1970654181 ± 0.0069142833 | 0.0880698755 ± 0.0029801289 |
| `without_flows` | naive | 5 | **0.7700035490 ± 0.0040878521** | 0.5490339084 ± 0.0100715333 | 0.1185365854 ± 0.0201570278 | null by construction | null by construction |
| `without_flows` | prismflow | 5 | **0.7677030484 ± 0.0038525715** | 0.5531766805 ± 0.0068354876 | 0.1268292683 ± 0.0136889904 | 2.1590483493 ± 0.0207787271 | 0.2254336037 ± 0.0146221517 |
| `without_dns` | naive | 5 | **0.7440774046 ± 0.0084693584** | 0.5531231410 ± 0.0054124376 | 0.1102439024 ± 0.0108255322 | null by construction | null by construction |
| `without_dns` | prismflow | 5 | **0.7447516055 ± 0.0082767688** | 0.5501986913 ± 0.0067016113 | 0.1043902439 ± 0.0134035431 | 2.4144224019 ± 0.0298687808 | 0.2014200679 ± 0.0155461106 |
| `with_auth` | naive | 5 | **0.7401838830 ± 0.0122072868** | 0.5534098751 ± 0.0068643974 | 0.1068292683 ± 0.0137323788 | null by construction | null by construction |
| `with_auth` | prismflow | 5 | **0.7419867355 ± 0.0092586512** | 0.5578001190 ± 0.0039122871 | 0.1156097561 ± 0.0078277105 | 2.8196540797 ± 0.0201698075 | 0.1940751504 ± 0.0054505720 |

Source for every cell: `results/lanl_v2/filled_tables.json`,
`table1_roc_auc_per_cell.per_cell.<arm>/<system>.<field>.{mean,std,n}`, which
aggregates `results/lanl_v2/sweep_default.json` over the five per-seed units.

**Fill rule (satisfied):** a row with `n seeds` below 5 would be reported as
`n = <k>, NOT A FINDING` with its ± withheld. No row needed that.

**Read the arm ordering before the system columns.** The best arm is
`without_auth` (0.7900 naive) and the worst is `with_auth` (0.7402 naive): on
this sample **removing the auth view raises ROC AUC by about 0.050**. That is
the opposite direction from Part 30, where auth presence was the strongest
trivial signal, and it is consistent with §4.1 — the selection now runs
through `proc`, and auth's features add noise to a ranking that no longer
needs them. `without_auth` also carries by far the highest attack recall
(0.354 against 0.107–0.127 elsewhere) and the highest balanced accuracy
(0.677 against 0.549–0.558).

The two secondary columns are **not** pinned: balanced accuracy sits at
0.549–0.678 and attack recall at 0.104–0.356, both well off the degenerate
values. See §5 — Amendment §10.3's premise does not hold in this range, and
these columns are secondary because of L16 and L23, not because they are
constant.

### Table 2 — Per-seed ROC AUC (40 cells)

Source: `results/lanl_v2/units_default/<arm>_<system>_seed<N>.json`,
field `roc_auc`. The per-cell arrays in `sweep_default.json`
(`per_cell.<cell>.roc_auc.per_seed`) are the same values aggregated.

This table exists for reproducibility, not for averaging: it is what lets a
reader see whether a mean hides a split population (and whether any seed sits
far enough out to drive the ±).

**FILLED 2026-11-22 at 40/40.**

| arm | system | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|---|
| `without_auth` | naive | 0.7869803253 | 0.7912215580 | 0.7891339940 | 0.7908043122 | 0.7920421062 |
| `without_auth` | prismflow | 0.7828828224 | 0.7867918341 | 0.7855183761 | 0.7862734580 | 0.7926657187 |
| `without_flows` | naive | 0.7632432495 | 0.7718997838 | 0.7710997519 | 0.7740374487 | 0.7697375111 |
| `without_flows` | prismflow | 0.7614367754 | 0.7679026712 | 0.7680550195 | 0.7692167844 | 0.7719039915 |
| `without_dns` | naive | 0.7308022083 | 0.7432551327 | 0.7440861856 | 0.7488014538 | 0.7534420423 |
| `without_dns` | prismflow | 0.7325428534 | 0.7472057428 | 0.7416364678 | 0.7475516606 | 0.7548213027 |
| `with_auth` | naive | 0.7201082109 | 0.7463095283 | 0.7373706127 | 0.7466818241 | 0.7504492390 |
| `with_auth` | prismflow | 0.7273084981 | 0.7392864439 | 0.7442268394 | 0.7490343001 | 0.7500775961 |

Source: `results/lanl_v2/filled_tables.json`,
`table2_per_seed.per_cell.<arm>/<system>.seed<N>`, read from
`results/lanl_v2/units_default/<arm>_<system>_seed<N>.json`, field `roc_auc`.

**No cell hides a split population.** Within every cell the five values span
at most 0.0303 (`with_auth`/naive, 0.7201 to 0.7504) and the spread is
monotone in nothing — seed 0 is the lowest value in 7 of 8 cells, which is a
seed effect and not a bimodality.

### Per-seed signs of PrismFlow − naive, reported BEFORE the means

Source: `filled_tables.json`,
`table2_per_seed.prismflow_minus_naive.<arm>`.

| arm | seeds + | seeds − | seeds exactly 0 | mean | std | reading |
|---|---|---|---|---|---|---|
| `without_auth` | 1 | **4** | 0 | **−0.0032100173** | 0.0021726225 | **negative**, \|mean\| > std |
| `without_flows` | 1 | **4** | 0 | −0.0023005006 | 0.0027377719 | **sign undetermined**, \|mean\| < std |
| `without_dns` | **3** | 2 | 0 | +0.0006742009 | 0.0025411558 | **sign undetermined**, \|mean\| < std |
| `with_auth` | **3** | 2 | 0 | +0.0018028525 | 0.0058650782 | **sign undetermined**, \|mean\| < std |

Per-seed differences, in seed order:

| arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| `without_auth` | −0.0040975029 | −0.0044297239 | −0.0036156179 | −0.0045308542 | +0.0006236125 |
| `without_flows` | −0.0018064741 | −0.0039971126 | −0.0030447324 | −0.0048206642 | +0.0021664805 |
| `without_dns` | +0.0017406451 | +0.0039506101 | −0.0024497178 | −0.0012497932 | +0.0013792603 |
| `with_auth` | +0.0072002873 | −0.0070230844 | +0.0068562267 | +0.0023524760 | −0.0003716429 |

**PrismFlow does not beat naive on any arm of this sample, and on one arm it
loses.** Only `without_auth` has `|mean| > std`, and its sign is **negative**:
PrismFlow is worse by 0.0032 ± 0.0022, negative in 4 of 5 seeds. The other
three arms are **sign undetermined** by this document's own rule (§3.3) — the
difference is smaller than the per-seed std, so neither direction is
established. `seeds_exactly_zero` is 0 in all four arms, so no "x of 5" gloss
is being misread here.

Stated plainly: on four arms, five seeds and one execution mode, the
dependence discount **changed nothing measurable in ranking quality**, and the
one arm where a direction is established runs against PrismFlow. The
largest-magnitude arm, `with_auth`, has the largest std (0.0059) and flips
sign twice across seeds — exactly the cancelling population the per-seed
reporting rule exists to expose.

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
re-derived set and whether it is the same five computers: **RE-DERIVED, and it
is the same five computers, in the same rank order.** Source:
`results/lanl_v2/filled_tables.json`, `table3_leave_host_out.hosts`.

| host | test rows removed | attack rows removed | benign rows removed | Part 30 benign |
|---|---|---|---|---|
| `C17693` | 147 | 137 | **10** | 1 |
| `C1015` | 270 | 12 | 258 | 1 |
| `C2388` | 295 | 10 | 285 | 3 |
| `C1936` | 205 | 7 | 198 | 0 |
| `C2079` | 8 | 7 | **1** | 3 |

**Rule (b) did restore benign contrast, and the figures say how much.**
`C17693` goes from 1 benign row to **10**, and the three mid-rank hosts from
0–3 rows to 198–285. The attack counts are unchanged for every host (137, 12,
10, 7, 7), as they must be — both designs keep every attack cell and the test
range is identical. `C2079` is the exception that proves the mechanism: it
gained no benign rows (3 → 1), and it is the one host whose removal barely
moves any cell.

**But 10 benign rows against 137 attack rows is still not a contrast.**
`C17693`'s within-host prevalence is 137/147 = 0.932 against a test-block
prevalence of 0.00485. This is the `k = 2` share showing up in the only place
it matters. (Arithmetic on the stated 1/4 share, **not a measurement**: a 1/1
build would have kept on the order of four times as many benign rows for this
host. What that build would actually have measured is **not measured** — it
was never run.)

Each cell is `mean(ROC AUC with host removed) − mean(ROC AUC all test)` over
the 5 seeds, matching Part 30's `change_from_all_test`. Report the per-seed
array beside the delta; one host is not a resample, so **no interval is
attached and the result is descriptive** (Part 30's own `interval` field says
exactly this).

**FILLED 2026-11-22 at 40/40.** Source: `results/lanl_v2/filled_tables.json`,
`table3_leave_host_out.per_cell.<arm>/<system>`:
`roc_auc_all_test.mean` and `per_host.<host>.change_from_all_test`.
The four right-hand columns are **exploratory, not pre-registered**, and must
be labelled as such wherever they are quoted.

| arm | system | all-test AUC | **C17693 Δ** (barred) | C1015 Δ | C2388 Δ | C1936 Δ | C2079 Δ |
|---|---|---|---|---|---|---|---|
| `without_auth` | naive | 0.790036 | **−0.099710** | −0.000511 | +0.002777 | +0.004168 | −0.003648 |
| `without_auth` | prismflow | 0.786826 | **−0.099784** | −0.000729 | +0.003972 | +0.004168 | −0.003703 |
| `without_flows` | naive | 0.770004 | **−0.103284** | −0.003892 | +0.006326 | +0.003083 | −0.003407 |
| `without_flows` | prismflow | 0.767703 | **−0.103105** | −0.003437 | +0.007529 | +0.002827 | −0.003409 |
| `without_dns` | naive | 0.744077 | **−0.108785** | +0.006904 | +0.003563 | +0.001708 | −0.004273 |
| `without_dns` | prismflow | 0.744752 | **−0.104055** | +0.004857 | +0.005311 | +0.002298 | −0.004225 |
| `with_auth` | naive | 0.740184 | **−0.113438** | +0.006529 | +0.003075 | +0.005185 | −0.004440 |
| `with_auth` | prismflow | 0.741987 | **−0.108746** | +0.003993 | +0.005275 | +0.005560 | −0.004366 |

**Pairwise against Part 30, as §3.2 requires** — all eight Part 30 deltas are
negative, all eight Part 38 deltas are negative and **larger in magnitude**:

| arm | system | Part 38 Δ | Part 30 Δ | growth in magnitude, \|Part 38\| − \|Part 30\| |
|---|---|---|---|---|
| `without_auth` | naive | −0.099710 | −0.051875 | **+0.047835** |
| `without_auth` | prismflow | −0.099784 | −0.050799 | **+0.048984** |
| `without_flows` | naive | −0.103284 | −0.043474 | **+0.059810** |
| `without_flows` | prismflow | −0.103105 | −0.043361 | **+0.059744** |
| `without_dns` | naive | −0.108785 | −0.035963 | **+0.072822** |
| `without_dns` | prismflow | −0.104055 | −0.036776 | **+0.067279** |
| `with_auth` | naive | −0.113438 | −0.037837 | **+0.075601** |
| `with_auth` | prismflow | −0.108746 | −0.040041 | **+0.068705** |

Part 30 band: 0.035963 to 0.051875. Part 38 band: **0.099710 to 0.113438** —
the two bands do not overlap, and Part 38's smallest cell is nearly twice
Part 30's largest. The four exploratory hosts move every cell by at most
0.0075 in absolute value, two orders of magnitude less than `C17693`.

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
| Part 38 `auth_present_indicator` test AUC | **0.6611838192** | `results/lanl_v2/h1_auth_presence.json`, `baselines.auth_present_indicator.test_auc` |
| Drop from Part 30 | **0.0024714022** | same file, `h1.drop_achieved` |
| **H1 verdict** | **DISCONFIRMED** | same file, `h1.verdict` |

**H1 is MEASURED, by a separate scalar pass, not by the sweep.**
`experiments/lanl_v2/h1_auth_presence.py` →
`results/lanl_v2/h1_auth_presence.json`.

The sweep does not compute it: `metrics.py:80` defines `auth_presence_auc` and
`run_lanl_v2.py` never calls it (its only metric calls are `m.evaluate` and
`m.constant_baseline_brier`, lines 448–449). The baseline is seed-free and is
a property of the *sample*, not of a trained unit, so it correctly does not
belong in `units_default/*.json`. The pass reads
`results/lanl_v2/samples_v2_days16.npz` alone and is **independent of the
sweep and of its execution mode** — no thread count applies, no ENIV is
involved, and it needed no training and no inference.

Orientation follows Part 30's rule exactly, by importing
`trivial_scalar_values` and `roc_auc_binary` from the frozen
`experiments/lanl/run_lanl.py` rather than re-implementing them: the sign is
chosen on the TRAIN block only and applied unchanged to test. Auth is column
**0** of `["auth", "proc", "flows", "dns"]`, fixed by the sample table's own
`meta.sources`. Verified: the pass re-runs bit-exact, and is unchanged under
`OMP_NUM_THREADS=1`.

**All three of Part 30's scalars, on Part 38's table.** Only
`auth_present_indicator` carries H1's bar; the other two are what make the
disconfirmation readable. Train rows 197,363, test rows 84,460, test
prevalence 0.0048543689.

| scalar | orientation | train AUC | Part 38 test AUC | Part 30 test AUC | change |
|---|---|---|---|---|---|
| **`auth_present_indicator`** (barred) | as-is | 0.6675392537 | **0.6611838192** | 0.6636552214 | **−0.0024714022** |
| `total_records_across_views` | as-is | 0.6918403740 | 0.5862805386 | 0.6842788175 | −0.0979982790 |
| `n_views_present` | as-is | 0.5645444187 | 0.4800433975 | 0.5624544156 | −0.0824110181 |

**Read the three together, because that is the finding.** The two
*volume/occupancy* scalars fell hard under the new selection — record volume by
0.098, and views-present by 0.082 to 0.4800, i.e. below chance. The
*auth-presence* scalar did **not** move: it fell by 0.0025, a fiftieth of the
required 0.05. So the new selection rule demonstrably changed what the
occupancy signals can do, and left auth presence intact.

**Do not infer H1 from the arm AUCs.** The `with_auth` − `without_auth` gap is
a different quantity (below) and cannot stand in for the baseline.

The arm gap the brief asked for is a **different** quantity and is reported
separately, as a descriptive figure with no pre-registered bar:

| quantity | value | source |
|---|---|---|
| `with_auth` − `without_auth` AUC, naive | **−0.0498525761** (0.7401838830 − 0.7900364591) | Table 1 |
| `with_auth` − `without_auth` AUC, prismflow | **−0.0448397063** (0.7419867355 − 0.7868264419) | Table 1 |
| per-seed sign counts for each | **5 of 5 negative in both systems** — naive per-seed −0.0668721144, −0.0449120297, −0.0517633813, −0.0441224881, −0.0415928672; prismflow −0.0555743242, −0.0475053902, −0.0412915367, −0.0372391579, −0.0425881226 | Table 2 |
| train/test prevalence, each arm | identical across arms by construction — test 0.0048543689320388345 on 84,460 rows, 410 attack; `n_train` 197,363 | `results/lanl_v2/h1_auth_presence.json` (`test_prevalence`, `n_train`, `n_test`); the arms differ only in the view mask, not in the sampled rows |

**The gap has the opposite sign to Part 30's framing, in 10 of 10 seed-pairs.**
Dropping the auth view **raises** test ROC AUC by 0.0499 (naive) and 0.0448
(prismflow), negative in every one of the five paired seeds in both systems.
Under Part 30 the `without_auth` arm was the *handicapped* arm; on Part 38's
sample it is the strongest. This is descriptive and carries **no
pre-registered bar** — it is not H1, and it cannot stand in for H1.

Note the arms share one sampled table, so the prevalence row is a check that
both arms were drawn as specified, not evidence about the network, and it
cannot carry a closing condition.

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
**Mode B columns filled 2026-11-22 at 40/40; mode A columns are the only
`pending mode A` cells left in this document.** Mode B AUC is Table 1; the
sign column is Table 2's per-arm `prismflow_minus_naive` under §3.3's rule — a
difference smaller than the per-seed std is **sign undetermined**, not
agreement.

| arm | system | mode B AUC (8 threads) | mode A AUC | Δ (A − B) | sign of PF − naive, B | sign of PF − naive, A | signs agree |
|---|---|---|---|---|---|---|---|
| `without_auth` | naive | 0.7900364591 ± 0.0020106638 | pending mode A | pending mode A | **negative** (−0.0032100173, std 0.0021726225, 4/5 seeds −) | pending mode A | pending mode A |
| `without_auth` | prismflow | 0.7868264419 ± 0.0035939990 | pending mode A | pending mode A | **negative** (same arm-level figure) | pending mode A | pending mode A |
| `without_flows` | naive | 0.7700035490 ± 0.0040878521 | pending mode A | pending mode A | **sign undetermined** (−0.0023005006, std 0.0027377719) | pending mode A | pending mode A |
| `without_flows` | prismflow | 0.7677030484 ± 0.0038525715 | pending mode A | pending mode A | **sign undetermined** | pending mode A | pending mode A |
| `without_dns` | naive | 0.7440774046 ± 0.0084693584 | pending mode A | pending mode A | **sign undetermined** (+0.0006742009, std 0.0025411558) | pending mode A | pending mode A |
| `without_dns` | prismflow | 0.7447516055 ± 0.0082767688 | pending mode A | pending mode A | **sign undetermined** | pending mode A | pending mode A |
| `with_auth` | naive | 0.7401838830 ± 0.0122072868 | pending mode A | pending mode A | **sign undetermined** (+0.0018028525, std 0.0058650782) | pending mode A | pending mode A |
| `with_auth` | prismflow | 0.7419867355 ± 0.0092586512 | pending mode A | pending mode A | **sign undetermined** | pending mode A | pending mode A |

**H3 cannot be confirmed even if mode A runs, and the mode B column is why.**
H3 tests whether the sign of PrismFlow − naive agrees across modes in all four
arms. On mode B **three of the four arms have no established sign**: the
difference is smaller than the per-seed std, which §3.3 requires be reported
as *sign undetermined*. A sign that does not exist on one side cannot agree or
disagree with the other. So mode A could at best decide the `without_auth`
arm, and H3's "all four arms" condition is **unreachable on this sample**
whatever mode A shows. This is a property of the measured effect sizes, not of
mode A's absence, and it is new information as of 40/40.

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

**FILLED 2026-11-22 at 40/40.** Source: `results/lanl_v2/filled_tables.json`,
`table6_operational.per_cell.<arm>/<system>`.

**`peak_rss_mb` is NOT aggregated, and the column below is not a mean.** It is
a **running maximum over the life of the process** that wrote the unit, so it
is non-decreasing within a launch and a unit's figure can be inherited from an
earlier unit in the same process. The column reports `n_distinct` and the
distinct values. Launch identity is **not measured** — the unit JSON records
no `process_id` and no `launch_time`, and file mtime is not a substitute
because a `git checkout` rewrites it.

| arm | system | unit seconds mean ± std | peak RSS MB — distinct values (not a mean) | rows ≥ 0.5, per seed | max prob_attack mean | degenerate at 0.5 |
|---|---|---|---|---|---|---|
| `without_auth` | naive | 796.5 ± 241.4 | 3: 513.3, 513.8, 523.1 | 145, 146, 147, 158, 144 | 0.915352 | **False** |
| `without_auth` | prismflow | 1291.5 ± 344.0 | 2: 513.8, 514.3 | 145, 149, 147, 143, 169 | 0.862739 | **False** |
| `without_flows` | naive | 714.0 ± 170.3 | 1: 513.8 | 1772, 1768, 1771, 1780, 1754 | 0.973320 | **False** |
| `without_flows` | prismflow | 884.5 ± 3.0 | 1: 513.8 | 1769, 1765, 1772, 1775, 1784 | 0.981022 | **False** |
| `without_dns` | naive | 546.0 ± 1.3 | 1: 513.8 | 382, 377, 389, 379, 379 | 0.993766 | **False** |
| `without_dns` | prismflow | 1845.8 ± 272.6 | 1: 513.8 | 377, 380, 376, 387, 372 | 0.987943 | **False** |
| `with_auth` | naive | 1051.3 ± 349.4 | 4: 496.7, 499.5, 499.6, 500.1 | 42, 45, 48, 52, 36 | 0.994147 | **False** |
| `with_auth` | prismflow | 1929.8 ± 410.7 | **5**: 501.7, 503.1, 528.8, 537.3, 538.6 | 47, 46, 51, 53, 44 | 0.988934 | **False** |

**The `peak_rss_mb` column is self-documenting about the defect it replaced.**
Three cells have `n_distinct = 1` — a single observation shared by five
units — where the old aggregation would have printed a mean with `std = 0.0`
and read as perfect reproducibility. `with_auth`/`prismflow` has
`n_distinct = 5` and is strictly increasing (501.7 → 538.6), which is the
running maximum visible directly: its seeds 2, 3 and 4 were trained in one
process on 2026-11-22 and each inherited the previous unit's peak.

**`unit_seconds` ± spans launches and is not a reproducibility figure.**
Compare `without_dns`/naive at **546.0 ± 1.3** (one launch) with
`with_auth`/prismflow at **1929.8 ± 410.7** (three launches) — the second ±
is dominated by machine state, not by seed. `with_auth`/naive is the clearest
case: its five seeds measured 653.8, 705.4, 1359.8, 1370.8 and 1166.7 seconds
across three launches, so its ± 349.4 is a cross-launch spread. Each value is
still a genuine per-unit measurement, so the mean remains a legitimate
*operational* summary; it is not evidence about determinism. Table 6 carries
no hypothesis, so nothing in §3 depends on either column.

**The §10.2 degeneracy diagnostic reads "not degenerate" in all 8 cells.**
`max_prob_attack` runs 0.862739 to 0.994147, every cell well above 0.5, and
`degenerate_any_seed` is `False` for all eight. This is the measured
confirmation of §5: Amendment §10.3's premise does not hold at days 1–16. It
is **a reading, not a failure**.

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

**Verdict logic, and the branch that fired:**

- **H1 CONFIRMED if** Part 38's `auth_present_indicator` test AUC
  **≤ 0.6136552214**, i.e. at least 0.05 below Part 30's 0.6636552213548774.
  The baseline is seed-free, so this is a single value and no ± applies.
- **H1 DISCONFIRMED if** the value is above that bar.

**H1 IS DISCONFIRMED. Measured test AUC 0.6611838192, a drop of 0.0024714022
against the 0.05 required** (`results/lanl_v2/h1_auth_presence.json`, `h1`).

Per PRE_REGISTRATION §1, this outcome was pre-declared as a finding, not a
null result: "If it does **not** fall, H1 is disconfirmed and the selection
signal has another source, which is a reportable finding in its own right."

**What the disconfirmation says, stated no more strongly than the measurement
supports.** Part 38's selection never consults the auth stream — rule (a) and
the §1.5 sampler audit are the evidence for that — and auth presence still
ranks the test block at 0.661. Under Part 30 the explanation on offer was the
selection rule: 410/410 attack cells had an auth record while about a third of
benign cells did not, so auth presence separated the classes. That explanation
is now **excluded for Part 38's sample**, and the separation is still there. The
association between auth presence and the attack label therefore looks like a
property of the data rather than an artefact of how controls were drawn.

Two limits on that reading, both of which must travel with it:

1. This is **one scalar on one sample table**, not a decomposition. Which
   non-auth signal carries the remaining separation is **not measured**. The
   drop in `total_records_across_views` (−0.098) and `n_views_present`
   (−0.082) shows the occupancy signals moved, which is evidence the rule did
   something, not evidence about what is left.
2. Part 38's sample "is not a subset or a superset of Part 30's, and the two
   sample tables are **not comparable row-for-row**" (PRE_REGISTRATION §2).
   The 0.6637 and the 0.6612 are two designs' values, so the *small
   difference* between them is not a measured effect size — only the
   pass/fail against the pre-registered bar is.
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

**H2 IS DISCONFIRMED. All eight of eight cells are at or above the bar, and
the smallest exceeds it by a factor of 2.8** (`results/lanl_v2/filled_tables.json`,
`table3_leave_host_out.h2_verdict`): `max_abs_delta` **0.1134384687**
(`with_auth`/naive), `min_abs_delta` **0.0997097614**
(`without_auth`/naive), against a bar of 0.036. `cells_at_or_above_bar` lists
all eight. The rule is existential, so one cell would have sufficed; eight did.

**The sharpness warning does not apply to this verdict, and that is worth
saying explicitly.** §3.2 warns that H2's bar sits inside Part 30's own band
(0.035963 to 0.051875), so a *confirmation* would have been uninformative.
That caveat protects against a false pass. Part 38 did not pass: its band is
**0.099710 to 0.113438**, which does not overlap Part 30's at all, and its
smallest cell is nearly twice Part 30's largest. The verdict does not rest on
the threshold — it rests on two non-overlapping bands, which is what §3.2's
pairwise requirement was for.

**The direction is the opposite of what H2 predicted, in all eight cells.**
H2 predicted the leave-host-out sensitivity would **shrink** because benign
contrast for `C17693` was restored. It **grew**, by +0.0478 to +0.0756 per
cell (Table 3, pairwise block). Removing one host now costs 0.10 to 0.11 AUC
where it cost 0.036 to 0.052 before.

**What this disconfirmation licenses, and what it does not.**

1. **It licenses:** on this build, restoring **a quarter** of `C17693`'s
   benign cells did not reduce the single-host sensitivity; it increased it.
   The realised share is **1/4** (§1.1, `k = 2`), and the verdict must read
   "1/4" — rule (b) kept 10 benign rows for `C17693` where Part 30 kept 1
   (Table 3, host block), against 137 attack rows unchanged.
2. **It does not license:** "restoring benign contrast does not help." That
   claim needs the 1/1 build, which was **not run**. At `k = 2` three quarters
   of the restored contrast was dropped again, so H2's own mechanism was
   tested at a quarter strength.
3. **It does not license** any statement about *why* sensitivity grew. A
   plausible reading is available and is **not measured**: Part 38's test
   block is 84,460 rows at prevalence 0.00485 against Part 30's 18,729 at
   0.0219, so `C17693`'s 137 attack rows are now a third of the attack class
   inside a four-and-a-half-times larger block, and removing them removes a
   larger share of the rankable positive signal. **No decomposition
   distinguishing that from a behavioural effect exists.** Do not assert it.

**No interval attaches.** One host is not a resample; the result is
descriptive. Part 30's own `step5_l27.interval` records the same, and the
per-seed arrays are in
`table3_leave_host_out.per_cell.<cell>.per_host.C17693.roc_auc_host_removed.per_seed`.

**A fourth figure belongs beside this verdict**, from the already-committed
scalar passes: `fallback_survivor_k2` (`crc32(computer | window) % 4 == 0`)
reaches test AUC **0.8670731707317073**, outranking every view occupancy
(`results/lanl_v2/h1_occupancy_decomposition.json`,
`whole_test_block.fallback_survivor_k2`). It is a prevalence artefact of the
same `k = 2` fallback — `crc32` is not a feature and no encoder sees it — so
it says the class balance on attack hosts was manufactured by a hash, and does
**not** say the model has a leak. See §4.2.

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

### L24 — still OPEN, and the reason is now measured rather than pending

**This clause has fired.** `auth_present_indicator` test AUC is
**0.6611838192**, above the 0.6136552214 bar: H1 disconfirmed, drop achieved
0.0024714022 of the 0.05 required. L24 is OPEN on a **measured
disconfirmation**, not on missing instrumentation.

#### 4.1 The second source is identified: `proc` occupancy, installed by rule (a)

Two further scalar passes, both seed-free, no training, no inference, both
bit-exact on re-run and unchanged under `OMP_NUM_THREADS=1`:
`results/lanl_v2/h1_occupancy_decomposition.json` and
`results/lanl_v2/h1_conditional_rates.json`.

**Every selection input, scored as its own trivial baseline on the test block**
(train-only orientation; `attack_host_indicator` excluded from the ranking as
tautological — it is 1 on every attack row by construction):

| scalar | test AUC | orientation | reading |
|---|---|---|---|
| `fallback_survivor_k2` | 0.8670731707 | negated | **artefact of the L27 fallback, see §4.2** |
| `occupancy_proc` | 0.7083759667 | negated | **the second source** |
| `auth_present_indicator` | 0.6611838192 | as-is | H1's barred scalar |
| `n_non_auth_views_present` | 0.6503477024 | negated | rule (a)'s own gate |
| `identity_hash_kept` | 0.5883878644 | as-is | host-split only, see below |
| `total_records_across_views` | 0.5862805386 | as-is | fell 0.098 from Part 30 |
| `occupancy_flows` | 0.5253242118 | as-is | weak |
| `occupancy_dns` | 0.5000356930 | negated | **at chance** |

**The conditional rates say which of these are data and which are design.**
Attack rows are *identical* on both sample tables for all four views — both
designs keep every attack cell and the test time range is unchanged, so the
attack population is literally the same 410 rows and only the benign controls
differ:

| view | Part 30 attack / benign | Part 38 attack / benign | attack rate | benign rate | gap change |
|---|---|---|---|---|---|
| auth | 410/410 · 12323/18319 | 410/410 · 56955/84050 | 1.000000 → 1.000000 | 0.672690 → 0.677632 | −0.004943 |
| proc | 204/410 · 12774/18319 | 204/410 · 76848/84050 | 0.497561 → 0.497561 | 0.697309 → **0.914313** | **−0.217004** |
| flows | 127/410 · 5149/18319 | 127/410 · 21778/84050 | 0.309756 → 0.309756 | 0.281074 → 0.259108 | +0.021967 |
| dns | 8/410 · 340/18319 | 8/410 · 1646/84050 | 0.019512 → 0.019512 | 0.018560 → 0.019584 | −0.001024 |

1. **L24's contingency SURVIVED, and that settles H1's reading.** The auth
   attack rate is still *exactly* 1.000000 on a sample whose selection never
   consulted the auth stream, and the benign rate moved by 0.005. So auth's
   label-adjacency is **a property of the data**, not an artefact of how Part
   30 drew its controls. This is the direct measurement behind H1's
   disconfirmation, not an inference from it.
2. **`proc`'s separation is NEW and is a selection artefact of rule (a).**
   `proc`'s attack rate is unchanged at 0.497561 while its benign rate jumped
   from 0.697309 to 0.914313, because rule (a) admits a cell only if it is
   occupied in at least one of `proc`, `flows`, `dns`. The gap more than
   doubled, −0.1998 → −0.4168. **Part 38 removed auth from the selection and
   installed a stronger `proc`-occupancy selection effect in its place. The
   selection did not drop; it moved.**

**The two passes cross-validate exactly, which is why the above can be relied
on.** For a binary indicator, `AUC = 0.5 + |attack_rate − benign_rate| / 2`.
The rate-implied AUCs from the conditional-rate pass reproduce the measured
AUCs from the decomposition pass to **0.00e+00** on all four views, and the
same algebra on Part 30's rates returns **0.6636552214** — L24's committed
0.6637. Two independent implementations and the prior Part's prose agree.

`identity_hash_kept` is an internal consistency check rather than a finding:
0.5883878644 on the whole block but **0.5037865749 within attack hosts**, so
its whole-block value came entirely from the attack-host / non-attack-host
split (every non-attack-host control satisfies `crc32 % 512 == 0` by
construction) and within hosts it sits at chance, as a hash must.

**Within attack hosts the picture is unchanged** — `proc` 0.7143004072, auth
0.6524885255 — which is the within-attack-host contrast PRE_REGISTRATION §2
names as the figure that matters. So the host-rule tautology is not what
produces either number.

#### 4.2 A separate finding, belonging to L27's rule, not L24's

`fallback_survivor_k2` (`crc32(computer|window) % 4 == 0`) outranks every view
occupancy at test AUC **0.8670731707**. **It is a prevalence artefact of the
`k = 2` size fallback and not a signal any model can use.** `crc32` is not a
feature and no encoder sees it. It scores as it does because attack cells were
kept unconditionally while attack-host benign cells were thinned to 1/4 by
that predicate, so the surviving benign rows satisfy a condition the attack
rows mostly fail. **What it says is that the class balance on attack hosts was
manufactured by a hash. What it does not say is that the model has a leak.**

This belongs beside §3.2's L27 verdict, because the fallback is L27's rule (b)
on its fallback branch. It is an additional reason the L27 verdict must read
"1/4" rather than "all".

#### 4.3 L24 status after these passes — NARROWED, still OPEN

**Narrowed, in two specific ways that did not hold before:**

- The "unidentified second source" of H1's disconfirmation **is identified**:
  `proc` occupancy, at 0.7084, stronger than auth's 0.6612, and attributable
  to rule (a) by the conditional rates.
- Auth's label-adjacency is **separated** from the selection question and
  confirmed as a data property, which is what L24's first bullet claims and
  what Part 38's design was built to test.

**Still OPEN, because:**

- **No design yet removes both selection effects.** Rule (a) had to gate on
  *something*, and whatever it gates on acquires the benign-side enrichment
  that `proc` now carries. L24's closing condition ("control cells drawn
  without reference to the `auth` stream") is met, and the finding is that
  meeting it is not sufficient to remove selection from the sample.
- **The arm AUCs are now in, and they do not rescue L24 — they sharpen it.**
  Dropping the auth view **raises** test ROC AUC by **0.0498525761** (naive)
  and **0.0448397063** (prismflow), negative in 5 of 5 paired seeds in both
  systems (Table 4 lower block, from Table 1 and Table 2). `without_auth` is
  the **best** arm on this sample, not the handicapped one. Under Part 30 the
  reading was "auth is label-adjacent, so `without_auth` is penalised"; on
  Part 38's sample `without_auth` wins, which is what the relocation of the
  selection effect to `proc` predicts. This is descriptive and carries no
  pre-registered bar.
- **Any status change requires the review gate** (PRE_REGISTRATION §7.6).
  Proposals go to `docs/proposed_limitations_update.md`.

**What must not be claimed.** That auth is "not label-adjacent" — both passes
point the other way. That rule (a) failed — it did what it was specified to
do, and the finding is about what that implies. That 0.6612 against 0.6637
measures an effect — the two tables are not row-comparable, so only the
pass/fail against the pre-registered bar is a result. That
`fallback_survivor_k2` is a leak — the model cannot see it.

### L27 still OPEN if

- **THIS CLAUSE HAS FIRED, in all eight cells.** The eight `C17693` deltas,
  with their per-seed arrays in
  `filled_tables.json`,
  `table3_leave_host_out.per_cell.<cell>.per_host.C17693.roc_auc_host_removed.per_seed`:

  | arm | system | Δ | per-seed AUC with host removed |
  |---|---|---|---|
  | `without_auth` | naive | −0.099710 | see path above, n = 5 |
  | `without_auth` | prismflow | −0.099784 | " |
  | `without_flows` | naive | −0.103284 | " |
  | `without_flows` | prismflow | −0.103105 | " |
  | `without_dns` | naive | −0.108785 | " |
  | `without_dns` | prismflow | −0.104055 | " |
  | `with_auth` | naive | **−0.113438** (largest) | 0.5980048747, 0.6409268960, 0.6217520263, 0.6331019766, 0.6399412978 |
  | `with_auth` | prismflow | −0.108746 | " |

  H2 is disconfirmed. Restoring a quarter of the benign contrast for that host
  did not shrink the leave-host-out sensitivity — **it grew it**, by +0.0478 to
  +0.0756 per cell against Part 30. The concentration of the attack class in
  one host remains a live threat to every per-arm AUC, and is now measured as
  a larger threat than Part 30 recorded.
- All eight pass but the smallest Part 30 cell (−0.035963) is not cleared by a
  margin → **this branch did not fire.** It would have required a
  confirmation; none occurred. Had it fired, the remedy was a bar set from the
  *median* or the *largest* of Part 30's band rather than its minimum, agreed
  at the review gate. Recorded because the branch was written before the
  numbers and must be reported as untaken rather than deleted.
- Part 38's re-derived top-5 host set differs from Part 30's → **this branch
  did not fire.** The re-derivation on Part 38's own test block returns the
  **same five computers in the same rank order** (Table 3, host block). The
  removed-row counts differ by design, as rule (b) requires.
- The realised share is **1/4** (`k = 2`), so L27's "separate host from
  behaviour" is tested at a quarter of the intended benign contrast whatever
  the deltas say. **And the thinning predicate is now measured as the
  strongest scalar on the test block** — `fallback_survivor_k2` at test AUC
  0.8670731707 (§4.2). It is not a signal a model can use, but it means the
  attack/benign balance on attack hosts is a hash's doing, which is exactly
  the contrast H2's mechanism relies on. Needed to test it at 1/1: raising the 400,000 sample bar,
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

Checked 2026-11-22 at 40/40.

- [x] Every `[pending]` replaced by a value with its JSON path, or by
      `not measured` with the reason. **The only remaining `[pending]` markers
      are Table 5's mode A columns, labelled `pending mode A`, plus the two
      occurrences in this document's own prose explaining the convention.**
- [x] Every cell has `n = 5`; any `n < 5` cell labelled `NOT A FINDING`. All
      eight arm × system cells are `n = 5`; no cell needed the label.
- [x] Per-seed sign counts reported before every aggregate mean — Table 2,
      with `seeds_exactly_zero` = 0 in all four arms.
- [x] No ± pools the two execution modes. Mode A has not been run, so no
      pooling was possible; Table 5's mode B and mode A columns are separate.
- [x] Every ENIV value carries its thread count — Table 1's header reads
      "8 threads", matching `torch_threads_observed`.
- [x] Mode A tables either filled or left `pending mode A` — never blank.
- [x] Table 3's `C17693` column distinguished from the four exploratory hosts,
      and the four are labelled exploratory at the table and in §3.2.
- [x] No number in this file lacks a committed source.
- [x] `PRE_REGISTRATION.md` and `docs/KNOWN_LIMITATIONS.md` unedited.
- [x] Status proposals written to `docs/proposed_limitations_update.md` only.
- [ ] **Resolve each number to an exact JSON path by hand before the report
      quotes it.** The repo's number check is a *presence* check — it confirms
      a number appears somewhere in a cited file, not that it is the field
      claimed. This box is deliberately left open.
