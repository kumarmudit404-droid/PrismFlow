# Part 38 — LANL redesign and retrain (`experiments/lanl_v2/`)

Status at the time of writing (2026-11-21): the mode `default` sweep is
**25 of 40 units** complete and still running. The three **scalar passes are
complete and committed**; every sweep table is pending.

Every number below cites the committed file and the field it came from. A
quantity with no committed path is written `not measured`.

---

## 1. Purpose

Part 30 measured PrismFlow on LANL `cyber1` and three limitations came out of
it that the design itself could not settle:

- **L24** — the `auth` view is label-adjacent, and dropping it does not drop
  the selection, because cells were chosen from all four views.
- **L27** — one host carries a third of the attack class with no benign
  contrast.
- **L28** — the execution mode changes the fitted weights, and a per-seed std
  computed inside one run understates the spread.

Each of those entries names a closing condition that is a condition on the
**design**, not on a number. Part 38 builds a sample that meets the first two
and runs the sweep in a second execution mode for the third. It does not edit
`docs/KNOWN_LIMITATIONS.md` and does not change any limitation's status;
proposals go to `docs/proposed_limitations_update.md` for the review gate.

## 2. Pre-registration

`experiments/lanl_v2/PRE_REGISTRATION.md`, written before any cell was
sampled, plus **Amendment 1** (dated 2026-11-06) at its section 10. Both are
**frozen**: nothing in this Part edits them, and the one finding that
contradicts the amendment's supporting premise is recorded separately in
`results/lanl_v2/SWEEP_EXECUTION_LOG.md` section 2 rather than retrofitted
into the amendment.

The three hypotheses, each with a bar taken from a committed number:

| | hypothesis | bar |
|---|---|---|
| **H1** | L24 — with controls drawn without reference to `auth`, the auth-presence-only baseline falls materially | drop of at least **0.05** from Part 30's **0.6636552213548774** |
| **H2** | L27 — with benign cells of attack hosts retained, dropping `C17693` changes test ROC AUC by less than 0.036, in all eight arm-and-system cells | **0.036**, the smallest of Part 30's committed per-arm drops |
| **H3** | L28 — the sign of the PrismFlow-minus-naive ROC AUC difference agrees across both execution modes, in all four arms | sign agreement, no magnitude bar |

**Primary metric is threshold-free ROC AUC.** Balanced accuracy at the default
0.5 threshold is secondary (Amendment 1, section 10.1). No hypothesis is
decided on a threshold metric.

**The selection rule.** Every `attack = 1` cell is kept, unchanged from Part
30. A cell is kept as a control iff **both**: (a) it is occupied in at least
one of `proc`, `flows`, `dns` — the `auth` stream is never consulted — and
(b) its computer has at least one attack cell in the read range, or else
`crc32(computer|window) % 512 == 0`. `auth` features are attached in a later
pass, so `auth` enters as features only.

**The size fallback fired.** `results/lanl_v2/build_v2_days16.json`,
`size_fallback`: `fallback_fired` true, `attack_host_benign_share_k` **2**,
`attack_host_benign_share` **`"1/4"`**, `rows_before_fallback` **979122**,
`rows_after_fallback` **281824**, `attack_host_benign_cells_before`
**929764**, `attack_host_benign_cells_dropped` **697298**, against
`max_total_bar` **400000**. So the realised attack-host benign share is
**1/4, not 1/1**, and every L27 statement must say so. The rule and its
fallback were both fixed before any count was seen.

Realised sample (`build_v2_days16.json`): `n_samples` **281823**, train
**197363** at prevalence **0.0022243277615358503**, test **84460** at
prevalence **0.0048543689320388345**, `attack_hosts` **302**.

## 3. Sampler audit

Recorded in `results/lanl_v2/SWEEP_EXECUTION_LOG.md` section 1, a dated
operational record that amends nothing. **Verdict: passed, no defect
detected.**

**Probe 1 — selection against Part 30's committed population**, three read
ranges, summed to the same range, against `results/lanl/results.json`
`population.per_view.<view>`. **Nine of nine exact:**

| range | `proc` | `flows` | `dns` |
|---|---|---|---|
| days 1–6 | 7,975,224 | 5,361,691 | 192,635 |
| days 1–7 | 9,637,185 | 6,502,720 | 229,895 |
| days 1–16 | 23,627,216 | 13,484,366 | 580,753 |

**Probe 2 — records scanned and attack counts at 16 days.** Matches Part 30
exactly: `proc` 115,159,740, `flows` 121,284,699, `dns` 1,055,902; attack
counts to the row, train 439 over 150 windows and test 410 over 141. Two
independent 11 GB scans reproduced `per_day`, `selection`, `n_samples`,
`attack_hosts` and `attack_cells_in_read_range` field for field, at peak RSS
2,366.4 MB against 2,367.7 MB.

**Probe 3 — synthetic audit**, `tests/unit/test_lanl_v2_sampler_synthetic.py`,
8 tests passing. Four synthetic `*.txt.gz` in the loader's wire format over 10
hosts, 50 attack cells inside the read range and 7 outside, under a throwaway
`raw_root`. `select_cells` is compared against a brute-force re-derivation of
the pre-registered rule **cell by cell** — the check a sum cannot give,
because a compensating pair of errors survives a sum.

Two caveats carried forward, neither resolved:

1. The brief asked for a 10-view toy. The loader's view set is fixed at four
   by `FEATURES`, `SOURCE_COLUMN` and `RECORD_WIDTH` in the frozen
   `prismflow/data/lanl_cyber1.py`, so four real views over 10 hosts was built
   instead.
2. The earlier smoke numbers were reported as void on account of a sampler
   defect. Amendment section 10.4 records that report as received and **not
   confirmed**: the section 7.1 reproduction gate passes on the committed
   sampler and **no defect has been located**. Nothing in this Part rests on
   those smoke numbers.

## 4. The three scalar passes

All three are seed-free, read only committed tables, and involve **no
training, no inference and no model**. Each re-runs bit-exact and is unchanged
under `OMP_NUM_THREADS=1`. Covered by `tests/unit/test_h1_scalar_passes.py`
(12 tests).

They exist because **the sweep does not compute H1**:
`experiments/lanl_v2/metrics.py:80` defines `auth_presence_auc` and
`run_lanl_v2.py` never calls it — its only metric calls are
`m.evaluate(labels, prob)` and `m.constant_baseline_brier(labels)` at lines
448–449.

### 4.1 H1 — the auth-presence baseline

`experiments/lanl_v2/h1_auth_presence.py` →
`results/lanl_v2/h1_auth_presence.json`. Orientation follows Part 30's rule by
importing `trivial_scalar_values` and `roc_auc_binary` from the frozen
`experiments/lanl/run_lanl.py`: the sign is chosen on the TRAIN block only and
applied unchanged to test. Auth is column **0** of
`["auth", "proc", "flows", "dns"]` (`auth_column_index`).

| scalar | orientation | train AUC | Part 38 test AUC | Part 30 test AUC | drop |
|---|---|---|---|---|---|
| **`auth_present_indicator`** | as-is | 0.6675392537222482 | **0.6611838191552647** | 0.6636552213548774 | **0.0024714021996127267** |
| `total_records_across_views` | as-is | 0.6918403739721934 | 0.5862805385876582 | 0.6842788175411642 | 0.09799827895350599 |
| `n_views_present` | as-is | 0.5645444186716992 | 0.4800433975130947 | 0.5624544155807845 | 0.08241101806768975 |

Fields: `baselines.<name>.{train_auc_as_measured, orientation_fixed_on_train,
test_auc, part30_test_auc, drop_from_part30}`. `n_train` 197363, `n_test`
84460.

**`h1.verdict` is `DISCONFIRMED`.** The bar was
`h1.bar_test_auc_at_or_below` **0.6136552213548774**; `h1.drop_achieved` is
**0.0024714021996127267** against `h1.min_drop_required` **0.05**. The
baseline is seed-free, so this is a single value and no `+/-` applies.
Pre-registration section 1 pre-declared this branch: "If it does **not** fall,
H1 is disconfirmed and the selection signal has another source, which is a
reportable finding in its own right."

### 4.2 The occupancy decomposition

`experiments/lanl_v2/h1_occupancy_decomposition.py` →
`results/lanl_v2/h1_occupancy_decomposition.json`. Every selection input
scored as its own trivial baseline. The two rule-(b) predicates are rebuilt
from frozen code rather than paraphrased: the host set is `read_redteam` →
`cells_in_read_range` → `attack_hosts` (`attack_hosts_in_read_range`
**302**), and the hash predicates use `control_divisor` **512** and
`fallback_k` **2**. The cell-key round-trip (`window_start` → window index,
`computer` str → bytes) is asserted, because an off-by-one there would hash a
different cell.

`whole_test_block.<scalar>.test_auc`, ranked by distance from chance
(`ranking_excluding_tautological`), beside
`within_attack_hosts.<scalar>.test_auc` (`n_test_on_attack_hosts` **70130** of
84460, prevalence **0.0058462854698417225**):

| scalar | whole test block | within attack hosts | orientation |
|---|---|---|---|
| `fallback_survivor_k2` | 0.8670731707317073 | 0.8670731707317073 | negated |
| `occupancy_proc` | 0.7083759666864962 | 0.7143004072037278 | negated |
| `auth_present_indicator` | 0.6611838191552647 | 0.6524885255306943 | as-is |
| `occupancy_auth` | 0.6611838191552647 | 0.6524885255306943 | as-is |
| `n_non_auth_views_present` | 0.6503477024419263 | 0.647878062773743 | negated |
| `identity_hash_kept` | 0.5883878643664485 | 0.5037865748709123 | as-is |
| `total_records_across_views` | 0.5862805385876584 | 0.569008332983502 | as-is |
| `occupancy_flows` | 0.5253242117787031 | 0.5347905559520311 | as-is |
| `occupancy_dns` | 0.5000356930398572 | 0.4996673103564082 | negated |
| `n_views_present` | 0.4800433975130947 | 0.4757941172354925 | as-is |
| `attack_host_indicator` | 0.5852468768590124 — **tautological, excluded** | 0.5 — constant | as-is |

`attack_host_indicator` is 1 on every attack row **by construction** (an
attack cell makes its own computer an attack host), so its AUC measures the
sampling rule and not a discovered signal. It is marked `tautological: true`
and excluded from the ranking; within attack hosts it is constant and its AUC
is 0.5 by definition. Pre-registration section 2 declares the underlying fact
and names the within-attack-host contrast as the figure that matters, which is
why both columns are reported.

`identity_hash_kept` is a consistency check rather than a result:
**0.5883878643664485** whole-block but **0.5037865748709123** within attack
hosts, so its whole-block value came entirely from the attack-host /
non-attack-host split — every non-attack-host control satisfies
`crc32 % 512 == 0` by construction — and within hosts it sits at chance, as a
hash must.

`positive_rate_test` and `distinct_values_test` are recorded beside every
scalar, because a near-constant indicator has an uninformative AUC.

### 4.3 The conditional rates

`experiments/lanl_v2/h1_conditional_rates.py` →
`results/lanl_v2/h1_conditional_rates.json`. Per-view record-presence
contingency on the TEST block of **both** sample tables.

**Part 30's side is recomputed, not quoted.** L24's 12,323 figure is prose in
`docs/KNOWN_LIMITATIONS.md` and is **not** a field in
`results/lanl/results.json`, whose cited paths
(`phase3_diagnostics.arm_coverage`, `.trivial_baselines`) carry neither count.
The pass recomputes it from `results/lanl/samples.npz` and gates on
reproducing L24's figures before any comparison:
`l24_reproduction_gate.status` is **`PASSED`** on all four checks
(410/410 attack, 12,323/18,319 benign).

`part30.per_view.<view>` and `part38.per_view.<view>`:

| view | Part 30 attack / benign | Part 38 attack / benign | attack rate | benign rate | rate gap |
|---|---|---|---|---|---|
| auth | 410/410 · 12323/18319 | 410/410 · 56955/84050 | 1.0 → 1.0 | 0.6726895572902452 → 0.6776323616894706 | 0.32731044270975485 → 0.3223676383105294 |
| proc | 204/410 · 12774/18319 | 204/410 · 76848/84050 | 0.4975609756097561 → 0.4975609756097561 | 0.6973088050657787 → 0.9143129089827484 | −0.19974782945602265 → −0.4167519333729923 |
| flows | 127/410 · 5149/18319 | 127/410 · 21778/84050 | 0.3097560975609756 → 0.3097560975609756 | 0.28107429444838694 → 0.2591076740035693 | 0.028681803112588666 → 0.050648423557406286 |
| dns | 8/410 · 340/18319 | 8/410 · 1646/84050 | 0.01951219512195122 → 0.01951219512195122 | 0.018559965063595175 → 0.019583581201665676 | 0.000952230058356044 → −7.138607971445693e-05 |

Part 30 test block `n_rows` 18729 at prevalence 0.0218911848; Part 38 `n_rows`
84460 at prevalence 0.0048543689320388345.

**`attack_rate_change` is 0.0 on all four views.** Both designs keep every
attack cell and the test time range is unchanged, so the attack population is
literally the same 410 rows and only the benign control population differs.
This is an internal check on both samplers, not a result.

`auth_verdict.still_exactly_one_on_part38` is **true**: the auth attack rate
is still exactly 1.0 on a sample whose selection never consulted the auth
stream, and `rate_gap_change` is **−0.004942804399225453**.

**The two passes cross-validate exactly.** For a binary indicator,
`AUC = 0.5 + |attack_rate − benign_rate| / 2`. The rate-implied AUCs reproduce
the measured AUCs in section 4.2 to **0.00e+00** on all four views, and the
same algebra on Part 30's `auth` rate gap returns **0.6636552213548774** —
L24's committed figure. Asserted in
`tests/unit/test_h1_scalar_passes.py::test_h1_occupancy_decomposition_rates_cross_validate`.

### 4.4 What the three passes establish, and what they do not

**Established.** L24's contingency survived a selection that never consults
`auth` (auth attack rate still exactly 1.0, benign rate moved
0.004942804399225453), so the auth-presence separation is a property of the
data rather than an artefact of how Part 30 drew its controls. H1 is
disconfirmed. The occupancy signal that now carries the most separation is
`proc` at 0.7083759666864962, above `auth`'s 0.6611838191552647, and the
conditional rates attribute it to rule (a): `proc`'s attack rate is unchanged
while its benign rate moved 0.6973088050657787 → 0.9143129089827484, so its
rate gap more than doubled. Within attack hosts the same ordering holds
(`proc` 0.7143004072037278, `auth` 0.6524885255306943), so the host-rule
tautology does not produce it. The two volume scalars fell
(`total_records_across_views` by 0.09799827895350599, `n_views_present` by
0.08241101806768975, the latter to below chance), which is the evidence that
the rule changed something.

**Not established.** Which part of `proc`'s 0.7083759666864962 is
selection-induced enrichment and which part is behavioural signal about the
label — **not measured**. `proc` occupancy is visible to the encoders through
the view mask, so unlike `fallback_survivor_k2` it cannot be set aside as
unusable by a model. Also **not measured**: the design-effect ENIV, which
`prismflow.evaluation.protocol.collect_predictions` does not expose (it
returns only batch-weighted scalars, not the dependence matrix, and is
frozen); every unit JSON records this in `eniv_design_effect_status`.

**`fallback_survivor_k2` at 0.8670731707317073 is a prevalence artefact of the
`k = 2` thinning, not a signal any model can use.** `crc32` is not a feature
and no encoder sees it. It scores as it does because attack cells were kept
unconditionally while attack-host benign cells were thinned by
`crc32 % 4 == 0`, so the surviving benign rows satisfy a predicate the attack
rows mostly fail. It says the class balance on attack hosts was manufactured
by a hash. It does not say the model has a leak. It belongs to L27's rule, not
L24's.

## 5. How to reproduce

Each pass takes seconds and needs no GPU. Run from the repository root.

```sh
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/h1_auth_presence.py
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/h1_occupancy_decomposition.py
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/h1_conditional_rates.py
```

Each rewrites its own JSON in place. To check determinism, copy the file
aside, re-run, and `diff`; to check thread invariance, re-run under
`OMP_NUM_THREADS=1` and `diff` again. Both checks are also asserted by the
test file:

```sh
PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest -q tests/unit/test_h1_scalar_passes.py
```

Inputs: `results/lanl_v2/samples_v2_days16.npz` (sha256
`03b7602a590f9c64196ceee2750981187f456541d81927cb69d7b7d6aa251a9a`,
`SWEEP_EXECUTION_LOG.md` section 5), `results/lanl/samples.npz` for Part 30's
side, and `data/raw/redteam.txt.gz` for the attack-host set.

The sweep itself, resumable — a unit whose JSON exists is skipped:

```sh
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/run_lanl_v2.py sweep \
    --tag v2_days16 --arms without_flows without_dns without_auth with_auth \
    --systems naive prismflow --seeds 0 1 2 3 4 --mode default
```

Python block-buffers stdout when it is not a tty, so a redirected sweep log
will look stalled. **`results/lanl_v2/units_default/` is the durable progress
record**, written once per unit.

## 6. What the sweep adds when it completes

40 units = 4 arms × 2 systems × 5 seeds, in **one** execution mode. Then
`experiments/lanl_v2/fill_analysis.py`, behind a 40-unit gate, writes
`results/lanl_v2/filled_tables.json`:

| table | content | status |
|---|---|---|
| 1 | ROC AUC mean ± sample std (ddof=1) per arm × system, 8 cells | pending the sweep |
| 2 | per-seed ROC AUC, 40 cells, plus per-seed signs of PrismFlow − naive | pending the sweep |
| 3 | **H2** — leave-host-out AUC delta, 8 cells × 5 hosts; only the `C17693` column is barred | pending the sweep |
| 4 | **H1** — the auth-presence baseline | **complete**, section 4.1 |
| 5 | **H3** — cross-mode delta | **pending mode A**, which has not been run |
| 6 | per-unit wall-clock, peak RSS, and the Amendment 10.2 degeneracy diagnostic | pending the sweep |

`fill_analysis.py` has five fatal gates: 40 matched JSON/npz pairs; every
unit's recorded `scores_sha256` matching the npz on disk; no npz carrying a
key named `vacuity` (L29 — this runner writes `confidence`); all units in one
mode with one agreed thread count; and the re-scored all-test ROC AUC equal to
the committed `roc_auc` field to 1e-9, so a leave-host-out delta cannot be an
artefact of two AUC implementations. A `--dry-run` flag computes the tables
over whatever units are present, stamps the whole document
`DRY_RUN_NOT_A_FINDING`, and writes a separate `_dryrun` filename.

**Mode A is not run.** `results/lanl_v2/sweep_estimate.json` puts it at
~2.0 h across 11 concurrent single-thread processes at ~5,372 MB. Without it
Table 5 stays `pending mode A` and H3 has no verdict. Pre-registration section
6 forbids pooling the two modes into a single `+/-`, and every ENIV figure is
reported with the thread count it was computed at.

A cell with fewer than 5 completed seeds is **not a finding** and no verdict
may be drawn from it.

## 7. Files

| path | what |
|---|---|
| `PRE_REGISTRATION.md` | frozen, with Amendment 1 at section 10 |
| `sampler.py` | the selection rule and the size fallback |
| `run_lanl_v2.py` | build, probe, smoke, and the resumable `sweep` |
| `metrics.py` | ROC AUC, balanced accuracy, ECE, and `auth_presence_auc` |
| `h1_auth_presence.py` | H1's three trivial baselines |
| `h1_occupancy_decomposition.py` | every selection input as its own baseline |
| `h1_conditional_rates.py` | the per-view contingency on both tables |
| `fill_analysis.py` | Tables 1, 2, 3, 6 behind a 40-unit gate |
| `ANALYSIS_TEMPLATE.md` | the tables, verdict logic and still-OPEN clauses |
| `SUNDAY_SCHEDULE.md` | the plan to the push; a planning document, not evidence |
| `sweep_block.sh` | a time-boxed block of single-unit sweep calls |
| `results/lanl_v2/SWEEP_EXECUTION_LOG.md` | the sampler audit, the range-dependence finding, and the launch record |
