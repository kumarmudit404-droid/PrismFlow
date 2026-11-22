# Part 38 — verdict worksheet

**What this file is.** A one-page decision sheet: for each of L24, L27 and
L28, the barred quantity, the bar, the measured value if it exists, and the
verdict. It is the thing to read when asking "has Part 38 decided this yet?"

**What this file is not.** It is **not** a second copy of the verdicts. The
reasoning, the caveats and the final-form prose already exist in two committed
places and are cited here rather than restated, because a verdict that lives
in three files can drift in two of them:

- `experiments/lanl_v2/ANALYSIS_TEMPLATE.md` §3 (verdict logic, written before
  the numbers), §4 (still-OPEN clauses), §5 (range dependence).
- `docs/part38_section.md` (the final-report prose, §4.1–4.3, §7).

Status proposals go to `docs/proposed_limitations_update.md` only. This file
changes no limitation's status: PRE_REGISTRATION §7.6 reserves that for the
full sweep plus a review gate.

Written 2026-11-22, with the mode `default` (B) sweep at **36 of 40 units**.

---

## 1. Decision table

| | barred quantity | bar | measured | verdict |
|---|---|---|---|---|
| **H1 / L24** | `auth_present_indicator` test ROC AUC, seed-free | **≤ 0.6136552214** (0.05 below Part 30's 0.6636552214) | **0.6611838192** | **DISCONFIRMED** — decided, see §2 |
| **H2 / L27** | all eight arm×system `C17693` leave-host-out deltas, absolute value | **< 0.036** on all eight; any one ≥ 0.036 disconfirms | not measured | **PENDING** — needs Table 3 at 40/40, see §3 |
| **H3 / L28** | sign of PrismFlow − naive test ROC AUC, agreeing across modes A and B in all four arms | sign agreement in 4/4 arms | not measured | **PENDING MODE A** — and unclosable by L28's own text, see §4 |

"not measured" above is an absence, not a zero, and is not estimated anywhere
in this file.

---

## 2. H1 / L24 — DECIDED, DISCONFIRMED

This verdict is final now and does not wait for the sweep. Its three inputs
are seed-free scalar passes over the committed sample table, with no training,
no inference and no model, each bit-exact on re-run and unchanged under
`OMP_NUM_THREADS=1`.

**Verdict: H1 DISCONFIRMED.** Measured `0.6611838191552647` against a bar of
`0.6136552213548774`; drop achieved `0.0024714021996127267` of the `0.05`
required — a fiftieth.
Source: `results/lanl_v2/h1_auth_presence.json`, `h1.verdict`,
`h1.part38_test_auc`, `h1.bar_test_auc_at_or_below`, `h1.drop_achieved`.

Pre-registered as a finding either way (PRE_REGISTRATION §1): "If it does
**not** fall, H1 is disconfirmed and the selection signal has another source,
which is a reportable finding in its own right."

**The other source is identified, not inferred.** Three figures carry it:

| figure | value | source |
|---|---|---|
| `occupancy_proc` test AUC (orientation `negated`) | 0.7083759666864962 | `h1_occupancy_decomposition.json`, `whole_test_block.occupancy_proc.test_auc` |
| `proc` benign presence rate, Part 30 → Part 38 | 0.697309 → 0.914313 | `h1_conditional_rates.json`, `per_view_rate_comparison.proc.{part30_benign_rate,part38_benign_rate}` |
| `proc` attack presence rate, unchanged | 0.4975609756097561 both | same file, `per_view_rate_comparison.proc.attack_rate_change` = 0.0 |

**The contingency L24 names survived.** `auth` attack presence rate is still
exactly `1.0` on a sample whose selection never read the auth stream; the
benign rate moved by `+0.004942804399225453`.
Source: `results/lanl_v2/h1_conditional_rates.json`, `auth_verdict`.

**L24's own reproduction gate PASSED** — 410/410 attack and 12,323/18,319
benign recomputed from `results/lanl/samples.npz`, matching L24's prose.
Source: same file, `l24_reproduction_gate.status`.

**Proposed status: OPEN, narrowed.** The closing condition is met — it is a
condition on the design, discharged by rule (a) — H1 is disconfirmed, and the
selection effect is **relocated to `proc`, not removed**. Rule (a) must gate
on something, and whatever it gates on acquires the benign-side enrichment
`proc` now carries.
Reasoning, and the two limits that must travel with the reading:
`ANALYSIS_TEMPLATE.md` §3.1, §4.1, §4.3; `docs/part38_section.md` §4.1–4.3, §7.

**Still outstanding for L24**, and not decided by this sheet: the `with_auth` −
`without_auth` arm gap (Table 4 lower block, from Table 1), pending the sweep.
It is a **different quantity** from H1 and cannot stand in for it.

---

## 3. H2 / L27 — PENDING the sweep

**Fill from:** `results/lanl_v2/filled_tables.json`,
`table3_leave_host_out.per_cell.<arm>/<system>.per_host.C17693`, read at
40/40. Eight cells: `{without_auth, without_flows, without_dns, with_auth}` ×
`{naive, prismflow}`.

**The rule is existential.** Any one of the eight at `|Δ| ≥ 0.036`
disconfirms; confirmation needs all eight below. So the verdict can become
determinable before 40/40 — and must still be read only from the committed
40/40 file.

**Three things the verdict text must contain.** None is optional; all three
are fixed in advance.

1. **The realised share is 1/4, not 1/1.** The §1.1 size fallback fired at
   `k = 2`: 697,298 of 929,764 attack-host benign cells were dropped to bring
   979,122 rows under the pre-registered 400,000 bar. A disconfirmation
   therefore shows that restoring **a quarter** of the benign contrast did not
   help — not that restoring it fails. Every L27 verdict must read "1/4"; one
   that says "all benign cells of attack hosts were retained" is false on this
   build.
2. **The bar sits inside Part 30's own band.** Part 30's eight committed
   `C17693` deltas run `−0.035962653276654266` to `−0.051874860066125961`
   (`results/lanl_reanalysis/results.json`,
   `step5_l27.per_arm.<arm>.<system>.per_host.C17693.change_from_all_test`),
   all negative. The smallest is **below** the 0.036 bar, so a Part 38 cell
   reproducing Part 30's weakest cell exactly would score as passing. Report
   the eight Part 38 deltas against the eight Part 30 deltas pairwise.
3. **No interval attaches.** One host is not a resample; the result is
   descriptive. Part 30's own `step5_l27.interval` records the same.

**A fourth figure belongs beside it, already committed.**
`fallback_survivor_k2` (`crc32(computer|window) % 4 == 0`) reaches test AUC
**0.8670731707317073**, outranking every view occupancy
(`h1_occupancy_decomposition.json`, `whole_test_block.fallback_survivor_k2`).
It is a prevalence artefact of the same `k = 2` fallback — `crc32` is not a
feature and no encoder sees it — so it says the class balance on attack hosts
was manufactured by a hash, and does **not** say the model has a leak.
See `ANALYSIS_TEMPLATE.md` §4.2.

**The four non-pre-registered hosts** (`C1015`, `C2388`, `C1936`, `C2079`)
cannot confirm or disconfirm H2 and must never be aggregated with the `C17693`
column. Part 38's host set must be **re-derived on Part 38's own test block**,
not copied from Part 30, and whether it is the same five computers is itself a
`[pending]` cell.

**Proposed status: pending.** NARROWED at most, even on a clean confirmation.

---

## 4. H3 / L28 — PENDING MODE A, and unclosable as written

**Status: PENDING.** Mode A (`threads1_concurrent11`) has not been run.
`results/lanl_v2/sweep_estimate.json` puts it at ~2.0 h across 11 concurrent
single-thread processes at ~5,372 MB. Table 5 stays `pending mode A` — not
zero, not blank, not missing.

**L28 cannot be CLOSED by its own literal text under this pre-registration,
and this is stated rather than finessed.** Its closing condition asks for a
std that covers "mode as well as seed" — a pooled ±. PRE_REGISTRATION §6
forbids exactly that pooling. The pre-registration substitutes H3, a
sign-agreement test, and that is the only barred L28 analysis Part 38 can
produce. Closing L28 on its own terms needs a second dated amendment and a
review-gate decision that a mode-pooled ± is legitimate — not a judgement this
document can make.

**Do not adopt a "within 3 stds" magnitude bar.** It is already failed by the
prior record: L28 records the same seed differing by **0.0140** in AUC across
modes (naive seed 0: 0.8375 and 0.8235), about **four** per-seed stds of that
arm and system (0.0033). A 3-std bar would score Part 30's own observation as
disconfirming. Any magnitude bar must be pre-registered at a value chosen
before mode A runs, and justified against that 4-std figure.

**Every ENIV figure carries its thread count.** Mode B observed 8 threads,
mode A sets 1; a committed `eniv_eigen` of 3.4396320614332856 at the machine
default is missed by 0.0107 at one thread (3.4289614712229826, Part 34). Never
compare ENIV across the two mode columns without naming each side's thread
count.

---

## 5. Range-dependence note — bears on how Table 1 is read

Recorded in `results/lanl_v2/SWEEP_EXECUTION_LOG.md` §2, a dated operational
record that amends nothing. Full text: `ANALYSIS_TEMPLATE.md` §5.

Amendment §10.3 states the model "emits an attack probability below 0.5 on
every test row", which pins balanced accuracy at exactly 0.5. **True at days
1–6 and 1–7; false at days 1–16, which is the range the sweep runs on.**

| range | test ROC AUC (seed 0 / seed 1) | balanced accuracy | attack recall |
|---|---|---|---|
| days 1–6 | 0.7319 / 0.7189 | 0.5000 | 0.0000 |
| days 1–7 | 0.7128 / 0.7089 | 0.5000 | 0.0000 |
| **days 1–16** | **0.7201 / 0.7463** | **0.5512 / 0.5549** | **0.1024 / 0.1098** |

Source: `results/lanl_v2/smoke.json`, `smoke_days7.json`, `smoke_days16.json`
— all `with_auth`, `naive`, mode `default`, **two seeds, one arm, one system,
and therefore not findings about any hypothesis.**

**Consequence for the verdicts:** do not describe Table 1's balanced accuracy
and attack recall columns as "pinned" or "degenerate by construction". On the
full range they move, so they carry some information — they are secondary
because of **L16** ("rank well and threshold badly") and **L23** ("weighted
precision collapses at the default threshold"), not because they are constant.
The primary threshold-free ROC AUC is unaffected and stands. The §10.2
degeneracy diagnostic is unaffected and will simply read "not degenerate",
which is a reading, not a failure.

Correcting the amendment would mean editing a dated document after numbers
were looked at, which the pre-registration forbids. A second dated amendment
is the user's call.

---

## 6. `peak_rss_mb` — a defect in Table 6's aggregation, not in any verdict

**No hypothesis is attached to Table 6, so nothing above depends on this.** It
is recorded here because the fix must land before Table 6 is quoted.

`peak_rss_mb` is a **high-water mark for the life of the process that wrote
the unit**, not a per-unit measurement. Measured across the 36 units on disk:

| value | units | what it is |
|---|---|---|
| 499.6 – 523.1, distinct per unit | 10 | the 7 Nov `sweep_block.sh` block — one short-lived process per unit, so these *are* per-unit peaks |
| **513.8**, identical | 24 | one long-lived 21 Nov process — a single HWM reported 24 times |
| **496.7**, identical | 2 | the 21 Nov 22:09 resume process — one HWM reported twice |

Source: `results/lanl_v2/units_default/*.json`, field `peak_rss_mb`.

**The brief calls this three regimes. It is one regime per launch**, and the
4-unit resume now running will add a fourth distinct value. A caption naming a
fixed count will be wrong again at the next launch, so the caption must state
the per-launch semantic instead.

**Why the mean is wrong and not merely noisy.** For the 24 identical values,
`agg()` is not averaging 24 measurements — it is counting **one** measurement
24 times, then reporting `std = 0.0`, which reads as perfect reproducibility
when it is a single observation. That is the same failure mode as a seed loop
that varies nothing.

**`unit_seconds` has a related but milder defect, found while measuring this.**
The same cell, arm and system was timed at 653.8 s / 705.4 s on 7 Nov and
1359.8 s / 1370.8 s on 21 Nov — a 2.0× machine-state difference across
launches (`with_auth_naive_seed{0,1}` vs `seed{2,3}`, same file and field). A
mean over those five seeds mixes two machine states. Unlike `peak_rss_mb`,
each value is still a genuine per-unit measurement, so a mean remains a
legitimate *operational* summary; it must simply be captioned as spanning
launches. It is also why the `PREDICT=1524` floor in `sweep_block.sh` sits
below the real `with_auth`/`prismflow` cost.

**Planned handling:** `experiments/lanl_v2/fill_analysis_patch.py`, not yet
applied. The launch identity is **not recorded in the unit JSON** — there is
no `process_id` and no `launch_time` field — so provenance that is not
derivable from committed data is reported as `not measured` rather than
invented.

---

## 7. What this sheet does not decide

- No limitation's status (PRE_REGISTRATION §7.6).
- Tables 1, 2, 3 and 6 — pending the sweep at 40/40.
- Table 5 and every cross-mode figure — pending mode A.
- The design-effect ENIV — not measured; `collect_predictions` is frozen and
  returns batch-weighted scalars with no dependence matrix.
- How much of `proc`'s 0.7083759667 is selection-induced enrichment versus
  behavioural signal — not measured, no decomposition exists, and unlike
  `fallback_survivor_k2` it **cannot** be set aside as unusable, because
  `proc` occupancy is visible to the encoders through the view mask.
- L27's contrast at the intended 1/1 share — not measured; testing it needs
  the 400,000 bar raised, which is a review-gate decision.
