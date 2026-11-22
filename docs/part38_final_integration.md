# Part 38 — final-report integration instruction

**What this file is.** The mechanics of landing Part 38 in
`docs/FINAL_REPORT.md`: where §4.4 goes, what else in the report has to move
when it does, and which placeholder is filled from which committed JSON path.

**What this file is not.** It is **not** the §4.4 prose. That prose is already
written, in final form, in **`docs/part38_section.md`** — 400 lines, every
number cited, absences marked `not measured`. Copying it here would make a
third copy of text that already has two homes (`part38_section.md` and
`experiments/lanl_v2/ANALYSIS_TEMPLATE.md` §3–§5), and the copy that drifts is
the one that gets quoted. **`part38_section.md` is the source; this file says
where to put it.**

Written 2026-11-22 with the mode `default` (B) sweep at **36 of 40 units**.
Nothing here may be applied to `FINAL_REPORT.md` until the sweep is at 40/40
and `results/lanl_v2/filled_tables.json` exists, because §4.4 carries table
references that would otherwise point at a file that is not there.

---

## 1. Insertion point

`docs/FINAL_REPORT.md` as committed:

```
line 239   ### 4.3 LANL cyber1 — clean-data evidence only
...
line 312   **Sources:** results/ait_clean/summary.md, ... results/lanl/population.json.
line 313   ---
line 314   ## 5. What works: detection
```

**§4.4 is inserted between line 312 and line 313** — after §4.3's `Sources:`
line, before the `---` that closes §4. It does not replace or edit §4.3: Part
30's table stands as the record of Part 30's design, and Part 38 is a
**different sample** that is not comparable row-for-row
(PRE_REGISTRATION §2).

Take the body of `docs/part38_section.md` from its `## 1. Headline` onward,
demote every heading by two levels (`## 1. Headline` → `#### 4.4.1 Headline`,
`### 4.1 H1 is disconfirmed…` → `##### H1 is disconfirmed…`), and drop that
file's own preamble lines 1–15, whose "integrates into §4" note becomes
redundant once it is in §4.

---

## 2. What else in the report moves

Four edits outside §4.4 itself. Each is mechanical; none is a status change.

| # | location | change | why |
|---|---|---|---|
| 1 | line 182, `## 4. Real-data results (Parts 28–30)` | → `(Parts 28–30, 38)` | the section now contains a Part 38 subsection |
| 2 | lines 283–298, §4.3's L24 and L27 bullets | append one sentence each: "Part 38 retested this on a new sample; see §4.4." | a reader at §4.3 must not take Part 30's figure as the last word |
| 3 | line 608, §10's L28 row | append: "Part 38 ran four arms in mode B only; mode A is not run, so L28 stays **not measured** for cross-mode spread on all four arms." | §10 is the Not-measured register and must name the new absence |
| 4 | `## How to re-check this report` (line 620) | add the four commands from `part38_section.md` §8 | the re-check block is the report's reproduction contract |

**Not touched:** `docs/KNOWN_LIMITATIONS.md`, `PRE_REGISTRATION.md`, and
every L24/L27/L28 **status**. PRE_REGISTRATION §7.6: this programme "does not
mark L24, L27 or L28 closed, resolved or narrowed. Status changes require the
full sweep and a review gate." Proposed status text is queued in
`docs/proposed_limitations_update.md` under `## Part 38 proposals`, marked
PROPOSAL, and goes nowhere else.

---

## 3. Placeholder fill map

Every placeholder in `part38_section.md`, with the exact path that fills it.
A placeholder still present when the report is read is **a defect in the
report, not a zero** — leave it as `not measured` with its reason rather than
filling it with anything derived, rounded or guessed.

### 3.1 Ready now — needs no sweep

| placeholder | fill with | source path |
|---|---|---|
| §4.1 H1 verdict | DISCONFIRMED, 0.6611838191552647 vs 0.6136552213548774 bar | `results/lanl_v2/h1_auth_presence.json`, `h1` |
| §4.2 contingency survived | attack rate 1.0 → 1.0, benign +0.004942804399225453 | `results/lanl_v2/h1_conditional_rates.json`, `auth_verdict` |
| §4.3 the `proc` relocation | 0.7083759666864962; benign rate 0.697309 → 0.914313 | `results/lanl_v2/h1_occupancy_decomposition.json`, `whole_test_block.occupancy_proc`; `h1_conditional_rates.json`, `per_view_rate_comparison.proc` |
| §4.4 cross-validation | rate-implied vs measured AUC agree to 0.00e+00 on all four views | `h1_occupancy_decomposition.json` + `h1_conditional_rates.json` |
| §5.4 range dependence | days 1–16 balanced acc 0.5512 / 0.5549, recall 0.1024 / 0.1098 | `results/lanl_v2/smoke_days16.json` (**2 seeds, 1 arm — not a finding**) |

These five are the whole of what Part 38 can state today, and `part38_section.md`
already states them. **No edit to that file is needed for this group.**

### 3.2 Pending the sweep at 40/40

| placeholder | fill with | source path |
|---|---|---|
| `[SWEEP_RESULT]` §5.1 "25 of 40" | `40 of 40` | `ls results/lanl_v2/units_default` |
| §5.1 table, rows 1/2/3/6 "pending the sweep" | the filled values | `results/lanl_v2/filled_tables.json` |
| Table 1 — AUC mean ± std, 8 rows | `per_cell.<arm>/<system>.roc_auc.{mean,std,n}` | `results/lanl_v2/sweep_default.json` |
| Table 2 — per-seed AUC, 40 cells | `roc_auc` per unit | `results/lanl_v2/units_default/<cell>.json` |
| Table 2 — per-seed signs | `seeds_positive`, `seeds_negative`, `seeds_exactly_zero` **before** any mean | `sweep_default.json`, `prismflow_minus_naive_roc_auc.<arm>` |
| Table 3 / H2 verdict | the eight `C17693` deltas | `filled_tables.json`, `table3_leave_host_out.per_cell.<arm>/<system>.per_host.C17693` |
| Table 4 lower block — arm gap | `with_auth` − `without_auth`, naive and prismflow, plus per-seed signs | Table 1 and Table 2 |
| Table 6 | per-launch, see §4 below | `filled_tables.json`, `table6_operational` |
| §6 row "Tables 1, 2, 3 and 6" | delete the row once filled | — |
| §7 L27 bullet "proposed pending" | the H2 verdict with its three mandatory caveats | `docs/part38_verdict_worksheet.md` §3 |

**Fill rules that are not negotiable** (ANALYSIS_TEMPLATE.md §6):
a cell with `n seeds` below 5 is `n = <k>, NOT A FINDING` and its ± is
withheld; per-seed signs are reported before every aggregate mean; every ENIV
value carries its thread count; nothing pools the two execution modes.

### 3.3 Pending mode A — may stay pending

| placeholder | status |
|---|---|
| Table 5, every cell | `pending mode A` — never 0, never blank |
| H3 verdict | not measured |
| §7 L28 bullet | proposed pending, NARROWED at best, **unclosable by L28's own text** under this pre-registration |

Mode A is ~2.0 h across 11 concurrent single-thread processes at ~5,372 MB
(`results/lanl_v2/sweep_estimate.json`). If it is dropped, §4.4 ships with
Table 5 `pending mode A` and that is a complete, honest section — **not an
incomplete one.** L28 could not be closed even by a clean H3 confirmation.

---

## 4. The one content change §4.4 needs that is not a placeholder

`part38_section.md` §5.1 lists Table 6 as "per-unit wall-clock, peak RSS,
degeneracy diagnostic — pending the sweep". That row needs a caveat added
before Table 6 is quoted anywhere:

> `peak_rss_mb` is a high-water mark for the life of the **process** that wrote
> the unit, not a per-unit measurement, so it is reported per seed and is not
> averaged across seeds. The 36 units on disk were written by three launches
> and carry three HWM values; the launch identity is **not recorded in the
> unit JSON**, so it is reported as not measured rather than reconstructed.

Measured basis, and why the fix cannot name a fixed number of regimes:
`docs/part38_verdict_worksheet.md` §6. The patch is
`experiments/lanl_v2/fill_analysis_patch.py`, not yet applied.

---

## 5. Apply order

1. Sweep reaches 40/40 → `sweep_default.json` rewritten by `aggregate_sweep`.
2. Apply `fill_analysis_patch.py` to `fill_analysis.py`.
3. Run `fill_analysis.py` → `results/lanl_v2/filled_tables.json`.
4. Fill `ANALYSIS_TEMPLATE.md` Tables 1, 2, 3, 6 and the §3 verdicts.
5. Fill `part38_section.md` §3.2 placeholders from the same file.
6. Insert §4.4 into `FINAL_REPORT.md` and apply §2's four edits.
7. Verify with the repo's own number check — and note it is a **presence**
   check, not an identity check: it confirms a number appears somewhere in a
   cited file, so resolve each one to an exact JSON path by hand.

Steps 1–5 are reversible. **Step 6 is the first that edits the report**, and
nothing above it should be merged ahead of it.
