# Part 38 — final-report integration instruction

**What this file is.** The mechanics of landing Part 38 in
`docs/FINAL_REPORT.md`: where §4.4 goes, what else in the report has to move
when it does, and which placeholder is filled from which committed JSON path.

**What this file is not.** It is **not** the §4.4 prose. That prose is already
written, in final form, in **`docs/part38_section.md`** — 617 lines, every
number cited, absences marked `not measured`. Copying it here would make a
third copy of text that already has two homes (`part38_section.md` and
`experiments/lanl_v2/ANALYSIS_TEMPLATE.md` §3–§5), and the copy that drifts is
the one that gets quoted. **`part38_section.md` is the source; this file says
where to put it.**

Written 2026-11-22 before the sweep finished; **updated the same day with mode
`default` (B) complete at 40 of 40 units.** Steps 1–5 of §5 are done, so
§4.4 is now ready to insert — `results/lanl_v2/filled_tables.json` is
committed and every table reference in §4.4 resolves. **Step 6 is the next
action and is the first that edits `FINAL_REPORT.md`.**

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

**Five** edits outside §4.4 itself. The first four are mechanical; the fifth
carries a result. None is a status change.

| # | location | change | why |
|---|---|---|---|
| 1 | line 182, `## 4. Real-data results (Parts 28–30)` | → `(Parts 28–30, 38)` | the section now contains a Part 38 subsection |
| 2 | lines 283–298, §4.3's L24 and L27 bullets | append one sentence each: "Part 38 retested this on a new sample; see §4.4." | a reader at §4.3 must not take Part 30's figure as the last word |
| 3 | line 608, §10's L28 row | append: "Part 38 ran four arms in mode B only; mode A is not run, so L28 stays **not measured** for cross-mode spread on all four arms — and Part 38's effect sizes make H3 unsatisfiable on its sample, so mode A would not supply a verdict either (§4.4)." | §10 is the Not-measured register and must name the new absence |
| 4 | `## How to re-check this report` (line 620) | add the commands from `part38_section.md` §8 — now **six**, the three scalar passes plus `fill_analysis.py` and the aggregate-only sweep re-run | the re-check block is the report's reproduction contract |
| 5 | §6 "What does not" and §8 "Limitations" | record that on Part 38's LANL sample **no arm favours the discount** — three of four sign-undetermined, `without_auth` negative at −0.0032 ± 0.0022 in 4 of 5 seeds | this is a null result for the report's central thesis and must not live only inside §4.4 |

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

These five were what Part 38 could state before the sweep finished, and
`part38_section.md` §4 states them. **Unchanged by the 40/40 fill** — they are
seed-free scalars over the sample table and no sweep result touches them, so
§4 of `part38_section.md` was not edited when §5 was filled.

### 3.2 DONE — filled at 40/40 on 2026-11-22

**This whole group is complete.** The sweep finished at 40/40 (*4 trained, 6
skipped, 7844s*), `fill_analysis.py` wrote
`results/lanl_v2/filled_tables.json`, and both `part38_section.md` and
`ANALYSIS_TEMPLATE.md` are filled and cross-checked against that file
programmatically with 0 mismatches.

| placeholder | filled with | source path |
|---|---|---|
| ~~`[SWEEP_RESULT]` §5.1 "25 of 40"~~ | `40 of 40` | done |
| ~~§5.1 table, rows 1/2/3/6~~ | **filled** | done |
| ~~Table 1 — AUC mean ± std, 8 rows~~ | all 8 cells `n = 5` | `filled_tables.json`, `table1_roc_auc_per_cell.per_cell.<arm>/<system>.roc_auc.{mean,std,n}` |
| ~~Table 2 — per-seed AUC, 40 cells~~ | **filled** | `table2_per_seed.per_cell.<arm>/<system>.seed<N>` |
| ~~Table 2 — per-seed signs~~ | **filled**, before every mean | `table2_per_seed.prismflow_minus_naive.<arm>` |
| ~~Table 3 / H2 verdict~~ | **DISCONFIRMED**, 8 of 8 cells | `table3_leave_host_out.h2_verdict` |
| ~~Table 4 lower block — arm gap~~ | −0.0498525761 / −0.0448397063, 5/5 seeds | Table 1 and Table 2 |
| ~~Table 6~~ | **filled**, `peak_rss_mb` per seed | `table6_operational` |
| ~~§6 row "Tables 1, 2, 3 and 6"~~ | row removed | done |

**Three results arrived that this map did not anticipate**, and §4.4 now
carries them — check they survive the insertion:

1. **Dropping the auth view raises ROC AUC** by 0.0498525761 (naive) and
   0.0448397063 (prismflow), negative in 5 of 5 paired seeds in both systems.
   `without_auth` is the best arm, which reverses §4.3's framing of Part 30.
   **Edit 2 in §2 below matters more because of this**, not less.
2. **H2 was disconfirmed in the opposite direction** — sensitivity grew to
   0.0997–0.1134 from Part 30's 0.0360–0.0519, bands not overlapping.
3. **No arm favours the discount.** Three of four arms are sign-undetermined
   and `without_auth` is negative. This is a **null result for the report's
   central thesis on this dataset** and §4.4 labels it as one. It belongs in
   `FINAL_REPORT.md` §8 (Limitations) and §6 (What does not), not only in §4.4
   — **edit 5 in §2, added for this, and the one most likely to be missed.**

§4.4's §7 L27 and L28 bullets are filled with their measured statuses;
`docs/proposed_limitations_update.md` is **not** yet updated to match, which is
step 7 of §5.

**Fill rules that are not negotiable** (ANALYSIS_TEMPLATE.md §6):
a cell with `n seeds` below 5 is `n = <k>, NOT A FINDING` and its ± is
withheld; per-seed signs are reported before every aggregate mean; every ENIV
value carries its thread count; nothing pools the two execution modes.

### 3.3 Pending mode A — may stay pending

| placeholder | status |
|---|---|
| Table 5, **mode A columns** | `pending mode A` — never 0, never blank. Its **mode B columns are filled** |
| H3 verdict | not measured — **and unreachable on this sample**, see below |
| §7 L28 bullet | proposed pending, NARROWED at best, **unclosable by L28's own text** under this pre-registration, **and now unclosable by H3 either** |

**New at 40/40, and it changes the decision about running mode A.** H3 needs
sign agreement across modes in **all four arms**, and on mode B three of the
four arms have **no established sign** — the PrismFlow − naive difference is
smaller than their own per-seed std. A sign that does not exist cannot agree,
so mode A could at best decide `without_auth` and H3's condition is
unsatisfiable on this sample whatever mode A shows. Running mode A is still
worth doing for the across-mode spread itself, which is what L28 is about; it
will **not** produce a verdict, and §4.4 says so.

Mode A is ~2.0 h across 11 concurrent single-thread processes at ~5,372 MB
(`results/lanl_v2/sweep_estimate.json`). If it is dropped, §4.4 ships with
Table 5 `pending mode A` and that is a complete, honest section — **not an
incomplete one.** L28 could not be closed even by a clean H3 confirmation.

---

## 4. The `peak_rss_mb` caveat — DONE, and sharper than planned

**Done.** `part38_section.md` §5.5 now carries it and `fill_analysis.py` was
patched (`ac98a1f`), so `peak_rss_mb` is emitted per seed with `n_distinct`
and is never averaged.

The planned wording said "a high-water mark for the life of the process …
three launches, three HWM values". **The sweep measured something stronger
and the shipped wording reflects it:** the figure is a **running maximum**, so
it is non-decreasing within a launch and a unit's number can be inherited from
an earlier unit in the same process. `with_auth_naive_seed4` recorded 499.5
and `with_auth_prismflow_seed2` recorded 528.8 **in the same process**, and
the OS counter `PeakWorkingSetSize` for that process read 537.3 MB before
`with_auth_prismflow_seed3` finished — which is exactly the figure that unit
then reported. So there is no fixed number of regimes, and not even one value
per launch.

Measured basis, and why the fix cannot name a fixed number of regimes:
`docs/part38_verdict_worksheet.md` §6. The patch is
`experiments/lanl_v2/fill_analysis_patch.py`, not yet applied.

---

## 5. Apply order

1. ~~Sweep reaches 40/40 → `sweep_default.json` rewritten by
   `aggregate_sweep`.~~ **DONE** — `11cf3af`, plus `a0f58ab` for the 12 bare
   `NaN` the generator still emits.
2. ~~Apply `fill_analysis_patch.py` to `fill_analysis.py`.~~ **DONE** —
   `ac98a1f`.
3. ~~Run `fill_analysis.py` → `results/lanl_v2/filled_tables.json`.~~
   **DONE** — `11cf3af`, 15 seconds, exit 0, no gate failures.
4. ~~Fill `ANALYSIS_TEMPLATE.md` Tables 1, 2, 3, 6 and the §3 verdicts.~~
   **DONE** — `9213d0d`.
5. ~~Fill `part38_section.md` placeholders from the same file.~~ **DONE** —
   `ab37aeb`.
6. **NEXT, and the first step that edits the report.** Insert §4.4 into
   `FINAL_REPORT.md` and apply all **five** of §2's edits — including edit 5,
   the null result in §6 and §8, which is the only one that carries a finding
   rather than a cross-reference.
7. Update `docs/proposed_limitations_update.md`: L27's proposal now needs the
   disconfirmation and its opposite direction; L28's needs the unreachability.
   **Not yet done**, and no limitation status may change without the review
   gate (PRE_REGISTRATION §7.6).
8. Verify with the repo's own number check — and note it is a **presence**
   check, not an identity check: it confirms a number appears somewhere in a
   cited file, so resolve each one to an exact JSON path by hand.

Steps 1–5 are done and were reversible. **Step 6 is the first that edits the
report**, and nothing above it should be merged ahead of it.
