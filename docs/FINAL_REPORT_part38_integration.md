# Part 38 → `docs/FINAL_REPORT.md`: integration preview

**A preview, not a patch. `docs/FINAL_REPORT.md` is NOT edited by this file.**
Written 2026-11-21 against `FINAL_REPORT.md` at **628 lines**, with the mode
`default` sweep at 26 of 40 units. Line numbers below are from that revision
and will shift if the report changes first; re-derive them before applying.

The prose to insert already exists as `docs/part38_section.md`. This file says
**where it goes, what it must not contradict, and what has to change
elsewhere in the report as a consequence** — which is the part that cannot be
done by copy-paste.

---

## 1. Where §4.4 goes

`FINAL_REPORT.md` §4, *Real-data results (Parts 28–30)*, currently runs:

| line | heading |
|---|---|
| 182 | `## 4. Real-data results (Parts 28–30)` |
| 187 | `### 4.1 AIT Log Data Set V2.0 — what the label actually is` |
| 224 | `### 4.2 AIT is solved by one view, so it is not evidence about fusion` |
| 239 | `### 4.3 LANL cyber1 — clean-data evidence only` |
| 310 | `**Sources:** ...` then `---` then `## 5.` at 314 |

**Insert `### 4.4 LANL redesigned — what changes when selection stops reading
`auth`` after line 310's `**Sources:**` paragraph and before the `---` that
closes §4.** It belongs inside §4 and after 4.3, because it is the same
dataset family measured under a different sampling rule and 4.3's numbers are
the thing it is compared against.

**Two header edits follow from the insertion:**

1. **Line 182** — `## 4. Real-data results (Parts 28–30)` becomes
   **`## 4. Real-data results (Parts 28–30, 38)`**. The section no longer
   covers only 28–30.
2. **Line 184-ish**, §4's two-line preamble ("Two real datasets. Neither is
   pooled with the other, and AIT's two directions are never averaged.") gains
   a third sentence: **"LANL is measured under two different sampling designs,
   Part 30's and Part 38's, and those two are not comparable row-for-row
   either."** Without it a reader will diff 4.3's table against 4.4's and
   believe the difference is a result.

## 2. What 4.4 must not contradict

**4.3 stays exactly as it is.** Its eight-row table is the record of Part 30's
design. Part 38 is a different sample — `PRE_REGISTRATION.md` §2: Part 38's
sample "is not a subset or a superset of Part 30's, and the two sample tables
are **not comparable row-for-row**." Any instinct to "update" 4.3's numbers
with Part 38's is wrong and must be resisted at review.

Five existing claims in the report that 4.4 touches, and how:

| existing claim | where | what 4.4 does to it |
|---|---|---|
| "Auth presence alone is a selection signal (L24). Every attack sample has an `auth` record, so the untrained `auth_present_indicator` reaches test AUC **0.6637**. The `without_auth` arm removes the auth features; it does not remove that selection." | §4.3, 4th governing fact | **Strengthens and relocates it.** On a sample that never consults `auth`, the attack rate is still exactly 1.0 and the baseline reads **0.6611838191552647**. But `proc` occupancy now reads **0.7083759666864962** — higher — so the sentence "it does not remove that selection" becomes true of Part 38 *for a different view*. |
| "One host carries a third of the attack class (L27) … Dropping its test rows lowers test ROC AUC by **0.036** to **0.052**." | §4.3, 2nd governing fact; §8 line ~507 | **Pending.** 4.4's H2 figure replaces nothing until Table 3 is filled at 40/40, and must carry the realised **1/4** share. |
| "The sample is a declared case-control sample, not the network (L23) … test prevalence **0.0219**." | §4.3 qualification | **Adds a second design.** Part 38's test prevalence is **0.0048543689320388345** on 84,460 rows. Both are artefacts of their own rules; neither is the network. |
| "Execution mode is a hidden factor (L28). A 5-seed std computed inside one run …" | §8 line ~520 | **Unchanged, and must gain a sentence** saying L28 cannot be closed by its own text under Part 38's pre-registration. See §4 below. |
| "An untrained single scalar reads most of that ranking. `total_records_across_views` … reaches test ROC AUC **0.6843** (L26)." | §4.3 | **Weakens on the new sample.** The same scalar falls to **0.5862805385876582**. That is evidence the new rule changed something, and it belongs in 4.4 rather than as an edit to 4.3. |

## 3. Tables 4.4 adds

Two are ready now; four wait on the sweep.

| table | content | source | status |
|---|---|---|---|
| **4.4a** | the three trivial baselines, Part 38 against Part 30, with train AUC and orientation | `results/lanl_v2/h1_auth_presence.json`, `baselines.<name>` | **ready** |
| **4.4b** | per-view attack/benign record-presence contingency on both test blocks | `results/lanl_v2/h1_conditional_rates.json`, `part30.per_view` / `part38.per_view` | **ready** |
| **4.4c** | every selection input as its own baseline, whole block and within attack hosts | `results/lanl_v2/h1_occupancy_decomposition.json` | **ready** |
| **4.4d** | ROC AUC mean ± std per arm × system, 8 rows | `filled_tables.json`, `table1_roc_auc_per_cell` | **pending the sweep** |
| **4.4e** | H2 — leave-`C17693`-out delta, 8 cells, against Part 30's eight | `filled_tables.json`, `table3_leave_host_out` | **pending the sweep** |
| **4.4f** | H3 — cross-mode sign agreement | — | **pending mode A** |

**4.4c should be trimmed, not pasted whole.** `part38_section.md` §4.3 carries
all ten scalars in two columns because the README needs the full instrument
record. The report wants the four rows that carry the argument —
`fallback_survivor_k2`, `occupancy_proc`, `auth_present_indicator`,
`occupancy_dns` — with a pointer to `experiments/lanl_v2/README.md` §4.2 for
the rest. The two instrument readings (`attack_host_indicator`,
`identity_hash_kept`) must survive the trim **as prose**, because dropping
them silently would leave the top row unexplained.

**4.4d's row order must match 4.3's** (`with_auth` naive, `with_auth`
PrismFlow, `without_auth` naive, …) so the two tables can be read against each
other, while the §4 preamble warns they are not comparable row-for-row.

## 4. Edits required elsewhere in the report

These are the ones that get forgotten. Each is a *consequence* of 4.4, not
part of it.

**§8, Limitations and threats to validity (line 481–544).**

- The **L24** paragraph (line ~494) currently ends with the `without_auth` arm
  removing features but not selection. Append: that a sample drawn without
  reference to `auth` now exists, that the auth-presence baseline did not fall
  (H1 disconfirmed, drop **0.0024714021996127267** against a required
  **0.05**), and that the selection effect relocated to `proc`. **L24 stays
  OPEN.**
- The **L27** paragraph (line ~507) gains the realised **1/4** share once
  Table 3 lands, and the note that H2's 0.036 bar sits inside Part 30's own
  band (smallest committed cell **−0.035962653276654266**).
- The **L28** paragraph (line ~520) gains one sentence that does **not** depend
  on mode A: **L28's closing condition asks for a std covering mode as well as
  seed, and Part 38's pre-registration §6 forbids pooling the two modes, so
  L28 cannot be closed on its own terms by this design.**

**§10, Not measured (line 584–619).** Add rows, in the existing
`| quantity | status |` format:

| quantity | status |
|---|---|
| Part 38's Tables 1, 2, 3 and 6 | **not measured** — the sweep is at 26 of 40 |
| Part 38's H2 verdict | **not measured** — pending Table 3 at 40/40 |
| Part 38's H3 verdict and every cross-mode figure | **not measured** — mode A not run |
| the design-effect ENIV on LANL v2 | **not measured** — `collect_predictions` returns batch-weighted scalars and no dependence matrix, and is frozen |
| how much of `proc`'s 0.7083759666864962 is selection-induced versus behavioural | **not measured** — no decomposition exists |
| L27's contrast at the intended 1/1 share | **not measured** — the fallback fired at `k = 2` |

**§9, Future work (line 545–583).** The honest next item from Part 38 is a
selection rule whose admission criterion is independent of every view's
occupancy — **and the entry should say that it may not exist on this data**,
since a cell with no occupied view carries no features. That is a stronger and
more useful statement than "try another sampling rule".

**"How to re-check this report" (line 620–628).** Add the three scalar passes
and their test, which reproduce 4.4a–c in seconds with no GPU:

```sh
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/h1_auth_presence.py
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/h1_occupancy_decomposition.py
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/h1_conditional_rates.py
PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest -q tests/unit/test_h1_scalar_passes.py
```

**§1 and §5 need nothing.** Part 38 is a sampling and selection result; it
makes no detection claim, so the thesis and the "what works" verdicts are
untouched.

## 5. How to integrate

1. **Re-derive the line numbers.** This preview is against the 628-line
   revision.
2. **Fill `experiments/lanl_v2/ANALYSIS_TEMPLATE.md` first**, from
   `filled_tables.json`, at 40/40. Do not write 4.4d or 4.4e from anything
   else — in particular not from a `--dry-run` output, which is stamped
   `DRY_RUN_NOT_A_FINDING` and is deleted rather than committed.
3. **Copy `docs/part38_section.md` §1 and §4 into 4.4**, trimming 4.4c per §3
   above. Its §2 (what Part 30 left open) duplicates §4.3 and §8 and should be
   cut to a cross-reference. Its §3 (the design) compresses to one paragraph
   plus the selection rule — the report is not the place for the full
   pre-registration.
4. **Carry §5.2's handling of the H2 preview verbatim.** The mid-sweep dry run
   is named, its output recorded as deliberately uncommitted, and **no figure
   from it appears.** The report must not acquire a number that has no
   committed path.
5. **Apply the §4 edits above**, then check §8 and §10 agree with 4.4. A
   reader who finds L24 described as closed in §8 and open in 4.4 will trust
   neither.
6. **Keep the two headers in §1 of this file.**
7. **Verify every number resolves.** Each figure in 4.4 must name its JSON
   path. The committed scalar-pass outputs are
   `results/lanl_v2/h1_auth_presence.json`,
   `h1_occupancy_decomposition.json`, `h1_conditional_rates.json`.

## 6. The one-sentence version, for the §4 preamble or an abstract

> Part 38 rebuilt the LANL sample so that control cells are chosen without
> consulting the `auth` stream — L24's written closing condition — and the
> auth-presence baseline did not fall: **0.6611838191552647** against Part
> 30's **0.6636552213548774**, a drop of **0.0024714021996127267** where the
> pre-registration required **0.05**. The volume-based selection signals did
> fall, and `proc` occupancy rose to **0.7083759666864962**, above `auth`'s
> own figure. **The selection did not drop; it moved.**

**If the write-up reads better than that, it is wrong.** The four things Part
38 has to say are: H1 disconfirmed, the selection effect relocated rather than
removed, L27's contrast tested at a quarter of its intended strength, and L28
unclosable by its own text under this pre-registration.
