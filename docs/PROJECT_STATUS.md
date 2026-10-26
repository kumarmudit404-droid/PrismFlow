# Project status

Final status as of 2026-10-25, with the limitation counts refreshed
2026-10-26 after Part 35 merged. Consistent with `docs/CONTRACT.md`,
`docs/KNOWN_LIMITATIONS.md` and `docs/v2-known-limitations.md`; where this file
and one of those disagree, those are authoritative and this one is wrong.

As merged, `docs/KNOWN_LIMITATIONS.md` stands at **32** entries — **1**
resolved, **3** closed and **28** open — and `docs/v2-known-limitations.md` is
unchanged at **9** — **2** resolved, **2** closed, **1** new and **5** open.
Resolved and closed are not the same marker and are never merged: resolved
means the defect was fixed, closed means the entry reached a terminal
statement without one. Counts from `web/data/limitations.json`, which
`web/build_data.py` derives from the two records.

**Status of the real-data track and the site (Parts 26-33), added 2026-10-25.**
The numbers in sections 1 to 3 below are the synthetic and V2 record and are
unchanged. What has happened since 2026-10-07 is status, not a revision of
them:

- **Parts 26-27 — AIT Log Data Set V2.0 loaded** and its data card written.
- **Part 28 — PrismFlow run on AIT clean data**, two directions, never pooled.
  On this label **one view (`dns`) solves the task**, so no Part 28 number is
  evidence about fusion (`docs/KNOWN_LIMITATIONS.md` L17).
- **Part 29 — chorus attack and detector on AIT.** The detector inverts under
  attack on real data, and the all-or-nothing protocol cannot speak to
  deployment (L18).
- **Part 30 — LANL cyber1**, four view configurations. **Clean-data evidence
  only**: tests B, D and the detector half of C were not run, and no
  robustness or attack claim carries over (L30). Its sample is a declared
  case-control sample (L23) with an execution-mode spread the per-seed std
  understates (L28).
- **Part 31 — the real-data section on the site**, with a per-dataset claim
  table whose verdicts are CONFIRMS, CONTRADICTS, NO EVIDENCE and needs more
  data.
- **Part 32 — a live diagnostic scoring service** (`serve/`) and the site's
  *Try it* panel. The site ships with the API switched **off**; the
  `Dockerfile` **has not been build-tested**
  (`docs/part32-live-service.md`).
- **Part 33 — `docs/FINAL_REPORT.md`**, every number resolved against a cited
  committed file by `docs/check_report_numbers.py`. Proposed changes to the
  limitations record are queued in `docs/proposed_limitations_update.md` and
  **have not been applied**.

**The headline verdicts are unchanged by the real data.** Detection is
evidenced **on synthetic data only**; the defence is **refuted on synthetic
data** and is **not validated on AIT or on LANL**.

Every figure below names the committed file it came from. Where no committed
file holds a number, this file says **not measured** and does not estimate one.

---

## In one paragraph

PrismFlow tests whether discounting fused confidence by estimated inter-view
dependence (ENIV) yields fusion robust to colluding or compromised views. On
synthetic data the **detection half is evidenced and the defence half is
refuted**: the estimator identifies collusion reliably, and the correction
driven from that signal made the attack succeed slightly *more* often than the
undiscounted baseline. On real retrieved evidence (V2) the question was never
reached: four of five angles retrieve almost nothing, so **no calibration
metric has been computed on real data, and none can be computed meaningfully
today**. The applied-to-benchmarks result — that a standard 6-view benchmark
carries about 3.3 effective views — stands on its own and is the project's most
transferable finding.

---

## 1. What is evidenced

Each row is 5 seeds with mean ± sample std, which is the project's minimum for
calling anything a finding (`docs/CONTRACT.md` §5).

### 1.1 A standard multi-view benchmark is far more redundant than its view count

| quantity | value | source |
|---|---|---|
| nominal views, Handwritten (UCI Multiple Features) | 6 | `docs/DATASETS.md` |
| measured ENIV | 3.28 ± 0.07 | `results/real/README.md` |
| efficiency ratio | 0.547 ± 0.011 | `results/real/README.md` |
| upper tail dependence λ_U at q = 0.90 | 0.127 ± 0.042 | `results/real/README.md` |
| test accuracy | 0.981 ± 0.003 | `docs/DATASETS.md` (and `handwritten/redundancy.json`) |
| ENIV after within-class feature permutation | 5.51 ± 0.08 | `results/real/README.md` |

5 seeds, 800 test samples, measured before any duplication, corruption or
attack. The permutation control makes the views independent by construction
while leaving sample size, feature dimension and marginals untouched, so **2.23
of the 2.72-view shortfall is measured redundancy and roughly half a view is
estimator floor**. Reproduced by `experiments/real/run_real.py`.

The redundancy is uneven: dropping `fou` or `mor` costs 0.74 and 0.68 effective
views, while dropping `fac` or `pix` — the two largest views — costs **0.23**
and 0.21, because those two are 0.852 dependent on each other
(`results/real/README.md`, the drop-one table).

> **Discrepancy, resolved 2026-10-08.** The `fac` drop-one figure had been
> given as **0.24** in two places. Computed per seed from the committed source
> (`results/real/handwritten/redundancy.json`, `per_seed[].eniv` minus
> `per_seed[].eniv_leave_one_out.fac`) it is **0.2349 ± 0.0305** over 5 seeds,
> i.e. **0.23 ± 0.03** at the drop-one table's rounding. Corrected to 0.23 in
> `web/index.html` (commit `164be26`) and in `docs/DATASETS.md`. An earlier
> version of this note also named the root `README.md`: that was wrong — the
> README never carried the figure, so nothing was changed there.

**What this is not.** Real views carry no ground-truth dependence, so ENIV is
*applied* here and not *validated* here. The estimator's only validation is
Part 05, on the synthetic generator where `rho` is a parameter that was set
(`docs/KNOWN_LIMITATIONS.md` L9).

### 1.2 Detection of collusion

| quantity | value | source |
|---|---|---|
| ENIV under coordinated attack | 3.36 → 2.70 | `docs/CONTRACT.md` §1 (Part 09); `results/chorus/` |
| dependence among compromised views | +0.7351 ± 0.0914 | `docs/CONTRACT.md` §1 (Part 09); `results/chorus/` |

5 seeds, `experiments/chorus/`. The estimator identifies collusion reliably.

### 1.3 One measured benefit of the discount under attack

| quantity | value | source |
|---|---|---|
| belief in the wrong class when the attack succeeds, ε = 1.0 | −0.0562 ± 0.0069, 5/5 seeds, bars separate | `docs/CONTRACT.md` §1 (Part 09) |

### 1.4 Calibration under view duplication

Under view duplication PrismFlow **preserves** resolution, accuracy and AURC
relative to naive fusion, and the robustness to added copies is a property of
the discount mechanism rather than of training adaptation — a naive model
frozen at k = 0, with the discount applied at inference only, reproduces it.
Figures, confounds and scope are in `docs/EVALUATION_PROTOCOL.md`, "Current
claim"; numbers in `results/calibration_duplicated/`.

---

## 2. What is refuted

### 2.1 The defence half of the hypothesis

| quantity | value | source |
|---|---|---|
| attacked cells where PrismFlow's attack success exceeded naive's | 12 of 12 | `docs/CONTRACT.md` §1 (Part 09) |
| difference at ε = 1.0 | +0.0267 ± 0.0071, 0/5 seeds better | `docs/CONTRACT.md` §1 (Part 09) |

Part 09 attacked all three systems white-box with k colluding views optimised
jointly to agree on a wrong class (`experiments/chorus/`, 5 seeds). **As
tested, the claim that the discount yields fusion robust to colluding views is
not supported.** The effect is small and unpaired bars overlap, but no cell at
any k, ε or β favours PrismFlow.

Detection and defence are separate capabilities and must be stated separately.
Only the first is evidenced. The correction fails because it is proportional
rather than capping, sublinear in α, and batch-level rather than per-sample
(`docs/KNOWN_LIMITATIONS.md` L4, and the L4 extension of 2026-09-18, which
refutes the shapes the original argument predicted).

### 2.2 The dependence signal is blind to uncoordinated compromise

| quantity | value | source |
|---|---|---|
| independent PGD attack success, ε = 2.0 | 0.4640 | `docs/CONTRACT.md` §1 (Part 09) |
| rho_bar movement over the same attack | −0.0290 ± 0.0251 | `docs/CONTRACT.md` §1 (Part 09) |

ENIV stayed flat. This is the boundary Part 08 measured directly: the
hypothesis is about *redundant* evidence, not *unreliable* evidence, because
noise makes a view less dependent rather than more
(`docs/KNOWN_LIMITATIONS.md` L3).

---

## 3. What was never measured

These are absences, not zeros. Nothing below has a committed number.

### 3.1 Calibration on real retrieved evidence — V2's central gap

| quantity | status | source |
|---|---|---|
| ECE on the 48-row V2 dataset | **not measured** | `docs/v2-known-limitations.md` V2-L6 |
| Brier score on the 48-row V2 dataset | **not measured** | `docs/v2-known-limitations.md` V2-L6 |
| conflict figures on the 48-row V2 dataset | **not measured** | `docs/v2-known-limitations.md` V2-L6 |

The harness **refused** to compute them rather than compute them on a handful
of rows, printing "Too few rows completed for metrics." That refusal is the
correct behaviour and is the honest summary of V2's state.

Why there is nothing to compute, from the committed 48-row run
(`results/v2/part24_pipeline_verification.json`, and the recovered first pass at
`results/v2/part24_pipeline_verification_pass1.json`):

| pass | rows with ≥1 angle | rows fused (≥2 angles producing claims) | rows scored |
|---|---|---|---|
| first, 2026-09-27 | 38/48 | 1/48 (row 032) | 1 |
| second, 2026-09-27 (quota-limited) | 20/48 | 0/48 | 0 |

Source: `docs/v2-known-limitations.md` V2-L6.

The single fused row is **not** a result. At n = 1, one pass, with a lexical
adjudicator standing in for Claude, row 032 demonstrates that the code path
executes end to end and nothing more. Its ENIV was **1.648** with confidence
**0.6731** in that pass (`docs/v2-known-limitations.md` V2-L6). The same row in
the later N=6 re-verification fused to **1.7634** with confidence **0.5580**
(`docs/v2-known-limitations.md`, status paragraph;
`results/v2/part24_reverify_n6.json`). **The two differ although the code is
identical**, because the reasoner sends no temperature and no seed and the
provider samples at its own default — which is itself a reason no single pass
can be a finding.

### 3.2 Per-angle retrieval, the upstream cause

**Corrected 2026-10-09.** This table previously reported the pre-fix run (tech
48/48 on 480 records, all arXiv, GitHub 0) as the current state. The V2-L2 and
V2-L3 connector fixes were re-run over all 48 rows on 2026-10-01, and the
postfix figures are these:

| angle | rows with ≥1 record | records | source connector |
|---|---|---|---|
| tech | 44/48 | 440 | GitHub 407 on 44 rows, arXiv 33 on 5 rows |
| market | 6/48 | 11 | NewsAPI |
| financial | 0/48 | 0 | yfinance |
| sentiment | 1/48 | 3 | StackExchange |
| regulatory | 0/48 | 0 | **no connector exists** |

Source: `results/v2/part24_pipeline_verification_postfix.json`, commit
`5ea0a17`, tabulated in `results/v2/coverage_funnel_old_vs_postfix.json`
(commit `305020c`); `docs/v2-known-limitations.md` V2-L1 carries the pre-fix
table alongside it.

GitHub went from 0 records to **407 on 44 of 48 rows**; arXiv fell from 480
records on 48 rows to **33 on 5**, because GitHub now holds the tech angle's
primary slot and the fallback chain stops there. That drop is the fallback
chain working, not a regression in the arXiv fix. The four thin angles did not
move in row coverage — market 6, financial 0, sentiment 1, regulatory 0 in both
runs. 440 is pre-budget: 387 records actually reached a reasoner (V2-L7).

Postfix, **3 of 48 rows retrieve nothing, 39 retrieve exactly one angle and 6
retrieve two**; by claims rather than retrieval it is 6, 40 and **2**. So on 46
of 48 rows there is still nothing for dependence, ENIV, the discount or fusion
to act on. Note the gap between the 6 rows that *retrieved* two angles and the
2 that *fused*: retrieval is not claims.

V2-L2 (GitHub returned 0 on every row) and V2-L3 (arXiv answered unrelated
queries with the same ten records) were both the same fault — a connector
handed a bag of words where it expects a query language — and are **resolved**.
At N=48 the fix raised fusible rows from 0 to 2 (rows 032 and 044, ENIV
**1.7052** and **1.657**). Fixing them advanced coverage only, and 2 of 48 is
not a distribution — see the calibration table above and V2-L6.

### 3.3 Other absences

| quantity | status | source |
|---|---|---|
| Claude as the fusion adjudicator (Part 23) | **not exercised** — the committed run used the offline `LexicalAdjudicator` | `docs/v2-known-limitations.md`, provenance table |
| gpt-4o as a reasoner | **not verified** — Groq verified the `OpenAIReasoner` *code path*, which says nothing about OpenAI's own models | `docs/part20-groq-verification.md`; `docs/v2-known-limitations.md` V2-L5 |
| Claude as a reasoner (`claude_reasoner.py`) | **not verified** | `docs/part20-groq-verification.md` |
| a 5-seed V2 evaluation | **not run** — the free Groq tier funds about two passes a day against the five the rule requires | `docs/v2-known-limitations.md` V2-L5 |
| regulatory-angle retrieval | **never built** — no connector exists in this checkout | `docs/v2-known-limitations.md` V2-L1 |
| financial-angle retrieval on this dataset | **structurally impossible** — the pitches are deliberately anonymised and yfinance needs a ticker | `docs/v2-known-limitations.md` V2-L4 |
| Reddit as the sentiment source | **unreachable from this network**, IP-scoped 403, measured 2026-09-24 | `prismflow/v2/angles/sentiment_angle.py` |
| attack success as a function of (k, rho) on the ENIV surface | **no such cell exists** in any committed file — the two experiments' `k` axes count different things | `web/README.md`, "The join the brief asked for does not exist" |
| human-subject evaluation | **not run** | `docs/KNOWN_LIMITATIONS.md` L15 |
| deployment evaluation | **not run** | `docs/KNOWN_LIMITATIONS.md` L15 |
| the site's achieved frame rate on the real GPU | **not measured** — the 60 Hz on record is a display refresh rate, not an achieved frame rate | `web/README.md` |
| hero-title contrast at 390px | **not measured** — at phone width the title does not overlap the prism, so there is no changed pixel to sample | `web/README.md` |

---

## 4. Standing limitations

`docs/KNOWN_LIMITATIONS.md` carries L1–L30 in full and
`docs/v2-known-limitations.md` carries V2-L1 through V2-L9. They are the
record; this is an index. The index below covers L1–L15 and V2-L1 to V2-L7;
L16–L30 (the real-data limitations of Parts 28-30) and V2-L8, V2-L9 are
indexed in `docs/FINAL_REPORT.md` sections 4, 7 and 8.

**V1, discovered as each Part ran:** L1 flat confidence under duplication
depends on training-time adaptation · L2 residual per-pair dependence
accumulates with view count · L3 the discount addresses redundant, not
unreliable, evidence · L4 the discount is proportional rather than capping · L5
the suspicion detector inherits the signal's blind spot · L6 the
shared/private objective does not fix which branch carries cross-view
information · L7 tail dependence inverts under strong attack · L8 detection and
correction fail together, because both read cross-view belief agreement.

**V1, standing limitations of the project as a whole:** L9 ENIV is validated
only on synthetic data · L10 the estimator is biased on finite batches · L11
tail dependence is unreliable below 50 exceedances · L12 the attacker
capabilities that defeat the defence are only partly tested · L13 V1 detaches
the discount, so it never shapes representation · L14 one dataset family and
five seeds that are five partitions of it · L15 no human-subject and no
deployment evaluation.

**V2:** V2-L1 one angle of five contributes · V2-L2 **resolved** · V2-L3
**resolved** · V2-L4 financial is structurally incompatible with the dataset ·
V2-L5 the free tier cannot fund a 5-seed evaluation · V2-L6 ECE and Brier
cannot yet be computed · V2-L7 some GitHub records exceed the whole token
budget and are dropped whole.

L14 is worth restating because it bounds everything in section 1: the five
seeds are five partitions of **one dataset family**, not five datasets.

---

## 5. What would change the picture, in order

From `docs/v2-known-limitations.md` V2-L6: **coverage first, then seeds, then
metrics**, in that order, because each is a precondition for the next.

1. **Coverage.** A connector whose queries match the dataset's language, for
   any of the four thin angles (V2-L1). Fixing the reasoner quota alone cannot
   create a second angle on a row where nothing was retrieved.
2. **Seeds.** Funded API capacity for 5 passes (V2-L5).
3. **Metrics.** Only then do ECE and Brier mean anything.

Separately, and independent of V2: the defence result (§2.1) would be revisited
by a **capping** rather than proportional discount, applied per-sample rather
than per-batch (`docs/KNOWN_LIMITATIONS.md` L4).

---

## 6. The site

`web/` is a static presentation layer over the committed files above. It
computes no metric, runs no experiment, and makes no network request. Every
figure on it carries a `source:` line naming the file it came from, and every
cell with no committed value renders as `not measured` rather than 0 or blank.
`web/build_data.py` copies the specific committed JSON the charts need into
`web/data/` and stamps each with a sha256. See `web/README.md`.
