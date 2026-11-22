# Part 38 — push checklist

Branch `part38-lanl`, merge-base with `main` is **`805d1fc`**. Audited
**2026-11-22** with the mode `default` (B) sweep **complete at 40 of 40
units**, §4.4 integrated into the final report, and the L24/L27/L28 status
updates applied.

**The audit in §4 has been EXECUTED**, unlike the 2026-11-21 revision of this
file, which was a plan. Results are recorded per category below.

---

## 0. Readiness

### Ready to push

- **Part 38 is closed.** Mode B sweep 40/40, all 40 `scores_sha256` verified
  against the on-disk npz, `filled_tables.json` reproduces bit-exact across
  two runs.
- **L24 and L27 verdicts are in place.** H1 **DISCONFIRMED** and H2
  **DISCONFIRMED** (the latter in the *opposite* direction to its
  prediction), both recorded in `docs/KNOWN_LIMITATIONS.md` with dated Part 38
  blocks. Both entries stay **OPEN**; nothing was marked RESOLVED or CLOSED.
- **The final report is integrated.** §4.4 inserted,
  `check_report_numbers.py` green at **759** numbers (327 before), every one
  resolving to a cited file.
- **A null result for the central thesis is recorded in the report body**, not
  only inside §4.4: on this LANL sample no arm favours the dependence
  discount (`docs/FINAL_REPORT.md` §6 and §8).

### Deferred, with runners in place and not executed

| item | blocks | state |
|---|---|---|
| Mode A (`threads1_concurrent11`) | **L28** verdict, and H3 | `experiments/lanl_v2/run_mode_a.sh` exists and is validated. ~2.0 h, ~5.4 GB across 11 processes. **H3 is unsatisfiable on this sample regardless** — three of four arms have no established sign for mode A to agree with |
| Part 37 item B | **L12** | scaffolded, not executed |
| Part 39 item A | **L16** | scaffolded, not executed |

### What the user does next

1. **Review `docs/FINAL_REPORT.md` §4.4 visually** — it is 616 inserted lines
   and the heading depth goes to `#####`. Check it reads as part of §4.
2. **Decide on the `web-backdrop` merge** (§7 below). This needs a visual
   check; it is not a correctness question.
3. **Push `part38-lanl`** (§6).
4. **Merge and push `web-backdrop`** only if step 2 approved it.

---

## 1. Commits on the branch since `main`

**48 commits**, `805d1fc..HEAD`, oldest first.

| # | SHA | subject |
|---|---|---|
| 1 | `406df50` | Part 38 step 1: pre-registration, before any sample or result |
| 2 | `d2dc1eb` | Part 38 step 1: the lanl_v2 sampler and runner, with its reproduction gate |
| 3 | `ff92183` | Part 38 step 1: amendment 1 — H2's primary metric is threshold-free ROC AUC |
| 4 | `13998e7` | Part 38 step 1: implement section 2's size fallback, and measure peak memory |
| 5 | `8ff1ed3` | Part 38 step 1: fix peak_rss_mb's handle, and add the step-rate probe |
| 6 | `c1683b9` | Part 38 step 1: re-run the 6-day smoke and the day-1 reproduction gate |
| 7 | `87e8ef9` | Part 38 step 1: the 16-day build completes, and a measured sweep estimate |
| 8 | `136ebcd` | Part 38 step 2: audit the sampler on synthetic capture files |
| 9 | `8f76c2c` | Part 38 step 2: the days 1-7 and days 1-16 probe cells |
| 10 | `6e500a1` | Part 38 step 3: the dated sweep execution log, before mode B is launched |
| 11 | `51f7ac5` | Part 38 step 3: the sweep subcommand, resumable, one execution mode at a time |
| 12 | `76e7726` | Part 38: preserve in-flight sweep state (10/40 units) |
| 13 | `72d3be9` | Part 38: analysis template, no numbers filled |
| 14 | `2ba2c41` | Part 38: H1 measured, and DISCONFIRMED |
| 15 | `0624966` | Part 38: occupancy decomposition after H1 disconfirmation |
| 16 | `7839181` | Part 38: conditional rates -- Part 30's 410/410 statistic on v2 |
| 17 | `02d690b` | Part 38: template records the two scalar passes and narrows L24 |
| 18 | `a8d0026` | Part 38: fill_analysis.py skeleton (do not run until 40/40) |
| 19 | `391c33a` | Part 38: unit tests for the three scalar passes |
| 20 | `c171171` | Part 38: fill_analysis.py dry-run flag for schema verification |
| 21 | `0d79110` | Part 38: proposed limitations update (review gate pending) |
| 22 | `b1d8fde` | Part 38: Sunday schedule draft |
| 23 | `1fa3bca` | Part 38: README documenting the three scalar passes |
| 24 | `693cfde` | Part 38: narrative section for final report |
| 25 | `62cab9c` | Part 38: Mode A launch script (do not run until mode B at 40/40) |
| 26 | `16a9107` | Part 38: push checklist for Sunday |
| 27 | `c0d7369` | Part 39 item A: scaffolding and pre-registration (not run) |
| 28 | `385d26f` | Part 37 item B: pre-registration and README (not run) |
| 29 | `040bafb` | Part 38: final report integration preview |
| 30 | `8453021` | Part 38: Sunday decision analysis |
| 31 | `e4efc7a` | Part 39 item A: runner (not executed, not tested) |
| 32 | `89bf16c` | Part 38: preserve 26 recovered units from shutdown (26/30 new, 10 already committed = 36/40) |
| 33 | `1a130b0` | Part 38: verdict worksheet (L24 ready; L27/L28 pending sweep) |
| 34 | `b85fb0e` | Part 38: final-report prose ready for integration |
| 35 | `9d3565a` | Part 38: peak_rss_mb fix patch (apply after sweep) |
| 36 | `239f349` | Part 38: peak_rss_mb is a running maximum, not one value per launch |
| 37 | `ac98a1f` | Part 38: peak_rss_mb preserved per-seed with regime provenance |
| 38 | `11cf3af` | Part 38: mode B sweep complete at 40/40 |
| 39 | `a0f58ab` | Part 38: sweep_default.json NaN -> null |
| 40 | `9213d0d` | Part 38: analysis template filled, L24/L27 verdicts recorded |
| 41 | `c4f5993` | Part 38: worksheet updated to the 40/40 verdicts |
| 42 | `b126c80` | Part 38: worksheet L24 arm gap filled, stale "pending the sweep" removed |
| 43 | `b829e24` | Part 38: worksheet 'does not decide' list no longer lists filled tables |
| 44 | `ab37aeb` | Part 38: section reconciled with the 40/40 verdicts |
| 45 | `7b3cd07` | Part 38: integration instruction updated for the completed sweep |
| 46 | `3435993` | Part 38: pairwise column relabelled to the quantity it holds |
| 47 | `22d284c` | Part 38: integrated into final report (§4.4) |
| 48 | `e57bd73` | Part 38: L24, L27 status updates; L28 deferred to future Part |

**PRE_REGISTRATION.md is touched by two commits, both legitimate.**
`406df50` **adds** the file (status `A`) and `ff92183` is the dated
**amendment 1** (status `M`). No later commit edits it — verified with
`git show --name-status`.

**No frozen module is touched.** `git diff --name-only main..HEAD -- prismflow/`
is empty.

---

## 2. Files changed

`git diff --stat main..HEAD`: **161 files changed, 18668 insertions(+),
6 deletions(-)**.

| directory | files |
|---|---|
| `results/lanl_v2` (top level) | 47 |
| `results/lanl_v2/units_default` | 40 |
| `results/lanl_v2/scores_default` | 40 |
| `experiments/lanl_v2` | 14 |
| `docs` | 9 |
| `experiments/part39_ait` | 4 |
| `tests/unit` | 3 |
| `results/lanl_v2/logs` | 2 |

---

## 3. Size

| quantity | value |
|---|---|
| tracked tree at HEAD | **82.1 MB** in **1285** files |
| `.git` directory | **77.2 MB** |
| largest tracked file | **4.4 MB**, `results/ait_attack/results.A_naive_checkpoint.json` |
| largest file added by this branch | **2.9 MB**, `results/lanl_v2/samples_v2_days16.npz` |
| the 40 mode-B score arrays | **18.3 MB** total, ~495 KB each |

**No single file exceeds 10 MB**, so `results/`'s no-LFS convention holds
without gzip. The 40 `scores_default/*.npz` are tracked individually rather
than as one archive, which is why the directory total passes while no member
does.

---

## 4. Secret and settings audit — EXECUTED 2026-11-22

| category | command | result |
|---|---|---|
| repo checker, real paths | `tools/check_secrets.py --all` | **PASS** — 975 files, 4 values loaded from `.env`, `clean`, exit 0 |
| `api_key` / `secret` / `token` / `credential` / `-----BEGIN` | `grep -IEn` over every one of the 161 files in `git diff --name-only main..HEAD` | **PASS** — 11 hits, **all prose about secret scanning** (this file, `part38_decisions.md`, `SUNDAY_SCHEDULE.md`, and one "token budget" in `FINAL_REPORT.md`). No value. |
| `.claude/settings*.json` | grep over the diff | **PASS** — not present |
| `.env`, `.pem`, `id_rsa`, `*.key`, `credentials` | grep over the diff | **PASS** — not present |
| 64-char hex strings | field-name resolution, not shape | **PASS, expected** — every one sits under `scores_sha256` (40 occurrences, one per unit JSON), `samples_sha256` (1) or `sha256` (1). These are **file digests**, which share a shape with several credential formats. Checked by resolving each to its JSON key rather than by eyeballing the string. |
| untracked-by-design dirs | grep over the diff | **PASS** — neither `Project backend details/` nor `docs/report/` appears |

**The scan ran against real repo paths**, not a flat `git log -p` dump: a flat
dump trips `check_secrets`' own allow-list and comes back clean when it is
not.

---

## 5. Working-tree state at audit time

**Not part of the push unless committed.**

**Four tracked files are deleted in the working tree and the deletions are
unstaged.** They predate this session and this branch:

```
 D PrismFlow_Master_Project_and_Claude_Prompts.docx
 D PrismFlow_Master_Project_and_Claude_Prompts.pdf
 D PrismFlow_Master_Project_and_Claude_Prompts.txt
 D PrismFlow_Technical_Overview.pdf
```

**Decide deliberately before the push.** They are not Part 38's business and
the default should be to leave them alone — but a `git add -A` or a `git
commit -a` would sweep all four deletions into a Part 38 commit, which would
be wrong. **Stage by explicit path on Sunday.**

**16 untracked result files** are waiting: 15 sweep unit JSONs and
`sweep_default.json`. They are the sweep's own output and get committed with
the rest at 40/40.

**Two untracked directories are not Part 38's and should stay untracked:**
`Project backend details/` and `docs/report/`.

## 6. Pre-push checklist

Run in order. Any failure stops the push.

**Content**

- [ ] `results/lanl_v2/units_default/` has **40** JSONs and
      `scores_default/` has **40** npz, as matched pairs.
- [ ] `fill_analysis.py` ran **without** `--dry-run`;
      `results/lanl_v2/filled_tables.json` exists; **no
      `filled_tables_dryrun.json` is tracked**.
- [ ] Every Table 1 cell has `n = 5`. Any cell below that is labelled
      `NOT A FINDING` and carries no verdict.
- [ ] `docs/part38_section.md` has no `[pending]` left in a section presented
      as a result, and no figure from the uncommitted dry run.
- [ ] L27's verdict carries both caveats: the realised **1/4** share, and
      Part 30's smallest committed cell (**−0.035962653276654266**) sitting
      *below* the 0.036 bar.
- [ ] L28 is **not** called closed anywhere.

**Repository**

- [x] `git diff main --stat -- experiments/lanl_v2/PRE_REGISTRATION.md`
      shows **only** `ff92183`'s amendment 1 (`406df50` adds the file).
      **This item changed on 2026-11-22:** it previously also required
      `docs/KNOWN_LIMITATIONS.md` to be untouched. The review gate authorised
      the L24/L27/L28 updates, so that file IS modified, by design, in
      `e57bd73`. All three entries stay **OPEN** and no heading gained
      RESOLVED or CLOSED — which is the property to check now.
- [ ] `docs/proposed_limitations_update.md` still contains
      `<!-- PART36:END -->` and all four earlier proposal sections.
- [ ] No frozen module is modified: `git diff main --name-only` contains
      nothing under `prismflow/`.
- [ ] The score-npz size decision (§3.2) is made explicitly.
- [ ] The four working-tree deletions (§5) are handled deliberately; nothing
      was staged with `-A` or committed with `-a`.

**Gates**

- [ ] `pytest -q` green, and the output was read. **Background it and start by
      15:30:** 1,285 tests, ~7 min warm and ~26 min cold, which exceeds the
      default command timeout.
- [ ] `tools/check_secrets.py --all` clean.
- [ ] Push scan run over a tree with real repo paths (§4.2), and each hex hit
      identified as a `scores_sha256` or the sample-table digest.
- [ ] Branch is `part38-lanl`; the merge target is agreed; **nothing was
      pushed before the review.**

---

## 7. The push, and the `web-backdrop` decision

**Nothing below has been run.** Claude Code did not push, merge or tag.

### 7.1 Push `part38-lanl`

The branch has **no upstream**, so the first push sets it:

```sh
git -C "E:/PERSONAL/project/PrIsM FlOw" push -u origin part38-lanl
```

Pre-flight, if you want the audit re-run from scratch rather than trusted:

```sh
git -C "E:/PERSONAL/project/PrIsM FlOw" status --short
PYTHONPATH=. ./.venv/Scripts/python.exe tools/check_secrets.py --all
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe docs/check_report_numbers.py
```

Expected: a clean tree except the two untracked directories
(`Project backend details/`, `docs/report/`), `check_secrets` `clean`, and
`check_report_numbers` reporting **759** numbers all resolving.

### 7.2 `web-backdrop` — only after a visual check

**This is a judgement, not a gate, and it is not Part 38's.** The backdrop
changes what the site looks like; nothing in this block measured it and no
test covers it. Look at it in a browser first.

```sh
git -C "E:/PERSONAL/project/PrIsM FlOw" checkout main
git -C "E:/PERSONAL/project/PrIsM FlOw" merge --no-ff web-backdrop
git -C "E:/PERSONAL/project/PrIsM FlOw" push origin main
```

`--no-ff` keeps the merge visible as one commit on `main`.

**Do not fold this into the `part38-lanl` push.** They are unrelated changes
and a visual regression should not be bisectable only through a results
branch. Merging `web-backdrop` into `main` while `part38-lanl` is still
unmerged is fine; they touch disjoint paths.
