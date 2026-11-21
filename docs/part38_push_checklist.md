# Part 38 — push checklist for Sunday

Branch `part38-lanl`, merge-base with `main` is **`805d1fc`**. Audited
2026-11-21 with the mode `default` sweep at **25 of 40** units.

**Nothing in this document has been executed.** The secret audit is a *plan*,
to be run Sunday. The sizes and file lists are read from git as it stands now
and will grow as the sweep lands.

---

## 1. Commits on the branch since `main`

**25 commits**, `805d1fc..HEAD`, oldest first.

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

**Commits still to come before the push:** the remaining sweep units and
`sweep_default.json`; `filled_tables.json` plus the Tables 1/2/3/6 verdict
writes; L27's proposal text; and — if mode A runs — `units_threads1/`,
`scores_threads1/`, `sweep_threads1.json` and H3.

## 2. Files changed

**83 files, 8,822 insertions, 0 deletions.** One modification, 82 additions.
No file is deleted by this branch.

| area | files | what |
|---|---|---|
| `experiments/lanl_v2/` | 15 | pre-registration, sampler, runner, metrics, the three scalar passes, `fill_analysis.py`, `run_mode_a.sh`, `sweep_block.sh`, README, analysis template, Sunday schedule |
| `results/lanl_v2/` | 64 | sample table, 10 unit JSONs, 10 score npz, 28 probe JSONs, 6 smoke score npz, 3 smoke JSONs, 2 reproduction JSONs, build report, sweep estimate, execution log, the three scalar-pass JSONs |
| `tests/unit/` | 3 | `test_lanl_v2_sampler.py`, `test_lanl_v2_sampler_synthetic.py`, `test_h1_scalar_passes.py` |
| `docs/` | 2 | `part38_section.md` (new), `proposed_limitations_update.md` (**modified — appended only**) |

**The single modified file is `docs/proposed_limitations_update.md`**, and it
was **appended to**, not rewritten: it already held 412 lines of Part
33/34/35/36 proposals, and the Part 38 section runs from a `---` to a
`<!-- PART38:END -->` marker, matching the existing `<!-- PART36:END -->`
convention. **Confirm at review that the Part 36 content is still intact.**

**`PRE_REGISTRATION.md` appears as `A` (added), not `M`.** It is added *by this
branch* at commit 1 and never modified afterwards — which is the frozen
discipline holding. `docs/KNOWN_LIMITATIONS.md` does not appear in the diff at
all.

## 3. Size

### 3.1 As it stands

| area | size |
|---|---|
| `scores_default/` (10 npz) | **4.36 MB** |
| other npz (`samples_v2_days16.npz` 2.94 MB + 6 smoke score npz) | **4.36 MB** |
| results JSON | 0.07 MB |
| `experiments/` | 0.21 MB |
| `docs/` | 0.07 MB |
| `tests/` | 0.03 MB |
| **total** | **9.11 MB** over 212 objects |

Largest blobs: `samples_v2_days16.npz` **3008.6 KB**, then the score npz at
**372.9–527.3 KB** each.

### 3.2 Projected

The 10 committed score npz average **0.436 MB**.

| scenario | projected branch size |
|---|---|
| mode B at 40/40, scores committed | **~22.2 MB** (+13.1 MB for 30 more npz) |
| the above plus mode A at 40/40 | **~39.6 MB** (+17.4 MB) |
| mode B at 40/40, **scores excluded** | **~9.1 MB** |

**This is the decision to make before the push, not during it.** The repo has
**no LFS** and the next-largest tracked file anywhere is 4.45 MB
(`results/ait_attack/results.A_naive_checkpoint.json`). The score npz are
already `savez_compressed`, so the gzip-plus-sha256-manifest remedy used for
Part 29's 25 MB **does not help here**.

Two facts that bear on the choice:

- **Precedent exists.** Part 30's `results/lanl/scores/*.npz` are tracked — 48
  files, 5.9 MB — so tracking score npz is the established pattern, not a new
  one. Part 38's are larger because its test block is 84,460 rows against Part
  30's 18,729.
- **They are not merely archival.** Table 3 (L27 / H2) is computed *from* these
  arrays by re-scoring with a host mask. Excluding them makes the
  leave-host-out analysis unreproducible without a full retrain.

### 3.3 `.gitignore` behaviour, already checked

`.gitignore:43` `results/*/*/*` ignores that depth and `:44` re-includes only
`*.json` (plus `.csv/.md/.png`). So:

| path | status |
|---|---|
| `results/lanl_v2/units_default/*.json` | **tracked normally** |
| `results/lanl_v2/scores_default/*.npz` | **ignored — needs `git add -f`** (as commit 12 did) |
| `results/lanl_v2/units_threads1/*.json` | **will track normally** |
| `results/lanl_v2/scores_threads1/*.npz` | **will be ignored — will need `-f`** |
| `results/lanl_v2/logs/*.log` | **ignored** — correct, they are operational |
| `results/lanl_v2/*.json` (top level) | **tracked** via `:31 !results/*/*.json` |

## 4. Secret and settings audit

### 4.1 Static path check — **done, clean**

`git diff --name-only 805d1fc..HEAD` filtered for
`settings|\.env|secret|credential|token|key|\.pem|id_rsa|\.claude/` returns
**no matches**. **No `.claude/settings*.json` and no `.env` is on this
branch.**

### 4.2 Content scan — **PLAN, to run Sunday**

Each commit in this block already ran `tools/check_secrets.py` against its
staged changes and each reported clean. That is **not** sufficient for a push:
staged-only scans never saw the 60 result files added by earlier commits.

Run, in this order:

```sh
PYTHONPATH=. ./.venv/Scripts/python.exe tools/check_secrets.py --all
```

`--all` reads every tracked file, and the check that matters is its second
one: it takes the real values out of `.env` and looks for those exact strings,
which catches a key with no recognisable prefix. It never prints the values.

Then the **push scan**, over the blobs this branch actually adds.

> **The push scan needs real repository paths.** A flat `git log -p` dump trips
> `check_secrets`' own allow-list, so the scan comes back clean when it is not.
> Mirror each added blob at its repo path — e.g. export
> `805d1fc..HEAD` into a temp tree with `git archive HEAD | tar -x -C <tmp>`,
> or check out the branch into a scratch worktree — and scan that tree.

**Expect two classes of benign hit and confirm each by eye rather than
waving it through:**

- **`scores_sha256` fields** in the 40 unit JSONs — 64-hex strings, which is
  the shape of several credential formats. They are file digests and are
  verifiable: each equals the sha256 of its `scores_default/<name>.npz`, and
  `fill_analysis.py` asserts exactly that.
- **the `samples_v2_days16.npz` sha256** in `SWEEP_EXECUTION_LOG.md` section 5,
  `03b7602a590f9c64196ceee2750981187f456541d81927cb69d7b7d6aa251a9a`.

**Any other hit stops the push.** Report it; do not rewrite history to hide it.

### 4.3 Data that must not be pushed

LANL `cyber1` raw data is **not** on this branch and must not be. Confirm:
`data/raw/` holds `auth.txt.gz` (7.6 GB), `proc.txt.gz` (2.4 GB),
`flows.txt.gz` (1.1 GB), `dns.txt.gz` (185 MB), `redteam.txt.gz` and
`data/raw/ait/` (2.2 GB per scenario). None appears in the diff. The derived
`samples_v2_days16.npz` **is** on the branch at 2.94 MB, which is the intended
artefact.

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

- [ ] `git diff main --stat -- experiments/lanl_v2/PRE_REGISTRATION.md docs/KNOWN_LIMITATIONS.md`
      is **empty**.
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
