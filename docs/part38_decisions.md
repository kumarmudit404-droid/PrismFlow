# Part 38 — two decisions for Sunday

Written 2026-11-21 with the mode `default` sweep at 26 of 40. Both decisions
are grounded in committed files; neither is executed here.

---

## Decision A — is `apply_mode`'s `concurrency = 11` a problem for L28's design?

### A.1 The premise has to be corrected first

**The brief asks about "`apply_mode`'s `concurrency=11` bug in mode B". There
is no such bug: mode B does not record 11.** `run_lanl_v2.py:132-147`:

```
if name == "threads1":
    torch.set_num_threads(1)
    concurrency = 11
elif name == "default":
    torch.set_num_threads(default)
    concurrency = 1
```

Mode B is `--mode default`, so it takes the second branch and records
`concurrency: 1`. Verified on disk, not inferred: **all 26 committed unit
JSONs contain `"concurrency": 1`**, e.g.
`results/lanl_v2/units_default/without_dns_prismflow_seed0.json`:

```
"mode": {"mode": "default", "torch_threads_observed": 8,
         "concurrency": 1, "machine_default_threads_measured": 8}
```

**So mode B records what it does and does what it was pre-registered to do.**
`PRE_REGISTRATION.md` §6 defines mode B as "machine-default thread count,
units run sequentially" — 8 threads observed, sequential, concurrency 1. All
three recorded fields are true of the run. **Nothing about mode B needs
changing and nothing about it threatens L28.**

### A.2 The real residue, which is about mode A

The `concurrency = 11` in the `threads1` branch is a **declaration, not an
implementation**. `sweep` trains its units sequentially in one process; it
spawns nothing. So if anyone runs

```sh
run_lanl_v2.py sweep --mode threads1 ...
```

directly, every resulting unit JSON would claim `"concurrency": 11` while the
run was in fact sequential at 1 thread. **That would be a false record** — and
it would be a third execution mode (`threads1_sequential`) that the
pre-registration never defined, recorded under mode A's name.

This has not happened: `results/lanl_v2/units_threads1/` does not exist.

### A.3 Impact on L28

**None, as things stand, and the design is intact in both dimensions.**
§6 pre-registers mode A and mode B as differing in **two** things together —
thread count *and* concurrency:

| | threads | concurrency |
|---|---|---|
| mode A | 1 | 11 processes |
| mode B | machine default (**measured at 8**) | sequential |

Mode B delivered both of its dimensions. Mode A will deliver both **provided
`experiments/lanl_v2/run_mode_a.sh` is used**, since that script supplies the
11 workers. Its header says so in its first paragraph, and its closing note
repeats it.

Note also what L28 is actually about: "**The thread count selects between
deterministic answers, it does not add noise**", re-measured in Part 34 as a
committed `eniv_eigen` of **3.4396320614332856** at the machine default
against **3.4289614712229826** at `OMP_NUM_THREADS=1`, a miss of **0.0107**.
The dominant factor L28 names is the **thread count**, which both modes set
explicitly and record as observed. Concurrency matters because it is what Part
30 run 2 actually did and §6 says mode A "reproduces the mode recorded in
`results/lanl/execution_mode.json` for Part 30 run 2" — so it is part of the
fidelity of the reproduction, not the mechanism under test.

### A.4 Options

| | option | cost | verdict |
|---|---|---|---|
| **1** | Accept mode B as-is and launch mode A only via `run_mode_a.sh`. | none | **recommended** |
| **2** | Re-run mode B "with true concurrency". | ~11 h of compute, and it would produce a mode **nobody pre-registered**: §6 defines mode B as *sequential*. | reject |
| **3** | Narrow L28's verdict to the dimensions actually varied. | none, but unnecessary — both dimensions *are* varied once mode A runs via the launcher. | reject as stated; see the amendment below |
| **4** | Add a guard to `run_lanl_v2.py` so `--mode threads1` refuses to run without the launcher. | edits the runner mid-sweep | reject for now, see A.6 |

### A.5 Recommendation — option 1, with one verification added

**Accept mode B. Launch mode A only through `run_mode_a.sh`. Add one check at
analysis time.**

The check: when Table 5 is written, **assert that mode A's recorded
`concurrency: 11` matches the number of workers that actually ran**, by
counting `results/lanl_v2/logs/mode_a_worker*.log` files and confirming it is
11. One line, and it converts a declaration into a verified record. Without it
the claim "11 concurrent processes" rests on nobody having run the bare
command — true today, unverifiable later.

Option 3 is rejected **as stated** but contains a real point that belongs in
the write-up regardless: L28's verdict should say which dimensions were varied
(thread count 8 → 1, concurrency 1 → 11, together) and should **not** claim to
separate their effects. Two factors moved at once, by design, because §6's
purpose was to reproduce Part 30's two runs rather than to decompose them. A
reader must not be left thinking the thread count alone was isolated.

### A.6 Why not patch the runner

`run_lanl_v2.py` is not frozen and the guard would be four lines. It is still
the wrong move today:

- The file has already produced **26 committed units**. Editing it mid-sweep
  makes the committed runner differ from the one that produced them, which
  costs more in reproducibility than the guard buys in safety. The live
  process holds its own copy, so an edit would not affect the running sweep —
  but the provenance claim would be muddied.
- The exposure is one command nobody has run, and `run_mode_a.sh`'s pre-flight
  already blocks the realistic path into it.

**Revisit after the push.** A guard belongs in the runner eventually; it does
not belong in the window before a release.

---

## Decision B — keep or drop the score `.npz` files from the push?

### B.1 The numbers

Measured from the branch (`git cat-file -s` over the diff against `805d1fc`):

| | size |
|---|---|
| branch now (10 score npz committed) | **9.11 MB** over 212 objects |
| `scores_default/` alone, 10 files | **4.36 MB**, averaging **0.436 MB** |
| largest single blob, `samples_v2_days16.npz` | **3008.6 KB** |
| `.git` on disk now | 71 MB |

Projections at that average:

| scenario | branch size |
|---|---|
| mode B at 40/40, scores kept | **~22.2 MB** |
| mode B + mode A, both kept | **~39.6 MB** |
| mode B at 40/40, scores dropped | **~9.1 MB** |

### B.2 What each set is actually needed for

This is the question that decides it, and the two sets differ:

**Mode B's 40 score npz are load-bearing.** Table 3 — H2, L27's whole
hypothesis — is computed **from** these arrays: `fill_analysis.py` re-scores
`roc_auc_binary(label[~mask], prob_attack[~mask])` with a host mask applied,
using the per-row `computer` column the npz carries. Dropping them makes H2
**unreproducible without re-running the sweep**, which is ~11 h of compute. It
would also strand the five leave-host-out hosts' deltas as numbers nobody
downstream can re-derive.

**Mode A's 40 score npz are not needed by anything pre-registered.** H3
compares the **sign** of the PrismFlow-minus-naive ROC AUC difference across
modes, and every ROC AUC it needs is a scalar field in the mode A unit JSONs.
Table 5 is likewise built from unit fields. A mode A leave-host-out analysis
would need them — and is not pre-registered.

### B.3 The constraint is a local convention, not a platform limit

Worth separating, because it changes the weight of the "10 MB" figure:

- The **10 MB** threshold is this repository's own convention. It is not a
  platform limit.
- GitHub blocks individual files above **100 MB** and warns above **50 MB**.
  The largest blob here is **3.0 MB** and the largest score npz is **0.52 MB**,
  so nothing is near either.
- A 22 MB branch addition is unremarkable for a git host. A 40 MB one is
  larger than ideal but not a problem.

So "22.2 MB breaks the convention" is true and "22.2 MB is a problem for the
push" is not. The convention exists to stop results directories becoming
dumping grounds, which is a good reason to think before adding 17 MB — and a
poor reason to delete the inputs to a hypothesis.

### B.4 Precedent and the remedies that do not apply

**Precedent is clear.** Part 30's `results/lanl/scores/*.npz` are tracked — 48
files, **5.9 MB**. Tracking LANL score npz is the established pattern. Part
38's are larger per file only because its test block is 84,460 rows against
Part 30's 18,729.

**Part 29's remedy does not transfer.** Part 29's 25 MB was resolved with
`gzip -9` plus a decompressed-sha256 manifest. These files are already written
by `np.savez_compressed`, so a second compression pass gains nothing. **No LFS
is configured** and configuring it the day before a push is not a change to
make under time pressure.

### B.5 Recommendation — keep mode B's, drop mode A's

**Keep all 40 of mode B's `scores_default/*.npz`** (`git add -f`, as commit
`76e7726` already did for the first 10). They are the inputs to H2 and the
only thing standing between a reader and an 11-hour retrain. **~22.2 MB.**

**Do not commit mode A's `scores_threads1/*.npz`.** Nothing pre-registered
reads them, and `.gitignore:43` already excludes them by default — so this
costs no action, only a decision not to force-add. **Saves ~17.4 MB.**

**One wart this creates, and how to handle it.** Mode A's unit JSONs will each
carry a `scores_sha256` field naming a file that is not in the repository. That
is a dangling reference and it should be stated rather than left to be
discovered: add one line to `experiments/lanl_v2/README.md` saying mode A's
score arrays are **not tracked**, that the recorded hash is a provenance record
of the local file, and that re-deriving them means re-running mode A
(~2.0 h — far cheaper than mode B's 11 h, which is itself part of why dropping
them is tolerable).

**If symmetry is preferred over 17 MB, keeping both sets is defensible** —
nothing breaks at 39.6 MB. But asymmetry here is *principled*: one set backs a
hypothesis and the other backs nothing, and that is a better rule than
"keep everything" or "keep nothing".

### B.6 What must happen either way

- **Decide before the push, not during it.** A `git add -f` under time
  pressure is how 17 MB arrives unnoticed.
- **Record the decision in the commit message** that adds the final 30, so the
  size is a choice on the record rather than an accident.
- **The push scan will flag the 40 `scores_sha256` fields** as 64-hex strings,
  which is also the shape of several credential formats. They are file digests
  and `fill_analysis.py` asserts each equals the sha256 of its npz. Confirm
  them by eye; do not wave them through.
