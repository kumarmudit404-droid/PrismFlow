# Part 38 — schedule to the Sunday 17:30 push

Draft written Saturday 2026-11-21 16:45 IST, with the mode `default` (B) sweep
at 25 of 40 units. Every time estimate names its basis; estimates built on a
cell with fewer than 5 measured units say so.

**This is a planning document, not evidence.** Nothing here is a finding and
nothing here may be cited as a measurement.

---

## 1. Expected completion tonight

### 1.1 The sweep

| | |
|---|---|
| State at 16:42 | **25 / 40** units, process 971 alive |
| Remaining | 15 units, **16,298 s = 4.53 h** |
| **ETA** | **~21:15 IST** |

From measured per-cell means in `results/lanl_v2/units_default/*.json`
(`unit_seconds`), 15 units at their own cell's mean:

| cell | left | measured mean | n measured | cost |
|---|---|---|---|---|
| `without_dns`/prismflow | 3 | 1769.7 s | 2 | 5309 s |
| `with_auth`/prismflow | 3 | 1486.0 s | 2 | 4458 s |
| `without_auth`/prismflow | 3 | 950.0 s | 2 | 2850 s |
| `with_auth`/naive | 3 | 679.6 s | 2 | 2039 s |
| `without_auth`/naive | 3 | 547.5 s | 2 | 1642 s |

**Every one of those five means rests on n = 2.** The three cells already at
n = 5 came in at 546.0 s (`without_dns`/naive, std 1.3), 714.0 s
(`without_flows`/naive, std 170.3) and 884.5 s (`without_flows`/prismflow,
std 3.0) — so a cell mean from two units can be off by minutes, and the
`without_flows`/naive spread shows machine state alone moves a unit by
~370 s.

**The ETA has slipped twice today, both times for understood reasons.**
`without_dns`/prismflow seed 1 came in well above seed 0, lifting that cell's
mean from 1397.5 s to 1769.7 s — and this afternoon's test runs, the three
thread-invariance subprocess passes and the `--dry-run` all competed with the
sweep for the same 8 threads. **Expect ~21:15 if the box is left alone from
here, later if it is not.** The sweep is the only thing that should be running
tonight.

### 1.2 After the sweep lands

| step | command | estimate | basis |
|---|---|---|---|
| Fill Tables 1, 2, 3, 6 | `fill_analysis.py` | **2–4 min** | the `--dry-run` over 24 units completed well inside a minute of CPU; the real run adds 16 units and 5 hosts each |
| Verdict writes into the template | manual | **20–30 min** | §3.2's L27 verdict plus §4's clauses; the fill rules are already fixed in `docs/proposed_limitations_update.md` so this is transcription, not analysis |
| L27 proposal text | manual | **15 min** | placeholder and both caveats already written |
| Commit | — | **5 min** | |

So **~22:00–22:15 for a complete mode-B analysis**, if the sweep lands at
21:15 and nothing fails its gate.

**Three things could stop it, all of them loud by design.** A `scores_sha256`
mismatch, a re-scored all-test AUC disagreeing with the committed `roc_auc`
beyond 1e-9, or a unit whose observed thread count differs. Each raises and
names the unit. If one fires, the right move is to report it, not to work
around it.

---

## 2. Sunday options, ranked

Only one of these fits comfortably before the report and push window. The
ranking is by *what the project gains per hour*, not by what is most
interesting.

### Option A — Mode A, for L28 — **recommended**

| | |
|---|---|
| Compute | **~2.0 h**, from `results/lanl_v2/sweep_estimate.json` |
| Resources | **~5,372 MB across 11 processes** — needs an idle box |
| New code | **none.** `--mode` already accepts it; `apply_mode` sets 1 thread and the runner records the observed count |
| Unblocks | Table 5, and the only barred L28 analysis this design can produce (H3 sign agreement) |

**Why first.** It is the only item on this list that turns an OPEN limitation
into a measured verdict, it needs no new code, and it is the one piece of
Part 38's own pre-registration still unexecuted — §6 specifies *two* modes and
one has been run. Leaving it undone means Part 38 does not do what it said it
would do.

**The honest limit, which must be stated whatever it shows:** L28's literal
closing condition asks for a std covering mode *as well as* seed, and §6
forbids pooling the modes. So mode A narrows L28; it cannot close it. See
`docs/proposed_limitations_update.md`, the L28 placeholder.

**Risk:** 5.4 GB across 11 processes on a 15.7 GB box with a browser open is
tight. Run it with nothing else live, and expect the wall-clock to exceed
2.0 h if it swaps.

### Option B — Part 39 item A, for L16

| | |
|---|---|
| Compute | **~15–20 min** — `experiments/ait_clean/README.md:466` records the full 5-seed AIT run at **~14 min CPU** |
| New code | **~370 lines** (driver ~180, config ~60, test ~50, README ~80) |
| Unblocks | L16's written closing condition: "a threshold chosen on the training scenario and reported as such" |

**Why second, not first.** Cheapest compute on the list and it delivers a
closing condition verbatim — but it needs a retrain, because **no AIT model
checkpoint and no per-row AIT probability is committed anywhere**
(`results/ait_clean/<direction>/<system>/` holds only `metrics.csv`,
`metrics.json`, `reliability_diagram.png`; there is no `.npz` or `.npy` under
`results/ait_clean/` or `results/ait_attack/`, and no `.pt`/`.ckpt` in the
repo). It also means writing a new experiment directory under Part 39's rules,
which includes pre-registering `experiments/part39_ait/config.yaml` **before**
running anything. That is the real cost: the 20 minutes of compute sits behind
an hour of correct scaffolding.

### Option C — Part 37 item B, for L12

| | |
|---|---|
| Compute | **~1.5–2.0 h**, extrapolated, not recorded |
| New attack code | **none** — `prismflow/attacks/adaptive.py` already implements `gamma · max(0, rho_hat(C) − tau)` with BPDA |
| New driver / config / test / README | **~490 lines** |
| Unblocks | L12's "a working gate-aware arm" plus the k sweep |

**Why third.** The compute figure is an extrapolation: `results/adaptive/
adaptive.json` records no wall-clock, so the estimate comes from the data size
(2,000 samples, batch 64, 30 steps) and Part 29's measured 6.0 s per PrismFlow
attack at batch 1200 against 14.3 s at batch 64. Two design decisions must be
made and pre-registered *before* anything runs:

1. **The epsilon grid.** L12's null exists because at eps 2.0 random
   perturbation already drives dependence to 0.0752, below every seed's tau,
   so the hinge is inactive. Choosing eps *because* it makes the penalty
   active is result-driven selection unless declared as a calibration pass.
2. **The k grid is {2, 3}, not {1, 2, 3}.** `compromised_agreement` returns
   zeros when `|C| < 2`, so at k = 1 the evasion term is identically zero and
   the arm reduces exactly to plain chorus — the same declared identity Part
   29 recorded for beta at k = 1.

### What not to do Sunday

**Part 37 item C** (detector-optimising attacker, ~120 lines of new attack
code plus a driver) and **Part 39 item C** (contaminated-fraction axis, needs
the AIT retrain plus attack generation, ~1.0–1.5 h). B + C of Part 37 together
is ~970 lines and 3–4 h of compute and does not fit a block alongside the
report. **Part 39 item E is blocked by data**: only `russellmitchell` and
`santos` are downloaded and both labels are solved by `dns` alone.

---

## 3. The video clips — **NOT SPECIFIED, needs your input**

**I cannot estimate this and will not invent a number.** There is no video or
clip work anywhere in the repository. The only video reference in the tree is
`web/README.md:162`, "**The hero: video removed, text centred, contrast
re-measured (2026-10-01)**" — a looping prism video that was *removed* from the
hero for a measured reason, not work pending.

So "the videos work (3 clips)" is something only you hold. To schedule it I
need:

- **What the three clips show** — the web demo, the live `/score` service, the
  diagnostics, something else?
- **What produces them** — screen capture, a script, an existing tool?
- **Where they go** — `web/`, the report, outside the repo entirely?
- **Whether they must be in the repo before the push.** This one decides
  whether they are on the critical path at all. If they live outside the repo,
  they do not block the 17:30 push and should be done *after* it.

**Provisional placement, to be corrected:** if the clips are screen captures
of the existing site and service, they need the box free (capture is
wall-clock, not CPU-bound) and do not conflict with a compute block. Slotting
them **Sunday 14:00–15:30** keeps the morning for Option A and leaves the
report window intact. **If they need rendering or editing, that estimate is
wrong and the schedule needs rebuilding around them.**

Note the repo convention if they do land in `web/`: binaries at that depth are
re-included only for `.json/.csv/.md/.png`
(`.gitignore:43–48`), and video files are large — `results/` already breaks
the 10 MB convention with no LFS configured. **Decide where the clips live
before recording them, not after.**

---

## 4. Report, merge and push windows

| window | work | notes |
|---|---|---|
| **Sun 09:00–11:30** | Option A (mode A), launched and running | needs the box otherwise idle |
| **Sun 11:30–12:30** | Table 5 fill + H3 verdict + L28 proposal text | fill rule already fixed |
| **Sun 12:30–14:00** | report updates: the three scalar passes, H1's disconfirmation, the `proc` finding, the `k=2` share, L27's verdict | |
| **Sun 14:00–15:30** | clips — **provisional, see §3** | |
| **Sun 15:30–16:30** | gates: `pytest -q` (background it — **1,285 tests, ~7 min warm, ~26 min cold**), `check_secrets.py --all`, push scan with real blob paths | |
| **Sun 16:30–17:00** | read the gate output, fix or report | |
| **Sun 17:00–17:30** | merge decision, then push | |

**Two scheduling facts worth pinning now.**

`pytest -q` over the whole suite **exceeds the default Bash timeout** — 1,285
tests, ~7 min warm and ~26 min cold — so it must be backgrounded and started
no later than 15:30, with doc edits happening alongside it rather than after.

The push scan needs **real repository paths**, not a flat `git log -p` dump: a
flat dump trips `check_secrets`' own allow-list and the scan looks clean when
it is not. Mirror the blobs at their repo paths.

---

## 5. What must be true at 17:00 Sunday to push at 17:30

A checklist, each item either true or not. **Any "not" means do not push; fix
or report, and push later.**

**The data**

1. `ls results/lanl_v2/units_default/ | wc -l` is **40**, and
   `scores_default/` is **40**, as matched pairs.
2. `fill_analysis.py` ran **without** `--dry-run` and
   `results/lanl_v2/filled_tables.json` exists. No `filled_tables_dryrun.json`
   is committed.
3. Every cell in Table 1 has **n = 5**. Any cell with fewer is labelled
   `NOT A FINDING` and no verdict is drawn from it.
4. All five of `fill_analysis.py`'s gates passed silently — sha256, mode,
   thread agreement, pairing, and the re-scored-AUC equality.

**The claims**

5. Every number in `ANALYSIS_TEMPLATE.md` and in the report resolves to an
   exact JSON path. No `[pending]` remains in a section presented as a result.
6. L27's verdict carries **both** caveats: the realised **1/4** share, and the
   bar sitting inside Part 30's band (smallest committed cell
   −0.035962653276654266, below the 0.036 bar).
7. L28 is **not** described as closed. If mode A ran, H3 is reported and L28
   is narrowed; if it did not, Table 5 still says `pending mode A`.
8. Per-seed **signs** are reported before every aggregate mean, and
   `seeds_exactly_zero` is reported as its own count.
9. No `+/-` pools the two execution modes. Every ENIV value carries its
   thread count.
10. The `proc`-occupancy finding is stated as a **selection artefact of rule
    (a)**, and `fallback_survivor_k2` is stated as **not a signal the model can
    see**. Neither is presented as detection.

**The repository**

11. `PRE_REGISTRATION.md` and `docs/KNOWN_LIMITATIONS.md` are **unedited** —
    `git diff main --stat` shows neither. All status changes live in
    `docs/proposed_limitations_update.md` marked PROPOSAL.
12. `pytest -q` is **green**, and its output was actually read.
13. `check_secrets.py --all` is clean, and the push scan ran against real repo
    paths.
14. The **~18 MB** `scores_default/` decision is made deliberately: 40 npz at
    ~450 KB each, over the repo's 10 MB convention, no LFS, and already
    `savez_compressed` so gzip does not help. Either commit them knowingly or
    exclude them knowingly — do not let the push decide by default.
15. The branch is `part38-lanl` and the merge target is understood. **Nothing
    has been pushed before the 17:00 review.**

**The honest-answer test, last**

16. Part 38's headline is stated as what it is: **H1 disconfirmed**, the
    selection effect **relocated to `proc`** rather than removed, L27's
    contrast tested at **a quarter** of its intended strength, and L28
    **unclosable by its own text** under this pre-registration. If the write-up
    reads better than that, it is wrong.
