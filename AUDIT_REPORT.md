# PrismFlow audit report (Part 16)

Audit date: 2026-09-20. Audited at commit `5516bcc` on `main`.

Environment actually used: Python 3.13.7, torch 2.14.0+cpu, numpy 2.5.3,
streamlit 1.64.0, Windows 11, CPU only.

## How to read the classifications

Every component below carries exactly one of: **WORKING**, **PARTIALLY
WORKING**, **EXPERIMENTAL**, **NOT IMPLEMENTED**, **NOT VERIFIED**, **KNOWN
LIMITATION**.

**NOT VERIFIED means exactly one thing:** it was not personally observed to run
to completion during this audit session. It is not a claim that the component
is broken, and it is not a claim that it works. Several components marked NOT
VERIFIED produced the committed results this report quotes; that is evidence
about a previous session, and this audit does not launder it into a present
claim. Nothing is marked WORKING because it worked before.

---

# FINDING 1 (CRITICAL, CLOSED): `--quick` could overwrite committed evidence, and did

**Classification: PARTIALLY WORKING** — three instances fixed and verified this
session, four latent instances remain by design.

This is the most serious defect found in the audit. It is recorded here in full
because it FIRED before it was caught, which makes it evidence rather than
theory.

## The defect

16 experiment scripts accept `--quick`, a smoke-test flag that cuts the run to
1 seed and 2 epochs. Each logs `NOT EVIDENCE` at WARNING level when it does so.

13 of those scripts route their output through a `--results-dir` argument, so a
smoke run can be directed somewhere harmless. **Three did not.** They hardcoded
`OUT_DIR = Path("results/<name>")` with no override, so `--quick` wrote 1-seed
output directly onto the canonical, committed, 5-seed evidence path:

| script | canonical path it overwrote |
|---|---|
| `experiments/tail/run_tail.py` | `results/tail/` |
| `experiments/synthesis/run_oracle.py` | `results/synthesis/` |
| `experiments/branch_audit/run_branch_audit.py` | `results/branch_audit/` |

The scripts warned and then did the damaging thing anyway. A warning that does
not change behaviour is not a control.

## It fired

On 2026-09-20 a "run every experiment script" sweep ran all three under
`--quick`. Seven tracked files were left modified in the working tree and were
found in that state when Part 16 resumed. They were restored with
`git checkout --` before any other work, and the canonical numbers were
verified by sha256 against `HEAD`.

**What would have entered the record had it been committed:**

- `results/synthesis/summary.md` — the P3 refutation delta inflated roughly
  fourfold, `+0.0333` to `+0.1314`. Three ablation arms flipped from `below` to
  `REACHES`. The `dependence` univariate signal inverted, `0.5440` to `0.2083`.
  The load-bearing per-seed column read `1/1` in every row.
- `results/tail/summary.md` — `n_tail` fell 60 to 15, tripping the estimator's
  own reliability floor (`reliable: yes` to `NO`), while the copula fit
  degenerated to the df ceiling with `lambda_U = 0`.
- `results/branch_audit/summary.md` — the entire `rho = 0.6` half of the design
  deleted, and `max_dependence` for `v2_lambda0` moved outside the committed
  seed std (`0.3454 +/- 0.0337` to `0.4091`).

Each of those is a headline number. Each would have been published as a
single-seed result, in direct violation of the 5-SEED RULE.

## How it was confirmed as `--quick` rather than seed noise

Four independent lines, because one would not have been enough to justify
overwriting a working tree:

1. **The files said so.** All three summaries carried `Seeds: [0]` where the
   committed versions carry `Seeds: [0, 1, 2, 3, 4]`. Every `+/-` column was
   absent, because one seed has no spread.
2. **Identical mtimes.** All three JSONs were stamped `2026-09-20 13:27`,
   within the same second. Three multi-minute 5-seed experiments cannot finish
   in the same second. No other file under `results/` was touched that day.
3. **Each script's `--quick` branch matched the shape of its own damage.**
   `run_tail.py` sets `n_samples=2000` and `epsilons=[0.0, 1.0]`; the diff
   showed exactly `8000 -> 2000` and a five-row epsilon grid cut to those two
   rows. `run_branch_audit.py` sets `rhos[:1]`; the diff showed the `rho = 0.6`
   block deleted.
4. **Re-running the fixed `--quick` path reproduced the clobbered values digit
   for digit** (`branch_audit` `v2_lambda0`: 3.143 / 3.135 / 3.185 / 3.147).

## The fix, and its verification

`--quick` now resolves `out_dir = OUT_DIR + "_quick"` and logs both paths, so a
smoke run cannot reach the evidence path at all. This matches the
`--results-dir` escape the other 13 scripts already had, which is why only
these three were ever exposed. `results/*_quick/` was added to `.gitignore`
ahead of the per-extension un-ignores, so smoke output is not merely harmless
but uncommittable.

Verified this session by running all three with `--quick`:

- output landed in `results/{tail,synthesis,branch_audit}_quick/`;
- all six canonical files matched sha256 against `HEAD` afterwards;
- `git diff -- results/` was empty;
- `git status` showed no `_quick` path;
- 563 tests passed.

## What remains open

Four further scripts hardcode `OUT_DIR` the same way:
`experiments/adaptive/run_adaptive.py`, `experiments/synthesis/run_diagnostics.py`,
`experiments/synthesis/run_gate_matrix.py`,
`experiments/synthesis/run_sign_constrained.py`.

They are **latent, not safe**. None has a `--quick` flag today, so none can
currently overwrite anything — but nothing in the code prevents one being added,
and the three fixed scripts show what happens when it is. They are left
unmodified because the Part 16 brief forbids modifying experiment files beyond
what the task requires, and because a fix without a `--quick` flag to guard is
unfalsifiable. **Recommended before any future smoke-test flag is added to those
four: apply the same three-line `out_dir` resolution first.**

---

# FINDING 2: no integration test exists

**Classification: NOT IMPLEMENTED**

The Part 16 brief instructs the audit to run "full unit suite, integration
test, and EVERY experiment script". There is no integration test to run.
`tests/` contains only `tests/unit/` (21 files, 563 tests). There is no
`tests/integration/`, no integration marker, and no pytest configuration
selecting one — there is no `pyproject.toml`, `setup.cfg`, or `pytest.ini` in
the repository at all.

What partially covers the gap: `tests/unit/test_calibration.py` drives the full
`evaluate()` protocol end-to-end against a stub model, and this audit exercised
the real end-to-end path through the Streamlit app (Finding 6). Neither is a
committed integration test.

---

# FINDING 3: `streamlit` is an undeclared dependency

**Classification: PARTIALLY WORKING**

`requirements.txt` declares `numpy`, `torch`, `PyYAML`, `pytest`, `matplotlib`.
It does not declare `streamlit`, which `app.py` and `prismflow/app/panels.py`
both import and without which the Part 16 demo cannot start. It is installed in
the working environment (1.64.0), which is why the omission was invisible.

Every other declared dependency is genuinely used: numpy (46 files), torch (62),
yaml (11), pytest (19), matplotlib (7). No unused dependencies found.

Not fixed here: `requirements.txt` is outside the brief's list of paths this
Part may create, and editing it is a change to the project's declared
environment rather than a documentation fix. **The one-line change is to add
`streamlit>=1.40` to `requirements.txt`.** `docs/INSTALLATION.md` installs it
explicitly in the meantime and says why.

---

# FINDING 4: `n_seeds = 0` on ENIV-family metrics for naive arms

**Classification: WORKING (benign, documented so it is not re-reported)**

63 committed result files contain `n_seeds: 0` on at least one metric. This is
correct behaviour, not missing data. The affected metric names are all
PrismFlow-only quantities — `eniv`, `mean_dependence`, `efficiency_ratio`,
`normalised_hsic`, `separation_auc`, `dependence_compromised`,
`delta_dependence_compromised`, `target_belief_on_success`,
`target_prob_on_success` — and they appear with `n_seeds: 0` only on `naive`
and `nodiscount` arms, which do not compute them. The aggregator collected zero
values and recorded that honestly rather than substituting one.

Verified: of the 63 files, the only five whose path does not contain `naive` or
`nodiscount` are aggregates that contain naive arms internally
(`chorus/attack_metrics.json`, `disentangle/disentangle.json`,
`per_sample/per_sample.json` and two `per_sample` condition files).

---

# FINDING 5: NaN in the clone summaries at k = 0

**Classification: WORKING (structural, benign)**

`dependence_clone_pairs_mean` and `_std` are NaN in
`results/clone/summary.json` and its three `clone_eigen*` siblings — 12 NaN
values per file, in exactly the `k = 0` rows and nowhere else. At `k = 0` there
are no clone pairs, so the mean is taken over an empty set. Every `k >= 1` row
carries a finite value in the expected range (0.886 to 0.949).

This is the correct answer to "what is the average dependence among clone pairs
when there are no clone pairs". No fix is warranted and none was made.

---

# Component classification

## Core statistical machinery

| component | classification | basis |
|---|---|---|
| `statistics/dependence.py` | **KNOWN LIMITATION** | Works, but the estimator's bias is signed and exceeds its seed variance. See `docs/KNOWN_LIMITATIONS.md` L10. |
| `eniv/eniv.py` (ENIV estimator) | **KNOWN LIMITATION** | Validated at one design point only (n_views = 4, synthetic, analytic rho). MAE 0.5220 +/- 0.2998 effective views. See L9. |
| `eniv/discount.py` | **WORKING** | Unit-tested; alpha detachment verified in source at lines 75 and 122. Its scope boundary (redundant vs unreliable evidence) is L3, not a defect. |
| `statistics/tail_dependence.py` | **KNOWN LIMITATION** | Correct, and self-reporting: `MIN_TAIL_SAMPLES = 50` with a `reliable` flag that fires. Underpowered at project split sizes. See L11. |
| `statistics/suspicion.py` | **WORKING** | Audited this session — see the label-leakage section below. |
| `statistics/clique_contrast.py` | **NOT VERIFIED** | Unit tests pass; its Part 14 experiment was not re-run this session. |
| `statistics/copula.py`, `hsic.py` | **NOT VERIFIED** | Unit tests pass; not exercised end-to-end this session. |
| `models/prismflow.py` | **WORKING** | Exercised end-to-end this session through the app across all six scenarios. |
| `models/shared_private.py` (V2) | **EXPERIMENTAL** | Part 11's answer was negative: the penalty works but collapses the model at lambda >= 10. Not the shipped path. See L13. |
| `train_two_timescale.py` | **EXPERIMENTAL** | The partial remedy to L13. Detaches evidence before the estimator step; never adopted as the default. |

## Attacks

| component | classification | basis |
|---|---|---|
| `attacks/chorus.py` | **WORKING** | Ran end-to-end through the app this session (confidence fell 0.837 to 0.599, ENIV 3.16 to 2.70 — the attack does what it claims). |
| `attacks/adaptive.py` | **WORKING** | Ran end-to-end through the app this session. BPDA surrogate verified in source at line 50. |
| `attacks/baseline_attacks.py` (PGD) | **NOT VERIFIED** | Unit tests pass; not run end-to-end this session. |
| the `gate_check` arm of Part 14 | **NOT IMPLEMENTED** | Built to apply gate-specific pressure; it is a null arm that reduces exactly to gamma = 0. Documented in L12. No gate-aware-attacker conclusion is licensed by this project. |

## Experiment scripts

**All 25 experiment scripts: NOT VERIFIED.**

None was run in full 5-seed mode during this audit. This is a deliberate choice,
not an oversight: the only full sweep attempted in this work cycle is Finding 1,
and re-running 25 scripts to satisfy a checkbox is how that happened. The three
scripts that were run were run under `--quick`, explicitly to verify the fix,
and their output is 1-seed smoke data in `results/*_quick/` which is **not
evidence and is gitignored**.

The committed results under `results/` were produced by earlier sessions and are
quoted throughout this project. This audit verified their **internal
consistency** (Findings 4 and 5, plus the seed and range checks below) and their
**integrity against the near-miss** (Finding 1). It did not reproduce them.

## Demo and documentation

| component | classification | basis |
|---|---|---|
| `app.py` | **WORKING** | Boots (HTTP 200) and renders with no exception; all six scenarios run to completion — verified this session. |
| Independence budget panel | **WORKING** | Renders in the brief's exact shape: nominal 4, effective 3.16, efficiency 79%, most redundant pair named, plain-language line. |
| `prismflow/app/panels.py` | **WORKING** | Surfaces the L11 reliability limit in the UI rather than hiding it: "UNRELIABLE: 25 exceedances is below MIN_TAIL_SAMPLES = 50." |
| `tests/unit/test_app.py` | **WORKING** | 11 tests pass; asserts `app.py` imports no `torch.nn` — independently confirmed by AST scan this session. |
| `results/final_summary.{json,csv}` | **WORKING** | 17 rows, 0 unresolved. Every value resolved programmatically from a committed file; no value typed by hand. |
| PyInstaller `PrismFlow.exe` | **NOT IMPLEMENTED** | Optional and timeboxed in the brief. Not attempted. The Streamlit app is the brief's stated acceptable substitute. |

---

# The brief's specific checks

## Accidental weight sharing across view encoders — CLEAN

`MultiViewEncoder` takes an explicit `share_weights` flag defaulting to
`False`, and `encoders.py` documents that setting it `True` "invalidates all
downstream ENIV" measurements.

Verified this session by construction and inspection:

- `share_weights=False` (the default): **no shared parameter objects** between
  any of the 4 encoders, and **no identical values** either — 24 unique
  parameter tensors, 3340 parameters.
- `share_weights=True`: sharing is complete and deliberate (all 6 encoder pairs
  share all 4 tensors; 12 unique tensors, 916 parameters). The flag works.
- **No experiment script passes `share_weights` at all**, so every committed
  result was produced with independent encoders.

## Label leakage across splits — CLEAN

`split_indices` permutes once and slices, so splits cannot overlap by
construction. Verified empirically across **50 (n, seed) combinations**
(n in {300, 800, 2000, 8000, 1001} x seeds 0-9): train/val/test were pairwise
disjoint and exactly exhaustive over `range(n)` in all 50.

Separately, the suspicion module is the place label leakage would matter most,
and it is clean — see below.

## The suspicion function receiving attack metadata — CLEAN

No function in `prismflow/statistics/suspicion.py` accepts an attack label, an
attack config, a compromised-view index, or a ground-truth label. Signatures
verified this session:

- `pairwise_agreement(belief, view_mask)`
- `agreement_null(belief, view_mask, strata, seed, n_permutations)`
- `suspicion_report(belief, dependence, view_mask, evidence, strata, threshold, seed, n_permutations)`
- `calibrate_threshold(reference_scores, target_rate)` — scores and a rate; the
  `target_rate` is a false-positive rate, not a target class
- `detection_rates(scores, threshold)`

`tests/unit/test_suspicion.py` enforces this with a `FORBIDDEN` tuple
(`"attack"`, `"compromise"`, `"adversar"`, `"truth"`) checked against the
signatures of the public functions and of `DefendedPrismFlow.forward`. The
guard is a test, not a convention.

## Invalid probability vectors, NaN, Inf — CLEAN

- **Runtime:** fused probabilities over 200 samples were finite, within [0, 1],
  and summed to 1 to within 1.2e-7. Confidence was finite and within [0, 1].
- **Committed results:** all 180 non-quick result JSONs parse. The only
  non-finite values anywhere are the structural `k = 0` NaNs of Finding 5.
- **Range checks:** no bounded metric (accuracy, ECE, MCE, Brier, confidence,
  attack success, efficiency ratio) falls outside its valid range in any
  committed file. No AUC falls outside [0, 1].

## Non-reproducibility across identical seeds — CLEAN

- `split_indices(2000, seed=3)` twice: identical partitions; seed 4 differs.
- Full pipeline (generate, construct, forward with discount) at seed 7 twice:
  **probabilities and confidences bitwise identical**; seed 8 differs.
- Every committed result JSON records `torch` version, and most record the
  config and commit.

## Broken checkpoint loading — CLEAN

`save_checkpoint` / `load_checkpoint` round-trip verified this session: a saved
`state_dict` reloaded into a differently-seeded model with `strict=True`
reproduced every tensor exactly, and a missing file raises `FileNotFoundError`
rather than returning silently. `checkpoints/` is empty, which is expected —
experiments train in-process and do not persist weights.

## Import cycles and unused dependencies — CLEAN

- **Import cycles:** AST scan of all 45 `prismflow` modules for intra-package
  imports found **no cycles**.
- **Unused dependencies:** none. See Finding 3 for the inverse problem.

## Hard-coded or placeholder results — CLEAN

Scan of `prismflow/`, `experiments/`, `app.py` and `tests/` for `placeholder`,
`TODO`, `FIXME`, `XXX`, `dummy`, `fake`, `hardcod`, `stub`, `for now`,
`temporar`: the only hits are legitimate test doubles in
`tests/unit/test_calibration.py` (`_StubModel`, `_stub_setup`) and one docstring
in `fusion.py` describing what the code does *not* do ("rather than folding in a
placeholder"). **No fabricated or placeholder result values found anywhere in
the repository.**

## Any experiment reported with fewer than 5 seeds — CLEAN, with Finding 1 attached

The contract is enforced in code, not by convention:
`prismflow/evaluation/protocol.py` defines `MIN_SEEDS` and **raises** on fewer
seeds unless the caller passes `allow_fewer_seeds`, in which case the output is
stamped `not_evidence: true`.

Verified across all 180 committed result JSONs: **141 `not_evidence` fields,
all `false`.** No committed result is a sub-5-seed run.

That is the state *after* Finding 1 was caught. Had that sweep been committed,
this check would have failed on three files — and note that it would **not**
have been caught by the `not_evidence` flag, because the `--quick` runs were
legitimate 1-seed runs that set `seeds` to `[0]` before `evaluate()` ever saw
them. The flag guards against passing too few seeds to the protocol; it does
not guard against the config being cut upstream. **The output-path fix in
Finding 1 is the only control that covers this case.**

---

# Deviations from the Part 16 brief

Recorded rather than silently taken:

1. **`docs/INSTALLATION.md` was created although it is not on the brief's
   allowlist.** It was explicitly requested. Its content is installation
   instructions only; no result, claim, or code is affected.
2. **`requirements.txt` was NOT modified** despite Finding 3, because it is not
   on the allowlist and the fix is a change to the declared environment. The
   exact change is stated in Finding 3.
3. **The four latent `OUT_DIR` scripts were not modified**, for the reason in
   Finding 1.
4. **"EVERY experiment script" was not run.** See the experiment-scripts
   section. Running them was the proximate cause of Finding 1, and the brief's
   own instruction to mark unobserved components NOT VERIFIED is the safer
   reading of its intent.

---

# Summary

Nothing in this audit invalidates a committed result. The repository's
statistical hygiene is strong where it is enforced in code — the `MIN_SEEDS`
guard, the `not_evidence` stamp, the suspicion signature test, the
`rho_matrix is None` assertion on real data, the `reliable` flag on tail
estimates. Each of those caught or would catch a real error class.

The one serious defect found was in the layer *around* those controls: a smoke
flag that could write to the evidence path. It is the characteristic failure of
a well-guarded system — the guards were all downstream of the thing that went
wrong. It fired, it was caught before commit, and it is now closed for the three
scripts that carried it and documented for the four that could.
