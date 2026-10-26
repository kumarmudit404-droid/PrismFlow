# Proposed changes to docs/KNOWN_LIMITATIONS.md

**P1, P2 and P3 were accepted and applied** to `docs/KNOWN_LIMITATIONS.md` in
Part 33, as L22's new bullet, L31 and L32, with one change to P3: the Python
line now says the tests ran under the development `.venv` (3.13.7), the same
minor version as the image base `python:3.13-slim` and a different build of
it, rather than "not the image's interpreter". The items below are the text as
proposed. This file is the proposal queue: each item names what writing
`docs/FINAL_REPORT.md` surfaced, what evidence supports it, and the exact text
proposed. Accepting or rejecting each one is the user's call.

Written 2026-10-25, against `docs/KNOWN_LIMITATIONS.md` at L1–L30.

---

## P1. L22 should name where the comparability caveat is now enforced

**Type:** cross-reference. No finding changes.

**Why.** L22 establishes that the suspicion score is a maximum over
off-diagonal pairs and therefore grows with view count, so a figure from one
width must not be differenced against another. Part 32 ships a service that
scores arrays uploaded by strangers at *whatever* view count they choose,
which is exactly the comparison L22 forbids. The service states the caveat in
every `/score` response as a literal field (`suspicion_comparability`,
`serve/scoring.py`), and `tests/serve/test_scoring.py` asserts it on both the
with-evidence and without-evidence branches.

**Proposed addition** to L22, after "**A clean clone with equal pair count has
not been run.**":

> - **The live diagnostic of Part 32 states this limit in every response.**
>   `serve/scoring.py` returns `suspicion_comparability` as a literal field on
>   every `/score` result, including the branch where no suspicion score could
>   be computed, so an uploader cannot read two runs of different widths as
>   comparable without first being told they are not.

---

## P2. Window length is a free parameter on both real datasets, and no sensitivity analysis exists

**Type:** proposed new standing limitation (L31 if accepted).

**Why.** Every real-data number in the project is at one window length per
dataset — 300 s on AIT in network mode, 300 s on LANL over days 1–16. The
window length sets the label distribution, the per-view missingness and the
size of the sample space, so it conditions every metric downstream of it.
`docs/part27-ait-datacard.md` records that `rewindow` can re-bin an AIT scan
to any multiple exactly, so the analysis is *possible*; it has not been run
against the metrics. L23 and L27 touch the consequences on LANL without
stating the parameter itself as a limitation.

This is currently stated only in `docs/FINAL_REPORT.md` section 8 and listed
in its section 10 as not measured. If it belongs in the record, it belongs
here rather than in a report that cites the record.

**Proposed text:**

> ## L31. One window length per real dataset, and no sensitivity analysis
>
> **Component:** every Part 28, 29 and 30 metric (`experiments/ait_clean/`,
> `experiments/ait_attack/`, `experiments/lanl/`).
>
> **Evidence:** `results/ait_clean/summary.md` and `results/lanl/results.json`
> (`window_seconds`); `docs/part27-ait-datacard.md`.
>
> - **AIT is reported at 300 s in network mode, LANL at 300 s over days
>   1–16.** No metric in either Part is reported at a second window length.
> - **The window length is not a neutral parameter.** It sets which lines fall
>   in a positive window, how often a view is empty and therefore masked, and
>   the size of the `(unit, window)` sample space that [[L23]]'s prevalence is
>   computed against.
> - **Re-binning is implemented and tested, but was never run against the
>   metrics.** `rewindow` re-bins one scan to any multiple exactly for
>   merge-closed features.
>
> **What this means.** Every real-data figure is conditional on one choice of
> window length, and the project has no measurement of how much any of them
> would move at another.
>
> **What would change our understanding.** The same metrics at two or more
> window lengths per dataset, reported separately and never pooled.

---

## P3. The Part 32 deployment artefact is untested, and nothing records that

**Type:** proposed new standing limitation (L32 if accepted), or a line in
`docs/PROJECT_STATUS.md` if a limitation is judged too heavy.

**Why.** Part 32 ships a `Dockerfile` as the deployment path for the live
diagnostic service. No `docker build` has ever been run against it — the
Docker daemon is not running on the development machine, and the CLI cannot
reach it. The service's Python code is tested; the image that would carry it
is not. `docs/part32-live-service.md` now says so in the deployment section,
but the standing record does not.

**Proposed text:**

> ## L32. Part 32's container image has never been built
>
> **Component:** the deployment path of the live diagnostic service
> (`Dockerfile`, `serve/requirements.txt`).
>
> **Evidence:** `docs/part32-live-service.md`, "Deploying to Hugging Face
> Spaces".
>
> - **No `docker build` has been run against the Dockerfile.** The pinned
>   dependency set, the CPU-only torch wheel and the non-root user are all
>   unexercised; the first build is the deployer's and it may fail.
> - **The service's own code is tested** under `tests/serve/`, at the Python
>   version of the development `.venv`, which is not the image's interpreter.
>
> **What this means.** No claim that the service runs in a container is
> supported by anything in this repository.
>
> **What would change our understanding.** One successful build, and a
> `/health` response from the running container.

---

## P4. L28's thread-count observation, re-measured in Part 34

**Type:** evidence added to an existing entry. The finding does not change and
nothing is resolved.

Part 34 step 6 ran `tests/unit/test_ait_attack.py::test_clean_dependence_matches_part28`
at two thread counts, on the frozen test exactly as committed. Both results:

| run | torch threads | result | `eniv_eigen` obtained |
|---|---|---|---|
| `OMP_NUM_THREADS=1` | 1 | **FAILED** (38.18 s) | 3.4289614712229826 |
| default | 8 | **passed** (44.78 s) | not printed on a pass |

Part 28's committed seed-0 value is **3.4396320614332856** and the test
asserts equality to `abs=1e-6`. The 1-thread run misses it by
**-0.010670590210303**, about **1.07e4** times the tolerance.

Two things this adds to L28 that it does not already say:

- **The 1-thread value is itself reproducible.** It equals
  `results/lanl/execution_mode.json`
  (`thread_count_reproducibility_observation.measured.omp_num_threads_1_eniv_eigen_obtained`)
  **to the last digit** -- 3.4289614712229826 both times, measured on a
  different day in a different session. So the thread count does not make the
  computation noisy; it selects between two deterministic answers. That is a
  stronger statement than "the std understates the spread", and it means a
  thread count is part of an experiment's identity, not a performance knob.
- **The default count on this machine is 8, not 12.** L28 and
  `execution_mode.json` both describe run 1 as "torch's default 12 threads".
  The default is a property of the machine, so "the default" is not a
  reproducible specification of a run; the number has to be recorded. The test
  passes at 8 and at 12, which says the committed value is not unique to 12,
  but it does not tell us the set of counts that reproduce it.

**Proposed text,** to be added to L28 as a bullet after "The same seed differs
across modes":

> - **The thread count selects between deterministic answers, it does not add
>   noise.** Re-measured in Part 34 at two counts on the unmodified test:
>   `OMP_NUM_THREADS=1` obtains `eniv_eigen` **3.4289614712229826** against
>   Part 28's committed **3.4396320614332856** and fails the `abs=1e-6`
>   assertion by **0.0107**; the machine default (**8** threads here) passes.
>   The 1-thread value matches the one recorded in
>   `results/lanl/execution_mode.json` to the last digit, from a separate
>   session, so each count reproduces its own value exactly. A reported figure
>   is therefore conditional on a thread count, and "the default" does not
>   name one: this machine's default is 8 while the run-1 record says 12.

**Status of L28: OPEN, unchanged.** What would change it is still seeds run
across at least two execution modes, which Part 34 did not do -- step 6 is one
seed of one diagnostic at two thread counts, not a replicate design.

---

## Rejected while writing the report

- **An entry for the AIT honest-duplication result.** Part 29's
  `sensor_group_check` has no within-group pair to score, which is already
  carried by L17 and by `docs/part27-ait-datacard.md` section 9. A new
  limitation would restate them.
- **An entry for the Part 32 service's unconditional dependence estimate.**
  The service already says so in every response (`conditioning`), and it is a
  property of running without a model rather than a defect of the method.

---

## P5. Proposed status of L25 and L32 after Part 34

Part 34's gate asks for a proposed status for the two limitations it set out
to close. Neither proposal edits `docs/KNOWN_LIMITATIONS.md`.

### L25 — proposed **RESOLVED**, with one residue named below

L25's "what would change our understanding" is *"a width resolution passed in
by the caller rather than looked up by source name, which would let one
implementation serve every dataset."* That now exists and is checked.

**Evidence.** `prismflow/training/` (`scaling.py`, `trainer.py`). The native
width of each view arrives as the `native_dims` argument; no module in the
package imports a dataset module or reads a source-name table — its only
first-party imports are `prismflow.data.loaders`, `prismflow.models.*`,
`prismflow.train` and `prismflow.utils.seed`. `tests/unit/test_training_agnostic.py`,
9 tests, shows the arithmetic is the same arithmetic and not a rewrite of it:

- `test_scaler_equals_run_lanl_mirror_bit_for_bit` — equal to
  `experiments/lanl/run_lanl.py`'s mirror on the LANL `with_auth` train block
  of `results/lanl/samples.npz`: `mean`, `std`, `native_dims` `[8, 5, 7, 2]`,
  `fit_rows` 42912, `present_rows_per_view`, and the scaled `[n, V, D]` array.
- `test_scaler_equals_part28_fit_scaler_bit_for_bit` — equal to Part 28's
  `fit_scaler` on the AIT widths `[9, 8, 9, 9, 10]`.
- `test_trainer_reproduces_committed_lanl_probabilities` — one training unit
  (LANL `with_auth` naive, seed 0) at the 1 thread
  `results/lanl/execution_mode.json` records, reproduces
  `results/lanl/scores/with_auth_naive_seed0.npz` exactly: `prob_attack` and
  `vacuity` bit-equal over 18729 test samples, largest absolute difference
  **0.0**. The assertion carries no tolerance; on failure it reports the
  largest difference and the thread count.

**The residue.** `experiments/ait_clean/run_ait_clean.py` and
`experiments/lanl/run_lanl.py` are frozen and were not modified, so the two
existing copies are still in the tree. What Part 34 removes is the *cause* —
the source-name lookup — and therefore the forward claim in L25's "what this
means": a third dataset no longer hits this wall. If L25 is read instead as
"two copies exist in the repository", it stays **OPEN** until a Part that is
authorised to edit those two files calls `prismflow.training`.

### L32 — proposed **OPEN**, unchanged

**Evidence.** `results/serve_docker/report.json` and
`docs/part32-live-service.md`. The Docker CLI (client 29.7.2, context
`desktop-linux`) answers, but no daemon is reachable: `docker info`,
`docker ps` and `docker desktop status` all fail on
`npipe:////./pipe/dockerDesktopLinuxEngine` — "The system cannot find the file
specified." Re-probed on 2026-10-26 at the start of the gate, with the same
result. No build was started, so every field of the step-3 report that would
describe the image is recorded as *not measured* rather than as a failure.
L32's closing condition — one successful build and a `/health` response from
the running container — is untouched.

**To close it:** `docker desktop start`, wait for the daemon, then re-run
`/part34` step 3.

---

<!-- PART35:BEGIN -->

## Part 35 proposals (L23, L26, L27, L28 `with_auth`, L29)

Written by `experiments/lanl_reanalysis/run.py` from `results/lanl_reanalysis/results.json`. Every figure is generated, not transcribed. `docs/KNOWN_LIMITATIONS.md` is unchanged.

### L29 - proposed **CLOSED**

L29's own "what would change our understanding" is "a test of each candidate cause, and a Brier below the constant baseline once the responsible one is fixed". All three named candidates were tested under a rule fixed before any number existed, one of them is responsible, and removing its rows does put Brier below the baseline. The responsible cause is NOT fixed, because the rows it names are unpredictable by construction - excluding them changes the evaluated population rather than the model - so the operative warning is unchanged. Proposed CLOSED as diagnosed; the user may prefer OPEN on the strength of the unfixed half, and that is their call.

> **L29 (update, Part 35).** Each candidate cause was corrected alone on the saved test-block scores, 5 seeds, both systems, against the fixed two-class constant baseline 0.0428. Pre-registered rule: a cause accounts for the failure only if correcting it alone puts Brier below 0.0428 in 5 of 5 seeds for both systems.
> - `without_auth`: 3600 of 18729 test rows have an all-False view mask (139 of them attack rows). On every one of them, in both systems and all 5 seeds, `prob_attack` is exactly 0.5 and the saved `vacuity_confidence` is exactly 0.0 - the least confident value it can take. A constant 0.5 scores the maximum two-class Brier of 0.5 per row, so those rows alone contribute 0.0961 to the arm's Brier, against a whole budget of 0.0428.
> - `without_auth` naive: uncorrected 0.1218 +/- 0.0006; present rows only 0.0318 +/- 0.0007 against that subset's own constant baseline 0.0352; absent rows 0.5000 +/- 0.0000; prior-shift ORACLE 0.2033 +/- 0.0015; cross-fitted Platt DIAGNOSTIC 0.0432 +/- 0.0000
> - `without_auth` prismflow: uncorrected 0.1219 +/- 0.0005; present rows only 0.0319 +/- 0.0006 against that subset's own constant baseline 0.0352; absent rows 0.5000 +/- 0.0000; prior-shift ORACLE 0.2038 +/- 0.0013; cross-fitted Platt DIAGNOSTIC 0.0432 +/- 0.0000
> - `without_flows`: 998 of 18729 test rows have an all-False view mask (0 of them attack rows). On every one of them, in both systems and all 5 seeds, `prob_attack` is exactly 0.5 and the saved `vacuity_confidence` is exactly 0.0 - the least confident value it can take. A constant 0.5 scores the maximum two-class Brier of 0.5 per row, so those rows alone contribute 0.0266 to the arm's Brier, against a whole budget of 0.0428.
> - `without_flows` naive: uncorrected 0.0628 +/- 0.0004; present rows only 0.0382 +/- 0.0004 against that subset's own constant baseline 0.0452; absent rows 0.5000 +/- 0.0000; prior-shift ORACLE 0.0882 +/- 0.0008; cross-fitted Platt DIAGNOSTIC 0.0427 +/- 0.0001
> - `without_flows` prismflow: uncorrected 0.0630 +/- 0.0006; present rows only 0.0384 +/- 0.0006 against that subset's own constant baseline 0.0452; absent rows 0.5000 +/- 0.0000; prior-shift ORACLE 0.0891 +/- 0.0011; cross-fitted Platt DIAGNOSTIC 0.0426 +/- 0.0001
> - Verdicts: `without_auth`: accounted for by (a) the all-False view mask; not accounted for by (b) the prior shift (ORACLE) (0/5 naive, 0/5 prismflow), (c) cross-fitted Platt (DIAGNOSTIC) (0/5 naive, 0/5 prismflow); `without_flows`: accounted for by (a) the all-False view mask; not accounted for by (b) the prior shift (ORACLE) (0/5 naive, 0/5 prismflow), (c) cross-fitted Platt (DIAGNOSTIC) (4/5 naive, 5/5 prismflow).
> - **"Accounts for" is not "fixed".** The rows cause (a) names are unpredictable by construction (Part 30 section 4.2): every model must score them identically whatever their label. Restricting the metric to the rows with a present view changes the evaluated population; it does not change a probability. A deployed detector still meets those cells and still emits 0.5 on them.
> - The prior-shift correction makes it WORSE in both failing arms, and it is an ORACLE (it uses the test prevalence). The Platt recalibration is a DIAGNOSTIC fitted on test rows and cross-fitted by computer fold, because `results/lanl/` holds no train-block scores; it does not meet the rule either. Neither is a deployable fix, so the operative sentence stands: **probability outputs of those arms should not be used.**
> - A route for Part 38, not run here: the protocol's own `vacuity_confidence` is already exactly 0.0 on precisely these rows, so abstaining on them is a selective-prediction decision the existing machinery can express, rather than a new model.
> - Evidence: `results/lanl_reanalysis/results.json`, `step2_l29`; `experiments/lanl_reanalysis/README.md` section 1.

### L23 - proposed **CLOSED**

L23 asks for "a measurement at, or reweighted to, the population prevalence". That measurement now exists, and it confirms the limitation rather than lifting it: the reweighting is only as good as the uniformity of the control hash, and its false-alarm resolution is 512 cells, so it still gives no operating point that transfers.

> **L23 (update, Part 35).** Metrics were recomputed under inverse-probability weights (attack 1, benign 512, the declared 1/512 control hash) on the saved test-block scores, 5 seeds per arm and system.
> - `without_auth` naive: weighted Brier 0.097901 +/- 0.000881, weighted prevalence 0.00004371, and at the 0.5 threshold weighted precision 0.000090 +/- 0.000000, recall 0.3912 +/- 0.0013, FPR 0.18904962 +/- 0.00007117 at 443290 +/- 167 weighted false alarms per test day
> - `without_auth` prismflow: weighted Brier 0.098170 +/- 0.000824, weighted prevalence 0.00004371, and at the 0.5 threshold weighted precision 0.000090 +/- 0.000001, recall 0.3907 +/- 0.0032, FPR 0.18901687 +/- 0.00004883 at 443213 +/- 114 weighted false alarms per test day
> - `without_flows` naive: weighted Brier 0.030717 +/- 0.000455, weighted prevalence 0.00004371, and at the 0.5 threshold weighted precision 0.000089 +/- 0.000008, recall 0.1112 +/- 0.0095, FPR 0.05490474 +/- 0.00008097 at 128742 +/- 190 weighted false alarms per test day
> - `without_flows` prismflow: weighted Brier 0.031319 +/- 0.000478, weighted prevalence 0.00004371, and at the 0.5 threshold weighted precision 0.000085 +/- 0.000011, recall 0.1073 +/- 0.0143, FPR 0.05499208 +/- 0.00008279 at 128947 +/- 194 weighted false alarms per test day
> - `without_dns` naive: weighted Brier 0.005472 +/- 0.000437, weighted prevalence 0.00004371, and at the 0.5 threshold weighted precision 0.001230 +/- 0.000236, recall 0.1263 +/- 0.0184, FPR 0.00451990 +/- 0.00035251 at 10598 +/- 827 weighted false alarms per test day
> - `without_dns` prismflow: weighted Brier 0.005936 +/- 0.000623, weighted prevalence 0.00004371, and at the 0.5 threshold weighted precision 0.001232 +/- 0.000152, recall 0.1244 +/- 0.0150, FPR 0.00441072 +/- 0.00012448 at 10342 +/- 292 weighted false alarms per test day
> - `with_auth` naive: weighted Brier 0.003592 +/- 0.000472, weighted prevalence 0.00004371, and at the 0.5 threshold weighted precision 0.009390 +/- 0.002252, recall 0.1273 +/- 0.0199, FPR 0.00062230 +/- 0.00020643 at 1459 +/- 484 weighted false alarms per test day
> - `with_auth` prismflow: weighted Brier 0.004228 +/- 0.000559, weighted prevalence 0.00004371, and at the 0.5 threshold weighted precision 0.010981 +/- 0.004255, recall 0.1444 +/- 0.0108, FPR 0.00062230 +/- 0.00019145 at 1459 +/- 449 weighted false alarms per test day
> - The weights reconstruct the occupied-cell population only as far as the crc32 control hash is uniform over cells; uniformity is assumed, not measured. A weighted false-positive count moves in steps of 512 cells over 4 test days, so **no alarm rate between 0 and 128 per day is representable** and all three pre-registered targets (1, 10 and 100 false alarms per test day) collapse onto the same zero-alarm threshold. The weighted precision of 1.0 there means no SAMPLED control fired, which estimates the population false-alarm count as 0 with a resolution of 512 cells; it does not mean a detector with no false alarms. A reweighted estimate is therefore not an operating point.
> - Population cells over the test days alone: between 6,554,950 and 15,730,712 (`results/lanl/population.json`, summed per day over days 13-16).
> - Evidence: `results/lanl_reanalysis/results.json`, `step3_l23`; `experiments/lanl_reanalysis/README.md` section 2.

### L26 - proposed **CLOSED**

L26 asks for "an unseen-computer result that clears the scalar by more than 0.01 with an interval attached". The interval is now attached, by the pre-registered 2000-replicate computer-cluster bootstrap, and the result confirms the limitation: NO arm-and-system pair clears that bar, and every paired model-minus-scalar interval on unseen computers includes 0. Closed as confirmed, with the numbers whichever way they fell.

> **L26 (update, Part 35).** A computer-cluster bootstrap (2000 replicates, fixed seed 12345, 8,350 test computers drawn with replacement) was run on the saved scores, paired against the untrained scalar inside each replicate. The scalar was recomputed from `samples.npz` with Part 30's train-fixed orientation and reproduces the committed test AUC 0.6843 before any interval was computed.
> - Scalar `total_records_across_views`: seen AUC 0.6405 [0.5347, 0.8990], unseen AUC 0.7563 [0.7030, 0.8078].
> - Paired model-minus-scalar on unseen computers, `without_auth/naive`: -0.0461 [-0.1127, 0.0224], **includes 0**
> - Paired model-minus-scalar on unseen computers, `without_auth/prismflow`: -0.0436 [-0.1106, 0.0250], **includes 0**
> - Paired model-minus-scalar on unseen computers, `without_flows/naive`: -0.0127 [-0.0728, 0.0479], **includes 0**
> - Paired model-minus-scalar on unseen computers, `without_flows/prismflow`: -0.0055 [-0.0625, 0.0534], **includes 0**
> - Paired model-minus-scalar on unseen computers, `without_dns/naive`: 0.0021 [-0.0682, 0.0697], **includes 0**
> - Paired model-minus-scalar on unseen computers, `without_dns/prismflow`: 0.0026 [-0.0675, 0.0703], **includes 0**
> - Paired model-minus-scalar on unseen computers, `with_auth/naive`: 0.0043 [-0.0634, 0.0714], **includes 0**
> - Paired model-minus-scalar on unseen computers, `with_auth/prismflow`: 0.0052 [-0.0639, 0.0723], **includes 0**
> - Above the scalar with an interval excluding 0: none. Below it with an interval excluding 0: none. Interval includes 0: `without_auth/naive`, `without_auth/prismflow`, `without_flows/naive`, `without_flows/prismflow`, `without_dns/naive`, `without_dns/prismflow`, `with_auth/naive`, `with_auth/prismflow`. L26's own clearance bar is +0.01 with an interval attached; the arm-and-system pairs that clear it are **none**.
> - Evidence: `results/lanl_reanalysis/results.json`, `step4_l26`; `experiments/lanl_reanalysis/README.md` section 3.

### L27 - proposed **OPEN**

L27's "what would change our understanding" is a DESIGN change - a sample that keeps all or a stated share of the benign cells of attack hosts. That needs a re-sample and a retrain, which is Part 38. Part 35 only widens the sensitivity measurement from one host to five, so L27 stays OPEN with new evidence attached.

> **L27 (new evidence, Part 35; still OPEN).** The leave-one-host-out measurement of section 7.9 was extended from the single top host to the 5 computers with the most test attack rows, for all four arms and both systems, on the saved scores (5 seeds, mean +/- std, no interval - one host is not a resample).
> - `C17693`: 137 attack and 1 benign test rows
> - `C1015`: 12 attack and 1 benign test rows
> - `C2388`: 10 attack and 3 benign test rows
> - `C1936`: 7 attack and 0 benign test rows
> - `C2079`: 7 attack and 3 benign test rows
> - Largest AUC change from removing one host: `without_auth/naive` minus `C17693` -0.0519; `without_auth/prismflow` minus `C17693` -0.0508; `without_flows/naive` minus `C17693` -0.0435.
> - The design change L27 names has not been made, so the limitation is unchanged. Evidence: `results/lanl_reanalysis/results.json`, `step5_l27`; `experiments/lanl_reanalysis/README.md` section 4.

### L28 - proposed **OPEN**

L28 asks for "seeds run across at least two execution modes, so the reported std covers mode as well as seed". Only `with_auth` has two modes in `results/lanl/`; the quantity is now reported per metric for that arm, and the other three arms have no second mode to compare, so they stay OPEN until Part 38 runs them.

> **L28 (new evidence, Part 35; still OPEN).** For `with_auth` only, the per-seed std within each execution mode, the spread across modes and the pooled spread are now reported per metric, both systems (`results/lanl/results.run1.json.gz` at 12 threads sequential against `results/lanl/results.json` at 1 thread, concurrency 11).
> - prismflow `attack_recall`: within-mode std 0.0196 (run 1) and 0.0146 (run 2); across-mode mean difference 0.0190; largest same-seed gap 0.0341; pooled std over the 10 values 0.0191, which is 0.98x the larger within-mode std
> - prismflow `balanced_accuracy`: within-mode std 0.0098 (run 1) and 0.0073 (run 2); across-mode mean difference 0.0095; largest same-seed gap 0.0170; pooled std over the 10 values 0.0095, which is 0.98x the larger within-mode std
> - naive `roc_auc`: within-mode std 0.0039 (run 1) and 0.0033 (run 2); across-mode mean difference -0.0031; largest same-seed gap 0.0140; pooled std over the 10 values 0.0038, which is 0.96x the larger within-mode std
> - prismflow `benign_recall`: within-mode std 0.0002 (run 1) and 0.0002 (run 2); across-mode mean difference -0.0001; largest same-seed gap 0.0002; pooled std over the 10 values 0.0002, which is 0.96x the larger within-mode std
> - prismflow `accuracy`: within-mode std 0.0004 (run 1) and 0.0003 (run 2); across-mode mean difference 0.0003; largest same-seed gap 0.0007; pooled std over the 10 values 0.0004, which is 0.96x the larger within-mode std
> - naive `brier`: within-mode std 0.0005 (run 1) and 0.0005 (run 2); across-mode mean difference 0.0001; largest same-seed gap 0.0010; pooled std over the 10 values 0.0005, which is 0.94x the larger within-mode std
> - naive `balanced_accuracy`: within-mode std 0.0081 (run 1) and 0.0088 (run 2); across-mode mean difference 0.0002; largest same-seed gap 0.0146; pooled std over the 10 values 0.0080, which is 0.91x the larger within-mode std
> - naive `attack_recall`: within-mode std 0.0163 (run 1) and 0.0176 (run 2); across-mode mean difference 0.0005; largest same-seed gap 0.0293; pooled std over the 10 values 0.0160, which is 0.91x the larger within-mode std
> - prismflow `brier`: within-mode std 0.0003 (run 1) and 0.0002 (run 2); across-mode mean difference -0.0003; largest same-seed gap 0.0007; pooled std over the 10 values 0.0003, which is 0.88x the larger within-mode std
> - prismflow `roc_auc`: within-mode std 0.0036 (run 1) and 0.0043 (run 2); across-mode mean difference -0.0003; largest same-seed gap 0.0042; pooled std over the 10 values 0.0038, which is 0.87x the larger within-mode std
> - naive `accuracy`: within-mode std 0.0003 (run 1) and 0.0003 (run 2); across-mode mean difference -0.0000; largest same-seed gap 0.0006; pooled std over the 10 values 0.0003, which is 0.87x the larger within-mode std
> - prismflow `prob_ece`: within-mode std 0.0007 (run 1) and 0.0011 (run 2); across-mode mean difference 0.0004; largest same-seed gap 0.0020; pooled std over the 10 values 0.0009, which is 0.79x the larger within-mode std
> - naive `benign_recall`: within-mode std 0.0001 (run 1) and 0.0002 (run 2); across-mode mean difference -0.0000; largest same-seed gap 0.0004; pooled std over the 10 values 0.0002, which is 0.76x the larger within-mode std
> - naive `prob_ece`: within-mode std 0.0002 (run 1) and 0.0009 (run 2); across-mode mean difference -0.0001; largest same-seed gap 0.0008; pooled std over the 10 values 0.0006, which is 0.69x the larger within-mode std
> - **The reading, which sharpens L28 rather than lifting it.** The two modes' 5-seed MEANS are close: the largest across-mode mean difference over these metrics is 0.0190, and pooling the 10 values does not widen the std at all - the pooled std is at most 0.98x the larger within-mode std. What moves is the SAME SEED: the largest same-seed gap is 0.0341, which is several within-mode stds. So the mode selects between deterministic answers per seed, as L28 already says, and the 5-seed mean absorbs most of it; a SINGLE-seed figure is the one that must not be read as mode-independent.
> - The three remaining arms (`without_auth`, `without_flows`, `without_dns`) were run in one mode only, so no cross-mode spread exists for them and L28 is not closed. Evidence: `results/lanl_reanalysis/results.json`, `step6_l28`; `experiments/lanl_reanalysis/README.md` section 5.

<!-- PART35:END -->
