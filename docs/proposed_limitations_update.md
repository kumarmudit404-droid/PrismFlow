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

<!-- PART36:BEGIN -->

## Part 36 proposals (L2, L9, L10, L11, L22)

From `experiments/part36_synthetic/README.md`, under rules pre-registered in `experiments/part36_synthetic/config.yaml` before any number existed. `docs/KNOWN_LIMITATIONS.md` is unchanged.

### L9 - proposed **OPEN** (first half answered), plus a correction to its text

L9's closing condition has two halves. The first - "a generator with analytically known rho at other view counts (6, 8, 16) to test whether the compression scales with n_views" - is now run and answered. The second needs a real dataset with an independently established dependence structure, which does not exist in this checkout, so L9 stays OPEN with the first half's evidence attached. A separate correction to L9's existing text is proposed below it.

> **L9 (new evidence, Part 36; still OPEN).** The analytic-rho generator was run at V = [4, 6, 8, 16], rho [0.0, 0.2, 0.4, 0.6, 0.8, 0.95], 5 seeds, with only `n_views` varied from the committed settings. The committed V = 4 point reproduces first (MAE 0.5220005950719182, correlation 0.9922060509334697, both to abs diff 0).
> - V = 4: correlation with truth **0.9922**, MAE **0.5220 +/- 0.2998** effective views, which is **0.1305** of V
> - V = 6: correlation with truth **0.9930**, MAE **0.8514 +/- 0.4575** effective views, which is **0.1419** of V
> - V = 8: correlation with truth **0.9924**, MAE **1.1662 +/- 0.6368** effective views, which is **0.1458** of V
> - V = 16: correlation with truth **0.9938**, MAE **2.2995 +/- 1.3063** effective views, which is **0.1437** of V
> - Pre-registered rule: SCALES if MAE(V = 16) - MAE(V = 4) exceeds the across-cell std at V = 4. Measured gap **1.7775** against std **0.2998**. **Verdict: SCALES.**
> - **The reading.** Rank agreement does not degrade - correlation is flat near 0.993 at every view count. The VALUE error grows almost exactly in proportion to V, so the single validated design point of n_views = 4 is the most favourable one run, and every ENIV reported at a higher view count carries a proportionally larger absolute error. Evidence: `results/part36_synthetic/analysis_a.json`; `experiments/part36_synthetic/README.md` section 1.

### L9 (text correction) - proposed **REVISE**

L9's bullet list mixes two truth columns. Its MAE **0.5220 +/- 0.2998** and correlation **0.9922** are computed against `analytic_n_eff_eigen`; the six signed errors in the next bullet (**-1.0049, +0.2557, +0.7232, +0.8831, +0.8756, +0.7714**) and the "true 1.039 at rho = 0.95" are computed against `analytic_n_eff`, a different column of the same CSV. Each number is correct on its own; together they read as one series and are not. Proposed replacement for the signed-error bullet, using the eigen column throughout so it matches the MAE above it:

> - The error is not noise around the truth, it is COMPRESSION toward the middle of the range: signed error against `analytic_n_eff_eigen` runs **-1.0049**, **-0.6443**, **-0.2587**, **+0.1117**, **+0.4520**, **+0.6604** at rho = 0.0, 0.2, 0.4, 0.6, 0.8, 0.95. The estimator overstates dependence when views are near-independent and understates it when they are strongly dependent, crossing between rho = 0.4 and rho = 0.6. (Against the other analytic column, `analytic_n_eff`, the same estimates give -1.0049, +0.2557, +0.7232, +0.8831, +0.8756, +0.7714 and the crossing falls between rho = 0.0 and rho = 0.2. The two columns must not be mixed in one series.) Evidence: `results/part36_synthetic/analysis_a.json`; `experiments/part36_synthetic/README.md` section 6.1.

### L2 - proposed **OPEN**, with its scope narrowed

Part 36 measured residual per-pair dependence against view count on the analytic-rho generator and found it FALLS, not accumulates. That does not overturn L2, which is about a different harness, but it does show L2's title claims more than L2's evidence supports. L2's untested causal link is untouched, so the limitation stays OPEN.

> **L2 (new evidence, Part 36; still OPEN).** Residual per-pair dependence at rho = 0.0 on the analytic-rho generator, where the true per-pair dependence is 0 and the added views are INDEPENDENT (5 seeds):
> - V = 4: chance-corrected **0.3345 +/- 0.0163**, uncorrected **0.0844 +/- 0.0336**
> - V = 6: chance-corrected **0.3139 +/- 0.0033**, uncorrected **0.0590 +/- 0.0050**
> - V = 8: chance-corrected **0.3071 +/- 0.0053**, uncorrected **0.0511 +/- 0.0106**
> - V = 16: chance-corrected **0.2732 +/- 0.0062**, uncorrected **0.0271 +/- 0.0068**
> - Pre-registered rule: ACCUMULATES only if the mean is monotonically non-decreasing in V and the V = 16 value exceeds the V = 4 value by more than the larger std. Monotone: no; gap **-0.0613** against **0.0163**. **Verdict: DOES NOT ACCUMULATE.**
> - **The reading, which narrows L2 rather than lifting it.** The residual falls with view count under both the corrected and the uncorrected estimator, so it is not an intrinsic property of the estimator that accumulates with width. L2's accumulation is specific to `clone_eigen_softcluster`, where the added views are CLONES: the title "Residual per-pair dependence accumulates with view count" should be scoped to that harness. The causal link to the confidence drift at rho = 0.0 remains UNTESTED - it needs the `clone_eigen_softcluster` harness and was declared out of scope before the run. Evidence: `results/part36_synthetic/analysis_a.json`; `experiments/part36_synthetic/README.md` section 2.

### L10 - proposed **OPEN**, sharpened

L10's closing condition is "a batch-size sweep holding rho fixed, to separate the sample-count component from the estimator's structural floor". The sweep was run (varying the ESTIMATION sample count, a deviation stated in `config.yaml` before the run), and the pre-registered rule returns STRUCTURAL FLOOR in every cell. It is proposed OPEN rather than CLOSED because a post-hoc check shows these points cannot distinguish a true floor from a slower-than-1/n decay - reporting it CLOSED would claim a separation the data does not deliver.

> **L10 (new evidence, Part 36; still OPEN).** Nested estimation subsamples of one trained model's test split per seed, n in [64, 128, 256, 512, 1024, 2048], rho fixed at 0.3, 5 seeds, paired within seed. Signed ENIV bias against `analytic_n_eff_eigen`:
> - `V4_dependent` (truth 3.10): bias **-1.2548 +/- 0.0781** at n = 64, **-0.2581 +/- 0.0232** at n = 2048 (a 4.9x reduction); fitted floor **0.2736 +/- 0.0348**, fitted c **65.1 +/- 6.5**, fit residual rms 0.0404; **STRUCTURAL FLOOR**
> - `V4_independent_null` (truth 4.00): bias **-2.1469 +/- 0.0454** at n = 64, **-0.7493 +/- 0.0352** at n = 2048 (a 2.9x reduction); fitted floor **0.7892 +/- 0.0331**, fitted c **91.3 +/- 2.2**, fit residual rms 0.0758; **STRUCTURAL FLOOR**
> - `V6_dependent` (truth 4.50): bias **-1.9311 +/- 0.1629** at n = 64, **-0.2589 +/- 0.0734** at n = 2048 (a 7.5x reduction); fitted floor **0.3220 +/- 0.0599**, fitted c **108.8 +/- 9.0**, fit residual rms 0.0994; **STRUCTURAL FLOOR**
> - `V6_independent_null` (truth 6.00): bias **-3.4630 +/- 0.1397** at n = 64, **-1.0979 +/- 0.0270** at n = 2048 (a 3.2x reduction); fitted floor **1.2255 +/- 0.0271**, fitted c **153.0 +/- 9.6**, fit residual rms 0.1671; **STRUCTURAL FLOOR**
> - **Both components are present.** The sample-count component is large and real: the bias shrinks by 2.9x to 7.5x between n = 64 and n = 2048. It does not reach zero: at n = 2048 the bias is still many across-seed stds from 0 in all 4 cells. That part is a measurement, not a fit.
> - **What the sweep cannot separate, and this is the sharpening.** The residuals of the pre-registered `floor + c / n` fit are sign-ordered rather than scattered, with the identical pattern in all 4 cells, and a floor-free power law `k * n^alpha` - one parameter fewer - fits the same 6 points comparably (better in 2 of 4 cells), with exponents **-0.459** (`V4_dependent`), **-0.308** (`V4_independent_null`), **-0.585** (`V6_dependent`), **-0.338** (`V6_independent_null`) - all far short of the -1 the pre-registered model assumes. So the fitted floor is an UPPER BOUND on the structural component over this n range, not an identified limit, and L10's "structural floor" is supported as "still far from zero at n = 2048" rather than as "non-zero in the limit". Closing it needs a sweep reaching an n where the decay visibly flattens. Evidence: `results/part36_synthetic/analysis_b.json`; `experiments/part36_synthetic/README.md` sections 3.3 and 3.4.

### L11 - proposed **CLOSED**

L11's own rule decides, and it was run at the 4x L11 asks for. Every epsilon's tail AUC stayed inside the committed band, so the negative result of Part 13 is substantive rather than underpowered. Proposed CLOSED as confirmed: the limitation's operative warning about n_tail near the floor is unchanged, but its outstanding question is answered.

> **L11 (update, Part 36).** Part 13's detection arm was re-run at `n_samples = 32000` against the committed 8000 - a test slice of 4800 rows against 1200 - with every other committed setting unchanged, 5 seeds. The committed tail AUCs reproduce first (0.4846145138888889 at eps 0.2 and 0.5856175694444444 at eps 2.0, both to abs diff 0).
> - eps 0.2: tail AUC **0.4938 +/- 0.0039** at 4x against **0.4846 +/- 0.0022** committed; n_tail at q = 0.95 **240.0 +/- 0.0**; std/mean **0.008**
> - eps 0.5: tail AUC **0.5007 +/- 0.0042** at 4x against **0.4801 +/- 0.0063** committed; n_tail at q = 0.95 **240.0 +/- 0.0**; std/mean **0.008**
> - eps 1.0: tail AUC **0.5554 +/- 0.0087** at 4x against **0.5370 +/- 0.0132** committed; n_tail at q = 0.95 **240.0 +/- 0.0**; std/mean **0.016**
> - eps 2.0: tail AUC **0.5568 +/- 0.0129** at 4x against **0.5856 +/- 0.0126** committed; n_tail at q = 0.95 **240.0 +/- 0.0**; std/mean **0.023**
> - Primary band (the committed values) [0.4801, 0.5856]: **SUBSTANTIVE**. Band as quoted in L11's text [0.4846, 0.5856]: **SUBSTANTIVE**.
> - **A correction to L11's own text.** L11 quotes the Part 13 span as "0.4846 +/- 0.0022 to 0.5856 +/- 0.0126", but the committed means in `results/tail/tail.json` are 0.4846 (eps 0.2), 0.4801 (eps 0.5), 0.5370 (eps 1.0) and 0.5856 (eps 2.0). The quoted floor of 0.4846 excludes the eps 0.5 point, so the span should read **0.4801 to 0.5856**. Part 36 pre-registered and evaluated both bands rather than picking one.
> - **The dispersion L11 names as the real problem is unchanged by the 4x.** Across-seed std as a fraction of the mean is still 0.008 to 0.023 at four times the split. Quadrupling buys exceedances, not agreement between seeds. Evidence: `results/part36_synthetic/analysis_c.json`; `experiments/part36_synthetic/README.md` section 4.

### L22 - proposed **CLOSED** as confirmed

L22 asks for "a clone condition whose clean reference has the same view count as the attacked model". That condition now exists and the figure is measured. The result confirms L22's direction of concern rather than lifting it, and L22's operative warning - that 0.8743 must not be differenced against a fixed-width result - is unchanged, so it closes as confirmed with a comparable number attached.

> **L22 (update, Part 36).** The same width-6 model scored two conditions, so both maximise their suspicion score over the same 30 ordered off-diagonal entries: ATTACKED (the 4-view dataset with 2 duplicated views appended, exactly as `clone_k2` does) and CLEAN-AT-WIDTH-6 (the 6-view dataset the model was trained on, uncorrupted). Committed Part 10 settings throughout, 5 seeds, paired within seed. The committed mixed-width figure reproduces bit-for-bit first (0.8743088888888889 +/- 0.05831081226634124, abs diff 0).
> - `clone_k2` against the committed 4-view reference (MIXED width): **0.8743 +/- 0.0583**
> - `clone_k2` against a clean 6-view reference (EQUAL width): **0.7652 +/- 0.0843**
> - Paired difference, equal minus mixed: **-0.1091 +/- 0.0443**, negative in 5 of 5 seeds, magnitude 2.5x its own std. **Verdict: WIDTH EXPLAINS PART OF THE COMMITTED FIGURE.**
> - Mean suspicion score: clean at width 4 **0.3590 +/- 0.0394**, clean at width 6 **0.4280 +/- 0.0337**, cloned at width 6 **0.5243 +/- 0.0203**. The width change alone lifts the clean reference by +0.0690, which is where the AUC difference comes from.
> - **The reading.** L22 is right that part of 0.8743 is the width: 0.1091 of it, or 29% of the committed figure's margin over chance. The rest is not. At equal width the detector still separates the cloned condition from a clean 6-view reference at **0.7652 +/- 0.0843**, so duplication is detected and not merely counted. **0.7652 is the number that may be differenced against a fixed-width result; 0.8743 still may not.** Evidence: `results/part36_synthetic/analysis_d.json`; `experiments/part36_synthetic/README.md` section 5.

<!-- PART36:END -->

---

## Part 38 proposals (L24 measured; L27 and L28 pending the sweep)

**APPLIED 2026-11-22 to `docs/KNOWN_LIMITATIONS.md`.** L24, L27 and L28 each
carry a dated Part 38 block and all three remain **OPEN**; no entry was marked
RESOLVED or CLOSED. L27's and L28's placeholder text below was superseded by
the measured values at 40/40 before being applied — read the entries in
`KNOWN_LIMITATIONS.md`, not the placeholders here.

**Gate provenance, stated because `PRE_REGISTRATION.md` §7.6 requires it.**
§7.6 conditions a status change on "the full sweep and a review gate". The
review gate is the user's explicit instruction of 2026-11-22 to apply these
proposals. **The sweep is complete for mode `default` (B) at 40 of 40 units
but NOT for mode A**, which §8 of the same file describes as part of the
planned sweep ("40 units x 2 modes"), so "the full sweep" in §7.6's sense is
satisfied only for mode B. L24's and L27's evidence does not depend on mode A
— L24 is three seed-free scalar passes and L27's H2 is a mode-B leave-host-out
measurement — and L28, the one entry that does depend on it, was left
UNCHANGED for exactly that reason.

**Original proposal header — not applied until the review gate per
`PRE_REGISTRATION.md` §7.6**, which states that this programme "does not mark L24, L27 or L28
closed, resolved or narrowed. Status changes require the full sweep and a
review gate."

Written 2026-11-21, while the mode `default` (B) sweep was still running. Only
**L24** has its evidence in: its three statistics are seed-free scalars over
committed tables and do not depend on the sweep. L27 and L28 are placeholders
below, with their fill rules stated so the text cannot be shaped by the
numbers when they land.

### What changed in Part 38

Three things, none of which was available to Parts 30 or 35.

**1. A sample whose controls are selected without reference to `auth`.** This
is L24's written closing condition. A cell enters as a control iff (a) it is
occupied in at least one of `proc`, `flows`, `dns` — the `auth` stream is
never consulted — and (b) it is on a host with at least one attack cell in the
read range, or else `crc32(computer|window) % 512 == 0`. `auth` features are
attached in a later pass. The step-2 sampler audit is the evidence that this
is what ran: nine of nine exact against Part 30's committed
`population.per_view.<view>` sums, records scanned matching to the row, two
independent 11 GB scans agreeing field for field, and
`tests/unit/test_lanl_v2_sampler_synthetic.py` comparing `select_cells`
against a brute-force re-derivation of the rule cell by cell
(`results/lanl_v2/SWEEP_EXECUTION_LOG.md` section 1).

**2. The size fallback fired at `k = 2`, so the realised attack-host benign
share is `1/4`, not `1/1`.** `results/lanl_v2/build_v2_days16.json`,
`size_fallback`: `fallback_fired` true, `attack_host_benign_share_k` 2,
`attack_host_benign_share` `"1/4"`, `rows_before_fallback` 979122,
`rows_after_fallback` 281824, `attack_host_benign_cells_before` 929764,
`attack_host_benign_cells_dropped` 697298, against `max_total_bar` 400000.
The rule and the fallback were both fixed before any count was seen, so this
is mechanical rather than a choice — but it means **697,298 of 929,764
attack-host benign cells were dropped**, and every L27 statement must read
"1/4".

**3. Three seed-free scalar passes, each bit-exact on re-run and unchanged
under `OMP_NUM_THREADS=1`**, covered by `tests/unit/test_h1_scalar_passes.py`
(12 tests): `results/lanl_v2/h1_auth_presence.json`,
`h1_occupancy_decomposition.json`, `h1_conditional_rates.json`. The last two
are independent implementations of one quantity — for a binary indicator
`AUC = 0.5 + |attack_rate - benign_rate| / 2` — and they agree to `0.00e+00`
on all four views, with the same algebra on Part 30's rates returning L24's
committed 0.6636552214.

**A separate finding that belongs to L27's rule, recorded here so it is not
lost:** `fallback_survivor_k2` (`crc32 % 4 == 0`) reaches test AUC
**0.8670731707**, higher than any view occupancy
(`h1_occupancy_decomposition.json`,
`whole_test_block.fallback_survivor_k2.test_auc`). **It is a prevalence
artefact of the `k = 2` thinning and not a signal any model can use** —
`crc32` is not a feature and no encoder sees it. Attack cells were kept
unconditionally while attack-host benign cells were thinned by that predicate,
so the surviving benign rows satisfy a condition the attack rows mostly fail.
It says the class balance on attack hosts was manufactured by a hash; it does
not say the model has a leak.

### L24 — proposed **OPEN**, narrowed, with its selection claim relocated

L24's closing condition is met by the design above. The measurement says the
condition was necessary but not sufficient, which is a narrowing and not a
close.

> **L24 (update, Part 38).** A sample whose control cells are drawn without
> any reference to the `auth` stream now exists, and it was measured. Three
> seed-free scalar passes over the committed sample table, no training and no
> inference; each re-runs bit-exact and is unchanged under
> `OMP_NUM_THREADS=1`.
>
> - **The auth-presence baseline did NOT fall.** Pre-registered H1 required a
>   drop of at least **0.05** from Part 30's **0.6636552213548774**, i.e. a
>   test AUC at or below **0.6136552213548774**. Measured:
>   **0.6611838191552647**, a drop of **0.0024714021996127267** — a fiftieth
>   of the bar. **H1 is DISCONFIRMED**
>   (`results/lanl_v2/h1_auth_presence.json`, `h1.verdict`,
>   `h1.drop_achieved`). The baseline is seed-free, so this is a single value
>   and no +/- applies.
> - **L24's contingency survived intact.** Every attack cell still holds an
>   `auth` record — attack rate **exactly 1.000000** — on a sample whose
>   selection never consulted the stream, and the benign rate moved only
>   **0.672690 to 0.677632** (`results/lanl_v2/h1_conditional_rates.json`,
>   `per_view_rate_comparison.auth`). Part 30's own 410/410 and 12,323/18,319
>   are recomputed from `results/lanl/samples.npz` and reproduced exactly as a
>   gate before the comparison is made (`l24_reproduction_gate.status`,
>   `PASSED`); the 12,323 figure is prose in L24 and is not a field in
>   `results/lanl/results.json`. **So the auth-presence separation is a
>   property of the data, not an artefact of how Part 30 drew its controls.**
>   L24's first bullet — the labels come from the same stream as a view, so
>   `auth` is label-adjacent — is strengthened, not weakened.
> - **The selection did not drop; it moved to `proc`.** Every selection input
>   scored as its own trivial baseline on the test block, train-only
>   orientation (`results/lanl_v2/h1_occupancy_decomposition.json`,
>   `whole_test_block.<scalar>.test_auc`): `occupancy_proc`
>   **0.7083759667**, `auth_present_indicator` **0.6611838192**,
>   `n_non_auth_views_present` **0.6503477024**, `occupancy_flows`
>   **0.5253242118**, `occupancy_dns` **0.5000356930** (at chance). `proc` now
>   carries MORE of the separation than `auth` does. The conditional rates say
>   why and attribute it to rule (a): `proc`'s attack rate is unchanged at
>   **0.497561** while its benign rate jumped **0.697309 to 0.914313**,
>   because rule (a) admits a cell only if `proc`, `flows` or `dns` is
>   occupied. Its rate gap more than doubled, **-0.199748 to -0.416752**.
> - **Not the host-rule tautology.** Within attack hosts (n = 70,130 test
>   rows of 84,460; 302 attack hosts in the read range) the picture is
>   unchanged: `occupancy_proc` **0.7143004072**, `occupancy_auth`
>   **0.6524885255** (`within_attack_hosts.<scalar>.test_auc`). That is the
>   within-attack-host contrast `PRE_REGISTRATION.md` section 2 names as the
>   figure that matters. `attack_host_indicator` is reported but excluded as
>   tautological — it is 1 on every attack row by construction and is constant
>   within hosts. `identity_hash_kept` is a consistency check that passed:
>   **0.5883878644** whole-block but **0.5037865749** within hosts, so its
>   whole-block value came entirely from the host split and within hosts it
>   sits at chance, as a hash must.
> - **The volume signals DID fall**, which is the evidence the rule changed
>   something: `total_records_across_views` **0.6842788175 to 0.5862805386**
>   and `n_views_present` **0.5624544156 to 0.4800433975**, the latter below
>   chance (`h1_auth_presence.json`, `baselines.<name>`).
>
> **What this means.** L24's closing condition is satisfied and the result is
> not the one the condition anticipated. Removing `auth` from selection
> removed the *volume* signals and left `auth`'s own contingency untouched,
> while installing a stronger `proc`-occupancy selection effect in its place.
> **Rule (a) has to gate on something, and whatever it gates on acquires the
> benign-side enrichment `proc` now carries.** So "sample selection survives
> the view drop" is still true of Part 38, for a different view. The two
> sample tables are not comparable row-for-row (`PRE_REGISTRATION.md` section
> 2), so **0.6612 against 0.6637 is not a measured effect size** — only the
> pass/fail against the pre-registered bar is a result.
>
> **What would change our understanding.** A selection rule whose admission
> criterion is independent of every view's occupancy — which on this data may
> not exist, since a cell with no occupied view carries no features — or a
> decomposition that separates `proc`'s selection-induced enrichment from any
> behavioural signal `proc` carries. The latter is the open question: `proc`
> occupancy is visible to the encoders through the view mask, unlike
> `fallback_survivor_k2`, so it cannot be dismissed as unusable.

**Proposed status line:** `L24` stays **OPEN**. Not resolved and not closed:
its closing condition is met, H1 is disconfirmed, and the selection effect is
relocated rather than removed.

### L27 — placeholder, fills from Table 3 when the sweep lands

**Do not write this section from a partial sweep.** The fill rule, fixed now:

- The barred quantity is the **`C17693`** column only, over the **eight**
  arm-and-system cells. H2 is confirmed iff all eight move by **less than
  0.036** in absolute value, and disconfirmed if any one moves by **0.036 or
  more** (`PRE_REGISTRATION.md` section 1, H2, verbatim). The other four hosts
  are not pre-registered for Part 38 and cannot confirm or disconfirm H2.
- Source when it exists: `results/lanl_v2/filled_tables.json`,
  `table3_leave_host_out.per_cell.<arm>/<system>.per_host.C17693`, produced by
  `experiments/lanl_v2/fill_analysis.py` behind a 40-unit gate.
- **Two caveats must appear beside whatever the number is.** First, the
  realised share is **1/4**, so a disconfirmation shows that restoring a
  quarter of the benign contrast did not help — narrower than H2's mechanism.
  Second, H2's 0.036 bar sits *inside* Part 30's own committed band: the eight
  `C17693` deltas run **-0.035962653276654266** to
  **-0.051874860066125961** (`results/lanl_reanalysis/results.json`,
  `step5_l27.per_arm.<arm>.<system>.per_host.C17693.change_from_all_test`),
  and the smallest is **below** the bar. A cell reproducing Part 30's weakest
  cell would be scored as passing, so "H2 confirmed" would not by itself mean
  the sensitivity shrank. Report all eight Part 38 deltas against all eight
  Part 30 deltas pairwise.
- One host is not a resample, so **no interval attaches** and the result is
  descriptive — Part 30's own `step5_l27.interval` records the same.

**Proposed status line: pending.** L27 stays **OPEN** until Table 3 is filled
at 40/40.

### L28 — placeholder, and its literal closing condition cannot be met under this design

**This needs saying regardless of what mode A eventually shows.** L28's
written closing condition is "Seeds run across at least two execution modes,
so the reported std covers mode as well as seed" — a **pooled** standard
deviation. `PRE_REGISTRATION.md` section 6 forbids exactly that: "The two
modes are **never pooled into a single +/-**: L28's finding is that a 5/5 seed
count inside one run shares one execution mode and is not independent evidence
about mode effects." The pre-registration substitutes **H3**, a sign-agreement
test on `roc_auc`, and that is the only barred L28 analysis Part 38 can
produce.

So **L28 cannot be CLOSED by its own text here.** Satisfying it would need a
review-gate decision that a mode-pooled +/- is legitimate, and a second dated
amendment. That is not a judgement this document can make.

Fill rule when mode A runs (~2.0 h, ~5.4 GB across 11 processes, needs an idle
box):

- H3 is confirmed iff the sign of the PrismFlow-minus-naive difference in test
  ROC AUC agrees across modes A and B in all four arms; disconfirmed if it
  flips in any arm. A sign resting on a difference smaller than the per-seed
  std is **sign undetermined**, not agreement. A cell with
  `seeds_exactly_zero > 0` is reported with that count, because an exact zero
  has no sign.
- The other five metrics are reported without a bar, which is where Part 30's
  disagreement lives (`balanced_accuracy`, `attack_recall` and `brier`
  disagreeing for `with_auth`; `results/lanl/execution_mode.json`).
- **Any magnitude bar must be pre-registered before mode A runs and justified
  against L28's own figure.** A "within 3 standard deviations" bar would be
  *already failed by the prior record*: L28 reports the same seed differing by
  **0.0140** in AUC across modes (naive seed 0: 0.8375 and 0.8235), which is
  "about **4** per-seed standard deviations of that same arm and system
  (**0.0033**)".
- Every ENIV figure carries its thread count. Mode B observed **8** threads,
  mode A sets **1**, and a committed `eniv_eigen` of **3.4396320614332856** at
  the machine default is missed by **0.0107** at one thread
  (**3.4289614712229826**, Part 34).

**Proposed status line: pending.** L28 stays **OPEN**, and is **NARROWED at
best** even on a clean H3 confirmation.

<!-- PART38:END -->
