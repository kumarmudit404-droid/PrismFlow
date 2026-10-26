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
