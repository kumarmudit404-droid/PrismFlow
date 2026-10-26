---
description: PrismFlow Part 34 - engineering defects (Part 33b fixes, L25, L32, harness timeout, comment fix)
---

PART 34 - ENGINEERING DEFECTS
=============================

READ FIRST: .claude/commands/limitations-index.md, docs/CONTRACT.md, docs/KNOWN_LIMITATIONS.md
(L22, L25, L27, L28, L31, L32), docs/part32-live-service.md. Use the prismflow-honest-reporting skill.

STANDING RULES: branch part34-defects from main; commit per step; do not push; do not deploy;
no secrets anywhere. Frozen modules (prismflow/models, statistics, eniv, evaluation, attacks, v2/)
are import-only. Earlier Parts' code is NOT modified except the edits authorised BY NAME in
steps 1, 4 and 5 below. docs/KNOWN_LIMITATIONS.md is not edited; proposals go to
docs/proposed_limitations_update.md.

STEP 0. Commit the new command files (.claude/commands/limitations-index.md, part34.md to
part39.md) unchanged.

STEP 1 - Part 33b fixes (AUTHORISED edit to web/build_data.py).
 a. Claim-table detail text names the system and subset. AIT detection cell: the figures are
    PrismFlow's, natural_windows.dnsteal_only attack windows against benign windows. LANL
    detection cell: the four per-arm figures are PrismFlow's; add naive's with_auth and
    without_flows values read from results/lanl/results.json. No verdict word or rule changes.
    web/data/real_summary.json may change only in those two detail strings and generated_utc.
 b. docs/FINAL_REPORT.md section 3 preamble: replace "Every row is 5 seeds" with what is true:
    3.1's correlation is one pooled value with no recorded seed count; 3.3's detector AUCs are
    5 seeds per results/comparison/detector.json; the other rows are 5 seeds. Cite the file.
 c. Section 4.3: cite results/lanl/results.json (phase4_closeout.drop_top_test_computer) for
    the 0.036-0.052 range. Section 8 and the Docker row of section 10: point to L31 and L32.
 d. docs/check_report_numbers.py passes.

STEP 2 - L25. Create a new package prismflow/training/ with a dataset-agnostic fit_scaler,
apply_scaler and trainer whose view widths are PASSED IN by the caller (no lookup through
ait_lds.FEATURES or any source-name table). Do not modify experiments/ait_clean/run_ait_clean.py
or experiments/lanl/run_lanl.py. Tests in tests/unit/test_training_agnostic.py:
 - the scaler fitted on the LANL train block of results/lanl/samples.npz equals run_lanl.py's
   mirror bit for bit, and the AIT scaler equals Part 28's fit_scaler bit for bit;
 - one training unit (LANL with_auth naive seed 0) reproduces the committed test probabilities
   in results/lanl/scores/ at the thread count recorded in results/lanl/execution_mode.json.
   If it cannot, report the largest difference and the thread count used; do not loosen the test
   silently.

STEP 3 - L32. Run `docker info`. If the daemon answers: build the image from Dockerfile; run it
with ALLOWED_ORIGIN unset on a free local port; GET /health; POST /score with
web/data/sample_ait_views.npz; confirm every literal caveat field is present; stop and remove the
container. Record build time, image size and both responses in results/serve_docker/report.json
and a short section in docs/part32-live-service.md. If the build fails, record the error
(trimmed), make the smallest Dockerfile fix as its own commit, and retry once. Never push the
image anywhere. If the daemon does not answer, write "Docker daemon not running - start Docker
Desktop and re-run step 3" and leave L32 OPEN.

STEP 4 - harness timeout (AUTHORISED edit to web/verify_page.py only). Make the socket and
call deadlines configurable by environment variable PRISMFLOW_VERIFY_TIMEOUT (default 300 s).
Nothing else in the file changes. web/check_prism_contrast.py must then pass as committed.

STEP 5 - comment fix (AUTHORISED, comments and docstrings only). In
experiments/ait_clean/config.yaml and the run_ait_clean.py docstring, correct "5 windows on
russellmitchell and 3 on santos" to the strict counts, after confirming them from the Part 27
data card or results/ait_window. git diff must show no change to any code line.

STEP 6 - report only. Run tests/unit/test_ait_attack.py::test_clean_dependence_matches_part28
with OMP_NUM_THREADS=1 and with the default. Report both. Do not edit the test. Add the result
under L28 in docs/proposed_limitations_update.md.

STEP 7. Rebuild web/dist with web/build_dist.py and check it serves locally. It is git-ignored:
build only, do not commit.

GATE: pytest -q, web/check_palette.py, web/check_prism_contrast.py, web/verify_page.py real and
reveal, check_secrets.py --all, push scan. If anything matches the push scan, stop and report.
Write the proposed status of L25 and L32 (RESOLVED / OPEN, with the evidence) to
docs/proposed_limitations_update.md. If every gate passes: git checkout main, git merge --no-ff
part34-defects, re-run pytest -q and the push scan. Do not push.
End with "PART 34 MERGED, NOT PUSHED <hash>" or "PART 34 NOT MERGED: <reason>", then the list of
proposed status changes.
