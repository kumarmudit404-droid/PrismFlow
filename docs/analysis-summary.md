# PrismFlow -- what the committed results actually show

Generated from committed files only. No number below was estimated, interpolated,
or carried over from memory of a prior session -- every figure cites the file and
field it came from. Where a requested breakdown does not exist in these files, it
is marked **not measured**, not filled in with a plausible guess.

## Methodology -- files read

- `data/v2/evaluation_queries.json` (48-row V2 dataset)
- `results/v2/part24_pipeline_verification.json` (48-row single-pass pipeline run)
- `results/v2/part24_pipeline_verification_pass1.json` (the FIRST single-pass run, kept
  as a second file; retrieval is identical to the run above cell for cell, but its
  claim counts are not -- it has 0 reasoner errors against the other run's 32, and
  it is the pass in which row 032 reached two angles and produced an ENIV)
- `results/chorus/attack_metrics.json` + `results/chorus/metadata.json` (Part 09 Chorus/PGD attack experiment, V1)
- `results/calibration_duplicated/summary.md` (V1 calibration under view duplication)
- `results/clone_eigen/summary.md`, `results/clone_eigen_perview/summary.md`, `results/clone_eigen_softcluster/summary.md` (V1 clone/eigen-ENIV experiments, three dependence-estimator variants)
- `docs/v2-known-limitations.md` (V2-L1 through V2-L7)

`results/v2/part24_reverify_n6.json` was also opened to confirm it is the same
single-pass design at N=6, not additional seeded evidence.

**Important finding about the source files themselves:** neither
`part24_pipeline_verification.json` nor its `_pass1` twin is Part 24 evidence.
Both carry the field `"is_part24_evidence": false` and `"passes": 1, "seeds":
null` in the file itself, with the label *"PIPELINE-VERIFICATION RUN -- NOT
PART 24 EVIDENCE... no seed loop: nothing on this path varies per seed."* This
directly matches `docs/v2-known-limitations.md` V2-L5: *"The free Groq tier
cannot fund a 5-seed evaluation."* There is no committed multi-seed V2 pipeline
run. All V2 pipeline numbers below are single-pass and are labelled
**NOT A PART 24 FINDING** throughout.

The actual 5-seed, mean +/- std evidence in this repo is all **V1** (the
frozen `app.py` / `prismflow/` system), from the Chorus/PGD attack experiment
and the calibration/clone-eigen experiments.

---

## 1. V1 metrics -- mean +/- std, n_seeds

All three families below report `seeds: [0, 1, 2, 3, 4]`, i.e. n_seeds = 5,
satisfying the project's 5-seed rule.

### Chorus/PGD attack experiment (`results/chorus/attack_metrics.json`, `metadata.json`)
Part 09: k compromised views perturbed jointly (Chorus) so they agree on a wrong
class, vs. independent PGD as the literature baseline. "Controlled research
simulation on this project's own models and synthetic data" (`metadata.json`
-> `config.experiment.description`, `note`). `success_rate` = attack success
rate (higher is worse for the defended system).

| condition | naive success_rate | prismflow success_rate | n_seeds (both) |
|---|---|---|---|
| clean | 0.000 +/- 0.000 | 0.000 +/- 0.000 | 5 |
| chorus_k1 | 0.222 +/- 0.086 | 0.253 +/- 0.091 | 5 |
| chorus_k2_beta0 | 0.370 +/- 0.099 | 0.389 +/- 0.105 | 5 |
| chorus_k2_beta5 | 0.353 +/- 0.120 | 0.396 +/- 0.103 | 5 |
| chorus_k2_eps0.2 | 0.042 +/- 0.021 | 0.043 +/- 0.019 | 5 |
| chorus_k2_eps0.5 | 0.216 +/- 0.074 | 0.235 +/- 0.065 | 5 |
| chorus_k2_eps1.0 | 0.455 +/- 0.116 | 0.481 +/- 0.113 | 5 |
| chorus_k2_eps2.0 | 0.597 +/- 0.128 | 0.613 +/- 0.129 | 5 |
| chorus_k3 | 0.559 +/- 0.134 | 0.581 +/- 0.110 | 5 |
| pgd_k2_eps0.2 | 0.037 +/- 0.019 | 0.038 +/- 0.022 | 5 |
| pgd_k2_eps0.5 | 0.185 +/- 0.063 | 0.201 +/- 0.057 | 5 |
| pgd_k2_eps1.0 | 0.369 +/- 0.098 | 0.387 +/- 0.106 | 5 |
| pgd_k2_eps2.0 | 0.464 +/- 0.111 | 0.491 +/- 0.112 | 5 |

(`results/chorus/attack_metrics.json` -> `summary["<cond>|naive"].success_rate`,
`summary["<cond>|prismflow"].success_rate`, each with its own `n_seeds` and `std` field.)

### Calibration under view duplication (`results/calibration_duplicated/summary.md`)
k = number of duplicated copies of view 0, on top of a 4-view base (rho = 0.3).

| k | system | accuracy | prob_ece | brier |
|---|---|---|---|---|
| 0 | naive | 0.8300 +/- 0.0791 | 0.0579 +/- 0.0094 | 0.2511 +/- 0.1019 |
| 0 | prismflow | 0.8300 +/- 0.0797 | 0.0637 +/- 0.0137 | 0.2512 +/- 0.0990 |
| 2 | naive | 0.8100 +/- 0.0783 | 0.0608 +/- 0.0146 | 0.2697 +/- 0.1037 |
| 2 | prismflow | 0.8260 +/- 0.0765 | 0.0602 +/- 0.0083 | 0.2588 +/- 0.0992 |
| 4 | naive | 0.8073 +/- 0.0854 | 0.0715 +/- 0.0199 | 0.2831 +/- 0.1109 |
| 4 | prismflow | 0.8227 +/- 0.0789 | 0.0664 +/- 0.0188 | 0.2633 +/- 0.0977 |

(all cells: `results/calibration_duplicated/summary.md`, tables "k = 0/2/4", n=5 seeds each.)

### Clone/eigen-ENIV experiments (three dependence-estimator variants)
`results/clone_eigen/summary.md`, `results/clone_eigen_perview/summary.md`,
`results/clone_eigen_softcluster/summary.md`. All: seeds [0,1,2,3,4], rho in {0.0, 0.5}, k = 0..4 duplicated views.
Representative row (rho=0.0, k=2, 6 views), accuracy / confidence:

| variant | naive acc | prismflow acc | naive conf | prismflow conf |
|---|---|---|---|---|
| clone_eigen | 0.745 +/- 0.070 | 0.751 +/- 0.063 | 0.677 +/- 0.057 | 0.602 +/- 0.049 |
| clone_eigen_perview | 0.745 +/- 0.070 | 0.749 +/- 0.066 | 0.677 +/- 0.057 | 0.585 +/- 0.051 |
| clone_eigen_softcluster | 0.745 +/- 0.070 | 0.748 +/- 0.072 | 0.677 +/- 0.057 | 0.641 +/- 0.052 |

Full per-k, per-rho tables for all three variants are in the respective `summary.md`
files; not reproduced in full here for length.

---

## 2. Every condition where PrismFlow did worse than naive

Using `success_rate` (fraction of attacks that succeeded) from
`results/chorus/attack_metrics.json`: **PrismFlow's success_rate is higher
than naive's in every single non-clean condition** -- all 12 of the 13
committed conditions besides `clean` (where both are 0.000). This holds for
every row in the table in section 1, each backed by 5 seeds.

No other worse-than-naive comparison is defined by the committed files for a
single scalar "did worse" metric outside of `success_rate` -- calibration and
clone-eigen tables show mixed results per metric (e.g. prismflow's
`brier_resolution` is sometimes lower than naive's, e.g. k=0:
0.4410 +/- 0.0957 vs naive 0.4442 +/- 0.0957 in `calibration_duplicated`,
k=0 table), but these are not framed as a single pass/fail comparison in the
source files, so no aggregate "worse than naive" count is reported for them.
**Per-metric worse/better calls beyond `success_rate` and the two tables
above: not measured** (would require re-deriving a judgment the source files
do not themselves make).

---

## 3. V2 dataset composition

`data/v2/evaluation_queries.json` -- 48 rows total. Per-row fields:
`id, idea_pitch, domain, actual_outcome, outcome_date, ground_truth_source,
conflict_expected, notes`.

By `domain`:

| domain | count |
|---|---|
| Tech-OSS | 28 |
| Startup | 10 |
| Financial-Product | 10 |

**No `angle` field exists on the dataset rows** -- angle assignment happens at
pipeline time, not in the dataset file. Composition by angle: **not measured**
(not a field in this file).

---

## 4. Per-angle retrieval coverage

From `results/v2/part24_pipeline_verification.json` (48 rows). **This file is
explicitly NOT A PART 24 FINDING** (`is_part24_evidence: false`, `passes: 1`,
`seeds: null`) -- single pass, GROQ `openai/gpt-oss-120b` via the OpenAIReasoner
code path, LexicalAdjudicator fusion. Reported here as a pipeline-behavior
snapshot, not as evidence:

| angle | rows with >=1 retrieved record | rows with >=1 claim |
|---|---|---|
| tech | 48 / 48 | 19 / 48 |
| market | 6 / 48 | 1 / 48 |
| sentiment | 1 / 48 | 0 / 48 |
| financial | 0 / 48 | 0 / 48 |
| regulatory | 0 / 48 | 0 / 48 |

This matches `docs/v2-known-limitations.md` V2-L1 ("Retrieval coverage: one
angle of five contributes") and V2-L4 (financial angle structurally
incompatible with this dataset by design conflict). Regulatory has zero
records because, per every row's own warning text, "no connector is
configured for the regulatory angle (slots tried: primary, secondary,
tertiary)" -- not a retrieval failure, an absent connector.

---

## 5. Zero-claim rows, split by cause

Same single-pass, NOT-A-PART-24-FINDING file (`results/v2/part24_pipeline_verification.json`),
`zero_claim_cause` field per row, 48 rows total:

| cause | rows | row IDs |
|---|---|---|
| had >=1 claim (not a zero-claim row) | 20 | (see file; not zero-claim) |
| `provider_error` | 26 | 006, 024, 026, 027, 028, 029, 030, 031, 032, 033, 034, 035, 036, 037, 038, 039, 041, 042, 043, 044, 045, 046, 047, 048, 049, 050 |
| `genuine` (evidence retrieved, reasoner ran, no claims extracted) | 2 | 017, 020 |

`provider_error` rows are dominated by Groq rate-limit (429) and malformed-JSON
(400) reasoner errors recorded per-angle in each row's `reasoner_error` field
(e.g. row 006: tech angle "BadRequestError 400: ...Failed to generate JSON";
row 024: tech angle "RateLimitError 429: ...Rate limit reached..."). No
`connector_error` or other named cause category appears in this file's
`zero_claim_cause` values -- only `null` (has claims), "provider_error", and
"genuine" occur.

---

## Not measured (explicitly requested, not present in committed files)

- Per-angle composition of the V2 dataset (no `angle` field on dataset rows).
- A multi-seed (n>=5), Part-24-qualifying V2 pipeline run -- none is committed;
  `docs/v2-known-limitations.md` V2-L5 states the free Groq tier cannot fund one.
- ECE / Brier score for the V2 pipeline -- V2-L6 states these cannot yet be
  computed as a direct consequence of V2-L5.
- A single unified "worse than naive" verdict across every calibration/clone-eigen
  metric (the source files report mixed per-metric results, not a pass/fail call).
