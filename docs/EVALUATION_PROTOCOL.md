# Evaluation protocol

The project's claim is about honest confidence, not accuracy, so evaluation
measures honesty directly. Every experiment from Part 08 on reports its metrics
through one function:

```python
from prismflow.evaluation import evaluate

report = evaluate(model, loader, experiment_id, seeds)
```

Experiment scripts must not compute their own metrics. If an experiment needs a
metric that does not exist, add it to `compute_metrics` in
`prismflow/evaluation/protocol.py`, with a test, so every number in the project
comes from the same code.

## Calling it

| argument | meaning |
|---|---|
| `model` | one trained model per seed: `{seed: model}` or `lambda seed: model`. A single model is rejected. |
| `loader` | the batches to score, per seed: `{seed: iterable}` or `lambda seed: iterable`. Use a callable for one-shot iterators such as `iter_batches`. |
| `experiment_id` | output subdirectory of `results/`. Use `/` to separate conditions, e.g. `robustness/missing_0.3/prismflow`. One call is one system under one condition. |
| `seeds` | at least 5 (contract rule). |

Keyword options: `results_dir` (default `results`), `n_bins` (default 15),
`coverages` (default 1.0, 0.9, 0.8, 0.5), `title` for the diagram, and
`allow_fewer_seeds` for smoke tests only. Fewer than 5 seeds sets
`not_evidence: true` in both output files and marks the diagram title; such
outputs must never be reported.

The model is run in eval mode under `torch.no_grad()` and restored to its
previous mode afterwards. It must return a `PrismFlowOutput`, or any object with
`probs [B, K]`, `confidence [B]` and `eniv` (an `ENIVResult` or `None`).

## Outputs

`results/<experiment_id>/`

- `metrics.json`: `summary` (mean, sample std with ddof=1, and n_seeds for every
  metric), `per_seed` rows, per-seed `reliability_bins` for both confidence
  definitions, and the settings used (`n_bins`, `coverages`, seeds). NaN is
  written as `null`.
- `metrics.csv`: one row per seed, then a `mean` row and a `std` row.
- `reliability_diagram.png`: one panel per confidence definition. Each point is
  a bin's mean accuracy across seeds, with the std across the seeds that populate
  that bin. The strip below shows mean sample count per bin, so bins backed by
  few samples are visible as such.

Metrics are computed within each seed and then summarised. Samples are never
pooled across seeds, because each seed is a different trained model.

## Two confidence definitions

The model reports two different quantities that are both called confidence.
The protocol scores both and keeps them apart.

| prefix | quantity | what it is |
|---|---|---|
| `prob_` | `max_k probs[k]` | expected Dirichlet probability of the predicted class. A probability of being right, so it is what ECE is defined for. Always at least 1/K. |
| `vacuity_` | `1 - uncertainty` (`PrismFlowOutput.confidence`) | the evidential "how much evidence" signal the clone experiment tracks. Not a probability of being right: a fully vacuous opinion scores 0 and still predicts at chance, 1/K. |

`vacuity_ece` is therefore expected to be non-zero even for a well-behaved
model. It says how far evidence mass is from matching accuracy, which is a
meaningful question, but it is not the standard calibration error. A write-up
must name which definition a number uses.

## Metrics

| name | definition | better |
|---|---|---|
| `accuracy` | fraction of argmax predictions that are correct | higher |
| `macro_precision`, `macro_recall`, `macro_f1` | unweighted mean over all K classes. A never-predicted class scores precision 0, and it still counts in the mean. | higher |
| `{p}_ece` | sum over bins of (n_b / N) \|accuracy_b - mean confidence_b\| | lower |
| `{p}_mce` | largest \|accuracy_b - mean confidence_b\| over non-empty bins | lower |
| `{p}_aurc` | area under the risk-coverage curve, the mean selective risk over coverages 1/N ... N/N | lower |
| `{p}_selective_risk@c` | error rate on the ceil(c * N) most confident samples | lower |
| `{p}_mean_confidence` | mean of that confidence | n/a |
| `brier` | mean_i sum_k (p_ik - y_ik)^2, range [0, 2] | lower |
| `brier_reliability` | calibration term | lower |
| `brier_resolution` | how much outcome frequency differs between forecast bins | higher |
| `brier_uncertainty` | spread of the labels alone, the same for every model on the same data | n/a |
| `brier_within_bin` | binning remainder (see below) | n/a |
| `eniv`, `mean_dependence`, `efficiency_ratio` | batch-size-weighted means of the model's ENIV diagnostics; NaN when the model does not compute ENIV | reported, not scored |

**Bins.** 15 equal-width bins on [0, 1] by default. Each bin is closed on the
right, and 0 goes in the first bin. ECE, MCE, the Brier decomposition and the
diagram all share this convention.

**Ties in selective prediction.** When samples share a confidence value, their
order is arbitrary. Each tied group contributes its errors at the group's mean
rate, so risk is the expectation over random tie-breaking and never depends on
storage order. A model with one constant confidence gets AURC equal to its error
rate.

## The Brier decomposition

This table is the project's main calibration argument. A single Brier number
cannot show whether a difference comes from calibration or discrimination, so
the components are always reported separately:

    brier = reliability - resolution + uncertainty + within_bin

Murphy's three-term identity is exact only when all forecasts in a bin share
one value. Continuous model probabilities do not, and binning them leaves a
remainder: within-bin forecast variance minus twice the within-bin covariance
(Stephenson, Coelho & Jolliffe, 2008). That remainder is reported as
`brier_within_bin` rather than folded into one of the named components, so
reliability and resolution are never quietly adjusted to force a sum. It is
exactly 0 for forecasts that are constant within bins, and it is usually small
with 15 bins. If it is not small relative to the difference being argued, say so.

The multiclass score is decomposed one class at a time (one-vs-rest), and the
per-class terms are summed. The sum reconstructs the multiclass Brier score
exactly (tested).

**Reading it.** `brier_uncertainty` is identical for every system on the same
data. A difference in `brier` between two systems is split between reliability
(calibration) and resolution (discrimination). Report both, with std across
seeds. Do not claim a reliability improvement unless the error bars separate.
Also state any condition where resolution got worse.

### Current claim (revised 2026-09-17)

The original framing was "PrismFlow improves reliability, not resolution."
**The evidence so far does not support it and points the other way.** Until new
evidence changes this, the claim is:

> Under view duplication, PrismFlow preserves resolution, accuracy and AURC
> relative to naive fusion. It does not show improved Brier reliability or ECE.
> Its reliability trends slightly worse as copies are added.

Evidence (5 seeds, rho = 0.3, 4 base views, k copies of view 0;
`experiments/calibration/README_duplicated.md`, `results/calibration_duplicated/`):

- Without duplicates (k = 0), PrismFlow and naive fusion cannot be told apart
  on any metric (`experiments/calibration/README.md`).
- As copies are added, the gap in PrismFlow's favour grows within every seed.
  At k = 4 the growth is Brier -0.0200 +/- 0.0118 and AURC -0.0149 +/- 0.0091
  (5/5 seeds each). The Brier gain is entirely resolution: +0.0275 +/- 0.0166
  (5/5). Naive fusion's accuracy falls -0.0227 +/- 0.0095 (5/5 seeds lower),
  while PrismFlow's falls -0.0073 +/- 0.0128.
- ECE does not separate at any k. At k = 4, prismflow - naive is
  -0.0051 +/- 0.0261, lower in 2/5 seeds.
- Brier reliability does not separate either, and it leans against PrismFlow.
  At k = 4, prismflow - naive is +0.0041 +/- 0.0099 (lower in 1/5 seeds).
  PrismFlow's own reliability worsens from k = 0 to k = 4 by +0.0090 +/- 0.0061
  (5/5 seeds).

Confounds:

- **Naive fusion trained on the copies. Tested; does not explain the result.**
  A naive model trained at k = 0 and frozen, with copies appended only at
  inference (`experiments/calibration/README_frozen_naive.md`), loses about the
  same resolution (-0.0340 +/- 0.0283) and accuracy (-0.0213 +/- 0.0257) as
  naive fusion trained on the copies. Training on copies mainly absorbs
  confidence inflation: +0.018 against +0.057, frozen higher in 5/5 seeds.
  Against the frozen model, PrismFlow still separates on resolution and AURC
  (5/5 seeds). Reliability is still null (-0.0002 +/- 0.0116).
- **PrismFlow trained under the discount. Tested: the robustness comes from the
  mechanism, and training adds a level correction.** The same frozen k = 0 naive
  model was run with the discount switched on at inference only, with no
  retraining (`experiments/calibration/README_frozen_discount.md`). Reproduction
  checks against both earlier runs were exact. Relative to the undiscounted
  frozen model, this mechanism-only condition protects against added copies as
  much as trained PrismFlow does. At k = 4:
  - resolution: +0.0306 +/- 0.0270 vs +0.0308 +/- 0.0278
  - AURC: -0.0175 +/- 0.0071 vs -0.0183 +/- 0.0091
  - Brier: -0.0301 +/- 0.0228 vs -0.0274 +/- 0.0241

  All of these hold in 5/5 seeds for both. Training contributes a roughly
  constant offset instead: the untrained discount costs Brier and resolution
  even with no duplicates (k = 0: Brier +0.0091 +/- 0.0040, 5/5 seeds). At
  k = 4, mechanism-only is still worse than PrismFlow on Brier by
  +0.0063 +/- 0.0039 (5/5 seeds). Reliability is null for the mechanism too.
  **Restriction:** the frozen copies share view 0's encoder, so this is exact
  feature duplication, the easiest case for the estimator. With near-duplicate
  copies (`naive_weights_discounted`, weights trained on duplicates), the
  untrained discount protected about half as much. Whether the mechanism alone
  suffices beyond exact duplication is open.

So the claim can be attributed as follows. Robustness of resolution, accuracy
and AURC to exact duplication is a property of the discount mechanism. Matching
the undiscounted baseline's level when there are no duplicates requires
training under the discount.

Scope: one synthetic condition, k up to 4, 300 test samples per seed. The ECE
and reliability nulls mean "not detected at this power", not "shown to be
zero". A write-up that makes a calibration claim must cite this section and
must not restore the original framing without new evidence.

## Tests

`tests/unit/test_calibration.py` checks:

- a perfectly calibrated predictor (binary and multiclass) has ECE near 0
- an overconfident predictor has high ECE (analytically about 0.35)
- the three components reconstruct Brier exactly for binned forecasts, and all
  four do for continuous multiclass forecasts
- a perfect confidence ranking has lower AURC than a random one, and a random
  ranking's AURC is close to the error rate
- the tie handling, selective risk at known coverages, and macro scores on a
  hand-worked example
- `evaluate` writes all three files, matches `compute_metrics` seed by seed,
  refuses fewer than 5 seeds, and runs on the real `PrismFlow` output
