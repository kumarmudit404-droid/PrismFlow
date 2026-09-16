# Clone experiment (Part 06)

**Question.** Does duplicating an existing view increase fused confidence,
even though the duplicate carries no new information?

**Verdict: half of the thesis reproduces.**

- **Naive fusion fabricates confidence, as predicted.** Every copy of view 0
  raises its confidence while accuracy stays flat, in 5/5 seeds at both rho
  values.
- **PrismFlow does not hold confidence flat. It over-corrects.** Its
  confidence *falls* with every copy, in 5/5 seeds at both rho values,
  because ENIV falls from ~3.5 toward 2.4 instead of staying at 4. The
  dependence engine detects the clones correctly (~0.92). The error is in how
  `prismflow/eniv/eniv.py` aggregates those correlations into an effective
  count (see "Why ENIV falls").

This is not the "both curves flat" failure case in the Part 06 spec. The naive
effect is real. But the predicted PrismFlow behaviour ("confidence
approximately flat; ENIV stays near 4") did not happen, and it should not be
reported as a success.

All numbers below come from `run_clone.py` executed on 2026-09-16:
5 seeds (0-4), mean +/- sample std. The full table is `results/clone/summary.md`
and per-seed rows are in `results/clone/runs.csv`.

## Design (fixed before any run)

- Base: 4 synthetic views at rho = 0.0 (primary) and rho = 0.5 (secondary).
  k = 0..4 bit-identical copies of view 0 appended via
  `prismflow.data.corruption.duplicate_view`.
- A model's encoder count is fixed at construction, so each (rho, k, seed)
  trains its own (4+k)-view model. Every copy gets its own encoder over the same
  input, so redundancy must be detected across differently-trained feature
  bases.
- **naive**: trained and evaluated with `use_discount=False`.
- **prismflow**: trained and evaluated with `use_discount=True`, from the same
  seed (identical initial weights).
- **naive_weights_discounted** (diagnostic): the naive model's weights,
  evaluated with the discount on. It separates the discount operator's effect
  from anything the model learns while training under it.
- Training settings match the Part 04 baseline (40 epochs, batch 64, lr 1e-3,
  KL anneal 10). ENIV is estimated per forward batch of 64, as the deployed
  model does it.

## Results

### Mean confidence

| rho | system | k=0 | k=4 | within-seed change k=0 to k=4 | seeds rising |
|---|---|---|---|---|---|
| 0.0 | naive | 0.649 +/- 0.059 | 0.700 +/- 0.056 | **+0.051 +/- 0.013** | 5/5 |
| 0.0 | prismflow | 0.612 +/- 0.041 | 0.485 +/- 0.038 | **-0.128 +/- 0.026** | 0/5 |
| 0.0 | naive weights + discount | 0.582 +/- 0.044 | 0.323 +/- 0.037 | -0.259 +/- 0.031 | 0/5 |
| 0.5 | naive | 0.761 +/- 0.096 | 0.789 +/- 0.101 | **+0.028 +/- 0.010** | 5/5 |
| 0.5 | prismflow | 0.645 +/- 0.076 | 0.560 +/- 0.062 | **-0.085 +/- 0.030** | 0/5 |
| 0.5 | naive weights + discount | 0.546 +/- 0.080 | 0.381 +/- 0.088 | -0.165 +/- 0.023 | 0/5 |

The across-seed std (~0.06-0.10) is larger than the naive rise. That spread is
seed-level difficulty, which is shared across k because each seed uses the same
data and initialisation for every k. The within-seed change is the correct test,
and its sign is unanimous.

Naive confidence gained per added copy (mean within-seed change):
rho=0.0: +0.0140, +0.0137, +0.0060, +0.0173.
rho=0.5: +0.0110, +0.0086, -0.0000, +0.0083.

Gap between the two primary systems, naive minus prismflow:

| rho | k=0 | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| 0.0 | 0.037 +/- 0.027 | 0.085 +/- 0.023 | 0.132 +/- 0.022 | 0.179 +/- 0.029 | 0.215 +/- 0.028 |
| 0.5 | 0.116 +/- 0.022 | 0.143 +/- 0.030 | 0.178 +/- 0.040 | 0.199 +/- 0.051 | 0.230 +/- 0.052 |

The gap grows with k, but the naive rise accounts for less than a quarter of
it. At rho=0.0, k=4, naive gained +0.051 and prismflow lost -0.128. Most of
the gap is PrismFlow's own confidence falling.

### Accuracy (unaffected, as expected)

| rho | system | k=0 | k=4 | within-seed change |
|---|---|---|---|---|
| 0.0 | naive | 0.743 +/- 0.072 | 0.737 +/- 0.085 | -0.006 +/- 0.022 (2/5 up) |
| 0.0 | prismflow | 0.755 +/- 0.067 | 0.735 +/- 0.063 | -0.020 +/- 0.024 (1/5 up) |
| 0.5 | naive | 0.811 +/- 0.083 | 0.802 +/- 0.077 | -0.009 +/- 0.024 (2/5 up) |
| 0.5 | prismflow | 0.817 +/- 0.070 | 0.795 +/- 0.072 | -0.022 +/- 0.021 (1/5 up) |

Naive accuracy is flat, so its confidence rise is pure fabrication. PrismFlow
shows a small, not clearly separated drift down (about 1 std).

### ENIV and discount factor (prismflow)

| rho | truth for every k | k=0 | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|---|
| 0.0 | 4.00 | 3.465 +/- 0.207 | 3.327 +/- 0.167 | 2.991 +/- 0.071 | 2.603 +/- 0.043 | 2.411 +/- 0.087 |
| 0.5 | 1.60 | 2.259 +/- 0.109 | 2.289 +/- 0.099 | 2.164 +/- 0.146 | 2.031 +/- 0.109 | 1.914 +/- 0.085 |

alpha (= ENIV / views) falls from 0.866 to 0.301 at rho=0.0, and from 0.565 to
0.239 at rho=0.5.

At k=0, rho=0.5, ENIV reads 2.26 against a truth of 1.60. This is the residual
attenuation already reported in Part 05.

## Why ENIV falls

**The dependence engine is not the problem.** Mean measured dependence
(prismflow):

| rho | pairs | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|---|
| 0.0 | clone group (view 0 and its copies) | 0.931 | 0.918 | 0.920 | 0.917 |
| 0.0 | distinct base views | 0.053 | 0.037 | 0.044 | 0.022 |
| 0.5 | clone group | 0.949 | 0.941 | 0.937 | 0.935 |
| 0.5 | distinct base views | 0.243 | 0.238 | 0.239 | 0.229 |

Copies read near-duplicate and independent views read near-independent.

**The aggregation is the problem.** `compute_eniv` averages all off-diagonal
pairs into a single rho_bar and applies n / (1 + (n-1) rho_bar). That formula
is exact only when every pair has the same correlation. A clone group violates
this: a few pairs sit near 1 and the rest near 0. Take the measured
clone/distinct values, count how many pairs of each kind there are, and apply
the formula. The result reproduces the observed ENIV almost exactly:

| rho | k | views | predicted from measured dependence | observed ENIV |
|---|---|---|---|---|
| 0.0 | 1 | 5 | 3.200 | 3.327 |
| 0.0 | 2 | 6 | 2.905 | 2.991 |
| 0.0 | 3 | 7 | 2.533 | 2.603 |
| 0.0 | 4 | 8 | 2.360 | 2.411 |
| 0.5 | 1 | 5 | 2.217 | 2.289 |
| 0.5 | 2 | 6 | 2.073 | 2.164 |
| 0.5 | 3 | 7 | 1.928 | 2.031 |
| 0.5 | 4 | 8 | 1.832 | 1.914 |

The prediction uses the non-clone pairs' mean as the value for every non-clone
pair, which is why it sits slightly below observed. At rho=0.0 the correct
answer is 4 for every k. Averaging dilutes the clone group across all pairs, so
each copy looks like a little dependence spread over every view. The discount
then penalises all views, not just the duplicated one.

`eniv.py` is a frozen module. Nothing here has been changed. Replacing the
equicorrelation design effect with an aggregation that respects block
structure is a design decision for the project owner, not something this Part
should do.

## Training under the discount partly compensates

At the same k, PrismFlow and the naive-weights-plus-discount diagnostic apply
nearly the same alpha (rho=0.0, k=4: 0.301 vs 0.302). PrismFlow is still more
confident: +0.162 +/- 0.005 at rho=0.0 and +0.179 +/- 0.043 at rho=0.5, in 5/5
seeds each. A model trained under the discount learns to emit more evidence,
which wins back part of the confidence the discount removes. Here that
softens the over-correction. It is also a route by which training can work
against the discount, and it deserves attention in the calibration work (Part 07).

## Diagnostic checklist

Required by the spec only if both curves were flat. Answered anyway:

1. **Baseline accuracy saturated?** No. It is 0.735-0.755 at rho=0.0 and
   0.788-0.817 at rho=0.5.
2. **Vacuity pinned near 0 or 1?** No. Mean uncertainty ranges from 0.211 to
   0.677 across all systems and conditions.
3. **Dependence ~1.0 for duplicated views?** Yes, approximately. Clone-group
   dependence is 0.886-0.949 across all conditions.
4. **Discount applied (alpha < 1)?** Yes. alpha ranges from 0.239 to 0.866
   (prismflow).

## Reproduce

```
python -m experiments.clone.run_clone      # ~55 min on CPU, writes results/clone/
python -m experiments.clone.plot_clone     # writes the four PNGs
```

## Figures

- `results/clone/confidence_vs_duplicates.png`
- `results/clone/eniv_vs_duplicates.png`
- `results/clone/uncertainty_vs_duplicates.png`
- `results/clone/accuracy_vs_duplicates.png`
