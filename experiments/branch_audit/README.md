# Where did the shared information go? (Part 11 follow-up)

Part 11 found that disentanglement pushes ENIV UP on data that genuinely is
dependent, and named one suspected cause. **That cause is wrong.** This audit
refutes it and identifies the actual one.

Nothing here is a fix, and no frozen module is modified. Part 11's modules are
imported and used exactly as committed; this experiment only applies the
project's own estimator to representations that were always present in the
forward pass.

## The two hypotheses

**H1, the README's.** "Nothing requires the shared branches to stay aligned
ACROSS views ... a projection is free to satisfy that constraint by rotating
each view's shared subspace independently, which destroys measurable cross-view
structure."

**H2, the one it overlooked.** The penalty requires only that `shared_v` be
independent of `private_v`. Nothing requires the branch *named* shared to be the
one carrying across-view information. If a model routes the common latent into
`private`, the penalty is satisfied, dependence read off `shared` collapses, and
ENIV rises for a measurement reason rather than because redundancy was removed.

## Verdict

**H1 is refuted, and it was refutable without running anything.** Dependence is
measured with CCA, which `prismflow/statistics/dependence.py` states is
"invariant to any invertible linear remapping of either side". A rotation is
exactly such a remapping, so the instrument cannot see the cause H1 proposes.
Measured: an independent random orthogonal rotation per view moves ENIV by
**0.0221 on average, max 0.1376, sign inconsistent**, against an effect to
explain of **+0.3441 to +0.5203**.

The rotation probe is guarded in `tests/unit/test_branch_audit.py`, because a
buggy near-identity rotation would produce the same null reading for the wrong
reason. The rotation is orthogonal to 2.4e-7, sits 1.39 from the identity, moves
the data by up to 31.0 — and changes CCA by 7.6e-05.

**H2 is not confirmed as stated, and the truth is worse than a swap.** There is
no consistent swap: averaged over seeds, `dep(private) - dep(shared)` is
+0.0047 (rho 0.3) and +0.0031 (rho 0.6) for the orthogonality penalty —
essentially zero.

But the per-seed values are large and **alternate in sign**:

| rho 0.3, orthogonality | seed 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| dep(private) - dep(shared) | +0.0941 | -0.0945 | +0.0714 | +0.0614 | -0.1091 |

Every run concentrates 0.06-0.12 more cross-view dependence in one branch than
the other. Which branch is not consistent, so the mean cancels to nothing. **The
5-seed rule is what made this visible; a mean alone reports "no effect".**

## The orientation is decided by initialisation

The sign pattern across seeds is `+ - + + -`, **identical for both rho values
and both penalty methods**:

| condition | seed 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| rho 0.3, hsic_l1 | + | - | + | + | - |
| rho 0.3, orthogonality | + | - | + | + | - |
| rho 0.6, hsic_l1 | + | - | + | + | - |
| rho 0.6, orthogonality | + | - | + | + | - |

Seeds 0, 2, 3 route cross-view information into `private`; seeds 1, 4 into
`shared`. Neither the true dependence in the data nor the choice of penalty
changes it.

The reason is that **independence is symmetric**. The objective asks for
`shared_v ⊥ private_v` and cannot distinguish the two arguments, so the two
labellings are interchangeable optima and initialisation picks one. The branch
names are aspirational, not enforced.

## The architecture is innocent; the penalty causes it

| rho 0.6 | dep(shared) | per-seed sd of delta |
|---|---|---|
| V1 (reference) | 0.3832 | n/a |
| v2_lambda0 | 0.3840 | 0.0045 |
| v2_hsic_l1 | 0.3148 | 0.0430 |
| v2_orthogonality | 0.2052 | 0.0988 |

With `lambda_2 = 0` the branches are indistinguishable and dep(shared) matches
V1 to 0.0008. Branch asymmetry and apparent dependence loss both scale with
penalty pressure. This rules out the extra projection layers as the cause.

## The information was moved, not destroyed — mostly

Measuring the same models on `concat(shared, private)`:

| rho | system | dependence recovered | ENIV inflation cut |
|---|---|---|---|
| 0.3 | v2_hsic_l1 | 76.0% | 76.0% |
| 0.3 | v2_orthogonality | 47.8% | 37.3% |
| 0.6 | v2_hsic_l1 | 75.5% | 76.5% |
| 0.6 | v2_orthogonality | 55.8% | 55.1% |

Most of the "missing" dependence is recoverable by reading both branches, which
is what makes this substantially a measurement artefact. The residual is real:
the penalty does reduce recoverable cross-view dependence as well as relabelling
it, and the direction is the dangerous one.

## Design

- Same data, training and settings as Part 11, so the trained models are the
  same objects; only the measurement differs.
- Four representations per model, one estimator configuration (cca /
  pairwise_holdout / 4 null permutations — the model's own forward settings):
  `shared`, `private`, `both`, `shared_rotated`.
- V1 is measured on its encoder features, as its own forward path does, and is
  the reference for "what the dependence actually is".
- `v2_hsic_l10` and `v2_hsic_l100` are excluded: Part 11 showed they collapse to
  chance accuracy and near-zero confidence, so "where their information went" is
  not a meaningful question.

## Limitations

- Absolute ENIV is not comparable to Part 11's table: `evaluate()` computes ENIV
  per batch and sample-weight-averages, while this audit pools the test split,
  and CCA's finite-sample bias and chance correction both depend on n. Ordering
  is reproduced; all comparisons here are within-audit.
- Only the mean off-diagonal dependence was retained, not the per-pair matrix,
  so whether routing also varies per VIEW within a run is not directly measured.
  `both` exceeding each individual branch in most seeds hints that it does.
- `dependence_on="features"` exists in the config and was not run; it may remove
  the artefact entirely by never reading a single branch.
- One synthetic linear-Gaussian generator, 5 seeds, 2 rho values.

## Reproduce

```
python -m experiments.branch_audit.run_branch_audit   # ~25 min on CPU
```

Writes `results/branch_audit/{branch_audit.json,summary.md}`.
