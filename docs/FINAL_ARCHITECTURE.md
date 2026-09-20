# PrismFlow: final architecture

The thesis, unchanged from the first commit:

> Agreement between views should contribute evidence in proportion to the
> effective independence of those views, not in proportion to how many views
> agree.

This document describes what was built, what each piece is for, and — because
the project ends on several negative results — what each piece turned out not
to do.

## The pipeline

```
views -> encoders -> evidence -> dependence -> ENIV -> discount -> fusion
                                                          |
                                                          +-> prediction
                                                          +-> confidence
                                                          +-> diagnostics
```

Read left to right, the claim is: the same evidence, fused under a correct
count of independent sources, yields a confidence that does not inflate when
sources are duplicated or made to collude.

### Stage by stage

| stage | module | shape in | shape out |
|---|---|---|---|
| encoders | `models/encoders.py` | `[B, V, D_view]` | `[B, V, D_feat]` |
| evidence | `models/evidence.py` | `[B, V, D_feat]` | `[B, V, K]` non-negative |
| dependence | `statistics/dependence.py` | `[B, V, D_feat]` | `[V, V]` matrix |
| ENIV | `eniv/eniv.py` | `[V, V]` | scalar in `[1, V]` |
| discount | `eniv/discount.py` | `[B, V, K]`, alpha | `[B, V, K]` scaled |
| fusion | `models/fusion.py` | `[B, V, K]` | `[B, K]` + vacuity |

`B` is batch, `V` views, `K` classes. The full convention, including the
missing-view mask, is in `docs/CONTRACT.md`.

## The four load-bearing design decisions

### 1. Dependence is measured on encoder features, not on raw views

`feature_dependence_matrix` reads the encoder outputs, with canonical
correlation as the default method, conditioned within the model's own
pre-fusion predicted class, and corrected against a permutation null.

*Why:* raw-view correlation measures how the data was generated; feature
correlation measures how much the model's own evidence streams actually
duplicate each other, which is the quantity the discount needs.

*What it cost:* the encoder contributes its own compression to the estimate.
On the Part 05 validation sweep, dependence measured on raw views achieves MAE
**0.1919 +/- 0.1494** effective views; the same estimator on encoder features
achieves **0.5220 +/- 0.2998**. Roughly two-thirds of the error is the encoder,
not ENIV. See `docs/KNOWN_LIMITATIONS.md` L9.

### 2. The discount is proportional, and its weights carry no gradient

`alpha_i = 1 / sum_j clip(R_ij, 0, 1)`, applied as either a Shafer belief
discount or an evidence scaling. `alpha = alpha.detach()` at
`eniv/discount.py:75` and `:122`.

*Why:* an estimator with a signed bias larger than its seed variance (L10)
should not be allowed to backpropagate into the representation it is measuring.
The measured payoff is stability — V1 has the steadiest across-seed ENIV of any
arm tried (std **0.0632** at rho = 0.6, against 0.0957-0.2590 for every V2
variant).

*What it cost:* PrismFlow is a post-hoc correction. It changes what the model
concludes from its evidence, never what evidence the encoders learn to produce.
A representation penalty *can* move dependence where the discount cannot
(**0.3832 +/- 0.0231** to **0.2052 +/- 0.0550** at rho = 0.6), but collapses the
model at higher weights. This is L13, and it bounds the design's ceiling.

### 3. Detection is label-free by construction, and tested for it

`statistics/suspicion.py` reads belief vectors, the dependence matrix, and a
stratum label taken from the model's own prediction. It never receives an
attack label, an attack config, or a compromised-view index.
`tests/unit/test_suspicion.py` enforces this against the actual function
signatures with a forbidden-substring tuple.

*Why:* a detector told where the attack is measures nothing.

*What it cost:* the label-free requirement is why the Part 10 `detector` signal
is excluded from the Part 14 deployable gate — it has no label-free sign. The
oracle gate (fitted *with* attack labels, AUC **0.7448 +/- 0.0592**) is
therefore an upper bound on what any legal gate could extract, never a
detection result.

### 4. Every estimator reports its own reliability

`TailEstimate.reliable` is `n_tail >= MIN_TAIL_SAMPLES` (50).
`RealMultiViewDataset.rho_matrix` is `None`, asserted by test, because real data
has no ground-truth dependence. `evaluate()` raises below `MIN_SEEDS` and
stamps `not_evidence` when overridden.

*Why:* the project's central quantity is a confidence. A confidence estimator
that cannot say when it is out of data is the exact failure it was built to fix.

*Where it shows:* the demo app displays the tail value and immediately labels it
"UNRELIABLE: 25 exceedances is below MIN_TAIL_SAMPLES = 50. Shown because
hiding it would be worse."

## Module map

```
prismflow/
  models/         encoders, evidence heads, fusion, PrismFlow, DefendedPrismFlow
                  shared_private.py + disentanglement_losses.py = the V2 arm (EXPERIMENTAL)
  eniv/           eniv.py (estimator), discount.py (alpha), per_sample.py, amortized.py
  statistics/     dependence.py, suspicion.py, tail_dependence.py, copula.py,
                  hsic.py, clique_contrast.py
  attacks/        chorus.py (the collusion attack), adaptive.py (BPDA adversary),
                  baseline_attacks.py (PGD control)
  data/           synthetic.py (rho is a parameter), real_datasets.py (rho is None),
                  dataset.py (splits), corruption.py, loaders.py
  evaluation/     protocol.py (the 5-seed gate), calibration.py, metrics.py
  app/            panels.py (demo rendering only)
  train.py, train_two_timescale.py
```

45 modules, ~7000 lines, 563 unit tests.

## What the architecture does NOT do

Stated here because the pipeline diagram invites each of these assumptions:

- **It does not detect unreliable views.** The discount reads cross-view
  dependence. A view that is individually corrupted but uncorrelated with the
  others moves that signal in the wrong direction. This is a structural
  boundary, not a tunable parameter (L3).
- **It does not survive an adaptive adversary.** One attack objective degrades
  the discount and the detection gate simultaneously, because both are
  functions of cross-view belief agreement (L8). At gamma = 5 the attacker
  gains success **0.6267 -> 0.6600** while pushing gate AUC **0.6743 ->
  0.5806**.
- **It does not shape representation learning.** See decision 2 and L13.
- **It is not validated on real data.** ENIV can be *applied* to real data but
  not *validated* there, because no real dataset has a known rho (L9, L14).
- **It has not been evaluated with humans or in deployment** (L15).

## Where the project ended

Three contributions stand: the ENIV estimator and discount, the Chorus
collusion attack, and the measurement that a standard 6-view benchmark carries
**3.28 +/- 0.07** effective views at an independence efficiency of **0.547 +/-
0.011**.

Two lines closed negatively. Tail dependence is not a sharper detector than
mean dependence (AUC **0.5856 +/- 0.0126** against **0.7085 +/- 0.1395** at
eps = 2.0, L7). And detection and correction fail together under one adaptive
objective, so they are not two defences (L8).

The redundancy-audit result is the one that generalises furthest and depends on
the fewest contested assumptions: it needs ENIV to be *ordinally* trustworthy,
which the validation sweep supports (correlation **0.9922**), rather than
*numerically* exact, which it does not.
