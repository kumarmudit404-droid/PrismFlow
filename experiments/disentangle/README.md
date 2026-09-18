# Shared / private disentanglement with HSIC (Part 11)

**Path note.** Part 11's allowed-paths list does not include a README, but the
brief requires one ("QUESTION TO ANSWER IN THE README"). This file resolves that
contradiction in favour of the brief's explicit instruction.

**Status: V2, and it stays OFF.** `SharedPrivateConfig.enabled` defaults to
False and `build_shared_private` returns the real V1 `PrismFlow` class when it
is, so a default run cannot differ from V1. V1's files are untouched — no change
to `prismflow/models/prismflow.py`, `prismflow/eniv/` or `prismflow/train.py`.

**Question (the brief's).** Does disentanglement change ENIV estimates, and if so
in which direction? The brief's plausible outcome was that separating private
information makes the shared components MORE correlated and therefore LOWERS
ENIV.

## Verdict: the direction is the opposite of the hypothesis, and it is unwelcome

**ENIV goes UP, not down, at both rho, for every variant that trains at all.**

| rho | v1 | v2_lambda0 | v2_hsic_l1 | v2_orthogonality |
|---|---|---|---|---|
| 0.3 | 3.3281 +/- 0.0750 | 3.2779 +/- 0.0984 | 3.4100 +/- 0.1469 | **3.4922 +/- 0.1362** |
| 0.6 | 3.0406 +/- 0.0632 | 2.9845 +/- 0.0957 | 3.2020 +/- 0.1526 | **3.4304 +/- 0.1507** |

The architecture alone (`v2_lambda0`) slightly *lowers* ENIV. The penalty
reverses that and pushes it above V1, and the effect is larger where there is
more true dependence to find: +0.08 at rho 0.3 against **+0.16** at rho 0.6 for
HSIC, and +0.16 against **+0.39** for the orthogonality penalty.

**Why this is a warning rather than a curiosity.** At rho = 0.6 the data
genuinely has substantial cross-view correlation, and V1 reports 3.04 of 4
effective views. After disentanglement the estimator reports 3.43 — it has moved
toward "these views are independent" on data that is *not*. ENIV becomes less
faithful to the generative truth, not more. Confidence follows it up, exactly as
the project's thesis would predict: evidence-mass confidence rises 0.7107 (V1) ->
0.7629 (HSIC) -> 0.7695 (orthogonality) at rho 0.6.

So disentanglement does not give the dependence estimator a cleaner signal. It
gives it a signal that reports less dependence, inflates ENIV, and inflates
confidence — the failure mode this project exists to object to, arriving through
a mechanism intended to help.

**The likely cause, and it is testable.** Dependence is measured on the shared
branch, which is a learned per-view projection. Nothing in the objective
requires the shared branches to stay aligned ACROSS views — the penalty only
separates shared from private WITHIN a view. A projection is free to satisfy
that constraint by rotating each view's shared subspace independently, which
destroys measurable cross-view structure without removing any redundancy from
the model. `disentanglement_losses.py` deliberately does not push shared
branches apart from each other, but it does not hold them together either. An
explicit cross-view alignment term on the shared branches is the obvious next
experiment and was not run here.

**ENIV stability gets worse, not better.** Across-seed std of ENIV:

| rho | v1 | v2_lambda0 | v2_hsic_l1 | v2_orthogonality |
|---|---|---|---|---|
| 0.3 | **0.0750** | 0.0984 | 0.1469 | 0.1362 |
| 0.6 | **0.0632** | 0.0957 | 0.1526 | 0.1507 |

V1 is the steadiest estimate in all four comparisons; every V2 variant roughly
doubles the spread. If the split were isolating a cleaner quantity, the estimate
should have become steadier. It did the reverse.

## HSIC destabilises training above lambda_2 = 1, and the default stays OFF

| lambda_2 | accuracy (rho 0.3) | ECE | evidence-mass confidence | normalised HSIC |
|---|---|---|---|---|
| 0 | 0.8233 +/- 0.0841 | 0.0581 | 0.7842 | 0.9588 +/- 0.0075 |
| 1 | 0.8240 +/- 0.0831 | 0.0620 | 0.7905 | 0.1824 +/- 0.0293 |
| 10 | **0.3273 +/- 0.0183** | 0.0147 | **0.0159** | 0.5043 +/- 0.1234 |
| 100 | **0.3273 +/- 0.0183** | 0.0148 | **0.0257** | 0.5158 +/- 0.0836 |

At lambda_2 >= 10 the model collapses to chance accuracy (1/3 with 3 classes) at
both rho. **The ECE of 0.0147 in those rows must not be read as good
calibration:** the model has also collapsed to near-total vacuity (confidence
0.016), so it is making no confident claims to be wrong about. A calibration
metric on a model that predicts nothing is not a calibration result. Note also
that the collapsed runs disentangle *worse* (nHSIC 0.50) than lambda_2 = 1
(0.18) — the degenerate solution is not even good at the objective it destroyed
the task for.

Per the brief's instruction, HSIC is therefore left disabled by default. The
usable setting on this data is lambda_2 = 1, which buys real disentanglement
(0.9588 -> 0.1824) at no measurable accuracy cost.

## A practical trap: the two penalties are not on the same scale

Raw biased RBF HSIC and the raw linear penalty differ by about four orders of
magnitude on the same data — **0.0079 against 92.2**, a factor of ~11,600. A
single lambda_2 therefore applies wildly unequal pressure. The first version of
this experiment used lambda_2 = 0.1 for both and the HSIC term was effectively
switched off (normalised HSIC moved 0.9894 -> 0.9863, i.e. not at all), which
would have been reported as "HSIC does not work". It was a scaling error, not a
result. Anyone implementing this should calibrate lambda_2 per estimator, or
normalise, and should check that the penalty is actually moving before
concluding anything about it.

## The orthogonality penalty beats HSIC here, and that is a real result

| rho | v2_hsic_l1 | v2_orthogonality |
|---|---|---|
| 0.3 | 0.1824 +/- 0.0293 | **0.0506 +/- 0.0180** |
| 0.6 | 0.1687 +/- 0.0259 | **0.0551 +/- 0.0206** |

This is not a metric artefact: **both columns are scored by normalised RBF
HSIC**, the kernel criterion, so the linear penalty is winning on the nonlinear
measure. Accuracy is comparable (0.8187 against 0.8240 at rho 0.3).

Two reasons, and they point in different directions for future work. The
practical one is the scale issue above: at lambda_2 = 0.1 the linear penalty
still applies far more pressure than HSIC does at lambda_2 = 1, so this is not a
like-for-like comparison of the criteria. The substantive one is that this
dataset's views are linear projections of a shared latent plus Gaussian noise,
so the dependence to be removed IS essentially linear, and a criterion that only
sees linear dependence loses nothing by being blind to the rest.

**What this does and does not say about HSIC.** It does not vindicate the
orthogonality penalty in general — `tests/unit/test_hsic.py` pins the case it
provably fails, where `y = x^2` is scored as near-independent (0.02) by the
linear criterion and as strongly dependent (0.44) by HSIC. It says that this
synthetic data does not contain that case, so the extra power buys nothing here
and costs tuning difficulty. HSIC's advantage should appear on data with
genuinely nonlinear shared/private structure. That data was not generated for
this Part, and until it is, "HSIC is better" is unevidenced in this project.

## Accuracy and calibration (the viable variants)

| rho | metric | v1 | v2_lambda0 | v2_hsic_l1 | v2_orthogonality |
|---|---|---|---|---|---|
| 0.3 | accuracy | 0.8300 +/- 0.0797 | 0.8233 +/- 0.0841 | 0.8240 +/- 0.0831 | 0.8187 +/- 0.0800 |
| 0.3 | ECE | 0.0637 +/- 0.0137 | 0.0581 +/- 0.0201 | 0.0620 +/- 0.0167 | 0.0522 +/- 0.0160 |
| 0.6 | accuracy | 0.7940 +/- 0.0665 | 0.7827 +/- 0.0761 | 0.7827 +/- 0.0615 | 0.7727 +/- 0.0815 |
| 0.6 | ECE | 0.0574 +/- 0.0084 | 0.0557 +/- 0.0136 | 0.0611 +/- 0.0188 | 0.0558 +/- 0.0156 |

No separation anywhere — every mean +/- std interval overlaps. V1 has the
highest accuracy in both rho and the V2 variants are 0.006-0.021 below it, but
within noise. Disentanglement is not bought at a measurable accuracy cost, and
it does not buy a measurable calibration gain either.

## Design

- Data and training identical to Parts 07-10: 4 views, 40 epochs, the project's
  own `compute_loss`, so V1 and V2 differ only by architecture and the penalty.
- `v2_lambda0` exists to separate the two: without it, any V1/V2 difference
  could be the extra projection layers rather than the disentanglement.
- Evidence is computed from `concat(shared, private)` by default, so the
  classifier keeps all the information V1 had and an accuracy difference cannot
  be blamed on discarding half the features.
- Dependence is measured on the SHARED branch, which is what makes the ENIV
  question answerable — it is measured on the representation the penalty shapes.
- Normalised HSIC (centred kernel alignment, in [0, 1]) is reported; raw HSIC is
  what training minimises, because normalising inside the loss would let the
  model shrink the denominator instead.

## Limitations

- One synthetic generator, linear-Gaussian by construction — the case where a
  kernel criterion has least to offer (see above).
- lambda_2 swept over three values for HSIC and one for orthogonality; the
  comparison between the two criteria is therefore not like-for-like.
- The shared branches are never required to align across views, which is the
  leading explanation for the ENIV result and is untested.
- ENIV is measured on the shared branch for V2 and on the full features for V1,
  which is the intended comparison but does mean the two are not reading the
  same representation. A `dependence_on="features"` ablation exists in the
  config and was not run.
- 300 test samples per seed, 5 seeds, mean +/- sample std.

## Reproduce

```
python -m experiments.disentangle.run_disentangle   # ~37 min on CPU
```

Writes `results/disentangle/{disentangle.json,summary.md}` and per-condition
evaluation outputs.
