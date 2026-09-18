# Tail dependence and copula measures (Part 13)

## The argument this Part was built to test

Pearson correlation and dCor measure AVERAGE co-movement. Adversarial events are
not average; they are extreme. The quantity that should matter is upper tail
dependence,

    lambda_U = lim_{q -> 1-}  P( U_2 > q | U_1 > q )

the probability that view 2 is extreme given that view 1 is extreme. The
Gaussian copula has lambda_U = 0 by construction: two variables can correlate at
0.9 and still be modelled as having zero probability of joint extreme behaviour.
That is the modelling failure that priced correlated defaults as independent
before 2008, and this project's fusion was expected to commit it in miniature —
measuring average agreement and staying blind to coordinated extreme agreement,
which is exactly what a chorus attack is.

**The expected result was that rho_bar would stay flat as the attack strengthens
while lambda_U rose sharply.** That is not what happens.

## Verdict: refuted, and the tail measure is the one that fails

On the colluding pair — the only pair the attack acts on — 5 seeds:

| epsilon | attack success | rho_bar (colluding) | lambda_U (colluding) |
|---|---|---|---|
| 0.0 | 0.0000 | 0.1146 +/- 0.0216 | 0.1367 +/- 0.0462 |
| 0.2 | 0.0512 | 0.7218 +/- 0.0364 | 0.6233 +/- 0.1310 |
| 0.5 | 0.2318 | 0.8234 +/- 0.0739 | **0.7767 +/- 0.0742** |
| 1.0 | 0.4097 | 0.7098 +/- 0.0544 | 0.2900 +/- 0.0830 |
| 2.0 | 0.5195 | **0.8322 +/- 0.0217** | **0.2367 +/- 0.1431** |

**Pearson rises and holds.** It reaches 0.72 at the weakest attack and stays
between 0.71 and 0.83 at every strength. It is at its maximum at epsilon = 2.0.

**lambda_U rises, peaks at epsilon = 0.5, then collapses** — 0.7767 down to
0.2367, a fall of 0.54 from its peak, while the attack is getting *stronger* and
succeeding more often (0.23 -> 0.52). The tail measure loses the attack exactly
when the attack is most dangerous.

The separation between colluding and honest pairs makes the practical point:

| epsilon | rho_bar comp/honest | lambda_U comp/honest |
|---|---|---|
| 0.5 | 5.28x | 6.47x |
| 2.0 | **2.60x** | **1.29x** |

At full strength lambda_U barely distinguishes a colluding pair from an honest
one. Pearson still separates them by a factor of 2.6.

## Why: the attack flattens the tail, and rank measures need variation

The leading explanation was testable and is supported. `tail_spread_compromised`
is the coefficient of variation of the colluding views' values INSIDE their own
upper tail — how much the extremes still differ from one another:

| epsilon | 0.0 | 0.2 | 0.5 | 1.0 | 2.0 |
|---|---|---|---|---|---|
| spread within tail | 0.2162 | 0.2373 | 0.2383 | 0.1411 | **0.0747** |
| lambda_U (colluding) | 0.1367 | 0.6233 | 0.7767 | 0.2900 | **0.2367** |

Across the attacked cells the two track each other at **r = 0.939** (r = 0.847
pooled per-seed, n = 20), while spread against Pearson is **r = -0.245** —
Pearson is indifferent to it.

The mechanism: a strong chorus attack drives the colluding views to a
near-constant extreme. Once every one of their large values is roughly the same
size, WHICH samples land in the top 5% is close to arbitrary, so the ranks in
the tail become noise and co-exceedance reverts toward the independence baseline
(honest pairs sit at 0.117-0.183, and the colluding pair falls to 0.237).

**The deeper reason is that a copula is scale-free BY CONSTRUCTION.** The rank
transform divides out each marginal — that is precisely what makes a copula a
pure dependence measure, and it is why the approach looked attractive. But this
attack's signature is largely a shift in LEVEL: the colluding views are pushed
to extreme evidence together. A measure that deliberately discards level
discards a large part of the evidence that an attack is underway. Pearson on
class-conditional residuals keeps it.

That is the finding. The invariance that makes copulas the right tool for
comparing dependence across differently-scaled variables is the same invariance
that blinds them here.

## Detection: the tail flag does not work

ROC of attacked against clean per-sample scores, same seeds, neither detector
told which views collude:

| epsilon | tail AUC | Part 10 rho_bar AUC | tail TPR@5%FPR | rho_bar TPR@5%FPR |
|---|---|---|---|---|
| 0.2 | 0.4846 +/- 0.0022 | 0.5769 +/- 0.1782 | 0.0640 | 0.4448 |
| 0.5 | 0.4801 +/- 0.0063 | 0.6572 +/- 0.0774 | 0.0675 | 0.6197 |
| 1.0 | 0.5370 +/- 0.0132 | 0.7083 +/- 0.1243 | 0.0423 | 0.6867 |
| 2.0 | 0.5856 +/- 0.0126 | 0.7085 +/- 0.1395 | 0.0377 | 0.6932 |

The tail flag is at or below chance at the two weakest attacks and reaches only
0.586 at the strongest. **Its TPR at a 5% false-positive rate is 0.038-0.068 —
indistinguishable from the false-positive rate itself, i.e. no signal.** The
Part 10 unexplained-agreement flag, which reads the ordinary dependence signal,
is better at every strength.

Note the shapes differ: the tail flag is weakest where lambda_U is strongest
(epsilon 0.5) and least-bad where lambda_U has collapsed. The per-sample score
(max over pairs of the joint rank) and the batch-level coefficient are not the
same statistic, and this Part does not establish why they move oppositely. It is
not claimed as understood.

## Tail-aware ENIV moves the wrong way

Substituting lambda_U for rho_bar in the design-effect form:

| epsilon | ENIV (standard) | tail ENIV |
|---|---|---|
| 0.0 | 3.1171 +/- 0.0462 | 2.8720 +/- 0.3705 |
| 0.5 | 2.8151 +/- 0.0852 | 2.2398 +/- 0.2780 |
| 1.0 | 2.6801 +/- 0.0454 | 3.1350 +/- 0.3281 |
| 2.0 | 2.4984 +/- 0.0478 | **3.3324 +/- 0.3370** |

Standard ENIV falls monotonically as the attack strengthens — it detects the
collusion, which is the one capability Part 09 credited it with. Tail ENIV falls
until epsilon 0.5 and then RISES above its own clean value. Driving the discount
from lambda_U would therefore *weaken* it at exactly the strengths where the
attack succeeds most often. This variant should not be adopted.

## The copula fit does find real tail structure — at moderate attack only

| epsilon | t - Gaussian loglik | fitted df | fitted lambda_U |
|---|---|---|---|
| 0.0 | 23.2476 +/- 34.9932 | 28.0 +/- 21.4 | 0.0161 +/- 0.0284 |
| 0.2 | 90.5256 +/- 65.5318 | 8.4 +/- 4.4 | 0.0782 +/- 0.0560 |
| 0.5 | **178.8098 +/- 56.7024** | **4.8 +/- 1.5** | 0.1298 +/- 0.0376 |
| 1.0 | 63.5835 +/- 21.8164 | 11.4 +/- 3.5 | 0.0564 +/- 0.0219 |
| 2.0 | 54.9670 +/- 22.0779 | 18.0 +/- 6.7 | 0.0474 +/- 0.0249 |

At epsilon 0.5 the data clearly prefers a tail-dependent model: df falls to 4.8
and the likelihood gain is over three standard deviations from zero. On clean
data the gain (23.2 +/- 35.0) overlaps zero, so no preference is established
there. The same non-monotonicity appears: the tail signature is strongest at
moderate attack and fades as the attack saturates.

So the copula machinery is working and is detecting something real. It is the
attack's strong regime that has no tail signature left to find.

## A caveat that did NOT bite, reported because it was expected to

Tail estimation is data-hungry: at q = 0.95 a 300-sample split yields 15
exceedances, below `MIN_TAIL_SAMPLES = 50`. Every estimate here reports its
exceedance count, and the full split carries 60.

Re-estimating each condition on a 300-sample subsample of the same split:

| epsilon | lambda_U at n=1200 | lambda_U at n=300 (n_tail=15) |
|---|---|---|
| 0.0 | 0.1372 +/- 0.0611 | 0.1400 +/- 0.0559 |
| 0.5 | 0.2694 +/- 0.0755 | 0.2467 +/- 0.0726 |
| 2.0 | 0.0700 +/- 0.0399 | 0.0733 +/- 0.0435 |

The seed-MEAN is robust to the smaller sample. This does not license using 15
exceedances: the reliability flag is still False, the result says nothing about
a single run's estimate, and averaging five seeds is doing work here that one
run would not do. But the honest report is that the expected instability did not
appear at this level of aggregation.

## Two things established along the way, both corrections to this Part's own setup

**lambda_U = 0 for a Gaussian copula is an ASYMPTOTIC statement and the approach
is slow.** Measured: a Gaussian pair at rho = 0.9 estimates lambda_U at 0.685
(q = 0.90), 0.628 (0.95), 0.570 (0.99), 0.530 (0.995). A single lambda_U at a
single threshold therefore does not demonstrate tail dependence. The
discriminator is the DECAY — Gaussian falls away, a t copula flattens to a
positive constant (0.468 -> 0.400 over the same thresholds). Pinned in
`tests/unit/test_tail_dependence.py`.

**The log-ratio estimator is not a variance reduction.** It uses the whole
sample through the empirical copula, which was assumed to make it steadier. It
does not: across-seed sd is slightly worse than the empirical estimator's
(n = 300: 0.0988 vs 0.0930; n = 1000: 0.0658 vs 0.0621; n = 3000: 0.0433 vs
0.0409) and it sits systematically ~0.03 lower. Both are reported because they
bracket the estimate, not because one dominates.

## Design

- Part 09's attack configuration exactly: chorus, k = 2 of 4 views, beta = 1.0,
  30 steps, epsilon in {0.2, 0.5, 1.0, 2.0} plus clean.
- **Both measures read the same scalar.** For each sample and view,
  `s[n, v]` is the evidence view v assigns to the FUSED PREDICTED class, and
  rho_bar and lambda_U are both computed on that s. Measuring one on features
  and the other on evidence would confound the measure with the representation.
  A single-seed check with total evidence mass (no class conditioning at all)
  gave identical lambda_U, so the scalar choice is not driving the result.
- The predicted class, never the target: `docs/CONTRACT.md` forbids the
  suspicion detector ground truth, and a diagnostic that needs to know the
  attack cannot run in deployment.
- **Colluding pairs are reported separately from honest pairs**, as Part 09 did.
  The attack coordinates 1 of the 6 view pairs, so an all-pairs mean divides the
  signal by six; those diluted means are kept in `summary.md` for continuity and
  are not the headline.
- `n_samples` raised to 8000 (test split 1200) so the tail estimate clears
  `MIN_TAIL_SAMPLES`. Part 09's 300-sample split gives 15 exceedances. Only the
  evaluation size changed; the attack is unmodified.
- No scipy in the project's dependencies, so `norm_ppf` (Acklam), the
  regularised incomplete beta (Lentz), `t_cdf` and `t_ppf` are implemented in
  `prismflow/statistics/copula.py` and pinned against closed forms (Cauchy,
  `math.erf`, the normal limit).

## Limitations

- **The honest-pair control is not inert.** rho_bar among honest pairs rises
  0.113 -> 0.320 as epsilon grows. Once the attack succeeds the fused prediction
  flips to the attacker's target, and conditioning on that class makes the
  honest views — all uniformly low on it — correlate. This is the collider
  effect `dependence.py` documents, and it means "honest" is not a clean
  baseline at high epsilon. The comp/honest ratios above are affected.
- One attack family, one synthetic generator, k = 2 of 4, one beta.
- lambda_U is measured on a scalar summary of evidence, not on the full
  K-dimensional evidence vector.
- The t-copula df is fitted on a 9-point grid; df is a nuisance parameter and
  its exact value is not a finding.
- Tail ENIV uses the design-effect form only. The eigenvalue form is justified
  by a variance decomposition of a correlation matrix, and a matrix of
  conditional exceedance probabilities is not one.
- Why the per-sample tail flag is weakest where the batch-level lambda_U is
  strongest is not explained here.

## Reproduce

```
python -m experiments.tail.run_tail     # ~10 min on CPU
python -m experiments.tail.plot_tail
```

Writes `results/tail/{tail.json,summary.md,tail_vs_correlation.png}`.
