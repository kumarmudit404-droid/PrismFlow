# The Chorus attack (Part 09)

**Scope and ethics.** A controlled research simulation. Every number here comes
from this project's own models on its own synthetic data, inside this
repository. No real systems, no live traffic, no intrusion tooling.

**Question.** If an attacker controls k of n views and optimises them *jointly*
so that they agree on a wrong class, does dependence-discounted fusion resist
it? Part 08 established that the discount is blind to individually noisy views
because noise makes a view *less* dependent (`docs/KNOWN_LIMITATIONS.md` L3).
Agreement is the one axis the discount can actually read, so this is the attack
that tests the defence rather than its scope boundary.

**Verdict.**

1. **The agreement term works as an attack.** Chorus beats independent PGD at
   every epsilon, by +0.1327 +/- 0.0262 success at epsilon 2.0 (5/5 seeds).
2. **The discount does not defend against it.** PrismFlow's attack success rate
   is *higher* than naive fusion's in 12 of 12 attacked cells. At epsilon 1.0
   it is +0.0267 +/- 0.0071 worse, 0/5 seeds better.
3. **ENIV detects the chorus and the detection buys nothing.** ENIV falls
   3.36 -> 2.70 as epsilon rises, and mean pairwise dependence among the
   compromised views rises +0.7351 +/- 0.0914. The estimator sees the collusion
   clearly. Success still goes up. **Detection without defence.**
4. **The headline blindness result belongs to PGD, not to chorus.** Independent
   PGD reaches 0.4640 success at epsilon 2.0 while rho_bar moves **-0.0290 +/-
   0.0251** and ENIV stays flat at 3.34. The average-case dependence measure is
   blind to the uncoordinated attack that works nearly as well. This is the
   result that motivates the tail-dependence work in Part 13.
5. **Both systems grow more confident as they become more wrong.** Evidence-mass
   confidence rises 0.779 -> 0.893 while accuracy falls 0.830 -> 0.409. This is
   the Part 08 failure arriving by a different route.
6. **One thing the discount does do:** when the attack succeeds, PrismFlow holds
   less belief in the wrong class (-0.0562 +/- 0.0069 at epsilon 1.0, 5/5 seeds,
   bars separate). It is wrong slightly more often, and slightly less sure of it.
7. **The mechanism is proportional scaling, and that is why detection fails to
   become defence** (section 6). Discounted belief in the attacker's class still
   reaches fusion at 0.555-0.730, above 0.5 on 68-90% of successful attacks and
   3.3-3.4x larger than the honest views' belief in the true class. Worse, the
   belief reduction is *sublinear* in alpha: alpha = 0.50 removes only 17.6% of
   belief, and alpha = 0.43 removes 12.8%, so the discount weakens exactly as
   the attack strengthens.

All numbers: `run_chorus.py` executed 2026-09-18, 5 seeds (0-4), mean +/- sample
std, 300 test samples per seed. Accuracy, ECE, Brier and ENIV come from
`prismflow.evaluation.evaluate`; attack diagnostics are computed in
`run_chorus.py` (protocol.py is frozen in this Part). Full tables:
`results/chorus/summary.md`. Per-cell outputs: `results/chorus/<cell>_<system>/`.

## Design (fixed before the run)

- **Threat model.** The attacker controls the first k views, sees the model
  white-box, and may perturb only its own views' inputs within
  `||delta||_inf <= epsilon`. Labels, weights and the other views are untouched.
- **Objective.** Maximise `sum_{v in C} log b_v[target] - beta * sum_{i,j in C}
  ||b_i - b_j||^2` over the compromised views jointly. The second term penalises
  disagreement *among the attacker's own views*. The attack does not evade the
  trust mechanism, it operates it.
- **Three systems**, each attacked white-box in the configuration it is
  evaluated in: `naive` (trained and evaluated without the discount),
  `prismflow_nodiscount` (trained with, evaluated without), `prismflow` (both).
- **Targets** are least-likely class, chosen before the attack starts and never
  equal to the predicted class, so the clean cell scores 0.0000 success by
  construction.
- Data and training identical to Parts 04/07/08: 4 views, rho = 0.3, 40 epochs.
  Per seed one naive and one PrismFlow model from the same seed, so initial
  weights are identical and only `use_discount` differs.
- Optimiser: projected gradient ascent with backtracking, 30 steps. A step that
  does not improve the objective is rejected and the step size halved, so
  accepted iterates improve monotonically. Backtracking makes the attack
  stronger, never weaker.
- Dependence is measured with the same estimator the model uses
  (`feature_dependence_matrix` on encoder features), so attacked and clean
  rho_bar are directly comparable.

## 1. Does the agreement term help the attacker?

Success rate, naive fusion, paired within seed:

| epsilon | independent PGD | chorus (beta=1) | chorus - pgd | seeds chorus higher |
|---|---|---|---|---|
| 0.2 | 0.0367 +/- 0.0194 | 0.0420 +/- 0.0214 | +0.0053 +/- 0.0056 | 4/5 |
| 0.5 | 0.1853 +/- 0.0627 | 0.2160 +/- 0.0744 | +0.0307 +/- 0.0153 | 5/5 |
| 1.0 | 0.3693 +/- 0.0979 | 0.4547 +/- 0.1163 | +0.0853 +/- 0.0319 | 5/5 |
| 2.0 | 0.4640 +/- 0.1108 | 0.5967 +/- 0.1283 | +0.1327 +/- 0.0262 | 5/5 |

**Yes, and the gain grows with the budget.** Unpaired bars overlap because
seed-to-seed variance dominates (std ~0.11 across seeds against a ~0.13 effect),
so the paired within-seed comparison is the one that carries the result: 5/5
seeds at every epsilon above 0.2.

**Sanity check passes.** `chorus_k2_beta0` scores 0.3700 +/- 0.0990 against
`pgd_k2_eps1.0`'s 0.3693 +/- 0.0979. With the agreement term off, the objective
separates per view and the joint optimiser reproduces independent PGD to three
decimal places. The gap at beta > 0 is therefore the agreement term doing work,
not a bookkeeping artefact.

**beta is non-monotone** (epsilon 1.0, naive):

| beta | success | dependence among compromised | rho_bar |
|---|---|---|---|
| 0 | 0.3700 +/- 0.0990 | 0.1246 | 0.1698 |
| 1 | **0.4547 +/- 0.1163** | 0.7076 | 0.2629 |
| 5 | 0.3533 +/- 0.1202 | 0.6800 | 0.2507 |

Weighting agreement too heavily costs the attacker more in targeted belief than
it buys in conflict reduction. beta = 1 is the best of the three tested.

## 2. Does the discount defend? (the main question)

Attack success, `prismflow` - `naive`, paired within seed. Negative would mean
the discount defends:

| cell | prismflow - naive | seeds where prismflow is better |
|---|---|---|
| chorus_k2_eps0.2 | +0.0007 +/- 0.0068 | 2/5 |
| chorus_k2_eps0.5 | +0.0187 +/- 0.0205 | 1/5 |
| chorus_k2_eps1.0 | +0.0267 +/- 0.0071 | **0/5** |
| chorus_k2_eps2.0 | +0.0167 +/- 0.0270 | 2/5 |
| pgd_k2_eps0.5 | +0.0153 +/- 0.0096 | **0/5** |
| pgd_k2_eps2.0 | +0.0267 +/- 0.0239 | 1/5 |
| chorus_k2_beta5 | +0.0427 +/- 0.0290 | 1/5 |
| chorus_k1 | +0.0313 +/- 0.0180 | **0/5** |
| chorus_k3 | +0.0213 +/- 0.0259 | 1/5 |

**No. The sign is against PrismFlow in all 12 attacked cells.** The effect is
small (+0.001 to +0.043) and unpaired bars overlap everywhere, so this should be
read as "the discount does not defend", not as "the discount makes things much
worse". But there is no cell, at any k, epsilon or beta, where it helps, and the
direction is consistent enough to rule out the hypothesis that it defends.

The mechanism is visible in the vacuity column: PrismFlow carries *more* vacuity
than naive fusion (0.1949 vs 0.1605 at epsilon 1.0) because the discount removes
evidence. That lowers the bar the attacker's targeted evidence has to clear.

## 3. Headline: rho_bar under attack versus clean

Within-seed shift against each system's own clean run, naive fusion:

| cell | rho_bar (all pairs) - clean | dependence among compromised - clean | ENIV | success |
|---|---|---|---|---|
| chorus eps 0.5 | +0.0330 +/- 0.0094 | +0.2664 +/- 0.1001 | 3.147 | 0.2160 |
| chorus eps 1.0 | +0.0762 +/- 0.0059 | +0.5438 +/- 0.1015 | 2.888 | 0.4547 |
| chorus eps 2.0 | +0.1532 +/- 0.0215 | +0.7351 +/- 0.0914 | 2.709 | 0.5967 |
| **pgd eps 1.0** | **-0.0167 +/- 0.0200** | -0.0381 +/- 0.0571 | 3.341 | 0.3693 |
| **pgd eps 2.0** | **-0.0290 +/- 0.0251** | +0.0002 +/- 0.1227 | 3.336 | 0.4640 |
| chorus k3 eps 1.0 | +0.1517 +/- 0.0311 | +0.3095 +/- 0.0438 | 2.884 | 0.5593 |

**The answer splits in two, and the split is the finding.**

*Against chorus, rho_bar is not blind.* It moves +0.153 at epsilon 2.0 and ENIV
falls from 3.355 to 2.709 — the estimator reports roughly two-thirds of a view's
worth of lost independence. The coordinated attack is plainly visible to the
average-case measure.

*Against independent PGD, rho_bar is blind.* At epsilon 2.0 PGD reaches 0.4640
success — 78% of what the fully coordinated attack achieves — while rho_bar
moves **-0.0290**, in the wrong direction, and ENIV sits at 3.336 against a
clean 3.355, i.e. flat. An attacker who simply does not coordinate defeats the
model 46% of the time and leaves the dependence diagnostic reading "nothing
happened".

So the measure is not blind to collusion; it is blind to the attack that does
not need collusion. Combined with section 2 — where seeing the chorus did not
help the model resist it — average-case pairwise dependence fails on both
counts: it misses the attack it cannot see, and it does not defend against the
attack it can. That is the direct motivation for the tail-dependence work in
Part 13, and it is a stronger motivation than the blanket "rho_bar barely moves"
the Part brief anticipated.

## 4. Confidence and calibration under attack

Naive fusion across the chorus epsilon sweep:

| cell | accuracy | ECE | evidence-mass confidence | belief in wrong class when successful |
|---|---|---|---|---|
| clean | 0.8300 +/- 0.0791 | 0.0579 +/- 0.0094 | 0.7790 +/- 0.0829 | n/a |
| eps 0.2 | 0.8100 +/- 0.0798 | 0.0598 +/- 0.0059 | 0.7212 +/- 0.0883 | 0.4767 +/- 0.0231 |
| eps 0.5 | 0.7093 +/- 0.0774 | 0.0798 +/- 0.0256 | 0.7406 +/- 0.0775 | 0.6325 +/- 0.0186 |
| eps 1.0 | 0.5307 +/- 0.1016 | 0.2875 +/- 0.0690 | 0.8395 +/- 0.0408 | 0.8042 +/- 0.0247 |
| eps 2.0 | 0.4087 +/- 0.1026 | 0.4800 +/- 0.0969 | 0.8928 +/- 0.0166 | 0.8937 +/- 0.0161 |

**Confidence rises as accuracy collapses.** Accuracy falls 0.421 from clean to
epsilon 2.0 while evidence-mass confidence *rises* 0.114 and ECE grows eightfold
to 0.480. When the attack succeeds the model holds 0.894 belief in the class the
attacker chose. PrismFlow behaves the same way (0.7499 -> 0.8681 confidence,
accuracy 0.8300 -> 0.3947). This is the same shared failure Part 08 found under
noise: evidence accumulates, and neither system asks whether it should.

**Where PrismFlow measurably helps.** Belief in the wrong class on successful
attacks, `prismflow` - `naive`, paired:

| cell | difference | seeds better |
|---|---|---|
| chorus_k2_eps0.5 | -0.0323 +/- 0.0095 | 5/5 |
| chorus_k2_eps1.0 | **-0.0562 +/- 0.0069** | 5/5 |
| chorus_k2_eps2.0 | -0.0343 +/- 0.0069 | 5/5 |
| chorus_k2_beta5 | -0.0315 +/- 0.0195 | 5/5 |
| pgd_k2_eps0.5 | -0.0275 +/- 0.0076 | 5/5 |

At epsilon 1.0 and 2.0 the unpaired bars separate as well (0.8042 +/- 0.0247 vs
0.7480 +/- 0.0246; 0.8937 +/- 0.0161 vs 0.8594 +/- 0.0157). This is the only
consistent benefit in the experiment: the discount does not stop the attack, but
it makes the model less certain of the attacker's answer when the attack lands.
PrismFlow also has slightly lower ECE than naive at high epsilon (0.4564 vs
0.4800 at epsilon 2.0), within overlapping bars, and slightly *worse* ECE on
clean input (0.0637 vs 0.0579).

## 5. How many views must the attacker hold?

At epsilon 1.0, beta 1, naive fusion:

| k | success | rho_bar - clean | ENIV |
|---|---|---|---|
| 1 | 0.2220 +/- 0.0858 | +0.0254 +/- 0.0197 | 3.213 |
| 2 | 0.4547 +/- 0.1163 | +0.0762 +/- 0.0059 | 2.888 |
| 3 | 0.5593 +/- 0.1337 | +0.1517 +/- 0.0311 | 2.884 |

Success roughly doubles from one view to two and then flattens, with 4 views and
2 clean ones still setting a ceiling near 0.56-0.60. `dependence_compromised` is
n/a at k = 1: a single view has no within-pairs to average, which is itself a
limitation of the diagnostic — the one-view attack is invisible to it by
construction, and rho_bar moves only +0.025 while the attack succeeds 22% of the
time.

**Does PrismFlow's disadvantage widen with k?** Paired within seed, same format
as the epsilon and beta tables above:

| k | naive | prismflow | prismflow - naive | seeds prismflow better | relative |
|---|---|---|---|---|---|
| 1 | 0.2220 +/- 0.0858 | 0.2533 +/- 0.0909 | +0.0313 +/- 0.0180 | 0/5 | +14.1% |
| 2 | 0.4547 +/- 0.1163 | 0.4813 +/- 0.1133 | +0.0267 +/- 0.0071 | 0/5 | +5.9% |
| 3 | 0.5593 +/- 0.1337 | 0.5807 +/- 0.1099 | +0.0213 +/- 0.0259 | 1/5 | +3.8% |

**Neither: it stays flat in absolute terms and narrows in relative terms.** The
absolute gap drifts down slightly (+0.031 -> +0.027 -> +0.021) but the standard
deviations (0.018-0.026) are as large as the drift, so the absolute gap should
be read as flat within noise. The relative gap does fall clearly, from +14.1% to
+3.8%, simply because the baseline attack succeeds more often as k grows: the
discount's fixed handicap is a larger fraction of a small success rate than of a
large one. The direction is against PrismFlow at every k, and 0/5 seeds favour
it at k = 1 and k = 2.

The practical reading is that PrismFlow's disadvantage is **not** a collusion-
scaling effect. If the discount were actively counterproductive against
coordination, the gap would widen with k, since more compromised views means
more measurable dependence and a heavier discount. It does not. The handicap
looks like a roughly constant cost of carrying extra vacuity (section 2), which
is present whether or not the attack is coordinated at all — consistent with the
identical gap under uncoordinated PGD (+0.0153 to +0.0267).

## 6. Why detection does not become defence: raw vs discounted evidence

Diagnostic run after the main experiment, on `prismflow` only, splitting
successful attacks from failed ones. **Method note:** the saved `.npz` files
hold only `delta` and the saved `metrics.json` only aggregate metrics, so this
could not be read off the stored artefacts. The models were reconstructed by
retraining with the same seeds and replaying the saved perturbations; the
reproduced per-seed success rates match `attack_metrics.json` **exactly on all
5 seeds in both cells**, so the reconstruction is faithful. No model or discount
code was changed.

Note the mechanism actually in use is `evidence_discount` with per-view
`soft_cluster_alpha`, not the retired V1 scalar `shafer_discount`.

Compromised views' mean belief in the target class:

| cell | alpha (compromised) | alpha (honest) | undiscounted | discounted | belief retained |
|---|---|---|---|---|---|
| chorus eps 1.0 | 0.5021 +/- 0.0092 | 0.6088 +/- 0.0434 | 0.6743 +/- 0.0560 | 0.5553 +/- 0.0611 | 82.4% |
| chorus eps 2.0 | 0.4331 +/- 0.0157 | 0.5712 +/- 0.0547 | 0.8368 +/- 0.0229 | 0.7301 +/- 0.0320 | 87.2% |

On failed attacks the same views carry only 0.1533 -> 0.1138 (eps 1.0) and
0.1127 -> 0.0831 (eps 2.0), so the quantity being discounted really does track
whether the attack worked.

What fusion actually receives on successful attacks:

| cell | compromised -> target | honest -> true | ratio | compromised u | honest u |
|---|---|---|---|---|---|
| chorus eps 1.0 | 0.5553 | 0.1668 +/- 0.0470 | **3.3x** | 0.4441 | 0.7765 |
| chorus eps 2.0 | 0.7301 | 0.2130 +/- 0.0457 | **3.4x** | 0.2696 | 0.7337 |

**Confirmed.** Discounted belief in the attacker's class stays above 0.5 on
average in both cells, and exceeds 0.5 on **67.6% +/- 12.9%** of successful
attacks at epsilon 1.0 and **90.1% +/- 1.5%** at epsilon 2.0. It arrives at
fusion 3.3-3.4x larger than the honest views' belief in the true class, and the
honest views are far more vacuous (u ~ 0.73-0.78) than the compromised ones
(u ~ 0.27-0.44). The attacker wins the fusion on magnitude.

**Two mechanisms, and the second is the more damaging.**

*Proportional, not capping.* Alpha roughly halves the compromised views'
evidence (0.50 and 0.43) but cannot bound the result. Closing a 3.3x margin
needs alpha near 0.25 with a floor, or an explicit cap on any single view's
contribution. A proportional factor scales the attacker's evidence down and
leaves it dominant.

*The reduction is sublinear in alpha, so it is even weaker than alpha suggests.*
Discounting scales the **evidence**, and belief is `e / (sum(e) + K)`, so
scaling evidence by alpha gives `alpha*e / (alpha*sum(e) + K)` — the denominator
shrinks too. At epsilon 1.0 an alpha of 0.502 removes only 17.6% of the belief;
at epsilon 2.0 an alpha of 0.433 removes 12.8%. **The stronger the attack, the
less of it a given alpha removes**, because high-evidence opinions are exactly
where this normalisation is most forgiving. The discount is weakest precisely
when it is needed most.

*Alpha is also per-view and batch-level*, not per-sample: every sample in a
batch receives the same factor, so the correction cannot concentrate on the
attacked samples even when the batch-level dependence signal fires.

This is the mechanical explanation for sections 2 and 3: the dependence signal
detects the collusion (ENIV 3.36 -> 2.70), and the correction it drives is the
wrong shape and the wrong size to neutralise it.

## Limitations

- One synthetic setting, 4 views, rho = 0.3, 300 test samples per seed, 5 seeds.
  Overlapping mean +/- std is a coarse test; the paired within-seed comparisons
  are what carry the conclusions here, and they are reported alongside.
- White-box, fixed-target, L-inf bounded, 30 PGD steps. A stronger optimiser, an
  adaptive target, or an attacker who also manipulates the dependence estimator
  itself would all be different experiments.
- The attacker perturbs *inputs*. An attacker who controls a view's encoder or
  its emitted evidence directly is unconstrained by epsilon and is not tested.
- Epsilon is in input units against view features with std ~1.3, so epsilon 2.0
  is a large perturbation, well beyond an imperceptibility budget. It is included
  to find the ceiling, not as a realistic threat.
- No defence was tuned against this attack. The claim is about the discount as
  built in Parts 01-07, not about dependence-based defences in general.

## Reproduce

```
python -m experiments.chorus.run_chorus            # ~15 min on CPU (3 min training, 12 min attacks)
python -m experiments.chorus.run_chorus --quick    # smoke test, NOT evidence
```

Perturbation tensors are written to `results/chorus/perturbations/*.npz`
(untracked by `.gitignore`; 180 files, one per cell x system x seed). Config,
commit and library versions are recorded in `results/chorus/metadata.json`.
