# Known limitations

Open issues that are measured or suspected but not resolved. Each entry names
the evidence and what would change our understanding of it.

L1-L8 were written as each Part found them and are ordered by discovery. L9-L15
were added in Part 16 to cover the standing limitations of the project as a
whole -- the ones that are true regardless of which experiment you read, and
which a reader is most likely to assume away.

| standing limitation | entry |
|---|---|
| ENIV is validated only on synthetic data with analytically known rho | **L9** |
| dependence is estimated from finite batches; the estimator's bias is signed, and larger than its seed variance | **L10** |
| tail dependence is unreliable below 50 exceedances | **L11** |
| the exact attacker capabilities that defeat the defence | **L12** (see also L4, L8) |
| V1 detaches ENIV, so it corrects inference but does not shape representation learning | **L13** |
| a single dataset family; no cross-domain evidence | **L14** |
| no human-subject and no deployment evaluation | **L15** |

## L1. Flat confidence under duplication depends on training-time adaptation

**Component:** soft-cluster discount weights, `alpha_i = 1 / sum_j clip(R_ij, 0, 1)`
(`soft_cluster_alpha`, commit 333f143), applied by scaling evidence. These are
not the BLUE inverse weights (0bdf92c, `per_view_alpha`, no longer used).

**Evidence:** `results/clone_eigen_softcluster/` (5 seeds, mean +/- sample std;
within-seed confidence change from k=0 to k=4 copies of view 0).

| rho | system | change at k=4 | seeds rising |
|---|---|---|---|
| 0.0 | naive | +0.051 +/- 0.013 | 5/5 |
| 0.0 | prismflow (trained and evaluated with discount) | +0.013 +/- 0.013 | 4/5 |
| 0.0 | naive weights + discount at inference only | -0.041 +/- 0.008 | 0/5 |
| 0.5 | naive | +0.028 +/- 0.010 | 5/5 |
| 0.5 | prismflow | +0.010 +/- 0.007 | 5/5 |
| 0.5 | naive weights + discount at inference only | -0.054 +/- 0.023 | 0/5 |

**What this shows.** The soft-cluster discount gives near-flat confidence under
duplication only together with training under that discount. The mechanism
alone, applied at inference to naively trained weights, still loses confidence
as copies are added: 0/5 seeds rising at k=4 at both rho. The loss is monotonic
in k at rho=0.5. At rho=0.0 it levels off between k=3 and k=4 (-0.043, then
-0.041). This is the same direction of failure the mechanism change was meant
to remove. It is much smaller than before: the same diagnostic at k=4 was
-0.259 / -0.165 in Part 06 (design-effect ENIV) and -0.123 / -0.099 in
`clone_eigen` (rho=0.0 / 0.5). At rho=0.5 it is not smaller than the per-view
run (`clone_eigen_perview`: -0.030 +/- 0.052).

**Interpretation (inferred, not measured in this run).** The trained model
appears to compensate for the discount it experienced during training, by
emitting more evidence, on this specific duplicate structure. Part 06 measured
this compensation for the original discount. The per-view evidence magnitudes
were not logged in this run.

**Open question.** It is not validated whether this compensation generalises
correctly to adversarially induced view correlation that is not literal
duplication. That is the scenario Part 09's Chorus attack tests against a
fixed trained model.

**Action.** If Part 09 shows the defence failing in a way that could trace back
to this, revisit this entry first, before assuming the attack itself is the
problem. A first check would be to compare the prismflow and
naive-weights-plus-discount systems under the attack.

## L2. Residual per-pair dependence accumulates with view count

From 333f143: residual ~0.05 per-pair measured dependence accumulates in the
soft cluster size, so an untouched view's alpha falls from ~0.88 to ~0.75 going
from 4 to 8 views (controlled harness). This may contribute to prismflow's
upward confidence drift at rho=0.0 in `clone_eigen_softcluster`
(+0.003, +0.007, +0.010, +0.013 for k=1..4). That link is untested.

## L3. The discount addresses redundant evidence, not unreliable evidence

**Scope boundary, not a defect.** The discount acts on views that are
correlated *with each other*. It has no purchase on a single view that is
individually noisy or corrupted but uncorrelated with anything.

**Evidence:** Part 08, `experiments/robustness/` (5 seeds, mean +/- sample std;
`results/robustness/noise_summary.md`).

- ENIV stays flat as view noise rises: 3.2-3.4 across every sigma in all three
  noise conditions. Noise makes a view LESS dependent on the others, not more,
  which is the opposite of what a dependence-based discount can detect.
- One noisy view (sigma 0.25 -> 4) costs naive fusion 0.098 accuracy while
  top-label confidence falls only 0.014, and evidence-mass confidence RISES
  0.016. The model stays confident while becoming wrong.
- PrismFlow does not separate from naive fusion at any sigma in any condition,
  and at sigma 4 it is worse on accuracy and Brier in 5/5 seeds.

**What this means.** This is a structural boundary of the dependence-based
approach, not a tunable parameter: no setting of the discount detects
individually unreliable evidence, because the signal it reads (cross-view
dependence) does not move in that case. Detecting unreliable-but-independent
views needs a different signal entirely, such as per-view reconstruction error
or out-of-distribution scoring on raw inputs. That is out of scope for this
project.

**Consequence for later Parts.** An attack that makes views AGREE (collusion,
duplication) is on the axis the discount can read. An attack that merely
degrades a view's quality is not, and PrismFlow should not be claimed to defend
against it. See `docs/CONTRACT.md` section 1.

### L3 extension (2026-09-18, Part 09): adversarial evidence, not just natural

Part 09 tested the axis this entry said was readable — an attacker perturbing
k views *jointly* so they agree on a wrong class — and the boundary turned out
to be wider than "unreliable evidence".

**Evidence:** Part 09, `experiments/chorus/` (5 seeds, mean +/- sample std;
`results/chorus/summary.md`).

- **Reading the axis does not confer defence.** Under the chorus attack ENIV
  falls 3.36 -> 2.70 and dependence among the compromised views rises
  +0.7351 +/- 0.0914, so the collusion is plainly detected. PrismFlow's attack
  success rate is nonetheless HIGHER than naive fusion's in 12/12 attacked
  cells (+0.0267 +/- 0.0071 at epsilon 1.0, 0/5 seeds better).
- **The blindness moves to uncoordinated attacks.** Independent PGD reaches
  0.4640 success at epsilon 2.0 — 78% of the coordinated attack — while rho_bar
  moves -0.0290 +/- 0.0251 (wrong direction) and ENIV stays flat at 3.34.
- **k = 1 is invisible by construction.** A single compromised view has no
  within-pairs, so `dependence_compromised` is undefined while the attack
  still succeeds 22% of the time.

**What this means.** The original L3 framing — "the discount addresses redundant
evidence, not unreliable evidence" — is too generous. Redundancy that is
*adversarially induced* is detected and still not defended against. The accurate
statement is that the discount addresses redundant evidence **arising
naturally**, and that detection of redundancy is not the same capability as
robustness to it. The mechanism is L4.

## L4. The discount is proportional, not capping, and weakens as evidence grows

**Scope boundary and defect boundary both.** This is the mechanism behind L3's
extension: it explains why detecting collusion does not neutralise it.

**Component:** `evidence_discount` with per-view `soft_cluster_alpha`
(the retired V1 scalar `shafer_discount` is not what runs).

**Evidence:** Part 09 diagnostic, `experiments/chorus/README.md` section 6
(5 seeds, `prismflow`, successful attacks only; models reconstructed by
deterministic retraining, replayed per-seed success rates match
`results/chorus/attack_metrics.json` exactly).

- On successful attacks, compromised views' discounted belief in the attacker's
  class is 0.5553 +/- 0.0611 (epsilon 1.0) and 0.7301 +/- 0.0320 (epsilon 2.0):
  above 0.5 on 67.6% and 90.1% of successes respectively.
- That belief reaches fusion **3.3-3.4x larger** than the honest views' belief
  in the true class (0.1668 and 0.2130), and the honest views are much more
  vacuous (u ~ 0.73-0.78 against 0.27-0.44).
- **The reduction is sublinear in alpha.** Discounting scales evidence, and
  belief is `e / (sum(e) + K)`, so alpha = 0.5021 removes only 17.6% of belief
  and alpha = 0.4331 removes 12.8%. The stronger the attack, the smaller the
  fraction a given alpha removes.
- Alpha is per-view and batch-level, not per-sample, so the correction cannot
  concentrate on attacked samples within a batch.

**What this means.** A proportional factor cannot bound any single view's
contribution. Neutralising a 3.3x margin would need alpha near 0.25 with a
floor, or an explicit cap on per-view contribution, or a per-sample rather than
per-batch factor. None of these is what V1 does, and the sublinearity means the
mechanism is weakest exactly where the evidence is largest — which is where an
attacker operates.

**What would change our understanding.** A capped or per-sample variant tested
on the same `results/chorus/perturbations/` tensors would show directly whether
the shape of the correction is the binding constraint, or whether dependence is
simply the wrong signal to drive a correction from at all.

### L4 extension (2026-09-18): that test was run, and the shapes above are refuted

The paragraph above proposed three remedies. All three were tested on exactly
those tensors and none of them works. **The speculation in "What this means" is
superseded by this subsection.**

**Evidence:** `experiments/chorus/README_correction_shapes.md` (5 seeds,
read-only replay; both validation gates pass — reconstructed per-seed success
matches `results/chorus/attack_metrics.json` exactly and the baseline row
reproduces section 6 to 4 decimals).

At eps 1.0, against a baseline margin of 3.33x and attack success 0.4813:

| shape | ratio | attack success |
|---|---|---|
| capped, belief <= 0.3 | 2.65x | 0.4780 |
| floor_alpha, alpha >= 0.25 | 3.33x | 0.4813 (exact no-op) |
| alpha_ceiling, alpha <= 0.25 | **4.01x** | **0.4933** |
| per_sample | 3.30x | **0.5120** |

- **A cap cannot be selective.** It is the only shape that moves absolute belief
  much (0.5553 -> 0.2650), but no correction knows which views are compromised,
  so it must apply to all of them. Honest views get clipped too (0.1668 ->
  0.1002), both sides fall, and success is unchanged.
- **"alpha near 0.25 with a floor" was the wrong prescription, in two ways.**
  Literally, a lower bound at 0.25 cannot bind, because the measured alphas
  (0.5021, 0.4331) already exceed it. Read as intended — force alpha DOWN —
  it makes things worse: the ratio rises to 4.01x and success rises to 0.4933.
  **Discounting harder transfers relative advantage to the attacker**, because
  the sublinearity recorded above cuts a larger fraction from low-evidence
  opinions, and the honest views are the low-evidence ones.
- **Per-sample granularity does nothing.** Ratio 3.30x against 3.33x, success
  up 0.031. (Note also that per-sample *dependence* is not estimable at all:
  dependence is a statistic across samples. The row uses a per-sample agreement
  proxy.)

**What this means, revised.** The binding constraint is not the SHAPE of the
correction but its SELECTIVITY. An oracle sweep — alpha applied only to the
compromised views, using ground-truth knowledge, an upper bound no deployable
mechanism can reach — shows the ordinary proportional mechanism does bind once
it is aimed: alpha 0.05 inverts the margin to 0.88x and cuts success 0.4813 ->
0.1840. It has to be extreme (a 20x reduction) and the attack still lands 18%
of the time, but it binds, and no blanket shape does.

**Consequence.** This moves the problem onto detection, and L5 measures that
detection at AUC 0.6328 for k=2 and 0.4870 — chance — for k=3. A correction that
must concentrate 20x on the right two views cannot be driven by a signal that is
at chance about which two they are. L3/L4, this extension, and L5 therefore
close a loop rather than pointing at a fix.

## L5. The suspicion detector inherits the dependence signal's blind spot, and inverts under missing evidence

**Component:** `prismflow/statistics/suspicion.py`, exposed as an alert by
`prismflow/models/defended.py`. The detector changes no prediction: it is an
alert, not a correction.

**Evidence:** Part 10, `experiments/comparison/` (5 seeds, ROC AUC against the
same seed's clean scores, thresholds calibrated per seed on clean data; paired
within-seed comparisons; `results/comparison/summary.md`).

- **Detection degrades monotonically with attack strength:** AUC 0.7215 (k=1)
  -> 0.6328 (k=2) -> 0.4870 (k=3, at chance), with k=1 above k=3 in 5/5 seeds
  (+0.2345 +/- 0.1115). The strongest attack is the LEAST detectable: at k=3
  accuracy has collapsed to 0.4213 and the detector is indistinguishable from
  a coin.
- **Inverted under missing views:** AUC 0.2568 (30% missing) and 0.1247 (50%
  missing). Scores are systematically LOWER than clean, so the flag fires LESS
  under missingness than under normal operation. At 50% missing it never fires
  at all, 5/5 seeds. A deployment relying on this flag would be most silent
  exactly when a third of its evidence is gone.
- **False-positive on honest redundancy:** `clone_k2` (duplication, no attacker)
  scores AUC 0.8743 +/- 0.0583, higher than every chorus condition tested
  (+0.2415 +/- 0.1164 against chorus k=2, 5/5 seeds). The detector cannot
  distinguish honest duplication from adversarial collusion — it detects the
  STATISTICAL SIGNATURE of coordinated agreement, not intent.

**Mechanistic link to L3 and L4.** The detector and the discount fail on
near-complementary attacks. Independent PGD succeeds by evading the dependence
signal entirely (L3); the detector catches PGD precisely because it watches that
same signal for agreement unexplained BY dependence (AUC 0.8396 +/- 0.0970,
+0.2068 +/- 0.0608 over chorus k=2, 5/5 seeds). Chorus succeeds partly by
manufacturing the dependence signal itself, and that is what buys it cover from
a detector watching for exactly that signal (AUC 0.6328, degrading further with
k). **Coordination functions partly as a stealth mechanism against
dependence-based detection.**

**What this means.** Detector and discount are complementary failure modes, not
redundant safeguards. Neither is currently validated as a defence: the discount
does not reduce attack success (L3 extension), and the detector is at chance on
the most damaging attack and inverted under missing evidence. Combining them —
an ensemble, or a handoff between the two signals — is untested. It is proposed
future work, not a result of this Part.

**What would change our understanding.** An attacker optimising against the
detector, rather than only against the model, would settle whether the k-trend
above is a ceiling or merely an artefact of an attack that was never trying to
hide. Section 1 of `experiments/comparison/README.md` notes that the evasive
move — raising measured dependence deliberately — appears to be partly available
to the chorus attacker already, for free.

## L6. The shared/private objective does not fix which branch carries cross-view information, and ENIV reads only one of them

**Component:** `prismflow/models/disentanglement_losses.py` and
`prismflow/models/shared_private.py` (V2, `enabled=False` by default).
`SharedPrivatePrismFlow.forward` measures dependence on the `shared` branch when
`dependence_on="shared"`, which is the default and is what Part 11's ENIV came
from. V1 is unaffected: this limitation applies only to a path that is off by
default.

**Evidence:** Part 11 follow-up, `experiments/branch_audit/` (5 seeds, rho in
{0.3, 0.6}, the model's own estimator — cca / pairwise_holdout / 4 null
permutations — applied to four representations of the same trained model;
`results/branch_audit/summary.md`).

Part 11 reported that disentanglement pushes ENIV UP (2.85 -> 3.37 at rho 0.6 in
this audit's measurement configuration) and named one suspected cause. That
cause is refuted and the actual one is different.

- **The proposed cause is refuted: the estimator is blind to it.** Part 11
  suggested the shared branches rotate independently across views, destroying
  measurable structure. Dependence is measured with CCA, which
  `prismflow/statistics/dependence.py` states is "invariant to any invertible
  linear remapping of either side" — and a rotation is one. Applying an
  independent random orthogonal rotation to each view's shared branch moves ENIV
  by **0.0221 on average (max 0.1376, sign inconsistent)**, against an effect to
  explain of **+0.3441 to +0.5203**. An order of magnitude too small, and in the
  wrong shape. `tests/unit/test_branch_audit.py` pins both halves: that the
  rotation is genuinely orthogonal and far from identity (it moves the data by
  up to 31.0), and that CCA changes by 7.6e-05 under it.
- **It is not a consistent branch swap either.** Averaged over seeds,
  dep(private) - dep(shared) is essentially zero: **+0.0047** (rho 0.3) and
  **+0.0031** (rho 0.6) for the orthogonality penalty. Taken on the mean alone,
  nothing is happening.
- **But the per-seed magnitude is large and the SIGN FLIPS.** The same
  difference is **+0.0941 -0.0945 +0.0714 +0.0614 -0.1091** across seeds (rho
  0.3, orthogonality). Each run puts roughly 0.06-0.12 more cross-view
  dependence in one branch than the other; which branch is not consistent. The
  mean is near zero because the population cancels, not because the effect is
  absent. **A 5-seed mean is exactly the statistic that hides this**, and the
  5-SEED RULE is what made it visible.
- **The orientation is set by INITIALISATION, not by the data or the penalty.**
  The sign pattern across seeds 0-4 is `+ - + + -` — **identical for both rho
  values and both penalty methods**:

  | condition | seed 0 | 1 | 2 | 3 | 4 |
  |---|---|---|---|---|---|
  | rho 0.3, hsic_l1 | + | - | + | + | - |
  | rho 0.3, orthogonality | + | - | + | + | - |
  | rho 0.6, hsic_l1 | + | - | + | + | - |
  | rho 0.6, orthogonality | + | - | + | + | - |

  Seeds 0, 2, 3 route the cross-view information into `private`; seeds 1, 4 into
  `shared`. The true dependence in the data (rho 0.3 vs 0.6) does not change it,
  and neither does which penalty is applied.
- **The architecture alone does nothing; the penalty causes it.** With
  `lambda_2 = 0` the branches are indistinguishable — dep(shared) 0.3840 against
  V1's 0.3832 at rho 0.6, and a per-seed difference of sd 0.0045. Branch
  asymmetry scales with penalty pressure: sd **0.0045 -> 0.0430 -> 0.0988** for
  lambda0 -> hsic_l1 -> orthogonality, and the apparent loss of dependence on
  `shared` tracks it, **0.000 -> 0.068 -> 0.178** below V1.
- **Most of the "missing" dependence is recoverable, so it was moved rather than
  destroyed.** Measuring the same models on `concat(shared, private)` recovers
  **75.5-76.0%** of the gap to V1 for HSIC and **47.8-55.8%** for orthogonality,
  and cuts the ENIV inflation by **76.0-76.5%** and **37.3-55.1%** respectively.

**What this means.** The disentanglement objective requires only that `shared_v`
and `private_v` be statistically independent, and independence is SYMMETRIC in
its two arguments. Nothing in the loss distinguishes the branch named "shared"
from the branch named "private" — the two are interchangeable optima, and
initialisation decides which one a run lands in. The names are aspirational, not
enforced. ENIV is then read off whichever branch happens to be labelled
"shared", so in the runs where initialisation routed cross-view information to
`private`, the estimator is reading a representation that genuinely does not
contain it and correctly reports few dependencies — about a representation
nobody should have been asking.

**Part 11's ENIV inflation is therefore substantially a measurement artefact,
but not entirely one.** Reading both branches removes 37-76% of it depending on
penalty and rho. The residual is a real reduction in recoverable cross-view
dependence, so the penalty does distort the representation as well as
relabelling it, and the direction is still the dangerous one: less measured
dependence, higher ENIV, higher confidence on data that has not become more
independent.

**Scope.** This does not touch V1 or any default run. It is a reason not to
enable V2 as it stands, and a specific defect to fix before the idea is retried.

**What would change our understanding.** An objective that breaks the symmetry —
a cross-view alignment term on the shared branches, which would make "shared"
mean "agrees with other views' shared" rather than merely "independent of my own
private" — is the obvious repair and is untested. Part 11's README proposed it
for the wrong reason; it remains the right experiment. Two further checks are
unrun: whether the routing varies per VIEW within a run as well as per run
(`both` exceeding each individual branch in most seeds hints that it does, but
the per-pair matrix was not retained), and whether `dependence_on="features"`,
which already exists in the config, removes the artefact entirely by never
reading a single branch.

**Measurement note.** Absolute ENIV here is not comparable to Part 11's table.
`evaluate()` computes ENIV per batch and sample-weight-averages it
(`prismflow/evaluation/protocol.py`), whereas this audit pools the test split
into one estimate; CCA's finite-sample bias and its permutation chance
correction both depend on n. The ordering is reproduced (v1 < hsic_l1 <
orthogonality at both rho), and every comparison above is within-audit, where n,
seeds and estimator settings are identical across the four representations.

## L7. Tail dependence is not the sharper instrument the argument predicted; it inverts under strong attack

**Component:** `prismflow/statistics/tail_dependence.py` and `copula.py`, added
in Part 13. Neither is wired into any default path: no model, discount or
detector reads them. This limitation is a reason not to adopt the variant, not a
defect in a running system.

**Evidence:** Part 13, `experiments/tail/` (5 seeds, Part 09's chorus
configuration — k = 2 of 4, beta 1.0, 30 steps; test split 1200, 60 exceedances
at q = 0.95; `results/tail/summary.md`).

- **The predicted result does not occur.** The Part was built expecting rho_bar
  to stay flat while lambda_U rose with attack strength. On the colluding pair,
  rho_bar reaches 0.7218 at epsilon 0.2 and holds 0.71-0.83 at every strength,
  peaking at **0.8322 at epsilon 2.0**. lambda_U peaks at **0.7767 at epsilon
  0.5** then collapses to **0.2367 at epsilon 2.0** — a fall of 0.54 while
  attack success rises 0.23 -> 0.52.
- **At full strength the tail measure barely discriminates.** The
  colluding-to-honest ratio at epsilon 2.0 is **2.60x for rho_bar against 1.29x
  for lambda_U**.
- **The mechanism is measured.** Spread inside the colluding views' own upper
  tail (coefficient of variation) falls **0.2383 -> 0.0747** across epsilon
  0.5 -> 2.0, tracking lambda_U at **r = 0.939** over the attacked cells
  (**r = 0.847** pooled per-seed, n = 20), while the same spread against rho_bar
  is **r = -0.245**. A strong attack drives the colluding views to a
  near-constant extreme; which samples land in the top 5% then becomes arbitrary
  and co-exceedance reverts toward the independence baseline.
- **The root cause is the property that made copulas attractive.** The rank
  transform divides out each marginal by construction. This attack's signature
  is largely a shift in LEVEL, so a scale-free measure discards a large part of
  the evidence. Pearson on class-conditional residuals retains it.
- **Tail-aware ENIV moves the wrong way.** Standard ENIV falls monotonically
  3.1171 -> 2.4984, detecting the collusion. Tail ENIV falls to 2.2398 at
  epsilon 0.5 then **rises to 3.3324 at epsilon 2.0, above its own clean value
  of 2.8720**. Driving the discount from lambda_U would weaken it where the
  attack succeeds most.
- **The tail flag has no detection signal.** TPR at 5% FPR is **0.038-0.068**,
  indistinguishable from the false-positive rate. AUC runs 0.4846 -> 0.5856
  against the Part 10 flag's 0.5769 -> 0.7085.

**What this means.** The blindness this Part set out to demonstrate is real in
theory and absent in this system's failure mode. The chorus attack does not hide
in the tail while leaving average co-movement untouched; it raises both, and
then destroys the tail statistic's own basis by flattening the extremes. Part
09's conclusion stands unchanged: the dependence signal detects collusion
reliably, and the correction driven from it is what fails (L4). Tail dependence
does not repair that, and the lambda_U-driven discount would make it worse.

**A separate trap, recorded because it would mislead anyone reading a single
number.** lambda_U = 0 for a Gaussian copula is ASYMPTOTIC, and the approach is
slow: at rho = 0.9 the estimate is 0.685, 0.628, 0.570, 0.530 at q = 0.90, 0.95,
0.99, 0.995. A single lambda_U at a single threshold does not demonstrate tail
dependence — only the decay curve does, contrasted against a t copula, which
flattens (0.468 -> 0.400). Pinned in `tests/unit/test_tail_dependence.py`.

**What would change our understanding.** The copula fit does find real structure
at moderate attack (t beats Gaussian by 178.81 +/- 56.70 loglik, df 4.8, at
epsilon 0.5; on clean data the gain of 23.25 +/- 34.99 overlaps zero), so the
machinery works and it is the strong regime that has no signature left. An
attack that coordinates views WITHOUT saturating them — constrained to preserve
within-tail variance — would settle whether lambda_U is blind in general or only
to this attack's shape. Two further gaps: lambda_U is measured on a scalar
summary rather than the full K-dimensional evidence, and it is not explained why
the per-sample tail flag is weakest exactly where the batch-level lambda_U is
strongest.

**Caveat on the control.** The honest-pair baseline is not inert — rho_bar among
honest pairs rises 0.113 -> 0.320 as the prediction flips to the attacker's
target and conditioning on that class correlates the honest views. This is the
collider effect `dependence.py` documents, and the comp/honest ratios above
inherit it.

## L8. Detection and correction fail together, because both read cross-view belief agreement

**Component:** the ENIV discount (`prismflow/models/prismflow.py`, via
`statistics/dependence.py::feature_dependence_matrix`) and the deployable
detection gate (`experiments/synthesis/`, via
`statistics/suspicion.py::pairwise_agreement`). These are two DIFFERENT
estimators in two different modules — the discount reads a batch-level matrix
built from features and evidence against a permutation null, the gate reads
per-sample cosine agreement between belief vectors. What they share is the
quantity, not the code: both are functions of cross-view belief agreement.
That makes the joint collapse below a property of the quantity rather than an
implementation artifact, which is the stronger reading. (`dependence.py` does
feed one gate signal — the Part 10 `detector` — but that signal is EXCLUDED
from the deployable gate for want of a label-free sign, so it is not the link.)

**Evidence:** Part 14, `experiments/adaptive/` (5 seeds, gamma in
{0, 1, 2, 5, 10, 20}, BPDA-enabled, transfer and gradient-free controls
verified healthy; `results/adaptive/`) and `experiments/synthesis/`
(16-cell matrix, `results/synthesis/gate_matrix.json`).

- **One objective evades both defences at once.** At gamma = 5 the attacker
  gains simultaneously on three axes previously investigated as separate
  questions: attack success **+0.0333 +/- 0.0237** (5/5 seeds up), measured
  dependence **-0.1057** (5/5 down), gate AUC **0.6743 -> 0.5806** (5/5
  down). The objective contains no gate term at any gamma.
- **The gate falls without being targeted.** Across the sweep gate AUC runs
  0.6743 -> 0.6135 -> 0.6109 -> 0.5806 -> 0.5186 -> 0.4874 against measured
  dependence 0.8120 -> 0.7456 -> 0.7483 -> 0.7063 -> 0.4223 -> 0.3343.
  Pearson r = **0.9578** across the six levels.
- **They co-vary strongly but NOT proportionally.** Normalised to their own
  ranges the two curves diverge by up to **0.28**. The gate degrades earlier
  (already -0.0608 at gamma = 1, where dependence has moved only -0.0663 of a
  much larger range) while dependence collapses later and faster (-0.2840 in
  the single step gamma 5 -> 10, against -0.0620 for the gate). A claim that
  the two track each other "almost exactly in shape" is not supported; what is
  supported is that both fall monotonically under one pressure, with the gate
  more sensitive early and dependence more sensitive late.

**What this means.** Detection and correction were investigated as separate
questions, reported in separate documents, and are not separate defences. A
future correction to either must account for the other degrading under the
same pressure, or it will claim an independence that does not exist. An
adversarially robust successor needs a second signal derived from something
other than cross-view agreement — not a differently-weighted combination of
the same one.

**What would change our understanding.** A genuinely orthogonal signal. L7
ruled out tail dependence. `clique_contrast` (Part 14) survives label-free
calibration but is computed from the same `pairwise_agreement` array as the
gate's `dependence` signal — a different aggregation of one measurement, which
is the likely reason it collapses in step. Untested directions: per-view
reconstruction error, provenance or timestamp metadata, and out-of-distribution
scoring on raw inputs — none of which read cross-view correlation at all.

**Caveat on one arm, and on an earlier overstatement.** The `gate_check` arm
was built to apply gate-specific pressure and DID NOT: it returned numbers
identical to the gradient-free `random_search` arm on all 5 seeds and all 3
metrics, because at epsilon 2.0 random perturbation drives measured dependence
to 0.0752, below every seed's tau, leaving `max(0, rho - tau)` inactive so the
gamma = 20 objective reduces exactly to gamma = 0. It is a null arm and cannot
be cited as evidence that a gate-aware attacker has nothing left to take; that
question is untested. The claim above rests on the sweep's gate column alone.

Separately, "the gate is at or below chance in 15 of 16 conditions" overstates
the matrix and should not be repeated. Measured: **10 of 16** cells at or below
0.52, **3** between 0.52 and 0.60, and **3** at or above 0.60 — the last three
all in the `chorus_k3` column (0.668 unstressed, 0.637 at missing_30, 0.624
under noise). The accurate statement is that only the `chorus_k3` column rises
meaningfully above chance anywhere, and only its unstressed cell clears 0.60 on
4 of 5 seeds.

## L9. ENIV is validated only on synthetic data, where rho is a parameter we set

**Component:** the ENIV estimator (`prismflow/eniv/`), and every number in this
project that depends on it.

**Evidence:** Part 05, `experiments/eniv_validation/` (5 seeds x 6 rho levels =
30 fits, `results/eniv_validation/estimator_vs_truth.csv`,
`results/eniv_validation/summary.json`). `docs/DATASETS.md` states the same
boundary from the data side.

The validation sweep is the ONLY evidence in this project that ENIV measures
what it claims to measure, and it exists only because the synthetic generator
lets us choose `rho` and compute `analytic_n_eff` in closed form. The sweep
covers a single design point: **n_views = 4**, rho in {0.0, 0.2, 0.4, 0.6, 0.8,
0.95}, 5 seeds each.

- The model's own estimator (`features_cca_holdout`) tracks the truth in RANK
  but not in VALUE: correlation with truth **0.9922**, mean absolute error
  **0.5220 +/- 0.2998** effective views. Against a true ENIV of 4.0 at rho = 0
  it reads **2.9951 +/- 0.0486**; against a true 1.039 at rho = 0.95 it reads
  **1.8104 +/- 0.0702**.
- The error is not noise around the truth, it is COMPRESSION toward the middle
  of the range: signed error runs **-1.0049** at rho = 0, then **+0.2557,
  +0.7232, +0.8831, +0.8756, +0.7714** as rho rises. The estimator understates
  independence when views are independent and overstates it when they are not.
- The best-case variant (`views_cca`, run on raw views rather than encoder
  features) reaches MAE **0.1919 +/- 0.1494** at correlation **0.9965**, so the
  compression is partly attributable to the encoder, not to ENIV alone. That
  variant is not what the model uses.

**What this means.** Every ENIV, efficiency ratio and discount magnitude
reported on real data in Part 15 is a measurement taken with an instrument
calibrated at one view count on one generator. `RealMultiViewDataset.rho_matrix`
is `None` and `tests/unit/test_real_datasets.py` asserts it, so the code cannot
quietly consume a measurement as a target -- but the reader can, and should not.
Claims of the form "these views carry N independent views of evidence" are
licensed only up to an error of roughly half an effective view, in a direction
that depends on where the true value sits.

**What would change our understanding.** A generator with analytically known
rho at other view counts (6, 8, 16) to test whether the compression scales with
n_views; and a real dataset with an independently established dependence
structure, which we do not have. See [[L10]] for the finite-sample component of
this error and [[L14]] for the single-dataset limitation.

## L10. Dependence is estimated from finite batches, and the estimator is biased

**Component:** `prismflow/statistics/dependence.py::feature_dependence_matrix`,
which every downstream quantity reads -- the discount, the Part 10 detector, the
Part 14 gate.

**Evidence:** Part 05 `results/eniv_validation/estimator_vs_truth.csv` (5 seeds
x 6 rho) and the Part 15 permutation-null control,
`results/real/handwritten/redundancy.json` (5 seeds).

- **The clean measurement of downward bias is the independence null.** Part 15
  permutes each view's encoder features within class, which removes cross-view
  coupling while leaving sample size, feature dimension, marginals and class
  structure untouched. On 6 genuinely independent views at n = 800, ENIV reads
  **5.5094 +/- 0.0777**, not 6.0. That shortfall of **0.49 effective views** is
  pure finite-sample artefact, measured, not estimated.
- **The paired rho_bar null says the mean bias is already corrected.** Under
  the same permutation rho_bar reads **0.0508 +/- 0.0116**, near zero. So the
  chance correction works on the pairwise quantity, and the residual deficit
  above is the eigenvalue form's own floor.
- **The bias is SIGNED, not uniformly downward.** Measured rho_bar against
  true rho on the synthetic sweep: **+0.3345** at rho = 0, **+0.2144** at 0.2,
  **+0.0860** at 0.4, then **-0.0374** at 0.6, **-0.1508** at 0.8, **-0.2203**
  at 0.95. It crosses zero between rho = 0.4 and rho = 0.6. Describing this as
  "a downward bias" is correct only for rho >= 0.6; below that the estimator
  reports dependence that is not there.
- **Seed variance is small relative to that bias.** Across-seed std of rho_bar
  is **0.0075 to 0.0235** over the whole sweep -- an order of magnitude below
  the bias at the extremes. Running more seeds does not reduce the error that
  matters here.

**What this means.** The dominant error in the dependence signal is systematic,
not statistical, so the 5-SEED RULE protects against the smaller of the two
problems. In the regime where PrismFlow is most often evaluated (rho = 0.3,
used by Parts 11-14), the estimator sits on the upward-bias side of the
crossing: it over-reports dependence, so the discount is applied slightly too
hard. Any claim that rests on the absolute level of measured dependence -- as
opposed to its direction of change under an intervention -- inherits this.

**What would change our understanding.** A batch-size sweep holding rho fixed,
to separate the sample-count component from the estimator's structural floor.
Not run: every experiment here uses `evaluation.batch_size = 512` and the
comparison was never made. Related: [[L9]], [[L11]].

## L11. Tail dependence is unreliable below 50 exceedances, and the threshold bites

**Component:** `prismflow/statistics/tail_dependence.py`, where
`MIN_TAIL_SAMPLES = 50` and `TailEstimate.reliable` is
`n_tail >= MIN_TAIL_SAMPLES`.

**Evidence:** Part 13, `experiments/tail/` (5 seeds,
`results/tail/tail.json`), including a pre-registered split-size control.

- At the Part 13 test split (8000 samples, 1200 in the evaluated slice) the
  headline quantile q = 0.95 leaves **n_tail = 60** exceedances -- clearing the
  threshold by 10. At Part 09's split size of 300 it leaves **n_tail = 15**,
  which the code flags `reliable = False`.
- **The failure is dispersion, not displacement.** At n = 300 the lambda_U
  means barely move (eps 0.0: **0.1372 +/- 0.0611** at n = 1200 against
  **0.1400 +/- 0.0559** at n = 300; eps 1.0: **0.0961 +/- 0.0495** against
  **0.1000 +/- 0.0377**). A reader checking only the mean would conclude the
  smaller split was fine. The std is 40-60% of the mean in both cases, which is
  the actual problem, and it is already large at n = 1200.
- The margin is thin by construction: 60 exceedances against a floor of 50 is
  a 20% margin, and it comes from a deliberately generous 1200-sample slice.
  Any real deployment slicing per-class or per-condition falls under the floor
  immediately -- Part 15's real-data audit had to drop the headline quantile to
  **q = 0.9** to reach **n_tail = 80** on 800 test samples.

**What this means.** Tail dependence is the most sample-hungry quantity in the
project and the one with the least statistical headroom. Its negative result in
Part 13 ([[L7]]) is safe, because an underpowered estimator failing to detect
is consistent with both "no signal" and "not enough data" -- but the converse
does not hold. A future POSITIVE tail result at n_tail near 50 should not be
believed without a larger split.

**What would change our understanding.** Re-running Part 13's detection arm at
4x the split size. If tail AUC stayed at the measured **0.4846 +/- 0.0022 to
0.5856 +/- 0.0126**, the negative result would be confirmed as substantive
rather than underpowered.

## L12. The attacker capabilities that defeat the defence are specific, and only partly tested

**Component:** the whole defence -- discount and gate together ([[L8]]).

**Evidence:** Part 14, `experiments/adaptive/` (5 seeds, k = 3, eps = 2.0,
beta = 1.0, 30 steps, BPDA through the dependence estimator;
`results/adaptive/adaptive.json`).

What is MEASURED as sufficient to evade:

- **Full white-box access to the model and the dependence estimator**, with a
  differentiable surrogate (BPDA, `prismflow/attacks/adaptive.py:50`). Under
  that access, gamma = 5 raises attack success to **0.6600 +/- 0.1161** from
  **0.6267 +/- 0.1352** while driving measured dependence to **0.7063 +/-
  0.0702** from **0.8120 +/- 0.0120** and gate AUC to **0.5806 +/- 0.0413**
  from **0.6743 +/- 0.0492**.
- **Control of exactly 3 of 4 views** at eps = 2.0. Both the view count and the
  budget are fixed across the entire sweep; neither was varied.

What is MEASURED as insufficient:

- **Gradient-free search.** The `random_search` arm reaches success **0.0773
  +/- 0.0439** against 0.6267 for the gradient attack -- an 8x gap. Gradient
  access is not a convenience here, it is the capability.
- **Transfer from a separate model.** **0.5793 +/- 0.1519**, below direct
  white-box, with dependence and gate barely moved (**0.8151 +/- 0.0298**,
  **0.6530 +/- 0.0461**).

What is **UNTESTED and must not be inferred**:

- **A gate-aware attacker.** The `gate_check` arm was built to supply this and
  is a null: it returns numbers identical to `random_search` on all 5 seeds and
  all 3 metrics, because at eps = 2.0 random perturbation already drives
  dependence to 0.0752, below every seed's tau, so `max(0, rho - tau)` is
  inactive and the gamma = 20 objective reduces exactly to gamma = 0. No
  conclusion about gate-aware attackers is licensed by this project.
- Attacker control of 2 or fewer views, or of more than 3; any budget other
  than eps = 2.0; any view count other than 4; adaptive attacks on real data.

**What this means.** "The discount is partially evadable" is a claim about one
cell -- 3-of-4 views, eps = 2.0, white-box with gradients, no gate term. The
sweep shows the attacker pays for evasion (success falls to **0.4967 +/-
0.1253** at gamma = 20, where dependence is crushed to **0.3343 +/- 0.0172**),
so there is a real trade-off curve, but its shape is known at one operating
point only.

**What would change our understanding.** A working gate-aware arm -- the
straightforward fix is to lower eps in that arm so the penalty is active -- and
a sweep over the number of compromised views.

## L13. V1 detaches the discount, so it corrects inference but never shapes representation

**Component:** `prismflow/eniv/discount.py`, lines 75 and 122:
`alpha = alpha.detach()` in both `shafer_discount` and the evidence-scaling
path. The discount weights carry no gradient, by design and by contract.

**Evidence:** Part 11, `experiments/disentangle/` and
`experiments/branch_audit/` (5 seeds each, `results/disentangle/summary.md`,
`results/branch_audit/summary.md`).

- **What the detached discount cannot do, a penalty can.** Adding an explicit
  representation-shaping term moves measured dependence substantially where the
  discount leaves it untouched: at rho = 0.6, mean dependence is **0.3832 +/-
  0.0231** for v1 and **0.2052 +/- 0.0550** for `v2_orthogonality` -- a drop of
  0.178, roughly half the v1 level. ENIV rises correspondingly, **2.8486 +/-
  0.0690** to **3.3689 +/- 0.1733**.
- **The architecture alone does nothing; the penalty does it.**
  `v2_lambda0` is the V2 architecture with the penalty switched off and it
  reproduces v1 almost exactly (dependence **0.3840 +/- 0.0392** against
  0.3832; ENIV **2.8458 +/- 0.1157** against 2.8486). This rules out the
  architecture as the cause.
- **It is not free, and it is not monotone in the penalty weight.** At
  lambda = 1 accuracy holds (**0.7827 +/- 0.0615** against v1's **0.7940 +/-
  0.0665** at rho = 0.6), but at lambda = 10 and 100 the model collapses to
  **0.3273 +/- 0.0183** with vacuity confidence at **0.0172 +/- 0.0020**. The
  window in which shaping helps without destroying the model was found by
  sweep, not by principle.
- **v1 is the steadiest on ENIV across seeds** (std **0.0632** at rho = 0.6
  against 0.0957-0.2590 for every V2 arm), which is the compensating property
  of not letting the estimator back-propagate.

**What this means.** PrismFlow as evaluated is a post-hoc correction: it changes
what the model concludes from its evidence, never what evidence the encoders
learn to produce. The redundancy it discounts is redundancy it had no hand in
creating or avoiding. This is a deliberate contract choice -- a detached alpha
cannot destabilise training through an estimator that is itself biased
([[L10]]) -- and the stability figure above is the evidence for it. But it
bounds the ceiling: the best this design can do is spend correctly whatever
independence the encoders happen to supply.

**What would change our understanding.** A V3 that lets a calibrated,
low-variance dependence estimate shape training without the lambda = 10
collapse. Part 11 did not find one; `prismflow/train_two_timescale.py` is the
closest attempt and detaches the evidence before the estimator step
(`train_two_timescale.py:250`).

## L14. One dataset family, and five seeds that are five partitions of it

**Component:** the entire real-data evidence base (Part 15,
`experiments/real/`).

**Evidence:** `results/real/handwritten/redundancy.json`,
`results/real/README.md` (5 seeds).

- **There is one real dataset: `handwritten`** (UCI multi-feature, 6 views:
  fou, fac, kar, pix, zer, mor). Every real-data claim in this project rests on
  it.
- **The 5 seeds do not do what 5 seeds usually do.** On synthetic data a seed
  regenerates the dataset; on real data it only re-partitions a fixed one
  (`docs/DATASETS.md` states this in the "what a seed varies" row). The quoted
  spreads -- ENIV **3.2820 +/- 0.0664**, rho_bar **0.5257 +/- 0.0146**,
  accuracy **0.9808 +/- 0.0034** -- are partition variance, NOT dataset
  variance. They say the measurement is stable on this data; they say nothing
  about how it would move on other data.
- **The headline result is plausibly domain-specific.** The measured redundancy
  (6 nominal views carrying **3.28** effective views, efficiency **0.547 +/-
  0.011**) is concentrated in one block of the dependence matrix: pix-fac
  **0.852**, pix-kar **0.823**, fac-kar **0.707**, against fou-kar **0.304**
  and mor-kar **0.291**. Those three high pairs are all pixel-derived
  descriptors of the same glyph. A benchmark whose views come from genuinely
  different sensors need not behave this way.

**What this means.** "A standard 6-view benchmark carries 3.28 effective views"
is a true statement about `handwritten` and a conjecture about anything else.
Reporting a +/- next to it invites the reader to treat it as a sampled estimate
over benchmarks, which it is not. The efficiency ratio in particular should
never be quoted as a property of multi-view data in general.

**What would change our understanding.** The same audit on two or three
unrelated multi-view benchmarks, ideally with views from different sensing
modalities rather than different transforms of one image. The audit code is
dataset-agnostic; only the dataset registry needs extending.

## L15. No human-subject evaluation and no deployment evaluation were run

**Component:** the project's claims about calibration and confidence, which are
evaluated only as numbers.

**Evidence:** none, and that is the entry. No user study, no decision-support
task, no expert assessment, and no deployment of any kind was conducted. This
is stated here rather than omitted because the project's central quantity is a
CONFIDENCE, and confidence is the one output whose value is realised by a human
or a downstream system acting on it.

What was measured instead, and what it does and does not cover:

- **Calibration error against ground-truth labels.** Part 11's table reports
  v1 prob_ECE **0.0637 +/- 0.0137** at rho = 0.3 and **0.0574 +/- 0.0084** at
  rho = 0.6 (`results/disentangle/summary.md`). This establishes that the
  numbers are statistically well-behaved.
- **Confidence flatness under duplication**, the Part 01-02 result ([[L1]]).
  This establishes the mechanism does what it was designed to do.

Neither establishes that a person reading a PrismFlow confidence makes better
decisions than one reading a naive fusion confidence, or that the diagnostics
(`prismflow/app/panels.py`) communicate the dependence structure to anyone. The
Part 16 demo app is an inspection surface, not an evaluated interface: it has
unit tests, no usability evidence.

**What this means.** Every claim in this repository is a claim about
measurements, not about utility. In particular, nothing here supports "this
makes multi-view systems safer to rely on" -- only "this makes one specific
statistical error smaller under conditions we enumerate".

**What would change our understanding.** A decision task where participants see
either naive or PrismFlow confidence on the same colluding-view inputs, scored
on whether they correctly withhold trust. That is a different project, and it
would need [[L14]] resolved first -- a human study on one synthetic generator
would not be worth running.
