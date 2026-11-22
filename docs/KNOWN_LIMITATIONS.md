# Known limitations

Open issues that are measured or suspected but not resolved. Each entry names
the evidence and what would change our understanding of it.

L1-L8 were written as each Part found them and are ordered by discovery, and so
were L16-L32 (L16-L22 in Parts 28-29, L23-L30 in Part 30, L31-L32 in Part 33
from what writing `docs/FINAL_REPORT.md` surfaced). L9-L15
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

## L2. Residual per-pair dependence accumulates with view count in the clone harness

From 333f143: residual ~0.05 per-pair measured dependence accumulates in the
soft cluster size, so an untouched view's alpha falls from ~0.88 to ~0.75 going
from 4 to 8 views (controlled harness). This may contribute to prismflow's
upward confidence drift at rho=0.0 in `clone_eigen_softcluster`
(+0.003, +0.007, +0.010, +0.013 for k=1..4). That link is untested.

**New evidence, Part 36; the entry stays OPEN, with its scope narrowed.**
Residual per-pair dependence was measured against view count on the
analytic-rho generator, at rho = 0.0, where the true per-pair dependence is 0
and the added views are INDEPENDENT rather than clones (5 seeds):

- V = 4: chance-corrected **0.3345 +/- 0.0163**, uncorrected
  **0.0844 +/- 0.0336**.
- V = 6: chance-corrected **0.3139 +/- 0.0033**, uncorrected
  **0.0590 +/- 0.0050**.
- V = 8: chance-corrected **0.3071 +/- 0.0053**, uncorrected
  **0.0511 +/- 0.0106**.
- V = 16: chance-corrected **0.2732 +/- 0.0062**, uncorrected
  **0.0271 +/- 0.0068**.
- Pre-registered rule: ACCUMULATES only if the mean is monotonically
  non-decreasing in V and the V = 16 value exceeds the V = 4 value by more than
  the larger std. Monotone: no; gap **-0.0613** against **0.0163**.
  **Verdict: DOES NOT ACCUMULATE.**
- **The reading, which narrows this entry rather than lifting it.** The
  residual FALLS with view count under both the corrected and the uncorrected
  estimator, so it is not an intrinsic property of the estimator that
  accumulates with width. The accumulation is specific to
  `clone_eigen_softcluster`, where the added views are CLONES -- hence the
  scoped title.
- **Why it stays OPEN.** The causal link to the confidence drift at rho = 0.0
  remains UNTESTED. It needs the `clone_eigen_softcluster` harness and was
  declared out of scope before the run.

**Evidence:** `results/part36_synthetic/analysis_a.json`;
`experiments/part36_synthetic/README.md` section 2.

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
lets us choose `rho` and compute the analytic truth in closed form. The sweep
covers a single design point: **n_views = 4**, rho in {0.0, 0.2, 0.4, 0.6, 0.8,
0.95}, 5 seeds each.

**Truth column.** Every figure in this entry -- the MAE, the correlation, the
signed errors and the Part 36 block below -- is computed against
`analytic_n_eff_eigen`, the eigenvalue form of the analytic truth. The CSV also
carries `analytic_n_eff`, a different column, and the two must never be mixed
in one series: over the same 30 fits this estimator scores MAE
**0.5220 +/- 0.2998** at correlation **0.9922** against `analytic_n_eff_eigen`
but MAE **0.7523 +/- 0.2476** at correlation **0.9069** against
`analytic_n_eff`. An earlier revision of this entry quoted the MAE from the
first column and the signed errors from the second.

- The model's own estimator (`features_cca_holdout`) tracks the truth in RANK
  but not in VALUE: correlation with truth **0.9922**, mean absolute error
  **0.5220 +/- 0.2998** effective views. Against a true ENIV of 4.0 at rho = 0
  it reads **2.9951 +/- 0.0486**; against a true 1.150 at rho = 0.95 it reads
  **1.8104 +/- 0.0702**.
- The error is not noise around the truth, it is COMPRESSION toward the middle
  of the range: signed error against `analytic_n_eff_eigen` runs **-1.0049**,
  **-0.6443**, **-0.2587**, **+0.1117**, **+0.4520**, **+0.6604** at
  rho = 0.0, 0.2, 0.4, 0.6, 0.8, 0.95. The estimator overstates dependence
  when views are near-independent and understates it when they are strongly
  dependent, crossing between rho = 0.4 and rho = 0.6.
- The best-case variant (`views_cca`, run on raw views rather than encoder
  features) reaches MAE **0.1919 +/- 0.1494** at correlation **0.9965**, so the
  compression is partly attributable to the encoder, not to ENIV alone. That
  variant is not what the model uses.

**New evidence, Part 36; the entry stays OPEN.** The analytic-rho generator was
run at V = [4, 6, 8, 16], rho [0.0, 0.2, 0.4, 0.6, 0.8, 0.95], 5 seeds, with
only `n_views` varied from the committed settings. The committed V = 4 point
reproduces first (MAE 0.5220005950719182, correlation 0.9922060509334697, both
to abs diff 0).

- V = 4: correlation with truth **0.9922**, MAE **0.5220 +/- 0.2998**
  effective views, which is **0.1305** of V.
- V = 6: correlation **0.9930**, MAE **0.8514 +/- 0.4575**, **0.1419** of V.
- V = 8: correlation **0.9924**, MAE **1.1662 +/- 0.6368**, **0.1458** of V.
- V = 16: correlation **0.9938**, MAE **2.2995 +/- 1.3063**, **0.1437** of V.
- Pre-registered rule: SCALES if MAE(V = 16) - MAE(V = 4) exceeds the
  across-cell std at V = 4. Measured gap **1.7775** against std **0.2998**.
  **Verdict: SCALES.**
- **The reading.** Rank agreement does not degrade -- correlation is flat near
  0.993 at every view count. The VALUE error grows almost exactly in
  proportion to V, so the single validated design point of n_views = 4 is the
  most favourable one run, and every ENIV reported at a higher view count
  carries a proportionally larger absolute error.
- **Why it stays OPEN.** Only the first half of the closing condition was
  answerable. The second half needs a real dataset with an independently
  established dependence structure, which does not exist in this checkout.

**Evidence:** `results/part36_synthetic/analysis_a.json`;
`experiments/part36_synthetic/README.md` section 1.

**What this means.** Every ENIV, efficiency ratio and discount magnitude
reported on real data in Part 15 is a measurement taken with an instrument
calibrated at one view count on one generator. `RealMultiViewDataset.rho_matrix`
is `None` and `tests/unit/test_real_datasets.py` asserts it, so the code cannot
quietly consume a measurement as a target -- but the reader can, and should not.
Claims of the form "these views carry N independent views of evidence" are
licensed only up to an error of roughly half an effective view, in a direction
that depends on where the true value sits.

**What would change our understanding.** The first half of this condition -- a
generator with analytically known rho at other view counts (6, 8, 16) -- was
run in Part 36 and is reported above: the compression does scale with n_views.
What remains is a real dataset with an independently established dependence
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

**New evidence, Part 36; the entry stays OPEN, sharpened.** Nested estimation
subsamples of one trained model's test split per seed, n in
[64, 128, 256, 512, 1024, 2048], rho fixed at 0.3, 5 seeds, paired within seed.
The sweep varies the ESTIMATION sample count, a deviation from the "batch-size
sweep" wording that was stated in `config.yaml` before the run. Signed ENIV
bias against `analytic_n_eff_eigen`:

- `V4_dependent` (truth 3.10): bias **-1.2548 +/- 0.0781** at n = 64,
  **-0.2581 +/- 0.0232** at n = 2048 (a 4.9x reduction); fitted floor
  **0.2736 +/- 0.0348**, fitted c **65.1 +/- 6.5**, fit residual rms 0.0404.
- `V4_independent_null` (truth 4.00): bias **-2.1469 +/- 0.0454** at n = 64,
  **-0.7493 +/- 0.0352** at n = 2048 (a 2.9x reduction); fitted floor
  **0.7892 +/- 0.0331**, fitted c **91.3 +/- 2.2**, fit residual rms 0.0758.
- `V6_dependent` (truth 4.50): bias **-1.9311 +/- 0.1629** at n = 64,
  **-0.2589 +/- 0.0734** at n = 2048 (a 7.5x reduction); fitted floor
  **0.3220 +/- 0.0599**, fitted c **108.8 +/- 9.0**, fit residual rms 0.0994.
- `V6_independent_null` (truth 6.00): bias **-3.4630 +/- 0.1397** at n = 64,
  **-1.0979 +/- 0.0270** at n = 2048 (a 3.2x reduction); fitted floor
  **1.2255 +/- 0.0271**, fitted c **153.0 +/- 9.6**, fit residual rms 0.1671.
- **Both components are present.** The sample-count component is large and
  real: the bias shrinks by 2.9x to 7.5x between n = 64 and n = 2048. It does
  not reach zero -- at n = 2048 the bias is still many across-seed stds from 0
  in all 4 cells. That part is a measurement, not a fit.
- **What the sweep cannot separate, and this is the sharpening.** The
  residuals of the pre-registered `floor + c / n` fit are sign-ordered rather
  than scattered, with the identical pattern in all 4 cells, and a floor-free
  power law `k * n^alpha` -- one parameter fewer -- fits the same 6 points
  comparably (better in 2 of 4 cells), with exponents **-0.459**
  (`V4_dependent`), **-0.308** (`V4_independent_null`), **-0.585**
  (`V6_dependent`) and **-0.338** (`V6_independent_null`) -- all far short of
  the -1 the pre-registered model assumes.
- **Why it stays OPEN.** The fitted floor is therefore an UPPER BOUND on the
  structural component over this n range, not an identified limit. The
  "structural floor" is supported as "still far from zero at n = 2048" rather
  than as "non-zero in the limit", which is weaker than CLOSED would claim.

**Evidence:** `results/part36_synthetic/analysis_b.json`;
`experiments/part36_synthetic/README.md` sections 3.3 and 3.4.

**What this means.** The dominant error in the dependence signal is systematic,
not statistical, so the 5-SEED RULE protects against the smaller of the two
problems. In the regime where PrismFlow is most often evaluated (rho = 0.3,
used by Parts 11-14), the estimator sits on the upward-bias side of the
crossing: it over-reports dependence, so the discount is applied slightly too
hard. Any claim that rests on the absolute level of measured dependence -- as
opposed to its direction of change under an intervention -- inherits this.

**What would change our understanding.** The sweep this entry asked for was run
in Part 36 and is reported above; it establishes an upper bound on the
structural component but cannot identify it, because over n = 64 to 2048 a
floor-free power law fits as well as a floor. What remains is a sweep reaching
an n where the decay visibly flattens. Related: [[L9]], [[L11]].

## L11. CLOSED 2026-11-05, confirmed not lifted. Tail dependence is unreliable below 50 exceedances, and the threshold bites

**Status: closed as CONFIRMED: the re-run exists and does not lift the
limitation.** L11 asked for Part 13's detection arm re-run at 4x the split
size, with the negative result to be confirmed as substantive if tail AUC
stayed inside the measured band. Part 36 ran it and it did, so the outstanding
question is answered. The entry's operative warning -- that a POSITIVE tail
result at n_tail near 50 should not be believed -- is unchanged, and the
dispersion it names as the real problem is unchanged too.

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

**New evidence, Part 36; this is what closes the entry.** Part 13's detection
arm was re-run at `n_samples = 32000` against the committed 8000 -- a test
slice of 4800 rows against 1200, and n_tail at q = 0.95 of **240** against 60
-- with every other committed setting unchanged, 5 seeds. The committed tail
AUCs reproduce first (0.4846145138888889 at eps 0.2 and 0.5856175694444444 at
eps 2.0, both to abs diff 0).

- eps 0.2: tail AUC **0.4938 +/- 0.0039** at 4x against **0.4846 +/- 0.0022**
  committed; n_tail **240.0 +/- 0.0**; std/mean **0.008**.
- eps 0.5: tail AUC **0.5007 +/- 0.0042** at 4x against **0.4801 +/- 0.0063**
  committed; n_tail **240.0 +/- 0.0**; std/mean **0.008**.
- eps 1.0: tail AUC **0.5554 +/- 0.0087** at 4x against **0.5370 +/- 0.0132**
  committed; n_tail **240.0 +/- 0.0**; std/mean **0.016**.
- eps 2.0: tail AUC **0.5568 +/- 0.0129** at 4x against **0.5856 +/- 0.0126**
  committed; n_tail **240.0 +/- 0.0**; std/mean **0.023**.
- Every epsilon's 4x mean lies inside the band, so the rule returns
  **SUBSTANTIVE**. The negative result of Part 13 is not a power artefact. At
  eps 2.0 the AUC in fact FELL with four times the data, from 0.5856 to
  0.5568.
- **A correction to this entry's own text.** The band was previously quoted
  here as "0.4846 +/- 0.0022 to 0.5856 +/- 0.0126", but the committed means in
  `results/tail/tail.json` are 0.4846 (eps 0.2), **0.4801** (eps 0.5), 0.5370
  (eps 1.0) and 0.5856 (eps 2.0). The quoted floor of 0.4846 excluded this
  entry's own eps 0.5 point, so the span is **0.4801 to 0.5856**. Part 36
  pre-registered and evaluated both the quoted and the committed band rather
  than picking one; the verdict is SUBSTANTIVE under each.
- **The dispersion is unchanged by the 4x.** Across-seed std as a fraction of
  the mean is still 0.008 to 0.023 at four times the split. Quadrupling buys
  exceedances, not agreement between seeds.

**Evidence:** `results/part36_synthetic/analysis_c.json`;
`experiments/part36_synthetic/README.md` section 4.

**What would change our understanding.** Nothing outstanding on the question
this entry posed. A positive tail result at n_tail near the floor of 50 would
still need a larger split before it was believed.

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

## L16. On real cross-scenario data, discrimination and calibration come apart, and the positive class is one or three episodes

**Component:** the real-data evidence base for window-level detection
(Part 28, `experiments/ait_clean/`).

**Evidence:** `results/ait_clean/results.json`,
`results/ait_clean/summary.md` (5 seeds, two directions, never pooled), and
`results/ait_clean/baselines.json` for the non-oracle reference below.

- **Accuracy does not clear the majority-class baseline in direction B.**
  Training on `santos` (60.2% attack) and testing on `russellmitchell`
  (90.0% attack), naive fusion scores accuracy 0.9100 +/- 0.1233 and
  PrismFlow 0.9366 +/- 0.0819 against a baseline of 0.9004. Both bars
  straddle it. Balanced accuracy (0.9500 +/- 0.0685 and 0.9648 +/- 0.0455)
  and ROC AUC (0.9999 +/- 0.0002 for both) clear 0.50 decisively, so the
  systems rank well and threshold badly.
- **The majority-class baseline is better calibrated than either system in
  direction B -- but it is an ORACLE reference, not a competitor.** It is
  handed the test set's own class prevalence and emits it as a constant
  forecast, so its Brier is exactly `2p(1-p)` by construction: 0.479198 at
  p = 0.601986 and 0.179301 at p = 0.900437, reproduced to machine precision
  in `results/ait_clean/results.json`
  (`directions.<name>.majority_class_baseline.brier`). Its ECE is 0.0000 for
  the same reason -- a constant forecast equal to the realised base rate is
  calibrated by definition. A deployed system does not know the test
  prevalence, so beating this baseline on Brier or ECE is not a bar a real
  detector must clear, and failing to beat it is not by itself a defect. It
  is reported because it bounds what a calibration number on this data can
  mean, not because it is a rival.
- **The non-oracle reference tells the same story and is a fair comparison.**
  `flat_all_views`, a logistic regression that sees only the training
  scenario, scores Brier **0.1264** and ECE **0.0765** in direction B
  (`results/ait_clean/baselines.json`,
  `directions.B_santos_to_rm.baselines.flat_all_views.brier` and
  `.prob_ece`), against naive fusion's 0.2237 +/- 0.0768 / 0.2066 +/- 0.0602
  and PrismFlow's 0.1884 +/- 0.0688 / 0.1921 +/- 0.0478. So both fusion
  systems are worse calibrated under this prevalence shift than a flat model
  with no oracle knowledge, and the ECE gap sits outside both systems'
  one-std ranges. See [[L17]].
- **Direction A does not show this**, and the two directions must not be
  averaged. On `santos` (prevalence 0.602) both systems beat the
  majority-class baseline on every metric, and PrismFlow beats naive fusion
  in 5/5 seeds on balanced accuracy (+0.0043 +/- 0.0015), Brier
  (-0.0111 +/- 0.0043) and ECE (-0.0120 +/- 0.0066).
- **The positive class is 3 episodes (`santos`) and 1 episode
  (`russellmitchell`).** Every +/- in Part 28 is a spread over 5
  initialisations on a fixed pair of scenarios, not over independent attacks.
- **The task is "was the DNS exfiltration tool running".** 1,026 of 1,031
  and 663 of 667 attack windows carry only dnsteal-family steps, and the
  multi-step intrusion is 4 and 3 windows respectively. No Part 28 metric is
  computed over those windows.

**What this means.** Part 28 is evidence that the pipeline runs on real
multi-sensor log data, and that the dependence estimator reads an eigen ENIV
of **2.9579 +/- 0.0638 to 3.4958 +/- 0.1015** across the two test scenarios
and two systems (design-effect ENIV **1.9982 +/- 0.0686 to
2.4811 +/- 0.1234**; `results/ait_clean/results.json`,
`directions.<name>.systems.<system>.diagnostics_summary.eniv_eigen` and
`.eniv_design_effect`). **ENIV is a dependence statistic, not a count of
informative views**: it measures how much the five views' encoder features
covary, and it is blind to whether a view carries any signal about the label
at all. A view that is pure noise is maximally independent of the others and
raises ENIV rather than lowering it ([[L3]]), and on this data four of the
five views are close to useless on the label while ENIV still reads near 3
([[L17]]). An ENIV of 3.0-3.5 must therefore never be read as "three to three
and a half views are contributing". Part 28 is not evidence of intrusion
detection, and in direction B it is not evidence that either system is usable
as a detector at a fixed threshold.

**What would change our understanding.** A threshold chosen on the training
scenario and reported as such, rather than argmax at 0.5; a third and fourth
scenario so the direction count exceeds the episode count; and a window
length at which more than one attack episode exists.

## L17. On AIT the label is solved by one view, so Part 28 is not evidence about fusion

**Component:** every Part 28 comparison between a fusion system and anything
else (`experiments/ait_clean/`), and the premise of Part 29 on this data.

**Evidence:** `results/ait_clean/baselines.json` (deterministic logistic
regression, one value per row, no seed spread -- see
`experiments/ait_clean/README.md` section 9), with the fusion rows quoted from
`results/ait_clean/results.json` (5 seeds).

- **One view solves the label.** `single_view_dns` -- a logistic regression on
  the nine `dns` features and that view's mask indicator, nothing else --
  scores balanced accuracy **0.9898** and ROC AUC **0.9993** on `santos`
  (prevalence 0.602, n = 1,108) and balanced accuracy **1.0000** and ROC AUC
  **1.0000** on `russellmitchell` (prevalence 0.900, n = 1,145)
  (`directions.<name>.baselines.single_view_dns.balanced_accuracy` and
  `.roc_auc`). On `russellmitchell` it is 1.0000 on ROC AUC, balanced
  accuracy, macro-F1, attack recall and benign recall simultaneously.
- **No other single view clears balanced accuracy 0.71 in either direction.**
  The best non-`dns` view is `suricata`, at 0.7061 on `santos` and 0.5115 on
  `russellmitchell`. `apache`, `auth` and `audit` sit between 0.3733 and
  0.5088 across both directions.
- **`apache`'s ROC AUC is below 0.5 in both directions**: 0.4258 on `santos`
  and 0.3425 on `russellmitchell`. Below chance on both test sets, fitted on
  the other scenario each time.
- **The flat model matches or beats both fusion systems.** In direction A,
  `flat_all_views` (all five views' features concatenated plus the five mask
  indicators) scores balanced accuracy **0.9804** and Brier **0.0295**,
  against naive fusion's 0.9444 +/- 0.0056 / 0.0819 +/- 0.0092 and
  PrismFlow's 0.9487 +/- 0.0061 / 0.0707 +/- 0.0071 -- above both by more
  than one 5-seed std on each. In direction B it scores balanced accuracy
  **0.9588**, inside both systems' one-std ranges (0.9500 +/- 0.0685 and
  0.9648 +/- 0.0455), so no separation there, with Brier **0.1264** and ECE
  **0.0765** against 0.2237 +/- 0.0768 / 0.2066 +/- 0.0602 and
  0.1884 +/- 0.0688 / 0.1921 +/- 0.0478.
- **The single-view `dns` model matches or exceeds both fusion systems on
  balanced accuracy in both directions**: 0.9898 against 0.9444 +/- 0.0056
  and 0.9487 +/- 0.0061 on `santos`; 1.0000 against 0.9500 +/- 0.0685 and
  0.9648 +/- 0.0455 on `russellmitchell`. It also matches or exceeds
  `flat_all_views` on balanced accuracy in both (0.9898 against 0.9804;
  1.0000 against 0.9588).

**What this means. Part 28 gives no evidence that multi-view fusion adds
anything on this label.** A single-view logistic regression on `dns` equals or
beats every fused system reported, and the data card predicted exactly this:
the exfiltration is visible directly in `dns` as query names up to 250
characters carrying the exfiltrated file name
(`docs/part27-ait-datacard.md` section 9, limitation 2). This is a property of
the label, not a result about the architecture -- a task one sensor can solve
cannot discriminate between ways of combining five. It does not show that
fusion is useless in general, and it does not bear on the project's thesis,
which concerns redundancy among views that all carry signal
([[L3]], [[L14]]). It does mean no Part 28 number should be cited as fusion
performance.

**It also conditions Part 29.** Part 29 attacks k colluding views on this
data. Because one view carries the label, **the result will depend almost
entirely on whether the compromised set includes `dns`**: compromising `dns`
attacks the only view with signal, while compromising any subset of `apache`,
`auth`, `audit` and `suricata` perturbs views that score 0.3733 to 0.7061
balanced accuracy on their own. Those are two different experiments and must
be reported as separate conditions with the compromised set named, not pooled
into a single "k views compromised" row. A Part 29 attack-success number that
does not say whether `dns` was in the compromised set is uninterpretable.

**What would change our understanding.** A label whose signal is spread across
views -- so that no single view solves it and fusion has something to do --
or more attack episodes, so the comparison rests on more than the 3 and 1
independent blocks these two scenarios provide. Either would make a
fusion-against-flat comparison on real data informative; neither exists in
the AIT scenarios loaded so far.

## L18. On real data the detector inverts under attack, and the protocol cannot speak to deployment

**Component:** the suspicion detector measured under attack on real data
(Part 29, `experiments/ait_attack/`), and the measurement design it is
measured by.

**Evidence:** `results/ait_attack/results.json`
(`directions.<name>.systems.prismflow.cells.<cell>.summary.detector_auc`,
5 seeds, two directions, never pooled), tabulated in
`experiments/ait_attack/README.md` sections 7 and 9.

- **The majority of all conditions sit below chance.** The detector scores
  below AUC 0.5 in **129 of 187** conditions in direction A
  (`russellmitchell` -> `santos`) and **159 of 187** in direction B
  (`santos` -> `russellmitchell`); the per-cell 5-seed mean AUC ranges
  **0.2384..0.7133** in A and **0.1314..0.6220** in B. The flag therefore
  fires LESS under attack than under clean operation across most of the grid.
- **Every window in the evaluated batch is attacked.** `attack: batch_size`
  is 1200 against n_test 1108 (A) and 1145 (B), so each direction is one
  batch and the perturbation is applied to all of it. There is no clean
  population inside an attacked run.
- **The AUC compares an attacked run with the clean run of the same
  windows.** It is a between-run comparison at fixed width, not a
  within-batch separation of contaminated from clean cases.
- **Partially contaminated batches were not tested.** No cell mixes attacked
  and clean windows and no cell varies the contaminated fraction; that axis
  is absent from the grid and is reported as not measured.

**What this means.** The below-chance AUCs are a real and reproducible
property of this all-or-nothing protocol, and **no claim about deployed
detection rates follows from them in either direction**. A deployed detector
sees a mostly-clean stream and must flag a small contaminated part of it;
that is the regime this Part does not enter. This extends [[L5]], which
describes a detector degrading to chance and inverting under missingness, by
showing the inversion on real data under attack as well -- and by bounding
what the measurement can mean.

**What would change our understanding.** A contaminated-fraction axis in the
grid, so the detector is scored on batches that are partly clean; and a
within-batch decision rule, so the AUC separates cases rather than runs.

## L19. The coordination term's effect depends on the view and the direction

**Component:** the chorus attack's agreement term `beta`
(`prismflow/attacks/chorus.py`, `chorus_objective`) as it acts on real data
(Part 29, `experiments/ait_attack/`).

**Evidence:** `results/ait_attack/results.json`, paired WITHIN SEED at
`chorus k=2 eps=1.0` and only then aggregated; tabulated per pair in
`experiments/ait_attack/README.md` section 11 and its sub-sections 11.1
and 11.2.

- **Raising `beta` from 0 to 1 increases attack success on all four
  `dns`-pair families in all four units, 5/5 seeds.** In direction B the
  paired shift is **+0.0786..+0.6837** for naive fusion and
  **+0.2795..+0.8225** for PrismFlow across the four `dns` pairs.
- **For `dns` pairs, PrismFlow minus naive changes sign with `beta`.** It is
  negative at `beta=0` in **8 of 8** pairs and positive at `beta=1` in
  **7 of 8**; the difference-in-differences -- the `beta` shift measured on
  PrismFlow minus the same shift measured on naive -- is positive in
  **8 of 8**. The four pairs per direction share the `dns` view and the same
  seeds, so these are **not independent** observations.
- **In direction B the `suricata` pairs without `dns` move the other way**,
  falling by about **0.65** (naive) and **0.70** (PrismFlow) as `beta` rises.
  In direction A the `suricata` family spans both signs, so the opposition is
  a direction-B result only.
- **No mechanism is measured.** No per-view alpha and no belief margin is
  recorded per cell, so nothing here says why `beta` moves a pair either way.

**What this means.** There is no single "effect of `beta`". A figure averaged
over the ten pairs would average opposed behaviours and describe neither, and
a figure averaged over the two directions would hide that the opposition only
holds in one of them. Any claim about the coordination term must name the
pair family and the direction it was measured in ([[L17]]).

**What would change our understanding.** A per-view alpha and belief-margin
trace per cell, which would let the sign be attributed rather than observed;
and a label whose signal is not carried by one view, so the `dns` and
`suricata` families are not two different experiments.

## L20. Part 09's L3 extension is only partially reproduced on AIT

**Component:** the comparison between PrismFlow and naive fusion under the
chorus attack on real data (Part 29, `experiments/ait_attack/`), against
Part 09's finding that PrismFlow was strictly worse in 12/12 attacked cells.

**Evidence:** `results/ait_attack/results.json`, `prismflow - naive` attack
success paired WITHIN SEED on the same cell key; tabulated and de-duplicated
in `experiments/ait_attack/README.md` section 9 under [[L3]]. The
de-duplication key is `(direction, cell_key)`, because the `eps=1.0 beta=1.0`
group and the epsilon sweep select identical cells.

- **Over 280 distinct chorus cells, both signs appear.** PrismFlow is higher
  than naive -- Part 09's sign -- in **118** cells, **60** of them with all 5
  seeds agreeing; it is lower in **162**, **95** of them unanimous. Nothing
  here is 12/12 either way.
- **Over 12 strata (sub-row x direction), the `dns` split flips the sign in
  2.** Read by majority instead of by presence, `dns`-outside is **never**
  higher-dominant (**0 of 12**).
- **Part 09's sign carries a unanimous majority in 5 rows, all of them with
  `dns` inside the compromised set**; the opposite sign carries one in 5 rows
  (3 with `dns` outside, 2 inside).
- **The cells are nested and share seeds.** They are nested compromised
  subsets evaluated on the same test windows with the same 5 seeds and the
  same trained models, so every count above is a DESCRIPTION of how the grid
  came out and **not a tally of independent trials**; no p-value or binomial
  reading may be built on it.
- **`pgd` is reported separately and is not evidence on the chorus claim.**
  Part 09's claim is about the chorus attack, so no count mixes the two
  attacks.

**What this means.** Part 09's blanket statement -- that reading the
dependence axis confers no defence -- must not be restated as if it held on
real data, and neither may its negation. Which sign a cell shows is decided
by the `dns` split, the direction and `epsilon`, not by the attack being
chorus. No mechanism is claimed: this Part measures fused attack success in
each arm and nothing inside the discount ([[L4]]).

**What would change our understanding.** Cells that are not nested subsets on
shared seeds, so the counts could support an inferential claim; and a
per-cell record of the discount's internals, so a sign could be explained
rather than located.

## L21. Fixed epsilon is not comparable across views or directions

**Component:** the epsilon grid shared by Part 09 and Part 29
(`experiments/ait_attack/config.yaml`), applied to views of different scale
and reach.

**Evidence:** `results/ait_attack/results.json`
(`directions.<name>.systems.naive.feature_scale.per_view`),
`results/ait_attack/padding_audit.json` for the reach figures, and
`results/ait_clean/baselines.md` for the clean single-view scores.

- **The same nominal epsilon is a different perturbation per direction.**
  `epsilon 1.0` equals **0.639** of the `dns` standard deviation in direction
  A and **1.373** in direction B -- **2.15x** larger relative to that view's
  own spread in B.
- **It is also a different perturbation per view.** In direction B at
  `chorus eps=1.0, k=1`, the fraction of each view's native feature cells the
  attack actually perturbed is **86.9%** for `suricata` against **3.7%** for
  `dns`. These are from a **single seed (seed 0)**, one observation each, and
  are not means.
- **The view with the highest attack success is not the view that carries the
  label.** Compromised alone in direction B, `suricata` reaches attack
  success **0.7541** (naive) and **0.5972** (PrismFlow), while `dns` reaches
  **0.0996 +/- 0.1248** (naive -- carried by one seed, 0.3223 of five, and
  **0.0439 +/- 0.0093** without it) and **0.0356 +/- 0.0231** (PrismFlow).
  On clean data `dns` solves the label and `suricata` barely beats chance:
  balanced accuracy **1.0000** against **0.5115** on `russellmitchell`.
- **Attack success per view is therefore confounded by reach**, and
  matched-reach -- for example a std-scaled epsilon -- is **not measured**.

**What this means.** Attack-success figures are not comparable across views
or across directions at fixed nominal epsilon, so `suricata`'s higher number
is not evidence that it is the better view to attack at equal effort: the two
cells are not equal effort. Clean single-view accuracy does not predict which
view an attacker should pick, which is why [[L17]] cannot be read off this
Part's attack grid.

**What would change our understanding.** An epsilon scaled per view, by that
view's own standard deviation or by a fixed perturbed-cell fraction, so two
views can be attacked at matched effort; and more than one seed for the reach
figures.

## L22. CLOSED 2026-11-05, confirmed not lifted. Part 10's clone comparison is not like-for-like

**Status: closed as CONFIRMED: the equal-width figure exists and does not lift
the limitation.** L22 asked for a clone condition whose clean reference has the
same view count as the attacked model. Part 36 built it and measured it, so a
comparable number now exists. It confirms this entry's direction of concern
rather than retiring it: part of 0.8743 really is the width, and the operative
warning -- that 0.8743 must not be differenced against any fixed-width result
-- is unchanged.

**Component:** Part 10's `clone_k2` detector condition
(`experiments/comparison/`), quoted in [[L5]] as AUC 0.8743 +/- 0.0583.

**Evidence:** confirmed from the code rather than inferred, at
`experiments/comparison/run_comparison.py:88` and `:69`,
`prismflow/data/corruption.py:15-20`, `run_comparison.py:232-234`, `:271`,
`:279-281` and `:292`; set out in `experiments/ait_attack/README.md`
section 8.

- **The attacked model and its clean reference have different widths.**
  `duplicate_view` APPENDS k copies of a view, so `clone_k2` runs
  **6 views** (4 + 2) while the reference it is scored against is the `clean`
  condition at k = 0 and therefore **4 views**.
- **The suspicion score is a MAXIMUM over off-diagonal entries, so width
  matters.** A 6-view model maximises over **30** ordered off-diagonal
  entries (6x5) and the 4-view reference over **12** (4x3) -- 15 against 6
  unordered pairs. A maximum over 30 draws is stochastically larger than one
  over 12 even when every per-pair distribution is identical.
- **A clean clone with equal pair count has now been run**, in Part 36; see
  the block below.
- **The live diagnostic of Part 32 states this limit in every response.**
  `serve/scoring.py` returns `suspicion_comparability` as a literal field on
  every `/score` result, including the branch where no suspicion score could
  be computed, so an uploader cannot read two runs of different widths as
  comparable without first being told they are not.

**What this means.** The **0.8743** figure compares different pair counts, so
part of it is the width change itself rather than detected duplication, and
it must not be differenced against any fixed-width result. Part 29 has no
such asymmetry -- every condition and its clean reference run the same
5-view model -- and it also runs no duplication condition, so the two Parts
can only be compared qualitatively ([[L5]], [[L8]]).

**New evidence, Part 36; this is what closes the entry.** The same width-6
model scored two conditions, so both maximise their suspicion score over the
same **30** ordered off-diagonal entries: ATTACKED (the 4-view dataset with 2
duplicated views appended, exactly as `clone_k2` does) and CLEAN-AT-WIDTH-6
(the 6-view dataset the model was trained on, uncorrupted). Committed Part 10
settings throughout, 5 seeds, paired within seed. The committed mixed-width
figure reproduces bit-for-bit first (0.8743088888888889 +/-
0.05831081226634124, abs diff 0).

- `clone_k2` against the committed 4-view reference (MIXED width):
  **0.8743 +/- 0.0583**.
- `clone_k2` against a clean 6-view reference (EQUAL width):
  **0.7652 +/- 0.0843**.
- Paired difference, equal minus mixed: **-0.1091 +/- 0.0443**, negative in
  **5 of 5** seeds, magnitude 2.5x its own std. **Verdict: WIDTH EXPLAINS PART
  OF THE COMMITTED FIGURE.**
- Mean suspicion score: clean at width 4 **0.3590 +/- 0.0394**, clean at width
  6 **0.4280 +/- 0.0337**, cloned at width 6 **0.5243 +/- 0.0203**. The width
  change alone lifts the clean reference by **+0.0690**, which is where the AUC
  difference comes from.
- **The reading.** Part of 0.8743 is the width: **0.1091** of it, or 29% of
  the committed figure's margin over chance. The rest is not. At equal width
  the detector still separates the cloned condition from a clean 6-view
  reference at **0.7652 +/- 0.0843**, so duplication is detected and not
  merely counted. **0.7652 is the number that may be differenced against a
  fixed-width result; 0.8743 still may not.**
- **Equal width controls view COUNT but not information content.** `clone_k2`
  carries 4 distinct views plus 2 duplicates while the clean width-6 reference
  carries 6 distinct views, so the two conditions differ in how much
  independent evidence they hold as well as in duplication. The 0.7652 figure
  therefore isolates the width artefact -- which is what L22 asked for -- but
  is not a pure like-for-like contrast, and no condition in this design can be:
  a clean reference cannot be both 6 views wide and built from 4 views' worth
  of information.

**Evidence:** `results/part36_synthetic/analysis_d.json`;
`experiments/part36_synthetic/README.md` section 5.

**What would change our understanding.** Nothing outstanding on the question
this entry posed. Separating duplication from information content, as opposed
to from width, would need a clean reference built by splitting 4 views' worth
of signal across 6 channels.

## L23. CLOSED 2026-10-26, confirmed not lifted. On LANL the reported prevalence is a declared case-control sample, not the network

**Status: closed as CONFIRMED: the measurement exists and does not lift the
limitation.** L23 asked for "a measurement at, or reweighted to, the
population prevalence". Part 35 recomputed the metrics under
inverse-probability weights (attack 1, benign 512, the declared 1/512 control
hash) on the saved test-block scores, 5 seeds per arm and system. That
measurement now exists, and it confirms the entry rather than retiring it.

- **Weighted prevalence 0.00004371**, against the sample's 0.0219. Weighted
  Brier: `without_auth` **0.097901 +/- 0.000881** (naive) and **0.098170 +/-
  0.000824** (prismflow); `without_flows` **0.030717 +/- 0.000455** and
  **0.031319 +/- 0.000478**; `without_dns` **0.005472 +/- 0.000437** and
  **0.005936 +/- 0.000623**; `with_auth` **0.003592 +/- 0.000472** and
  **0.004228 +/- 0.000559**.
- **Weighted precision collapses at the default threshold.** `without_auth`
  **0.000090**, `without_flows` **0.000085 to 0.000089**, `without_dns`
  **0.001230 to 0.001232**, `with_auth` **0.009390 to 0.010981**, at weighted
  false-alarm counts of **1,459** to **443,290** per test day.
- **The reweighting is only as good as the hash.** The weights reconstruct the
  occupied-cell population only as far as the crc32 control hash is uniform
  over cells, and uniformity is assumed here, not measured.
- **Its false-alarm resolution is 512 cells.** A weighted false-positive count
  moves in steps of **512** cells over 4 test days, so **no alarm rate between
  0 and 128 per day is representable**, and all three pre-registered targets
  (**1**, **10** and **100** false alarms per test day) collapse onto the same
  zero-alarm threshold. The weighted precision of **1.0** there means no
  SAMPLED control fired, which estimates the population false-alarm count as 0
  with a resolution of 512 cells; it does not mean a detector with no false
  alarms.

**So a reweighted estimate is not an operating point**, and the entry's
original reading stands: no population operating point has been measured on
LANL. The population cell count is still bounded, not counted -- between
**6,554,950** and **15,730,712** over the test days
(`results/lanl/population.json`).

**Evidence:** `results/lanl_reanalysis/results.json` (`step3_l23`);
`experiments/lanl_reanalysis/README.md` section 2.

The original entry is kept below, because the sampling decision it recorded is
unchanged.

**Component:** Part 30's LANL cyber1 sample (`experiments/lanl/`, 300 s
windows, days 1-16), and every threshold metric computed on it.

**Evidence:** `results/lanl/population.json`; `experiments/lanl/README.md`
section 11.

- **The sample space is far larger than the sample.** On LANL cyber1 at 300 s
  the `(computer, window)` sample space over days 1-16 is between
  **23,627,216** and **60,634,229** occupied cells, against **849** attack
  cells (**439** train, **410** test): a population prevalence between
  **1.4e-5** and **3.6e-5**.
- **The sample is declared case-control**: all cases plus a deterministic
  1/512 hash of controls. Its test prevalence is **0.0219**. Over the test
  days (13-16) alone the occupied cells are between **6,554,950** and
  **15,730,712**, so the matching population rate is **2.6e-5** to **6.3e-5**
  and the sample prevalence is about **350x** to **840x** higher.

**What this means.** The accuracy, Brier and ECE Part 30 reports describe that
sample and not the network, and no population operating point has been
measured on LANL.

**What would change our understanding.** A measurement at, or reweighted to,
the population prevalence, which would give an operating point that transfers
to the network.

## L24. On LANL the `auth` view is label-adjacent, and dropping it does not drop the selection

**Component:** Part 30's `with_auth` arms, and the `without_auth` arm offered
as the independent-evidence comparison (`experiments/lanl/`).

**Evidence:** `results/lanl/results.json` (`phase3_diagnostics.arm_coverage`,
`phase3_diagnostics.trivial_baselines`) and `results/lanl/samples.npz`;
`experiments/lanl/README.md` sections 7.1 and 11.

- **The labels come from the same stream as a view.** LANL's red-team labels
  are derived from authentication records, so the `auth` view is
  label-adjacent. Part 30 measures both arms, so the WITHOUT-auth arm is the
  one that bears on detection from evidence independent of the label source.
- **That arm is itself limited.** **139 of 410** of its test attack samples
  (**33.9%**) have no record in any remaining view, so they are unpredictable
  by construction and cap the arm's attainable recall before training.
- **Sample selection survives the view drop.** Every attack sample has an
  `auth` record while about a third of benign samples do not (**410 of 410**
  attack against **12,323 of 18,319** benign in the test block), so auth
  presence alone reaches test AUC **0.6637**.

**New evidence, Part 38 (2026-11-22); the entry stays OPEN, narrowed.** The
closing condition below was built and measured: a sample whose control cells
are drawn without any reference to the `auth` stream. Three seed-free scalar
passes over the committed sample table, no training and no inference; each
re-runs bit-exact and is unchanged under `OMP_NUM_THREADS=1`
(`tests/unit/test_h1_scalar_passes.py`, 12 tests).

- **The auth-presence baseline did NOT fall. H1 is DISCONFIRMED.**
  Pre-registered H1 required a drop of at least **0.05** from Part 30's
  **0.6636552213548774**, i.e. a test AUC at or below **0.6136552213548774**.
  Measured **0.6611838191552647**, a drop of **0.0024714021996127267** -- a
  fiftieth of the bar (`results/lanl_v2/h1_auth_presence.json`, `h1.verdict`,
  `h1.drop_achieved`). The baseline is seed-free, so this is a single value
  and no +/- applies.
- **L24's contingency survived intact, so it is a property of the data.**
  Every attack cell still holds an `auth` record -- attack rate exactly
  **1.0** -- on a sample whose selection never consulted the stream, and the
  benign rate moved only **0.6726895572902452** to **0.6776323616894706**
  (`results/lanl_v2/h1_conditional_rates.json`,
  `per_view_rate_comparison.auth`). This entry's **410 of 410** and
  **12323 of 18319** are recomputed from `results/lanl/samples.npz` and
  reproduced exactly as a gate before any comparison is made
  (`l24_reproduction_gate.status` `PASSED`). The first bullet above -- the
  labels come from the same stream as a view -- is strengthened, not
  weakened.
- **The selection did not drop; it moved to `proc`.** Every selection input
  scored as its own trivial baseline on the test block, train-only
  orientation (`results/lanl_v2/h1_occupancy_decomposition.json`,
  `whole_test_block.<scalar>.test_auc`): `occupancy_proc`
  **0.7083759666864962**, `auth_present_indicator` **0.6611838191552647**,
  `n_non_auth_views_present` **0.6503477024**, `occupancy_flows`
  **0.5253242118**, `occupancy_dns` **0.5000356930** (at chance). `proc` now
  carries MORE of the separation than `auth` does. The conditional rates
  attribute it to rule (a): `proc`'s attack rate is unchanged at
  **0.4975609756097561** while its benign rate jumped **0.6973088050657787**
  to **0.9143129089827484**, because rule (a) admits a cell only if `proc`,
  `flows` or `dns` is occupied. Its rate gap more than doubled,
  **-0.19974782945602265** to **-0.4167519333729923**.
- **Not the host-rule tautology.** Within attack hosts the picture is
  unchanged: `occupancy_proc` **0.7143004072**, `occupancy_auth`
  **0.6524885255** (`within_attack_hosts.<scalar>.test_auc`).
- **The volume signals DID fall**, which is the evidence the rule changed
  something: `total_records_across_views` **0.6842788175** to
  **0.5862805386** and `n_views_present` **0.5624544156** to
  **0.4800433975**, the latter below chance (`h1_auth_presence.json`,
  `baselines.<name>`).

**Why it stays OPEN.** The closing condition is satisfied and the result is
not the one the condition anticipated. Removing `auth` from selection removed
the *volume* signals and left `auth`'s own contingency untouched, while
installing a stronger `proc`-occupancy selection effect in its place. Rule (a)
has to gate on something, and whatever it gates on acquires the benign-side
enrichment `proc` now carries -- so "sample selection survives the view drop"
is still true of Part 38, for a different view. The two sample tables are not
comparable row-for-row (`experiments/lanl_v2/PRE_REGISTRATION.md` section 2),
so **0.6611838191552647 against 0.6636552213548774 is not a measured effect
size** -- only the pass/fail against the pre-registered bar is a result.

**What would additionally change our understanding.** A selection rule whose
admission criterion is independent of every view's occupancy -- which on this
data may not exist, since a cell with no occupied view carries no features --
or a decomposition separating `proc`'s selection-induced enrichment from any
behavioural signal `proc` carries. The latter is the live question: `proc`
occupancy is visible to the encoders through the view mask.

**Evidence:** `results/lanl_v2/h1_auth_presence.json`,
`results/lanl_v2/h1_occupancy_decomposition.json`,
`results/lanl_v2/h1_conditional_rates.json`; `docs/FINAL_REPORT.md` section
4.4.4.

**What this means.** The `without_auth` arm removes the auth features but not
this selection, because cells were chosen from all four views.

**What would change our understanding.** A sample whose control cells are
drawn without reference to the `auth` stream, which would separate the view
drop from the selection.

## L25. RESOLVED 2026-10-26. Part 28's scaler and trainer are not reusable across datasets

**Status: resolved** by `prismflow/training/` (Part 34 step 2, commit
`7a5602b`), a dataset-agnostic `fit_scaler`, `apply_scaler` and trainer whose
view widths arrive as the `native_dims` argument. No module in the package
imports a dataset module or reads a source-name table; its only first-party
imports are `prismflow.data.loaders`, `prismflow.models.*`, `prismflow.train`
and `prismflow.utils.seed`. So the entry's closing condition -- a width
resolution passed in by the caller -- is met, and a third dataset no longer
hits this wall.

The equivalence is bit-for-bit rather than approximate
(`tests/unit/test_training_agnostic.py`, 9 tests): the scaler equals
`experiments/lanl/run_lanl.py`'s mirror on the `with_auth` train block of
`results/lanl/samples.npz` (`mean`, `std`, `native_dims` `[8, 5, 7, 2]`,
`fit_rows` 42912, `present_rows_per_view` and the scaled `[n, V, D]` array),
and equals Part 28's `fit_scaler` on the AIT widths `[9, 8, 9, 9, 10]`; one
training unit (LANL `with_auth` naive, seed 0, at the 1 thread
`results/lanl/execution_mode.json` records) reproduces
`results/lanl/scores/with_auth_naive_seed0.npz` exactly, `prob_attack` and
`vacuity` bit-equal over 18729 test samples, largest absolute difference
**0.0**, with no tolerance in the assertion.

**The residue.** The duplicated copies in Parts 28 and 30 remain:
`experiments/ait_clean/run_ait_clean.py` and `experiments/lanl/run_lanl.py`
are frozen and were not modified, so what Part 34 removed is the cause -- the
source-name lookup -- and not the two existing copies, which a Part authorised
to edit those files would have to replace with a call to `prismflow.training`.

The original entry is kept below, because the duplication it recorded was
real.

**Component:** Part 28's `fit_scaler`, `apply_scaler` and `train_ait`
(`experiments/ait_clean/run_ait_clean.py`).

**Evidence:** `experiments/lanl/run_lanl.py`, which mirrors all three;
`experiments/lanl/README.md` section 11.

- **Each resolves native view widths through `ait_lds.FEATURES`**, keyed by
  AIT source names.
- **Part 30 mirrored them rather than patching Part 28.**

**What this means.** Any future Part on a third dataset will hit the same
wall.

**What would change our understanding.** A width resolution passed in by the
caller rather than looked up by source name, which would let one
implementation serve every dataset.

## L26. CLOSED 2026-10-26, confirmed not lifted. On LANL the models' advantage over a trivial scalar is confined to seen attack computers

**Status: closed as CONFIRMED: the interval is attached and the clearance bar
is not met.** L26 asked for "an unseen-computer result that clears the scalar
by more than 0.01 with an interval attached". Part 35 ran the pre-registered
computer-cluster bootstrap (2000 replicates, fixed seed 12345, 8,350 test
computers drawn with replacement), pairing each arm and system against the
untrained scalar inside each replicate. The scalar was recomputed from
`samples.npz` with Part 30's train-fixed orientation and reproduces the
committed test AUC **0.6843** before any interval was computed.

- **No arm-and-system pair clears the scalar by 0.01 on unseen computers.**
  The largest paired model-minus-scalar difference is **+0.0052** (`with_auth`
  prismflow); the smallest is **-0.0461** (`without_auth` naive).
- **All eight paired intervals include 0.** `without_auth` naive **-0.0461
  [-0.1127, 0.0224]**; `without_auth` prismflow **-0.0436 [-0.1106, 0.0250]**;
  `without_flows` naive **-0.0127 [-0.0728, 0.0479]**; `without_flows`
  prismflow **-0.0055 [-0.0625, 0.0534]**; `without_dns` naive **0.0021
  [-0.0682, 0.0697]**; `without_dns` prismflow **0.0026 [-0.0675, 0.0703]**;
  `with_auth` naive **0.0043 [-0.0634, 0.0714]**; `with_auth` prismflow
  **0.0052 [-0.0639, 0.0723]**.
- **A 0.01 difference is not resolvable on this test set.** The paired
  intervals are about **+/-0.07** wide, seven times the clearance bar they
  were asked to test.
- **So the result is "not shown to clear the scalar", not "equal to the
  scalar".** The data are consistent with the models being better than the
  scalar on unseen computers, with their being worse, and with their being the
  same; this test set cannot tell those apart.
- **The interval covers test-set sampling only.** It resamples test computers.
  It does not cover seed, execution mode (L28), or the sample design (L24,
  L27).

**Evidence:** `results/lanl_reanalysis/results.json` (`step4_l26`);
`experiments/lanl_reanalysis/README.md` section 3.

The original entry is kept below, because the confinement it recorded is
unchanged.

**Component:** Part 30's four arms on LANL cyber1, compared against the
untrained scalar `total_records_across_views`.

**Evidence:** `results/lanl/results.json`,
`phase3_diagnostics.seen_unseen` and
`phase4_closeout.trivial_scalar_seen_unseen`; `experiments/lanl/README.md`
sections 7.3, 7.7 and 11.

- **All-test, the models lead.** `total_records_across_views`, oriented on
  train only, reaches test ROC AUC **0.6843** over all **410** test attack
  samples, against **0.7755 to 0.8389** for the trained arms.
- **Split by whether the attack computer also has attack samples in train**,
  the scalar reaches **0.6405** on the **255** seen samples and **0.7563** on
  the **155** unseen ones, while the models reach **0.8153 to 0.8875** on seen
  and **0.7102 to 0.7614** on unseen.
- **On unseen computers the lead is gone.** `without_auth` naive,
  `without_auth` prismflow and `without_flows` naive are below the scalar;
  `without_flows` prismflow, `without_dns` naive, `without_dns` prismflow,
  `with_auth` naive and `with_auth` prismflow are within 0.01 of it.
- **The verdict is not uniform within `without_flows`**, so these are stated
  per arm and system, not per arm.
- **One fixed split, no interval.**

**What this means.** On LANL cyber1 the trained models' ranking advantage over
a single untrained scalar is confined to attack computers already seen as
attacked in train.

**What would change our understanding.** An unseen-computer result that clears
the scalar by more than 0.01 with an interval attached.

## L27. On LANL one host carries a third of the attack class, with no benign contrast

**Component:** Part 30's LANL sample construction (all cases, benign cells at
1/512 by hash).

**Evidence:** `results/lanl/results.json`
(`phase3_diagnostics.host_audit`, `phase4_closeout.top_attack_computers`,
`phase4_closeout.drop_top_test_computer`); `experiments/lanl/README.md`
sections 7.9 and 11.

- **One computer dominates the attack class.** `C17693` accounts for **279 of
  the 849** attack samples (**142 of 439** in train, **137 of 410** in test).
- **It has almost no benign contrast.** **1** benign sample in train and **1**
  in test, because benign cells are kept at 1/512 by hash.
- **Removing it moves the headline.** Dropping every test row of `C17693`
  lowers test ROC AUC by **0.036 to 0.052** depending on arm and system.
- **Host overlap across blocks.** **35** attack computers appear in both
  blocks and **104** are test-only.

**New evidence, Part 35; the entry stays OPEN.** The leave-one-host-out
measurement was extended from the single top host to the **5** computers with
the most test attack rows, for all four arms and both systems, on the saved
test-block scores (5 seeds, mean +/- std).

- **Removing `C17693` (137 of 410 test attack rows, and 1 benign) costs 0.036
  to 0.052 AUC across all eight arm-and-system pairs.** The extremes are
  `without_dns` prismflow **-0.0360** and `without_auth` naive **-0.0519**.
- **Removing any of the other four top hosts changes AUC by at most 0.007.**
  `C1015` (12 attack test rows), `C2388` (10), `C1936` (7) and `C2079` (7)
  together give 32 host-arm-system cells whose largest absolute change is
  **0.0069**.
- **No interval is attached: this is descriptive.** One host is not a
  resample, so these figures carry no interval and are not a significance
  test.
- **Why it stays OPEN.** L27's closing condition is a DESIGN change -- a
  sample that keeps all or a stated share of the benign cells of attack hosts.
  That needs a re-sample and a retrain, which is Part 38. Part 35 only widened
  the sensitivity measurement from one host to five.

**Evidence:** `results/lanl_reanalysis/results.json` (`step5_l27`);
`experiments/lanl_reanalysis/README.md` section 4.

**New evidence, Part 38 (2026-11-22); the entry stays OPEN. H2 is
DISCONFIRMED, in the opposite direction to its prediction.** The closing
condition below was built: a design that keeps a stated share of the benign
cells of attack hosts. The share realised was **1/4**, not 1/1 -- the
pre-registered size fallback fired at `k = 2`, dropping **697298** of
**929764** attack-host benign cells to bring **979122** rows under the
**400000** bar (`results/lanl_v2/build_v2_days16.json`, `size_fallback`).
Every statement below is therefore about a quarter of the benign contrast,
not all of it.

- **All eight `C17693` cells moved by 0.036 or more, so H2 is DISCONFIRMED.**
  H2 was confirmed only if all eight arm-and-system cells moved by less than
  **0.036** in absolute value. Measured band **0.09970976139929666** to
  **0.11343846872488195** (`results/lanl_v2/filled_tables.json`,
  `table3_leave_host_out.h2_verdict.min_abs_delta` and `.max_abs_delta`;
  `cells_at_or_above_bar` lists all eight). The rule is existential, so one
  cell would have sufficed; eight fired.
- **The sensitivity GREW rather than shrank, which is the reverse of H2's
  mechanism.** H2 predicted that restoring benign contrast for `C17693` would
  shrink the leave-host-out delta. Part 30's eight deltas run
  **-0.035962653276654266** to **-0.051874860066125961**; Part 38's run
  **-0.09970976139929666** to **-0.11343846872488195**. The two bands **do
  not overlap**, and Part 38's smallest cell is nearly twice Part 30's
  largest. In magnitude the deltas grew by **+0.047835** to **+0.075601** per
  cell. The verdict therefore does not rest on the 0.036 threshold -- which
  sits inside Part 30's own band -- but on two non-overlapping bands.
- **Rule (b) did restore contrast, and the figures say how much.** `C17693`
  goes from **1** benign test row under Part 30 to **10** under Part 38,
  against **137** attack rows unchanged
  (`table3_leave_host_out.hosts`). Its within-host prevalence is still
  **0.932**, against a test-block prevalence of **0.00485**. Ten benign rows
  against 137 attack rows is not a contrast.
- **What this disconfirmation licenses, and what it does not.** It licenses:
  on this build, restoring **a quarter** of `C17693`'s benign cells did not
  reduce the single-host sensitivity; it increased it. It does **not** license
  "restoring benign contrast does not help" -- that claim needs the 1/1 build,
  which was **not run**. It does **not** license any statement about *why* the
  sensitivity grew: Part 38's test block is **84460** rows at prevalence
  **0.00485** against Part 30's **18729** at **0.0219**, so removing
  `C17693`'s attack rows removes a larger share of the rankable positive
  signal, and **no decomposition distinguishing that from a behavioural
  effect exists**. Do not assert one.
- **No interval attaches.** One host is not a resample; the result is
  descriptive, as Part 30's own `step5_l27.interval` records.
- **Why it stays OPEN.** The closing condition asks for "all **or a stated
  share**". A stated share was delivered and measured, and it was **1/4** --
  so the design question this entry poses is answered at quarter strength,
  and the 1/1 build that would answer it fully has not been run.

**Evidence:** `results/lanl_v2/filled_tables.json` (`table3_leave_host_out`);
`results/lanl_v2/build_v2_days16.json` (`size_fallback`);
`docs/FINAL_REPORT.md` section 4.4.5.2.

**What this means.** A model can therefore learn that host's activity profile
as the attack class without learning attack behaviour.

**What would change our understanding.** A design that keeps all or a stated
share of the benign cells of attack hosts, which would be needed to separate
host from behaviour.

## L28. On LANL the execution mode changes the fitted weights, and the per-seed std understates the spread

**Component:** Part 30's torch thread count and process concurrency
(`results/lanl/execution_mode.json`), and every per-seed standard deviation
computed inside a single run.

**Evidence:** `results/lanl/results.json`, `execution_mode` including
`execution_mode.thread_count_reproducibility_observation`;
`results/lanl/results.run1.json.gz`; `experiments/lanl/README.md` section 11.

- **The torch thread count changes the fitted weights.** Runs 1 (12 threads,
  sequential) and 2 (1 thread, 11 concurrent) of the `with_auth` arm agree in
  the sign of the PrismFlow-minus-naive difference for `roc_auc`, `prob_ece`,
  `benign_recall` and `accuracy` (`roc_auc` **+0.0069, +0.0097**; `prob_ece`
  **-0.0013, -0.0008**), but disagree for `balanced_accuracy` (**-0.0004,
  +0.0088**), `attack_recall` (**-0.0010, +0.0176**) and `brier` (**+0.0000,
  -0.0004**).
- **The same seed differs across modes.** Up to **0.0140** in AUC (naive
  seed 0: **0.8375** and **0.8235**), about **4** per-seed standard deviations
  of that same arm and system (**0.0033**).
- **A seed count such as 5/5 inside one run shares one execution mode** and is
  not independent evidence about mode effects.
- **The thread count selects between deterministic answers, it does not add
  noise.** Re-measured in Part 34 at two counts on the unmodified test
  `tests/unit/test_ait_attack.py::test_clean_dependence_matches_part28`:
  `OMP_NUM_THREADS=1` obtains `eniv_eigen` **3.4289614712229826** against
  Part 28's committed **3.4396320614332856** and fails the `abs=1e-6`
  assertion by **0.0107**; the machine default (**8** threads here) passes.
  The 1-thread value matches the one recorded in
  `results/lanl/execution_mode.json` to the last digit, from a separate
  session, so each count reproduces its own value exactly. A reported figure
  is therefore conditional on a thread count, and "the default" does not
  name one: this machine's default is 8 while the run-1 record says 12.

**New evidence, Part 35; the entry stays OPEN.** For `with_auth` -- the only
arm with two execution modes in `results/lanl/` -- the per-seed std within each
mode, the spread across modes and the pooled spread are now reported per
metric for both systems (`results/lanl/results.run1.json.gz` at 12 threads
sequential against `results/lanl/results.json` at 1 thread, concurrency 11).
Both halves of the result matter, and they point in opposite directions.

- **The 5-seed MEANS barely move across modes, so pooling the modes does not
  widen the std.** The largest across-mode mean difference over these metrics
  is **0.0190** (prismflow `attack_recall`), and the pooled std over all 10
  values is **0.79x to 0.98x** the larger within-mode std for PrismFlow
  (**0.87x to 0.98x** over the six metrics other than `prob_ece`, which is
  **0.79x**) and **0.69x to 0.96x** for naive. Pooling the two modes did not
  widen the std in any of the 14 cells.
- **The SAME SEED moves by several within-mode stds.** The largest same-seed
  naive `roc_auc` gap is **0.0140**, against a within-mode std of **0.0033**
  (run 2) to **0.0039** (run 1) -- about four stds. The largest same-seed gap
  over all metrics is **0.0341** (prismflow `attack_recall`).
- **The reading, which sharpens L28 rather than lifting it.** The mode selects
  between deterministic answers per seed, as this entry already says, and the
  5-seed MEAN absorbs most of it. It is a SINGLE-seed figure that must not be
  read as mode-independent.
- **Why it stays OPEN.** `without_auth`, `without_flows` and `without_dns`
  were run in one execution mode only, so no cross-mode spread exists for
  them, and the entry's closing condition -- seeds run across at least two
  execution modes -- is met for one arm of four. The other three wait on
  Part 38.

**Evidence:** `results/lanl_reanalysis/results.json` (`step6_l28`);
`experiments/lanl_reanalysis/README.md` section 5.

**Part 38 (2026-11-22): UNCHANGED, still OPEN.** Part 38 ran all four arms in
mode `default` (B) only, 40 of 40 units. **Mode A was not run under the Part 38
deadline**, so no cross-mode spread exists for `without_auth`,
`without_flows` or `without_dns`, and this entry's status is exactly as Part 35
left it. `experiments/lanl_v2/run_mode_a.sh` exists and is validated; a future
Part will run it.

Two things Part 38 did establish about what mode A could settle:

- **H3, the pre-registered substitute for the pooled +/-, is unsatisfiable on
  this sample whatever mode A shows.** H3 required the sign of the
  PrismFlow-minus-naive ROC AUC difference to agree across modes in **all
  four** arms. On mode B, three of the four arms have **no established sign**
  -- the difference is smaller than their own per-seed std
  (`results/lanl_v2/filled_tables.json`,
  `table2_per_seed.prismflow_minus_naive`). A sign that does not exist cannot
  agree, so mode A can at best decide `without_auth`. Running mode A remains
  worth doing for the across-mode spread itself, which is what this entry is
  about; it will not produce an H3 verdict.
- **This entry cannot be CLOSED by its own text under the Part 38
  pre-registration.** The closing condition below asks for a std that "covers
  mode as well as seed" -- a **pooled** std.
  `experiments/lanl_v2/PRE_REGISTRATION.md` section 6 forbids pooling the two
  modes into a single +/-. Closing it would need a review-gate decision that a
  mode-pooled +/- is legitimate, and a further dated amendment.

**Evidence:** `results/lanl_v2/filled_tables.json`; `docs/FINAL_REPORT.md`
section 4.4.5.3.

**What this means.** Differences of order 0.01 in `balanced_accuracy`,
`attack_recall` and `brier` are not established, and the per-seed std
understates replicate spread. Related: Part 28's committed ENIV is
reproducible only at the thread count that produced it.

**What would change our understanding.** Seeds run across at least two
execution modes, so the reported std covers mode as well as seed.

## L29. CLOSED 2026-10-26, diagnosed not fixed. On two LANL arms the probabilities are worse than a constant

**Status: closed as DIAGNOSED, not as fixed.** L29 asked for "a test of each
candidate cause, and a Brier below the constant baseline once the responsible
one is fixed". Part 35 tested all three candidates on the saved test-block
scores, 5 seeds, both systems, against the fixed two-class constant baseline
**0.0428**, under a rule written down before any number existed: a cause
accounts for the failure only if correcting it alone puts Brier below 0.0428
in 5 of 5 seeds for both systems.

- **Responsible: the all-False view mask.** `without_auth` has **3600 of
  18729** test rows with no present view (**139** of them attack rows);
  `without_flows` has **998** (**0** attack). On every one of those rows, in
  both systems and all 5 seeds, `prob_attack` is exactly **0.5**, which scores
  the maximum two-class Brier of **0.5** per row -- **0.0961** of
  `without_auth`'s Brier and **0.0266** of `without_flows`'s, against a whole
  budget of 0.0428. Restricted to rows with a present view, Brier falls to
  **0.0318 +/- 0.0007** and **0.0319 +/- 0.0006** (`without_auth` naive and
  prismflow) and **0.0382 +/- 0.0004** and **0.0384 +/- 0.0006**
  (`without_flows`), against those subsets' own constant baselines **0.0352**
  and **0.0452** -- below 0.0428 in 5 of 5 seeds, both systems, both arms.
- **Not responsible: the prior shift, and it is an ORACLE.** Correcting from
  train to test prevalence makes both arms WORSE (`without_auth` **0.2033 +/-
  0.0015** and **0.2038 +/- 0.0013**; `without_flows` **0.0882 +/- 0.0008**
  and **0.0891 +/- 0.0011**): 0 of 5 seeds in every cell. It also uses the
  test prevalence, so it is not available to a deployed detector.
- **Not responsible: recalibration, and it is a DIAGNOSTIC.** Cross-fitted
  Platt by computer fold reaches **0.0432 +/- 0.0000** on `without_auth` (0 of
  5, both systems) and **0.0427 +/- 0.0001** and **0.0426 +/- 0.0001** on
  `without_flows` (4 of 5 naive, 5 of 5 prismflow), so it fails the
  both-systems rule. `results/lanl/` holds no train-block scores, so it is
  fitted on test rows and cross-fitted by computer fold. It is not a
  deployable recalibration.

**"Accounts for" is not "fixed", which is why this closes as diagnosed and not
as resolved.** The rows the responsible cause names are unpredictable by
construction (L24; Part 30 section 4.2): every model must score them
identically whatever their label. Restricting the metric to rows with a
present view changes the evaluated population; it does not change a
probability, and a deployed detector still meets those cells and still emits
0.5 on them. **So the operative warning is unchanged: the probability outputs
of the `without_auth` and `without_flows` arms must not be used.**

**The 0.5/0.0 pair on those rows, and what it is not.** On rows where every
view of an arm is absent the saved scores carry `prob_attack` **0.5** and the
npz column named `vacuity` **0.0**. A vacuity of 0.0 with no evidence would be
an output defect, but that is not what the column holds: `run_lanl.py` writes
`collected["vacuity_confidence"]` into the npz key `"vacuity"`
(`experiments/lanl/run_lanl.py` line 2533), and that array is
`PrismFlowOutput.confidence`, defined as `1.0 - fused_uncertainty`
(`prismflow/models/prismflow.py` line 187). So 0.0 there is **minimum
confidence on maximum vacuity**, which is the correct output for a row with no
evidence, and `prismflow/models` is not implicated. What is wrong is the npz
key: a reader who takes `vacuity = 0.0` at face value reads "no ignorance" on
a row that has no evidence at all. That mislabel is in
`experiments/lanl/run_lanl.py`, which is FROZEN. **Reported, not patched;
decision pending with the user.**

**A route for Part 38, not run here.** Because the saved confidence is already
exactly 0.0 on precisely these rows, abstaining on them is a
selective-prediction decision the existing machinery can express, rather than
a new model.

**Evidence:** `results/lanl_reanalysis/results.json` (`step2_l29`);
`experiments/lanl_reanalysis/README.md` section 1.

The original entry is kept below, because the failure it recorded is real and
is not fixed.

**Component:** the probability outputs of both systems in Part 30's
`without_auth` and `without_flows` arms.

**Evidence:** `results/lanl/results.json`, per-arm `brier` and `roc_auc`;
`experiments/lanl/README.md` section 11.

- **Brier above the constant-prevalence baseline of 0.0428.** In
  `without_auth` **0.1218** and **0.1219**, in `without_flows` **0.0628** and
  **0.0630**, for both systems. Brier here is the two-class form,
  twice the binary form; the constant-prevalence baseline is
  **2*pi*(1-pi) = 0.0428** at test prevalence **0.0219**.
- **Ranking is unaffected.** ROC AUC is **0.7755 to 0.8233**.
- **Cause not diagnosed.** Candidates, none tested: rows with an all-False
  view mask, no validation split, and the shift in sample prevalence from
  train (**1.0%**) to test (**2.2%**).

**What this means.** Probability outputs of those arms should not be used.

**What would change our understanding.** A test of each candidate cause, and a
Brier below the constant baseline once the responsible one is fixed.

## L30. What Part 30 does not establish

**Component:** the scope of Part 30 as a whole, against Part 29's test battery
and against L1-L22.

**Evidence:** `experiments/lanl/README.md` sections 9 and 11;
`results/lanl/results.json`, the per-unit test-A populations and
`phase4_closeout.views_present_shares`.

- **Most of Part 29's battery was not run on LANL.** Tests B (chorus attack),
  D (masked-view detector AUC) and the detector half of C were not run.
- **No robustness claim in L1-L22 has been tested on this dataset**, and
  Part 30 is clean-data evidence only.
- **Part 29 test A is not a stable signal.** ENIV is lower and mean pairwise
  dependence higher on attack windows in `without_auth`, `without_dns` and
  `with_auth`, and both reverse in `without_flows` for both systems (ENIV
  **+0.2636** naive, **5/5** seeds and **+0.2116** prismflow, **5/5** seeds).
- **The two classes also differ in how many rows can carry a pair at all.**
  In `with_auth`, **66.1%** of attack rows against **54.3%** of benign rows
  have two or more views.

**What this means.** Part 30 is clean-data evidence on a second dataset
family; no robustness or attack claim carries over to LANL.

**What would change our understanding.** Running tests B, C and D on LANL, and
a test-A reading that holds its sign across all four arms.

## L31. One window length per real dataset, and no sensitivity analysis

**Component:** every Part 28, 29 and 30 metric (`experiments/ait_clean/`,
`experiments/ait_attack/`, `experiments/lanl/`).

**Evidence:** `results/ait_clean/summary.md` and `results/lanl/results.json`
(`window_seconds`); `docs/part27-ait-datacard.md`.

- **AIT is reported at 300 s in network mode, LANL at 300 s over days
  1-16.** No metric in either Part is reported at a second window length.
- **The window length is not a neutral parameter.** It sets which lines fall
  in a positive window, how often a view is empty and therefore masked, and
  the size of the `(unit, window)` sample space that [[L23]]'s prevalence is
  computed against.
- **Re-binning is implemented and tested, but was never run against the
  metrics.** `rewindow` re-bins one scan to any multiple exactly for
  merge-closed features.

**What this means.** Every real-data figure is conditional on one choice of
window length, and the project has no measurement of how much any of them
would move at another.

**What would change our understanding.** The same metrics at two or more
window lengths per dataset, reported separately and never pooled.

## L32. Part 32's container image has never been built

**Component:** the deployment path of the live diagnostic service
(`Dockerfile`, `serve/requirements.txt`).

**Evidence:** `docs/part32-live-service.md`, "Deploying to Hugging Face
Spaces".

- **No `docker build` has been run against the Dockerfile.** The pinned
  dependency set, the CPU-only torch wheel and the non-root user are all
  unexercised; the first build is the deployer's and it may fail.
- **The service's own code is tested** under `tests/serve/`, but those tests
  ran under the development `.venv` (Python **3.13.7**), the same minor
  version as the image base `python:3.13-slim` and a different build of it.

**What this means.** No claim that the service runs in a container is
supported by anything in this repository.

**What would change our understanding.** One successful build, and a
`/health` response from the running container.
