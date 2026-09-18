# Known limitations

Open issues that are measured or suspected but not resolved. Each entry names
the evidence and what would change our understanding of it.

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
