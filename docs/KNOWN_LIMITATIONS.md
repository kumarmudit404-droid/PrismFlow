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
