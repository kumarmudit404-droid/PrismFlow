# Correction shapes: does a capped or per-sample discount bind? (L4 follow-up)

**Read-only diagnostic.** Nothing here is wired into the model. `discount.py`,
`eniv.py`, `fusion.py` and `prismflow.py` are called, never modified, and no new
attack was run: the analysis replays the perturbation tensors Part 09 already
saved in `results/chorus/perturbations/`.

**Question.** L4 records that the current discount is *proportional* (alpha
scales evidence) and *sublinear in its own strength* — alpha = 0.50 removes only
17.6% of belief at eps 1.0, alpha = 0.43 removes 12.8% at eps 2.0 — so it is
weakest exactly where evidence is strongest. L4 then speculated that "alpha near
0.25 with a floor, or an explicit cap on per-view contribution, or a per-sample
rather than per-batch factor" might neutralise the attack. This tests all three.

**Verdict: all three are refuted. None of them inverts the margin, and none
meaningfully reduces attack success.** Worse, the obvious fix — discounting
harder — makes the margin *worse*, not better. The binding constraint is not the
shape of the correction. It is that a correction which cannot tell a compromised
view from an honest one must hit both.

| shape | attack success, eps 1.0 | vs current |
|---|---|---|
| current | 0.4813 +/- 0.1133 | — |
| capped (belief <= 0.3) | 0.4780 +/- 0.1263 | -0.003 |
| floor_alpha (alpha >= 0.25) | 0.4813 +/- 0.1133 | 0.000 (no-op) |
| alpha_ceiling (alpha <= 0.25) | 0.4933 +/- 0.1166 | **+0.012, worse** |
| per_sample | 0.5120 +/- 0.1055 | **+0.031, worse** |

## Validation

Two gates had to pass before any number above was written, and the script
aborts rather than print if either fails:

1. **Replay fidelity.** Reconstructed per-seed attack success matches
   `results/chorus/attack_metrics.json` exactly, all 5 seeds, both cells.
2. **Baseline reproduction.** The `current` row reproduces
   `experiments/chorus/README.md` section 6 to 4 decimal places
   (0.5553 / 0.1668 at eps 1.0; 0.7301 / 0.2130 at eps 2.0).

**Sample set is held fixed.** "Successful attacks" means the samples the model
as it stands today gets wrong — the same set section 6 characterised — so every
shape is compared on identical samples. The `attack success` column is a
separate, additional measurement: beliefs re-fused under each shape, then the
prediction recomputed.

## Results

### chorus_k2_eps1.0

| shape | compromised -> target | honest -> true | ratio | margin inverted? | attack success |
|---|---|---|---|---|---|
| current | 0.5553 +/- 0.0611 | 0.1668 +/- 0.0470 | 3.33x | no | 0.4813 +/- 0.1133 |
| capped | 0.2650 +/- 0.0083 | 0.1002 +/- 0.0201 | 2.65x | no | 0.4780 +/- 0.1263 |
| floor_alpha | 0.5553 +/- 0.0611 | 0.1668 +/- 0.0470 | 3.33x | no | 0.4813 +/- 0.1133 |
| alpha_ceiling | 0.4171 +/- 0.0589 | 0.1040 +/- 0.0342 | **4.01x** | no | 0.4933 +/- 0.1166 |
| per_sample | 0.5568 +/- 0.0622 | 0.1687 +/- 0.0484 | 3.30x | no | 0.5120 +/- 0.1055 |

### chorus_k2_eps2.0

| shape | compromised -> target | honest -> true | ratio | margin inverted? | attack success |
|---|---|---|---|---|---|
| current | 0.7301 +/- 0.0320 | 0.2130 +/- 0.0457 | 3.43x | no | 0.6133 +/- 0.1293 |
| capped | 0.2854 +/- 0.0025 | 0.1197 +/- 0.0207 | 2.38x | no | 0.5673 +/- 0.1349 |
| floor_alpha | 0.7301 +/- 0.0320 | 0.2130 +/- 0.0457 | 3.43x | no | 0.6133 +/- 0.1293 |
| alpha_ceiling | 0.6321 +/- 0.0441 | 0.1428 +/- 0.0378 | **4.42x** | no | 0.6313 +/- 0.1312 |
| per_sample | 0.7510 +/- 0.0364 | 0.2201 +/- 0.0513 | 3.41x | no | 0.6587 +/- 0.1187 |

## Why each one fails

**capped — the cap cannot be selective.** It is the only shape that moves
absolute belief substantially (0.5553 -> 0.2650 at eps 1.0). But a cap is a
blanket rule: no correction knows which views are compromised, so it must apply
to all of them. Honest views are confident on the samples where they are right,
and get clipped too — honest belief in the true class falls 0.1668 -> 0.1002.
Both sides are cut, the ratio only drops 3.33x -> 2.65x, and attack success is
unchanged within noise (-0.003). **The cap removes the honest views' ability to
outvote at the same time as it removes the attacker's ability to dominate.**

**floor_alpha — a no-op on this data, by construction.** Measured alphas on the
compromised views are 0.5021 and 0.4331, both already above 0.25, so a lower
bound at 0.25 cannot bind. The numbers are bit-identical to `current`. The
brief's wording ("min alpha = 0.25") and L4's wording ("alpha near 0.25 with a
floor") point in opposite directions; both readings were computed rather than
silently resolved, and `alpha_ceiling` is the one that tests the intent.

**alpha_ceiling — discounting harder makes the margin WORSE.** Forcing alpha to
<= 0.25 reduces the attacker's belief (0.5553 -> 0.4171) but reduces the honest
views' belief proportionally *more* (0.1668 -> 0.1040), so the ratio rises from
3.33x to **4.01x** (and 3.43x to 4.42x at eps 2.0) and attack success goes *up*.
This is L4's sublinearity operating in reverse and is the sharpest result here:
because belief is `e / (sum(e) + K)`, scaling evidence down costs a low-evidence
opinion a larger fraction of its belief than a high-evidence one. The honest
views are the low-evidence ones. **Turning up the discount transfers relative
advantage to the attacker.** Any future proposal to "just discount more" is
refuted by this row.

**per_sample — no effect, slightly negative.** Ratio 3.30x against 3.33x, attack
success *up* 0.031. On an attacked sample the compromised pair agrees, which
lowers their per-sample alpha, but the honest views disagree with everything,
which raises theirs toward 1 and leaves them barely discounted — and the
sublinearity above means the attacker still loses less belief than the honest
views do in relative terms. Per-sample granularity does not change the shape
problem; it just varies alpha faster.

**Caveat on "per-sample dependence".** Dependence is a statistic *across*
samples; a single sample has no correlation. So this row uses a per-sample
redundancy *proxy* — the cosine between belief vectors, the same statistic Part
10's detector uses — fed to `soft_cluster_alpha`. Read it as "what a per-sample
agreement-driven correction would do", not as per-sample dependence, which is
not measurable.

## What the binding constraint actually is

To separate "wrong shape" from "wrong selectivity", the same sweep was run with
an **oracle**: alpha applied *only* to the compromised views, using ground-truth
knowledge of which they are. This is not a defence and could never be deployed —
it is an upper bound on what any selective correction could achieve.

| oracle alpha on compromised views only | comp -> target | ratio | inverted? | attack success |
|---|---|---|---|---|
| 0.25 | 0.4171 +/- 0.0589 | 2.50x | no | 0.3880 +/- 0.0982 |
| 0.10 | 0.2446 +/- 0.0434 | 1.47x | no | 0.2620 +/- 0.0837 |
| 0.05 | 0.1465 +/- 0.0290 | **0.88x** | **YES** | 0.1840 +/- 0.0713 |
| 0.01 | 0.0352 +/- 0.0079 | **0.21x** | **YES** | 0.0947 +/- 0.0325 |

(eps 1.0; eps 2.0 needs alpha 0.01 to invert, reaching 0.38x and success 0.1560.)

**Selectivity is the binding constraint, not shape.** A *selective* proportional
discount works: the same proportional mechanism L4 criticised inverts the margin
and cuts attack success from 0.4813 to 0.1840 — provided it is pointed only at
the compromised views. It needs to be extreme (alpha ~0.05, a 20x reduction) and
even then the attack still lands 18% of the time, against a clean-input floor of
0. But it binds, and no blanket shape does.

This relocates the problem, and the relocation is unwelcome. Selectivity
requires knowing which views to discount, which is detection — and Part 10
measured detection on this exact attack family at AUC 0.6328 for k=2, falling to
**0.4870 (chance) at k=3**, while flagging honest duplication harder than any
attack. A correction that needs to concentrate 20x on the right two views cannot
be driven by a signal that is at chance about which views those are.

So the three parts now form a closed loop:

- **L3/L4:** the discount detects coordinated collusion but does not defend.
- **This diagnostic:** no blanket correction shape fixes that; only selective
  correction does, and only at extreme strength.
- **L5/Part 10:** the detection needed to be selective is at chance exactly where
  the attack is most damaging.

## Limitations

- Two cells (chorus k=2 at eps 1.0 and 2.0), 5 seeds, 300 test samples per seed.
  k=1 and k=3 were not run; the brief scoped this to the section 6 cells.
- The sample set is fixed by the current model, so the belief columns answer
  "what would the margin have been", not "what would the attack have done" — the
  `attack success` column answers the latter and is the stronger measurement.
- Corrections are applied post hoc at inference. A model *trained* under a capped
  or selective discount could adapt to it, as Part 06 and L1 found for the
  existing discount. That is a training experiment, not this one, and it could
  change these numbers in either direction.
- CAP = 0.3 and the alpha bound of 0.25 are single points, chosen from the
  brief. The oracle sweep varies alpha; the cap was not swept.
- The oracle rows use ground-truth compromise knowledge and are an upper bound
  only. Nothing here is a defence, and none of it is wired into the model.

## Reproduce

```
python -m experiments.chorus.analyze_correction_shapes   # ~5 min on CPU
```

Writes `results/chorus/correction_shapes.json`. Both validation gates run on
every invocation; the script raises rather than report unvalidated numbers.
