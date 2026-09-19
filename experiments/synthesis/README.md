# Limits of the defence: the oracle-to-deployable gap

**This line of work is CLOSED as a negative result.** Nothing in this
directory is a working defence, and no number in it may be summarised as
one. What it produced is a measurement of the distance between what an
oracle with attack labels can see and what a gate that must calibrate
without them can actually reach.

Everything here is on the Part 02 synthetic generator, 5 seeds (0-4), mean
and sample standard deviation, per-seed values retained in the JSON beside
every aggregate.

---

## The correction that matters most

> The 0.6743 selectivity AUC reported for the deployable gate in the
> sign-constrained ceiling measurement (`results/synthesis/sign_constrained.json`)
> was measured on a single condition: chorus k=3, epsilon=2.0, no stress --
> the exact condition the gate's constituent signals (`dependence`,
> `clique_contrast`) were developed and validated against throughout this
> session. The 16-cell matrix shows this does NOT generalize: the gate is at
> or below chance in 15 of 16 conditions tested, including every other chorus
> k, every PGD condition, and both missing-view stress levels. 0.6743 should
> be read as the gate's PEAK observed performance on its own development
> condition, not as its deployable ceiling. This is the single most important
> correction this session produced.

`experiments/synthesis/PREREGISTRATION_missing_views.md` quotes 0.6743
without this qualification. That file is deliberately left unedited: it is a
pre-registration, it was committed before the matrix ran, and revising it
after the fact would destroy the only property that makes it worth having.
Read it as a record of what was believed beforehand.

---

## What was run, in order

| stage | question | result |
|---|---|---|
| Arm B oracle | is the information present at all? | yes -- 0.7448 +/- 0.0592, label-fitted |
| P3 ablation | is clique contrast load-bearing? | not in a linear gate; yes in a rank gate |
| diagnostics | is the tail contribution a leak? | no -- real per-sample content |
| sign-constrained | what can a label-free gate reach? | 0.6743 on one condition |
| 16-cell matrix | does that hold anywhere else? | **no -- chance in 15 of 16** |

### Arm B: the oracle ceiling

A logistic gate fitted WITH attack labels and scored in-sample, on four
per-sample signals. An upper bound on what any legal gate could extract, run
before building one so that a low ceiling would stop the work early.

    oracle selectivity AUC   0.7448 +/- 0.0592
    leave-one-seed-out       0.7379 +/- 0.0616
    pooled in-sample         0.7430

The gate passed its 0.60 go-threshold. This is the number that later turned
out not to transfer.

### P3: the pre-registered ablation

P3 predicted that no combination of the existing signals reaches 0.70
without clique contrast.

| arm | linear | rank |
|---|---|---|
| all four | 0.7448 +/- 0.0592 | 0.7528 +/- 0.0617 |
| existing three | 0.7115 +/- 0.0593 | 0.6736 +/- 0.0697 |
| existing two | 0.6724 +/- 0.0591 | -- |
| clique alone | 0.5034 +/- 0.1668 | 0.6408 +/- 0.0551 |

**P3 is refuted on its own terms**: it was pre-registered against the linear
oracle, and there `existing_three` reaches 0.7115. But the substance flips
with the fit. Under a rank fit clique contrast adds +0.0792 +/- 0.0388,
positive on all 5 seeds, and `existing_three` falls below 0.70.

`clique_only` at 0.5034 is not a signal at chance -- it is a cancelling
population. The single-feature linear fit assigns a NEGATIVE weight on seeds
0, 1 and 3, landing at exactly `1 - univariate`. The linear fit was
cancelling a rank-based signal, not reading an absent one.

### Diagnostics: the tail alarm was a false alarm

`tail` is a pre-registered negative control (Part 13: TPR@5%FPR 0.038-0.068).
It contributed +0.0391 to the linear oracle, more than clique contrast did,
which `run_oracle.py`'s own docstring registers as a setup alarm.

| fit | tail column | AUC |
|---|---|---|
| dependence + detector | absent | 0.6724 +/- 0.0591 |
| + tail | as measured | 0.7115 +/- 0.0593 |
| + tail | condition MEAN only | 1.0000 +/- 0.0000 |
| + tail | within-condition centred | 0.6976 +/- 0.0602 |

The surrogate row is the positive control: a true batch-level constant scores
1.0000, so the test could detect the leak it was looking for. The real tail
does not behave that way. Between-condition share of tail variance is
0.0068 +/- 0.0033, and tail cannot separate clone_k2 from clone_k3 (same
label, different condition): 0.5044 +/- 0.0206.

So the contribution is genuine per-sample content: +0.0252 +/- 0.0084
survives within-condition centring, positive on all 5 seeds. `tail` is a
SUPPRESSOR -- univariate 0.4543, below chance, but informative given the
other signals. Part 13's marginal negative result is not contradicted.

Recorded as a deployment caveat: `tail_dependence.pseudo_observations` ranks
WITHIN the batch it is handed, so `tail` is not strictly per-sample. Harmless
in this design, not harmless in streaming deployment.

### Sign-constrained: what survives without labels

Three of the four signals owe their usable direction to attack labels, so
Arm B's ceiling cannot be inherited by a gate that will not have them.

**`detector` and `tail` have no defensible a priori sign in the useful
direction.** Both design rationales give POSITIVE (`suspicion.py`: "what is
suspicious is agreement in EXCESS of both"). Both are empirically backwards:
univariate 0.4023 and 0.4543. Their only correct sign is the negative one the
oracle learned from labels. They are therefore EXCLUDED rather than guessed.

| variant | AUC (on chorus k=3, eps 2.0, unstressed) |
|---|---|
| dependence + clique_contrast | 0.6743 +/- 0.0492 |
| clique alone | 0.6408 +/- 0.0551 |
| dependence alone | 0.5440 +/- 0.0301 |
| all four, design-rationale signs | 0.5130 +/- 0.0196 |

Deploying all four with the only signs a label-free gate could justify gives
0.5130 -- near chance, and worse than clique contrast alone. Removing label
help costs the two-signal gate nothing (+0.0013 +/- 0.0085); it costs the
four-signal set everything (+0.2325 +/- 0.0714).

**Read every number in this table subject to the correction at the top.**

### The 16-cell matrix: it does not generalise

Gate fixed, never refitted per condition. Negatives are clean + clone_k2 +
clone_k3 under the same stress as the attack they face.

**Gate selectivity AUC**

| stress | chorus_k1 | chorus_k2 | chorus_k3 | pgd_k2 |
|---|---|---|---|---|
| none | 0.483 | 0.520 | **0.668** | 0.499 |
| missing_30 | 0.469 | 0.534 | 0.637 | 0.490 |
| missing_50 | 0.441 | 0.502 | 0.568 | 0.461 |
| noisy_1 | 0.507 | 0.510 | 0.624 | 0.490 |

Per-seed, seeds above 0.60: chorus_k1 **0/5**, chorus_k2 **0/5**, pgd_k2
**0/5**, chorus_k3 (unstressed) 4/5. `chorus_k1` is below chance in three of
four stress levels.

**Undefined rate** -- its own column, never folded into dropped rows

| stress | chorus_k1 | chorus_k2 | chorus_k3 | pgd_k2 |
|---|---|---|---|---|
| none | 0.000 | 0.000 | 0.000 | 0.000 |
| missing_30 | 0.345 | 0.350 | 0.346 | 0.345 |
| missing_50 | 0.706 | 0.709 | 0.708 | 0.706 |
| noisy_1 | 0.006 | 0.006 | 0.006 | 0.006 |

**Four-signal oracle upper bound** (label-fitted, NOT deployable)

| stress | chorus_k1 | chorus_k2 | chorus_k3 | pgd_k2 |
|---|---|---|---|---|
| none | 0.766 | 0.734 | 0.716 | 0.827 |
| missing_30 | 0.764 | 0.759 | 0.708 | 0.843 |
| missing_50 | 0.772 | 0.767 | 0.720 | 0.884 |
| noisy_1 | 0.630 | 0.626 | 0.714 | 0.766 |

---

## The gap, which is the contribution

The information is present in every cell. An oracle with attack labels finds
it, and finds MORE of it under PGD than under chorus. A gate calibrating
without those labels reaches none of it.

| cell | oracle | gate | gap |
|---|---|---|---|
| none / pgd_k2 | 0.827 | 0.499 | **+0.328** |
| missing_30 / pgd_k2 | 0.843 | 0.490 | **+0.353** |
| missing_50 / pgd_k2 | 0.884 | 0.461 | **+0.423** |

**The gap widens as conditions get harder.** That is the opposite of what a
usable defence does, and it is the finding. An evaluation that reported only
the oracle's 0.7448 would have claimed a defence that does not exist.

PGD is the sharpest case. Part 09 established that the dependence signal is
blind to uncoordinated compromise, and the Part 10 scope note required a PGD
column in any matrix for exactly that reason. This matrix extends the result:
**clique contrast is blind to it too.** Both deployable signals miss the
attack the oracle finds easiest.

---

## Pre-registered missing-views verdict: SEVERE

Criteria fixed in `PREREGISTRATION_missing_views.md` before the matrix ran.
Both clauses triggered: undefined rate 0.3502 (bar 0.15) and gate AUC 0.4686
(bar 0.60).

> The deployable gate's primary signal is structurally compromised under the
> exact condition (missing views) where L5 already showed the alternative
> signal (detector) failing. Both halves of Arm A's signal set have a known
> blind spot, and they do not overlap.

The structural prediction, stated in the pre-registration before the run, was
close to exact. `clique_contrast` requires at least three usable views; with
V = 4 and views absent at 0.3 independently, P(fewer than 3 usable) = 0.3483.
Observed: 0.345. At 50%: predicted 0.6875, observed 0.706.

---

## Verification of the empty-slice warning

`run_oracle.py:149` raises `RuntimeWarning: Mean of empty slice` under
missing views, where a sample can retain no measurable off-diagonal pair.

**Verified: no value in `gate_matrix.json` is affected by this warning.**
Checked three ways, on numpy 2.5.3:

- In all 80 per-seed cells, total undefined rate equals the clique-contrast
  undefined rate exactly. The union of NaN across the four signals is
  therefore identical to clique contrast's NaN set.
- Reproduced directly at missing_30 and missing_50: rows with NaN
  `dependence` are a strict subset of rows with NaN `clique_contrast`
  (28 of 91, and 104 of 205). Rows NaN in dependence but finite in clique,
  which would be rows whose value could change: **0**.
- Every row that survives into a reported AUC has finite dependence, backed
  by at least 6 measurable ordered pairs of 12.

`np.nanmean` returns NaN for an all-NaN slice on this version, and the mean
of the available entries otherwise -- the intended semantics. The warning
fires only on rows already dropped as clique-undefined.

Caveat that remains: on defined rows under view loss, `dependence` is
averaged over 6 ordered pairs rather than 12, so the estimate is noisier
there even though it is correct by the signal's definition. The warning
should still be handled explicitly rather than emitted; it is reported here
and not silently patched.

---

## Files

| path | what it is |
|---|---|
| `run_oracle.py` | Arm B oracle ceiling + P3 ablation |
| `run_diagnostics.py` | tail leak test, rank-vs-linear, NaN drop pattern |
| `run_sign_constrained.py` | the label-free ceiling |
| `run_gate_matrix.py` | the 16-cell attack x stress matrix |
| `PREREGISTRATION_missing_views.md` | bands fixed before the matrix ran |

Results in `results/synthesis/`. `prismflow/statistics/clique_contrast.py`
carries the estimator; `tests/unit/test_clique_contrast.py` pins it against
constructed structures (19 tests).

## What this does not tell you

The matrix uses non-adaptive attackers. An attacker aware of the ENIV
discount is a separate, still-open question and is Part 14's actual specified
work (`prismflow/attacks/adaptive.py`, `experiments/adaptive/`). Given the
gate is already at chance against a non-adaptive attacker, adapting against
the GATE is not worth pursuing further; adapting against the DISCOUNT
mechanism is.
