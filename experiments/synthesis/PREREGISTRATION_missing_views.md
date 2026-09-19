# Pre-registration: clique contrast under missing views

Written and committed BEFORE the condition matrix is run. Nothing in this
file may be revised after the numbers exist. If the outcome does not fit a
band below, that is reported as an unanticipated outcome rather than by
adjusting a band.

## What is being tested

`clique_contrast` requires at least THREE usable views per sample: with two,
each view has a single comparison, and one comparison cannot be split into a
non-empty "top g" and a non-empty "rest". Such samples return NaN --
undefined, not zero (see `prismflow/statistics/clique_contrast.py`).

The deployable gate is `dependence + clique_contrast`, a priori signs
(+1, +1), equal weights, ranks calibrated on honest rows only, no
per-condition refitting. Its measured selectivity AUC is 0.6743 +/- 0.0492
(`results/synthesis/sign_constrained.json`).

Missing views attack that three-view requirement directly. The 30%-missing
condition is also where L5 recorded the Part 10 detector inverting -- the
other signal considered for this gate.

## Measured baseline, fixed before the run

Undefined rate on the Arm B conditions: **1 row in 300 on clone_k3**
(0.33%), on 2 of 5 seeds; **zero** on chorus_k3 and clone_k2; zero for the
other three signals. Source: `results/synthesis/diagnostics.json`.

## Structural prediction, stated before the run

With V = 4 and each view independently absent at rate 0.3, the probability
that a sample retains fewer than three usable views is

    P(>= 2 absent) = C(4,2)(0.3^2)(0.7^2) + C(4,3)(0.3^3)(0.7) + 0.3^4
                   = 0.2646 + 0.0756 + 0.0081
                   = 0.3483

So masking ALONE predicts roughly 35% undefined at the 30% condition, before
any near-vacuous-belief effect. If the observed rate lands far below this,
the masking is not reaching the signal the way this analysis assumes, and
that discrepancy is itself reportable.

## Bands

Evaluated at the 30%-missing condition, over 5 seeds, mean reported with
sample std, and per-seed values reported alongside.

| band | criterion |
|---|---|
| **SEVERE** | undefined rate > 15%, OR two-signal AUC on DEFINED rows < 0.60 |
| **MATERIAL** | undefined rate > 5% (and not SEVERE) |
| **CLEAN** | neither |

The AUC clause forces SEVERE in any band: a gate that still scores most rows
but scores them below 0.60 has failed just as surely as one that cannot
score them at all.

## What gets stated on each outcome

**On SEVERE** -- state exactly this, as a headline finding and not a
footnote:

> The deployable gate's primary signal is structurally compromised under the
> exact condition (missing views) where L5 already showed the alternative
> signal (detector) failing. Both halves of Arm A's signal set have a known
> blind spot, and they do not overlap.

**On MATERIAL** -- state that the signal degrades measurably under view loss,
give the rate and the AUC on defined rows, and state that it does not meet
the pre-registered bar for structural compromise.

**On CLEAN** -- state plainly, and with the same prominence, that the
deployable gate is robust to 30% view loss: undefined rate at or below 5%
and AUC on defined rows at or above 0.60. This outcome is NOT to be
undersold because it is the less dramatic one. A robustness result that
survives a pre-registered attempt to break it is a finding.

## Reporting rules that apply to all outcomes

- Undefined rate is its own column. It is never folded into "dropped rows".
- AUC on defined rows is reported next to the undefined rate, never alone:
  an AUC computed after discarding a third of the samples is not comparable
  to one computed on all of them, and stating it without the rate beside it
  would misrepresent it.
- Per-seed values before any aggregate mean, for both quantities.
