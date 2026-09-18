# Suspicion detector (Part 10)

**Scope note.** The Part 10 brief specifies a 5-system x 8-condition comparison
matrix alongside the detector. That matrix was deliberately deferred: Part 09
found the discount does not defend against colluding views, and a matrix built
to display a defence that does not exist answers the wrong question. The
detector — the half of the system Part 09 showed to be working — got the Part
instead, and is evaluated as a detector: ROC AUC, detection at a calibrated
false-positive budget, per condition. An independent-PGD condition was added to
the brief's list, because Part 09 showed it is the attack the dependence signal
cannot see.

**Question.** The discount reads inter-view dependence. Can the same statistics
support an *alert* — "these views agree more than their dependence structure
explains" — even though they do not support an automatic correction?

**Verdict.**

1. **The detector inverts the discount's blind spot, and that is the headline.**
   Independent PGD, the attack Part 09 showed rho_bar cannot see at all, is the
   single most detectable attack here: AUC **0.8396 +/- 0.0970** against
   chorus k=2's **0.6328 +/- 0.0920**. Paired within seed the gap is
   **+0.2068 +/- 0.0608, 5/5 seeds**. The discount and the detector fail on
   opposite attacks.
2. **Coordination buys concealment.** Detection falls monotonically as more
   views collude: AUC 0.7215 (k=1) -> 0.6328 (k=2) -> **0.4870 (k=3)**, with
   k=1 above k=3 in 5/5 seeds (+0.2345 +/- 0.1115). At k=3 the detector is at
   chance while accuracy has collapsed to 0.4213. **The strongest attack is the
   least detectable one.**
3. **The detector is INVERTED under missing views.** AUC 0.2568 (30% missing)
   and 0.1247 (50%). Scores are systematically *lower* than clean, so at any
   threshold masked input is flagged *less* often than clean input. At 50%
   missing the detection rate is exactly 0.0000 in 5/5 seeds.
4. **It flags redundancy, not adversaries.** The second-most-flagged condition
   is `clone_k2` (AUC 0.8743 +/- 0.0583) — honest duplication with no attacker
   present, and it outscores every chorus condition (+0.2415 +/- 0.1164 against
   chorus k=2, 5/5 seeds). The statistic answers "is there unexplained
   agreement?", not "is there an attacker?"
5. **Detection rates are unstable across seeds.** At the 5% budget the standard
   deviations are 0.18-0.29 against means of 0.36-0.65. The AUC ordering is
   stable; the operating-point numbers are not, and should not be quoted as
   single figures.

All numbers: `run_comparison.py` executed 2026-09-18, 5 seeds (0-4), mean +/-
sample std, 300 test samples per seed. Task metrics come from
`prismflow.evaluation.evaluate`. Full tables: `results/comparison/summary.md`
and `matrix.csv`; per-condition outputs in `results/comparison/<condition>/`.

## What the detector measures

For each view pair, the cosine between their belief vectors,

    r_ij(x) = sum_k b_i[k] b_j[k] / sqrt(a_ii(x) a_jj(x))

compared against what the dependence structure accounts for,

    expected_ij = null_ij + clip(R_ij, 0, 1) * (1 - null_ij)

where `null_ij` is the same within-stratum permutation null the dependence
estimator itself uses. The score is the maximum of `r_ij - expected_ij` over
available pairs. Full derivation in `prismflow/statistics/suspicion.py`.

**Isolation.** The detector receives beliefs, the view mask, the dependence
matrix and the model's own pre-fusion strata. It never receives an attack label,
attack config, compromise mask or corruption flag, and `tests/unit/test_suspicion.py`
asserts that against every public signature in the module — including a test
that sweeps names added later. This experiment *does* know which condition is
which, because scoring a detector requires it; evaluation-time knowledge of the
label is not detector-time knowledge of it.

**Thresholds** are calibrated per seed on that seed's own clean scores, so the
`clean` row reads 0.0500 and 0.0040 by construction and every other row is a
true-positive rate at that false-positive budget.

## Results

| condition | ROC AUC | detection @ 5% FPR | detection @ 1% FPR | accuracy | ENIV |
|---|---|---|---|---|---|
| clean | 0.5000 +/- 0.0000 | 0.0500 +/- 0.0000 | 0.0040 +/- 0.0055 | 0.8300 | 3.3281 |
| clone_k2 | 0.8743 +/- 0.0583 | 0.6453 +/- 0.1822 | 0.6453 +/- 0.1822 | 0.7807 | 3.8059 |
| missing_30 | **0.2568 +/- 0.0618** | 0.0461 +/- 0.0778 | 0.0461 +/- 0.0778 | 0.8027 | 3.1304 |
| missing_50 | **0.1247 +/- 0.0319** | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.7360 | 2.9655 |
| noisy_1 | 0.7486 +/- 0.0544 | 0.3760 +/- 0.2787 | 0.3760 +/- 0.2787 | 0.7800 | 3.3438 |
| chorus_k1 | 0.7215 +/- 0.0512 | 0.3700 +/- 0.2459 | 0.3700 +/- 0.2459 | 0.6807 | 3.2127 |
| chorus_k2 | 0.6328 +/- 0.0920 | 0.4280 +/- 0.1970 | 0.4260 +/- 0.2000 | 0.5073 | 2.9048 |
| chorus_k3 | **0.4870 +/- 0.1096** | 0.3613 +/- 0.1268 | 0.3613 +/- 0.1268 | 0.4213 | 2.8887 |
| pgd_k2 | **0.8396 +/- 0.0970** | 0.5807 +/- 0.2897 | 0.5807 +/- 0.2897 | 0.5880 | 3.3435 |

### 1. Why PGD is easier to detect than the chorus

The two attacks reach comparable damage (accuracy 0.588 against 0.507) by
opposite routes, and the detector's premise divides them:

| | measured R among compromised views | detector AUC |
|---|---|---|
| independent PGD | stays near clean (Part 09: rho_bar -0.0290) | 0.8396 |
| chorus | rises sharply (Part 09: +0.7351 at eps 2.0) | 0.6328 |

PGD produces agreement *without* dependence, so none of it is explained and all
of it shows up as excess. The chorus raises measured dependence in step with the
agreement it manufactures, so the expectation rises with the observation and
the excess stays small. **The agreement term the attacker adds to succeed also
buys it cover from the detector that is watching for agreement.** That is not a
defect in the implementation; it is what "agreement beyond what dependence
explains" necessarily means, and `tests/unit/test_suspicion.py` pins the
behaviour with an explicit test.

Combined with Part 09 this gives a clean division of failure:

| attack | discount defends? | dependence signal sees it? | detector sees it? |
|---|---|---|---|
| chorus (coordinated) | no (12/12 cells worse) | **yes** (ENIV 3.36 -> 2.70) | weakly, and worse as k grows |
| independent PGD | no | **no** (rho_bar wrong direction) | **yes** (AUC 0.84) |

Neither instrument covers both, and their coverage is close to complementary.
An alerting system that ran both signals — measured dependence for coordinated
compromise, unexplained agreement for uncoordinated compromise — would cover
more than either alone. That is a concrete, testable proposal for a later Part,
not something this Part evidences.

### 2. The k trend, paired within seed

Per-seed AUC, showing the decline is not a single-seed artefact:

| condition | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 |
|---|---|---|---|---|---|
| chorus_k1 | 0.659 | 0.756 | 0.744 | 0.774 | 0.674 |
| chorus_k2 | 0.571 | 0.690 | 0.563 | 0.768 | 0.573 |
| chorus_k3 | 0.466 | 0.518 | 0.319 | 0.621 | 0.511 |

k=1 exceeds k=3 in 5/5 seeds (+0.2345 +/- 0.1115). More colluding views means
more of the agreement is absorbed into the measured dependence structure, so
detectability falls exactly as the attack's damage rises.

### 3. Where the detector points the wrong way

Under missing views the score does not merely fail to rise, it *falls*: mean
score 0.3590 (clean) -> 0.1566 (30% missing) -> 0.0244 (50%). Fewer co-present
pairs, more vacuity, and a lower dependence estimate all push the observed
cosine down. A threshold tuned for a 5% clean false-positive rate therefore
flags masked input **less** often than clean input, and at 50% missing it never
fires at all. Any deployment reading this flag as "something is wrong" would be
most silent exactly when a third of the evidence is gone.

`noisy_1` behaves in the opposite direction (AUC 0.7486) and is worth a note,
because Part 08 established the *discount* is blind to noise. The detector is
partly sensitive to it, but by an indirect route: noise lowers the measured
dependence, which lowers the expectation, while the three honest views go on
agreeing as before. The excess rises because the expectation fell, not because
the agreement rose. It is detection of a kind, but not of the noisy view.

## Where PrismFlow loses

Generated from the numbers by `build_table.py`, not written by hand, so it
cannot quietly omit a losing condition. The full list is in
`results/comparison/summary.md`. In summary:

- **Detector inverted** on `missing_30` (AUC 0.2568) and `missing_50` (0.1247).
- **Detector at chance** on `chorus_k3` (0.4870) — the most damaging attack
  tested, accuracy 0.4213.
- **Highest false alarm on a benign condition**: `clone_k2` at AUC 0.8743
  outscores every adversarial chorus condition.
- **Accuracy is worse than clean in every condition**, by -0.0273 (missing 30%)
  to -0.4087 (chorus k=3), and **ECE is worse than clean in every condition**,
  by +0.0026 (noisy) to +0.2928 (chorus k=3).
- Adding the detector changes no prediction at all, so none of Part 09's
  negative defence results are improved by this Part. They are unchanged by
  construction, and a test asserts it.

## Limitations

- One synthetic setting, 4 views, rho = 0.3, 300 test samples per seed, 5 seeds.
- Operating-point detection rates have standard deviations of 0.18-0.29 and
  should not be quoted as point estimates; the AUC ordering is the stable result.
- `detection @ 5% FPR` and `@ 1% FPR` are nearly equal in most rows because the
  score distribution is bimodal: once a sample clears the clean maximum, lowering
  the budget removes few detections.
- Thresholds are calibrated on clean data from the *same* generator and seed.
  A deployment calibrating on a different distribution would do worse.
- The clone condition uses a 6-view model, so its ENIV and accuracy are not
  directly comparable with the 4-view rows; only its detector scores are.
- No attacker in this Part optimises *against* the detector. An attacker aware
  of it would have an obvious move — raise measured dependence deliberately —
  and section 1 suggests that move is already partly available for free.

## Reproduce

```
python -m experiments.comparison.run_comparison   # ~14 min on CPU
python -m experiments.comparison.build_table      # writes matrix.csv + summary.md
```

`run_comparison.py` checkpoints `progress.json` after every condition, so an
interrupted run keeps its completed work.
