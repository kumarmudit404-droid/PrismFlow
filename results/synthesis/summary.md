# Part 14 Arm B: oracle ceiling (NOT a detector)

Fitted WITH attack labels and evaluated in-sample. This is an upper
bound on what any legal gate could extract, never a detection result.

Seeds: [0, 1, 2, 3, 4]. Primary metric: selectivity AUC, chorus k=3 eps=2.0 vs clone k=[2, 3].

Attack success: 0.6267 +/- 0.1352

## Decision

| quantity | value |
|---|---|
| oracle selectivity AUC (per-seed) | **0.7448 +/- 0.0592** |
| go threshold | 0.6 |
| pooled in-sample | 0.7430 |
| leave-one-seed-out | 0.7379 +/- 0.0616 |
| **decision** | **PROCEED to Arm A** |

## Each signal alone (univariate selectivity AUC)

| signal | AUC | note |
|---|---|---|
| dependence | 0.5440 +/- 0.0301 | quantity the discount acts on |
| detector | 0.4023 +/- 0.0715 | Part 10, at chance on k=3 vs clean (L5) |
| tail | 0.4543 +/- 0.0166 | PRE-REGISTERED NEGATIVE CONTROL (Part 13: no signal) |
| clique_contrast | 0.6408 +/- 0.0551 | new in Part 14 |

AUC below 0.5 means the signal prefers to flag HONEST duplication over
the attack -- the L5 failure, restated on the selectivity metric.

## P3 ablation (threshold 0.7)

Pre-registered prediction 3: neither the existing signals nor their combination reaches selectivity AUC 0.7 without clique contrast.

The threshold applies to the arms WITHOUT clique contrast (existing_three, existing_two); P3 holds only if every one of
them stays below 0.7. Arms that include clique contrast are
shown for contrast and are not what the prediction is about.

| arm | signals | AUC | vs 0.7 | seeds >= 0.7 |
|---|---|---|---|---|
| all_four | dependence, detector, tail, clique_contrast | 0.7448 +/- 0.0592 | **REACHES** | 4/5 |
| existing_three (P3 arm) | dependence, detector, tail | 0.7115 +/- 0.0593 | **REACHES** | 3/5 |
| existing_two (P3 arm) | dependence, detector | 0.6724 +/- 0.0591 | below | 3/5 |
| clique_only | clique_contrast | 0.5034 +/- 0.1668 | below | 0/5 |
| clique_plus_dependence | dependence, clique_contrast | 0.5990 +/- 0.0407 | below | 0/5 |

**P3 REFUTED.** all_four minus existing_three = +0.0333 selectivity AUC.

The per-seed column is the load-bearing one: a mean below threshold
with seeds on both sides is not the same result as one where no seed
crosses. Read it before quoting the mean.

