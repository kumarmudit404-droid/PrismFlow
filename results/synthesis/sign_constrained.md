# Part 14: the sign-constrained (label-free) ceiling

Seeds: [0, 1, 2, 3, 4]. No labels used for sign, weight or rank
calibration. Equal-weight sum of signed ranks, ranks calibrated on the
HONEST rows only. Threshold of reference: 0.70.

## What a gate without attack labels can reach

| variant | signals | selectivity AUC |
|---|---|---|
| deployable_dep_clique | dependence, clique_contrast | 0.6743 +/- 0.0492 |
| clique_alone | clique_contrast | 0.6408 +/- 0.0551 |
| dependence_alone | dependence | 0.5440 +/- 0.0301 |
| all_four_design_signs | dependence, detector, tail, clique_contrast | 0.5130 +/- 0.0196 |

## The cost of removing label help

| set | label-fitted (rank) | sign-constrained | gap |
|---|---|---|---|
| deployable_dep_clique | 0.6756 +/- 0.0430 | 0.6743 +/- 0.0492 | +0.0013 |
| all_four | 0.7455 +/- 0.0570 | 0.5130 +/- 0.0196 | +0.2325 |

## Signs

dependence +1 (established since Part 04), clique_contrast +1
(mechanistic, from the module docstring, pre-data).

detector and tail have NO defensible a priori sign in the useful
direction: their design rationale gives +1, their univariate AUCs are
0.4023 and 0.4543 -- backwards. Their only known-correct sign is
label-derived, which Arm A cannot obtain. all_four_design_signs uses
the design-rationale signs to show what that costs; it is not an
endorsement of deploying them.

