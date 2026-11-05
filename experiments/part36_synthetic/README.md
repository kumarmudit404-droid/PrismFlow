# Part 36 - synthetic estimator and detector checks (L2, L9, L10, L11, L22)

Pre-registered in `experiments/part36_synthetic/config.yaml`, committed before any number in `results/part36_synthetic/` existed. Every analysis, metric, threshold, decision rule and fallback subset was fixed there first, and every verdict below is the one its rule gives, whichever way the numbers fell.

This file is generated from `results/part36_synthetic/*.json` by `experiments/part36_synthetic/render.py`, so no figure in it is typed by hand.

5 seeds everywhere (`[0, 1, 2, 3, 4]`), mean +/- sample standard deviation (numpy ddof=1), paired within seed where two conditions are compared. Frozen modules and earlier experiments are imported, never edited.

## 0. Reproduction first

Each committed value was re-derived at the ORIGINAL settings before the analysis that depends on it was allowed to start. The tolerance is half a unit in the last decimal place the committed record reports, fixed in `config.yaml` before the run.

| quantity | source | committed | measured | abs diff | tol | pass |
|---|---|---|---|---|---|---|
| `eniv_features_cca_holdout.correlation_with_truth` | `results/eniv_validation/summary.json` | 0.9922060509 | 0.9922060509 | 0 | 5e-05 | yes |
| `eniv_features_cca_holdout.mae` | `results/eniv_validation/summary.json` | 0.5220005951 | 0.5220005951 | 0 | 5e-05 | yes |
| `eniv_features_cca_holdout.mae_std` | `results/eniv_validation/summary.json` | 0.2998212817 | 0.2998212817 | 0 | 5e-05 | yes |
| `summary.clone_k2.auc.mean` | `results/comparison/detector.json` | 0.8743088889 | 0.8743088889 | 0 | 5e-05 | yes |
| `summary.clone_k2.auc.std` | `results/comparison/detector.json` | 0.05831081227 | 0.05831081227 | 0 | 5e-05 | yes |
| `detection.eps0.2.tail.auc.mean` | `results/tail/tail.json` | 0.4846145139 | 0.4846145139 | 0 | 5e-05 | yes |
| `detection.eps2.0.tail.auc.mean` | `results/tail/tail.json` | 0.5856175694 | 0.5856175694 | 0 | 5e-05 | yes |

All of these reproduce to the printed precision, and the Part 10 and Part 05 quantities reproduce bit-for-bit (abs diff exactly 0), which is a stronger statement than the tolerance asked for. Everything below is therefore interpretable.

## 1. A - L9: does the compression scale with view count?

Generator: `experiments.eniv_validation.run_validation.run_cell`, imported unchanged, at the committed settings with only `n_views` varied. Truth: `analytic_n_eff_eigen(rho, V)`. Primary estimator: `eniv_features_cca_holdout`. View counts [4, 6, 8, 16], rho grid [0.0, 0.2, 0.4, 0.6, 0.8, 0.95], 5 seeds, so 30 cells per view count. Ran the full pre-registered grid: yes.

### 1.1 Accuracy against the analytic truth, per view count

| V | estimator | correlation with truth | MAE +/- std | MAE / V |
|---|---|---|---|---|
| 4 | `eniv_features_cca_holdout` **(primary)** | 0.9922 | 0.5220 +/- 0.2998 | 0.1305 |
| 4 | `eniv_views_cca` | 0.9965 | 0.1919 +/- 0.1494 | 0.0480 |
| 4 | `eniv_model_forward` | 0.9724 | 0.6872 +/- 0.4323 | 0.1718 |
| 6 | `eniv_features_cca_holdout` **(primary)** | 0.9930 | 0.8514 +/- 0.4575 | 0.1419 |
| 6 | `eniv_views_cca` | 0.9963 | 0.3129 +/- 0.2540 | 0.0521 |
| 6 | `eniv_model_forward` | 0.9716 | 1.1634 +/- 0.7189 | 0.1939 |
| 8 | `eniv_features_cca_holdout` **(primary)** | 0.9924 | 1.1662 +/- 0.6368 | 0.1458 |
| 8 | `eniv_views_cca` | 0.9963 | 0.4350 +/- 0.3530 | 0.0544 |
| 8 | `eniv_model_forward` | 0.9702 | 1.6239 +/- 1.0144 | 0.2030 |
| 16 | `eniv_features_cca_holdout` **(primary)** | 0.9938 | 2.2995 +/- 1.3063 | 0.1437 |
| 16 | `eniv_views_cca` | 0.9962 | 0.9424 +/- 0.7542 | 0.0589 |
| 16 | `eniv_model_forward` | 0.9781 | 3.3776 +/- 1.9756 | 0.2111 |

Correlation with truth is flat across view count (0.9922 at V = 4, 0.9930 at V = 6, 0.9924 at V = 8, 0.9938 at V = 16), so the RANK agreement L9 already reports does not degrade as views are added. The VALUE error does: MAE rises from 0.5220 at V = 4 to 2.2995 at V = 16, a factor of 4.41 across a 4x rise in view count. As a fraction of V the error is near-constant (0.1305 at V = 4 against 0.1437 at V = 16), which is the shape of a proportional compression rather than a fixed offset. The same holds for the two non-primary variants, so it is not an artefact of the estimator the model happens to use.

### 1.2 Signed error per rho, which shows the direction

| V | quantity | rho 0.0 | rho 0.2 | rho 0.4 | rho 0.6 | rho 0.8 | rho 0.95 |
|---|---|---|---|---|---|---|---|
| 4 | analytic truth | 4.000 | 3.400 | 2.800 | 2.200 | 1.600 | 1.150 |
| 4 | signed error | -1.005 +/- 0.049 | -0.644 +/- 0.048 | -0.259 +/- 0.052 | 0.112 +/- 0.022 | 0.452 +/- 0.040 | 0.660 +/- 0.070 |
| 6 | analytic truth | 6.000 | 5.000 | 4.000 | 3.000 | 2.000 | 1.250 |
| 6 | signed error | -1.571 +/- 0.016 | -0.937 +/- 0.026 | -0.305 +/- 0.063 | 0.315 +/- 0.075 | 0.837 +/- 0.072 | 1.143 +/- 0.073 |
| 8 | analytic truth | 8.000 | 6.600 | 5.200 | 3.800 | 2.400 | 1.350 |
| 8 | signed error | -2.152 +/- 0.037 | -1.157 +/- 0.067 | -0.323 +/- 0.034 | 0.515 +/- 0.079 | 1.213 +/- 0.073 | 1.637 +/- 0.082 |
| 16 | analytic truth | 16.000 | 13.000 | 10.000 | 7.000 | 4.000 | 1.750 |
| 16 | signed error | -4.103 +/- 0.094 | -1.937 +/- 0.146 | -0.408 +/- 0.231 | 1.196 +/- 0.155 | 2.622 +/- 0.095 | 3.531 +/- 0.095 |

The sign flips from negative to positive as rho rises at every view count: the estimator understates the effective view count when views are near-independent and overstates it when they are strongly dependent. That is compression toward the middle of the range, and the magnitude at both ends grows with V. **These errors are against `analytic_n_eff_eigen`.** Section 6.1 reports a defect in L9's own text on exactly this point.

### 1.3 Pre-registered rule and verdict (L9)

> SCALES if MAE(V_max) - MAE(V_min) > across-cell std at V_min; FLAT if every MAE is within one V_min std of MAE(V_min); else MIXED

MAE(V = 16) - MAE(V = 4) = 1.7775, against an across-cell std at V = 4 of 0.2998. The gap exceeds the spread by a factor of 5.9, and MAE is monotonically increasing over all four view counts. **Verdict: SCALES.**

This answers the FIRST half of L9's closing condition and answers it in the direction that makes L9 worse, not better: the single validated design point of n_views = 4 is the most favourable one run. It does not close L9. The second half needs a real dataset with an independently established dependence structure, which does not exist in this checkout and was recorded as out of reach in `config.yaml` (`cannot`) before anything ran.

## 2. A - L2: does residual per-pair dependence accumulate with view count?

Measured at rho = 0.0, where the true per-pair dependence is 0, so the whole reading is residual. Both the chance-corrected estimator the model uses and the uncorrected `_nc` variant are reported, because L2's quoted figure of ~0.05 per pair is the scale of the uncorrected one and the corrected one sits six times higher.

| V | `rho_bar_features_cca_holdout` (primary) | `rho_bar_features_cca_holdout_nc` |
|---|---|---|
| 4 | 0.3345 +/- 0.0163 | 0.0844 +/- 0.0336 |
| 6 | 0.3139 +/- 0.0033 | 0.0590 +/- 0.0050 |
| 8 | 0.3071 +/- 0.0053 | 0.0511 +/- 0.0106 |
| 16 | 0.2732 +/- 0.0062 | 0.0271 +/- 0.0068 |

> ACCUMULATES if the mean rho_bar at rho = 0.0 is monotonically non-decreasing in V and the V_max value exceeds the V_min value by more than the larger of the two stds

Monotonically non-decreasing in V: no. V = 16 minus V = 4 = -0.0613, against a larger std of 0.0163. **Verdict: DOES NOT ACCUMULATE.** The residual FALLS with view count, and it falls under both variants, so the direction does not depend on the chance correction.

**This does not contradict L2 on L2's own harness, and it must not be read as doing so.** L2's accumulation claim is about `clone_eigen_softcluster`, where the views added are clones and the soft cluster size grows with them. Part 36 measures the analytic-rho generator, where the views added are INDEPENDENT. The two readings are consistent: adding independent views dilutes the mean per-pair residual, adding clones does not. What Part 36 does establish is that the residual is not an intrinsic property of the estimator that accumulates with width whatever the added views are, so L2's title overstates the scope of its own evidence. L2's causal link to the confidence drift at rho = 0.0 was declared out of scope in `config.yaml` before the run and is still untested.

## 3. B - L10: the sample-count component against the structural floor

**Pre-registered deviation, stated as one.** The sweep varies the ESTIMATION sample count, not `TrainConfig.batch_size`. Batch size drives both training and estimation, so sweeping it would confound "the estimator saw fewer samples" with "a differently trained model", and L10's question is about the estimator. One model is trained per seed at the committed settings with `n_samples = 16000`, and the dependence matrix is then estimated on NESTED random subsamples of that model's test split, so the sweep is paired within seed and the only thing changing along it is how many rows the estimator sees. rho is fixed at 0.3. Ran the full pre-registered grid: yes.

The `independent_null` arm is the same sweep on rho = 0.0 data, which is where any deficit is pure finite-sample artefact; it is the synthetic analogue of the Part 15 permutation null L10 quotes.

### 3.1 Signed ENIV bias against the analytic truth, per estimation n

| cell | truth | n = 64 | n = 128 | n = 256 | n = 512 | n = 1024 | n = 2048 |
|---|---|---|---|---|---|---|---|
| `V4_dependent` | 3.10 | -1.2548 +/- 0.0781 | -0.8359 +/- 0.0745 | -0.5698 +/- 0.0454 | -0.4035 +/- 0.0341 | -0.3209 +/- 0.0410 | -0.2581 +/- 0.0232 |
| `V4_independent_null` | 4.00 | -2.1469 +/- 0.0454 | -1.5991 +/- 0.0514 | -1.2434 +/- 0.0402 | -0.9739 +/- 0.0406 | -0.8295 +/- 0.0263 | -0.7493 +/- 0.0352 |
| `V6_dependent` | 4.50 | -1.9311 +/- 0.1629 | -1.3146 +/- 0.1353 | -0.8407 +/- 0.0692 | -0.5544 +/- 0.0889 | -0.3786 +/- 0.0401 | -0.2589 +/- 0.0734 |
| `V6_independent_null` | 6.00 | -3.4630 +/- 0.1397 | -2.6413 +/- 0.0522 | -2.0287 +/- 0.0363 | -1.5555 +/- 0.0274 | -1.2726 +/- 0.0253 | -1.0979 +/- 0.0270 |

### 3.2 The pre-registered separation, abs(bias)(n) = floor + c / n

| cell | fitted floor +/- std | fitted c +/- std | fit residual rms | abs bias at n = 64 | abs bias at n = 2048 | ratio |
|---|---|---|---|---|---|---|
| `V4_dependent` | 0.2736 +/- 0.0348 | 65.1 +/- 6.5 | 0.0404 | 1.2548 | 0.2581 | 4.86x |
| `V4_independent_null` | 0.7892 +/- 0.0331 | 91.3 +/- 2.2 | 0.0758 | 2.1469 | 0.7493 | 2.87x |
| `V6_dependent` | 0.3220 +/- 0.0599 | 108.8 +/- 9.0 | 0.0994 | 1.9311 | 0.2589 | 7.46x |
| `V6_independent_null` | 1.2255 +/- 0.0271 | 153.0 +/- 9.6 | 0.1671 | 3.4630 | 1.0979 | 3.15x |

### 3.3 Pre-registered rule and verdict (L10)

> DOMINATED BY SAMPLE COUNT if the fitted floor is within one across-seed std of 0 and the smallest-n bias is more than twice the largest-n bias; STRUCTURAL FLOOR if the floor exceeds one std of 0 and the largest-n bias is still more than that std from 0

| cell | fitted floor within one std of 0 | verdict |
|---|---|---|
| `V4_dependent` | no | **STRUCTURAL FLOOR** |
| `V4_independent_null` | no | **STRUCTURAL FLOOR** |
| `V6_dependent` | no | **STRUCTURAL FLOOR** |
| `V6_independent_null` | no | **STRUCTURAL FLOOR** |

**Both components are present, and the answer is the same in 4 of 4 cells: STRUCTURAL FLOOR.** The sample-count component is large - the bias shrinks by a factor of 2.9 to 7.5 between n = 64 and n = 2048 - so L10's "estimated from finite batches" is confirmed. But it does not go to zero: at the largest n run, the bias is still 0.2581 (11 across-seed stds from 0) in `V4_dependent`, 0.7493 (21 across-seed stds from 0) in `V4_independent_null`, 0.2589 (4 across-seed stds from 0) in `V6_dependent`, 1.0979 (41 across-seed stds from 0) in `V6_independent_null`. That part is model-free: it is a measurement at a fixed n, not a fitted limit.

### 3.4 POST-HOC robustness check, NOT pre-registered

The fitted floor should not be read as the estimator's asymptotic bias, and this check is the reason. The residuals of the pre-registered `floor + c / n` fit are not scattered - their signs along increasing n are the same in all four cells, `-+++--` (`V4_dependent`), `-+++--` (`V4_independent_null`), `-+++--` (`V6_dependent`), `-+++--` (`V6_independent_null`), which is the signature of a mis-specified decay rather than noise. A floor-free power law `abs(bias) = k * n^alpha` fits the same points with these exponents:

| cell | power-law exponent alpha | power-law residual rms | `floor + c / n` residual rms |
|---|---|---|---|
| `V4_dependent` | -0.459 | 0.0487 | 0.0404 |
| `V4_independent_null` | -0.308 | 0.0780 | 0.0758 |
| `V6_dependent` | -0.585 | 0.0159 | 0.0994 |
| `V6_independent_null` | -0.338 | 0.0768 | 0.1671 |

Every exponent is well short of the -1 the pre-registered model assumes. The floor-free power law, which has one fewer parameter, fits better in 2 of 4 cells (`V6_dependent`, `V6_independent_null`) and never worse by more than 0.0084 rms. So these 6 points do NOT distinguish "a non-zero floor plus a 1/n term" from "a single term decaying more slowly than 1/n and reaching zero". The fitted floor is an upper bound on the structural component over this n range, not an identified limit - and it exceeds the measured bias at the largest n in every cell, which is the same point seen from the other side. **The verdict in section 3.3 stands as pre-registered**: it was fixed before any number existed and is reported as the rule gives it. But the claim it licenses is "the bias is still far from zero at n = 2048", which is model-free, not "the bias has a non-zero limit", which is not. Separating those needs a sweep reaching an n where the decay visibly flattens, and this one does not.

## 4. C - L11: Part 13's detection arm at 4x the split size

`experiments.tail.run_tail` imported cell by cell, in its own per-seed order, at `n_samples = 32000` against the committed 8000. Test slice 1200 rows -> 4800 rows. 5 seeds. Epsilons run: [0.0, 0.2, 0.5, 1.0, 2.0]; full pre-registered grid: yes.

### 4.1 Tail AUC at the committed size and at 4x

| epsilon | tail AUC, committed size | tail AUC, 4x | std / mean at 4x | n_tail at q = 0.95, 4x | rho_bar AUC, 4x |
|---|---|---|---|---|---|
| 0.2 | 0.4846 +/- 0.0022 | 0.4938 +/- 0.0039 | 0.008 | 240.0 +/- 0.0 | 0.7244 +/- 0.0531 |
| 0.5 | 0.4801 +/- 0.0063 | 0.5007 +/- 0.0042 | 0.008 | 240.0 +/- 0.0 | 0.5957 +/- 0.0376 |
| 1.0 | 0.5370 +/- 0.0132 | 0.5554 +/- 0.0087 | 0.016 | 240.0 +/- 0.0 | 0.6746 +/- 0.1071 |
| 2.0 | 0.5856 +/- 0.0126 | 0.5568 +/- 0.0129 | 0.023 | 240.0 +/- 0.0 | 0.6152 +/- 0.1561 |

The 4x is real where it matters: n_tail at the headline quantile rises from the committed 60 to the figure in the table above, so the estimator is no longer sitting on L11's 50-exceedance floor with a 20% margin.

### 4.2 Pre-registered rule and verdict (L11), on BOTH bands

> SUBSTANTIVE if every epsilon's 4x tail AUC mean lies inside the band; UNDERPOWERED-AFTER-ALL if any mean moves outside it by more than its own across-seed std

**Two bands are pre-registered on purpose.** L11's text quotes Part 13's tail AUC span as 0.4846 to 0.5856, but the committed values in `results/tail/tail.json` run from 0.4801 to 0.5856 - L11's quoted floor EXCLUDES one of its own four points (eps 0.5). A rule built on the quoted floor would score a re-run as "outside the band" against a floor the original experiment already breached, so both are evaluated and the committed-values band is primary. Section 6.2 proposes the correction.

| band | range | epsilons inside | epsilons outside by more than their own std | verdict |
|---|---|---|---|---|
| primary | [0.4801, 0.5856] | eps0.2, eps0.5, eps1.0, eps2.0 | none | **SUBSTANTIVE** |
| quoted | [0.4846, 0.5856] | eps0.2, eps0.5, eps1.0, eps2.0 | none | **SUBSTANTIVE** |

**The dispersion L11 names as the actual problem is unchanged by the 4x.** Across-seed std as a fraction of the mean runs from 0.008 (eps0.2) to 0.023 (eps2.0) at 4x. Quadrupling the split buys exceedances, not agreement between seeds, which is consistent with L11's own reading that the failure is dispersion rather than displacement.

## 5. D - L22: the clone comparison at equal width

`experiments.comparison.run_comparison` imported unchanged, at the committed Part 10 settings. The design was pre-registered in `config.yaml` after reading the committed runner rather than inferring it: the same width-6 model scores two conditions, so both maximise their suspicion score over the same 30 ordered off-diagonal entries and the width change L22 names is removed by construction.

| condition | model width | data | corruption | mean suspicion score |
|---|---|---|---|---|
| `clean_w4` | 4 | 4-view | none | 0.3590 +/- 0.0394 |
| `clean_w6` | 6 | 6-view | none | 0.4280 +/- 0.0337 |
| `clone_k2` | 6 | 4-view | duplicate_view k=2 | 0.5243 +/- 0.0203 |

### 5.1 Detector AUC, both references

| comparison | detector AUC |
|---|---|
| `clone_k2` against `clean_w4` (mixed width, the committed figure) | 0.8743 +/- 0.0583 |
| `clone_k2` against `clean_w6` (equal width, new) | 0.7652 +/- 0.0843 |
| paired difference, equal minus mixed, within seed | -0.1091 +/- 0.0443 |

| seed | mixed-width AUC | equal-width AUC | paired difference |
|---|---|---|---|
| 0 | 0.8007 | 0.6626 | -0.1380 |
| 1 | 0.9188 | 0.7605 | -0.1583 |
| 2 | 0.9413 | 0.8734 | -0.0679 |
| 3 | 0.8331 | 0.7092 | -0.1239 |
| 4 | 0.8777 | 0.8203 | -0.0574 |

The paired difference is negative in 5 of 5 seeds, so the mean is not a cancelling population averaging to a drop.

### 5.2 Pre-registered rule and verdict (L22)

> part explained if the equal-width AUC is below the committed AUC by more than the paired difference's own std; none if the paired mean is within one std of 0; a higher equal-width AUC is reported as L22's direction not confirmed

Equal-width AUC 0.7652 against the committed mixed-width 0.8743; paired difference -0.1091 +/- 0.0443, whose magnitude exceeds its own std by a factor of 2.5. **Verdict: WIDTH EXPLAINS PART OF THE COMMITTED FIGURE.**

**What this does and does not say.** The width change accounts for 0.1091 AUC, which is 29% of the committed figure's margin over chance. L22's direction of concern is CONFIRMED: part of 0.8743 was the width. The rest is not - at equal width the detector still separates the cloned condition from a clean 6-view reference at 0.7652 +/- 0.0843, well above chance, so duplication is detected and not merely counted. L22's operative warning is unchanged: the committed 0.8743 is measured at mixed widths and must not be differenced against a fixed-width result. 0.7652 is the number that may be.

## 6. Two defects found in `docs/KNOWN_LIMITATIONS.md` itself

Both are in a frozen document. Neither was fixed here. Both are REPORTED, with proposed replacement text in `docs/proposed_limitations_update.md` for the user to apply.

### 6.1 L9 quotes two different truth columns in one bullet list

`results/eniv_validation/estimator_vs_truth.csv` carries two analytic columns, `analytic_n_eff` and `analytic_n_eff_eigen`, which differ at every rho above 0. L9's MAE of **0.5220 +/- 0.2998** and correlation of **0.9922** are against `analytic_n_eff_eigen` (the other column gives MAE 0.7523 and correlation 0.9069, so the attribution is not in doubt). The six signed errors in the next bullet - **-1.0049, +0.2557, +0.7232, +0.8831, +0.8756, +0.7714** - are against `analytic_n_eff`, and so is the "true 1.039 at rho = 0.95" in the bullet before them (the eigen truth there is 1.15). All of these numbers are individually correct; they are not mutually comparable.

Against the eigen column throughout, the V = 4 signed errors are **-1.0049** at rho = 0.0, **-0.6443** at rho = 0.2, **-0.2587** at rho = 0.4, **+0.1117** at rho = 0.6, **+0.4520** at rho = 0.8, **+0.6604** at rho = 0.95. The qualitative claim L9 draws - compression toward the middle - survives under either column, and so does L9's sentence "understates independence when views are independent and overstates it when they are not". What moves is the crossing point: under the eigen column the sign flips between rho = 0.4 and rho = 0.6, two grid steps later than the rho = 0.0 to rho = 0.2 crossing the quoted series shows.

### 6.2 L11 quotes a tail AUC floor that excludes one of its own points

L11's closing condition reads "If tail AUC stayed at the measured **0.4846 +/- 0.0022 to 0.5856 +/- 0.0126**". The four committed tail AUC means in `results/tail/tail.json` are 0.4846 (eps 0.2), 0.4801 (eps 0.5), 0.5370 (eps 1.0) and 0.5856 (eps 2.0), so the committed span is 0.4801 to 0.5856 and the quoted floor of 0.4846 sits ABOVE the eps 0.5 point it is meant to cover. Part 36 evaluated both bands rather than choosing one; section 4.2 reports both.

## 7. Proposed status per limitation

These are PROPOSALS. `docs/KNOWN_LIMITATIONS.md` is not edited by Part 36; the replacement text is in `docs/proposed_limitations_update.md` for the user to apply or reject.

| limitation | what Part 36 measured | proposed status | why |
|---|---|---|---|
| [[L9]] | SCALES - MAE 0.5220 at V = 4 to 2.2995 at V = 16 | **OPEN (first half answered)** | the view-count half of the closing condition is answered, and answered against the estimator; the real-data half has no dataset in this checkout |
| [[L2]] | DOES NOT ACCUMULATE on the analytic-rho generator (-0.0613 from V = 4 to V = 16) | **OPEN (scope correction proposed)** | the residual does not accumulate when the added views are independent, so L2's title overstates its harness; the causal link to the confidence drift is still untested |
| [[L10]] | STRUCTURAL FLOOR in 4 of 4 cells | **OPEN (sharpened)** | both components are measured and the bias at the largest n is many stds from 0, but six points cannot separate a true floor from a slower-than-1/n decay (section 3.4) |
| [[L11]] | SUBSTANTIVE on the primary band [0.4801, 0.5856] | **CLOSED as confirmed** | L11's own rule, run at the 4x it asks for |
| [[L22]] | WIDTH EXPLAINS PART OF THE COMMITTED FIGURE - equal-width AUC 0.7652 against the committed 0.8743 | **CLOSED as confirmed** | the equal-width clone condition L22 asks for now exists; the limitation's factual claims all stand and the comparable number is measured |

## 8. What Part 36 did not do

- L9 second half: a real dataset with an independently established dependence structure does not exist in this checkout, so the external- validity half of L9 cannot be addressed here and L9 cannot be closed on the strength of Part 36 alone.
- L2's causal link: L2 says the connection between residual per-pair dependence and prismflow's confidence drift at rho=0.0 is UNTESTED. Part 36 measures the residual dependence against view count. It does not test the link to the drift, which needs the clone_eigen_softcluster harness, so that half stays open.
- The post-hoc power-law check of section 3.4 is NOT pre-registered and is labelled as such wherever it appears. It changes no verdict.
- No frozen module and no earlier experiment's code was edited. Every generator, trainer and detector used here was imported.

## 9. Files

| path | what it is |
|---|---|
| `experiments/part36_synthetic/config.yaml` | the pre-registration |
| `experiments/part36_synthetic/run_part36.py` | the runner, which applies the rules |
| `experiments/part36_synthetic/render.py` | this README's generator |
| `tests/unit/test_part36_synthetic.py` | tests for the decision rules |
| `results/part36_synthetic/reproduction_eniv_4view.json` | reproduction eniv 4view |
| `results/part36_synthetic/analysis_a.json` | analysis a |
| `results/part36_synthetic/analysis_b.json` | analysis b |
| `results/part36_synthetic/analysis_c.json` | analysis c |
| `results/part36_synthetic/analysis_d.json` | analysis d |

