# Robustness experiments (Part 08)

**Questions.** Does PrismFlow degrade more gracefully than naive fusion when
views go missing or turn noisy? Does confidence fall appropriately as inputs
are lost? And does an imputation baseline buy accuracy at the cost of honesty?

**Verdict.**

1. **PrismFlow is never better than naive fusion here, on any metric, at any
   missing rate or noise level, by the error-bar rule.** Every mean +/- std
   interval overlaps. Under the weaker paired test it is *worse* on ECE at 30%
   and 50% missing (0/5 seeds better), and worse on accuracy and Brier at the
   highest noise levels (5/5 seeds).
2. **Confidence falls appropriately as views go missing**, for both systems,
   and roughly tracks the accuracy drop.
3. **Confidence does not fall as views turn noisy.** Accuracy drops by up to
   0.17 while top-label confidence moves by at most 0.015 and evidence-mass
   confidence *rises*. Both systems do this. The discount does not detect a
   noisy view at all: ENIV is flat in sigma.
4. **The imputation prediction reproduces in mechanism but not in premise.**
   Imputing a missing view as the mean of the others inflates confidence and
   raises ECE and Brier reliability, as predicted. But it does not improve
   accuracy: it makes accuracy substantially worse, so the specific claim
   "ECE rises even where accuracy improves" is not what happens here.
   **Caveat: this is a statement about this dataset, not about imputation.**
   Here the views are independent random projections of a shared latent, so a
   mean of the available views is not a plausible substitute view and the
   encoder receives an out-of-distribution input. The premise half of the
   hypothesis is untested until it is run where views share a feature space
   and imputation can actually recover accuracy.

All numbers come from `run_missing.py` and `run_noise.py` executed on
2026-09-17: 5 seeds (0-4), mean +/- sample std, 300 test samples per seed.
Every metric was computed by `prismflow.evaluation.evaluate`. Full tables are
`results/robustness/missing_summary.md` and `noise_summary.md`; per-condition
outputs are in `results/robustness/<condition>_<system>/`.

## Design (fixed before the run)

- Data and training identical to the Part 04 baseline and Part 07: 4 views,
  rho = 0.3, 40 epochs. Per seed, one naive and one PrismFlow model, same seed,
  so initial weights are identical and only `use_discount` differs.
- **Models are trained on clean data.** Corruption is applied at test time only,
  so every condition scores the same trained models. This measures robustness to
  degradation the model was not trained for.
- Corruption seeds depend on (seed, batch index, condition), so both systems see
  exactly the same dropped entries and the same noise draws.
- Missing: `drop_views` at 0/10/30/50/70%, each (sample, view) independently.
  A sample can lose every view; it then receives the vacuous opinion (uniform
  probabilities, confidence 0), which is the honest output. At 70% about 24% of
  samples are in that state.
- Noise: `add_noise` on view 0 (`one_noisy`) or views 0 and 1 (`two_noisy`), and
  view 0's content replaced by pure noise (`one_severe`), with sigma in
  {0.25, 0.5, 1, 2, 4}. View features have std ~1.3, so the sweep spans about
  0.2x to 3x the view scale.
- Imputation: a missing view's input is replaced by the mean of that sample's
  available view inputs and marked present, then scored with the naive model
  (`imputed_naive`) and with PrismFlow (`imputed_prismflow`). Samples with no
  available view stay fully missing.

## 1. Missing views

| missing rate | accuracy (naive / prismflow) | ECE | Brier reliability |
|---|---|---|---|
| 0% | 0.8300 +/- 0.0791 / 0.8300 +/- 0.0797 | 0.0579 +/- 0.0094 / 0.0637 +/- 0.0137 | 0.0310 +/- 0.0067 / 0.0282 +/- 0.0081 |
| 10% | 0.8133 +/- 0.0741 / 0.8160 +/- 0.0761 | 0.0643 +/- 0.0103 / 0.0664 +/- 0.0116 | 0.0328 +/- 0.0044 / 0.0302 +/- 0.0079 |
| 30% | 0.7973 +/- 0.0695 / 0.8007 +/- 0.0698 | 0.0782 +/- 0.0219 / 0.1023 +/- 0.0245 | 0.0395 +/- 0.0112 / 0.0422 +/- 0.0106 |
| 50% | 0.7493 +/- 0.0596 / 0.7493 +/- 0.0618 | 0.0982 +/- 0.0136 / 0.1301 +/- 0.0255 | 0.0375 +/- 0.0111 / 0.0450 +/- 0.0128 |
| 70% | 0.6300 +/- 0.0755 / 0.6373 +/- 0.0728 | 0.0757 +/- 0.0338 / 0.0856 +/- 0.0313 | 0.0275 +/- 0.0050 / 0.0301 +/- 0.0081 |

Figures: `ece_vs_missing.png` (headline), `missing_overview.png`.

**No separation anywhere.** Every mean +/- std interval overlaps, so nothing
here may be reported as a difference between the systems.

**Where PrismFlow is not better** (required by the protocol; paired
within-seed, bars still overlap):

| condition | metric | prismflow - naive | seeds better |
|---|---|---|---|
| 30% missing | ECE | +0.0241 +/- 0.0070 | 0/5 |
| 50% missing | ECE | +0.0320 +/- 0.0141 | 0/5 |
| 50% missing | Brier reliability | +0.0075 +/- 0.0083 | 1/5 |
| sigma 4, one_noisy | accuracy | -0.0193 +/- 0.0083 | 0/5 |
| sigma 4, one_noisy | Brier | +0.0201 +/- 0.0087 | 0/5 |
| sigma 4, one_severe | accuracy | -0.0293 +/- 0.0157 | 0/5 |
| sigma 4, one_severe | Brier | +0.0420 +/- 0.0103 | 0/5 |
| sigma 4, one_severe | ECE | +0.0134 +/- 0.0143 | 0/5 |

PrismFlow's ECE penalty at 30-50% missing has the same cause as its behaviour
elsewhere: it is consistently less confident (top-label confidence lower in 5/5
seeds at every rate), and both systems are already *under*confident once views
start disappearing. Discounting further moves it away from calibration, not
toward it.

**ENIV under masking is not trustworthy.** It falls 3.33 -> 3.10 -> 2.85 from 0%
to 50%, then jumps back to 3.34 at 70%. With most views absent, each pairwise
dependence estimate rests on few samples that have both views present. The 70%
value should not be read as "dependence recovered".

## 2. Does confidence fall appropriately? (secondary hypothesis)

Figure: `confidence_vs_missing.png`.

| missing rate | accuracy (naive) | top-label confidence | confidence - accuracy | 1 - uncertainty |
|---|---|---|---|---|
| 0% | 0.8300 | 0.8072 +/- 0.0557 | -0.023 | 0.7790 +/- 0.0829 |
| 10% | 0.8133 | 0.7854 +/- 0.0528 | -0.028 | 0.7365 +/- 0.0811 |
| 30% | 0.7973 | 0.7385 +/- 0.0491 | -0.059 | 0.6521 +/- 0.0744 |
| 50% | 0.7493 | 0.6623 +/- 0.0510 | -0.087 | 0.5189 +/- 0.0775 |
| 70% | 0.6300 | 0.5686 +/- 0.0427 | -0.061 | 0.3632 +/- 0.0627 |

**Yes, and the slope is plausible.** Losing 70% of view entries costs 0.200
accuracy and 0.239 top-label confidence, so confidence falls slightly faster
than accuracy: both systems drift from mildly underconfident to more
underconfident, never overconfident. Evidence-mass confidence (1 - uncertainty)
falls much harder, from 0.779 to 0.363, which is the appropriate response for a
quantity that measures how much evidence arrived rather than how likely the
answer is. PrismFlow's curve is the same shape, shifted down by about 0.02-0.03.

Two floors are worth noting when reading the curve. Top-label probability cannot
fall below 1/K = 0.333, and a sample with no views left produces exactly that,
with confidence 0. So the 70% point is partly a mixture of vacuous samples
rather than a smooth degradation.

## 3. Noise

Figures: `noise_one_noisy.png`, `noise_two_noisy.png`, `noise_one_severe.png`.

Naive fusion, one noisy view:

| sigma | accuracy | ECE | top-label confidence | 1 - uncertainty | ENIV (prismflow) |
|---|---|---|---|---|---|
| 0.25 | 0.8320 +/- 0.0777 | 0.0587 +/- 0.0107 | 0.8078 +/- 0.0549 | 0.7793 +/- 0.0816 | 3.3495 +/- 0.0810 |
| 0.5 | 0.8253 +/- 0.0763 | 0.0475 +/- 0.0158 | 0.8079 +/- 0.0560 | 0.7811 +/- 0.0848 | 3.3119 +/- 0.1124 |
| 1 | 0.8247 +/- 0.0787 | 0.0559 +/- 0.0197 | 0.8048 +/- 0.0516 | 0.7816 +/- 0.0853 | 3.3799 +/- 0.1066 |
| 2 | 0.7953 +/- 0.0693 | 0.0570 +/- 0.0138 | 0.7997 +/- 0.0521 | 0.7872 +/- 0.0893 | 3.3486 +/- 0.1069 |
| 4 | 0.7340 +/- 0.0728 | 0.1009 +/- 0.0374 | 0.7936 +/- 0.0487 | 0.7955 +/- 0.0856 | 3.3130 +/- 0.0552 |

**No separation between the systems at any sigma in any condition.**

**The real finding here is a shared failure.** As one view's noise goes from
0.25 to 4, naive accuracy falls 0.098 while top-label confidence falls 0.014 and
evidence-mass confidence *rises* 0.016. With two noisy views accuracy falls
0.168 and ECE reaches 0.1725 +/- 0.0659. Under `one_severe`, where view 0 carries
no signal at all, naive accuracy falls 0.090 and confidence again rises slightly.
A noisy view still produces evidence, and Dempster fusion adds it. Neither system
notices, and PrismFlow's discount cannot help, because **ENIV is flat in sigma**
(3.2-3.4 across every sigma in all three conditions): noise makes a view *less* dependent on the
others, not more, so a dependence-based discount is the wrong instrument for it.
This is the same miscalibration the project objects to under duplication, arriving
by a different route, and PrismFlow as built does not address it.

## 4. Imputation baseline (third hypothesis)

Figure: `imputation_vs_missing.png`.

`imputed_naive` against plain masking (`naive`), paired within seed:

| missing rate | accuracy | ECE | Brier reliability | top-label confidence |
|---|---|---|---|---|
| 30% | -0.0493 +/- 0.0243 (0/5 better) | -0.0070 +/- 0.0430 | +0.0036 +/- 0.0155 | +0.0206 +/- 0.0234 |
| 50% | -0.0987 +/- 0.0154 (0/5) | -0.0007 +/- 0.0350 | +0.0069 +/- 0.0166 | +0.0388 +/- 0.0297 |
| 70% | -0.0927 +/- 0.0266 (0/5) | +0.0236 +/- 0.0598 | +0.0219 +/- 0.0244 | **+0.0555 +/- 0.0416 (bars separate)** |

**Partly reproduces.** The predicted mechanism is visible: the imputed view is a
deterministic function of the others, it adds belief mass without adding
information, and confidence rises because of it. At 70% missing, imputation
raises top-label confidence by +0.0555 (the only separated comparison in the
whole missing-view experiment), raises evidence-mass confidence from 0.363 to
0.538, and raises ECE from 0.0757 to 0.0994 and Brier reliability from 0.0275 to
0.0494. ECE rises with missing rate for `imputed_naive` beyond the 10% point
(0.058 -> 0.056 -> 0.071 -> 0.098 -> 0.099).

**The premise does not hold.** The prediction was that this would happen *even
where accuracy improves*. Here accuracy does not improve: imputation costs about
0.09-0.10 accuracy at 50-70% missing, in 5/5 seeds, and Brier is worse at every
rate (+0.0952 +/- 0.0223 at 70%, 0/5 seeds better). So this is not the novel
"buys accuracy, pays in honesty" trade the hypothesis describes; it is worse on
both counts. The likely reason is specific to this data: views are independent
random projections of a shared latent, so the mean of three views is not a
plausible fourth view, and the encoder receives an input unlike anything it
trained on. A dataset where views share a feature space would test the
hypothesis more fairly.

**Where the discount does something useful.** Against `imputed_naive`,
`imputed_prismflow` lowers ECE by -0.0282 +/- 0.0146 at 50% (5/5 seeds) and
-0.0361 +/- 0.0260 at 70% (5/5), lowers Brier reliability (5/5 at 50%), and
lowers confidence by -0.0474 +/- 0.0047 at 70% (bars separate). ENIV also drops
to 2.80 +/- 0.13 for imputed inputs at 70%, against 3.34 for masked ones: the
estimator detects that the reconstructed view is redundant. Accuracy is
unchanged. This is the one place in Part 08 where the discount measurably
improves calibration, and the ECE and reliability gains are still within
overlapping bars.

## Limitations

- One synthetic setting, 300 test samples per seed, 5 seeds. Non-overlap of
  mean +/- std is a coarse test; "no separation" means "not detected here".
- Models are trained on clean data only. Training with missing or noisy views
  would be a different, and probably more favourable, experiment for both
  systems.
- ENIV under heavy masking is estimated from few co-present samples (see above).
- The imputation is input-space and crude by construction (see above).

## Reproduce

```
python -m experiments.robustness.run_missing        # ~6 min on CPU
python -m experiments.robustness.run_noise          # ~8 min on CPU
python -m experiments.robustness.plot_robustness    # writes the seven PNGs
```
