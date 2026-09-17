# Frozen-naive control (Part 07 followup 2)

**Question.** In the calibration-under-duplication run, the naive system at k
copies was *trained* on data containing those copies. Does its partial
robustness come from training-time adaptation to duplicates? Would a naive
model that never saw a duplicate degrade much more sharply?

**Verdict: training adaptation absorbs confidence inflation, not
discrimination loss.**

- **Resolution and accuracy do not collapse more sharply for naive_frozen.**
  From k = 0 to k = 4, resolution falls -0.0340 for naive_frozen and -0.0307
  for naive. Accuracy falls -0.0213 and -0.0227. The within-seed difference
  between the two is null on both. The discrimination loss is what duplication
  itself costs a naive fuser, and training on the copies does not recover it.
- **Confidence inflation is what training absorbs.** naive_frozen's top-label
  confidence rises +0.0571 by k = 4 (5/5 seeds), about three times naive's
  +0.0184. It is higher than naive's in 5/5 seeds at both k = 2 and k = 4.
  naive_frozen ends clearly overconfident: confidence minus accuracy is +0.056
  at k = 4, against +0.018 for naive and -0.028 for PrismFlow.
- **That extra confidence only partly reaches the calibration metrics.** ECE
  worsens more for naive_frozen than for naive in 4/5 seeds (+0.0150 +/- 0.0172
  more at k = 4), but not separably. Brier reliability shows no difference
  (+0.0040 +/- 0.0083).

**For the earlier finding:** the confound does not explain it away. Against a
model that never adapted to duplicates, PrismFlow's advantage is the same
shape as before and slightly larger. It separates on resolution and AURC in
5/5 seeds and leans on ECE in 4/5. Brier reliability is still null, even
against naive_frozen.

All numbers come from `run_calibration_frozen_naive.py` executed on
2026-09-17: seeds 0-4, mean +/- sample std, 300 test samples per seed. Every
metric was computed by `prismflow.evaluation.evaluate`. naive_frozen outputs
are in `results/calibration_frozen_naive/k<k>_naive_frozen/`. The comparison
systems are read from the per-seed rows in `results/calibration_duplicated/`.
The full tables are in `results/calibration_frozen_naive/summary.md`.

## Design (fixed before the run)

- Base data, seeds and training settings are read from `config_duplicated.yaml`:
  4 views, rho = 0.3, k = 0, 2, 4 copies of view 0, 40 epochs.
- **naive_frozen**: for each seed, the naive model is trained at k = 0 with the
  same function and seed as the duplicated run. Its weights are then frozen.
  A model's encoder count is fixed at construction, so for k copies it is
  widened into a (4 + k)-view naive model. Slots 0-3 hold the trained weights
  unchanged. Each of the k extra slots holds an exact copy of view 0's trained
  encoder and evidence head. A duplicated input therefore produces evidence
  bit-identical to view 0's, and Dempster fusion counts it 1 + k times. This is
  asserted on every evaluation batch. Nothing is trained after k = 0.
- **Reproduction check (passed).** The retrained k = 0 models must match
  `calibration_duplicated/k0_naive` exactly before any comparison is made.
  Measured max absolute difference, over every seed and compared metric: 0.
- Comparison systems (naive and prismflow trained with the copies, and
  naive weights + discount) are not retrained. Their per-seed `evaluate()`
  rows from the duplicated run are paired with naive_frozen by seed.

## Change from k = 0, within seed

| metric | system | k = 2 | k = 4 |
|---|---|---|---|
| accuracy | naive_frozen | -0.0133 +/- 0.0155 (1/5 up) | -0.0213 +/- 0.0257 (1/5 up) |
| accuracy | naive | -0.0200 +/- 0.0133 (0/5 up) | -0.0227 +/- 0.0095 (0/5 up) |
| accuracy | prismflow | -0.0040 +/- 0.0119 (2/5 up) | -0.0073 +/- 0.0128 (2/5 up) |
| resolution | naive_frozen | -0.0155 +/- 0.0145 (1/5 up) | -0.0340 +/- 0.0283 (0/5 up) |
| resolution | naive | -0.0176 +/- 0.0123 (1/5 up) | -0.0307 +/- 0.0200 (1/5 up) |
| resolution | prismflow | -0.0016 +/- 0.0068 (3/5 up) | -0.0032 +/- 0.0068 (2/5 up) |
| top-label confidence | naive_frozen | +0.0402 +/- 0.0084 (5/5 up) | +0.0571 +/- 0.0141 (5/5 up) |
| top-label confidence | naive | +0.0117 +/- 0.0092 (5/5 up) | +0.0184 +/- 0.0084 (5/5 up) |
| top-label confidence | prismflow | +0.0039 +/- 0.0094 (4/5 up) | +0.0087 +/- 0.0110 (4/5 up) |
| ECE | naive_frozen | +0.0086 +/- 0.0195 (4/5 up) | +0.0285 +/- 0.0319 (4/5 up) |
| ECE | naive | +0.0028 +/- 0.0154 (4/5 up) | +0.0136 +/- 0.0215 (4/5 up) |
| ECE | prismflow | -0.0035 +/- 0.0161 (2/5 up) | +0.0027 +/- 0.0267 (2/5 up) |
| reliability | naive_frozen | +0.0042 +/- 0.0048 (4/5 up) | +0.0061 +/- 0.0069 (3/5 up) |
| reliability | naive | +0.0011 +/- 0.0065 (3/5 up) | +0.0021 +/- 0.0065 (3/5 up) |
| reliability | prismflow | +0.0059 +/- 0.0089 (4/5 up) | +0.0090 +/- 0.0061 (5/5 up) |
| Brier | naive_frozen | +0.0199 +/- 0.0167 (4/5 up) | +0.0395 +/- 0.0280 (5/5 up) |
| Brier | naive | +0.0187 +/- 0.0087 (5/5 up) | +0.0320 +/- 0.0166 (5/5 up) |
| Brier | prismflow | +0.0075 +/- 0.0041 (5/5 up) | +0.0121 +/- 0.0105 (5/5 up) |
| AURC | naive_frozen | +0.0133 +/- 0.0059 (5/5 up) | +0.0258 +/- 0.0106 (5/5 up) |
| AURC | naive | +0.0110 +/- 0.0047 (5/5 up) | +0.0224 +/- 0.0112 (5/5 up) |
| AURC | prismflow | +0.0036 +/- 0.0024 (5/5 up) | +0.0075 +/- 0.0052 (5/5 up) |

## Does naive_frozen degrade more sharply than naive?

(naive_frozen change) - (naive change), within seed:

| metric | k = 2 | k = 4 |
|---|---|---|
| accuracy | +0.0067 +/- 0.0085 | +0.0013 +/- 0.0291 |
| resolution | +0.0021 +/- 0.0095 | -0.0032 +/- 0.0187 |
| top-label confidence | **+0.0285 +/- 0.0151 (5/5 higher)** | **+0.0387 +/- 0.0185 (5/5 higher)** |
| ECE | +0.0058 +/- 0.0074 (4/5 higher) | +0.0150 +/- 0.0172 (4/5 higher) |
| reliability | +0.0031 +/- 0.0066 | +0.0040 +/- 0.0083 |
| Brier | +0.0013 +/- 0.0095 | +0.0075 +/- 0.0210 |
| AURC | +0.0024 +/- 0.0045 | +0.0034 +/- 0.0074 |

Only confidence separates.

## At k = 4: naive_frozen against both systems

| metric | naive_frozen | naive | prismflow | frozen - prismflow, paired | seeds frozen worse |
|---|---|---|---|---|---|
| accuracy | 0.8087 +/- 0.0678 | 0.8073 +/- 0.0854 | 0.8227 +/- 0.0789 | -0.0140 +/- 0.0205 | 4/5 |
| top-label confidence | 0.8643 +/- 0.0430 | 0.8256 +/- 0.0568 | 0.7943 +/- 0.0576 | +0.0700 +/- 0.0186 | n/a (5/5 higher) |
| ECE | 0.0865 +/- 0.0297 | 0.0715 +/- 0.0199 | 0.0664 +/- 0.0188 | +0.0201 +/- 0.0234 | 4/5 |
| Brier | 0.2906 +/- 0.1044 | 0.2831 +/- 0.1109 | 0.2633 +/- 0.0977 | +0.0273 +/- 0.0245 | 4/5 |
| reliability | 0.0370 +/- 0.0119 | 0.0331 +/- 0.0113 | 0.0372 +/- 0.0041 | -0.0002 +/- 0.0116 | 2/5 |
| resolution | 0.4103 +/- 0.0929 | 0.4135 +/- 0.1039 | 0.4378 +/- 0.0969 | -0.0275 +/- 0.0242 | **5/5** |
| AURC | 0.0866 +/- 0.0566 | 0.0831 +/- 0.0582 | 0.0677 +/- 0.0487 | +0.0189 +/- 0.0097 | **5/5** |

Uncertainty is 0.6648 +/- 0.0014 in every cell. `brier_within_bin` is at most
0.0012 in magnitude.

## Reading it

- **The discrimination cost of duplication is structural, not learned.** A
  naive fuser that counts one view 1 + k times loses about as much resolution
  and accuracy whether or not it trained on the copies. This supports the
  earlier hypothesis that the over-weighted view dominates the fused opinion.
  It is still inferred, not measured: per-view evidence magnitudes were not
  logged.
- **Training on duplicates teaches naive to damp confidence, about two thirds of
  the inflation** (+0.018 against +0.057). So the trained naive baseline in the
  duplicated run was a *stronger* calibration baseline than a truly
  duplicate-unaware model. That makes the calibration comparison there
  conservative, and it partly explains why ECE did not separate.
- **Even so, the extra overconfidence barely shows in ECE or reliability** at
  this sample size. Confidence rises 5/5, and ECE rises 4/5 but inside noise.
  Reliability does not move at all. A confidence shift of about 0.04 is not
  enough to separate these metrics with 300 test samples and 5 seeds.
- **Brier reliability is null against every naive variant**, and PrismFlow's
  own reliability still worsens with k (+0.0090, 5/5). The "improves
  reliability, not resolution" claim is not rescued by this control. The
  measured advantage remains resolution and AURC.

## What this control does not isolate

PrismFlow here is still trained on the copies, so its advantage combines the
discount mechanism with its own training adaptation (see
`docs/KNOWN_LIMITATIONS.md` L1). The clean mechanism-only test is the frozen
k = 0 naive model with the discount switched on at inference. That was not
run. The naive_frozen widening is also a stronger form of duplication than
the trained systems see: the copies share view 0's encoder exactly. In the
trained models, each copy has its own separately trained encoder.

## Limitations

- One base condition (rho = 0.3, 4 views, view 0 duplicated), k up to 4.
- 300 test samples per seed, 5 seeds. The nulls mean "not detected at this
  power".
- The comparison systems come from the earlier run's saved per-seed rows. That
  pairing is valid only because the k = 0 reproduction was exact.

## Reproduce

```
python -m experiments.calibration.run_calibration_duplicated    # prerequisite, ~20 min
python -m experiments.calibration.run_calibration_frozen_naive  # ~2 min on CPU
```
