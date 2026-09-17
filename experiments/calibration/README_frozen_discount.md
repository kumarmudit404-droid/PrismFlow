# Mechanism-only control (Part 07 followup 3)

**Question.** Under duplication, PrismFlow keeps resolution, accuracy and AURC
better than naive fusion. Is that the discount mechanism itself, or PrismFlow's
training adapting to a discount it experiences?

**Condition.** `naive_frozen_discounted`: the naive model trained at k = 0,
weights frozen, widened to 4 + k views exactly as in the frozen-naive control
(copy slots clone view 0's trained encoder and head). The only change is that
the discount is switched ON at inference. No retraining of any kind. Compared
within seed against `naive_frozen`, the same model with the discount off, the
difference is the discount's effect alone.

**Verdict: the mechanism alone reproduces PrismFlow's duplication robustness.
Training contributes a fixed level offset, not the robustness.**

- **Robustness to added copies (the slope): the mechanism matches PrismFlow.**
  From k = 0 to k = 4, relative to naive_frozen, the mechanism alone and trained
  PrismFlow protect by almost the same amounts:
  - resolution: +0.0306 vs +0.0308, 5/5 seeds each
  - AURC: -0.0175 vs -0.0183, 5/5 each
  - Brier: -0.0301 vs -0.0274, 5/5 each
  - accuracy: +0.0127 vs +0.0140, 4/5 each
  - ECE: -0.0290 vs -0.0258, 4/5 vs 5/5
- **Level: the untrained discount carries a constant cost, and training
  removes it.** At k = 0, with no duplicates, the discount at inference costs
  Brier +0.0091 +/- 0.0040 and resolution -0.0089 +/- 0.0042, both 5/5 seeds
  worse. The offset persists at k = 4: mechanism-only minus PrismFlow is Brier
  +0.0063 +/- 0.0039 (worse in 5/5) and AURC +0.0032 +/- 0.0038 (worse in 4/5).
- **Reliability: null throughout,** for the mechanism as for every other
  system.

This is **not** the pattern the earlier "discount without training" conditions
suggested. See "Why the earlier conditions looked different" below.

All numbers come from `run_calibration_frozen_discount.py` executed on
2026-09-17: seeds 0-4, mean +/- sample std, 300 test samples per seed. Every
metric was computed by `prismflow.evaluation.evaluate`. Outputs are in
`results/calibration_frozen_discount/k<k>_naive_frozen_discounted/`. The full
tables, including all five systems at each k, are in
`results/calibration_frozen_discount/summary.md`.

## Design (fixed before the run)

- Base data, seeds and training settings are read from `config_duplicated.yaml`:
  4 views, rho = 0.3, k = 0, 2, 4 copies of view 0.
- The k = 0 naive models are retrained from the same seeds with the same
  function. Nothing is optimised afterwards.
- Copy-slot evidence is asserted bit-identical to view 0's on every
  evaluation batch.
- **Reproduction checks (both passed, max absolute difference 0):**
  - at k = 0, the condition equals `calibration_duplicated/k0_naive_weights_discounted`
    (the same trained model with the same discount);
  - with the discount toggled off, k = 0 equals `calibration_frozen_naive/k0_naive_frozen`.
  So the weights are exactly those of the earlier runs, and the per-seed pairing
  with them is valid.

## Change from k = 0 to k = 4, within seed

| metric | frozen + discount | frozen, no discount | prismflow | naive |
|---|---|---|---|---|
| resolution | -0.0034 +/- 0.0109 | -0.0340 +/- 0.0283 | -0.0032 +/- 0.0068 | -0.0307 +/- 0.0200 |
| accuracy | -0.0087 +/- 0.0077 | -0.0213 +/- 0.0257 | -0.0073 +/- 0.0128 | -0.0227 +/- 0.0095 |
| AURC | +0.0083 +/- 0.0048 | +0.0258 +/- 0.0106 | +0.0075 +/- 0.0052 | +0.0224 +/- 0.0112 |
| Brier | +0.0094 +/- 0.0090 | +0.0395 +/- 0.0280 | +0.0121 +/- 0.0105 | +0.0320 +/- 0.0166 |
| ECE | -0.0005 +/- 0.0149 | +0.0285 +/- 0.0319 | +0.0027 +/- 0.0267 | +0.0136 +/- 0.0215 |
| reliability | +0.0060 +/- 0.0035 | +0.0061 +/- 0.0069 | +0.0090 +/- 0.0061 | +0.0021 +/- 0.0065 |
| top-label confidence | +0.0125 +/- 0.0062 | +0.0571 +/- 0.0141 | +0.0087 +/- 0.0110 | +0.0184 +/- 0.0084 |

The mechanism-only degradation profile is nearly identical to PrismFlow's.

## Protection relative to naive_frozen

(system's change from k = 0) - (naive_frozen's change from k = 0), within seed.
The sign convention is "better": positive for resolution and accuracy, negative
for the others.

| metric | system | k = 2 | k = 4 |
|---|---|---|---|
| resolution | frozen + discount | +0.0164 +/- 0.0143 (4/5) | +0.0306 +/- 0.0270 (5/5) |
| resolution | prismflow | +0.0140 +/- 0.0163 (3/5) | +0.0308 +/- 0.0278 (5/5) |
| accuracy | frozen + discount | +0.0100 +/- 0.0156 (4/5) | +0.0127 +/- 0.0213 (4/5) |
| accuracy | prismflow | +0.0093 +/- 0.0086 (4/5) | +0.0140 +/- 0.0164 (4/5) |
| AURC | frozen + discount | -0.0086 +/- 0.0038 (5/5) | -0.0175 +/- 0.0071 (5/5) |
| AURC | prismflow | -0.0098 +/- 0.0059 (5/5) | -0.0183 +/- 0.0091 (5/5) |
| Brier | frozen + discount | -0.0166 +/- 0.0132 (5/5) | -0.0301 +/- 0.0228 (5/5) |
| Brier | prismflow | -0.0124 +/- 0.0138 (4/5) | -0.0274 +/- 0.0241 (5/5) |
| ECE | frozen + discount | -0.0113 +/- 0.0208 (3/5) | -0.0290 +/- 0.0216 (4/5) |
| ECE | prismflow | -0.0121 +/- 0.0178 (4/5) | -0.0258 +/- 0.0145 (5/5) |
| reliability | frozen + discount | +0.0006 +/- 0.0059 | -0.0001 +/- 0.0091 |
| reliability | prismflow | +0.0017 +/- 0.0109 | +0.0029 +/- 0.0094 |

(n/5) = seeds in the "better" direction.

## Level: mechanism-only minus PrismFlow, same seed

| metric | k = 0 | k = 2 | k = 4 |
|---|---|---|---|
| Brier | +0.0090 +/- 0.0024 (5/5 worse) | +0.0048 +/- 0.0031 (5/5 worse) | +0.0063 +/- 0.0039 (5/5 worse) |
| resolution | -0.0056 +/- 0.0037 (4/5 worse) | -0.0032 +/- 0.0036 (4/5 worse) | -0.0058 +/- 0.0098 (4/5 worse) |
| AURC | +0.0024 +/- 0.0020 (4/5 worse) | +0.0035 +/- 0.0038 (4/5 worse) | +0.0032 +/- 0.0038 (4/5 worse) |
| ECE | +0.0121 +/- 0.0088 (4/5 worse) | +0.0129 +/- 0.0147 (3/5 worse) | +0.0089 +/- 0.0205 (3/5 worse) |
| top-label confidence | -0.0191 +/- 0.0024 (5/5 lower) | -0.0082 +/- 0.0085 (4/5 lower) | -0.0154 +/- 0.0106 (5/5 lower) |

The gap is roughly constant in k. It is present before any duplicate exists.

## Reading it

- **The discount mechanism is what makes the system robust to duplicated
  views.** With no training adaptation at all, it gives the same protection
  against added copies as trained PrismFlow on resolution, accuracy, AURC and
  Brier.
- **What training under the discount buys is a level correction.** Applied to
  weights that never experienced it, the discount also scales evidence of
  non-duplicated views, which costs resolution and lowers confidence even at
  k = 0. PrismFlow's training recovers that constant cost. This is consistent
  with the compensation described in `docs/KNOWN_LIMITATIONS.md` L1 (the model
  emits more evidence to win back discounted confidence). Per-view evidence was
  not logged, so this is inferred.
- **Reliability is not improved by the mechanism either.** The calibration
  claim stays as revised: preserved resolution/accuracy/AURC, no reliability
  gain.

## Why the earlier conditions looked different

The earlier "discount without training" diagnostic,
`naive_weights_discounted`, applied the discount to a naive model trained *on
the duplicates*. In that model, every copy has its own separately trained
encoder, so copies are near-duplicates in feature space, not exact ones. It
got about half the resolution protection (-0.0160 change at k = 4, against
-0.0034 here). Its ENIV at k = 4 is 3.88 (alpha 0.485), against 3.57
(alpha 0.446) here. That means it discounted less.

In this control the copies share view 0's encoder, so their features are
exact duplicates. That is the easiest case for the dependence estimator.
**So "the mechanism alone suffices" is established for exact feature
duplication only.** When copies have diverged into near-duplicates, the
mechanism-only result is weaker, which matches the near-duplicate limit in
`docs/KNOWN_LIMITATIONS.md`. The clone experiment's L1 finding (confidence
falls under the untrained discount) concerns vacuity confidence on
duplicate-trained weights. It is not contradicted, but it is not the whole
picture either.

## Limitations

- One base condition (rho = 0.3, 4 views, view 0 duplicated), k up to 4. 300
  test samples per seed, 5 seeds.
- Exact feature duplication only, as above.
- The attribution of the level offset to evidence compensation is inferred, not
  measured.

## Reproduce

```
python -m experiments.calibration.run_calibration_duplicated      # prerequisite, ~20 min
python -m experiments.calibration.run_calibration_frozen_naive    # prerequisite, ~2 min
python -m experiments.calibration.run_calibration_frozen_discount # ~2 min on CPU
```
