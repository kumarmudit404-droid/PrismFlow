# Calibration experiment (Part 07 protocol)

**Question.** Is PrismFlow's confidence more honest than naive fusion's? In
particular, does the discount improve Brier **reliability** (calibration)
without costing **resolution** (discrimination)?

**Verdict: not shown in this setting.** With 4 views at rho = 0.3 and no
duplicated views, PrismFlow and naive fusion cannot be told apart on any
calibration metric. Every paired difference is smaller than its std across
seeds. Switching the discount on at test time only (naive weights) makes Brier
score slightly but consistently worse, 5/5 seeds, and almost all of that loss
comes from resolution. This result does not support the reliability claim, and
it does not refute it. The claim is about redundant views, and this setting has
none.

All numbers below come from `run_calibration.py` executed on 2026-09-17:
seeds 0-4, mean +/- sample std, 300 test samples per seed. Every metric was
computed by `prismflow.evaluation.evaluate`. The full table is
`results/calibration/summary.md`, and per-system outputs are in
`results/calibration/<system>/`.

## Design (fixed before the run)

- Data: default synthetic generator, 4 views, rho = 0.3,
  signal_strength 1.15 (the Part 04 baseline setting). No duplicates, no attacks.
- Training settings match the Part 04 baseline and the clone experiment: 40
  epochs, batch 64, lr 1e-3, KL anneal 10, per-view loss weight 1.0. Fixed
  epoch count, no model selection.
- **naive**: trained and evaluated with `use_discount=False`.
- **prismflow**: trained and evaluated with `use_discount=True`, from the same
  seed. Initial weights are identical to naive's.
- **naive_weights_discounted**: a copy of the trained naive model, evaluated
  with the discount on. It isolates the discount operator from anything
  learned while training under it.
- Evaluation: the test split, batches of 64 (ENIV is estimated per batch, as in
  deployment), 15 bins, coverages 1.0/0.9/0.8/0.5.

## Results

`prob_` metrics use the top-label probability, `max_k probs[k]`. `vacuity_`
metrics use 1 - uncertainty, which is not a probability of being right (see
`docs/EVALUATION_PROTOCOL.md`).

| metric | naive | prismflow | naive weights + discount |
|---|---|---|---|
| accuracy | 0.830 +/- 0.079 | 0.830 +/- 0.080 | 0.828 +/- 0.079 |
| prob_ece | 0.0579 +/- 0.0094 | 0.0637 +/- 0.0137 | 0.0758 +/- 0.0179 |
| prob_mce | 0.294 +/- 0.131 | 0.256 +/- 0.095 | 0.299 +/- 0.090 |
| brier | 0.2511 +/- 0.1019 | 0.2512 +/- 0.0990 | 0.2602 +/- 0.0981 |
| brier_reliability (lower better) | 0.0310 +/- 0.0067 | 0.0282 +/- 0.0081 | 0.0316 +/- 0.0054 |
| brier_resolution (higher better) | 0.4442 +/- 0.0957 | 0.4410 +/- 0.0957 | 0.4354 +/- 0.0984 |
| brier_uncertainty | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 | 0.6648 +/- 0.0014 |
| brier_within_bin | -0.0004 +/- 0.0008 | -0.0008 +/- 0.0014 | -0.0007 +/- 0.0017 |
| prob_aurc | 0.0608 +/- 0.0499 | 0.0602 +/- 0.0491 | 0.0625 +/- 0.0505 |
| selective risk @ 0.5 coverage | 0.0507 +/- 0.0555 | 0.0480 +/- 0.0549 | 0.0520 +/- 0.0563 |
| vacuity_mean_confidence | 0.779 +/- 0.083 | 0.750 +/- 0.082 | 0.712 +/- 0.085 |
| vacuity_ece | 0.0905 +/- 0.0245 | 0.0909 +/- 0.0202 | 0.1220 +/- 0.0245 |
| ENIV | n/a | 3.33 +/- 0.08 | 3.36 +/- 0.08 |

The across-seed std mostly reflects how hard each seed's data is, and every
system shares that difficulty. The paired within-seed differences are the
correct comparison:

| metric | prismflow - naive | seeds lower | naive weights + discount - naive | seeds lower |
|---|---|---|---|---|
| prob_ece | +0.0058 +/- 0.0157 | 2/5 | +0.0179 +/- 0.0191 | 1/5 |
| brier | +0.0002 +/- 0.0043 | 3/5 | **+0.0091 +/- 0.0040** | 0/5 |
| brier_reliability | -0.0027 +/- 0.0069 | 4/5 | +0.0006 +/- 0.0063 | 3/5 |
| brier_resolution | -0.0033 +/- 0.0041 | 4/5 | **-0.0089 +/- 0.0042** | 5/5 |
| prob_aurc | -0.0006 +/- 0.0016 | 3/5 | **+0.0018 +/- 0.0008** | 0/5 |
| accuracy | +0.0000 +/- 0.0075 | 2/5 | -0.0020 +/- 0.0051 | 2/5 |

## Reading it

- **PrismFlow vs naive: no separable difference.** Reliability is lower in 4/5
  seeds, but the mean change (-0.0027) is less than half its std (0.0069).
  Resolution is also lower in 4/5 seeds, by a similar amount. Total Brier is
  unchanged (+0.0002). ECE is slightly *higher* on average, but inside noise.
  None of this is evidence of improved reliability.
- **The discount alone, applied to naive weights, hurts.** Brier rises in 5/5
  seeds, and nearly all of the rise is lost resolution (-0.0089, 5/5).
  Reliability does not move. AURC also worsens in 5/5 seeds. So in this
  setting, the discount's per-view reweighting blurs which samples are easy
  without making probabilities more honest. Training under the discount
  (prismflow) recovers most of that loss. This matches the
  compensation effect reported in the clone experiment.
- **Vacuity confidence falls under the discount**, as it should (0.779 to 0.750
  to 0.712). That drop does not show up as lower top-label calibration error.
  `vacuity_ece` is essentially unchanged for prismflow and worse for the
  test-time-only discount.
- **PrismFlow is already slightly underconfident** on top-label probability.
  Most reliability-diagram bins sit above the diagonal. Lowering confidence
  further cannot improve calibration here.
- `brier_within_bin` has a mean of at most 0.0008 in magnitude, with a std of
  at most 0.0017. That is small next to the reliability and resolution values,
  but not next to the smallest PrismFlow-vs-naive differences, which are null
  anyway.

## Limitations

- **One data condition, and the least favourable one for the claim.** The
  thesis concerns agreement between *redundant* views. At rho = 0.3 with 4
  distinct views, ENIV is about 3.3 of 4, so the discount is mild (alpha about
  0.83), and a null result is the expected outcome. The informative test is
  calibration under duplication or collusion, for example the clone
  experiment's k copies. That run has not been done.
- 300 test samples per seed spread over 15 bins. Bins below confidence 0.5 hold
  about 10 samples each, which makes MCE noisy (std 0.09-0.13). Do not read
  anything into MCE differences here.
- 5 seeds. The paired stds are large relative to the PrismFlow-vs-naive
  differences, so detecting an effect of that size would need more seeds.

## Reproduce

```
python -m experiments.calibration.run_calibration    # ~2 min on CPU, writes results/calibration/
```

Outputs:

- `results/calibration/{naive,prismflow,naive_weights_discounted}/metrics.json`
- `results/calibration/{naive,prismflow,naive_weights_discounted}/metrics.csv`
- `results/calibration/{naive,prismflow,naive_weights_discounted}/reliability_diagram.png`
- `results/calibration/summary.md`: the side-by-side and paired tables, read from
  the three reports
- `results/calibration/config_used.json`
