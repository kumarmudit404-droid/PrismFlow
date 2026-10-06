# PrismFlow

Dependence-Discounted Evidential Fusion under Collusive View Compromise.

A multi-view learning research project testing the hypothesis that agreement
between views should contribute evidence in proportion to the effective
independence of those views, not in proportion to how many views agree.

See `docs/CONTRACT.md` for the full project contract (data flow, component
classification, tensor shape conventions, and reproducibility rules).

## How much redundancy does a standard multi-view benchmark contain?

Multi-view methods are compared on benchmarks by their view count. Nobody
reports how many of those views are actually independent. On Handwritten (UCI
Multiple Features), the canonical 6-view benchmark, they are not six.

| dataset | nominal views | measured ENIV | efficiency ratio | lambda_U |
|---|---|---|---|---|
| handwritten | 6 | 3.28 +/- 0.07 | 0.547 +/- 0.011 | 0.127 +/- 0.042 |

5 seeds, mean +/- sample std, 800 test samples, test accuracy 0.981 +/- 0.003.
Measured before any duplication, corruption or attack is applied.
ENIV is the effective number of independent views; lambda_U is upper tail
dependence at q = 0.90.

**Six views behave like three and a quarter.** That is not the estimator's
noise floor: permuting each view's features within class, which makes the
views independent by construction while leaving the sample size, the feature
dimension and the marginals untouched, moves ENIV to **5.51 +/- 0.08** of 6.
So 2.23 of the 2.72-view shortfall is measured redundancy and roughly half a
view is floor.

The redundancy is not spread evenly. Dropping `fou` or `mor` costs 0.74 and
0.68 effective views; dropping `fac` or `pix` -- the two largest views, at 216
and 240 features -- costs 0.24 and 0.21, because they are 0.85 dependent on
each other. Keeping `pix` alongside `fac` buys about a fifth of an independent
witness. View count and view width both overstate what those two contribute.

This reframes gains reported on such benchmarks: a method claiming to exploit
six views may be exploiting three. `results/real/README.md` has the full
tables, `docs/DATASETS.md` the datasets and their limits, and
`experiments/real/run_real.py` reproduces it.

**What this is not.** Real views carry no ground-truth dependence, so ENIV is
APPLIED here and not VALIDATED here. The only validation of the estimator is
Part 05, on the synthetic generator, where `rho` is a parameter that was set.
See `docs/DATASETS.md`.

## Status

Parts 01-15 are implemented; `parts/` holds each Part's brief and
`results/<experiment>/` its findings. Two results qualify the project's
hypothesis and are stated in `docs/CONTRACT.md`: the dependence signal
DETECTS collusion reliably, and the discount driven from it does not defend
against it (Part 09). `docs/KNOWN_LIMITATIONS.md` carries the rest.

## Setup

```bash
pip install -r requirements.txt
pytest
```

### Checking for secrets before a commit

`tools/check_secrets.py` looks for two things: strings shaped like a provider
key, and the **actual values** in your `.env`, which is what catches a key with
no recognisable prefix. It never prints a value — a hit names the variable and
where it turned up.

```bash
python tools/check_secrets.py          # staged changes
python tools/check_secrets.py --all    # every tracked file
```

To run it on every commit, install it as a local hook. This is deliberately
not installed for you, and it is not committed — a hook that appears in a
working copy without the owner choosing it is a surprise, and `.git/hooks/` is
not tracked anyway:

```bash
printf '#!/bin/sh\nexec python tools/check_secrets.py\n' > .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit
```

A non-zero exit refuses the commit. If a finding is a false positive, add the
exact `(path, text)` pair to `ACKNOWLEDGED` in the script rather than loosening
a pattern, so the same string elsewhere is still reported.

## Layout

```
prismflow/
  data/        multi-view dataset generation and loading
  models/      encoders and evidence heads
  statistics/  dependence estimation, calibration, suspicion detection
  eniv/        Effective Number of Independent Views estimator + discount
  attacks/     colluding / compromised view generators
  evaluation/  multi-seed experiment harness
  utils/       config, seed, logging, device, checkpoint
  app/         demo / interface code
configs/       YAML experiment configs
docs/          project contract
tests/         unit and integration tests
```
