# PrismFlow

Dependence-Discounted Evidential Fusion under Collusive View Compromise.

A multi-view learning research project testing the hypothesis that agreement
between views should contribute evidence in proportion to the effective
independence of those views, not in proportion to how many views agree.

See `docs/CONTRACT.md` for the full project contract (data flow, component
classification, tensor shape conventions, and reproducibility rules).

## Status

This is the Part 01 skeleton: contract, package layout, and shared
utilities (config, seeding, logging, device, checkpointing) only. No
modeling code has been implemented yet.

## Setup

```bash
pip install -r requirements.txt
pytest
```

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
