# baseline

Naive Dempster fusion baseline (no dependence discounting).

- views: 4, classes: 3, rho: 0.3
- samples: 2000, epochs: 40
- seeds: [0, 1, 2, 3, 4]

| split | metric | mean | std |
|-------|--------|------|-----|
| val | accuracy | 0.8087 | 0.0932 |
| val | mean_confidence | 0.7748 | 0.0956 |
| val | mean_uncertainty | 0.2252 | 0.0956 |
| test | accuracy | 0.8300 | 0.0791 |
| test | mean_confidence | 0.7790 | 0.0829 |
| test | mean_uncertainty | 0.2210 | 0.0829 |
