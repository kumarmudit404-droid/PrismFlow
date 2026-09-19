# Part 14: adaptive adversary

Seeds: [0, 1, 2, 3, 4]. Attack: k=3, eps=2.0, beta=1.0, 30 steps.
tau per seed = clean measured dependence among the compromised views:
the attacker asks to look no more dependent than an honest baseline.

## Sweep: does knowing the discount help?

| gamma | attack success | measured dependence (final) | deployable gate AUC |
|---|---|---|---|
| 0 | 0.6267 +/- 0.1352 | 0.8120 +/- 0.0120 | 0.6743 +/- 0.0492 |
| 1 | 0.6540 +/- 0.1221 | 0.7456 +/- 0.0226 | 0.6135 +/- 0.0505 |
| 2 | 0.6567 +/- 0.1421 | 0.7483 +/- 0.0232 | 0.6109 +/- 0.0460 |
| 5 | 0.6600 +/- 0.1161 | 0.7063 +/- 0.0702 | 0.5806 +/- 0.0413 |
| 10 | 0.5680 +/- 0.1005 | 0.4223 +/- 0.0775 | 0.5186 +/- 0.0464 |
| 20 | 0.4967 +/- 0.1253 | 0.3343 +/- 0.0172 | 0.4874 +/- 0.0540 |

gamma = 0 IS the Part 09 Chorus attack (pinned in tests/unit/test_adaptive.py).

## Controls

| arm | attack success | measured dependence | gate AUC |
|---|---|---|---|
| transfer | 0.5793 +/- 0.1519 | 0.8151 +/- 0.0298 | 0.6530 +/- 0.0461 |
| random_search | 0.0773 +/- 0.0439 | 0.0752 +/- 0.0708 | 0.5441 +/- 0.0276 |
| gate_check | 0.0773 +/- 0.0439 | 0.0752 +/- 0.0708 | 0.5441 +/- 0.0276 |

