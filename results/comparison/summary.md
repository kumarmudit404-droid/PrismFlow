# Suspicion detector (Part 10)

Seeds: [0, 1, 2, 3, 4]. Mean +/- sample std across seeds. Detection thresholds are calibrated per seed on that seed's own CLEAN scores, so the `clean` row reads 0.05 and 0.01 by construction and every other row is a true-positive rate at that false-positive budget. AUC is against the same seed's clean scores; 0.5 is chance.

The detector never sees which condition it is scoring. This table does, because scoring a detector requires it.

## Detection

| condition | ROC AUC | detection @ 5% FPR | detection @ 1% FPR | mean suspicion score |
|---|---|---|---|---|
| clean | 0.5000 +/- 0.0000 | 0.0500 +/- 0.0000 | 0.0040 +/- 0.0055 | +0.3590 +/- 0.0394 |
| clone_k2 | 0.8743 +/- 0.0583 | 0.6453 +/- 0.1822 | 0.6453 +/- 0.1822 | +0.5243 +/- 0.0203 |
| missing_30 | 0.2568 +/- 0.0618 | 0.0461 +/- 0.0778 | 0.0461 +/- 0.0778 | +0.1566 +/- 0.0153 |
| missing_50 | 0.1247 +/- 0.0319 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | +0.0244 +/- 0.0197 |
| noisy_1 | 0.7486 +/- 0.0544 | 0.3760 +/- 0.2787 | 0.3760 +/- 0.2787 | +0.4167 +/- 0.0332 |
| chorus_k1 | 0.7215 +/- 0.0512 | 0.3700 +/- 0.2459 | 0.3700 +/- 0.2459 | +0.4223 +/- 0.0381 |
| chorus_k2 | 0.6328 +/- 0.0920 | 0.4280 +/- 0.1970 | 0.4260 +/- 0.2000 | +0.3845 +/- 0.0177 |
| chorus_k3 | 0.4870 +/- 0.1096 | 0.3613 +/- 0.1268 | 0.3613 +/- 0.1268 | +0.3672 +/- 0.0240 |
| pgd_k2 | 0.8396 +/- 0.0970 | 0.5807 +/- 0.2897 | 0.5807 +/- 0.2897 | +0.4719 +/- 0.0339 |

## Task metrics under each condition

| condition | accuracy | ECE | Brier reliability | mean confidence | ENIV |
|---|---|---|---|---|---|
| clean | 0.8300 +/- 0.0797 | 0.0637 +/- 0.0137 | 0.0282 +/- 0.0081 | 0.7499 +/- 0.0816 | 3.3281 +/- 0.0750 |
| clone_k2 | 0.7807 +/- 0.0633 | 0.0801 +/- 0.0171 | 0.0339 +/- 0.0066 | 0.7136 +/- 0.0764 | 3.8059 +/- 0.1124 |
| missing_30 | 0.8027 +/- 0.0657 | 0.1062 +/- 0.0227 | 0.0400 +/- 0.0091 | 0.5993 +/- 0.0682 | 3.1304 +/- 0.0961 |
| missing_50 | 0.7360 +/- 0.0573 | 0.1104 +/- 0.0285 | 0.0388 +/- 0.0127 | 0.4780 +/- 0.0614 | 2.9655 +/- 0.0919 |
| noisy_1 | 0.7800 +/- 0.0844 | 0.0663 +/- 0.0227 | 0.0339 +/- 0.0094 | 0.7652 +/- 0.0818 | 3.3438 +/- 0.0987 |
| chorus_k1 | 0.6807 +/- 0.1024 | 0.1140 +/- 0.0469 | 0.0606 +/- 0.0286 | 0.8114 +/- 0.0516 | 3.2127 +/- 0.0872 |
| chorus_k2 | 0.5073 +/- 0.1018 | 0.2624 +/- 0.0700 | 0.1587 +/- 0.0653 | 0.8051 +/- 0.0433 | 2.9048 +/- 0.0608 |
| chorus_k3 | 0.4213 +/- 0.0925 | 0.3565 +/- 0.0835 | 0.2719 +/- 0.1035 | 0.7676 +/- 0.0216 | 2.8887 +/- 0.0606 |
| pgd_k2 | 0.5880 +/- 0.1023 | 0.2081 +/- 0.0774 | 0.1146 +/- 0.0571 | 0.8625 +/- 0.0309 | 3.3435 +/- 0.1186 |

## Where PrismFlow loses

- **missing_30**: detector INVERTED, AUC 0.2568 +/- 0.0618. Scores are systematically LOWER than clean, so the flag points the wrong way: at any threshold this condition is flagged less often than clean input is.
- **missing_50**: detector INVERTED, AUC 0.1247 +/- 0.0319. Scores are systematically LOWER than clean, so the flag points the wrong way: at any threshold this condition is flagged less often than clean input is.
- **chorus_k3**: detector at chance, AUC 0.4870 +/- 0.1096. Agreement under this condition is not distinguishable from clean.
- **clone_k2**: accuracy 0.7807 against 0.8300 clean (-0.0493).
- **clone_k2**: ECE 0.0801 against 0.0637 clean (+0.0164).
- **missing_30**: accuracy 0.8027 against 0.8300 clean (-0.0273).
- **missing_30**: ECE 0.1062 against 0.0637 clean (+0.0425).
- **missing_50**: accuracy 0.7360 against 0.8300 clean (-0.0940).
- **missing_50**: ECE 0.1104 against 0.0637 clean (+0.0467).
- **noisy_1**: accuracy 0.7800 against 0.8300 clean (-0.0500).
- **noisy_1**: ECE 0.0663 against 0.0637 clean (+0.0026).
- **chorus_k1**: accuracy 0.6807 against 0.8300 clean (-0.1493).
- **chorus_k1**: ECE 0.1140 against 0.0637 clean (+0.0503).
- **chorus_k2**: accuracy 0.5073 against 0.8300 clean (-0.3227).
- **chorus_k2**: ECE 0.2624 against 0.0637 clean (+0.1987).
- **chorus_k3**: accuracy 0.4213 against 0.8300 clean (-0.4087).
- **chorus_k3**: ECE 0.3565 against 0.0637 clean (+0.2928).
- **pgd_k2**: accuracy 0.5880 against 0.8300 clean (-0.2420).
- **pgd_k2**: ECE 0.2081 against 0.0637 clean (+0.1444).
