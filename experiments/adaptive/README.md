# Part 14: adaptive adversary

**The discount is partially evadable.** An attacker that knows the ENIV
discount exists and optimises against it does strictly better than the Part 09
Chorus attack: higher attack success, lower measured dependence, and a lower
score from the detection gate, all at once, on all 5 seeds.

Threat model unchanged: controlled research simulation on our own models and
synthetic data. The attacker controls k = 3 of 4 views, sees the model
white-box, and perturbs only its own views within `|| delta ||_inf <= 2.0`.

## The objective

    maximise over {delta_v : v in C}
        sum_{v in C} log b_v[target]                  targeted
        - beta  * sum_{i,j in C} || b_i - b_j ||^2    agreement
        - gamma * max(0, rho_hat(C) - tau)            EVASION

`gamma = 0` reproduces the Chorus attack exactly -- pinned by
`test_gamma_zero_reproduces_chorus_exactly`, which asserts an identical
delta. Everything below is therefore attributable to the evasion term alone.

`tau` is set per seed to the CLEAN measured dependence among the views the
attacker controls (0.197 to 0.296 across seeds). The attacker is asking to
look no more dependent than an honest baseline, not to look independent in
some absolute sense.

## Sweep

5 seeds, mean +/- sample std.

| gamma | attack success | measured dependence | deployable gate AUC |
|---|---|---|---|
| 0 | 0.6267 +/- 0.1352 | 0.8120 +/- 0.0120 | 0.6743 +/- 0.0492 |
| 1 | 0.6540 +/- 0.1221 | 0.7456 +/- 0.0226 | 0.6135 +/- 0.0505 |
| 2 | 0.6567 +/- 0.1421 | 0.7483 +/- 0.0232 | 0.6109 +/- 0.0460 |
| **5** | **0.6600 +/- 0.1161** | **0.7063 +/- 0.0702** | **0.5806 +/- 0.0413** |
| 10 | 0.5680 +/- 0.1005 | 0.4223 +/- 0.0775 | 0.5186 +/- 0.0464 |
| 20 | 0.4967 +/- 0.1253 | 0.3343 +/- 0.0172 | 0.4874 +/- 0.0540 |

Reporting success alone would be unreadable, which is why measured dependence
is beside it. Flat success with flat rho would mean the attacker never
managed to evade; what actually happens is that rho moves a long way and
success moves with it.

### The evadable region: gamma = 2 to 5

Against gamma = 0, at gamma = 5:

| quantity | change | per-seed |
|---|---|---|
| attack success | **+0.0333 +/- 0.0237** | +0.037, +0.020, +0.017, +0.073, +0.020 -- **5/5 up** |
| measured dependence | **-0.1057** | 5/5 down |
| gate AUC | 0.6743 -> 0.5806 | 5/5 down |

The attacker gains on every axis simultaneously, on every seed. It attacks
more successfully AND looks more independent to the estimator AND scores
lower on the detector. gamma = 2 shows the same pattern (success +0.0300
+/- 0.0217, 5/5 up). gamma = 1 is 4/5 up with one tie.

Per the brief's instruction: **the success rate rises with gamma, so the
defence is partially evadable, and this is stated rather than buried.** The
rise is small in absolute terms (+0.033) but it is consistent across every
seed and it comes together with a large drop in the quantity the defence
measures.

### The overshoot region: gamma >= 10

Past gamma = 5 the attacker pays for evasion. At gamma = 20 measured
dependence is crushed to 0.3343 -- essentially down to tau -- but success
falls to 0.4967, below the Chorus baseline (-0.1300 +/- 0.0239, 5/5 down).

This is the tension the objective encodes: the beta term pushes the
attacker's views together and the gamma term pulls them apart, because the
estimator measures roughly what beta maximises. A large gamma wins the
evasion and loses the attack. The defence is not free to evade, but its price
is finite and the cheapest point on the curve is a net gain for the attacker.

## Controls

These decide whether the sweep can be believed at all.

| arm | attack success | measured dependence | gate AUC |
|---|---|---|---|
| white-box (gamma=0) | 0.6267 +/- 0.1352 | 0.8120 +/- 0.0120 | 0.6743 +/- 0.0492 |
| transfer | 0.5793 +/- 0.1519 | 0.8151 +/- 0.0298 | 0.6530 +/- 0.0461 |
| random search (gradient-free) | 0.0773 +/- 0.0439 | 0.0752 +/- 0.0708 | 0.5441 +/- 0.0276 |

**Transfer** crafts on the undefended model (identical weights, discount off)
and applies to the defended one. It reaches 0.5793 against the white-box
attack's 0.6267 -- slightly WORSE, which is the healthy ordering. Transfer
beating white-box would have meant the white-box gradient was masked.

**Random search** is gradient-free with 64 restarts in the same epsilon ball.
It reaches 0.0773 against the gradient attack's 0.6267. The gradient attack
is real, not an artifact of a broken gradient.

### BPDA was required, and it is load-bearing

`PrismFlow.forward` computes the dependence matrix, ENIV and alpha inside
`torch.no_grad()` (`prismflow/models/prismflow.py:147`). That makes
`d(rho_hat)/d(delta)` identically zero. An attacker differentiating naively
through the model would find the evasion term unreachable and would conclude
the defence resists evasion -- a conclusion produced entirely by OUR
implementation rather than by the defence.

The evasion term therefore uses a Backward Pass Differentiable Approximation:
forward value is the model's true measurement, backward pass uses a
differentiable surrogate (mean pairwise cosine agreement among the
compromised views). `test_measured_dependence_is_detached` and
`test_bpda_restores_the_gradient` pin both halves. Without BPDA the entire
sweep would have been flat and would have read as a robustness result.

### One arm did not test what it was built to test

`gate_check` was intended as a gradient-free probe aimed at the deployable
gate. It returned numbers **identical to `random_search` on all 5 seeds and
all three metrics**, because at epsilon = 2.0 random perturbations collapse
measured dependence to 0.0752 -- far below every seed's tau -- so
`max(0, rho - tau)` is zero and the gamma = 20 objective reduces exactly to
the gamma = 0 one. The arm is reported as uninformative rather than presented
as a separate result.

The gate thread is closed by the sweep's gate column instead, which is a
stronger close: an adaptive attacker drives the deployable gate from 0.6743
to 0.4874 -- below chance -- **without targeting the gate at all**. There is
nothing left to adapt against.

## Limits of the defence

1. **It is partially evadable.** At gamma = 5 an attacker aware of the
   discount improves attack success on 5/5 seeds while reducing the measured
   dependence the discount acts on by 0.106 and the detector's AUC by 0.094.

2. **What it measures is what it can be fooled on.** The estimator reads
   agreement among views; the evasion term is a direct penalty on that same
   reading. There is no separation between the signal and the thing an
   attacker can manipulate.

3. **Its price on the attacker is finite and small.** Full evasion
   (gamma = 20, rho down to tau) costs 0.130 of attack success. An attacker
   who wants to evade completely can, and still lands at 0.4967 success.

4. **The detection half degrades faster than the attack half.** Across the
   sweep the gate falls 0.6743 -> 0.4874 while success falls only
   0.6267 -> 0.4967. Evasion hurts detection more than it hurts the attack.

5. **Scope.** Synthetic data, 4 views, k = 3, one epsilon, one attack family.
   `experiments/synthesis/` already showed the detection side does not
   generalise across attacks; nothing here contradicts that, and this file
   does not extend to real data (Part 15).

## Files

| path | what |
|---|---|
| `prismflow/attacks/adaptive.py` | objective, BPDA, gradient-free control |
| `tests/unit/test_adaptive.py` | 12 tests, including the two BPDA pins |
| `run_adaptive.py` | sweep + three control arms |

Results in `results/adaptive/`. Related: `experiments/synthesis/README.md`
(the gate thread, closed as a negative result).
