# Per-sample ENIV (Part 12)

**Path note.** As in Part 11, the allowed-paths list for this Part does not
include a README. The auto-generated `results/per_sample/summary.md` carries the
tables but no verdict, and a table without an interpretation is the one thing
this project's contract does not accept as a finding. This file supplies the
verdict; it adds no new numbers, and every figure below is read from
`results/per_sample/per_sample.json`.

## The question

On `mixed`, dependence is INPUT-CONDITIONAL: 20% of samples have their views
drawn from a single shared latent (rho 0.95) and the other 80% have independent
views (rho 0). A global dependence matrix must describe that dataset with one
number, and that number is wrong about every sample in it — too high for the
independent majority, far too low for the collapsed minority. Per-sample
`n_eff(x)` *can* represent the difference. Whether it *does* is the Part.

The claim is therefore the SEPARATION AUC: how well the per-sample estimate
tells collapsed samples from independent ones. Everything else is whether it
costs anything.

## Verdict: the negative half is clean, the positive half is weak

| system | separation AUC (mixed) | per-seed |
|---|---|---|
| global | 0.5242 +/- 0.0469 | .514 .535 .556 .568 .449 |
| per_sample | **0.6039 +/- 0.0720** | .635 .613 **.496** .584 .692 |

**The negative half holds cleanly.** The global matrix sits at chance, which is
exactly what an input-conditional generator should do to a single number. That
part of the Part's premise is confirmed rather than assumed: a global estimate
genuinely cannot represent this structure.

**The positive half is weaker than the framing invites.** 0.6039 is above
chance, and by a margin the seed spread does not erase in the mean. But one seed
of five lands at **0.4964 — below chance**. A detector that is at or under
coin-flip on 20% of training runs is not a usable detector, and the honest
summary is "per-sample ENIV carries a weak, unreliable signal about which
samples are collapsed", not "per-sample ENIV works".

This is reported as measured. It is not rounded up.

## It is the cheaper system, not the dearer one

| condition | global (s) | per_sample (s) |
|---|---|---|
| homogeneous | 0.1549 +/- 0.0166 | **0.0444 +/- 0.0227** |
| mixed | 0.2389 +/- 0.0157 | **0.0295 +/- 0.0163** |

An amortised estimator replaces a per-batch matrix computation with a forward
pass, so this is expected, but it is worth stating: the weak separation is not
bought with compute. It is ~8x faster on mixed.

## It buys nothing else, and is marginally worse at one thing

| condition | metric | global | per_sample |
|---|---|---|---|
| homogeneous | accuracy | 0.8333 +/- 0.0748 | 0.8320 +/- 0.0768 |
| homogeneous | ECE | 0.0679 +/- 0.0121 | 0.0661 +/- 0.0140 |
| mixed | accuracy | 0.7080 +/- 0.0655 | 0.7073 +/- 0.0682 |
| mixed | ECE | 0.0741 +/- 0.0157 | 0.0711 +/- 0.0186 |
| mixed | chorus attack success | **0.6673 +/- 0.0438** | 0.6820 +/- 0.0421 |

Accuracy and calibration are indistinguishable — every interval overlaps.
Chorus-attack success is slightly HIGHER under per-sample, i.e. per-sample ENIV
is very slightly *worse* at resisting the attack, not better. Consistent with
Part 09: the discount does not defend, and making it per-sample does not change
that. The separation AUC is the only thing that moved.

## ENIV stability: better on mixed, worse on homogeneous

Across-seed std of the per-run ENIV mean:

| condition | global | per_sample |
|---|---|---|
| homogeneous | **0.0769** | 0.0972 |
| mixed | 0.1281 | **0.0512** |

No clean story. Per-sample is steadier exactly where dependence is
heterogeneous, which is the case it was built for, and less steady on the
homogeneous data where a single global number is already the right model. That
is the direction one would want, but with 5 seeds it is one comparison each way
and should not be leaned on.

## The gaming audit: the failure mode it was built to catch is ABSENT

`train_two_timescale` trains an amortised estimator against the encoder. The
standing risk is that the encoder learns to fool the estimator rather than to
produce independent representations. The audit logs, every epoch, the
dependence the trained estimator reports against the dependence the untrainable
V1 estimator measures on held-out data. A widening gap is the failure.

**The gap does not widen. Its magnitude shrinks, from ~0.3 to ~0.05 over 40
epochs, in all 10 runs.** That is the amortised estimator converging toward the
honest measurement. On this evidence the per-sample result is not an artefact of
estimator capture, which is the precondition for believing it at all.

**What the audit does show, and it is not nothing.** The gap drifts from
negative to positive in 10 of 10 runs, and 8 of 10 final gaps are positive at
roughly +0.04. At convergence the trained estimator reports slightly LESS
dependence than the honest audit measures. That is the direction that inflates
ENIV, and per-sample ENIV mean is indeed above global (3.6527 vs 3.5082 on
mixed; 3.4649 vs 3.3310 on homogeneous).

Small, but it points the same way Part 11 did, by a completely different
mechanism: both V2 systems end up under-reporting dependence and reporting more
effective views than the V1 estimator does. See `docs/KNOWN_LIMITATIONS.md`.

## Design

- 4 views, 40 epochs, 300 test samples per seed, 5 seeds, mean +/- sample std.
- `homogeneous`: rho 0.3 throughout. `mixed`: 20% of samples collapsed at rho
  0.95, 80% at rho 0.0.
- Two-timescale training: 5 encoder steps per 1 estimator step.
- Chorus attack: k=2, epsilon=1.0, beta=1.0, 30 steps.
- Separation AUC is computed on negated `n_eff(x)`, since collapsed samples
  should have FEWER effective views; it is n/a on `homogeneous`, which has no
  two groups to separate.

## Limitations

- One synthetic generator, and the collapsed/independent split is a designed
  mixture rather than anything naturally occurring.
- 0.60 AUC with a sub-chance seed is not a deployable detector and no claim is
  made that it is.
- The attack comparison is a single attack at one strength.
- The terminal positive audit gap is small and its effect on ENIV is not
  separately isolated from the architecture change.

## Reproduce

```
python -m experiments.per_sample.run_per_sample
```

Writes `results/per_sample/{per_sample.json,summary.md}`, the per-condition
evaluation outputs, and the audit logs plus `audit_gap.png`.
