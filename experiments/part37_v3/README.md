# Part 37 item B — the gate-aware attacker (L12)

**Status: scaffolding only. NOTHING HAS BEEN RUN.** No attack executed, no
epsilon probed, no cell measured. This directory holds a pre-registration and
this file. Every number below is a **reference from an earlier committed
Part**, cited to its path.

**Directory name.** The review brief said `experiments/adaptive_v2/`. This is
`experiments/part37_v3/` because Part 37's standing rules permit creating only
`prismflow/v3/**`, `tests/unit/test_v3_*.py`, `experiments/part37_v3/**` and
`results/part37_v3/**`. Scaffolding at a path the Part may not write to would
be orphaned the moment Part 37 runs.

---

## 1. Purpose

Remove one sentence from `docs/KNOWN_LIMITATIONS.md` L12: **"No conclusion
about gate-aware attackers is licensed by this project."**

L12 lists what is measured as sufficient to evade the defence (full white-box
access with a differentiable surrogate; control of exactly 3 of 4 views at
eps = 2.0), what is measured as insufficient (gradient-free search, 8× worse;
transfer from a separate model), and what is **untested and must not be
inferred**. The first untested item is a gate-aware attacker, and the reason is
not that nobody tried — the `gate_check` arm exists and is a **null**.

`results/adaptive/adaptive.json`, 5 seeds:

| arm | success | `rho_final` | `gate_auc` |
|---|---|---|---|
| `gate_check` | 0.0773333340883255 ± 0.04386849096046346 | 0.0752225685864687 ± 0.07079799297281554 | 0.544142315657971 ± 0.027580741938847887 |
| `random_search` | 0.0773333340883255 ± 0.04386849096046346 | 0.0752225685864687 ± 0.07079799297281554 | 0.544142315657971 ± 0.027580741938847887 |

Bit-identical. `tau` is **0.29635801911354065** (`per_seed[*].tau`), the arm's
random perturbation at eps = 2.0 drives measured dependence to **0.0752**, and
`max(0, rho − tau)` is therefore identically zero: the gamma = 20 objective
reduces exactly to gamma = 0. L12's suggested fix is to **lower eps so the
penalty is active**, and to **sweep k**, which Part 12 never varied ("Both the
view count and the budget are fixed across the entire sweep; neither was
varied").

## 2. Scope

| | |
|---|---|
| epsilon | **0.2, 0.5, 1.0, 2.0** — Part 09's committed grid, unchanged |
| k | **2, 3** — k = 1 excluded by derivation, see §4 |
| gamma | **0.0, 1.0, 5.0, 20.0** — a subset of Part 12's committed six |
| seeds | 0, 1, 2, 3, 4 |
| cells | 4 × 2 × 4 = **32** |
| attack runs | **160** |

**No retraining and no new attack code.** The attack runs against the models
Part 12's recipe produces for the same five seeds, and
`prismflow/attacks/adaptive.py` already implements the objective —
`AdaptiveConfig(gamma, tau, use_bpda)` and
`gamma · max(0, rho_hat(C) − tau)` with BPDA through the dependence estimator.
`prismflow/attacks/` is frozen and import-only. What this item needs is a
**driver**, a config and a test; `experiments/adaptive/run_adaptive.py` (313
lines) is the template.

**Expected compute: 1.5–2.0 h. This is an extrapolation, not a measurement.**
`results/adaptive/adaptive.json` records no wall-clock. The basis is the data
size (2,000 samples at batch 64 ≈ 32 batches, 30 steps) and Part 29's measured
6.0 s per PrismFlow attack at batch 1200 against 14.3 s at batch 64, where the
dependence estimator's permutation null runs once per batch. **Time one seed of
one cell before launching the rest** and replace this figure with the measured
one.

## 3. The epsilon grid is Part 09's, and that is the point

Part 09's committed grid `[0.2, 0.5, 1.0, 2.0]`
(`experiments/ait_attack/config.yaml`, `attack.epsilon_grid`) is adopted
unchanged **because it predates this question**. A grid picked now, knowing
the hinge needs `rho > 0.2964`, would be result-driven selection — choosing
the budget that makes the mechanism look alive.

So **hinge activation is a measured outcome, reported per cell**. Every cell
records `rho_initial`, `rho_final` and that seed's `tau`, and is classified
**hinge active** (the gamma term contributed gradient) or **hinge inactive**
(the arm reduces to its own gamma = 0 counterpart by derivation, reported as a
declared identity and excluded from the hypothesis test).

**If every cell comes back hinge-inactive, that is the finding, not a failed
run:** no epsilon in Part 09's grid keeps dependence above tau for a
gradient-free attacker on this data, so L12's suggested fix does not work and
the gate-aware question stays open for a different reason than before. **No
epsilon may be added after seeing results.**

## 4. k = 1 is excluded by derivation

`compromised_agreement` in `prismflow/attacks/adaptive.py` returns
`torch.zeros(())` when the compromised set has fewer than two members — it is a
mean over ordered off-diagonal pairs, and a one-element set has none. So at
k = 1 the surrogate dependence is identically 0, hence below tau, hence the
hinge is inactive and the gamma term is identically zero: **the gate-aware arm
is bit-identical to plain chorus at k = 1, for every gamma.** Beta is inert at
k = 1 for the same structural reason, which Part 29 already recorded.

Running k = 1 across four gammas would record one number four times and
present a tautology as a sweep. One k = 1, gamma = 0 cell is kept as a
**declared identity check** — it must match plain chorus bit-for-bit, and a
mismatch stops the item.

The generator has `n_views` **4** (`config.data.n_views`), so k = 3 is the
"3 of 4" cell L12's existing finding sits in and k = 2 is the new one. k = 4 is
excluded: with every view compromised there is no honest view left, so it is
not a defence question.

## 5. The hypothesis, and the multiplicity problem in the brief's version

**H1.** A gate-aware attacker achieves higher attack success than plain chorus
at the same k and the same epsilon, paired within seed.

The brief states H1 as holding "on at least 1 of the measured cells". With
**24** non-zero-gamma cells, at least one positive mean difference is close to
certain under a null, so that is not a test. The pre-registration records the
brief's wording and bars a **stricter primary rule**: a confirming cell must be
hinge active **and** have a positive mean difference **and** a positive sign in
**at least 4 of 5 seeds** **and** a mean difference exceeding its own sample
std.

That last condition is the binding one here. The committed per-seed spreads are
wide — success **0.6267 ± 0.1352** at gamma = 0 against **0.6600 ± 0.1161** at
gamma = 5 (L12) — so a mean difference of a few points sits inside one std and
must not be called an effect. The count of cells with a positive mean is
reported as a secondary figure so the multiplicity is visible rather than
hidden.

**The whole trade-off curve is reported, not its peak.** L12's existing finding
is that the attacker *pays* for evasion: at gamma = 20 success falls to
**0.4967 ± 0.1253** while dependence is crushed to **0.3343 ± 0.0172**.
Reporting only the best gamma would delete that, so every gamma is reported for
every (eps, k).

## 6. Reproduction

Not yet runnable — no driver exists in this directory. When it does:

```sh
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/part37_v3/run_gate_aware.py
```

**Inputs:** Part 12's recipe and tau, from `results/adaptive/adaptive.json`
(`config`, `per_seed[*].tau`); the frozen
`prismflow/attacks/{chorus,adaptive}.py`; and Part 09's epsilon grid from
`experiments/ait_attack/config.yaml`.

**Expected outputs:**

| path | content |
|---|---|
| `results/part37_v3/units/eps<E>_k<K>_gamma<G>_seed<N>.json` | per run: success, `rho_initial`, `rho_final`, `tau`, hinge classification, `gate_auc`, `steps_accepted`, `steps_rejected` |
| `results/part37_v3/item_b.json` | per cell: mean ± sample std (ddof=1), per-seed signs, paired difference against gamma = 0, hinge classification, H1 verdict |

**Four stopping rules**, all in the pre-registration: the k = 1 identity check;
the gamma = 0 identity check (every gamma = 0 cell must equal plain chorus at
the same eps and k bit-for-bit, since the objective reduces exactly); tau
provenance (read per seed from Part 12's calibration, not re-fitted — seed 0
must reproduce **0.29635801911354065**); and the budget rule (if the projected
total exceeds 4 hours, run all four epsilons at k = 3 and gamma ∈ {0, 5} and
mark the rest OPEN).

## 7. What this item does NOT do

**Item C — the detector-optimising attacker (L5) — is a separate Part and must
not be folded in.** It needs new objective code in `prismflow/v3/`, because the
detector's statistic is a **max** excess over pairs against
`expected_ij = null_ij + clip(R_ij, 0, 1) · (1 − null_ij)`
(`prismflow/statistics/suspicion.py`), not the mean cosine the existing
surrogate computes. Earlier pricing put B + C together at ~970 new lines and
3–4 h of compute, which does not fit one block.

**Item A is already closed by committed work** and is not re-run.
`results/chorus/correction_shapes.json` holds `capped`, `floor_alpha`,
`alpha_ceiling`, `per_sample` and a four-point oracle sweep at eps 1.0 and 2.0,
5 seeds, read-only replay, both gates passing. `floor_alpha` is bit-identical
to `current` on all five fields and all 5 seeds, because the measured alphas
(0.5021, 0.4331) already exceed the 0.25 bound — so "cap with a floor"
collapses to `capped` by derivation and there is no new cell to run.

## 8. What this item cannot show, whatever it measures

- **It cannot show the discount is a defence.** L3's extension already records
  that the discount does not reduce attack success. A gate-aware attacker
  succeeding tells us about the attacker; one failing does not make the gate a
  defence, because success at gamma = 0 is already high.
- **It cannot speak to the detector.** `gate_auc` is the gate, not
  `prismflow/statistics/suspicion.py`. That is item C.
- **It cannot generalise off synthetic data.** 2,000 samples, 4 synthetic views
  at `rho` 0.3. L12's "adaptive attacks on real data" stays untested and
  nothing here transfers to AIT or LANL.
- **It cannot empty L12's untested list.** After this item the curve is known
  at 4 epsilons × 2 k values. "Any budget other than eps = 2.0" and "any view
  count other than 4" shorten; they do not disappear.

**No limitation's status changes here.** Proposals go to
`docs/proposed_limitations_update.md` for the review gate.
