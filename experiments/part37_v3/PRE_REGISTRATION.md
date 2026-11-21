# Part 37 item B — PRE-REGISTRATION: the gate-aware attacker (L12)

**Written 2026-11-21, before anything was run.** No attack has been executed,
no epsilon has been probed and no cell has been measured for this item.
Everything below was fixed before any Part 37 number existed.

**Directory note.** The review brief for this scaffolding said
`experiments/adaptive_v2/`. This is `experiments/part37_v3/` instead, because
Part 37's standing rules permit creating only `prismflow/v3/**`,
`tests/unit/test_v3_*.py`, `experiments/part37_v3/**` and
`results/part37_v3/**`. Scaffolding at a path the Part may not write to would
be orphaned the moment Part 37 runs.

---

## 0. What this closes, and what it does not

**L12's written closing condition**, from `docs/KNOWN_LIMITATIONS.md`: *"A
working gate-aware arm — the straightforward fix is to lower eps in that arm so
the penalty is active — and a sweep over the number of compromised views."*

L12 currently states, in its own words, that **"No conclusion about gate-aware
attackers is licensed by this project."** This item is item **B** of Part 37
and addresses exactly that sentence.

**Item C is NOT in scope** and must not be folded in. Item C is an attacker
optimising against the **detector's suspicion score**, which needs new
objective code in `prismflow/v3/` (the detector's statistic is a *max* excess
over pairs against `expected_ij = null_ij + clip(R_ij,0,1)·(1−null_ij)`, not
the mean cosine the existing surrogate computes). It is a separate
pre-registration.

**Item A is already closed by committed work** and is not re-run:
`results/chorus/correction_shapes.json` holds `capped`, `floor_alpha`,
`alpha_ceiling`, `per_sample` and a four-point oracle sweep at eps 1.0 and
2.0, 5 seeds, read-only replay. `floor_alpha` is bit-identical to `current` on
all five fields and all 5 seeds because the measured alphas (0.5021, 0.4331)
already exceed the 0.25 bound, so "cap with a floor" collapses to `capped` by
derivation.

## 1. Why the existing arm is a null, from the committed numbers

`results/adaptive/adaptive.json`, Part 12, 5 seeds, k = 3, eps = 2.0,
beta = 1.0, 30 steps, BPDA through the dependence estimator:

| arm | success | `rho_final` | `gate_auc` |
|---|---|---|---|
| `gate_check` | 0.0773333340883255 ± 0.04386849096046346 | 0.0752225685864687 ± 0.07079799297281554 | 0.544142315657971 ± 0.027580741938847887 |
| `random_search` | 0.0773333340883255 ± 0.04386849096046346 | 0.0752225685864687 ± 0.07079799297281554 | 0.544142315657971 ± 0.027580741938847887 |

**Bit-identical on all three metrics.** The reason is the hinge. The adaptive
objective adds `− gamma · max(0, rho_hat(C) − tau)`, and `tau` is
**0.29635801911354065** (`results/adaptive/adaptive.json`, `per_seed[*].tau`).
At eps = 2.0 the arm's random perturbation drives measured dependence to
**0.0752**, far below tau, so `max(0, rho − tau)` is identically zero and the
gamma = 20 objective reduces **exactly** to gamma = 0.

Note what this does *not* say. The **gradient** adaptive arm at eps = 2.0
reaches `rho_final` **0.8120 ± 0.0120**, well above tau, and its gamma sweep
does move: gamma = 5 raises success to **0.6600 ± 0.1161** from **0.6267 ±
0.1352** while driving dependence to **0.7063 ± 0.0702** and gate AUC to
**0.5806 ± 0.0413** from **0.6743 ± 0.0492**; gamma = 20 pays for evasion with
success **0.4967 ± 0.1253** at dependence **0.3343 ± 0.0172** (L12). So a
trade-off curve exists for the gradient attacker at one operating point. What
is missing is the **gradient-free, gate-aware** arm at a budget where its own
perturbation does not destroy the signal the gate watches, and any variation in
`k` at all — L12 records that "Both the view count and the budget are fixed
across the entire sweep; neither was varied."

## 2. The epsilon grid, and why it is not chosen to activate the hinge

**Pre-registered grid: `[0.2, 0.5, 1.0, 2.0]`.**

This is **Part 09's committed epsilon grid, unchanged**
(`experiments/ait_attack/config.yaml`, `attack.epsilon_grid`, carried there
from Part 09). It is adopted **because** it predates this question: a grid
chosen now, after knowing that the hinge needs `rho > 0.2964`, would be
result-driven selection of exactly the kind this document exists to prevent.
Part 09's grid spans an order of magnitude below the eps = 2.0 that produced
the null, which is what L12's "lower eps" asks for, and it was fixed before
anyone looked at tau.

**Hinge activation is a MEASURED OUTCOME, reported per cell, not a
precondition for inclusion.** Every cell records `rho_initial`, `rho_final`
and that seed's `tau`, and is classified:

- **hinge active** — `rho` exceeds `tau` at some accepted iterate, so the
  gamma term contributed gradient;
- **hinge inactive** — `rho` never exceeds `tau`, so the arm reduces to its
  own gamma = 0 counterpart **by derivation**. Such a cell is reported as a
  declared identity, exactly as Part 29 reported beta at k = 1, and is
  **excluded from H1's test** because it carries no information about
  gate-aware attackers.

**If every cell is hinge-inactive, that is the result**, and it is reported as
one: "no epsilon in Part 09's committed grid keeps measured dependence above
tau for a gradient-free attacker on this data, so L12's suggested fix does not
work and the gate-aware question stays open." That outcome is not a failed
run; it is a measurement about the suggested fix. **No epsilon outside the
grid may be added after seeing results.**

## 3. The k grid: {2, 3} only

**k = 1 is excluded, by derivation, not by choice.**
`prismflow/attacks/adaptive.py`'s `compromised_agreement` returns
`torch.zeros(())` when `chosen.shape[1] < 2` — the surrogate is a mean over
ordered off-diagonal pairs and a one-element compromised set has none. So at
k = 1:

- `rho_hat(C)` surrogate is identically 0, hence below tau, hence the hinge is
  inactive and the gamma term is identically zero;
- the gate-aware arm is therefore **bit-identical to plain chorus at k = 1**,
  for every gamma;
- and beta is inert at k = 1 for the same structural reason, which Part 29
  already recorded ("beta is INERT at k = 1: the agreement term sums over
  ordered pairs within the compromised set and there are none").

Running k = 1 would record the same number three or four times and present a
tautology as a sweep. **The exclusion is stated here so that the k grid is not
read as cherry-picked**, and one k = 1 cell at gamma = 0 is kept as a
**declared identity check** — it must equal the plain chorus k = 1 result
bit-for-bit, and a mismatch stops the item as a defect.

The synthetic generator has `n_views` **4**
(`results/adaptive/adaptive.json`, `config.data.n_views`), so k = 3 is "3 of
4" — the cell L12's existing finding sits in — and k = 2 is the new one.
k = 4 is excluded: with every view compromised there is no honest view left
and the fused prediction is not a defence question.

## 4. The gamma grid: {0.0, 1.0, 5.0, 20.0}

A four-value subset of Part 12's committed grid
`[0.0, 1.0, 2.0, 5.0, 10.0, 20.0]` (`results/adaptive/adaptive.json`,
`gammas`), keeping:

- **0.0**, the mandatory reference — the objective reduces to plain chorus
  exactly, which is what makes any change across gamma attributable to the
  evasion term alone;
- **1.0** and **20.0**, the endpoints of the committed non-zero range;
- **5.0**, the point at which L12 reports the gradient arm's success peaking.

2.0 and 10.0 are dropped to keep the cell count at the pre-registered budget.
**The subset is fixed now and is not revisited after seeing results.**

## 5. Scope

| | |
|---|---|
| epsilon | 0.2, 0.5, 1.0, 2.0 |
| k | 2, 3 |
| gamma | 0.0, 1.0, 5.0, 20.0 |
| seeds | 0, 1, 2, 3, 4 |
| **cells** | 4 × 2 × 4 = **32** |
| **attack runs** | 32 × 5 = **160** |

Inherited unchanged from Part 12 (`results/adaptive/adaptive.json`,
`config`): `n_views` 4, `rho` 0.3, `n_samples` 2000, 40 epochs, batch 64,
lr 0.001, `anneal_epochs` 10, attack `steps` 30, `beta` 1.0,
`evaluation.batch_size` 512, and BPDA on (`use_bpda=True`).

**No retraining.** The attack runs against the models Part 12's recipe
produces for the same five seeds; `prismflow/attacks/adaptive.py` and
`prismflow/attacks/chorus.py` are **frozen and import-only**. No new attack
code is needed for this item — `AdaptiveConfig(gamma, tau, use_bpda)` already
implements the objective.

**Expected compute: 1.5–2.0 h.** This is an **extrapolation, not a measured
figure**: `results/adaptive/adaptive.json` records no wall-clock. The basis is
the data size (2,000 samples at batch 64 is ~32 batches, 30 steps) and Part
29's measured 6.0 s per PrismFlow attack at batch 1200 against 14.3 s at batch
64. **Time one seed of one cell before launching the rest** and replace this
estimate with the measurement.

## 6. Hypothesis

**H1.** A gate-aware attacker achieves **higher attack success** than plain
chorus at the **same k and the same epsilon**.

Paired comparison, within seed: for each (eps, k, gamma > 0) cell, the
difference is `success(gamma) − success(gamma = 0)` at that same eps and k,
seed by seed.

**The brief's rule and the multiplicity problem with it.** The brief states H1
as holding "on at least 1 of the measured cells". With **24** non-zero-gamma
cells, finding one cell whose mean difference is positive is close to certain
under a null, so "at least one cell" is not a test. It is recorded here as
stated, and a **stricter primary rule is pre-registered alongside it**:

- **PRIMARY (barred).** H1 is **CONFIRMED** iff at least one
  (eps, k, gamma > 0) cell is **hinge active** and satisfies *all three*:
  1. mean paired difference **> 0**;
  2. the sign is positive in **at least 4 of 5 seeds**;
  3. the mean difference exceeds **its own sample std** (ddof = 1).
- **SECONDARY (reported, no bar).** The count of cells with a positive mean
  difference, out of the number tested, so the multiplicity is visible rather
  than hidden. The brief's "at least 1 cell" reading is reported as this count
  being ≥ 1 and is explicitly **not** the verdict.
- **H1 is DISCONFIRMED** iff no hinge-active cell satisfies the primary rule.
- **H1 is NOT MEASURED** iff no cell is hinge active (see §2).

Condition 3 is the one that matters on this data: the committed per-seed stds
are large — success **0.6267 ± 0.1352** at gamma = 0 and **0.6600 ± 0.1161**
at gamma = 5 — so a mean difference of a few points is well inside one std and
must not be called an effect.

**Secondary observations, no bar attached, reported per cell:** `rho_initial`,
`rho_final`, that seed's `tau`, the hinge-active classification, `gate_auc`,
`steps_accepted` and `steps_rejected`, and the success of plain chorus at the
same (eps, k) for the paired reference.

**The trade-off curve is reported, not just its peak.** L12's existing finding
is that the attacker *pays* for evasion (success 0.4967 at gamma = 20 where
dependence is crushed to 0.3343). Reporting only the best gamma would hide
that. Every gamma in the grid is reported for every (eps, k).

## 7. Write-down rules

1. **5 seeds, mean ± sample std (ddof = 1), per cell.** A cell with fewer than
   5 completed seeds is **not a finding**.
2. **Paired within seed**, always. The gamma = 0 reference is the same seed's
   own plain-chorus run at the same eps and k, never a pooled mean.
3. **Per-seed sign counts before every aggregate mean**, with
   `exactly_zero` as its own category.
4. **No pooling across eps, across k, or across gamma.** 32 cells, 32 reports.
5. **Hinge-inactive cells are excluded from H1 and reported as declared
   identities**, with the derivation stated, never silently dropped.
6. **No result-driven selection.** The eps grid, the k grid, the gamma grid,
   the primary rule and the cell structure are fixed by this document. If the
   grid turns out to contain no hinge-active cell, that is the finding; no
   epsilon is added afterwards.
7. **A result that fails its rule is reported as a failure, in the same detail
   as a success** — Part 37's standing rule.
8. **Frozen modules are import-only.** `prismflow/attacks/` is not edited. Any
   new mechanism would live in `prismflow/v3/`, and this item needs none.
9. **No limitation's status changes here.** Proposals go to
   `docs/proposed_limitations_update.md` for the review gate.

## 8. Stopping rules

1. **k = 1 identity check.** The one k = 1, gamma = 0 cell must equal plain
   chorus at k = 1 bit-for-bit. A mismatch stops the item: it would mean the
   gate term is not inert where §3 derives that it must be.
2. **gamma = 0 identity check.** Every gamma = 0 cell must equal the plain
   chorus result at the same eps and k bit-for-bit, since the objective reduces
   exactly. A mismatch stops the item.
3. **tau provenance.** `tau` must be read per seed from the same calibration
   Part 12 used, not re-fitted. If it cannot be reproduced as
   **0.29635801911354065** for seed 0, stop and report.
4. **Budget.** Time one seed of one cell first. If the projected total exceeds
   **4 hours**, run the smallest pre-registered subset — all four epsilons at
   k = 3 and gamma ∈ {0, 5} — and mark the remainder **OPEN**.

## 9. What this item cannot show

- **It cannot show the discount is a defence.** L3's extension already records
  that the discount does not reduce attack success. A gate-aware attacker
  succeeding tells us more about the attacker; a gate-aware attacker failing
  does not make the gate a defence, because success at gamma = 0 is already
  high.
- **It cannot speak to the detector.** `gate_auc` is the gate, not
  `prismflow/statistics/suspicion.py`. The detector-optimising attacker is
  item C.
- **It cannot generalise off synthetic data.** 2,000 samples, 4 synthetic
  views at `rho` 0.3. L12's "adaptive attacks on real data" stays untested,
  and no result here transfers to AIT or LANL.
- **It cannot widen the operating point beyond its own grid.** After this item
  the curve is known at 4 epsilons × 2 k values, not at "any budget" — L12's
  list of untested capabilities shortens, it does not empty.
