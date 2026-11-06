# Part 38 sweep execution log — 2026-11-06

A dated operational record, written before mode B was launched. It is **not**
part of the pre-registration and amends nothing: `PRE_REGISTRATION.md` stays
dated and frozen, and `docs/KNOWN_LIMITATIONS.md` is not edited.

---

## 1. Sampler audit: passed

Days 1–7, days 1–16 and synthetic data are all clean. No defect detected.

**Selection pass against Part 30's committed `population_cells_per_day`**
(`results/lanl/results.json`, `population.per_view.<view>`), summed to the
same read range. Nine of nine exact:

| range | proc | flows | dns |
|---|---|---|---|
| days 1–6 | 7,975,224 | 5,361,691 | 192,635 |
| days 1–7 | 9,637,185 | 6,502,720 | 229,895 |
| days 1–16 | 23,627,216 | 13,484,366 | 580,753 |

Records scanned also match Part 30 exactly at 16 days (proc 115,159,740,
flows 121,284,699, dns 1,055,902), and the days 1–16 build reproduced the one
committed earlier field for field — `per_day`, `selection`, `n_samples`,
`attack_hosts` and `attack_cells_in_read_range` identical across two
independent 11 GB scans, peak RSS 2,366.4 MB against 2,367.7 MB. Attack counts
match Part 30 to the row: train 439 over 150 windows, test 410 over 141.

**Synthetic audit**, `tests/unit/test_lanl_v2_sampler_synthetic.py`, 8 tests
passing. Four synthetic `*.txt.gz` in the loader's wire format over 10 hosts,
50 attack cells inside the read range and 7 outside, under a throwaway
`raw_root`. `select_cells` is compared against a brute-force re-derivation of
the pre-registered rule **cell by cell** — the check Part 30's committed sums
cannot give, because a compensating pair of errors would survive a sum.
`attack_hosts` matches the input attack set, the two out-of-range attack hosts
are not promoted, `cells_in_read_range` returns exactly the 50, the host rule
fires only on benign cells of in-range attack hosts, and L24's condition holds
even for an auth-only *benign cell of an attack host* — rule (a) beats rule
(b).

The brief asked for a 10-view toy. The loader's view set is fixed at four by
`FEATURES`, `SOURCE_COLUMN` and `RECORD_WIDTH` in the frozen
`prismflow/data/lanl_cyber1.py`, so four real views over 10 hosts was built
instead; that is what the three listed assertions need.

---

## 2. RANGE-DEPENDENCE FINDING

**Amendment section 10.3's premise holds on the reduced ranges and NOT in the
range the sweep runs on.**

| range | test ROC AUC (seed 0 / seed 1) | balanced accuracy | attack recall |
|---|---|---|---|
| days 1–6 | 0.7319 / 0.7189 | 0.5000 | 0.0000 |
| days 1–7 | 0.7128 / 0.7089 | 0.5000 | 0.0000 |
| **days 1–16** | **0.7201 / 0.7463** | **0.5512 / 0.5549** | **0.1024 / 0.1098** |

Evidence: `results/lanl_v2/smoke.json`, `smoke_days7.json`,
`smoke_days16.json`. All three cells are `with_auth`, `naive`, mode
`default`, seeds 0 and 1 — two seeds, one arm, one system, and therefore not
findings about any hypothesis.

Section 10.3 states that the model "emits an attack probability below 0.5 on
every test row", which pins balanced accuracy at exactly 0.5. That is true at
days 1–6 (max `prob_attack` 0.158440, zero rows at or above 0.5) and at days
1–7. **At days 1–16 it is false**: balanced accuracy moves off 0.5 and attack
recall is roughly 0.10. The sweep runs on days 1–16, so the structural premise
does not hold in the target range.

**The primary metric is unaffected and stands.** Threshold-free ROC AUC is
justified by **L16** ("rank well and threshold badly") and **L23** ("weighted
precision collapses at the default threshold"), both recorded before Part 38
and neither contingent on the model being degenerate. What the finding removes
is one supporting observation, not the reason. The per-unit degeneracy
diagnostic added by section 10.2 is also unaffected: it is defined as a
measurement per unit, and it will simply read "not degenerate" on days 1–16.

**The amendment remains dated and frozen.** This finding is recorded here, not
retrofitted into section 10. Correcting section 10.3 would mean editing a
dated amendment after numbers were looked at, which is exactly what the
pre-registration forbids. A second dated amendment is the user's call.

---

## 3. Sweep estimates carried in from step 1

From `results/lanl_v2/sweep_estimate.json`, built from the completed 16-day
build and 16 timed probes on that table, plus two measured 11-way concurrency
tests. Measured on an i5-13420H, 8 physical cores, 12 logical, 15.7 GB RAM.

| mode | schedule | estimated wall-clock | peak RSS |
|---|---|---|---|
| **B** `threadsdefault_sequential` | 40 units, sequential, 8 threads | **~9.9 h** | **505 MB** (1 process) |
| **A** `threads1_concurrent11` | 40 units, 11 concurrent, 1 thread | **~2.0 h** | **5,372 MB** (11 processes) |

Per-unit training spans 453–1,524 s by arm and system. Observed variance on
the one cell measured twice end to end: **679.4 s and 942.6 s** for
`with_auth` / `naive` / mode B, a spread of **263 s** against a 696 s
estimate, attributable to other load and thermal state on a laptop CPU.

**Mode B is the launched path.** Mode A is not run in this step: it needs an
idle box for 5.4 GB across 11 processes, and the two modes are never combined
in one run — pre-registration section 6 forbids pooling them into a single
+/- in any case.

**The estimates are a lower bound,** for the reason recorded in
`sweep_estimate.json`: Part 30's recorded per-unit seconds run 14.0x–23.8x its
own committed solo training projection, where Part 38 measures a 2.60x
contention penalty. `run_lanl.py` times the whole unit including per-unit
diagnostics while the Part 38 probe times `train` alone, so the excess is
per-unit work the probe does not run, machine state, or both. It is not
measured and is not folded in.

---

## 4. Declared gap in the ENIV diagnostics

Pre-registration section 3 asks for eigen ENIV and mean pairwise dependence,
with the design effect "reported beside it, never gated on".

`prismflow.models.PrismFlow.forward` calls `compute_eniv(matrix, present)` at
its default method, which is **eigen**, and
`prismflow.evaluation.protocol.collect_predictions` returns only the
batch-weighted scalars `eniv`, `mean_dependence` and `efficiency_ratio` — not
the dependence matrix. Both files are frozen. So **the design-effect ENIV is
not captured by this code path** and the sweep records it as not measured
rather than deriving it from a scalar. `design_effect_n_eff` needs the full
matrix, which would mean a second eval loop outside the frozen collector.

The headline method is eigen, which is what the pre-registration gates on, so
no hypothesis depends on the gap. Both are `null` for the `naive` system by
construction: `output.eniv` is `None` when `use_discount` is false.

---

## 5. Launch record

| | |
|---|---|
| Launched | 2026-11-06 |
| Branch | `part38-lanl` |
| Mode | B — `default`, machine-default thread count, sequential |
| Units | 4 arms x 2 systems x 5 seeds = 40 |
| Sample table | `results/lanl_v2/samples_v2_days16.npz`, sha256 `03b7602a590f9c64196ceee2750981187f456541d81927cb69d7b7d6aa251a9a` |
| Per-unit output | `results/lanl_v2/units_default/<arm>_<system>_seed<N>.json` |
| Per-unit scores | `results/lanl_v2/scores_default/<arm>_<system>_seed<N>.npz` |
| Aggregate | `results/lanl_v2/sweep_default.json` |
| Not run here | mode A, and every cross-mode comparison (H3) |
