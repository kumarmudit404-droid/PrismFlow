# Part 38 — LANL redesign and retrain

**Stand-alone section, written 2026-11-21.** It integrates into
`docs/FINAL_REPORT.md` as a subsection of **§4, Real-data results**, after
§4.3 (`LANL cyber1 — clean-data evidence only`), whose figures it does not
replace: Part 30's table stands as the record of Part 30's design. Part 38 is
a **different sample**, and the two are not comparable row-for-row.

At the time of writing the mode `default` sweep is **25 of 40 units** and
still running, so this section reports the **three scalar passes only**.
Every arm figure, and H2 and H3, are marked pending. Every number cites the
committed file and field it came from; absences say **not measured** and are
not estimated.

---

## 1. Headline

Part 38 built the sample Part 30's limitations asked for — controls chosen
without consulting the `auth` stream, and benign cells of attack hosts
retained — and measured what that changed.

**It removed the volume signals, left `auth`'s own label-adjacency exactly
where it was, and installed a stronger `proc`-occupancy selection effect in
its place.** The pre-registered prediction that the auth-presence baseline
would fall materially is **disconfirmed**. So the finding is not that the
selection was removed; it is that **the selection moved**, and that meeting
L24's written closing condition turns out not to be sufficient to remove
selection from a LANL sample at all.

Two further results carry no good news and are stated as they are. The
pre-registered size fallback fired, so L27's contrast is being tested at **a
quarter** of its intended strength. And L28 **cannot be closed by its own
text** under this pre-registration, because its closing condition asks for a
standard deviation pooled across execution modes and the pre-registration
forbids that pooling.

## 2. What Part 30 left open

Three limitations from `docs/KNOWN_LIMITATIONS.md`, each naming a closing
condition that is a condition on the **design**, not on a number:

| | the claim | the written closing condition |
|---|---|---|
| **L24** | the `auth` view is label-adjacent, and dropping it does not drop the selection | "A sample whose control cells are drawn without reference to the `auth` stream, which would separate the view drop from the selection." |
| **L27** | one host carries a third of the attack class, with no benign contrast | "A design that keeps all or a stated share of the benign cells of attack hosts, which would be needed to separate host from behaviour." |
| **L28** | the execution mode changes the fitted weights, and the per-seed std understates the spread | "Seeds run across at least two execution modes, so the reported std covers mode as well as seed." |

Under Part 30's design, every one of the 410 test attack samples held an
`auth` record while about a third of benign samples did not, so the untrained
`auth_present_indicator` reached test ROC AUC **0.6637** without being fitted
at all (`results/lanl/results.json`,
`phase3_diagnostics.trivial_baselines.baselines.auth_present_indicator.test_auc`
= **0.6636552213548774**). The `without_auth` arm removed the auth *features*;
it could not remove that *selection*, because cells had been chosen from all
four views.

## 3. What Part 38 was designed to test

Pre-registered in `experiments/lanl_v2/PRE_REGISTRATION.md` before any cell was
sampled, with Amendment 1 (dated 2026-11-06) at its section 10. Both are
frozen.

The time grid, split, feature list, label and window length are unchanged from
`configs/data/lanl.yaml`. What changed is **which cells enter the sample**.
Every `attack = 1` cell is kept, as in Part 30. A cell is kept as a control
iff **both**:

- **(a)** it is occupied in at least one of `proc`, `flows`, `dns` — **the
  `auth` stream is never consulted**; `auth` features are attached in a later
  pass, so `auth` enters as features only. This is how L24's condition is met.
- **(b)** its computer has at least one attack cell in the read range, in
  which case all its benign cells are kept; otherwise it is kept iff
  `crc32(computer|window) % 512 == 0`, Part 30's rule at its unchanged
  divisor. This is how L27's condition is met.

Three hypotheses, each barred against a committed number:

| | prediction | bar |
|---|---|---|
| **H1** (L24) | the auth-presence-only baseline falls materially | a drop of at least **0.05** from **0.6636552213548774** |
| **H2** (L27) | dropping `C17693`'s test rows changes test ROC AUC by less than 0.036, in all eight arm-and-system cells | **0.036**, the smallest of Part 30's committed per-arm drops |
| **H3** (L28) | the sign of the PrismFlow-minus-naive ROC AUC difference agrees across both execution modes, in all four arms | sign agreement; no magnitude bar |

**The primary metric is threshold-free ROC AUC.** Balanced accuracy at the
default 0.5 threshold is secondary (Amendment 1, section 10.1), justified by
L16 ("rank well and threshold badly") and L23 ("weighted precision collapses
at the default threshold"), both recorded before Part 38.

**The sampler was audited before anything was trained**
(`results/lanl_v2/SWEEP_EXECUTION_LOG.md` section 1, verdict passed): nine of
nine exact against Part 30's committed `population.per_view.<view>` sums over
three read ranges; records scanned matching Part 30 exactly at 16 days
(`proc` 115,159,740, `flows` 121,284,699, `dns` 1,055,902); attack counts to
the row; two independent 11 GB scans agreeing field for field; and
`tests/unit/test_lanl_v2_sampler_synthetic.py` comparing `select_cells`
against a brute-force re-derivation of the rule **cell by cell**, which is the
check a sum cannot give.

**The size fallback fired, and this bounds what L27 can show.**
`results/lanl_v2/build_v2_days16.json`, `size_fallback`:
`attack_host_benign_share_k` **2**, `attack_host_benign_share` **`"1/4"`**,
`rows_before_fallback` **979122**, `rows_after_fallback` **281824**,
`attack_host_benign_cells_before` **929764**,
`attack_host_benign_cells_dropped` **697298**, against `max_total_bar`
**400000**. The bar and the fallback were both fixed before any count was
seen, so this is mechanical rather than a choice — but **697,298 of 929,764
attack-host benign cells were dropped**, and L27's "keeps all or a stated
share" is met on its *stated share* branch at **1/4**, not on "all".

Realised sample: `n_samples` **281823**; train **197363** at prevalence
**0.0022243277615358503**; test **84460** at prevalence
**0.0048543689320388345**; `attack_hosts` **302**.

## 4. What the three scalar passes showed

Three seed-free passes over the committed sample tables. **No training, no
inference, no model**; each re-runs bit-exact and is unchanged under
`OMP_NUM_THREADS=1`; all three are covered by
`tests/unit/test_h1_scalar_passes.py` (12 tests). They are separate programs
because **the sweep does not compute H1**:
`experiments/lanl_v2/metrics.py:80` defines `auth_presence_auc` and
`run_lanl_v2.py` never calls it.

Orientation follows Part 30's rule by *importing* `trivial_scalar_values` and
`roc_auc_binary` from the frozen `experiments/lanl/run_lanl.py` rather than
reimplementing them — the sign is chosen on the train block only and applied
unchanged to test — so Part 38's values and Part 30's 0.6637 come from the
same code.

### 4.1 H1 is disconfirmed: the auth-presence baseline did not fall

`results/lanl_v2/h1_auth_presence.json`, `baselines.<name>`:

| scalar | train AUC | Part 38 test AUC | Part 30 test AUC | drop |
|---|---|---|---|---|
| **`auth_present_indicator`** | 0.6675392537222482 | **0.6611838191552647** | 0.6636552213548774 | **0.0024714021996127267** |
| `total_records_across_views` | 0.6918403739721934 | 0.5862805385876582 | 0.6842788175411642 | 0.09799827895350599 |
| `n_views_present` | 0.5645444186716992 | 0.4800433975130947 | 0.5624544155807845 | 0.08241101806768975 |

All three oriented `as-is`. The bar was a test AUC at or below
**0.6136552213548774** (`h1.bar_test_auc_at_or_below`); the drop achieved was
**0.0024714021996127267** against a required **0.05**
(`h1.drop_achieved`, `h1.min_drop_required`). **`h1.verdict` is
`DISCONFIRMED`.** The baseline is seed-free, so this is a single value and no
`+/-` applies.

**The two volume scalars did fall, and hard** — record volume by
**0.09799827895350599** and views-present by **0.08241101806768975**, the
latter to **0.4800433975130947**, below chance. So the new selection rule
demonstrably changed what occupancy-volume signals can do. It left auth
presence where it was.

### 4.2 The contingency behind that AUC survived intact

`results/lanl_v2/h1_conditional_rates.json`. An AUC is a ranking; it does not
say whether the underlying contingency survived. L24's claim is a contingency,
so it was recomputed on both tables — and **Part 30's side is recomputed, not
quoted**, because L24's 12,323 figure is prose in `KNOWN_LIMITATIONS.md` and
is **not a field** in `results/lanl/results.json`. The pass gates on
reproducing it from `results/lanl/samples.npz` before any comparison:
`l24_reproduction_gate.status` = **`PASSED`** on all four checks.

| view | Part 30 attack / benign | Part 38 attack / benign | attack rate | benign rate |
|---|---|---|---|---|
| auth | 410/410 · 12323/18319 | 410/410 · 56955/84050 | 1.0 → **1.0** | 0.6726895572902452 → 0.6776323616894706 |
| proc | 204/410 · 12774/18319 | 204/410 · 76848/84050 | 0.4975609756097561 → 0.4975609756097561 | 0.6973088050657787 → **0.9143129089827484** |
| flows | 127/410 · 5149/18319 | 127/410 · 21778/84050 | 0.3097560975609756 → 0.3097560975609756 | 0.28107429444838694 → 0.2591076740035693 |
| dns | 8/410 · 340/18319 | 8/410 · 1646/84050 | 0.01951219512195122 → 0.01951219512195122 | 0.018559965063595175 → 0.019583581201665676 |

**`attack_rate_change` is 0.0 on all four views.** Both designs keep every
attack cell and the test time range is unchanged, so the attack population is
literally the same 410 rows and only the benign control population differs.
That is an internal check on both samplers, not a result.

**`auth_verdict.still_exactly_one_on_part38` is true.** Every attack cell
still holds an `auth` record — attack rate **exactly 1.0** — on a sample whose
selection never consulted the stream, and the benign rate moved by
**0.004942804399225453**. **So the auth-presence separation is a property of
the data, not an artefact of how Part 30 drew its controls.** L24's first
claim — the labels derive from authentication records, so `auth` is
label-adjacent — is strengthened by Part 38, not weakened.

### 4.3 The selection moved to `proc`

`results/lanl_v2/h1_occupancy_decomposition.json`. Every selection input
scored as its own trivial baseline, with the rule-(b) predicates rebuilt from
frozen code rather than paraphrased (host set via `read_redteam` →
`cells_in_read_range` → `attack_hosts`, `attack_hosts_in_read_range` **302**;
hash predicates at `control_divisor` **512** and `fallback_k` **2**).

`whole_test_block.<scalar>.test_auc`, beside
`within_attack_hosts.<scalar>.test_auc` (`n_test_on_attack_hosts` **70130** of
84460):

| scalar | whole test block | within attack hosts |
|---|---|---|
| `fallback_survivor_k2` | 0.8670731707317073 | 0.8670731707317073 |
| **`occupancy_proc`** | **0.7083759666864962** | **0.7143004072037278** |
| `auth_present_indicator` | 0.6611838191552647 | 0.6524885255306943 |
| `n_non_auth_views_present` | 0.6503477024419263 | 0.647878062773743 |
| `identity_hash_kept` | 0.5883878643664485 | 0.5037865748709123 |
| `total_records_across_views` | 0.5862805385876584 | 0.569008332983502 |
| `occupancy_flows` | 0.5253242117787031 | 0.5347905559520311 |
| `occupancy_dns` | 0.5000356930398572 | 0.4996673103564082 |
| `n_views_present` | 0.4800433975130947 | 0.4757941172354925 |

**`proc` occupancy now carries more of the separation than `auth` does**, and
the conditional rates say why: `proc`'s attack rate is unchanged at
**0.4975609756097561** while its benign rate rose from
**0.6973088050657787** to **0.9143129089827484**, because rule (a) admits a
cell only if `proc`, `flows` or `dns` is occupied. Its rate gap more than
doubled, **−0.19974782945602265 → −0.4167519333729923**. `occupancy_dns` sits
at chance (**0.5000356930398572**) and `occupancy_flows` is weak.

**This is not the host-rule tautology.** Within attack hosts, where that
tautology is held constant, the same ordering holds: `proc`
**0.7143004072037278** above `auth` **0.6524885255306943**. That is the
within-attack-host contrast `PRE_REGISTRATION.md` section 2 names as the
figure that matters.

Two entries in the table are instrument readings rather than findings, and are
labelled as such in the committed file:

- **`attack_host_indicator`** (whole block **0.5852468768590124**) is marked
  `tautological: true` and excluded from the ranking. It is 1 on every attack
  row by construction, since an attack cell makes its own computer an attack
  host, and within attack hosts it is constant with AUC 0.5 by definition.
- **`identity_hash_kept`** is a consistency check that passed:
  **0.5883878643664485** whole-block against **0.5037865748709123** within
  attack hosts. Its whole-block value came entirely from the attack-host /
  non-attack-host split, since every non-attack-host control satisfies
  `crc32 % 512 == 0` by construction, and within hosts it sits at chance — as
  a hash must.

**`fallback_survivor_k2` tops the table at 0.8670731707317073 and is not a
signal any model can use.** `crc32` is not a feature and no encoder sees it.
It scores as it does because attack cells were kept unconditionally while
attack-host benign cells were thinned by `crc32 % 4 == 0`, so the surviving
benign rows satisfy a predicate the attack rows mostly fail. **It says the
class balance on attack hosts was manufactured by a hash. It does not say the
model has a leak.** It bears on L27's rule, not L24's.

### 4.4 The two passes cross-validate exactly

For a binary indicator, `AUC = 0.5 + |attack_rate − benign_rate| / 2`. The
rate-implied AUCs from §4.2 reproduce the measured AUCs from §4.3 to
**0.00e+00** on all four views, and the same algebra on Part 30's `auth` rate
gap returns **0.6636552213548774** — L24's committed figure. Two independent
implementations and the prior Part's record agree. Asserted in
`tests/unit/test_h1_scalar_passes.py::test_h1_occupancy_decomposition_rates_cross_validate`.

## 5. What is pending

### 5.1 The sweep tables

40 units = 4 arms × 2 systems × 5 seeds in **one** execution mode. At the time
of writing: **25 of 40**. `experiments/lanl_v2/fill_analysis.py`, behind a
40-unit gate, then writes `results/lanl_v2/filled_tables.json`.

| table | content | status |
|---|---|---|
| 1 | ROC AUC mean ± sample std (ddof=1) per arm × system | **pending the sweep** |
| 2 | per-seed ROC AUC, 40 cells, plus per-seed signs of PrismFlow − naive | **pending the sweep** |
| 3 | **H2** — leave-host-out delta, 8 cells × 5 hosts; only `C17693` is barred | **pending the sweep** |
| 4 | **H1** — the auth-presence baseline | **complete**, §4.1 |
| 5 | **H3** — cross-mode delta | **pending mode A**, not run |
| 6 | per-unit wall-clock, peak RSS, degeneracy diagnostic | **pending the sweep** |

A cell with fewer than five completed seeds is **not a finding** and no
verdict may be drawn from it.

### 5.2 The H2 preview — an operational note, NOT evidence

A mid-sweep `--dry-run` of `fill_analysis.py` exercised the table code against
the units then on disk. **Its output was deliberately not committed and no
figure from it appears in this section or anywhere else.** It is named here
only because of one piece of logic worth knowing in advance:

**H2's rule is existential** — "H2 is disconfirmed if **any** of the eight
arm-and-system cells still moves by 0.036 or more" (PRE_REGISTRATION section
1). Three of the eight cells reached five seeds before the others, so H2's
verdict may become determinable before the sweep reaches 40/40. The dry run's
reading pointed toward **disconfirmation**, and that reading has **no
committed path and must not be reported as a result**. The verdict belongs to
`results/lanl_v2/filled_tables.json`,
`table3_leave_host_out.per_cell.<arm>/<system>.per_host.C17693`, read at
40/40.

When it is written, **two caveats must travel with it**, both fixed in advance
in `docs/proposed_limitations_update.md`:

1. **The realised share is 1/4.** A disconfirmation therefore shows that
   restoring a *quarter* of the benign contrast did not help — a narrower
   claim than H2's mechanism makes.
2. **H2's bar sits inside Part 30's own band.** The eight committed `C17693`
   deltas run **−0.035962653276654266** to **−0.051874860066125961**
   (`results/lanl_reanalysis/results.json`,
   `step5_l27.per_arm.<arm>.<system>.per_host.C17693.change_from_all_test`),
   and the smallest is **below** the 0.036 bar. A cell reproducing Part 30's
   weakest cell would be scored as passing, so a confirmation would not by
   itself mean the sensitivity shrank.

One host is not a resample, so **no interval attaches** and the result is
descriptive — Part 30's own `step5_l27.interval` records the same.

### 5.3 Mode A, and why L28 cannot be closed here

Mode A (`threads1_concurrent11`) has not been run.
`results/lanl_v2/sweep_estimate.json` puts it at **~2.0 h** across 11
concurrent single-thread processes at **~5,372 MB**. It needs no new code.
Without it, Table 5 stays `pending mode A` and H3 has no verdict.

**And a clean H3 confirmation would still not close L28.** L28's written
closing condition asks for a standard deviation that covers "mode as well as
seed" — a *pooled* `+/-`. PRE_REGISTRATION section 6 forbids exactly that:
"The two modes are **never pooled into a single +/-**: L28's finding is that a
5/5 seed count inside one run shares one execution mode and is not independent
evidence about mode effects." The pre-registration substitutes H3, a
**sign-agreement** test, and that is the only barred L28 analysis this design
can produce. Closing L28 on its own terms would need a review-gate decision
that a mode-pooled `+/-` is legitimate, and a second dated amendment.

**Any magnitude bar on the cross-mode comparison must be pre-registered before
mode A runs.** A "within three standard deviations" bar would be already
failed by the prior record: L28 reports the same seed differing by **0.0140**
in AUC across modes (naive seed 0: **0.8375** and **0.8235**), about **four**
per-seed standard deviations of that arm and system (**0.0033**).

Every ENIV figure is reported with the thread count it was computed at. Mode B
observed **8** threads and mode A sets **1**, and a committed `eniv_eigen` of
**3.4396320614332856** at the machine default is missed by **0.0107** at one
thread (**3.4289614712229826**, Part 34).

### 5.4 A premise of Amendment 1 does not hold in the range the sweep runs on

Recorded in `results/lanl_v2/SWEEP_EXECUTION_LOG.md` section 2 rather than
retrofitted into the dated amendment. Amendment section 10.3 states that the
model "emits an attack probability below 0.5 on every test row", pinning
balanced accuracy at exactly 0.5. That is true at days 1–6 and days 1–7 and
**false at days 1–16**, which is the range the sweep runs on: balanced
accuracy moves off 0.5 and attack recall is roughly 0.10
(`results/lanl_v2/smoke.json`, `smoke_days7.json`, `smoke_days16.json` — two
seeds, one arm, one system, and **therefore not findings about any
hypothesis**).

**The primary metric is unaffected.** Threshold-free ROC AUC is justified by
L16 and L23, neither of which is contingent on the model being degenerate.
What the finding removes is one supporting observation, not the reason.
Correcting the amendment would mean editing a dated document after numbers
were looked at, which the pre-registration forbids.

## 6. Not measured

Absences, not zeros. None of these is estimated anywhere in this section.

| quantity | status |
|---|---|
| Tables 1, 2, 3 and 6 | **not measured** — the sweep is at 25 of 40 |
| H2's verdict | **not measured** — pending Table 3 at 40/40 |
| H3's verdict, and every cross-mode figure | **not measured** — mode A not run |
| the design-effect ENIV | **not measured** — `prismflow.evaluation.protocol.collect_predictions` returns batch-weighted scalars and no dependence matrix, and is frozen; recorded per unit in `eniv_design_effect_status` |
| how much of `proc`'s 0.7083759666864962 is selection-induced enrichment versus behavioural signal | **not measured** — no decomposition exists; `proc` occupancy *is* visible to the encoders through the view mask, so unlike `fallback_survivor_k2` it cannot be set aside as unusable |
| L27's contrast at the intended 1/1 share | **not measured** — the fallback fired at `k = 2`; testing 1/1 would need the pre-registered 400,000 bar raised, which is a review-gate decision |
| eigen ENIV for the `naive` system | **not applicable** — `output.eniv` is `None` when `use_discount` is false, and is reported `null` rather than 0 |

## 7. Status of the three limitations

**No limitation's status is changed by this section.** PRE_REGISTRATION
section 7.6: this programme "does not mark L24, L27 or L28 closed, resolved or
narrowed. Status changes require the full sweep and a review gate." Proposed
text for all three is queued in `docs/proposed_limitations_update.md` under
`## Part 38 proposals`, marked PROPOSAL.

- **L24 — proposed OPEN, narrowed.** Its closing condition is met; H1 is
  disconfirmed; the selection effect is relocated to `proc` rather than
  removed. **Rule (a) has to gate on something, and whatever it gates on
  acquires the benign-side enrichment `proc` now carries** — so "sample
  selection survives the view drop" remains true of Part 38, for a different
  view. What would change this: a selection rule whose admission criterion is
  independent of every view's occupancy, which on this data may not exist,
  since a cell with no occupied view carries no features.
- **L27 — proposed pending.** Fills from Table 3 at 40/40, with the 1/4 share
  and the bar-inside-the-band caveat attached.
- **L28 — proposed pending, and NARROWED at best.** Unclosable by its own text
  under this pre-registration, whatever mode A shows.

## 8. How to re-check this section

```sh
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/h1_auth_presence.py
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/h1_occupancy_decomposition.py
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/h1_conditional_rates.py
PYTHONPATH=. ./.venv/Scripts/python.exe -m pytest -q tests/unit/test_h1_scalar_passes.py
```

Each pass rewrites its own JSON in place; copy the file aside and `diff` to
confirm it reproduces, and re-run under `OMP_NUM_THREADS=1` to confirm thread
invariance. `experiments/lanl_v2/README.md` section 5 has the inputs and their
hashes.
