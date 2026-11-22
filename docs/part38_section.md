# Part 38 — LANL redesign and retrain

**Stand-alone section, written 2026-11-21.** It integrates into
`docs/FINAL_REPORT.md` as a subsection of **§4, Real-data results**, after
§4.3 (`LANL cyber1 — clean-data evidence only`), whose figures it does not
replace: Part 30's table stands as the record of Part 30's design. Part 38 is
a **different sample**, and the two are not comparable row-for-row.

**Updated 2026-11-22: the mode `default` (B) sweep is complete at 40 of 40
units.** H1 and H2 are both decided and both **disconfirmed**; H3 is pending
mode A and is additionally **unreachable on this sample**, for a reason the
sweep measured. Every number cites the committed file and field it came from;
absences say **not measured** and are not estimated.

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

**The sweep then disconfirmed the second prediction in the opposite
direction.** H2 predicted that restoring `C17693`'s benign contrast would
shrink the leave-host-out sensitivity. It **grew** — all eight arm-and-system
cells move by **0.0997 to 0.1134** against a 0.036 bar, where Part 30's own
eight cells moved by 0.0360 to 0.0519. The two bands do not overlap. So the
one-host concentration that L27 names is now measured as a **larger** threat
to every per-arm AUC than Part 30 recorded, on a design built to reduce it.

**And the fusion thesis gains no support here.** Across four arms, five seeds
and one execution mode, the dependence discount changed nothing measurable in
ranking quality: three of four arms show a PrismFlow-minus-naive difference
**smaller than their own per-seed std** (reported as *sign undetermined*), and
the one arm with an established direction — `without_auth`, −0.0032 ± 0.0022,
negative in 4 of 5 seeds — runs **against** PrismFlow.

Two further results carry no good news and are stated as they are. The
pre-registered size fallback fired, so L27's contrast was tested at **a
quarter** of its intended strength. And L28 **cannot be closed by its own
text** under this pre-registration, because its closing condition asks for a
standard deviation pooled across execution modes and the pre-registration
forbids that pooling — and as of 40/40, running mode A could not close it
either, because H3's "all four arms" condition has no sign to agree with in
three of them.

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

## 5. What the sweep showed, and what is still pending

### 5.1 The sweep tables

40 units = 4 arms × 2 systems × 5 seeds in **one** execution mode.
**Complete at 40 of 40** (`results/lanl_v2/logs/sweep_resume2.out`: *4
trained, 6 skipped, 7844s*). `experiments/lanl_v2/fill_analysis.py`, behind
its 40-unit gate, wrote `results/lanl_v2/filled_tables.json`.

| table | content | status |
|---|---|---|
| 1 | ROC AUC mean ± sample std (ddof=1) per arm × system | **filled**, every cell `n = 5` |
| 2 | per-seed ROC AUC, 40 cells, plus per-seed signs of PrismFlow − naive | **filled** |
| 3 | **H2** — leave-host-out delta, 8 cells × 5 hosts; only `C17693` is barred | **filled**, H2 **DISCONFIRMED**, §5.2 |
| 4 | **H1** — the auth-presence baseline | **complete**, §4.1 |
| 5 | **H3** — cross-mode delta | mode B columns filled; mode A columns **pending mode A**, not run |
| 6 | per-unit wall-clock, peak RSS, degeneracy diagnostic | **filled**, §5.5 |

Full tables with per-cell citations: `experiments/lanl_v2/ANALYSIS_TEMPLATE.md`
§2. Source for every value below:
`results/lanl_v2/filled_tables.json`.

**Table 1 — test ROC AUC, mean ± sample std over 5 seeds, mode B, 8 threads**
(`table1_roc_auc_per_cell.per_cell.<arm>/<system>.roc_auc`):

| arm | naive | prismflow |
|---|---|---|
| `without_auth` | **0.7900364591 ± 0.0020106638** | 0.7868264419 ± 0.0035939990 |
| `without_flows` | 0.7700035490 ± 0.0040878521 | 0.7677030484 ± 0.0038525715 |
| `without_dns` | 0.7440774046 ± 0.0084693584 | 0.7447516055 ± 0.0082767688 |
| `with_auth` | 0.7401838830 ± 0.0122072868 | 0.7419867355 ± 0.0092586512 |

Every cell is `n = 5`, so the 5-SEED RULE is satisfied for every row. Each ±
is **initialisation-and-batch-order variance on a fixed sample** — a seed
cannot redraw or re-partition this data (PRE_REGISTRATION §4) — so it is not
dataset variance and not a spread over independent attacks.

**Dropping the `auth` view raises ROC AUC, which reverses Part 30's framing.**
`with_auth` − `without_auth` is **−0.0498525761** (naive) and
**−0.0448397063** (prismflow), negative in **5 of 5** paired seeds in both
systems. Under Part 30 `without_auth` was the handicapped arm; on Part 38's
sample it is the **best** arm, and it also carries the highest attack recall
(0.3537 ± 0.0112 against 0.1044–0.1268 elsewhere) and the highest balanced
accuracy (0.6768 ± 0.0056 against 0.5490–0.5578). This is descriptive, carries
**no pre-registered bar**, is **not H1**, and cannot stand in for H1 — but it
is what relocating the selection effect to `proc` (§4.3) predicts.

**Table 2 — PrismFlow − naive, per-seed signs reported before the means**
(`table2_per_seed.prismflow_minus_naive.<arm>`):

| arm | seeds + | seeds − | seeds exactly 0 | mean | std | reading |
|---|---|---|---|---|---|---|
| `without_auth` | 1 | **4** | 0 | **−0.0032100173** | 0.0021726225 | **negative**, \|mean\| > std |
| `without_flows` | 1 | **4** | 0 | −0.0023005006 | 0.0027377719 | **sign undetermined** |
| `without_dns` | **3** | 2 | 0 | +0.0006742009 | 0.0025411558 | **sign undetermined** |
| `with_auth` | **3** | 2 | 0 | +0.0018028525 | 0.0058650782 | **sign undetermined** |

**No arm favours the discount, and the one arm with an established direction
runs against it.** Only `without_auth` has `|mean| > std`, and its sign is
negative. The other three are *sign undetermined* — the difference is smaller
than the per-seed std, so neither direction is established, and reporting the
mean alone would misrepresent them. `seeds_exactly_zero` is **0** in all four
arms, so no "x of 5 agree" gloss is being misread here. `with_auth` has the
largest magnitude **and** the largest std, and flips sign twice across its
five seeds — the cancelling population the per-seed rule exists to expose.

**This is a null result for the thesis on this dataset, and it is reported as
one.** It is four arms, five seeds and **one** execution mode on a single
sample, so it is not evidence that the discount cannot help anywhere; it is
evidence that on Part 38's LANL sample it did not.

### 5.2 H2 is disconfirmed, and the sensitivity grew rather than shrank

**H2's rule is existential** — "H2 is disconfirmed if **any** of the eight
arm-and-system cells still moves by 0.036 or more" (PRE_REGISTRATION section
1). One cell would have sufficed. **All eight fired.** Read at 40/40 from
`results/lanl_v2/filled_tables.json`,
`table3_leave_host_out.per_cell.<arm>/<system>.per_host.C17693.change_from_all_test`,
with Part 30's committed value for the same cell beside it
(`results/lanl_reanalysis/results.json`,
`step5_l27.per_arm.<arm>.<system>.per_host.C17693.change_from_all_test`):

| arm | system | Part 38 Δ | Part 30 Δ | growth in magnitude, \|Part 38\| − \|Part 30\| |
|---|---|---|---|---|
| `without_auth` | naive | −0.099710 | −0.051875 | **+0.047835** |
| `without_auth` | prismflow | −0.099784 | −0.050799 | **+0.048984** |
| `without_flows` | naive | −0.103284 | −0.043474 | **+0.059810** |
| `without_flows` | prismflow | −0.103105 | −0.043361 | **+0.059744** |
| `without_dns` | naive | −0.108785 | −0.035963 | **+0.072822** |
| `without_dns` | prismflow | −0.104055 | −0.036776 | **+0.067279** |
| `with_auth` | naive | **−0.113438** | −0.037837 | **+0.075601** |
| `with_auth` | prismflow | −0.108746 | −0.040041 | **+0.068705** |

`h2_verdict.verdict` = **`DISCONFIRMED`**; `max_abs_delta` **0.1134384687**,
`min_abs_delta` **0.0997097614**, against a bar of **0.036**.

**H2 predicted the sensitivity would shrink. It grew, in all eight cells.**
Removing one host now costs 0.0997–0.1134 AUC where Part 30 recorded
0.0360–0.0519. Part 38's band and Part 30's band **do not overlap**, and
Part 38's smallest cell is nearly twice Part 30's largest.

**The bar-inside-the-band caveat does not weaken this verdict**, and that is
worth stating because it was written in advance expecting it might. Part 30's
own band runs **−0.035962653276654266** to **−0.051874860066125961** and its
smallest cell sits *below* the 0.036 bar, so a Part 38 cell reproducing
Part 30's weakest would have scored as passing. That caveat guards against a
false **pass**. There was no pass. The verdict rests on two non-overlapping
bands, not on the threshold.

**Rule (b) did restore benign contrast — the host set was re-derived on
Part 38's own test block, not copied, and returns the same five computers in
the same rank order** (`table3_leave_host_out.hosts`):

| host | test rows removed | attack removed | benign removed | Part 30 benign |
|---|---|---|---|---|
| `C17693` | 147 | 137 | **10** | 1 |
| `C1015` | 270 | 12 | 258 | 1 |
| `C2388` | 295 | 10 | 285 | 3 |
| `C1936` | 205 | 7 | 198 | 0 |
| `C2079` | 8 | 7 | **1** | 3 |

`C17693` holds **10** benign rows where Part 30 held **1**, with its 137
attack rows unchanged. The attack counts are identical for every host, as they
must be — both designs keep every attack cell and the test range is the same.
`C2079` is the host that gained no benign rows, and it is the one whose
removal barely moves any cell (−0.0034 to −0.0044). **So contrast was restored
and the sensitivity still grew.**

**Three limits on what this disconfirmation licenses.**

1. **The realised share is 1/4.** It shows that restoring **a quarter** of
   `C17693`'s benign contrast did not help — a narrower claim than H2's
   mechanism makes. It does **not** show that restoring benign contrast fails;
   the 1/1 build was never run, and what it would have measured is **not
   measured**.
2. **It says nothing about *why* the sensitivity grew.** A plausible reading
   exists and is **not measured**: Part 38's test block is 84,460 rows at
   prevalence 0.00485 against Part 30's 18,729 at 0.0219, so `C17693`'s 137
   attack rows are a third of the attack class inside a four-and-a-half times
   larger block, and removing them removes a larger share of the rankable
   positive signal. No decomposition distinguishes that from a behavioural
   effect. It is not asserted here.
3. **No interval attaches.** One host is not a resample; the result is
   descriptive — Part 30's own `step5_l27.interval` records the same. The
   per-seed arrays are at
   `table3_leave_host_out.per_cell.<cell>.per_host.C17693.roc_auc_host_removed.per_seed`.

**The four other hosts are exploratory, not pre-registered**, and must never
be aggregated with the `C17693` column. They move every cell by at most
0.0075 in absolute value — two orders of magnitude less than `C17693`.

**And `fallback_survivor_k2` belongs beside this verdict** (§4.3): the `k = 2`
thinning predicate is the strongest scalar on the test block at
**0.8670731707317073**. It is not a signal a model can use, but it means the
attack/benign balance on attack hosts is a hash's doing — which is exactly the
contrast H2's mechanism relies on.

### 5.3 Mode A, and why L28 cannot be closed here

Mode A (`threads1_concurrent11`) has not been run.
`results/lanl_v2/sweep_estimate.json` puts it at **~2.0 h** across 11
concurrent single-thread processes at **~5,372 MB**. It needs no new code.
Without it, Table 5's mode A columns stay `pending mode A` and H3 has no
verdict.

**New at 40/40: H3 is unreachable on this sample whatever mode A shows, and
this changes what running it could buy.** H3 requires the sign of
PrismFlow − naive to agree across both modes **in all four arms**. On mode B,
**three of the four arms have no established sign** — the difference is
smaller than their own per-seed std, which this programme reports as *sign
undetermined* rather than as agreement (§5.1, Table 2). A sign that does not
exist on one side cannot agree with the other, so mode A could at best decide
the `without_auth` arm, and the "all four arms" condition cannot be satisfied.

This is a property of the **measured effect sizes**, not of mode A's absence,
and it is not a reason to skip mode A: the across-mode spread is still the
figure L28 is about, and it would still be worth measuring. It is a reason not
to expect a verdict from it.

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

### 5.5 Table 6, and a measurement defect it exposes

`table6_operational.per_cell`. No hypothesis is attached to any of it.

**The §10.2 degeneracy diagnostic reads "not degenerate" in all eight cells.**
`max_prob_attack` runs **0.862739 to 0.994147**, every cell well above 0.5,
and `degenerate_any_seed` is `false` for all eight. This is the measured
confirmation of §5.4: Amendment §10.3's premise does not hold at days 1–16.
**It is a reading, not a failure.**

**`peak_rss_mb` is not aggregated, because it cannot honestly be.** It is a
**running maximum over the life of the process** that wrote the unit, so it is
non-decreasing within a launch and a unit's figure can be inherited from an
earlier unit in the same process. Measured directly during this sweep:
`with_auth_naive_seed4` recorded **499.5** and `with_auth_prismflow_seed2`
**528.8** in the same process, and the OS counter `PeakWorkingSetSize` for
that process read **537.3 MB** before `with_auth_prismflow_seed3` finished —
which is exactly the figure that unit then reported. The
`with_auth`/`prismflow` cell's five values are **501.7, 503.1, 528.8, 537.3,
538.6**: strictly increasing, five distinct values, written by three launches.

`fill_analysis.py` was changed for this: `peak_rss_mb` left `TABLE6_FIELDS`
and is emitted per seed with `n_distinct` and `distinct_values`. Before that,
three of the eight cells would have printed a mean with **`std = 0.0`** over
five units that shared **one** observation — reading as perfect
reproducibility when it was a single measurement counted five times.
**Launch identity is `not measured`**: the unit JSON records no `process_id`
and no `launch_time`, and file mtime is not a substitute because a
`git checkout` rewrites it. It is reported as absent rather than
reconstructed.

**`unit_seconds` ± spans launches and is not a reproducibility figure.**
`without_dns`/naive reads **546.0 ± 1.3** (one launch) against
`with_auth`/prismflow at **1929.8 ± 410.7** (three launches). `with_auth`/naive
is the clearest case: its five seeds measured 653.8, 705.4, 1359.8, 1370.8 and
1166.7 seconds across three launches, so its ± 349.4 is machine state, not
seed. Each value is still a genuine per-unit measurement, so the mean remains
a legitimate *operational* summary — it is not evidence about determinism.

## 6. Not measured

Absences, not zeros. None of these is estimated anywhere in this section.

| quantity | status |
|---|---|
| H3's verdict, and every cross-mode figure | **not measured** — mode A not run. And H3 is **unreachable** on this sample regardless (§5.3) |
| why the leave-host-out sensitivity grew rather than shrank | **not measured** — no decomposition separates the 4.5× larger test block from a behavioural effect (§5.2) |
| which launch wrote any given `peak_rss_mb` | **not measured** — no `process_id` and no `launch_time` field in the unit JSON; mtime is not a substitute (§5.5) |
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
- **L27 — proposed OPEN, and the reason is now a measured disconfirmation
  rather than missing instrumentation.** H2 is disconfirmed in all eight
  cells; rule (b) restored `C17693`'s benign rows from 1 to 10 and the
  leave-host-out sensitivity **grew** from Part 30's 0.0360–0.0519 to
  0.0997–0.1134. The closing condition's "all **or a stated share**" branch is
  met at **1/4**, so the condition as written is satisfied and the finding is
  that satisfying it did not separate host from behaviour. What would change
  this: the same design at the **1/1** share, which needs the pre-registered
  400,000-row bar raised — a review-gate decision and a new dated amendment,
  not a parameter to change quietly. The uncapped build is 979,122 rows.
- **L28 — proposed pending, NARROWED at best, and now unclosable by two
  separate routes.** Its own text asks for a mode-pooled ± that the
  pre-registration forbids; and as of 40/40, H3 — the substitute the
  pre-registration offers — is **unreachable on this sample**, because three
  of four arms have no established sign for mode A to agree with (§5.3).
  Running mode A remains worth doing for the across-mode spread itself; it
  will not produce a verdict.

## 8. How to re-check this section

The three scalar passes of §4, and their tests:

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

The §5 tables, which need **no training** — Tables 1, 2 and 6 aggregate the
committed unit JSONs, and Table 3 re-scores the committed
`scores_default/*.npz` with a host mask:

```sh
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/fill_analysis.py
```

It takes about 15 seconds, gates on 40 units, and rewrites
`results/lanl_v2/filled_tables.json`; `diff` against the committed copy.
Re-running the sweep itself reproduces nothing cheaply — 40 units is
roughly 12 hours of training — so the audit path is the saved scores, not a
retrain:

```sh
PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/run_lanl_v2.py sweep \
    --tag v2_days16 --arms with_auth --systems naive prismflow --seeds 0 1 2 3 4 --mode default
```

With all 40 unit JSONs present this skips every unit and only re-runs
`aggregate_sweep`, which is the cheap way to regenerate
`results/lanl_v2/sweep_default.json`. **Run it at the machine default thread
count** — a committed `eniv_eigen` misses a `1e-6` check at
`OMP_NUM_THREADS=1` (3.4289614712229826 against 3.4396320614332856, Part 34).
Note `aggregate_sweep` writes bare `NaN` for the std of an empty cell, which
is invalid strict JSON; the committed file has those 12 replaced with `null`,
so expect that diff and do not treat it as a change in a measurement.
