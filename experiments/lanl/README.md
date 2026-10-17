# Part 30 - PrismFlow on LANL cyber1 (days 1-16)

**STATUS: INCOMPLETE.** Only ['with_auth'] of ['with_auth', 'without_auth'] view configurations finished. **without_auth did NOT run** and every figure below covers the finished configuration only. Reason, measured not estimated: the streaming pass took 390 s and the finished configuration's 10 models took 3348 s more (`results/lanl/results.json`, `elapsed_seconds` minus `population.scan_seconds`), because the discounted system costs about 1.7x the undiscounted one per model on this data. A second configuration needs the same again, which does not fit this Part's 90-minute budget. Seeds were NOT reduced and the day range was NOT shrunk to make it fit; the configuration is simply missing.

Every number below is generated from a file under `results/lanl/` by `experiments/lanl/run_lanl.py --readme-only`. None is typed by hand. Figures that were not produced say `not measured` and never 0 or blank.

## 0. Readings chosen where the brief is silent

`.claude/commands/part30.md` is authoritative wherever it speaks. It is silent on the following, so each was decided BEFORE any model ran and committed in `configs/data/lanl.yaml` (commit 1 of this branch).

| ambiguity | what part30.md says | reading taken |
|---|---|---|
| day boundary | nothing | `day = floor((t-1)/86400)+1`, so day 1 is t in [1, 86400] |
| window length | "window from config, default = the window chosen for AIT" | **300 s**, the Part 27 review-gate choice carried by `configs/data/ait.yaml` and `experiments/ait_clean/config.yaml` (`chose_at: part27 review gate`). Not a Part 30 invention. |
| which computer owns an auth/flow record | nothing | the **source** computer (`attributed_computer: source`); destination columns still feed distinct-value features |
| what "a red-team event falls inside it" means for a (computer, window) sample | "Labels from redteam.txt" | **attack = 1 if an event in that window names that computer as source or destination**. The window-only alternative is counted and reported in section 3 rather than discarded. |
| features | nothing | the simplest count-based per-window aggregates, listed in `configs/data/lanl.yaml` `features:` before any run |
| train/test split fractions and gap length | "Split by TIME with a gap" | the pre-registered seconds in section 1 |
| sample universe | "Sample = (computer, time window)" | kept, but the cell count forced a declared case-control sample - section 2 |

Two further readings are forced by this Part's own standing rules rather than by part30.md:

- **No validation split and no early stopping.** The instruction was to carve validation from inside the train block *by the same mechanism Part 28 uses, and if Part 28 has none, add none*. Part 28 has none (`experiments/ait_clean/run_ait_clean.py::train_ait`: "There is no validation split and no early stopping"), so none was added.
- **Three Part 28 functions could not be imported.** `fit_scaler`, `apply_scaler` and `train_ait` resolve native view widths through `prismflow.data.ait_lds.FEATURES`, which is keyed by AIT source names: `FEATURES['proc']` and `FEATURES['flows']` do not exist and `FEATURES['auth']`/`FEATURES['dns']` carry AIT's widths, not LANL's. They are mirrored in `run_lanl.py` on LANL widths with Part 28's hyperparameters read from `config.yaml`. This is a reuse limitation of the Part 28 code, reported and **not patched** - `run_ait_clean.py` is untouched. Every metric, by contrast, IS Part 28's own imported function.

## 1. Day-to-seconds mapping and the split

Window grid: `window = floor((t-1)/300)`, so window 0 is t in [1, 300]. Every split boundary below is an exact multiple of 300 s, so **no window straddles a boundary** - asserted in `tests/data/test_lanl_cyber1.py::test_every_split_boundary_lands_on_a_window_edge`.

| block | days | seconds | windows | used for |
|---|---|---|---|---|
| train | 1-10 | 1 .. 864,000 | 0 .. 2,879 | fitting, scaling, training |
| gap | 11-12 | 864,001 .. 1,036,800 | 2,880 .. 3,455 | **dropped, never read into a sample** |
| test | 13-16 | 1,036,801 .. 1,382,400 | 3,456 .. 4,607 | scoring only |
| unused | 17-30 | 1,382,401 .. 2,592,000 | - | not read |

Red-team events inside the gap: **0**. Samples built from the gap: **0**.

Nothing from the test block is used for fitting, scaling, feature selection, thresholds or model selection. The scaler is fitted on train rows only and refitted per view configuration (`results/lanl/results.json`, `configurations.<name>.scaler`).

## 2. Why this is a case-control sample

part30.md fixes the sample as `(computer, time window)`. Measured over days 1-16, per view (`results/lanl/population.json`, `per_view`):

| view | records streamed | occupied (computer, window) cells |
|---|---:|---:|
| auth | 277,999,264 | 22,941,894 |
| proc | 115,159,740 | 23,627,216 |
| flows | 121,284,699 | 13,484,366 |
| dns | 1,055,902 | 580,753 |

The union over views is **not directly counted** (`results/lanl/population.json`, `estimated_population_cells_union`): it is bounded below by 23,627,216 and above by 60,634,229. Either way the sample universe is tens of millions of cells, against 439 train and 410 test attack cells - a population prevalence near 2e-5.

Two consequences, both measurements rather than preferences:

- 40 epochs over ~2e7 samples does not fit this Part's time budget.
- a uniform subsample that *does* fit leaves single-digit attack cells, which trips this Part's own stop condition of >= 10 attack windows per split.

So the sample is a **declared case-control sample**: every attack cell is kept, and a benign cell is kept iff `crc32(computer|window) % 512 == 0`. That is a deterministic hash of the cell identity - not a draw from a generator - so it is identical on every rerun and on every experiment seed, needs no prior knowledge of the cell universe, and reads neither a label nor a feature value.

**What it costs.** Sample prevalence is an artefact of the divisor and is NOT population prevalence. Accuracy, Brier and ECE therefore describe this sample, not the network, and are marked `(sample prevalence)` wherever they appear. Balanced accuracy, ROC AUC and per-class recall are the figures that survive the reweighting.

## 3. Attack-window counts per split

| block | samples | attack samples | benign samples | distinct attack windows | sample prevalence |
|---|---:|---:|---:|---:|---:|
| train | 42,912 | 439 | 42,473 | **150** | 0.01023 |
| test | 18,729 | 410 | 18,319 | **141** | 0.02189 |


**Reconciliation with the label file.** The red-team file names a (computer, window) cell; a sample exists only where some view actually recorded an event for that cell, so the two counts need not agree (`results/lanl/label_audit.json`):

| block | attack cells named by redteam | attack samples built | named but unobserved by every view | distinct computers named |
|---|---:|---:|---:|---:|
| train | 440 | 439 | 1 | 198 |
| test | 410 | 410 | 0 | 139 |

An unobserved cell is a computer the red team touched in a window in which none of auth, proc, flows or dns logged anything for it. It is dropped, not counted as benign - counting it benign would label a known attack as clean.

Under the window-only reading the same blocks would carry 150 (train) and 141 (test) attacked windows, with every computer present in those windows relabelled attack. That reading was not scored.

Pre-registered stop condition: at least 10 distinct attack windows in each of train and test. **Both pass** (150 and 141). The day range was not widened and no count was repaired (`results/lanl/split.json`).

### Caveat (i): red-team events are concentrated

The label is not spread evenly across either block. Per day (`results/lanl/split.json`, `per_split.<block>.per_day`):

| block | day | samples | attack samples | attack windows |
|---|---:|---:|---:|---:|
| train | 1 | 3,491 | 0 | 0 |
| train | 2 | 4,210 | 18 | 8 |
| train | 3 | 4,061 | 22 | 11 |
| train | 4 | 3,665 | 0 | 0 |
| train | 5 | 3,660 | 0 | 0 |
| train | 6 | 4,654 | 36 | 18 |
| train | 7 | 4,928 | 6 | 2 |
| train | 8 | 4,749 | 2 | 1 |
| train | 9 | 5,145 | 351 | 108 |
| train | 10 | 4,349 | 4 | 2 |
| test | 13 | 4,805 | 222 | 64 |
| test | 14 | 4,897 | 109 | 43 |
| test | 15 | 4,736 | 37 | 15 |
| test | 16 | 4,291 | 42 | 19 |

A block's attack evidence is therefore dominated by a few days, and any per-seed spread below says nothing about variation across attack episodes. Red-team events per day across the whole file, including the days this Part does not use, are in `results/lanl/population.json` (`redteam_events_per_day`).

The window-only label reading (section 0) was **not scored**: only the primary (computer, window) reading was trained and evaluated. The alternative is defined in `configs/data/lanl.yaml` `label.also_report` so the choice is on record, but no metric here rests on it.

### Caveat (ii): the split is fixed, so the std is training randomness only

The split is fixed by time and the case-control sample is a deterministic hash, so a seed CANNOT redraw or re-partition the data. A seed selects exactly two things: parameter initialisation (`set_seed`) and minibatch order (`iter_batches(shuffle=True, seed=seed+epoch)`). **Every +/- below is initialisation-and-batch-order variance on one fixed split.** It is not dataset variance and it is not a spread over independent attacks.

### Caveat (iii): the auth view is label-adjacent

LANL's red-team file records authentication events, so the auth view contains the very records the labels are derived from. Any auth-view result is therefore label-adjacent and cannot be read as detection from independent evidence. That is why part30.md step 2 requires both configurations, and why the WITHOUT-auth column is the one to read for a detection claim.

### Caveat (iv): "benign" means "no known red-team event"

A sample is (computer, 300 s window). attack = 1 means a redteam.txt.gz event in that window names that computer as its source or its destination. benign = 0 means NO KNOWN red-team event does, which is not the same as verified clean: the red-team file covers only the activity LANL's red team recorded, so an unlabelled compromise inside the capture is counted as benign here.

Unlike AIT, cyber1 ships no label-coverage mask, so there is no `label_known` filter to apply and no window can be marked unexamined. An unlabelled compromise inside the capture is counted as benign here, which inflates the benign class and can only depress measured precision-like quantities.

## 4. Part 28's metrics, repeated on LANL

Identical settings to Part 28, read from `experiments/lanl/config.yaml`: 40 epochs, batch 64, lr 0.001, anneal 10, hidden [64, 64], feature_dim 32. Seeds [0, 1, 2, 3, 4]. Not re-tuned for LANL.

Every row is `prismflow.evaluation.protocol.compute_metrics` plus Part 28's `part28_extras`, called through Part 28's imported `metric_row`, and cross-checked against `prismflow.evaluation.evaluate`'s own per-seed rows (`run_lanl.py::cross_check`; a mismatch above 1e-9 raises).

### WITH auth (auth, proc, flows, dns)

Test set: n = 18,729, 410 attack, 18,319 benign, sample prevalence 0.02189 (case-control; see section 2).

| metric | naive fusion | V1 PrismFlow | majority baseline | paired PrismFlow - naive |
|---|---|---|---|---|
| balanced accuracy | 0.5585 +/- 0.0081 | 0.5581 +/- 0.0098 | 0.5000 | -0.0004 +/- 0.0098 (2/5 seeds favour PrismFlow) |
| ROC AUC | 0.8308 +/- 0.0039 | 0.8378 +/- 0.0036 | 0.5000 | +0.0069 +/- 0.0040 (5/5 seeds favour PrismFlow) |
| attack recall | 0.1176 +/- 0.0163 | 0.1166 +/- 0.0196 | 0.0000 | -0.0010 +/- 0.0196 (2/5 seeds favour PrismFlow) |
| benign recall | 0.9995 +/- 0.0001 | 0.9996 +/- 0.0002 | 1.0000 | +0.0001 +/- 0.0001 (4/5 seeds favour PrismFlow) |
| accuracy *(sample prevalence)* | 0.9802 +/- 0.0003 | 0.9802 +/- 0.0004 | 0.9781 | +0.0001 +/- 0.0004 (2/5 seeds favour PrismFlow) |
| Brier *(sample prevalence)* | 0.0369 +/- 0.0005 | 0.0369 +/- 0.0003 | 0.0428 | +0.0000 +/- 0.0003 (3/5 seeds favour PrismFlow) |
| ECE (prob) *(sample prevalence)* | 0.0085 +/- 0.0002 | 0.0072 +/- 0.0007 | 0.0000 | -0.0013 +/- 0.0006 (5/5 seeds favour PrismFlow) |
| ENIV (eigen) | 3.4939 +/- 0.0556 | 3.5337 +/- 0.0637 | not measured | +0.0397 +/- 0.1102 (3/5 seeds favour PrismFlow) |

Seeds reported: naive 5, PrismFlow 5, of 5 configured. Mean +/- std over 5 seeds, so these are findings under the 5-seed rule - subject to every caveat in section 3.

Majority-class baseline predicts the test set's majority class for every window with a constant probability vector, so its accuracy IS the prevalence, its balanced accuracy is 0.5 and its ROC AUC is 0.5 - the number an accuracy figure must beat before it describes anything but class balance (`experiments/ait_clean/run_ait_clean.py::majority_baseline`).

Per-view missing fraction (`results/lanl/results.json`, `configurations.with_auth.missingness`):

| view | train | test |
|---|---:|---:|
| auth | 0.3464 | 0.3201 |
| proc | 0.3328 | 0.3071 |
| flows | 0.5640 | 0.7183 |
| dns | 0.9839 | 0.9814 |

Views missing in more than 50% of train windows: flows, dns.

**Not run, and it is in the brief.** Part 28's step 5 says that when a view is missing in more than half the windows, a second configuration WITHOUT that view should also be run. That applies here to flows, dns and was **not run** - the time budget did not reach it (see the status line). So this Part does not know whether the result above survives dropping flows or dns. `dns` in particular is present in under 2% of windows, so its encoder sees almost nothing and its contribution to the fusion is close to untested here.

## 5. Part 29 test A - ENIV on real attack vs real benign windows

ENIV and mean pairwise dependence on real attack vs real benign windows of the test block. No synthetic perturbation. LANL's red-team file has no attack-step taxonomy, so unlike Part 29 on AIT there is no dnsteal-only/multi-step split to report; this is one attack population, not two pooled.

### WITH auth (auth, proc, flows, dns)

n = 410 attack windows, 18,319 benign windows, **never pooled**.

| system | quantity | attack windows | benign windows | paired attack - benign |
|---|---|---|---|---|
| naive | ENIV (eigen) | 3.2911 +/- 0.3673 | 3.5261 +/- 0.0607 | -0.2350 +/- 0.3985 (1/5 seeds favour PrismFlow) |
| naive | mean pairwise dependence | 0.1691 +/- 0.1575 | 0.1365 +/- 0.0138 | +0.0326 +/- 0.1694 (3/5 seeds favour PrismFlow) |
| prismflow | ENIV (eigen) | 3.3241 +/- 0.1489 | 3.5613 +/- 0.0610 | -0.2372 +/- 0.1565 (0/5 seeds favour PrismFlow) |
| prismflow | mean pairwise dependence | 0.1381 +/- 0.1078 | 0.1269 +/- 0.0228 | +0.0111 +/- 0.1158 (2/5 seeds favour PrismFlow) |

Read the WITHOUT-auth block for a detection claim: with auth present, the views contain the records the labels came from (caveat iii).

Sign convention: `paired_difference` marks a *fall* in mean dependence as favouring PrismFlow, because `LOWER_IS_BETTER` contains `mean_dependence_eigen`. For test A the quantity of interest is whether dependence is HIGHER on attack windows than benign ones, so read the signed mean, not the favouring count, in that row.

## 6. View-pair dependence (Part 29 test C, first half)

The strict same-sensor pair does not exist here, as it did not on AIT - but for a different reason. AIT's `sensor_groups()` returns a single-member group (`results/ait_attack/results.json`, `sensor_group_check.verdict`: "NO EVIDENCE"); LANL cyber1 ships **no sensor-group definition at all**, so there is no within-group pair to score. Verdict: **NO EVIDENCE** for strict same-sensor pairs. Nothing was substituted under that name.

Below is a DIFFERENT and separately labelled quantity - **different sensors, naturally dependent** - the measured pair dependence on clean test data, from `results/lanl/results.json`, `configurations.<name>.systems.<system>.pair_table`.

### WITH auth (auth, proc, flows, dns)

| pair | naive fusion | V1 PrismFlow |
|---|---|---|
| auth-dns | 0.2603 +/- 0.0688 | 0.2464 +/- 0.0791 |
| auth-proc | 0.2215 +/- 0.0208 | 0.2252 +/- 0.0122 |
| auth-flows | 0.2111 +/- 0.0215 | 0.1472 +/- 0.0256 |
| proc-dns | 0.1263 +/- 0.0415 | 0.0912 +/- 0.0603 |
| proc-flows | 0.1101 +/- 0.0148 | 0.1016 +/- 0.0109 |
| flows-dns | 0.0375 +/- 0.0525 | 0.0469 +/- 0.0479 |

Most dependent pair: **auth-dns** (naive), **auth-dns** (PrismFlow). Least dependent: **flows-dns** (naive), **flows-dns** (PrismFlow). The two systems agree on both ends.

Scoring the suspicion detector on the most and least dependent of these pairs was **not run** (see section 8), so this Part states no verdict on whether natural dependence raises the detector the way Part 10's exact clone did.

## 7. Cross-dataset table (part30.md step 5)

synthetic vs AIT vs LANL, for the DETECTION claim and the honest-duplication false positive. Every cell cites a `results/` file or says `not measured`. Nothing is pooled across datasets and nothing is averaged across AIT's two directions.

### 7.1 DETECTION: does dependence among the implicated views rise?

| dataset | condition | mean dependence | ENIV | source |
|---|---|---|---|---|
| synthetic | clean | 0.1849 +/- 0.0361 (`dependence_all_pairs`) | 3.3281 +/- 0.0750 | `results/chorus/attack_metrics.json`, `summary.clean|prismflow` |
| synthetic | chorus k=2, eps 1.0 | 0.6970 +/- 0.0380 (`dependence_compromised`) | 2.9048 +/- 0.0608 | `results/chorus/attack_metrics.json`, `summary.chorus_k2_eps1.0|prismflow` |
| AIT (A) | attack (dnsteal-only) | 0.2906 +/- 0.0274 | 3.2868 +/- 0.1345 | `results/ait_attack/results.json`, `directions.A_rm_to_santos.systems.prismflow.natural_windows.dnsteal_only.summary` |
| AIT (A) | benign | 0.3273 +/- 0.0124 | 3.1749 +/- 0.0749 | `results/ait_attack/results.json`, `directions.A_rm_to_santos.systems.prismflow.natural_windows.benign.summary` |
| AIT (B) | attack (dnsteal-only) | 0.3619 +/- 0.0196 | 2.8917 +/- 0.0823 | `results/ait_attack/results.json`, `directions.B_santos_to_rm.systems.prismflow.natural_windows.dnsteal_only.summary` |
| AIT (B) | benign | 0.2482 +/- 0.0249 | 3.1458 +/- 0.0856 | `results/ait_attack/results.json`, `directions.B_santos_to_rm.systems.prismflow.natural_windows.benign.summary` |
| LANL (with_auth) | attack | 0.1381 +/- 0.1078 | 3.3241 +/- 0.1489 | `results/lanl/results.json`, `configurations.with_auth.part29_test_a_natural_attack_windows.per_system.prismflow.populations.attack_windows.summary` |
| LANL (with_auth) | benign | 0.1269 +/- 0.0228 | 3.5613 +/- 0.0610 | `results/lanl/results.json`, `configurations.with_auth.part29_test_a_natural_attack_windows.per_system.prismflow.populations.benign_windows.summary` |

AIT's attack population is dnsteal-only, which Part 29 forbids pooling with its multi-step windows; LANL's red-team file has no step taxonomy, so LANL's single attack population is not the same object as either of AIT's. The rows are side by side, not comparable cell by cell.

### 7.2 Honest-duplication false positive

| dataset | quantity | value | source |
|---|---|---|---|
| synthetic | exact duplication, k=0: mean dependence, all pairs / clone pairs | 0.0537 +/- 0.0251 / not measured (no clone pairs exist at k=0) | `results/clone/summary.json`, `summary[]` where k=0, rho=0, system=prismflow |
| synthetic | exact duplication, k=2: mean dependence, all pairs / clone pairs | 0.2037 +/- 0.0092 / 0.9175 +/- 0.0159 | `results/clone/summary.json`, `summary[]` where k=2, rho=0, system=prismflow |
| synthetic | Part 10 clone_k2 detector AUC | not measured in this Part | not re-run here; Part 10's own results file holds it |
| AIT | strict same-sensor pair | NO EVIDENCE | `results/ait_attack/results.json`, `sensor_group_check.verdict` |
| LANL | strict same-sensor pair | NO EVIDENCE - cyber1 ships no sensor-group definition, so no within-group pair exists | `configs/data/lanl.yaml`; section 6 above |
| LANL | detector AUC on naturally dependent pairs | not measured | detector not scored (section 8) |

## 8. What was not run, and why

part30.md step 4 asks for Part 28's metrics **and Part 29's tests A-D**. Part 28's metrics and test A ran. B, C and D did not, and no partial substitute is reported in their place:

- **B_chorus_attack** - NOT RUN. Part 29's grid is k in {1,2,3} x Part 09's epsilon and beta grid, retrained per cell, which does not fit this Part's time budget. No partial grid is reported, because a subset of the cells is not the test.
- **C_honest_duplication** - PARTIALLY RUN. The strict same-sensor pair does not exist here: LANL cyber1 ships no sensor-group definition, so there is no within-group pair to score and the answer is NO EVIDENCE, exactly as on AIT but for a different reason. The 'different sensors, naturally dependent' half needs the most/least dependent pairs, which ARE written to systems.<name>.pair_table above; scoring the detector on them was not run.
- **D_missingness** - NOT RUN as a detector-AUC test. The per-view missing fractions on train and test ARE measured above under 'missingness'; the 30%/50% masked detector AUC of Part 10 was not run.

These are **INCOMPLETE**, not negative results. Nothing below should be read as evidence about chorus robustness, the suspicion detector or masked-view behaviour on LANL.

## 9. What this design cannot show

- **No population-level operating point.** The case-control sample fixes prevalence by construction, so no threshold, alert rate or precision figure here transfers to the real network.
- **No spread over attack episodes.** One fixed split, and the +/- is initialisation-and-batch-order variance (caveat ii). The red-team activity is concentrated in a few days (caveat i), so even the test block is close to a handful of episodes.
- **No independence from the label source in the WITH-auth arm.** Caveat iii.
- **No claim that benign windows are clean.** Caveat iv.
- **No cross-dataset significance.** Three datasets, one split each, different label semantics; section 7 is a side-by-side table, not a test.
- **No detection-vs-defence separation on LANL.** That separation is Part 09's and rests on the chorus grid, which was not run here.

## 10. Proposed limitations (NOT applied)

`docs/KNOWN_LIMITATIONS.md` and `docs/CONTRACT.md` are unchanged by this Part. The following text is proposed for the user to approve or reject; it has not been written to either file.

> **L23 (proposed).** On LANL cyber1 the `(computer, window)` sample space is ~2e7 cells at 300 s over days 1-16 against ~850 red-team cells, a prevalence near 2e-5. Part 30 therefore reports a declared case-control sample (all cases, deterministic 1/512 hash of controls), so its accuracy, Brier and ECE describe that sample and not the network. No population operating point has been measured on LANL.

> **L24 (proposed).** LANL's red-team labels are derived from authentication records, so the `auth` view is label-adjacent. Part 30 reports WITH-auth and WITHOUT-auth separately; only the WITHOUT-auth arm bears on detection from independent evidence.

> **L25 (proposed).** Part 28's `fit_scaler`, `apply_scaler` and `train_ait` are not reusable across datasets: each resolves native view widths through `ait_lds.FEATURES`, keyed by AIT source names. Part 30 mirrored them rather than patching Part 28. Any future Part on a third dataset will hit the same wall.

## 11. Files

| path | what it holds |
|---|---|
| `configs/data/lanl.yaml` | pre-registered day, window, split, features, label rule, control divisor, stop conditions |
| `prismflow/data/lanl_cyber1.py` | streaming read-only loader, `LANLMultiViewDataset` |
| `experiments/lanl/config.yaml` | Part 28's training and evaluation settings, unchanged |
| `experiments/lanl/run_lanl.py` | runner and this README's generator |
| `tests/data/test_lanl_cyber1.py` | loader tests |
| `results/lanl/population.json` | records streamed and occupied cells per view, per day |
| `results/lanl/split.json` | per-split and per-day sample and attack counts, attack blocks |
| `results/lanl/label_audit.json` | both label readings per block, from `redteam.txt.gz` alone |
| `results/lanl/results.json` | every metric, diagnostic, pair table and paired difference |

Run: 3739 s wall, scan 390 s on 4 worker processes. Commit `0086b7981f6b3a12e7f74f92fce768cd2e30885f`.

`data/raw/` was read only. Nothing in it was extracted, renamed, moved or deleted, and `auth.txt.gz` was not modified.
