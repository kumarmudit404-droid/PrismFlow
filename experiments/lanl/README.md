# Part 30 - PrismFlow on LANL cyber1 (days 1-16)

**STATUS: COMPLETE (for what it reports).** Part 28's metric set and Part 29's test A ran on all 4 view configurations (with_auth, without_auth, without_dns, without_flows) over 5 seeds and both systems. Part 29's tests B, D and the second half of C did NOT run (section 9), so this Part is complete for what it reports and INCOMPLETE against part30.md step 4's full A-D list. Read section 9 before citing anything here as a Part 29 repeat.

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

Every row is `prismflow.evaluation.protocol.compute_metrics` plus Part 28's `part28_extras`, called through Part 28's imported `metric_row`.

**Execution mode, disclosed.** Units of work - one `(arm, system, seed)` each - were run as **parallel** processes, 11 at a time, with **1 torch thread per process**. Hyperparameters and seeds 0-4 are unchanged; parallelism is across units only, and no unit shares state with another beyond the read-only sample table.

| torch threads | ms/step (200 real steps, idle machine) | projected naive model |
|---:|---:|---:|
| 1 | 3.94 | 106 s |
| 2 | 4.00 | 107 s |
| 4 | 4.97 | 133 s |
| 12 | 9.25 | 248 s |

200 real training steps of a with_auth naive model on the real train block, 10 warm-up steps discarded. torch's default thread count (12) is 2.35x slower per step than 1 thread: tiny batches of a 4-view MLP spend more time in thread synchronisation than in arithmetic. Run 1 ran at the default and its measured 253 s/naive model matches this benchmark's 248 s projection at 12 threads, which is the check that the benchmark reflects reality.

The benchmark's throughput figure was a PROJECTION that assumed no contention between concurrent units. What the chosen mode actually delivered, measured from the `seconds` field of every unit file in `results/lanl/units/`:

| quantity | projected before the run | measured after it |
|---|---:|---:|
| seconds per unit | 140.2 | 1768.8 |
| throughput vs run 1 sequential | 26.32x | 2.09x |

A unit takes LONGER in this mode than it did sequentially - 1769 s against run 1's 336 s - because 11 of them share the box. The gain is throughput, not latency. The projection above overstated that gain because it assumed no contention between concurrent units; the decision rule (at least 2x run 1's sequential throughput) is met on the MEASURED figure as well as the projected one, so the mode choice stands on what happened rather than on the forecast.

**Reported, not fixed - a frozen-module observation.** A committed Part 28 ENIV value is reproducible only at the torch thread count that produced it. tests/unit/test_ait_attack.py::test_clean_dependence_matches_part28 re-computes eniv_eigen and asserts it equals Part 28's committed seed-0 value to abs=1e-6. REPORTED, NOT FIXED. Frozen-module observation; no frozen code was changed and no committed value was regenerated.

| condition | `eniv_eigen` | against committed | asserted tolerance |
|---|---:|---:|---:|
| torch default (12 threads) | reproduces the committed value | - | 1e-06 |
| `OMP_NUM_THREADS=1` | 3.4289614712 | 3.4396320614 | 1e-06 |

Part 30 run 1 ran at torch's default 12 threads and run 2 at 1 thread. The thread count changes the order in which sums are accumulated during training, so it changes the fitted weights and then every diagnostic computed from them. The difference is about 0.3% of ENIV on AIT - four orders of magnitude above the 1e-6 tolerance that test asserts - which is the scale to keep in mind when reading this Part's run-1-vs-run-2 replicate table and its seed-0 equivalence check. It is arithmetic, not a quality difference between the systems.

The Part 30 gate test suite is therefore run at the DEFAULT thread count, which is the count the committed values were produced at. Under OMP_NUM_THREADS=1 that one test fails; that is the test detecting a thread change, not a defect introduced by Part 30.

A second thread benchmark was taken on 2026-10-18 and is **excluded**. It is kept in `execution_mode.json` so the exclusion is auditable, not hidden: CONTAMINATED, not a clean benchmark. No PrismFlow process was running (all 22 orphans had been terminated), but this box carried 20-33% load from unrelated desktop applications throughout. The readings are non-monotonic in thread count and 12 threads alone spans 11.1-35.7 ms/step across three identical repeats, so this run cannot discriminate 1 from 2 from 4 threads. The one signal it reproduces is that 12 threads (torch's default here) is never faster and is sometimes 3x slower. The thread choice below was therefore taken from the 2026-10-17 idle-machine benchmark, which is the only clean one, and was fixed before this run's units were launched.

**Process audit before this run's measurements (part30.md Phase 0b).** 24 python processes were found. The other 22 processes were 11 live units of the Phase 2 scheduler (phase2.sh, supervisor PID 18016, started 15:22:36) working the brief's own arm priority order. They are NOT orphans: the supervisor shell was alive, the queue was advancing, and a 40-second CPU sample confirmed every one of them accumulating CPU time. Killing them would have discarded in-flight work on the arms the brief owes.

- Terminated: run_lanl.py --aggregate (2 processes, PID 9016 and 19268, started 15:40:22, still running 40 minutes later on a then-incomplete set of arms). Its command line shows run_lanl.py, its parent session was gone, and its output would have been superseded by the final aggregate, so it was terminated. Nothing it had written was kept.

*On the idle-machine requirement:* The brief's Phase 1 asks for a thread benchmark on an IDLE machine. The machine was NOT idle in this session and was not made idle, because idling it meant killing the owed arms mid-flight. The conservative reading taken, recorded here rather than resolved by asking: Phase 1 was already executed and committed before this session - the thread count and the parallel-vs-sequential decision were fixed from the 2026-10-17 idle-machine benchmark and had already been applied to every unit on disk - so repeating it could only have produced a second contaminated reading at the cost of the arms. No new benchmark was run in this session and no benchmark number here comes from a loaded box except the one already labelled CONTAMINATED and excluded.

**Equivalence check (part30.md Phase 1).** Seed 0, `with_auth`, both systems, re-run in the chosen mode and compared with the same seed of run 1, which ran sequentially at torch's default thread count. Run 1's per-seed values are read from `results/lanl/results.run1.json.gz`.

| system | metric | run 1 (seed 0) | run 2 (seed 0) | absolute difference |
|---|---|---:|---:|---:|
| naive | balanced_accuracy | 0.5643 | 0.5582 | 0.0061 |
| naive | roc_auc | 0.8375 | 0.8235 | 0.0140 |
| prismflow | balanced_accuracy | 0.5618 | 0.5716 | 0.0097 |
| prismflow | roc_auc | 0.8413 | 0.8402 | 0.0011 |

Same seed, same hyperparameters, same sample table and the same time split. A non-zero difference here is execution mode and floating-point reduction order, not a different experiment: run 1 ran sequentially at torch's default 12 threads and run 2 at 1 thread, which changes the order in which sums are accumulated. The difference is NOT evidence about either system's quality.

### 4.1 All arms, side by side

Paired within seed: the same 5 seeds, the same rows, the same split. Prevalence-dependent metrics are flagged - the sample is case-control at about 2% attack prevalence against about 2e-5 in the population, so ECE here describes the sample.

| arm | views | system | balanced accuracy | ROC AUC | ECE *(sample prevalence)* | attack recall |
|---|---|---|---|---|---|---|
| without_auth | proc, flows, dns | naive | 0.5260 +/- 0.0006 | 0.7755 +/- 0.0025 | 0.0947 +/- 0.0011 | 0.0522 +/- 0.0013 |
| without_auth | proc, flows, dns | prismflow | 0.5258 +/- 0.0016 | 0.7775 +/- 0.0030 | 0.0943 +/- 0.0014 | 0.0517 +/- 0.0032 |
| without_flows | auth, proc, dns | naive | 0.5552 +/- 0.0049 | 0.8137 +/- 0.0022 | 0.0330 +/- 0.0008 | 0.1107 +/- 0.0098 |
| without_flows | auth, proc, dns | prismflow | 0.5517 +/- 0.0067 | 0.8233 +/- 0.0031 | 0.0320 +/- 0.0007 | 0.1039 +/- 0.0134 |
| without_dns | auth, proc, flows | naive | 0.5617 +/- 0.0093 | 0.8336 +/- 0.0029 | 0.0101 +/- 0.0007 | 0.1239 +/- 0.0185 |
| without_dns | auth, proc, flows | prismflow | 0.5591 +/- 0.0051 | 0.8389 +/- 0.0058 | 0.0093 +/- 0.0004 | 0.1185 +/- 0.0101 |
| with_auth | auth, proc, flows, dns | naive | 0.5587 +/- 0.0088 | 0.8278 +/- 0.0033 | 0.0084 +/- 0.0009 | 0.1180 +/- 0.0176 |
| with_auth | auth, proc, flows, dns | prismflow | 0.5676 +/- 0.0073 | 0.8375 +/- 0.0043 | 0.0076 +/- 0.0011 | 0.1356 +/- 0.0146 |

Paired difference, PrismFlow minus naive, within seed:

| arm | balanced accuracy | ROC AUC | ECE | attack recall |
|---|---|---|---|---|
| without_auth | -0.0002 +/- 0.0016 (2/5 seeds favour PrismFlow) | +0.0019 +/- 0.0025 (4/5 seeds favour PrismFlow) | -0.0004 +/- 0.0006 (3/5 seeds favour PrismFlow) | -0.0005 +/- 0.0032 (1/5 seeds favour PrismFlow) |
| without_flows | -0.0035 +/- 0.0065 (1/5 seeds favour PrismFlow) | +0.0096 +/- 0.0013 (5/5 seeds favour PrismFlow) | -0.0010 +/- 0.0006 (5/5 seeds favour PrismFlow) | -0.0068 +/- 0.0131 (1/5 seeds favour PrismFlow) |
| without_dns | -0.0026 +/- 0.0125 (2/5 seeds favour PrismFlow) | +0.0053 +/- 0.0070 (4/5 seeds favour PrismFlow) | -0.0008 +/- 0.0004 (5/5 seeds favour PrismFlow) | -0.0054 +/- 0.0250 (2/5 seeds favour PrismFlow) |
| with_auth | +0.0088 +/- 0.0046 (5/5 seeds favour PrismFlow) | +0.0097 +/- 0.0063 (5/5 seeds favour PrismFlow) | -0.0008 +/- 0.0016 (4/5 seeds favour PrismFlow) | +0.0176 +/- 0.0095 (5/5 seeds favour PrismFlow) |

A seed count of 5/5 means every seed moved the same way; it does NOT mean the effect is larger than test-set sampling noise. Section 7.4's bootstrap interval is the number to read for that.

Brier against the majority baseline, which predicts the test block's class prevalence for every window. Its Brier is the score of a CONSTANT forecast, so a trained model above it is worse than that constant:

| arm | system | Brier *(sample prevalence)* | constant-prevalence baseline | above the baseline |
|---|---|---:|---:|---|
| without_auth | naive | 0.1218 +/- 0.0006 | 0.0428 | **YES** |
| without_auth | prismflow | 0.1219 +/- 0.0005 | 0.0428 | **YES** |
| without_flows | naive | 0.0628 +/- 0.0004 | 0.0428 | **YES** |
| without_flows | prismflow | 0.0630 +/- 0.0006 | 0.0428 | **YES** |
| without_dns | naive | 0.0386 +/- 0.0007 | 0.0428 | no |
| without_dns | prismflow | 0.0386 +/- 0.0002 | 0.0428 | no |
| with_auth | naive | 0.0370 +/- 0.0005 | 0.0428 | no |
| with_auth | prismflow | 0.0366 +/- 0.0002 | 0.0428 | no |

**Flagged: `without_auth`, `without_flows` are above the constant-prevalence baseline on Brier**, for both systems in each flagged arm. A probability forecast worse than a constant is a calibration failure, not a ranking failure - the same arms' ROC AUC is well above 0.5. **The cause is not diagnosed here.** Nothing in this Part isolates whether it comes from the empty-tensor rows of section 4.2, from the absence of a validation split, or from the case-control prevalence the forecast is trained against.

**Part 28 step 5 drop-view configurations.** Step 5 asks for a second configuration without any view missing in more than 50% of train windows. On this sample table that is 2 views - `dns` 98.4%, `flows` 56.4% of train windows missing. Configurations `without_dns`, `without_flows` ran, over all 5 seeds and both systems; they are in the table above. **No configuration dropping ALL of `dns`, `flows` at once was run**, so the arms above each drop one of them and never both; this Part does not know what the remaining views do with both absent.

### 4.2 Samples an arm cannot predict by construction

Shared sample table: all arms use the SAME rows, labels and split. An arm drops a view's feature columns and encoder, never a row. A row with no present view among the arm's views is unpredictable by construction - all models score it identically.

| arm | train: attack with no present view | test: attack with no present view | test: benign with none |
|---|---:|---:|---:|
| without_auth | 49 of 439 | **139 of 410** (33.9%) | 3,461 |
| without_flows | 0 of 439 | **0 of 410** (0.0%) | 998 |
| without_dns | 0 of 439 | **0 of 410** (0.0%) | 73 |
| with_auth | 0 of 439 | **0 of 410** (0.0%) | 0 |

This is a ceiling, not a tuning problem. In **without_auth**, 139 test attack samples have an all-zero feature tensor and an all-False view mask, so every model - and every future model - must give them the same score as each other and as the equally empty benign samples. The arm's attainable attack recall is capped below 1 before training starts, and its AUC carries a block of tied scores. Read that arm's numbers as a floor on what the remaining views can do, not as a measurement of what the views cannot.

### WITHOUT auth (proc, flows, dns)

Test set: n = 18,729, 410 attack, 18,319 benign, sample prevalence 0.02189 (case-control; see section 2).

| metric | naive fusion | V1 PrismFlow | majority baseline | paired PrismFlow - naive |
|---|---|---|---|---|
| balanced accuracy | 0.5260 +/- 0.0006 | 0.5258 +/- 0.0016 | 0.5000 | -0.0002 +/- 0.0016 (2/5 seeds favour PrismFlow) |
| ROC AUC | 0.7755 +/- 0.0025 | 0.7775 +/- 0.0030 | 0.5000 | +0.0019 +/- 0.0025 (4/5 seeds favour PrismFlow) |
| attack recall | 0.0522 +/- 0.0013 | 0.0517 +/- 0.0032 | 0.0000 | -0.0005 +/- 0.0032 (1/5 seeds favour PrismFlow) |
| benign recall | 0.9999 +/- 0.0001 | 0.9999 +/- 0.0000 | 1.0000 | +0.0000 +/- 0.0001 (2/5 seeds favour PrismFlow) |
| accuracy *(sample prevalence)* | 0.9791 +/- 0.0000 | 0.9792 +/- 0.0001 | 0.9781 | +0.0000 +/- 0.0001 (3/5 seeds favour PrismFlow) |
| Brier *(sample prevalence)* | 0.1218 +/- 0.0006 | 0.1219 +/- 0.0005 | 0.0428 | +0.0001 +/- 0.0002 (1/5 seeds favour PrismFlow) |
| ECE (prob) *(sample prevalence)* | 0.0947 +/- 0.0011 | 0.0943 +/- 0.0014 | 0.0000 | -0.0004 +/- 0.0006 (3/5 seeds favour PrismFlow) |
| ENIV (eigen) | 2.8570 +/- 0.0386 | 2.8666 +/- 0.0485 | not measured | +0.0096 +/- 0.0485 (2/5 seeds favour PrismFlow) |

Seeds reported: naive 5, PrismFlow 5, of 5 configured. Mean +/- std over 5 seeds, so these are findings under the 5-seed rule - subject to every caveat in section 3.

Majority-class baseline predicts the test set's majority class for every window with a constant probability vector, so its accuracy IS the prevalence, its balanced accuracy is 0.5 and its ROC AUC is 0.5 - the number an accuracy figure must beat before it describes anything but class balance (`experiments/ait_clean/run_ait_clean.py::majority_baseline`).

Per-view missing fraction (`results/lanl/results.json`, `configurations.without_auth.missingness`):

| view | train | test |
|---|---:|---:|
| proc | 0.3328 | 0.3071 |
| flows | 0.5640 | 0.7183 |
| dns | 0.9839 | 0.9814 |

Views missing in more than 50% of train windows: flows, dns.

Part 28's step 5 drop-view configurations for `flows`, `dns` are stated once in section 4, above, rather than repeated per arm.

### WITHOUT flows (auth, proc, dns)

Test set: n = 18,729, 410 attack, 18,319 benign, sample prevalence 0.02189 (case-control; see section 2).

| metric | naive fusion | V1 PrismFlow | majority baseline | paired PrismFlow - naive |
|---|---|---|---|---|
| balanced accuracy | 0.5552 +/- 0.0049 | 0.5517 +/- 0.0067 | 0.5000 | -0.0035 +/- 0.0065 (1/5 seeds favour PrismFlow) |
| ROC AUC | 0.8137 +/- 0.0022 | 0.8233 +/- 0.0031 | 0.5000 | +0.0096 +/- 0.0013 (5/5 seeds favour PrismFlow) |
| attack recall | 0.1107 +/- 0.0098 | 0.1039 +/- 0.0134 | 0.0000 | -0.0068 +/- 0.0131 (1/5 seeds favour PrismFlow) |
| benign recall | 0.9996 +/- 0.0001 | 0.9995 +/- 0.0001 | 1.0000 | -0.0001 +/- 0.0001 (1/5 seeds favour PrismFlow) |
| accuracy *(sample prevalence)* | 0.9801 +/- 0.0002 | 0.9799 +/- 0.0003 | 0.9781 | -0.0002 +/- 0.0002 (0/5 seeds favour PrismFlow) |
| Brier *(sample prevalence)* | 0.0628 +/- 0.0004 | 0.0630 +/- 0.0006 | 0.0428 | +0.0001 +/- 0.0005 (2/5 seeds favour PrismFlow) |
| ECE (prob) *(sample prevalence)* | 0.0330 +/- 0.0008 | 0.0320 +/- 0.0007 | 0.0000 | -0.0010 +/- 0.0006 (5/5 seeds favour PrismFlow) |
| ENIV (eigen) | 2.5997 +/- 0.0631 | 2.6206 +/- 0.0770 | not measured | +0.0209 +/- 0.1034 (3/5 seeds favour PrismFlow) |

Seeds reported: naive 5, PrismFlow 5, of 5 configured. Mean +/- std over 5 seeds, so these are findings under the 5-seed rule - subject to every caveat in section 3.

Majority-class baseline predicts the test set's majority class for every window with a constant probability vector, so its accuracy IS the prevalence, its balanced accuracy is 0.5 and its ROC AUC is 0.5 - the number an accuracy figure must beat before it describes anything but class balance (`experiments/ait_clean/run_ait_clean.py::majority_baseline`).

Per-view missing fraction (`results/lanl/results.json`, `configurations.without_flows.missingness`):

| view | train | test |
|---|---:|---:|
| auth | 0.3464 | 0.3201 |
| proc | 0.3328 | 0.3071 |
| dns | 0.9839 | 0.9814 |

Views missing in more than 50% of train windows: dns.

Part 28's step 5 drop-view configurations for `dns` are stated once in section 4, above, rather than repeated per arm.

### WITHOUT dns (auth, proc, flows)

Test set: n = 18,729, 410 attack, 18,319 benign, sample prevalence 0.02189 (case-control; see section 2).

| metric | naive fusion | V1 PrismFlow | majority baseline | paired PrismFlow - naive |
|---|---|---|---|---|
| balanced accuracy | 0.5617 +/- 0.0093 | 0.5591 +/- 0.0051 | 0.5000 | -0.0026 +/- 0.0125 (2/5 seeds favour PrismFlow) |
| ROC AUC | 0.8336 +/- 0.0029 | 0.8389 +/- 0.0058 | 0.5000 | +0.0053 +/- 0.0070 (4/5 seeds favour PrismFlow) |
| attack recall | 0.1239 +/- 0.0185 | 0.1185 +/- 0.0101 | 0.0000 | -0.0054 +/- 0.0250 (2/5 seeds favour PrismFlow) |
| benign recall | 0.9995 +/- 0.0003 | 0.9996 +/- 0.0001 | 1.0000 | +0.0001 +/- 0.0003 (2/5 seeds favour PrismFlow) |
| accuracy *(sample prevalence)* | 0.9803 +/- 0.0006 | 0.9803 +/- 0.0002 | 0.9781 | +0.0000 +/- 0.0006 (3/5 seeds favour PrismFlow) |
| Brier *(sample prevalence)* | 0.0386 +/- 0.0007 | 0.0386 +/- 0.0002 | 0.0428 | +0.0000 +/- 0.0008 (4/5 seeds favour PrismFlow) |
| ECE (prob) *(sample prevalence)* | 0.0101 +/- 0.0007 | 0.0093 +/- 0.0004 | 0.0000 | -0.0008 +/- 0.0004 (5/5 seeds favour PrismFlow) |
| ENIV (eigen) | 2.6509 +/- 0.0162 | 2.6824 +/- 0.0301 | not measured | +0.0315 +/- 0.0448 (4/5 seeds favour PrismFlow) |

Seeds reported: naive 5, PrismFlow 5, of 5 configured. Mean +/- std over 5 seeds, so these are findings under the 5-seed rule - subject to every caveat in section 3.

Majority-class baseline predicts the test set's majority class for every window with a constant probability vector, so its accuracy IS the prevalence, its balanced accuracy is 0.5 and its ROC AUC is 0.5 - the number an accuracy figure must beat before it describes anything but class balance (`experiments/ait_clean/run_ait_clean.py::majority_baseline`).

Per-view missing fraction (`results/lanl/results.json`, `configurations.without_dns.missingness`):

| view | train | test |
|---|---:|---:|
| auth | 0.3464 | 0.3201 |
| proc | 0.3328 | 0.3071 |
| flows | 0.5640 | 0.7183 |

Views missing in more than 50% of train windows: flows.

Part 28's step 5 drop-view configurations for `flows` are stated once in section 4, above, rather than repeated per arm.

### WITH auth (auth, proc, flows, dns)

Test set: n = 18,729, 410 attack, 18,319 benign, sample prevalence 0.02189 (case-control; see section 2).

| metric | naive fusion | V1 PrismFlow | majority baseline | paired PrismFlow - naive |
|---|---|---|---|---|
| balanced accuracy | 0.5587 +/- 0.0088 | 0.5676 +/- 0.0073 | 0.5000 | +0.0088 +/- 0.0046 (5/5 seeds favour PrismFlow) |
| ROC AUC | 0.8278 +/- 0.0033 | 0.8375 +/- 0.0043 | 0.5000 | +0.0097 +/- 0.0063 (5/5 seeds favour PrismFlow) |
| attack recall | 0.1180 +/- 0.0176 | 0.1356 +/- 0.0146 | 0.0000 | +0.0176 +/- 0.0095 (5/5 seeds favour PrismFlow) |
| benign recall | 0.9994 +/- 0.0002 | 0.9995 +/- 0.0002 | 1.0000 | +0.0001 +/- 0.0002 (2/5 seeds favour PrismFlow) |
| accuracy *(sample prevalence)* | 0.9801 +/- 0.0003 | 0.9806 +/- 0.0003 | 0.9781 | +0.0005 +/- 0.0001 (5/5 seeds favour PrismFlow) |
| Brier *(sample prevalence)* | 0.0370 +/- 0.0005 | 0.0366 +/- 0.0002 | 0.0428 | -0.0004 +/- 0.0003 (5/5 seeds favour PrismFlow) |
| ECE (prob) *(sample prevalence)* | 0.0084 +/- 0.0009 | 0.0076 +/- 0.0011 | 0.0000 | -0.0008 +/- 0.0016 (4/5 seeds favour PrismFlow) |
| ENIV (eigen) | 3.5185 +/- 0.0718 | 3.5627 +/- 0.0429 | not measured | +0.0442 +/- 0.1069 (3/5 seeds favour PrismFlow) |

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

Part 28's step 5 drop-view configurations for `flows`, `dns` are stated once in section 4, above, rather than repeated per arm.

### 4.3 Run 1 as a replicate of with_auth

Same seeds, same hyperparameters, same sample rule; run 1 ran sequentially at torch's default thread count, run 2 in this Part's chosen mode. Differences are execution mode and floating-point reduction order, not a different experiment.

Run 1's record is `results/lanl/results.run1.json.gz`; its sha256 is in `results/lanl/results_manifest.json` and was **verified** on reading. Run 1 was commit `0086b7981f6b3a12e7f74f92fce768cd2e30885f`.

| metric | system | run 1 | run 2 | run 2 - run 1 |
|---|---|---:|---:|---:|
| balanced_accuracy | naive | 0.5585 | 0.5587 | +0.0002 |
| balanced_accuracy | prismflow | 0.5581 | 0.5676 | +0.0095 |
| roc_auc | naive | 0.8308 | 0.8278 | -0.0031 |
| roc_auc | prismflow | 0.8378 | 0.8375 | -0.0003 |
| prob_ece | naive | 0.0085 | 0.0084 | -0.0001 |
| prob_ece | prismflow | 0.0072 | 0.0076 | +0.0004 |
| attack_recall | naive | 0.1176 | 0.1180 | +0.0005 |
| attack_recall | prismflow | 0.1166 | 0.1356 | +0.0190 |

PrismFlow minus naive, paired within seed, in each run separately:

| metric | run 1 paired difference | run 2 paired difference | sign differs between runs |
|---|---:|---:|---|
| balanced_accuracy | -0.0004 +/- 0.0098 (2/5) | +0.0088 +/- 0.0046 (5/5) | **YES** |
| roc_auc | +0.0069 +/- 0.0040 (5/5) | +0.0097 +/- 0.0063 (5/5) | no |
| prob_ece | -0.0013 +/- 0.0006 (5/5) | -0.0008 +/- 0.0016 (4/5) | no |
| attack_recall | -0.0010 +/- 0.0196 (2/5) | +0.0176 +/- 0.0095 (5/5) | **YES** |
| brier | +0.0000 +/- 0.0003 (3/5) | -0.0004 +/- 0.0003 (5/5) | **YES** |
| benign_recall | +0.0001 +/- 0.0001 (4/5) | +0.0001 +/- 0.0002 (2/5) | no |
| accuracy | +0.0001 +/- 0.0004 (2/5) | +0.0005 +/- 0.0001 (5/5) | no |

Metrics whose paired sign differs between the two runs: `balanced_accuracy`, `attack_recall`, `brier`. PrismFlow minus naive, paired within seed, in each run separately. A seed count inside ONE run shares that run's execution mode for both systems, so it is evidence about the two systems under that mode and NOT independent evidence about mode effects. Only the two columns side by side speak to the mode, and with one run each they do so without a spread.

These two runs are **replicates, not reproductions**. Same seeds and same hyperparameters, but a different torch thread count changes the order of floating-point reductions, so the trained weights differ and a seed does not reproduce bit-exactly across modes. Where a run-1-to-run-2 difference is comparable to or larger than that metric's per-seed std, the seed std is understating the spread an independent re-run would show.

## 5. Part 29 test A - ENIV on real attack vs real benign windows

ENIV and mean pairwise dependence on real attack vs real benign test windows. No synthetic perturbation. LANL's red-team file has no attack-step taxonomy, so unlike Part 29 on AIT there is no dnsteal-only/multi-step split; this is one attack population, not two pooled.

### WITHOUT auth (proc, flows, dns)

n = 410 attack windows, 18,319 benign windows, **never pooled**.

| system | quantity | attack windows | benign windows | paired attack - benign |
|---|---|---|---|---|
| naive | ENIV (eigen) | 2.7369 +/- 0.1115 | 2.8649 +/- 0.0460 | -0.1279 +/- 0.1202 (1/5 seeds > 0) |
| naive | mean pairwise dependence | 0.1315 +/- 0.0558 | 0.0527 +/- 0.0184 | +0.0789 +/- 0.0626 (4/5 seeds > 0) |
| prismflow | ENIV (eigen) | 2.7649 +/- 0.1341 | 2.8752 +/- 0.0454 | -0.1103 +/- 0.1224 (1/5 seeds > 0) |
| prismflow | mean pairwise dependence | 0.1175 +/- 0.0671 | 0.0573 +/- 0.0201 | +0.0602 +/- 0.0601 (4/5 seeds > 0) |

### WITHOUT flows (auth, proc, dns)

n = 410 attack windows, 18,319 benign windows, **never pooled**.

| system | quantity | attack windows | benign windows | paired attack - benign |
|---|---|---|---|---|
| naive | ENIV (eigen) | 2.8562 +/- 0.0927 | 2.5926 +/- 0.0584 | +0.2636 +/- 0.0923 (5/5 seeds > 0) |
| naive | mean pairwise dependence | 0.0717 +/- 0.0461 | 0.1956 +/- 0.0277 | -0.1238 +/- 0.0469 (0/5 seeds > 0) |
| prismflow | ENIV (eigen) | 2.8216 +/- 0.1872 | 2.6100 +/- 0.0845 | +0.2116 +/- 0.1611 (5/5 seeds > 0) |
| prismflow | mean pairwise dependence | 0.0789 +/- 0.1045 | 0.1882 +/- 0.0419 | -0.1092 +/- 0.0932 (0/5 seeds > 0) |

### WITHOUT dns (auth, proc, flows)

n = 410 attack windows, 18,319 benign windows, **never pooled**.

| system | quantity | attack windows | benign windows | paired attack - benign |
|---|---|---|---|---|
| naive | ENIV (eigen) | 2.5849 +/- 0.0518 | 2.7294 +/- 0.0148 | -0.1444 +/- 0.0558 (0/5 seeds > 0) |
| naive | mean pairwise dependence | 0.1634 +/- 0.0440 | 0.1304 +/- 0.0070 | +0.0330 +/- 0.0441 (5/5 seeds > 0) |
| prismflow | ENIV (eigen) | 2.5993 +/- 0.0790 | 2.7491 +/- 0.0132 | -0.1498 +/- 0.0702 (0/5 seeds > 0) |
| prismflow | mean pairwise dependence | 0.1495 +/- 0.0493 | 0.1189 +/- 0.0074 | +0.0306 +/- 0.0492 (3/5 seeds > 0) |

### WITH auth (auth, proc, flows, dns)

n = 410 attack windows, 18,319 benign windows, **never pooled**.

| system | quantity | attack windows | benign windows | paired attack - benign |
|---|---|---|---|---|
| naive | ENIV (eigen) | 3.2664 +/- 0.2609 | 3.5612 +/- 0.0742 | -0.2948 +/- 0.2598 (1/5 seeds > 0) |
| naive | mean pairwise dependence | 0.1337 +/- 0.1376 | 0.1276 +/- 0.0231 | +0.0061 +/- 0.1374 (2/5 seeds > 0) |
| prismflow | ENIV (eigen) | 3.1816 +/- 0.1844 | 3.5619 +/- 0.0355 | -0.3803 +/- 0.2005 (0/5 seeds > 0) |
| prismflow | mean pairwise dependence | 0.1731 +/- 0.1486 | 0.1258 +/- 0.0148 | +0.0473 +/- 0.1533 (4/5 seeds > 0) |

Read the WITHOUT-auth block for a detection claim: with auth present, the views contain the records the labels came from (caveat iii).

Sign convention: the last column is attack minus benign, and the count beside it is the number of seeds whose own difference is above zero. `paired_difference`'s own favouring count is NOT shown here: it reads a fall in `mean_dependence_eigen` as favouring PrismFlow, because that key is in `LOWER_IS_BETTER`, which answers a different question from the one test A asks.

**Signs of the attack-minus-benign differences, per arm and system.** A `+` means the quantity is higher on attack windows:

| arm | system | ENIV (eigen) | mean pairwise dependence |
|---|---|:--:|:--:|
| without_auth | naive | - | + |
| without_auth | prismflow | - | + |
| without_flows | naive | + | - |
| without_flows | prismflow | + | - |
| without_dns | naive | - | + |
| without_dns | prismflow | - | + |
| with_auth | naive | - | + |
| with_auth | prismflow | - | + |

The most common pattern over the arms is ENIV `-` with dependence `+`, and `without_flows` is opposite to it. An arm that reverses sign while the others agree is not explained by this Part.

Dependence and ENIV are PAIR statistics, so only a test row with at least two of the arm's views present contributes anything to them. How many rows that is, per class:

| arm | attack rows with >= 2 views | share of attack | benign rows with >= 2 views | share of benign |
|---|---:|---:|---:|---:|
| without_auth | 67 of 410 | 16.3% | 3,315 of 18,319 | 18.1% |
| without_flows | 208 of 410 | 50.7% | 8,022 of 18,319 | 43.8% |
| without_dns | 270 of 410 | 65.9% | 9,857 of 18,319 | 53.8% |
| with_auth | 271 of 410 | 66.1% | 9,942 of 18,319 | 54.3% |

A pair statistic needs two present views in the same row, so a row with 0 or 1 present view contributes nothing to dependence or ENIV.

## 6. View-pair dependence (Part 29 test C, first half)

The strict same-sensor pair does not exist here, as it did not on AIT - but for a different reason. AIT's `sensor_groups()` returns a single-member group (`results/ait_attack/results.json`, `sensor_group_check.verdict`: "NO EVIDENCE"); LANL cyber1 ships **no sensor-group definition at all**, so there is no within-group pair to score. Verdict: **NO EVIDENCE** for strict same-sensor pairs. Nothing was substituted under that name.

Below is a DIFFERENT and separately labelled quantity - **different sensors, naturally dependent** - the measured pair dependence on clean test data, from `results/lanl/results.json`, `configurations.<name>.systems.<system>.pair_table`.

### WITHOUT auth (proc, flows, dns)

| pair | naive fusion | V1 PrismFlow |
|---|---|---|
| proc-flows | 0.0905 +/- 0.0086 | 0.0857 +/- 0.0101 |
| proc-dns | 0.0856 +/- 0.0618 | 0.0784 +/- 0.0698 |
| flows-dns | -0.0122 +/- 0.0255 | 0.0188 +/- 0.0122 |

Most dependent pair: **proc-flows** (naive), **proc-flows** (PrismFlow). Least dependent: **flows-dns** (naive), **flows-dns** (PrismFlow). The two systems agree on both ends.

### WITHOUT flows (auth, proc, dns)

| pair | naive fusion | V1 PrismFlow |
|---|---|---|
| auth-dns | 0.2783 +/- 0.0556 | 0.2529 +/- 0.0637 |
| auth-proc | 0.2045 +/- 0.0147 | 0.2039 +/- 0.0152 |
| proc-dns | 0.1032 +/- 0.0470 | 0.1002 +/- 0.0499 |

Most dependent pair: **auth-dns** (naive), **auth-dns** (PrismFlow). Least dependent: **proc-dns** (naive), **proc-dns** (PrismFlow). The two systems agree on both ends.

### WITHOUT dns (auth, proc, flows)

| pair | naive fusion | V1 PrismFlow |
|---|---|---|
| auth-proc | 0.2259 +/- 0.0146 | 0.2206 +/- 0.0133 |
| auth-flows | 0.1713 +/- 0.0316 | 0.1490 +/- 0.0359 |
| proc-flows | 0.1199 +/- 0.0118 | 0.0970 +/- 0.0116 |

Most dependent pair: **auth-proc** (naive), **auth-proc** (PrismFlow). Least dependent: **proc-flows** (naive), **proc-flows** (PrismFlow). The two systems agree on both ends.

### WITH auth (auth, proc, flows, dns)

| pair | naive fusion | V1 PrismFlow |
|---|---|---|
| auth-dns | 0.2537 +/- 0.0712 | 0.2486 +/- 0.0332 |
| auth-proc | 0.2178 +/- 0.0119 | 0.2219 +/- 0.0109 |
| auth-flows | 0.1971 +/- 0.0351 | 0.1429 +/- 0.0164 |
| proc-flows | 0.1026 +/- 0.0130 | 0.0956 +/- 0.0123 |
| proc-dns | 0.0926 +/- 0.0763 | 0.0746 +/- 0.0342 |
| flows-dns | 0.0303 +/- 0.0126 | 0.0343 +/- 0.0344 |

Most dependent pair: **auth-dns** (naive), **auth-dns** (PrismFlow). Least dependent: **flows-dns** (naive), **flows-dns** (PrismFlow). The two systems agree on both ends.

Scoring the suspicion detector on the most and least dependent of these pairs was **not run** (see section 9), so this Part states no verdict on whether natural dependence raises the detector the way Part 10's exact clone did.

## 7. Diagnostics (no new training)

### 7.1 Trivial baselines: what a single scalar already achieves

Each row is a SINGLE SCALAR, no training. Orientation (sign) is chosen on the train block only and then applied unchanged to test.

| scalar | orientation (fixed on train) | train AUC | **test AUC** |
|---|---|---:|---:|
| `total_records_across_views` | as-is | 0.8164 | **0.6843** |
| `auth_present_indicator` | as-is | 0.6750 | **0.6637** |
| `n_views_present` | as-is | 0.6281 | **0.5625** |

Every attack sample in both blocks has an `auth` record, while about a third of benign samples have none: train 439 of 439 attack (100.0%) against 27609 of 42473 benign (65.0%); test 410 of 410 attack (100.0%) against 12323 of 18319 benign (67.3%). Counted from `results/lanl/samples.npz`, locating the `auth` view by name in the stored sources list. That single split is the whole of `auth_present_indicator`, which is why its test AUC (0.6637) is identical in the all-test, seen and unseen groups of section 7.7: the scalar is constant across the attack class, so subsetting the attacks cannot move it.

**Read this before any model number below.** The best single scalar is `total_records_across_views` at test AUC 0.6843; the best model is without_dns / prismflow at 0.8389, a difference of +0.1546.

That difference is an all-test figure. Split by whether the attack sample's computer also has attack samples in train (section 7.3's definition), `total_records_across_views` reaches 0.6405 on the 255 seen attack samples and 0.7563 on the 155 unseen ones (section 7.7), while the models reach 0.8153 to 0.8875 on seen and 0.7102 to 0.7614 on unseen (section 7.3). Against the scalar's unseen AUC of 0.7563, of the 8 arm-and-system pairs 0 are above by more than 0.01 AUC, 5 are within 0.01 of it and 3 are below it by more than 0.01. The all-test ranking of models over this scalar therefore does not hold inside the unseen group.

| arm | system | all-test AUC | seen AUC | unseen AUC | unseen - scalar unseen | model unseen vs scalar unseen |
|---|---|---:|---:|---:|---:|---|
| without_auth | naive | 0.7755 | 0.8153 | 0.7102 | -0.0461 | below |
| without_auth | prismflow | 0.7775 | 0.8168 | 0.7126 | -0.0436 | below |
| without_flows | naive | 0.8137 | 0.8563 | 0.7435 | -0.0127 | below |
| without_flows | prismflow | 0.8233 | 0.8674 | 0.7508 | -0.0055 | within 0.01 of |
| without_dns | naive | 0.8336 | 0.8793 | 0.7584 | +0.0021 | within 0.01 of |
| without_dns | prismflow | 0.8389 | 0.8875 | 0.7589 | +0.0026 | within 0.01 of |
| with_auth | naive | 0.8278 | 0.8686 | 0.7606 | +0.0043 | within 0.01 of |
| with_auth | prismflow | 0.8375 | 0.8837 | 0.7614 | +0.0052 | within 0.01 of |

Scalar row for comparison: `total_records_across_views` reaches 0.6843 over all 410 test attack samples, 0.6405 seen and 0.7563 unseen. It is untrained, so it has no seed and no std; the model columns are 5-seed means whose stds are in sections 4.1 and 7.3. One fixed split, no interval.

### 7.2 Host audit: how concentrated the attack labels are

| block | attack samples | attack computers | top-1 share | top-4 share |
|---|---:|---:|---:|---:|
| train | 439 | 198 | 0.3235 | 0.3645 |
| test | 410 | 139 | 0.3341 | 0.4049 |

Attack computers appearing in BOTH blocks: **35**; test-only attack computers: 104. A computer with attack samples in BOTH blocks lets a model recognise the host rather than the behaviour, which is why the seen/unseen split below is reported.

### 7.3 Seen vs unseen computers

'seen' = the attack sample's computer also has at least one attack sample in the train block. Each group is scored against all benign test samples. Descriptive, not a controlled comparison.

| arm | system | seen AUC | unseen AUC | n seen | n unseen |
|---|---|---|---|---:|---:|
| without_auth | naive | 0.8153 +/- 0.0026 | 0.7102 +/- 0.0035 | 255 | 155 |
| without_auth | prismflow | 0.8168 +/- 0.0022 | 0.7126 +/- 0.0052 | 255 | 155 |
| without_flows | naive | 0.8563 +/- 0.0028 | 0.7435 +/- 0.0055 | 255 | 155 |
| without_flows | prismflow | 0.8674 +/- 0.0022 | 0.7508 +/- 0.0073 | 255 | 155 |
| without_dns | naive | 0.8793 +/- 0.0031 | 0.7584 +/- 0.0036 | 255 | 155 |
| without_dns | prismflow | 0.8875 +/- 0.0047 | 0.7589 +/- 0.0082 | 255 | 155 |
| with_auth | naive | 0.8686 +/- 0.0033 | 0.7606 +/- 0.0046 | 255 | 155 |
| with_auth | prismflow | 0.8837 +/- 0.0046 | 0.7614 +/- 0.0051 | 255 | 155 |

### 7.4 Paired cluster bootstrap over test computers

2000 replicates, fixed seed 12345, clusters = test computers, paired (same resampled computers for both systems, every seed). Measures test-set sampling variability; the per-seed std measures training randomness. They are not interchangeable.

| arm | mean paired AUC difference (PrismFlow - naive) | 95% CI | test computers |
|---|---:|---|---:|
| without_auth | +0.0019 | [-0.0005, +0.0063] **includes 0** | 8,350 |
| without_flows | +0.0096 | [+0.0052, +0.0142] excludes 0 | 8,350 |
| without_dns | +0.0053 | [+0.0010, +0.0083] excludes 0 | 8,350 |
| with_auth | +0.0097 | [+0.0027, +0.0138] excludes 0 | 8,350 |

A per-seed std and this interval answer different questions. The std says how much the answer moves when the initialisation and batch order change on THIS test set; the interval says how much it would move if the test computers had been a different draw. An interval that includes 0 means the sign of the difference is not established by this test set, whatever the seed counts say.

### 7.5 How a missing view enters dependence and ENIV

A missing view is carried as a BOOLEAN MASK, not as a zero-filled observation. `prismflow/statistics/dependence.py::feature_dependence_matrix` receives `view_mask` and `prismflow/statistics/dependence.py::available_views` reduces it to the per-view availability vector that `prismflow/eniv/eniv.py::compute_eniv` is given, so an absent view is excluded from the pair statistics rather than contributing a zero row. The feature row of an absent view IS zero in the tensor (`run_lanl.py::apply_scaler` leaves `views[~rows, v] = 0.0`), but the encoder never consumes it because `MultiViewEncoder` gates on the mask. Consequence for the pair table: a dns pair is computed over the samples where BOTH views of the pair are present, not over all samples, so the dns pairs rest on the ~2% of samples that have a dns record.

Part 30 close-out, read-only: every figure is computed from results/lanl/samples.npz and results/lanl/scores/*.npz. No model was trained and no view was rescanned. Results only - no interpretation is offered in this block.

### 7.6 The 4 most-attacked computers, as sampled

| computer | train attack | train benign | test attack | test benign | attack train+test |
|---|---:|---:|---:|---:|---:|
| `C17693` | 142 | 1 | 137 | 1 | 279 |
| `C1015` | 1 | 6 | 12 | 1 | 13 |
| `C2388` | 3 | 4 | 10 | 3 | 13 |
| `C395` | 5 | 4 | 7 | 0 | 12 |

Attack computers in the sample table: 302. Counts are rows of the shared sample table, which is case-control: every attack cell is kept and a benign cell is kept iff crc32(computer|window) % 512 == 0, so the benign column is kept controls and not the computer's benign windows on the network.

### 7.7 The 7.1 scalars inside the 7.3 seen/unseen groups

Same scalars and same train-only orientation as 7.1; same seen/unseen definition and same all-benign comparison set as 7.3. Untrained, so there is no seed and no std.

| scalar | orientation (fixed on train) | seen test AUC | unseen test AUC |
|---|---|---:|---:|
| `total_records_across_views` | as-is | 0.6405 | 0.7563 |
| `auth_present_indicator` | as-is | 0.6637 | 0.6637 |
| `n_views_present` | as-is | 0.4895 | 0.6824 |

n = 255 seen attack rows, 155 unseen attack rows, each against all 18,319 benign test rows - the same comparison sets as 7.3.

### 7.8 `without_auth` over its predictable test rows only

Test rows with at least one of this arm's views present. The excluded rows are unpredictable by construction, not misclassified. Attack recall is at a 0.5 probability threshold, which for two classes is the argmax.

Rows kept: 15,129 of 18,729 test rows (271 attack, 14,858 benign). Rows excluded: 3,600, of which 139 are attack rows.

| system | ROC AUC (rows with a present view) | attack recall (same rows) |
|---|---|---|
| naive | 0.8556 +/- 0.0050 | 0.0790 +/- 0.0020 |
| prismflow | 0.8594 +/- 0.0053 | 0.0782 +/- 0.0048 |

Beside these, the same arm over ALL test rows is in section 4.1.

### 7.9 Test AUC with the most-attacked test computer removed

All test rows of one computer removed, attack and benign together. Descriptive: one host is not a resample, so this is not an interval.

Removed: every test row of `C17693` - 138 rows (137 attack, 1 benign), leaving 18,591 test rows of which 273 are attack rows.

| arm | system | ROC AUC, that computer removed | ROC AUC, all test rows (4.1) |
|---|---|---|---|
| without_auth | naive | 0.7237 +/- 0.0037 | 0.7755 +/- 0.0025 |
| without_auth | prismflow | 0.7267 +/- 0.0047 | 0.7775 +/- 0.0030 |
| without_flows | naive | 0.7702 +/- 0.0042 | 0.8137 +/- 0.0022 |
| without_flows | prismflow | 0.7800 +/- 0.0042 | 0.8233 +/- 0.0031 |
| without_dns | naive | 0.7976 +/- 0.0036 | 0.8336 +/- 0.0029 |
| without_dns | prismflow | 0.8021 +/- 0.0092 | 0.8389 +/- 0.0058 |
| with_auth | naive | 0.7899 +/- 0.0044 | 0.8278 +/- 0.0033 |
| with_auth | prismflow | 0.7974 +/- 0.0064 | 0.8375 +/- 0.0043 |

## 8. Cross-dataset table (part30.md step 5)

synthetic vs AIT vs LANL, for the DETECTION claim and the honest-duplication false positive. Every cell cites a `results/` file or says `not measured`. Nothing is pooled across datasets and nothing is averaged across AIT's two directions.

### 8.1 DETECTION: does dependence among the implicated views rise?

| dataset | condition | mean dependence | ENIV | source |
|---|---|---|---|---|
| synthetic | clean | 0.1849 +/- 0.0361 (`dependence_all_pairs`) | 3.3281 +/- 0.0750 | `results/chorus/attack_metrics.json`, `summary.clean|prismflow` |
| synthetic | chorus k=2, eps 1.0 | 0.6970 +/- 0.0380 (`dependence_compromised`) | 2.9048 +/- 0.0608 | `results/chorus/attack_metrics.json`, `summary.chorus_k2_eps1.0|prismflow` |
| AIT (A) | attack (dnsteal-only) | 0.2906 +/- 0.0274 | 3.2868 +/- 0.1345 | `results/ait_attack/results.json`, `directions.A_rm_to_santos.systems.prismflow.natural_windows.dnsteal_only.summary` |
| AIT (A) | benign | 0.3273 +/- 0.0124 | 3.1749 +/- 0.0749 | `results/ait_attack/results.json`, `directions.A_rm_to_santos.systems.prismflow.natural_windows.benign.summary` |
| AIT (B) | attack (dnsteal-only) | 0.3619 +/- 0.0196 | 2.8917 +/- 0.0823 | `results/ait_attack/results.json`, `directions.B_santos_to_rm.systems.prismflow.natural_windows.dnsteal_only.summary` |
| AIT (B) | benign | 0.2482 +/- 0.0249 | 3.1458 +/- 0.0856 | `results/ait_attack/results.json`, `directions.B_santos_to_rm.systems.prismflow.natural_windows.benign.summary` |
| LANL (without_auth) | attack | 0.1175 +/- 0.0671 | 2.7649 +/- 0.1341 | `results/lanl/results.json`, `configurations.without_auth.part29_test_a_natural_attack_windows.per_system.prismflow.populations.attack_windows.summary` |
| LANL (without_auth) | benign | 0.0573 +/- 0.0201 | 2.8752 +/- 0.0454 | `results/lanl/results.json`, `configurations.without_auth.part29_test_a_natural_attack_windows.per_system.prismflow.populations.benign_windows.summary` |
| LANL (without_flows) | attack | 0.0789 +/- 0.1045 | 2.8216 +/- 0.1872 | `results/lanl/results.json`, `configurations.without_flows.part29_test_a_natural_attack_windows.per_system.prismflow.populations.attack_windows.summary` |
| LANL (without_flows) | benign | 0.1882 +/- 0.0419 | 2.6100 +/- 0.0845 | `results/lanl/results.json`, `configurations.without_flows.part29_test_a_natural_attack_windows.per_system.prismflow.populations.benign_windows.summary` |
| LANL (without_dns) | attack | 0.1495 +/- 0.0493 | 2.5993 +/- 0.0790 | `results/lanl/results.json`, `configurations.without_dns.part29_test_a_natural_attack_windows.per_system.prismflow.populations.attack_windows.summary` |
| LANL (without_dns) | benign | 0.1189 +/- 0.0074 | 2.7491 +/- 0.0132 | `results/lanl/results.json`, `configurations.without_dns.part29_test_a_natural_attack_windows.per_system.prismflow.populations.benign_windows.summary` |
| LANL (with_auth) | attack | 0.1731 +/- 0.1486 | 3.1816 +/- 0.1844 | `results/lanl/results.json`, `configurations.with_auth.part29_test_a_natural_attack_windows.per_system.prismflow.populations.attack_windows.summary` |
| LANL (with_auth) | benign | 0.1258 +/- 0.0148 | 3.5619 +/- 0.0355 | `results/lanl/results.json`, `configurations.with_auth.part29_test_a_natural_attack_windows.per_system.prismflow.populations.benign_windows.summary` |

AIT's attack population is dnsteal-only, which Part 29 forbids pooling with its multi-step windows; LANL's red-team file has no step taxonomy, so LANL's single attack population is not the same object as either of AIT's. The rows are side by side, not comparable cell by cell.

### 8.2 Honest-duplication false positive

| dataset | quantity | value | source |
|---|---|---|---|
| synthetic | exact duplication, k=0: mean dependence, all pairs / clone pairs | 0.0537 +/- 0.0251 / not measured (no clone pairs exist at k=0) | `results/clone/summary.json`, `summary[]` where k=0, rho=0, system=prismflow |
| synthetic | exact duplication, k=2: mean dependence, all pairs / clone pairs | 0.2037 +/- 0.0092 / 0.9175 +/- 0.0159 | `results/clone/summary.json`, `summary[]` where k=2, rho=0, system=prismflow |
| synthetic | Part 10 clone_k2 detector AUC | not measured in this Part | not re-run here; Part 10's own results file holds it |
| AIT | strict same-sensor pair | NO EVIDENCE | `results/ait_attack/results.json`, `sensor_group_check.verdict` |
| LANL | strict same-sensor pair | NO EVIDENCE - cyber1 ships no sensor-group definition, so no within-group pair exists | `configs/data/lanl.yaml`; section 6 above |
| LANL | detector AUC on naturally dependent pairs | not measured | detector not scored (section 9) |

## 9. What was not run, and why

part30.md step 4 asks for Part 28's metrics **and Part 29's tests A-D**. Part 28's metrics and test A ran. B, C and D did not, and no partial substitute is reported in their place:

- **B_chorus_attack** - NOT RUN. Part 29's grid is k in {1,2,3} x Part 09's epsilon and beta grid, retrained per cell, which does not fit this Part's time budget. No partial grid is reported, because a subset of the cells is not the test.
- **C_honest_duplication** - PARTIALLY RUN. The strict same-sensor pair does not exist here: LANL cyber1 ships no sensor-group definition, so there is no within-group pair to score and the answer is NO EVIDENCE. The 'different sensors, naturally dependent' half needs the most/least dependent pairs, which ARE written to systems.<name>.pair_table; scoring the detector on them was not run.
- **D_missingness** - NOT RUN as a detector-AUC test. Per-view missing fractions on train and test ARE measured above, and whole arms without flows and without dns ARE run, which is a stronger answer than masking at 30%/50%; but Part 10's masked detector AUC itself was not computed.

These are **INCOMPLETE**, not negative results. Nothing below should be read as evidence about chorus robustness, the suspicion detector or masked-view behaviour on LANL.

## 10. What this design cannot show

- **No population-level operating point.** The case-control sample fixes prevalence by construction, so no threshold, alert rate or precision figure here transfers to the real network.
- **No spread over attack episodes.** One fixed split, and the +/- is initialisation-and-batch-order variance (caveat ii). The red-team activity is concentrated in a few days (caveat i), so even the test block is close to a handful of episodes.
- **No independence from the label source in the WITH-auth arm.** Caveat iii.
- **No claim that benign windows are clean.** Caveat iv.
- **No cross-dataset significance.** Three datasets, one split each, different label semantics; section 7 is a side-by-side table, not a test.
- **No detection-vs-defence separation on LANL.** That separation is Part 09's and rests on the chorus grid, which was not run here.

## 11. Proposed limitations (NOT applied)

`docs/KNOWN_LIMITATIONS.md` and `docs/CONTRACT.md` are unchanged by this Part. The following text is proposed for the user to approve or reject; it has not been written to either file.

> **L23 (proposed).** On LANL cyber1 at 300 s the `(computer, window)` sample space is between 23,627,216 and 60,634,229 occupied cells over days 1-16 (`results/lanl/population.json`), against 439 train and 410 test attack cells - a population prevalence near 2e-5. Part 30 therefore reports a declared case-control sample (all cases, deterministic 1/512 hash of controls) whose test prevalence is 0.0219, about 1095x the population rate. Its accuracy, Brier and ECE therefore describe that sample and not the network, and no population operating point has been measured on LANL.

> **L24 (proposed).** LANL's red-team labels are derived from authentication records, so the `auth` view is label-adjacent. Part 30 measures both arms, so the WITHOUT-auth arm is the one that bears on detection from evidence independent of the label source. That arm is itself limited: 139 of 410 of its test attack samples (33.9%) have no record in any remaining view, so they are unpredictable by construction and cap the arm's attainable recall before training. Every attack sample has an `auth` record while about a third of benign samples do not (410 of 410 attack against 12323 of 18319 benign in the test block), so auth presence alone reaches test AUC 0.6637, and the WITHOUT-auth arm removes the auth features but not this selection, because cells were chosen from all four views.

> **L25 (proposed).** Part 28's `fit_scaler`, `apply_scaler` and `train_ait` are not reusable across datasets: each resolves native view widths through `ait_lds.FEATURES`, keyed by AIT source names. Part 30 mirrored them rather than patching Part 28. Any future Part on a third dataset will hit the same wall.

> **L26 (proposed).** Seen and unseen attack computers. On LANL cyber1 the trained models' ranking advantage over a single untrained scalar is confined to attack computers already seen as attacked in train. `total_records_across_views`, oriented on train only, reaches test ROC AUC 0.6843 over all 410 test attack samples, against 0.7755 to 0.8389 for the trained arms. Split by whether the attack computer also has attack samples in train, the scalar reaches 0.6405 on the 255 seen samples and 0.7563 on the 155 unseen ones, while the models reach 0.8153 to 0.8875 on seen and 0.7102 to 0.7614 on unseen (sections 7.3 and 7.7). On unseen computers `without_auth` naive, `without_auth` prismflow, `without_flows` naive are below it; `without_flows` prismflow, `without_dns` naive, `without_dns` prismflow, `with_auth` naive, `with_auth` prismflow are within 0.01 of it. The verdict is not uniform within `without_flows`, so these are stated per arm and system, not per arm. One fixed split, no interval. (`results/lanl/results.json`, `phase3_diagnostics.seen_unseen` and `phase4_closeout.trivial_scalar_seen_unseen`.)

> **L27 (proposed).** Host concentration and no within-host benign contrast. One computer, `C17693`, accounts for 279 of the 849 attack samples (142 of 439 in train, 137 of 410 in test) and has 1 benign sample in train and 1 in test, because benign cells are kept at 1/512 by hash. A model can therefore learn that host's activity profile as the attack class without learning attack behaviour. Removing every test row of `C17693` lowers test ROC AUC by 0.036 to 0.052 depending on arm and system (section 7.9). 35 attack computers appear in both blocks and 104 are test-only. A design that keeps all or a stated share of the benign cells of attack hosts would be needed to separate host from behaviour.

> **L28 (proposed).** Execution mode and replicate spread. The torch thread count changes the fitted weights. Runs 1 (12 threads, sequential) and 2 (1 thread, 11 concurrent) of the with_auth arm agree in the sign of the PrismFlow-minus-naive difference for `roc_auc`, `prob_ece`, `benign_recall`, `accuracy` (`roc_auc` +0.0069, +0.0097; `prob_ece` -0.0013, -0.0008), but disagree for `balanced_accuracy` (-0.0004, +0.0088), `attack_recall` (-0.0010, +0.0176), `brier` (+0.0000, -0.0004). The same seed differs by up to 0.0140 in AUC across modes (naive seed 0: 0.8375 and 0.8235), about 4 per-seed standard deviations of that same arm and system (0.0033). A seed count such as 5/5 inside one run shares one execution mode and is not independent evidence about mode effects. Differences of order 0.01 in `balanced_accuracy`, `attack_recall`, `brier` are not established, and the per-seed std understates replicate spread. Related: Part 28's committed ENIV is reproducible only at the thread count that produced it (`results/lanl/results.json`, `execution_mode.thread_count_reproducibility_observation`).

> **L29 (proposed).** Probabilities worse than a constant. In `without_auth` (0.1218 and 0.1219) and `without_flows` (0.0628 and 0.0630) the Brier score of both systems is above the constant-prevalence baseline of 0.0428, while ROC AUC is 0.7755 to 0.8233. Cause not diagnosed. Candidates, none tested: rows with an all-False view mask, no validation split, and the shift in sample prevalence from train (1.0%) to test (2.2%). Probability outputs of those arms should not be used.

> **L30 (proposed).** What Part 30 does not establish. Part 29 tests B (chorus attack), D (masked-view detector AUC) and the detector half of C were not run on LANL (section 9); no robustness claim in L1-L22 has been tested on this dataset, and Part 30 is clean-data evidence only. Part 29 test A is not a stable signal: ENIV is lower and mean pairwise dependence higher on attack windows in `without_auth`, `without_dns`, `with_auth`, and both reverse in `without_flows` for both systems (ENIV +0.2636 naive, 5/5 seeds and +0.2116 prismflow, 5/5 seeds). The two classes also differ in how many rows can carry a pair at all (`with_auth`: 66.1% of attack rows against 54.3% of benign rows have two or more views).

## 12. Files

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
| `results/lanl/samples.npz` | the whole 61,641-row sample table, built once; every arm reads it so all arms share identical rows |
| `results/lanl/units/*.json` | one file per (arm, system, seed): metrics, diagnostics, test-A populations, timing |
| `results/lanl/scores/*.npz` | per-sample TEST scores keyed by (computer, window_start), per arm, system and seed |
| `results/lanl/execution_mode.json` | the thread benchmark and the parallel-vs-sequential decision |
| `results/lanl/results.run1.json.gz` | run 1, immutable, with its sha256 in `results_manifest.json` |
| `results/lanl/results.json` | every metric, diagnostic, pair table and paired difference |

**Wall-clock, and what each number measures.** `results/lanl/results.json` `elapsed_seconds` = 753 s. results.json `elapsed_seconds` times the AGGREGATE step only - `run_lanl.py --aggregate`: reading the 40 unit files, samples.npz and scores/*.npz, computing the Phase 3 and Phase 4 diagnostics including the 2000-replicate cluster bootstrap, and writing results.json and this README. It does NOT include any training. The sample-table scan was 234 s on 4 worker processes, once, before any arm ran.

Training wall-clock, from the unit files only:

| quantity | value |
|---|---|
| units | 40 |
| first unit start | not recorded in unit files |
| last unit end | not recorded in unit files |
| total wall-clock | not recorded in unit files |
| sum of unit seconds | 70,753 s (19.7 process-hours) |
| longest single unit | 4,597 s |
| shortest single unit | 472 s |
| concurrency limit | 11 |
| waves (ceiling of units over concurrency) | 4 |

Read from the `seconds` field of every file in results/lanl/units/. Unit files carry no start or end timestamp, so first unit start, last unit end and total wall-clock are NOT RECORDED in results/lanl/ and are not estimated here. The wave count is the arithmetic ceiling of units over the concurrency limit, not an observed schedule: units have unequal lengths, so waves overlap in practice.

Commit `c96e4dfde138ccad66b9acb71d57f84c2e546689`.

`data/raw/` was read only. Nothing in it was extracted, renamed, moved or deleted, and `auth.txt.gz` was not modified.
