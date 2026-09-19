# Datasets

PrismFlow runs on two kinds of data, and they answer different questions.
Confusing them would let a measurement be read as a validation, so the
distinction is stated first and everything else follows from it.

| | synthetic (`prismflow/data/synthetic.py`) | real (`prismflow/data/real_datasets.py`) |
|---|---|---|
| cross-view dependence | a parameter, `rho`, that we set | unknown; there is no parameter |
| ground truth for ENIV | `analytic_n_eff`, `analytic_n_eff_eigen` | none exists |
| what a seed varies | the generated dataset | the train/val/test partition |
| ENIV can be | VALIDATED (Part 05) and applied | applied only |

## Names used here, and what they are in the brief's language

Two short names appear in the code, the config and the result JSON. They are
internal identifiers, not new ideas, and this is what they stand for. Part 16
should audit against the right-hand column.

| name in code / config / JSON | what it is |
|---|---|
| stage `redundancy`, "redundancy audit" | the per-dataset `nominal views / measured ENIV / efficiency ratio / lambda_U` table that the Part 15 brief requires ("REPORT THIS SEPARATELY - IT IS ITSELF A CONTRIBUTION") |
| `eniv_independence_null`, `rho_bar_independence_null`, "independence null" | the permutation-null independence control described below -- NOT requested by any brief, added because the brief's claim does not follow from the ENIV number alone |

The identifiers are left as they are because they are keys in result files
that are already committed; renaming them would invalidate recorded evidence
to fix a label. `lambda_U` is a Part 13 quantity and appears in the table
because the Part 15 brief's table has a `lambda_U` column.

## THE LIMITATION, STATED PLAINLY

**On real data there is no ground-truth dependence, so ENIV cannot be
validated here -- only applied.**

The synthetic generator is the project's calibration standard for exactly one
reason: we choose `rho`, so we can ask whether the estimator recovers a number
we already know. Part 05 does that, and the ENIV validation sweep is the only
evidence in this project that ENIV measures what it claims to measure.

A real benchmark has no such number. Its views are dependent to some degree
that was never set by anyone and cannot be read off the data, so there is
nothing for an estimate to be checked against. Every ENIV, efficiency ratio
and lambda_U reported on real data in Part 15 is therefore a **measurement
taken with an instrument validated elsewhere**, not a test of the instrument.

Concretely, none of the following are licensed by any real-data result:

- that ENIV is accurate on real data;
- that the redundancy it reports is the true redundancy;
- that agreement between a real benchmark's views is as redundant, or as
  independent, as the number says.

`RealMultiViewDataset.rho_matrix` is `None` for this reason, and
`tests/unit/test_real_datasets.py` asserts it. It is not an estimate standing
in for a target. Code reaching for ground truth on real data fails loudly
instead of quietly consuming a measurement as if it were one.

### The one control that does travel

Part 15's redundancy audit cannot validate ENIV, but it can bound one specific
alternative explanation. ENIV is biased downward at finite sample size -- the
eigenvalue form caps eigenvalues at 1, which can only remove mass, so
estimation noise in the off-diagonals pushes the count below the view count
even at true independence (`prismflow/eniv/eniv.py`, "finite samples"). A
reading of 3-point-something out of 6 is therefore not by itself evidence of
redundancy.

So the audit measures the estimator's own reading under an explicit
independence null: each view's encoder features and evidence are permuted
**within class**, which destroys cross-view coupling while leaving the sample
size, the feature dimension, the marginals and the class structure untouched.
The observed ENIV is then reported against that null rather than against 6.

This rules out small-sample bias as the whole story. It does not establish
that the remaining gap is the true dependence, and must not be presented as
doing so.

## Handwritten (UCI Multiple Features, `mfeat`)

The Part 15 brief's first preference, and the canonical 6-view benchmark of
the multi-view literature.

| | |
|---|---|
| source | <https://archive.ics.uci.edu/dataset/72/multiple+features> |
| samples | 2000 (200 per class, stored in class-block order) |
| classes | 10 (handwritten digits 0-9) |
| views | 6 |

| view | file | dim | features |
|---|---|---|---|
| fou | `mfeat-fou` | 76 | Fourier coefficients of the character shapes |
| fac | `mfeat-fac` | 216 | profile correlations |
| kar | `mfeat-kar` | 64 | Karhunen-Loeve coefficients |
| pix | `mfeat-pix` | 240 | pixel averages in 2x3 windows |
| zer | `mfeat-zer` | 47 | Zernike moments |
| mor | `mfeat-mor` | 6 | morphological features |

All six views are extracted from **the same 2000 binary images**. That is the
structural reason to expect dependence before measuring any: these are six
descriptions of one object, not six independent observations of it.

Files are fetched on first use into `data/raw/handwritten/` (gitignored) and
verified against the SHA-256 digests recorded in
`prismflow/data/real_datasets.py`. A digest mismatch is an error rather than
something to overwrite: it means the upstream distribution changed under a
result that has already been reported.

### Measured redundancy

From `results/real/handwritten/redundancy.json`, 5 seeds, 800 test samples,
before any duplication, corruption or attack. Full tables in
`results/real/README.md`.

| nominal views | measured ENIV | efficiency ratio | lambda_U (q=0.90) |
|---|---|---|---|
| 6 | 3.28 +/- 0.07 | 0.547 +/- 0.011 | 0.127 +/- 0.042 |

| | value |
|---|---|
| test accuracy | 0.981 +/- 0.003 |
| rho_bar (observed) | 0.526 +/- 0.015 |
| rho_bar (independence null) | 0.051 +/- 0.012 |
| ENIV (independence null) | 5.51 +/- 0.08 |
| ENIV deficit against that null | -2.23 +/- 0.06 |
| ENIV, design-effect form | 1.65 +/- 0.03 |
| tail ENIV | 3.72 +/- 0.45 |

Read this as: of the 2.72 effective views the benchmark falls short by, about
0.49 is the estimator's floor at this sample size and feature dimension, and
about 2.23 is measured redundancy. The design-effect form reads much lower
(1.65) because it assumes exchangeability, and these views are conspicuously
not exchangeable -- `fac`, `kar`, `pix` and `zer` form a dependent block while
`fou` and `mor` sit outside it. `prismflow/eniv/eniv.py` explains why the
eigenvalue form is the default for exactly this structure.

The per-view breakdown is the practically useful part: removing `fou` costs
0.74 effective views and removing `mor` costs 0.68, while removing `fac` costs
0.24 and removing `pix` costs 0.21 -- although `fac` and `pix` are the two
largest views, at 216 and 240 features, and are 0.85 dependent on each other.
View count and view width both overstate what those two views contribute.

### Why only one dataset

The brief lists Caltech101-7/20, Scene15, BBCSport and Reuters as further
candidates. None were added, for reasons that are practical rather than
scientific and are recorded here so the gap is not mistaken for a choice:

- **BBCSport, 3Sources and the other mlg.ucd.ie collections** -- the host was
  unreachable from this environment (connection timeout on every file).
- **Caltech101-7/20, Scene15** -- distributed as MATLAB `.mat` files, which
  need `scipy` (v7) or `h5py` (v7.3). Neither is in `requirements.txt`, and
  Part 15's allowed-paths list does not include it, so adding a dependency was
  out of scope.
- **Reuters multilingual (UCI)** -- 5 views, but the distribution is a
  multi-hundred-megabyte archive; out of proportion to a one-day sweep.

The redundancy table is therefore a single row. A single benchmark cannot
establish that multi-view benchmarks in general are redundant; it establishes
it for this one. Additional rows are the obvious extension.

## Preprocessing

Applied in `prismflow.data.real_datasets.build_real_dataset`, in this order.

1. **Split first.** `split_indices(n, seed=seed)` -- the project's own
   deterministic partition.
2. **Standardise on train only.** Each view's native features are z-scored
   using the mean and standard deviation of the TRAINING rows. No test
   statistic reaches the scaler. Columns with near-zero training variance are
   centred and left unscaled rather than divided by a noise floor.
3. **Pad to a common width.** Views are right-padded with zeros to
   `D = max_v d_v` (240 here), so the `[B, V, D]` tensor convention in
   `docs/CONTRACT.md` is unchanged.

Standardisation is not cosmetic. The six views span four orders of magnitude
(morphological features reach 1.8e4, Fourier coefficients sit below 1), and
the attack budget `epsilon` is an L-infinity bound in feature units -- without
it, the same epsilon would be a different attack in every view.

### Padding is inert, by construction

`MultiViewEncoder` already supported heterogeneous view widths before Part 15:
each view's `EncoderConfig` declares its own `input_dim <= D` and the encoder
slices `views[..., v, :input_dim]`. `real_datasets.view_configs` builds those
configs from the native widths, so:

- no encoder ever reads a padded column;
- no gradient reaches one;
- the Chorus/PGD `sign()` step leaves them at exactly zero, because the
  gradient there is exactly zero.

Building the encoder configs by hand with a uniform `input_dim` would feed one
view's features to another view's encoder. Use `view_configs`.

## What a seed varies, and why the spread is narrower here

On synthetic data the seed redraws the dataset and the split is held fixed
(`TrainConfig.split_seed`). Real data cannot be redrawn, so on real data the
seed drives the train/val/test partition, the scaler fitted on it, model
initialisation and batch order.

Five seeds on real data are therefore **five resamplings of the same 2000
patterns**, not five datasets. The standard deviations reported across them
are split-and-init variance. They are not comparable to the synthetic parts'
spreads, which include data variance, and they should be expected to be
narrower. Do not read a tight real-data interval as a stronger result than a
wider synthetic one.

## Two splits, and why

| stage | split | test n | reason |
|---|---|---|---|
| Part 10 comparison matrix | 0.70 / 0.15 / 0.15 | 300 | Part 10's split, unchanged, so the matrix is comparable cell for cell |
| redundancy audit, Part 13 tail | 0.50 / 0.10 / 0.40 | 800 | tail estimation needs exceedances |

Tail dependence is estimated from the points above a high quantile, so it is
data-hungry by construction. `tail_dependence.MIN_TAIL_SAMPLES` is 50. Part 13
bought that sample by raising the generator's `n_samples` to 8000; on real
data that lever does not exist -- there are 2000 patterns and no more -- so the
sample is bought from the split instead.

At 800 test points, `q = 0.90` gives 80 exceedances and clears the threshold;
`q = 0.95` gives 40 and does **not**. The headline quantile on real data is
therefore **0.90**, where Part 13's was 0.95. `q = 0.95` is still reported,
with its exceedance count and its reliability flag, so the shortfall is
visible rather than hidden. Part 13's 300-sample subsample column is also
kept, so the cost of a small split stays measured rather than asserted.

ENIV is affected by the same sample size in the same direction (its
finite-sample bias is downward), which is the second reason the redundancy
audit uses the wide split: a 300-point estimate would understate the effective
view count and overstate the redundancy being reported.

### Two clean ENIV numbers, and why they differ

This is the one place the record reports the same quantity twice, so it is
stated explicitly rather than left to be discovered as an apparent
contradiction. Baseline ENIV on Handwritten appears as **3.28 +/- 0.07** and
as **3.35 +/- 0.09**. Both are correct and neither is a restatement of the
other: they are clean-condition ENIV measured on the two different splits
above, by different models trained on different amounts of data.

| number | source | split | test n | train n | accuracy |
|---|---|---|---|---|---|
| ENIV 3.28 +/- 0.07, ratio 0.547 | `handwritten/redundancy.json` | 0.50/0.10/0.40 | 800 | 1000 | 0.981 +/- 0.003 |
| ENIV 3.35 +/- 0.09, ratio 0.558 | `handwritten/comparison.json`, `clean` cell | 0.70/0.15/0.15 | 300 | 1400 | 0.987 +/- 0.004 |

The headline table quotes the first, because it is the one measured with
enough test samples to carry the `lambda_U` column beside it. The comparison
matrix quotes the second, because its whole point is to be cell-for-cell
comparable with Part 10, which used that split. Quote whichever the context
requires, name the split when you do, and do not average them.

## The attack budget does not transfer by name

`epsilon` in `ChorusConfig` is an L-infinity bound applied per feature, and the
attack step is `sign(gradient)`, so a compromised view moves by `epsilon` in
almost every one of its dimensions. Parts 09 and 13 ran at `d_view = 16`; the
compromised views here are `fou` at 76 features and `fac` at 216. The same
`epsilon = 1.0` therefore buys several times the L2 perturbation it bought on
synthetic data.

The epsilon grid is kept identical to Part 13 anyway, because changing it would
make the two Parts incomparable in a different and less visible way. Instead
the realised mean L2 norm of the perturbation is recorded per cell
(`delta_l2_mean` in `results/real/handwritten/tail.json`, and a column in
`results/real/README.md`), so the comparison can be made on the quantity that
actually transfers rather than on a shared label. Attack success rates at a
given epsilon are NOT comparable between the synthetic Parts and this one.

Padded columns are exempt: their gradient is exactly zero, so `sign()` returns
zero and they are never perturbed.

## Adding a dataset

Add a `RealDatasetSpec` to `REGISTRY` in `prismflow/data/real_datasets.py`
with one `ViewSpec` per view, including each file's SHA-256. If its labels are
not in class-block order, the label construction in `load_raw` needs a branch.
Then add its name to `data.datasets` in `experiments/real/config.yaml`.
`tests/unit/test_real_datasets.py::test_registry_specs_are_internally_consistent`
checks the spec; the schema tests need no change, because they compare against
`Batch`'s own dataclass fields rather than a written-down list.
