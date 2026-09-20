# Reproducibility

What is guaranteed, what is not, and the one mistake that will silently destroy
your committed results.

## READ THIS BEFORE RUNNING ANY EXPERIMENT

**`--quick` is a smoke test. Its output is not evidence and must never be
reported.** It cuts the run to 1 seed and 2 epochs, and on some scripts it also
shrinks the dataset and the sweep grid.

Until 2026-09-20, three scripts wrote `--quick` output directly onto the
canonical results path, silently overwriting committed 5-seed evidence. That
happened — see Finding 1 in `AUDIT_REPORT.md` for exactly what it destroyed and
how close it came to being published.

It is now fixed. `--quick` writes to `results/<name>_quick/`, which is
gitignored. But the habit that caused it is worth keeping:

> **Never run a sweep of "every experiment script" without checking
> `git status` afterwards and reading the diff.** A changed results file is a
> claim about your data. Treat it as one.

Four scripts still hardcode their output path and have no `--quick` flag today:
`run_adaptive.py`, `run_diagnostics.py`, `run_gate_matrix.py`,
`run_sign_constrained.py`. If you add a smoke flag to any of them, apply the
`out_dir` fix from `run_tail.py` in the same commit.

## The 5-seed rule

Every experiment runs a minimum of 5 seeds and reports mean and sample standard
deviation (ddof = 1). **A single-seed result is not evidence and must never be
reported as a finding.**

This is enforced in code, not by convention.
`prismflow/evaluation/protocol.py` raises below `MIN_SEEDS` unless the caller
passes `allow_fewer_seeds`, and any output produced that way is stamped
`not_evidence: true` in both the JSON and the CSV.

Verified at the Part 16 audit: all 180 committed result files carry
`not_evidence: false`.

Note the gap this leaves. The guard catches too few seeds being *passed to the
protocol*. It does not catch the config being cut *upstream*, which is exactly
what `--quick` does — those are legitimate 1-seed runs as far as `evaluate()`
can tell. The output-path separation above is the only control that covers it.

## Determinism

`prismflow/utils/seed.py::set_seed` seeds `random`, `PYTHONHASHSEED`, numpy,
torch CPU and all CUDA devices, and with `deterministic=True` (the default) sets
`cudnn.deterministic`, disables `cudnn.benchmark`, sets
`CUBLAS_WORKSPACE_CONFIG=:4096:8` and enables
`torch.use_deterministic_algorithms`.

Verified during the Part 16 audit on this environment:

- `split_indices(2000, seed=3)` called twice returns identical partitions; seed
  4 differs.
- The full pipeline — generate, construct, forward with discount — at seed 7
  twice returns **bitwise identical** probabilities and confidences. Seed 8
  differs.

**What this does not promise.** Bit-identical reproduction across a different
torch version, a different BLAS build, a different CPU architecture, or CPU
versus GPU is *not* guaranteed and was not tested. Expect agreement to the
reported standard deviations, not to the last digit, on any other machine.

Every result JSON records the torch version it was produced under. The
committed results were produced under **torch 2.14.0+cpu, numpy 2.5.3, Python
3.13.7**.

## Splits

`prismflow/data/dataset.py::split_indices` permutes once against a seeded
generator and slices, at 70/15/15 by default. Splits are therefore disjoint and
exhaustive by construction, not by check.

Verified at audit across 50 `(n, seed)` combinations: train/val/test were
pairwise disjoint and exactly covered `range(n)` in every case. There is no
label leakage across splits.

**A seed means different things on different data**, and conflating the two
would turn a measurement into a validation:

| | synthetic | real |
|---|---|---|
| what a seed varies | the generated dataset | the train/val/test partition |
| ground truth for ENIV | `analytic_n_eff` | none exists |
| ENIV can be | validated and applied | applied only |

So a `+/-` on a synthetic result is dataset variance, and a `+/-` on a real
result is partition variance. See `docs/DATASETS.md` and
`docs/KNOWN_LIMITATIONS.md` L14.

## Running experiments

All scripts are modules and are run from the repository root:

```bash
python -m experiments.eniv_validation.run_validation      # full, 5 seeds
python -m experiments.tail.run_tail --quick               # smoke, NOT evidence
```

13 of the 16 `--quick`-capable scripts accept `--results-dir` to redirect
output. The remaining three (`run_tail`, `run_oracle`, `run_branch_audit`)
redirect automatically under `--quick` and write to the canonical path
otherwise.

Scripts with a `config.yaml` beside them (`calibration`, `chorus`, `clone`,
`real`, `robustness`) read it; the rest carry a `CONFIG` dict at module top.
Either way the resolved config is written into the output JSON, so a result
file always states the parameters that produced it.

## Verifying a result you did not produce

1. Open the result JSON and read `config`, `torch`, and `seeds`.
2. Confirm `not_evidence` is `false` and `seeds` has at least 5 entries.
3. Check `results/final_summary.json` — it is regenerated from the committed
   files programmatically, so a value that disagrees with it indicates a file
   changed without the summary being rebuilt.
4. For tail-dependence quantities, check `n_tail` against `MIN_TAIL_SAMPLES =
   50` and the `reliable` flag. An unreliable estimate is reported rather than
   hidden, so it is your job not to quote it.

## Known reproducibility limits

These are properties of the measurements, not bugs. Full detail in
`docs/KNOWN_LIMITATIONS.md`.

- **The dependence estimator's bias exceeds its seed variance** (L10). Bias runs
  `+0.3345` at rho = 0 to `-0.2203` at rho = 0.95; across-seed std is `0.0075`
  to `0.0235`. Running more seeds tightens the error bar around a number that
  is systematically displaced. More seeds will not fix it.
- **Tail dependence is underpowered at project split sizes** (L11). At the
  headline quantile the full split leaves 60 exceedances against a floor of 50;
  a 300-sample split leaves 15. The failure shows as dispersion, not as a
  shifted mean, so checking the mean alone will not reveal it.
- **ENIV is validated at one design point** (L9): `n_views = 4`, synthetic,
  analytic rho. Nothing establishes its accuracy at other view counts.
