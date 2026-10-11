# Part 27 - AIT window length and sample mode

What this directory is: one script, `run_window_table.py`, that reads the Part
26 loader and writes `results/ait_window/window_table.{json,csv,md}`. Its job
is to put the window-length and sample-mode choice in front of the user with
the numbers attached. **It chooses nothing.**

## What was run

16 rows: 2 scenarios x 2 sample modes x 4 window lengths (30, 60, 300, 900 s).
Eight direct scans - one per (scenario, window) - at 46.8 to 50.0 s each
(`window_table.json`, `scan_seconds`); both sample modes are built from each
scan, so the logs are read once per window length rather than once per row.

The four lengths are exact multiples of 30 s, so `rewindow` could have re-binned
one 30 s scan into all of them. Each length was scanned directly anyway: it is
what the Part 27 brief asks for, and it keeps the table from depending on the
re-bin identity even though that identity is tested
(`tests/data/test_ait_lds.py::test_rewindow_equals_a_direct_scan`).

**No training, no prediction, no fusion, no ENIV, no accuracy, no
calibration.** Every column is a count, a fraction of counts, a run length or
an attack-step name.

**Nothing here is a 5-seed finding, and nothing here needs to be.** A scan is
deterministic: the same scenario, window and mode produce byte-identical
tensors (verified on real data in `docs/part26-ait-loader.md` section 17), so
each of the 16 rows is one exact measurement, not one sample of a distribution.
There is no seed to vary and no `+/-` to report. The 5-seed rule applies from
Part 28, where training introduces one.

## Column definitions that are not self-evident

| column | definition | why it is defined that way |
|---|---|---|
| `attack_fraction_of_labelled` | attack / `label_known` samples | A window no label-covered file contributed to is `-1 not labelled`, not benign. Dividing by `n_samples` would silently turn unexamined windows into negatives - 8,187 of them in russellmitchell host mode at 300 s. |
| `attack_runs` | maximal runs of consecutive attack windows **on the time grid**, per host in host mode, pooled | 664 consecutive windows of one exfiltration tool are one episode, not 664 observations. A dropped empty window breaks a run instead of joining the windows either side of it. |
| `attack_windows_dnsteal_only` | attack windows whose only steps are in `{dnsteal, dnsteal-received, dnsteal-dropped, attacker}` | Sits beside the attack fraction so neither can be quoted without the other. The set is named explicitly in the script, because which steps belong to one campaign is a reading of the scenario and not the loader's call. |
| `mean_events_per_present_window` | in-grid events of that source / samples where `view_mask` is True | Events outside the declared scenario interval (158,966 santos, 169,150 russellmitchell) are not in any window and must not inflate a per-window mean. |
| `usable_both_classes` | at least 30 attack **and** 30 benign `label_known` windows | The brief's threshold. It is met by all 16 rows, so it does not discriminate - see the trade-offs below. |

## Cross-check against Part 26

The 300 s rows reproduce the committed Part 26 counts exactly, for both
scenarios and both modes:

| row | this table | `data/processed/ait_<scenario>_<mode>_300s/manifest.json` |
|---|---|---|
| russellmitchell network 300 s | 1,152 / 1,031 / 114 / 7 | 1,152 / 1,031 / 114 / 7 |
| russellmitchell host 300 s | 9,834 / 1,034 / 613 / 8,187 | 9,834 / 1,034 / 613 / 8,187 |
| santos network 300 s | 1,152 / 667 / 441 / 44 | 1,152 / 667 / 441 / 44 |
| santos host 300 s | 9,779 / 669 / 939 / 8,171 | 9,779 / 669 / 939 / 8,171 |

(samples / attack / benign / not labelled.) These are two independent scans of
the same bytes by the same code, so agreement is a consistency check, not new
evidence.

## The trade-offs

All figures from `results/ait_window/window_table.md`. **This section explains;
it does not choose.**

### Short windows (30-60 s): the views go dark

Mean views present falls to **2.95 of 5** (russellmitchell network 30 s) and
**2.81** (santos 30 s), against 4.05 and 3.99 at 300 s. The sparse sources are
the ones that disappear: `auth` is missing in **93.6%** of 30 s network windows
in both scenarios and `audit` in 80.9% / 69.7%, because PAM and auditd emit a
handful of records a minute. At 30 s, `auth` averages 7.2 events in the windows
where it appears at all.

This cuts both ways for PrismFlow. Missingness is exactly what `view_mask` and
the ENIV discount are built to handle, so short windows give that machinery the
most structure to work with - and they also give every view fewer events to
compute a feature from, so the features themselves get noisier.

Short windows also make the **class balance worse, not better**: the attack
fraction of labelled windows *rises* to 0.956 (russellmitchell network 30 s)
and 0.741 (santos 30 s), because the exfiltration is continuous and so lands in
nearly every short window. The unlabelled count rises with it - 742 and 2,547
windows at 30 s network, against 7 and 44 at 300 s - because a short window is
more likely to contain no line from any label-covered file at all.

### Long windows (900 s): the missingness structure collapses

At 900 s network, **only `apache` is ever missing** (0.161 russellmitchell,
0.286 santos); `auth`, `dns`, `audit` and `suricata` are present in every
window. Mean views present is 4.84 and 4.71 of 5. A fusion method whose point
is handling absent views has almost nothing to handle here.

The sample count drops to **384** per scenario, and russellmitchell's benign
class drops to **40 windows**. The attack fraction barely moves (0.896 and
0.583 at 900 s against 0.900 and 0.602 at 300 s) - the usual "long windows
dilute the attack" intuition does not apply when the positive label is a
continuously-running tool.

### Host mode: more samples, fewer views, far less ground truth

Host mode multiplies samples by about 8.5 (9,834 and 9,779 at 300 s) and cuts
mean views present to **1.75 / 1.74 of 5**, because only 3 of 10 hosts run
Apache and only 2 run a resolver - those views are missing for every sample
from every other host regardless of what happened. Per-view missing fractions
are 0.74-0.82 at 300 s.

It is also where ground truth thins out most: **8,187 of 9,834** and **8,171 of
9,779** samples are not labelled at 300 s, leaving 1,647 and 1,608 usable,
because only 3 of 10 hosts carry a labelled file.

In exchange it gives the most balanced labelled sets in the whole table -
santos host at 900 s is 0.280 attack (226 attack, 581 benign) and
russellmitchell host at 900 s is 0.429 - since a host that logged nothing
attack-related in a window becomes a benign sample in its own right.

### Independent attack episodes: the one thing no window length fixes

| scenario | mode | 30 s | 60 s | 300 s | 900 s |
|---|---|---|---|---|---|
| russellmitchell | network | **1** | **1** | **1** | **1** |
| russellmitchell | host | 4 | 4 | 4 | 4 |
| santos | network | 5 | 4 | **3** | 3 |
| santos | host | 8 | 7 | 5 | 5 |

russellmitchell's 10,305 attack windows at 30 s are **one** contiguous block of
the same exfiltration. Shortening the window multiplies the attack windows and
does not add an episode. So:

- The positive class's effective sample size is the run count, not the
  attack-window count.
- The brief's 30-attack / 30-benign threshold is satisfied by all 16 rows and
  is therefore **not the binding constraint**; the episode count is.
- A random window split trains and tests on neighbours inside one run.
- A chronological split on santos is close to a label split, because the one
  long run starts at the first window of the grid.

### And the label means something narrower than "attack"

In every one of the 16 rows, **98.2% to 99.9%** of attack windows carry only
dnsteal-family steps. The multi-step intrusion the dataset's own record
declares as its attack time - 98 minutes on russellmitchell, 44 on santos -
reaches 13 windows at 30 s and 4-5 at 300 s. `docs/part27-ait-datacard.md`
section 6 states this with both scenarios' attacker configuration beside it.

## Reproducing

```
python experiments/ait_window/run_window_table.py            # ~7 min, 8 scans
python experiments/ait_window/run_window_table.py --scenarios santos --windows 300
```

Requires `python scripts/fetch_ait.py <scenario>` to have run (Part 26); the
script prints `PENDING: <scenario> is not fetched` and omits that scenario
rather than guessing at its rows, and stops outright if neither is present.

Outputs: `results/ait_window/window_table.json` (every column, full run-length
lists, provenance, scan times), `.csv` (one flat row per
scenario/mode/window) and `.md` (the eight tables printed at the review gate).
