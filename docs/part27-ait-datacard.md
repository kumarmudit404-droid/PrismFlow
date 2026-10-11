# AIT Log Data Set v2.0 - data card (PrismFlow Parts 26-29)

Scope: the two scenarios PrismFlow has loaded, `russellmitchell` and `santos`.
Statistics here are counts and fractions of counts only. **No experiment, no
training, no ENIV, no accuracy and no calibration number appears in this
Part**, and none has been measured on this data.

Every figure cites its source: `results/ait_window/window_table.json` (this
Part), `data/processed/ait_<scenario>_<mode>_300s/manifest.json` and
`data/raw/ait/<scenario>/manifest.json` (Part 26, gitignored; the CRC-32 and
SHA-256 there tie a number to specific bytes), the archive's own
`processing/config/attacker.yaml` and `dataset.yaml`, or the Zenodo record
metadata.

---

## 1. Source and identity

| field | value |
|---|---|
| name | AIT Log Data Set V2.0 |
| DOI | [10.5281/zenodo.5789064](https://doi.org/10.5281/zenodo.5789064) |
| record | <https://zenodo.org/records/5789064> |
| version | `v2_0` |
| published | 2022-02-24 |
| creators | Landauer, Max; Skopik, Florian; Frank, Maximilian; Hotwagner, Wolfgang; Wurzenberger, Markus (AIT Austrian Institute of Technology); Rauber, Andreas (Vienna University of Technology) |
| keywords | log data, intrusion detection |
| scenarios in record | `fox`, `harrison`, `russellmitchell`, `santos`, `shaw`, `wardbeck`, `wheeler`, `wilson` (8) |
| archives used here | `russellmitchell.zip` 7,132,670,599 B; `santos.zip` 10,004,843,924 B |

Source of this table: the Zenodo REST record for 5789064 (`metadata.title`,
`.version`, `.publication_date`, `.creators`, `.keywords`, `.license`, and
`files[].key`/`.size`), read at fetch time.

**The data is synthetic.** The record states it plainly: "This repository
contains synthetic log data suitable for evaluation of intrusion detection
systems, federated learning, and alert aggregation", collected from eight
testbeds with simulated normal user behaviour and a scripted attack sequence.
It is real log *format* and real tooling output from a real (virtual) network;
it is not production traffic, and no claim here should be read as one.

## 2. Licence and required attribution

**Licence: Creative Commons Attribution-NonCommercial-ShareAlike 4.0
International (CC BY-NC-SA 4.0)** - `metadata.license.id` is `cc-by-nc-sa-4.0`
on the Zenodo record. Licence text: <https://creativecommons.org/licenses/by-nc-sa/4.0/>.

What that binds this project to: attribution, non-commercial use only, and
share-alike on anything derived from the data. No raw or derived AIT data is
committed to this repository - `data/raw/` and `data/processed/` are gitignored
- so what PrismFlow redistributes is code and statistics, not the dataset.

Attribution text to use:

> AIT Log Data Set V2.0. Max Landauer, Florian Skopik, Maximilian Frank,
> Wolfgang Hotwagner, Markus Wurzenberger, Andreas Rauber. Zenodo, 2022.
> doi:10.5281/zenodo.5789064. Licensed CC BY-NC-SA 4.0.

**The record asks for two papers to be cited** ("If you use the dataset, please
cite the following publications", verbatim from the record description):

> [1] M. Landauer, F. Skopik, M. Frank, W. Hotwagner, M. Wurzenberger, and
> A. Rauber. "Maintainable Log Datasets for Evaluation of Intrusion Detection
> Systems". IEEE Transactions on Dependable and Secure Computing, vol. 20,
> no. 4, pp. 3466-3482, doi:10.1109/TDSC.2022.3201582.

> [2] M. Landauer, F. Skopik, M. Wurzenberger, W. Hotwagner and A. Rauber,
> "Have it Your Way: Generating Customized Log Datasets With a Model-Driven
> Simulation Testbed", IEEE Transactions on Reliability, vol. 70, no. 1,
> pp. 402-415, March 2021, doi:10.1109/TR.2020.3031317.

Both must appear in any PrismFlow write-up that reports an AIT number,
including Part 33.

## 3. Scenarios used

| | `russellmitchell` | `santos` |
|---|---|---|
| simulation interval (record, and the archive's `dataset.yaml`) | 2022-01-21 00:00 - 2022-01-25 00:00 | 2022-01-14 00:00 - 2022-01-18 00:00 |
| **attack time declared by the record** | **2022-01-24 03:01 - 04:39** (98 min) | **2022-01-17 11:15 - 11:59** (44 min) |
| scan volume (record) | Low | Low |
| unpacked size (record) | 14 GB | 17 GB |
| archive size (Zenodo `files[].size`) | 7,132.7 MB | 10,004.8 MB |
| entries fetched by Part 26 | 91 | 86 |
| hosts producing a parsed view | 10 | 10 |
| window grid at 300 s | 1,152 windows | 1,152 windows |

`russellmitchell` is PrismFlow's training scenario and `santos` the held-out
one (`configs/data/ait.yaml`: `scenario`, `held_out_scenario`); Part 28 runs
both directions. The other six scenarios are not loaded, and nothing here
establishes that they behave like these two.

## 4. Views

Five views, one per log-source **type**, never per host, so the same view list
loads any scenario (`prismflow/data/ait_lds.py`, `SOURCES`;
`results/ait_window/window_table.json`, `view_identity`):

| view | index | sensor | native width | same_sensor_group |
|---|---|---|---|---|
| `apache` | 0 | web server (Apache2 access log) | 9 | - |
| `auth` | 1 | host authentication (PAM / sshd via syslog) | 8 | - |
| `dns` | 2 | DNS resolver (dnsmasq) | 9 | - |
| `audit` | 3 | Linux kernel audit subsystem (auditd) | 9 | - |
| `suricata` | 4 | network IDS (Suricata, AF-packet tap) | 10 | **`network`** |

Tensors are `[B, V, D]` with `D = 10`, each view right-padded and declaring its
native `input_dim` so no encoder reads a padded column (CONTRACT section 4).
The view signature is **identical across both scenarios**, compared per row
rather than assumed (`view_identity.identical_across_scenarios: true`).

**Same-sensor group.** `sensor_groups()` reports `{"network": [4]}`:
`suricata` and the raw packet capture `pcap` are the same AF-packet tap on the
same interface and are therefore not independent observations. Only `suricata`
is built; `pcap` is 7.0 GB (russellmitchell) / 10.1 GB (santos) of the archive
and no view parses it. The group exists so that a future packet view joins an
existing statement about shared sensors instead of arriving as an apparently
independent view. **Part 26 and 27 report the grouping and do not act on it.**

Twenty-one further source types are present and unparsed, including three AIT
labels (`apache_error`, `openvpn`, `monitoring`); the full census is in
`docs/part26-ait-loader.md` sections 4 and 14.

## 5. Features

8-10 simple statistics per view per window - event counts, distinct-value
counts, failure/error rates, byte and name-length magnitudes. Counts and
magnitudes are `log1p`-transformed, fractions are left in `[0, 1]`, and
**no feature reads a label**; that separation is checked behaviourally
(`tests/data/test_ait_lds.py::test_no_label_derived_column_in_any_view`). No
standardisation is applied, because a scaler must be fitted on training indices
only and that is Part 28's business. The full list is
`docs/part26-ait-loader.md` section 6.

Features are a first documented choice, not a tuned one, and no
feature-selection or representation-quality evidence exists for them.

## 6. Label method, and what the label means

**Ground truth comes only from the official `labels/` tree**: one JSON object
per line naming a 1-based line number in the corresponding `gather/` file, the
attack steps for that line, and the labelling rules that fired. Nothing else is
used - not filenames, not attack time ranges, not the attacker host's own logs.
A window is `attack = 1` if **any** labelled line falls in it, and the step
names are kept per window.

### The label is "was the DNS exfiltration tool running", not APT detection

This is the single most important fact on this card, and it has to be stated
beside every number computed from these labels:

| | `russellmitchell` | `santos` |
|---|---|---|
| attack windows, 300 s network | 1,031 of 1,152 | 667 of 1,152 |
| attack fraction of **labelled** windows | 0.900 | 0.602 |
| attack windows carrying **only** dnsteal-family steps | **1,026** | **663** |
| dnsteal-only share of attack windows | **0.995** | **0.994** |
| attack windows carrying any other step | 5 | 4 |

`results/ait_window/window_table.json`, rows at `window_seconds: 300`,
`sample_mode: network`. dnsteal-family is the set
`{dnsteal, dnsteal-received, dnsteal-dropped, attacker}`, named explicitly in
`experiments/ait_window/run_window_table.py` because which steps belong to one
campaign is a reading of the scenario, not something the loader should decide.

The dnsteal-only share is **0.982 to 0.999 in all 16 rows of the window
table** - every scenario, mode and window length. The cause is in the archive's
own attacker configuration:

| `processing/config/attacker.yaml` | `russellmitchell` | `santos` |
|---|---|---|
| `dnsteal.start` | 2022-01-20T12:48:24Z | 2022-01-13T13:20:50Z |
| `dnsteal.stop` | 2022-01-24T13:50:03Z | 2022-01-17T09:48:02Z |
| `exfiltration_service.stop` | 2022-01-24T13:50:39Z | 2022-01-16T07:16:18Z |
| `dnsteal.domain` | `email-19.kennedy-mendoza.info` | `kiirjekuimkpcrcioastvkeodpnimx.biz` |

On `russellmitchell` the exfiltration spans the whole declared interval, so the
labels do too. On `santos` it stops 1 day 16.7 h before the capture ends, which
is the only reason that scenario has 441 benign windows at 300 s.

### The record's own attack time is the intrusion, and it is minutes long

The non-dnsteal attack windows coincide **exactly** with the attack interval
the Zenodo record declares (measured from
`data/processed/ait_<scenario>_network_300s/attack_steps.json` and
`window_start.npy`):

| scenario | record's declared attack time | 300 s windows carrying a non-dnsteal step |
|---|---|---|
| `russellmitchell` | 2022-01-24 03:01 - 04:39 | 03:00, 03:35, 03:55, 04:35 (+ 13:50, `exfiltration-service` stop) |
| `santos` | 2022-01-17 11:15 - 11:59 | 11:15, 11:20, 11:55 (+ 2022-01-16 07:15, `exfiltration-service` stop) |

So the dataset's authors describe the attack as a 44-98 minute multi-step
intrusion, while "any labelled line in the window" marks 58-90% of the capture.
Both readings come from the same official labels; they are answers to different
questions. **A window-level binary task built on "any labelled line" is
therefore not the dataset's declared attack interval**, and Part 28 must say
which question it is answering. An alternative task definition - positives
restricted to the declared attack interval - exists and is cheap; Part 27
reports that it exists and does not choose it.

### Three label states, because absence of a label is not evidence of benign

The record says unlabelled lines "can be considered to be labeled as normal".
That holds **line-wise inside a log the labelling examined**. It does not
extend to a log the labelling never looked at, and `suricata` is exactly that:
the archive's `rules/` carries labelling rules for `0_auth`, `apache`, `audit`,
`dnsmasq`, `monitoring` and `openvpn`, and **none for suricata**. There is no
suricata label file in either scenario.

So the loader carries `labels` in `{1 attack, 0 benign, -1 not labelled}` plus
a `label_known` mask:

- a window a **label-covered file** contributed to, with no labelled line, is
  benign - those lines were examined and marked clean;
- a window **no label-covered file** contributed to is **not labelled** - the
  labelling never examined it, and scoring it as benign would invent negatives.

**Part 28 and 29 must filter on `label_known`.** Unlabelled counts at 300 s:
7 of 1,152 (russellmitchell network), 44 of 1,152 (santos network), 8,187 of
9,834 (russellmitchell host) and 8,171 of 9,779 (santos host). Every
`attack_fraction` in the window table is over `label_known` samples only.

### Label coverage, per file and not per source

AIT labels 8 files on 5 hosts in each scenario, covering six source types, of
which Part 26 parses five. Coverage is per **file**: `intranet_server`'s
labelled Apache log is `...-access.log.2` on russellmitchell and the current
`...-access.log` on santos, while the other rotations of the same log on the
same host carry no labels.

| parsed view | label files | labelled hosts | labelled attack lines (rm / santos) |
|---|---|---|---|
| `apache` | 1 | `intranet_server` | 7,695 / 7,794 |
| `auth` | 1 | `intranet_server` | 8 / 12 |
| `dns` | 1 | `inet-firewall` | 54,035 / 39,426 |
| `audit` | 2 | `internal_share`, `intranet_server` | 11 / 21 |
| `suricata` | **0** | **-** | **0 / 0** |

`docs/part26-ait-loader.md` sections 5 and 15. Totals: 61,749 labelled attack
lines on russellmitchell, 47,253 on santos, dominated in both by the one DNS
log (87.5% and 83.4%).

**`suricata` is an unlabelled source.** It is the only network-sensor view, its
features are usable, and its quiet windows are unexamined rather than clean. It
can never contribute to a window's benign status.

## 7. Sample construction

A log data set has no natural row, so the unit of analysis is constructed. Both
readings are implemented, reported, and **not chosen here**:

| | `sample_mode: network` | `sample_mode: host` |
|---|---|---|
| one sample is | one network-wide time window | one (host, time window) |
| a view aggregates | that source type across every host producing it | that source type on that host alone |
| distinct-value features | union of the value sets across hosts | that host's own set |
| a view is missing when | no host logged that source in the window | that host did not log that source in the window |

Window length is a config key, swept over 30 / 60 / 300 / 900 s in
`results/ait_window/window_table.md`. Empty windows are dropped
(`drop_empty_windows: true`), which is why a dropped window breaks a contiguous
run rather than joining the windows either side of it.

## 8. Independent attack episodes: the number that limits everything

Attack windows are not independent observations. Maximal runs of consecutive
attack windows on the time grid (`window_table.json`, `attack_runs`):

| scenario | mode | 30 s | 60 s | 300 s | 900 s |
|---|---|---|---|---|---|
| `russellmitchell` | network | **1** run (10,305) | **1** (5,153) | **1** (1,031) | **1** (344) |
| `russellmitchell` | host | 4 runs (10,305/7/2/1) | 4 (5,153/4/2/1) | 4 (1,031/1/1/1) | 4 (344/1/1/1) |
| `santos` | network | 5 runs (6,633/8/2/1/1) | 4 (3,317/4/2/1) | **3** (664/2/1) | 3 (222/1/1) |
| `santos` | host | 8 runs | 7 runs | 5 runs | 5 runs |

`russellmitchell` has **one** contiguous attack episode at every window length
in network mode: 10,305 attack windows at 30 s are one block of the same
exfiltration. `santos` has three to five. No window length changes this by more
than a handful, because it is a property of the scenario and not of the
binning.

Consequences, all of which belong next to any Part 28 number:

- The effective positive sample size is the **run count**, not the
  attack-window count. A `+/-` over 5 seeds is a spread over initialisations,
  **not** over independent attacks.
- A **random** window split trains and tests on neighbouring windows of one
  run, which leaks.
- A **chronological** split on `santos` puts nearly all attack windows before
  nearly all benign windows, so early/late is close to a label split.
- The brief's own usability threshold - at least 30 attack and 30 benign
  windows - is met by **all 16 rows** (`window_table.md` section 7), so it is
  not the binding constraint here. The episode count is.

## 9. Known limitations

1. **Synthetic data from a simulation testbed**, stated by the record itself.
   Normal behaviour is simulated; the attack is scripted.
2. **The window label is close to a single-view signal.** 98-99.9% of attack
   windows in every row carry only the DNS-exfiltration steps, and that
   exfiltration is visible directly in `dns` as query names up to 250
   characters carrying the exfiltrated file name. A window-level result on this
   task is weak evidence about multi-view fusion.
3. **The declared attack interval is 44-98 minutes**, so the multi-step
   intrusion reaches 3-5 windows of 1,152 at 300 s. Anything said about *that*
   activity is a handful-of-windows result.
4. **`suricata`, the only network-sensor view, is unlabelled** in both
   scenarios.
5. **One to five independent attack episodes per scenario** (section 8).
6. **Partial label coverage**: 5 of 10 hosts and 8 files per scenario. In host
   mode that leaves 83% of samples unlabelled at 300 s.
7. **`rho_matrix` is `None`.** Real views have no ground-truth cross-view
   dependence, so ENIV can be APPLIED on this data and never VALIDATED on it
   (`docs/KNOWN_LIMITATIONS.md` L9, `docs/DATASETS.md`).
8. **L14 is not closed by loading this data.** L14 asks for the redundancy
   audit on unrelated benchmarks; Parts 26-27 report counts only. Whether these
   five sensor views are less redundant than `handwritten`'s six pixel-derived
   ones is Part 28's measurement.
9. **No packet-capture view**, so the `network` same-sensor group has one
   member and the near-duplicate view pair that would most directly exercise
   the project's redundancy thesis does not exist yet.
10. **158,966 (santos) and 169,150 (russellmitchell) events fall outside the
    declared interval** and are excluded by the default grid. No labelled
    attack line is among them; `time_range: observed` would change the sample
    counts.
11. **Two scenarios of eight**, and the window table's 16 rows are one scan
    each - deterministic, so there is nothing to average over, and no row here
    is a 5-seed finding or claims to be.
