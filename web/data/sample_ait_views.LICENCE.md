# `sample_ait_views.npz` — what it is, and the licence it carries

**Derived from the AIT Log Data Set V2.0. Licensed CC BY-NC-SA 4.0.**

This file is a derivative of a CC BY-NC-SA 4.0 dataset, so the share-alike
term reaches it: it is offered under the same licence, for non-commercial use,
with the attribution below kept intact.

## It contains no raw logs

The arrays are per-window aggregate counts produced by Part 26's loader
(`prismflow/data/ait_lds.py`), not log lines. There is no message text, no
hostname, no IP address, no user name, no URL and no timestamp in the file —
`window_start` and the labels were deliberately left out, so the sample cannot
be realigned against the source capture.

| array | shape | what it is |
|---|---|---|
| `features` | `[400, 5, 10]` | per-window counts for five sensors × ten features |
| `view_mask` | `[400, 5]` | which sensors produced any record in that window |

Provenance: the first 400 windows of
`data/processed/ait_santos_network_300s/views.npy`, scenario `santos`, 300 s
windows, sources `apache, auth, dns, audit, suricata`, taken as a contiguous
head slice so the subset is stated rather than selected. The source directory
is gitignored; only this derived sample is committed.

sha256 of this file: `59b94e02a7a987256ee1913fa4f7b036d4411ef59ec469dae9d4b9e03b80ffe1`

## Required attribution

Quoted verbatim from `docs/part27-ait-datacard.md` section 2, which records it
from the Zenodo record:

> AIT Log Data Set V2.0. Max Landauer, Florian Skopik, Maximilian Frank,
> Wolfgang Hotwagner, Markus Wurzenberger, Andreas Rauber. Zenodo, 2022.
> doi:10.5281/zenodo.5789064. Licensed CC BY-NC-SA 4.0.

The record asks for two papers to be cited by anyone using the dataset:

> [1] M. Landauer, F. Skopik, M. Frank, W. Hotwagner, M. Wurzenberger, and
> A. Rauber. "Maintainable Log Datasets for Evaluation of Intrusion Detection
> Systems". IEEE Transactions on Dependable and Secure Computing, vol. 20,
> no. 4, pp. 3466-3482, doi:10.1109/TDSC.2022.3201582.

> [2] M. Landauer, F. Skopik, M. Wurzenberger, W. Hotwagner and A. Rauber,
> "Have it Your Way: Generating Customized Log Datasets With a Model-Driven
> Simulation Testbed", IEEE Transactions on Reliability, vol. 70, no. 1,
> pp. 402-415, March 2021, doi:10.1109/TR.2020.3031317.

Licence text: <https://creativecommons.org/licenses/by-nc-sa/4.0/>
Source record: <https://zenodo.org/records/5789064>

Nothing above is written for this file: every line is read from
`docs/part27-ait-datacard.md`, which took it from the Zenodo record.
