---
description: PrismFlow real-data track (Parts 26-33) - order, downloads, gates
---

# PrismFlow real-data track - index (Parts 26-33)

Do NOT implement anything from this file. Print it back to the user as a checklist,
then check which prerequisites below are met on disk and report each as MET / NOT MET.

## Order (one command per Claude Code session)
| Command | What it does | Needs | Stop for review? |
|---|---|---|---|
| /part26 | AIT-LDS v2.0 loader for russellmitchell (selective download) | internet | no |
| /part26 santos | same loader, second scenario (test set) | Part 26 | no |
| /part27 | window x sample-mode table + data card | both scenarios | YES - user picks window and mode |
| /part28 <window> <mode> | train naive + PrismFlow on AIT (clean) | Part 27 + window chosen | no |
| /part29 <window> <mode> | chorus attack, detector, honest-duplication, missingness on AIT | Part 28 | YES - verdict review |
| /part30 <days> | LANL cyber1 loader + repeat of 28-29 | LANL files fully downloaded | YES |
| /part31 | add real-data results to the existing web/ site | 28-29 results | YES - visual review |
| /part32 | live scoring API (serve/) + "Try it" section on the same web/ site | Part 31 | no |
| /part33 | final report | everything above | YES |

## Downloads (raw data goes in data/raw/, which is gitignored)
- AIT Log Data Set v2.0: https://zenodo.org/records/5789064 (CC BY-NC-SA 4.0). Two scenarios are
  needed for a leakage-free train/test split (train on one, test on another).
- LANL Comprehensive Multi-Source: https://csr.lanl.gov/data/cyber1/ (CC0, request form).
- Partially downloaded files end in .fdmdownload. Never parse a file until its download has finished
  and `gzip -t` / `unzip -t` passes.

## Prerequisite checks to run
1. data/raw/ contains a finished AIT scenario zip (no .fdmdownload suffix) and it passes `unzip -t`.
2. A second AIT scenario is present (needed by Part 28).
3. data/raw/ LANL files (auth, proc, flows, dns, redteam .txt.gz) finished and pass `gzip -t`.
4. Free disk space on the drive is at least 2x the largest unpacked scenario.
5. `pytest -q` passes on the current branch before any Part starts.
