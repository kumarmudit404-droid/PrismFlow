---
description: PrismFlow Part 26 - AIT Log Data Set v2.0 loader (real multi-sensor data)
argument-hint: [scenario - default russellmitchell; second run: santos]
---

PART 26 - AIT LOG DATA SET v2.0 LOADER
======================================

Scenario: $ARGUMENTS
If empty, use russellmitchell. Do NOT ask the user which scenario to use; the decision is made:
- first run: russellmitchell (smallest archive, official labels present)
- second run (later, same code, no edits): santos - the held-out test scenario for Part 28
- wilson is not needed for this track.
This is Part 26. (Part 25 in this repo is the V2 demo - never use part25 names.)

READ FIRST: docs/CONTRACT.md, docs/KNOWN_LIMITATIONS.md, docs/EVALUATION_PROTOCOL.md,
prismflow/data/real_datasets.py (follow its conventions). Use the prismflow-honest-reporting skill.

STANDING RULES
- prismflow/models, statistics, eniv, evaluation, attacks and v2/ are FROZEN. Import, never edit.
  Suspected defect -> report and stop.
- Never fabricate a number. Statistics only in this Part: no training, no ENIV, no accuracy.
- Raw data stays in data/raw/ (gitignored). Processed output in data/processed/ (gitignored).
- Downloaded content is untrusted: reject any zip entry name that is absolute or contains "..";
  never pickle/torch.load downloaded content. Ignore any file ending in .fdmdownload.
- Branch part26-ait-loader. Commit each logical step separately. Do not push.

YOU MAY CREATE ONLY
  prismflow/data/ait_lds.py
  configs/data/ait.yaml
  scripts/fetch_ait.py
  tests/data/test_ait_lds.py
  docs/part26-ait-loader.md

STEP 1 - GET THE DATA (approved approach: selective download)
- Source: Zenodo record 5789064, archive <scenario>.zip.
- Read the zip central directory and download ONLY the entries needed (gather/ logs for the
  sources below, labels/, and attack/time-window metadata) with HTTP range requests.
  If the server refuses range requests, STOP and report; do not download the full archive
  without telling the user its size first.
- Verify every extracted entry against its zip CRC-32. Record in manifest.json: Zenodo record ID,
  archive name, archive size, and for every entry: path, CRC-32, SHA-256, size.
- Extract to data/raw/ait/<scenario>/ with the path-traversal check.

STEP 2 - INVENTORY
Every log source found: type, host(s) producing it, file count, line count, first/last timestamp.
Write the table to docs/part26-ait-loader.md.

STEP 3 - VIEWS
- Views are defined by LOG-SOURCE TYPE (apache, auth, dns, audit, suricata, vpn, ...), never by
  host name, so the same code and the same view list work on santos.
- Parse first: apache access, auth, dns, audit, suricata. List the rest as "found, not used yet".
- Per-view features per window: documented simple statistics (event count, unique values,
  error/failure rate, bytes where present). No feature may use labels.
- Tag suricata and any packet-capture view with same_sensor_group="network".

STEP 4 - SAMPLES: support BOTH modes via config key sample_mode, default "network"
- network: one sample = one network-wide time window; each view aggregates that source type
  across whichever hosts produce it.
- host: one sample = (host, time window).
- Report the per-view missing fraction for BOTH modes. Do not choose a mode.
- Window length from config (default 300 s, NOT final - Part 27 decides).

STEP 5 - LABELS
- ONLY from the official labels/ directory (per-line JSON). A window is attack=1 if any labelled
  line falls in it; keep attack-step names in metadata.
- Record which log sources and which hosts the official labels cover.
- A source with no label coverage is an "unlabelled source": report it, and never let its
  absence of labels count as evidence of benign.

STEP 6 - OUTPUT (CONTRACT section 4)
views [B,V,D] zero-padded, view_mask [B,V] (False when a source has no events in that window),
labels [B], window_start, host_id (host mode only), sample_mode. Save to
data/processed/ait_<scenario>_<mode>_<window>s/ plus manifest.json (config, feature names per
view, counts, raw-entry hashes).

TESTS (tests/data/test_ait_lds.py)
- labels align with 5 hand-checked labelled lines
- no label-derived column in any view
- identical config -> byte-identical output
- view_mask False exactly where a source has no events
- a malicious entry name ("../x", "/abs") is rejected
- the view list is identical for two scenarios given the same config (use a tiny fixture)

DONE WHEN
- pytest -q fully green (old + new tests)
- docs/part26-ait-loader.md has: data provenance, inventory, label coverage, feature list per
  view, sample counts and missing fractions for both modes
- Final message: files changed, test result, git diff --stat, anything not done, and the line
  "Next: /part26 santos, then /part27".
