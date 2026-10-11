# Part 26 - AIT Log Data Set v2.0 loader

Scenarios: **russellmitchell** (part A, first run) and **santos** (part B,
second run, no code or config edit between them). This Part delivers a loader
for the AIT Log Data Set v2.0 and its dataset statistics, and nothing else.

- **Part A - russellmitchell**, sections 1-11 below.
- **Part B - santos**, from section 12. Part B is the held-out scenario for
  Part 28 and was loaded with the same committed code; where a Part A statement
  is superseded, Part B says so.

**No experiment was run. No ENIV, dependence, accuracy or calibration number
appears here**, by the brief's instruction and because none has been measured
on this data. Every figure below is a count, a timestamp or a fraction of
counts, produced by `prismflow/data/ait_lds.py` and recorded in
`data/processed/ait_<scenario>_<mode>_300s/manifest.json`.

Code: `prismflow/data/ait_lds.py`, `scripts/fetch_ait.py`,
`configs/data/ait.yaml`, `tests/data/test_ait_lds.py` (58 tests).

---

# Part A - russellmitchell

## 0. The one thing to read before Part 27 or 28 (russellmitchell)

At 300 s network windows, **1,031 of the 1,152 windows are labelled attack**:
89.5% of all windows, or **90.0% of the 1,145 windows the labelling covers**
(1,031 attack + 114 benign; 7 windows are unlabelled). That number is not the
useful quantity, and taking it at face value would misread the dataset.

A window is attack if **any** labelled line falls inside it, and
`processing/config/attacker.yaml` puts the `dnsteal` DNS-exfiltration run at
**2022-01-20T12:48 to 2022-01-24T13:50** - essentially the entire capture. So:

| | attack windows |
|---|---|
| carrying only dnsteal-family steps | **1,026** |
| carrying at least one other step | **5** |

The dnsteal family counted here is `dnsteal`, `dnsteal-received`,
`dnsteal-dropped`, `attacker` (`AITSamples.attack_windows_beyond` takes the set
from the caller; this is the set used). Full breakdown over attack windows:

| attack step | windows | | attack step | windows |
|---|---|---|---|---|
| `attacker` | 1,031 | | `dns_scan` | 1 |
| `dnsteal` | 1,031 | | `traceroute` | 1 |
| `dnsteal-received` | 1,031 | | `attacker_http` | 1 |
| `foothold` | 3 | | `dirb` | 1 |
| `network_scan` | 2 | | `service_scan` | 1 |
| `escalate` | 2 | | `webshell_cmd` | 1 |
| `dnsteal-dropped` | 1 | | `webshell_upload` | 1 |
| `exfiltration-service` | 1 | | `wpscan` | 1 |
| `attacker_change_user` | 1 | | `escalated_command` | 1 |
| `escalated_sudo_command` | 1 | | `escalated_sudo_session` | 1 |

**What this means for the window-level task.** A classifier that always
predicts "attack" scores 90.0% on the labelled windows, and the label is
almost entirely "was the continuous DNS exfiltration running" - a signal
visible in one view (`dns`), where the exfiltration shows up directly as very
long query names. The multi-step intrusion activity that would actually
require several views to see is confined to **5 of 1,152 windows**. A
window-level binary result on this scenario, at this window length, would
therefore say very little about multi-view fusion.

This is reported here rather than discovered during Part 28. It is a property
of the scenario and the window-level framing, not a defect in the loader, and
Part 27 should weigh it when choosing the window length and the task.

---

## 1. Why this dataset

`docs/KNOWN_LIMITATIONS.md` **L14** is the standing limitation this Part exists
to start closing:

> There is one real dataset: `handwritten` ... The measured redundancy (6
> nominal views carrying 3.28 effective views) is concentrated in one block of
> the dependence matrix: pix-fac 0.852, pix-kar 0.823, fac-kar 0.707 ... Those
> three high pairs are all pixel-derived descriptors of the same glyph. **A
> benchmark whose views come from genuinely different sensors need not behave
> this way.**

L14 asks for the same audit on unrelated benchmarks, "ideally with views from
different sensing modalities rather than different transforms of one image".
AIT is that: a web server's access log, a host's PAM authentication log, a DNS
resolver's query log, a kernel audit trail and a network IDS are five different
instruments observing one network over four days.

Two consequences that are **not** results of this Part:

- Whether these views are less redundant than `handwritten`'s is open. Part 28
  measures it; no ENIV is reported here.
- `rho_matrix` is `None`, as for `handwritten`. Real views have no
  ground-truth cross-view dependence, so ENIV can be APPLIED here and never
  VALIDATED here (`docs/DATASETS.md`, "THE LIMITATION, STATED PLAINLY";
  `docs/KNOWN_LIMITATIONS.md` L9).

---

## 2. Data provenance

| field | value |
|---|---|
| Zenodo record | 5789064 |
| archive | `russellmitchell.zip` |
| archive size | 7,132.7 MB |
| entries in archive | 14,911 |
| entries fetched | **91** |
| transferred, compressed | **86.7 MB** |
| written to disk, uncompressed | 2,322.0 MB |
| manifest entries with CRC-32 + SHA-256 | 91 |
| fetch complete | true |

The per-scenario archives on record 5789064 run 7.1 GB (russellmitchell) to
26.5 GB (wilson), and most of that is raw packet capture no Part 26 view
parses. Zenodo's file endpoint honours HTTP range requests, so
`scripts/fetch_ait.py` reads the archive's ZIP central directory over ranges
and fetches only the members the loader's own registry recognises - 91 of
14,911 entries, 86.7 MB instead of 7.1 GB.

Integrity and safety:

- Every member is verified against the **CRC-32 in the archive's own central
  directory** before being written. The write goes to a `.part` file renamed
  only after the CRC matches, so an interrupted fetch cannot leave a short file
  that a later run's hash check would be asked to trust.
- `data/raw/ait/russellmitchell/manifest.json` records the Zenodo record ID,
  archive name and size, and each entry's path, CRC-32, SHA-256, uncompressed
  size and compressed size.
- Member names are untrusted remote input. `safe_extract_path` rejects
  absolute POSIX paths, drive-qualified and UNC paths, backslash separators,
  NUL bytes and any `..` component, then re-checks the resolved path against
  the destination. 12 cases are tested.
- Nothing unpickles, executes or `torch.load`s downloaded content;
  `load_processed` reads `.npy` with `allow_pickle=False`.
- If the endpoint ever stops honouring ranges, the fetch raises
  `RangeNotSupported` rather than falling back to a multi-gigabyte download.
- A partially fetched scenario is **refused** by `scan_scenario`, because its
  missing views would be missing for want of a download rather than for want
  of events.

`data/raw/` and `data/processed/` are gitignored; the hashes in the manifests
are what ties a reported number to specific bytes.

---

## 3. Inventory: parsed sources

One view per log source type. Window 300 s, grid from the archive's own
`dataset.yaml` (2022-01-21T00:00:00Z to 2022-01-25T00:00:00Z, 1,152 windows).

| source | sensor | hosts | files | lines | events parsed | unparsed | outside grid | first event | last event |
|---|---|---|---|---|---|---|---|---|---|
| `apache` | web server (Apache2 access log) | 3 | 30 | 103,337 | 103,337 | 0 | 0 | 2022-01-21 05:55:22Z | 2022-01-24 21:57:49Z |
| `auth` | host authentication (PAM / sshd via syslog) | 10 | 20 | 6,818 | 6,818 | 0 | 1,499 | 2022-01-20 11:12:22Z | 2022-01-25 07:02:47Z |
| `dns` | DNS resolver (dnsmasq) | 2 | 2 | 540,882 | 540,882 | 0 | 37,384 | 2022-01-20 11:21:42Z | 2022-01-25 06:56:31Z |
| `audit` | Linux kernel audit subsystem (auditd) | 7 | 7 | 18,740 | 18,740 | 0 | 0 | 2022-01-21 00:00:01Z | 2022-01-24 23:45:01Z |
| `suricata` | network IDS (Suricata, AF-packet tap) | 7 | 8 | 1,115,793 | 1,115,793 | 0 | 130,267 | 2022-01-20 13:13:05Z | 2022-01-25 06:57:01Z |

**1,785,570 events parsed, 0 lines unparsed.** The zero is load-bearing: two
format variants were found only because the unparsed count was reported rather
than assumed to be zero.

- `dnsmasq` logs **without a hostname field**
  (`Jan 21 00:00:09 dnsmasq[3468]: query[A] ...`), so a syslog pattern that
  requires one classifies every one of the 540,882 DNS lines as unparsed.
- `cloud_share`'s only non-empty Apache log is `other_vhosts_access.log.{1..4}`
  (27,500 requests) in Debian's **combined-vhost** format, where the first
  field is `hostname:port` and not the client. Requiring the client-first shape
  dropped that host from the apache view entirely while reporting the lines
  merely as "unparsed".

**169,150 events fall outside the window grid** (`auth` 1,499, `dns` 37,384,
`suricata` 130,267). They are counted, not silently dropped. The cause is that
logging and the attack both start before the declared scenario interval:
Suricata's first event is 2022-01-20T13:13Z and dnsteal starts
2022-01-20T12:48Z, while `dataset.yaml` declares the scenario from
2022-01-21T00:00Z. `time_range: observed` is available if Part 27 wants the
grid taken from the first and last parsed event instead. **No labelled attack
line falls outside the grid or fails to parse** - the loader reports
`labelled_attack_lines_in_grid`, `_unparsed` and `_outside_grid` per file for
exactly this check.

### Hosts producing each parsed source

| source | hosts |
|---|---|
| `apache` | `cloud_share`, `intranet_server`, `webserver` |
| `auth` | `cloud_share`, `davey_mail`, `inet-dns`, `inet-firewall`, `internal_share`, `intranet_server`, `mail`, `morris_mail`, `vpn`, `webserver` |
| `dns` | `inet-dns`, `inet-firewall` |
| `audit` | `cloud_share`, `inet-firewall`, `internal_share`, `intranet_server`, `mail`, `vpn`, `webserver` |
| `suricata` | `cloud_share`, `inet-firewall`, `internal_share`, `intranet_server`, `mail`, `vpn`, `webserver` |

Ten hosts in total, and only three run Apache - which is why the host-mode
mask below is so structured. **No attacker or client host contributes to any
parsed view**: `attacker_0`, `ext_user_*`, `remote_employee_*` and
`internal_employee_*` produce none of these five source types, so no
attacker-side log enters a defender view. This is a fact about this scenario,
checked against the archive census, not something the loader enforces.

---

## 4. Found, not used yet

Every log source type present in the archive that Part 26 does not parse. The
census comes from the archive's ZIP central directory (recorded as
`archive_sources` in the raw manifest), not from disk - a disk census would
report all of these as absent, since their files were never fetched. Line
counts and timestamps are therefore **not available** for these rows and are
not estimated.

| source | sensor | files | hosts | archive bytes | why not used |
|---|---|---|---|---|---|
| `pcap` | Suricata AF-packet tap (`same_sensor_group: network`) | 11 | 7 | 6,961.9 MB | raw packet capture; 4.4 GB compressed for the labelled hosts alone, and no Part 26 view parses it |
| `suricata_stats` | Suricata (`same_sensor_group: network`) | 8 | 7 | 1,689.8 MB | capture and decoder counters every ~8 s |
| `monitoring` | host metrics agent (metricbeat via Logstash) | 158 | 1 | 1,842.9 MB | labelled by AIT for one host-day only |
| `attacker_wpdiscuz` | attacker tooling (wpDiscuz exploit) | 4 | 1 | 871.7 MB | attacker-side ground truth by construction; not a defender view |
| `dnsteal` | attacker tooling (dnsteal) | 34 | 1 | 512.2 MB | attacker-side ground truth by construction; not a defender view |
| `journal` | systemd journal | 18 | 7 | 159.4 MB | binary; needs `journalctl` |
| `attacker_downloads` | attacker tooling | 45 | 4 | 109.0 MB | attacker-side ground truth by construction |
| `logstash_self` | Logstash's own pipeline log | 15 | 1 | 67.6 MB | shipper self-log |
| `mail` | Exim4 / sendmail | 42 | 13 | 31.9 MB | SMTP delivery, queue and rejection records |
| `horde` | webmail (Horde) | 6 | 3 | 13.5 MB | application log |
| `syslog` | host syslog | 43 | 10 | 5.3 MB | overlaps `auth`, `kern` and daemon output |
| `user_log` | host syslog (user facility) | 6 | 3 | 2.2 MB | |
| `messages` | host syslog (catch-all) | 6 | 3 | 2.3 MB | |
| `suricata_fast` | Suricata (`same_sensor_group: network`) | 8 | 7 | 2.1 MB | one-line alert summaries; redundant with `suricata` by construction |
| `suricata_self` | Suricata (`same_sensor_group: network`) | 14 | 7 | 1.5 MB | engine self-log |
| `openvpn` | VPN gateway | 1 | 1 | 0.5 MB | **labelled by AIT**; a candidate sixth parsed view |
| `apache_error` | Apache2 error log | 22 | 3 | 0.4 MB | **labelled by AIT**; the brief's starting set is the access log |
| `kern` | kernel ring buffer | 2 | 1 | 0.2 MB | |
| `redis` | Redis | 3 | 1 | <0.1 MB | |
| `shorewall` | host firewall init | 2 | 1 | <0.1 MB | |
| `attacks_log` | attacker tooling | 1 | 1 | <0.1 MB | attacker-side ground truth by construction |

All 516 `gather/*/logs/` files in the archive classify into one of the 26
source types; the census reports an `__unclassified__` bucket and it is
**empty**, so the inventory above has no hole.

### Same-sensor group

`suricata` and `pcap` both carry `same_sensor_group: "network"` - they are the
same AF-packet tap on the same interface, so they are not independent
observations. `sensor_groups()` exposes this and the processed manifest records
it per view. Only `suricata` is currently in the group; the group exists so
that when a packet-capture view is built it joins an existing statement about
shared sensors instead of arriving as an apparently independent view. Part 26
reports the grouping and does not act on it.

---

## 5. Label coverage

Ground truth comes **only** from the official `labels/` tree: per-line JSON
giving the 1-based line numbers of attack lines in a specific log file. Nothing
else is used - not attack time ranges, not filenames, not the attacker host's
own logs.

AIT labels **8 files across 5 hosts**, covering **six** source types. Part 26
parses five, and the three label files whose source it does not parse refer to
logs the selective fetch never downloaded; they are recorded and kept out of
the usable map rather than retained with line numbers pointing at nothing.

| label file | source type | usable here |
|---|---|---|
| `labels/inet-firewall/logs/dnsmasq.log` | `dns` | yes |
| `labels/intranet_server/logs/apache2/intranet.smith.russellmitchell.com-access.log.2` | `apache` | yes |
| `labels/intranet_server/logs/auth.log` | `auth` | yes |
| `labels/intranet_server/logs/audit/audit.log` | `audit` | yes |
| `labels/internal_share/logs/audit/audit.log` | `audit` | yes |
| `labels/intranet_server/logs/apache2/intranet.smith.russellmitchell.com-error.log.2` | `apache_error` | no - log not fetched |
| `labels/vpn/logs/openvpn.log` | `openvpn` | no - log not fetched |
| `labels/monitoring/logs/logstash/intranet-server/2022-01-24-system.cpu.log` | `monitoring` | no - log not fetched |

| parsed source | label files | hosts covered | labelled attack lines |
|---|---|---|---|
| `apache` | 1 | `intranet_server` | 7,695 |
| `auth` | 1 | `intranet_server` | 8 |
| `dns` | 1 | `inet-firewall` | 54,035 |
| `audit` | 2 | `internal_share`, `intranet_server` | 11 |
| `suricata` | 0 | - | 0 |

**61,749 labelled attack lines**, of which 54,035 (87.5%) are the dnsteal
exfiltration in one DNS log on one host.

### Suricata is an unlabelled source

The archive's `rules/` directory carries labelling rules for `0_auth`,
`apache`, `audit`, `dnsmasq`, `monitoring` and `openvpn` - and **none for
suricata**. There is no suricata label file anywhere in the scenario.
`suricata` is therefore reported as an `unlabelled_parsed_source`. Its features
are usable, and it is the only network-sensor view, but its quiet windows are
not evidence of benign and it contributes nothing to any window's benign
status.

### Absence of a label is not evidence of benign

This asymmetry forces a third label state:

- A window a **label-covered file** contributed to, with no labelled line in
  it, is benign: the labelling pass examined those lines and marked none.
- A window **no label-covered file** contributed to is **not labelled**. The
  labelling never examined it.

So `labels` is three-state - `1` attack, `0` benign, `-1` not labelled - with a
`label_known` boolean array beside it. **Part 28 must filter on `label_known`
before training.** Reading `labels == 0` as benign without it would invent
negatives out of unexamined windows; in host mode at 300 s that would be 8,187
of 9,834 samples.

Coverage is tracked **per file, not per (host, source)**: AIT labels
`intranet_server`'s `...-access.log.2` but not `.1`, `.3` or `.4` of the same
log on the same host.

---

## 6. Features per view

Every feature is one of three kinds, and the kind fixes its transform:

| kind | what it is | transform |
|---|---|---|
| `count` | an event or distinct-value count | `log1p(count)` |
| `fraction` | a share of the window's events | none; already in [0, 1] |
| `value` | a magnitude in native units (bytes, characters) | `log1p(value)` |

| view | D | features |
|---|---|---|
| `apache` | 9 | `n_requests`, `n_unique_client`, `n_unique_path`, `n_unique_agent`, `frac_status_4xx`, `frac_status_5xx`, `frac_method_post`, `bytes_total`, `bytes_mean` |
| `auth` | 8 | `n_events`, `n_unique_program`, `n_unique_pid`, `frac_session_open`, `frac_session_close`, `frac_failure`, `frac_sudo`, `frac_ssh` |
| `dns` | 9 | `n_events`, `n_unique_qname`, `n_unique_client`, `n_unique_qtype`, `frac_query`, `frac_answered`, `frac_forwarded`, `qname_len_mean`, `qname_len_max` |
| `audit` | 9 | `n_events`, `n_unique_type`, `n_unique_comm`, `n_unique_uid`, `n_unique_event_id`, `frac_failed`, `frac_syscall`, `frac_execve`, `frac_user_record` |
| `suricata` | 10 | `n_events`, `n_unique_event_type`, `n_unique_src_ip`, `n_unique_dest_ip`, `n_unique_dest_port`, `n_unique_signature`, `frac_alert`, `frac_http`, `frac_dns`, `flow_bytes_total` |

`D = 10`, the widest view; narrower views are right-padded with zeros and each
view's `EncoderConfig` declares its own native `input_dim` (`view_configs`), so
no encoder reads a padded column - the same arrangement as
`real_datasets.view_configs`.

**No feature reads a label.** Features are computed in `scan_scenario` from
`gather/` log line content only; labels are read separately from `labels/` and
land in `Accumulator.attack_steps`, `.attack_lines` and `.covered_events`,
which `Accumulator.vector()` never touches. This is checked behaviourally: the
test moves the attack to a different line, rebuilds, and requires the feature
bytes to be **identical** while the label array **changes**.

**No standardisation is applied.** Unlike the `handwritten` loader, this one
does not z-score, because a scaler must be fitted on training indices only and
the split is Part 28's business. The manifest records each feature's name, kind
and description so a later Part scales from a known starting point.

`qname_len_mean` and `qname_len_max` are in the `dns` view because
DNS-exfiltration tooling smuggles bytes out in subdomain labels and so produces
very long query names. That is a property of the line, not of any label - but
see section 0: it is also why the window-level task on this scenario is close
to a single-view one.

---

## 7. Samples and missing fractions, both modes

A log data set has no natural row, so the unit of analysis is constructed, and
the choice changes what a view is. Both readings are implemented and both are
reported. **Part 27 decides; this Part does not choose.**

| | `sample_mode: network` (default) | `sample_mode: host` |
|---|---|---|
| one sample is | one network-wide time window | one (host, time window) |
| a view aggregates | that source type across every host producing it | that source type on that host alone |
| distinct-value features | union of the value sets across hosts | that host's own set |
| a view is missing when | no host logged that source in the window | that host did not log that source in the window |

The union matters and is tested: summing per-host distinct counts instead of
unioning the value sets would make one client address seen on two hosts read as
two clients, so every distinct-value feature would scale with however many
hosts happened to be logging.

### Window 300 s

| mode | samples | attack | benign | not labelled | mean views present |
|---|---|---|---|---|---|
| network | 1,152 | 1,031 | 114 | 7 | 4.05 of 5 |
| host | 9,834 | 1,034 | 613 | 8,187 | 1.75 of 5 |

| view | network: present | network: **missing fraction** | host: present | host: **missing fraction** |
|---|---|---|---|---|
| `apache` | 939 | **0.185** | 2,056 | **0.791** |
| `auth` | 603 | **0.477** | 2,537 | **0.742** |
| `dns` | 1,149 | **0.003** | 2,294 | **0.767** |
| `audit` | 825 | **0.284** | 2,211 | **0.775** |
| `suricata` | 1,152 | **0.000** | 8,064 | **0.180** |

The two modes differ by a factor of 2.3 in mean views present (4.05 against
1.75), and per-view missing fractions differ by more than two orders of
magnitude (`dns`: 0.0026 against 0.7667, a ratio of 294).
In network mode a view is rarely absent because some host is almost always
logging; in host mode the mask is structural - only 3 of 10 hosts run Apache,
and only 2 run a DNS resolver, so those views are missing for every sample from
every other host regardless of what happened.

Host mode also pays for the narrow label coverage: **8,187 of 9,834 samples are
not labelled**, leaving 1,647 usable, because only 3 of 10 hosts carry a
labelled file.

### Other window lengths

| window s | windows on grid | network samples | network attack | network not labelled | host samples | host attack | host not labelled |
|---|---|---|---|---|---|---|---|
| 300 | 1,152 | 1,152 | 1,031 | 7 | 9,834 | 1,034 | 8,187 |
| 600 | 576 | 576 | 516 | 0 | 5,210 | 519 | 4,175 |
| 1,800 | 192 | 192 | 172 | 0 | 1,920 | 175 | 1,429 |
| 3,600 | 96 | 96 | 86 | 0 | 960 | 89 | 672 |

Per-view missing fractions at each length are in `report.json`'s sweep and in
each written manifest. The direction is uniform: longer windows mean fewer
samples, fewer missing views, and a slightly higher attack rate as a share of
samples (network mode: 89.5% at 300 s, 89.6% at 3,600 s), with the benign
count falling from 114 to 10. No length escapes the problem in section 0.

`window_seconds: 300` is the brief's default and **is not final**. The sweep
above is included because it is cheap, not because this Part is choosing:
`rewindow` re-bins one 300 s scan to any multiple exactly, since every feature
is merge-closed (counts sum, value sets union, maxima max, fractions derived
from merged counts). A test requires byte-identical tensors against a direct
scan at the target length in both modes, so this is an identity and not an
approximation. One scan of the 2.3 GB took 29 s.

---

## 8. Output

`data/processed/ait_<scenario>_<mode>_<window>s/` (gitignored), per CONTRACT
section 4. Written for both modes at 300 s:
`ait_russellmitchell_network_300s`, `ait_russellmitchell_host_300s`.

| file | shape | dtype | meaning |
|---|---|---|---|
| `views.npy` | `[B, V, D]` = `[1152, 5, 10]` network | float32 | features, zero-padded on D |
| `view_mask.npy` | `[B, V]` | bool | True where that source reported >= 1 event |
| `labels.npy` | `[B]` | int64 | 1 attack, 0 benign, **-1 not labelled** |
| `label_known.npy` | `[B]` | bool | filter on this before training |
| `window_start.npy` | `[B]` | float64 | window start, epoch seconds UTC |
| `host_id.npy` | `[B]` | int64 | host mode only; index into `hosts` |
| `manifest.json` | | | config, feature list per view, counts, label coverage, attack-step frequency, per-file statistics, raw-entry hashes |
| `attack_steps.json` | | | attack step names per sample, where any |

An identical config produces byte-identical arrays and manifests; file
discovery is sorted by `(source, host, relpath)` so no filesystem ordering
reaches the output. Changing any config key changes the output directory name,
so an existing result is never silently overwritten by a different config.

---

## 9. Views are defined by log-source TYPE, never by host

`SOURCES` keys off filename shape anchored on the `logs/` segment
(`logs/apache2/*access.log*`, `logs/auth.log*`, `logs/dnsmasq.log*`,
`logs/audit/audit.log*`, `logs/suricata/eve.json*`). No pattern mentions a host
name, so the same view list and the same code load `santos` - different
machines, different names - with no edit. `classify_log_path` is the single
place a path becomes a source type, and `scripts/fetch_ait.py` calls it too, so
the downloader and the parser cannot disagree about which files matter.

Checked by `test_view_list_is_identical_for_two_scenarios`, which builds two
fixture scenarios whose hosts are `web01`/`fw01` and `srv-a`/`srv-b` and
requires identical view lists and identical tensors; and by
`test_every_source_pattern_is_mutually_exclusive`, since a path matching two
patterns would make the view order ambiguous.

---

## 10. Reproducing this

```
python scripts/fetch_ait.py russellmitchell          # 86.7 MB, resumable
python -m pytest tests/data/test_ait_lds.py -q       # 58 tests
```

Then, from `configs/data/ait.yaml`:

```python
from prismflow.data.ait_lds import build_ait_dataset, build_samples, load_ait_config, write_processed
config = load_ait_config()                            # scenario, window, mode
dataset, samples, scan = build_ait_dataset(config)
write_processed(samples, scan)                        # network mode
write_processed(build_samples(scan, "host"), scan)    # host mode, same scan
```

The held-out scenario is `santos` (`configs/data/ait.yaml`,
`held_out_scenario`): `python scripts/fetch_ait.py santos` and then the same
build with `scenario: santos`.

---

## 11. Limitations of this Part

1. **The window-level task is close to degenerate on this scenario** (section
   0): 90.0% of labelled windows are attack, and 1,026 of the 1,031 attack
   windows carry only the continuous dnsteal steps. Five windows contain the rest of the intrusion.
2. **Suricata is unlabelled**, so the one network-sensor view can never
   contribute to a benign determination, and its quiet windows are unexamined
   rather than clean.
3. **One scenario.** This closes none of L14 on its own - that needs `santos`
   loaded and audited with the same code, which is why the loader is
   scenario-agnostic.
4. **No packet-capture view**, so the `network` same-sensor group has one
   member and the near-duplicate view pair that would most directly exercise
   the project's redundancy thesis does not exist yet. The cost is 4.4 GB
   compressed and a pcap reader.
5. **Features are a first, documented choice**, not a tuned one: 8-10 simple
   count and rate statistics per view. No feature-selection or
   representation-quality evidence exists for them.
6. **169,150 events sit outside the declared scenario interval** and are
   excluded by the default grid. No labelled attack line is among them, but the
   `time_range: observed` alternative would change the sample counts above.
7. **`rho_matrix` is `None` and ENIV cannot be validated here**, only applied
   (L9, `docs/DATASETS.md`).

---
---

# Part B - santos

The second run of the brief: the **held-out scenario for Part 28**, loaded with
**no edit to `prismflow/data/ait_lds.py`, `scripts/fetch_ait.py`,
`configs/data/ait.yaml` or `tests/data/test_ait_lds.py`**. The only input that
changed is the scenario name, passed as the documented override
`load_ait_config(scenario="santos")` (`configs/data/ait.yaml` still declares
`scenario: russellmitchell` and `held_out_scenario: santos`).

As in part A: **no experiment was run, and no ENIV, dependence, accuracy or
calibration number appears.** Every figure in part B is a count, a timestamp or
a fraction of counts, produced by the committed loader and recorded in
`data/processed/ait_santos_network_300s/manifest.json`,
`data/processed/ait_santos_host_300s/manifest.json` and
`data/raw/ait/santos/manifest.json` (all three gitignored; the SHA-256 and
CRC-32 in the raw manifest are what tie these numbers to specific bytes).

---

## 12. The one thing to read before Part 27 or 28 (santos)

**santos has a real benign population and russellmitchell did not.** At 300 s
network windows:

| | santos | russellmitchell |
|---|---|---|
| samples | 1,152 | 1,152 |
| attack | **667** | 1,031 |
| benign | **441** | 114 |
| not labelled | 44 | 7 |
| attack share of labelled windows | **60.2%** (667 / 1,108) | 90.0% (1,031 / 1,145) |
| attack windows carrying only dnsteal-family steps | **663** | 1,026 |
| attack windows carrying any other step | **4** | 5 |
| mean views present | 3.99 of 5 | 4.05 of 5 |

(santos from `data/processed/ait_santos_network_300s/manifest.json`,
`statistics`; russellmitchell from the part A table in section 7 and its own
manifest.)

Why the benign windows exist. `processing/config/attacker.yaml` declares
`exfiltration_service.stop: 2022-01-16T07:16:18.019Z` while the capture runs to
2022-01-18T00:00Z, so the continuous DNS exfiltration ends **1 day 16.7 h
before the data does**. The last window carrying a dnsteal-family label starts
2022-01-16T07:15:00Z, and `gather/internal_share/logs/audit/audit.log` lines
451-452 (`SERVICE_START` / `SERVICE_STOP`, `audit(1642317378.019)` =
2022-01-16T07:16:18Z) are labelled `exfiltration-service`. In russellmitchell
the same tool ran to the end of the capture, which is why 90% of its windows
are attack.

**But the window label is still almost entirely "was dnsteal running".** 663 of
the 667 attack windows carry only `attacker` / `dnsteal` / `dnsteal-received`.
The multi-step intrusion - `foothold`, `network_scan`, `traceroute`,
`dns_scan`, `service_scan`, `dirb`, `wpscan`, `attacker_http`,
`webshell_upload`, `webshell_cmd`, `escalate`, `attacker_change_user`,
`escalated_*` - is confined to **three windows**: those starting
2022-01-17T11:15:00Z, 11:20:00Z and 11:55:00Z. A fourth window
(2022-01-16T07:15:00Z) carries `exfiltration-service` alongside the dnsteal
steps. So the activity that would actually require several views to see is 3-4
windows of 1,152, exactly as in part A section 0.

**The labels are a step function in time, and that decides how Part 28 may
split.** Measured from `labels.npy` and `window_start.npy` of
`ait_santos_network_300s`: the attack windows form **three runs** - 664
consecutive windows from the first window of the grid (2022-01-14T00:00:00Z) to
2022-01-16T07:20:00Z, then 2 windows at 2022-01-17T11:15Z, then 1 at
2022-01-17T11:55Z. Benign windows run from 2022-01-16T07:20:00Z to the end.
Consequences, both structural rather than tunable:

- A **random** window split puts adjacent windows of the same 664-window run in
  train and test. Any feature that drifts slowly (log volume, host set) then
  leaks.
- A **chronological** split on this scenario puts nearly all attack windows in
  the earlier part and nearly all benign windows in the later part, so an
  early/late split is close to a label split.

Neither is a defect in the loader; both are properties of a scenario in which
one tool ran for the first 58% of the capture. Part 27 should state which split
it intends before Part 28 reports any accuracy on this data.

---

## 13. Data provenance (santos)

| field | value |
|---|---|
| Zenodo record | 5789064 |
| archive | `santos.zip` |
| archive size | 10,004.8 MB (10.0 GB) |
| entries in archive | 14,908 |
| entries fetched | **86** |
| transferred, compressed | **89.2 MB** |
| written to disk, uncompressed | 2,351.5 MB |
| manifest entries with CRC-32 + SHA-256 | 86 |
| fetch complete | true |
| scan time, one pass over 2.35 GB | 48.0 s |

Source: `data/raw/ait/santos/manifest.json` (`archive_size_bytes`,
`archive_entries`, `selected_entries`, `selected_compressed_bytes`,
`selected_uncompressed_bytes`, `complete`, `entries`).

The selective fetch worked exactly as in part A and needed no change: Zenodo
served every range request (`--dry-run` reported `86 of 14908 entries selected,
89.2 MB compressed -> 2351.5 MB on disk (archive is 10.0 GB)` before anything
was downloaded), all 86 members were verified against the archive's own CRC-32
before being written, and 86 of 86 are recorded with CRC-32 and SHA-256. The
brief's stop condition - range requests refused, archive size unreported - was
not reached, so nothing needed to be asked.

**89.2 MB transferred instead of 10.0 GB.** 10,118 MB of the archive is raw
packet capture that no Part 26 view parses (section 14).

---

## 14. Inventory: parsed sources (santos)

One view per log source type, the same five as part A. Window 300 s, grid from
the archive's own `dataset.yaml` (**2022-01-14T00:00:00Z to
2022-01-18T00:00:00Z**, 1,152 windows).

| source | sensor | hosts | files | lines | events parsed | unparsed | outside grid | first event | last event |
|---|---|---|---|---|---|---|---|---|---|
| `apache` | web server (Apache2 access log) | 3 | 25 | 210,121 | 210,121 | 0 | 0 | 2022-01-14 05:24:13Z | 2022-01-17 22:21:53Z |
| `auth` | host authentication (PAM / sshd via syslog) | 10 | 20 | 6,744 | 6,744 | 0 | 1,430 | 2022-01-13 10:44:57Z | 2022-01-18 06:01:19Z |
| `dns` | DNS resolver (dnsmasq) | 2 | 2 | 567,569 | 567,569 | 0 | 36,107 | 2022-01-13 11:35:21Z | 2022-01-18 06:00:03Z |
| `audit` | Linux kernel audit subsystem (auditd) | 7 | 7 | 24,042 | 24,042 | 0 | 0 | 2022-01-14 00:00:01Z | 2022-01-17 23:45:11Z |
| `suricata` | network IDS (Suricata, AF-packet tap) | 7 | 8 | 1,129,009 | 1,129,009 | 0 | 121,429 | 2022-01-13 14:33:23Z | 2022-01-18 05:41:42Z |

**1,937,485 events parsed, 0 lines unparsed**, from
`data/processed/ait_santos_network_300s/manifest.json` (`source_files`, summed
per source by `ait_lds.inventory`).

The zero matters more here than in part A, because it is the part-A format work
being tested on data it was not written against:

- `cloud_share`'s only non-empty Apache logs are
  `other_vhosts_access.log.{0..3}` in Debian's **combined-vhost** format
  (44,244 lines across four files). The part-A parser change reads them, so
  `cloud_share` is in the santos apache view. Before that change this host
  would have contributed 44,244 "unparsed" lines and no events.
- santos `dnsmasq.log` has the same **no-hostname** syslog shape, and all
  567,569 DNS lines parse.

**158,966 events fall outside the window grid** (`auth` 1,430, `dns` 36,107,
`suricata` 121,429), counted and reported rather than dropped. Same cause as
part A: logging starts before the declared interval (Suricata's first event is
2022-01-13T14:33Z, `dataset.yaml` declares the scenario from
2022-01-14T00:00Z). **No labelled attack line falls outside the grid or fails
to parse** - `labelled_attack_lines_outside_grid` and
`labelled_attack_lines_unparsed` are 0 for all five sources.

### Hosts producing each parsed source (santos)

| source | hosts |
|---|---|
| `apache` | `cloud_share`, `intranet_server`, `webserver` |
| `auth` | `cloud_share`, `hayes_mail`, `inet-dns`, `inet-firewall`, `internal_share`, `intranet_server`, `mail`, `rogersturnbull_mail`, `vpn`, `webserver` |
| `dns` | `inet-dns`, `inet-firewall` |
| `audit` | `cloud_share`, `inet-firewall`, `internal_share`, `intranet_server`, `mail`, `vpn`, `webserver` |
| `suricata` | `cloud_share`, `inet-firewall`, `internal_share`, `intranet_server`, `mail`, `vpn`, `webserver` |

Ten hosts, and **two of them are named differently from every russellmitchell
host** (`hayes_mail`, `rogersturnbull_mail` against `davey_mail`,
`morris_mail`). Nothing in the loader noticed, which is the point of section 9:
`SOURCES` keys off the path shape under `logs/`, never off the host directory.
As in part A, **no attacker or client host contributes to any parsed view** -
checked against the whole-archive census, not just what was fetched:
`archive_sources` lists exactly these ten hosts for the five parsed source
types, so no attacker-side log could have entered a defender view even though
the fetch selects by source type regardless of host.

### Found, not used yet (santos)

21 unparsed source types are present, censused from the archive's ZIP central
directory (`archive_sources` in the raw manifest), not from disk. The largest:

| source | sensor | files | hosts | archive bytes |
|---|---|---|---|---|
| `pcap` | Suricata AF-packet tap (`same_sensor_group: network`) | 13 | 7 | 10,118.0 MB |
| `monitoring` | host metrics agent (metricbeat via Logstash) | 158 | 1 | 1,813.8 MB |
| `suricata_stats` | Suricata (`same_sensor_group: network`) | 8 | 7 | 1,638.0 MB |
| `attacker_wpdiscuz` | attacker tooling (wpDiscuz exploit) | 4 | 1 | 670.7 MB |
| `dnsteal` | attacker tooling (dnsteal) | 36 | 1 | 230.2 MB |
| `attacker_downloads` | attacker tooling | 59 | 6 | 178.2 MB |
| `journal` | systemd journal (binary) | 18 | 7 | 159.4 MB |
| `logstash_self` | Logstash's own pipeline log | 15 | 1 | 66.0 MB |
| `mail` | Exim4 / sendmail | 47 | 21 | 57.2 MB |
| `horde` | webmail (Horde) | 6 | 3 | 35.7 MB |

The remaining 11 (`suricata_fast` 9.8 MB, `syslog` 8.0 MB, `messages` 3.6 MB,
`user_log` 3.5 MB, `suricata_self` 1.5 MB, `openvpn` 0.6 MB, `apache_error`
0.4 MB, `kern` 0.2 MB, and `shorewall`, `redis`, `attacks_log` each under
0.1 MB) are in the raw manifest's census with the same fields. Reasons for
exclusion are unchanged from part A section 4; `pcap` is 10.1 GB here against
7.0 GB in russellmitchell.

**All 528 `gather/*/logs/` files in the santos archive classify into one of the
registry's 26 source types**: the census carries no `__unclassified__` bucket
(`archive_census` adds that key only when something fails to classify), so the
inventory has no hole on a scenario the registry was not written against.

`sensor_groups()` reports `{"network": [4]}` for santos exactly as for
russellmitchell - `suricata` at view index 4, with `pcap` reserved for the same
group. Part 26 reports the grouping and does not act on it.

---

## 15. Label coverage (santos)

Ground truth comes **only** from the official `labels/` tree. santos labels
**8 files across 5 hosts**, covering **six** source types - the same count and
the same six types as russellmitchell.

| label file | source type | usable here |
|---|---|---|
| `labels/inet-firewall/logs/dnsmasq.log` | `dns` | yes |
| `labels/intranet_server/logs/apache2/intranet.smith.santos.com-access.log` | `apache` | yes |
| `labels/intranet_server/logs/auth.log` | `auth` | yes |
| `labels/intranet_server/logs/audit/audit.log` | `audit` | yes |
| `labels/internal_share/logs/audit/audit.log` | `audit` | yes |
| `labels/intranet_server/logs/apache2/intranet.smith.santos.com-error.log` | `apache_error` | no - log not fetched |
| `labels/vpn/logs/openvpn.log` | `openvpn` | no - log not fetched |
| `labels/monitoring/logs/logstash/intranet-server/2022-01-17-system.cpu.log` | `monitoring` | no - log not fetched |

| parsed source | label files | hosts covered | labelled attack lines |
|---|---|---|---|
| `apache` | 1 | `intranet_server` | 7,794 |
| `auth` | 1 | `intranet_server` | 12 |
| `dns` | 1 | `inet-firewall` | 39,426 |
| `audit` | 2 | `internal_share`, `intranet_server` | 21 |
| `suricata` | 0 | - | 0 |

**47,253 labelled attack lines**, of which 39,426 (83.4%) are in the one DNS
log on one host. All 47,253 are parsed and inside the window grid.

**Suricata is an unlabelled source here too.** `rules/` carries `0_auth.yaml`,
`apache.yaml`, `audit.yaml`, `dnsmasq.yaml`, `monitoring.yaml` and
`openvpn.yaml` and no suricata rules; there is no suricata label file anywhere
in the scenario, so `unlabelled_parsed_sources == ["suricata"]`. Its quiet
windows are unexamined, not benign, and it contributes to no window's benign
status.

The three-state label (`1` attack, `0` benign, `-1` not labelled) and the
`label_known` array work as in part A. **Part 28 must filter on
`label_known`**: in santos host mode that is 8,171 of 9,779 samples not
labelled, leaving 1,608 usable.

Coverage is per file, not per (host, source), and santos shows it on a
different file than russellmitchell did: `intranet_server`'s apache label file
is the **current** `intranet.smith.santos.com-access.log` (9,462 lines, 7,794
labelled), while `.log.1`, `.log.2` and `.log.3` of the same log on the same
host (6,322 lines together) carry no labels.

### The five hand-checked labelled lines

One per usable label file, read from `labels/` and compared against the raw
`gather/` line it names:

| file | line | attack steps | raw line (truncated) |
|---|---|---|---|
| `gather/inet-firewall/logs/dnsmasq.log` | 1 | `attacker`, `dnsteal`, `dnsteal-received` | `Jan 14 00:00:09 dnsmasq[14919]: query[A] 3x6-.401-.ScGCirxO*xUUf11BDIVfdMkwtfLsy2uc...` |
| `gather/intranet_server/logs/apache2/intranet.smith.santos.com-access.log` | 431 | `foothold`, `service_scan` | `172.21.128.119 - - [17/Jan/2022:11:21:38 +0000] "GET / HTTP/1.0" 200 17642 "-" "-"` |
| `gather/intranet_server/logs/auth.log` | 185 | `attacker_change_user`, `escalate` | `Jan 17 11:58:17 intranet-server su[20749]: Successful su for gmorgan by www-data` |
| `gather/intranet_server/logs/audit/audit.log` | 1969 | `attacker_change_user`, `escalate` | `type=USER_AUTH msg=audit(1642420697.656:2274): ... acct="gmorgan" exe="/bin/su"` |
| `gather/internal_share/logs/audit/audit.log` | 451 | `attacker`, `dnsteal`, `exfiltration-service` | `type=SERVICE_START msg=audit(1642317378.019:579): pid=1 uid=0 ... unit=ut` |

Each line exists, is non-blank, carries at least one step name, and is
consistent with its step: a dnsteal query whose 250-character name carries the
exfiltrated file name `customers_2013.xlsx` and the attacker domain
`kiirjekuimkpcrcioastvkeodpnimx.biz` (`processing/config/attacker.yaml`,
`dnsteal.domain`), the first HTTP request of the `foothold` scan, the `su` to
`gmorgan` that `escalate` performs, its audit counterpart, and the exfiltration
service's own start record. Reproduce with `load_labels` / `read_label_file` from
`prismflow/data/ait_lds.py`; the committed test
`test_real_labelled_lines_are_attack_lines_in_their_files` performs the same
check automatically, pinned to `data/raw/ait/russellmitchell` (part A).

---

## 16. Samples and missing fractions, both modes (santos)

Both modes implemented, both reported, **neither chosen** - Part 27 decides.

### Window 300 s

| mode | samples | attack | benign | not labelled | mean views present |
|---|---|---|---|---|---|
| network | 1,152 | 667 | 441 | 44 | 3.99 of 5 |
| host | 9,779 | 669 | 939 | 8,171 | 1.74 of 5 |

| view | network: present | network: **missing fraction** | host: present | host: **missing fraction** |
|---|---|---|---|---|
| `apache` | 809 | **0.298** | 1,813 | **0.815** |
| `auth` | 603 | **0.477** | 2,540 | **0.740** |
| `dns` | 1,116 | **0.031** | 2,175 | **0.778** |
| `audit` | 917 | **0.204** | 2,380 | **0.757** |
| `suricata` | 1,152 | **0.000** | 8,064 | **0.175** |

From `statistics.views[*].missing_fraction` in each mode's manifest. The two
modes differ by a factor of 2.30 in mean views present (3.99 against 1.74) and
by 24.9x on `dns` (0.0312 against 0.7776) - the same structural story as part
A, on a different scenario: in network mode some host is almost always logging,
while in host mode only 3 of 10 hosts run Apache and only 2 run a resolver.

Host mode again pays for narrow label coverage: **8,171 of 9,779 samples are
not labelled**, leaving 1,608 usable, because only 3 of 10 hosts carry a
labelled file. Its attack share of labelled samples is 669 / 1,608 = **41.6%**,
against 60.2% in network mode.

### Other window lengths

One scan, re-binned with `rewindow` (a tested identity for these merge-closed
features, not an approximation):

| window s | windows on grid | net samples | net attack | net benign | net unlabelled | net mean views | host samples | host attack | host benign | host unlabelled |
|---|---|---|---|---|---|---|---|---|---|---|
| 300 | 1,152 | 1,152 | 667 | 441 | 44 | 3.99 | 9,779 | 669 | 939 | 8,171 |
| 600 | 576 | 576 | 335 | 235 | 6 | 4.29 | 5,197 | 337 | 682 | 4,178 |
| 1,800 | 192 | 192 | 113 | 79 | 0 | 4.73 | 1,920 | 115 | 373 | 1,432 |
| 3,600 | 96 | 96 | 57 | 39 | 0 | 4.75 | 960 | 59 | 229 | 672 |

Per-view missing fractions at each length, and every other figure in part B,
are in `data/processed/ait_santos_report.json` (gitignored), written by the one
run that produced the two processed directories.

`window_sweep_seconds` also lists 60 s; it is **not reported** because
`rewindow` only re-bins upward and a 60 s table would need its own scan. That
is a gap in this sweep, not a measured zero.

Longer windows move santos the same way as russellmitchell - fewer samples,
fewer missing views - and they also **shrink the multi-view part of the
label**: attack windows carrying a non-dnsteal step go 4 (300 s), 4 (600 s), 3
(1,800 s), 2 (3,600 s) in network mode, because the three intrusion windows
merge into their neighbours. The attack share of labelled windows stays near
60% at every length (60.2%, 58.8%, 58.9%, 59.4%).

---

## 17. Output and reproducibility (santos)

`data/processed/ait_santos_network_300s/` and
`data/processed/ait_santos_host_300s/` (both gitignored), same files and shapes
as part A section 8: `views.npy` `[1152, 5, 10]` network and `[9779, 5, 10]`
host, `view_mask.npy`, `labels.npy`, `label_known.npy`, `window_start.npy`,
`host_id.npy` (host mode), `manifest.json`, `attack_steps.json`.

Three checks were run against the written santos arrays rather than assumed
from part A:

1. **Byte-identical on a second scan.** A second independent `scan_scenario` +
   `build_samples` + `write_processed` into a scratch directory produced files
   byte-identical to the committed run: 7 of 7 files in network mode, 8 of 8 in
   host mode (`filecmp.cmpfiles(..., shallow=False)`, no mismatches, no
   errors).
2. **Masked views are exactly the empty ones.** Every `view_mask == False` row
   of `views.npy` is all zero, and every `view_mask == True` row has a positive
   `log1p(event count)` in its first column - 4,597 of 4,597 present views in
   network mode, 16,972 of 16,972 in host mode.
3. **Label alignment**, section 15.

Reproduce:

```
python scripts/fetch_ait.py santos                   # 89.2 MB, resumable
python -m pytest tests/data/test_ait_lds.py -q       # 58 tests
```

```python
from prismflow.data.ait_lds import (
    build_samples, load_ait_config, raw_manifest, scan_scenario, write_processed,
)
config = load_ait_config(scenario="santos")           # window 300 s, mode network
scan = scan_scenario(config)                          # 48 s over 2.35 GB
raw = raw_manifest(config)
write_processed(build_samples(scan, "network"), scan, raw)
write_processed(build_samples(scan, "host"), scan, raw)
```

---

## 18. What part B settles, and what it does not

**Settled.** The loader is scenario-agnostic in the way section 9 claimed: a
second AIT scenario - different hosts, different dates, a 10.0 GB archive, 528
log files - loaded with no edit to any committed file, produced the identical
view list (`apache`, `auth`, `dns`, `audit`, `suricata`, `D = 10`, native
widths 9/8/9/9/10), classified every log file in the archive, and left 0 lines
unparsed. Part A section 11.3 ("one scenario") is closed as far as *loading*
goes.

**Not settled, and not touched by this run:**

1. **L14 is not closed.** L14 asks for the redundancy audit - dependence
   matrix, ENIV, efficiency - on unrelated benchmarks. Part 26 reports counts
   only. Nothing here says whether these five sensor views are less redundant
   than `handwritten`'s six pixel-derived ones; Part 28 measures that.
   `rho_matrix` is `None` for santos as for russellmitchell, so ENIV can be
   applied and never validated on this data (L9, `docs/DATASETS.md`).
2. **The window-level task is still close to a single-view one** (section 12):
   663 of 667 attack windows are dnsteal-only, and dnsteal is directly visible
   in `dns` as very long query names. santos is better than russellmitchell
   because it has 441 benign windows, not because the attack is more
   multi-view.
3. **Suricata, the only network-sensor view, is unlabelled** in both scenarios.
4. **No packet-capture view**, so the `network` same-sensor group still has one
   member - and santos makes that more expensive, not less: `pcap` is 10,118 MB
   here.
5. **Features are part A's first documented choice**, unchanged and untuned. No
   feature-selection or representation-quality evidence exists for them on
   either scenario.
6. **The committed real-data tests are pinned to
   `data/raw/ait/russellmitchell`** (`REAL_SCENARIO_ROOT`,
   `tests/data/test_ait_lds.py:78`). Re-pointing them at santos, or
   parametrising them over both, is an edit the second run of this brief
   forbids; the santos equivalents of those three checks were run by hand and
   are recorded in sections 15 and 17 with the code to repeat them.
7. **Two scenarios of eight.** `fox`, `harrison`, `wardbeck`, `shaw`,
   `wheeler` and `wilson` are unloaded, and nothing here establishes that they
   behave like these two.
