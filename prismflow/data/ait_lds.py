"""Part 26: AIT Log Data Set v2.0 as a PrismFlow multi-view dataset.

This is the project's second real dataset family, and the first whose views
come from genuinely different sensors rather than different transforms of one
image. `docs/KNOWN_LIMITATIONS.md` L14 names that gap precisely: "One dataset
family, and five seeds that are five partitions of it", and asks for "views
from different sensing modalities". A web server's access log, a host's PAM
authentication log, a DNS resolver's query log, a kernel audit trail and a
network IDS are five such modalities observing one network.

This Part builds the loader and reports dataset statistics. It runs no
experiment, trains nothing, and computes no ENIV, accuracy or dependence
figure -- those belong to Part 28 and later.

WHAT A SAMPLE IS
----------------
A log data set has no natural row. AIT ships raw log files per host, so the
unit of analysis has to be constructed, and the choice changes what "view"
means. Part 26 therefore implements BOTH readings and reports both, rather
than silently fixing one (Part 27 decides):

- `sample_mode="network"` (default): one sample is one network-wide time
  window. Each view aggregates one log-source TYPE across every host that
  produces it. Views are rarely missing, because some host is almost always
  logging; the question a view answers is "what did the DNS resolver see on
  this network in this window".
- `sample_mode="host"`: one sample is (host, time window). A view is present
  only when that host runs that source, so the mask is heavily structured --
  only one host in russellmitchell runs Apache, so the apache view is missing
  for every sample from every other host.

Neither is more correct. They answer different questions, and the missing-view
fraction differs by an order of magnitude between them, which is exactly why
the brief asks for both.

VIEWS ARE DEFINED BY SOURCE TYPE, NOT BY HOST
---------------------------------------------
`SOURCES` keys off filename shape (`logs/apache2/*access.log*`,
`logs/auth.log*`, ...), never off a host name, so the same view list and the
same code load `santos` -- whose hosts are different machines with different
names -- without an edit. `classify_log_path` is the single place a path
becomes a source type, and `scripts/fetch_ait.py` calls it too, so the
downloader and the parser cannot disagree about which files matter.

GROUND TRUTH COMES ONLY FROM labels/
------------------------------------
AIT publishes an official `labels/` tree: for each labelled log file, a
JSON-per-line file giving the 1-based line numbers that belong to an attack
step. Part 26 uses nothing else -- not attack time ranges, not filenames, not
the attacker host's own logs. A window is attack=1 when at least one labelled
line falls inside it.

That coverage is PARTIAL, and the asymmetry matters:

- A labelled source's window with no labelled line in it is evidence of
  benign, because the labelling pass examined those lines and marked none.
- An UNLABELLED source carries no such evidence. Suricata is unlabelled in
  this dataset: `rules/` contains labelling rules for apache, auth, audit,
  dnsmasq, openvpn and monitoring, and none for suricata. Its features are
  still usable, but its quiet windows say nothing about whether an attack was
  under way. `LabelCoverage` records which sources and hosts the labelling
  covers, the manifest stores it, and `unlabelled_sources` is reported rather
  than folded into the benign count.

NO FEATURE READS A LABEL
------------------------
Features are computed from `data/raw/ait/<scenario>/gather/` only; labels are
read from `.../labels/` only, by a separate function, after features exist.
The two paths never meet. `tests/data/test_ait_lds.py` asserts this
behaviourally -- it perturbs the label files and checks the feature tensor is
byte-identical -- rather than trusting this paragraph.

CONVENTIONS FOLLOWED FROM prismflow/data/real_datasets.py
---------------------------------------------------------
Heterogeneous view widths are right-padded to D = max_v d_v and each view's
`EncoderConfig` declares its own `input_dim`, so no encoder ever reads a
padded column (`view_configs`). `rho_matrix` is None: real views have no
ground-truth dependence, and an estimate must never stand in for a target
(`docs/DATASETS.md`). Raw data lives in gitignored `data/raw/`, processed
tensors in gitignored `data/processed/`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import torch

from prismflow.data.dataset import Batch, Sample
from prismflow.models.encoders import EncoderConfig

DEFAULT_RAW_ROOT = Path("data/raw/ait")
DEFAULT_PROCESSED_ROOT = Path("data/processed")

# Window length in seconds. 300 is the brief's default and is NOT the final
# choice: Part 27 reports the window-length table and decides.
DEFAULT_WINDOW_SECONDS = 300
SAMPLE_MODES = ("network", "host")


# ---------------------------------------------------------------------------
# Log-source registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogSource:
    """One log-source TYPE: how to recognise it, and what it observes.

    `pattern` matches the path of a file inside the scenario archive, relative
    to the scenario root, and is anchored on the `logs/` segment so it is
    independent of the host directory between `gather/` and `logs/`.
    """

    name: str
    pattern: str
    sensor: str
    description: str
    # Views from one physical sensor are not independent observations. Suricata
    # and the raw packet capture it writes are the same AF-packet tap on the
    # same interface; grouping them here keeps that visible to anything reading
    # the metadata, instead of leaving two apparently separate views.
    same_sensor_group: str | None = None
    parsed: bool = False
    note: str = ""

    @property
    def regex(self) -> re.Pattern[str]:
        return re.compile(self.pattern)


# Order is the canonical view order: it fixes the V axis of every tensor this
# module writes, so it must not be reordered without reprocessing. The five
# `parsed=True` sources are the brief's starting set; everything below them is
# recorded as found and left unparsed, so the inventory is complete without
# anything being quietly consumed.
SOURCES: tuple[LogSource, ...] = (
    LogSource(
        name="apache",
        pattern=r"(?:^|/)logs/apache2/[^/]*access\.log(?:\.\d+)?$",
        sensor="web server (Apache2 access log)",
        description="HTTP requests served: method, path, status, response bytes, user agent",
        parsed=True,
    ),
    LogSource(
        name="auth",
        pattern=r"(?:^|/)logs/auth\.log(?:\.\d+)?$",
        sensor="host authentication (PAM / sshd via syslog)",
        description="login, sudo, session and SSH authentication events",
        parsed=True,
    ),
    LogSource(
        name="dns",
        pattern=r"(?:^|/)logs/dnsmasq\.log(?:\.\d+)?$",
        sensor="DNS resolver (dnsmasq)",
        description="DNS queries, replies, forwards and cached answers",
        parsed=True,
    ),
    LogSource(
        name="audit",
        pattern=r"(?:^|/)logs/audit/audit\.log(?:\.\d+)?$",
        sensor="Linux kernel audit subsystem (auditd)",
        description="syscall, execve, path and user-credential audit records",
        parsed=True,
    ),
    LogSource(
        name="suricata",
        pattern=r"(?:^|/)logs/suricata/eve\.json(?:\.\d+)?$",
        sensor="network IDS (Suricata, AF-packet tap)",
        description="IDS events: alerts, flows, HTTP/DNS/TLS metadata, anomalies",
        same_sensor_group="network",
        parsed=True,
    ),
    # ---- found, not used yet -------------------------------------------
    LogSource(
        name="pcap",
        pattern=r"(?:^|/)logs/suricata/log\.pcap(?:\.\d+)?$",
        sensor="network IDS (Suricata, AF-packet tap)",
        description="raw packet capture written by the same Suricata instance",
        same_sensor_group="network",
        note=(
            "not fetched: 4.4 GB compressed for russellmitchell's labelled hosts "
            "alone. Shares `network` with suricata, so adding it needs no redesign."
        ),
    ),
    LogSource(
        name="apache_error",
        pattern=r"(?:^|/)logs/apache2/(?:[^/]*-)?error\.log(?:\.\d+)?$",
        sensor="web server (Apache2 error log)",
        description="Apache error and PHP warning lines",
        note="labelled by AIT, but the brief's starting set is the access log",
    ),
    LogSource(
        name="suricata_fast",
        pattern=r"(?:^|/)logs/suricata/fast\.log(?:\.\d+)?$",
        sensor="network IDS (Suricata, AF-packet tap)",
        description="one-line alert summaries, a subset of eve.json",
        same_sensor_group="network",
        note="redundant with the suricata view by construction",
    ),
    LogSource(
        name="suricata_stats",
        pattern=r"(?:^|/)logs/suricata/stats\.log(?:\.\d+)?$",
        sensor="network IDS (Suricata, AF-packet tap)",
        description="capture and decoder counters emitted every ~8 s",
        same_sensor_group="network",
    ),
    LogSource(
        name="suricata_self",
        pattern=r"(?:^|/)logs/suricata/suricata(?:-start)?\.log(?:\.\d+)?$",
        sensor="network IDS (Suricata, AF-packet tap)",
        description="Suricata's own engine log",
        same_sensor_group="network",
    ),
    LogSource(
        name="openvpn",
        pattern=r"(?:^|/)logs/openvpn\.log(?:\.\d+)?$",
        sensor="VPN gateway (OpenVPN)",
        description="VPN session establishment and teardown",
        note="labelled by AIT; a candidate sixth parsed view",
    ),
    LogSource(
        name="monitoring",
        pattern=r"(?:^|/)logs/logstash/[^/]+/\d{4}-\d{2}-\d{2}-system\.[^/]+\.log$",
        sensor="host metrics agent (metricbeat via Logstash)",
        description="per-host CPU, memory, network, process and service metrics",
        note="labelled by AIT for one host-day only (see label coverage)",
    ),
    LogSource(
        name="logstash_self",
        pattern=r"(?:^|/)logs/logstash/logstash-[^/]+\.log$",
        sensor="log shipper (Logstash)",
        description="Logstash's own pipeline log",
    ),
    LogSource(
        name="syslog",
        pattern=r"(?:^|/)logs/syslog(?:\.\d+)?$",
        sensor="host syslog",
        description="general system log; overlaps auth, kern and daemon output",
    ),
    LogSource(
        name="kern",
        pattern=r"(?:^|/)logs/kern\.log(?:\.\d+)?$",
        sensor="host kernel ring buffer",
        description="kernel and netfilter messages",
    ),
    LogSource(
        name="messages",
        pattern=r"(?:^|/)logs/messages(?:\.\d+)?$",
        sensor="host syslog",
        description="RHEL-style catch-all syslog facility",
    ),
    LogSource(
        name="user_log",
        pattern=r"(?:^|/)logs/user\.log(?:\.\d+)?$",
        sensor="host syslog",
        description="user-facility syslog",
    ),
    LogSource(
        name="journal",
        pattern=r"(?:^|/)logs/journal/[^/]+/[^/]+\.journal~?$",
        sensor="host systemd journal",
        description="binary systemd journal; needs journalctl to read",
    ),
    LogSource(
        name="mail",
        pattern=(
            r"(?:^|/)logs/(?:mail\.(?:log|info|warn|err)(?:\.\d+)?"
            r"|exim4/[^/]+|sm\.log(?:\.\d+)?)$"
        ),
        sensor="mail transfer agent (Exim4 / sendmail)",
        description="SMTP delivery, queue and rejection records",
    ),
    LogSource(
        name="horde",
        pattern=r"(?:^|/)logs/horde/[^/]+$",
        sensor="webmail application (Horde)",
        description="webmail application log",
    ),
    LogSource(
        name="redis",
        pattern=r"(?:^|/)logs/redis/[^/]+$",
        sensor="cache server (Redis)",
        description="Redis server log",
    ),
    LogSource(
        name="shorewall",
        pattern=r"(?:^|/)logs/shorewall-init\.log(?:\.\d+)?$",
        sensor="host firewall (Shorewall)",
        description="firewall initialisation log",
    ),
    LogSource(
        name="dnsteal",
        pattern=r"(?:^|/)logs/dnsteal(?:\.log)?(?:/[^/]+)?$",
        sensor="attacker tooling (dnsteal)",
        description="DNS-exfiltration tool output on the attacker host",
        note="attacker-side ground truth by construction; not a defender view",
    ),
    LogSource(
        name="attacker_wpdiscuz",
        pattern=r"(?:^|/)logs/ait\.aecid\.attacker\.wpdiscuz/[^/]+$",
        sensor="attacker tooling (wpDiscuz exploit)",
        description="attacker-side exploit tool output",
        note="attacker-side ground truth by construction; not a defender view",
    ),
    LogSource(
        name="attacker_downloads",
        pattern=r"(?:^|/)logs/downloads/[^/]+$",
        sensor="attacker tooling",
        description="files staged or retrieved by the attacker",
        note="attacker-side ground truth by construction; not a defender view",
    ),
    LogSource(
        name="attacks_log",
        pattern=r"(?:^|/)logs/attacks\.log(?:\.\d+)?$",
        sensor="attacker tooling",
        description="attacker's own run log",
        note="attacker-side ground truth by construction; not a defender view",
    ),
)

SOURCES_BY_NAME: dict[str, LogSource] = {s.name: s for s in SOURCES}

#: Source types Part 26 parses into views, in canonical view order.
PARSED_SOURCES: tuple[str, ...] = tuple(s.name for s in SOURCES if s.parsed)

#: Source types found in the archive but deliberately left unparsed.
UNPARSED_SOURCES: tuple[str, ...] = tuple(s.name for s in SOURCES if not s.parsed)


def classify_log_path(path: str) -> str | None:
    """Return the source-type name for an archive path, or None.

    `path` may be a full archive member name (`gather/<host>/logs/...`) or any
    suffix of one; matching is anchored on the `logs/` segment so the host
    directory name is never consulted. The first match in `SOURCES` order
    wins, and the patterns are mutually exclusive by construction (asserted in
    `tests/data/test_ait_lds.py`).
    """
    normalised = path.replace("\\", "/")
    for source in SOURCES:
        if source.regex.search(normalised):
            return source.name
    return None


def host_of(path: str) -> str | None:
    """Return the host directory of a `gather/<host>/...` archive path."""
    parts = path.replace("\\", "/").split("/")
    if len(parts) >= 3 and parts[0] == "gather":
        return parts[1]
    return None


# ---------------------------------------------------------------------------
# Scenario discovery
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LogFile:
    """One concrete log file on disk, with the source type it belongs to."""

    path: Path
    relpath: str
    host: str
    source: str


def _relpath(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def discover_log_files(scenario_root: Path | str) -> list[LogFile]:
    """Every `gather/` log file under a scenario root, classified by source.

    The result is sorted by (source, host, relpath) so that every downstream
    pass visits files in the same order regardless of filesystem order. That
    ordering is what makes `write_processed` byte-reproducible.
    """
    root = Path(scenario_root)
    gather = root / "gather"
    if not gather.is_dir():
        raise FileNotFoundError(
            f"{gather} does not exist. Run `python scripts/fetch_ait.py "
            f"{root.name}` first."
        )

    found: list[LogFile] = []
    for path in sorted(gather.rglob("*")):
        if not path.is_file() or path.name.endswith(".part"):
            continue
        relpath = _relpath(root, path)
        source = classify_log_path(relpath)
        host = host_of(relpath)
        if source is None or host is None:
            continue
        found.append(LogFile(path=path, relpath=relpath, host=host, source=source))
    found.sort(key=lambda f: (f.source, f.host, f.relpath))
    return found


def scenario_time_range(scenario_root: Path | str) -> tuple[float, float]:
    """Read the official scenario start/end from `dataset.yaml`, as epoch seconds.

    AIT ships this file with the archive, so the evaluated interval is the
    dataset's own statement of it rather than something inferred from whichever
    files happened to be downloaded. The timestamps are naive and the logs are
    UTC, so they are read as UTC.
    """
    path = Path(scenario_root) / "dataset.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; it carries the scenario time range")

    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, _, raw = line.partition(":")
        values[key.strip()] = raw.strip().strip("'\"")
    try:
        start = datetime.fromisoformat(values["start"]).replace(tzinfo=timezone.utc)
        end = datetime.fromisoformat(values["end"]).replace(tzinfo=timezone.utc)
    except (KeyError, ValueError) as exc:
        raise RuntimeError(f"{path} has no readable start/end: {values}") from exc
    return start.timestamp(), end.timestamp()


# ---------------------------------------------------------------------------
# Timestamps
# ---------------------------------------------------------------------------

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Two shapes occur in this dataset, and the hostname field is what differs:
#
#   auth.log     "Jan 23 06:25:06 inet-firewall CRON[22309]: pam_unix(...): ..."
#   dnsmasq.log  "Jan 21 00:00:09 dnsmasq[3468]: query[A] foo.example from 10.0.0.1"
#
# dnsmasq is configured here to log without a hostname, so the host group is
# optional. It is tried first (greedy `(?:...)?`), which keeps auth.log parsing
# `inet-firewall` as the host and `CRON` as the program; on a dnsmasq line that
# branch cannot match -- a hostname must be followed by whitespace and `[` is
# next -- so it falls through to program-first. Requiring the hostname instead
# would silently classify every DNS line as unparsed.
SYSLOG_RE = re.compile(
    r"^(?P<month>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\s+"
    r"(?:(?P<host>[^\s:\[]+)\s+)?"
    r"(?P<program>[^\s\[:]+)(?:\[(?P<pid>\d+)\])?:\s*(?P<message>.*)$"
)


class SyslogClock:
    """Turns year-less syslog timestamps into epoch seconds.

    Syslog's `Mmm DD HH:MM:SS` carries no year, so one has to come from
    outside. The scenario's own `dataset.yaml` start supplies it. Lines within
    a file are chronological, so a month that goes backwards means the year
    rolled over; that is carried rather than assumed not to happen, because
    `auth.log.1` can straddle a new year in principle even though
    russellmitchell (2022-01-21 to 2022-01-25) does not.

    One clock per file: the carry must not leak between files.
    """

    def __init__(self, year: int):
        self.year = year
        self._previous_month: int | None = None

    def epoch(self, month: int, day: int, hour: int, minute: int, second: int) -> float:
        if self._previous_month is not None and month < self._previous_month:
            self.year += 1
        self._previous_month = month
        return datetime(
            self.year, month, day, hour, minute, second, tzinfo=timezone.utc
        ).timestamp()


# "192.168.10.4 - - [21/Jan/2022:00:00:01 +0000] "GET /x HTTP/1.1" 200 1234 "-" "UA""
APACHE_RE = re.compile(
    r"^(?P<client>\S+)\s+\S+\s+(?P<user>\S+)\s+"
    r"\[(?P<day>\d{2})/(?P<month>[A-Z][a-z]{2})/(?P<year>\d{4}):"
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})\s+(?P<offset>[+-]\d{4})\]\s+"
    r'"(?P<method>[A-Z!-~]*)\s*(?P<path>\S*)\s*(?P<protocol>[^"]*)"\s+'
    r"(?P<status>\d{3})\s+(?P<bytes>\d+|-)"
    r'(?:\s+"(?P<referer>[^"]*)"\s+"(?P<agent>[^"]*)")?'
)

# "type=SYSCALL msg=audit(1642636800.123:456): arch=c000003e syscall=59 ..."
AUDIT_RE = re.compile(
    r"^type=(?P<type>\S+)\s+msg=audit\((?P<epoch>\d+\.\d+):(?P<serial>\d+)\):\s*(?P<rest>.*)$"
)
AUDIT_FIELD_RE = re.compile(r"(\b[a-z_]+)=((?:\"[^\"]*\")|(?:\S+))")

# dnsmasq, after the syslog prefix: "query[A] foo.example.com from 10.0.0.5"
DNS_QUERY_RE = re.compile(r"^query\[(?P<qtype>[A-Z0-9]+)\]\s+(?P<name>\S+)\s+from\s+(?P<client>\S+)")
DNS_REPLY_RE = re.compile(r"^(?:reply|cached)\s+(?P<name>\S+)\s+is\s+(?P<answer>\S+)")
DNS_FORWARD_RE = re.compile(r"^forwarded\s+(?P<name>\S+)\s+to\s+(?P<server>\S+)")


def _parse_iso8601(text: str) -> float | None:
    """Parse a Suricata `timestamp` field to epoch seconds.

    Suricata writes `2022-01-21T00:00:01.123456+0000`. `fromisoformat` accepts
    `+00:00` but, before 3.11, not `+0000`; the colon is inserted so the same
    code works on either, and a trailing `Z` is handled too.
    """
    if not text:
        return None
    candidate = text.strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    elif len(candidate) > 5 and candidate[-5] in "+-" and candidate[-3] != ":":
        candidate = candidate[:-2] + ":" + candidate[-2:]
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


# ---------------------------------------------------------------------------
# Official labels
# ---------------------------------------------------------------------------


@dataclass
class LabelCoverage:
    """Which (source, host) pairs the official labelling actually examined.

    This exists because absence of a label is not evidence of benign. AIT
    labels a SUBSET of the log files: for russellmitchell, `rules/` carries
    labelling rules for apache, auth, audit, dnsmasq, openvpn and monitoring,
    and the `labels/` tree covers eight files on five hosts. Suricata -- a
    parsed view -- has no labelling rules and no label file at all.

    So a window with no labelled line is benign ONLY with respect to the
    sources the labelling covers. For an unlabelled source the same quiet
    window is simply unexamined, and the two must not be added together.
    """

    #: relpath of every label file found, mapped to its source type
    files: dict[str, str] = field(default_factory=dict)
    #: source type -> hosts whose files of that type are labelled
    hosts_by_source: dict[str, set[str]] = field(default_factory=dict)
    #: parsed source types with no label file anywhere in the scenario
    unlabelled_sources: tuple[str, ...] = ()

    def covers(self, source: str, host: str) -> bool:
        return host in self.hosts_by_source.get(source, set())

    def as_dict(self) -> dict[str, Any]:
        return {
            "label_files": dict(sorted(self.files.items())),
            "labelled_hosts_by_source": {
                source: sorted(hosts)
                for source, hosts in sorted(self.hosts_by_source.items())
            },
            "unlabelled_parsed_sources": list(self.unlabelled_sources),
            "note": (
                "A window is benign only with respect to sources listed in "
                "labelled_hosts_by_source. Sources in "
                "unlabelled_parsed_sources were never examined by the "
                "labelling pass; their quiet windows are not evidence of "
                "benign."
            ),
        }


def read_label_file(path: Path) -> dict[int, list[str]]:
    """Read one AIT label file: 1-based line number -> attack-step names.

    The official format is one JSON object per line, each carrying a `line`
    number into the corresponding `gather/` file and a `labels` list. Only
    lines with a non-empty `labels` list are kept, so the returned mapping
    contains attack lines alone and its size is the labelled-line count.

    A label file is data, not code: it is parsed with `json.loads` per line
    and anything unparseable raises rather than being skipped, because a
    silently dropped label would turn an attack window into a benign one.
    """
    labelled: dict[int, list[str]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for position, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{position} is not valid JSON: {exc}") from exc
            line_no = record.get("line")
            if line_no is None:
                raise RuntimeError(f"{path}:{position} has no `line` field: {record!r}")
            names = record.get("labels") or []
            if isinstance(names, str):
                names = [names]
            if names:
                labelled[int(line_no)] = sorted(str(n) for n in names)
    return labelled


def load_labels(
    scenario_root: Path | str, parsed_sources: Sequence[str] = PARSED_SOURCES
) -> tuple[dict[str, dict[int, list[str]]], LabelCoverage]:
    """Load the whole official `labels/` tree.

    Returns `(labels_by_gather_relpath, coverage)`. A label file at
    `labels/<rest>` describes `gather/<rest>`, so the key is the gather-side
    relpath and lines up with `discover_log_files`.
    """
    root = Path(scenario_root)
    labels_dir = root / "labels"
    if not labels_dir.is_dir():
        raise FileNotFoundError(
            f"{labels_dir} does not exist. The official labels directory is the "
            "only ground truth Part 26 uses; without it there is nothing to label "
            "windows with."
        )

    by_gather: dict[str, dict[int, list[str]]] = {}
    coverage = LabelCoverage()
    for path in sorted(labels_dir.rglob("*")):
        if not path.is_file():
            continue
        relpath = _relpath(root, path)
        source = classify_log_path(relpath)
        host = relpath.split("/")[1] if relpath.count("/") >= 1 else None
        if source is None or host is None:
            continue
        gather_relpath = "gather/" + relpath[len("labels/") :]
        by_gather[gather_relpath] = read_label_file(path)
        coverage.files[relpath] = source
        coverage.hosts_by_source.setdefault(source, set()).add(host)

    coverage.unlabelled_sources = tuple(
        source for source in parsed_sources if source not in coverage.hosts_by_source
    )
    return by_gather, coverage


# ---------------------------------------------------------------------------
# Per-view features
# ---------------------------------------------------------------------------

# Every feature is one of three kinds, and the kind fixes its transform:
#
#   count     an event or distinct-value count, emitted as log1p(count).
#             Raw counts span orders of magnitude between a quiet window and
#             a scan, and log1p keeps a zero at zero.
#   fraction  a share of the window's events, already in [0, 1].
#   value     a magnitude in native units (bytes, characters), also log1p.
#
# No feature is derived from a label, a filename, a host name or an attack
# time range. The accumulator a feature reads is built only from log line
# content.
COUNT = "count"
FRACTION = "fraction"
VALUE = "value"


@dataclass(frozen=True)
class Feature:
    name: str
    kind: str
    description: str


FEATURES: dict[str, tuple[Feature, ...]] = {
    "apache": (
        Feature("n_requests", COUNT, "HTTP requests logged in the window"),
        Feature("n_unique_client", COUNT, "distinct client addresses"),
        Feature("n_unique_path", COUNT, "distinct request paths"),
        Feature("n_unique_agent", COUNT, "distinct User-Agent strings"),
        Feature("frac_status_4xx", FRACTION, "share of 4xx responses"),
        Feature("frac_status_5xx", FRACTION, "share of 5xx responses"),
        Feature("frac_method_post", FRACTION, "share of POST requests"),
        Feature("bytes_total", VALUE, "summed response size"),
        Feature("bytes_mean", VALUE, "mean response size"),
    ),
    "auth": (
        Feature("n_events", COUNT, "authentication log lines in the window"),
        Feature("n_unique_program", COUNT, "distinct syslog programs: sshd, sudo, CRON"),
        Feature("n_unique_pid", COUNT, "distinct process ids"),
        Feature("frac_session_open", FRACTION, "share of session-opened lines"),
        Feature("frac_session_close", FRACTION, "share of session-closed lines"),
        Feature("frac_failure", FRACTION, "share of failure, invalid or denied lines"),
        Feature("frac_sudo", FRACTION, "share of sudo lines"),
        Feature("frac_ssh", FRACTION, "share of sshd lines"),
    ),
    "dns": (
        Feature("n_events", COUNT, "resolver log lines in the window"),
        Feature("n_unique_qname", COUNT, "distinct queried names"),
        Feature("n_unique_client", COUNT, "distinct querying clients"),
        Feature("n_unique_qtype", COUNT, "distinct query types: A, AAAA, TXT"),
        Feature("frac_query", FRACTION, "share of query lines"),
        Feature("frac_answered", FRACTION, "share of reply or cached lines"),
        Feature("frac_forwarded", FRACTION, "share of forwarded lines"),
        Feature("qname_len_mean", VALUE, "mean queried-name length in characters"),
        Feature("qname_len_max", VALUE, "longest queried name in characters"),
    ),
    "audit": (
        Feature("n_events", COUNT, "auditd records in the window"),
        Feature("n_unique_type", COUNT, "distinct record types: SYSCALL, PATH"),
        Feature("n_unique_comm", COUNT, "distinct process names"),
        Feature("n_unique_uid", COUNT, "distinct user ids"),
        Feature("n_unique_event_id", COUNT, "distinct audit event serials"),
        Feature("frac_failed", FRACTION, "share of records reporting success=no"),
        Feature("frac_syscall", FRACTION, "share of SYSCALL records"),
        Feature("frac_execve", FRACTION, "share of EXECVE records"),
        Feature("frac_user_record", FRACTION, "share of USER_ records: auth, login, acct"),
    ),
    "suricata": (
        Feature("n_events", COUNT, "IDS events in the window"),
        Feature("n_unique_event_type", COUNT, "distinct event types"),
        Feature("n_unique_src_ip", COUNT, "distinct source addresses"),
        Feature("n_unique_dest_ip", COUNT, "distinct destination addresses"),
        Feature("n_unique_dest_port", COUNT, "distinct destination ports"),
        Feature("n_unique_signature", COUNT, "distinct alert signatures"),
        Feature("frac_alert", FRACTION, "share of events that are alerts"),
        Feature("frac_http", FRACTION, "share of http events"),
        Feature("frac_dns", FRACTION, "share of dns events"),
        Feature("flow_bytes_total", VALUE, "summed flow byte counts where reported"),
    ),
}

#: D in the contract's `[B, V, D]`: the widest view, with narrower views
#: right-padded. No encoder reads a padded column (see `view_configs`).
FEATURE_DIM = max(len(FEATURES[s]) for s in PARSED_SOURCES)


class Accumulator:
    """Running statistics for one (window, host, source), built from lines only.

    Distinct-value features need the actual values, not just counts, because
    `network` mode takes the UNION across hosts rather than the sum: two hosts
    that both saw the same client address contribute one distinct client, not
    two. Sets are therefore kept and merged, and collapsed to counts only when
    the feature vector is built.

    `attack_steps` and `attack_lines` are label-derived and are the only
    label-derived state here. `vector()` never reads them; that separation is
    what the no-leakage test checks.
    """

    __slots__ = (
        "n", "counts", "sums", "maxima", "uniques",
        "attack_steps", "attack_lines", "covered_events",
    )

    def __init__(self) -> None:
        self.n = 0
        self.counts: Counter[str] = Counter()
        self.sums: Counter[str] = Counter()
        self.maxima: dict[str, float] = {}
        self.uniques: dict[str, set[str]] = {}
        self.attack_steps: set[str] = set()
        self.attack_lines = 0
        # Events that came from a file the official labelling examined.
        # Coverage is per FILE, not per (host, source): AIT labels
        # `...access.log.2` but not `.1`, `.3` or `.4` of the same log on the
        # same host. A window is benign only if some covered file contributed
        # to it, so this count -- not the source name -- decides that.
        self.covered_events = 0

    def hit(self, key: str, amount: int = 1) -> None:
        self.counts[key] += amount

    def add(self, key: str, amount: float) -> None:
        self.sums[key] += amount

    def peak(self, key: str, amount: float) -> None:
        current = self.maxima.get(key)
        if current is None or amount > current:
            self.maxima[key] = amount

    def see(self, key: str, value: str) -> None:
        bucket = self.uniques.get(key)
        if bucket is None:
            bucket = self.uniques[key] = set()
        # Interned, because the same value recurs across thousands of cells and
        # each occurrence would otherwise be a separate string object. One
        # address, port, program name or event type is seen in most windows of
        # a four-day capture, and russellmitchell's Suricata logs alone carry
        # several million events; interning collapses those to one object per
        # distinct value instead of one per (window, host, source) it appears
        # in. The sets retain the values regardless, so nothing is kept alive
        # that would otherwise be freed.
        bucket.add(sys.intern(value))

    def merge(self, other: "Accumulator") -> None:
        self.n += other.n
        self.counts.update(other.counts)
        self.sums.update(other.sums)
        for key, value in other.maxima.items():
            self.peak(key, value)
        for key, values in other.uniques.items():
            bucket = self.uniques.get(key)
            if bucket is None:
                self.uniques[key] = set(values)
            else:
                bucket |= values
        self.attack_steps |= other.attack_steps
        self.attack_lines += other.attack_lines
        self.covered_events += other.covered_events

    # -- feature derivation ------------------------------------------------

    def _share(self, key: str) -> float:
        return (self.counts.get(key, 0) / self.n) if self.n else 0.0

    def _distinct(self, key: str) -> int:
        return len(self.uniques.get(key, ()))

    def vector(self, source: str) -> list[float]:
        """The documented feature vector for `source`, in `FEATURES` order."""
        raw = self.raw(source)
        out: list[float] = []
        for feature in FEATURES[source]:
            value = float(raw[feature.name])
            out.append(value if feature.kind == FRACTION else math.log1p(max(value, 0.0)))
        return out

    def raw(self, source: str) -> dict[str, float]:
        """Untransformed feature values, for the inventory tables and tests."""
        if source == "apache":
            return {
                "n_requests": self.n,
                "n_unique_client": self._distinct("client"),
                "n_unique_path": self._distinct("path"),
                "n_unique_agent": self._distinct("agent"),
                "frac_status_4xx": self._share("status_4xx"),
                "frac_status_5xx": self._share("status_5xx"),
                "frac_method_post": self._share("method_post"),
                "bytes_total": self.sums.get("bytes", 0.0),
                "bytes_mean": (self.sums.get("bytes", 0.0) / self.n) if self.n else 0.0,
            }
        if source == "auth":
            return {
                "n_events": self.n,
                "n_unique_program": self._distinct("program"),
                "n_unique_pid": self._distinct("pid"),
                "frac_session_open": self._share("session_open"),
                "frac_session_close": self._share("session_close"),
                "frac_failure": self._share("failure"),
                "frac_sudo": self._share("sudo"),
                "frac_ssh": self._share("ssh"),
            }
        if source == "dns":
            queries = self.counts.get("query", 0)
            return {
                "n_events": self.n,
                "n_unique_qname": self._distinct("qname"),
                "n_unique_client": self._distinct("client"),
                "n_unique_qtype": self._distinct("qtype"),
                "frac_query": self._share("query"),
                "frac_answered": self._share("answered"),
                "frac_forwarded": self._share("forwarded"),
                "qname_len_mean": (
                    self.sums.get("qname_len", 0.0) / queries if queries else 0.0
                ),
                "qname_len_max": self.maxima.get("qname_len", 0.0),
            }
        if source == "audit":
            return {
                "n_events": self.n,
                "n_unique_type": self._distinct("type"),
                "n_unique_comm": self._distinct("comm"),
                "n_unique_uid": self._distinct("uid"),
                "n_unique_event_id": self._distinct("event_id"),
                "frac_failed": self._share("failed"),
                "frac_syscall": self._share("syscall"),
                "frac_execve": self._share("execve"),
                "frac_user_record": self._share("user_record"),
            }
        if source == "suricata":
            return {
                "n_events": self.n,
                "n_unique_event_type": self._distinct("event_type"),
                "n_unique_src_ip": self._distinct("src_ip"),
                "n_unique_dest_ip": self._distinct("dest_ip"),
                "n_unique_dest_port": self._distinct("dest_port"),
                "n_unique_signature": self._distinct("signature"),
                "frac_alert": self._share("alert"),
                "frac_http": self._share("http"),
                "frac_dns": self._share("dns"),
                "flow_bytes_total": self.sums.get("flow_bytes", 0.0),
            }
        raise ValueError(f"no feature definition for source {source!r}")


# ---------------------------------------------------------------------------
# Line parsers: one per parsed source
# ---------------------------------------------------------------------------
#
# Each parser takes a raw line and a per-file clock, and returns the event
# timestamp in epoch seconds plus the accumulator updates the line implies, or
# None when the line is not an event of that source (a continuation line, a
# blank, a rotation banner). Unparsed lines are counted per file and reported;
# they are never silently dropped, because a parser that quietly matched
# nothing would read as a quiet network.


def _parse_apache(line: str, clock: SyslogClock, acc: Accumulator) -> float | None:
    match = APACHE_RE.match(line)
    if not match:
        return None
    month = MONTHS.get(match["month"])
    if month is None:
        return None
    # Apache logs its own UTC offset, so it is used rather than assumed.
    offset = match["offset"]
    shift = int(offset[1:3]) * 3600 + int(offset[3:5]) * 60
    if offset[0] == "-":
        shift = -shift
    stamp = datetime(
        int(match["year"]), month, int(match["day"]),
        int(match["hour"]), int(match["minute"]), int(match["second"]),
        tzinfo=timezone.utc,
    ).timestamp() - shift

    acc.n += 1
    acc.see("client", match["client"])
    acc.see("path", match["path"])
    acc.see("agent", match["agent"] or "-")
    status = match["status"]
    if status.startswith("4"):
        acc.hit("status_4xx")
    elif status.startswith("5"):
        acc.hit("status_5xx")
    if match["method"] == "POST":
        acc.hit("method_post")
    size = match["bytes"]
    if size != "-":
        acc.add("bytes", float(size))
    return stamp


_AUTH_FAILURE = re.compile(
    r"authentication failure|failed password|invalid user|permission denied"
    r"|access denied|failure;|auth could not identify",
    re.IGNORECASE,
)


def _parse_auth(line: str, clock: SyslogClock, acc: Accumulator) -> float | None:
    match = SYSLOG_RE.match(line)
    if not match:
        return None
    month = MONTHS.get(match["month"])
    if month is None:
        return None
    stamp = clock.epoch(
        month, int(match["day"]), int(match["hour"]),
        int(match["minute"]), int(match["second"]),
    )

    program = match["program"]
    message = match["message"]
    acc.n += 1
    acc.see("program", program)
    if match["pid"]:
        acc.see("pid", match["pid"])
    if "session opened" in message:
        acc.hit("session_open")
    if "session closed" in message:
        acc.hit("session_close")
    if _AUTH_FAILURE.search(message):
        acc.hit("failure")
    if program.startswith("sudo"):
        acc.hit("sudo")
    if program.startswith("sshd"):
        acc.hit("ssh")
    return stamp


def _parse_dns(line: str, clock: SyslogClock, acc: Accumulator) -> float | None:
    match = SYSLOG_RE.match(line)
    if not match:
        return None
    month = MONTHS.get(match["month"])
    if month is None:
        return None
    stamp = clock.epoch(
        month, int(match["day"]), int(match["hour"]),
        int(match["minute"]), int(match["second"]),
    )

    message = match["message"]
    acc.n += 1
    query = DNS_QUERY_RE.match(message)
    if query:
        name = query["name"]
        acc.hit("query")
        acc.see("qname", name)
        acc.see("qtype", query["qtype"])
        acc.see("client", query["client"])
        # Query-name length is the DNS-exfiltration signal: a tool that
        # smuggles bytes out in subdomain labels produces long names. It is a
        # property of the line, not of any label.
        acc.add("qname_len", len(name))
        acc.peak("qname_len", len(name))
        return stamp
    reply = DNS_REPLY_RE.match(message)
    if reply:
        acc.hit("answered")
        acc.see("qname", reply["name"])
        return stamp
    forwarded = DNS_FORWARD_RE.match(message)
    if forwarded:
        acc.hit("forwarded")
        acc.see("qname", forwarded["name"])
    return stamp


_AUDIT_USER_TYPES = (
    "USER_AUTH", "USER_ACCT", "USER_LOGIN", "USER_START", "USER_END",
    "USER_CMD", "USER_ERR", "USER_AVC", "USER_MGMT", "USER_CHAUTHTOK",
    "CRED_ACQ", "CRED_DISP", "CRED_REFR", "LOGIN",
)


def _parse_audit(line: str, clock: SyslogClock, acc: Accumulator) -> float | None:
    match = AUDIT_RE.match(line)
    if not match:
        return None
    record_type = match["type"]
    stamp = float(match["epoch"])

    acc.n += 1
    acc.see("type", record_type)
    acc.see("event_id", match["serial"])
    if record_type == "SYSCALL":
        acc.hit("syscall")
    elif record_type == "EXECVE":
        acc.hit("execve")
    if record_type in _AUDIT_USER_TYPES:
        acc.hit("user_record")

    for key, value in AUDIT_FIELD_RE.findall(match["rest"]):
        if key == "comm":
            acc.see("comm", value.strip('"'))
        elif key in ("uid", "auid"):
            acc.see("uid", value)
        elif key == "success" and value.strip('"') == "no":
            acc.hit("failed")
        elif key == "res" and value.strip('"') in ("failed", "0"):
            acc.hit("failed")
    return stamp


def _parse_suricata(line: str, clock: SyslogClock, acc: Accumulator) -> float | None:
    stripped = line.strip()
    if not stripped or stripped[0] != "{":
        return None
    try:
        event = json.loads(stripped)
    except json.JSONDecodeError:
        # A truncated final line is the one tolerable case here, and it is
        # counted as unparsed rather than guessed at.
        return None
    stamp = _parse_iso8601(event.get("timestamp", ""))
    if stamp is None:
        return None

    event_type = str(event.get("event_type", "unknown"))
    acc.n += 1
    acc.see("event_type", event_type)
    if event_type == "alert":
        acc.hit("alert")
        signature = (event.get("alert") or {}).get("signature")
        if signature:
            acc.see("signature", str(signature))
    elif event_type == "http":
        acc.hit("http")
    elif event_type == "dns":
        acc.hit("dns")

    for key in ("src_ip", "dest_ip"):
        value = event.get(key)
        if value:
            acc.see(key, str(value))
    port = event.get("dest_port")
    if port is not None:
        acc.see("dest_port", str(port))

    flow = event.get("flow")
    if isinstance(flow, dict):
        total = 0.0
        for key in ("bytes_toserver", "bytes_toclient"):
            value = flow.get(key)
            if isinstance(value, (int, float)):
                total += float(value)
        if total:
            acc.add("flow_bytes", total)
    return stamp


PARSERS = {
    "apache": _parse_apache,
    "auth": _parse_auth,
    "dns": _parse_dns,
    "audit": _parse_audit,
    "suricata": _parse_suricata,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AITConfig:
    """Everything that determines the processed output.

    The whole object is written into the output manifest, so a processed
    directory always carries the configuration that produced it. Two runs with
    an equal config produce byte-identical arrays
    (`tests/data/test_ait_lds.py::test_identical_config_is_byte_identical`).
    """

    scenario: str
    window_seconds: int = DEFAULT_WINDOW_SECONDS
    sample_mode: str = "network"
    #: Parsed source types, in view order. Defaults to the brief's five.
    sources: tuple[str, ...] = PARSED_SOURCES
    #: "scenario" takes the window grid from the archive's own dataset.yaml;
    #: "observed" takes it from the first and last event actually parsed.
    #: The default is the dataset's own statement of its interval.
    time_range: str = "scenario"
    #: Drop windows in which no view has any event. In network mode these are
    #: gaps in the whole capture; keeping them would add samples whose
    #: view_mask is entirely False and whose features are all zero.
    drop_empty_windows: bool = True
    raw_root: str = str(DEFAULT_RAW_ROOT)

    def __post_init__(self) -> None:
        if self.sample_mode not in SAMPLE_MODES:
            raise ValueError(
                f"sample_mode must be one of {SAMPLE_MODES}, got {self.sample_mode!r}"
            )
        if self.time_range not in ("scenario", "observed"):
            raise ValueError(
                f"time_range must be 'scenario' or 'observed', got {self.time_range!r}"
            )
        if self.window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {self.window_seconds}")
        unknown = [s for s in self.sources if s not in FEATURES]
        if unknown:
            raise ValueError(
                f"no feature definition for {unknown}; parsed sources are {PARSED_SOURCES}"
            )

    @property
    def scenario_root(self) -> Path:
        return Path(self.raw_root) / self.scenario

    @property
    def output_name(self) -> str:
        return f"ait_{self.scenario}_{self.sample_mode}_{self.window_seconds}s"

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "window_seconds": self.window_seconds,
            "sample_mode": self.sample_mode,
            "sources": list(self.sources),
            "time_range": self.time_range,
            "drop_empty_windows": self.drop_empty_windows,
            "raw_root": self.raw_root,
        }


# ---------------------------------------------------------------------------
# One streaming pass over the raw logs
# ---------------------------------------------------------------------------


@dataclass
class FileStats:
    """What one log file contributed. The inventory table is built from these."""

    relpath: str
    host: str
    source: str
    lines: int = 0
    events: int = 0
    unparsed: int = 0
    out_of_range: int = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None
    label_covered: bool = False
    labelled_lines: int = 0
    attack_lines_in_range: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.relpath,
            "host": self.host,
            "source": self.source,
            "lines": self.lines,
            "events_parsed": self.events,
            "lines_unparsed": self.unparsed,
            "events_outside_window_grid": self.out_of_range,
            "first_timestamp": _iso(self.first_timestamp),
            "last_timestamp": _iso(self.last_timestamp),
            "label_file_present": self.label_covered,
            "labelled_attack_lines": self.labelled_lines,
        }


def _iso(stamp: float | None) -> str | None:
    if stamp is None:
        return None
    return datetime.fromtimestamp(stamp, tz=timezone.utc).isoformat()


@dataclass
class ScanResult:
    """The single parse of a scenario, from which both sample modes are built.

    Cells are keyed `(window_index, host, source)`. Both modes are assembled
    from this one structure, so the 2.3 GB of logs are read once per
    (scenario, window length) rather than once per mode.
    """

    config: AITConfig
    cells: dict[tuple[int, str, str], Accumulator]
    files: list[FileStats]
    coverage: LabelCoverage
    origin: float
    n_windows: int
    hosts: tuple[str, ...]
    #: True only when the raw fetch manifest says every selected entry
    #: verified. False also covers "no manifest", i.e. a hand-built fixture.
    fetch_complete: bool = False

    @property
    def window_seconds(self) -> int:
        return self.config.window_seconds

    def window_start(self, index: int) -> float:
        return self.origin + index * self.window_seconds


def scan_scenario(
    config: AITConfig,
    progress: bool = False,
    allow_incomplete_fetch: bool = False,
) -> ScanResult:
    """Parse every file of every configured source into per-window cells.

    Memory is bounded by the number of (window, host, source) cells and the
    distinct values each holds, not by the size of the logs: lines are consumed
    one at a time and never retained. Suricata's `eve.json` is 405 MB on one
    host, so this matters.

    A PARTIAL fetch is refused. `scripts/fetch_ait.py` rewrites its manifest
    after every member and records `complete`, and a scan over a half-fetched
    scenario produces event counts, attack-window counts and missing-view
    fractions that look entirely reasonable and are simply wrong -- the
    missing views are missing because the file has not arrived yet, not
    because the sensor was quiet. That is the kind of silent wrongness this
    project's rules exist to prevent, so it raises unless the caller says
    otherwise, and either way the answer is recorded in the processed
    manifest.
    """
    fetch_complete = _require_complete_fetch(config, allow_incomplete_fetch)
    root = config.scenario_root
    wanted = set(config.sources)
    files = [f for f in discover_log_files(root) if f.source in wanted]
    if not files:
        raise FileNotFoundError(
            f"no files of sources {sorted(wanted)} under {root}/gather. "
            f"Run `python scripts/fetch_ait.py {config.scenario}` first."
        )

    labels_by_file, coverage = load_labels(root, config.sources)

    if config.time_range == "scenario":
        start, end = scenario_time_range(root)
    else:
        start, end = _observed_range(files, config)
    origin = start
    n_windows = max(1, math.ceil((end - start) / config.window_seconds))

    year = datetime.fromtimestamp(origin, tz=timezone.utc).year
    cells: dict[tuple[int, str, str], Accumulator] = {}
    stats: list[FileStats] = []

    for log in files:
        parser = PARSERS[log.source]
        attack_lines = labels_by_file.get(log.relpath)
        covered = log.relpath in labels_by_file
        record = FileStats(
            relpath=log.relpath,
            host=log.host,
            source=log.source,
            label_covered=covered,
            labelled_lines=len(attack_lines) if attack_lines else 0,
        )
        # One clock per file: the syslog year-rollover carry must not leak
        # between files, which are each independently chronological.
        clock = SyslogClock(year)

        with log.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_no, line in enumerate(handle, start=1):
                record.lines += 1
                # One scratch accumulator per line, merged into its window's
                # cell only once the line is known to be an in-range event.
                scratch = Accumulator()
                stamp = parser(line, clock, scratch)
                if stamp is None:
                    record.unparsed += 1
                    continue
                record.events += 1
                if record.first_timestamp is None or stamp < record.first_timestamp:
                    record.first_timestamp = stamp
                if record.last_timestamp is None or stamp > record.last_timestamp:
                    record.last_timestamp = stamp

                index = int((stamp - origin) // config.window_seconds)
                if index < 0 or index >= n_windows:
                    record.out_of_range += 1
                    continue

                key = (index, log.host, log.source)
                cell = cells.get(key)
                if cell is None:
                    cell = cells[key] = Accumulator()
                cell.merge(scratch)
                if covered:
                    cell.covered_events += 1
                    names = attack_lines.get(line_no) if attack_lines else None
                    if names:
                        cell.attack_lines += 1
                        cell.attack_steps.update(names)
                        record.attack_lines_in_range += 1

        stats.append(record)
        if progress:
            print(
                f"  {log.relpath}: {record.events} events, "
                f"{record.unparsed} unparsed, {record.labelled_lines} labelled",
                flush=True,
            )

    hosts = tuple(sorted({f.host for f in files}))
    return ScanResult(
        config=config,
        cells=cells,
        files=stats,
        coverage=coverage,
        origin=origin,
        n_windows=n_windows,
        hosts=hosts,
        fetch_complete=fetch_complete,
    )


def _require_complete_fetch(config: AITConfig, allow_incomplete: bool) -> bool:
    """Return whether the raw fetch is known complete; raise if it is not.

    Three states, and they are different. `True` means the fetch manifest says
    every selected entry verified. `None` means there is no manifest at all,
    which is the case for a hand-built fixture and is allowed. `False` means
    the fetcher stopped part-way, and that is refused.
    """
    manifest = raw_manifest(config)
    if not manifest:
        return False
    complete = manifest.get("complete")
    if complete is False and not allow_incomplete:
        fetched = len(manifest.get("entries", []))
        selected = manifest.get("selected_entries")
        raise RuntimeError(
            f"{config.scenario_root / 'manifest.json'} reports an incomplete "
            f"fetch: {fetched} of {selected} selected entries verified. Scanning "
            "now would report missing views that are merely unfetched. Finish "
            f"`python scripts/fetch_ait.py {config.scenario}` (it resumes), or "
            "pass allow_incomplete_fetch=True and say so in the write-up."
        )
    return bool(complete)


def _observed_range(files: Sequence[LogFile], config: AITConfig) -> tuple[float, float]:
    """First and last event timestamp across the configured files.

    Used only for `time_range="observed"`. It costs a second read of every
    file, which is why the scenario's own declared range is the default.
    """
    low: float | None = None
    high: float | None = None
    # The declared scenario start is still the only source of a syslog year,
    # since syslog lines carry none, so dataset.yaml is read even here.
    year = datetime.fromtimestamp(
        scenario_time_range(config.scenario_root)[0], tz=timezone.utc
    ).year
    for log in files:
        parser = PARSERS[log.source]
        clock = SyslogClock(year)
        with log.path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stamp = parser(line, clock, Accumulator())
                if stamp is None:
                    continue
                if low is None or stamp < low:
                    low = stamp
                if high is None or stamp > high:
                    high = stamp
    if low is None or high is None:
        raise RuntimeError("no parseable event in any configured file")
    return low, high + 1.0


# ---------------------------------------------------------------------------
# Samples: both modes from one scan
# ---------------------------------------------------------------------------

#: `labels` uses this for a window the official labelling never examined.
#: It is NOT a class. A window with no labelled source present is not benign;
#: it is unknown, and Part 28 must filter on `label_known` before training.
LABEL_UNKNOWN = -1


@dataclass
class AITSamples:
    """The contract's tensors for one (scenario, mode, window length).

    `views` is `[B, V, D]`, `view_mask` is `[B, V]`, `labels` is `[B]`
    (CONTRACT section 4). `label_known` carries the third label state that a
    partially-labelled dataset forces; see `LABEL_UNKNOWN`.
    """

    views: np.ndarray            # [B, V, D] float32
    view_mask: np.ndarray        # [B, V] bool
    labels: np.ndarray           # [B] int64, in {-1, 0, 1}
    label_known: np.ndarray      # [B] bool
    window_start: np.ndarray     # [B] float64, epoch seconds UTC
    host_id: np.ndarray | None   # [B] int64, host mode only
    hosts: tuple[str, ...]       # host_id -> name
    sources: tuple[str, ...]     # view index -> source name
    attack_steps: list[list[str]]
    config: AITConfig

    def __len__(self) -> int:
        return int(self.views.shape[0])

    @property
    def n_views(self) -> int:
        return int(self.views.shape[1])

    def statistics(self) -> dict[str, Any]:
        """Dataset statistics only -- no model quantity is computed here."""
        total = len(self)
        present = self.view_mask.sum(axis=0)
        return {
            "n_samples": total,
            "n_views": self.n_views,
            "feature_dim": int(self.views.shape[2]),
            "window_seconds": self.config.window_seconds,
            "sample_mode": self.config.sample_mode,
            "n_attack_windows": int((self.labels == 1).sum()),
            "n_benign_windows": int((self.labels == 0).sum()),
            "n_unlabelled_windows": int((self.labels == LABEL_UNKNOWN).sum()),
            "n_hosts": len(self.hosts),
            "views": {
                source: {
                    "present_windows": int(present[index]),
                    "missing_windows": int(total - present[index]),
                    "missing_fraction": (
                        float(total - present[index]) / total if total else None
                    ),
                    "feature_names": [f.name for f in FEATURES[source]],
                    "feature_dim": len(FEATURES[source]),
                }
                for index, source in enumerate(self.sources)
            },
            "mean_views_present": (
                float(self.view_mask.sum(axis=1).mean()) if total else None
            ),
            "fully_masked_windows": int((~self.view_mask.any(axis=1)).sum()),
        }


def build_samples(scan: ScanResult, sample_mode: str | None = None) -> AITSamples:
    """Assemble one mode's tensors from an already-scanned scenario.

    network mode merges every host's cell for a source into one view, taking
    the union of distinct-value sets rather than the sum of their sizes.
    host mode keeps them separate and emits one sample per (host, window).
    """
    mode = sample_mode or scan.config.sample_mode
    if mode not in SAMPLE_MODES:
        raise ValueError(f"sample_mode must be one of {SAMPLE_MODES}, got {mode!r}")
    sources = tuple(scan.config.sources)
    config = (
        scan.config
        if mode == scan.config.sample_mode
        else _with_mode(scan.config, mode)
    )

    # Group cells into (sample key) -> source -> merged accumulator. The key is
    # the window alone in network mode, and (host, window) in host mode, which
    # is the only difference between the two.
    grouped: dict[tuple, dict[str, Accumulator]] = {}
    for (index, host, source), cell in scan.cells.items():
        key = (index,) if mode == "network" else (host, index)
        bucket = grouped.setdefault(key, {})
        merged = bucket.get(source)
        if merged is None:
            merged = bucket[source] = Accumulator()
        merged.merge(cell)

    if mode == "host":
        keys = sorted(grouped, key=lambda k: (k[0], k[1]))
        hosts = tuple(sorted({str(k[0]) for k in grouped}))
    else:
        keys = sorted(grouped, key=lambda k: k[0])
        hosts = scan.hosts

    if not scan.config.drop_empty_windows:
        keys = _all_keys(scan, mode, hosts)

    host_index = {name: position for position, name in enumerate(hosts)}
    n_samples = len(keys)
    n_views = len(sources)

    views = np.zeros((n_samples, n_views, FEATURE_DIM), dtype=np.float32)
    view_mask = np.zeros((n_samples, n_views), dtype=bool)
    labels = np.full(n_samples, LABEL_UNKNOWN, dtype=np.int64)
    label_known = np.zeros(n_samples, dtype=bool)
    window_start = np.zeros(n_samples, dtype=np.float64)
    host_id = np.zeros(n_samples, dtype=np.int64) if mode == "host" else None
    attack_steps: list[list[str]] = []

    for row, key in enumerate(keys):
        bucket = grouped.get(key, {})
        index = key[-1]
        window_start[row] = scan.window_start(index)
        if host_id is not None:
            host_id[row] = host_index[str(key[0])]

        covered_events = 0
        attack_lines = 0
        steps: set[str] = set()
        for column, source in enumerate(sources):
            cell = bucket.get(source)
            if cell is None or cell.n == 0:
                continue
            vector = cell.vector(source)
            views[row, column, : len(vector)] = vector
            view_mask[row, column] = True
            covered_events += cell.covered_events
            attack_lines += cell.attack_lines
            steps |= cell.attack_steps

        # The three-state label. `covered_events` counts events from files the
        # official labelling examined; without one, a quiet window carries no
        # evidence either way and must not be scored as benign.
        if attack_lines > 0:
            labels[row] = 1
            label_known[row] = True
        elif covered_events > 0:
            labels[row] = 0
            label_known[row] = True
        else:
            labels[row] = LABEL_UNKNOWN
            label_known[row] = False
        attack_steps.append(sorted(steps))

    return AITSamples(
        views=views,
        view_mask=view_mask,
        labels=labels,
        label_known=label_known,
        window_start=window_start,
        host_id=host_id,
        hosts=tuple(hosts),
        sources=sources,
        attack_steps=attack_steps,
        config=config,
    )


def rewindow(scan: ScanResult, window_seconds: int) -> ScanResult:
    """Re-bin an existing scan to a LONGER window that is a multiple of its own.

    Every feature in `FEATURES` is merge-closed: event and distinct-value
    counts are sums and unions, byte and length totals are sums, the maximum is
    a maximum, and every fraction is derived from the merged counts at
    `vector()` time rather than averaged. So merging the cells of five
    consecutive 60 s windows gives exactly the cells of one 300 s window
    starting at the same origin -- not an approximation of them.

    That makes a window-length sweep one parse instead of one parse per length,
    which matters because scanning russellmitchell means reading 2.3 GB and
    json-decoding several million Suricata events.
    `tests/data/test_ait_lds.py::test_rewindow_equals_a_direct_scan` checks the
    equality against a direct scan rather than taking the argument on trust.

    Only exact multiples are allowed. A non-multiple would split events across
    a boundary that the finer scan already committed to, which would silently
    produce windows that are not the requested length.
    """
    factor, remainder = divmod(window_seconds, scan.window_seconds)
    if remainder or factor < 1:
        raise ValueError(
            f"{window_seconds}s is not a positive multiple of this scan's "
            f"{scan.window_seconds}s window; re-scan instead of re-binning"
        )
    if factor == 1:
        return scan

    cells: dict[tuple[int, str, str], Accumulator] = {}
    for (index, host, source), cell in scan.cells.items():
        key = (index // factor, host, source)
        merged = cells.get(key)
        if merged is None:
            merged = cells[key] = Accumulator()
        merged.merge(cell)

    return ScanResult(
        config=_with_window(scan.config, window_seconds),
        cells=cells,
        files=scan.files,
        coverage=scan.coverage,
        origin=scan.origin,
        n_windows=math.ceil(scan.n_windows / factor),
        hosts=scan.hosts,
        fetch_complete=scan.fetch_complete,
    )


def _with_window(config: AITConfig, window_seconds: int) -> AITConfig:
    return AITConfig(
        scenario=config.scenario,
        window_seconds=window_seconds,
        sample_mode=config.sample_mode,
        sources=config.sources,
        time_range=config.time_range,
        drop_empty_windows=config.drop_empty_windows,
        raw_root=config.raw_root,
    )


def _with_mode(config: AITConfig, mode: str) -> AITConfig:
    """The same config with a different sample_mode, so one scan serves both."""
    return AITConfig(
        scenario=config.scenario,
        window_seconds=config.window_seconds,
        sample_mode=mode,
        sources=config.sources,
        time_range=config.time_range,
        drop_empty_windows=config.drop_empty_windows,
        raw_root=config.raw_root,
    )


def _all_keys(scan: ScanResult, mode: str, hosts: Sequence[str]) -> list[tuple]:
    """Every window on the grid, including those no view reported.

    Only used when `drop_empty_windows` is False.
    """
    if mode == "network":
        return [(index,) for index in range(scan.n_windows)]
    return [
        (host, index)
        for host in sorted(hosts)
        for index in range(scan.n_windows)
    ]


# ---------------------------------------------------------------------------
# Processed output
# ---------------------------------------------------------------------------

ARRAY_FILES = {
    "views": "views.npy",
    "view_mask": "view_mask.npy",
    "labels": "labels.npy",
    "label_known": "label_known.npy",
    "window_start": "window_start.npy",
    "host_id": "host_id.npy",
}


def write_processed(
    samples: AITSamples,
    scan: ScanResult,
    raw_manifest: dict[str, Any] | None = None,
    processed_root: Path | str = DEFAULT_PROCESSED_ROOT,
) -> Path:
    """Write one mode's tensors and manifest to `data/processed/<output_name>/`.

    Arrays go out as `.npy`, which is a fixed header plus the raw buffer, so an
    identical config yields byte-identical files. Nothing here is pickled:
    `allow_pickle` is left at its default False on read, and the arrays are
    plain numeric types.
    """
    destination = Path(processed_root) / samples.config.output_name
    destination.mkdir(parents=True, exist_ok=True)

    np.save(destination / ARRAY_FILES["views"], samples.views)
    np.save(destination / ARRAY_FILES["view_mask"], samples.view_mask)
    np.save(destination / ARRAY_FILES["labels"], samples.labels)
    np.save(destination / ARRAY_FILES["label_known"], samples.label_known)
    np.save(destination / ARRAY_FILES["window_start"], samples.window_start)
    if samples.host_id is not None:
        np.save(destination / ARRAY_FILES["host_id"], samples.host_id)

    (destination / "manifest.json").write_text(
        json.dumps(_manifest(samples, scan, raw_manifest), indent=2, sort_keys=False) + "\n"
    )
    (destination / "attack_steps.json").write_text(
        json.dumps(
            {
                str(row): steps
                for row, steps in enumerate(samples.attack_steps)
                if steps
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def _manifest(
    samples: AITSamples,
    scan: ScanResult,
    raw_manifest: dict[str, Any] | None,
) -> dict[str, Any]:
    raw = raw_manifest or {}
    return {
        "part": 26,
        "dataset": "AIT Log Data Set v2.0",
        "scenario": samples.config.scenario,
        "config": samples.config.as_dict(),
        "window_grid": {
            "origin": _iso(scan.origin),
            "n_windows_on_grid": scan.n_windows,
            "window_seconds": scan.window_seconds,
            "source_of_range": (
                "dataset.yaml start/end"
                if samples.config.time_range == "scenario"
                else "first and last parsed event"
            ),
        },
        "arrays": {
            "views": {
                "file": ARRAY_FILES["views"],
                "shape": list(samples.views.shape),
                "dtype": str(samples.views.dtype),
                "layout": "[B, V, D] per CONTRACT section 4, zero-padded on D",
            },
            "view_mask": {
                "file": ARRAY_FILES["view_mask"],
                "shape": list(samples.view_mask.shape),
                "dtype": "bool",
                "meaning": "True where that source reported at least one event",
            },
            "labels": {
                "file": ARRAY_FILES["labels"],
                "shape": list(samples.labels.shape),
                "dtype": "int64",
                "values": {"1": "attack", "0": "benign", "-1": "not labelled"},
            },
            "label_known": {
                "file": ARRAY_FILES["label_known"],
                "shape": list(samples.label_known.shape),
                "dtype": "bool",
                "meaning": (
                    "True where at least one label-covered file contributed an "
                    "event. Filter on this before training: labels == 0 is "
                    "benign only where this is True."
                ),
            },
            "window_start": {
                "file": ARRAY_FILES["window_start"],
                "shape": list(samples.window_start.shape),
                "dtype": "float64",
                "meaning": "window start, epoch seconds UTC",
            },
            **(
                {
                    "host_id": {
                        "file": ARRAY_FILES["host_id"],
                        "shape": list(samples.host_id.shape),
                        "dtype": "int64",
                        "meaning": "index into `hosts`",
                    }
                }
                if samples.host_id is not None
                else {}
            ),
        },
        "hosts": list(samples.hosts),
        "views": [
            {
                "index": index,
                "source": source,
                "sensor": SOURCES_BY_NAME[source].sensor,
                "same_sensor_group": SOURCES_BY_NAME[source].same_sensor_group,
                "features": [
                    {"name": f.name, "kind": f.kind, "description": f.description}
                    for f in FEATURES[source]
                ],
            }
            for index, source in enumerate(samples.sources)
        ],
        "statistics": samples.statistics(),
        "label_coverage": scan.coverage.as_dict(),
        "source_files": [record.as_dict() for record in scan.files],
        "raw_fetch": {
            "zenodo_record": raw.get("zenodo_record"),
            "archive": raw.get("archive"),
            "archive_size_bytes": raw.get("archive_size_bytes"),
            "complete": scan.fetch_complete,
            "entries": raw.get("entries", []),
        },
        "no_label_in_features": (
            "Features are computed in scan_scenario from gather/ log line "
            "content only. Labels are read separately from labels/ and are "
            "merged into Accumulator.attack_steps / attack_lines / "
            "covered_events, which Accumulator.vector() never reads."
        ),
    }


def load_processed(
    directory: Path | str, processed_root: Path | str = DEFAULT_PROCESSED_ROOT
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read back a processed directory. `allow_pickle` stays False."""
    path = Path(directory)
    if not path.is_dir():
        path = Path(processed_root) / directory
    arrays = {
        key: np.load(path / filename, allow_pickle=False)
        for key, filename in ARRAY_FILES.items()
        if (path / filename).is_file()
    }
    manifest = json.loads((path / "manifest.json").read_text())
    return arrays, manifest


# ---------------------------------------------------------------------------
# Dataset, behind the existing interface
# ---------------------------------------------------------------------------


class AITMultiViewDataset:
    """In-memory AIT dataset, interface-identical to `MultiViewDataset`.

    Same attributes and the same `get_sample` / `get_batch` return types as
    the synthetic loader and `RealMultiViewDataset`, so Part 28 can train on
    it without touching an experiment script.

    `rho_matrix` is None for the reason given in
    `prismflow/data/real_datasets.py` and `docs/DATASETS.md`: real views have
    no ground-truth cross-view dependence, and this attribute must fail loudly
    rather than hand an estimate to code that wants a target.

    UNLIKE the handwritten loader, this one does NOT z-score. Standardisation
    has to be fitted on training indices only, and the split is Part 28's
    business; features here are the documented raw statistics with the log1p
    transform named in `FEATURES`. The one-line transform is in the manifest so
    a later Part scales from a known starting point rather than guessing.
    """

    def __init__(self, samples: AITSamples):
        self.views = torch.as_tensor(samples.views, dtype=torch.float32)  # [N, V, D]
        self.view_mask = torch.as_tensor(samples.view_mask, dtype=torch.bool)  # [N, V]
        self.labels = torch.as_tensor(samples.labels, dtype=torch.long)  # [N]
        self.label_known = torch.as_tensor(samples.label_known, dtype=torch.bool)
        self.window_start = torch.as_tensor(samples.window_start, dtype=torch.float64)
        self.host_id = (
            torch.as_tensor(samples.host_id, dtype=torch.long)
            if samples.host_id is not None
            else None
        )
        self.rho_matrix = None
        self.config = samples.config
        self.samples = samples
        self.view_dims = tuple(len(FEATURES[s]) for s in samples.sources)
        # Attack vs not, over the windows the labelling covers. A model trained
        # on this must be given only `label_known` rows; see LABEL_UNKNOWN.
        self.n_classes = 2
        self.sources = samples.sources

    def __len__(self) -> int:
        return int(self.views.shape[0])

    @property
    def n_views(self) -> int:
        return int(self.views.shape[1])

    def get_sample(self, index: int) -> Sample:
        return Sample(
            sample_id=index,
            views=self.views[index],
            label=int(self.labels[index]),
            view_mask=self.view_mask[index],
            metadata={
                "dataset": f"ait_{self.config.scenario}",
                "sample_mode": self.config.sample_mode,
                "window_seconds": self.config.window_seconds,
                "window_start": float(self.window_start[index]),
                "label_known": bool(self.label_known[index]),
                "attack_steps": list(self.samples.attack_steps[index]),
                "host": (
                    self.samples.hosts[int(self.host_id[index])]
                    if self.host_id is not None
                    else None
                ),
            },
        )

    def get_batch(self, indices) -> Batch:
        idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        return Batch(
            views=self.views[idx],
            view_mask=self.view_mask[idx],
            labels=self.labels[idx],
            sample_ids=idx,
        )


def view_configs(
    sources: Sequence[str] = PARSED_SOURCES,
    hidden_dims: Sequence[int] = (64, 64),
    feature_dim: int = 32,
    dropout: float = 0.0,
) -> list[EncoderConfig]:
    """One `EncoderConfig` per view, each with that view's NATIVE input_dim.

    Views here are 8 to 10 features wide and padded to D = `FEATURE_DIM`.
    Declaring the native width per view is what keeps the padded columns
    unreachable, exactly as in `real_datasets.view_configs`.
    """
    return [
        EncoderConfig(
            input_dim=len(FEATURES[source]),
            hidden_dims=list(hidden_dims),
            feature_dim=feature_dim,
            dropout=dropout,
        )
        for source in sources
    ]


def sensor_groups(sources: Sequence[str] = PARSED_SOURCES) -> dict[str, list[int]]:
    """Group name -> view indices that share one physical sensor.

    Views in the same group are not independent observations. Part 26 reports
    this; it does not act on it. Suricata is alone in `network` until the
    packet-capture view is built, and the group is there so that view joins an
    existing statement rather than arriving as an apparently unrelated one.
    """
    groups: dict[str, list[int]] = {}
    for index, source in enumerate(sources):
        group = SOURCES_BY_NAME[source].same_sensor_group
        if group:
            groups.setdefault(group, []).append(index)
    return groups


def build_ait_dataset(
    config: AITConfig, progress: bool = False
) -> tuple[AITMultiViewDataset, AITSamples, ScanResult]:
    """Scan a scenario and build one mode's dataset.

    Returns the dataset, its samples and the scan, so a caller that wants the
    other mode can call `build_samples(scan, "host")` without re-reading the
    logs.
    """
    scan = scan_scenario(config, progress=progress)
    samples = build_samples(scan)
    return AITMultiViewDataset(samples), samples, scan


def raw_manifest(config: AITConfig) -> dict[str, Any]:
    """The fetch manifest for a scenario, or {} when it has not been fetched."""
    path = config.scenario_root / "manifest.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def inventory(scan: ScanResult) -> list[dict[str, Any]]:
    """Per-source inventory rows for `docs/part26-ait-loader.md` (Step 2)."""
    rows: list[dict[str, Any]] = []
    for source in scan.config.sources:
        records = [f for f in scan.files if f.source == source]
        if not records:
            continue
        firsts = [f.first_timestamp for f in records if f.first_timestamp is not None]
        lasts = [f.last_timestamp for f in records if f.last_timestamp is not None]
        rows.append(
            {
                "source": source,
                "sensor": SOURCES_BY_NAME[source].sensor,
                "same_sensor_group": SOURCES_BY_NAME[source].same_sensor_group,
                "hosts": sorted({f.host for f in records}),
                "n_hosts": len({f.host for f in records}),
                "n_files": len(records),
                "lines": sum(f.lines for f in records),
                "events_parsed": sum(f.events for f in records),
                "lines_unparsed": sum(f.unparsed for f in records),
                "events_outside_grid": sum(f.out_of_range for f in records),
                "first_timestamp": _iso(min(firsts)) if firsts else None,
                "last_timestamp": _iso(max(lasts)) if lasts else None,
                "label_files": sum(1 for f in records if f.label_covered),
                "labelled_attack_lines": sum(f.labelled_lines for f in records),
            }
        )
    return rows


def found_not_used(scenario_root: Path | str) -> list[dict[str, Any]]:
    """Source types present in the scenario that Part 26 does not parse.

    Reported so the inventory is complete: a source absent from this list and
    from `inventory()` is a source the registry does not know about, which is
    a defect rather than a silent omission.

    The census comes from the FETCH MANIFEST's `archive_sources`, which
    `scripts/fetch_ait.py` builds from the archive's central directory. It has
    to: Part 26 fetches five source types and leaves twenty-one unfetched, so
    a census taken by walking `data/raw/` would report every unparsed source
    as absent -- exactly the sources this list is for. Only when no manifest
    census exists does it fall back to what is on disk, and `census_source`
    says which happened so the reader is not left guessing.

    `lines` and timestamps are absent for these rows and stay absent: they
    cannot be known without the files, and this Part does not estimate them.
    """
    root = Path(scenario_root)
    rows: list[dict[str, Any]] = []

    manifest_path = root / "manifest.json"
    census: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            census = json.loads(manifest_path.read_text()).get("archive_sources") or {}
        except json.JSONDecodeError:
            census = {}

    if census:
        for source in UNPARSED_SOURCES:
            entry = census.get(source)
            if not entry:
                continue
            spec = SOURCES_BY_NAME[source]
            rows.append(
                {
                    "source": source,
                    "sensor": spec.sensor,
                    "description": spec.description,
                    "same_sensor_group": spec.same_sensor_group,
                    "n_files": entry.get("n_files"),
                    "n_hosts": entry.get("n_hosts"),
                    "archive_bytes": entry.get("uncompressed_bytes"),
                    "hosts": entry.get("hosts", []),
                    "fetched": False,
                    "note": spec.note,
                    "census_source": "archive central directory",
                }
            )
        unclassified = census.get("__unclassified__")
        if unclassified:
            rows.append(
                {
                    "source": "__unclassified__",
                    "sensor": "unknown",
                    "description": (
                        "log files under gather/*/logs/ that no SOURCES pattern "
                        "matches. A non-zero count here is a registry gap."
                    ),
                    "same_sensor_group": None,
                    "n_files": unclassified.get("n_files"),
                    "n_hosts": 0,
                    "archive_bytes": 0,
                    "hosts": [],
                    "fetched": False,
                    "note": "; ".join(unclassified.get("examples", [])[:5]),
                    "census_source": "archive central directory",
                }
            )
        return rows

    present: dict[str, list[LogFile]] = {}
    for log in discover_log_files(root):
        present.setdefault(log.source, []).append(log)
    for source in UNPARSED_SOURCES:
        logs = present.get(source)
        if not logs:
            continue
        spec = SOURCES_BY_NAME[source]
        rows.append(
            {
                "source": source,
                "sensor": spec.sensor,
                "description": spec.description,
                "same_sensor_group": spec.same_sensor_group,
                "n_files": len(logs),
                "n_hosts": len({log.host for log in logs}),
                "archive_bytes": sum(
                    log.path.stat().st_size for log in logs if log.path.is_file()
                ),
                "hosts": sorted({log.host for log in logs}),
                "fetched": True,
                "note": spec.note,
                "census_source": "files on disk (no manifest census)",
            }
        )
    return rows


# ---------------------------------------------------------------------------
# configs/data/ait.yaml
# ---------------------------------------------------------------------------

DEFAULT_CONFIG_PATH = Path("configs/data/ait.yaml")


def load_ait_config(
    path: Path | str = DEFAULT_CONFIG_PATH, **overrides: Any
) -> AITConfig:
    """Build an `AITConfig` from `configs/data/ait.yaml`, with overrides.

    The YAML holds more than the config object needs -- the Zenodo record, the
    held-out scenario, Part 27's window sweep -- so only the keys `AITConfig`
    declares are taken and the rest are left for the callers that want them
    (`load_ait_settings`). An unknown override is an error rather than a
    silently ignored typo.
    """
    settings = load_ait_settings(path)
    taken = {
        "scenario": settings.get("scenario", "russellmitchell"),
        "window_seconds": int(settings.get("window_seconds", DEFAULT_WINDOW_SECONDS)),
        "sample_mode": str(settings.get("sample_mode", "network")),
        "sources": tuple(settings.get("sources", PARSED_SOURCES)),
        "time_range": str(settings.get("time_range", "scenario")),
        "drop_empty_windows": bool(settings.get("drop_empty_windows", True)),
        "raw_root": str(settings.get("raw_root", DEFAULT_RAW_ROOT)),
    }
    unknown = sorted(set(overrides) - set(taken))
    if unknown:
        raise ValueError(f"unknown AITConfig override(s): {unknown}; known: {sorted(taken)}")
    taken.update(overrides)
    if "sources" in overrides:
        taken["sources"] = tuple(overrides["sources"])
    return AITConfig(**taken)


def load_ait_settings(path: Path | str = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """The whole YAML as a plain mapping, for keys outside `AITConfig`."""
    import yaml

    text = Path(path).read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} root must be a mapping, got {type(data)!r}")
    return data
