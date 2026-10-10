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
