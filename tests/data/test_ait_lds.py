"""Part 26: the AIT loader's load-bearing claims, checked rather than asserted.

Most of these run on a FIXTURE scenario built on disk by `make_scenario`, not
on the 86.7 MB download: `data/raw/` is gitignored, so a fresh clone has
nothing until `scripts/fetch_ait.py` has run. The fixture is written here, so
every labelled line in it is hand-checked by construction -- the test states
which line of which file is an attack and the loader has to agree.

Two tests additionally hand-check the REAL AIT labels, and skip with a reason
when the scenario has not been fetched.

The fixture deliberately uses host names that look nothing like
russellmitchell's (`web01`, `fw01`, and `srv-a`/`srv-b` in the second
scenario), because the point of the view registry is that views are keyed off
log-source TYPE and never off host name. A test that reused the real host
names could not tell the difference.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import torch

from prismflow.data.ait_lds import (
    FEATURE_DIM,
    FEATURES,
    LABEL_UNKNOWN,
    PARSED_SOURCES,
    SOURCES,
    SOURCES_BY_NAME,
    UNPARSED_SOURCES,
    AITConfig,
    AITMultiViewDataset,
    build_samples,
    classify_log_path,
    discover_log_files,
    found_not_used,
    host_of,
    load_ait_config,
    load_labels,
    load_processed,
    rewindow,
    scan_scenario,
    scenario_time_range,
    sensor_groups,
    view_configs,
    write_processed,
)
from prismflow.data.dataset import Batch, Sample

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from fetch_ait import (  # noqa: E402
    ZipMember,
    archive_census,
    parse_central_directory,
    safe_extract_path,
)

# 2022-01-21T00:00:00Z, the russellmitchell scenario start, so the fixture
# sits on the same clock as the real data.
T0 = datetime(2022, 1, 21, tzinfo=timezone.utc)
EPOCH0 = T0.timestamp()
WINDOW = 300

REAL_SCENARIO_ROOT = Path("data/raw/ait/russellmitchell")


# ---------------------------------------------------------------------------
# Fixture scenario
# ---------------------------------------------------------------------------


def _syslog_stamp(offset: int) -> str:
    moment = datetime.fromtimestamp(EPOCH0 + offset, tz=timezone.utc)
    return moment.strftime("%b %d %H:%M:%S")


def _apache_stamp(offset: int) -> str:
    moment = datetime.fromtimestamp(EPOCH0 + offset, tz=timezone.utc)
    return moment.strftime("%d/%b/%Y:%H:%M:%S +0000")


def _iso_stamp(offset: int) -> str:
    moment = datetime.fromtimestamp(EPOCH0 + offset, tz=timezone.utc)
    return moment.strftime("%Y-%m-%dT%H:%M:%S.000000+0000")


def make_scenario(
    root: Path,
    name: str,
    web_host: str = "web01",
    fw_host: str = "fw01",
    attack_label: str = "fixture_exploit",
) -> Path:
    """Write a small but structurally faithful scenario under `root/name`.

    The layout mirrors the archive: `dataset.yaml`, `gather/<host>/logs/...`
    and a `labels/` tree whose paths shadow the gather paths. What each window
    contains is stated in `EXPECTED` below and is what the tests check.

    Window 0 (offset 0-299):   apache, auth, dns, audit, suricata
    Window 1 (offset 300-599): apache (the ATTACK line), suricata
    Window 2 (offset 600-899): auth only -- and auth is NOT label-covered here,
                               so this window must come out label -1, not 0.
    """
    scenario = root / name
    gather = scenario / "gather"
    (scenario).mkdir(parents=True, exist_ok=True)

    (scenario / "dataset.yaml").write_text(
        "name: processed_{}_scenario\n"
        "start: '2022-01-21T00:00:00'\n"
        "end: '2022-01-21T00:15:00'\n".format(name)
    )

    # --- apache: label-covered, one attack line in window 1 --------------
    apache_dir = gather / web_host / "logs" / "apache2"
    apache_dir.mkdir(parents=True, exist_ok=True)
    apache_lines = [
        f'10.0.0.5 - - [{_apache_stamp(10)}] "GET /index.html HTTP/1.1" 200 1024 "-" "Mozilla/5.0"',
        f'10.0.0.6 - - [{_apache_stamp(20)}] "GET /about HTTP/1.1" 404 199 "-" "Mozilla/5.0"',
        f'10.0.0.5 - - [{_apache_stamp(310)}] "POST /wp-admin/x.php HTTP/1.1" 500 66 "-" "sqlmap/1.5"',
        f'10.0.0.7 - - [{_apache_stamp(320)}] "GET /index.html HTTP/1.1" 200 1024 "-" "Mozilla/5.0"',
    ]
    apache_name = f"vhost.{name}.test-access.log.2"
    (apache_dir / apache_name).write_text("\n".join(apache_lines) + "\n")

    # Line 3 is the attack. The label file names it by 1-based line number,
    # which is the official format.
    label_dir = scenario / "labels" / web_host / "logs" / "apache2"
    label_dir.mkdir(parents=True, exist_ok=True)
    (label_dir / apache_name).write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"line": 1, "labels": [], "rules": {}},
                {"line": 2, "labels": [], "rules": {}},
                {"line": 3, "labels": [attack_label], "rules": {attack_label: ["r1"]}},
                {"line": 4, "labels": [], "rules": {}},
            ]
        )
        + "\n"
    )

    # --- auth: NOT label-covered, present in windows 0 and 2 -------------
    auth_dir = gather / web_host / "logs"
    auth_dir.mkdir(parents=True, exist_ok=True)
    (auth_dir / "auth.log").write_text(
        f"{_syslog_stamp(30)} {web_host} CRON[101]: pam_unix(cron:session): "
        "session opened for user root by (uid=0)\n"
        f"{_syslog_stamp(40)} {web_host} sshd[202]: Failed password for invalid user "
        "admin from 10.0.0.9 port 22 ssh2\n"
        f"{_syslog_stamp(650)} {web_host} CRON[303]: pam_unix(cron:session): "
        "session closed for user root\n"
    )

    # --- dns: dnsmasq, no hostname field, window 0 only ------------------
    dns_dir = gather / fw_host / "logs"
    dns_dir.mkdir(parents=True, exist_ok=True)
    (dns_dir / "dnsmasq.log").write_text(
        f"{_syslog_stamp(50)} dnsmasq[3468]: query[A] short.example.com from 10.0.0.5\n"
        f"{_syslog_stamp(51)} dnsmasq[3468]: forwarded short.example.com to 192.168.1.1\n"
        f"{_syslog_stamp(52)} dnsmasq[3468]: reply short.example.com is 93.184.216.34\n"
        f"{_syslog_stamp(60)} dnsmasq[3468]: query[TXT] "
        "aVeryLongExfiltrationLabel.chunk2.secret.example.net from 10.0.0.5\n"
    )

    # --- audit: window 0 only --------------------------------------------
    audit_dir = gather / fw_host / "logs" / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "audit.log").write_text(
        f"type=USER_ACCT msg=audit({EPOCH0 + 70:.3f}:149): pid=17710 uid=0 auid=4294967295 "
        "ses=4294967295 msg='op=PAM:accounting acct=\"root\" exe=\"/usr/sbin/cron\" "
        "hostname=? addr=? terminal=cron res=success'\n"
        f"type=SYSCALL msg=audit({EPOCH0 + 80:.3f}:150): arch=c000003e syscall=59 "
        'success=no exit=-2 comm="bash" exe="/bin/bash" uid=1000\n'
    )

    # --- suricata: windows 0 and 1, no label file anywhere ---------------
    suricata_dir = gather / fw_host / "logs" / "suricata"
    suricata_dir.mkdir(parents=True, exist_ok=True)
    events = [
        {
            "timestamp": _iso_stamp(90),
            "event_type": "flow",
            "src_ip": "10.0.0.5",
            "dest_ip": "93.184.216.34",
            "dest_port": 443,
            "flow": {"bytes_toserver": 500, "bytes_toclient": 1500},
        },
        {
            "timestamp": _iso_stamp(95),
            "event_type": "http",
            "src_ip": "10.0.0.5",
            "dest_ip": "10.0.0.1",
            "dest_port": 80,
        },
        {
            "timestamp": _iso_stamp(330),
            "event_type": "alert",
            "src_ip": "10.0.0.5",
            "dest_ip": "10.0.0.1",
            "dest_port": 80,
            "alert": {"signature": "ET WEB_SERVER SQL Injection"},
        },
    ]
    (suricata_dir / "eve.json").write_text(
        "\n".join(json.dumps(event) for event in events) + "\n"
    )

    return scenario


#: What `make_scenario` puts in each window, per source. The loader has to
#: agree with this, and it is the ground truth for the mask test.
EXPECTED_PRESENCE = {
    0: {"apache", "auth", "dns", "audit", "suricata"},
    1: {"apache", "suricata"},
    2: {"auth"},
}
#: Window 1 holds the attack line; window 0 is covered-and-clean; window 2 has
#: no label-covered source present at all.
EXPECTED_LABELS = {0: 0, 1: 1, 2: LABEL_UNKNOWN}


def _row_of_window(samples, window_index: int) -> int:
    """The sample row holding a given window.

    Deliberately exact rather than `np.isclose`: window starts are epoch
    seconds around 1.64e9, and `isclose`'s default relative tolerance of 1e-5
    is about 16000 seconds there, so it matches every window in this fixture.
    """
    offsets = (samples.window_start - EPOCH0) / WINDOW
    rows = np.flatnonzero(np.rint(offsets).astype(int) == window_index)
    assert len(rows) == 1, f"window {window_index} maps to rows {rows.tolist()}"
    return int(rows[0])


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    make_scenario(tmp_path, "alpha")
    return tmp_path


@pytest.fixture
def fixture_config(fixture_root: Path) -> AITConfig:
    return AITConfig(
        scenario="alpha", window_seconds=WINDOW, raw_root=str(fixture_root)
    )


# ---------------------------------------------------------------------------
# Registry: views are keyed off source type, not host
# ---------------------------------------------------------------------------


def test_every_source_pattern_is_mutually_exclusive():
    """No path may classify as two sources, or the view order is ambiguous."""
    paths = [
        "gather/h/logs/apache2/site-access.log",
        "gather/h/logs/apache2/site-access.log.2",
        "gather/h/logs/apache2/site-error.log.2",
        "gather/h/logs/apache2/error.log",
        "gather/h/logs/auth.log",
        "gather/h/logs/auth.log.1",
        "gather/h/logs/dnsmasq.log",
        "gather/h/logs/audit/audit.log",
        "gather/h/logs/suricata/eve.json",
        "gather/h/logs/suricata/fast.log",
        "gather/h/logs/suricata/stats.log",
        "gather/h/logs/suricata/log.pcap.1642684610",
        "gather/h/logs/openvpn.log",
        "gather/h/logs/syslog.1",
        "gather/h/logs/kern.log",
        "gather/h/logs/mail.log",
        "gather/h/logs/logstash/node/2022-01-24-system.cpu.log",
    ]
    for path in paths:
        matched = [s.name for s in SOURCES if s.regex.search(path)]
        assert len(matched) == 1, f"{path} matched {matched}"


def test_apache_view_is_access_log_only():
    assert classify_log_path("gather/h/logs/apache2/x-access.log.2") == "apache"
    assert classify_log_path("gather/h/logs/apache2/x-error.log.2") == "apache_error"
    assert classify_log_path("gather/h/logs/apache2/error.log") == "apache_error"


def test_suricata_and_pcap_share_the_network_sensor_group():
    assert SOURCES_BY_NAME["suricata"].same_sensor_group == "network"
    assert SOURCES_BY_NAME["pcap"].same_sensor_group == "network"
    groups = sensor_groups(PARSED_SOURCES)
    assert groups["network"] == [PARSED_SOURCES.index("suricata")]


def test_classification_ignores_the_host_directory():
    """Same log, different host: same source. This is what lets santos load."""
    for host in ("intranet_server", "srv-a", "totally-different-name", "x"):
        assert classify_log_path(f"gather/{host}/logs/auth.log") == "auth"
        assert host_of(f"gather/{host}/logs/auth.log") == host


def test_feature_dim_is_the_widest_parsed_view():
    widths = [len(FEATURES[s]) for s in PARSED_SOURCES]
    assert FEATURE_DIM == max(widths)
    assert [c.input_dim for c in view_configs()] == widths


def test_parsed_and_unparsed_sources_partition_the_registry():
    assert set(PARSED_SOURCES) | set(UNPARSED_SOURCES) == {s.name for s in SOURCES}
    assert not set(PARSED_SOURCES) & set(UNPARSED_SOURCES)
    for source in PARSED_SOURCES:
        assert source in FEATURES


def test_view_list_is_identical_for_two_scenarios(tmp_path: Path):
    """The brief's cross-scenario check: different hosts, same views.

    `alpha` runs web01/fw01 and `beta` runs srv-a/srv-b. If any view were keyed
    off a host name, the two would not produce the same view list, and santos
    would need a code change.
    """
    make_scenario(tmp_path, "alpha", web_host="web01", fw_host="fw01")
    make_scenario(tmp_path, "beta", web_host="srv-a", fw_host="srv-b")

    built = {}
    for name in ("alpha", "beta"):
        config = AITConfig(
            scenario=name, window_seconds=WINDOW, raw_root=str(tmp_path)
        )
        built[name] = build_samples(scan_scenario(config))

    assert built["alpha"].sources == built["beta"].sources == PARSED_SOURCES
    assert built["alpha"].views.shape[1:] == built["beta"].views.shape[1:]
    assert built["alpha"].view_mask.shape[1:] == built["beta"].view_mask.shape[1:]
    # The two fixtures differ only in host names, so the tensors must match.
    assert np.array_equal(built["alpha"].views, built["beta"].views)
    assert np.array_equal(built["alpha"].labels, built["beta"].labels)
    assert built["alpha"].hosts != built["beta"].hosts


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_labels_align_with_hand_checked_lines(fixture_config: AITConfig):
    """Line 3 of the apache log is the attack, and nothing else is.

    Every line of the fixture is written in this file, so this is the hand
    check: the loader must put the attack in window 1 and leave windows 0 and
    2 unlabelled-as-attack.
    """
    scan = scan_scenario(fixture_config)
    samples = build_samples(scan, "network")

    assert len(samples) == 3
    for row in range(3):
        index = int(round((samples.window_start[row] - EPOCH0) / WINDOW))
        assert samples.labels[row] == EXPECTED_LABELS[index], (
            f"window {index} labelled {samples.labels[row]}, "
            f"expected {EXPECTED_LABELS[index]}"
        )

    attack_rows = np.flatnonzero(samples.labels == 1)
    assert attack_rows.tolist() == [1]
    assert samples.attack_steps[1] == ["fixture_exploit"]
    assert samples.attack_steps[0] == []
    assert samples.attack_steps[2] == []


def test_five_hand_checked_label_records_are_read_verbatim(tmp_path: Path):
    """Five specific (file, line) pairs, four of them attacks, one not."""
    scenario = make_scenario(tmp_path, "alpha")
    target = (
        scenario
        / "labels"
        / "web01"
        / "logs"
        / "apache2"
        / "vhost.alpha.test-access.log.2"
    )
    target.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"line": 1, "labels": ["step_a"], "rules": {}},
                {"line": 2, "labels": [], "rules": {}},
                {"line": 3, "labels": ["step_b", "step_c"], "rules": {}},
                {"line": 4, "labels": ["step_d"], "rules": {}},
            ]
        )
        + "\n"
    )
    labels, coverage = load_labels(scenario)
    key = "gather/web01/logs/apache2/vhost.alpha.test-access.log.2"

    assert labels[key] == {
        1: ["step_a"],
        3: ["step_b", "step_c"],
        4: ["step_d"],
    }
    assert 2 not in labels[key], "a line with an empty labels list is not an attack"
    assert coverage.covers("apache", "web01")


def test_unlabelled_source_never_counts_as_benign(fixture_config: AITConfig):
    """Window 2 has only `auth`, which has no label file: it must be unknown.

    This is the asymmetry the brief insists on. Treating window 2 as benign
    would invent a negative out of a source the labelling never examined.
    """
    scan = scan_scenario(fixture_config)
    samples = build_samples(scan, "network")

    assert samples.labels[2] == LABEL_UNKNOWN
    assert not samples.label_known[2]
    assert samples.label_known[0] and samples.label_known[1]
    # suricata and auth are parsed but carry no label file in the fixture.
    assert set(scan.coverage.unlabelled_sources) == {"auth", "dns", "audit", "suricata"}
    assert samples.statistics()["n_unlabelled_windows"] == 1
    assert samples.statistics()["n_benign_windows"] == 1
    assert samples.statistics()["n_attack_windows"] == 1


def test_label_coverage_is_per_file_not_per_source(tmp_path: Path):
    """AIT labels `access.log.2` but not `.1`. Coverage must follow the file."""
    scenario = make_scenario(tmp_path, "alpha")
    sibling = (
        scenario / "gather" / "web01" / "logs" / "apache2" / "vhost.alpha.test-access.log.1"
    )
    sibling.write_text(
        f'10.0.0.8 - - [{_apache_stamp(700)}] "GET /old HTTP/1.1" 200 10 "-" "curl/7"\n'
    )
    config = AITConfig(scenario="alpha", window_seconds=WINDOW, raw_root=str(tmp_path))
    scan = scan_scenario(config)

    covered = {f.relpath: f.label_covered for f in scan.files}
    assert covered["gather/web01/logs/apache2/vhost.alpha.test-access.log.2"] is True
    assert covered["gather/web01/logs/apache2/vhost.alpha.test-access.log.1"] is False

    # Window 2 now has apache events, but only from the UNCOVERED rotation, so
    # it still must not be scored benign.
    samples = build_samples(scan, "network")
    row = _row_of_window(samples, 2)
    assert samples.view_mask[row, PARSED_SOURCES.index("apache")]
    assert samples.labels[row] == LABEL_UNKNOWN
    assert not samples.label_known[row]


def test_incomplete_fetch_is_refused(tmp_path: Path):
    """A half-fetched scenario must not produce statistics that look fine.

    The failure mode this guards is specific: missing views would be missing
    because the file had not arrived, not because the sensor was quiet, and
    the resulting missing-view fractions would be entirely plausible.
    """
    scenario = make_scenario(tmp_path, "alpha")
    config = AITConfig(scenario="alpha", window_seconds=WINDOW, raw_root=str(tmp_path))
    (scenario / "manifest.json").write_text(
        json.dumps({"complete": False, "selected_entries": 91, "entries": [{"path": "x"}]})
    )

    with pytest.raises(RuntimeError, match="incomplete fetch"):
        scan_scenario(config)

    # The escape hatch works, and the scan records that it was used.
    scan = scan_scenario(config, allow_incomplete_fetch=True)
    assert scan.fetch_complete is False


def test_complete_fetch_is_recorded(tmp_path: Path):
    scenario = make_scenario(tmp_path, "alpha")
    config = AITConfig(scenario="alpha", window_seconds=WINDOW, raw_root=str(tmp_path))

    # No manifest at all: allowed (a fixture), but not claimed as complete.
    assert scan_scenario(config).fetch_complete is False

    (scenario / "manifest.json").write_text(
        json.dumps({"complete": True, "selected_entries": 1, "entries": [{"path": "x"}]})
    )
    scan = scan_scenario(config)
    assert scan.fetch_complete is True
    assert rewindow(scan, 600).fetch_complete is True

    destination = write_processed(
        build_samples(scan), scan, raw_manifest={"complete": True}, processed_root=tmp_path / "o"
    )
    _, manifest = load_processed(destination)
    assert manifest["raw_fetch"]["complete"] is True


def test_missing_labels_directory_is_an_error(tmp_path: Path):
    scenario = make_scenario(tmp_path, "alpha")
    for path in sorted((scenario / "labels").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    (scenario / "labels").rmdir()
    config = AITConfig(scenario="alpha", window_seconds=WINDOW, raw_root=str(tmp_path))
    with pytest.raises(FileNotFoundError, match="labels"):
        scan_scenario(config)


# ---------------------------------------------------------------------------
# No label leakage into features
# ---------------------------------------------------------------------------


def test_no_label_derived_column_in_any_view(tmp_path: Path):
    """Rewrite the labels, rebuild, and require the features to be identical.

    This is stronger than inspecting the feature list: if any feature read a
    label -- directly, or through a count that happened to include label state
    -- moving the attack to a different line would move a feature value. The
    labels themselves must change, or the test would pass trivially.
    """
    scenario = make_scenario(tmp_path, "alpha")
    config = AITConfig(scenario="alpha", window_seconds=WINDOW, raw_root=str(tmp_path))
    before = build_samples(scan_scenario(config), "network")

    label_file = (
        scenario / "labels" / "web01" / "logs" / "apache2" / "vhost.alpha.test-access.log.2"
    )
    # Move the attack from line 3 (window 1) to line 1 (window 0), and add a
    # second step, so both the label vector and the step names must change.
    label_file.write_text(
        "\n".join(
            json.dumps(record)
            for record in [
                {"line": 1, "labels": ["moved_step", "extra_step"], "rules": {}},
                {"line": 2, "labels": [], "rules": {}},
                {"line": 3, "labels": [], "rules": {}},
                {"line": 4, "labels": [], "rules": {}},
            ]
        )
        + "\n"
    )
    after = build_samples(scan_scenario(config), "network")

    assert before.views.tobytes() == after.views.tobytes(), (
        "a feature changed when only the labels changed: some view is "
        "label-derived"
    )
    assert before.view_mask.tobytes() == after.view_mask.tobytes()
    assert before.window_start.tobytes() == after.window_start.tobytes()
    # And the labels really did move, so the check above is not vacuous.
    assert not np.array_equal(before.labels, after.labels)
    assert after.labels[0] == 1 and after.labels[1] == 0
    assert after.attack_steps[0] == ["extra_step", "moved_step"]


def test_feature_names_contain_no_label_vocabulary():
    """A cheap structural guard beside the behavioural one above."""
    forbidden = ("label", "attack", "malicious", "ground_truth", "is_attack")
    for source, features in FEATURES.items():
        for feature in features:
            lowered = feature.name.lower()
            for word in forbidden:
                assert word not in lowered, f"{source}.{feature.name}"


# ---------------------------------------------------------------------------
# view_mask
# ---------------------------------------------------------------------------


def test_view_mask_is_false_exactly_where_a_source_has_no_events(
    fixture_config: AITConfig,
):
    samples = build_samples(scan_scenario(fixture_config), "network")

    for row in range(len(samples)):
        index = int(round((samples.window_start[row] - EPOCH0) / WINDOW))
        present = {
            source
            for column, source in enumerate(samples.sources)
            if samples.view_mask[row, column]
        }
        assert present == EXPECTED_PRESENCE[index], f"window {index}"


def test_masked_views_are_all_zero(fixture_config: AITConfig):
    samples = build_samples(scan_scenario(fixture_config), "network")
    absent = ~samples.view_mask
    assert np.all(samples.views[absent] == 0.0)


def test_padded_columns_are_always_zero(fixture_config: AITConfig):
    """Narrow views are right-padded; no encoder reads those columns."""
    samples = build_samples(scan_scenario(fixture_config), "network")
    for column, source in enumerate(samples.sources):
        width = len(FEATURES[source])
        assert np.all(samples.views[:, column, width:] == 0.0), source


# ---------------------------------------------------------------------------
# Both sample modes
# ---------------------------------------------------------------------------


def test_both_modes_build_from_one_scan(fixture_config: AITConfig):
    scan = scan_scenario(fixture_config)
    network = build_samples(scan, "network")
    host = build_samples(scan, "host")

    assert network.config.sample_mode == "network"
    assert host.config.sample_mode == "host"
    assert network.host_id is None
    assert host.host_id is not None
    # Two hosts, so host mode has at least as many samples as network mode.
    assert len(host) >= len(network)
    assert network.sources == host.sources


def test_host_mode_splits_views_by_host(fixture_config: AITConfig):
    """web01 never produces dns; fw01 never produces apache."""
    scan = scan_scenario(fixture_config)
    host = build_samples(scan, "host")

    apache = host.sources.index("apache")
    dns = host.sources.index("dns")
    for row in range(len(host)):
        name = host.hosts[int(host.host_id[row])]
        if name == "web01":
            assert not host.view_mask[row, dns]
        if name == "fw01":
            assert not host.view_mask[row, apache]


def test_network_mode_unions_distinct_values_across_hosts(tmp_path: Path):
    """Two hosts seeing the same client is one distinct client, not two.

    If network mode summed per-host distinct counts instead of unioning the
    value sets, the same client address observed on two hosts would read as
    two clients, and every distinct-value feature would be inflated by
    however many hosts happened to be logging.
    """
    scenario = tmp_path / "dup"
    (scenario).mkdir(parents=True)
    (scenario / "dataset.yaml").write_text(
        "name: x\nstart: '2022-01-21T00:00:00'\nend: '2022-01-21T00:05:00'\n"
    )
    (scenario / "labels").mkdir()
    for host in ("a", "b"):
        directory = scenario / "gather" / host / "logs" / "apache2"
        directory.mkdir(parents=True)
        # The SAME client and the SAME path on both hosts.
        (directory / "v-access.log").write_text(
            f'10.0.0.5 - - [{_apache_stamp(10)}] "GET /same HTTP/1.1" 200 100 "-" "UA"\n'
        )

    config = AITConfig(scenario="dup", window_seconds=WINDOW, raw_root=str(tmp_path))
    scan = scan_scenario(config)
    network = build_samples(scan, "network")

    column = network.sources.index("apache")
    names = [f.name for f in FEATURES["apache"]]
    clients = network.views[0, column, names.index("n_unique_client")]
    requests = network.views[0, column, names.index("n_requests")]

    assert clients == pytest.approx(np.log1p(1.0)), "distinct clients must be unioned"
    assert requests == pytest.approx(np.log1p(2.0)), "request counts must be summed"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_rewindow_equals_a_direct_scan(tmp_path: Path):
    """Re-binning a 60 s scan to 300 s must equal scanning at 300 s.

    `rewindow` exists so a window-length sweep costs one parse of the 2.3 GB
    of logs instead of one per length. That is only sound because every
    feature is merge-closed, and this is the check: the re-binned tensors have
    to be byte-identical to the direct scan's, not merely close. If a feature
    were ever added that averages instead of summing, this fails.
    """
    make_scenario(tmp_path, "alpha")
    fine = scan_scenario(
        AITConfig(scenario="alpha", window_seconds=60, raw_root=str(tmp_path))
    )
    direct = scan_scenario(
        AITConfig(scenario="alpha", window_seconds=300, raw_root=str(tmp_path))
    )
    rebinned = rewindow(fine, 300)

    for mode in ("network", "host"):
        want = build_samples(direct, mode)
        got = build_samples(rebinned, mode)
        assert got.views.tobytes() == want.views.tobytes(), mode
        assert got.view_mask.tobytes() == want.view_mask.tobytes(), mode
        assert got.labels.tobytes() == want.labels.tobytes(), mode
        assert got.label_known.tobytes() == want.label_known.tobytes(), mode
        assert got.window_start.tobytes() == want.window_start.tobytes(), mode
        assert got.attack_steps == want.attack_steps, mode


def test_rewindow_rejects_non_multiples(tmp_path: Path):
    make_scenario(tmp_path, "alpha")
    scan = scan_scenario(
        AITConfig(scenario="alpha", window_seconds=300, raw_root=str(tmp_path))
    )
    assert rewindow(scan, 300) is scan
    for bad in (301, 450, 299, 0, -300):
        with pytest.raises(ValueError, match="multiple"):
            rewindow(scan, bad)


def test_identical_config_is_byte_identical(tmp_path: Path):
    make_scenario(tmp_path, "alpha")
    config = AITConfig(scenario="alpha", window_seconds=WINDOW, raw_root=str(tmp_path))

    first = tmp_path / "out1"
    second = tmp_path / "out2"
    scan_a = scan_scenario(config)
    scan_b = scan_scenario(config)
    path_a = write_processed(build_samples(scan_a), scan_a, processed_root=first)
    path_b = write_processed(build_samples(scan_b), scan_b, processed_root=second)

    written = sorted(p.name for p in path_a.glob("*.npy"))
    assert written, "no arrays were written"
    for name in written:
        assert (path_a / name).read_bytes() == (path_b / name).read_bytes(), name
    assert (path_a / "manifest.json").read_text() == (path_b / "manifest.json").read_text()
    assert (path_a / "attack_steps.json").read_text() == (
        path_b / "attack_steps.json"
    ).read_text()


def test_processed_round_trips(tmp_path: Path):
    make_scenario(tmp_path, "alpha")
    config = AITConfig(scenario="alpha", window_seconds=WINDOW, raw_root=str(tmp_path))
    scan = scan_scenario(config)
    samples = build_samples(scan)
    destination = write_processed(samples, scan, processed_root=tmp_path / "out")

    arrays, manifest = load_processed(destination)
    assert np.array_equal(arrays["views"], samples.views)
    assert np.array_equal(arrays["view_mask"], samples.view_mask)
    assert np.array_equal(arrays["labels"], samples.labels)
    assert np.array_equal(arrays["label_known"], samples.label_known)
    assert manifest["config"]["sample_mode"] == "network"
    assert manifest["part"] == 26
    assert [v["source"] for v in manifest["views"]] == list(PARSED_SOURCES)


def test_output_name_separates_mode_and_window(tmp_path: Path):
    base = dict(scenario="alpha", raw_root=str(tmp_path))
    assert AITConfig(**base, window_seconds=300).output_name == "ait_alpha_network_300s"
    assert (
        AITConfig(**base, window_seconds=600, sample_mode="host").output_name
        == "ait_alpha_host_600s"
    )


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def test_bad_config_values_are_rejected(tmp_path: Path):
    with pytest.raises(ValueError, match="sample_mode"):
        AITConfig(scenario="x", sample_mode="per_host")
    with pytest.raises(ValueError, match="time_range"):
        AITConfig(scenario="x", time_range="whenever")
    with pytest.raises(ValueError, match="window_seconds"):
        AITConfig(scenario="x", window_seconds=0)
    with pytest.raises(ValueError, match="feature definition"):
        AITConfig(scenario="x", sources=("apache", "pcap"))


def test_repo_config_file_loads():
    config = load_ait_config()
    assert config.scenario == "russellmitchell"
    assert config.sample_mode == "network"
    assert config.sources == PARSED_SOURCES


# ---------------------------------------------------------------------------
# Untrusted archive entry names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "../x",
        "../../etc/passwd",
        "gather/../../x",
        "/abs",
        "//server/share/x",
        "C:/Windows/x",
        "c:x",
        "..\\x",
        "gather\\h\\logs\\auth.log",
        "gather/h/logs/\x00auth.log",
        "",
        "gather/h/logs/",
    ],
)
def test_malicious_entry_names_are_rejected(tmp_path: Path, name: str):
    with pytest.raises(ValueError):
        safe_extract_path(tmp_path, name)


def test_ordinary_entry_names_resolve_under_the_destination(tmp_path: Path):
    for name in (
        "dataset.yaml",
        "gather/h/logs/auth.log",
        "labels/h/logs/apache2/site-access.log.2",
    ):
        resolved = safe_extract_path(tmp_path, name)
        assert resolved.is_relative_to(tmp_path.resolve())
        assert resolved.name == name.rsplit("/", 1)[-1]


def test_archive_census_counts_unfetched_sources(tmp_path: Path):
    """The census must come from the archive, not from what was fetched.

    Part 26 fetches five source types and leaves the rest alone, so a census
    taken from disk would report every unparsed source as absent -- which is
    exactly the opposite of what the "found, not used yet" list is for. The
    archive here contains a pcap and a syslog that no Part 26 view parses, and
    both must still appear.
    """
    members = [
        ZipMember("gather/h1/logs/auth.log", 8, 1, 10, 100, 0),
        ZipMember("gather/h2/logs/auth.log", 8, 2, 10, 200, 1000),
        ZipMember("gather/h1/logs/suricata/log.pcap.1642684610", 8, 3, 10, 9999, 2000),
        ZipMember("gather/h1/logs/syslog.1", 8, 4, 10, 50, 3000),
        ZipMember("gather/h1/configs/dnsmasq.conf", 8, 5, 10, 7, 4000),
        ZipMember("labels/h1/logs/auth.log", 8, 6, 10, 9, 5000),
    ]
    census = archive_census(members)

    assert census["auth"]["n_files"] == 2
    assert census["auth"]["n_hosts"] == 2
    assert census["auth"]["uncompressed_bytes"] == 300
    assert census["pcap"]["n_files"] == 1
    assert census["syslog"]["n_files"] == 1
    # configs/ is not a log, and labels/ is not a gather entry.
    assert "__unclassified__" not in census
    assert set(census) == {"auth", "pcap", "syslog"}


def test_found_not_used_prefers_the_manifest_census(tmp_path: Path):
    """With a manifest census, unfetched sources are listed and marked so."""
    scenario = make_scenario(tmp_path, "alpha")
    (scenario / "manifest.json").write_text(
        json.dumps(
            {
                "archive_sources": {
                    "pcap": {
                        "n_files": 11,
                        "n_hosts": 7,
                        "hosts": ["a"],
                        "uncompressed_bytes": 6_961_900_000,
                    },
                    "openvpn": {
                        "n_files": 1,
                        "n_hosts": 1,
                        "hosts": ["vpn"],
                        "uncompressed_bytes": 540_000,
                    },
                }
            }
        )
    )
    rows = {row["source"]: row for row in found_not_used(scenario)}

    assert rows["pcap"]["n_files"] == 11
    assert rows["pcap"]["fetched"] is False
    assert rows["pcap"]["same_sensor_group"] == "network"
    assert rows["pcap"]["census_source"] == "archive central directory"
    assert rows["openvpn"]["n_files"] == 1
    # A parsed source must never show up in the "not used" list.
    assert not set(rows) & set(PARSED_SOURCES)


def test_found_not_used_falls_back_to_disk_without_a_census(tmp_path: Path):
    scenario = make_scenario(tmp_path, "alpha")
    extra = scenario / "gather" / "fw01" / "logs" / "openvpn.log"
    extra.write_text("Jan 21 00:00:01 fw01 openvpn[1]: x\n")
    rows = {row["source"]: row for row in found_not_used(scenario)}

    assert rows["openvpn"]["n_files"] == 1
    assert rows["openvpn"]["fetched"] is True
    assert "no manifest census" in rows["openvpn"]["census_source"]


def test_central_directory_parses_a_real_zip(tmp_path: Path):
    """The zip reader is checked against a zip that zipfile wrote.

    The remote path cannot be exercised offline, but the parsing can: names,
    sizes, CRC-32s and local-header offsets must all come back matching what
    `zipfile` recorded, for both stored and deflated members.
    """
    archive = tmp_path / "sample.zip"
    payloads = {
        "dataset.yaml": b"start: '2022-01-21T00:00:00'\n",
        "gather/h/logs/auth.log": b"Jan 21 00:00:01 h CRON[1]: x\n" * 500,
        "labels/h/logs/auth.log": b'{"line": 1, "labels": ["a"]}\n',
    }
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("dataset.yaml", payloads["dataset.yaml"], zipfile.ZIP_STORED)
        handle.writestr(
            "gather/h/logs/auth.log", payloads["gather/h/logs/auth.log"], zipfile.ZIP_DEFLATED
        )
        handle.writestr(
            "labels/h/logs/auth.log", payloads["labels/h/logs/auth.log"], zipfile.ZIP_DEFLATED
        )

    raw = archive.read_bytes()
    start = raw.rfind(b"PK\x05\x06")
    import struct

    cd_size, cd_offset = struct.unpack("<II", raw[start + 12 : start + 20])
    members = parse_central_directory(raw[cd_offset : cd_offset + cd_size])

    assert [m.name for m in members] == list(payloads)
    with zipfile.ZipFile(archive) as handle:
        for member in members:
            info = handle.getinfo(member.name)
            assert member.crc32 == info.CRC
            assert member.compressed_size == info.compress_size
            assert member.uncompressed_size == info.file_size
            assert member.header_offset == info.header_offset
            assert member.method == info.compress_type


# ---------------------------------------------------------------------------
# Dataset interface
# ---------------------------------------------------------------------------


def test_dataset_matches_the_existing_interface(fixture_config: AITConfig):
    """Same return types and shapes as the synthetic and handwritten loaders."""
    scan = scan_scenario(fixture_config)
    samples = build_samples(scan)
    dataset = AITMultiViewDataset(samples)

    assert len(dataset) == len(samples)
    assert dataset.n_views == len(PARSED_SOURCES)
    assert dataset.rho_matrix is None, "real views have no ground-truth dependence"

    sample = dataset.get_sample(0)
    assert isinstance(sample, Sample)
    assert sample.views.shape == (dataset.n_views, FEATURE_DIM)
    assert sample.view_mask.shape == (dataset.n_views,)
    assert sample.view_mask.dtype == torch.bool
    assert "label_known" in sample.metadata
    assert "attack_steps" in sample.metadata

    batch = dataset.get_batch([0, 1])
    assert isinstance(batch, Batch)
    assert batch.views.shape == (2, dataset.n_views, FEATURE_DIM)
    assert batch.view_mask.shape == (2, dataset.n_views)
    assert batch.labels.shape == (2,)
    assert batch.sample_ids.shape == (2,)


def test_batch_fields_match_the_synthetic_loader():
    """Compare field by field, so a new Batch field is picked up automatically."""
    import dataclasses

    fields = {f.name for f in dataclasses.fields(Batch)}
    assert fields == {"views", "view_mask", "labels", "sample_ids"}


# ---------------------------------------------------------------------------
# Real data, skipped when it has not been fetched
# ---------------------------------------------------------------------------

needs_real_data = pytest.mark.skipif(
    not (REAL_SCENARIO_ROOT / "labels").is_dir(),
    reason=(
        "data/raw/ is gitignored; run `python scripts/fetch_ait.py "
        "russellmitchell` to exercise the real-data checks"
    ),
)


@needs_real_data
def test_real_scenario_time_range_is_read_from_dataset_yaml():
    start, end = scenario_time_range(REAL_SCENARIO_ROOT)
    assert datetime.fromtimestamp(start, tz=timezone.utc).isoformat().startswith("2022-01-21")
    assert datetime.fromtimestamp(end, tz=timezone.utc).isoformat().startswith("2022-01-25")


@needs_real_data
def test_real_label_files_parse_and_cover_known_sources():
    labels, coverage = load_labels(REAL_SCENARIO_ROOT)
    assert labels, "no label files were read"

    # Every USABLE label file must map to a gather path on disk, or its line
    # numbers refer to a file nothing can read.
    for gather_relpath in labels:
        assert (REAL_SCENARIO_ROOT / gather_relpath).is_file(), gather_relpath

    # AIT labels six source types; Part 26 parses five. The label files for the
    # sources it does not parse are present and their logs were not fetched,
    # which is expected and must be recorded rather than dropped.
    unfetched = set(coverage.files_without_log.values())
    assert unfetched, "every label file had a log, so this scenario changed"
    assert not unfetched & set(PARSED_SOURCES), (
        f"a PARSED source has a label file with no log on disk: {unfetched}"
    )

    # Suricata is parsed as a view but is not labelled anywhere in this
    # dataset: rules/ carries no suricata rules and there is no label file.
    assert coverage.unlabelled_sources == ("suricata",)
    assert "suricata" not in coverage.hosts_by_source
    assert not (REAL_SCENARIO_ROOT / "rules" / "suricata.yaml").exists()


@needs_real_data
def test_real_labelled_lines_are_attack_lines_in_their_files():
    """Hand-check five real labelled records against the raw line they name."""
    labels, _ = load_labels(REAL_SCENARIO_ROOT)
    checked = 0
    for gather_relpath, attack_lines in sorted(labels.items()):
        if not attack_lines:
            continue
        raw = (REAL_SCENARIO_ROOT / gather_relpath).read_text(
            encoding="utf-8", errors="replace"
        ).splitlines()
        for line_no in sorted(attack_lines)[:5]:
            assert 1 <= line_no <= len(raw), (
                f"{gather_relpath}: label names line {line_no} but the file has "
                f"{len(raw)} lines"
            )
            assert raw[line_no - 1].strip(), f"{gather_relpath}:{line_no} is blank"
            assert attack_lines[line_no], f"{gather_relpath}:{line_no} has no step name"
            checked += 1
        if checked >= 5:
            break
    assert checked >= 5, f"only {checked} labelled lines were available to check"


@needs_real_data
def test_real_discovery_classifies_every_gather_log_file():
    """Nothing fetched may be unclassifiable, or the inventory has a hole."""
    files = discover_log_files(REAL_SCENARIO_ROOT)
    assert files
    assert {f.source for f in files} <= {s.name for s in SOURCES}
    assert set(PARSED_SOURCES) <= {f.source for f in files}
