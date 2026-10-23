"""Part 30 - tests for the LANL cyber1 loader.

These tests never touch `data/raw/`. Every stream test writes a tiny gzip file
into pytest's `tmp_path` and points the loader at that, so the real capture
stays read-only and the suite stays fast.
"""

from __future__ import annotations

import dataclasses
import gzip
import zlib

import numpy as np
import pytest
import torch
import yaml

from prismflow.data.lanl_cyber1 import (
    CLASS_ATTACK,
    CLASS_BENIGN,
    FEATURES,
    NOT_COUNTS,
    RECORD_WIDTH,
    SOURCE_COLUMN,
    GzipStreamError,
    LANLMultiViewDataset,
    TimestampInversion,
    _keep,
    _stream,
    build_lanl_samples,
    load_lanl_config,
    read_redteam,
    scan_view,
    subset,
    view_configs,
)

CONFIG = "configs/data/lanl.yaml"


# ---------------------------------------------------------------------------
# Config and grid arithmetic
# ---------------------------------------------------------------------------


def test_config_loads_and_matches_feature_constant():
    cfg = load_lanl_config(CONFIG)
    assert cfg.views == ("auth", "proc", "flows", "dns")
    for view in cfg.views:
        assert tuple(yaml.safe_load(open(CONFIG).read())["features"][view]) == FEATURES[view]


def test_window_and_day_arithmetic_are_one_indexed_from_t_equals_one():
    cfg = load_lanl_config(CONFIG)
    # Day 1 is t in [1, 86400]; window 0 is t in [1, 300].
    assert cfg.day_of(1) == 1
    assert cfg.day_of(86400) == 1
    assert cfg.day_of(86401) == 2
    assert cfg.window_of(1) == 0
    assert cfg.window_of(300) == 0
    assert cfg.window_of(301) == 1


def test_feature_dim_is_at_least_the_widest_view():
    cfg = load_lanl_config(CONFIG)
    assert cfg.feature_dim >= max(len(FEATURES[v]) for v in cfg.views)


def test_every_split_boundary_lands_on_a_window_edge():
    """A window must belong to exactly one split, so no window may straddle."""
    cfg = load_lanl_config(CONFIG)
    for name, bounds in cfg.split.items():
        assert (bounds["start"] - 1) % cfg.window_seconds == 0, name
        assert bounds["end"] % cfg.window_seconds == 0, name


def test_splits_are_contiguous_disjoint_and_gap_separates_them():
    cfg = load_lanl_config(CONFIG)
    train, gap, test = cfg.split["train"], cfg.split["gap"], cfg.split["test"]
    assert train["end"] < gap["start"] <= gap["end"] < test["start"]
    assert gap["start"] == train["end"] + 1
    assert test["start"] == gap["end"] + 1
    assert cfg.read_until == test["end"]


def test_block_of_window_assigns_each_window_to_one_block():
    cfg = load_lanl_config(CONFIG)
    assert cfg.block_of_window(0) == "train"
    assert cfg.block_of_window(cfg.split["train"]["end"] // cfg.window_seconds - 1) == "train"
    assert cfg.block_of_window(cfg.split["gap"]["start"] // cfg.window_seconds) == "gap"
    assert cfg.block_of_window((cfg.split["test"]["start"] - 1) // cfg.window_seconds) == "test"
    assert cfg.block_of_window(cfg.read_until // cfg.window_seconds) == "unused"


def test_no_window_index_maps_to_two_blocks():
    cfg = load_lanl_config(CONFIG)
    last = cfg.read_until // cfg.window_seconds
    blocks = [cfg.block_of_window(w) for w in range(last)]
    # Exactly one contiguous run per block name, i.e. no interleaving.
    changes = sum(1 for a, b in zip(blocks, blocks[1:]) if a != b)
    assert changes == 2
    assert set(blocks) == {"train", "gap", "test"}


# ---------------------------------------------------------------------------
# The control rule
# ---------------------------------------------------------------------------


def test_every_attack_cell_is_kept_regardless_of_the_hash():
    cells = {(b"C1", 7), (b"C2", 9)}
    for cell in cells:
        assert _keep(cell, cells, divisor=10**9) is True


def test_control_rule_is_deterministic_and_label_free():
    cell = (b"C1234", 99)
    first = _keep(cell, set(), 512)
    assert first == _keep(cell, set(), 512)
    # The rule reads only the cell identity: the attack set changes nothing for
    # a cell that is not in it.
    assert first == _keep(cell, {(b"OTHER", 1)}, 512)
    assert first == (zlib.crc32(b"C1234|99") % 512 == 0)


def test_control_rule_keeps_roughly_one_in_divisor():
    kept = sum(_keep((f"C{i}".encode(), i % 4608), set(), 512) for i in range(200_000))
    # Binomial(200000, 1/512) has mean ~390 and sd ~20; this is a very wide band.
    assert 250 < kept < 550


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def _write(tmp_path, name, lines):
    path = tmp_path / name
    with gzip.open(path, "wt") as fh:
        fh.write("".join(line + "\n" for line in lines))
    return path


def test_stream_reads_a_prefix_and_stops_at_read_until(tmp_path):
    path = _write(tmp_path, "dns.txt.gz", ["1,C1,C2", "400,C1,C3", "5000,C9,C9"])
    seen = [int(f[0]) for f, _ in _stream(path, "dns", 3, until=1000)]
    assert seen == [1, 400]


def test_stream_raises_on_timestamp_inversion_naming_the_line(tmp_path):
    path = _write(tmp_path, "dns.txt.gz", ["1,C1,C2", "900,C1,C3", "800,C1,C4"])
    with pytest.raises(TimestampInversion) as exc:
        list(_stream(path, "dns", 3, until=None))
    message = str(exc.value)
    assert "line 3" in message and "800" in message and "900" in message


def test_stream_rejects_a_wrong_width_record(tmp_path):
    path = _write(tmp_path, "dns.txt.gz", ["1,C1,C2", "2,C1"])
    with pytest.raises(ValueError, match="expected 3 fields"):
        list(_stream(path, "dns", 3, until=None))


def test_stream_raises_on_a_corrupt_archive(tmp_path):
    path = tmp_path / "dns.txt.gz"
    path.write_bytes(b"\x1f\x8b\x08\x00 not actually gzip payload")
    with pytest.raises((GzipStreamError, ValueError, EOFError, OSError)):
        list(_stream(path, "dns", 3, until=None))


def test_source_columns_and_widths_match_the_documented_schema():
    assert RECORD_WIDTH == {"auth": 9, "proc": 5, "flows": 9, "dns": 3}
    # auth: time,su,du,src,dst,...  proc: time,user,computer,...
    # flows: time,dur,src,sport,...  dns: time,src,resolved
    assert SOURCE_COLUMN == {"auth": 3, "proc": 2, "flows": 2, "dns": 1}


# ---------------------------------------------------------------------------
# Aggregation and features
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny(tmp_path):
    """A four-view capture small enough to reason about by hand.

    Window 0 is t in [1, 300]; C1 is the attributed computer throughout, and
    the control divisor is 1 so every cell is kept.
    """
    _write(tmp_path, "auth.txt.gz", [
        "1,U1@D,U1@D,C1,C2,NTLM,Network,LogOn,Success",
        "2,U2@D,U1@D,C1,C3,Kerberos,Network,LogOn,Fail",
        "3,U1@D,U1@D,C1,C2,NTLM,Network,LogOff,Success",
    ])
    _write(tmp_path, "proc.txt.gz", [
        "1,U1@D,C1,P1,Start",
        "2,U1@D,C1,P2,Start",
        "3,U1@D,C1,P1,End",
    ])
    _write(tmp_path, "flows.txt.gz", [
        "1,10,C1,80,C2,N1,6,5,500",
        "2,20,C1,80,C3,N2,6,7,700",
    ])
    _write(tmp_path, "dns.txt.gz", ["1,C1,C9", "2,C1,C9", "3,C1,C8"])
    _write(tmp_path, "redteam.txt.gz", ["2,U2@D,C1,C3"])
    base = load_lanl_config(CONFIG)
    return dataclasses.replace(
        base, raw_root=tmp_path, read_until=300, control_divisor=1
    )


def test_auth_aggregates_counts_distincts_and_failure_rate(tiny):
    rt = read_redteam(tiny)
    scan = scan_view("auth", tiny, rt["cells"])
    vector = scan["kept"][(b"C1", 0)]
    names = FEATURES["auth"]
    got = dict(zip(names, vector))
    # Counts are log1p transformed; failure_rate is not.
    assert got["n_events"] == pytest.approx(np.log1p(3))
    assert got["n_distinct_src_user"] == pytest.approx(np.log1p(2))   # U1, U2
    assert got["n_distinct_dst_user"] == pytest.approx(np.log1p(1))   # U1
    assert got["n_distinct_dst_computer"] == pytest.approx(np.log1p(2))  # C2, C3
    assert got["n_success"] == pytest.approx(np.log1p(2))
    assert got["n_failure"] == pytest.approx(np.log1p(1))
    assert got["n_distinct_auth_type"] == pytest.approx(np.log1p(2))
    assert got["failure_rate"] == pytest.approx(1 / 3)


def test_failure_rate_is_the_only_untransformed_feature(tiny):
    assert NOT_COUNTS == {"failure_rate"}
    rt = read_redteam(tiny)
    vector = scan_view("auth", tiny, rt["cells"])["kept"][(b"C1", 0)]
    index = FEATURES["auth"].index("failure_rate")
    assert 0.0 <= vector[index] <= 1.0


def test_proc_flows_and_dns_aggregates(tiny):
    rt = read_redteam(tiny)
    proc = dict(zip(FEATURES["proc"], scan_view("proc", tiny, rt["cells"])["kept"][(b"C1", 0)]))
    assert proc["n_events"] == pytest.approx(np.log1p(3))
    assert proc["n_distinct_process"] == pytest.approx(np.log1p(2))
    assert proc["n_start"] == pytest.approx(np.log1p(2))
    assert proc["n_end"] == pytest.approx(np.log1p(1))

    flows = dict(zip(FEATURES["flows"], scan_view("flows", tiny, rt["cells"])["kept"][(b"C1", 0)]))
    assert flows["n_events"] == pytest.approx(np.log1p(2))
    assert flows["sum_packets"] == pytest.approx(np.log1p(12))
    assert flows["sum_bytes"] == pytest.approx(np.log1p(1200))
    assert flows["sum_duration"] == pytest.approx(np.log1p(30))
    assert flows["n_distinct_dst_port"] == pytest.approx(np.log1p(2))

    dns = dict(zip(FEATURES["dns"], scan_view("dns", tiny, rt["cells"])["kept"][(b"C1", 0)]))
    assert dns["n_events"] == pytest.approx(np.log1p(3))
    assert dns["n_distinct_resolved"] == pytest.approx(np.log1p(2))  # C9, C8


def test_population_cells_counts_occupied_cells_not_records(tiny):
    rt = read_redteam(tiny)
    scan = scan_view("auth", tiny, rt["cells"])
    assert scan["records"] == 3
    assert scan["population_cells"] == 1   # one (computer, window) cell


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def test_redteam_marks_both_source_and_destination_computers(tiny):
    rt = read_redteam(tiny)
    assert (b"C1", 0) in rt["cells"]
    assert (b"C3", 0) in rt["cells"]
    assert rt["windows"] == {0}


def test_duplicate_redteam_rows_are_kept_and_collapse(tmp_path):
    _write(tmp_path, "redteam.txt.gz", ["5,U1@D,C1,C2", "5,U1@D,C1,C2"])
    cfg = dataclasses.replace(load_lanl_config(CONFIG), raw_root=tmp_path)
    rt = read_redteam(cfg)
    assert rt["rows"] == 2              # both rows read, nothing dropped
    assert rt["cells"] == {(b"C1", 0), (b"C2", 0)}   # and a window is binary


def test_label_is_attack_only_for_the_named_computers(tiny):
    rt = read_redteam(tiny)
    scans = {v: scan_view(v, tiny, rt["cells"]) for v in tiny.views}
    samples = build_lanl_samples(tiny, scans, rt)
    by_computer = dict(zip(samples.computer, samples.labels))
    assert by_computer["C1"] == CLASS_ATTACK
    # C2 appears only as an auth/flows DESTINATION, so it is never an
    # attributed computer and never becomes a sample of its own here.
    assert set(samples.computer) == {"C1"}


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def test_build_places_each_view_in_its_own_slot_and_pads_with_zeros(tiny):
    rt = read_redteam(tiny)
    scans = {v: scan_view(v, tiny, rt["cells"]) for v in tiny.views}
    samples = build_lanl_samples(tiny, scans, rt)
    assert samples.views.shape == (1, 4, tiny.feature_dim)
    assert samples.view_mask.shape == (1, 4)
    assert samples.view_mask.all()
    for vi, view in enumerate(tiny.views):
        native = len(FEATURES[view])
        assert np.all(samples.views[0, vi, native:] == 0.0), view


def test_gap_and_out_of_range_windows_never_become_samples(tiny):
    """A cell in the gap is dropped at assembly, not filtered downstream."""
    rt = read_redteam(tiny)
    scans = {v: scan_view(v, tiny, rt["cells"]) for v in tiny.views}
    gap_window = tiny.split["gap"]["start"] // tiny.window_seconds
    # Beyond the TEST block's end, i.e. days 17-30. Derived from the split, not
    # from `tiny.read_until`, which this fixture shortens to one window.
    unused_window = tiny.split["test"]["end"] // tiny.window_seconds + 10
    scans["dns"]["kept"][(b"CGAP", gap_window)] = np.zeros(2, dtype=np.float32)
    scans["dns"]["kept"][(b"CFAR", unused_window)] = np.zeros(2, dtype=np.float32)
    samples = build_lanl_samples(tiny, scans, rt)
    assert "CGAP" not in samples.computer
    assert "CFAR" not in samples.computer
    assert set(samples.block) <= {"train", "test"}


def test_window_start_is_the_first_second_of_the_window(tiny):
    rt = read_redteam(tiny)
    scans = {v: scan_view(v, tiny, rt["cells"]) for v in tiny.views}
    samples = build_lanl_samples(tiny, scans, rt)
    assert samples.window_start[0] == 1.0


# ---------------------------------------------------------------------------
# Dataset interface parity with the AIT loader
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset(tiny):
    rt = read_redteam(tiny)
    scans = {v: scan_view(v, tiny, rt["cells"]) for v in tiny.views}
    return LANLMultiViewDataset(build_lanl_samples(tiny, scans, rt))


def test_dataset_exposes_the_same_attributes_as_the_ait_loader(dataset):
    from prismflow.data.ait_lds import AITMultiViewDataset

    required = [
        "views", "view_mask", "labels", "label_known", "window_start",
        "host_id", "rho_matrix", "config", "samples", "view_dims",
        "n_classes", "sources",
    ]
    for name in required:
        assert hasattr(dataset, name), name
        assert hasattr(AITMultiViewDataset, "__init__")
    assert dataset.n_classes == 2
    assert dataset.rho_matrix is None      # must fail loudly, never estimated
    assert dataset.view_dims == tuple(len(FEATURES[s]) for s in dataset.sources)
    assert dataset.views.dtype == torch.float32
    assert dataset.view_mask.dtype == torch.bool
    assert dataset.labels.dtype == torch.int64


def test_get_sample_and_get_batch_shapes(dataset):
    sample = dataset.get_sample(0)
    assert sample.views.shape == (dataset.n_views, dataset.views.shape[2])
    assert sample.metadata["dataset"] == "lanl_cyber1"
    assert sample.metadata["sample_mode"] == "computer_window"
    assert sample.metadata["block"] in ("train", "test")
    batch = dataset.get_batch([0])
    assert batch.views.shape == (1, dataset.n_views, dataset.views.shape[2])
    assert batch.labels.shape == (1,)


def test_label_known_is_all_true_because_this_dataset_has_no_coverage_mask(dataset):
    assert bool(dataset.label_known.all())


# ---------------------------------------------------------------------------
# The WITHOUT-auth configuration
# ---------------------------------------------------------------------------


def test_subset_rebuilds_the_tensor_without_the_dropped_view(dataset):
    without = subset(dataset, ("proc", "flows", "dns"))
    assert without.sources == ("proc", "flows", "dns")
    assert without.n_views == 3
    assert without.view_dims == (5, 7, 2)
    # The kept views keep their values, in the new order.
    original = dataset.views.numpy()
    assert np.allclose(without.views.numpy()[:, 0, :], original[:, 1, :])
    assert without.labels.tolist() == dataset.labels.tolist()


def test_view_configs_declare_each_views_native_width(dataset):
    configs = view_configs(dataset.sources, hidden_dims=[64, 64], feature_dim=32)
    assert [c.input_dim for c in configs] == [8, 5, 7, 2]
    assert all(c.feature_dim == 32 for c in configs)
    # Dropping auth must drop its encoder, not merely mask it.
    assert [c.input_dim for c in view_configs(("proc", "flows", "dns"))] == [5, 7, 2]


def test_class_codes_match_part_28():
    from experiments.ait_clean.run_ait_clean import CLASS_ATTACK as A
    from experiments.ait_clean.run_ait_clean import CLASS_BENIGN as B

    assert (CLASS_BENIGN, CLASS_ATTACK) == (B, A)
