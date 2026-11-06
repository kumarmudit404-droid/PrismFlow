"""Part 38 step 2: the sampler audited end to end on SYNTHETIC capture files.

WHY THIS EXISTS, against the tests already in test_lanl_v2_sampler.py
--------------------------------------------------------------------
Those tests pin the selection PREDICATES as pure functions. They cannot catch
a defect in `select_cells` itself -- the streaming pass that decides which
cells the predicate is even asked about, caches the decision per window, and
resets that cache when the window advances. On the real capture that pass can
only be checked against Part 30's committed aggregates, which agree but are
sums: a compensating pair of errors would survive them.

Here the ground truth is constructed, so the whole pass is checkable cell by
cell. Four synthetic `*.txt.gz` files are written in the real wire format
(comma separated, exact field counts, non-decreasing timestamps) under a
throwaway `raw_root`, and `select_cells` is compared against a brute-force
re-derivation of the pre-registered rule over the same occupancy.

NOTE ON "10 VIEWS". The brief asked for a 10-view toy. The LANL loader's view
set is fixed at four by `FEATURES`, `SOURCE_COLUMN` and `RECORD_WIDTH` in the
frozen `prismflow/data/lanl_cyber1.py`, so a 10-view table is not
constructible through it without editing a frozen module. This builds the
four real views over **10 synthetic hosts** instead, which is what the three
listed assertions actually need.
"""

from __future__ import annotations

import dataclasses
import gzip
import zlib

import pytest

from prismflow.data.lanl_cyber1 import load_lanl_config
from experiments.lanl_v2 import sampler as sp

WINDOW = 300
DAY = 86400
READ_UNTIL = 2 * DAY                     # windows 0-575
DIVISOR = 4                              # small, so the hash path is exercised
HOSTS = [f"C{i:02d}".encode() for i in range(1, 11)]

#: C10 is occupied in `auth` and NOWHERE else. Under L24's closing condition
#: it must never enter the sample, whatever auth says.
AUTH_ONLY_HOST = HOSTS[9]

#: A benign cell of an ATTACK host that is auth-only. Rule (a) must beat rule
#: (b): being an attack host does not buy a cell that no non-auth view saw.
AUTH_ONLY_CELL_OF_ATTACK_HOST = (HOSTS[0], 200)


def _attack_cells():
    """50 attack cells inside the read range, plus 7 outside it.

    Source and destination are the same computer in every redteam row, so a
    row contributes exactly one cell and the count is exact -- `read_redteam`
    adds both columns.
    """
    inside = ([(HOSTS[0], 3 + 5 * i) for i in range(20)]       # C01, 20 cells
              + [(HOSTS[1], 300 + 7 * i) for i in range(20)]   # C02, 20 cells
              + [(HOSTS[2], 40 + 11 * i) for i in range(10)])  # C03, 10 cells
    outside = ([(HOSTS[7], 900 + i) for i in range(4)]         # C08, day 4
               + [(HOSTS[8], 920 + i) for i in range(3)])      # C09, day 4
    assert len(inside) == 50 and len(outside) == 7
    assert all(w * WINDOW + 1 <= READ_UNTIL for _, w in inside)
    assert all(w * WINDOW + 1 > READ_UNTIL for _, w in outside)
    return inside, outside


def _occupancy():
    """Which (host, window) cells each view holds. The constructed truth.

    Spread so that every branch of the rule has cells on both sides of it:
    attack hosts and non-attack hosts, cells the hash keeps and cells it does
    not, and cells that only `auth` ever saw.
    """
    proc, flows, dns, auth = set(), set(), set(), set()
    inside, _outside = _attack_cells()
    for host_i, host in enumerate(HOSTS[:9]):          # C10 deliberately absent
        for w in range(host_i, 576, 17):
            proc.add((host, w))
            if w % 2 == 0:
                flows.add((host, w))
            if w % 5 == 0:
                dns.add((host, w))
    proc.update(inside)                                # every case is occupied
    auth.update(proc)
    auth.update((AUTH_ONLY_HOST, w) for w in range(0, 576, 9))
    auth.add(AUTH_ONLY_CELL_OF_ATTACK_HOST)
    assert AUTH_ONLY_CELL_OF_ATTACK_HOST not in proc | flows | dns
    return {"proc": proc, "flows": flows, "dns": dns, "auth": auth}


def _write_view(path, cells, width, source_column):
    """One synthetic view file, in the loader's wire format.

    Sorted by time because `_stream` raises TimestampInversion rather than
    sorting, which is the loader's contract and not something to work around.

    Every filler column is b"1": `flows` sums duration, packets and bytes with
    int(), so a non-numeric filler fails inside the frozen accumulator. The
    distinct-value features then all have cardinality 1, which is irrelevant
    here -- this file audits SELECTION, not feature arithmetic.
    """
    with gzip.open(path, "wb") as handle:
        for host, window in sorted(cells, key=lambda c: (c[1], c[0])):
            fields = [b"1"] * width
            fields[0] = str(window * WINDOW + 1).encode()
            fields[source_column] = host
            handle.write(b",".join(fields) + b"\n")


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory):
    root = tmp_path_factory.mktemp("lanl_synthetic")
    occ = _occupancy()
    for view, width, col in (("proc", 5, 2), ("flows", 9, 2),
                             ("dns", 3, 1), ("auth", 9, 3)):
        _write_view(root / f"{view}.txt.gz", occ[view], width, col)

    inside, outside = _attack_cells()
    with gzip.open(root / "redteam.txt.gz", "wb") as handle:
        for host, window in sorted(inside + outside, key=lambda c: c[1]):
            t = str(window * WINDOW + 1).encode()
            handle.write(b",".join([t, b"u@D", host, host]) + b"\n")

    cfg = dataclasses.replace(
        load_lanl_config(),
        raw_root=root,
        read_until=READ_UNTIL,
        control_divisor=DIVISOR,
        split={"train": {"start": 1, "end": DAY, "days": [1, 1]},
               # an empty gap: `block_of_window` tries train, gap, test in
               # order and an end below start can never match.
               "gap": {"start": DAY + 1, "end": DAY, "days": []},
               "test": {"start": DAY + 1, "end": READ_UNTIL, "days": [2, 2]}},
    )
    return cfg, occ, set(inside), set(outside)


def _hash_keeps(cell):
    return zlib.crc32(cell[0] + b"|" + str(cell[1]).encode()) % DIVISOR == 0


# ---------------------------------------------------------------------------
# the three assertions the brief asks for
# ---------------------------------------------------------------------------

def test_read_redteam_recovers_exactly_the_constructed_cells(synthetic):
    """The label source first: everything below is meaningless if this slips."""
    from prismflow.data.lanl_cyber1 import read_redteam
    cfg, _occ, inside, outside = synthetic
    got = read_redteam(cfg)
    assert got["rows"] == 57
    assert got["cells"] == inside | outside
    assert len(inside) == 50


def test_attack_hosts_matches_the_input_attack_set(synthetic):
    """Assertion 1. The derived host set is exactly the hosts that were attacked."""
    cfg, _occ, inside, outside = synthetic
    assert sp.attack_hosts(inside | outside) == {HOSTS[0], HOSTS[1], HOSTS[2],
                                                 HOSTS[7], HOSTS[8]}
    in_range = sp.cells_in_read_range(cfg, inside | outside)
    assert sp.attack_hosts(in_range) == {HOSTS[0], HOSTS[1], HOSTS[2]}
    # The out-of-range attack hosts must NOT be promoted to attack hosts: that
    # is the whole reason cells_in_read_range exists.
    assert HOSTS[7] not in sp.attack_hosts(in_range)
    assert HOSTS[8] not in sp.attack_hosts(in_range)


def test_cells_in_read_range_returns_the_right_subset(synthetic):
    """Assertion 2. Exactly the 50 constructed in-range cells, no more, no less."""
    cfg, _occ, inside, outside = synthetic
    got = sp.cells_in_read_range(cfg, inside | outside)
    assert got == inside
    assert len(got) == 50
    assert got & outside == set()


def test_host_rule_fires_only_on_benign_cells_of_attack_hosts(synthetic):
    """Assertion 3, checked against the streaming pass, not just the predicate.

    A cell is in the sample for exactly one of three reasons: it is a case, the
    hash kept it, or the host rule kept it. Every cell in the third group must
    be a benign cell of an in-range attack host.
    """
    cfg, _occ, inside, outside = synthetic
    all_cells = inside | outside
    hosts = sp.attack_hosts(sp.cells_in_read_range(cfg, all_cells))
    selected = sp.select_cells(cfg, all_cells, hosts, sp.v2_keep,
                              sp.NON_AUTH_VIEWS)["selected"]

    by_host_rule = {c for c in selected
                    if c not in all_cells and not _hash_keeps(c)}
    assert by_host_rule, "the host rule never fired, so this proves nothing"
    for cell in by_host_rule:
        assert cell[0] in hosts, cell
    # and it is not vacuous in the other direction either: some attack-host
    # cells are kept that the hash alone would have dropped.
    assert any(c[0] == HOSTS[0] for c in by_host_rule)
    assert any(c[0] == HOSTS[1] for c in by_host_rule)


# ---------------------------------------------------------------------------
# the streaming pass against a brute-force re-derivation
# ---------------------------------------------------------------------------

def test_select_cells_equals_a_brute_force_rederivation(synthetic):
    """Cell by cell, not aggregate: the check Part 30's committed sums cannot give."""
    cfg, occ, inside, outside = synthetic
    all_cells = inside | outside
    hosts = sp.attack_hosts(sp.cells_in_read_range(cfg, all_cells))

    non_auth = occ["proc"] | occ["flows"] | occ["dns"]
    in_range = {c for c in non_auth if c[1] * WINDOW + 1 <= READ_UNTIL}
    expected = {c for c in in_range
                if c in all_cells or c[0] in hosts or _hash_keeps(c)} | all_cells

    got = sp.select_cells(cfg, all_cells, hosts, sp.v2_keep, sp.NON_AUTH_VIEWS)
    assert got["selected"] == expected
    assert got["occupied_cells_per_streamed_view"] == {
        v: len({c for c in occ[v] if c[1] * WINDOW + 1 <= READ_UNTIL})
        for v in sp.NON_AUTH_VIEWS}


def test_auth_is_never_consulted_for_selection(synthetic):
    """L24's closing condition, on cells constructed to be auth-only.

    C10 exists only in `auth`, and one benign cell of attack host C01 exists
    only in `auth`. Neither may enter the sample -- the second is the sharper
    case, because rule (b) would otherwise have kept it.
    """
    cfg, occ, inside, outside = synthetic
    all_cells = inside | outside
    hosts = sp.attack_hosts(sp.cells_in_read_range(cfg, all_cells))
    selected = sp.select_cells(cfg, all_cells, hosts, sp.v2_keep,
                              sp.NON_AUTH_VIEWS)["selected"]

    assert AUTH_ONLY_HOST in {c for c, _w in occ["auth"]}
    assert not any(c[0] == AUTH_ONLY_HOST for c in selected)
    assert AUTH_ONLY_CELL_OF_ATTACK_HOST in occ["auth"]
    assert AUTH_ONLY_CELL_OF_ATTACK_HOST[0] in hosts
    assert AUTH_ONLY_CELL_OF_ATTACK_HOST not in selected


def test_part30_rule_on_the_same_synthetic_capture(synthetic):
    """The reproduction rule must keep cases plus the hash, and nothing else.

    Run over all four views, as Part 30 did, so an auth-only cell the hash
    keeps IS selected here. That contrast with the test above is exactly the
    selection change L24 asks for.
    """
    cfg, occ, inside, outside = synthetic
    all_cells = inside | outside
    hosts = sp.attack_hosts(sp.cells_in_read_range(cfg, all_cells))
    got = sp.select_cells(cfg, all_cells, hosts, sp.part30_keep, cfg.views)

    occupied = {c for v in cfg.views for c in occ[v]
                if c[1] * WINDOW + 1 <= READ_UNTIL}
    expected = {c for c in occupied
                if c in all_cells or _hash_keeps(c)} | all_cells
    assert got["selected"] == expected
    assert any(c[0] == AUTH_ONLY_HOST for c in got["selected"])


# ---------------------------------------------------------------------------
# end to end: features, labels and the blocks
# ---------------------------------------------------------------------------

def test_build_labels_exactly_the_50_in_range_cases(synthetic):
    """Through the frozen builder: 50 attack rows, and the blocks split by time."""
    import numpy as np
    from prismflow.data.lanl_cyber1 import read_redteam
    cfg, _occ, inside, outside = synthetic
    all_cells = inside | outside
    hosts = sp.attack_hosts(sp.cells_in_read_range(cfg, all_cells))
    selected = sp.select_cells(cfg, all_cells, hosts, sp.v2_keep,
                               sp.NON_AUTH_VIEWS)["selected"]
    samples = sp.build(cfg, sp.restrict_to_blocks(cfg, selected),
                       read_redteam(cfg))

    assert int((samples.labels == 1).sum()) == 50
    rows = {(c.encode(), int(w) // WINDOW)
            for c, w in zip(samples.computer, samples.window_start)}
    assert rows & outside == set()                 # day 4 is outside every block
    assert inside <= rows
    # Day 1 is train, day 2 is test, and no row lands anywhere else.
    block = np.asarray(samples.block)
    for i, w in enumerate(samples.window_start):
        assert block[i] == ("train" if w <= DAY else "test")
    assert set(block) == {"train", "test"}
