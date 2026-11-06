"""Part 38 step 1: the redesigned sampler's selection rules.

The reproduction gate in `run_lanl_v2.py reproduce` is the end-to-end check
and needs an 11 GB scan. These are the checks that do not: the rules are pure
functions of a cell, an attack set and a host set, so they can be pinned
exactly without reading a byte of the capture.

`test_part30_keep_matches_the_frozen_rule` is the important one. The
reproduction gate only means something if the rule it reproduces is really
Part 30's, so the re-implementation is compared against
`prismflow.data.lanl_cyber1._keep` itself rather than against a description
of it.
"""

from __future__ import annotations

import zlib

from prismflow.data.lanl_cyber1 import _keep
from experiments.lanl_v2 import sampler as sp


def _cells(n: int = 4000):
    return [(f"C{i}".encode(), i * 7 % 2048) for i in range(n)]


def test_part30_keep_matches_the_frozen_rule():
    """Identical verdict on every cell, cases and controls, at Part 30's 512."""
    cells = _cells()
    attack = set(cells[:17])
    hosts = sp.attack_hosts(attack)
    for cell in cells:
        assert sp.part30_keep(cell, attack, hosts, 512) == _keep(cell, attack, 512)


def test_part30_keep_ignores_the_host_set():
    """The reproduction rule must not acquire L27's behaviour by accident."""
    cells = _cells(500)
    attack = {cells[0]}
    hosts = {c[0] for c in cells}          # every host "is" an attack host
    for cell in cells:
        assert sp.part30_keep(cell, attack, hosts, 512) == _keep(cell, attack, 512)


def test_v2_keeps_every_cell_of_an_attack_host():
    """L27's closing condition: the stated share for attack hosts is 1/1."""
    attack = {(b"C17693", 10)}
    hosts = sp.attack_hosts(attack)
    assert hosts == {b"C17693"}
    for window in range(2000):
        assert sp.v2_keep((b"C17693", window), attack, hosts, 512) is True


def test_v2_leaves_non_attack_hosts_on_part30s_hash():
    """A host with no attack cell keeps the unchanged crc32 % 512 rule."""
    attack = {(b"C17693", 10)}
    hosts = sp.attack_hosts(attack)
    kept = 0
    cells = [(b"C999", w) for w in range(5120)]
    for cell in cells:
        got = sp.v2_keep(cell, attack, hosts, 512)
        assert got == _keep(cell, attack, 512)
        kept += got
    # Not asserted as exactly 1/512: crc32 is not uniform over this tiny
    # family. Asserted as "a small minority", which is what the rule is for.
    assert 0 < kept < len(cells) // 50


def test_v2_keeps_every_case():
    """Cases are kept whatever their host and whatever the hash says."""
    attack = {(b"Cx", 1), (b"Cy", 2)}
    hosts = sp.attack_hosts(attack)
    for cell in attack:
        assert sp.v2_keep(cell, attack, hosts, 512) is True


def test_no_hash_divisor_admits_almost_nothing():
    """The feature pass disables the hash; only crc32 == 0 could slip through.

    `feature_scans` also intersects back onto the selection, so this is the
    first of two guards, not the only one.
    """
    admitted = [c for c in _cells(20000)
                if zlib.crc32(c[0] + b"|" + str(c[1]).encode()) % sp.NO_HASH == 0]
    assert admitted == []


def test_non_auth_views_excludes_auth():
    """L24: the selection pass may not stream the auth file."""
    assert "auth" not in sp.NON_AUTH_VIEWS
    assert set(sp.NON_AUTH_VIEWS) == {"proc", "flows", "dns"}


def test_cells_in_read_range_filters_by_read_until():
    """L27's host set is read-range-scoped, not whole-file.

    `read_redteam` streams with until=None by design, so an unfiltered set
    would make every benign cell of a host attacked on an unread day a
    control. Measured on the real file: 305 hosts unfiltered against 24 in a
    6-day range.
    """
    import dataclasses
    from prismflow.data.lanl_cyber1 import load_lanl_config

    cfg = dataclasses.replace(load_lanl_config(), read_until=6 * 86400)
    inside = (b"C1", 10)                      # t = 3001
    outside = (b"C2", 5000)                   # t = 1500001, past day 6
    got = sp.cells_in_read_range(cfg, {inside, outside})
    assert got == {inside}
    assert sp.attack_hosts(got) == {b"C1"}


def test_read_range_boundary_is_inclusive():
    """The last window that starts inside the range is in the range."""
    import dataclasses
    from prismflow.data.lanl_cyber1 import load_lanl_config

    cfg = dataclasses.replace(load_lanl_config(), read_until=86400)
    last = (b"C1", 287)                       # t = 86101, inside day 1
    first_after = (b"C1", 288)                # t = 86401, day 2
    got = sp.cells_in_read_range(cfg, {last, first_after})
    assert got == {last}
