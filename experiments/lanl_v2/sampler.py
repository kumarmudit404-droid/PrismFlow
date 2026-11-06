"""Part 38 step 1: the redesigned LANL cell sampler (L24, L27).

WHAT IS NEW, against Part 30
----------------------------
Part 30 selected cells with one rule applied inside each view's own scan
(`prismflow.data.lanl_cyber1.scan_view` -> `_keep`): all cases, plus benign
cells at `crc32(computer|window) % 512 == 0`. Because the rule ran inside
EVERY view's scan, a cell occupied only in `auth` entered the sample. That is
the selection L24 names: auth presence alone reached test AUC 0.6637.

This module splits selection from feature extraction into two passes.

    pass 1   decide WHICH cells are in the sample, reading only the streams
             the pre-registration allows the decision to see
    pass 2   extract features for exactly those cells, from all four views

`auth` is read in pass 2 only. It therefore contributes FEATURES and never
SELECTION, which is L24's written closing condition.

WHY `scan_view` IS CALLED RATHER THAN COPIED
--------------------------------------------
`scan_view`'s second job is the per-view accumulators and the log1p/failure
rate finalisation -- the actual feature definitions, which must not be
reimplemented here or the two Parts would measure different features. It
takes the case set as a parameter and `_keep` returns True for any cell in
it, so passing the SELECTED set as `attack_cells` and a divisor no hash can
satisfy makes it extract features for precisely the selected cells. Nothing
in `prismflow/` is modified; `experiments/lanl/run_lanl.py` is not touched.

The divisor trick is belt-and-braces only: `select_only` is re-intersected
with the selection afterwards and a disagreement raises, so a hash collision
at 0 cannot quietly widen the sample.
"""

from __future__ import annotations

import dataclasses
import zlib
from collections import defaultdict

from prismflow.data.lanl_cyber1 import (
    RECORD_WIDTH,
    SOURCE_COLUMN,
    LANLConfig,
    _stream,
    build_lanl_samples,
    scan_view,
)

# Cells whose crc32 is exactly 0 are the only ones a divisor of 2**32 admits.
# Used to disable the hash in the feature pass, never to select anything.
NO_HASH = 2 ** 32

NON_AUTH_VIEWS = ("proc", "flows", "dns")


def cells_in_read_range(cfg: LANLConfig, attack_cells) -> set:
    """The attack cells whose window starts at or before `cfg.read_until`.

    `prismflow.data.lanl_cyber1.read_redteam` streams redteam.txt.gz with
    `until=None` -- deliberately, because Part 30 needed every labelled cell
    and a cell outside the read range simply never met a view scan. Here the
    distinction matters: the pre-registration defines an attack host as one
    with an attack cell IN THE READ RANGE, so an unfiltered set would keep
    every benign cell of a host whose only attack lands on a day this run
    never reads.
    """
    return {(c, w) for c, w in attack_cells
            if w * cfg.window_seconds + 1 <= cfg.read_until}


def attack_hosts(attack_cells: set[tuple[bytes, int]]) -> set[bytes]:
    """Every computer with at least one attack cell in the set given (L27).

    Pass `cells_in_read_range(...)` output, not `read_redteam`'s raw set.
    """
    return {computer for computer, _window in attack_cells}


def part30_keep(cell, attack_cells, hosts, divisor):
    """Part 30's rule, re-implemented for the reproduction gate.

    `hosts` is accepted and ignored so both rules share one signature.
    """
    if cell in attack_cells:
        return True
    return zlib.crc32(cell[0] + b"|" + str(cell[1]).encode()) % divisor == 0


def v2_keep(cell, attack_cells, hosts, divisor):
    """The pre-registered rule. Cases, all benign cells of attack hosts, hash rest.

    The non-auth condition is NOT here: it is enforced by which views pass 1
    streams. Keeping it out of the predicate is deliberate -- the predicate
    cannot see a view name, so it cannot accidentally come to depend on one.
    """
    if cell in attack_cells:
        return True
    if cell[0] in hosts:
        return True
    return zlib.crc32(cell[0] + b"|" + str(cell[1]).encode()) % divisor == 0


def select_cells(cfg: LANLConfig, attack_cells, hosts, keep, views,
                 divisor: int | None = None) -> dict:
    """Pass 1. Stream `views` and return the selected cell set plus counters.

    Only selected cells are retained, so memory is bounded by the sample size
    and not by the ~2.3e7 cells each view occupies. The decision is cached per
    (window, computer) exactly as `scan_view` caches it, so crc32 runs about
    once per cell rather than once per record.

    Cases are added unconditionally at the end: an attack cell occupied in no
    streamed view is still a case, and dropping it would silently change the
    label distribution the pre-registration fixed.
    """
    divisor = cfg.control_divisor if divisor is None else divisor
    selected: set[tuple[bytes, int]] = set()
    records = defaultdict(int)
    occupied_cells = defaultdict(int)

    for view in views:
        path = cfg.raw_root / f"{view}.txt.gz"
        source = SOURCE_COLUMN[view]
        width = RECORD_WIDTH[view]
        decision: dict[bytes, bool] = {}
        seen_this_window: set[bytes] = set()
        current_window = -1
        for fields, _lineno in _stream(path, view, width, cfg.read_until):
            records[view] += 1
            window = cfg.window_of(int(fields[0]))
            if window != current_window:
                occupied_cells[view] += len(seen_this_window)
                seen_this_window = set()
                decision = {}
                current_window = window
            computer = fields[source]
            seen_this_window.add(computer)
            take = decision.get(computer)
            if take is None:
                take = keep((computer, window), attack_cells, hosts, divisor)
                decision[computer] = take
            if take:
                selected.add((computer, window))
        occupied_cells[view] += len(seen_this_window)

    # Cases are in the sample whatever pass 1 streamed.
    before = len(selected)
    selected |= set(attack_cells)
    return {
        "selected": selected,
        "n_selected": len(selected),
        "cases_added_outside_streamed_views": len(selected) - before,
        "records_scanned": dict(records),
        "occupied_cells_per_streamed_view": dict(occupied_cells),
        "streamed_views": list(views),
        "divisor": divisor,
    }


def feature_scans(cfg: LANLConfig, selected: set) -> dict:
    """Pass 2. Features for exactly the selected cells, from every view.

    The returned `kept` sets are intersected back onto `selected`: `scan_view`
    could in principle admit a cell whose crc32 is 0, and a sample that grew
    by a hash collision would not be the pre-registered sample.
    """
    cfg_nohash = dataclasses.replace(cfg, control_divisor=NO_HASH)
    scans = {}
    for view in cfg.views:
        scan = scan_view(view, cfg_nohash, selected)
        extra = set(scan["kept"]) - selected
        if extra:
            scan["kept"] = {c: v for c, v in scan["kept"].items() if c in selected}
            scan["hash_collisions_dropped"] = len(extra)
        scans[view] = scan
    return scans


def build(cfg: LANLConfig, selected: set, redteam: dict):
    """Join pass-2 scans into the [N, V, D] table, via the frozen builder."""
    return build_lanl_samples(cfg, feature_scans(cfg, selected), redteam)


def per_day_counts(samples) -> dict:
    """Samples, attack samples and distinct attack windows per day, per block.

    The shape of `results/lanl/results.json`
    `split_report.per_split.<block>.per_day`, so the reproduction gate can
    compare like with like.
    """
    out: dict[str, dict] = {}
    windows: dict[tuple[str, int], set] = defaultdict(set)
    for i, block in enumerate(samples.block):
        t = int(samples.window_start[i])
        day = samples.config.day_of(t)
        bucket = out.setdefault(block, {}).setdefault(
            day, {"samples": 0, "attack_samples": 0, "attack_windows": 0})
        bucket["samples"] += 1
        if samples.labels[i] == 1:
            bucket["attack_samples"] += 1
            windows[(block, day)].add(samples.config.window_of(t))
    for (block, day), ws in windows.items():
        out[block][day]["attack_windows"] = len(ws)
    return out
