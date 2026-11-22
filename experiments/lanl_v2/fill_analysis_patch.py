"""Patch for the `peak_rss_mb` aggregation defect in `fill_analysis.py`.

NOT APPLIED. Run `python experiments/lanl_v2/fill_analysis_patch.py --apply`
after the sweep reaches 40/40. With no flag it only self-checks.

THE DEFECT
----------
`peak_rss_mb` is a high-water mark for the life of the *process* that wrote
the unit, not a per-unit measurement, so `agg()` must not average it.
Measured over the 36 units on disk (`units_default/*.json`, `peak_rss_mb`):

    499.6 - 523.1, distinct per unit   10 units   7 Nov sweep_block.sh:
                                                  one short-lived process per
                                                  unit, so these ARE per-unit
    513.8, identical                   24 units   one long-lived 21 Nov
                                                  process: one HWM, 24 times
    496.7, identical                    2 units   the 21 Nov 22:09 resume

For the 24-unit group `agg()` is not averaging 24 measurements; it counts ONE
measurement 24 times and reports `std = 0.0`, which reads as perfect
reproducibility when it is a single observation -- the same failure mode as a
seed loop that varies nothing.

ONE REGIME PER LAUNCH, NOT THREE. The review brief calls this three regimes.
The 4-unit resume launched 2026-11-22 11:50 adds a fourth distinct value, so a
caption naming a fixed count is wrong at the next launch. The caption states
the per-launch semantic instead.

WHAT IS NOT MEASURED, AND IS NOT INVENTED
-----------------------------------------
The brief asks for a `process_id` or `launch_time` field "so the regime is
traceable". Neither is a field in the unit JSON, and the runner that writes
them is not edited by this patch. File mtime is not a substitute: a `git
checkout` rewrites it, which is exactly what happened to 10 of these 36 units
on 2026-11-22 11:47. So launch identity is reported as `not measured` rather
than reconstructed. Identical values across seeds are the only evidence that
two units shared a process, and that is what `n_distinct` reports.

`unit_seconds` IS LEFT AGGREGATED, DELIBERATELY
-----------------------------------------------
It carries a related but milder defect: `with_auth_naive_seed{0,1}` were timed
at 653.8 s / 705.4 s on 7 Nov and `seed{2,3}` at 1359.8 s / 1370.8 s on 21 Nov
-- a 2.0x machine-state difference across launches, same field, same cell. But
each value is still a genuine per-unit measurement, so a mean stays a
legitimate *operational* summary (Table 6 carries no hypothesis). It gets a
caption saying it spans launches, not a structural change. This is also why the
`PREDICT=1524` floor in `sweep_block.sh` sits below the real
`with_auth`/`prismflow` cost.
"""
from __future__ import annotations

import sys
from pathlib import Path

TARGET = Path(__file__).resolve().parent / "fill_analysis.py"

# --- edit 1: drop peak_rss_mb from the aggregated fields -------------------

OLD_FIELDS = '''TABLE6_FIELDS = ("unit_seconds", "peak_rss_mb", "rows_at_or_above_threshold",
                 "max_prob_attack", "all_false_mask_rows")'''

NEW_FIELDS = '''#: `peak_rss_mb` is DELIBERATELY ABSENT: it is a per-launch high-water mark,
#: not a per-unit measurement, so `agg()` must not mean it across seeds. It is
#: emitted per seed by `table6()` instead. See fill_analysis_patch.py.
TABLE6_FIELDS = ("unit_seconds", "rows_at_or_above_threshold",
                 "max_prob_attack", "all_false_mask_rows")

PEAK_RSS_SEMANTIC = (
    "peak_rss_mb is a high-water mark for the life of the PROCESS that wrote "
    "the unit, not a per-unit measurement. Reported per seed and never "
    "averaged: units written by one long-lived process all carry that "
    "process's single HWM, so a mean over them counts one measurement n times "
    "and its std is 0.0 by construction rather than by reproducibility. One "
    "HWM per launch -- n_distinct is the number of launches that wrote this "
    "cell, not a fixed number of regimes.")

PEAK_RSS_LAUNCH_ID = (
    "not measured: the unit JSON records no process_id and no launch_time, so "
    "which launch produced a given peak_rss_mb is not derivable from "
    "committed data. File mtime is not a substitute -- a git checkout "
    "rewrites it. Identical values across seeds are the only evidence that "
    "two units shared a process.")

UNIT_SECONDS_SEMANTIC = (
    "unit_seconds is aggregated, but the seeds of a cell may span launches "
    "with different machine states: with_auth/naive seeds 0,1 measured 653.8s "
    "and 705.4s while seeds 2,3 measured 1359.8s and 1370.8s, a 2.0x "
    "difference in the same field of the same cell. Each value is a genuine "
    "per-unit measurement, so the mean is a legitimate operational summary; "
    "it is not a reproducibility figure.")'''

# --- edit 2: emit peak_rss_mb per seed, with what provenance exists --------

OLD_TABLE6 = '''        cell["seeds"] = list(seeds)
        cell["degenerate_at_threshold_per_seed"] = ['''

NEW_TABLE6 = '''        peaks = [units[n].get("peak_rss_mb") for n in names]
        distinct = sorted({float(v) for v in peaks if v is not None})
        cell["peak_rss_mb_per_seed"] = {
            "per_seed": [None if v is None else float(v) for v in peaks],
            "aggregated": False,
            "n_distinct": len(distinct),
            "distinct_values": distinct,
            "semantic": PEAK_RSS_SEMANTIC,
            "launch_identity": PEAK_RSS_LAUNCH_ID,
        }
        cell["seeds"] = list(seeds)
        cell["degenerate_at_threshold_per_seed"] = ['''

# --- edit 3: the Table 6 caption ------------------------------------------

OLD_WHAT = '''        "what": "Operational record and the Amendment 10.2 threshold-degeneracy "
                "diagnostic. No hypothesis is attached to any of it.",'''

NEW_WHAT = '''        "what": "Operational record and the Amendment 10.2 threshold-degeneracy "
                "diagnostic. No hypothesis is attached to any of it.",
        "peak_rss_mb_semantic": PEAK_RSS_SEMANTIC,
        "peak_rss_mb_launch_identity": PEAK_RSS_LAUNCH_ID,
        "unit_seconds_semantic": UNIT_SECONDS_SEMANTIC,'''

EDITS = ((OLD_FIELDS, NEW_FIELDS), (OLD_TABLE6, NEW_TABLE6), (OLD_WHAT, NEW_WHAT))


def self_check(src: str) -> None:
    """Every anchor must appear exactly once, and the patch must be idempotent.

    A silent no-match would leave `peak_rss_mb` aggregated while the commit
    message claimed it was fixed, which is the one outcome worth asserting
    against.
    """
    for old, new in EDITS:
        assert src.count(old) == 1, f"anchor not unique ({src.count(old)}x): {old[:60]!r}"
        assert new not in src, f"already applied: {new[:60]!r}"
    assert '"peak_rss_mb"' in OLD_FIELDS and '"peak_rss_mb"' not in NEW_FIELDS, \
        "edit 1 must remove peak_rss_mb from the aggregated tuple"


def apply(src: str) -> str:
    out = src
    for old, new in EDITS:
        out = out.replace(old, new, 1)
    return out


def main(*argv: str) -> int:
    # fill_analysis.py is CRLF on this box; the anchors above are LF. Normalise
    # for matching and restore the original ending on write, so the diff is the
    # three edits and not every line in the file.
    raw = TARGET.read_bytes().decode("utf-8")
    crlf = "\r\n" in raw
    src = raw.replace("\r\n", "\n")
    self_check(src)
    if "--apply" not in argv:
        print("self-check OK: 3 anchors found, none applied yet.")
        print("re-run with --apply to patch", TARGET.name)
        return 0
    out = apply(src)
    TARGET.write_bytes((out.replace("\n", "\r\n") if crlf else out).encode("utf-8"))
    print("patched", TARGET.name, "-- now run py_compile and fill_analysis.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
