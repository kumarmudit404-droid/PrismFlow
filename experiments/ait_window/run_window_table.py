"""Part 27: the AIT window-length / sample-mode table.

STATISTICS ONLY. Nothing here trains, predicts, fuses or computes ENIV: every
column is a count, a fraction of counts, a timestamp or a list of attack-step
names, read out of the Part 26 loader (`prismflow/data/ait_lds.py`, imported,
never edited). Part 27 reports the table; the window and sample mode are the
user's decision at the review gate.

WHAT A ROW IS
    one (scenario, sample_mode, window_seconds) combination, scanned directly
    at that window length. The four lengths are exact multiples of each other,
    so `rewindow` could re-bin one 30 s scan into all of them, but each length
    is scanned on its own here: a direct scan is what the brief asks for, it
    costs ~50 s per scenario per length, and it removes the re-bin identity
    from the set of things the table depends on.

THREE COLUMNS THAT ARE NOT OBVIOUS
    attack_fraction      counted over `label_known` samples only. A window no
                         label-covered file contributed to is NOT benign; it is
                         unexamined (`labels == -1`), and dividing by all
                         samples would quietly turn those into negatives.
    attack_runs          maximal runs of consecutive attack windows ON THE TIME
                         GRID (and per host in host mode). The run COUNT, not
                         the attack-window count, is how many independent
                         attack episodes a row contains: 664 consecutive
                         windows of one exfiltration tool are one episode, not
                         664 observations.
    dnsteal_only         attack windows whose ONLY attack steps are
                         dnsteal-family. Sits beside attack_fraction so neither
                         can be quoted without the other, because on both
                         scenarios the window label is close to "was the DNS
                         exfiltration tool running".

Usage:
    python experiments/ait_window/run_window_table.py
    python experiments/ait_window/run_window_table.py --scenarios santos
    python experiments/ait_window/run_window_table.py --windows 300
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from prismflow.data.ait_lds import (  # noqa: E402
    FEATURES,
    LABEL_UNKNOWN,
    AITSamples,
    ScanResult,
    build_samples,
    load_ait_config,
    load_ait_settings,
    raw_manifest,
    scan_scenario,
    sensor_groups,
)

#: Window lengths, seconds. Part 27's brief; 30 is the shortest and every
#: longer one is an exact multiple of it.
WINDOWS: tuple[int, ...] = (30, 60, 300, 900)

#: Both sample modes. Part 27 reports both and chooses neither.
MODES: tuple[str, ...] = ("network", "host")

SCENARIOS: tuple[str, ...] = ("russellmitchell", "santos")

#: The continuously-running DNS exfiltration campaign's step names. Named
#: explicitly here, not inferred: which steps belong to one campaign is a
#: reading of the scenario, and `docs/part27-ait-datacard.md` states this set.
DNSTEAL_FAMILY: frozenset[str] = frozenset(
    {"dnsteal", "dnsteal-received", "dnsteal-dropped", "attacker"}
)

#: A row is "usable" for a binary window-level task only if both classes have
#: at least this many windows. A reported threshold, not a choice.
MIN_PER_CLASS = 30

OUT_DIR = _REPO_ROOT / "results" / "ait_window"


# ---------------------------------------------------------------------------
# Contiguous runs
# ---------------------------------------------------------------------------


def contiguous_runs(
    window_start: np.ndarray,
    flags: np.ndarray,
    window_seconds: int,
    host_id: np.ndarray | None = None,
) -> list[int]:
    """Lengths of maximal runs of `flags` that are adjacent on the time grid.

    Adjacent means the next sample starts exactly one window later, so a
    dropped empty window (`drop_empty_windows: true`) breaks a run instead of
    joining the windows either side of it. In host mode runs are computed per
    host and pooled, because two hosts' windows at the same instant are not
    consecutive observations of one thing.

    Returned longest-first. The list's LENGTH is the number of independent
    episodes; its values are how long each lasted.
    """
    if window_start.size == 0:
        return []
    groups: list[np.ndarray] = []
    if host_id is None:
        groups.append(np.arange(window_start.size))
    else:
        for host in np.unique(host_id):
            groups.append(np.flatnonzero(host_id == host))

    lengths: list[int] = []
    for rows in groups:
        order = rows[np.argsort(window_start[rows], kind="stable")]
        current = 0
        previous_start: float | None = None
        for row in order:
            if not flags[row]:
                if current:
                    lengths.append(current)
                current = 0
                previous_start = None
                continue
            start = float(window_start[row])
            if current and previous_start is not None and (
                abs(start - previous_start - window_seconds) > 1e-6
            ):
                lengths.append(current)
                current = 0
            current += 1
            previous_start = start
        if current:
            lengths.append(current)
    return sorted(lengths, reverse=True)


# ---------------------------------------------------------------------------
# One row
# ---------------------------------------------------------------------------


def events_in_grid(scan: ScanResult) -> dict[str, int]:
    """In-grid event count per source, summed over the scan's cells.

    Taken from the cells rather than from `FileStats.events` so that events
    outside the window grid -- 158,966 on santos, 169,150 on russellmitchell --
    cannot inflate a per-window mean for windows they are not in.
    """
    totals: dict[str, int] = {source: 0 for source in scan.config.sources}
    for (_, _, source), cell in scan.cells.items():
        totals[source] = totals.get(source, 0) + cell.n
    return totals


def describe(samples: AITSamples, scan: ScanResult) -> dict[str, Any]:
    """Every Part 27 column for one (scenario, mode, window)."""
    labels = samples.labels
    mask = samples.view_mask
    n = len(samples)
    attack = labels == 1
    benign = labels == 0
    unlabelled = labels == LABEL_UNKNOWN
    n_attack = int(attack.sum())
    n_benign = int(benign.sum())
    n_known = int(samples.label_known.sum())

    attack_runs = contiguous_runs(
        samples.window_start, attack, samples.config.window_seconds, samples.host_id
    )
    benign_runs = contiguous_runs(
        samples.window_start, benign, samples.config.window_seconds, samples.host_id
    )

    dnsteal_only = 0
    other_step = 0
    for row, steps in enumerate(samples.attack_steps):
        if labels[row] != 1:
            continue
        if set(steps) <= DNSTEAL_FAMILY:
            dnsteal_only += 1
        else:
            other_step += 1

    totals = events_in_grid(scan)
    views: dict[str, Any] = {}
    for column, source in enumerate(samples.sources):
        present = int(mask[:, column].sum())
        views[source] = {
            "present_samples": present,
            "missing_samples": n - present,
            "missing_fraction": (n - present) / n if n else None,
            "events_in_grid": totals.get(source, 0),
            "mean_events_per_present_window": (
                totals.get(source, 0) / present if present else None
            ),
            "mean_events_per_sample": totals.get(source, 0) / n if n else None,
            "feature_dim": len(FEATURES[source]),
        }

    steps = samples.attack_step_frequency()
    return {
        "scenario": samples.config.scenario,
        "sample_mode": samples.config.sample_mode,
        "window_seconds": samples.config.window_seconds,
        "n_windows_on_grid": scan.n_windows,
        "n_samples": n,
        "n_attack": n_attack,
        "n_benign": n_benign,
        "n_not_labelled": int(unlabelled.sum()),
        "n_label_known": n_known,
        # Over label_known only. Dividing by n_samples would read every
        # unexamined window as a negative.
        "attack_fraction_of_labelled": (n_attack / n_known if n_known else None),
        "attack_fraction_of_all_samples": (n_attack / n if n else None),
        "attack_runs": {"count": len(attack_runs), "lengths": attack_runs},
        "benign_runs": {"count": len(benign_runs), "lengths": benign_runs},
        "dnsteal_family": sorted(DNSTEAL_FAMILY),
        "attack_windows_dnsteal_only": dnsteal_only,
        "attack_windows_dnsteal_only_fraction": (
            dnsteal_only / n_attack if n_attack else None
        ),
        "attack_windows_with_other_step": other_step,
        "mean_views_present": (
            float(mask.sum(axis=1).mean()) if n else None
        ),
        "views": views,
        "view_order": list(samples.sources),
        "n_hosts": len(samples.hosts),
        "attack_steps_present": sorted(steps),
        "n_attack_steps_present": len(steps),
        "attack_step_frequency": steps,
        "usable_both_classes": n_attack >= MIN_PER_CLASS and n_benign >= MIN_PER_CLASS,
    }


# ---------------------------------------------------------------------------
# Derived summaries
# ---------------------------------------------------------------------------


def smallest_usable_window(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Smallest window leaving >= MIN_PER_CLASS attack AND benign windows.

    Reported per (scenario, mode) with the two counts that decide it. When no
    length qualifies the row says so and names the best length tried; it never
    extrapolates to a length that was not run.
    """
    out: list[dict[str, Any]] = []
    pairs = sorted({(row["scenario"], row["sample_mode"]) for row in rows})
    for scenario, mode in pairs:
        candidates = sorted(
            (r for r in rows if r["scenario"] == scenario and r["sample_mode"] == mode),
            key=lambda r: r["window_seconds"],
        )
        qualifying = [r for r in candidates if r["usable_both_classes"]]
        if qualifying:
            best = qualifying[0]
            out.append(
                {
                    "scenario": scenario,
                    "sample_mode": mode,
                    "smallest_usable_window_seconds": best["window_seconds"],
                    "n_attack": best["n_attack"],
                    "n_benign": best["n_benign"],
                    "threshold": MIN_PER_CLASS,
                    "windows_tried": [r["window_seconds"] for r in candidates],
                }
            )
        else:
            richest = max(candidates, key=lambda r: min(r["n_attack"], r["n_benign"]))
            out.append(
                {
                    "scenario": scenario,
                    "sample_mode": mode,
                    "smallest_usable_window_seconds": None,
                    "note": "none of the lengths tried",
                    "best_window_seconds": richest["window_seconds"],
                    "n_attack": richest["n_attack"],
                    "n_benign": richest["n_benign"],
                    "threshold": MIN_PER_CLASS,
                    "windows_tried": [r["window_seconds"] for r in candidates],
                }
            )
    return out


def view_identity(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Are the views identical across scenarios? Compared, not assumed."""
    by_scenario: dict[str, set[tuple]] = {}
    for row in rows:
        signature = tuple(
            (source, row["views"][source]["feature_dim"]) for source in row["view_order"]
        )
        by_scenario.setdefault(row["scenario"], set()).add(signature)
    inconsistent = {s: len(v) for s, v in by_scenario.items() if len(v) > 1}
    signatures = {s: sorted(v) for s, v in by_scenario.items()}
    distinct = {next(iter(v)) for v in by_scenario.values() if len(v) == 1}
    agree = len(distinct) == 1 and not inconsistent
    # With one scenario in the table there is nothing to compare, and reporting
    # "identical" would answer a question that was not asked.
    comparable = len(by_scenario) >= 2
    return {
        "n_scenarios_compared": len(by_scenario),
        "identical_across_scenarios": (agree if comparable else None),
        "not_comparable_reason": (
            None if comparable else "only one scenario is present in this table"
        ),
        "signature_per_scenario": {
            s: [list(map(list, sig)) for sig in sigs] for s, sigs in signatures.items()
        },
        "differences": (
            [] if agree else ["view signature differs; see signature_per_scenario"]
        ),
        "sensor_groups": sensor_groups(rows[0]["view_order"]) if rows else {},
    }


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "scenario",
    "sample_mode",
    "window_seconds",
    "n_windows_on_grid",
    "n_samples",
    "n_attack",
    "n_benign",
    "n_not_labelled",
    "n_label_known",
    "attack_fraction_of_labelled",
    "attack_runs_count",
    "attack_runs_lengths",
    "benign_runs_count",
    "benign_runs_lengths",
    "attack_windows_dnsteal_only",
    "attack_windows_dnsteal_only_fraction",
    "attack_windows_with_other_step",
    "mean_views_present",
    "n_attack_steps_present",
    "attack_steps_present",
    "usable_both_classes",
]


def _runs_text(lengths: Sequence[int], limit: int = 12) -> str:
    head = "/".join(str(x) for x in lengths[:limit])
    return head + (f"/... (+{len(lengths) - limit} more)" if len(lengths) > limit else "")


def write_csv(rows: Sequence[dict[str, Any]], views: Sequence[str], path: Path) -> None:
    columns = list(CSV_COLUMNS)
    for source in views:
        columns += [f"missing_fraction_{source}", f"mean_events_present_{source}"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in CSV_COLUMNS if key in row}
            flat["attack_runs_count"] = row["attack_runs"]["count"]
            flat["attack_runs_lengths"] = _runs_text(row["attack_runs"]["lengths"])
            flat["benign_runs_count"] = row["benign_runs"]["count"]
            flat["benign_runs_lengths"] = _runs_text(row["benign_runs"]["lengths"])
            flat["attack_steps_present"] = " ".join(row["attack_steps_present"])
            for source in views:
                stats = row["views"][source]
                flat[f"missing_fraction_{source}"] = stats["missing_fraction"]
                flat[f"mean_events_present_{source}"] = stats[
                    "mean_events_per_present_window"
                ]
            writer.writerow(flat)


def _fraction(value: float | None, digits: int = 3) -> str:
    return "not measured" if value is None else f"{value:.{digits}f}"


def _number(value: float | None, digits: int = 1) -> str:
    return "not measured" if value is None else f"{value:,.{digits}f}"


def write_markdown(
    rows: Sequence[dict[str, Any]],
    views: Sequence[str],
    identity: dict[str, Any],
    usable: Sequence[dict[str, Any]],
    provenance: dict[str, Any],
    path: Path,
) -> None:
    lines: list[str] = []
    add = lines.append
    add("# Part 27 - AIT window length and sample mode")
    add("")
    add(
        "Statistics only. No training, no ENIV, no accuracy: every number below "
        "is a count or a fraction of counts, produced by "
        "`experiments/ait_window/run_window_table.py` over the Part 26 loader."
    )
    add("")
    add(
        "`attack_fraction` is over **label_known samples only** - a window no "
        "label-covered file contributed to is unexamined, not benign."
    )
    add("")

    add("## 1. Samples and labels")
    add("")
    add(
        "| scenario | mode | window s | grid | samples | attack | benign | not labelled "
        "| attack frac (labelled) | dnsteal-only attack | dnsteal-only frac | attack w/ other step |"
    )
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in rows:
        add(
            f"| {row['scenario']} | {row['sample_mode']} | {row['window_seconds']} "
            f"| {row['n_windows_on_grid']:,} | {row['n_samples']:,} | {row['n_attack']:,} "
            f"| {row['n_benign']:,} | {row['n_not_labelled']:,} "
            f"| {_fraction(row['attack_fraction_of_labelled'])} "
            f"| {row['attack_windows_dnsteal_only']:,} "
            f"| {_fraction(row['attack_windows_dnsteal_only_fraction'])} "
            f"| {row['attack_windows_with_other_step']:,} |"
        )
    add("")
    add(
        f"dnsteal-family = {', '.join('`' + s + '`' for s in sorted(DNSTEAL_FAMILY))}. "
        "An attack window counted in `dnsteal-only` carries no other attack step."
    )
    add("")

    add("## 2. Contiguous runs on the time grid")
    add("")
    add(
        "The run **count** is how many independent attack episodes the row "
        "contains; the lengths are how long each lasted, in windows. Host mode "
        "runs are per host and pooled. A dropped empty window breaks a run."
    )
    add("")
    add(
        "| scenario | mode | window s | attack runs | attack run lengths "
        "| benign runs | benign run lengths |"
    )
    add("|---|---|---|---|---|---|---|")
    for row in rows:
        add(
            f"| {row['scenario']} | {row['sample_mode']} | {row['window_seconds']} "
            f"| **{row['attack_runs']['count']}** | {_runs_text(row['attack_runs']['lengths'], 8)} "
            f"| {row['benign_runs']['count']} | {_runs_text(row['benign_runs']['lengths'], 8)} |"
        )
    add("")

    add("## 3. Per-view missing fraction (view_mask False)")
    add("")
    add(
        "| scenario | mode | window s | mean views present | "
        + " | ".join(views)
        + " |"
    )
    add("|---|---|---|---|" + "---|" * len(views))
    for row in rows:
        cells = " | ".join(
            _fraction(row["views"][source]["missing_fraction"]) for source in views
        )
        add(
            f"| {row['scenario']} | {row['sample_mode']} | {row['window_seconds']} "
            f"| {_number(row['mean_views_present'], 2)} | {cells} |"
        )
    add("")

    add("## 4. Per-view mean events per window where the view is present")
    add("")
    add(
        "In-grid events only, summed from the scan's cells, so events outside "
        "the declared scenario interval cannot inflate a per-window mean."
    )
    add("")
    add("| scenario | mode | window s | " + " | ".join(views) + " |")
    add("|---|---|---|" + "---|" * len(views))
    for row in rows:
        cells = " | ".join(
            _number(row["views"][source]["mean_events_per_present_window"])
            for source in views
        )
        add(
            f"| {row['scenario']} | {row['sample_mode']} | "
            f"{row['window_seconds']} | {cells} |"
        )
    add("")

    add("## 5. Attack steps reaching at least one attack window")
    add("")
    add("| scenario | mode | window s | distinct steps | steps |")
    add("|---|---|---|---|---|")
    for row in rows:
        add(
            f"| {row['scenario']} | {row['sample_mode']} | {row['window_seconds']} "
            f"| {row['n_attack_steps_present']} "
            f"| {', '.join('`' + s + '`' for s in row['attack_steps_present'])} |"
        )
    add("")

    add("## 6. Are the views identical across the two scenarios?")
    add("")
    if identity["identical_across_scenarios"] is None:
        add(
            f"**Not comparable here: {identity['not_comparable_reason']}.** The "
            "signature below is what this table measured."
        )
    else:
        add(
            f"**{'Yes' if identity['identical_across_scenarios'] else 'No'}.** "
            "Compared as (source name, native feature width) in view order, for "
            "every row."
        )
    add("")
    for scenario, signatures in identity["signature_per_scenario"].items():
        for signature in signatures:
            pretty = ", ".join(f"`{name}`:{dim}" for name, dim in signature)
            add(f"- `{scenario}`: {pretty}")
    if identity["differences"]:
        add("")
        for difference in identity["differences"]:
            add(f"- DIFFERENCE: {difference}")
    add("")
    add(f"Same-sensor groups: `{identity['sensor_groups']}` (view indices).")
    add("")

    add(f"## 7. Smallest window leaving at least {MIN_PER_CLASS} of each class")
    add("")
    add(
        "`label_known` samples only. A reported threshold, not a choice: the "
        "window and mode decision stays with the user at the review gate."
    )
    add("")
    add(
        "| scenario | mode | smallest usable window s | attack there | benign there "
        "| windows tried |"
    )
    add("|---|---|---|---|---|---|")
    for entry in usable:
        window = entry["smallest_usable_window_seconds"]
        label = (
            f"**{window}**"
            if window is not None
            else f"**none of the lengths tried** (best {entry['best_window_seconds']} s)"
        )
        add(
            f"| {entry['scenario']} | {entry['sample_mode']} | {label} "
            f"| {entry['n_attack']:,} | {entry['n_benign']:,} "
            f"| {', '.join(str(w) for w in entry['windows_tried'])} |"
        )
    add("")

    add("## 8. Provenance")
    add("")
    add("| scenario | archive | archive size | entries fetched | fetch complete |")
    add("|---|---|---|---|---|")
    for scenario, raw in provenance.items():
        size = raw.get("archive_size_bytes")
        add(
            f"| {scenario} | `{raw.get('archive', 'not measured')}` "
            f"| {_number((size or 0) / 1e6)} MB | {raw.get('selected_entries', 'not measured')} "
            f"| {raw.get('complete', 'not measured')} |"
        )
    add("")
    add(
        "Raw and processed data are gitignored; the CRC-32 and SHA-256 in each "
        "`data/raw/ait/<scenario>/manifest.json` tie these numbers to specific "
        "bytes."
    )
    add("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_rows(
    scenarios: Iterable[str], windows: Iterable[int], progress: bool = True
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, float]]:
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    timings: dict[str, float] = {}
    for scenario in scenarios:
        provenance[scenario] = raw_manifest(load_ait_config(scenario=scenario))
        for window in windows:
            config = load_ait_config(scenario=scenario, window_seconds=window)
            started = time.time()
            scan = scan_scenario(config)
            elapsed = time.time() - started
            timings[f"{scenario}@{window}s"] = round(elapsed, 1)
            if progress:
                print(f"{scenario} {window}s: scanned in {elapsed:.1f}s", flush=True)
            for mode in MODES:
                row = describe(build_samples(scan, mode), scan)
                rows.append(row)
                if progress:
                    print(
                        f"  {mode}: {row['n_samples']:,} samples, "
                        f"{row['n_attack']:,} attack / {row['n_benign']:,} benign / "
                        f"{row['n_not_labelled']:,} unlabelled, "
                        f"{row['attack_runs']['count']} attack runs",
                        flush=True,
                    )
            del scan
    return rows, provenance, timings


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--windows", nargs="*", type=int, default=None)
    parser.add_argument("--out", default=str(OUT_DIR))
    args = parser.parse_args(argv)

    settings = load_ait_settings()
    raw_root = Path(str(settings.get("raw_root", "data/raw/ait")))
    wanted = list(args.scenarios or SCENARIOS)
    present = [s for s in wanted if (raw_root / s / "labels").is_dir()]
    missing = [s for s in wanted if s not in present]
    if not present:
        print(
            "STOP: no fetched scenario found under "
            f"{raw_root}. Run `python scripts/fetch_ait.py <scenario>` first."
        )
        return 2
    for scenario in missing:
        print(f"PENDING: {scenario} is not fetched; it is absent from this table.")

    windows = tuple(args.windows or WINDOWS)
    rows, provenance, timings = build_rows(present, windows)
    views = list(rows[0]["view_order"])
    identity = view_identity(rows)
    usable = smallest_usable_window(rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "part": 27,
        "statistics_only": True,
        "what_no_number_here_is": (
            "no training, no prediction, no fusion, no ENIV, no accuracy, no "
            "calibration: counts and fractions of counts only"
        ),
        "scenarios": present,
        "scenarios_pending": missing,
        "windows_seconds": list(windows),
        "sample_modes": list(MODES),
        "label_states": {"1": "attack", "0": "benign", "-1": "not labelled"},
        "attack_fraction_denominator": "label_known samples only",
        "dnsteal_family": sorted(DNSTEAL_FAMILY),
        "min_per_class_threshold": MIN_PER_CLASS,
        "scan_seconds": timings,
        "provenance": provenance,
        "view_identity": identity,
        "smallest_usable_window": usable,
        "rows": rows,
    }
    (out / "window_table.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(rows, views, out / "window_table.csv")
    write_markdown(rows, views, identity, usable, provenance, out / "window_table.md")
    print(f"wrote {out / 'window_table.json'}")
    print(f"wrote {out / 'window_table.csv'}")
    print(f"wrote {out / 'window_table.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
