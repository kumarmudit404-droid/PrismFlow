"""Fill ANALYSIS_TEMPLATE.md's Tables 1, 2, 3 and 6 from the committed sweep.

DO NOT RUN UNTIL THE SWEEP IS 40/40. The gate below refuses to run on a
partial sweep and says so, because a mean over a partial cell is not a
finding (CLAUDE.md, 5-SEED RULE) and a table built from one would have to be
thrown away rather than topped up.

NO TRAINING, NO INFERENCE, NO MODEL. Tables 1, 2 and 6 read the committed
per-unit JSON fields. Table 3 re-scores the committed per-row arrays in
`results/lanl_v2/scores_default/*.npz` with a host mask applied, which is
arithmetic over saved probabilities -- the models are never rebuilt and are
not needed.

WHAT IS NOT HERE, and why

- **Table 4 (L24)** is already measured and committed by three separate
  scalar passes: `h1_auth_presence.py`, `h1_occupancy_decomposition.py`,
  `h1_conditional_rates.py`. It does not depend on the sweep and is not
  recomputed here.
- **Table 5 (L28)** needs execution mode A, which has not been run. It stays
  `pending mode A`. Pre-registration section 6 forbids pooling the two modes
  into a single +/-, so this script never aggregates across modes and only
  ever reads `units_default/`, which is mode `default` (mode B) alone.

INTEGRITY CHECKS, all fatal

1. Exactly 40 unit JSONs and 40 score npz, as matched pairs.
2. Every unit's `scores_sha256` matches the sha256 of its npz on disk.
3. No npz carries a key named `vacuity` (L29: the column is a CONFIDENCE and
   this runner writes `confidence`; the inverted name is the defect, in a
   frozen file, and must not reappear here).
4. Every unit's `mode.mode` is `default` and its observed thread count agrees
   with every other unit's. A figure is conditional on a thread count.
5. Table 3's recomputed all-test ROC AUC matches the committed `roc_auc`
   field. This is the check that the re-score path and the sweep's own metric
   path agree before any host is removed; without it, a leave-host-out delta
   could be an artefact of two different AUC implementations.

ROC AUC here comes from the frozen `experiments/lanl/run_lanl.py`'s
`roc_auc_binary`, the same function the Part 38 scalar passes use and the same
one Part 30's committed `step5_l27` deltas were computed with, so Table 3's
deltas are comparable to Part 30's cell for cell.

    PYTHONPATH=. ./.venv/Scripts/python.exe experiments/lanl_v2/fill_analysis.py
"""

from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from experiments.lanl.run_lanl import roc_auc_binary                # noqa: E402

OUT = ROOT / "results" / "lanl_v2"
UNITS = OUT / "units_default"
SCORES = OUT / "scores_default"

ARMS = ("without_auth", "without_flows", "without_dns", "with_auth")
SYSTEMS = ("naive", "prismflow")
SEEDS = (0, 1, 2, 3, 4)
EXPECTED_UNITS = len(ARMS) * len(SYSTEMS) * len(SEEDS)
MODE = "default"

#: Part 30's committed rule, results/lanl_reanalysis/results.json,
#: step5_l27.host_rule -- quoted so the re-derivation is auditable.
HOST_RULE = ("The 5 computers with the most TEST rows of label attack, ranked "
             "by that count, ties broken by computer name ascending.")
N_HOSTS = 5
#: Part 30's committed per-cell C17693 deltas, READ FROM THE FILE rather than
#: transcribed: a hardcoded copy can drift from the source and a rounded copy
#: makes the pairwise comparison wrong in the last digits that matter here,
#: given H2's bar sits inside Part 30's own band.
PART30_L27 = ROOT / "results" / "lanl_reanalysis" / "results.json"
PART30_HOST = "C17693"
#: PRE_REGISTRATION.md section 1, H2.
H2_BAR = 0.036
#: Metrics carried into Table 1 beside the primary. ROC AUC is primary;
#: balanced accuracy and attack recall are SECONDARY (Amendment 1, 10.1).
TABLE1_FIELDS = ("roc_auc", "balanced_accuracy", "attack_recall",
                 "benign_recall", "brier", "prob_ece",
                 "eniv_eigen", "mean_dependence", "efficiency_ratio")
#: `peak_rss_mb` is DELIBERATELY ABSENT: it is a per-launch high-water mark,
#: not a per-unit measurement, so `agg()` must not mean it across seeds. It is
#: emitted per seed by `table6()` instead. See fill_analysis_patch.py.
TABLE6_FIELDS = ("unit_seconds", "rows_at_or_above_threshold",
                 "max_prob_attack", "all_false_mask_rows")

PEAK_RSS_SEMANTIC = (
    "peak_rss_mb is a RUNNING MAXIMUM over the life of the process that wrote "
    "the unit, not a per-unit measurement. Within one launch it is "
    "non-decreasing, so a unit's figure includes memory peaks caused by "
    "EARLIER units in the same process: measured 499.5 then 528.8 for "
    "with_auth naive_seed4 then prismflow_seed2 in one process on 2026-11-22. "
    "Reported per seed and never averaged -- units written by one long-lived "
    "process can all carry that process's plateau, so a mean over them counts "
    "one observation n times and its std is 0.0 by construction rather than "
    "by reproducibility. n_distinct counts the distinct values in this cell, "
    "which is a lower bound on the launches that wrote it and not a fixed "
    "number of regimes.")

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
    "it is not a reproducibility figure.")


def unit_name(arm: str, system: str, seed: int) -> str:
    return f"{arm}_{system}_seed{seed}"


def part30_c17693() -> dict[tuple[str, str], float]:
    """Part 30's eight committed C17693 deltas, at full precision."""
    per_arm = json.loads(PART30_L27.read_text())["step5_l27"]["per_arm"]
    out = {}
    for arm, systems in per_arm.items():
        for system, cell in systems.items():
            host = cell["per_host"].get(PART30_HOST)
            if host is not None:
                out[(arm, system)] = host["change_from_all_test"]
    return out


def gate(dry_run: bool = False) -> tuple[dict[str, dict], dict]:
    """Refuse to run on a partial sweep, unless this is an explicit dry run.

    Returns the loaded units and the cell map {(arm, system): [seeds]} the
    tables iterate. In a real run that map is always the full 8 x 5; in a dry
    run it is whatever is on disk, which is why every dry-run figure is
    flagged and written to a separate filename.

    Every other check -- sha256, mode, thread agreement, json/npz pairing --
    applies in BOTH modes. A dry run relaxes the seed count, nothing else.
    """
    present = sorted(p.stem for p in UNITS.glob("*.json"))
    wanted = [unit_name(a, s, d) for a in ARMS for s in SYSTEMS for d in SEEDS]
    missing = [n for n in wanted if n not in set(present)]
    extra = [n for n in present if n not in set(wanted)]
    if (missing or len(present) != EXPECTED_UNITS) and not dry_run:
        raise SystemExit(
            "sweep incomplete, do not run yet: %d of %d units present, %d "
            "missing.\n  missing: %s%s"
            % (len(present), EXPECTED_UNITS, len(missing),
               ", ".join(missing[:8]),
               " ..." if len(missing) > 8 else ""))
    if extra:
        raise SystemExit(
            "unexpected unit files, refusing to aggregate: %s"
            % ", ".join(extra))

    units: dict[str, dict] = {}
    for name in [n for n in wanted if n in set(present)]:
        row = json.loads((UNITS / f"{name}.json").read_text())
        npz = SCORES / f"{name}.npz"
        if not npz.exists():
            raise SystemExit(
                f"sweep incomplete, do not run yet: {name}.json exists but "
                f"{npz.name} does not")
        got = hashlib.sha256(npz.read_bytes()).hexdigest()
        if got != row["scores_sha256"]:
            raise SystemExit(
                f"{npz.name} does not match the sha256 its unit JSON "
                f"records.\n  recorded: {row['scores_sha256']}\n  on disk:  "
                f"{got}\nRefusing to aggregate scores that have changed since "
                "they were written.")
        if row["mode"]["mode"] != MODE:
            raise SystemExit(
                f"{name} was run in mode {row['mode']['mode']!r}, not "
                f"{MODE!r}. Pre-registration section 6 forbids pooling modes.")
        units[name] = row

    threads = {u["mode"]["torch_threads_observed"] for u in units.values()}
    if len(threads) != 1:
        raise SystemExit(
            "units disagree on the observed thread count (%s). Every figure "
            "is conditional on a thread count and ENIV reproduces only at the "
            "count it was computed at; refusing to pool." % sorted(threads))

    cells = {}
    for arm in ARMS:
        for system in SYSTEMS:
            seeds = [d for d in SEEDS if unit_name(arm, system, d) in units]
            if seeds:
                cells[(arm, system)] = seeds
    return units, cells


def agg(values: list[float | None]) -> dict:
    """mean +/- sample std (ddof=1), the rule aggregate_sweep declares.

    A field that is None for every seed -- `eniv_eigen` on the naive system,
    where `output.eniv` is None because `use_discount` is false -- is reported
    as not measured rather than as 0.
    """
    kept = [v for v in values if v is not None]
    if not kept:
        return {"mean": None, "std": None, "n": 0,
                "status": "not measured: null for every seed in this cell"}
    return {
        "mean": float(st.fmean(kept)),
        "std": float(st.stdev(kept)) if len(kept) > 1 else None,
        "n": len(kept),
        "per_seed": [None if v is None else float(v) for v in values],
    }


def load_scores(name: str) -> dict:
    with np.load(SCORES / f"{name}.npz", allow_pickle=True) as z:
        if "vacuity" in z.files:
            raise SystemExit(
                f"{name}.npz carries a key named 'vacuity' (L29). This "
                "runner writes 'confidence'; refusing to read a file that "
                "reintroduced the inverted name.")
        return {
            "computer": np.asarray([str(c) for c in z["computer"]], dtype=object),
            "label": z["label"].astype(int),
            "prob_attack": z["prob_attack"].astype(np.float64),
        }


def table1(units, cells) -> dict:
    out = {
        "what": "ROC AUC and companions, mean +/- sample std (ddof=1) over the "
                "5 seeds, per arm x system. Mode `default` (B) only.",
        "primary_metric": "roc_auc",
        "secondary_note": "balanced_accuracy and attack_recall are SECONDARY "
                          "(Amendment 1, section 10.1). No hypothesis is "
                          "decided on a threshold metric.",
        "per_cell": {},
    }
    for (arm, system), seeds in cells.items():
        cell = {}
        for field in TABLE1_FIELDS:
            cell[field] = agg([units[unit_name(arm, system, d)].get(field)
                               for d in seeds])
        cell["n_seeds"] = len(seeds)
        cell["seeds"] = list(seeds)
        if len(seeds) < len(SEEDS):
            cell["NOT_A_FINDING"] = (
                "n = %d of %d seeds. A mean over a partial cell is not a "
                "finding (5-SEED RULE)." % (len(seeds), len(SEEDS)))
        out["per_cell"][f"{arm}/{system}"] = cell
    return out


def table2(units, cells) -> dict:
    """Per-seed ROC AUC, and the per-seed SIGN of PrismFlow - naive.

    Signs before means: a cancelling population reads as "no effect" if only
    the mean is shown, and an exact zero has no sign and is counted
    separately rather than folded into either side.
    """
    out = {
        "what": "Per-seed test ROC AUC, 40 cells, and the per-seed paired "
                "difference. Signs are reported BEFORE the aggregate mean.",
        "per_cell": {}, "prismflow_minus_naive": {},
    }
    for (arm, system), seeds in cells.items():
        out["per_cell"][f"{arm}/{system}"] = {
            f"seed{d}": units[unit_name(arm, system, d)]["roc_auc"]
            for d in seeds}
    for arm in ARMS:
        # A paired difference needs the SAME seed on both systems.
        paired = sorted(set(cells.get((arm, "prismflow"), []))
                        & set(cells.get((arm, "naive"), [])))
        if not paired:
            out["prismflow_minus_naive"][arm] = {
                "status": "not measured: no seed is complete on both systems"}
            continue
        diffs = [units[unit_name(arm, "prismflow", d)]["roc_auc"]
                 - units[unit_name(arm, "naive", d)]["roc_auc"] for d in paired]
        out["prismflow_minus_naive"][arm] = {
            "seeds_paired": paired,
            "per_seed": diffs,
            "seeds_positive": sum(d > 0 for d in diffs),
            "seeds_negative": sum(d < 0 for d in diffs),
            "seeds_exactly_zero": sum(d == 0 for d in diffs),
            "mean": float(st.fmean(diffs)),
            "std": float(st.stdev(diffs)) if len(diffs) > 1 else None,
            "sign_note": "A difference smaller than the per-seed std is "
                         "reported as sign undetermined, not as agreement.",
        }
    return out


def derive_hosts(units) -> dict:
    """Part 30's host rule, re-derived on PART 38's test block.

    Not copied: section 2's rule (b) keeps all benign cells of attack hosts
    (at the realised 1/4 share), so the removed-row counts differ from Part
    30's by design, and the five computers need not be the same five.
    """
    scores = load_scores(next(iter(units)))
    attack = scores["label"] == 1
    names, counts = np.unique(scores["computer"][attack], return_counts=True)
    order = sorted(zip(names, counts), key=lambda kv: (-kv[1], kv[0]))
    chosen = [str(n) for n, _ in order[:N_HOSTS]]
    detail = []
    for host in chosen:
        m = scores["computer"] == host
        detail.append({
            "computer": host,
            "test_rows_removed": int(m.sum()),
            "test_attack_rows_removed": int((m & attack).sum()),
            "test_benign_rows_removed": int((m & ~attack).sum()),
        })
    return {
        "host_rule": HOST_RULE,
        "re_derived_on": "Part 38's own test block, not copied from Part 30",
        "hosts": detail,
        "part30_hosts": ["C17693", "C1015", "C2388", "C1936", "C2079"],
        "same_five_as_part30": chosen == ["C17693", "C1015", "C2388",
                                          "C1936", "C2079"],
        "n_attack_hosts_in_test": int(names.size),
    }


def table3(units, cells, hosts) -> dict:
    part30 = part30_c17693()
    out = {
        "what": "L27 leave-host-out test ROC AUC delta, 8 arm x system rows x "
                "%d hosts. Re-scored from the committed per-row arrays; no "
                "model is rebuilt." % N_HOSTS,
        "interval": "NONE. One host is not a resample, so no interval is "
                    "attached and the result is descriptive. Part 30 records "
                    "the same.",
        "pre_registration_status": (
            "Only the C17693 column carries H2's bar. The other hosts are NOT "
            "pre-registered for Part 38 and are a descriptive mirror of Part "
            "30's committed 40-cell table; they cannot confirm or disconfirm "
            "H2 and must not be aggregated with the C17693 column."),
        "h2_bar": H2_BAR,
        "part30_c17693_source": "results/lanl_reanalysis/results.json, step5_l27",
        "hosts": hosts,
        "per_cell": {},
    }
    host_names = [h["computer"] for h in hosts["hosts"]]
    for (arm, system), seeds in cells.items():
        per_host: dict[str, dict] = {}
        all_test_per_seed = []
        for seed in seeds:
            name = unit_name(arm, system, seed)
            s = load_scores(name)
            full = roc_auc_binary(s["label"], s["prob_attack"])
            committed = units[name]["roc_auc"]
            if not np.isclose(full, committed, rtol=0, atol=1e-9):
                raise SystemExit(
                    f"{name}: re-scored all-test ROC AUC {full!r} does "
                    f"not match the committed {committed!r}. A "
                    "leave-host-out delta computed against a different "
                    "AUC implementation would not be comparable to Part "
                    "30's. Stopping.")
            all_test_per_seed.append(float(full))
            for host in host_names:
                keep = s["computer"] != host
                lab = s["label"][keep]
                if len(np.unique(lab)) != 2:
                    per_host.setdefault(host, {"per_seed": []})
                    per_host[host]["per_seed"].append(None)
                    per_host[host]["status"] = (
                        "not measured: removing this host leaves the test "
                        "block one-class, so no AUC is defined")
                    continue
                auc = roc_auc_binary(lab, s["prob_attack"][keep])
                per_host.setdefault(host, {"per_seed": []})
                per_host[host]["per_seed"].append(float(auc))

        all_test = agg(all_test_per_seed)
        cell = {"roc_auc_all_test": all_test, "per_host": {}}
        for host in host_names:
            removed = agg(per_host[host]["per_seed"])
            entry = {"roc_auc_host_removed": removed}
            if removed["mean"] is not None and all_test["mean"] is not None:
                entry["change_from_all_test"] = (
                    removed["mean"] - all_test["mean"])
            else:
                entry["change_from_all_test"] = None
                entry["status"] = per_host[host].get(
                    "status", "not measured")
            if host == "C17693" and entry["change_from_all_test"] is not None:
                delta = abs(entry["change_from_all_test"])
                p30 = part30.get((arm, system))
                entry["barred"] = True
                entry["h2_abs_delta"] = delta
                entry["h2_cell_passes"] = delta < H2_BAR
                entry["part30_change_from_all_test"] = p30
                entry["vs_part30"] = (
                    None if p30 is None else delta - abs(p30))
            else:
                entry["barred"] = False
            cell["per_host"][host] = entry
        out["per_cell"][f"{arm}/{system}"] = cell

    barred = [(c, v["per_host"]["C17693"]) for c, v in out["per_cell"].items()
              if "C17693" in v["per_host"]]
    measured = [(c, e) for c, e in barred if e.get("h2_abs_delta") is not None]
    if len(measured) != len(ARMS) * len(SYSTEMS):
        out["h2_verdict"] = {
            "verdict": "NOT MEASURED",
            "why": "C17693 is not among Part 38's top-%d attack hosts, or its "
                   "delta is undefined in some cell. H2 names that host "
                   "explicitly; a different host cannot stand in for it."
                   % N_HOSTS,
            "cells_measured": len(measured),
        }
    else:
        failing = [c for c, e in measured if not e["h2_cell_passes"]]
        out["h2_verdict"] = {
            "verdict": "CONFIRMED" if not failing else "DISCONFIRMED",
            "rule": "H2 is disconfirmed if ANY of the eight arm-and-system "
                    "cells moves by %.3f or more in absolute value." % H2_BAR,
            "cells_at_or_above_bar": failing,
            "max_abs_delta": max(e["h2_abs_delta"] for _, e in measured),
            "min_abs_delta": min(e["h2_abs_delta"] for _, e in measured),
            "sharpness_warning": (
                "Part 30's own committed C17693 band is 0.035963 to 0.051875 "
                "over the same eight cells, and its smallest cell "
                "(without_dns/naive, -0.035963) is BELOW this 0.036 bar. A "
                "Part 38 cell reproducing Part 30's weakest cell would be "
                "scored as passing, so a CONFIRMED verdict does not by itself "
                "mean the sensitivity shrank. Read the per-cell vs_part30 "
                "column."),
            "realised_share_caveat": (
                "The k=2 size fallback fired, so the realised attack-host "
                "benign share is 1/4, not 1/1. A disconfirmation shows that "
                "restoring a QUARTER of the benign contrast did not help, "
                "which is narrower than H2's mechanism."),
        }
    return out


def table6(units, cells) -> dict:
    out = {
        "what": "Operational record and the Amendment 10.2 threshold-degeneracy "
                "diagnostic. No hypothesis is attached to any of it.",
        "peak_rss_mb_semantic": PEAK_RSS_SEMANTIC,
        "peak_rss_mb_launch_identity": PEAK_RSS_LAUNCH_ID,
        "unit_seconds_semantic": UNIT_SECONDS_SEMANTIC,
        "degeneracy_definition": (
            "A unit whose max prob_attack is below 0.5 is degenerate at the "
            "default threshold, which pins balanced accuracy at exactly 0.5 "
            "whatever the ranking underneath it."),
        "per_cell": {},
    }
    for (arm, system), seeds in cells.items():
        names = [unit_name(arm, system, d) for d in seeds]
        cell = {f: agg([units[n].get(f) for n in names])
                for f in TABLE6_FIELDS}
        peaks = [units[n].get("peak_rss_mb") for n in names]
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
        cell["degenerate_at_threshold_per_seed"] = [
            units[n]["max_prob_attack"] < 0.5 for n in names]
        cell["degenerate_any_seed"] = any(
            cell["degenerate_at_threshold_per_seed"])
        out["per_cell"][f"{arm}/{system}"] = cell
    return out


DRY_RUN_FLAG = "--dry-run"
DRY_RUN_STAMP = "DRY_RUN_NOT_A_FINDING"


def output_path(dry_run: bool) -> Path:
    """The dry-run filename carries `_dryrun`, so the two can never be confused.

    Asserted rather than trusted: a dry run that overwrote
    `filled_tables.json` would put partial-cell means where the real tables
    go, which is exactly the mistake the 5-SEED RULE exists to prevent.
    """
    path = OUT / ("filled_tables_dryrun.json" if dry_run
                  else "filled_tables.json")
    if dry_run and "_dryrun" not in path.name:
        raise AssertionError("dry-run output filename must contain '_dryrun'")
    if not dry_run and "_dryrun" in path.name:
        raise AssertionError("a real run must not write a _dryrun filename")
    return path


def main(*argv: str) -> int:
    dry_run = DRY_RUN_FLAG in argv
    unknown = [a for a in argv if a != DRY_RUN_FLAG]
    if unknown:
        raise SystemExit("unknown argument(s): %s. Only %s is accepted."
                         % (", ".join(unknown), DRY_RUN_FLAG))
    units, cells = gate(dry_run=dry_run)
    if dry_run:
        assert DRY_RUN_FLAG in argv, (
            "dry-run output may only be produced under the explicit "
            f"{DRY_RUN_FLAG} flag")
        if len(units) == EXPECTED_UNITS:
            print("note: the sweep is complete (%d units). A dry run is still "
                  "a dry run -- run without %s for the real tables."
                  % (EXPECTED_UNITS, DRY_RUN_FLAG))
    hosts = derive_hosts(units)
    filled = {
        "part": 38,
        "mode": MODE,
        "mode_note": (
            "Mode `default` is pre-registration section 6's mode B "
            "(machine-default threads, sequential). Mode A has not been run. "
            "The two modes are NEVER pooled into a single +/-."),
        "torch_threads_observed": next(iter(units.values()))[
            "mode"]["torch_threads_observed"],
        "n_units": len(units),
        "aggregation_rule": (
            "mean +/- sample std (ddof=1) over the 5 seeds, within one "
            "execution mode. A seed varies parameter initialisation and "
            "minibatch order only, on a fixed sample, so every +/- is "
            "initialisation-and-batch-order variance."),
        "table4_elsewhere": (
            "L24 is measured by results/lanl_v2/h1_auth_presence.json, "
            "h1_occupancy_decomposition.json and h1_conditional_rates.json. "
            "It does not depend on the sweep and is not recomputed here."),
        "table5_pending": "pending mode A",
        "table1_roc_auc_per_cell": table1(units, cells),
        "table2_per_seed": table2(units, cells),
        "table3_leave_host_out": table3(units, cells, hosts),
        "table6_operational": table6(units, cells),
    }
    if dry_run:
        filled = {
            "status": DRY_RUN_STAMP,
            "what_this_is": (
                "SCHEMA VERIFICATION ONLY. Every figure below is computed "
                "over the units that happened to be on disk mid-sweep, so "
                "cells have fewer than 5 seeds and NO NUMBER HERE IS A "
                "FINDING. It exists to exercise the table code before the "
                "sweep lands, nothing else. Do not cite it, do not commit it."),
            "n_units_present": len(units),
            "n_units_expected": EXPECTED_UNITS,
            "cells_complete": sorted(f"{a}/{s}" for (a, s), d in cells.items()
                                     if len(d) == len(SEEDS)),
            "cells_partial": sorted(f"{a}/{s}" for (a, s), d in cells.items()
                                    if len(d) < len(SEEDS)),
            **{k: v for k, v in filled.items()},
        }
    path = output_path(dry_run)
    path.write_text(json.dumps(filled, indent=1))

    t1 = filled["table1_roc_auc_per_cell"]["per_cell"]
    print("Filled from %d units, mode %s, %d threads observed"
          % (len(units), MODE, filled["torch_threads_observed"]))
    print("\nTable 1 -- ROC AUC per arm x system")
    print("%-28s %6s %14s %12s" % ("arm/system", "n", "mean", "std"))
    for cell, v in t1.items():
        r = v["roc_auc"]
        print("%-28s %6d %14.10f %12s%s"
              % (cell, r["n"], r["mean"],
                 "n/a" if r["std"] is None else "%.10f" % r["std"],
                 "   " + DRY_RUN_STAMP if "NOT_A_FINDING" in v else ""))
    print("\nTable 2 -- per-seed sign of PrismFlow - naive")
    for arm, v in filled["table2_per_seed"]["prismflow_minus_naive"].items():
        if "status" in v:
            print("  %-16s %s" % (arm, v["status"]))
            continue
        print("  %-16s +%d/-%d/0:%d  mean %+.10f  std %s  (seeds %s)"
              % (arm, v["seeds_positive"], v["seeds_negative"],
                 v["seeds_exactly_zero"], v["mean"],
                 "n/a" if v["std"] is None else "%.10f" % v["std"],
                 v["seeds_paired"]))
    t3 = filled["table3_leave_host_out"]
    print("\nTable 3 -- hosts re-derived on Part 38's test block "
          "(same five as Part 30: %s)" % t3["hosts"]["same_five_as_part30"])
    for h in t3["hosts"]["hosts"]:
        print("  %-10s %6d rows  %5d attack  %5d benign"
              % (h["computer"], h["test_rows_removed"],
                 h["test_attack_rows_removed"], h["test_benign_rows_removed"]))
    print("\n  H2 verdict: %s" % t3["h2_verdict"]["verdict"])
    for cell, v in t3["per_cell"].items():
        e = v["per_host"].get("C17693")
        if e and e.get("h2_abs_delta") is not None:
            print("    %-28s delta %+.10f  abs %.10f  passes %s  "
                  "(Part 30 %+.6f)"
                  % (cell, e["change_from_all_test"], e["h2_abs_delta"],
                     e["h2_cell_passes"], e["part30_change_from_all_test"]))
    # path.name, never a literal: the first draft hardcoded
    # "filled_tables.json" here and reported that name after correctly
    # writing filled_tables_dryrun.json, which is worse than a wrong
    # filename -- it is a correct file with a lying receipt.
    print("\nwrote %s" % path.relative_to(ROOT).as_posix())
    print("Table 4 is elsewhere (three scalar passes); Table 5 pending mode A.")
    if dry_run:
        print("\n%s -- %d of %d units. Complete cells: %s. Partial cells: %s."
              % (DRY_RUN_STAMP, len(units), EXPECTED_UNITS,
                 ", ".join(filled["cells_complete"]) or "none",
                 ", ".join(filled["cells_partial"]) or "none"))
        print("No number above is a finding. Re-run without %s once the "
              "sweep reports %d/%d." % (DRY_RUN_FLAG, EXPECTED_UNITS,
                                        EXPECTED_UNITS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
