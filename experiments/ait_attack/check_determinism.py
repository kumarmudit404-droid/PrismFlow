"""Part 29 step 0: is a Part 29 re-run bit-reproducible?

The 2026-10-13 run of this experiment died after three of its four units (see
`experiments/ait_attack/README.md`, "Retained run evidence"). The runner has no
resume path -- `main()` builds its results dict from scratch and loops every
(direction, system) unconditionally -- so the 2026-10-14 relaunch recomputed
those three units. That gives a free determinism check: the same commit, the
same config and the same seeds, run twice, hours apart.

This script compares the two files over the three SHARED units and writes
`results/ait_attack/determinism_check.json`. It reports a maximum absolute
difference per metric family AND an exhaustive leaf-by-leaf diff, so a
difference cannot hide in a field the families do not name. It rounds nothing.

Read-only with respect to both results files.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from experiments.ait_attack.results_io import load_results, resolve

#: Logical names, not paths: `load_results` prefers the raw `.json` on disk and
#: falls back to the committed `.json.gz`, verifying either against
#: `results_manifest.json` before it is used.
NEW = "results.json"
OLD = "results.3of4units_20261014_0022.json"
OUT = Path("results/ait_attack/determinism_check.json")

SHARED = (
    ("A_rm_to_santos", "naive"),
    ("A_rm_to_santos", "prismflow"),
    ("B_santos_to_rm", "naive"),
)

#: The families the Part 29 post-run brief asks for, by the per-seed field
#: each one is actually stored under in `results.json`.
FAMILIES = {
    "success_rate": ("attack_success",),
    "detector_auc": ("detector_auc",),
    "eniv": ("eniv_eigen", "eniv_design_effect"),
    "accuracy": ("accuracy",),
}


def family_diff(new: dict, old: dict) -> dict:
    """Max |difference| and differing-cell count, per family, over shared units."""
    out = {}
    for family, fields in FAMILIES.items():
        worst = 0.0
        worst_where = None
        cells = set()
        none_mismatch = 0
        compared = 0
        for dname, sname in SHARED:
            ncells = new["directions"][dname]["systems"][sname]["cells"]
            ocells = old["directions"][dname]["systems"][sname]["cells"]
            for key in sorted(set(ncells) & set(ocells)):
                nps, ops = ncells[key]["per_seed"], ocells[key]["per_seed"]
                if len(nps) != len(ops):
                    cells.add(f"{dname}/{sname}/{key}")
                    continue
                for field in fields:
                    for i, (a, b) in enumerate(zip(nps, ops)):
                        va, vb = a.get(field), b.get(field)
                        if va is None and vb is None:
                            continue
                        if (va is None) != (vb is None):
                            none_mismatch += 1
                            cells.add(f"{dname}/{sname}/{key}")
                            continue
                        compared += 1
                        delta = abs(float(va) - float(vb))
                        if delta > 0:
                            cells.add(f"{dname}/{sname}/{key}")
                        if delta > worst:
                            worst = delta
                            worst_where = f"{dname}/{sname} {key} seed{i} {field}"
        out[family] = {
            "fields": list(fields),
            "max_abs_diff": worst,
            "max_abs_diff_at": worst_where,
            "n_values_compared": compared,
            "n_cells_differing": len(cells),
            "cells_differing": sorted(cells),
            "none_vs_value_mismatches": none_mismatch,
        }
    return out


def exhaustive_diff(new: dict, old: dict) -> dict:
    """Every leaf under the shared units, not only the named families."""
    diffs: list[dict] = []
    counts = defaultdict(int)

    def walk(a, b, path: str) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    diffs.append({"path": f"{path}/{key}",
                                  "only_in": "new" if key in a else "old"})
                else:
                    walk(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                diffs.append({"path": path, "new": f"len {len(a)}", "old": f"len {len(b)}"})
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]")
        else:
            counts["leaves"] += 1
            numeric = (isinstance(a, (int, float)) and isinstance(b, (int, float))
                       and not isinstance(a, bool) and not isinstance(b, bool))
            if numeric:
                counts["numeric"] += 1
            if a != b:
                diffs.append({"path": path, "new": repr(a), "old": repr(b)})

    for dname, sname in SHARED:
        walk(new["directions"][dname]["systems"][sname],
             old["directions"][dname]["systems"][sname], f"{dname}/{sname}")
    # The A-direction paired block exists in both files and is derived from
    # both systems, so a difference there would catch an aggregation change
    # that the per-cell walk above could miss.
    walk(new["directions"]["A_rm_to_santos"].get("paired_prismflow_minus_naive"),
         old["directions"]["A_rm_to_santos"].get("paired_prismflow_minus_naive"),
         "A_rm_to_santos/paired_prismflow_minus_naive")

    return {
        "n_leaves_compared": counts["leaves"],
        "n_numeric_leaves_compared": counts["numeric"],
        "n_differences": len(diffs),
        "differences": diffs[:200],
        "differences_truncated": len(diffs) > 200,
    }


def main() -> None:
    new_path, old_path = resolve(NEW), resolve(OLD)
    if old_path is None:
        raise SystemExit(f"{OLD} is missing (neither .json nor .json.gz); it is the "
                         "retained 3-unit checkpoint")
    if new_path is None:
        raise SystemExit(f"{NEW} is missing (neither .json nor .json.gz); "
                         "run the experiment")
    new = load_results(NEW)
    old = load_results(OLD)

    report = {
        "question": ("Do the three units shared by the died 2026-10-13 run and the "
                     "completed 2026-10-14 run agree, given the runner has no resume "
                     "and recomputed them from scratch?"),
        "new_file": str(new_path), "old_file": str(old_path),
        "new_commit": new.get("commit"), "old_commit": old.get("commit"),
        "new_seeds": new.get("seeds"), "old_seeds": old.get("seeds"),
        "shared_units": [f"{d}/{s}" for d, s in SHARED],
        "unit_present_only_in_new": [
            f"{d}/{s}" for d in new["directions"]
            for s in new["directions"][d]["systems"]
            if (d, s) not in SHARED
        ],
        "per_family": family_diff(new, old),
        "exhaustive": exhaustive_diff(new, old),
    }
    report["bit_identical_on_shared_units"] = (
        report["exhaustive"]["n_differences"] == 0
        and all(f["max_abs_diff"] == 0 and f["n_cells_differing"] == 0
                for f in report["per_family"].values())
    )
    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}")
    print(f"bit_identical_on_shared_units: {report['bit_identical_on_shared_units']}")
    for name, f in report["per_family"].items():
        print(f"  {name:14} max|diff|={f['max_abs_diff']!r} "
              f"cells_differing={f['n_cells_differing']} "
              f"values_compared={f['n_values_compared']}")
    ex = report["exhaustive"]
    print(f"  exhaustive: {ex['n_differences']} differences over "
          f"{ex['n_leaves_compared']} leaves ({ex['n_numeric_leaves_compared']} numeric)")


if __name__ == "__main__":
    main()
