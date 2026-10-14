"""Part 29 step 2: distinct detector scores, tie fraction and detector AUC on
the PrismFlow arm.

WHY THIS SCRIPT EXISTS AT ALL
-----------------------------
`run_ait_attack.py` computes the per-window suspicion score vector for every
condition, uses it for `detector_auc` and the budgeted detection rates, and then
DISCARDS it: `record()` strips every key beginning with `_`, including
`_scores`, before serialisation (`run_ait_attack.py:860`). So `results.json`
carries the AUC but not the score distribution behind it, and the number of
distinct scores and the fraction of tied scores cannot be read back from it.

Recomputing all 187 conditions x 5 seeds x 2 directions costs about 3.3 hours at
the measured 6.3 s per cell. This script therefore measures a BOUNDED, LABELLED
SUBSET at full 5-seed depth rather than a single seed over everything: a
single-seed number would not be evidence under the contract's 5-SEED RULE.

SCOPE -- read before quoting anything from the output
-----------------------------------------------------
Conditions covered: the 25 enumerated view subsets at chorus epsilon 1.0
beta 1.0, plus `clean` and both missingness rates = 28 of 187 conditions.
Seeds: all 5. Directions: both. Every other condition's distinct-score count and
tie fraction is NOT MEASURED, and `detector_auc` for all 187 is already in
`results.json` and should be read from there, not from here.

CROSS-CHECK
-----------
Determinism was verified bit-exact over the three units shared by the died and
completed runs (`check_determinism.py`,
`results/ait_attack/determinism_check.json`), so a `detector_auc` recomputed
here must EQUAL the committed value for the same condition and seed. That
equality is asserted per condition and recorded as
`matches_committed_exactly`; it is a check on this script, not an assumption.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from experiments.ait_attack.run_ait_attack import (CONFIG_PATH, cell_key, detector_auc,
                                                   enumerate_subsets, prepare_direction,
                                                   run_cell, train_ait)
from prismflow.models.defended import DefendedPrismFlow

COMMITTED = Path("results/ait_attack/results.json")
OUT = Path("results/ait_attack/detector_scores_subset.json")
EPSILON, BETA = 1.0, 1.0


def score_stats(scores) -> dict:
    """Distinct-score count and tie fraction for one score vector.

    `tie_fraction` is the fraction of FINITE scores whose value occurs more than
    once, so a vector of all-distinct scores reads 0.0 and a constant vector
    reads 1.0. Non-finite scores are counted and excluded, never imputed.
    """
    scores = np.asarray(scores, dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        return {"n_windows": int(scores.size), "n_finite": 0, "n_distinct": 0,
                "tie_fraction": None, "largest_tie_group": None,
                "status": "not measured: no finite scores"}
    counts = Counter(finite.tolist())
    tied = sum(c for c in counts.values() if c > 1)
    return {
        "n_windows": int(scores.size),
        "n_finite": int(finite.size),
        "n_distinct": int(len(counts)),
        "tie_fraction": float(tied / finite.size),
        "largest_tie_group": int(max(counts.values())),
        "min": float(finite.min()),
        "max": float(finite.max()),
    }


def aggregate(rows: list[dict], committed_cell: dict | None) -> dict:
    """5-seed mean and population std, plus the committed-AUC cross-check."""
    aucs = np.array([r["detector_auc"] for r in rows], dtype=np.float64)
    distinct = np.array([r["n_distinct"] for r in rows], dtype=np.float64)
    ties = np.array([r["tie_fraction"] for r in rows], dtype=np.float64)
    entry = {
        "n_seeds": len(rows),
        "n_windows": rows[0]["n_windows"],
        "n_distinct_mean": float(distinct.mean()),
        "n_distinct_std": float(distinct.std(ddof=0)),
        "n_distinct_min": int(distinct.min()),
        "n_distinct_max": int(distinct.max()),
        "tie_fraction_mean": float(ties.mean()),
        "tie_fraction_std": float(ties.std(ddof=0)),
        "largest_tie_group_max": int(max(r["largest_tie_group"] for r in rows)),
        "detector_auc_mean": float(aucs.mean()),
        "detector_auc_std": float(aucs.std(ddof=0)),
        "dns_compromised": rows[0].get("dns_compromised"),
        "per_seed": rows,
    }
    if committed_cell is not None:
        committed = [ps.get("detector_auc") for ps in committed_cell["per_seed"]]
        if all(c is not None for c in committed):
            live = [r["detector_auc"] for r in rows]
            entry["committed_auc_per_seed"] = committed
            entry["matches_committed_exactly"] = bool(
                all(a == b for a, b in zip(live, committed)))
            entry["max_abs_auc_diff_vs_committed"] = float(
                max(abs(a - b) for a, b in zip(live, committed)))
    return entry


def main() -> None:
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    committed = json.loads(COMMITTED.read_text())
    out = {
        "scope": ("BOUNDED SUBSET: the 25 enumerated view subsets at chorus "
                  f"epsilon {EPSILON} beta {BETA}, plus clean and both missingness "
                  "rates = 28 of 187 conditions. All 5 seeds, both directions. "
                  "Every other condition's distinct-score count and tie fraction "
                  "is NOT MEASURED here; detector_auc for all 187 conditions is in "
                  "results/ait_attack/results.json."),
        "why_recomputed": ("run_ait_attack.py:860 strips _scores before "
                           "serialisation, so the score distribution is absent "
                           "from results.json."),
        "seeds": cfg["seeds"],
        "system": "prismflow (use_discount=True; the naive arm has no detector)",
        "tie_fraction_definition": ("fraction of finite scores whose value occurs "
                                    "more than once"),
        "directions": {},
    }

    for direction in cfg["directions"]:
        dname = direction["name"]
        prepared = prepare_direction(direction, cfg, {})
        sources = list(prepared["test"].sources)
        subsets = enumerate_subsets(sources, cfg["attack"]["k_grid"])
        per_condition: dict[str, list[dict]] = {}
        started = time.time()

        for seed in cfg["seeds"]:
            model = train_ait(prepared["train"], prepared["train_index"], seed, True, cfg)
            defended = DefendedPrismFlow(
                model, suspicion_permutations=cfg["detector"]["permutations"],
                suspicion_seed=seed,
            )
            clean = run_cell(defended, prepared["test"], prepared["test_index"],
                             cfg, kind="clean", seed=seed)
            clean_scores = clean["_scores"]
            # 0.5 by construction: the clean condition scored against itself.
            per_condition.setdefault("clean", []).append(
                {**score_stats(clean_scores), "detector_auc": 0.5, "seed": seed})

            for entry in subsets:
                cell = run_cell(defended, prepared["test"], prepared["test_index"], cfg,
                                kind="chorus", subset=entry["subset"],
                                epsilon=EPSILON, beta=BETA, seed=seed)
                per_condition.setdefault(cell_key("chorus", entry, EPSILON, BETA), []).append({
                    **score_stats(cell["_scores"]),
                    "detector_auc": float(detector_auc(cell["_scores"], clean_scores)),
                    "seed": seed,
                    "dns_compromised": entry["dns_compromised"],
                })

            for rate in cfg["missingness"]["rates"]:
                cell = run_cell(defended, prepared["test"], prepared["test_index"], cfg,
                                kind="missing", seed=seed, missing_rate=rate)
                per_condition.setdefault(f"missing_{int(round(rate * 100))}", []).append({
                    **score_stats(cell["_scores"]),
                    "detector_auc": float(detector_auc(cell["_scores"], clean_scores)),
                    "seed": seed,
                })
            print(f"{dname} seed {seed} done ({time.time() - started:.0f}s)", flush=True)

        cells = committed["directions"][dname]["systems"]["prismflow"]["cells"]
        out["directions"][dname] = {
            key: aggregate(rows, cells.get(key)) for key, rows in per_condition.items()
        }
        print(f"{dname} complete in {time.time() - started:.0f}s", flush=True)

    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT}")
    mismatched = [
        f"{d}/{k}" for d, conds in out["directions"].items() for k, v in conds.items()
        if v.get("matches_committed_exactly") is False
    ]
    print(f"conditions whose AUC disagrees with results.json: {len(mismatched)}")
    for name in mismatched:
        print(f"  MISMATCH {name}")


if __name__ == "__main__":
    main()
