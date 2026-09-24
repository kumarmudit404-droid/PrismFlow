"""Planted-redundancy experiment for the Part 22 ENIV discount.

    .venv\\Scripts\\python.exe experiments/v2/test_eniv_redundancy.py

This one IS evidence, and it runs under the 5-SEED RULE: 20 queries x 5
conditions x 5 seeds, reporting mean and standard deviation over seed means.
Its matrices are synthetic by design -- a planted-redundancy test works because
the true structure is known before measuring. What is measured is the
AGGREGATION, and that measurement is a finding.


THE SEED LOOP, AGAIN
--------------------
The brief's experiment calls ``np.random.seed(seed)`` and then builds three
hardcoded matrices -- ``np.eye(5)``, a literal array, ``0.7*ones + 0.3*eye`` --
with no ``np.random.*`` draw anywhere after the seed call. Verified against the
brief's code as written: all five seeds produce bitwise-identical matrices, the
per-seed ENIVs are [5.0, 5.0, 5.0, 5.0, 5.0], [2.604167] x 5 and [1.315789] x 5,
and the reported standard deviation is exactly 0.0. It is the identical defect
found in Part 21, one part later.

0.0 is worse than a single-seed number, because it reads as a remarkably stable
estimator rather than as an absent measurement, and CLAUDE.md's 5-SEED RULE is
satisfied in form while producing nothing. A seed has to select something.

Here each condition is a BAND of off-diagonal dependence rather than one fixed
matrix. Each seed draws 20 query matrices, each with every off-diagonal entry
drawn independently from its band. The spread across seeds then describes real
variation in how the aggregation behaves over the range of matrices a condition
covers, which is the quantity a standard deviation is supposed to describe.

The bands are disjoint -- [0.00, 0.10], [0.30, 0.50], [0.70, 0.90] -- so the
monotonicity check still separates three genuinely distinct regimes. Nothing in
the perturbation can reorder them: the largest independent draw is below the
smallest moderate draw.


WHAT ELSE THE BRIEF GETS WRONG HERE
-----------------------------------
- Its fourth condition, ``decreasing_eniv``, is initialised in the results dict
  and never filled. ``np.mean([])`` is nan, so the saved JSON would carry a nan
  for a required condition -- the same never-populated-condition bug as Part 21.
  It is implemented here as a per-seed sweep over a correlation grid.

- Its predicted values do not match its own matrices. It says moderate gives
  "ENIV ~ 2, discount ~ 0.4" (actual: 2.6042 / 0.5208) and high gives
  "ENIV ~ 0.5, discount ~ 0.1 (capped)" (actual: 1.3158 / 0.2632, which never
  reaches the floor). The floor is therefore untested by the brief's own design.

- It builds its reports with ``type('DependenceReport', (), {...})()``, a
  duck-typed stub that bypasses every validation in the real dataclass. Real
  DependenceReport objects are constructed here, so the experiment exercises
  the runtime rather than a lookalike.

- It checks monotonicity only on the mean over seeds. A per-seed check is
  stronger and costs nothing: a population that cancels reads as no effect.
  Both are reported; the gate is decided on the per-seed result.


THE CLIQUE CONDITION IS THE POINT
---------------------------------
``disjoint_clique`` is not in the brief. It is the structure the two
aggregations disagree about, and the reason ``method="eigen"`` is the default:
two near-duplicate angles beside three independent ones, which is exactly what
Part 19's fallback chain produces when it quietly points two angles at one
source. Under exchangeability the design effect and the eigenvalue form agree;
this condition is where they come apart, so it is reported under both.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from prismflow.v2.dependence import DependenceReport  # noqa: E402
from prismflow.v2.statistics import (  # noqa: E402
    ENIV_METHODS,
    compute_discount_factor,
    compute_semantic_eniv,
    per_angle_discount,
)

N_SEEDS = 5
N_QUERIES = 20
N_ANGLES = 5
ANGLE_NAMES = ["tech", "market", "financial", "regulatory", "sentiment"]
OUTPUT = Path("results/v2/eniv_redundancy_experiment.json")

#: Baseline ENIV below this fails the gate (the brief's threshold).
BASELINE_ENIV_MIN = 3.5

#: The three ordered bands. Every off-diagonal entry of a condition's matrix is
#: drawn uniformly from its band, independently per pair, per query, per seed.
#: Disjoint by construction so the ordering cannot be an artefact of the draw.
BANDS: Dict[str, Tuple[float, float]] = {
    "independent": (0.00, 0.10),
    "moderate_correlation": (0.30, 0.50),
    "high_correlation": (0.70, 0.90),
}

#: Condition 4: the correlation grid the discount is swept along. Wider than
#: the three bands and overlapping them, so monotonicity is tested across the
#: whole range rather than only at three points.
SWEEP_GRID: Tuple[float, ...] = (0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
SWEEP_JITTER = 0.05

#: Condition 5: two angles near-duplicated, three independent of them and of
#: each other. The non-exchangeable structure.
CLIQUE_MEMBERS = (0, 1)
CLIQUE_BAND = (0.85, 0.98)
CLIQUE_BACKGROUND = (0.00, 0.10)


# --- building matrices -----------------------------------------------------


def symmetric_from_band(
    rng: np.random.Generator, n: int, low: float, high: float
) -> np.ndarray:
    """A valid dependence matrix with every off-diagonal drawn from [low, high]."""
    matrix = np.eye(n, dtype=np.float64)
    i, j = np.triu_indices(n, k=1)
    draws = rng.uniform(low, high, size=i.size)
    matrix[i, j] = draws
    matrix[j, i] = draws
    return matrix


def clique_matrix(rng: np.random.Generator, n: int) -> np.ndarray:
    """Near-duplicate pair inside an otherwise independent set."""
    matrix = symmetric_from_band(rng, n, *CLIQUE_BACKGROUND)
    a, b = CLIQUE_MEMBERS
    value = float(rng.uniform(*CLIQUE_BAND))
    matrix[a, b] = value
    matrix[b, a] = value
    return matrix


def as_report(matrix: np.ndarray) -> DependenceReport:
    """Wrap a matrix in the real validated dataclass, not a duck-typed stub."""
    n = matrix.shape[0]
    return DependenceReport(
        angle_names=ANGLE_NAMES[:n],
        n_angles=n,
        dependence_matrix=matrix,
    )


def measure(matrix: np.ndarray) -> Dict[str, Dict[str, float]]:
    """ENIV and discount under both aggregations."""
    report = as_report(matrix)
    out: Dict[str, Dict[str, float]] = {}
    for method in ENIV_METHODS:
        eniv = compute_semantic_eniv(report, method=method)
        out[method] = {
            "eniv": eniv,
            "discount": compute_discount_factor(eniv, report.n_angles),
        }
    return out


# --- reporting helpers -----------------------------------------------------


def mean_sd(values: Sequence[float]) -> Dict[str, float]:
    values = list(values)
    return {
        "mean": float(statistics.fmean(values)),
        "sd": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarise(
    per_seed_queries: Dict[str, List[List[float]]],
) -> Dict[str, Dict[str, object]]:
    """Mean +/- sd over SEED MEANS, plus the pooled range over every query."""
    summary: Dict[str, Dict[str, object]] = {}
    for key, seeds in per_seed_queries.items():
        seed_means = [float(statistics.fmean(queries)) for queries in seeds]
        pooled = [v for queries in seeds for v in queries]
        summary[key] = {
            **mean_sd(seed_means),
            "per_seed_mean": seed_means,
            "pooled_min": float(min(pooled)),
            "pooled_max": float(max(pooled)),
            "n_queries_total": len(pooled),
        }
    return summary


def non_increasing(values: Sequence[float], tolerance: float = 1e-9) -> bool:
    return all(b <= a + tolerance for a, b in zip(values, values[1:]))


# --- the experiment --------------------------------------------------------


def run_eniv_redundancy_experiment(
    num_seeds: int = N_SEEDS, num_queries: int = N_QUERIES
) -> Dict[str, object]:
    started = time.time()

    # condition -> method -> seed -> [per-query value]
    eniv: Dict[str, Dict[str, List[List[float]]]] = {}
    discount: Dict[str, Dict[str, List[List[float]]]] = {}
    conditions = list(BANDS) + ["disjoint_clique"]
    for condition in conditions:
        eniv[condition] = {m: [] for m in ENIV_METHODS}
        discount[condition] = {m: [] for m in ENIV_METHODS}

    # condition 4, per seed: the sweep's discount curve, averaged over queries
    sweep_curves: Dict[str, List[List[float]]] = {m: [] for m in ENIV_METHODS}
    sweep_monotone: Dict[str, List[bool]] = {m: [] for m in ENIV_METHODS}

    # the per-angle alphas on the clique condition, to show what the scalar hides
    clique_alphas: List[List[float]] = []

    for seed in range(num_seeds):
        # A Generator, not np.random.seed: the global seed is what let the
        # brief's loop look seeded while drawing nothing.
        rng = np.random.default_rng(seed)

        seed_eniv = {c: {m: [] for m in ENIV_METHODS} for c in conditions}
        seed_discount = {c: {m: [] for m in ENIV_METHODS} for c in conditions}
        seed_sweep = {m: [] for m in ENIV_METHODS}

        for _ in range(num_queries):
            for condition, (low, high) in BANDS.items():
                matrix = symmetric_from_band(rng, N_ANGLES, low, high)
                measured = measure(matrix)
                for method in ENIV_METHODS:
                    seed_eniv[condition][method].append(measured[method]["eniv"])
                    seed_discount[condition][method].append(
                        measured[method]["discount"]
                    )

            matrix = clique_matrix(rng, N_ANGLES)
            measured = measure(matrix)
            for method in ENIV_METHODS:
                seed_eniv["disjoint_clique"][method].append(measured[method]["eniv"])
                seed_discount["disjoint_clique"][method].append(
                    measured[method]["discount"]
                )
            clique_alphas.append(
                [per_angle_discount(as_report(matrix))[name] for name in ANGLE_NAMES]
            )

            # Condition 4: one sweep per query, jittered per rung so the curve
            # is not the same seven numbers every time.
            curve = {m: [] for m in ENIV_METHODS}
            for rho in SWEEP_GRID:
                low = max(0.0, rho - SWEEP_JITTER)
                high = min(1.0, rho + SWEEP_JITTER)
                swept = measure(symmetric_from_band(rng, N_ANGLES, low, high))
                for method in ENIV_METHODS:
                    curve[method].append(swept[method]["discount"])
            for method in ENIV_METHODS:
                seed_sweep[method].append(curve[method])

        for condition in conditions:
            for method in ENIV_METHODS:
                eniv[condition][method].append(seed_eniv[condition][method])
                discount[condition][method].append(seed_discount[condition][method])

        for method in ENIV_METHODS:
            # Average the query curves into one curve for this seed, then ask
            # whether THAT seed's curve is monotone.
            curve = [
                float(statistics.fmean(rung))
                for rung in zip(*seed_sweep[method])
            ]
            sweep_curves[method].append(curve)
            sweep_monotone[method].append(non_increasing(curve))

    # --- summaries ---------------------------------------------------------

    summary: Dict[str, object] = {}
    for method in ENIV_METHODS:
        summary[method] = {
            "eniv": summarise({c: eniv[c][method] for c in conditions}),
            "discount": summarise({c: discount[c][method] for c in conditions}),
        }

    sweep_summary = {}
    for method in ENIV_METHODS:
        curves = sweep_curves[method]
        sweep_summary[method] = {
            "grid": list(SWEEP_GRID),
            "discount_mean_by_rung": [
                float(statistics.fmean(rung)) for rung in zip(*curves)
            ],
            "discount_sd_by_rung": [
                float(statistics.stdev(rung)) if len(rung) > 1 else 0.0
                for rung in zip(*curves)
            ],
            "per_seed_curves": curves,
            "per_seed_monotone": sweep_monotone[method],
            "seeds_monotone": int(sum(sweep_monotone[method])),
            "mean_curve_monotone": non_increasing(
                [float(statistics.fmean(rung)) for rung in zip(*curves)]
            ),
        }

    alphas = np.asarray(clique_alphas, dtype=float)
    clique_detail = {
        "members": [ANGLE_NAMES[i] for i in CLIQUE_MEMBERS],
        "per_angle_alpha_mean": {
            name: float(alphas[:, i].mean()) for i, name in enumerate(ANGLE_NAMES)
        },
        "per_angle_alpha_sd": {
            name: float(alphas[:, i].std(ddof=1)) for i, name in enumerate(ANGLE_NAMES)
        },
    }

    # --- gate --------------------------------------------------------------
    #
    # Judged on "eigen", V1's validated aggregation. The design-effect column is
    # reported beside it but does not decide the gate.

    gate: Dict[str, object] = {}
    for method in ENIV_METHODS:
        disc = summary[method]["discount"]
        baseline = summary[method]["eniv"]["independent"]["mean"]

        # Per-seed ordering first: a mean can be monotone while individual
        # seeds are not.
        per_seed_ordered = [
            ind > mod > high
            for ind, mod, high in zip(
                disc["independent"]["per_seed_mean"],
                disc["moderate_correlation"]["per_seed_mean"],
                disc["high_correlation"]["per_seed_mean"],
            )
        ]
        gate[method] = {
            "baseline_eniv": baseline,
            "baseline_eniv_min": BASELINE_ENIV_MIN,
            "baseline_pass": bool(baseline >= BASELINE_ENIV_MIN),
            "discount_independent": disc["independent"]["mean"],
            "discount_moderate": disc["moderate_correlation"]["mean"],
            "discount_high": disc["high_correlation"]["mean"],
            "monotonic_on_means": bool(
                disc["independent"]["mean"]
                > disc["moderate_correlation"]["mean"]
                > disc["high_correlation"]["mean"]
            ),
            "monotonic_per_seed": per_seed_ordered,
            "seeds_monotonic": int(sum(per_seed_ordered)),
            "sweep_seeds_monotone": sweep_summary[method]["seeds_monotone"],
            "sweep_monotonic_all_seeds": bool(
                all(sweep_summary[method]["per_seed_monotone"])
            ),
        }
        gate[method]["passed"] = bool(
            gate[method]["baseline_pass"]
            and all(per_seed_ordered)
            and gate[method]["sweep_monotonic_all_seeds"]
        )

    decided_on = "eigen"
    payload = {
        "part": 22,
        "experiment": "eniv_redundancy",
        "n_seeds": num_seeds,
        "n_queries": num_queries,
        "n_angles": N_ANGLES,
        "bands": {k: list(v) for k, v in BANDS.items()},
        "sweep_grid": list(SWEEP_GRID),
        "sweep_jitter": SWEEP_JITTER,
        "gate_decided_on": decided_on,
        "summary": summary,
        "sweep": sweep_summary,
        "clique": clique_detail,
        "gate": gate,
        "passed": gate[decided_on]["passed"],
        "elapsed_seconds": round(time.time() - started, 2),
    }
    return payload


def _print(payload: Dict[str, object]) -> None:
    n_seeds = payload["n_seeds"]
    n_queries = payload["n_queries"]
    print(
        f"Part 22 -- ENIV redundancy: {n_queries} queries x "
        f"{len(payload['bands']) + 1} conditions x {n_seeds} seeds"
    )
    print()
    for method in ENIV_METHODS:
        marker = "  <- gate" if method == payload["gate_decided_on"] else ""
        print(f"[{method}]{marker}")
        summary = payload["summary"][method]
        print(
            f"  {'condition':22s} {'ENIV mean +/- sd':>22s} "
            f"{'discount mean +/- sd':>24s}"
        )
        for condition in summary["eniv"]:
            e = summary["eniv"][condition]
            d = summary["discount"][condition]
            print(
                f"  {condition:22s} "
                f"{e['mean']:>11.4f} +/- {e['sd']:<7.4f} "
                f"{d['mean']:>13.4f} +/- {d['sd']:<7.4f}"
            )
        sweep = payload["sweep"][method]
        rungs = " ".join(f"{v:.3f}" for v in sweep["discount_mean_by_rung"])
        print(f"  sweep discount by rho {sweep['grid']}:")
        print(f"    {rungs}   monotone in {sweep['seeds_monotone']}/{n_seeds} seeds")
        print()

    clique = payload["clique"]
    print(
        f"clique condition: {clique['members'][0]} and {clique['members'][1]} "
        "near-duplicated, three angles independent"
    )
    print("  per-angle alpha (V1 soft-cluster), showing what one scalar hides:")
    for name, value in clique["per_angle_alpha_mean"].items():
        sd = clique["per_angle_alpha_sd"][name]
        member = " (clique)" if name in clique["members"] else ""
        print(f"    {name:12s} {value:.4f} +/- {sd:.4f}{member}")
    print()

    for method in ENIV_METHODS:
        g = payload["gate"][method]
        mark = "PASS" if g["passed"] else "FAIL"
        print(
            f"gate [{method}]: {mark}  baseline ENIV {g['baseline_eniv']:.4f} "
            f"(>= {g['baseline_eniv_min']}), band order in "
            f"{g['seeds_monotonic']}/{n_seeds} seeds, sweep monotone in "
            f"{g['sweep_seeds_monotone']}/{n_seeds} seeds"
        )
    print()
    if payload["passed"]:
        print(f"GATE PASSED on {payload['gate_decided_on']} -- Part 23 may proceed.")
    else:
        print(
            f"GATE FAILED on {payload['gate_decided_on']} -- STOP. "
            "Do not proceed to Part 23."
        )


def main() -> int:
    payload = run_eniv_redundancy_experiment()
    _print(payload)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUTPUT}")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
