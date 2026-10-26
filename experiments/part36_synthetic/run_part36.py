"""Part 36: synthetic estimator and detector checks (L2, L9, L10, L11, L22).

Every setting, threshold and decision rule comes from config.yaml, which was
committed before this file existed. Nothing here chooses a number; it reads
the pre-registration, runs the cells, and prints the verdict the rule gives.

Frozen modules and earlier experiments are IMPORTED, never edited:
  experiments.eniv_validation.run_validation  -> analysis A and the 4-view repro
  experiments.tail.run_tail                   -> analysis C and the tail repro
  experiments.comparison.run_comparison       -> analysis D and the clone repro
  experiments.calibration.run_calibration     -> the trainer both of those use

Usage:
    python -m experiments.part36_synthetic.run_part36 --only a
    ... --only b | c | d
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from experiments.calibration.run_calibration import build_dataset as cal_build_dataset
from experiments.calibration.run_calibration import train as cal_train
from experiments.comparison.run_comparison import roc_auc as cmp_roc_auc
from experiments.comparison.run_comparison import run_condition
from experiments.eniv_validation.run_validation import (
    _aggregate,
    collect_evidence,
    run_cell,
    train_model,
)
from experiments.tail import run_tail as tailmod
from prismflow.data.synthetic import analytic_n_eff_eigen
from prismflow.eniv.eniv import compute_eniv
from prismflow.models.defended import DefendedPrismFlow
from prismflow.statistics.dependence import feature_dependence_matrix
from prismflow.train import TrainConfig
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

HERE = Path(__file__).parent
CFG = yaml.safe_load((HERE / "config.yaml").read_text(encoding="utf-8"))
OUT = Path(CFG["outputs"]["results_dir"])
SEEDS = CFG["seeds"]
LOGGER = get_logger("prismflow.part36")

# The primary estimator, named in config.yaml analysis_a.primary_estimator.
PRIMARY = CFG["analysis_a"]["primary_estimator"]


def mean_std(values) -> dict:
    """Mean and sample std (ddof=1), the project's convention."""
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return {
        "mean": statistics.mean(vals) if vals else float("nan"),
        "std": statistics.stdev(vals) if len(vals) > 1 else float("nan"),
        "n": len(vals),
    }


def json_safe(value):
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(float(value))
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write(name: str, payload: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
    LOGGER.info("wrote %s", path)


def budget_decision(seconds_one_seed: float, label: str) -> bool:
    """True if the full grid fits the pre-registered limit. Never a guess."""
    projected = seconds_one_seed * len(SEEDS) / 60.0
    limit = CFG["budget"]["limit_minutes_per_analysis"]
    LOGGER.info(
        "%s: one seed took %.0fs, projection %.1f min against a %d min limit",
        label, seconds_one_seed, projected, limit,
    )
    return projected <= limit


# --------------------------------------------------------------------------
# STEP 0 and analysis A share their V = 4 cells: the committed 4-view point IS
# the V = 4 row, so it is run once and checked twice.
# --------------------------------------------------------------------------
def eniv_seed(view_counts, rho_grid, seed) -> tuple[list[dict], dict]:
    rows, timings = [], {}
    for n_views in view_counts:
        config = TrainConfig(
            experiment_id="part36_synthetic", n_views=n_views, epochs=40, n_samples=2000
        )
        started = time.perf_counter()
        for rho in rho_grid:
            rows.append(run_cell(rho, seed, config, LOGGER))
        timings[f"V{n_views}_seed{seed}"] = time.perf_counter() - started
    return rows, timings


def check_reproduction(rows_v4: list[dict]) -> dict:
    """The committed 4-view quantities, re-derived at the original settings."""
    check = {c["name"]: c for c in CFG["reproduction"]["checks"]}["eniv_validation_4view"]
    agg = _aggregate(rows_v4, PRIMARY)
    entries = []
    for q in check["quantities"]:
        measured = agg[q["path"][1]]
        entries.append({
            "quantity": ".".join(q["path"]), "committed": q["committed"],
            "measured": measured, "tol": q["tol"],
            "abs_diff": abs(measured - q["committed"]),
            "pass": bool(abs(measured - q["committed"]) <= q["tol"]),
        })
    return {"eniv_validation_4view": {"source": check["source"], "quantities": entries}}


def analysis_a() -> dict:
    """STEP 0 then A. The V = 4 cells run at the committed rho grid whatever the
    budget decides for the new view counts, because the reproduction check is
    only a reproduction at the original settings."""
    cfg = CFG["analysis_a"]

    # STEP 0: the committed 4-view point, at the committed settings, first.
    rows, timings = [], {}
    for seed in SEEDS:
        seed_rows, seed_timings = eniv_seed([4], cfg["rho_grid"], seed)
        rows.extend(seed_rows)
        timings.update(seed_timings)
    reproduction = check_reproduction(rows)
    # Written before the check can stop the run, so a failure leaves a record.
    write("reproduction_eniv_4view.json", reproduction)
    _report_failures(reproduction["eniv_validation_4view"])

    # Budget: one seed of the new view counts at the full rho grid, measured.
    new_views = [v for v in cfg["view_counts_full"] if v != 4]
    probe_rows, probe_timings = eniv_seed(new_views, cfg["rho_grid"], SEEDS[0])
    timings.update(probe_timings)
    full = budget_decision(sum(probe_timings.values()), "analysis A")
    if full:
        view_counts, rho_grid = cfg["view_counts_full"], cfg["rho_grid"]
    else:
        view_counts = cfg["view_counts_smallest_subset"]
        rho_grid = cfg["rho_grid_smallest_subset"]
        LOGGER.warning("analysis A: running the smallest pre-registered subset")
    rows.extend(
        r for r in probe_rows if r["n_views"] in view_counts and r["rho"] in rho_grid
    )

    for seed in SEEDS[1:]:
        more_rows, more_timings = eniv_seed(
            [v for v in view_counts if v != 4], rho_grid, seed
        )
        rows.extend(more_rows)
        timings.update(more_timings)

    per_view = {}
    for n_views in view_counts:
        cells = [r for r in rows if r["n_views"] == n_views and r["rho"] in rho_grid]
        entry = {
            "n_cells": len(cells),
            "estimators": {key: _aggregate(cells, key) for key in cfg["reported_estimators"]},
            "signed_error_per_rho": {
                str(rho): mean_std([
                    r[PRIMARY] - r["analytic_n_eff_eigen"] for r in cells if r["rho"] == rho
                ])
                for rho in rho_grid
            },
            "truth_per_rho": {str(rho): analytic_n_eff_eigen(rho, n_views) for rho in rho_grid},
            "estimate_per_rho": {
                str(rho): mean_std([r[PRIMARY] for r in cells if r["rho"] == rho])
                for rho in rho_grid
            },
            "accuracy_per_rho": {
                str(rho): mean_std([r["accuracy"] for r in cells if r["rho"] == rho])
                for rho in rho_grid
            },
        }
        zero = [r for r in cells if r["rho"] == 0.0]
        if zero:
            entry["residual_rho_bar_at_rho0"] = {
                "features_cca_holdout": mean_std(
                    [r["rho_bar_features_cca_holdout"] for r in zero]
                ),
                "features_cca_holdout_nc": mean_std(
                    [r["rho_bar_features_cca_holdout_nc"] for r in zero]
                ),
            }
        per_view[str(n_views)] = entry

    return {
        "ran_full_grid": full, "view_counts": view_counts, "rho_grid": rho_grid,
        "primary_estimator": PRIMARY, "reproduction": reproduction,
        "per_view_count": per_view,
        "rows": rows, "timings_seconds": timings,
        "l9_verdict": verdict_l9(per_view, view_counts),
        "l2_verdict": verdict_l2(per_view, view_counts),
    }


def verdict_l9(per_view: dict, view_counts: list[int]) -> dict:
    """config.yaml analysis_a.decision_rule_l9, applied literally."""
    base = per_view[str(view_counts[0])]["estimators"][PRIMARY]
    base_mae, base_std = base["mae"], base["mae_std"]
    top_mae = per_view[str(view_counts[-1])]["estimators"][PRIMARY]["mae"]
    scales = top_mae - base_mae > base_std
    flat = all(
        abs(per_view[str(v)]["estimators"][PRIMARY]["mae"] - base_mae) <= base_std
        for v in view_counts
    )
    return {
        "rule": "SCALES if MAE(V_max) - MAE(V_min) > across-cell std at V_min; "
                "FLAT if every MAE is within one V_min std of MAE(V_min); else MIXED",
        "mae_by_view_count": {
            str(v): per_view[str(v)]["estimators"][PRIMARY]["mae"] for v in view_counts
        },
        "mae_std_by_view_count": {
            str(v): per_view[str(v)]["estimators"][PRIMARY]["mae_std"] for v in view_counts
        },
        "v_min_mae": base_mae, "v_min_mae_std": base_std,
        "v_max_minus_v_min": top_mae - base_mae,
        "verdict": "SCALES" if scales else ("FLAT" if flat else "MIXED"),
    }


def verdict_l2(per_view: dict, view_counts: list[int]) -> dict:
    """config.yaml analysis_a.decision_rule_l2, applied literally."""
    key = "residual_rho_bar_at_rho0"
    if not all(key in per_view[str(v)] for v in view_counts):
        return {"verdict": "NOT MEASURED", "reason": "rho = 0.0 was not in the grid run"}
    values = [per_view[str(v)][key]["features_cca_holdout"] for v in view_counts]
    means = [e["mean"] for e in values]
    monotone = all(b >= a - 1e-12 for a, b in zip(means, means[1:]))
    gap = means[-1] - means[0]
    larger_std = max(values[0]["std"], values[-1]["std"])
    return {
        "rule": "ACCUMULATES if the mean rho_bar at rho = 0.0 is monotonically "
                "non-decreasing in V and the V_max value exceeds the V_min value by "
                "more than the larger of the two stds",
        "rho_bar_by_view_count": {str(v): e for v, e in zip(view_counts, values)},
        "monotone_non_decreasing": monotone,
        "v_max_minus_v_min": gap, "larger_std": larger_std,
        "l2_quoted_per_pair_figure": 0.05,
        "verdict": "ACCUMULATES" if (monotone and gap > larger_std) else "DOES NOT ACCUMULATE",
    }


# --------------------------------------------------------------------------
# B - L10: the estimation sample-count sweep.
# --------------------------------------------------------------------------
def _train_for_b(n_views: int, rho: float, seed: int):
    config = TrainConfig(
        experiment_id="part36_synthetic", n_views=n_views, rho=rho,
        epochs=40, n_samples=CFG["analysis_b"]["n_samples"],
    )
    model, dataset, splits = train_model(config, seed)
    evidence, features, masks, _ = collect_evidence(
        model, dataset, splits["test"], config.batch_size
    )
    return features, evidence, masks


def _sweep(features, evidence, masks, n_grid, seed) -> dict:
    """ENIV and rho_bar on NESTED random subsamples of one model's test split."""
    rng = np.random.default_rng(36000 + seed)
    order = rng.permutation(features.shape[0])
    out = {}
    for n in n_grid:
        if n > order.size:
            out[str(n)] = {
                "eniv": None, "rho_bar": None, "n_rows": int(order.size),
                "note": "not measured: the test split has fewer rows than this n",
            }
            continue
        idx = torch.as_tensor(np.ascontiguousarray(order[:n]))
        matrix = feature_dependence_matrix(
            features[idx], evidence[idx], masks[idx],
            method="cca", conditioning="pairwise_holdout", seed=seed,
        )
        result = compute_eniv(matrix)
        out[str(n)] = {
            "eniv": float(result.effective_views),
            "rho_bar": float(result.mean_dependence),
            "n_rows": int(n),
        }
    return out


def _fit_floor(n_grid, abs_bias) -> dict:
    """|bias|(n) = floor + c / n by least squares. Reported with its residual."""
    pairs = [(n, b) for n, b in zip(n_grid, abs_bias) if b is not None and np.isfinite(b)]
    if len(pairs) < 2:
        return {"floor": None, "c": None, "residual_rms": None, "n_points": len(pairs),
                "note": "not measured: fewer than two usable points"}
    ns = np.array([p[0] for p in pairs], dtype=float)
    ys = np.array([p[1] for p in pairs], dtype=float)
    design = np.column_stack([np.ones_like(ns), 1.0 / ns])
    coef, *_ = np.linalg.lstsq(design, ys, rcond=None)
    residual = ys - design @ coef
    return {
        "floor": float(coef[0]), "c": float(coef[1]),
        "residual_rms": float(np.sqrt((residual ** 2).mean())),
        "n_points": len(pairs),
    }


def analysis_b() -> dict:
    cfg = CFG["analysis_b"]
    arms = [("dependent", cfg["rho_fixed"]), ("independent_null", 0.0)]

    started = time.perf_counter()
    per_seed = {}
    for n_views in cfg["view_counts"]:
        for arm, rho in arms:
            features, evidence, masks = _train_for_b(n_views, rho, SEEDS[0])
            per_seed[(n_views, arm, SEEDS[0])] = _sweep(
                features, evidence, masks, cfg["estimation_n_full"], SEEDS[0]
            )
    one_seed = time.perf_counter() - started
    full = budget_decision(one_seed, "analysis B")
    n_grid = cfg["estimation_n_full"] if full else cfg["estimation_n_smallest_subset"]
    if not full:
        LOGGER.warning("analysis B: running the smallest pre-registered subset")
        per_seed = {k: {str(n): v[str(n)] for n in n_grid} for k, v in per_seed.items()}

    for n_views in cfg["view_counts"]:
        for arm, rho in arms:
            for seed in SEEDS[1:]:
                features, evidence, masks = _train_for_b(n_views, rho, seed)
                per_seed[(n_views, arm, seed)] = _sweep(
                    features, evidence, masks, n_grid, seed
                )
            LOGGER.info("analysis B: V=%d %s done", n_views, arm)

    cells = {}
    for n_views in cfg["view_counts"]:
        for arm, rho in arms:
            truth = analytic_n_eff_eigen(rho, n_views)
            fits = [
                _fit_floor(n_grid, [
                    None if per_seed[(n_views, arm, s)][str(n)]["eniv"] is None
                    else abs(per_seed[(n_views, arm, s)][str(n)]["eniv"] - truth)
                    for n in n_grid
                ])
                for s in SEEDS
            ]
            per_n = {}
            for n in n_grid:
                vals = [per_seed[(n_views, arm, s)][str(n)] for s in SEEDS]
                per_n[str(n)] = {
                    "eniv": mean_std([v["eniv"] for v in vals]),
                    "rho_bar": mean_std([v["rho_bar"] for v in vals]),
                    "signed_eniv_bias": mean_std([
                        None if v["eniv"] is None else v["eniv"] - truth for v in vals
                    ]),
                }
            floors = mean_std([f["floor"] for f in fits])
            cells[f"V{n_views}_{arm}"] = {
                "n_views": n_views, "rho": rho, "analytic_n_eff_eigen": truth,
                "per_estimation_n": per_n,
                "fit_floor": floors,
                "fit_c": mean_std([f["c"] for f in fits]),
                "fit_residual_rms": mean_std([f["residual_rms"] for f in fits]),
                "per_seed_fits": fits,
                "verdict": verdict_l10(per_n, floors, n_grid),
            }

    return {
        "ran_full_grid": full, "estimation_n": n_grid,
        "view_counts": cfg["view_counts"], "arms": [a for a, _ in arms],
        "n_samples": cfg["n_samples"], "cells": cells,
        "per_seed": {f"V{k[0]}_{k[1]}_seed{k[2]}": v for k, v in per_seed.items()},
        "timings_seconds": {"one_seed_all_cells": one_seed},
    }


def verdict_l10(per_n: dict, floors: dict, n_grid: list[int]) -> dict:
    """config.yaml analysis_b.decision_rule, applied literally."""
    small = abs(per_n[str(n_grid[0])]["signed_eniv_bias"]["mean"])
    large = abs(per_n[str(n_grid[-1])]["signed_eniv_bias"]["mean"])
    floor, floor_std = floors["mean"], floors["std"]
    floor_at_zero = abs(floor) <= floor_std
    sample_driven = floor_at_zero and small > 2.0 * large
    structural = (not floor_at_zero) and large > floor_std
    return {
        "rule": "DOMINATED BY SAMPLE COUNT if the fitted floor is within one "
                "across-seed std of 0 and the smallest-n bias is more than twice the "
                "largest-n bias; STRUCTURAL FLOOR if the floor exceeds one std of 0 "
                "and the largest-n bias is still more than that std from 0",
        "abs_bias_smallest_n": small, "abs_bias_largest_n": large,
        "fitted_floor": floor, "fitted_floor_std": floor_std,
        "floor_within_one_std_of_zero": floor_at_zero,
        "verdict": "DOMINATED BY SAMPLE COUNT" if sample_driven
                   else ("STRUCTURAL FLOOR" if structural else "INCONCLUSIVE"),
    }


# --------------------------------------------------------------------------
# C - L11: Part 13's detection arm at 4x the split size, and its reproduction
#     at the committed size.
# --------------------------------------------------------------------------
def tail_detection(n_samples: int, epsilons: list[float]) -> dict:
    """Part 13's detection arm, imported cell by cell, at a chosen split size.

    Replicates run_tail.main's per-seed order exactly (set_seed, build_dataset,
    train_model, whole_split, subsample draw) so the committed size reproduces
    bit-for-bit rather than approximately. eps 0.0 is always included: it is the
    clean reference every AUC is measured against.
    """
    config = json.loads(json.dumps(tailmod.CONFIG))
    config["data"] = dict(config["data"], n_samples=n_samples)
    needed = sorted({0.0, *epsilons})

    per_eps = {f"eps{e}": [] for e in needed}
    roc = {f"eps{e}": [] for e in needed if e != 0.0}
    timings, n_test_rows = {}, None

    for seed in SEEDS:
        started = time.perf_counter()
        set_seed(seed)
        dataset, splits, base = tailmod.build_dataset(seed, config["data"])
        model = tailmod.train_model(dataset, splits, base, config["training"], seed)
        views, view_mask, _ = tailmod.whole_split(
            dataset, splits["test"], config["evaluation"]["batch_size"]
        )
        n_test_rows = int(views.shape[0])
        rng = np.random.default_rng(config["tail"]["subsample_seed"] + seed)
        subsample_idx = rng.choice(
            n_test_rows, size=min(config["tail"]["subsample_n"], n_test_rows), replace=False
        )

        clean_scores = None
        for epsilon in needed:
            if epsilon == 0.0:
                attacked = views
                compromised = tuple(range(config["attack"]["k"]))
                target = None
            else:
                attack = tailmod.chorus_attack(
                    model, views, view_mask,
                    tailmod.ChorusConfig(
                        k=config["attack"]["k"], epsilon=epsilon,
                        beta=config["attack"]["beta"], steps=config["attack"]["steps"],
                        seed=seed,
                    ),
                )
                attacked = views + attack.delta
                compromised, target = attack.compromised, attack.target

            row = tailmod.measure_condition(
                model, attacked, view_mask, config, subsample_idx, compromised, target
            )
            scores = row.pop("_scores")
            if epsilon == 0.0:
                clean_scores = scores
            else:
                roc[f"eps{epsilon}"].append({
                    detector: {
                        "auc": tailmod.roc_auc(scores[detector], clean_scores[detector]),
                        "tpr_at_fpr05": tailmod.detection_at_fpr(
                            scores[detector], clean_scores[detector], 0.05
                        ),
                    }
                    for detector in ("tail", "rho_bar")
                })
            row["seed"] = seed
            per_eps[f"eps{epsilon}"].append(row)
        timings[f"seed{seed}"] = time.perf_counter() - started
        LOGGER.info(
            "tail n_samples=%d seed=%d done in %.0fs",
            n_samples, seed, timings[f"seed{seed}"],
        )

    headline = str(config["tail"]["headline_quantile"])
    detection = {
        cell: {
            det: {
                metric: mean_std([e[det][metric] for e in entries])
                for metric in ("auc", "tpr_at_fpr05")
            }
            for det in ("tail", "rho_bar")
        }
        for cell, entries in roc.items() if entries
    }
    protocol = {
        cell: {
            "n_tail_min": mean_std([
                r["tail"][headline]["empirical"]["n_tail_min"] for r in rows
            ]),
            "lambda_u_compromised": mean_std([r["lambda_u_compromised"] for r in rows]),
            "attack_success": mean_std([r["attack_success"] for r in rows]),
        }
        for cell, rows in per_eps.items() if rows
    }
    return {
        "n_samples": n_samples, "n_test_rows": n_test_rows,
        "epsilons_run": needed, "detection": detection, "protocol": protocol,
        "timings_seconds": timings,
    }


def analysis_c() -> dict:
    cfg = CFG["analysis_c"]
    # The committed-size reproduction doubles as the timing probe: it is the
    # same code path on a quarter of the data, so the 4x projection comes from
    # a measurement of the same work rather than from a guess about it.
    repro = tail_detection(cfg["n_samples_committed"], cfg["epsilons_full"])
    checks = {c["name"]: c for c in CFG["reproduction"]["checks"]}["tail_detection"]
    repro_entries = []
    for q in checks["quantities"]:
        cell, _, _, field = q["path"][1], q["path"][2], q["path"][3], q["path"][4]
        measured = repro["detection"][cell]["tail"]["auc"][field]
        repro_entries.append({
            "quantity": ".".join(str(p) for p in q["path"]), "committed": q["committed"],
            "measured": measured, "tol": q["tol"],
            "abs_diff": abs(measured - q["committed"]),
            "pass": bool(abs(measured - q["committed"]) <= q["tol"]),
        })
    reproduction = {"source": checks["source"], "quantities": repro_entries}
    if any(not e["pass"] for e in repro_entries):
        return {"reproduction": reproduction, "committed_size": repro,
                "stopped": "reproduction failed; the 4x run was not started"}

    one_seed_4x = 4.0 * max(repro["timings_seconds"].values())
    full = budget_decision(one_seed_4x, "analysis C (4x, scaled from the repro)")
    epsilons = cfg["epsilons_full"] if full else cfg["epsilons_smallest_subset"]
    if not full:
        LOGGER.warning("analysis C: running the smallest pre-registered subset")
    at_4x = tail_detection(cfg["n_samples_full"], epsilons)

    return {
        "ran_full_grid": full, "epsilons_run": epsilons,
        "reproduction": reproduction,
        "committed_size": repro, "four_x": at_4x,
        "verdict": verdict_l11(repro, at_4x),
    }


def verdict_l11(repro: dict, at_4x: dict) -> dict:
    """config.yaml analysis_c.decision_rule, for BOTH pre-registered bands."""
    cfg = CFG["analysis_c"]
    out = {
        "rule": "SUBSTANTIVE if every epsilon's 4x tail AUC mean lies inside the "
                "band; UNDERPOWERED-AFTER-ALL if any mean moves outside it by more "
                "than its own across-seed std",
        "primary_band": cfg["band_as_committed_in_results"],
        "quoted_band": cfg["band_as_quoted_in_l11"],
        "per_epsilon": {}, "bands": {},
    }
    for cell, entry in at_4x["detection"].items():
        auc = entry["tail"]["auc"]
        out["per_epsilon"][cell] = {
            "tail_auc_4x": auc,
            "tail_auc_committed_size": repro["detection"].get(cell, {}).get("tail", {}).get("auc"),
            "rho_bar_auc_4x": entry["rho_bar"]["auc"],
            "std_over_mean": abs(auc["std"] / auc["mean"]) if auc["mean"] else None,
            "n_tail_min_4x": at_4x["protocol"][cell]["n_tail_min"],
        }
    for label, band in (("primary", out["primary_band"]), ("quoted", out["quoted_band"])):
        low, high = band
        inside, outside_by_more_than_std = [], []
        for cell, entry in out["per_epsilon"].items():
            mean, std = entry["tail_auc_4x"]["mean"], entry["tail_auc_4x"]["std"]
            if low <= mean <= high:
                inside.append(cell)
            elif min(abs(mean - low), abs(mean - high)) > std:
                outside_by_more_than_std.append(cell)
        out["bands"][label] = {
            "band": band, "inside": inside,
            "outside_by_more_than_own_std": outside_by_more_than_std,
            "verdict": ("SUBSTANTIVE" if len(inside) == len(out["per_epsilon"])
                        else ("UNDERPOWERED-AFTER-ALL" if outside_by_more_than_std
                              else "OUTSIDE THE BAND BY LESS THAN ONE STD")),
        }
    return out


# --------------------------------------------------------------------------
# D - L22: the clone comparison at equal width, and the committed clone_k2
#     figure reproduced.
# --------------------------------------------------------------------------
def analysis_d() -> dict:
    import experiments.comparison.run_comparison as cmp

    cfg = CFG["analysis_d"]
    base = json.loads(json.dumps(cmp.CONFIG))
    data4 = dict(base["data"])
    data6 = dict(base["data"], n_views=6)
    clean_cond = {"name": "clean", "kind": "clean", "k": 0}
    clone_cond = {"name": "clone_k2", "kind": "clone", "k": 2}
    eval_cfg = base["evaluation"]

    per_seed, timings = [], {}
    for seed in SEEDS:
        started = time.perf_counter()
        # Same construction as run_comparison.main: one dataset per width per
        # seed, trained by the same imported trainer at the same settings.
        dataset4, splits4 = cal_build_dataset(seed, data4)
        dataset6, splits6 = cal_build_dataset(seed, data6)
        defended = {}
        for width, source, source_splits, data_cfg in (
            (4, dataset4, splits4, data4), (6, dataset6, splits6, data6)
        ):
            model = cal_train(source, source_splits, seed, True, data_cfg, base["training"])
            model.use_discount = True
            model.eval()
            defended[width] = DefendedPrismFlow(
                model,
                suspicion_threshold=base["detector"]["threshold"],
                suspicion_permutations=base["detector"]["permutations"],
                suspicion_seed=seed,
            )

        _, s_clean_w4 = run_condition(
            defended[4], dataset4, splits4[eval_cfg["split"]], clean_cond,
            base["attack"], eval_cfg["batch_size"], seed,
        )
        _, s_clone = run_condition(
            defended[6], dataset4, splits4[eval_cfg["split"]], clone_cond,
            base["attack"], eval_cfg["batch_size"], seed,
        )
        _, s_clean_w6 = run_condition(
            defended[6], dataset6, splits6[eval_cfg["split"]], clean_cond,
            base["attack"], eval_cfg["batch_size"], seed,
        )

        def finite_mean(scores):
            finite = scores[np.isfinite(scores)]
            return float(finite.mean()) if finite.size else float("nan")

        auc_mixed = cmp_roc_auc(s_clone, s_clean_w4)
        auc_equal = cmp_roc_auc(s_clone, s_clean_w6)
        per_seed.append({
            "seed": seed,
            "auc_clone_vs_clean_w4_mixed_width": auc_mixed,
            "auc_clone_vs_clean_w6_equal_width": auc_equal,
            "paired_difference_equal_minus_mixed": auc_equal - auc_mixed,
            "mean_score_clean_w4": finite_mean(s_clean_w4),
            "mean_score_clean_w6": finite_mean(s_clean_w6),
            "mean_score_clone_k2": finite_mean(s_clone),
        })
        timings[f"seed{seed}"] = time.perf_counter() - started
        LOGGER.info(
            "seed %d: mixed-width AUC %.4f, equal-width AUC %.4f (%.0fs)",
            seed, auc_mixed, auc_equal, timings[f"seed{seed}"],
        )

    keys = [k for k in per_seed[0] if k != "seed"]
    summary = {k: mean_std([r[k] for r in per_seed]) for k in keys}

    check = {c["name"]: c for c in CFG["reproduction"]["checks"]}["comparison_clone_k2"]
    repro = []
    for q in check["quantities"]:
        measured = summary["auc_clone_vs_clean_w4_mixed_width"][q["path"][-1]]
        repro.append({
            "quantity": ".".join(q["path"]), "committed": q["committed"],
            "measured": measured, "tol": q["tol"],
            "abs_diff": abs(measured - q["committed"]),
            "pass": bool(abs(measured - q["committed"]) <= q["tol"]),
        })

    return {
        "conditions": cfg["conditions"], "per_seed": per_seed, "summary": summary,
        "reproduction": {"source": check["source"], "quantities": repro},
        "verdict": verdict_l22(summary),
        "timings_seconds": timings,
    }


def verdict_l22(summary: dict) -> dict:
    """config.yaml analysis_d.decision_rule, applied literally."""
    diff = summary["paired_difference_equal_minus_mixed"]
    mixed = summary["auc_clone_vs_clean_w4_mixed_width"]["mean"]
    equal = summary["auc_clone_vs_clean_w6_equal_width"]["mean"]
    if equal > mixed:
        verdict = "HIGHER AT EQUAL WIDTH - L22's direction of concern NOT CONFIRMED"
    elif mixed - equal > diff["std"]:
        verdict = "WIDTH EXPLAINS PART OF THE COMMITTED FIGURE"
    elif abs(diff["mean"]) <= diff["std"]:
        verdict = "WIDTH EXPLAINS NONE OF IT"
    else:
        verdict = "LOWER AT EQUAL WIDTH BY LESS THAN ONE PAIRED STD"
    return {
        "rule": "part explained if the equal-width AUC is below the committed AUC by "
                "more than the paired difference's own std; none if the paired mean is "
                "within one std of 0; a higher equal-width AUC is reported as L22's "
                "direction not confirmed",
        "mixed_width_auc": mixed, "equal_width_auc": equal,
        "paired_difference": diff, "verdict": verdict,
    }


def _report_failures(reproduction: dict) -> None:
    failures = [q for q in reproduction["quantities"] if not q["pass"]]
    if failures:
        LOGGER.error(
            "REPRODUCTION FAILED against %s: %s",
            reproduction["source"], json.dumps(json_safe(failures), indent=2),
        )
        raise SystemExit(2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Part 36 runner (see config.yaml).")
    parser.add_argument("--only", required=True, choices=["a", "b", "c", "d"])
    args = parser.parse_args()

    meta = {"torch": torch.__version__, "numpy": np.__version__, "seeds": SEEDS}

    if args.only == "a":
        write("analysis_a.json", {**meta, **analysis_a()})
    elif args.only == "b":
        write("analysis_b.json", {**meta, **analysis_b()})
    elif args.only == "c":
        result = analysis_c()
        write("analysis_c.json", {**meta, **result})
        _report_failures(result["reproduction"])
    else:
        result = analysis_d()
        write("analysis_d.json", {**meta, **result})
        _report_failures(result["reproduction"])


if __name__ == "__main__":
    main()
