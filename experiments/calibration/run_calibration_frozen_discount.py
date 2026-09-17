"""Mechanism-only control: does the discount help without any training adaptation?

The last calibration confound. PrismFlow's resolution/AURC advantage under
duplication could come from the discount mechanism itself, or from PrismFlow's
training adapting to a discount it experiences. This condition removes all
training adaptation, both to duplicates and to the discount:

  naive_frozen_discounted  the naive model trained at k = 0, weights frozen,
                           widened to 4 + k views exactly as in
                           run_calibration_frozen_naive.py (copy slots clone
                           view 0's trained encoder and head), with the
                           discount switched ON at inference only.

It is the naive_frozen model with one flag changed, so the paired comparison
naive_frozen_discounted - naive_frozen is the discount's effect alone.

Only this system is scored here, via evaluate(). Comparison systems are read
from per-seed rows already written by evaluate():
  naive_frozen                  results/calibration_frozen_naive/k<k>_naive_frozen/
  naive, prismflow,
  naive_weights_discounted      results/calibration_duplicated/k<k>_<system>/

Reproduction checks (the script refuses to compare if either fails):
  - k = 0 must equal calibration_duplicated/k0_naive_weights_discounted, which
    is the same trained model with the same discount;
  - with the discount toggled off, k = 0 must equal naive_frozen at k = 0.

Usage:
    python -m experiments.calibration.run_calibration_frozen_discount
    python -m experiments.calibration.run_calibration_frozen_discount --quick --results-dir <scratch>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from experiments.calibration.run_calibration import build_dataset
from experiments.calibration.run_calibration_duplicated import train
from experiments.calibration.run_calibration_frozen_naive import (
    CHANGE_METRICS,
    DUPLICATED_ID,
    REPRODUCTION_TOLERANCE,
    _CheckedModel,
    _fmt,
    _stats,
    widen_frozen,
)
from prismflow.data.corruption import duplicate_view
from prismflow.data.loaders import iter_batches
from prismflow.evaluation import compute_metrics, evaluate
from prismflow.evaluation.protocol import collect_predictions  # reproduction check only, not reported
from prismflow.utils.logging import get_logger

CONFIG_PATH = Path(__file__).with_name("config_duplicated.yaml")
FROZEN_ID = "calibration_frozen_naive"

SYSTEM = "naive_frozen_discounted"
SOURCES = {
    "naive_frozen": (FROZEN_ID, "naive_frozen"),
    "naive": (DUPLICATED_ID, "naive"),
    "prismflow": (DUPLICATED_ID, "prismflow"),
    "naive_weights_discounted": (DUPLICATED_ID, "naive_weights_discounted"),
}
TABLE_SYSTEMS = (SYSTEM, "naive_frozen", "prismflow", "naive", "naive_weights_discounted")

METRICS = (
    "accuracy",
    "prob_mean_confidence",
    "prob_ece",
    "brier",
    "brier_reliability",
    "brier_resolution",
    "brier_uncertainty",
    "brier_within_bin",
    "prob_aurc",
    "vacuity_mean_confidence",
    "vacuity_ece",
    "eniv",
    "efficiency_ratio",
)


def _load_rows(results_dir: Path, experiment_id: str, k: int, system: str) -> dict[int, dict]:
    path = results_dir / experiment_id / f"k{k}_{system}" / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the earlier calibration followups first")
    report = json.loads(path.read_text(encoding="utf-8"))
    return {row["seed"]: row for row in report["per_seed"]}


def _max_diff(a: dict, b: dict, seeds) -> float:
    return max(abs(a[s][m] - b[s][m]) for s in seeds for m in CHANGE_METRICS)


def render_markdown(rows, config, experiment_id):
    ks, seeds = config["duplicates"], config["seeds"]
    lines = [
        "# Mechanism-only control: frozen naive + discount at inference",
        "",
        f"Seeds: {seeds}. Base: {config['data']['n_views']} views, rho = {config['data']['rho']}; "
        f"k copies of view {config['source_view']}. `{SYSTEM}` from "
        f"`results/{experiment_id}/k<k>_{SYSTEM}/`; `naive_frozen` from `results/{FROZEN_ID}/`; "
        f"the rest from `results/{DUPLICATED_ID}/`. Mean +/- sample std across seeds.",
        "",
    ]

    def paired(k, a, b, metric):
        diffs = [rows[k][a][s][metric] - rows[k][b][s][metric] for s in seeds]
        return _fmt(*_stats(diffs), signed=True), f"{sum(d < 0 for d in diffs)}/{len(diffs)}"

    for k in ks:
        lines += [
            f"## k = {k}",
            "",
            "| metric | " + " | ".join(TABLE_SYSTEMS) + " |",
            "|---|" + "---|" * len(TABLE_SYSTEMS),
        ]
        for metric in METRICS:
            cells = [_fmt(*_stats([rows[k][s][seed][metric] for seed in seeds])) for s in TABLE_SYSTEMS]
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Paired within-seed differences:",
            "",
            f"| metric | {SYSTEM} - naive_frozen | seeds lower | {SYSTEM} - prismflow | seeds lower |",
            "|---|---|---|---|---|",
        ]
        for metric in CHANGE_METRICS:
            lines.append(
                f"| {metric} | " + " | ".join([*paired(k, SYSTEM, "naive_frozen", metric),
                                             *paired(k, SYSTEM, "prismflow", metric)]) + " |"
            )
        lines.append("")

    lines += [
        f"## Change from k = {ks[0]}, within seed",
        "",
        "| metric | system | " + " | ".join(f"k={k} - k={ks[0]}" for k in ks[1:]) + " |",
        "|---|---|" + "---|" * (len(ks) - 1),
    ]
    for metric in CHANGE_METRICS:
        for system in TABLE_SYSTEMS:
            cells = []
            for k in ks[1:]:
                diffs = [rows[k][system][s][metric] - rows[ks[0]][system][s][metric] for s in seeds]
                cells.append(f"{_fmt(*_stats(diffs), signed=True)} ({sum(d > 0 for d in diffs)}/{len(diffs)} up)")
            lines.append(f"| {metric} | {system} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += [
        "## How much of PrismFlow's protection does the mechanism alone provide?",
        "",
        "Within seed, change from k = 0 minus naive_frozen's change from k = 0. "
        "For resolution/accuracy, positive = less degradation than the undiscounted frozen model; "
        "for Brier/AURC/ECE/reliability, negative = less degradation.",
        "",
        "| metric | system | " + " | ".join(f"k={k}" for k in ks[1:]) + " |",
        "|---|---|" + "---|" * (len(ks) - 1),
    ]
    for metric in CHANGE_METRICS:
        for system in (SYSTEM, "prismflow"):
            cells = []
            for k in ks[1:]:
                diffs = [
                    (rows[k][system][s][metric] - rows[ks[0]][system][s][metric])
                    - (rows[k]["naive_frozen"][s][metric] - rows[ks[0]]["naive_frozen"][s][metric])
                    for s in seeds
                ]
                cells.append(f"{_fmt(*_stats(diffs), signed=True)} ({sum(d > 0 for d in diffs)}/{len(diffs)} up)")
            lines.append(f"| {metric} | {system} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the mechanism-only discount control.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-id", default="calibration_frozen_discount")
    parser.add_argument(
        "--comparison-dir", default=None, help="results dir holding the earlier runs; defaults to --results-dir"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test only: 2 epochs, no comparison -- NOT evidence, never report",
    )
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_id = args.experiment_id
    if args.quick:
        config["training"]["epochs"] = 2
        experiment_id = f"{experiment_id}_SMOKE_TEST_NOT_EVIDENCE"
    if len(config["seeds"]) < 5:
        raise ValueError("contract requires at least 5 seeds")

    logger = get_logger("prismflow.calibration_frozen_discount")
    seeds, ks = config["seeds"], config["duplicates"]
    if ks[0] != 0:
        raise ValueError("duplicates must start at k = 0, where the frozen model is trained")
    eval_cfg = config["evaluation"]
    source_view = config["source_view"]
    base_views = config["data"]["n_views"]
    comparison_dir = Path(args.comparison_dir or args.results_dir)

    # Same function, seed and data as calibration_frozen_naive: the same k = 0 model.
    datasets, frozen = {}, {}
    for seed in seeds:
        started = time.perf_counter()
        dataset, splits = build_dataset(seed, config["data"])
        datasets[seed] = (dataset, splits)
        frozen[seed] = train(dataset, splits, 0, source_view, seed, False, config["data"], config["training"])
        logger.info("seed %d: k=0 naive trained (%.0fs)", seed, time.perf_counter() - started)

    def loader_for(k):
        def loader(seed):
            dataset, splits = datasets[seed]
            for batch in iter_batches(dataset, splits[eval_cfg["split"]], eval_cfg["batch_size"]):
                yield duplicate_view(batch, source_view, k)
        return loader

    rows = {k: {} for k in ks}
    undiscounted_k0 = {}
    for k in ks:
        models = {}
        for seed in seeds:
            wide = widen_frozen(frozen[seed], k, source_view, config["data"])
            if k == 0 and not args.quick:
                # Discount off: must be the naive_frozen model exactly.
                collected = collect_predictions(wide, loader_for(0)(seed))
                undiscounted_k0[seed] = compute_metrics(
                    collected["labels"], collected["probs"], collected["vacuity_confidence"],
                    n_bins=eval_cfg["n_bins"], coverages=eval_cfg["coverages"],
                )
            wide.use_discount = True
            models[seed] = _CheckedModel(wide, base_views, source_view)

        report = evaluate(
            models,
            loader_for(k),
            f"{experiment_id}/k{k}_{SYSTEM}",
            seeds,
            results_dir=args.results_dir,
            n_bins=eval_cfg["n_bins"],
            coverages=eval_cfg["coverages"],
            title=f"{experiment_id}: k={k}, {SYSTEM}",
        )
        rows[k][SYSTEM] = {row["seed"]: row for row in report["per_seed"]}
        summary = report["summary"]
        logger.info(
            "k=%d %s acc=%.4f ece=%.4f reliability=%.4f resolution=%.4f aurc=%.4f eniv=%.3f",
            k, SYSTEM, summary["accuracy"]["mean"], summary["prob_ece"]["mean"],
            summary["brier_reliability"]["mean"], summary["brier_resolution"]["mean"],
            summary["prob_aurc"]["mean"], summary["eniv"]["mean"],
        )

    if args.quick:
        logger.info("quick mode: comparison and summary skipped")
        return

    for k in ks:
        for name, (source_id, system) in SOURCES.items():
            rows[k][name] = _load_rows(comparison_dir, source_id, k, system)

    checks = {
        "k0_vs_calibration_duplicated_naive_weights_discounted": _max_diff(
            rows[0][SYSTEM], rows[0]["naive_weights_discounted"], seeds
        ),
        "k0_discount_off_vs_naive_frozen": _max_diff(
            {s: {**undiscounted_k0[s], "seed": s} for s in seeds}, rows[0]["naive_frozen"], seeds
        ),
    }
    for name, value in checks.items():
        logger.info("reproduction check %s: max abs diff = %.3g", name, value)
        if value > REPRODUCTION_TOLERANCE:
            raise RuntimeError(f"reproduction check {name} failed (max diff {value:.3g}); comparison voided")

    out_dir = Path(args.results_dir) / experiment_id
    markdown = render_markdown(rows, config, experiment_id)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    (out_dir / "config_used.json").write_text(
        json.dumps({**config, "reproduction_max_abs_diff": checks}, indent=2), encoding="utf-8"
    )
    print()
    print(markdown)


if __name__ == "__main__":
    main()
