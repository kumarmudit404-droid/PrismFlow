"""Part 08 missing-view experiment: naive fusion vs PrismFlow as views vanish.

Per seed, one naive and one PrismFlow model are trained on clean data (same
seed, identical initial weights; only use_discount differs). Views are then
dropped at test time at each missing rate. Each (sample, view) is dropped
independently (prismflow.data.corruption.drop_views), with a mask that depends
only on (seed, batch index, rate), so every system sees exactly the same
missing entries.

Systems:
  naive              masked views are absent (vacuous) in Dempster fusion
  prismflow          same, with the ENIV discount
  imputed_naive      missing view inputs replaced by the mean of the sample's
                     available view inputs, then scored by the naive model
  imputed_prismflow  the same imputed inputs, scored by PrismFlow

Every metric comes from prismflow.evaluation.evaluate. Outputs go to
results/<id>/missing_r<rate>_<system>/ plus missing_summary.md, which only reads
the reports evaluate() returned.

Usage:
    python -m experiments.robustness.run_missing
    python -m experiments.robustness.run_missing --quick --results-dir <scratch>
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import torch
import yaml

from experiments.calibration.run_calibration import build_dataset, train
from prismflow.data.corruption import drop_views
from prismflow.data.dataset import Batch
from prismflow.data.loaders import iter_batches
from prismflow.evaluation import evaluate
from prismflow.utils.logging import get_logger

CONFIG_PATH = Path(__file__).with_name("config.yaml")

# Metrics the robustness tables report. `vacuity_mean_confidence` is
# 1 - mean fused uncertainty, so uncertainty is read off it directly.
METRICS = (
    "accuracy",
    "prob_ece",
    "brier_reliability",
    "brier_resolution",
    "brier",
    "prob_mean_confidence",
    "vacuity_mean_confidence",
    "eniv",
)


def load_config(quick: bool):
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_id = config["experiment"]["id"]
    if quick:
        config["training"]["epochs"] = 2
        experiment_id = f"{experiment_id}_SMOKE_TEST_NOT_EVIDENCE"
    if len(config["seeds"]) < 5:
        raise ValueError("contract requires at least 5 seeds")
    return config, experiment_id


def train_models(config, logger):
    """{seed: (dataset, splits)}, {system: {seed: model}} for naive and prismflow."""
    datasets, models = {}, {"naive": {}, "prismflow": {}}
    for seed in config["seeds"]:
        started = time.perf_counter()
        dataset, splits = build_dataset(seed, config["data"])
        datasets[seed] = (dataset, splits)
        models["naive"][seed] = train(dataset, splits, seed, False, config["data"], config["training"])
        models["prismflow"][seed] = train(dataset, splits, seed, True, config["data"], config["training"])
        logger.info("seed %d: naive and prismflow trained (%.0fs)", seed, time.perf_counter() - started)
    return datasets, models


def batch_seed(seed: int, batch_index: int, salt: int) -> int:
    """Deterministic corruption seed, distinct per (seed, batch, condition level)."""
    return seed * 1_000_003 + batch_index * 7_919 + salt


def impute_missing(batch: Batch) -> Batch:
    """Replace each missing view's input by the mean of the sample's available view inputs.

    Samples with no available view are left fully missing.
    """
    mask = batch.view_mask
    weights = mask.unsqueeze(-1).to(batch.views.dtype)
    count = weights.sum(dim=1, keepdim=True)
    mean = (batch.views * weights).sum(dim=1, keepdim=True) / count.clamp_min(1.0)
    has_any = mask.any(dim=1)

    fill = (~mask) & has_any.unsqueeze(1)
    views = torch.where(fill.unsqueeze(-1), mean.expand_as(batch.views), batch.views)
    new_mask = mask | fill
    return Batch(views=views, view_mask=new_mask, labels=batch.labels.clone(), sample_ids=batch.sample_ids.clone())


def summarise_table(reports: dict, systems, levels, level_name: str, label) -> list[str]:
    """Markdown rows: one table per metric, mean +/- std per (level, system)."""
    lines = []
    for metric in METRICS:
        lines += [
            f"### {metric}",
            "",
            f"| {level_name} | " + " | ".join(systems) + " |",
            "|---|" + "---|" * len(systems),
        ]
        for level in levels:
            cells = []
            for system in systems:
                entry = reports[(level, system)]["summary"][metric]
                mean, std = entry["mean"], entry["std"]
                cells.append("n/a" if math.isnan(mean) else f"{mean:.4f} +/- {std:.4f}")
            lines.append(f"| {label(level)} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def comparison_rows(reports: dict, a: str, b: str, levels, label, seeds) -> list[str]:
    """a vs b per level: mean +/- std for each, whether mean +/- std bars overlap, paired diff."""
    lines = [
        f"| level | metric | {a} | {b} | bars overlap | paired {a} - {b} | seeds lower |",
        "|---|---|---|---|---|---|---|",
    ]
    for level in levels:
        for metric in ("accuracy", "prob_ece", "brier_reliability", "brier", "prob_mean_confidence"):
            ea = reports[(level, a)]["summary"][metric]
            eb = reports[(level, b)]["summary"][metric]
            overlap = abs(ea["mean"] - eb["mean"]) <= ea["std"] + eb["std"]
            ra = {row["seed"]: row[metric] for row in reports[(level, a)]["per_seed"]}
            rb = {row["seed"]: row[metric] for row in reports[(level, b)]["per_seed"]}
            diffs = [ra[s] - rb[s] for s in seeds]
            lines.append(
                f"| {label(level)} | {metric} | {ea['mean']:.4f} +/- {ea['std']:.4f} | "
                f"{eb['mean']:.4f} +/- {eb['std']:.4f} | {'yes' if overlap else '**no**'} | "
                f"{statistics.mean(diffs):+.4f} +/- {statistics.stdev(diffs):.4f} | "
                f"{sum(d < 0 for d in diffs)}/{len(diffs)} |"
            )
    return lines


def main():
    parser = argparse.ArgumentParser(description="Run the Part 08 missing-view experiment.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="smoke test only: 2 epochs -- NOT evidence")
    args = parser.parse_args()

    config, experiment_id = load_config(args.quick)
    logger = get_logger("prismflow.robustness.missing")
    seeds, eval_cfg = config["seeds"], config["evaluation"]
    rates = config["missing"]["rates"]

    datasets, trained = train_models(config, logger)
    systems = ["naive", "prismflow"]
    if config["missing"]["imputation"]:
        systems += ["imputed_naive", "imputed_prismflow"]

    reports = {}
    for rate_index, rate in enumerate(rates):
        def loader(seed, rate=rate, rate_index=rate_index, impute=False):
            dataset, splits = datasets[seed]
            batches = iter_batches(dataset, splits[eval_cfg["split"]], eval_cfg["batch_size"])
            for batch_index, batch in enumerate(batches):
                batch = drop_views(batch, rate, seed=batch_seed(seed, batch_index, rate_index))
                yield impute_missing(batch) if impute else batch

        for system in systems:
            base = system.removeprefix("imputed_")
            impute = system.startswith("imputed_")
            reports[(rate, system)] = evaluate(
                trained[base],
                lambda seed, impute=impute, loader=loader: loader(seed, impute=impute),
                f"{experiment_id}/missing_r{rate:.2f}_{system}",
                seeds,
                results_dir=args.results_dir,
                n_bins=eval_cfg["n_bins"],
                coverages=eval_cfg["coverages"],
                title=f"missing rate {rate:.0%}: {system}",
            )
            s = reports[(rate, system)]["summary"]
            logger.info(
                "rate=%.1f %-18s acc=%.4f ece=%.4f rel=%.4f conf=%.4f vac_conf=%.4f eniv=%.3f",
                rate, system, s["accuracy"]["mean"], s["prob_ece"]["mean"], s["brier_reliability"]["mean"],
                s["prob_mean_confidence"]["mean"], s["vacuity_mean_confidence"]["mean"], s["eniv"]["mean"],
            )

    def label(rate):
        return f"{rate:.0%}"

    lines = [
        "# Missing-view experiment",
        "",
        f"Seeds: {seeds}. Models trained on clean data; views dropped at test time. "
        "Mean +/- sample std across seeds, from `metrics.json` in each "
        f"`results/{experiment_id}/missing_r<rate>_<system>/`. `prob_` = top-label probability; "
        "`vacuity_mean_confidence` = 1 - mean fused uncertainty.",
        "",
        "## All metrics",
        "",
        *summarise_table(reports, systems, rates, "missing rate", label),
        "## prismflow vs naive",
        "",
        "`bars overlap` compares mean +/- std intervals. 'no' is the only case that may be "
        "reported as a difference.",
        "",
        *comparison_rows(reports, "prismflow", "naive", rates, label, seeds),
        "",
    ]
    if config["missing"]["imputation"]:
        lines += [
            "## imputed_naive vs naive",
            "",
            *comparison_rows(reports, "imputed_naive", "naive", rates, label, seeds),
            "",
            "## imputed_prismflow vs imputed_naive",
            "",
            *comparison_rows(reports, "imputed_prismflow", "imputed_naive", rates, label, seeds),
            "",
        ]

    out_dir = Path(args.results_dir) / experiment_id
    (out_dir / "missing_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", out_dir / "missing_summary.md")


if __name__ == "__main__":
    main()
