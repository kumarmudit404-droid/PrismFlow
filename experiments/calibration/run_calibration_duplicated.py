"""Calibration under duplication: does PrismFlow stay honest when a view is cloned?

Same three systems and base data as run_calibration.py, now with k
bit-identical copies of one view appended (the clone experiment's duplication
logic). A copy carries no new information, so naive fusion should grow
overconfident as k rises; the discount should prevent that.

For each k, every seed trains a (base_views + k)-view naive model and a
PrismFlow model from the same seed (identical initial weights). The third
system is a copy of the naive model with the discount on at test time only.
Duplication is applied identically in training and in the evaluation batches.
Every metric comes from `prismflow.evaluation.evaluate`.

Outputs: results/<experiment_id>/k<k>_<system>/ (metrics.json, metrics.csv,
reliability_diagram.png), plus summary.md, which only reads the reports
evaluate() returned. The flat k<k>_<system> naming keeps outputs within the
two directory levels .gitignore tracks.

k = 0 takes exactly the code path of run_calibration.py, so its numbers must
reproduce results/calibration/.

Usage:
    python -m experiments.calibration.run_calibration_duplicated
    python -m experiments.calibration.run_calibration_duplicated --quick --results-dir <scratch>
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from pathlib import Path

import torch
import yaml

from experiments.calibration.run_calibration import SYSTEMS, build_dataset
from prismflow.data.corruption import duplicate_view
from prismflow.data.loaders import iter_batches
from prismflow.evaluation import evaluate
from prismflow.train import TrainConfig, build_model, compute_loss
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

CONFIG_PATH = Path(__file__).with_name("config_duplicated.yaml")

# Rows of the per-k tables, in reading order. Keys are evaluate() metric names.
REPORTED = (
    "accuracy",
    "prob_mean_confidence",
    "prob_ece",
    "prob_mce",
    "brier",
    "brier_reliability",
    "brier_resolution",
    "brier_uncertainty",
    "brier_within_bin",
    "prob_aurc",
    "vacuity_mean_confidence",
    "vacuity_ece",
    "vacuity_aurc",
    "eniv",
    "efficiency_ratio",
)

PAIRED = (
    "prob_ece",
    "brier",
    "brier_reliability",
    "brier_resolution",
    "prob_aurc",
    "prob_mean_confidence",
    "accuracy",
)


def train(dataset, splits, k, source_view, seed, use_discount, data_cfg, cfg):
    """Train one (base_views + k)-view model on data with k copies of the source."""
    train_config = TrainConfig(
        n_views=data_cfg["n_views"] + k,
        rho=data_cfg["rho"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        anneal_epochs=cfg["anneal_epochs"],
        per_view_loss_weight=cfg["per_view_loss_weight"],
        use_discount=use_discount,
    )

    # Same seed for both trained systems -> identical initial weights.
    set_seed(seed)
    model = build_model(train_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config.lr)

    for epoch in range(train_config.epochs):
        model.train()
        for batch in iter_batches(
            dataset, splits["train"], train_config.batch_size, shuffle=True, seed=seed + epoch
        ):
            batch = duplicate_view(batch, source_view, k)
            targets = torch.nn.functional.one_hot(
                batch.labels, num_classes=train_config.n_classes
            ).to(batch.views.dtype)
            output = model(batch.views, batch.view_mask)
            loss = compute_loss(
                output,
                targets,
                epoch=epoch,
                anneal_epochs=train_config.anneal_epochs,
                per_view_loss_weight=train_config.per_view_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def _stats(values):
    values = [v for v in values if not math.isnan(v)]
    if not values:
        return float("nan"), float("nan")
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else float("nan"))


def _fmt(mean, std, signed=False):
    if math.isnan(mean):
        return "n/a"
    head = f"{mean:+.4f}" if signed else f"{mean:.4f}"
    return head if math.isnan(std) else f"{head} +/- {std:.4f}"


def _per_seed(report, metric):
    return {row["seed"]: row[metric] for row in report["per_seed"]}


def render_markdown(reports, config, experiment_id):
    """reports[k][system] -> the dict evaluate() returned."""
    ks = config["duplicates"]
    lines = [
        "# Calibration under duplication",
        "",
        f"Seeds: {config['seeds']}. Base: {config['data']['n_views']} views, rho = "
        f"{config['data']['rho']}; k copies of view {config['source_view']}. Mean +/- sample "
        f"std across seeds, from `results/{experiment_id}/k<k>_<system>/metrics.json`. "
        "`prob_` = top-label probability; `vacuity_` = 1 - uncertainty.",
        "",
    ]

    for k in ks:
        lines += [
            f"## k = {k} ({config['data']['n_views'] + k} views)",
            "",
            "| metric | " + " | ".join(SYSTEMS) + " |",
            "|---|" + "---|" * len(SYSTEMS),
        ]
        for metric in REPORTED:
            cells = []
            for system in SYSTEMS:
                entry = reports[k][system]["summary"][metric]
                cells.append(_fmt(entry["mean"], entry["std"]))
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")

        lines += [
            "",
            "Paired within-seed differences (system minus naive):",
            "",
            "| metric | prismflow - naive | seeds lower | naive_weights_discounted - naive | seeds lower |",
            "|---|---|---|---|---|",
        ]
        for metric in PAIRED:
            base = _per_seed(reports[k]["naive"], metric)
            cells = []
            for system in ("prismflow", "naive_weights_discounted"):
                other = _per_seed(reports[k][system], metric)
                diffs = [other[s] - base[s] for s in config["seeds"]]
                cells += [_fmt(*_stats(diffs), signed=True), f"{sum(d < 0 for d in diffs)}/{len(diffs)}"]
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")

    # How each system's metric moves as copies are added, within seed.
    lines += [
        f"## Change from k = {ks[0]}, within seed",
        "",
        "Same seed means same base data at every k.",
        "",
        "| metric | system | " + " | ".join(f"k={k} - k={ks[0]}" for k in ks[1:]) + " |",
        "|---|---|" + "---|" * (len(ks) - 1),
    ]
    for metric in ("prob_ece", "brier_reliability", "brier_resolution", "prob_mean_confidence", "accuracy"):
        for system in SYSTEMS:
            base = _per_seed(reports[ks[0]][system], metric)
            cells = []
            for k in ks[1:]:
                other = _per_seed(reports[k][system], metric)
                diffs = [other[s] - base[s] for s in config["seeds"]]
                cells.append(f"{_fmt(*_stats(diffs), signed=True)} ({sum(d > 0 for d in diffs)}/{len(diffs)} up)")
            lines.append(f"| {metric} | {system} | " + " | ".join(cells) + " |")
    lines.append("")

    # Difference-in-differences: does the naive-vs-prismflow gap widen with k?
    lines += [
        "## Gap growth: (prismflow - naive) at k minus the same at k = 0, within seed",
        "",
        "| metric | " + " | ".join(f"k={k}" for k in ks[1:]) + " |",
        "|---|" + "---|" * (len(ks) - 1),
    ]
    for metric in ("prob_ece", "brier", "brier_reliability", "brier_resolution", "prob_aurc"):
        def gap(k):
            p, n = _per_seed(reports[k]["prismflow"], metric), _per_seed(reports[k]["naive"], metric)
            return {s: p[s] - n[s] for s in config["seeds"]}

        g0 = gap(ks[0])
        cells = []
        for k in ks[1:]:
            gk = gap(k)
            diffs = [gk[s] - g0[s] for s in config["seeds"]]
            cells.append(f"{_fmt(*_stats(diffs), signed=True)} ({sum(d < 0 for d in diffs)}/{len(diffs)} lower)")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the calibration-under-duplication experiment.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-id", default=None, help="defaults to config_duplicated.yaml's id")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test only: 5 seeds, 2 epochs -- NOT evidence, never report",
    )
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_id = args.experiment_id or config["experiment"]["id"]
    if args.quick:
        config["training"]["epochs"] = 2
        experiment_id = f"{experiment_id}_SMOKE_TEST_NOT_EVIDENCE"
    if len(config["seeds"]) < 5:
        raise ValueError("contract requires at least 5 seeds")

    logger = get_logger("prismflow.calibration_duplicated")
    seeds = config["seeds"]
    eval_cfg = config["evaluation"]
    source_view = config["source_view"]

    datasets = {}
    for seed in seeds:
        dataset, splits = build_dataset(seed, config["data"])
        datasets[seed] = (dataset, splits)

    # evaluate() writes each k's outputs as soon as that k finishes, so a run
    # that dies part-way still leaves every completed k on disk.
    reports = {}
    for k in config["duplicates"]:
        models = {system: {} for system in SYSTEMS}
        for seed in seeds:
            started = time.perf_counter()
            dataset, splits = datasets[seed]
            naive = train(dataset, splits, k, source_view, seed, False, config["data"], config["training"])
            discounted = copy.deepcopy(naive)
            discounted.use_discount = True
            models["naive"][seed] = naive
            models["naive_weights_discounted"][seed] = discounted
            models["prismflow"][seed] = train(
                dataset, splits, k, source_view, seed, True, config["data"], config["training"]
            )
            logger.info("k=%d seed %d trained (%.0fs)", k, seed, time.perf_counter() - started)

        def loader(seed, k=k):
            dataset, splits = datasets[seed]
            for batch in iter_batches(dataset, splits[eval_cfg["split"]], eval_cfg["batch_size"]):
                yield duplicate_view(batch, source_view, k)

        reports[k] = {}
        for system in SYSTEMS:
            reports[k][system] = evaluate(
                models[system],
                loader,
                f"{experiment_id}/k{k}_{system}",
                seeds,
                results_dir=args.results_dir,
                n_bins=eval_cfg["n_bins"],
                coverages=eval_cfg["coverages"],
                title=f"{experiment_id}: k={k}, {system}",
            )
            summary = reports[k][system]["summary"]
            logger.info(
                "k=%d %-24s ece=%.4f reliability=%.4f resolution=%.4f aurc=%.4f",
                k, system, summary["prob_ece"]["mean"], summary["brier_reliability"]["mean"],
                summary["brier_resolution"]["mean"], summary["prob_aurc"]["mean"],
            )

    out_dir = Path(args.results_dir) / experiment_id
    markdown = render_markdown(reports, config, experiment_id)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    (out_dir / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print()
    print(markdown)


if __name__ == "__main__":
    main()
