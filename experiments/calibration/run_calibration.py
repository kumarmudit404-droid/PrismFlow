"""The calibration experiment: is PrismFlow's confidence more honest?

Three systems on identical data and seeds, every metric computed by
`prismflow.evaluation.evaluate` (the Part 07 protocol) and nothing else:

  naive                     trained and evaluated with use_discount=False
  prismflow                 trained and evaluated with use_discount=True
  naive_weights_discounted  a copy of the naive model, discount on at test time

The two trained systems start from the same seed, so their initial weights
are identical and the only difference is the use_discount flag. The third
separates what the discount operator does from what training under it teaches.

Outputs go to results/<experiment_id>/<system>/ (metrics.json, metrics.csv,
reliability_diagram.png), plus a side-by-side summary.md that only reads the
summaries evaluate() returned.

Usage:
    python -m experiments.calibration.run_calibration
    python -m experiments.calibration.run_calibration --quick --results-dir <scratch>
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

from prismflow.data.dataset import MultiViewDataset, split_indices
from prismflow.data.loaders import iter_batches
from prismflow.data.synthetic import SyntheticConfig
from prismflow.evaluation import evaluate
from prismflow.train import TrainConfig, build_model, compute_loss
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

CONFIG_PATH = Path(__file__).with_name("config.yaml")

SYSTEMS = ("naive", "prismflow", "naive_weights_discounted")

# Rows of summary.md, in reading order. Keys are evaluate() metric names.
REPORTED = (
    "accuracy",
    "macro_f1",
    "prob_ece",
    "prob_mce",
    "brier",
    "brier_reliability",
    "brier_resolution",
    "brier_uncertainty",
    "brier_within_bin",
    "prob_aurc",
    "prob_selective_risk@1.00",
    "prob_selective_risk@0.90",
    "prob_selective_risk@0.80",
    "prob_selective_risk@0.50",
    "vacuity_mean_confidence",
    "vacuity_ece",
    "vacuity_aurc",
    "eniv",
    "efficiency_ratio",
)

# Paired within-seed differences against naive, for the terms the claim rests on.
PAIRED = ("prob_ece", "brier", "brier_reliability", "brier_resolution", "prob_aurc", "accuracy")


def build_dataset(seed: int, data_cfg: dict):
    dataset = MultiViewDataset(SyntheticConfig(n_views=data_cfg["n_views"], rho=data_cfg["rho"], seed=seed))
    return dataset, split_indices(len(dataset), seed=TrainConfig().split_seed)


def train(dataset, splits, seed: int, use_discount: bool, data_cfg: dict, cfg: dict):
    train_config = TrainConfig(
        n_views=data_cfg["n_views"],
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


def _fmt(entry: dict | None) -> str:
    if entry is None or math.isnan(entry["mean"]):
        return "n/a"
    mean, std = entry["mean"], entry["std"]
    return f"{mean:.4f}" if math.isnan(std) else f"{mean:.4f} +/- {std:.4f}"


def render_markdown(reports: dict, config: dict, experiment_id: str) -> str:
    seeds = config["seeds"]
    lines = [
        "# Calibration experiment",
        "",
        f"Seeds: {seeds}. Mean +/- sample std across seeds, from each system's "
        f"`results/{experiment_id}/<system>/metrics.json`. `prob_` metrics use the "
        "top-label probability; `vacuity_` metrics use 1 - uncertainty.",
        "",
        "| metric | " + " | ".join(SYSTEMS) + " |",
        "|---|" + "---|" * len(SYSTEMS),
    ]
    for metric in REPORTED:
        cells = [_fmt(reports[s]["summary"].get(metric)) for s in SYSTEMS]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Paired within-seed differences (system minus naive)",
        "",
        "Same seed means same data and, for prismflow, same initial weights.",
        "",
        "| metric | prismflow - naive | seeds lower | naive_weights_discounted - naive | seeds lower |",
        "|---|---|---|---|---|",
    ]
    naive_rows = {row["seed"]: row for row in reports["naive"]["per_seed"]}
    for metric in PAIRED:
        cells = []
        for system in ("prismflow", "naive_weights_discounted"):
            diffs = [row[metric] - naive_rows[row["seed"]][metric] for row in reports[system]["per_seed"]]
            mean = statistics.mean(diffs)
            std = statistics.stdev(diffs) if len(diffs) > 1 else float("nan")
            cells += [f"{mean:+.4f} +/- {std:.4f}", f"{sum(d < 0 for d in diffs)}/{len(diffs)}"]
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the calibration experiment.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-id", default=None, help="defaults to config.yaml's id")
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

    logger = get_logger("prismflow.calibration")
    seeds = config["seeds"]
    eval_cfg = config["evaluation"]

    datasets, models = {}, {system: {} for system in SYSTEMS}
    for seed in seeds:
        started = time.perf_counter()
        dataset, splits = build_dataset(seed, config["data"])
        datasets[seed] = (dataset, splits[eval_cfg["split"]])

        naive = train(dataset, splits, seed, False, config["data"], config["training"])
        discounted = copy.deepcopy(naive)
        discounted.use_discount = True
        models["naive"][seed] = naive
        models["naive_weights_discounted"][seed] = discounted
        models["prismflow"][seed] = train(dataset, splits, seed, True, config["data"], config["training"])
        logger.info("seed %d trained (%.0fs)", seed, time.perf_counter() - started)

    def loader(seed):
        dataset, indices = datasets[seed]
        return iter_batches(dataset, indices, eval_cfg["batch_size"])

    reports = {}
    for system in SYSTEMS:
        reports[system] = evaluate(
            models[system],
            loader,
            f"{experiment_id}/{system}",
            seeds,
            results_dir=args.results_dir,
            n_bins=eval_cfg["n_bins"],
            coverages=eval_cfg["coverages"],
            title=f"{experiment_id}: {system}",
        )
        logger.info("wrote %s", reports[system]["output_dir"])

    out_dir = Path(args.results_dir) / experiment_id
    markdown = render_markdown(reports, config, experiment_id)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    (out_dir / "config_used.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print()
    print(markdown)


if __name__ == "__main__":
    main()
