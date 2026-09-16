"""The clone experiment: does a duplicated view manufacture confidence?

Start from `base_views` views, append k bit-identical copies of one view, and
compare fused confidence between naive Dempster fusion and PrismFlow's
ENIV-discounted fusion. A copy carries zero new information, so any confidence
it adds is fabricated.

DESIGN DECISIONS (fixed before any clone run was executed)
----------------------------------------------------------
A model's encoder count is fixed at construction, so every k gets its own
(base_views + k)-view model, trained on the duplicated data. Each copy has its
own encoder over identical input -- what a duplicated sensor feed looks like
to a real multi-view system -- so the estimator has to recognise redundancy
across differently-trained feature bases, not merely spot identical tensors.

Each system is trained SEPARATELY from the same seed, with its own
use_discount flag. PrismFlow is therefore evaluated as it would be deployed.

A third, diagnostic-only system evaluates the NAIVE model's weights with the
discount switched on at inference. Comparing it against both primaries
separates two effects that the headline comparison conflates: what the
discount operator does, and whether training under the discount teaches the
model to compensate (e.g. by emitting more evidence to win back confidence).

Usage:
    python -m experiments.clone.run_clone
    python -m experiments.clone.run_clone --quick --results-dir <scratch>
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from prismflow.data.corruption import duplicate_view
from prismflow.data.dataset import MultiViewDataset, split_indices
from prismflow.data.loaders import iter_batches
from prismflow.data.synthetic import SyntheticConfig, analytic_n_eff_eigen
from prismflow.eniv.eniv import eigen_n_eff
from prismflow.train import TrainConfig, build_model, compute_loss
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

CONFIG_PATH = Path(__file__).with_name("config.yaml")

METRICS = (
    "accuracy",
    "mean_confidence",
    "mean_uncertainty",
    "eniv",
    "mean_dependence",
    "alpha",
    "dependence_clone_pairs",
    "dependence_distinct_pairs",
)

SYSTEMS = ("naive", "prismflow", "naive_weights_discounted")


def build_dataset(rho: float, seed: int, base_views: int, split_seed: int):
    dataset = MultiViewDataset(SyntheticConfig(n_views=base_views, rho=rho, seed=seed))
    return dataset, split_indices(len(dataset), seed=split_seed)


def train(dataset, splits, n_views_total, k, source_view, seed, use_discount, cfg):
    """Train one (base_views + k)-view model on data with k copies of the source."""
    train_config = TrainConfig(
        n_views=n_views_total,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        anneal_epochs=cfg["anneal_epochs"],
        per_view_loss_weight=cfg["per_view_loss_weight"],
        use_discount=use_discount,
    )

    # Same seed for both systems -> identical initial weights; the only
    # difference between them is the use_discount flag.
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

    return model


def _weighted_nanmean(matrices, weights):
    stacked = np.stack(matrices)
    w = np.asarray(weights, dtype=np.float64)[:, None, None]
    present = ~np.isnan(stacked)
    total = np.where(present, stacked, 0.0) * w
    denom = present * w
    with np.errstate(invalid="ignore"):
        return total.sum(axis=0) / denom.sum(axis=0)


def _pair_mean(matrix, pairs):
    values = [matrix[i, j] for i, j in pairs if not np.isnan(matrix[i, j])]
    return float(np.mean(values)) if values else float("nan")


@torch.no_grad()
def evaluate(model, dataset, indices, k, source_view, base_views, batch_size, discount):
    """Evaluate with the discount path forced on or off.

    With discount=False no ENIV is computed, so those fields are NaN.
    """
    model.eval()
    previous = model.use_discount
    model.use_discount = discount
    try:
        correct = total = 0
        confidence = uncertainty = 0.0
        eniv = dependence = alpha = 0.0
        matrices, weights = [], []

        for batch in iter_batches(dataset, indices, batch_size):
            batch = duplicate_view(batch, source_view, k)
            output = model(batch.views, batch.view_mask)
            n = len(batch)

            correct += int((output.prediction == batch.labels).sum())
            total += n
            confidence += float(output.confidence.sum())
            uncertainty += float(output.uncertainty.sum())

            if discount:
                eniv += output.eniv.effective_views * n
                dependence += output.eniv.mean_dependence * n
                alpha += output.eniv.efficiency_ratio * n
                matrices.append(output.dependence_matrix.cpu().numpy())
                weights.append(n)
    finally:
        model.use_discount = previous

    result = {
        "accuracy": correct / total,
        "mean_confidence": confidence / total,
        "mean_uncertainty": uncertainty / total,
    }

    if discount:
        matrix = _weighted_nanmean(matrices, weights)
        clone_group = [source_view] + list(range(base_views, base_views + k))
        clone_pairs = [
            (a, b) for i, a in enumerate(clone_group) for b in clone_group[i + 1 :]
        ]
        distinct_pairs = [(i, j) for i in range(base_views) for j in range(i + 1, base_views)]
        result.update(
            eniv=eniv / total,
            mean_dependence=dependence / total,
            alpha=alpha / total,
            dependence_clone_pairs=_pair_mean(matrix, clone_pairs),
            dependence_distinct_pairs=_pair_mean(matrix, distinct_pairs),
        )
    else:
        result.update({m: float("nan") for m in METRICS if m not in result})

    return result


def population_eniv(rho, k, base_views, source_view):
    """Eigen ENIV of the exact population dependence structure for this cell.

    Base views equicorrelated at rho, plus k perfect copies of the source.
    At rho=0 this equals the base truth for every k; at rho>0 it does not,
    because copies of a view that already correlates with the others are not
    fully absorbed by the eigenvalue form.
    """
    size = base_views + k
    matrix = np.full((size, size), rho)
    np.fill_diagonal(matrix, 1.0)
    group = [source_view] + list(range(base_views, size))
    for a in group:
        for b in group:
            matrix[a, b] = 1.0
    return eigen_n_eff(matrix)


def run_cell(rho, k, seed, config, logger):
    base_views = config["base_views"]
    source_view = config["source_view"]
    train_cfg = config["training"]
    eval_batch = config["evaluation"]["batch_size"]
    split_seed = TrainConfig().split_seed

    dataset, splits = build_dataset(rho, seed, base_views, split_seed)
    test_idx = splits[config["evaluation"]["split"]]
    n_views_total = base_views + k

    started = time.perf_counter()
    naive_model = train(dataset, splits, n_views_total, k, source_view, seed, False, train_cfg)
    prismflow_model = train(dataset, splits, n_views_total, k, source_view, seed, True, train_cfg)

    def score(model, discount):
        return evaluate(model, dataset, test_idx, k, source_view, base_views, eval_batch, discount)

    naive = score(naive_model, discount=False)
    naive_discounted = score(naive_model, discount=True)
    prismflow = score(prismflow_model, discount=True)

    # The naive system ignores ENIV, but what the estimator WOULD measure on
    # its features is still the right diagnostic for its row.
    for key in ("eniv", "mean_dependence", "dependence_clone_pairs", "dependence_distinct_pairs"):
        naive[key] = naive_discounted[key]
    naive["alpha"] = 1.0

    truth = {
        "eniv_truth_base": analytic_n_eff_eigen(rho, base_views),
        "eniv_truth_population": population_eniv(rho, k, base_views, source_view),
    }

    rows = []
    for system, metrics in (
        ("naive", naive),
        ("prismflow", prismflow),
        ("naive_weights_discounted", naive_discounted),
    ):
        rows.append(
            {"rho": rho, "k": k, "n_views": n_views_total, "seed": seed, "system": system,
             **metrics, **truth}
        )

    logger.info(
        "rho=%.1f k=%d seed=%d  conf naive=%.3f prismflow=%.3f  acc naive=%.3f prismflow=%.3f  "
        "eniv=%.2f  dep(clones)=%.2f  (%.0fs)",
        rho, k, seed,
        naive["mean_confidence"], prismflow["mean_confidence"],
        naive["accuracy"], prismflow["accuracy"],
        prismflow["eniv"], prismflow["dependence_clone_pairs"],
        time.perf_counter() - started,
    )
    return rows


def summarise(rows):
    """Mean and sample std (ddof=1) across seeds for every condition and metric."""
    summary = []
    keys = sorted({(r["rho"], r["k"], r["system"]) for r in rows})
    for rho, k, system in keys:
        cell = [r for r in rows if (r["rho"], r["k"], r["system"]) == (rho, k, system)]
        entry = {"rho": rho, "k": k, "n_views": cell[0]["n_views"], "system": system, "n_seeds": len(cell)}
        for metric in METRICS:
            values = [r[metric] for r in cell if not np.isnan(r[metric])]
            entry[f"{metric}_mean"] = statistics.mean(values) if values else float("nan")
            entry[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else float("nan")
        summary.append(entry)
    return summary


def render_markdown(summary, config):
    lines = [
        "# Clone experiment",
        "",
        f"Seeds: {config['seeds']}. Values are mean +/- sample std across seeds.",
        "",
    ]
    for rho in config["rhos"]:
        lines += [
            f"## rho = {rho}",
            "",
            "| k | views | system | accuracy | confidence | uncertainty | ENIV | alpha | dep(clones) | dep(distinct) |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for entry in summary:
            if entry["rho"] != rho:
                continue

            def fmt(metric):
                mean, std = entry[f"{metric}_mean"], entry[f"{metric}_std"]
                if np.isnan(mean):
                    return "n/a"
                return f"{mean:.3f} +/- {std:.3f}" if not np.isnan(std) else f"{mean:.3f}"

            lines.append(
                f"| {entry['k']} | {entry['n_views']} | {entry['system']} | {fmt('accuracy')} | "
                f"{fmt('mean_confidence')} | {fmt('mean_uncertainty')} | {fmt('eniv')} | "
                f"{fmt('alpha')} | {fmt('dependence_clone_pairs')} | {fmt('dependence_distinct_pairs')} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the clone experiment.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="output subdirectory; defaults to config.yaml's id. Use a new id to "
        "re-run without overwriting a result an earlier write-up cites.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="smoke test only: 1 seed, 2 epochs -- NOT evidence, never report",
    )
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_id = args.experiment_id or config["experiment"]["id"]

    if args.quick:
        config["seeds"] = [0]
        config["rhos"] = [0.0]
        config["duplicates"] = [0, 2]
        config["training"]["epochs"] = 2
        experiment_id = f"{experiment_id}_SMOKE_TEST_NOT_EVIDENCE"
    elif len(config["seeds"]) < 5:
        raise ValueError("contract requires at least 5 seeds")

    logger = get_logger("prismflow.clone")
    rows = []
    for rho in config["rhos"]:
        for k in config["duplicates"]:
            for seed in config["seeds"]:
                rows.extend(run_cell(rho, k, seed, config, logger))

    out_dir = Path(args.results_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = summarise(rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"config": config, "summary": summary}, indent=2), encoding="utf-8"
    )
    markdown = render_markdown(summary, config)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")

    logger.info("wrote %s", out_dir)
    print()
    print(markdown)


if __name__ == "__main__":
    main()
