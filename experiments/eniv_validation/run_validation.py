"""Does the ENIV estimator recover a quantity we already know?

This is the experiment Part 02 exists to make possible. The synthetic
generator is built around a rho we choose, so `analytic_n_eff(rho, V)` is not
a baseline to beat -- it is the right answer, known in advance. If the
estimator does not track it, every downstream claim in this project is
unfalsifiable, and the honest move is to stop rather than proceed to Part 06.

Three estimates are reported side by side, because "the estimator failed" and
"the data did not contain what we thought" are different failures with
different fixes:

  view-space (CCA)   dependence measured on the RAW generated views, after
                     centring by true label. This never touches the model, so
                     it isolates generator fidelity: if this does not track
                     rho, the data is wrong, not the estimator.

  evidence-space     dependence measured on a trained model's per-view
                     evidence -- the quantity the deployed system actually
                     discounts on. This is the number that matters.

  conditioning modes both "global" and "pairwise_holdout", since the collider
                     bias in the former is large enough to move ENIV (see
                     prismflow/statistics/dependence.py).

Usage:
    python -m experiments.eniv_validation.run_validation
    python -m experiments.eniv_validation.run_validation --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from prismflow.data.dataset import MultiViewDataset, split_indices
from prismflow.data.loaders import iter_batches
from prismflow.data.synthetic import (
    SyntheticConfig,
    analytic_n_eff,
    empirical_cross_view_correlation,
)
from prismflow.eniv.eniv import compute_eniv, mean_off_diagonal
from prismflow.statistics.dependence import dependence_matrix
from prismflow.train import TrainConfig, build_model, compute_loss
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

RHO_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
SEEDS = (0, 1, 2, 3, 4)
N_VIEWS = 4

ESTIMATORS = (
    ("pearson_global", "pearson", "global"),
    ("pearson_holdout", "pearson", "pairwise_holdout"),
    ("dcor_global", "dcor", "global"),
    ("dcor_holdout", "dcor", "pairwise_holdout"),
)


def train_model(config: TrainConfig, seed: int):
    """Train a naive (undiscounted) model and return it with its data.

    Discounting stays off: the estimator is being validated against the
    evidence a plain model produces, and feeding ENIV back into training
    during its own validation would be circular.
    """
    set_seed(seed)

    dataset = MultiViewDataset(
        SyntheticConfig(
            n_views=config.n_views,
            n_classes=config.n_classes,
            n_samples=config.n_samples,
            d_latent=config.d_latent,
            d_view=config.d_view,
            rho=config.rho,
            noise_std=config.noise_std,
            signal_strength=config.signal_strength,
            seed=seed,
        )
    )
    splits = split_indices(len(dataset), seed=config.split_seed)

    model = build_model(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    for epoch in range(config.epochs):
        model.train()
        for batch in iter_batches(
            dataset, splits["train"], config.batch_size, shuffle=True, seed=seed + epoch
        ):
            targets = torch.nn.functional.one_hot(
                batch.labels, num_classes=config.n_classes
            ).to(batch.views.dtype)

            output = model(batch.views, batch.view_mask)
            loss = compute_loss(
                output,
                targets,
                epoch=epoch,
                anneal_epochs=config.anneal_epochs,
                per_view_loss_weight=config.per_view_loss_weight,
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, dataset, splits


@torch.no_grad()
def collect_evidence(model, dataset, indices, batch_size):
    model.eval()
    evidence, masks, labels = [], [], []
    for batch in iter_batches(dataset, indices, batch_size):
        output = model(batch.views, batch.view_mask)
        evidence.append(output.per_view_evidence)
        masks.append(output.view_mask)
        labels.append(batch.labels)
    return torch.cat(evidence), torch.cat(masks), torch.cat(labels)


def view_space_reference(dataset) -> tuple[float, float]:
    """Dependence measured on raw views, centred by true label.

    Uses the frozen Part 02 canonical-correlation routine, which is
    projection-invariant -- necessary here because each view is seen through
    its own random A_v, so axis-aligned correlation would understate the
    shared structure.

    Computed over the WHOLE dataset, not the test split: this measures a
    property of the generated data rather than of the model, so there is no
    held-out set to respect, and canonical correlation is badly upward-biased
    at small n (it maximises over projections). Even at n=2000 against d=16
    expect some inflation -- read this column as a loose sanity check on the
    generator, not as a precise second estimate.
    """
    views = dataset.views.numpy()
    labels = dataset.labels.numpy()

    centered = views.copy()
    for stratum in np.unique(labels):
        rows = labels == stratum
        centered[rows] -= centered[rows].mean(axis=0, keepdims=True)

    matrix = empirical_cross_view_correlation(centered)
    return mean_off_diagonal(matrix), compute_eniv(matrix).effective_views


def run_cell(rho: float, seed: int, config: TrainConfig, logger) -> dict:
    cell_config = TrainConfig(**{**config.__dict__, "rho": rho})
    model, dataset, splits = train_model(cell_config, seed)

    evidence, masks, labels = collect_evidence(
        model, dataset, splits["test"], cell_config.batch_size
    )

    row = {
        "rho": rho,
        "seed": seed,
        "n_views": cell_config.n_views,
        "analytic_n_eff": analytic_n_eff(rho, cell_config.n_views),
    }

    for name, method, conditioning in ESTIMATORS:
        matrix = dependence_matrix(
            evidence, masks, method=method, conditioning=conditioning, seed=seed
        )
        result = compute_eniv(matrix)
        row[f"eniv_{name}"] = result.effective_views
        row[f"rho_bar_{name}"] = result.mean_dependence

    view_rho, view_eniv = view_space_reference(dataset)
    row["rho_bar_views_cca"] = view_rho
    row["eniv_views_cca"] = view_eniv

    correct = 0
    with torch.no_grad():
        for batch in iter_batches(dataset, splits["test"], cell_config.batch_size):
            correct += int((model(batch.views, batch.view_mask).prediction == batch.labels).sum())
    row["accuracy"] = correct / len(splits["test"])

    logger.info(
        "rho=%.2f seed=%d  acc=%.3f  analytic=%.3f  pearson_holdout=%.3f  views_cca=%.3f",
        rho,
        seed,
        row["accuracy"],
        row["analytic_n_eff"],
        row["eniv_pearson_holdout"],
        row["eniv_views_cca"],
    )
    return row


def _aggregate(rows: list[dict], key: str) -> dict:
    """MAE against the analytic value, plus the spread across all cells."""
    errors = [abs(row[key] - row["analytic_n_eff"]) for row in rows]
    estimates = [row[key] for row in rows]
    truth = [row["analytic_n_eff"] for row in rows]

    centered_est = [value - statistics.mean(estimates) for value in estimates]
    centered_truth = [value - statistics.mean(truth) for value in truth]
    numerator = sum(a * b for a, b in zip(centered_est, centered_truth))
    denominator = (
        sum(a * a for a in centered_est) * sum(b * b for b in centered_truth)
    ) ** 0.5

    return {
        "mae": statistics.mean(errors),
        "mae_std": statistics.stdev(errors) if len(errors) > 1 else 0.0,
        "correlation_with_truth": numerator / denominator if denominator > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ENIV against analytic n_eff.")
    parser.add_argument("--quick", action="store_true", help="fewer epochs, for smoke testing")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    logger = get_logger("prismflow.eniv_validation")

    config = TrainConfig(
        experiment_id="eniv_validation",
        n_views=N_VIEWS,
        epochs=8 if args.quick else 40,
        n_samples=600 if args.quick else 2000,
    )

    rows = [
        run_cell(rho, seed, config, logger) for rho in RHO_GRID for seed in SEEDS
    ]

    out_dir = Path(args.results_dir) / "eniv_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "estimator_vs_truth.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    estimator_keys = [f"eniv_{name}" for name, _, _ in ESTIMATORS] + ["eniv_views_cca"]
    summary = {key: _aggregate(rows, key) for key in estimator_keys}
    summary["per_rho"] = {
        str(rho): {
            key: statistics.mean([r[key] for r in rows if r["rho"] == rho])
            for key in estimator_keys + ["analytic_n_eff", "accuracy"]
        }
        for rho in RHO_GRID
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("wrote %s", csv_path)

    print()
    print(f"{'estimator':24s} {'MAE':>8s} {'std':>8s} {'corr w/ truth':>14s}")
    print("-" * 58)
    for key in estimator_keys:
        stats = summary[key]
        print(
            f"{key:24s} {stats['mae']:8.3f} {stats['mae_std']:8.3f} "
            f"{stats['correlation_with_truth']:14.3f}"
        )

    print()
    header = f"{'rho':>6s} {'analytic':>9s} " + " ".join(f"{k.replace('eniv_',''):>16s}" for k in estimator_keys)
    print(header)
    print("-" * len(header))
    for rho in RHO_GRID:
        cells = summary["per_rho"][str(rho)]
        line = f"{rho:6.2f} {cells['analytic_n_eff']:9.3f} "
        line += " ".join(f"{cells[k]:16.3f}" for k in estimator_keys)
        print(line)


if __name__ == "__main__":
    main()
