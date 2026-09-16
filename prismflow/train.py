"""Training loop for the Part 04 naive baseline.

Runs `encoders -> evidence -> Dempster fusion -> prediction` with no
dependence discounting, over a seed list, and writes per-seed plus aggregated
metrics to `results/<experiment_id>/`.

Per the contract, a run is only evidence if it covers at least 5 seeds and
reports mean and standard deviation; `run_experiment` refuses fewer.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from prismflow.data.dataset import MultiViewDataset, split_indices
from prismflow.data.loaders import iter_batches
from prismflow.data.synthetic import SyntheticConfig
from prismflow.models.encoders import EncoderConfig
from prismflow.models.evidence import edl_loss, evidence_to_alpha
from prismflow.models.prismflow import PrismFlow, PrismFlowOutput
from prismflow.utils.device import get_device
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

MIN_SEEDS = 5

# TrainConfig mirrors the generator's data parameters so one object describes a
# run end to end. Those defaults are READ FROM SyntheticConfig rather than
# retyped: they were duplicated once, and a later recalibration of
# signal_strength silently never reached a single training run because
# TrainConfig still carried the superseded value. Deriving them makes that
# class of drift structurally impossible.
_GENERATOR_DEFAULTS = SyntheticConfig()


@dataclass
class TrainConfig:
    experiment_id: str = "baseline"

    # data -- mirrors SyntheticConfig, see note above
    n_views: int = _GENERATOR_DEFAULTS.n_views
    n_classes: int = _GENERATOR_DEFAULTS.n_classes
    n_samples: int = _GENERATOR_DEFAULTS.n_samples
    d_latent: int = _GENERATOR_DEFAULTS.d_latent
    d_view: int = _GENERATOR_DEFAULTS.d_view
    rho: float = _GENERATOR_DEFAULTS.rho
    noise_std: float = _GENERATOR_DEFAULTS.noise_std
    signal_strength: float = _GENERATOR_DEFAULTS.signal_strength
    split_seed: int = 12345

    # model
    hidden_dims: list = field(default_factory=lambda: [64, 64])
    feature_dim: int = 32
    dropout: float = 0.0
    use_discount: bool = False

    # optimisation
    epochs: int = 40
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    anneal_epochs: int = 10
    per_view_loss_weight: float = 1.0

    device: str = "auto"


def build_model(config: TrainConfig) -> PrismFlow:
    view_configs = [
        EncoderConfig(
            input_dim=config.d_view,
            hidden_dims=list(config.hidden_dims),
            feature_dim=config.feature_dim,
            dropout=config.dropout,
        )
        for _ in range(config.n_views)
    ]
    return PrismFlow(
        view_configs,
        n_classes=config.n_classes,
        use_discount=config.use_discount,
    )


def compute_loss(
    output: PrismFlowOutput,
    targets_onehot: torch.Tensor,
    epoch: int,
    anneal_epochs: int,
    per_view_loss_weight: float,
) -> torch.Tensor:
    """EDL loss on the fused opinion, plus a per-view term.

    The per-view term matters beyond regularisation: Part 05 estimates
    cross-view dependence from per-view evidence, and a head supervised only
    through the fusion is free to emit arbitrary evidence as long as the fused
    result is right. That would leave the dependence estimator measuring an
    artefact of the fusion rather than a property of the views.
    """
    total = edl_loss(output.fused_alpha, targets_onehot, epoch, anneal_epochs)

    if per_view_loss_weight != 0.0:
        per_view_alpha = evidence_to_alpha(output.per_view_evidence)
        per_view_targets = targets_onehot.unsqueeze(1).expand_as(per_view_alpha)
        per_view = edl_loss(
            per_view_alpha, per_view_targets, epoch, anneal_epochs, reduction="none"
        )
        mask = output.view_mask.to(per_view.dtype)
        available = mask.sum().clamp_min(1.0)
        total = total + per_view_loss_weight * (per_view * mask).sum() / available

    return total


@torch.no_grad()
def evaluate(model: PrismFlow, dataset, indices, batch_size: int, device) -> dict:
    model.eval()

    n_correct = 0
    n_total = 0
    confidence_sum = 0.0
    uncertainty_sum = 0.0

    for batch in iter_batches(dataset, indices, batch_size):
        views = batch.views.to(device)
        view_mask = batch.view_mask.to(device)
        labels = batch.labels.to(device)

        output = model(views, view_mask)

        n_correct += int((output.prediction == labels).sum())
        n_total += len(labels)
        confidence_sum += float(output.confidence.sum())
        uncertainty_sum += float(output.uncertainty.sum())

    return {
        "accuracy": n_correct / n_total,
        "mean_confidence": confidence_sum / n_total,
        "mean_uncertainty": uncertainty_sum / n_total,
        "n_samples": n_total,
    }


def train_one_seed(config: TrainConfig, seed: int, logger=None) -> dict:
    """Train a single run end to end and return its test/val metrics."""
    set_seed(seed)
    device = get_device(config.device)

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

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in iter_batches(
            dataset, splits["train"], config.batch_size, shuffle=True, seed=seed + epoch
        ):
            views = batch.views.to(device)
            view_mask = batch.view_mask.to(device)
            labels = batch.labels.to(device)
            targets_onehot = torch.nn.functional.one_hot(
                labels, num_classes=config.n_classes
            ).to(views.dtype)

            output = model(views, view_mask)
            loss = compute_loss(
                output,
                targets_onehot,
                epoch=epoch,
                anneal_epochs=config.anneal_epochs,
                per_view_loss_weight=config.per_view_loss_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.detach())
            n_batches += 1

        if logger is not None and (epoch + 1) % 10 == 0:
            logger.info(
                "seed %d epoch %d/%d  loss=%.4f",
                seed,
                epoch + 1,
                config.epochs,
                epoch_loss / max(n_batches, 1),
            )

    val_metrics = evaluate(model, dataset, splits["val"], config.batch_size, device)
    test_metrics = evaluate(model, dataset, splits["test"], config.batch_size, device)

    return {
        "seed": seed,
        "val": val_metrics,
        "test": test_metrics,
    }


def aggregate(values: list[float]) -> dict:
    """Mean and sample standard deviation (ddof=1) across seeds."""
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def run_experiment(
    config: TrainConfig,
    seeds: list[int],
    results_dir: str | Path = "results",
    logger=None,
) -> dict:
    if len(seeds) < MIN_SEEDS:
        raise ValueError(
            f"contract requires at least {MIN_SEEDS} seeds, got {len(seeds)}. "
            "Single-seed and few-seed runs are not evidence."
        )

    logger = logger or get_logger("prismflow.train")
    per_seed = [train_one_seed(config, seed, logger=logger) for seed in seeds]

    summary = {}
    for split in ("val", "test"):
        summary[split] = {
            metric: aggregate([run[split][metric] for run in per_seed])
            for metric in ("accuracy", "mean_confidence", "mean_uncertainty")
        }

    report = {
        "experiment_id": config.experiment_id,
        "config": asdict(config),
        "seeds": seeds,
        "per_seed": per_seed,
        "summary": summary,
    }

    out_dir = Path(results_dir) / config.experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_render_summary(report), encoding="utf-8")

    logger.info("wrote %s", out_dir / "metrics.json")
    return report


def _render_summary(report: dict) -> str:
    config = report["config"]
    lines = [
        f"# {report['experiment_id']}",
        "",
        "Naive Dempster fusion baseline (no dependence discounting).",
        "",
        f"- views: {config['n_views']}, classes: {config['n_classes']}, rho: {config['rho']}",
        f"- samples: {config['n_samples']}, epochs: {config['epochs']}",
        f"- seeds: {report['seeds']}",
        "",
        "| split | metric | mean | std |",
        "|-------|--------|------|-----|",
    ]
    for split, metrics in report["summary"].items():
        for name, stats in metrics.items():
            lines.append(
                f"| {split} | {name} | {stats['mean']:.4f} | {stats['std']:.4f} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the PrismFlow naive baseline.")
    parser.add_argument("--experiment-id", default="baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--rho", type=float, default=0.3)
    parser.add_argument("--n-views", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--anneal-epochs", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    config = TrainConfig(
        experiment_id=args.experiment_id,
        rho=args.rho,
        n_views=args.n_views,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        anneal_epochs=args.anneal_epochs,
        device=args.device,
    )

    report = run_experiment(config, args.seeds, results_dir=args.results_dir)

    print()
    print(_render_summary(report))


if __name__ == "__main__":
    main()
