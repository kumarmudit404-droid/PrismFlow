"""Part 11: does HSIC disentanglement change what the dependence estimator sees?

V1 (no disentanglement) against V2 (shared/private branches with an HSIC
penalty), at rho in {0.3, 0.6}, 5 seeds.

THE QUESTION. The split is meant to give the dependence estimator a cleaner
signal: if the private branch absorbs each view's idiosyncratic part, what
remains in `shared` is more nearly common across views. The expected
consequence is that MEASURED DEPENDENCE RISES AND ENIV FALLS. That would not be
a bug -- it would be the split working, and ENIV finally reporting the overlap
that was previously diluted by private information. The experiment measures the
direction and reports it either way.

WHAT IS COMPARED

  v1              PrismFlow, discount on. The V1 path, untouched.
  v2_lambda0      V2 architecture, lambda_2 = 0. Isolates the ARCHITECTURE
                  (the extra projection layers, the narrower evidence input)
                  from the PENALTY. Without this row, any V1/V2 difference
                  could be either.
  v2_hsic         V2 with the HSIC penalty at lambda_2.
  v2_orthogonality  V2 with the linear penalty, to show what the weaker
                  criterion buys.

Reporting HSIC: the raw biased estimate is not comparable across runs because it
scales with the kernel and with each representation's own variability, so the
NORMALISED value (centred kernel alignment, in [0, 1]) is what is reported. The
raw value is what training minimises.

ENIV STABILITY is the across-seed standard deviation of ENIV, which is the
quantity the brief asks about: a split that gives the estimator a cleaner signal
should make it not just different but steadier.

Outputs under results/disentangle/:
    disentangle.json   per-seed and summarised, every system and rho
    summary.md         the tables
    <rho>_<system>/    metrics.json, metrics.csv, reliability_diagram.png

Usage:
    python -m experiments.disentangle.run_disentangle
    python -m experiments.disentangle.run_disentangle --quick   # NOT evidence
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from prismflow.data.dataset import MultiViewDataset, split_indices
from prismflow.data.loaders import iter_batches
from prismflow.data.synthetic import SyntheticConfig
from prismflow.evaluation import evaluate
from prismflow.models.disentanglement_losses import total_loss
from prismflow.models.encoders import EncoderConfig
from prismflow.models.shared_private import SharedPrivateConfig, build_shared_private
from prismflow.train import TrainConfig, build_model, compute_loss
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

CONFIG = {
    "experiment": {"id": "disentangle"},
    "seeds": [0, 1, 2, 3, 4],
    "rhos": [0.3, 0.6],
    "data": {"n_views": 4},
    "training": {
        "epochs": 40, "batch_size": 64, "lr": 1e-3,
        "anneal_epochs": 10, "per_view_loss_weight": 1.0,
    },
    # lambda_2 is SWEPT for HSIC. Raw biased RBF HSIC is ~1e4 times smaller than
    # the raw linear penalty on the same data (measured: 0.0079 against 92.2), so
    # one shared lambda_2 does not apply comparable pressure to the two methods --
    # at lambda_2 = 0.1 the HSIC term is effectively off. This is a practical trap
    # for anyone implementing the idea and is reported in the README.
    "v2": {"shared_dim": 16, "private_dim": 16,
           "lambda_hsic": [1.0, 10.0, 100.0], "lambda_orthogonality": 0.1},
    "evaluation": {"split": "test", "n_bins": 15, "coverages": [1.0, 0.9, 0.8, 0.5]},
}

SYSTEMS = (
    "v1", "v2_lambda0",
    "v2_hsic_l1", "v2_hsic_l10", "v2_hsic_l100",
    "v2_orthogonality",
)
METRICS = ("accuracy", "prob_ece", "brier_reliability", "vacuity_mean_confidence", "eniv")


def build_dataset(seed: int, rho: float, data_cfg: dict):
    """Same construction as Parts 07-10, with rho swept."""
    base = TrainConfig(n_views=data_cfg["n_views"], rho=rho)
    dataset = MultiViewDataset(SyntheticConfig(n_views=base.n_views, rho=rho, seed=seed))
    return dataset, split_indices(len(dataset), seed=base.split_seed), base


def make_model(system: str, base: TrainConfig, v2_cfg: dict):
    view_configs = [
        EncoderConfig(input_dim=base.d_view, hidden_dims=base.hidden_dims, feature_dim=base.feature_dim)
        for _ in range(base.n_views)
    ]
    if system == "v1":
        return build_model(TrainConfig(n_views=base.n_views, rho=base.rho, use_discount=True))

    if system == "v2_lambda0":
        method, lambda_2 = "hsic", 0.0
    elif system == "v2_orthogonality":
        method, lambda_2 = "orthogonality", v2_cfg["lambda_orthogonality"]
    else:
        method = "hsic"
        lambda_2 = float(system.removeprefix("v2_hsic_l"))

    config = SharedPrivateConfig(
        enabled=True,
        shared_dim=v2_cfg["shared_dim"],
        private_dim=v2_cfg["private_dim"],
        method=method,
        lambda_2=lambda_2,
    )
    return build_shared_private(view_configs, base.n_classes, config, use_discount=True)


def train_system(system, model, dataset, splits, base, training, seed):
    """One training run.

    The task loss is the project's own `compute_loss`, identical for every
    system, so V1 and V2 differ only by the architecture and the added penalty.
    """
    lambda_2 = getattr(getattr(model, "config", None), "lambda_2", 0.0)
    optimiser = torch.optim.Adam(model.parameters(), lr=training["lr"])

    history = []
    for epoch in range(training["epochs"]):
        model.train()
        epoch_penalty, batches = 0.0, 0
        for batch in iter_batches(
            dataset, splits["train"], training["batch_size"], shuffle=True, seed=seed + epoch
        ):
            targets = torch.nn.functional.one_hot(
                batch.labels, num_classes=base.n_classes
            ).to(batch.views.dtype)
            output = model(batch.views, batch.view_mask)
            task = compute_loss(
                output, targets, epoch=epoch,
                anneal_epochs=training["anneal_epochs"],
                per_view_loss_weight=training["per_view_loss_weight"],
            )
            report = getattr(output, "disentanglement", None)
            if report is not None:
                epoch_penalty += float(report.loss.detach())
                loss = total_loss(task, report, lambda_2)
            else:
                loss = task
            if not torch.isfinite(loss):
                raise RuntimeError(f"{system}: non-finite loss at epoch {epoch} (seed {seed})")
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            batches += 1
        history.append(epoch_penalty / max(batches, 1))

    model.eval()
    return history


@torch.no_grad()
def measure_disentanglement(model, dataset, indices, batch_size) -> float:
    """Normalised HSIC between the branches, averaged over evaluation batches."""
    values = []
    for batch in iter_batches(dataset, indices, batch_size):
        output = model(batch.views, batch.view_mask)
        if getattr(output, "disentanglement", None) is None:
            return float("nan")
        if not math.isnan(output.disentanglement.normalised):
            values.append(output.disentanglement.normalised)
    return float(np.mean(values)) if values else float("nan")


def summarise(rows, keys):
    out = {}
    for key in keys:
        values = [r[key] for r in rows if not (isinstance(r[key], float) and math.isnan(r[key]))]
        out[key] = {
            "mean": statistics.mean(values) if values else float("nan"),
            "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
            "n_seeds": len(values),
        }
    return out


def fmt(entry):
    if entry is None or math.isnan(entry["mean"]):
        return "n/a"
    mean, std = entry["mean"], entry["std"]
    return f"{mean:.4f}" if math.isnan(std) else f"{mean:.4f} +/- {std:.4f}"


def json_safe(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.integer)):
        return json_safe(float(value))
    return value


def main():
    parser = argparse.ArgumentParser(description="Part 11: HSIC disentanglement.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="smoke test -- NOT evidence")
    args = parser.parse_args()

    config = json.loads(json.dumps(CONFIG))
    experiment_id = config["experiment"]["id"]
    if args.quick:
        config["training"]["epochs"] = 2
        config["rhos"] = config["rhos"][:1]
        experiment_id = f"{experiment_id}_SMOKE_TEST_NOT_EVIDENCE"
    if len(config["seeds"]) < 5:
        raise ValueError("contract requires at least 5 seeds")

    logger = get_logger("prismflow.disentangle")
    seeds, training, eval_cfg = config["seeds"], config["training"], config["evaluation"]
    out_dir = Path(args.results_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol, detail = {}, {}
    for rho in config["rhos"]:
        for system in SYSTEMS:
            started = time.perf_counter()
            models, batches_by_seed, rows = {}, {}, []

            for seed in seeds:
                set_seed(seed)
                dataset, splits, base = build_dataset(seed, rho, config["data"])
                model = make_model(system, base, config["v2"])
                history = train_system(system, model, dataset, splits, base, training, seed)

                models[seed] = model
                batches_by_seed[seed] = list(
                    iter_batches(dataset, splits[eval_cfg["split"]], training["batch_size"])
                )
                rows.append({
                    "seed": seed,
                    "normalised_hsic": measure_disentanglement(
                        model, dataset, splits[eval_cfg["split"]], training["batch_size"]
                    ),
                    "penalty_first_epoch": history[0],
                    "penalty_last_epoch": history[-1],
                })

            name = f"rho{rho}_{system}"
            report = evaluate(
                models, lambda seed: iter(batches_by_seed[seed]),
                f"{experiment_id}/{name}", seeds,
                results_dir=args.results_dir,
                n_bins=eval_cfg["n_bins"], coverages=eval_cfg["coverages"], title=name,
            )
            protocol[name] = report["summary"]
            detail[name] = {
                "per_seed": rows,
                "summary": summarise(rows, ("normalised_hsic", "penalty_first_epoch", "penalty_last_epoch")),
                "eniv_per_seed": [r["eniv"] for r in report["per_seed"]],
            }
            logger.info(
                "%-22s acc=%.4f ece=%.4f eniv=%.4f (sd %.4f) nHSIC=%.4f (%.0fs)",
                name, report["summary"]["accuracy"]["mean"], report["summary"]["prob_ece"]["mean"],
                report["summary"]["eniv"]["mean"], report["summary"]["eniv"]["std"],
                detail[name]["summary"]["normalised_hsic"]["mean"], time.perf_counter() - started,
            )

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only
        commit = None

    (out_dir / "disentangle.json").write_text(
        json.dumps(json_safe({
            "config": config, "commit": commit, "torch": torch.__version__,
            "protocol": protocol, "detail": detail,
        }), indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Shared / private disentanglement with HSIC (Part 11)",
        "",
        f"Seeds: {seeds}. Mean +/- sample std across seeds. `v2_lambda0` is the V2 "
        "architecture with the penalty switched off, so the architecture and the "
        "penalty can be told apart. Normalised HSIC is a centred kernel alignment "
        "in [0, 1]; lower means better disentangled.",
        "",
    ]
    for metric in ("accuracy", "prob_ece", "eniv", "vacuity_mean_confidence"):
        lines += ["", f"## {metric}", "", "| rho | " + " | ".join(SYSTEMS) + " |",
                  "|---|" + "---|" * len(SYSTEMS)]
        for rho in config["rhos"]:
            cells = [fmt(protocol[f"rho{rho}_{s}"][metric]) for s in SYSTEMS]
            lines.append(f"| {rho} | " + " | ".join(cells) + " |")

    lines += ["", "## ENIV stability (across-seed std, lower = steadier)", "",
              "| rho | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    for rho in config["rhos"]:
        cells = [f"{protocol[f'rho{rho}_{s}']['eniv']['std']:.4f}" for s in SYSTEMS]
        lines.append(f"| {rho} | " + " | ".join(cells) + " |")

    lines += ["", "## Normalised HSIC between shared and private", "",
              "| rho | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    for rho in config["rhos"]:
        cells = [fmt(detail[f"rho{rho}_{s}"]["summary"]["normalised_hsic"]) for s in SYSTEMS]
        lines.append(f"| {rho} | " + " | ".join(cells) + " |")
    lines.append("")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
