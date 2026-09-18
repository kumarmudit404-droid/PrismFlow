"""Part 12: global ENIV against per-sample ENIV.

THE CONTRAST THAT IS THE POINT OF THE PART

`mixed` is a dataset where dependence is INPUT-CONDITIONAL: 20% of samples have
their views generated from a single shared latent draw (rho high), and the other
80% have independent views (rho 0). A global correlation matrix must report one
number for that dataset, and that number is wrong about every sample in it --
too high for the independent majority, far too low for the collapsed minority.
Per-sample ENIV can represent the difference. Whether it DOES is what this
measures.

WHAT IS COMPARED

  global       V1: one statistical dependence matrix per batch, V1 discount.
  per_sample   V2: the amortized estimator, trained two-timescale against the
               V1 estimator, producing n_eff(x) and a per-sample alpha.

On: ENIV stability across seeds, calibration, Chorus-attack detection, and
wall-clock cost. Plus, for `mixed` only, whether the per-sample estimate
SEPARATES the two groups -- the AUC of n_eff(x) between collapsed and
independent samples. That separation is the claim; everything else is whether it
costs anything.

GAMING AUDIT. `train_two_timescale` logs, every epoch, the dependence its
trained estimator reports and the dependence the untrainable V1 estimator
measures on held-out data. The gap between them is written to
`results/per_sample/audit_<condition>_seed<seed>.json` and plotted to
`audit_gap.png`. A widening gap is the encoder learning to fool the estimator
rather than learning independent representations. That plot is the evidence;
without it a per-sample result should not be believed.

Usage:
    python -m experiments.per_sample.run_per_sample
    python -m experiments.per_sample.run_per_sample --quick   # NOT evidence
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

from prismflow.attacks.chorus import ChorusConfig, chorus_attack
from prismflow.data.dataset import Batch, MultiViewDataset, split_indices
from prismflow.data.loaders import iter_batches
from prismflow.data.synthetic import SyntheticConfig
from prismflow.eniv.amortized import AmortizedConfig, AmortizedDependence
from prismflow.eniv.per_sample import compute_per_sample_eniv
from prismflow.evaluation import evaluate
from prismflow.statistics.dependence import available_views, feature_dependence_matrix
from prismflow.eniv.eniv import compute_eniv
from prismflow.train import TrainConfig, build_model
from prismflow.train_two_timescale import TwoTimescaleConfig, train_two_timescale
from prismflow.utils.logging import get_logger
from prismflow.utils.seed import set_seed

CONFIG = {
    "experiment": {"id": "per_sample"},
    "seeds": [0, 1, 2, 3, 4],
    "data": {"n_views": 4, "rho": 0.3},
    "mixed": {"collapsed_fraction": 0.2, "collapsed_rho": 0.95, "background_rho": 0.0},
    "training": {
        "epochs": 40, "batch_size": 64, "encoder_lr": 1e-3, "estimator_lr": 1e-3,
        "anneal_epochs": 10, "per_view_loss_weight": 1.0,
        "encoder_steps": 5, "estimator_steps": 1,
    },
    "attack": {"k": 2, "epsilon": 1.0, "beta": 1.0, "steps": 30},
    "evaluation": {"split": "test", "n_bins": 15, "coverages": [1.0, 0.9, 0.8, 0.5]},
    "conditions": ["homogeneous", "mixed"],
}

SYSTEMS = ("global", "per_sample")


class PerSampleDiscountModel(torch.nn.Module):
    """The trained model, but with the discount driven by n_eff(x).

    Without this, `global` and `per_sample` would share a forward pass and their
    calibration and attack columns would be identical by construction -- the two
    systems would differ only in a diagnostic readout. Here the amortized
    estimator's per-sample alpha actually feeds `evidence_discount`, which
    already accepts a [B, V] factor, so the fusion path is unchanged.

    Defined in the experiment rather than in `prismflow/models/` because Part 12
    may not add a model file, and because V1's `PrismFlow` must keep its global
    path as the default.
    """

    def __init__(self, model, estimator, method: str = "eigen"):
        super().__init__()
        self.model = model
        self.estimator = estimator
        self.method = method
        self.n_views = model.n_views
        self.n_classes = model.n_classes

    def forward(self, views, view_mask=None):
        from prismflow.eniv.discount import evidence_discount
        from prismflow.models.evidence import evidence_to_opinion, opinion_to_probs
        from prismflow.models.fusion import fuse_opinions
        from prismflow.models.prismflow import PrismFlowOutput

        if view_mask is None:
            view_mask = torch.ones(
                views.shape[0], self.n_views, dtype=torch.bool, device=views.device
            )
        features = self.model.encoder(views, view_mask)
        evidence = self.model.evidence_head(features, view_mask)

        with torch.no_grad():
            matrix = self.estimator(evidence, view_mask)
            result = compute_per_sample_eniv(
                matrix.cpu().numpy(), view_mask.cpu().numpy(), method=self.method
            )
            alpha = torch.as_tensor(result.alpha, dtype=evidence.dtype, device=evidence.device)

        belief, uncertainty = evidence_to_opinion(evidence_discount(evidence, alpha))
        fused_belief, fused_uncertainty = fuse_opinions(belief, uncertainty, view_mask)
        probs = opinion_to_probs(fused_belief, fused_uncertainty)

        return PrismFlowOutput(
            prediction=probs.argmax(dim=-1),
            probs=probs,
            confidence=1.0 - fused_uncertainty,
            uncertainty=fused_uncertainty,
            per_view_evidence=evidence,
            per_view_belief=belief,
            per_view_uncertainty=uncertainty,
            view_mask=view_mask,
            dependence_matrix=matrix.mean(dim=0),
            eniv=None,
        )


def make_mixed_dataset(seed: int, data_cfg: dict, mixed_cfg: dict):
    """Views collapsed onto one latent for a minority of samples.

    Built by overwriting a fraction of an ordinary dataset's views with copies
    of one view plus small noise, so the collapsed samples are genuinely
    redundant across views while the rest are untouched. The flag is returned
    separately and is used ONLY for evaluation, never by any estimator.
    """
    base = TrainConfig(n_views=data_cfg["n_views"], rho=mixed_cfg["background_rho"])
    dataset = MultiViewDataset(
        SyntheticConfig(n_views=base.n_views, rho=mixed_cfg["background_rho"], seed=seed)
    )

    generator = torch.Generator().manual_seed(seed + 9973)
    n = len(dataset)
    collapsed = torch.zeros(n, dtype=torch.bool)
    collapsed[torch.randperm(n, generator=generator)[: int(mixed_cfg["collapsed_fraction"] * n)]] = True

    views = dataset.views.clone()
    source = views[collapsed][:, 0:1, :]
    jitter = (1.0 - mixed_cfg["collapsed_rho"]) * torch.randn(
        source.shape[0], base.n_views, source.shape[-1], generator=generator
    )
    views[collapsed] = source.expand(-1, base.n_views, -1) + jitter
    dataset.views = views
    return dataset, split_indices(n, seed=base.split_seed), base, collapsed


def build_condition(condition: str, seed: int, config: dict):
    if condition == "mixed":
        return make_mixed_dataset(seed, config["data"], config["mixed"])
    base = TrainConfig(n_views=config["data"]["n_views"], rho=config["data"]["rho"])
    dataset = MultiViewDataset(
        SyntheticConfig(n_views=base.n_views, rho=config["data"]["rho"], seed=seed)
    )
    return dataset, split_indices(len(dataset), seed=base.split_seed), base, None


@torch.no_grad()
def per_sample_eniv_values(model, estimator, dataset, indices, batch_size):
    """n_eff(x) for every sample, in dataset order over `indices`."""
    values = []
    for batch in iter_batches(dataset, indices, batch_size):
        features = model.encoder(batch.views, batch.view_mask)
        evidence = model.evidence_head(features, batch.view_mask)
        matrix = estimator(evidence, batch.view_mask).cpu().numpy()
        values.append(compute_per_sample_eniv(matrix, batch.view_mask.cpu().numpy()).effective_views)
    return np.concatenate(values) if values else np.zeros(0)


@torch.no_grad()
def global_eniv_values(model, dataset, indices, batch_size, null_permutations=8):
    """One ENIV per batch from the V1 estimator, repeated for that batch's samples."""
    values = []
    for batch in iter_batches(dataset, indices, batch_size):
        features = model.encoder(batch.views, batch.view_mask)
        evidence = model.evidence_head(features, batch.view_mask)
        matrix = feature_dependence_matrix(
            features, evidence, batch.view_mask,
            method=model.dependence_method,
            conditioning=model.dependence_conditioning,
            seed=model.dependence_seed,
            null_permutations=null_permutations,
        )
        present = available_views(batch.view_mask, n_views=model.n_views)
        values.append(np.full(len(batch), compute_eniv(matrix, present).effective_views))
    return np.concatenate(values) if values else np.zeros(0)


def roc_auc(positive, negative) -> float:
    positive = np.asarray([v for v in positive if np.isfinite(v)], dtype=np.float64)
    negative = np.asarray([v for v in negative if np.isfinite(v)], dtype=np.float64)
    if positive.size == 0 or negative.size == 0:
        return float("nan")
    combined = np.concatenate([positive, negative])
    order = combined.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, combined.size + 1)
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    summed = np.zeros(counts.size)
    np.add.at(summed, inverse, ranks)
    ranks = (summed / counts)[inverse]
    rank_sum = ranks[: positive.size].sum()
    return float((rank_sum - positive.size * (positive.size + 1) / 2) / (positive.size * negative.size))


def attacked_batches(model, dataset, indices, attack_cfg, batch_size, seed):
    config = ChorusConfig(
        k=attack_cfg["k"], epsilon=attack_cfg["epsilon"], beta=attack_cfg["beta"],
        steps=attack_cfg["steps"], seed=seed,
    )
    out = []
    for batch in iter_batches(dataset, indices, batch_size):
        delta = chorus_attack(model, batch.views, batch.view_mask, config).delta
        out.append(Batch(
            views=batch.views + delta, view_mask=batch.view_mask.clone(),
            labels=batch.labels.clone(), sample_ids=batch.sample_ids.clone(),
        ))
    return out


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
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def plot_audit(audit_by_key, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - plotting is optional
        return False

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for key, log in audit_by_key.items():
        axes[0].plot(log["epoch"], log["trained_dependence"], alpha=0.6, label=f"{key} trained")
        axes[0].plot(log["epoch"], log["audited_dependence"], "--", alpha=0.6, label=f"{key} audited")
        axes[1].plot(log["epoch"], log["gap"], alpha=0.8, label=key)

    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("mean off-diagonal dependence")
    axes[0].set_title("Trained estimator vs held-out auditor")
    axes[0].legend(fontsize=6)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("audited - trained")
    axes[1].set_title("Gaming gap (positive = estimator under-reports)")
    axes[1].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
    return True


def main():
    parser = argparse.ArgumentParser(description="Part 12: per-sample ENIV.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="smoke test -- NOT evidence")
    args = parser.parse_args()

    config = json.loads(json.dumps(CONFIG))
    experiment_id = config["experiment"]["id"]
    if args.quick:
        config["training"]["epochs"] = 2
        config["attack"]["steps"] = 3
        experiment_id = f"{experiment_id}_SMOKE_TEST_NOT_EVIDENCE"
    if len(config["seeds"]) < 5:
        raise ValueError("contract requires at least 5 seeds")

    logger = get_logger("prismflow.per_sample")
    seeds, training, eval_cfg = config["seeds"], config["training"], config["evaluation"]
    out_dir = Path(args.results_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    protocol, detail, audits = {}, {}, {}
    for condition in config["conditions"]:
        models, estimators, datasets, flags = {}, {}, {}, {}

        for seed in seeds:
            set_seed(seed)
            dataset, splits, base, collapsed = build_condition(condition, seed, config)
            datasets[seed] = (dataset, splits, base)
            flags[seed] = collapsed

            model = build_model(TrainConfig(n_views=base.n_views, rho=base.rho, use_discount=True))
            estimator = AmortizedDependence(
                base.n_views, base.n_classes, AmortizedConfig(enabled=True)
            )
            started = time.perf_counter()
            log = train_two_timescale(
                model, estimator, dataset, splits, base.n_classes,
                TwoTimescaleConfig(
                    encoder_steps=training["encoder_steps"],
                    estimator_steps=training["estimator_steps"],
                    encoder_lr=training["encoder_lr"], estimator_lr=training["estimator_lr"],
                    epochs=training["epochs"], batch_size=training["batch_size"],
                    anneal_epochs=training["anneal_epochs"],
                    per_view_loss_weight=training["per_view_loss_weight"],
                ),
                seed=seed,
            )
            models[seed], estimators[seed] = model, estimator
            audits[f"{condition}_seed{seed}"] = log.as_dict()
            (out_dir / f"audit_{condition}_seed{seed}.json").write_text(
                json.dumps(json_safe(log.as_dict()), indent=2), encoding="utf-8"
            )
            logger.info(
                "%s seed %d trained (%.0fs) final gap %+.4f max gap %+.4f",
                condition, seed, time.perf_counter() - started, log.final_gap, log.max_gap,
            )

        for system in SYSTEMS:
            rows, batches_by_seed, scored = [], {}, {}
            started = time.perf_counter()
            for seed in seeds:
                dataset, splits, base = datasets[seed]
                indices = splits[eval_cfg["split"]]
                # The per_sample system is evaluated with the per-sample alpha
                # ACTUALLY driving the discount, not merely reported beside it.
                scored[seed] = (
                    PerSampleDiscountModel(models[seed], estimators[seed])
                    if system == "per_sample" else models[seed]
                )
                scored[seed].eval()
                batches_by_seed[seed] = list(
                    iter_batches(dataset, indices, training["batch_size"])
                )

                timed = time.perf_counter()
                if system == "per_sample":
                    values = per_sample_eniv_values(
                        models[seed], estimators[seed], dataset, indices, training["batch_size"]
                    )
                else:
                    values = global_eniv_values(
                        models[seed], dataset, indices, training["batch_size"]
                    )
                elapsed = time.perf_counter() - timed

                row = {
                    "seed": seed,
                    "eniv_mean": float(np.nanmean(values)) if values.size else float("nan"),
                    "eniv_spread": float(np.nanstd(values)) if values.size else float("nan"),
                    "eniv_seconds": elapsed,
                    "separation_auc": float("nan"),
                    "attack_success": float("nan"),
                }

                collapsed = flags[seed]
                if collapsed is not None:
                    subset = collapsed[torch.as_tensor(indices)].numpy().astype(bool)
                    if subset.any() and (~subset).any():
                        # Lower n_eff on the collapsed group is the correct
                        # direction, so the AUC is taken on negated values.
                        row["separation_auc"] = roc_auc(-values[subset], -values[~subset])

                # White-box against the system being evaluated, as in Part 09.
                attacked = attacked_batches(
                    scored[seed], dataset, indices, config["attack"], training["batch_size"], seed
                )
                with torch.no_grad():
                    wrong = sum(
                        int((scored[seed](b.views, b.view_mask).prediction != b.labels).sum())
                        for b in attacked
                    )
                    total = sum(len(b) for b in attacked)
                row["attack_success"] = wrong / total if total else float("nan")
                rows.append(row)

            name = f"{condition}_{system}"
            report = evaluate(
                scored, lambda seed: iter(batches_by_seed[seed]),
                f"{experiment_id}/{name}", seeds,
                results_dir=args.results_dir,
                n_bins=eval_cfg["n_bins"], coverages=eval_cfg["coverages"], title=name,
            )
            protocol[name] = report["summary"]
            detail[name] = {
                "per_seed": rows,
                "summary": summarise(
                    rows, ("eniv_mean", "eniv_spread", "eniv_seconds", "separation_auc", "attack_success")
                ),
                "wall_clock_total": time.perf_counter() - started,
            }
            logger.info(
                "%-22s eniv=%s spread=%s sep_auc=%s acc=%.4f ece=%.4f t=%.2fs",
                name,
                fmt(detail[name]["summary"]["eniv_mean"]),
                fmt(detail[name]["summary"]["eniv_spread"]),
                fmt(detail[name]["summary"]["separation_auc"]),
                report["summary"]["accuracy"]["mean"], report["summary"]["prob_ece"]["mean"],
                detail[name]["summary"]["eniv_seconds"]["mean"],
            )

    plotted = plot_audit(audits, out_dir / "audit_gap.png")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only
        commit = None

    (out_dir / "per_sample.json").write_text(
        json.dumps(json_safe({
            "config": config, "commit": commit, "torch": torch.__version__,
            "protocol": protocol, "detail": detail, "audit": audits,
            "audit_plot": bool(plotted),
        }), indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Per-sample ENIV (Part 12)",
        "",
        f"Seeds: {seeds}. Mean +/- sample std across seeds.",
        "",
        "| condition | system | ENIV mean | ENIV spread (within run) | separation AUC | seconds | accuracy | ECE |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for condition in config["conditions"]:
        for system in SYSTEMS:
            name = f"{condition}_{system}"
            summary = detail[name]["summary"]
            lines.append(
                f"| {condition} | {system} | {fmt(summary['eniv_mean'])} | {fmt(summary['eniv_spread'])} | "
                f"{fmt(summary['separation_auc'])} | {fmt(summary['eniv_seconds'])} | "
                f"{fmt(protocol[name]['accuracy'])} | {fmt(protocol[name]['prob_ece'])} |"
            )

    lines += ["", "## ENIV stability across seeds (std of the per-run mean)", "",
              "| condition | " + " | ".join(SYSTEMS) + " |", "|---|" + "---|" * len(SYSTEMS)]
    for condition in config["conditions"]:
        cells = [f"{detail[f'{condition}_{s}']['summary']['eniv_mean']['std']:.4f}" for s in SYSTEMS]
        lines.append(f"| {condition} | " + " | ".join(cells) + " |")

    lines += ["", "## Gaming audit (audited - trained dependence)", "",
              "| condition | seed | final gap | max gap |", "|---|---|---|---|"]
    for key, log in audits.items():
        condition, _, seed = key.partition("_seed")
        gaps = log["gap"]
        lines.append(f"| {condition} | {seed} | {gaps[-1]:+.4f} | {max(gaps):+.4f} |")
    lines += ["", f"Audit plot written: {plotted}.", ""]

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
