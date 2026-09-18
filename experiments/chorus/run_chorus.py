"""Part 09: the Chorus attack against naive fusion and PrismFlow.

Controlled research simulation. Everything runs against this project's own
models on synthetic data, inside this repository.

Three systems, each attacked white-box in the configuration it is evaluated in:

  naive                 trained without the discount, evaluated without it
  prismflow_nodiscount  trained with the discount, evaluated with it OFF
  prismflow             trained with the discount, evaluated with it ON

For every cell in config.yaml, perturbations are optimised per seed, then:

  - standard metrics (accuracy, ECE, Brier, confidence, ENIV) come from
    `prismflow.evaluation.evaluate`, which scores the attacked batches;
  - attack-specific diagnostics (success rate, belief in the wrong class,
    dependence among the compromised views, rho_bar) are computed here, since
    they are not calibration metrics and protocol.py is frozen in this Part.

Dependence is measured with the same estimator the model uses
(`feature_dependence_matrix` on encoder features), so the attacked and clean
rho_bar are directly comparable.

Outputs under results/<id>/:
    <cell>_<system>/            metrics.json, metrics.csv, reliability_diagram.png
    attack_metrics.json         per-seed and summarised attack diagnostics
    summary.md                  the tables
    metadata.json               config, code versions, per-cell attack settings
    perturbations/<cell>_<system>_seed<seed>.npz   (untracked bulk artefact)

Usage:
    python -m experiments.chorus.run_chorus
    python -m experiments.chorus.run_chorus --quick --results-dir <scratch>
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
import yaml

from experiments.calibration.run_calibration import build_dataset, train
from prismflow.attacks.baseline_attacks import pgd_attack
from prismflow.attacks.chorus import ChorusConfig, choose_targets, chorus_attack, select_views
from prismflow.data.dataset import Batch
from prismflow.data.loaders import iter_batches
from prismflow.eniv.eniv import compute_eniv
from prismflow.evaluation import evaluate
from prismflow.models.prismflow import FORWARD_NULL_PERMUTATIONS
from prismflow.statistics.dependence import available_views, feature_dependence_matrix
from prismflow.utils.logging import get_logger

CONFIG_PATH = Path(__file__).with_name("config.yaml")

ATTACK_METRICS = (
    "success_rate",
    "target_belief_on_success",
    "target_prob_on_success",
    "vacuity",
    "dependence_compromised",
    "dependence_all_pairs",
    "eniv_measured",
)

PROTOCOL_METRICS = ("accuracy", "prob_ece", "brier_reliability", "vacuity_mean_confidence", "eniv")


def build_system(models, spec, seed):
    """The trained model for one system, with its evaluation-time discount flag set."""
    model = models["prismflow" if spec["trained_with_discount"] else "naive"][seed]
    model.use_discount = spec["evaluate_with_discount"]
    model.eval()
    return model


@torch.no_grad()
def measure(model, batch: Batch, compromised) -> dict:
    """Attack diagnostics for one batch, plus the dependence the estimator sees."""
    output = model(batch.views, batch.view_mask)
    features = model.encoder(batch.views, batch.view_mask)
    matrix = feature_dependence_matrix(
        features,
        output.per_view_evidence,
        batch.view_mask,
        method=model.dependence_method,
        conditioning=model.dependence_conditioning,
        seed=model.dependence_seed,
        null_permutations=FORWARD_NULL_PERMUTATIONS,
    )
    present = available_views(batch.view_mask, n_views=model.n_views)
    eniv = compute_eniv(matrix, present)

    return {
        "probs": output.probs.cpu().numpy(),
        "prediction": output.prediction.cpu().numpy(),
        "belief": output.per_view_belief.cpu().numpy(),
        "uncertainty": output.uncertainty.cpu().numpy(),
        "matrix": matrix,
        "eniv_measured": float(eniv.effective_views),
        "n": len(batch),
    }


def _nanmean(values) -> float:
    present = [v for v in values if not math.isnan(v)]
    return float(np.mean(present)) if present else float("nan")


def pair_mean(matrix, pairs) -> float:
    """Mean dependence over the given view pairs; NaN when none are measurable."""
    values = [matrix[i][j] for i, j in pairs if not np.isnan(matrix[i][j])]
    return float(np.mean(values)) if values else float("nan")


def pairs_within(views) -> list[tuple[int, int]]:
    views = list(views)
    return [(i, j) for a, i in enumerate(views) for j in views[a + 1 :]]


def all_pairs(n_views: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_views) for j in range(i + 1, n_views)]


def attack_seed(model, dataset, indices, cell, attack_cfg, batch_size, seed):
    """Run one cell against one system for one seed. Returns (batches, diagnostics, delta)."""
    config = ChorusConfig(
        k=cell["k"],
        epsilon=cell["epsilon"],
        beta=cell["beta"],
        steps=attack_cfg["steps"],
        target_strategy=attack_cfg["target_strategy"],
        fixed_target=attack_cfg["fixed_target"],
        seed=seed,
    )
    compromised = select_views(model.n_views, config)

    attacked, rows, deltas = [], [], []
    for batch in iter_batches(dataset, indices, batch_size):
        if cell["attack"] == "none":
            delta = torch.zeros_like(batch.views)
            target = choose_targets(model, batch.views, batch.view_mask, config)
            history = []
        else:
            run = chorus_attack if cell["attack"] == "chorus" else pgd_attack
            result = run(model, batch.views, batch.view_mask, config)
            delta, target, history = result.delta, result.target, result.history

        perturbed = Batch(
            views=batch.views + delta,
            view_mask=batch.view_mask.clone(),
            labels=batch.labels.clone(),
            sample_ids=batch.sample_ids.clone(),
        )
        attacked.append(perturbed)
        deltas.append(delta.cpu().numpy())

        stats = measure(model, perturbed, compromised)
        success = stats["prediction"] == target.cpu().numpy()
        target_index = target.cpu().numpy()
        rows.append(
            {
                "n": stats["n"],
                "successes": int(success.sum()),
                "target_prob_sum": float(stats["probs"][np.arange(stats["n"]), target_index][success].sum()),
                "target_belief_sum": float(
                    stats["belief"][np.arange(stats["n"]), :, :][success][:, list(compromised), :][
                        np.arange(int(success.sum()))[:, None],
                        np.arange(len(compromised))[None, :],
                        target_index[success][:, None],
                    ].mean(axis=1).sum()
                    if success.any()
                    else 0.0
                ),
                "vacuity_sum": float(stats["uncertainty"].sum()),
                "matrix": stats["matrix"],
                "eniv_measured": stats["eniv_measured"],
                "objective_first": history[0] if history else float("nan"),
                "objective_last": history[-1] if history else float("nan"),
            }
        )

    total = sum(row["n"] for row in rows)
    successes = sum(row["successes"] for row in rows)
    weighted = lambda key: float(np.average([row[key] for row in rows], weights=[row["n"] for row in rows]))

    # Batch-size-weighted mean dependence matrix, ignoring entries a batch could
    # not measure. Pair-set means are taken from it afterwards, so the clean run
    # can be re-read for whichever views a later cell compromises.
    stacked = np.stack([row["matrix"] for row in rows])
    weights = np.asarray([row["n"] for row in rows], dtype=np.float64)[:, None, None]
    measurable = ~np.isnan(stacked)
    with np.errstate(invalid="ignore"):
        matrix = (np.where(measurable, stacked, 0.0) * weights).sum(axis=0) / (measurable * weights).sum(axis=0)

    diagnostics = {
        "success_rate": successes / total,
        "target_prob_on_success": (
            sum(row["target_prob_sum"] for row in rows) / successes if successes else float("nan")
        ),
        "target_belief_on_success": (
            sum(row["target_belief_sum"] for row in rows) / successes if successes else float("nan")
        ),
        "vacuity": sum(row["vacuity_sum"] for row in rows) / total,
        "dependence_compromised": pair_mean(matrix, pairs_within(compromised)),
        "dependence_all_pairs": pair_mean(matrix, all_pairs(matrix.shape[0])),
        "eniv_measured": weighted("eniv_measured"),
        "matrix": matrix.tolist(),
        # The clean cell runs no optimiser, so it has no objective trace.
        "objective_first": _nanmean([row["objective_first"] for row in rows]),
        "objective_last": _nanmean([row["objective_last"] for row in rows]),
        "compromised": list(compromised),
    }
    return attacked, diagnostics, np.concatenate(deltas, axis=0)


def summarise(per_seed: list[dict], keys) -> dict:
    out = {}
    for key in keys:
        values = [row[key] for row in per_seed if not (isinstance(row[key], float) and math.isnan(row[key]))]
        out[key] = {
            "mean": statistics.mean(values) if values else float("nan"),
            "std": statistics.stdev(values) if len(values) > 1 else float("nan"),
            "n_seeds": len(values),
        }
    return out


def _fmt(entry, signed=False):
    if entry is None or math.isnan(entry["mean"]):
        return "n/a"
    mean, std = entry["mean"], entry["std"]
    head = f"{mean:+.4f}" if signed else f"{mean:.4f}"
    return head if math.isnan(std) else f"{head} +/- {std:.4f}"


def render_markdown(attack_summary, protocol_summary, config, experiment_id):
    systems = list(config["systems"])
    cells = [cell["name"] for cell in config["cells"]]
    lines = [
        "# Chorus attack",
        "",
        f"Seeds: {config['seeds']}. Mean +/- sample std across seeds. Attack-specific "
        "diagnostics are computed by `run_chorus.py`; accuracy, ECE and ENIV come from "
        f"`prismflow.evaluation.evaluate` (`results/{experiment_id}/<cell>_<system>/`). "
        "Each system is attacked white-box in the configuration it is evaluated in.",
        "",
        "## Attack success rate",
        "",
        "| cell | " + " | ".join(systems) + " |",
        "|---|" + "---|" * len(systems),
    ]
    for cell in cells:
        lines.append(
            f"| {cell} | " + " | ".join(_fmt(attack_summary[(cell, s)]["success_rate"]) for s in systems) + " |"
        )

    for metric in ("target_prob_on_success", "vacuity", "dependence_compromised", "dependence_all_pairs", "eniv_measured"):
        lines += [
            "",
            f"## {metric}",
            "",
            "| cell | " + " | ".join(systems) + " |",
            "|---|" + "---|" * len(systems),
        ]
        for cell in cells:
            lines.append(
                f"| {cell} | " + " | ".join(_fmt(attack_summary[(cell, s)][metric]) for s in systems) + " |"
            )

    for metric in PROTOCOL_METRICS:
        lines += [
            "",
            f"## {metric} (protocol)",
            "",
            "| cell | " + " | ".join(systems) + " |",
            "|---|" + "---|" * len(systems),
        ]
        for cell in cells:
            lines.append(
                f"| {cell} | " + " | ".join(_fmt(protocol_summary[(cell, s)][metric]) for s in systems) + " |"
            )

    # Headline: does average-case dependence move at all under a successful attack?
    lines += [
        "",
        "## Headline: rho_bar under attack vs clean, within seed",
        "",
        "| cell | system | rho_bar (all pairs) - clean | rho_bar (compromised) - clean | success rate |",
        "|---|---|---|---|---|",
    ]
    for cell in cells:
        if cell == "clean":
            continue
        for system in systems:
            row = attack_summary[(cell, system)]
            lines.append(
                f"| {cell} | {system} | {_fmt(row['delta_dependence_all_pairs'], signed=True)} | "
                f"{_fmt(row['delta_dependence_compromised'], signed=True)} | "
                f"{_fmt(row['success_rate'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run the Part 09 Chorus attack experiment.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="smoke test only: 2 epochs, 3 steps -- NOT evidence")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_id = config["experiment"]["id"]
    if args.quick:
        config["training"]["epochs"] = 2
        config["attack"]["steps"] = 3
        config["cells"] = config["cells"][:3]
        experiment_id = f"{experiment_id}_SMOKE_TEST_NOT_EVIDENCE"
    if len(config["seeds"]) < 5:
        raise ValueError("contract requires at least 5 seeds")

    logger = get_logger("prismflow.chorus")
    seeds, eval_cfg = config["seeds"], config["evaluation"]
    out_dir = Path(args.results_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    perturbation_dir = out_dir / "perturbations"
    if config["outputs"]["save_perturbations"]:
        perturbation_dir.mkdir(exist_ok=True)

    datasets, models = {}, {"naive": {}, "prismflow": {}}
    for seed in seeds:
        started = time.perf_counter()
        dataset, splits = build_dataset(seed, config["data"])
        datasets[seed] = (dataset, splits)
        models["naive"][seed] = train(dataset, splits, seed, False, config["data"], config["training"])
        models["prismflow"][seed] = train(dataset, splits, seed, True, config["data"], config["training"])
        logger.info("seed %d trained (%.0fs)", seed, time.perf_counter() - started)

    attack_summary, protocol_summary, per_seed_store = {}, {}, {}
    for cell in config["cells"]:
        for system, spec in config["systems"].items():
            started = time.perf_counter()
            attacked_by_seed, rows = {}, []
            for seed in seeds:
                model = build_system(models, spec, seed)
                dataset, splits = datasets[seed]
                batches, diagnostics, delta = attack_seed(
                    model, dataset, splits[eval_cfg["split"]], cell, config["attack"],
                    eval_cfg["batch_size"], seed,
                )
                attacked_by_seed[seed] = batches
                rows.append({"seed": seed, **diagnostics})
                if config["outputs"]["save_perturbations"] and cell["attack"] != "none":
                    np.savez_compressed(
                        perturbation_dir / f"{cell['name']}_{system}_seed{seed}.npz",
                        delta=delta.astype(np.float32),
                        compromised=np.asarray(diagnostics["compromised"]),
                    )

            report = evaluate(
                {seed: build_system(models, spec, seed) for seed in seeds},
                lambda seed: iter(attacked_by_seed[seed]),
                f"{experiment_id}/{cell['name']}_{system}",
                seeds,
                results_dir=args.results_dir,
                n_bins=eval_cfg["n_bins"],
                coverages=eval_cfg["coverages"],
                title=f"{cell['name']}: {system}",
            )
            protocol_summary[(cell["name"], system)] = report["summary"]
            per_seed_store[f"{cell['name']}|{system}"] = rows

            summary = summarise(rows, ATTACK_METRICS)
            # rho_bar shift against this system's own clean run, paired by seed.
            if cell["name"] != "clean":
                clean_rows = {r["seed"]: r for r in per_seed_store[f"clean|{system}"]}
                compromised = rows[0]["compromised"]
                references = {
                    "dependence_all_pairs": lambda m: pair_mean(m, all_pairs(len(m))),
                    # The clean cell compromises nothing, so its reference value
                    # is re-read for exactly the views THIS cell attacks.
                    "dependence_compromised": lambda m: pair_mean(m, pairs_within(compromised)),
                }
                for key, reference in references.items():
                    clean_value = {seed: reference(row["matrix"]) for seed, row in clean_rows.items()}
                    diffs = [
                        r[key] - clean_value[r["seed"]]
                        for r in rows
                        if not (math.isnan(r[key]) or math.isnan(clean_value[r["seed"]]))
                    ]
                    summary[f"delta_{key}"] = {
                        "mean": statistics.mean(diffs) if diffs else float("nan"),
                        "std": statistics.stdev(diffs) if len(diffs) > 1 else float("nan"),
                        "n_seeds": len(diffs),
                    }
            attack_summary[(cell["name"], system)] = summary

            logger.info(
                "%-18s %-20s success=%.3f vacuity=%.3f dep(C)=%.3f rho_bar=%.3f eniv=%.2f acc=%.3f ece=%.3f (%.0fs)",
                cell["name"], system, summary["success_rate"]["mean"], summary["vacuity"]["mean"],
                summary["dependence_compromised"]["mean"], summary["dependence_all_pairs"]["mean"],
                summary["eniv_measured"]["mean"], report["summary"]["accuracy"]["mean"],
                report["summary"]["prob_ece"]["mean"], time.perf_counter() - started,
            )

    def json_safe(value):
        if isinstance(value, float) and math.isnan(value):
            return None
        if isinstance(value, dict):
            return {k: json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [json_safe(v) for v in value]
        return value

    (out_dir / "attack_metrics.json").write_text(
        json.dumps(
            json_safe(
                {
                    "summary": {f"{cell}|{system}": values for (cell, system), values in attack_summary.items()},
                    "per_seed": per_seed_store,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only
        commit = None
    (out_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": config,
                "commit": commit,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "forward_null_permutations": FORWARD_NULL_PERMUTATIONS,
                "note": "controlled research simulation; own models, synthetic data",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    markdown = render_markdown(attack_summary, protocol_summary, config, experiment_id)
    (out_dir / "summary.md").write_text(markdown, encoding="utf-8")
    logger.info("wrote %s", out_dir)


if __name__ == "__main__":
    main()
