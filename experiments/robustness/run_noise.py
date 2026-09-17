"""Part 08 noise experiment: naive fusion vs PrismFlow as views become noisy.

Same trained models as run_missing.py (clean data, one naive and one PrismFlow
per seed, identical initial weights). Corruption is applied at test time only,
with a seed that depends on (seed, batch index, condition, sigma), so both
systems see identical corrupted inputs.

Conditions (config.yaml):
  one_noisy    N(0, sigma^2) added to view 0            (corruption.add_noise)
  two_noisy    the same noise added to views 0 and 1, independently drawn
  one_severe   view 0's content replaced by N(0, sigma^2): no signal left

View features have std ~1.3, so the sigma sweep spans roughly 0.2x to 3x the
view scale. Every metric comes from prismflow.evaluation.evaluate. Outputs go to
results/<id>/noise_<condition>_s<sigma>_<system>/ plus noise_summary.md.

Usage:
    python -m experiments.robustness.run_noise
    python -m experiments.robustness.run_noise --quick --results-dir <scratch>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from experiments.robustness.run_missing import (
    METRICS,
    batch_seed,
    comparison_rows,
    load_config,
    summarise_table,
    train_models,
)
from prismflow.data.corruption import add_noise
from prismflow.data.dataset import Batch
from prismflow.data.loaders import iter_batches
from prismflow.evaluation import evaluate
from prismflow.utils.logging import get_logger

SYSTEMS = ("naive", "prismflow")


def blank_views(batch: Batch, views) -> Batch:
    """Zero the named views' inputs, so that adding noise leaves noise alone."""
    new_views = batch.views.clone()
    for view in views:
        new_views[:, view, :] = 0.0
    return Batch(
        views=new_views,
        view_mask=batch.view_mask.clone(),
        labels=batch.labels.clone(),
        sample_ids=batch.sample_ids.clone(),
    )


def corrupt(batch: Batch, spec: dict, sigma: float, seed: int) -> Batch:
    if spec["mode"] == "replace":
        batch = blank_views(batch, spec["views"])
    elif spec["mode"] != "add":
        raise ValueError(f"unknown noise mode {spec['mode']!r}")
    for offset, view in enumerate(spec["views"]):
        batch = add_noise(batch, view, sigma, seed=seed + 101 * offset)
    return batch


def main():
    parser = argparse.ArgumentParser(description="Run the Part 08 noise experiment.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--quick", action="store_true", help="smoke test only: 2 epochs -- NOT evidence")
    args = parser.parse_args()

    config, experiment_id = load_config(args.quick)
    logger = get_logger("prismflow.robustness.noise")
    seeds, eval_cfg = config["seeds"], config["evaluation"]
    sigmas = config["noise"]["sigmas"]
    conditions = config["noise"]["conditions"]

    datasets, trained = train_models(config, logger)

    reports = {}
    for condition_index, (condition, spec) in enumerate(conditions.items()):
        for sigma_index, sigma in enumerate(sigmas):
            def loader(seed, spec=spec, sigma=sigma, salt=1_000 * condition_index + sigma_index):
                dataset, splits = datasets[seed]
                batches = iter_batches(dataset, splits[eval_cfg["split"]], eval_cfg["batch_size"])
                for batch_index, batch in enumerate(batches):
                    yield corrupt(batch, spec, sigma, seed=batch_seed(seed, batch_index, salt))

            for system in SYSTEMS:
                reports[(condition, sigma, system)] = evaluate(
                    trained[system],
                    loader,
                    f"{experiment_id}/noise_{condition}_s{sigma:g}_{system}",
                    seeds,
                    results_dir=args.results_dir,
                    n_bins=eval_cfg["n_bins"],
                    coverages=eval_cfg["coverages"],
                    title=f"{condition}, sigma={sigma:g}: {system}",
                )
                s = reports[(condition, sigma, system)]["summary"]
                logger.info(
                    "%-10s sigma=%-4g %-9s acc=%.4f ece=%.4f rel=%.4f conf=%.4f vac_conf=%.4f eniv=%.3f",
                    condition, sigma, system, s["accuracy"]["mean"], s["prob_ece"]["mean"],
                    s["brier_reliability"]["mean"], s["prob_mean_confidence"]["mean"],
                    s["vacuity_mean_confidence"]["mean"], s["eniv"]["mean"],
                )

    lines = [
        "# Noise experiment",
        "",
        f"Seeds: {seeds}. Models trained on clean data; noise applied at test time. "
        "Mean +/- sample std across seeds, from `metrics.json` in each "
        f"`results/{experiment_id}/noise_<condition>_s<sigma>_<system>/`. View features have "
        "std ~1.3. `one_severe` replaces view 0's content with noise, leaving no signal.",
        "",
    ]
    for condition in conditions:
        condition_reports = {
            (sigma, system): reports[(condition, sigma, system)] for sigma in sigmas for system in SYSTEMS
        }
        lines += [
            f"## {condition} (views {conditions[condition]['views']}, mode {conditions[condition]['mode']})",
            "",
            *summarise_table(condition_reports, list(SYSTEMS), sigmas, "sigma", lambda s: f"{s:g}"),
            "### prismflow vs naive",
            "",
            "`bars overlap` compares mean +/- std intervals. 'no' is the only case that may be "
            "reported as a difference.",
            "",
            *comparison_rows(condition_reports, "prismflow", "naive", sigmas, lambda s: f"{s:g}", seeds),
            "",
        ]

    out_dir = Path(args.results_dir) / experiment_id
    (out_dir / "noise_summary.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s", out_dir / "noise_summary.md")


if __name__ == "__main__":
    main()
