"""Figures for the Part 08 robustness experiments.

Reads the metrics.json files that evaluate() wrote under results/<id>/ and plots
mean +/- sample std across seeds. It computes nothing: every value plotted is a
summary field from the protocol's own output.

Writes to results/<id>/:
    ece_vs_missing.png          headline: ECE vs missing rate, both systems
    confidence_vs_missing.png   confidence and accuracy vs missing rate
    missing_overview.png        six metrics vs missing rate
    imputation_vs_missing.png   imputation baselines against masking
    noise_<condition>.png       six metrics vs sigma, one file per condition

Series carry a colour, a marker shape and a direct label: the green/orange pair
sits in the 6-8 Delta E band under protanopia, so colour alone never encodes
identity here.

Usage:
    python -m experiments.robustness.plot_robustness
    python -m experiments.robustness.plot_robustness --results-dir <scratch>
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

CONFIG_PATH = Path(__file__).with_name("config.yaml")

# Surface and ink tokens match prismflow/evaluation/calibration.py.
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Validated categorical palette (all pairs, light and dark surfaces).
SERIES = {
    "naive": ("#2a78d6", "o", "naive"),
    "prismflow": ("#d1701a", "s", "prismflow"),
    "imputed_naive": ("#2f8f5b", "^", "imputed naive"),
    "imputed_prismflow": ("#b8388f", "D", "imputed prismflow"),
}

PRETTY = {
    "accuracy": "accuracy",
    "prob_ece": "ECE (top-label)",
    "brier_reliability": "Brier reliability (lower better)",
    "brier_resolution": "Brier resolution (higher better)",
    "brier": "Brier score",
    "prob_mean_confidence": "mean top-label confidence",
    "vacuity_mean_confidence": "mean confidence (1 - uncertainty)",
    "eniv": "ENIV (effective views)",
}

OVERVIEW = (
    "prob_ece",
    "accuracy",
    "brier_reliability",
    "prob_mean_confidence",
    "vacuity_mean_confidence",
    "eniv",
)


def read_summary(results_dir: Path, experiment_id: str, name: str) -> dict:
    path = results_dir / experiment_id / name / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run the experiment first")
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def series_values(summaries, metric):
    """(means, stds) with None -> nan, so a metric a system never reports is skipped."""
    means, stds = [], []
    for summary in summaries:
        entry = summary[metric]
        means.append(math.nan if entry["mean"] is None else entry["mean"])
        stds.append(0.0 if entry["std"] is None else entry["std"])
    return means, stds


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def draw(ax, x, metric, by_system, systems, label_last=True):
    style(ax)
    # Direct labels are staggered: at high missing rates the curves converge and
    # unstaggered labels overprint each other.
    offsets = [5.5 * (i - (len(systems) - 1) / 2) for i in range(len(systems))]
    for index, system in enumerate(systems):
        colour, marker, label = SERIES[system]
        means, stds = series_values(by_system[system], metric)
        if all(math.isnan(m) for m in means):
            continue
        ax.errorbar(
            x, means, yerr=stds, color=colour, marker=marker, markersize=6, linewidth=2,
            capsize=3, elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.0,
            label=label, zorder=3,
        )
        if label_last:
            last = next((i for i in range(len(means) - 1, -1, -1) if not math.isnan(means[i])), None)
            if last is not None:
                ax.annotate(
                    label, (x[last], means[last]), textcoords="offset points",
                    xytext=(6, offsets[index]), color=colour, fontsize=8, va="center", clip_on=False,
                )
    ax.margins(x=0.14)
    ax.set_title(PRETTY.get(metric, metric), color=INK_PRIMARY, fontsize=10.5, loc="left")


def finish(fig, path, title, subtitle, axes, xlabel):
    """Title block, one legend, and an x-label on the bottom row.

    Offsets are in inches converted to figure fractions, so a one-row and a
    three-row figure get the same visual spacing instead of the same fraction.
    """
    height = fig.get_figheight()
    for ax in axes[-3:] if len(axes) > 3 else axes:
        ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=9.5)
    fig.tight_layout(rect=(0, 0.07, 1, 1 - 0.85 / height))

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), frameon=False,
        labelcolor=INK_SECONDARY, fontsize=9, bbox_to_anchor=(0.5, 0.005),
    )
    fig.text(0.01, 1 - 0.30 / height, title, color=INK_PRIMARY, fontsize=12.5, ha="left", va="top")
    fig.text(0.01, 1 - 0.62 / height, subtitle, color=INK_SECONDARY, fontsize=9, ha="left", va="top")
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


def grid_figure(x, by_system, systems, metrics, path, title, subtitle, xlabel, xticks=None):
    rows = (len(metrics) + 2) // 3
    fig, axes = plt.subplots(rows, 3, figsize=(13.5, 3.6 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    flat = [ax for row in axes for ax in row]
    for ax, metric in zip(flat, metrics):
        draw(ax, x, metric, by_system, systems)
        if xticks is not None:
            ax.set_xticks(x)
            ax.set_xticklabels(xticks)
    for ax in flat[len(metrics):]:
        ax.set_visible(False)
    return finish(fig, path, title, subtitle, flat[:len(metrics)], xlabel)


def main():
    parser = argparse.ArgumentParser(description="Plot the Part 08 robustness results.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-id", default=None, help="defaults to config.yaml's id")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    experiment_id = args.experiment_id or config["experiment"]["id"]
    results_dir = Path(args.results_dir)
    out_dir = results_dir / experiment_id
    seeds = config["seeds"]
    subtitle = f"{len(seeds)} seeds, mean +/- sample std; models trained on clean data"

    # --- missing views -------------------------------------------------
    rates = config["missing"]["rates"]
    masked = ["naive", "prismflow"]
    imputed = masked + (["imputed_naive", "imputed_prismflow"] if config["missing"]["imputation"] else [])
    by_system = {
        system: [read_summary(results_dir, experiment_id, f"missing_r{rate:.2f}_{system}") for rate in rates]
        for system in imputed
    }
    ticks = [f"{rate:.0%}" for rate in rates]

    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    fig.patch.set_facecolor(SURFACE)
    draw(ax, rates, "prob_ece", by_system, masked)
    ax.set_xticks(rates)
    ax.set_xticklabels(ticks)
    ax.set_ylabel("ECE (lower is better)", color=INK_SECONDARY, fontsize=9.5)
    ax.set_title("")
    written = [
        finish(fig, out_dir / "ece_vs_missing.png", "Calibration error as views go missing",
               subtitle, [ax], "missing rate")
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, metric in zip(axes, ("prob_mean_confidence", "vacuity_mean_confidence")):
        draw(ax, rates, metric, by_system, masked)
        style(ax)
        means, stds = series_values(by_system["naive"], "accuracy")
        ax.errorbar(
            rates, means, yerr=stds, color=INK_MUTED, marker="o", markersize=5, linewidth=1.6,
            linestyle="--", capsize=3, elinewidth=1, label="accuracy (naive)", zorder=2,
        )
        ax.annotate("accuracy", (rates[-1], means[-1]), textcoords="offset points", xytext=(6, 0),
                    color=INK_MUTED, fontsize=8, va="center", clip_on=False)
        ax.set_xticks(rates)
        ax.set_xticklabels(ticks)
    written.append(
        finish(fig, out_dir / "confidence_vs_missing.png",
               "Does confidence fall as views are lost?",
               subtitle + "; accuracy shown for reference", list(axes), "missing rate")
    )

    written.append(
        grid_figure(rates, by_system, masked, OVERVIEW, out_dir / "missing_overview.png",
                    "Missing views: naive fusion vs PrismFlow", subtitle, "missing rate", ticks)
    )
    if config["missing"]["imputation"]:
        written.append(
            grid_figure(rates, by_system, imputed,
                        ("prob_ece", "accuracy", "brier_reliability", "prob_mean_confidence",
                         "vacuity_mean_confidence", "eniv"),
                        out_dir / "imputation_vs_missing.png",
                        "Imputing a missing view as the mean of the others", subtitle,
                        "missing rate", ticks)
        )

    # --- noise ---------------------------------------------------------
    sigmas = config["noise"]["sigmas"]
    for condition, spec in config["noise"]["conditions"].items():
        by_system = {
            system: [
                read_summary(results_dir, experiment_id, f"noise_{condition}_s{sigma:g}_{system}")
                for sigma in sigmas
            ]
            for system in masked
        }
        mode = "replaced by noise" if spec["mode"] == "replace" else "noise added"
        written.append(
            grid_figure(sigmas, by_system, masked, OVERVIEW, out_dir / f"noise_{condition}.png",
                        f"Noise condition: {condition}",
                        f"views {spec['views']} {mode}; {subtitle}", "sigma",
                        [f"{sigma:g}" for sigma in sigmas])
        )

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
