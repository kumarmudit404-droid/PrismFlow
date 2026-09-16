"""Plots for the clone experiment.

Reads results/clone/runs.csv and writes, into the same directory:

    confidence_vs_duplicates.png
    eniv_vs_duplicates.png
    uncertainty_vs_duplicates.png
    accuracy_vs_duplicates.png

Each figure has one panel per rho (same metric, same y-range, one axis each --
never a dual axis). Values are mean +/- sample std across seeds; every point is
read from runs.csv, nothing is computed that the run did not produce.

Identity is never carried by color alone: each system has its own marker and
line style, a legend, and a direct label at the end of its line. The diagnostic
aqua sits below 3:1 contrast on the surface, which is what makes those labels
mandatory rather than decorative; results/clone/summary.md is the table view.

Usage:
    python -m experiments.clone.plot_clone
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from prismflow.data.synthetic import analytic_n_eff, analytic_n_eff_eigen  # noqa: E402

# Minimum vertical gap between direct labels at line ends, in pixels.
_LABEL_GAP_PX = 13

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots 1-3 in fixed order; validated all-pairs (light surface).
SYSTEMS = (
    ("naive", "Naive Dempster fusion", "#2a78d6", "o", "-"),
    ("prismflow", "PrismFlow (ENIV discount)", "#eb6834", "s", "-"),
    ("naive_weights_discounted", "Naive weights + discount at inference", "#1baf7a", "^", "--"),
)

FIGURES = (
    ("mean_confidence", "Mean confidence", "confidence_vs_duplicates.png", None),
    ("eniv", "ENIV (effective independent views)", "eniv_vs_duplicates.png", "eniv"),
    ("mean_uncertainty", "Mean uncertainty (vacuity)", "uncertainty_vs_duplicates.png", None),
    ("accuracy", "Test accuracy", "accuracy_vs_duplicates.png", None),
)


def load(csv_path: Path):
    grouped = defaultdict(list)
    base_views = None
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rho, k = float(row["rho"]), int(row["k"])
            base_views = int(row["n_views"]) - k
            grouped[(rho, row["system"], k)].append(row)
    return grouped, base_views


def series(grouped, rho, system, metric):
    ks = sorted({k for (r, s, k) in grouped if r == rho and s == system})
    xs, means, stds = [], [], []
    for k in ks:
        values = [float(r[metric]) for r in grouped[(rho, system, k)]]
        values = [v for v in values if v == v]  # drop NaN
        if not values:
            continue
        xs.append(k)
        means.append(statistics.mean(values))
        stds.append(statistics.stdev(values) if len(values) > 1 else 0.0)
    return xs, means, stds


def style_axis(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)


def place_end_labels(ax, ends, dpi):
    """Direct-label each line end, nudging labels apart when ends converge.

    Labels are spread in screen space from the top down so that no two sit
    closer than _LABEL_GAP_PX; each stays as near its own line end as that
    allows.
    """
    ax.relim()
    ax.autoscale_view()
    placed = []
    for x, y, text in sorted(ends, key=lambda e: -e[1]):
        pixel_y = ax.transData.transform((x, y))[1]
        target = pixel_y
        if placed and placed[-1] - target < _LABEL_GAP_PX:
            target = placed[-1] - _LABEL_GAP_PX
        placed.append(target)
        offset_points = (target - pixel_y) * 72.0 / dpi
        ax.annotate(
            text, xy=(x, y), xytext=(8, offset_points), textcoords="offset points",
            color=INK_SECONDARY, fontsize=8.5, va="center",
        )


def draw(grouped, base_views, metric, label, filename, reference, out_dir, n_seeds):
    rhos = sorted({r for (r, _, _) in grouped})
    fig, axes = plt.subplots(1, len(rhos), figsize=(6.2 * len(rhos), 4.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    axes = list(axes) if len(rhos) > 1 else [axes]

    for ax, rho in zip(axes, rhos):
        style_axis(ax)
        if metric == "eniv":
            systems = [s for s in SYSTEMS if s[0] != "naive"]
        else:
            systems = SYSTEMS

        if reference == "eniv":
            # Copies add nothing, so the true effective count is that of the
            # base views alone, under compute_eniv's default eigen definition.
            truth = analytic_n_eff_eigen(rho, base_views)
            ax.axhline(truth, color=INK_SECONDARY, linewidth=1.4, linestyle="-", zorder=1)
            ax.text(
                0.02, truth, f"  truth for any k: {truth:.2f}  (eigen ENIV of base views)",
                color=INK_SECONDARY, fontsize=8.5, va="bottom",
                transform=ax.get_yaxis_transform(),
            )

            # Context only, and only where it differs: at rho=0 both
            # definitions give base_views and a second line would just overlap.
            context = analytic_n_eff(rho, base_views)
            if abs(context - truth) > 1e-9:
                ax.axhline(context, color=GRID, linewidth=1.0, linestyle="--", zorder=1)
                ax.text(
                    0.02, context,
                    f"  design effect {context:.2f} (different definition, not a target)",
                    color=INK_MUTED, fontsize=8, va="top",
                    transform=ax.get_yaxis_transform(),
                )

        ends = []
        for system, name, color, marker, linestyle in systems:
            xs, means, stds = series(grouped, rho, system, metric)
            if not xs:
                continue
            ax.errorbar(
                xs, means, yerr=stds, color=color, marker=marker, linestyle=linestyle,
                linewidth=2, markersize=6, capsize=3, elinewidth=1,
                markeredgecolor=SURFACE, markeredgewidth=1.2, label=name, zorder=3,
            )
            short = name.split(" (")[0].replace("Naive weights + discount at inference", "Naive + discount")
            ends.append((xs[-1], means[-1], short))

        ax.set_title(f"rho = {rho}", color=INK_PRIMARY, fontsize=11, loc="left")
        ax.set_xlabel("copies of view 0 appended (k)", color=INK_SECONDARY, fontsize=9.5)
        ax.set_xticks(sorted({k for (r, _, k) in grouped if r == rho}))
        ax.set_xlim(-0.3, max(k for (r, _, k) in grouped if r == rho) + 1.4)
        place_end_labels(ax, ends, fig.dpi)

    axes[0].set_ylabel(label, color=INK_SECONDARY, fontsize=9.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=len(labels), frameon=False,
        fontsize=9, labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 1.0),
    )
    fig.suptitle(
        f"{label} vs duplicated views  ({n_seeds} seeds, mean +/- sd)",
        color=INK_PRIMARY, fontsize=12, x=0.01, y=1.07, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    path = out_dir / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description="Plot the clone experiment.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-id", default="clone")
    args = parser.parse_args()

    out_dir = Path(args.results_dir) / args.experiment_id
    csv_path = out_dir / "runs.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run run_clone.py first")

    grouped, base_views = load(csv_path)
    n_seeds = len(next(iter(grouped.values())))

    for metric, label, filename, reference in FIGURES:
        print("wrote", draw(grouped, base_views, metric, label, filename, reference, out_dir, n_seeds))


if __name__ == "__main__":
    main()
