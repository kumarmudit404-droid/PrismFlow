"""Plot estimated ENIV against the analytic design effect, with error bars.

Reads `results/eniv_validation/estimator_vs_truth.csv` and writes
`estimator_vs_truth.png` next to it.

The analytic curve is drawn as the reference, not as one series among many:
it is the known right answer, and the question the plot has to answer at a
glance is how far the estimators sit from it and whether they bend the same
way.

Usage:
    python -m experiments.eniv_validation.plot_validation
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display on the machines this runs on
import matplotlib.pyplot as plt  # noqa: E402

SERIES = [
    ("eniv_pearson_holdout", "Pearson (pairwise holdout)", "#1f77b4", "o"),
    ("eniv_pearson_global", "Pearson (global)", "#ff7f0e", "s"),
    ("eniv_dcor_holdout", "dCor (pairwise holdout)", "#2ca02c", "^"),
    ("eniv_dcor_global", "dCor (global)", "#d62728", "v"),
    ("eniv_views_cca", "View-space CCA (data check)", "#8c564b", "D"),
]


def load(csv_path: Path):
    grouped = defaultdict(lambda: defaultdict(list))
    analytic = {}

    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rho = float(row["rho"])
            analytic[rho] = float(row["analytic_n_eff"])
            for key, _, _, _ in SERIES:
                grouped[key][rho].append(float(row[key]))

    return sorted(analytic), analytic, grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ENIV estimator validation.")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.results_dir) / "eniv_validation"
    csv_path = out_dir / "estimator_vs_truth.csv"
    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found -- run run_validation.py first")

    rhos, analytic, grouped = load(csv_path)
    n_seeds = len(next(iter(grouped[SERIES[0][0]].values())))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13, 5.2))

    left.plot(
        rhos,
        [analytic[r] for r in rhos],
        color="black",
        linewidth=2.5,
        linestyle="--",
        label="analytic n_eff (truth)",
        zorder=5,
    )
    for key, label, color, marker in SERIES:
        means = [statistics.mean(grouped[key][r]) for r in rhos]
        stds = [
            statistics.stdev(grouped[key][r]) if len(grouped[key][r]) > 1 else 0.0
            for r in rhos
        ]
        left.errorbar(
            rhos, means, yerr=stds, label=label, color=color, marker=marker,
            capsize=4, markersize=6, linewidth=1.6, alpha=0.9,
        )

    left.set_xlabel("true cross-view correlation  rho")
    left.set_ylabel("effective number of independent views")
    left.set_title(f"ENIV vs analytic design effect ({n_seeds} seeds, mean +/- sd)")
    left.legend(fontsize=8)
    left.grid(alpha=0.3)

    limit = max(max(analytic.values()), 4.2)
    right.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=2, label="perfect")
    for key, label, color, marker in SERIES:
        means = [statistics.mean(grouped[key][r]) for r in rhos]
        stds = [
            statistics.stdev(grouped[key][r]) if len(grouped[key][r]) > 1 else 0.0
            for r in rhos
        ]
        right.errorbar(
            [analytic[r] for r in rhos], means, yerr=stds, label=label, color=color,
            marker=marker, capsize=4, markersize=6, linestyle="none", alpha=0.9,
        )

    right.set_xlabel("analytic n_eff (truth)")
    right.set_ylabel("estimated n_eff")
    right.set_title("Estimated vs true  (above the line = under-discounting)")
    right.legend(fontsize=8)
    right.grid(alpha=0.3)

    figure.tight_layout()
    png_path = out_dir / "estimator_vs_truth.png"
    figure.savefig(png_path, dpi=150)
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
