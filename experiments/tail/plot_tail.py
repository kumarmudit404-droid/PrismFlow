"""The Part 13 figure: average co-movement against joint extreme agreement.

One figure, four panels, from `results/tail/tail.json`:

  1  rho_bar and lambda_U against attack strength -- THE headline. If the
     standard measure is flat while the tail measure climbs, this panel is the
     argument; if both move together, the panel says so just as plainly.
  2  ENIV against tail-aware ENIV, the same contrast in the units the fusion
     rule actually consumes.
  3  Detection: tail flag vs the Part 10 rho_bar flag, AUC against epsilon.
  4  The sample-size caveat: lambda_U at the full split against a 300-sample
     subsample, with the exceedance count annotated.

Error bars are +/- one sample std across seeds. Panels whose estimate rests on
fewer than MIN_TAIL_SAMPLES exceedances are marked, because an unmarked
unreliable point in a headline figure is how a caveat gets lost.

Usage:
    python -m experiments.tail.plot_tail
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from prismflow.statistics.tail_dependence import MIN_TAIL_SAMPLES  # noqa: E402

RESULTS = Path("results/tail")
RHO_COLOUR = "#B45309"
TAIL_COLOUR = "#1D4ED8"


def series(protocol, cells, *path):
    """(means, stds) for one metric across the epsilon cells, None-safe."""
    means, stds = [], []
    for cell in cells:
        node = protocol[cell]
        for key in path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        mean = node.get("mean") if isinstance(node, dict) else None
        std = node.get("std") if isinstance(node, dict) else None
        means.append(float("nan") if mean is None else float(mean))
        stds.append(0.0 if std is None else float(std))
    return means, stds


def main():
    payload = json.loads((RESULTS / "tail.json").read_text(encoding="utf-8"))
    config, protocol, detection = payload["config"], payload["protocol"], payload["detection"]
    epsilons = config["attack"]["epsilons"]
    cells = [f"eps{e}" for e in epsilons]

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    figure.suptitle(
        "Part 13 - average co-movement is not joint extreme agreement\n"
        f"Chorus attack, k={config['attack']['k']}, {len(config['seeds'])} seeds, "
        f"q={config['tail']['headline_quantile']}, bars = +/-1 sd across seeds",
        fontsize=12,
    )

    # ---- panel 1: the headline -------------------------------------------
    ax = axes[0][0]
    rho_mean, rho_std = series(protocol, cells, "rho_bar_compromised")
    tail_mean, tail_std = series(protocol, cells, "lambda_u_compromised")
    rho_h, _ = series(protocol, cells, "rho_bar_honest")
    tail_h, _ = series(protocol, cells, "lambda_u_honest")
    n_tail, _ = series(protocol, cells, "n_tail_min")

    ax.errorbar(epsilons, rho_mean, yerr=rho_std, marker="o", color=RHO_COLOUR,
                capsize=3, label=r"$\bar{\rho}$ colluding pair")
    ax.errorbar(epsilons, tail_mean, yerr=tail_std, marker="s", color=TAIL_COLOUR,
                capsize=3, label=r"$\lambda_U$ colluding pair")
    ax.plot(epsilons, rho_h, marker=".", linestyle=":", color=RHO_COLOUR, alpha=0.6,
            label=r"$\bar{\rho}$ honest pairs")
    ax.plot(epsilons, tail_h, marker=".", linestyle=":", color=TAIL_COLOUR, alpha=0.6,
            label=r"$\lambda_U$ honest pairs")
    ax.set_xlabel(r"attack strength $\epsilon$")
    ax.set_ylabel("dependence")
    ax.set_title(
        f"1. Pearson HOLDS, tail dependence COLLAPSES\n(colluding pair; "
        f"n_tail = {n_tail[0]:.0f} per estimate)",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---- panel 2: what fusion consumes -----------------------------------
    ax = axes[0][1]
    eniv_mean, eniv_std = series(protocol, cells, "eniv")
    teniv_mean, teniv_std = series(protocol, cells, "tail_eniv")
    ax.errorbar(epsilons, eniv_mean, yerr=eniv_std, marker="o", color=RHO_COLOUR,
                capsize=3, label="ENIV (Part 09 features)")
    ax.errorbar(epsilons, teniv_mean, yerr=teniv_std, marker="s", color=TAIL_COLOUR,
                capsize=3, label=r"tail ENIV ($\lambda_U$)")
    ax.axhline(config["data"]["n_views"], color="grey", linestyle=":", linewidth=1,
               label=f"nominal V = {config['data']['n_views']}")
    ax.set_xlabel(r"attack strength $\epsilon$")
    ax.set_ylabel("effective number of independent views")
    ax.set_title("2. In the units the discount consumes")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ---- panel 3: detection ----------------------------------------------
    ax = axes[1][0]
    attacked = [c for c in cells if c in detection]
    attacked_eps = [e for e, c in zip(epsilons, cells) if c in detection]
    if attacked:
        for detector, colour, label in (
            ("tail", TAIL_COLOUR, r"$\lambda_U$ co-exceedance flag"),
            ("rho_bar", RHO_COLOUR, "Part 10 unexplained-agreement flag"),
        ):
            means = [detection[c][detector]["auc"]["mean"] or float("nan") for c in attacked]
            stds = [detection[c][detector]["auc"]["std"] or 0.0 for c in attacked]
            ax.errorbar(attacked_eps, means, yerr=stds, marker="o", color=colour,
                        capsize=3, label=label)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1, label="chance")
    ax.set_xlabel(r"attack strength $\epsilon$")
    ax.set_ylabel("ROC AUC (attacked vs clean)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("3. Detection, attacked against clean")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # ---- panel 4: WHY the tail measure collapses --------------------------
    ax = axes[1][1]
    spread, spread_std = series(protocol, cells, "tail_spread_compromised")
    lam, lam_std = series(protocol, cells, "lambda_u_compromised")

    ax.errorbar(epsilons, lam, yerr=lam_std, marker="s", color=TAIL_COLOUR,
                capsize=3, label=r"$\lambda_U$ colluding pair")
    ax.set_xlabel(r"attack strength $\epsilon$")
    ax.set_ylabel(r"$\lambda_U$", color=TAIL_COLOUR)
    ax.tick_params(axis="y", labelcolor=TAIL_COLOUR)

    twin = ax.twinx()
    twin.errorbar(epsilons, spread, yerr=spread_std, marker="D", color="#047857",
                  capsize=3, linestyle="--", label="spread inside the tail (CV)")
    twin.set_ylabel("coefficient of variation within tail", color="#047857")
    twin.tick_params(axis="y", labelcolor="#047857")

    handles = ax.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + twin.get_legend_handles_labels()[1]
    ax.legend(handles, labels, fontsize=8, loc="upper right")
    ax.set_title(
        "4. Why: the attack FLATTENS the tail\n"
        "rank-based measures need variation they no longer have",
        fontsize=10,
    )
    ax.grid(alpha=0.3)

    figure.tight_layout(rect=(0, 0, 1, 0.94))
    out = RESULTS / "tail_vs_correlation.png"
    figure.savefig(out, dpi=160)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
