"""Calibration metrics: ECE, MCE, Brier score and its Murphy decomposition.

All functions are post-hoc statistical estimators over numpy arrays. Nothing
here touches a model.

Binning convention (shared by ECE, MCE, the Brier decomposition and the
reliability diagram): `n_bins` equal-width bins on [0, 1]. Bin b covers
(b/M, (b+1)/M], except the first, which also includes 0.

BRIER DECOMPOSITION, AND WHY THERE IS A FOURTH TERM
---------------------------------------------------
Murphy's decomposition, Brier = reliability - resolution + uncertainty, is an
identity only when every forecast in a bin has the same value. Continuous model
probabilities do not, so binning them leaves a within-bin remainder
(Stephenson, Coelho & Jolliffe 2008):

    Brier = reliability - resolution + uncertainty + within_bin
    within_bin = within-bin forecast variance - 2 * within-bin covariance

`within_bin` is reported rather than absorbed into one of the other three, so
the three named components are never silently adjusted. With forecasts that are
constant within bins it is exactly 0, and the classical three-term identity
holds.

The multiclass Brier score, mean_i sum_k (p_ik - y_ik)^2 (range [0, 2]), is
decomposed one class at a time (one-vs-rest) and the per-class components are
summed. That sum reconstructs the multiclass score exactly.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_N_BINS = 15


def _validate_probs(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be [N, K], got shape {probs.shape}")
    if labels.shape != (probs.shape[0],):
        raise ValueError(f"labels must be [N] matching probs, got {labels.shape}")
    if probs.shape[0] == 0:
        raise ValueError("need at least one sample")
    if labels.min() < 0 or labels.max() >= probs.shape[1]:
        raise ValueError("labels out of range for the number of classes")
    return probs, labels


def bin_indices(confidence: np.ndarray, n_bins: int = DEFAULT_N_BINS) -> np.ndarray:
    """Bin index in [0, n_bins) for each value in [0, 1] (see module docstring)."""
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    confidence = np.asarray(confidence, dtype=np.float64)
    if np.any(confidence < 0.0) or np.any(confidence > 1.0) or np.any(np.isnan(confidence)):
        raise ValueError("confidence values must lie in [0, 1]")
    return np.clip(np.ceil(confidence * n_bins).astype(np.int64) - 1, 0, n_bins - 1)


@dataclass
class ReliabilityBins:
    """Per-bin statistics for a reliability diagram. Empty bins hold NaN."""

    bin_lower: np.ndarray  # [M]
    bin_upper: np.ndarray  # [M]
    count: np.ndarray  # [M]
    mean_confidence: np.ndarray  # [M]
    accuracy: np.ndarray  # [M]

    def to_dict(self) -> dict:
        return {key: np.asarray(value).tolist() for key, value in asdict(self).items()}


def reliability_bins(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> ReliabilityBins:
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    if confidence.shape != correct.shape or confidence.ndim != 1:
        raise ValueError("confidence and correct must be matching 1-D arrays")

    index = bin_indices(confidence, n_bins)
    count = np.bincount(index, minlength=n_bins).astype(np.int64)
    conf_sum = np.bincount(index, weights=confidence, minlength=n_bins)
    correct_sum = np.bincount(index, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_confidence = np.where(count > 0, conf_sum / count, np.nan)
        accuracy = np.where(count > 0, correct_sum / count, np.nan)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    return ReliabilityBins(edges[:-1], edges[1:], count, mean_confidence, accuracy)


def expected_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> float:
    """ECE = sum_b (n_b / N) |accuracy_b - mean_confidence_b|."""
    bins = reliability_bins(confidence, correct, n_bins)
    present = bins.count > 0
    gaps = np.abs(bins.accuracy[present] - bins.mean_confidence[present])
    return float(np.sum(bins.count[present] * gaps) / np.sum(bins.count))


def maximum_calibration_error(
    confidence: np.ndarray, correct: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> float:
    """MCE = max over non-empty bins of |accuracy_b - mean_confidence_b|."""
    bins = reliability_bins(confidence, correct, n_bins)
    present = bins.count > 0
    return float(np.max(np.abs(bins.accuracy[present] - bins.mean_confidence[present])))


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean_i sum_k (p_ik - y_ik)^2, in [0, 2]."""
    probs, labels = _validate_probs(probs, labels)
    onehot = np.eye(probs.shape[1])[labels]
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


@dataclass
class BrierDecomposition:
    brier: float
    reliability: float  # lower is better: calibration error of the forecasts
    resolution: float  # higher is better: how far outcomes differ between bins
    uncertainty: float  # property of the labels alone, not of the model
    within_bin: float  # binning remainder, 0 for forecasts constant within bins

    def reconstruction(self) -> float:
        return self.reliability - self.resolution + self.uncertainty + self.within_bin

    def to_dict(self) -> dict:
        return asdict(self)


def _binary_decomposition(forecast: np.ndarray, outcome: np.ndarray, n_bins: int):
    n = forecast.shape[0]
    index = bin_indices(forecast, n_bins)
    count = np.bincount(index, minlength=n_bins)
    present = count > 0
    forecast_mean = np.bincount(index, weights=forecast, minlength=n_bins)[present] / count[present]
    outcome_mean = np.bincount(index, weights=outcome, minlength=n_bins)[present] / count[present]
    base_rate = outcome.mean()

    reliability = np.sum(count[present] * (forecast_mean - outcome_mean) ** 2) / n
    resolution = np.sum(count[present] * (outcome_mean - base_rate) ** 2) / n
    uncertainty = base_rate * (1.0 - base_rate)

    # Map each sample back to its bin's means (bins renumbered to present-only).
    position = np.cumsum(present) - 1
    f_dev = forecast - forecast_mean[position[index]]
    o_dev = outcome - outcome_mean[position[index]]
    within_bin = (np.sum(f_dev**2) - 2.0 * np.sum(f_dev * o_dev)) / n
    return reliability, resolution, uncertainty, within_bin


def brier_decomposition(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = DEFAULT_N_BINS
) -> BrierDecomposition:
    """Murphy decomposition of the multiclass Brier score (see module docstring)."""
    probs, labels = _validate_probs(probs, labels)
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probs must lie in [0, 1]")
    onehot = np.eye(probs.shape[1])[labels]

    totals = np.zeros(4)
    for k in range(probs.shape[1]):
        totals += _binary_decomposition(probs[:, k], onehot[:, k], n_bins)

    return BrierDecomposition(
        brier=brier_score(probs, labels),
        reliability=float(totals[0]),
        resolution=float(totals[1]),
        uncertainty=float(totals[2]),
        within_bin=float(totals[3]),
    )


# --- reliability diagram ------------------------------------------------

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_COLOR = "#2a78d6"


def plot_reliability_diagram(
    panels: list[dict],
    path,
    title: str,
):
    """Draw one reliability panel per confidence definition and save to `path`.

    Each panel dict holds:
        label        panel title, e.g. "top-label probability"
        bins         list of ReliabilityBins, one per seed (same n_bins)
    Per bin, the point is the mean accuracy across the seeds whose bin is
    non-empty, with sample std as the error bar (none when only one seed
    populates the bin). The lower strip shows the mean sample count per bin, so
    a bin backed by a handful of samples is visible as such.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, len(panels), figsize=(5.0 * len(panels), 5.6), sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1.0]}, squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)

    for column, panel in enumerate(panels):
        per_seed = panel["bins"]
        top, bottom = axes[0][column], axes[1][column]
        for ax in (top, bottom):
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(AXIS)
            ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)

        lower, upper = per_seed[0].bin_lower, per_seed[0].bin_upper
        centers = (lower + upper) / 2.0
        width = upper[0] - lower[0]
        accuracy = np.stack([b.accuracy for b in per_seed])
        confidence = np.stack([b.mean_confidence for b in per_seed])
        counts = np.stack([b.count for b in per_seed]).astype(np.float64)

        populated = np.sum(~np.isnan(accuracy), axis=0)
        present = populated > 0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN bins
            acc_mean = np.nanmean(accuracy, axis=0)
            conf_mean = np.nanmean(confidence, axis=0)
            acc_std = np.where(populated > 1, np.nanstd(accuracy, axis=0, ddof=1), 0.0)

        top.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=1.0, linestyle="--", zorder=1)
        top.text(0.60, 0.55, "perfect calibration", color=INK_MUTED, fontsize=8, ha="left", va="top")
        top.errorbar(
            conf_mean[present], acc_mean[present], yerr=acc_std[present],
            color=SERIES_COLOR, marker="o", linestyle="-", linewidth=2, markersize=5,
            capsize=3, elinewidth=1, markeredgecolor=SURFACE, markeredgewidth=1.0, zorder=3,
        )
        top.set_xlim(0, 1)
        top.set_ylim(0, 1)
        top.set_title(panel["label"], color=INK_PRIMARY, fontsize=11, loc="left")
        top.set_ylabel("accuracy in bin", color=INK_SECONDARY, fontsize=9.5)

        bottom.bar(centers, counts.mean(axis=0), width=width * 0.9, color=AXIS, zorder=2)
        bottom.set_ylabel("mean count", color=INK_SECONDARY, fontsize=9.5)
        bottom.set_xlabel("mean confidence in bin", color=INK_SECONDARY, fontsize=9.5)

    fig.suptitle(
        f"{title}  ({len(panels[0]['bins'])} seeds, mean +/- sd per bin)",
        color=INK_PRIMARY, fontsize=12, x=0.01, ha="left",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path
