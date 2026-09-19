"""Clique contrast: internal agreement MINUS external agreement.

A STATISTICAL ESTIMATOR per docs/CONTRACT.md: no learnable parameters, no
nn.Module, no gradients, and no access to attack ground truth.

WHY THIS SIGNAL EXISTS
----------------------
Every dependence-style signal in this project so far is PAIRWISE AND SYMMETRIC:
the dependence matrix, the ENIV aggregation over it, the Part 10
unexplained-agreement score, and Part 13's co-exceedance. A pairwise symmetric
statistic sees "a clique of views that agree with each other" and cannot ask
whether that clique also agrees with everyone ELSE.

That is exactly the distinction the project keeps failing on. L5 records the
symptom: honest duplication (clone_k2) scores 0.8743 on the suspicion detector,
HIGHER than every adversarial chorus condition tested. Honest redundancy and
adversarial collusion both produce an internally-agreeing clique, so no measure
of internal agreement alone can separate them.

What differs is the clique's relationship to the REST of the system. Honest
duplicates agree with each other and with the other views, because they are all
reporting the same true class. A colluding clique agrees with each other and
DISSENTS from the others, because it is reporting a class the honest views do
not support.

THE STATISTIC
-------------
For each view v, its agreements with the other views are sorted descending and
split at every possible point g:

    contrast_v = max over g of [ mean(top g) - mean(the rest) ]

and the reported value is the maximum over views. The split point is not fixed
because the clique size k is not known -- taking the best split is what lets one
statistic cover k = 1, 2 or 3 without being told which.

Reading it:

    ~0          this view's agreement is undifferentiated; it agrees with
                everyone about equally (honest duplication, or clean data)
    clearly >0  this view has a subset it agrees with markedly more than the
                rest -- a clique that is NOT carrying the rest with it

NOT A DETECTOR ON ITS OWN, AND NOT VALIDATED AS ONE. This module supplies a
signal. Whether it separates honest duplication from collusion ON REAL DATA is
an experimental question; `tests/unit/test_clique_contrast.py` only pins that it
reads the constructed structures correctly, which is a precondition for trusting
any experiment that uses it, not a result about attacks.
"""

from __future__ import annotations

import numpy as np

__all__ = ["clique_contrast", "clique_contrast_matrix", "per_view_contrast"]


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu()
    return np.asarray(array, dtype=np.float64)


def per_view_contrast(stacked: np.ndarray) -> np.ndarray:
    """Best-split contrast for every view -> [..., V].

    stacked  [..., V, V] agreement or dependence, diagonal ignored, NaN allowed
             for unmeasurable pairs.

    A view whose comparisons are entirely unmeasurable yields NaN rather than 0:
    refusing to score is not the same as scoring "no clique", which would quietly
    license confidence the data does not support.
    """
    if stacked.ndim < 2 or stacked.shape[-1] != stacked.shape[-2]:
        raise ValueError(f"expected [..., V, V], got shape {stacked.shape}")
    n_views = stacked.shape[-1]
    if n_views < 3:
        # With two views there is no "rest" to contrast against.
        return np.full(stacked.shape[:-1], np.nan)

    off_diagonal = ~np.eye(n_views, dtype=bool)
    off = stacked[..., off_diagonal].reshape(*stacked.shape[:-2], n_views, n_views - 1)

    # Sort descending with unmeasurable pairs sunk to the end.
    filled = np.where(np.isfinite(off), off, -np.inf)
    ordered = -np.sort(-filled, axis=-1)
    measurable = np.isfinite(ordered)

    best = np.full(ordered.shape[:-1], -np.inf, dtype=np.float64)
    for split in range(1, n_views - 1):
        top, rest = ordered[..., :split], ordered[..., split:]
        top_ok, rest_ok = np.isfinite(top), np.isfinite(rest)
        top_n, rest_n = top_ok.sum(axis=-1), rest_ok.sum(axis=-1)

        top_mean = np.where(top_ok, top, 0.0).sum(axis=-1) / np.maximum(top_n, 1)
        rest_mean = np.where(rest_ok, rest, 0.0).sum(axis=-1) / np.maximum(rest_n, 1)

        usable = (top_n > 0) & (rest_n > 0)
        best = np.maximum(best, np.where(usable, top_mean - rest_mean, -np.inf))

    # A view with fewer than two measurable comparisons has no contrast.
    enough = measurable.sum(axis=-1) >= 2
    return np.where(enough & np.isfinite(best), best, np.nan)


def clique_contrast(agreement, view_mask=None) -> np.ndarray:
    """Per-sample clique contrast from [B, V, V] agreement -> [B].

    Intended input is `suspicion.pairwise_agreement(belief)`, so this reads the
    same per-sample agreement the Part 10 detector reads -- the two signals
    differ in what they ASK of it, not in the data they see.

    UNDEFINED, NOT ZERO. A sample needs at least three usable views: with two,
    each view has a single comparison, and one comparison cannot be split into
    a non-empty "top g" and a non-empty "rest". Such a sample returns NaN. It
    must NOT return 0.0 -- 0.0 is a meaningful reading in this signal's range
    ("agrees with everyone about equally") and would assert something the data
    does not support. Callers are expected to drop NaN rows and COUNT them:
    the samples that land here are the ones where the model's evidence is
    near-vacuous, so the drops are not randomly distributed.
    """
    stacked = _to_numpy(agreement)
    if stacked.ndim != 3:
        raise ValueError(f"expected [B, V, V], got shape {stacked.shape}")

    if view_mask is not None:
        mask = _to_numpy(view_mask).astype(bool)
        if mask.shape != stacked.shape[:2]:
            raise ValueError(f"view_mask must be {stacked.shape[:2]}, got {mask.shape}")
        both = mask[:, :, None] & mask[:, None, :]
        stacked = np.where(both, stacked, np.nan)

    contrasts = per_view_contrast(stacked)

    # np.nanmax warns on an all-NaN slice, which is a case this function
    # EXPECTS (see the docstring) rather than an error. Taking the max over
    # -inf-filled values gives the identical result wherever any view is
    # measurable, and lets the undefined rows be named explicitly instead of
    # arriving as a RuntimeWarning from inside a library call.
    measurable = np.isfinite(contrasts)
    out = np.where(measurable, contrasts, -np.inf).max(axis=-1)
    return np.where(measurable.any(axis=-1), out, np.nan)


def clique_contrast_matrix(matrix) -> float:
    """Structural clique contrast from a single [V, V] dependence matrix.

    The batch-level counterpart of `clique_contrast`, for reading a dependence
    matrix directly rather than per-sample agreement.
    """
    stacked = _to_numpy(matrix)
    if stacked.ndim != 2 or stacked.shape[0] != stacked.shape[1]:
        raise ValueError(f"expected a square [V, V] matrix, got {stacked.shape}")
    contrasts = per_view_contrast(stacked)
    finite = contrasts[np.isfinite(contrasts)]
    return float(finite.max()) if finite.size else float("nan")
