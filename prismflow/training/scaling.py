"""Per-(view, native feature) z-scoring, with the widths passed in.

Line-for-line the arithmetic of `run_ait_clean.fit_scaler` / `apply_scaler`
and their `run_lanl` mirrors, with one change: `native_dims` is an argument
instead of `[len(FEATURES[s]) for s in dataset.sources]`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def fit_scaler(views: np.ndarray, view_mask: np.ndarray,
               native_dims: Sequence[int], epsilon: float) -> dict:
    """Mean and std per (view, native feature) over rows where the view is PRESENT.

    An absent view's row is all zeros and carries no observation, so including
    it would pull the mean toward zero by an amount that depends only on the
    missing rate. Padded columns beyond a view's native width are not fitted
    and are left at 0 by `apply_scaler`.

    `views` is [n, V, D] and `view_mask` is [n, V], already subset to the rows
    being fitted on -- the caller does the index selection, because which rows
    are the train block is a dataset question.
    """
    if views.ndim != 3:
        raise ValueError("views must be [n, V, D], got %r" % (views.shape,))
    if view_mask.shape != views.shape[:2]:
        raise ValueError("view_mask %r does not match views %r"
                         % (view_mask.shape, views.shape))
    n_views, dim = views.shape[1], views.shape[2]
    if len(native_dims) != n_views:
        raise ValueError("native_dims has %d entries for %d views"
                         % (len(native_dims), n_views))
    if any(d > dim for d in native_dims):
        raise ValueError("a native width exceeds the padded width %d: %r"
                         % (dim, list(native_dims)))

    native = [int(d) for d in native_dims]
    mean = np.zeros((n_views, dim), dtype=np.float64)
    std = np.ones((n_views, dim), dtype=np.float64)
    present_counts = []
    for v in range(n_views):
        rows = view_mask[:, v]
        present_counts.append(int(rows.sum()))
        if rows.sum() < 2:
            # Cannot fit a scale from fewer than two observations. Leaving
            # mean 0 / std 1 is the identity, not an estimate.
            continue
        block = views[rows, v, : native[v]].astype(np.float64)
        mean[v, : native[v]] = block.mean(axis=0)
        std[v, : native[v]] = block.std(axis=0, ddof=0)
    std = np.where(std > float(epsilon), std, 1.0)
    return {
        "mean": mean,
        "std": std,
        "native_dims": native,
        "fit_rows": int(views.shape[0]),
        "present_rows_per_view": present_counts,
    }


def apply_scaler(views: np.ndarray, view_mask: np.ndarray,
                 scaler: dict) -> np.ndarray:
    """z-score the native columns of PRESENT views; leave everything else at 0.

    Returns a new float32 [n, V, D] array. The caller wraps it back into
    whichever dataset class it uses, which is the only dataset-specific step
    and is one line at the call site.
    """
    views = np.asarray(views).copy()
    for v, native in enumerate(scaler["native_dims"]):
        rows = view_mask[:, v]
        if not rows.any():
            continue
        block = views[rows, v, :native].astype(np.float64)
        block = (block - scaler["mean"][v, :native]) / scaler["std"][v, :native]
        views[rows, v, :native] = block.astype(np.float32)
        # An absent view contributes nothing (the encoder never feeds it
        # forward), so its row stays the zero vector it arrived as.
        views[~rows, v] = 0.0
    return views.astype(np.float32)
