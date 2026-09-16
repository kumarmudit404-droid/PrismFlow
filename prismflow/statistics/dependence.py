"""Cross-view dependence estimation from per-view evidence.

A STATISTICAL ESTIMATOR per docs/CONTRACT.md: no learnable parameters, no
nn.Module, no gradients. Everything here runs in numpy on detached inputs.

WHY CLASS-CONDITIONAL RESIDUALS
-------------------------------
Views SHOULD agree when they are all correct, because that is the task
working as intended. What we need to detect is agreement BEYOND what the
label explains -- two views that move together even after you account for the
class they are both reporting. Unconditioned correlation cannot tell those
apart: it scores a well-functioning ensemble and a room full of copies of the
same sensor identically, and the whole project rests on that distinction.

So before correlating, subtract the per-class mean from each view's evidence
and correlate what is left over.

Conditioning uses the PREDICTED class, not the true label. Ground-truth labels
are unavailable at inference, and a dependence estimate that needs them could
not run in deployment (nor, later, inside the suspicion detector, which the
contract forbids from seeing ground truth).

COLLIDER BIAS, AND WHY THERE ARE TWO CONDITIONING MODES
-------------------------------------------------------
The predicted class is a function of every view at once, so conditioning on it
conditions on a collider: knowing the batch predicted class k, a low reading
from view i implies a higher reading from view j to compensate. That induces
NEGATIVE correlation between views that are genuinely independent. Measured on
pure noise it is around -0.24 at V=2, -0.11 at V=4, -0.05 at V=8 -- it decays
like 1/(V-1) but never vanishes.

The direction is the dangerous one. A negative bias understates dependence,
which inflates ENIV, which weakens the discount and leaves the model
overconfident -- the exact failure this project exists to prevent.

  "global"            conditions on the class predicted from all views. This
                      is the literal specification and the cheapest option,
                      and it carries the bias described above.

  "pairwise_holdout"  for each pair (i, j), conditions on the class predicted
                      from the OTHER views only. The conditioning variable is
                      then not a function of either view being correlated, so
                      the collider is broken. Falls back to "global" when
                      fewer than three views are available, since holding out
                      both leaves nothing to predict from.

Supplying `classes` explicitly bypasses both modes.

METHODS
-------
`pearson` detects linear dependence only -- it is fast, unbiased around zero,
and is the quantity the design-effect formula in `eniv.py` was derived for.

`dcor` (distance correlation) is zero if and only if the variables are
independent, so it catches non-linear coupling Pearson misses. It is not on
the same scale as Pearson: for jointly Gaussian variables dCor is strictly
below |r|, so feeding it to the same design-effect formula understates
dependence. Treat the two as different instruments, not interchangeable ones.
"""

from __future__ import annotations

import numpy as np

_VALID_METHODS = ("pearson", "dcor")
_VALID_CONDITIONING = ("global", "pairwise_holdout")

# Below this many jointly-available samples a pairwise correlation is noise.
_MIN_PAIR_SAMPLES = 8

# dCor builds an [n, n] distance matrix per view; subsample above this.
_DCOR_MAX_SAMPLES = 512


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def predicted_classes(evidence, view_mask=None) -> np.ndarray:
    """Pre-fusion class prediction: argmax of evidence summed over views.

    Used as the conditioning variable when no explicit class is supplied. It
    is deliberately cheap and label-free -- it only has to define the strata
    the residuals are taken within, not to be the model's final answer.
    """
    evidence = _to_numpy(evidence)
    if view_mask is not None:
        evidence = evidence * _to_numpy(view_mask)[:, :, None]
    return evidence.sum(axis=1).argmax(axis=-1)


def class_conditional_residuals(
    evidence, classes, view_mask=None
) -> tuple[np.ndarray, np.ndarray]:
    """Subtract the per-class, per-view, per-dimension mean.

    evidence:  [B, V, K]
    classes:   [B] integer stratum for each sample
    view_mask: [B, V] boolean, True = view present

    Returns (residuals [B, V, K], valid [B, V] boolean).

    A stratum with only one available sample for a given view carries no
    information once centred (its residual is exactly zero), so it is marked
    invalid rather than contributing a spurious zero to the correlation.
    """
    evidence = _to_numpy(evidence)
    classes = np.asarray(_to_numpy(classes), dtype=np.int64)

    if evidence.ndim != 3:
        raise ValueError(f"evidence must be [B, V, K], got {evidence.shape}")
    n_samples, n_views, _ = evidence.shape
    if classes.shape != (n_samples,):
        raise ValueError(f"classes must be [{n_samples}], got {classes.shape}")

    if view_mask is None:
        available = np.ones((n_samples, n_views), dtype=bool)
    else:
        available = _to_numpy(view_mask).astype(bool)
        if available.shape != (n_samples, n_views):
            raise ValueError(
                f"view_mask must be [{n_samples}, {n_views}], got {available.shape}"
            )

    residuals = np.zeros_like(evidence)
    valid = np.zeros((n_samples, n_views), dtype=bool)

    for stratum in np.unique(classes):
        in_stratum = classes == stratum
        for view in range(n_views):
            rows = in_stratum & available[:, view]
            count = int(rows.sum())
            if count < 2:
                continue
            block = evidence[rows, view, :]
            residuals[rows, view, :] = block - block.mean(axis=0, keepdims=True)
            valid[rows, view] = True

    return residuals, valid


def _pearson_pair(residual_a: np.ndarray, residual_b: np.ndarray) -> float:
    """Mean per-dimension Pearson correlation between two [n, K] residuals.

    Each dimension is standardised before the correlation is taken, which
    makes the flattened Pearson exactly the unweighted mean of the per-class
    correlations -- so one high-variance evidence dimension cannot dominate
    the view-level number.
    """
    std_a = residual_a.std(axis=0)
    std_b = residual_b.std(axis=0)
    usable = (std_a > 1e-12) & (std_b > 1e-12)
    if not usable.any():
        return np.nan

    z_a = residual_a[:, usable] / std_a[usable]
    z_b = residual_b[:, usable] / std_b[usable]
    per_dim = (z_a * z_b).mean(axis=0)
    return float(np.clip(per_dim.mean(), -1.0, 1.0))


def _double_center(matrix: np.ndarray) -> np.ndarray:
    row = matrix.mean(axis=0, keepdims=True)
    col = matrix.mean(axis=1, keepdims=True)
    return matrix - row - col + matrix.mean()


def _dcor_pair(residual_a: np.ndarray, residual_b: np.ndarray, rng) -> float:
    """Distance correlation, zero iff the two residual sets are independent."""
    n_samples = residual_a.shape[0]
    if n_samples > _DCOR_MAX_SAMPLES:
        picked = rng.choice(n_samples, size=_DCOR_MAX_SAMPLES, replace=False)
        residual_a = residual_a[picked]
        residual_b = residual_b[picked]

    dist_a = np.linalg.norm(residual_a[:, None, :] - residual_a[None, :, :], axis=-1)
    dist_b = np.linalg.norm(residual_b[:, None, :] - residual_b[None, :, :], axis=-1)

    centered_a = _double_center(dist_a)
    centered_b = _double_center(dist_b)

    dcov2 = (centered_a * centered_b).mean()
    dvar_a = (centered_a * centered_a).mean()
    dvar_b = (centered_b * centered_b).mean()

    denom = np.sqrt(dvar_a * dvar_b)
    if denom <= 1e-12:
        return np.nan
    return float(np.clip(np.sqrt(max(dcov2, 0.0) / denom), 0.0, 1.0))


def _holdout_classes(evidence, view_mask, exclude) -> np.ndarray | None:
    """Class predicted from every view except those in `exclude`."""
    keep = [v for v in range(evidence.shape[1]) if v not in exclude]
    if not keep:
        return None
    sub_mask = None if view_mask is None else view_mask[:, keep]
    return predicted_classes(evidence[:, keep, :], sub_mask)


def dependence_matrix(
    evidence,
    view_mask=None,
    classes=None,
    method: str = "pearson",
    conditioning: str = "global",
    seed: int = 0,
) -> np.ndarray:
    """Symmetric [V, V] cross-view dependence, unit diagonal.

    evidence:     [B, V, K] per-view evidence
    view_mask:    [B, V] boolean, True = view present
    classes:      [B] explicit conditioning strata; overrides `conditioning`
    method:       "pearson" or "dcor"
    conditioning: "global" or "pairwise_holdout" (see module docstring)

    A pair of views with fewer than 8 jointly-available samples yields NaN
    rather than a number. Refusing to estimate is not the same as estimating
    zero: zero would claim the pair is independent and quietly license full
    confidence. Callers decide what to do with an unmeasurable pair.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    if conditioning not in _VALID_CONDITIONING:
        raise ValueError(
            f"conditioning must be one of {_VALID_CONDITIONING}, got {conditioning!r}"
        )

    evidence = _to_numpy(evidence)
    if evidence.ndim != 3:
        raise ValueError(f"evidence must be [B, V, K], got {evidence.shape}")
    n_views = evidence.shape[1]

    mask = None if view_mask is None else _to_numpy(view_mask).astype(bool)
    explicit = classes is not None
    global_classes = (
        np.asarray(_to_numpy(classes), dtype=np.int64)
        if explicit
        else predicted_classes(evidence, mask)
    )

    rng = np.random.default_rng(seed)
    matrix = np.eye(n_views, dtype=np.float64)

    for i in range(n_views):
        for j in range(i + 1, n_views):
            pair_classes = global_classes
            if not explicit and conditioning == "pairwise_holdout":
                held_out = _holdout_classes(evidence, mask, {i, j})
                if held_out is not None:
                    pair_classes = held_out

            pair_evidence = evidence[:, [i, j], :]
            pair_mask = None if mask is None else mask[:, [i, j]]
            residuals, valid = class_conditional_residuals(
                pair_evidence, pair_classes, pair_mask
            )

            both = valid[:, 0] & valid[:, 1]
            if int(both.sum()) < _MIN_PAIR_SAMPLES:
                value = np.nan
            elif method == "pearson":
                value = _pearson_pair(residuals[both, 0, :], residuals[both, 1, :])
            else:
                value = _dcor_pair(residuals[both, 0, :], residuals[both, 1, :], rng)

            matrix[i, j] = value
            matrix[j, i] = value

    return matrix


def available_views(view_mask=None, n_views: int | None = None) -> np.ndarray:
    """Boolean [V] -- which views are present for at least one sample."""
    if view_mask is None:
        if n_views is None:
            raise ValueError("provide view_mask or n_views")
        return np.ones(n_views, dtype=bool)
    return _to_numpy(view_mask).astype(bool).any(axis=0)
