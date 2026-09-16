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

`cca` (first canonical correlation) is linear like Pearson but invariant to
the basis each side is expressed in. Use it when the two tensors have no
shared axis convention -- encoder features, where each view has its own
independently-trained encoder. Do not use it on evidence, where the class axes
already correspond and Pearson is both cheaper and unbiased.

CHANCE CORRECTION (null_permutations)
-------------------------------------
`cca` and `dcor` do not read zero when the inputs are independent. CCA
maximises a correlation over projection directions, and dCor is a
non-negative functional, so both report a positive value on pure noise. The
bias grows roughly with feature_dim / n_samples -- with 32-dimensional
features and a few hundred samples it is large enough to invent dependence
where there is none, which makes the discount fire on genuinely independent
views.

Setting `null_permutations > 0` measures that floor instead of assuming it
away: the pairing between the two views is shuffled within each conditioning
stratum, the same statistic is recomputed, and the observed value is rescaled
so the null maps to 0 and perfect dependence still maps to 1. The correction
is empirical, so it adapts to whatever n and dimension the caller actually
has rather than relying on an asymptotic formula that does not hold here.

Cost is roughly (1 + null_permutations) times the uncorrected estimate. It is
off by default because `pearson` on evidence does not need it.

CHOOSING A METHOD BY INPUT
--------------------------
  evidence [B, V, K]            axes aligned (class k is class k everywhere)
                                -> pearson
  features [B, V, feature_dim]  axes arbitrary per view
                                -> cca or dcor; pearson reads near zero
                                   regardless of the true dependence
"""

from __future__ import annotations

import numpy as np

_VALID_METHODS = ("pearson", "dcor", "cca")
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


def _sym_inv_sqrt(matrix: np.ndarray, reg: float) -> np.ndarray:
    matrix = matrix + reg * np.eye(matrix.shape[0])
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    eigenvalues = np.clip(eigenvalues, 1e-10, None)
    return eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T


def _cca_pair(residual_a: np.ndarray, residual_b: np.ndarray, reg: float = 1e-2) -> float:
    """First canonical correlation between two [n, F] residual sets.

    Necessary whenever the two tensors do not share a basis. Per-view encoders
    are trained independently, so feature dimension 5 of view i has no
    relationship to dimension 5 of view j -- an axis-aligned correlation
    averages genuine shared structure against arbitrary misalignment and
    reports near zero. Canonical correlation is invariant to any invertible
    linear remapping of either side, so it recovers the shared subspace
    whatever basis each encoder happened to learn.

    Evidence does not have this problem: class k means the same thing in every
    view, so its axes are aligned by construction and Pearson is appropriate.

    CAUTION: this maximises a correlation over projection directions, so it is
    upward-biased at finite n, roughly as sqrt(F / n). With F=32 and a few
    hundred samples that bias is substantial, and it inflates most where the
    true correlation is lowest.
    """
    n_samples = residual_a.shape[0]
    std_a = residual_a.std(axis=0)
    std_b = residual_b.std(axis=0)
    usable_a = std_a > 1e-12
    usable_b = std_b > 1e-12
    if not usable_a.any() or not usable_b.any():
        return np.nan

    z_a = residual_a[:, usable_a] / std_a[usable_a]
    z_b = residual_b[:, usable_b] / std_b[usable_b]

    cov_aa = z_a.T @ z_a / n_samples
    cov_bb = z_b.T @ z_b / n_samples
    cov_ab = z_a.T @ z_b / n_samples

    whitened = _sym_inv_sqrt(cov_aa, reg) @ cov_ab @ _sym_inv_sqrt(cov_bb, reg)
    singular_values = np.linalg.svd(whitened, compute_uv=False)
    if singular_values.size == 0:
        return np.nan
    return float(np.clip(singular_values[0], 0.0, 1.0))


def _double_center(matrix: np.ndarray) -> np.ndarray:
    row = matrix.mean(axis=0, keepdims=True)
    col = matrix.mean(axis=1, keepdims=True)
    return matrix - row - col + matrix.mean()


def _centered_distances(residual: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(residual[:, None, :] - residual[None, :, :], axis=-1)
    return _double_center(distances)


def _dcor_from_centered(centered_a: np.ndarray, centered_b: np.ndarray) -> float:
    dcov2 = (centered_a * centered_b).mean()
    dvar_a = (centered_a * centered_a).mean()
    dvar_b = (centered_b * centered_b).mean()

    denom = np.sqrt(dvar_a * dvar_b)
    if denom <= 1e-12:
        return np.nan
    return float(np.clip(np.sqrt(max(dcov2, 0.0) / denom), 0.0, 1.0))


def _dcor_pair(residual_a: np.ndarray, residual_b: np.ndarray) -> float:
    """Distance correlation, zero iff the two residual sets are independent."""
    return _dcor_from_centered(_centered_distances(residual_a), _centered_distances(residual_b))


def _within_stratum_permutation(strata: np.ndarray, rng) -> np.ndarray:
    """Shuffle sample order independently inside each conditioning stratum.

    Permuting within strata rather than globally keeps every marginal the
    estimator sees -- class proportions, per-stratum residual scale -- exactly
    as it found them, and destroys only the pairing between the two views.
    That is the null we want: "same data, same shapes, no cross-view
    relationship".
    """
    order = np.arange(strata.shape[0])
    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        order[rows] = rng.permutation(rows)
    return order


def _pair_statistic(method: str, residual_a: np.ndarray, residual_b: np.ndarray) -> float:
    if method == "pearson":
        return _pearson_pair(residual_a, residual_b)
    if method == "cca":
        return _cca_pair(residual_a, residual_b)
    return _dcor_pair(residual_a, residual_b)


def _permutation_null(
    method: str,
    residual_a: np.ndarray,
    residual_b: np.ndarray,
    strata: np.ndarray,
    rng,
    n_permutations: int,
) -> float:
    """Mean statistic under the independence null, at this n and dimension.

    dCor gets a shortcut: permuting a symmetric distance matrix's rows and
    columns by the same permutation commutes with double-centring, so the
    expensive [n, n] matrices are built once and each permutation is a cheap
    reindex rather than a full recompute.
    """
    if method == "dcor":
        centered_a = _centered_distances(residual_a)
        centered_b = _centered_distances(residual_b)
        values = []
        for _ in range(n_permutations):
            order = _within_stratum_permutation(strata, rng)
            values.append(_dcor_from_centered(centered_a, centered_b[np.ix_(order, order)]))
    else:
        values = [
            _pair_statistic(method, residual_a, residual_b[_within_stratum_permutation(strata, rng)])
            for _ in range(n_permutations)
        ]

    finite = [v for v in values if not np.isnan(v)]
    return float(np.mean(finite)) if finite else np.nan


def _chance_correct(observed: float, null: float) -> float:
    """Rescale so the independence null maps to 0 and perfect dependence to 1.

        corrected = (observed - null) / (1 - null)

    A plain subtraction would fix the zero point but drag the top of the scale
    down with it, turning a genuine correlation of 1.0 into 1 - null. This is
    the same chance correction the adjusted Rand index uses.
    """
    if np.isnan(observed) or np.isnan(null):
        return observed
    denominator = 1.0 - null
    if denominator <= 1e-9:
        return 0.0
    return float(np.clip((observed - null) / denominator, -1.0, 1.0))


def _holdout_classes(class_source, view_mask, exclude) -> np.ndarray | None:
    """Class predicted from every view except those in `exclude`."""
    keep = [v for v in range(class_source.shape[1]) if v not in exclude]
    if not keep:
        return None
    sub_mask = None if view_mask is None else view_mask[:, keep]
    return predicted_classes(class_source[:, keep, :], sub_mask)


def _dependence_core(
    signal,
    class_source,
    view_mask,
    classes,
    method: str,
    conditioning: str,
    seed: int,
    null_permutations: int = 0,
) -> np.ndarray:
    """Shared machinery: correlate `signal`, stratify by `class_source`.

    The two are separated because the tensor worth correlating and the tensor
    that can name a class are not always the same one -- encoder features
    carry the dependence but have no class axis to argmax over.
    """
    if method not in _VALID_METHODS:
        raise ValueError(f"method must be one of {_VALID_METHODS}, got {method!r}")
    if conditioning not in _VALID_CONDITIONING:
        raise ValueError(
            f"conditioning must be one of {_VALID_CONDITIONING}, got {conditioning!r}"
        )

    signal = _to_numpy(signal)
    if signal.ndim != 3:
        raise ValueError(f"signal must be [B, V, F], got {signal.shape}")

    class_source = _to_numpy(class_source)
    if class_source.shape[:2] != signal.shape[:2]:
        raise ValueError(
            f"class_source [B, V] {class_source.shape[:2]} must match signal "
            f"{signal.shape[:2]}"
        )

    n_views = signal.shape[1]
    mask = None if view_mask is None else _to_numpy(view_mask).astype(bool)

    explicit = classes is not None
    global_classes = (
        np.asarray(_to_numpy(classes), dtype=np.int64)
        if explicit
        else predicted_classes(class_source, mask)
    )

    rng = np.random.default_rng(seed)
    matrix = np.eye(n_views, dtype=np.float64)

    for i in range(n_views):
        for j in range(i + 1, n_views):
            pair_classes = global_classes
            if not explicit and conditioning == "pairwise_holdout":
                held_out = _holdout_classes(class_source, mask, {i, j})
                if held_out is not None:
                    pair_classes = held_out

            pair_mask = None if mask is None else mask[:, [i, j]]
            residuals, valid = class_conditional_residuals(
                signal[:, [i, j], :], pair_classes, pair_mask
            )

            both = valid[:, 0] & valid[:, 1]
            if int(both.sum()) < _MIN_PAIR_SAMPLES:
                matrix[i, j] = matrix[j, i] = np.nan
                continue

            residual_a = residuals[both, 0, :]
            residual_b = residuals[both, 1, :]
            strata = pair_classes[both]

            # dCor builds an [n, n] matrix per side; subsample once here so the
            # observed statistic and its null are computed on identical rows.
            if method == "dcor" and residual_a.shape[0] > _DCOR_MAX_SAMPLES:
                picked = rng.choice(residual_a.shape[0], _DCOR_MAX_SAMPLES, replace=False)
                residual_a = residual_a[picked]
                residual_b = residual_b[picked]
                strata = strata[picked]

            value = _pair_statistic(method, residual_a, residual_b)

            if null_permutations > 0:
                null = _permutation_null(
                    method, residual_a, residual_b, strata, rng, null_permutations
                )
                value = _chance_correct(value, null)

            matrix[i, j] = value
            matrix[j, i] = value

    return matrix


def dependence_matrix(
    evidence,
    view_mask=None,
    classes=None,
    method: str = "pearson",
    conditioning: str = "global",
    seed: int = 0,
    null_permutations: int = 0,
) -> np.ndarray:
    """Symmetric [V, V] cross-view dependence measured on EVIDENCE.

    evidence:          [B, V, K] per-view evidence
    view_mask:         [B, V] boolean, True = view present
    classes:           [B] explicit conditioning strata; overrides `conditioning`
    method:            "pearson", "cca", or "dcor"
    conditioning:      "global" or "pairwise_holdout" (see module docstring)
    null_permutations: >0 applies the chance correction (see module docstring)

    A pair of views with fewer than 8 jointly-available samples yields NaN
    rather than a number. Refusing to estimate is not the same as estimating
    zero: zero would claim the pair is independent and quietly license full
    confidence. Callers decide what to do with an unmeasurable pair.
    """
    return _dependence_core(
        evidence, evidence, view_mask, classes, method, conditioning, seed,
        null_permutations,
    )


def feature_dependence_matrix(
    features,
    evidence,
    view_mask=None,
    classes=None,
    method: str = "pearson",
    conditioning: str = "global",
    seed: int = 0,
    null_permutations: int = 0,
) -> np.ndarray:
    """Cross-view dependence measured on ENCODER FEATURES.

    features:  [B, V, feature_dim] MultiViewEncoder output, before the
               evidence head
    evidence:  [B, V, K] used ONLY to name the conditioning class, never
               correlated

    WHY MEASURE BEFORE THE EVIDENCE HEAD
    ------------------------------------
    The evidence head is a K-dimensional bottleneck trained for
    classification. Its job is to keep class-discriminative variation and
    discard the rest -- but the shared nuisance factor that makes two views
    redundant is, by construction, exactly the non-class-discriminative part.
    The head is therefore trained to throw away the very signal ENIV needs,
    and softplus then rectifies whatever survives.

    Two further effects compound on evidence and not on features: K is
    typically far smaller than feature_dim (3 vs 32 here), so the per-
    dimension correlations being averaged are fewer and noisier; and the
    model's own classification error acts as per-view noise that is
    uncorrelated across views, diluting measured dependence by an amount that
    tracks accuracy -- which itself varies across a rho sweep.

    Features are not a free lunch: they are higher-dimensional, so the
    estimate is more expensive (markedly so for dCor) and carries more
    variance per sample. Pair this with `null_permutations > 0` -- the
    upward bias that correction removes scales with feature_dim / n_samples,
    so it is worst exactly here.
    """
    return _dependence_core(
        features, evidence, view_mask, classes, method, conditioning, seed,
        null_permutations,
    )


def available_views(view_mask=None, n_views: int | None = None) -> np.ndarray:
    """Boolean [V] -- which views are present for at least one sample."""
    if view_mask is None:
        if n_views is None:
            raise ValueError("provide view_mask or n_views")
        return np.ones(n_views, dtype=bool)
    return _to_numpy(view_mask).astype(bool).any(axis=0)
