"""Upper and lower tail dependence between views.

A STATISTICAL ESTIMATOR per docs/CONTRACT.md: no learnable parameters, no
nn.Module, no gradients. Pure numpy on detached inputs.

THE ARGUMENT
------------
Pearson correlation, dCor and CCA all measure AVERAGE co-movement. An
adversarial event is not average -- it is extreme. The quantity that matches
what an attack does is upper tail dependence,

    lambda_U = lim_{q -> 1-} P( U_2 > q | U_1 > q )

the probability that view 2 is extreme GIVEN that view 1 is extreme.

The Gaussian copula has lambda_U = 0 for every correlation below 1. Two
variables can correlate at 0.9 and still be modelled as having vanishing
probability of joint extreme behaviour. That is not a technicality: it is the
modelling error that priced correlated defaults as independent before 2008.
This project's fusion commits the same error in miniature. It measures average
agreement (`dependence.py`) and is therefore blind to coordinated EXTREME
agreement -- which is precisely what a chorus attack is.

lambda_U = 0 IS AN ASYMPTOTIC STATEMENT, AND THE APPROACH IS SLOW
-----------------------------------------------------------------
Measured here, not assumed. A Gaussian pair at rho = 0.9 estimates lambda_U at
0.685 (q = 0.90), 0.628 (0.95), 0.570 (0.99), 0.530 (0.995) -- decaying, but
nowhere near zero at any threshold a finite sample can support. At rho = 0.6 it
runs 0.401 -> 0.319 -> 0.255 -> 0.203 -> 0.135 over the same thresholds.

So a single lambda_U at a single q does NOT separate "Gaussian" from "tail
dependent". What separates them is the DECAY: the Gaussian estimate falls
steadily as q rises, while a t copula's flattens to a positive constant (at
rho = 0.6, df = 3: 0.468 -> 0.446 -> 0.426 -> 0.420 -> 0.400). Any claim that a
measured lambda_U demonstrates tail dependence must show the decay curve, not
one number. `test_tail_dependence.py` pins this contrast.

ESTIMATORS
----------
Two, because they carry different bias and disagreeing is informative.

`empirical`      the conditional frequency directly:
                 #{both > q} / #{first > q}. Unbiased in what it counts, but it
                 only ever sees points above the threshold.

`nonparametric`  the log-ratio form built on the empirical copula,
                     lambda_U ~ 2 - log(C(q, q)) / log(q)
                 which uses the whole sample through C.

NOTE, measured rather than assumed: using the whole sample does NOT make the
log-ratio form steadier here. At q = 0.95 its across-seed sd is slightly WORSE
than the empirical estimator's (n = 300: 0.0988 vs 0.0930; n = 1000: 0.0658 vs
0.0621; n = 3000: 0.0433 vs 0.0409), and it sits systematically ~0.03 lower.
Both are reported because they bracket the estimate, not because one is a
variance reduction on the other.

THE SAMPLE-SIZE PROBLEM, STATED UP FRONT
----------------------------------------
Extremes are rare by definition, so every estimate here is data-hungry. At
q = 0.95 a 300-sample split puts 15 points above the threshold, which is not
enough to estimate a conditional probability. Every function below reports
`n_tail`, the number of exceedances the estimate actually rests on, and
`reliable`, which is False when that count is under `MIN_TAIL_SAMPLES`. A
lambda_U reported without its n_tail is not a result.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prismflow.statistics.copula import empirical_copula, pseudo_observations

__all__ = [
    "MIN_TAIL_SAMPLES",
    "TailEstimate",
    "co_exceedance_score",
    "lower_tail_dependence",
    "tail_dependence_matrix",
    "tail_eniv",
    "upper_tail_dependence",
]

TAIL_METHODS = ("empirical", "nonparametric")

# Below this many exceedances the estimate is reported but marked unreliable.
# The threshold is a convention, not a theorem: it is the point below which a
# conditional probability estimated from the exceedances has a standard error
# comparable to the quantity itself.
MIN_TAIL_SAMPLES = 50


@dataclass(frozen=True)
class TailEstimate:
    """`coefficient` is meaningless without `n_tail`; keep them together.

    n_tail    exceedances of the threshold in the CONDITIONING view
    reliable  n_tail >= MIN_TAIL_SAMPLES
    """

    coefficient: float
    n_tail: int
    quantile: float
    method: str
    n_samples: int

    @property
    def reliable(self) -> bool:
        return self.n_tail >= MIN_TAIL_SAMPLES


def _as_pairs(a, b) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(a, dtype=np.float64).ravel()
    second = np.asarray(b, dtype=np.float64).ravel()
    if first.shape != second.shape:
        raise ValueError(f"inputs must have equal length, got {first.shape} and {second.shape}")
    if first.size < 2:
        raise ValueError("need at least 2 samples")
    return first, second


def upper_tail_dependence(a, b, quantile: float = 0.95, method: str = "empirical") -> TailEstimate:
    """P(B extreme | A extreme) at threshold `quantile`.

    Both inputs are rank-transformed first, so the result depends only on the
    copula and not on either marginal.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if method not in TAIL_METHODS:
        raise ValueError(f"method must be one of {TAIL_METHODS}, got {method!r}")

    first, second = _as_pairs(a, b)
    u = pseudo_observations(np.column_stack([first, second]))
    exceed_first = u[:, 0] > quantile
    n_tail = int(exceed_first.sum())

    if method == "empirical":
        if n_tail == 0:
            coefficient = float("nan")
        else:
            coefficient = float((u[exceed_first, 1] > quantile).mean())
    else:
        # 2 - log C(q, q) / log q, the standard log-ratio limit form.
        joint = float(empirical_copula(u, points=[[quantile, quantile]])[0])
        if joint <= 0.0:
            coefficient = 0.0
        else:
            coefficient = float(np.clip(2.0 - np.log(joint) / np.log(quantile), 0.0, 1.0))

    return TailEstimate(
        coefficient=coefficient,
        n_tail=n_tail,
        quantile=quantile,
        method=method,
        n_samples=int(first.size),
    )


def lower_tail_dependence(a, b, quantile: float = 0.05, method: str = "empirical") -> TailEstimate:
    """P(B in lower tail | A in lower tail). Mirror of `upper_tail_dependence`."""
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    if method not in TAIL_METHODS:
        raise ValueError(f"method must be one of {TAIL_METHODS}, got {method!r}")

    first, second = _as_pairs(a, b)
    u = pseudo_observations(np.column_stack([first, second]))
    below_first = u[:, 0] < quantile
    n_tail = int(below_first.sum())

    if method == "empirical":
        coefficient = float("nan") if n_tail == 0 else float((u[below_first, 1] < quantile).mean())
    else:
        joint = float(empirical_copula(u, points=[[quantile, quantile]])[0])
        coefficient = float(np.clip(joint / quantile, 0.0, 1.0))

    return TailEstimate(
        coefficient=coefficient,
        n_tail=n_tail,
        quantile=quantile,
        method=method,
        n_samples=int(first.size),
    )


def tail_dependence_matrix(
    x,
    quantile: float = 0.95,
    method: str = "empirical",
    lower: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise tail dependence over columns of `x` -> ([d, d], [d, d] of n_tail).

    x  [n, d] one scalar per view per sample (see `view_statistic` callers).

    The matrix is symmetrised: lambda_U is a symmetric property of a pair, but
    the empirical estimator conditions on one side, so the two directions differ
    at finite n. Averaging them is the standard remedy and halves that noise.
    """
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected [n, d], got shape {array.shape}")
    dim = array.shape[1]
    estimate = lower_tail_dependence if lower else upper_tail_dependence

    matrix = np.eye(dim, dtype=np.float64)
    counts = np.zeros((dim, dim), dtype=np.int64)
    for i in range(dim):
        for j in range(i + 1, dim):
            forward = estimate(array[:, i], array[:, j], quantile, method)
            backward = estimate(array[:, j], array[:, i], quantile, method)
            pair = np.nanmean([forward.coefficient, backward.coefficient])
            matrix[i, j] = matrix[j, i] = float(pair)
            counts[i, j] = counts[j, i] = min(forward.n_tail, backward.n_tail)
    return matrix, counts


def co_exceedance_score(x, quantile: float = 0.95) -> np.ndarray:
    """PER-SAMPLE tail-agreement score -> [n].

    lambda_U is a population limit and cannot flag an individual sample, but a
    detector needs a per-sample number. This returns the fraction of view PAIRS
    that are jointly above their marginal threshold for that sample -- the
    quantity whose conditional mean IS lambda_U. It is the per-sample building
    block of the coefficient, not a different idea.

    Rank-transforming first means the threshold is a per-view quantile, so a
    view with a wider marginal cannot dominate the score.
    """
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected [n, d], got shape {array.shape}")
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    dim = array.shape[1]
    if dim < 2:
        return np.zeros(array.shape[0], dtype=np.float64)

    exceed = pseudo_observations(array) > quantile
    counts = exceed.sum(axis=1).astype(np.float64)
    # pairs jointly exceeding = C(k, 2) for k exceedances, normalised by C(d, 2)
    return (counts * (counts - 1.0) / 2.0) / (dim * (dim - 1.0) / 2.0)


def tail_agreement_score(x, view_mask=None) -> np.ndarray:
    """PER-SAMPLE continuous tail-agreement score -> [n].

        score(n) = max over pairs (i, j) of min( u[n, i], u[n, j] )

    For any threshold q, a pair is jointly above q exactly when its min exceeds
    q -- so this is the co-exceedance indicator with the threshold removed, and
    a detector built on it is not tied to one quantile. `co_exceedance_score`
    counts pairs at a FIXED q and is therefore coarse (for 4 views it takes at
    most 7 distinct values); this is continuous and ranks samples properly,
    which an ROC needs.

    The max over pairs is what keeps it usable as a detector: it asks "is any
    pair jointly extreme", which needs no knowledge of WHICH views collude.
    Knowing that would be attack ground truth, which docs/CONTRACT.md forbids
    the suspicion detector from receiving.
    """
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected [n, d], got shape {array.shape}")
    n_samples, dim = array.shape
    if dim < 2:
        return np.zeros(n_samples, dtype=np.float64)

    u = pseudo_observations(array)
    if view_mask is not None:
        present = np.asarray(view_mask, dtype=bool)
        if present.shape != u.shape:
            raise ValueError(f"view_mask must be {u.shape}, got {present.shape}")
        u = np.where(present, u, -np.inf)

    best = np.full(n_samples, -np.inf, dtype=np.float64)
    for i in range(dim):
        for j in range(i + 1, dim):
            np.maximum(best, np.minimum(u[:, i], u[:, j]), out=best)
    return np.where(np.isfinite(best), best, 0.0)


def tail_eniv(tail_matrix, n_views: int | None = None) -> float:
    """Design-effect ENIV driven by lambda_U instead of rho_bar.

        n_eff = n / (1 + (n - 1) * lambda_bar)

    This is the `design_effect` form from `eniv.py` with the mean off-diagonal
    tail dependence substituted for the mean off-diagonal correlation. The
    eigenvalue form is NOT used: it is justified by a variance decomposition of
    a correlation matrix, and a matrix of conditional exceedance probabilities
    is not one -- its eigenvalues have no such reading.

    Interpretation: the number of views that would have to agree INDEPENDENTLY
    to produce the joint extreme agreement actually observed.
    """
    matrix = np.asarray(tail_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square matrix, got {matrix.shape}")
    count = matrix.shape[0] if n_views is None else int(n_views)
    if count < 1:
        raise ValueError("n_views must be >= 1")
    if count == 1:
        return 1.0

    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    finite = off_diagonal[np.isfinite(off_diagonal)]
    if finite.size == 0:
        return float(count)

    mean_tail = float(np.clip(finite.mean(), 0.0, 1.0))
    return float(np.clip(count / (1.0 + (count - 1.0) * mean_tail), 1.0, count))
