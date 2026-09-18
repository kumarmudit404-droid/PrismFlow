"""Empirical copulas and parametric copula fits.

A STATISTICAL ESTIMATOR per docs/CONTRACT.md: no learnable parameters, no
nn.Module, no gradients. Pure numpy on detached inputs.

WHY A COPULA AT ALL
-------------------
A copula is the dependence structure of a joint distribution with the marginals
divided out. That separation is the point here: the project's existing measures
(Pearson, dCor, CCA) mix the shape of each view's marginal with how the views
move together, and two systems with identical average correlation can have
completely different joint EXTREME behaviour.

The Gaussian copula has upper tail dependence exactly zero for every
correlation below 1. Two views can correlate at 0.9 and still be modelled as
having vanishing probability of joint extreme agreement. The t copula, with
the same correlation matrix, has strictly positive tail dependence controlled
by its degrees of freedom. Fitting both to the same data and comparing the
log-likelihoods asks the data which of those two worlds it lives in.

See `tail_dependence.py` for the coefficient itself and why it is the quantity
an attack moves.

NO SCIPY
--------
The project's dependencies are numpy, torch, PyYAML, pytest and matplotlib, and
requirements.txt is not in this Part's scope. Everything needed here -- the
normal and Student-t quantile functions and the regularised incomplete beta --
is therefore implemented in this module against published algorithms, with
accuracy pinned in tests/unit/test_tail_dependence.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

__all__ = [
    "GaussianCopulaFit",
    "TCopulaFit",
    "empirical_copula",
    "fit_gaussian_copula",
    "fit_t_copula",
    "kendall_tau_matrix",
    "norm_ppf",
    "pseudo_observations",
    "t_cdf",
    "t_ppf",
]

# Acklam's rational approximation to the inverse standard normal CDF.
# Absolute relative error < 1.15e-9 over the whole open interval.
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)

_P_LOW = 0.02425


def norm_ppf(p):
    """Inverse standard normal CDF, vectorised (Acklam 2003)."""
    p = np.asarray(p, dtype=np.float64)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("norm_ppf requires 0 < p < 1; rank-transform first")

    out = np.empty_like(p)
    lower = p < _P_LOW
    upper = p > 1.0 - _P_LOW
    middle = ~(lower | upper)

    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(p[lower]))
        out[lower] = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / (
            (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        )
    if np.any(upper):
        q = np.sqrt(-2.0 * np.log(1.0 - p[upper]))
        out[upper] = -(
            ((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]
        ) / ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0)
    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q
        out[middle] = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / (
            ((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0
        )
    return out


def _betacf(a: float, b: float, x: float, max_iter: int = 300, eps: float = 3e-14) -> float:
    """Continued fraction for the incomplete beta (Lentz's method, NR 6.4)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(x, df: float):
    """Student-t CDF via the regularised incomplete beta."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.ravel()
    out = np.empty_like(flat)
    for i, value in enumerate(flat):
        half = 0.5 * _betai(0.5 * df, 0.5, df / (df + value * value))
        out[i] = half if value <= 0 else 1.0 - half
    return out.reshape(x.shape)


@lru_cache(maxsize=32)
def _t_quantile_grid(df: float) -> tuple[np.ndarray, np.ndarray]:
    """Cached (cdf, x) lookup for one df, used to invert the CDF by interpolation.

    `t_cdf` costs one incomplete-beta evaluation PER ELEMENT, so inverting it by
    per-element bisection would cost (elements x iterations) of them -- minutes
    per copula fit at n = 1200 with a df grid. The CDF depends only on df, so it
    is tabulated once per df and reused for every call.

    The grid is geometric in |x| rather than uniform: the quantile function is
    steepest in the tails, which is exactly where a copula fit reads it, and a
    uniform grid would resolve the bulk finely and the tails not at all.
    """
    positive = np.geomspace(1e-6, 1e4, 30_000)
    x_grid = np.concatenate([-positive[::-1], [0.0], positive])
    return t_cdf(x_grid, df), x_grid


def t_ppf(p, df: float):
    """Student-t quantile, by interpolating the cached CDF table for `df`.

    Monotone interpolation of a monotone function, so the result is monotone in
    p by construction. Accuracy is pinned against `t_cdf` in
    tests/unit/test_tail_dependence.py.
    """
    p = np.asarray(p, dtype=np.float64)
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("t_ppf requires 0 < p < 1")
    cdf_grid, x_grid = _t_quantile_grid(float(df))
    return np.interp(p, cdf_grid, x_grid).reshape(p.shape)


def pseudo_observations(x) -> np.ndarray:
    """Rank-transform each column to (0, 1): the empirical copula's input.

    u_ij = rank(x_ij) / (n + 1), ranks within column j. The (n + 1) divisor --
    rather than n -- is what keeps every value strictly inside the open unit
    interval, so the quantile functions used by the parametric fits never see 0
    or 1. Ties take their average rank.

    This is what makes the result a DEPENDENCE measurement: any strictly
    increasing change to a view's marginal leaves u unchanged, so nothing below
    can be moved by rescaling a single view.
    """
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected [n, d], got shape {array.shape}")
    n_samples = array.shape[0]
    if n_samples < 2:
        raise ValueError("need at least 2 samples to rank-transform")

    ranks = np.empty_like(array)
    for column in range(array.shape[1]):
        values = array[:, column]
        order = np.argsort(values, kind="mergesort")
        plain = np.empty(n_samples, dtype=np.float64)
        plain[order] = np.arange(1, n_samples + 1, dtype=np.float64)

        # Average ranks within ties, so a constant column maps to a constant
        # 0.5 rather than to an arbitrary permutation.
        unique, inverse = np.unique(values, return_inverse=True)
        if unique.size != n_samples:
            sums = np.bincount(inverse, weights=plain)
            counts = np.bincount(inverse)
            plain = (sums / counts)[inverse]
        ranks[:, column] = plain

    return ranks / (n_samples + 1.0)


def empirical_copula(u, points=None) -> np.ndarray:
    """C_n(w) = (1/n) * #{i : u_i <= w componentwise}.

    With `points=None` the copula is evaluated at the pseudo-observations
    themselves, which is what the diagonal-based tail estimators need.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"expected [n, d], got shape {u.shape}")
    grid = u if points is None else np.atleast_2d(np.asarray(points, dtype=np.float64))
    if grid.shape[1] != u.shape[1]:
        raise ValueError("points must have the same dimension as u")

    n_samples = u.shape[0]
    out = np.empty(grid.shape[0], dtype=np.float64)
    for i, point in enumerate(grid):
        out[i] = np.all(u <= point, axis=1).mean()
    return out if n_samples else out


def kendall_tau_matrix(x) -> np.ndarray:
    """Pairwise Kendall's tau-b, [d, d].

    Used to seed the copula correlation matrix through R = sin(pi/2 * tau),
    which is the standard rank-based estimator for elliptical copulas. It is
    invariant to the marginals and far more robust than a Pearson estimate on
    the transformed scores, which matters because the tail is where the fit is
    being judged.
    """
    array = np.asarray(x, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"expected [n, d], got shape {array.shape}")
    n_samples, dim = array.shape
    tau = np.eye(dim, dtype=np.float64)

    for i in range(dim):
        for j in range(i + 1, dim):
            a = array[:, i][:, None] - array[:, i][None, :]
            b = array[:, j][:, None] - array[:, j][None, :]
            upper = np.triu(np.ones((n_samples, n_samples), dtype=bool), k=1)
            sa, sb = np.sign(a[upper]), np.sign(b[upper])
            concordant = float(np.sum(sa * sb))
            n_a = float(np.sum(sa != 0))
            n_b = float(np.sum(sb != 0))
            denominator = math.sqrt(n_a * n_b) if n_a > 0 and n_b > 0 else 0.0
            value = concordant / denominator if denominator > 0 else 0.0
            tau[i, j] = tau[j, i] = float(np.clip(value, -1.0, 1.0))
    return tau


def _nearest_correlation(matrix: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to the nearest positive-definite correlation.

    sin(pi/2 * tau) applied entrywise need not be positive semi-definite at
    finite n, and the copula likelihoods require an invertible R.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    clipped = vectors @ np.diag(np.clip(values, floor, None)) @ vectors.T
    scale = np.sqrt(np.diag(clipped))
    scale[scale < 1e-12] = 1.0
    correlation = clipped / np.outer(scale, scale)
    np.fill_diagonal(correlation, 1.0)
    return correlation


@dataclass(frozen=True)
class GaussianCopulaFit:
    """Zero upper tail dependence BY CONSTRUCTION, whatever `correlation` says."""

    correlation: np.ndarray
    log_likelihood: float
    n_samples: int
    upper_tail_dependence: float = 0.0


@dataclass(frozen=True)
class TCopulaFit:
    """`df` controls the tail: smaller df, heavier joint extremes."""

    correlation: np.ndarray
    df: float
    log_likelihood: float
    n_samples: int
    upper_tail_dependence: float = float("nan")


def _gaussian_loglik(scores: np.ndarray, correlation: np.ndarray) -> float:
    dim = correlation.shape[0]
    sign, logdet = np.linalg.slogdet(correlation)
    if sign <= 0:
        return float("-inf")
    inverse = np.linalg.inv(correlation)
    quadratic = np.einsum("ij,jk,ik->i", scores, inverse - np.eye(dim), scores)
    return float(-0.5 * scores.shape[0] * logdet - 0.5 * quadratic.sum())


def fit_gaussian_copula(u) -> GaussianCopulaFit:
    """Gaussian copula by inversion of Kendall's tau.

    The log-likelihood is the COPULA density's, i.e. the joint density with the
    marginal contributions divided out, so it is directly comparable with
    `fit_t_copula` on the same pseudo-observations.
    """
    u = np.asarray(u, dtype=np.float64)
    scores = norm_ppf(u)
    correlation = _nearest_correlation(np.sin(0.5 * math.pi * kendall_tau_matrix(u)))
    return GaussianCopulaFit(
        correlation=correlation,
        log_likelihood=_gaussian_loglik(scores, correlation),
        n_samples=u.shape[0],
    )


def _t_loglik(u: np.ndarray, correlation: np.ndarray, df: float) -> float:
    n_samples, dim = u.shape
    scores = t_ppf(u, df)
    sign, logdet = np.linalg.slogdet(correlation)
    if sign <= 0:
        return float("-inf")
    inverse = np.linalg.inv(correlation)
    quadratic = np.einsum("ij,jk,ik->i", scores, inverse, scores)

    log_numerator = (
        math.lgamma(0.5 * (df + dim)) - math.lgamma(0.5 * df)
        - 0.5 * dim * math.log(df * math.pi) - 0.5 * logdet
        - 0.5 * (df + dim) * np.log1p(quadratic / df)
    )
    log_marginals = (
        math.lgamma(0.5 * (df + 1.0)) - math.lgamma(0.5 * df)
        - 0.5 * math.log(df * math.pi)
        - 0.5 * (df + 1.0) * np.log1p(scores * scores / df)
    ).sum(axis=1)
    return float(np.sum(log_numerator - log_marginals))


def fit_t_copula(u, df_grid=None) -> TCopulaFit:
    """t copula: R from Kendall's tau, df by profile likelihood over a grid.

    The grid is deliberate. df enters the likelihood only through quantile
    transforms that have to be inverted numerically here, so a derivative-based
    search would be both slower and less robust than evaluating a short grid,
    and df is a nuisance parameter whose exact value is not the finding.

    The fitted tail dependence is the closed form

        lambda_U = 2 * t_{df+1}( -sqrt((df + 1)(1 - r) / (1 + r)) )

    averaged over off-diagonal pairs -- strictly positive for every finite df,
    which is exactly what the Gaussian alternative cannot represent.
    """
    u = np.asarray(u, dtype=np.float64)
    grid = [2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 25.0, 50.0] if df_grid is None else list(df_grid)
    correlation = _nearest_correlation(np.sin(0.5 * math.pi * kendall_tau_matrix(u)))

    best_df, best_loglik = float("nan"), float("-inf")
    for df in grid:
        value = _t_loglik(u, correlation, float(df))
        if value > best_loglik:
            best_df, best_loglik = float(df), value

    dim = correlation.shape[0]
    tails = []
    for i in range(dim):
        for j in range(i + 1, dim):
            r = float(np.clip(correlation[i, j], -0.999999, 0.999999))
            argument = -math.sqrt((best_df + 1.0) * (1.0 - r) / (1.0 + r))
            tails.append(2.0 * float(t_cdf(np.array([argument]), best_df + 1.0)[0]))

    return TCopulaFit(
        correlation=correlation,
        df=best_df,
        log_likelihood=best_loglik,
        n_samples=u.shape[0],
        upper_tail_dependence=float(np.mean(tails)) if tails else float("nan"),
    )
