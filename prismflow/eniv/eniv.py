"""ENIV: Effective Number of Independent Views.

A STATISTICAL ESTIMATOR per docs/CONTRACT.md. There is deliberately no
nn.Module, no parameter, and no gradient anywhere in this file: ENIV is a
measurement taken of the model, not a part of the model. See the stop-gradient
note in `discount.py` for why that separation is load-bearing rather than
stylistic.

Two aggregations from a [V, V] dependence matrix are implemented.

EIGENVALUE ENIV (default)
-------------------------
    n_eff = sum_i min(lambda_i, 1)

over the eigenvalues of the dependence matrix. Each independent direction of
variation contributes at most one witness; a direction shared by several views
counts once, however many views load on it.

DESIGN EFFECT (kept for comparison, method="design_effect")
-----------------------------------------------------------
    n_eff = n / (1 + (n - 1) * rho_bar),  rho_bar = mean off-diagonal

Borrowed from survey sampling. It is exact only under EXCHANGEABILITY, where
every pair of views is equally correlated.

WHY THE DEFAULT CHANGED
-----------------------
The clone and Chorus experiments break exchangeability by construction: a
clique of near-duplicate views sits alongside views that are independent of
it. Averaging every pair into one rho_bar spreads the clique's high
correlation across every pair, so views that were never duplicated get
discounted too. In the Part 06 clone experiment (4 independent views plus k
copies of view 0), the design effect fell from ~3.5 to 2.4 as k went 0 to 4,
even though the true count stays 4 at every k. PrismFlow's confidence fell
with each copy instead of holding flat. Fed the measured clone and non-clone
dependence, the design-effect formula reproduced that decline almost exactly
(2.36 predicted vs 2.41 observed at k=4). So the error was in the
aggregation, not the dependence measurement.

On the same structure the eigenvalue form is exact for exact duplicates: an
uncorrelated block of g identical views has eigenvalues g and 0 (g-1 times),
which contributes exactly 1.

Known limits, measured rather than assumed:

  near-duplicates    a block correlated at c < 1 contributes 1 + (g-1)(1-c),
                     not 1. At the c ~ 0.92 measured on clones, each extra
                     copy adds ~0.08: small, and in the safe direction
                     relative to the bias it replaces.
  finite samples     sum(lambda) = V always, so capping at 1 can only remove
                     mass. Sampling noise spreads the eigenvalues and biases
                     n_eff DOWN at independence: at 300 samples, independent
                     views read 3.84 of 4 and 7.48 of 8.
  correlated base    copies of a view that already correlates with the others
                     are not fully absorbed. With base views at rho=0.5,
                     exact copies of one view move the population value from
                     2.50 to 3.00 over k=0..4. "One witness per block
                     regardless of size" holds for blocks independent of the
                     rest, not in general.

The two forms define different quantities, not better and worse estimates of
one. For equicorrelated views the design effect is n/(1+(n-1)rho) and the
eigenvalue form is n-(n-1)rho (`analytic_n_eff` vs `analytic_n_eff_eigen` in
prismflow.data.synthetic). Validate each against its own ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ENIV_METHODS = ("eigen", "design_effect")


@dataclass(frozen=True)
class ENIVResult:
    """
    nominal_views     n -- the count of AVAILABLE views
    effective_views   n_eff, clamped to [1, nominal_views]
    efficiency_ratio  n_eff / n, the Shafer discount factor (0, 1]
    mean_dependence   rho_bar, mean off-diagonal dependence over available views
                      (reported by both methods as a diagnostic)
    """

    nominal_views: int
    effective_views: float
    efficiency_ratio: float
    mean_dependence: float


def mean_off_diagonal(dependence: np.ndarray) -> float:
    """rho_bar over the upper triangle, ignoring unmeasurable (NaN) pairs.

    Returns 0.0 when nothing is measurable -- with one view, or none, there is
    no pair to be redundant with.
    """
    dependence = np.asarray(dependence, dtype=np.float64)
    n_views = dependence.shape[0]
    if n_views < 2:
        return 0.0

    upper = dependence[np.triu_indices(n_views, k=1)]
    measurable = upper[~np.isnan(upper)]
    if measurable.size == 0:
        return 0.0
    return float(measurable.mean())


def design_effect_n_eff(dependence: np.ndarray) -> float:
    """n / (1 + (n-1) * rho_bar). Exact only when all pairs are equally correlated."""
    n_views = dependence.shape[0]
    rho_bar = mean_off_diagonal(dependence)
    denominator = 1.0 + (n_views - 1) * rho_bar
    if denominator <= 1e-12:
        # rho_bar <= -1/(n-1): pathological anti-correlation. The design
        # effect is undefined there; cap at the ceiling rather than divide.
        return float(n_views)
    return n_views / denominator


def eigen_n_eff(dependence: np.ndarray) -> float:
    """sum_i min(lambda_i, 1) over the eigenvalues of the dependence matrix.

    Before the eigendecomposition, the matrix is repaired in two ways.

    Unmeasurable (NaN) pairs are imputed with the mean of the measurable
    off-diagonals. The eigendecomposition cannot skip an entry the way the
    design effect's average can, and imputing 0 would assert independence
    that was never measured.

    Negative eigenvalues are clipped to 0. A pairwise dependence matrix built
    from separate estimates is not guaranteed positive semi-definite, and a
    negative eigenvalue there is estimation error, not negative information.
    Letting it count would subtract witnesses that were never shown to be
    missing.
    """
    matrix = np.array(dependence, dtype=np.float64)
    n_views = matrix.shape[0]
    if n_views == 1:
        return 1.0

    missing = np.isnan(matrix)
    if missing.any():
        matrix[missing] = mean_off_diagonal(dependence)
    np.fill_diagonal(matrix, 1.0)
    matrix = 0.5 * (matrix + matrix.T)

    eigenvalues = np.linalg.eigvalsh(matrix)
    return float(np.clip(eigenvalues, 0.0, 1.0).sum())


def compute_eniv(dependence, available=None, method: str = "eigen") -> ENIVResult:
    """Effective number of independent views from a [V, V] dependence matrix.

    dependence: [V, V] symmetric, unit diagonal (from `dependence_matrix`)
    available:  optional [V] boolean mask of which views are present; ENIV is
                computed over the available sub-matrix only, because a missing
                view contributes no evidence and must not dilute the count of
                the views that do.
    method:     "eigen" (default) or "design_effect" -- see the module docstring

    `mean_dependence` is reported as measured (it can be slightly negative
    from sampling noise around true independence), while `effective_views` is
    clamped to [1, n] so the discount factor stays in (0, 1].
    """
    if method not in ENIV_METHODS:
        raise ValueError(f"method must be one of {ENIV_METHODS}, got {method!r}")

    dependence = np.asarray(dependence, dtype=np.float64)
    if dependence.ndim != 2 or dependence.shape[0] != dependence.shape[1]:
        raise ValueError(f"dependence must be square [V, V], got {dependence.shape}")

    if available is not None:
        available = np.asarray(available, dtype=bool)
        if available.shape != (dependence.shape[0],):
            raise ValueError(
                f"available must be [{dependence.shape[0]}], got {available.shape}"
            )
        dependence = dependence[np.ix_(available, available)]

    n_views = int(dependence.shape[0])
    if n_views == 0:
        return ENIVResult(0, 0.0, 0.0, 0.0)

    if method == "eigen":
        effective = eigen_n_eff(dependence)
    else:
        effective = design_effect_n_eff(dependence)

    effective = float(np.clip(effective, 1.0, float(n_views)))

    return ENIVResult(
        nominal_views=n_views,
        effective_views=effective,
        efficiency_ratio=effective / n_views,
        mean_dependence=mean_off_diagonal(dependence),
    )


def analytic_eniv(rho: float, n_views: int) -> float:
    """Design-effect ground truth at a known rho.

    Mirrors `prismflow.data.synthetic.analytic_n_eff`; kept here so the ENIV
    module can state its own ground truth without importing the data layer.
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if n_views < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    return n_views / (1.0 + (n_views - 1) * rho)


def analytic_eniv_eigen(rho: float, n_views: int) -> float:
    """Eigenvalue-ENIV ground truth at a known rho: n - (n-1) * rho.

    Mirrors `prismflow.data.synthetic.analytic_n_eff_eigen`.
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if n_views < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    return n_views - (n_views - 1) * rho
