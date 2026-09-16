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
    # The discount actually applied, one entry per view (see per_view_alpha).
    # efficiency_ratio is kept for reporting only; it is no longer what the
    # model multiplies by.
    per_view_alpha: tuple[float, ...] | None = None


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


# --- per-view discount factors -------------------------------------------
#
# WHY PER-VIEW
# A single alpha = ENIV / n applied to every view has the same flaw one layer
# downstream that the design effect had one layer up: structured information
# collapsed into one scalar and spread over units that are not structurally
# alike. Duplicating view 0 lowered the alpha of views 1-3, which had not
# changed. With ENIV held exactly at 4, fused confidence still fell from 0.975
# to 0.910 over k = 0..4 copies.
#
# WHY BLUE ROW SUMS, NOT 1/VIF
# 1/VIF_i = 1 / (R^-1)_ii is the share of view i NOT explained by the others.
# Two exact copies explain each other completely, so both go to ~0: the
# cluster is erased rather than counted once, and the first copy deletes a
# genuinely informative view. Without a ridge the same formula clips to 1 and
# discounts nothing. Measured on 4 views + k copies with the structure given
# exactly, fused confidence moved -0.026 over k = 0..4 (flat after k=1, but at
# the level of the three remaining views).
#
# The weights of the best linear unbiased combination of correlated
# estimates, w = R^-1 1, have the property actually wanted: an independent
# view keeps weight 1, and a cluster of g exact copies gets 1/g each, summing
# to exactly one witness. This holds with or without the ridge.
#
# WHY THE WEIGHTS SCALE EVIDENCE, NOT BELIEF
# The fusion rule is not linear in Shafer-discounted beliefs. g copies each
# belief-discounted by 1/g fused to 0.47-0.56 confidence against 0.73 for one
# undiscounted copy. Scaling evidence instead came closest (0.78-0.83), and
# on the 4 views + k copies test drifted +0.006 over k = 0..4, the smallest of
# every mechanism measured (current uniform: -0.066; per-view VIF: -0.026;
# BLUE on beliefs: -0.013). It is not exactly flat: the fusion rule does not
# add evidence exactly either. See prismflow/eniv/discount.py.


def _repaired_dependence(dependence: np.ndarray) -> np.ndarray:
    """Symmetric, unit-diagonal copy with NaN pairs imputed by the measured mean."""
    matrix = np.array(dependence, dtype=np.float64)
    missing = np.isnan(matrix)
    if missing.any():
        matrix[missing] = mean_off_diagonal(dependence)
    np.fill_diagonal(matrix, 1.0)
    return 0.5 * (matrix + matrix.T)


def _regularised_inverse(matrix: np.ndarray, ridge: float) -> np.ndarray:
    # Exact duplicates make the matrix singular along the duplicated
    # direction; the ridge keeps the inverse finite there.
    return np.linalg.pinv(matrix + ridge * np.eye(matrix.shape[0]))


_ALPHA_FLOOR = 1e-6


def per_view_alpha(dependence, available=None, ridge: float = 1e-6) -> np.ndarray:
    """Per-view discount alpha_i = sum_j ((R + ridge I)^-1)_ij, clipped to (0, 1].

    The row sums of the inverse dependence matrix: the best-linear-unbiased
    weights for combining correlated views. Independent view -> 1; each member
    of a cluster of g exact copies -> 1/g. See the block comment above for why
    this and not 1/VIF.

    For equicorrelated views the weight is 1 / (1 + (n-1) rho), and the weights
    sum to n / (1 + (n-1) rho), which is the design effect. The sum only equals
    the effective count for that structure. What matters here is that it
    splits weight WITHIN a redundant cluster and leaves unrelated views alone.

    Clipping: a row sum above 1 means the view is negatively related to the
    others. It is capped at 1 because the discount must never amplify
    evidence. A row sum at or below 0 is floored at 1e-6, i.e. the view's
    evidence is effectively ignored.

    available: optional [V] mask. Weights are computed over the available
               sub-matrix; unavailable views get 1.0, which is irrelevant
               because they carry zero evidence anyway.
    """
    dependence = np.asarray(dependence, dtype=np.float64)
    if dependence.ndim != 2 or dependence.shape[0] != dependence.shape[1]:
        raise ValueError(f"dependence must be square [V, V], got {dependence.shape}")
    n_views = dependence.shape[0]

    present = np.ones(n_views, dtype=bool) if available is None else np.asarray(available, dtype=bool)
    if present.shape != (n_views,):
        raise ValueError(f"available must be [{n_views}], got {present.shape}")

    alpha = np.ones(n_views, dtype=np.float64)
    if present.sum() == 0:
        return alpha

    sub = _repaired_dependence(dependence[np.ix_(present, present)])
    weights = _regularised_inverse(sub, ridge).sum(axis=1)
    alpha[present] = np.clip(weights, _ALPHA_FLOOR, 1.0)
    return alpha


def vif_alpha(dependence, ridge: float = 1e-6) -> np.ndarray:
    """alpha_i = 1 / VIF_i = 1 / ((R + ridge I)^-1)_ii, clipped to (0, 1].

    NOT used by the model. Kept as the documented counterexample: it sends
    every member of an exact-duplicate cluster, the original included, to ~0.
    For equicorrelated views VIF = (1 + (n-2) rho) / ((1 - rho)(1 + (n-1) rho)).
    """
    sub = _repaired_dependence(dependence)
    diagonal = np.diag(_regularised_inverse(sub, ridge))
    return np.clip(1.0 / diagonal, _ALPHA_FLOOR, 1.0)


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
