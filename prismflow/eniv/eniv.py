"""ENIV: Effective Number of Independent Views.

A STATISTICAL ESTIMATOR per docs/CONTRACT.md. There is deliberately no
nn.Module, no parameter, and no gradient anywhere in this file: ENIV is a
measurement taken of the model, not a part of the model. See the stop-gradient
note in `discount.py` for why that separation is load-bearing rather than
stylistic.

The design effect is borrowed from survey sampling, where it answers the same
question in a different costume: given n observations that are correlated
rather than independent, how many independent observations would have carried
the same information?

    n_eff = n / (1 + (n - 1) * rho_bar)

At rho_bar = 0 this returns n (every view its own witness); at rho_bar = 1 it
returns 1 (they are all the same witness wearing different hats).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ENIVResult:
    """
    nominal_views     n in the design effect -- the count of AVAILABLE views
    effective_views   n_eff, clamped to [1, nominal_views]
    efficiency_ratio  n_eff / n, the Shafer discount factor (0, 1]
    mean_dependence   rho_bar, mean off-diagonal dependence over available views
    """

    nominal_views: int
    effective_views: float
    efficiency_ratio: float
    mean_dependence: float


def mean_off_diagonal(dependence: np.ndarray) -> float:
    """rho_bar over the upper triangle, ignoring unmeasurable (NaN) pairs.

    Returns 0.0 when nothing is measurable -- with one view, or none, there is
    no pair to be redundant with, and the design effect collapses to n_eff = n
    either way.
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


def compute_eniv(dependence, available=None) -> ENIVResult:
    """Effective number of independent views from a [V, V] dependence matrix.

    dependence: [V, V] symmetric, unit diagonal (from `dependence_matrix`)
    available:  optional [V] boolean mask of which views are present; the
                design effect is computed over the available sub-matrix only,
                because a missing view contributes no evidence and must not
                dilute the count of the views that do.

    `mean_dependence` is reported as measured (it can be slightly negative
    from sampling noise around true independence), while `effective_views` is
    clamped to [1, n]. Reporting the raw rho_bar keeps the diagnostic honest;
    clamping n_eff keeps the discount factor in (0, 1].
    """
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

    rho_bar = mean_off_diagonal(dependence)

    denominator = 1.0 + (n_views - 1) * rho_bar
    if denominator <= 1e-12:
        # rho_bar <= -1/(n-1): pathological anti-correlation. The design
        # effect is undefined there; cap at the ceiling rather than divide.
        effective = float(n_views)
    else:
        effective = n_views / denominator

    effective = float(np.clip(effective, 1.0, float(n_views)))

    return ENIVResult(
        nominal_views=n_views,
        effective_views=effective,
        efficiency_ratio=effective / n_views,
        mean_dependence=rho_bar,
    )


def analytic_eniv(rho: float, n_views: int) -> float:
    """Design effect at a known rho -- the value `compute_eniv` must recover.

    Mirrors `prismflow.data.synthetic.analytic_n_eff`; kept here so the ENIV
    module can state its own ground truth without importing the data layer.
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if n_views < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    return n_views / (1.0 + (n_views - 1) * rho)
