"""ENIV for semantic views: turning the Part 21 matrix into a discount.

A STATISTICAL ESTIMATOR per docs/CONTRACT.md section 3. No parameters, no
gradient, nothing learnable -- this is a measurement taken of the angles, not a
part of any model.

Part 22's stated goal is to EXTEND V1's ENIV to the semantic setting, so the
aggregation is imported from ``prismflow.eniv.eniv`` rather than rewritten here.
V1 is frozen and read-only; one instrument, used twice, cannot drift from itself.


THE BRIEF'S FORMULA IS THE AGGREGATION V1 MEASURED AND REPLACED
---------------------------------------------------------------
The brief specifies ``ENIV = n**2 / sum(D)``. Expand the denominator: for a
unit-diagonal matrix, ``sum(D) = n + n(n-1)*rho_bar``, so

    n**2 / sum(D)  ==  n / (1 + (n-1) * rho_bar)

which is the DESIGN EFFECT, agreeing to 8.9e-16 over 2000 random symmetric
matrices. That is ``design_effect_n_eff`` in V1, and V1's module docstring
records why it stopped being the default after Part 06: it is exact only under
EXCHANGEABILITY, where every pair of views is equally correlated. Averaging
every pair into one rho_bar spreads a redundant clique's correlation across
pairs that are not redundant, so views nobody duplicated get discounted anyway.

The V2 setting breaks exchangeability by construction. Five angles over
different connectors are not equally correlated: Part 19's fallback chain can
quietly point two angles at one source while the other three stay independent,
and Part 21's citation estimator exists precisely to catch that. A clique beside
independent angles is the structure the design effect handles worst.

Measured here on V1's Part 06 structure -- 4 independent angles plus k exact
duplicates of angle 0, where the true effective count is 4 at every k:

    k    n    design effect    eigenvalue
    0    4           4.0000        4.0000
    1    5           3.5714        4.0000
    2    6           3.0000        4.0000
    3    7           2.5789        4.0000
    4    8           2.2857        4.0000

The design effect loses 1.7 witnesses that were never there to lose. The
eigenvalue form is exact.

So ``method="eigen"`` is the default and is what the gate is judged on.
``method="design_effect"`` computes the brief's formula exactly and is kept
because the experiment reports both: a claim that one aggregation is wrong is
worth more with the other one beside it than as an assertion.

WHAT THE EIGENVALUE FORM DOES NOT FIX
-------------------------------------
``sum_i min(lambda_i, 1)`` holds the effective COUNT at 4 in the table above,
but ``discount = ENIV / n`` still falls (1.00 -> 0.50 over k = 0..4) because n
grows with each duplicate. V1 found the same thing one layer down and replaced
the single scalar with per-view factors; ``per_angle_discount`` below is that
mechanism, and it is the one Part 23 should fuse with. The scalar is implemented
as specified because the brief's gate is defined on it, not because it is the
better instrument.


NO DEFAULT FOR n
----------------
The brief writes ``compute_discount_factor(eniv, n: int = 5)``. Five is the
number of angles PrismFlow plans for, not the number the report describes:
Part 21 EXCLUDES angles that produced no claims, so ``report.n_angles`` is the
count that survived, and on a live run today four of five angles are empty. A
default of 5 silently divides a 2-angle ENIV by 5. ``n`` is therefore required,
and ``eniv_report`` reads it off the report so the caller cannot mismatch them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

# V1, frozen and read-only. Imported rather than reimplemented -- see above.
from prismflow.eniv.eniv import (
    design_effect_n_eff,
    eigen_n_eff,
    mean_off_diagonal,
    soft_cluster_alpha,
)
from prismflow.v2.dependence.models import DependenceReport

#: Aggregations from a dependence matrix to an effective view count.
ENIV_METHODS = ("eigen", "design_effect")

#: The brief's floor on the discount. An effective view count can fall below
#: n/10, but the discount stops there: driving evidence to zero on the strength
#: of an estimate is a stronger statement than this instrument supports (see
#: DependenceReport's note that it is ordinal, not absolute).
DISCOUNT_FLOOR = 0.1


@dataclass(frozen=True)
class SemanticENIVResult:
    """An ENIV measurement and everything needed to report it honestly.

    nominal_angles    n -- angles present in the matrix, AFTER Part 21's
                      exclusion of angles that produced no claims
    effective_angles  ENIV, clamped to [1, n]
    discount_factor   max(DISCOUNT_FLOOR, ENIV / n), capped at 1.0
    mean_dependence   rho_bar over distinct pairs, as a diagnostic
    method            which aggregation produced effective_angles
    excluded_angles   named here so a reader can see what n does not count
    """

    nominal_angles: int
    effective_angles: float
    discount_factor: float
    mean_dependence: float
    method: str
    excluded_angles: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, object]:
        return {
            "nominal_angles": self.nominal_angles,
            "effective_angles": self.effective_angles,
            "discount_factor": self.discount_factor,
            "mean_dependence": self.mean_dependence,
            "method": self.method,
            "excluded_angles": list(self.excluded_angles),
        }


def compute_semantic_eniv(
    dependence_report: DependenceReport, method: str = "eigen"
) -> float:
    """Effective number of independent angles from a semantic dependence matrix.

    Args:
        dependence_report: from ``aggregate_dependence``. Only the angles it
            kept are counted; excluded angles are absent from the matrix.
        method: ``"eigen"`` (default, V1's validated aggregation) or
            ``"design_effect"`` (the brief's ``n**2 / sum(D)``).

    Returns:
        ENIV, clamped to [1, n]. Zero angles returns 0.0 -- no angle spoke, so
        there is no view for the count to be about.
    """
    if method not in ENIV_METHODS:
        raise ValueError(f"method must be one of {ENIV_METHODS}, got {method!r}")

    n = dependence_report.n_angles
    if n == 0:
        return 0.0

    matrix = np.asarray(dependence_report.dependence_matrix, dtype=np.float64)

    if method == "eigen":
        effective = eigen_n_eff(matrix)
    else:
        # Algebraically n**2 / sum(D) for a unit-diagonal matrix; V1's form is
        # used so the two aggregations cannot diverge in their edge handling.
        effective = design_effect_n_eff(matrix)

    return float(np.clip(effective, 1.0, float(n)))


def compute_discount_factor(eniv: float, n: int) -> float:
    """Confidence discount ``max(0.1, ENIV / n)``, capped at 1.0.

    ``n`` is required: see the module docstring on why the brief's default of 5
    is unsafe once Part 21 starts excluding silent angles.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        # No angle spoke. Nothing to be confident from; nothing to discount.
        return 0.0
    if not np.isfinite(eniv):
        raise ValueError(f"eniv must be finite, got {eniv}")
    return float(min(1.0, max(DISCOUNT_FLOOR, eniv / n)))


def apply_eniv_discount(claim_confidence: float, discount_factor: float) -> float:
    """Scale a claim's confidence by the discount.

    Both arguments are range-checked. The brief multiplies unconditionally, so a
    confidence of 1.4 or a discount of 2.0 would pass straight through and land
    in a reported number as a probability above one.
    """
    if not 0.0 <= claim_confidence <= 1.0:
        raise ValueError(f"claim_confidence must be in [0, 1], got {claim_confidence}")
    if not 0.0 <= discount_factor <= 1.0:
        raise ValueError(f"discount_factor must be in [0, 1], got {discount_factor}")
    return float(claim_confidence * discount_factor)


def per_angle_discount(dependence_report: DependenceReport) -> Dict[str, float]:
    """Per-angle discount alpha_i = 1 / sum_j clip(D_ij, 0, 1), keyed by angle.

    V1's ``soft_cluster_alpha``: one over the angle's soft cluster size, i.e.
    how many angles -- itself included -- it is redundant with. A cluster of g
    duplicates gets 1/g each, summing to one witness, while an angle nobody
    duplicated keeps 1.0.

    NOT what the Part 22 gate is defined on. It is provided because the single
    scalar ``ENIV / n`` is the mechanism V1 measured and replaced (fused
    confidence fell 0.975 -> 0.910 over 0..4 copies with ENIV held exactly at
    4), and Part 23 fuses per angle, so it will need a per-angle factor.
    """
    n = dependence_report.n_angles
    if n == 0:
        return {}
    alphas = soft_cluster_alpha(
        np.asarray(dependence_report.dependence_matrix, dtype=np.float64)
    )
    return {
        name: float(alpha)
        for name, alpha in zip(dependence_report.angle_names, alphas)
    }


def eniv_report(
    dependence_report: DependenceReport, method: str = "eigen"
) -> SemanticENIVResult:
    """The whole measurement in one object, for logging and for results JSON."""
    n = dependence_report.n_angles
    eniv = compute_semantic_eniv(dependence_report, method=method)
    rho_bar = (
        mean_off_diagonal(
            np.asarray(dependence_report.dependence_matrix, dtype=np.float64)
        )
        if n
        else 0.0
    )
    return SemanticENIVResult(
        nominal_angles=n,
        effective_angles=eniv,
        discount_factor=compute_discount_factor(eniv, n),
        mean_dependence=float(rho_bar),
        method=method,
        excluded_angles=tuple(dependence_report.excluded_angles),
    )


__all__ = [
    "ENIV_METHODS",
    "DISCOUNT_FLOOR",
    "SemanticENIVResult",
    "compute_semantic_eniv",
    "compute_discount_factor",
    "apply_eniv_discount",
    "per_angle_discount",
    "eniv_report",
]
