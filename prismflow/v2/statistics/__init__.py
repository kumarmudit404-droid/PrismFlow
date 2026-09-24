"""ENIV and the confidence discount for semantic views.

Part 22. Takes the DependenceReport from Part 21 and answers the question the
thesis rests on: how many independent witnesses are actually here? Everything
downstream multiplies by the answer.

Two things to know before reading a number out of this module:

- The default aggregation is V1's EIGENVALUE form, not the brief's
  ``n**2 / sum(D)``. Those two are not variants of one estimator: the brief's
  formula is algebraically the design effect, which V1's Part 06 measured to
  lose 1.7 witnesses on a structure whose true count never moved. Both are
  implemented and the Part 22 experiment reports both.
- The scalar ``ENIV / n`` discounts every angle equally, including angles
  nothing duplicated. ``per_angle_discount`` is the per-angle form V1 replaced
  it with, and is what Part 23 should fuse with.

Downstream parts import from here, not from the submodule.
"""

from .eniv import (
    DISCOUNT_FLOOR,
    ENIV_METHODS,
    SemanticENIVResult,
    apply_eniv_discount,
    compute_discount_factor,
    compute_semantic_eniv,
    eniv_report,
    per_angle_discount,
)

__all__ = [
    "compute_semantic_eniv",
    "compute_discount_factor",
    "apply_eniv_discount",
    "per_angle_discount",
    "eniv_report",
    "SemanticENIVResult",
    "ENIV_METHODS",
    "DISCOUNT_FLOOR",
]
