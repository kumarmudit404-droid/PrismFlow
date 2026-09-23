"""Combining the three estimators into the matrix Part 22 discounts against.

An ENGINEERING component per docs/CONTRACT.md section 3.

EMPTY ANGLES ARE EXCLUDED, NOT SCORED
-------------------------------------
An angle that produced no claims has no dependence with anything -- the quantity
is undefined, not zero. The brief has no exclusion and its citation estimator
returns 1.0 for two angles that cite nothing, so five silent angles would come
back mutually identical.

Scoring them 0 instead is worse, and worse in the dangerous direction. Zero
dependence reads as "a fully independent view", so Part 22 would count a silent
angle toward the effective number of views, discount less, and report higher
confidence -- which is precisely the failure this project exists to prevent.
This matters concretely rather than hypothetically: Part 19 ships four of five
angles without connectors, so on a live run today four angles are empty.

So they are dropped from the matrix and named in ``excluded_angles``. The report
then describes only the angles that actually said something, and Part 22 cannot
inflate its view count with silence.

WEIGHTS ARE VALIDATED
---------------------
The brief neither checks that the weights sum to 1 nor that their keys are
estimator names. Weights summing above 1 push every off-diagonal to the [0, 1]
clip, so the matrix saturates and every pair of angles looks maximally
dependent; a misspelled key silently drops an estimator's contribution to zero
while the other weights keep their magnitudes. Both produce a plausible-looking
matrix, which is the problem.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from prismflow.v2.reasoners.models import Claim, ClaimSet

from .estimators import (
    CitationEstimator,
    EmbeddingEstimator,
    LexicalEstimator,
)
from .models import DependenceReport

logger = logging.getLogger("prismflow.v2.dependence")

#: The brief's weights. Embedding carries most of it because it is the only
#: estimator that sees meaning; citation is weighted next because when it does
#: fire it is the most decisive signal (a shared source is shared information,
#: not a coincidence); lexical is a control and is weighted least.
DEFAULT_WEIGHTS: Dict[str, float] = {
    "embedding": 0.5,
    "citation": 0.3,
    "lexical": 0.2,
}

WEIGHT_TOLERANCE = 1e-6


def validate_weights(weights: Mapping[str, float]) -> Dict[str, float]:
    """Check weights are a probability-like mix over known estimators."""
    if not weights:
        raise ValueError("weights must not be empty")
    unknown = sorted(set(weights) - set(DEFAULT_WEIGHTS))
    if unknown:
        raise ValueError(
            f"unknown estimator weight(s) {unknown}; known estimators are "
            f"{sorted(DEFAULT_WEIGHTS)}. A misspelled key would silently "
            "contribute nothing while the others keep their weight."
        )
    missing = sorted(set(DEFAULT_WEIGHTS) - set(weights))
    if missing:
        raise ValueError(
            f"no weight given for {missing}; pass 0.0 explicitly to disable an "
            "estimator, so that disabling one is visible in the report"
        )
    negative = {k: v for k, v in weights.items() if v < 0}
    if negative:
        raise ValueError(f"weights must be non-negative, got {negative}")
    total = float(sum(weights.values()))
    if abs(total - 1.0) > WEIGHT_TOLERANCE:
        raise ValueError(
            f"weights must sum to 1.0, got {total:.6f}. Summing above 1 pushes "
            "every pair into the [0, 1] clip, so the matrix saturates and all "
            "angles look maximally dependent."
        )
    return {key: float(value) for key, value in weights.items()}


def aggregate_dependence(
    claimsets: Sequence[ClaimSet],
    weights: Optional[Mapping[str, float]] = None,
    *,
    estimators: Optional[Mapping[str, Any]] = None,
    include_empty: bool = False,
) -> DependenceReport:
    """Estimate pairwise dependence between the angles in ``claimsets``.

    Args:
        claimsets: one ClaimSet per angle.
        weights: mix over ``embedding``, ``citation`` and ``lexical``. Must name
            all three and sum to 1.
        estimators: pre-built estimator instances, keyed by name. The experiment
            passes a truncated embedding estimator this way, and tests pass one
            with a deterministic encoder so the suite needs no model download.
        include_empty: keep angles that produced no claims. Off by default --
            see the module docstring for why scoring silence is unsafe.

    Returns:
        A validated DependenceReport over the angles that were kept.
    """
    weights = validate_weights(weights if weights is not None else DEFAULT_WEIGHTS)

    names = [cs.angle_name for cs in claimsets]
    if len(set(names)) != len(names):
        raise ValueError(
            f"duplicate angle name(s) in {names}; two angles sharing a name "
            "cannot be distinguished in the matrix"
        )

    warnings: List[str] = []
    excluded: List[str] = []
    kept: List[ClaimSet] = []
    for claimset in claimsets:
        if claimset.claims or include_empty:
            kept.append(claimset)
        else:
            excluded.append(claimset.angle_name)

    if excluded:
        warnings.append(
            f"excluded {len(excluded)} angle(s) with no claims "
            f"({', '.join(excluded)}); an angle that said nothing has undefined "
            "dependence, and scoring it 0 would let Part 22 count it as an "
            "independent view"
        )
    failed = [cs.angle_name for cs in kept if not cs.succeeded]
    if failed:
        warnings.append(
            f"{', '.join(failed)} reported a reasoner error but still carried "
            "claims; treat their dependence with care"
        )

    claims_by_angle: Dict[str, List[Claim]] = {
        cs.angle_name: list(cs.claims) for cs in kept
    }
    angle_names = list(claims_by_angle)
    n = len(angle_names)

    built = dict(estimators) if estimators else {}
    built.setdefault("embedding", EmbeddingEstimator())
    built.setdefault("citation", CitationEstimator())
    built.setdefault("lexical", LexicalEstimator())

    per_estimator: Dict[str, np.ndarray] = {}
    for name, estimator in built.items():
        if name not in weights:
            continue
        matrix, estimator_warnings = estimator.estimate(claims_by_angle)
        per_estimator[name] = np.asarray(matrix, dtype=float)
        warnings.extend(f"{name}: {line}" for line in estimator_warnings)

    if n == 0:
        combined = np.zeros((0, 0), dtype=float)
    else:
        combined = np.zeros((n, n), dtype=float)
        for name, matrix in per_estimator.items():
            combined += weights[name] * matrix
        combined = np.clip(combined, 0.0, 1.0)
        combined = (combined + combined.T) / 2.0
        np.fill_diagonal(combined, 1.0)

    report = DependenceReport(
        angle_names=angle_names,
        n_angles=n,
        dependence_matrix=combined,
        per_estimator=per_estimator,
        warnings=warnings,
        weights=weights,
        excluded_angles=excluded,
    )
    logger.info("dependence: %s", report.summary())
    return report


# --- the gate: do the three estimators rank pairs the same way? ---------


def estimator_agreement(
    report: DependenceReport,
) -> Tuple[Optional[float], Dict[str, float], List[str]]:
    """Rank agreement between the estimators, over the same pairs.

    The Part 21 gate is "three estimators agree on ranking", and nothing in the
    brief computes it -- the specified aggregator averages the three and never
    compares them. Averaging estimators that disagree produces a confident
    number out of a contradiction, so the comparison is the point.

    Spearman rather than Pearson because the claim being tested is about ORDER:
    the estimators are on different scales (a citation Jaccard and an embedding
    cosine are not commensurable), so requiring their magnitudes to agree would
    be the wrong test. What must hold is that they rank the same pairs as most
    and least dependent.

    Returns:
        ``(mean_rho, per_pair_rho, notes)``. ``mean_rho`` is None when the
        comparison is undefined -- fewer than three angles gives fewer than
        three pairs, and a rank correlation over one or two points is not a
        measurement.
    """
    from scipy.stats import spearmanr

    notes: List[str] = []
    names = sorted(report.per_estimator)
    if len(names) < 2:
        return None, {}, ["fewer than two estimators; nothing to compare"]

    n_pairs = report.n_angles * (report.n_angles - 1) // 2
    if n_pairs < 3:
        return None, {}, [
            f"{report.n_angles} angles give {n_pairs} pair(s); a rank "
            "correlation needs at least 3 points to mean anything"
        ]

    i, j = np.triu_indices(report.n_angles, k=1)
    vectors = {name: report.per_estimator[name][i, j] for name in names}

    per_pair: Dict[str, float] = {}
    for a_index, a in enumerate(names):
        for b in names[a_index + 1 :]:
            left, right = vectors[a], vectors[b]
            if np.allclose(left, left[0]) or np.allclose(right, right[0]):
                # A constant vector has no ranking, so rho is undefined rather
                # than zero. This is the ordinary case for the citation
                # estimator, which is all-zeros whenever no two angles share a
                # source -- reporting that as "disagreement" would be wrong.
                notes.append(
                    f"{a} vs {b}: one estimator is constant across all pairs, "
                    "so rank correlation is undefined"
                )
                continue
            rho = float(spearmanr(left, right).statistic)
            if np.isnan(rho):
                notes.append(f"{a} vs {b}: rank correlation returned NaN")
                continue
            per_pair[f"{a}~{b}"] = rho

    if not per_pair:
        return None, {}, notes or ["no estimator pair yielded a defined rho"]
    mean_rho = float(np.mean(list(per_pair.values())))
    return mean_rho, per_pair, notes


__all__ = [
    "DEFAULT_WEIGHTS",
    "aggregate_dependence",
    "validate_weights",
    "estimator_agreement",
]
