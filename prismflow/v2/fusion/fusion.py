"""``PrismFusion``: apply the discount, name the disagreements, aggregate.

An ENGINEERING component per docs/CONTRACT.md section 3. The judgement it
delegates (conflict adjudication) lives in ``conflicts.py``.

This is where the thesis is finally spent. Parts 19-21 gather evidence and
measure how much of it is the same evidence twice; Part 22 turns that into a
discount; this part multiplies by it and says what the system believes.


n IS THE NUMBER OF ANGLES THAT SPOKE
------------------------------------
The brief calls ``compute_discount_factor(eniv, n=5)``. Five is what PrismFlow
plans for, not what the report describes: Part 21 excludes angles that produced
no claims, and on a live run today four of five angles have no connector. With
two angles surviving, ``n=5`` divides their ENIV by five and reports a discount
of 0.2 where the measurement supports 0.5. ``n`` comes from
``dependence_report.n_angles`` here, and a claim-bearing angle missing from the
report is an error rather than a default -- see ``_alignment_error``.


THE DISCOUNT IS PER ANGLE
-------------------------
The brief applies one scalar to every claim. V1 measured that mechanism and
replaced it: with ENIV held at exactly 4, a single ``alpha = ENIV/n`` still
pulled fused confidence from 0.975 to 0.910 over 0..4 duplicates of one view,
because it penalises views nobody duplicated. Part 22 carries V1's per-angle
soft-cluster factor forward for exactly this call site, and it is the default
here. ``use_per_angle_discount=False`` restores the brief's scalar; both numbers
are recorded in the result either way, so the choice is visible rather than
buried.


AGGREGATION, AND WHAT THE BRIEF ASKED FOR THAT CANNOT BE BUILT
---------------------------------------------------------------
Goal 3 asks for a weighted mean with "weights: recency (newer angles weighted
higher), source diversity", and then specifies ``np.mean`` over every claim --
unweighted, with no recency and no diversity anywhere in the code.

Recency cannot be implemented as written: an angle has no age. Records have
dates, claims cite records, but a ClaimSet carries no timestamp and nothing
upstream propagates one, so "newer angles" names a quantity that does not exist
in the pipeline. Rather than invent it, this aggregates with the weights the
project actually measured: each claim is weighted by its angle's dependence
discount, so an angle that is one of two near-duplicates contributes about half
as much as an independent one. That IS the source-diversity weighting the goal
asks for, computed rather than asserted. Recency is not implemented and is
reported as not implemented.

A plain mean also has a failure the weighted form does not: an angle that
returns twelve claims outvotes one that returns two, so verbosity becomes
evidence. Weights are normalised per angle for the same reason.


THE ERROR FIELD SURVIVES
------------------------
The brief's discount loop rebuilds every ClaimSet and omits ``error``, so a
failed angle emerges marked successful. Nothing is rebuilt here; failures are
collected into ``failed_angles``, stated in the audit trail, and -- by default --
their claims are excluded from the aggregate, because an angle that reported an
API error did not produce a considered opinion.
"""

from __future__ import annotations

import inspect
import logging
import time
from typing import Callable, Dict, List, Optional, Sequence

from prismflow.v2.dependence.models import DependenceReport
from prismflow.v2.reasoners.models import Claim, ClaimSet
from prismflow.v2.statistics import (
    apply_eniv_discount,
    compute_discount_factor,
    compute_semantic_eniv,
    per_angle_discount,
)

from .conflicts import LexicalAdjudicator, conflict_candidates
from .models import ClaimConflict, FusedClaim, FusedRecommendation

logger = logging.getLogger("prismflow.v2.fusion")


class PrismFusion:
    """Fuse one query's angle claims into a discounted recommendation."""

    def __init__(
        self,
        *,
        adjudicator: Optional[Callable] = None,
        encoder: Optional[Callable] = None,
        use_per_angle_discount: bool = True,
        drop_failed_angles: bool = True,
        eniv_method: str = "eigen",
    ) -> None:
        """
        Args:
            adjudicator: decides each candidate pair. ``None`` uses
                ``LexicalAdjudicator``, the offline fallback -- pass
                ``ClaudeAdjudicator()`` for the real judgement. May be sync or
                async; both are awaited correctly.
            encoder: text -> vectors, for candidate generation. Defaults to
                Part 21's shared sentence-transformer.
            use_per_angle_discount: per-angle factors (default) or the brief's
                single scalar.
            drop_failed_angles: exclude claims from angles whose reasoner
                errored. See the module docstring.
            eniv_method: passed to Part 22. ``"eigen"`` is V1's validated form.
        """
        self.adjudicator = adjudicator if adjudicator is not None else LexicalAdjudicator()
        self.encoder = encoder
        self.use_per_angle_discount = use_per_angle_discount
        self.drop_failed_angles = drop_failed_angles
        self.eniv_method = eniv_method

    # -- the call ---------------------------------------------------------

    async def fuse(
        self,
        query: str,
        claimsets: Sequence[ClaimSet],
        dependence_report: DependenceReport,
        eniv: Optional[float] = None,
    ) -> FusedRecommendation:
        """Discount, adjudicate, aggregate.

        Args:
            query: the original query text.
            claimsets: one ClaimSet per angle, as Part 20 returned them.
            dependence_report: Part 21's report over the angles that spoke.
            eniv: Part 22's value. Recomputed from the report when omitted, so a
                caller cannot pass a number that disagrees with the matrix.
        """
        started = time.time()
        audit: List[str] = []

        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        if eniv is None:
            eniv = compute_semantic_eniv(
                dependence_report, method=self.eniv_method
            )
            audit.append(
                f"ENIV computed from the report ({self.eniv_method}): {eniv:.4f}"
            )
        else:
            eniv = float(eniv)
            audit.append(f"ENIV supplied by caller: {eniv:.4f}")

        n = dependence_report.n_angles
        scalar_discount = compute_discount_factor(eniv, n)
        alphas = per_angle_discount(dependence_report)
        audit.append(
            f"n={n} angle(s) in the dependence report, scalar discount "
            f"max(0.1, {eniv:.4f}/{n}) = {scalar_discount:.4f}"
        )
        if dependence_report.excluded_angles:
            audit.append(
                "excluded from the view count by Part 21 (no claims): "
                + ", ".join(dependence_report.excluded_angles)
            )

        # -- which angles contribute -------------------------------------

        contributing, failed, audit_lines = self._select(claimsets, dependence_report)
        audit.extend(audit_lines)

        # -- discount ------------------------------------------------------

        fused_claims: List[FusedClaim] = []
        for claimset in contributing:
            angle = claimset.angle_name
            factor = (
                alphas.get(angle, scalar_discount)
                if self.use_per_angle_discount
                else scalar_discount
            )
            for claim in claimset.claims:
                fused_claims.append(
                    FusedClaim(
                        angle_name=angle,
                        claim=claim,
                        raw_confidence=claim.confidence,
                        discount_applied=factor,
                        discounted_confidence=apply_eniv_discount(
                            claim.confidence, factor
                        ),
                    )
                )
        mode = "per-angle" if self.use_per_angle_discount else "single scalar"
        audit.append(
            f"applied the {mode} ENIV discount to {len(fused_claims)} claim(s) "
            f"from {len(contributing)} angle(s)"
        )
        if self.use_per_angle_discount and alphas:
            audit.append(
                "per-angle discount: "
                + ", ".join(f"{a}={v:.3f}" for a, v in sorted(alphas.items()))
            )

        # -- conflicts -----------------------------------------------------

        claims_by_angle: Dict[str, List[Claim]] = {
            cs.angle_name: list(cs.claims) for cs in contributing
        }
        candidates, candidate_warnings = conflict_candidates(
            claims_by_angle, encoder=self.encoder
        )
        audit.extend(candidate_warnings)
        audit.append(
            f"{len(candidates)} cross-angle pair(s) similar enough to adjudicate "
            "(a LOW similarity means unrelated, not conflicting -- see "
            "fusion/conflicts.py)"
        )

        conflicts = await self._adjudicate(candidates)
        contradictions = [c for c in conflicts if c.is_contradiction]
        unadjudicated = [c for c in conflicts if c.verdict == "unadjudicated"]
        audit.append(
            f"adjudicator {self._adjudicator_name()}: "
            f"{len(contradictions)} contradiction(s), "
            f"{len(unadjudicated)} pair(s) left unadjudicated"
        )

        # -- aggregate ------------------------------------------------------

        overall = self._aggregate(fused_claims, alphas)
        audit.append(f"overall confidence {overall:.4f}")
        audit.append(
            "recency weighting from goal 3 is NOT implemented: a ClaimSet "
            "carries no timestamp, so 'newer angles' names a quantity the "
            "pipeline does not have"
        )

        adjudicator_tokens_in = int(getattr(self.adjudicator, "tokens_input", 0) or 0)
        adjudicator_tokens_out = int(
            getattr(self.adjudicator, "tokens_output", 0) or 0
        )

        recommendation = FusedRecommendation(
            query_text=query,
            fused_claims=fused_claims,
            overall_confidence=overall,
            conflicts=conflicts,
            audit_trail=audit,
            eniv=float(eniv),
            discount_factor=scalar_discount,
            per_angle_discount=alphas,
            n_angles=n,
            excluded_angles=list(dependence_report.excluded_angles),
            failed_angles=failed,
            latency_seconds=time.time() - started,
            # Every angle's cost is counted, including angles whose claims were
            # dropped: a failed call is billed too, and a cost line that hides
            # the failures understates the bill.
            tokens_input=sum(cs.tokens_input for cs in claimsets),
            tokens_output=sum(cs.tokens_output for cs in claimsets),
            fusion_tokens_input=adjudicator_tokens_in,
            fusion_tokens_output=adjudicator_tokens_out,
            adjudicator=self._adjudicator_name(),
        )
        logger.info("fusion: %s", recommendation.summary())
        return recommendation

    # -- pieces -------------------------------------------------------------

    def _select(
        self, claimsets: Sequence[ClaimSet], report: DependenceReport
    ) -> tuple[List[ClaimSet], List[str], List[str]]:
        """Angles that contribute claims, angles that failed, and the audit."""
        audit: List[str] = []
        failed = [cs.angle_name for cs in claimsets if not cs.succeeded]
        if failed:
            audit.append(
                f"{len(failed)} angle(s) reported a reasoner error: "
                + ", ".join(
                    f"{cs.angle_name} ({cs.error})"
                    for cs in claimsets
                    if not cs.succeeded
                )
            )

        contributing: List[ClaimSet] = []
        for claimset in claimsets:
            if not claimset.claims:
                continue
            if self.drop_failed_angles and not claimset.succeeded:
                audit.append(
                    f"{claimset.angle_name}: {len(claimset.claims)} claim(s) "
                    "dropped -- the reasoner errored, so this is not a "
                    "considered opinion"
                )
                continue
            contributing.append(claimset)

        known = set(report.angle_names)
        orphans = [cs.angle_name for cs in contributing if cs.angle_name not in known]
        if orphans:
            raise ValueError(self._alignment_error(orphans, report))
        return contributing, failed, audit

    @staticmethod
    def _alignment_error(orphans: List[str], report: DependenceReport) -> str:
        return (
            f"angle(s) {orphans} carry claims but are absent from the dependence "
            f"report, which describes {report.angle_names}. Fusing them would "
            "mean discounting them by a factor measured for a different set of "
            "angles, or defaulting them to 1.0 -- no discount at all, which is "
            "the unsafe direction. Rebuild the report from the same claimsets."
        )

    async def _adjudicate(
        self, candidates: Sequence[ClaimConflict]
    ) -> List[ClaimConflict]:
        if not candidates:
            return []
        result = self.adjudicator(candidates)
        if inspect.isawaitable(result):
            result = await result
        return list(result)

    def _adjudicator_name(self) -> str:
        return getattr(
            self.adjudicator, "NAME", type(self.adjudicator).__name__
        )

    @staticmethod
    def _aggregate(
        fused_claims: Sequence[FusedClaim], alphas: Dict[str, float]
    ) -> float:
        """Angle-weighted mean of discounted confidences.

        Each ANGLE gets one unit of weight, split across its claims, and that
        unit is scaled by the angle's dependence discount. So a verbose angle
        does not outvote a terse one, and a redundant angle does not count twice.
        """
        if not fused_claims:
            return 0.0

        by_angle: Dict[str, List[FusedClaim]] = {}
        for fused in fused_claims:
            by_angle.setdefault(fused.angle_name, []).append(fused)

        total_weight = 0.0
        total = 0.0
        for angle, claims in by_angle.items():
            weight = alphas.get(angle, 1.0) / len(claims)
            for fused in claims:
                total += weight * fused.discounted_confidence
                total_weight += weight
        if total_weight <= 0.0:
            return 0.0
        return float(min(1.0, max(0.0, total / total_weight)))


__all__ = ["PrismFusion"]
