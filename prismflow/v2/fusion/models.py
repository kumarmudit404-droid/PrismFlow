"""What fusion produces: discounted claims, named conflicts, and the bill.

An ENGINEERING component per docs/CONTRACT.md section 3.

THE BRIEF LOSES THE ATTRIBUTION IT THEN REPORTS
-----------------------------------------------
``FusedRecommendation.fused_claims: List[Claim]`` is every angle's claims
concatenated into one flat list, and the conflict strings that go beside it read
``"Claim 3 and 7 may conflict"``. Those indices point into a list whose origin
has already been discarded: nothing in the returned object says which angle
claim 3 came from, so the one thing a reader wants from a conflict -- who
disagreed with whom -- cannot be recovered. ``FusedClaim`` keeps the angle, and
``ClaimConflict`` names both sides.

THE BRIEF DROPS THE ERROR FIELD
-------------------------------
Its discount loop rebuilds each ClaimSet field by field and omits ``error``. Part
20 added that field on purpose: without it an angle whose API key is wrong and an
angle whose evidence genuinely supported nothing are indistinguishable, and
"treating a 401 as 'this angle found nothing' would let an outage look like a
considered opinion". Rebuilding the object without ``error`` puts the defect back
one part later -- every failed angle comes out of the discount loop marked
successful. Nothing here rebuilds a ClaimSet; failures are carried in
``failed_angles`` and stated in the audit trail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from prismflow.v2.reasoners.models import Claim

#: Published per-million-token rates, USD, for the models V2 can be pointed at.
#: Used only for the ESTIMATE in ``estimate_cost_usd`` -- this is arithmetic over
#: self-reported token counts, not a metered bill.
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

#: Verdicts an adjudicator may return for a candidate pair.
VERDICTS = ("contradiction", "agreement", "unrelated", "unadjudicated")


def estimate_cost_usd(
    tokens_input: int, tokens_output: int, model_name: str
) -> Optional[float]:
    """Cost estimate at published rates, or None for an unpriced model.

    The brief computes ``tokens_input * 0.0001 + tokens_output * 0.0003``, which
    is $100 per million input tokens and $300 per million output -- 50x and 30x
    the published Sonnet 5 rates of $2 and $10. A cost column overstated by a
    factor of fifty is worse than no cost column, because it will be quoted.

    Returns None rather than guessing when the model is not in the table: an
    invented rate is the defect being fixed.
    """
    rates = MODEL_PRICING.get(model_name)
    if rates is None:
        return None
    input_rate, output_rate = rates
    return (tokens_input / 1_000_000) * input_rate + (
        tokens_output / 1_000_000
    ) * output_rate


@dataclass(frozen=True)
class FusedClaim:
    """One claim, with the angle that made it and what the discount did to it."""

    angle_name: str
    claim: Claim
    raw_confidence: float
    discount_applied: float
    discounted_confidence: float

    @property
    def text(self) -> str:
        return self.claim.text

    @property
    def cited_ids(self) -> List[str]:
        return list(self.claim.cited_ids)

    def to_dict(self) -> Dict[str, object]:
        return {
            "angle_name": self.angle_name,
            "text": self.claim.text,
            "raw_confidence": self.raw_confidence,
            "discount_applied": self.discount_applied,
            "discounted_confidence": self.discounted_confidence,
            "cited_ids": list(self.claim.cited_ids),
        }


@dataclass(frozen=True)
class ClaimConflict:
    """Two claims from different angles that may disagree, and who said what.

    ``similarity`` is the embedding cosine that made the pair a CANDIDATE. It is
    not the evidence of conflict -- see ``conflicts.py`` on why a low similarity
    means "unrelated", not "contradictory". ``verdict`` is the adjudication.
    """

    angle_a: str
    angle_b: str
    text_a: str
    text_b: str
    similarity: float
    verdict: str
    rationale: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(
                f"verdict must be one of {VERDICTS}, got {self.verdict!r}"
            )

    @property
    def is_contradiction(self) -> bool:
        return self.verdict == "contradiction"

    def summary(self) -> str:
        return (
            f"{self.angle_a} vs {self.angle_b} [{self.verdict}, sim="
            f"{self.similarity:.2f}]: {self.text_a[:60]!r} / {self.text_b[:60]!r}"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "angle_a": self.angle_a,
            "angle_b": self.angle_b,
            "text_a": self.text_a,
            "text_b": self.text_b,
            "similarity": self.similarity,
            "verdict": self.verdict,
            "rationale": self.rationale,
        }


@dataclass
class FusedRecommendation:
    """Everything fusion concluded, and everything it needed to conclude it."""

    query_text: str
    fused_claims: List[FusedClaim]
    overall_confidence: float
    conflicts: List[ClaimConflict]
    audit_trail: List[str]
    eniv: float
    discount_factor: float
    per_angle_discount: Dict[str, float]
    n_angles: int
    excluded_angles: List[str] = field(default_factory=list)
    failed_angles: List[str] = field(default_factory=list)
    latency_seconds: float = 0.0
    tokens_input: int = 0
    tokens_output: int = 0
    fusion_tokens_input: int = 0
    fusion_tokens_output: int = 0
    adjudicator: str = "none"

    @property
    def contradictions(self) -> List[ClaimConflict]:
        return [c for c in self.conflicts if c.is_contradiction]

    @property
    def undiscounted_confidence(self) -> float:
        """The same aggregate before the discount, so the effect is visible."""
        if not self.fused_claims:
            return 0.0
        return sum(fc.raw_confidence for fc in self.fused_claims) / len(
            self.fused_claims
        )

    def cost_estimate_usd(self, model_name: str) -> Optional[float]:
        return estimate_cost_usd(self.tokens_input, self.tokens_output, model_name)

    def summary(self) -> str:
        return (
            f"{len(self.fused_claims)} claim(s) from {self.n_angles} angle(s), "
            f"ENIV {self.eniv:.2f}, confidence {self.overall_confidence:.3f} "
            f"(undiscounted {self.undiscounted_confidence:.3f}), "
            f"{len(self.contradictions)} contradiction(s) of "
            f"{len(self.conflicts)} candidate(s), {self.latency_seconds:.2f}s"
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "query_text": self.query_text,
            "fused_claims": [fc.to_dict() for fc in self.fused_claims],
            "overall_confidence": self.overall_confidence,
            "undiscounted_confidence": self.undiscounted_confidence,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "n_contradictions": len(self.contradictions),
            "audit_trail": list(self.audit_trail),
            "eniv": self.eniv,
            "discount_factor": self.discount_factor,
            "per_angle_discount": dict(self.per_angle_discount),
            "n_angles": self.n_angles,
            "excluded_angles": list(self.excluded_angles),
            "failed_angles": list(self.failed_angles),
            "latency_seconds": self.latency_seconds,
            "tokens_input": self.tokens_input,
            "tokens_output": self.tokens_output,
            "fusion_tokens_input": self.fusion_tokens_input,
            "fusion_tokens_output": self.fusion_tokens_output,
            "adjudicator": self.adjudicator,
        }


__all__ = [
    "MODEL_PRICING",
    "VERDICTS",
    "estimate_cost_usd",
    "FusedClaim",
    "ClaimConflict",
    "FusedRecommendation",
]
