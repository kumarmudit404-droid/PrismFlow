"""Fusion: spend the discount, name the disagreements, state a confidence.

Part 23. Takes Part 20's ClaimSets, Part 21's DependenceReport and Part 22's
ENIV, and produces one FusedRecommendation with a full audit trail.

Three things to know before reading a number out of this module:

- The conflict detector compares similarity the other way round from the brief.
  Low cosine similarity means two claims are about DIFFERENT things; a
  contradiction is two claims about the SAME thing that cannot both hold.
  Measured on the project's own encoder, the brief's ``sim < 0.3`` rule flagged
  0 of 5 planted contradictions and 5 of 5 unrelated pairs. See
  ``conflicts.py``.
- Similarity only nominates candidates. Whether a near pair agrees or disagrees
  is adjudicated -- by Claude when credentials exist, otherwise by a
  deliberately conservative lexical fallback that reports what it could not
  decide instead of calling it agreement.
- The discount is per angle by default, not one scalar, because V1 measured the
  scalar to penalise angles nothing duplicated.

Downstream parts import from here, not from the submodules.
"""

from .conflicts import (
    ADJUDICATION_SYSTEM,
    MAX_CANDIDATES,
    PARAPHRASE_CEILING,
    SIMILARITY_FLOOR,
    ClaudeAdjudicator,
    LexicalAdjudicator,
    conflict_candidates,
)
from .fusion import PrismFusion
from .models import (
    MODEL_PRICING,
    VERDICTS,
    ClaimConflict,
    FusedClaim,
    FusedRecommendation,
    estimate_cost_usd,
)

__all__ = [
    # orchestration
    "PrismFusion",
    # conflicts
    "conflict_candidates",
    "LexicalAdjudicator",
    "ClaudeAdjudicator",
    "SIMILARITY_FLOOR",
    "PARAPHRASE_CEILING",
    "MAX_CANDIDATES",
    "ADJUDICATION_SYSTEM",
    # data
    "FusedRecommendation",
    "FusedClaim",
    "ClaimConflict",
    "VERDICTS",
    "MODEL_PRICING",
    "estimate_cost_usd",
]
