"""Semantic dependence between angles: the measurement the discount rests on.

Part 21. Takes the ClaimSets from Part 20 and reports, for each pair of angles,
how much they are saying the same thing -- by meaning (embeddings), by evidence
(shared citations) and by vocabulary (TF-IDF). Part 22 turns that matrix into
the ENIV discount.

Three things to know before reading a number out of this module:

- The third estimator is named ``lexical``, not ``mi``. The brief calls it
  mutual information; the specified code computes a TF-IDF cosine. Renamed
  rather than mislabelled, because a results table saying "mutual information"
  is a claim about methodology.
- Citation overlap is structurally 0 between angles that do not share a
  connector, because their id spaces do not intersect. It is a source-sharing
  detector, and it is the estimator that would catch Part 19's fallback chain
  quietly pointing two angles at one source.
- Angles that produced no claims are excluded from the matrix rather than
  scored, so a silent angle cannot be counted as an independent view.

Downstream parts import from here, not from the submodules.
"""

from .aggregator import (
    DEFAULT_WEIGHTS,
    aggregate_dependence,
    estimator_agreement,
    validate_weights,
)
from .estimators import (
    DEFAULT_EMBEDDING_MODEL,
    ESTIMATORS,
    CitationEstimator,
    EmbeddingEstimator,
    LexicalEstimator,
    cosine,
)
from .models import TOLERANCE, DependenceReport

__all__ = [
    # aggregation
    "aggregate_dependence",
    "estimator_agreement",
    "validate_weights",
    "DEFAULT_WEIGHTS",
    # estimators
    "EmbeddingEstimator",
    "CitationEstimator",
    "LexicalEstimator",
    "ESTIMATORS",
    "DEFAULT_EMBEDDING_MODEL",
    "cosine",
    # data
    "DependenceReport",
    "TOLERANCE",
]
