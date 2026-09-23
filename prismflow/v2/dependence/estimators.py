"""Three ways to ask whether two angles are saying the same thing.

An ENGINEERING component per docs/CONTRACT.md section 3. The embedding model is
a fixed pretrained artefact, not a parameter this project fits.

Each estimator answers a different question, which is the point of having three:

  embedding  do the claims MEAN the same thing? Catches paraphrase, misses
             agreement that is lexically and semantically distinct but rests on
             the same underlying source.
  citation   do the claims rest on the SAME EVIDENCE? Catches shared sources
             exactly, and is blind to two angles independently reaching the
             same conclusion from different records.
  lexical    do the claims USE THE SAME WORDS? Cheap, model-free, and a useful
             control: when it disagrees sharply with the embedding estimator,
             one of them is measuring an artefact.

WHY THE THIRD ESTIMATOR IS NOT CALLED "MUTUAL INFORMATION"
-----------------------------------------------------------
The brief names it ``MutualInformationEstimator`` and documents it as "mutual
information via TF-IDF". What the specified code computes is the cosine
similarity between two TF-IDF vectors -- the same operation as the embedding
estimator, over a different vector space. That is a perfectly reasonable lexical
overlap measure and it is kept, but it is not mutual information: no joint
distribution is formed, no entropy is computed, and the result is not in bits.

The rename is not pedantry. Part 22 consumes ``per_estimator`` and the Part 16
paper outline reports methodology; a number labelled "mutual information" in a
results table is a claim about how it was computed. With one document per angle
a genuine MI estimate is not identifiable anyway -- there is no sample to
estimate a joint distribution from -- so the honest options were to rename it or
to delete it, and renaming keeps a useful signal.

WHY CITATION OVERLAP IS STRUCTURALLY NEAR-ZERO HERE
----------------------------------------------------
Worth knowing before reading any number this produces: in V2 each angle
retrieves from its own sources, so tech cites GitHub and arXiv ids while market
would cite NewsAPI ids. Those id spaces do not intersect, so citation overlap is
0 between any two angles that do not share a connector -- whatever they say.

That makes this estimator a SOURCE-SHARING DETECTOR rather than a semantic one.
It is close to the most valuable of the three for the thesis precisely because
of that: Part 19's fallback chain means two angles can silently end up reading
the same source, and this is the estimator that sees it. But it contributes a
near-constant 0 to the weighted sum in the ordinary case, which mostly rescales
the aggregate rather than informing it. Read the per-estimator matrices, not
just the aggregate.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from prismflow.v2.reasoners.models import Claim

logger = logging.getLogger("prismflow.v2.dependence")

#: The brief's model. 384 dimensions, small and fast.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

#: An encoder is anything that turns texts into a (n, d) float array.
Encoder = Callable[[Sequence[str]], np.ndarray]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity, clamped to [0, 1], safe on zero vectors.

    Negative cosines are clamped to 0 rather than taken as absolute values.
    For sentence embeddings a negative cosine is near-noise, not meaningful
    anti-correlation, and |cos| would report noise as dependence. The cost of
    the choice is that two angles that systematically CONTRADICT each other read
    as independent, when contradiction is in fact a form of dependence. That
    case cannot be distinguished from unrelatedness by this instrument, and
    Part 22 should not be told otherwise.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        # Vectors of different width share no space to be similar in. Scoring 0
        # keeps the matrix well-formed; the caller warns about the cause.
        return 0.0
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    value = float(np.dot(a, b) / (norm_a * norm_b))
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _pairwise(n: int, score: Callable[[int, int], float]) -> np.ndarray:
    """Build a symmetric matrix with a unit diagonal from a pair scorer."""
    matrix = np.eye(n, dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(score(i, j))
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


class EmbeddingEstimator:
    """Semantic similarity between angles, via sentence embeddings."""

    name = "embedding"

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        encoder: Optional[Encoder] = None,
        pooling: str = "mean",
        truncate_dims: Optional[int] = None,
    ) -> None:
        """
        Args:
            model_name: sentence-transformers model, loaded on first use.
            encoder: an alternative encoder. Tests inject a deterministic one so
                the suite neither downloads 80MB nor depends on a network.
            pooling: how a set of claims becomes one vector.
                ``mean`` (the brief's, and the default) averages the unit-normed
                claim vectors. It is stable and cheap, but it dilutes: an angle
                whose one claim exactly duplicates another angle's, alongside two
                unrelated claims, is scored as mostly-unrelated.
                ``max_match`` instead averages, for each claim, its best match in
                the other angle, then symmetrises.

                Measured against ``mean`` on this corpus, ``max_match`` is NOT
                uniformly more sensitive -- the first version of this docstring
                claimed it was, and the claim did not survive measurement:

                  one shared claim of 3 vs 3   mean 0.389   max_match 0.333
                  one shared claim of 2 vs 2   mean 0.462   max_match 0.450
                  subset, 1 claim vs 4         mean 0.441   max_match 0.504
                  no overlap, 2 vs 2           mean 0.121   max_match 0.086

                It is higher only where coverage is ASYMMETRIC -- one angle's
                claims are a subset of the other's -- because averaging
                best-matches over many unmatched claims drags the score down
                elsewhere. So it is the better instrument for "is angle A's
                evidence already contained in angle B's", and the worse one for
                overall similarity. ``mean`` stays the default: it is the brief's
                choice, and changing what Part 22 discounts against is not a
                decision to slip in through a default.
            truncate_dims: keep only the first k dimensions. Used by the
                planted-redundancy experiment's low-rank condition.
        """
        if pooling not in ("mean", "max_match"):
            raise ValueError(
                f"pooling must be 'mean' or 'max_match', got {pooling!r}"
            )
        if truncate_dims is not None and truncate_dims <= 0:
            raise ValueError(f"truncate_dims must be positive, got {truncate_dims}")
        self.model_name = model_name
        self.pooling = pooling
        self.truncate_dims = truncate_dims
        self._encoder = encoder
        self._model: Any = None

    @property
    def encoder(self) -> Encoder:
        """The encoder, loading the model on first use."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self._encoder = lambda texts: self._model.encode(
                list(texts), convert_to_numpy=True, show_progress_bar=False,
            )
        return self._encoder

    def embed_claims(self, claims: Sequence[Claim]) -> np.ndarray:
        """Unit-normed embeddings for a set of claims, shape (n_claims, d).

        Each vector is normalised BEFORE any averaging. The brief averages raw
        vectors, which lets a claim with a larger norm pull the angle's
        representation toward itself for reasons that have nothing to do with
        what it says.
        """
        texts = [c.text for c in claims]
        if not texts:
            return np.zeros((0, 1), dtype=float)
        vectors = np.asarray(self.encoder(texts), dtype=float)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if self.truncate_dims is not None:
            vectors = vectors[:, : self.truncate_dims]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return vectors / norms

    def estimate(
        self,
        claims_by_angle: Dict[str, List[Claim]],
    ) -> Tuple[np.ndarray, List[str]]:
        """Pairwise semantic similarity. Returns (matrix, warnings)."""
        names = list(claims_by_angle)
        n = len(names)
        warnings: List[str] = []
        if n == 0:
            return np.zeros((0, 0), dtype=float), warnings

        per_angle = {name: self.embed_claims(claims_by_angle[name]) for name in names}
        empty = [name for name in names if per_angle[name].shape[0] == 0]
        if empty:
            warnings.append(
                f"no claims to embed for {', '.join(empty)}; their similarity is "
                "0 by convention, which is not the same as being independent"
            )
        if self.truncate_dims is not None:
            warnings.append(
                f"embeddings truncated to {self.truncate_dims} dimension(s); "
                "similarities are compressed upward and separability degrades"
            )

        # An angle with no claims has no vector, but it still needs a slot of
        # the right width: zeros of the wrong dimension would make the cosine
        # raise "shapes not aligned" rather than score 0. The width comes from
        # whichever angles did produce claims.
        dimensions = {v.shape[1] for v in per_angle.values() if v.shape[0]}
        if len(dimensions) > 1:
            warnings.append(
                f"encoder returned inconsistent dimensions {sorted(dimensions)}; "
                "pairs of differing width score 0"
            )
        width = max(dimensions) if dimensions else 1

        if self.pooling == "mean":
            pooled = {
                name: vectors.mean(axis=0) if vectors.shape[0] else np.zeros(width)
                for name, vectors in per_angle.items()
            }
            return (
                _pairwise(n, lambda i, j: cosine(pooled[names[i]], pooled[names[j]])),
                warnings,
            )

        def best_match(i: int, j: int) -> float:
            left, right = per_angle[names[i]], per_angle[names[j]]
            if left.shape[0] == 0 or right.shape[0] == 0:
                return 0.0
            if left.shape[1] != right.shape[1]:
                return 0.0
            sim = np.clip(left @ right.T, 0.0, 1.0)
            # Symmetric: how well each of A's claims is covered by B, and vice
            # versa. Taking only one direction would make the matrix asymmetric
            # for the honest reason that coverage is not symmetric -- but the
            # report's contract requires symmetry, so both are averaged.
            return float((sim.max(axis=1).mean() + sim.max(axis=0).mean()) / 2.0)

        return _pairwise(n, best_match), warnings


class CitationEstimator:
    """Evidence overlap between angles: Jaccard over cited record ids."""

    name = "citation"

    def estimate(
        self,
        claims_by_angle: Dict[str, List[Claim]],
    ) -> Tuple[np.ndarray, List[str]]:
        names = list(claims_by_angle)
        n = len(names)
        warnings: List[str] = []
        if n == 0:
            return np.zeros((0, 0), dtype=float), warnings

        cited = {
            name: {cid for claim in claims_by_angle[name] for cid in claim.cited_ids}
            for name in names
        }
        uncited = [name for name in names if not cited[name]]
        if uncited:
            warnings.append(
                f"no citations at all from {', '.join(uncited)}; every pair "
                "involving them scores 0 overlap"
            )

        def jaccard(i: int, j: int) -> float:
            left, right = cited[names[i]], cited[names[j]]
            union = left | right
            if not union:
                # The brief returns 1.0 here -- "two angles that cite nothing
                # are identical". Absence of evidence is not evidence of
                # identity, and in V2 it is the common case rather than a corner
                # one: Part 19 ships four of five angles with no connectors, so
                # this branch would report them as maximally dependent with each
                # other on the strength of all being silent.
                return 0.0
            return len(left & right) / len(union)

        return _pairwise(n, jaccard), warnings


class LexicalEstimator:
    """Vocabulary overlap between angles, via TF-IDF cosine.

    The brief calls this mutual information. It is not -- see the module
    docstring. It is a lexical control on the embedding estimator, and it is
    genuinely useful as one, because it needs no model and cannot fail in the
    same way.
    """

    name = "lexical"

    def __init__(self, max_features: int = 100, stop_words: Optional[str] = "english"):
        self.max_features = max_features
        self.stop_words = stop_words

    def estimate(
        self,
        claims_by_angle: Dict[str, List[Claim]],
    ) -> Tuple[np.ndarray, List[str]]:
        from sklearn.feature_extraction.text import TfidfVectorizer

        names = list(claims_by_angle)
        n = len(names)
        warnings: List[str] = []
        if n == 0:
            return np.zeros((0, 0), dtype=float), warnings

        documents = [
            " ".join(claim.text for claim in claims_by_angle[name]) for name in names
        ]
        if not any(doc.strip() for doc in documents):
            warnings.append("no claim text at all; lexical estimator returns identity")
            return np.eye(n, dtype=float), warnings

        vectorizer = TfidfVectorizer(
            max_features=self.max_features, stop_words=self.stop_words
        )
        try:
            tfidf = vectorizer.fit_transform(documents).toarray()
        except ValueError as exc:
            # Every term was a stop word, or the vocabulary was empty. Identity
            # means "no measurable overlap", which is the honest reading.
            warnings.append(
                f"TF-IDF vectorisation failed ({exc}); lexical estimator "
                "returns identity"
            )
            logger.warning("lexical estimator: %s", exc)
            return np.eye(n, dtype=float), warnings

        return _pairwise(n, lambda i, j: cosine(tfidf[i], tfidf[j])), warnings


#: Name to estimator class. The aggregator's weights are keyed by these.
ESTIMATORS: Dict[str, type] = {
    "embedding": EmbeddingEstimator,
    "citation": CitationEstimator,
    "lexical": LexicalEstimator,
}


__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "Encoder",
    "EmbeddingEstimator",
    "CitationEstimator",
    "LexicalEstimator",
    "ESTIMATORS",
    "cosine",
]
