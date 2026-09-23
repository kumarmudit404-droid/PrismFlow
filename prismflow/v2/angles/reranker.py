"""Reranking records against a query: BM25, or a cross-encoder.

An ENGINEERING component per docs/CONTRACT.md section 3. Connectors return
records in whatever order the source chose (GitHub by stars, arXiv by its own
relevance); this module reorders them by relevance to the actual query and
writes a comparable ``relevance_score``.

WHAT THE BRIEF'S VERSION DID
----------------------------
``corpus = [r.title.split() + r.snippet_tokens.split() for r in records]``

``snippet_tokens`` is an ``int`` -- a count, not text -- so this raises
``AttributeError: 'int' object has no attribute 'split'`` on the first record,
every time. The cross-encoder path has the same mistake more quietly:
``f"{r.title} {r.snippet_tokens}"`` interpolates the number into the text and
scores the query against "hello-world 12".

There was no snippet text on NormalizedRecord to use. That is why Part 19
began with an authorised amendment to Part 17 adding ``snippet``; see
``connectors/base.py``. Reranking now scores title + snippet, which on real
arXiv results is ~10x the text that title alone would have given.

RANKING MUST BE DETERMINISTIC
-----------------------------
docs/CONTRACT.md section 5 requires reproducibility, and ranking feeds Part 21's
dependence estimates -- if two runs order the same records differently, apparent
dependence between angles becomes an artefact of tie-breaking. So: Python's sort
is stable and is fed the records in retrieval order, ties therefore keep
retrieval order, and when BM25 scores everything zero (no term overlap at all)
the original order is preserved rather than shuffled.

RECORDS ARE NOT MUTATED
-----------------------
The brief assigns ``record.relevance_score = score`` in place. Its own
``AngleEvidence`` then reports those same objects as ``raw_records``, described
as unranked -- so "raw" would carry ranked scores. Worse, ``MockGitHubConnector``
hands out the same objects on every call and the Part 18 cache hands out records
that a caller may hold elsewhere, so in-place mutation leaks across calls. Here
``dataclasses.replace`` produces new records and the inputs are left alone.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

from prismflow.v2.connectors.base import NormalizedRecord

logger = logging.getLogger("prismflow.v2.angles")

#: Ranking methods this module understands.
RERANK_METHODS = ("bm25", "cross-encoder", "none")

#: Cross-encoder used when ``method="cross-encoder"``. Downloading it needs
#: network access and roughly 80MB on first use, which is why failure falls
#: back rather than raising.
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@dataclass
class RerankOutcome:
    """The reranked records, plus what actually happened while ranking.

    ``method_used`` can differ from the method requested: a cross-encoder that
    cannot load falls back to BM25, and BM25 that cannot run falls back to
    retrieval order. The angle puts ``warning`` into its AngleEvidence so a
    degraded ranking is visible downstream instead of only in a log.
    """

    records: List[NormalizedRecord]
    method_used: str
    warning: Optional[str] = None
    all_zero: bool = False


def document_text(record: NormalizedRecord) -> str:
    """The text a record is ranked on: its title and its snippet."""
    return f"{record.title} {record.snippet}".strip()


def tokenize(text: str) -> List[str]:
    """Lowercased word tokens.

    Casefolding matters more than it looks: BM25 treats "Adversarial" and
    "adversarial" as different terms, so a capitalised title would not match a
    lowercase query. The brief's bare ``.split()`` keeps case and so silently
    under-matches.
    """
    return [token for token in text.lower().split() if token]


def rerank_records(
    query: str,
    records: Sequence[NormalizedRecord],
    method: str = "bm25",
) -> List[NormalizedRecord]:
    """Rerank ``records`` by relevance to ``query``. Never raises.

    Kept to the signature the brief specifies, for call sites that only want
    the list. ``rerank_with_outcome`` additionally reports how the ranking went,
    which is what the angles use.
    """
    return rerank_with_outcome(query, records, method).records


def rerank_with_outcome(
    query: str,
    records: Sequence[NormalizedRecord],
    method: str = "bm25",
) -> RerankOutcome:
    """Rerank, reporting the method actually used and any degradation.

    Standing rule 4 says a failed rerank must warn and return unsorted rather
    than raise. Implemented with one deliberate strengthening: a cross-encoder
    that cannot load falls back to BM25 rather than to unsorted, because BM25
    needs no model and no network, and discarding a ranking that is free to
    compute would lose evidence quality for no reason. Only if BM25 also fails
    does the order come back untouched.
    """
    records = list(records)
    if not records:
        return RerankOutcome(records=[], method_used="none")
    if method not in RERANK_METHODS:
        warning = (
            f"unknown rerank method {method!r}; leaving records in retrieval "
            f"order (known methods: {', '.join(RERANK_METHODS)})"
        )
        logger.warning(warning)
        return RerankOutcome(records=records, method_used="none", warning=warning)
    if method == "none":
        return RerankOutcome(records=records, method_used="none")

    if method == "cross-encoder":
        try:
            return _rerank_cross_encoder(query, records)
        except Exception as exc:  # model download, import, or inference
            warning = (
                f"cross-encoder rerank unavailable ({type(exc).__name__}: "
                f"{exc}); falling back to bm25"
            )
            logger.warning(warning)
            try:
                outcome = _rerank_bm25(query, records)
            except Exception as bm25_exc:
                fallback_warning = (
                    f"{warning}; bm25 also failed "
                    f"({type(bm25_exc).__name__}: {bm25_exc}); leaving records "
                    "in retrieval order"
                )
                logger.warning(fallback_warning)
                return RerankOutcome(
                    records=records, method_used="none",
                    warning=fallback_warning,
                )
            outcome.warning = warning
            return outcome

    try:
        return _rerank_bm25(query, records)
    except Exception as exc:
        warning = (
            f"bm25 rerank failed ({type(exc).__name__}: {exc}); leaving "
            "records in retrieval order"
        )
        logger.warning(warning)
        return RerankOutcome(records=records, method_used="none", warning=warning)


def _rerank_bm25(
    query: str,
    records: List[NormalizedRecord],
) -> RerankOutcome:
    """BM25 over title + snippet. No model, no network."""
    from rank_bm25 import BM25Okapi

    corpus = [tokenize(document_text(record)) for record in records]
    if not any(corpus):
        # Every document is empty, so BM25Okapi would divide by a zero average
        # document length. Nothing to rank on; say so instead of failing.
        warning = "every record has empty title and snippet; nothing to rank on"
        logger.warning(warning)
        return RerankOutcome(
            records=records, method_used="none", warning=warning, all_zero=True,
        )

    bm25 = BM25Okapi(corpus)
    scores = [float(s) for s in bm25.get_scores(tokenize(query))]
    return _apply_scores(records, scores, "bm25")


def _rerank_cross_encoder(
    query: str,
    records: List[NormalizedRecord],
) -> RerankOutcome:
    """Cross-encoder reranking: heavier, and needs the model present."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(CROSS_ENCODER_MODEL)
    pairs = [[query, document_text(record)] for record in records]
    raw = model.predict(pairs)
    scores = [float(s) for s in raw]

    # Cross-encoder logits are unbounded and can be negative, so they are
    # min-max scaled into [0, 1] to stay comparable with BM25's normalisation.
    low, high = min(scores), max(scores)
    spread = high - low
    if spread <= 0:
        scaled = [0.0 for _ in scores]
    else:
        scaled = [(s - low) / spread for s in scores]
    return _apply_scores(records, scaled, "cross-encoder", prescaled=True)


def _apply_scores(
    records: List[NormalizedRecord],
    scores: Sequence[float],
    method: str,
    *,
    prescaled: bool = False,
) -> RerankOutcome:
    """Sort by score and return new records carrying it.

    Scores are normalised to [0, 1] by the maximum, so ``relevance_score`` means
    "share of the best match in this result set". It is a within-set ranking
    signal and is NOT comparable across queries or across angles -- Part 23
    must not treat 0.8 from one angle as equal evidence to 0.8 from another.
    """
    all_zero = False
    if prescaled:
        normalised = list(scores)
    else:
        top = max(scores) if scores else 0.0
        if top <= 0.0:
            # No query term appears in any document. Every record is equally
            # (ir)relevant, so retrieval order is preserved rather than
            # arbitrarily permuted by equal scores.
            normalised = [0.0 for _ in scores]
            all_zero = True
        else:
            normalised = [max(0.0, s) / top for s in scores]

    paired: List[Tuple[NormalizedRecord, float]] = [
        (replace(record, relevance_score=score), score)
        for record, score in zip(records, normalised)
    ]

    # Stable sort: equal scores keep retrieval order, so the result is
    # reproducible run to run.
    paired.sort(key=lambda pair: pair[1], reverse=True)
    return RerankOutcome(
        records=[record for record, _ in paired],
        method_used="none" if all_zero else method,
        all_zero=all_zero,
    )


__all__ = [
    "RERANK_METHODS",
    "CROSS_ENCODER_MODEL",
    "RerankOutcome",
    "rerank_records",
    "rerank_with_outcome",
    "document_text",
    "tokenize",
]
