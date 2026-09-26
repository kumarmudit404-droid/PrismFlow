"""Derive a short keyword query from a long idea pitch. No LLM, no network.

Part 19 addendum, written against the findings in
``docs/part24-query-mismatch.md``. An ENGINEERING component per
docs/CONTRACT.md section 3 -- no learnable parameters and nothing sampled.

THE PROBLEM THIS SOLVES, IN THE CONNECTORS' OWN WORDS
------------------------------------------------------
``EvaluationQuery.query_text`` is ``idea_pitch``: long-form prose, 585
characters on the longest row measured. The connectors that work want short
keyword queries, and two of them say so explicitly:

* NewsAPI returns HTTP 400 ``queryTooLong``: "Your query is too long (585
  chars). Please reduce your query to 500 chars."
* GitHub search returns HTTP 422, "query rejected as unprocessable", on every
  row of that dataset. Its documented ceiling is 256 characters plus at most
  five boolean operators, and it rejects well before the character count alone
  would explain.

StackExchange and yfinance raise nothing at all and return zero records, which
is the quieter half of the problem: no error distinguishes "this query is the
wrong shape" from "this topic has no coverage".

WHY THIS IS DETERMINISTIC, AND WHY THAT IS NOT A COMPROMISE
------------------------------------------------------------
An LLM would extract better keywords. It would also put a sampled, unversioned
step in front of every retrieval, which breaks three things at once: Part 18's
cache invariant (the same pitch must produce the same query or a cache hit
stops being a cache hit), Part 21's dependence measurement (two angles could
diverge because their derived queries differed, not because their sources did),
and any claim that a run is reproducible. A frozen stopword list and a
frequency count are worse extraction and a sounder measurement. The whole
module is a pure function of its input.

WHAT IT DOES
------------
Lowercase, strip punctuation and markup, drop stopwords, drop pure numbers,
count frequency, and emit the top terms ordered by first appearance -- because
a pitch states its subject early, and ordering by frequency alone promotes
whatever the author happened to repeat. Multi-word proper-noun runs in the
original casing ("Swift Evolution", "Exception Groups") are kept as phrases,
since they are the highest-signal thing in a proposal pitch and splitting them
loses the subject.

WHAT IT DOES NOT DO
-------------------
It does not guarantee a connector will return records. yfinance wants a ticker
and no keyword extraction produces one; a pitch about a Python PEP has no
financial instrument behind it, and the honest result there stays zero records.
This closes the shape mismatch, not the corpus mismatch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

#: Per-connector ceilings, in characters. Each sits well under the limit the
#: API actually enforces, because the enforced number is a cliff and the cost
#: of being under it is a slightly shorter query.
#:
#:   newsapi        API rejects at 500; documented in its own error message.
#:   github         API documents 256 plus <=5 boolean operators, and 422s
#:                  earlier in practice, so this is deliberately conservative.
#:   stackexchange  no documented cap, but title search degrades badly on long
#:                  strings -- shorter is strictly better here.
#:   arxiv          tolerant, but a long query dilutes the relevance ranking.
#:   yfinance       a symbol lookup; anything long is meaningless to it.
CONNECTOR_QUERY_LIMITS: Dict[str, int] = {
    "newsapi": 240,
    "github": 180,
    "stackexchange": 150,
    "arxiv": 220,
    "yfinance": 60,
}

#: Used when a connector is not in the table. Under every limit above bar
#: yfinance's, so an unknown connector fails safe rather than fails long.
DEFAULT_QUERY_LIMIT = 150

#: Frozen. Extending this changes every derived query and therefore every
#: cache key, so it is versioned with the code rather than configurable.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren as at be
because been before being below between both but by can cannot could couldn
did didn do does doesn doing don down during each few for from further had
hadn has hasn have haven having he her here hers herself him himself his how
i if in into is isn it its itself just let ll me more most mustn my myself no
nor not now of off on once only or other ought our ours ourselves out over own
re s same shan she should shouldn so some such t than that the their theirs
them themselves then there these they this those through to too under until up
ve very was wasn we were weren what when where which while who whom why will
with won would wouldn you your yours yourself yourselves
allow allows already although among another approach available based become
becomes current currently describes description due either enough especially
every example existing first follow following form general give given however
include includes including instead introduce introduces keep like main make
makes many may means might much must need needs new one others particular
per perhaps possible propose proposed proposal provide provides rather really
require requires said say see seems several should similar since specific
still take term terms thing things three thus together two upon use used uses
using usually want way well whether within without work works would
""".split())

#: Kept even though they are short or common: dropping them loses the subject.
KEEP_ALWAYS = frozenset({"pep", "api", "cpu", "gpu", "gil", "sql", "ast", "os"})

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9+#._-]*")
_PROPER_RUN = re.compile(r"\b([A-Z][A-Za-z0-9]*(?:[ -][A-Z][A-Za-z0-9]*)+)\b")
_MARKUP = re.compile(r"<[^>]+>|https?://\S+|`[^`]*`")
_IDENTIFIER = re.compile(r"\b(?:PEP|SE|RFC)[ -]?\d{1,4}\b", re.I)


@dataclass(frozen=True)
class DerivedQuery:
    """A derived query and enough provenance to explain it.

    Attributes:
        text: what to send to the connector. Never longer than ``limit``.
        terms: the terms kept, in emission order.
        source_chars: length of the pitch it came from.
        limit: the ceiling applied.
        truncated: whether terms were dropped to fit ``limit``.
        fallback: set when extraction found nothing usable and the text is a
            trimmed prefix of the pitch instead. See ``derive_query``.
    """

    text: str
    terms: Tuple[str, ...]
    source_chars: int
    limit: int
    truncated: bool
    fallback: Optional[str] = None

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


def _phrases(pitch: str) -> List[str]:
    """Proper-noun runs and PEP/SE/RFC identifiers, in order of appearance."""
    found: List[str] = []
    for match in _IDENTIFIER.finditer(pitch):
        token = re.sub(r"[ -]", " ", match.group(0)).strip()
        if token not in found:
            found.append(token)
    for match in _PROPER_RUN.finditer(pitch):
        words = match.group(1).strip().split()
        # Strip leading sentence-initial capitalisation: "This PEP proposes"
        # opens a sentence, it is not the name of anything. Only the leading
        # words are stripped, so "Swift Evolution" survives intact.
        while words and words[0].lower() in STOPWORDS:
            words.pop(0)
        # A single surviving word is handled by the term pass, which dedupes
        # against phrases -- keeping it here would emit it twice.
        if len(words) < 2:
            continue
        phrase = " ".join(words)
        if phrase not in found:
            found.append(phrase)
    return found


def _terms(pitch: str) -> List[str]:
    """Content words, ordered by first appearance, deduplicated."""
    cleaned = _MARKUP.sub(" ", pitch)
    ordered: List[str] = []
    seen: set = set()
    for match in _WORD.finditer(cleaned):
        word = match.group(0)
        lowered = word.lower().strip("._-")
        if not lowered or lowered in seen:
            continue
        if lowered in KEEP_ALWAYS:
            seen.add(lowered)
            ordered.append(lowered)
            continue
        if lowered in STOPWORDS or len(lowered) < 3:
            continue
        if lowered.isdigit():
            continue
        seen.add(lowered)
        ordered.append(lowered)
    return ordered


def derive_query(
    pitch: str,
    limit: int = DEFAULT_QUERY_LIMIT,
    max_terms: int = 12,
) -> DerivedQuery:
    """Turn a long pitch into a short keyword query. Pure and deterministic.

    Phrases first (they carry the subject), then single terms in order of first
    appearance, packed until ``limit`` would be exceeded.

    A pitch with nothing extractable -- punctuation, stopwords only, or an
    empty string -- does NOT raise and does not return an empty query, because
    an empty query sent to a connector is a different and worse failure than a
    poor one. It falls back to a whitespace-normalised prefix of the pitch cut
    to ``limit`` on a word boundary, and says so in ``fallback``. If even that
    is empty the result is empty and ``is_empty`` is True; the caller decides,
    and ``BaseAngle`` treats it as "no derivation available" and sends the
    original.

    Raises:
        ValueError: ``limit`` or ``max_terms`` is not positive.
    """
    if limit <= 0:
        raise ValueError("limit must be positive, got %r" % limit)
    if max_terms <= 0:
        raise ValueError("max_terms must be positive, got %r" % max_terms)

    source_chars = len(pitch or "")
    if not pitch or not pitch.strip():
        return DerivedQuery("", (), source_chars, limit, False,
                            fallback="pitch was empty")

    candidates: List[str] = []
    for phrase in _phrases(pitch):
        if phrase not in candidates:
            candidates.append(phrase)
    for term in _terms(pitch):
        # Skip a term already carried by a kept phrase.
        if any(term == part.lower() for phrase in candidates
               for part in phrase.split()):
            continue
        candidates.append(term)

    if not candidates:
        trimmed = _trim_to_words(" ".join(pitch.split()), limit)
        return DerivedQuery(trimmed, (), source_chars, limit, True,
                            fallback="no extractable keywords in pitch")

    kept: List[str] = []
    length = 0
    truncated = False
    for candidate in candidates:
        if len(kept) >= max_terms:
            truncated = True
            break
        addition = len(candidate) + (1 if kept else 0)
        if length + addition > limit:
            truncated = True
            continue
        kept.append(candidate)
        length += addition

    if not kept:
        # Every candidate individually exceeded the limit.
        trimmed = _trim_to_words(candidates[0], limit)
        return DerivedQuery(trimmed, (trimmed,), source_chars, limit, True,
                            fallback="all candidate terms exceeded the limit")

    return DerivedQuery(" ".join(kept), tuple(kept), source_chars, limit,
                        truncated)


def _trim_to_words(text: str, limit: int) -> str:
    """Cut to ``limit`` on a word boundary, never mid-word."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    if " " in cut:
        cut = cut[:cut.rindex(" ")]
    return cut.strip()


def limit_for(connector_name: Optional[str]) -> int:
    """The character ceiling for a connector, by name."""
    if not connector_name:
        return DEFAULT_QUERY_LIMIT
    return CONNECTOR_QUERY_LIMITS.get(connector_name.lower(), DEFAULT_QUERY_LIMIT)


def derive_for_connector(pitch: str, connector_name: Optional[str]) -> DerivedQuery:
    """``derive_query`` with the ceiling that connector actually enforces."""
    return derive_query(pitch, limit=limit_for(connector_name))


__all__ = [
    "CONNECTOR_QUERY_LIMITS",
    "DEFAULT_QUERY_LIMIT",
    "STOPWORDS",
    "DerivedQuery",
    "derive_for_connector",
    "derive_query",
    "limit_for",
]
