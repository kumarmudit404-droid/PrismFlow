"""arXiv connector -- the Tech angle's literature evidence.

No credential is required, which makes this the one connector that works on a
fresh checkout with an empty .env. It is therefore the connector the live
integration tests lean on hardest.

A TRANSPORT DETAIL THAT LOOKS LIKE A BUG
----------------------------------------
arXiv's export host answers ``406 Not Acceptable`` to requests that send
``Accept-Encoding: identity`` -- which is precisely what ``urllib`` sends by
default. ``requests`` and ``aiohttp`` both send ``gzip, deflate`` and get 200.
Measured on 2026-09-23, all three with the same User-Agent and URL.

The practical consequence: this connector must not be "simplified" onto
urllib, and a future 406 here is a transport-header problem rather than a
malformed query. The default AiohttpClient is correct as-is; the note exists
so the next person does not spend an hour on it.

PROSE IS NOT A SEARCH QUERY -- THE V2-L3 ROOT CAUSE
---------------------------------------------------
This connector used to send ``all:{query}`` verbatim. When ``query`` was a prose
sentence, arXiv bound none of its terms and answered with its own default
top-relevance listing: the same ten big-collaboration physics papers (GWTC-4.0,
GWTC-5.0, ATLAS detector performance, KAGRA dark matter, IceCube follow-ups) for
every unrelated query. That is V2-L3, and it reached this connector because
``BaseAngle`` only derives a query when one EXCEEDS the connector's limit -- and
this connector's 220-character limit is the most permissive in the table, so 36
of the 48 Part 24 pitches were short enough to arrive as raw prose.

Measured live on 2026-09-28, rows 020 / 046 / 049, k=10, mean over the three
pairwise comparisons. "generic" is the fraction of returned titles that are
big-collaboration physics; "term-hit" the fraction whose title contains at least
one of that row's own query terms:

    strategy                      mean n   overlap   term-hit   generic
    as-built, raw prose            10.0     10.00      0.07      1.00
    terms, space separated         10.0      0.00      0.97      0.00
    terms, each all:-prefixed, OR  10.0      0.00      0.97      0.00   <- this
    terms, each all:-prefixed, AND  0.0      0.00      0.00      0.00
    top-4 ANDed                     0.3      0.00      1.00      0.00
    top-6 ORed                     10.0      0.00      0.93      0.00

Overlap 10.00 means all three unrelated queries returned the identical ten
records. ANDing is unusable: arXiv's ``all:`` fields are conjunctive enough that
four terms already reduce most rows to zero.

Space-separated and explicitly-ORed scored identically, which says arXiv's
implicit operator here is OR. The explicit form is used anyway: it does not rely
on undocumented default behaviour, and it is visible in the URL when someone
reads a request log.

The fix is in this connector rather than in the length gate, because a connector
should never accept prose in a field that takes a query language -- regardless
of how short the prose happens to be. ``BaseAngle`` is untouched.

WHY http AND NOT https FOR THE EXPORT HOST
------------------------------------------
``export.arxiv.org`` is the documented API host and answers on both schemes.
``arxiv.org/api/query`` did not resolve from this machine (DNS failure,
measured), so the export host is used explicitly rather than the bare domain
the brief quotes.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Any, List, Optional, Sequence
from urllib.parse import quote_plus

from .base import (
    AngleConnector,
    HTTPClient,
    NormalizedRecord,
    RawResponse,
    Record,
    count_tokens,
    utcnow,
)
from .errors import ParseError

logger = logging.getLogger("prismflow.v2.connectors.arxiv")

API_URL = "http://export.arxiv.org/api/query"

#: arXiv asks for no more than one request every three seconds.
DEFAULT_RATE_PER_MIN = 20

#: Brief: snippet is the summary truncated to 500 characters.
SNIPPET_CHARS = 500

#: Ceiling and term cap applied when reducing an incoming query to terms.
#: 220 matches CONNECTOR_QUERY_LIMITS["arxiv"] so a query that HAS been derived
#: upstream passes through this reduction unchanged; 12 matches derive_query's
#: own default. Neither number is a new tuning knob -- they exist so this
#: connector never depends on whether derivation ran before it.
SEARCH_QUERY_LIMIT = 220
MAX_SEARCH_TERMS = 12

_ATOM = "{http://www.w3.org/2005/Atom}"


class ArxivConnector(AngleConnector):
    """Query the arXiv Atom API."""

    def __init__(
        self,
        rate_limit_per_min: int = DEFAULT_RATE_PER_MIN,
        *,
        http_client: Optional[HTTPClient] = None,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            "arxiv",
            rate_limit_per_min,
            http_client=http_client,
            max_retries=max_retries,
        )

    def build_search_query(self, query: str) -> str:
        """Turn whatever arrives into a search_query arXiv can actually bind.

        THIS EXISTS BECAUSE ``all:{query}`` WAS WRONG, AND V2-L3 WAS THE COST.
        See the module docstring's "PROSE IS NOT A SEARCH QUERY" section for the
        measurement. In short: pasting a prose sentence after ``all:`` binds
        nothing, and arXiv answers with its default top-relevance listing -- the
        same ten big-collaboration physics papers for every unrelated query.

        A caller that supplies its own field prefix knows arXiv's syntax and is
        passed through untouched. Everything else is reduced to terms and ORed
        with an explicit ``all:`` on each one, so no term's binding depends on
        arXiv's undocumented handling of a bare word after a prefixed one.

        Reduction reuses ``derive_query`` rather than re-implementing term
        extraction here: it is the module that owns stopwords and phrase
        detection, it is already tested, and running it over an
        already-derived term string is idempotent in practice.
        """
        if ":" in query:
            return query

        from prismflow.v2.angles.query_derivation import derive_query

        derived = derive_query(
            query, limit=SEARCH_QUERY_LIMIT, max_terms=MAX_SEARCH_TERMS
        )
        terms = [t for t in derived.terms if t.strip()]
        if not terms:
            # Nothing extractable. A trimmed prose prefix is a poor query, but
            # an empty search_query is a 400, and dropping the call silently
            # would report "no records" for a request never made.
            return f"all:{derived.text or query}"

        # A multi-word phrase MUST be quoted: unquoted, arXiv reads only its
        # first word and silently widens the query.
        return " OR ".join(
            f'all:"{t}"' if " " in t else f"all:{t}" for t in terms
        )

    async def fetch(self, query: str, k: int) -> RawResponse:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")
        search = self.build_search_query(query)
        url = (
            f"{API_URL}?search_query={quote_plus(search)}"
            f"&start=0&max_results={int(k)}"
            "&sortBy=relevance&sortOrder=descending"
        )
        return await self._get(url)

    def parse(self, raw: RawResponse) -> List[Record]:
        try:
            root = ET.fromstring(raw.body_text)
        except ET.ParseError as exc:
            raise ParseError(
                f"response body is not well-formed XML: {exc}",
                source=self.name,
            ) from exc

        fetched_at = utcnow()
        records: List[Record] = []
        for entry in root.findall(f"{_ATOM}entry"):
            raw_id = _text(entry, f"{_ATOM}id")
            arxiv_id = _arxiv_id(raw_id)
            summary = _text(entry, f"{_ATOM}summary") or ""
            author = None
            author_el = entry.find(f"{_ATOM}author")
            if author_el is not None:
                author = _text(author_el, f"{_ATOM}name")
            records.append(
                Record(
                    id=arxiv_id,
                    title=_collapse(_text(entry, f"{_ATOM}title") or ""),
                    url=(f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id
                         else (raw_id or "")),
                    snippet=_collapse(summary)[:SNIPPET_CHARS],
                    source_name=self.name,
                    fetched_at=fetched_at,
                    extra={
                        "author": author,
                        "published_date": _parse_arxiv_date(
                            _text(entry, f"{_ATOM}published")
                        ),
                        "summary_full": _collapse(summary),
                    },
                )
            )
        return records

    def normalize(self, records: Sequence[Record]) -> List[NormalizedRecord]:
        normalized: List[NormalizedRecord] = []
        for rec in records:
            normalized.append(
                NormalizedRecord(
                    id=rec.id,
                    title=rec.title,
                    url=rec.url,
                    snippet=rec.snippet,
                    snippet_tokens=count_tokens(rec.snippet),
                    source=rec.source_name,
                    published_date=rec.extra.get("published_date"),
                    author=rec.extra.get("author"),
                    relevance_score=1.0,  # reranked in Part 19
                )
            )
        return normalized


# --- helpers -----------------------------------------------------------


def _text(element: Any, path: str) -> Optional[str]:
    found = element.find(path)
    if found is None or found.text is None:
        return None
    return found.text.strip()


def _collapse(text: str) -> str:
    """arXiv wraps titles and abstracts at source width; unwrap them."""
    return " ".join(text.split())


def _arxiv_id(raw_id: Optional[str]) -> str:
    """``http://arxiv.org/abs/2401.01234v2`` -> ``2401.01234v2``."""
    if not raw_id:
        return ""
    return raw_id.rstrip("/").rsplit("/", 1)[-1]


def _parse_arxiv_date(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("unparseable arXiv timestamp: %r", value)
        return None


__all__ = ["ArxivConnector", "API_URL", "DEFAULT_RATE_PER_MIN",
           "SEARCH_QUERY_LIMIT", "MAX_SEARCH_TERMS"]
