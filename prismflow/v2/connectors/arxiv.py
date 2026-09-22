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

    async def fetch(self, query: str, k: int) -> RawResponse:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")
        # A bare term is not a valid search_query; arXiv wants a field prefix.
        search = query if ":" in query else f"all:{query}"
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


__all__ = ["ArxivConnector", "API_URL", "DEFAULT_RATE_PER_MIN"]
