"""GitHub connector -- the Tech angle's repository evidence.

RATE LIMITS: THE SPEC'S NUMBERS ARE THE WRONG ONES
--------------------------------------------------
The Part 17 brief says 30 req/min unauthenticated and 60 with a token. Those
are GitHub's CORE REST limits (60/hr unauth, 5000/hr authenticated, which is
where "60" comes from). The SEARCH endpoints this connector uses are governed
separately and far more tightly:

    /search/*   10 requests/minute unauthenticated
                30 requests/minute authenticated

Measured against the live API on 2026-09-23, an unauthenticated search
returned ``x-ratelimit-limit: 10``. Defaulting to the brief's 30/min
unauthenticated would exceed the real limit threefold and earn a 403 storm
partway through any Part 19 fan-out. The defaults below follow the measured
limits; ``rate_limit_per_min`` is still a constructor argument, so the brief's
values can be restored explicitly if GitHub's documented limits change.

WHY A PLACEHOLDER TOKEN IS TREATED AS NO TOKEN
----------------------------------------------
``.env`` ships from a template whose GITHUB_TOKEN is the literal string
``ghp_...``. Passing that through produces ``Authorization: Bearer ghp_...``,
which GitHub answers with 401 -- measured, not assumed. Unauthenticated
search, by contrast, answers 200. So a template placeholder is strictly worse
than sending nothing, and the failure is confusing: the connector looks
broken when in fact the credential was never filled in.

``_resolve_token`` therefore rejects template-shaped values and falls back to
unauthenticated access with a warning. A real token is still strongly
preferred -- it triples the search rate limit -- but its absence degrades
throughput rather than breaking the connector.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence
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

logger = logging.getLogger("prismflow.v2.connectors.github")

API_ROOT = "https://api.github.com"

#: Measured search limits (see module docstring), not the core REST limits.
SEARCH_RATE_UNAUTHENTICATED = 10
SEARCH_RATE_AUTHENTICATED = 30

#: Substrings that mark a value as an unfilled template placeholder.
_PLACEHOLDER_MARKERS = ("...", "<", "your_", "xxx", "changeme")


def _resolve_token(explicit: Optional[str]) -> Optional[str]:
    """Return a usable token, or None if what we have is a placeholder."""
    token = explicit if explicit is not None else os.getenv("GITHUB_TOKEN")
    if token is None:
        return None
    token = token.strip()
    if not token:
        return None
    lowered = token.lower()
    if any(marker in lowered for marker in _PLACEHOLDER_MARKERS):
        logger.warning(
            "GITHUB_TOKEN looks like an unfilled template placeholder (%r); "
            "falling back to unauthenticated access at %d req/min. Fill it in "
            "to get %d req/min.",
            token,
            SEARCH_RATE_UNAUTHENTICATED,
            SEARCH_RATE_AUTHENTICATED,
        )
        return None
    return token


class GitHubConnector(AngleConnector):
    """Search GitHub repositories for the Tech angle."""

    def __init__(
        self,
        token: Optional[str] = None,
        rate_limit_per_min: Optional[int] = None,
        *,
        http_client: Optional[HTTPClient] = None,
        max_retries: int = 3,
    ) -> None:
        self.token = _resolve_token(token)
        if rate_limit_per_min is None:
            rate_limit_per_min = (
                SEARCH_RATE_AUTHENTICATED if self.token
                else SEARCH_RATE_UNAUTHENTICATED
            )
        super().__init__(
            "github",
            rate_limit_per_min,
            http_client=http_client,
            max_retries=max_retries,
        )

    @property
    def authenticated(self) -> bool:
        return self.token is not None

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    async def fetch(self, query: str, k: int) -> RawResponse:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")
        per_page = min(int(k), 100)  # GitHub caps per_page at 100
        url = (
            f"{API_ROOT}/search/repositories"
            f"?q={quote_plus(query)}&sort=stars&order=desc&per_page={per_page}"
        )
        return await self._get(url, headers=self._headers())

    def parse(self, raw: RawResponse) -> List[Record]:
        try:
            payload: Any = json.loads(raw.body_text)
        except json.JSONDecodeError as exc:
            raise ParseError(
                f"response body is not JSON: {exc}", source=self.name
            ) from exc
        if not isinstance(payload, dict):
            raise ParseError(
                f"expected a JSON object, got {type(payload).__name__}",
                source=self.name,
            )
        items = payload.get("items")
        if items is None:
            raise ParseError(
                "response has no 'items' key; "
                f"keys present: {sorted(payload)[:8]}",
                source=self.name,
            )
        if not isinstance(items, list):
            raise ParseError(
                f"'items' is {type(items).__name__}, expected list",
                source=self.name,
            )

        fetched_at = utcnow()
        records: List[Record] = []
        for item in items:
            if not isinstance(item, dict):
                raise ParseError(
                    f"item is {type(item).__name__}, expected object",
                    source=self.name,
                )
            owner = item.get("owner") or {}
            records.append(
                Record(
                    id=str(item.get("id", "")),
                    title=str(item.get("name") or item.get("full_name") or ""),
                    url=str(item.get("html_url", "")),
                    snippet=str(item.get("description") or ""),
                    source_name=self.name,
                    fetched_at=fetched_at,
                    extra={
                        "author": (owner.get("login") if isinstance(owner, dict)
                                   else None),
                        "published_date": _parse_github_date(
                            item.get("created_at")
                        ),
                        "stars": item.get("stargazers_count"),
                        "full_name": item.get("full_name"),
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
                    # Part 19 reranks. Until then every record is equally
                    # relevant; inventing a score here would be a fabricated
                    # ordering that later parts would silently inherit.
                    relevance_score=1.0,
                )
            )
        return normalized


def _parse_github_date(value: Any) -> Optional[datetime]:
    """Parse GitHub's ISO-8601 'Z' timestamps into aware datetimes."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("unparseable GitHub timestamp: %r", value)
        return None


__all__ = ["GitHubConnector", "SEARCH_RATE_UNAUTHENTICATED",
           "SEARCH_RATE_AUTHENTICATED"]
