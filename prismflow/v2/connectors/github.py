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

WHY EIGHT TERMS RETRIEVE NOTHING -- THE V2-L2 ROOT CAUSE
--------------------------------------------------------
GitHub's search ANDs space-separated terms across repository name, description
and README. This connector used to forward whatever it was handed, which after
Part 19's derivation was 8-12 terms, so every one of the 48 Part 24 rows came
back HTTP **200** with ``total_count: 0``. Not a 422, not a 403, not a 401: a
valid search that legitimately matches nothing, which is why it was silent.

THE CREDENTIAL WAS NOT THE CAUSE. ``docs/v2-known-limitations.md`` recorded that
``GITHUB_TOKEN`` looked like a placeholder and called it "the first thing to
check". It was checked, on 2026-09-28, and it is not the cause: UNAUTHENTICATED
search returns 200 with ``x-ratelimit-limit: 10`` and real results for a
one-term query. The placeholder still costs three-fold throughput, and
``_resolve_token`` still rejects it, but it never had anything to do with the
zeros.

Measured live on 2026-09-28, six rows, two per domain (002, 032, 020, 003, 046,
049), phrases quoted, k=10. "overlap" is the mean pairwise count of shared
repositories -- the number that has to be 0, or unrelated rows are getting the
same evidence:

    strategy                    rows with records   median total_count   overlap
    as-built, 8-12 ANDed              0 of 6                  0            0.00
    all terms ORed                    0 of 6        HTTP 422 (>5 ops)       n/a
    top-1                             6 of 6            1,541,034          2.00
    top-2 ANDed                       4 of 6                100            0.00  <- this
    top-3 ANDed                       2 of 6                  0            0.00
    top-3 ORed                        6 of 6              1,618,457        3.33
    top-3 ORed in:name,description    6 of 6                983,735        3.33

Top-1 and any OR variant retrieve plenty and retrieve the WRONG thing: totals in
the millions, dominated by whichever mega-repository matches any single term, and
a non-zero overlap between unrelated rows. Top-3 is too tight. Top-2 is the only
setting that returns records while keeping overlap at zero and total_count in the
tens-to-hundreds, i.e. a genuinely specific result set.

Two rows still return zero at top-2 (002 "Bell Media" + mobile-only, 003
"AVFoundation Camera" + platform). Those are honest zeros for a specific term
pair, not the systematic zero this replaces.

KNOWN AND NOT FIXED HERE: relevance is limited by what leads the derived term
list. Commit 81c9550 records that all 32 newly sourced pitches follow one of two
templates -- 12 open "Pitch a", 20 open "Build a" -- so "build" or "pitch" often
takes one of the two AND slots. Dropping those verbs was measured (same 4 of 6
coverage, same 0.00 overlap, median total 100 -> 60, and visibly more on-topic
descriptions on 3 of the 4 rows that return records), but it is a STOPWORDS
change, which rewrites every derived query and therefore every Part 18 cache
key. It is not required to fix V2-L2, so it is reported and left out rather than
bundled into a fix commit.

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

#: How many terms to AND. 2 is measured, not chosen -- see the module docstring.
AND_TERMS = 2

#: Character ceiling for the reduction. Matches CONNECTOR_QUERY_LIMITS["github"]
#: so an already-derived query is not re-cut to a different length here.
Q_LIMIT = 180

#: GitHub rejects a query with more than five AND/OR/NOT operators (HTTP 422,
#: "More than five AND / OR / NOT operators were used." -- measured). The
#: conjunction above uses spaces, not operators, so it cannot trip this; the
#: constant is recorded because it is what rules OR-joining out entirely.
MAX_BOOLEAN_OPERATORS = 5


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

    def build_q(self, query: str) -> str:
        """Reduce an incoming query to the few terms GitHub can actually match.

        THIS EXISTS BECAUSE GITHUB ANDs, AND V2-L2 WAS THE COST. See the module
        docstring's "WHY EIGHT TERMS RETRIEVE NOTHING" section. GitHub treats
        space-separated terms as a conjunction over repository name, description
        and README, so the 8-12 term strings this connector used to forward
        matched no repository on any of the 48 Part 24 rows -- HTTP 200,
        ``total_count: 0``, every time.

        A caller using GitHub's own qualifier syntax (``language:``, ``stars:``,
        ``in:``) knows what it is doing and is passed through untouched.

        Reduction reuses ``derive_query`` for term extraction, so stopwords and
        phrase detection are not re-implemented here.
        """
        if ":" in query:
            return query

        from prismflow.v2.angles.query_derivation import derive_query

        derived = derive_query(query, limit=Q_LIMIT, max_terms=AND_TERMS)
        terms = [t for t in derived.terms if t.strip()]
        if not terms:
            return derived.text or query

        # Quoted, because an unquoted multi-word phrase becomes N separate
        # ANDed words and tightens the conjunction without meaning to.
        return " ".join(f'"{t}"' if " " in t else t for t in terms)

    async def fetch(self, query: str, k: int) -> RawResponse:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")
        per_page = min(int(k), 100)  # GitHub caps per_page at 100
        url = (
            f"{API_ROOT}/search/repositories"
            f"?q={quote_plus(self.build_q(query))}"
            f"&sort=stars&order=desc&per_page={per_page}"
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
                    snippet=rec.snippet,
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
           "SEARCH_RATE_AUTHENTICATED", "AND_TERMS", "Q_LIMIT",
           "MAX_BOOLEAN_OPERATORS"]
