"""StackExchange connector -- the Sentiment angle's discussion evidence.

Part 17 interface. Extends ``base.py``; that file is unchanged.

WHY STACKEXCHANGE AND NOT REDDIT
--------------------------------
The Part 19 brief names Reddit (PRAW) as the Sentiment angle's primary source.
Measured on 2026-09-24 from this connection, Reddit is not usable here and the
failure is not a credential problem:

    DNS / TCP:443 / TLS        clean for both www.reddit.com and
                               oauth.reddit.com (TLSv1.3, Fastly addresses)
    POST /api/v1/access_token  HTTP 401, 41-byte body -- a genuine auth
                               challenge, so the token endpoint is reachable
    GET oauth.reddit.com/...   HTTP 403, "your request has been blocked due to
                               a network policy", with ANY bearer token
    GET www.reddit.com/*.json  HTTP 403, "You've been blocked by network
                               security" -- identical with a Chrome UA, so the
                               block is IP-scoped, not User-Agent-based

That combination is the worst shape for a connector: PRAW would authenticate
successfully and then 403 on every data call. Filling REDDIT_CLIENT_ID and
REDDIT_CLIENT_SECRET would not change it.

StackExchange is substituted because the angle takes injected connectors, so
nothing in Parts 19-23 changes -- only what sits behind the slot. Q&A with
community scores is a defensible sentiment signal: a score is an aggregate
judgement by readers, which is closer to what this angle is for than a raw
comment count would be.

WHAT GOES IN ``snippet`` -- A JUDGMENT CALL, THOUGH A SMALLER ONE THAN YFINANCE'S
--------------------------------------------------------------------------------
Unlike Yahoo, StackExchange returns real prose, so this case is closer to
arXiv's: the body is the text and it needs truncating, not inventing. Two
decisions were still made and neither is given by the source.

1. THE BODY IS HTML AND IS STRIPPED. ``filter=withbody`` returns markup --
   measured tags: p, a, blockquote, br, em, strong, h*. Left in, BM25 would
   score on tag names and Part 20's reasoner would read angle brackets as
   content. Tags are stripped and whitespace collapsed before truncation.

2. A BOUNDED SENTIMENT PREFIX IS PREPENDED:

       [score +42 | 7 answers | accepted | 51k views] How do I ...

   The score IS the sentiment measurement, and NormalizedRecord has nowhere
   else to put it -- Part 19 reranks on ``snippet`` and Part 20 reads
   ``snippet``, so a signal that lives only in ``extra`` is invisible to both.
   The prefix is fixed-shape and capped, so it cannot grow with the post. The
   full numbers also ride in ``Record.extra`` for anything that wants them
   unparsed.

THE FOUR DEFECT-CLASS CHECKS
----------------------------
1. CACHE-KEY COLLISION (Part 18 class) -- FOUND, the same shape as yfinance's.
   ``name="stackexchange"`` is distinct from github/arxiv/yfinance, so the
   cross-connector collision does not arise. But the SITE is part of what was
   asked: the same query against ``stackoverflow`` and against ``politics``
   returns entirely different posts, and the Part 18 key is
   ``(connector_name, sha256(query)[:16])`` with no site component. As a
   constructor argument the site would be invisible to the key and the second
   caller would be served the first's answer. It is therefore encoded in the
   query string (``"inflation@economics"``), exactly as the period is for
   yfinance.

2. RATE LIMITING -- DRIVEN BY THE API'S OWN FIELDS, NOT A GUESS, AND THE BASE
   CLASS GETS IT WRONG FOR THIS SOURCE.

   StackExchange signals throttling in two places, neither of which is a 429:

   a. ``backoff`` in the body of a SUCCESSFUL 200 response. It is an integer
      number of seconds, and the API documents that violating it earns a
      throttle_violation on the next call. A success that carries a throttle
      instruction has no equivalent in the other connectors, so
      ``_throttle`` is extended to wait out a pending backoff before the
      normal rate-limit spacing.

   b. ``error_id: 502, error_name: "throttle_violation"``, served with HTTP
      502. That is the defect: ``AngleConnector._check_status`` maps any
      status >= 500 to ``UpstreamError``, so an explicit "you are being
      throttled" would be reported to Part 19 as "the source is broken".
      Both are retried, so nothing hangs, but the operator would be told the
      wrong thing and the ``retry_after`` would be discarded.
      ``_check_status`` is overridden here to read the JSON error body --
      the full id list was fetched from the API's own ``/2.3/errors``
      endpoint rather than recalled:

          400 bad_parameter          402 invalid_access_token
          401 access_token_required  403 access_denied
          404 no_method              405 key_required
          406 access_token_compromised  407 write_failed
          409 duplicate_request      500 internal_error
          502 throttle_violation     503 temporarily_unavailable

   QUOTA, MEASURED: anonymous access gives ``quota_max: 300`` per day per IP,
   and it decrements one per request (observed 300 -> 294 over six probe
   calls). ``quota_remaining`` is read off every response and logged, and
   drops below ``QUOTA_WARN_THRESHOLD`` are warned about loudly.

   DOES A stackapps.com KEY BECOME NECESSARY? Not for this part, and yes for
   Part 24 -- stated rather than assumed either way. 300 requests/day is
   ample for unit tests (which make none), the live test (one) and manual
   smoke runs (a handful). It is NOT enough for the Part 24 harness as
   scoped: 50 queries x 5 seeds x 4 randomisations is 1000 pipeline runs, and
   even one StackExchange call per run exceeds the anonymous quota by 3.3x on
   the first day. A key raises the ceiling to 10,000/day and is requested at
   stackapps.com with no review step. ``api_key`` is therefore already a
   constructor argument and is threaded to the ``key`` parameter, defaulting
   to None so nothing about today's behaviour depends on having one.

3. TIMEZONE -- CONFIRMED, AND DIFFERENT AGAIN FROM BOTH OTHER SOURCES.
   ``creation_date`` is a Unix epoch INTEGER (measured: ``1642619486``), not
   an ISO string. It is UTC by definition, but it arrives with no tzinfo at
   all, so ``datetime.fromtimestamp`` would silently apply THIS MACHINE's
   local zone -- on this box, +05:30 -- and every StackExchange date would
   land five and a half hours early against arXiv and GitHub. It is parsed
   with an explicit ``tz=timezone.utc``. Three sources, three different date
   conventions: arXiv ISO-with-Z, yfinance exchange-local, StackExchange
   naive epoch.

4. OVERSIZE (Part 19 class) -- FOUND, and the raw body really does breach the
   budget. Measured over 30 top-voted "machine learning" questions: bodies
   ranged 194 to 11834 characters, median 931. The largest is 3455 tokens as
   raw HTML and 2783 tokens stripped -- both over the entire 2000-token angle
   budget, so Part 19 would drop that record and warn. Tag-stripping alone is
   NOT sufficient; the text is truncated to ``SNIPPET_CHARS`` as well, which
   holds the worst case to roughly 150 tokens. ``_assert_snippet_bounded``
   pins the invariant at construction.
"""

from __future__ import annotations

import html
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

from .base import (
    AngleConnector,
    HTTPClient,
    NormalizedRecord,
    RawResponse,
    Record,
    count_tokens,
    utcnow,
)
from .errors import AuthError, ParseError, QueryError, RateLimitError, UpstreamError

logger = logging.getLogger("prismflow.v2.connectors.stackexchange")

API_URL = "https://api.stackexchange.com/2.3/search/advanced"

#: 300/day anonymous works out to roughly 12/hour if spread evenly. 30/min is
#: the spacing between bursts, not a daily budget -- the daily ceiling is
#: enforced by the API and surfaced through ``quota_remaining``.
DEFAULT_RATE_PER_MIN = 30

#: Default StackExchange site. Overridable per query via ``"query@site"``.
DEFAULT_SITE = "stackoverflow"

#: Separator between the search terms and the site in a query string.
#: Present so the site lands inside the Part 18 cache key -- see note 1.
SITE_SEP = "@"

#: Returns the post body. Without it the API omits body entirely and every
#: snippet would be the title alone -- the same 10%-of-the-signal problem the
#: Part 17 amendment was made to fix.
BODY_FILTER = "withbody"

#: Truncation cap on the stripped body. Measured worst case is 10510 stripped
#: characters (2783 tokens) against a 2000-token angle budget -- see note 4.
SNIPPET_CHARS = 600

#: Warn when the daily anonymous quota gets this low.
QUOTA_WARN_THRESHOLD = 50

#: StackExchange error ids, from the API's own /2.3/errors endpoint.
THROTTLE_ERROR_ID = 502
AUTH_ERROR_IDS = frozenset({401, 402, 403, 405, 406})
QUERY_ERROR_IDS = frozenset({400, 404, 409})


class StackExchangeConnector(AngleConnector):
    """Search one StackExchange site's questions.

    Anonymous by default. ``api_key`` raises the daily quota from 300 to
    10,000 and is not needed below Part 24 scale -- see the module docstring.
    """

    def __init__(
        self,
        rate_limit_per_min: int = DEFAULT_RATE_PER_MIN,
        *,
        http_client: Optional[HTTPClient] = None,
        max_retries: int = 3,
        default_site: str = DEFAULT_SITE,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(
            "stackexchange",
            rate_limit_per_min,
            http_client=http_client,
            max_retries=max_retries,
        )
        if not default_site or not default_site.strip():
            raise ValueError("default_site must be a non-empty string")
        self.default_site = default_site.strip()
        self.api_key = api_key
        self.quota_remaining: Optional[int] = None
        #: monotonic deadline set from a 200 response's ``backoff`` field
        self._backoff_until = 0.0

    # -- throttling ------------------------------------------------------

    async def _throttle(self) -> None:
        """Wait out any API-instructed backoff, then the normal spacing.

        StackExchange can return ``backoff`` on a SUCCESSFUL response, which
        means "do not call again for N seconds". Ignoring it earns a
        throttle_violation on the next request, so it is honoured here rather
        than discovered later. See defect-class note 2a.
        """
        import asyncio

        wait = self._backoff_until - time.monotonic()
        if wait > 0:
            logger.warning(
                "stackexchange: API-instructed backoff, sleeping %.1fs", wait
            )
            await asyncio.sleep(wait)
        await super()._throttle()

    def _check_status(self, raw: RawResponse) -> None:
        """Map StackExchange's error envelope onto the Part 17 hierarchy.

        Overridden because this source does NOT use 429 for throttling: it
        sends ``error_id: 502`` with HTTP 502, which the base class would
        classify as UpstreamError. See defect-class note 2b.
        """
        if 200 <= raw.status_code < 300:
            return

        error_id, error_name, message = _error_envelope(raw.body_text)
        if error_id is None:
            super()._check_status(raw)
            return

        detail = f"{error_name or 'error'} ({error_id}): {message or 'no message'}"
        if error_id == THROTTLE_ERROR_ID:
            raise RateLimitError(
                f"throttled by StackExchange -- {detail}",
                source=self.name,
                status_code=raw.status_code,
                retry_after=_retry_after_from_message(message),
            )
        if error_id in AUTH_ERROR_IDS:
            hint = (
                " -- a stackapps.com key is required for this call"
                if error_id == 405
                else ""
            )
            raise AuthError(
                f"{detail}{hint}", source=self.name, status_code=raw.status_code
            )
        if error_id in QUERY_ERROR_IDS:
            raise QueryError(
                detail, source=self.name, status_code=raw.status_code
            )
        raise UpstreamError(detail, source=self.name, status_code=raw.status_code)

    # -- fetch -----------------------------------------------------------

    async def fetch(self, query: str, k: int) -> RawResponse:
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")

        terms, site = self._split_query(query)
        params = {
            "order": "desc",
            "sort": "relevance",
            "q": terms,
            "site": site,
            "pagesize": min(int(k), 100),
            "filter": BODY_FILTER,
        }
        if self.api_key:
            params["key"] = self.api_key
        raw = await self._get(f"{API_URL}?{urlencode(params)}")
        self._record_quota_and_backoff(raw)
        return raw

    def _split_query(self, query: str) -> Tuple[str, str]:
        """``"inflation@economics"`` -> ``("inflation", "economics")``.

        The site is parsed out of the query rather than taken as a keyword
        argument so it is inside the string the Part 18 cache hashes.
        """
        # The separator flag, not the emptiness of ``terms``, decides which
        # case this is: rpartition returns terms="" BOTH when there is no
        # separator and when one leads the string, so testing ``terms`` alone
        # accepts "@stackoverflow" as a search for the literal text
        # "@stackoverflow" instead of rejecting it.
        terms, sep, suffix = query.strip().rpartition(SITE_SEP)
        if not sep:
            terms, site = query.strip(), self.default_site
        else:
            site = suffix.strip() or self.default_site
        terms = terms.strip()
        if not terms:
            raise ValueError(f"query has no search terms before {SITE_SEP!r}")
        return terms, site

    def _record_quota_and_backoff(self, raw: RawResponse) -> None:
        """Read the two throttling fields off a successful response body."""
        try:
            body = json.loads(raw.body_text)
        except (TypeError, ValueError):
            return  # parse() will raise on this; nothing to record
        if not isinstance(body, dict):
            return

        quota = body.get("quota_remaining")
        if isinstance(quota, int):
            self.quota_remaining = quota
            if quota <= QUOTA_WARN_THRESHOLD:
                logger.warning(
                    "stackexchange: only %d of %s daily requests remain; a "
                    "stackapps.com key raises the ceiling to 10000",
                    quota,
                    body.get("quota_max", "?"),
                )
            else:
                logger.info("stackexchange: quota_remaining=%d", quota)

        backoff = body.get("backoff")
        if isinstance(backoff, (int, float)) and backoff > 0:
            self._backoff_until = time.monotonic() + float(backoff)
            logger.warning(
                "stackexchange: API returned backoff=%ss on a 200; next "
                "request deferred accordingly",
                backoff,
            )

    # -- parse -----------------------------------------------------------

    def parse(self, raw: RawResponse) -> List[Record]:
        try:
            body = json.loads(raw.body_text)
        except (TypeError, ValueError) as exc:
            raise ParseError(
                f"response body is not valid JSON: {exc}", source=self.name
            ) from exc
        if not isinstance(body, dict) or "items" not in body:
            raise ParseError(
                "response body has no 'items' key", source=self.name
            )

        fetched_at = utcnow()
        records: List[Record] = []
        for item in body.get("items") or []:
            if not isinstance(item, dict):
                raise ParseError(
                    f"item is not an object: {item!r}", source=self.name
                )
            question_id = item.get("question_id")
            if question_id is None:
                raise ParseError("item has no question_id", source=self.name)

            title = _clean(item.get("title") or "")
            snippet = _compose_snippet(item, title)
            _assert_snippet_bounded(snippet)
            owner = item.get("owner") or {}
            records.append(
                Record(
                    id=str(question_id),
                    title=title,
                    url=str(item.get("link") or ""),
                    snippet=snippet,
                    source_name=self.name,
                    fetched_at=fetched_at,
                    extra={
                        "author": owner.get("display_name") or None,
                        "published_date": epoch_to_utc(item.get("creation_date")),
                        "last_activity": epoch_to_utc(
                            item.get("last_activity_date")
                        ),
                        "score": item.get("score"),
                        "answer_count": item.get("answer_count"),
                        "is_answered": item.get("is_answered"),
                        "view_count": item.get("view_count"),
                        "tags": list(item.get("tags") or []),
                    },
                )
            )
        return records

    # -- normalize -------------------------------------------------------

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


def epoch_to_utc(value: Any) -> Optional[datetime]:
    """Unix epoch seconds -> tz-aware UTC datetime.

    ``tz=timezone.utc`` is explicit and load-bearing: ``fromtimestamp`` without
    it applies the LOCAL zone, which on this machine is +05:30, and would put
    every StackExchange post five and a half hours earlier than an arXiv paper
    published at the same instant. See defect-class note 3.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError, OverflowError):
        logger.debug("unparseable StackExchange epoch: %r", value)
        return None


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    """Strip HTML tags, unescape entities, collapse whitespace."""
    return _WS_RE.sub(" ", html.unescape(_TAG_RE.sub(" ", text or ""))).strip()


def _sentiment_prefix(item: Dict[str, Any]) -> str:
    """``[score +42 | 7 answers | accepted | 51k views]``. Fixed shape, capped.

    The score is the sentiment measurement this angle exists to read, and
    NormalizedRecord has no field for it -- see the module docstring.
    """
    bits: List[str] = []
    score = item.get("score")
    if isinstance(score, int):
        bits.append(f"score {score:+d}")
    answers = item.get("answer_count")
    if isinstance(answers, int):
        bits.append(f"{answers} answer" + ("s" if answers != 1 else ""))
    if item.get("is_answered"):
        bits.append("accepted")
    views = item.get("view_count")
    if isinstance(views, int):
        bits.append(f"{_short_count(views)} views")
    return f"[{' | '.join(bits)}] " if bits else ""


def _short_count(value: int) -> str:
    for limit, suffix in ((1_000_000, "M"), (1_000, "k")):
        if abs(value) >= limit:
            return f"{value / limit:.0f}{suffix}"
    return str(value)


def _compose_snippet(item: Dict[str, Any], title: str) -> str:
    """Bounded sentiment prefix + stripped body, truncated. See note 4."""
    prefix = _sentiment_prefix(item)
    text = _clean(item.get("body") or "") or title
    room = max(0, SNIPPET_CHARS - len(prefix))
    return (prefix + text[:room]).strip()


def _assert_snippet_bounded(snippet: str) -> None:
    """Guard the note-4 invariant where the cause is one function away."""
    if len(snippet) > SNIPPET_CHARS:
        raise ParseError(
            f"generated snippet is {len(snippet)} chars, over the "
            f"{SNIPPET_CHARS} cap; _compose_snippet gained an unbounded field",
            source="stackexchange",
        )


def _error_envelope(body_text: str) -> Tuple[Optional[int], str, str]:
    """``{"error_id": 502, ...}`` -> ``(502, name, message)``, or (None, '', '')."""
    try:
        body = json.loads(body_text)
    except (TypeError, ValueError):
        return None, "", ""
    if not isinstance(body, dict):
        return None, "", ""
    error_id = body.get("error_id")
    if not isinstance(error_id, int):
        return None, "", ""
    return error_id, str(body.get("error_name") or ""), str(
        body.get("error_message") or ""
    )


def _retry_after_seconds_from(value: str) -> Optional[float]:
    match = re.search(r"(\d+)\s*second", value or "", re.I)
    return float(match.group(1)) if match else None


def _retry_after_from_message(message: str) -> Optional[float]:
    """StackExchange puts the wait in prose, not a Retry-After header."""
    return _retry_after_seconds_from(message)


__all__ = [
    "StackExchangeConnector",
    "API_URL",
    "DEFAULT_RATE_PER_MIN",
    "DEFAULT_SITE",
    "SITE_SEP",
    "SNIPPET_CHARS",
    "QUOTA_WARN_THRESHOLD",
    "epoch_to_utc",
]
