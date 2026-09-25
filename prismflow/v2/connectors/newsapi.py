"""NewsAPI connector -- the Market angle's coverage evidence.

Part 17 interface. Extends ``base.py``; that file is unchanged.

The Market angle's brief names NewsAPI primary, Google Trends secondary and
CrunchBase tertiary. Only the primary is built here. The angle's own file
already carries the warning that NewsAPI and CrunchBase both aggregate press
releases, so a fallback from one to the other buys far less independence than
the two names suggest -- Part 21 measures that rather than assuming it.

``/v2/everything`` is the endpoint, not ``/v2/top-headlines``: this angle is
queried with a topic, and top-headlines is a country/category feed whose ``q``
is a filter over a small front page rather than a search over the corpus.

WHAT GOES IN ``snippet`` -- A SMALLER JUDGMENT CALL THAN YFINANCE'S
------------------------------------------------------------------
NewsAPI returns prose, so this is closer to arXiv than to yfinance: the text
exists and needs cleaning, not inventing. Three decisions were still made.

1. ``description`` AND ``content`` ARE BOTH USED, in that order. Measured over
   98 live articles: description median 212 chars, content median 214, and
   they overlap but are not identical -- content is the article's opening body
   text, description is the editorial summary. Using description alone throws
   away half the available signal in a 1500-token budget that can afford both.

2. THE OUTLET IS PREPENDED as a bounded, fixed-shape prefix:

       [The Verge] Electric air taxis get the green light ...

   For a market angle, WHO is covering a story is part of the evidence, not
   metadata: one outlet is a mention, twelve outlets is coverage. Part 19
   reranks on ``snippet`` and Part 20 reads ``snippet``, so an outlet that
   lives only in ``extra`` is invisible to both. ``source.name`` is used rather
   than ``source.id`` because ``id`` is null for 69 of 98 measured articles.

3. ``author`` FALLS BACK TO THE OUTLET. The journalist is the better value for
   NormalizedRecord.author, but it is null or empty in 7 of 98 articles;
   an outlet is a truer answer than None for a record that plainly has a
   publisher.

THE FOUR DEFECT-CLASS CHECKS
----------------------------
1. CACHE-KEY DISTINCTNESS (Part 18 class) -- CLEAN AT THE TOP LEVEL, FOUND ONE
   LEVEL DOWN, and there is a third case here that the other connectors do not
   have.

   ``name="newsapi"`` is distinct from github/arxiv/yfinance/stackexchange, so
   the cross-connector collision does not arise. The name is also not free:
   ``CONNECTOR_TTL_SECONDS`` in Part 18 already carries ``"newsapi": 900``, so
   this string and no other picks up the intended TTL.

   One level down, the LANGUAGE is part of what was asked and is not in the
   Part 18 key ``(connector_name, sha256(query)[:16])``. ``"inflation"`` in
   English and in German return different corpora entirely; as a constructor
   argument the language would be invisible to the key and the second caller
   would be served the first's answer for 900 seconds. It is therefore encoded
   in the query string (``"inflation@de"``), exactly as the period is for
   yfinance and the site for StackExchange.

   The third case is this source's own: ``sortBy`` ALSO changes the result set
   and is deliberately NOT a per-call knob. It is pinned to ``SORT_BY`` for the
   same reason -- a per-call sort that is not in the query string is a silent
   cache collision, and encoding two knobs in one suffix makes the query
   unreadable. Pinned and stated beats configurable and invisible. Part 19
   reranks locally anyway, so the source's ordering is an input to reranking
   rather than the final answer.

2. RATE LIMITING -- READ OFF THE PUBLISHED PLAN, THEN MEASURED, AND THE BASE
   CLASS IS WRONG FOR THIS SOURCE IN A WAY THAT COSTS A WHOLE DAY.

   THE FREE-TIER CAP IS A DAILY ONE. newsapi.org/pricing documents the
   Developer plan as "100 requests per day" with "No extra requests
   available" -- not a per-minute or per-second ceiling. It also restricts the
   plan to development use, enables CORS for localhost only, and limits the
   archive to "articles up to a month old".

   THERE IS NOTHING TO READ. Measured on a live 200: the only non-standard
   response header is ``X-Cached-Result``. There is no ``X-RateLimit-Remaining``
   and no quota field in the body. This is the opposite of StackExchange, which
   reports ``quota_remaining`` on every response. So the connector cannot know
   how much of the day it has left, and ``DAILY_REQUEST_CAP`` exists to warn
   from its OWN count (``request_count``, inherited) rather than pretending the
   source told us. That count is per process and resets on restart, which is
   stated here because it is a real limitation, not a bug to be surprised by.

   THE BASE-CLASS DEFECT. ``AngleConnector._check_status`` maps 429 to
   ``RateLimitError``, which is retried with exponential backoff. NewsAPI uses
   429 for two different conditions (newsapi.org/docs/errors):

       rateLimited        "You have been rate limited. Back off for a while"
       apiKeyExhausted    "Your API key has no more requests available"

   Backing off and retrying is exactly right for the first and exactly wrong
   for the second: a Developer key that has spent its 100 requests does not
   recover in 2, 4 or 8 seconds -- it recovers tomorrow. Retried, the
   connector would burn its backoff budget and then report a rate limit, when
   what the operator needs to hear is "the day's quota is gone". ``_check_status``
   is overridden to read the JSON ``code`` field and raise ``AuthError`` for
   ``apiKeyExhausted``, which the base class does not retry.

   INVALID PARAMETERS ARE SILENTLY IGNORED -- MEASURED, AND THIS IS THE WORST
   FAILURE SHAPE OF THE THREE CONNECTORS SO FAR. ``sortBy=definitely-not-a-sort``
   returns HTTP 200 with a normal result set, and ``language=zz`` returns
   HTTP 200 ``{"status":"ok","totalResults":0,"articles":[]}``. A typo'd
   language therefore produces an empty result that is indistinguishable from
   "there is no news on this topic" -- the angle would report zero evidence
   and nothing anywhere would say why. Both are validated client-side against
   the documented value lists before the request is built, because the API will
   not do it for us.

3. TIMEZONE -- THE PARSE IS THE EASY HALF; THE STALENESS WINDOW IS THE FINDING.

   ``publishedAt`` is ISO-8601 with a literal ``Z`` (measured:
   ``"2026-09-24T03:34:00Z"``), so it is unambiguous once ``Z`` is translated
   into ``+00:00`` -- ``datetime.fromisoformat`` on Python 3.10 and earlier
   rejects the ``Z`` form outright rather than mis-reading it. That matches
   arXiv and differs from yfinance (exchange-local) and StackExchange (naive
   epoch): four sources, four conventions.

   The real hazard is the ENDPOINT-SPECIFIC STALENESS WINDOW. The Developer
   plan documents a 24-hour article delay, and it is exactly true: over 98
   articles sorted by ``publishedAt`` descending, the NEWEST was 24.09 hours
   old, the oldest 28.60, median 26.94. So every Market record is at least a
   day behind every Tech, Financial and Sentiment record retrieved in the same
   run, and Part 21 compares dates across angles. That is a property of the
   plan, not a parsing bug, so it is surfaced rather than corrected.

   THE DELAY IS A FLOOR, NOT THE AGE OF WHAT COMES BACK -- and conflating the
   two produced a defect in the first draft of this file, caught by the smoke
   run. Because ``sortBy`` is pinned to ``relevancy`` (note 1), the returned
   page is the most RELEVANT articles, not the most recent: across four live
   smoke queries the freshest record on the page was 38.8h, 69.1h, 165.1h and
   351.5h old, every one of them a normal result for a topic whose best
   coverage is not from yesterday. A warning keyed to "older than the 24h
   delay" therefore fired on every single query and meant nothing.

   So ``staleness_hours`` is REPORTED, not judged, and the only staleness
   warning left is the one that is a genuine contract violation: the plan
   serves "articles up to a month old", so anything past
   ``ARCHIVE_LIMIT_HOURS`` means the plan or the endpoint changed under us.

4. OVERSIZE (Part 19 class) -- THE SOURCE ALREADY TRUNCATES, WHICH IS ITSELF
   THE FINDING, AND THE GUARD STAYS ANYWAY.

   Measured over 98 live articles: title 19-159 chars, description 11-260,
   content 57-215, and description+content never exceeded 475. Nothing
   NewsAPI returns on this plan can breach the Market angle's 1500-token
   budget. That is the opposite of StackExchange, whose worst real post was
   2783 tokens stripped.

   Two things follow, and neither is "no guard needed".

   a. ``content`` IS PRE-TRUNCATED AND SAYS SO IN BAND. 93 of 98 values end in
      a marker like ``"... bought [+3123 chars]"``. That marker is not article
      text: BM25 would score on the token "chars", and Part 20's reasoner would
      read a truncation artifact as content. It is stripped, and the character
      count it carries is kept in ``extra`` as ``content_truncated_chars``,
      where it is a fact about the article rather than words in the snippet.

   b. THE 200-CHAR CAP IS OBSERVED BEHAVIOUR, NOT A CONTRACT. It appears
      nowhere in the documentation as a guarantee and does not apply to paid
      plans. A guard that exists only because today's responses happen to be
      small is a guard that fails the first time the plan changes, so the
      snippet is truncated to ``SNIPPET_CHARS`` and ``_assert_snippet_bounded``
      pins the invariant at construction. The test for it runs against a
      fixture whose lengths are synthetic, and the test suite says so.

   U+FFFD IS ON THE WIRE. One of the 98 measured articles contains a literal
   replacement character in its text. Verified as the source's own damage and
   not a decoding artifact of ours: ``Content-Type`` is
   ``application/json; charset=utf-8``, the body decodes as strict UTF-8
   without error, and the U+FFFD survives that strict decode -- so NewsAPI's
   ingestion mangled the text upstream and is faithfully serving the mangled
   form. It is stripped in cleaning, because a replacement character is noise
   in a BM25 index and noise in a reasoner prompt.

5. CREDENTIAL HYGIENE -- NOT ONE OF THE FOUR, FOUND BY THE SMOKE RUN, AND THE
   REASON THE KEY TRAVELS IN A HEADER.

   NewsAPI accepts the key either as an ``apiKey`` QUERY PARAMETER or as an
   ``X-Api-Key`` header. The first draft used the query parameter, which the
   documentation presents first. ``AngleConnector._get`` logs every request as
   ``logger.info("%s: GET %s", name, url)``, and every backoff at WARNING with
   the same url -- so a transient DNS failure during the smoke run printed the
   live API key three times, in full, at WARNING level. Logs go to files, get
   pasted into issues, and are the one place a credential should never be.

   The header form fixes it from this side without touching Part 17: the url the
   base class logs no longer contains the secret. This is a property of THIS
   connector rather than of the logging -- see the frozen-module note below,
   because StackExchange passes its ``key`` exactly as this draft did.

A NOTE ON A FROZEN MODULE, REPORTED AND NOT TOUCHED
---------------------------------------------------
``CONNECTOR_TTL_SECONDS["newsapi"] = 900`` (Part 18) was chosen on the
reasoning that "headlines turn over within the hour". Against the measured
24-hour delay, a 900-second TTL re-fetches up to four times an hour for data
that cannot change more than once a day on this plan -- roughly 96 requests a
day against a 100-request ceiling, from cache misses alone. That is a defect in
a previously-passing module. Per the FROZEN MODULES rule it is REPORTED here
and NOT changed: Part 18 is untouched by this commit.

A SECOND ONE, SAME RULE. ``AngleConnector._get`` (Part 17) logs the full request
url at INFO on every call and at WARNING on every backoff, so any connector that
carries a credential in its query string writes that credential to the log. This
connector avoids it with a header (note 5), but ``StackExchangeConnector`` puts
``key`` in the query string -- harmless today only because that key is None and a
stackapps key is not yet needed. If Part 24 adds one, it will be logged.
REPORTED, not fixed: ``base.py`` and ``stackexchange.py`` are untouched by this
commit.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
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

logger = logging.getLogger("prismflow.v2.connectors.newsapi")

API_URL = "https://newsapi.org/v2/everything"

#: The free Developer plan's ceiling is DAILY (100/day, documented at
#: newsapi.org/pricing), so a per-minute figure is only burst spacing. 20/min
#: is deliberately slower than StackExchange's 30: the daily budget here is a
#: third of the size, and nothing in the response tells us how much is left.
DEFAULT_RATE_PER_MIN = 20

#: Documented free-tier ceiling. Enforced by the API; counted here only so the
#: connector can warn from its own tally -- see defect-class note 2.
DAILY_REQUEST_CAP = 100

#: Warn once the process has spent this many of its own requests.
REQUEST_WARN_THRESHOLD = 80

#: Documented article delay on the Developer plan, and measured at 24.09h for
#: the freshest of 98 articles under ``sortBy=publishedAt``. A FLOOR on the age
#: of any record this connector can return. See defect-class note 3.
ARTICLE_DELAY_HOURS = 24.0

#: The Developer plan serves "articles up to a month old" (newsapi.org/pricing),
#: so nothing older than this should ever come back. Unlike the delay, this is a
#: CEILING, and it is the only staleness figure worth warning about -- see
#: defect-class note 3 for why the age of a returned page is not.
ARCHIVE_LIMIT_HOURS = 24.0 * 31

#: Default search language. Overridable per query via ``"query@de"``.
DEFAULT_LANGUAGE = "en"

#: Separator between the search terms and the language in a query string.
#: Present so the language lands inside the Part 18 cache key -- see note 1.
LANG_SEP = "@"

#: The documented ``language`` values. Validated client-side because an
#: unknown code returns HTTP 200 with zero results rather than an error --
#: see defect-class note 2.
VALID_LANGUAGES = frozenset(
    {"ar", "de", "en", "es", "fr", "he", "it", "nl", "no", "pt", "ru", "sv",
     "ud", "zh"}
)

#: Pinned rather than exposed per call: a per-call sort would not be in the
#: query string and so would be invisible to the Part 18 key. See note 1.
#: ``relevancy`` because the angle is queried with a topic and Part 19 reranks
#: the result locally regardless.
SORT_BY = "relevancy"

#: Documented maximum for ``pageSize`` on /v2/everything.
MAX_PAGE_SIZE = 100

#: NewsAPI accepts the key either as an ``apiKey`` query parameter or in this
#: header. The header is used, and note 5 explains why that is not cosmetic.
API_KEY_HEADER = "X-Api-Key"

#: Truncation cap on the composed snippet. The measured worst real case is 475
#: chars of description+content, so this is not binding today -- it binds if
#: the plan's undocumented ~200-char content cap ever lifts. See note 4b.
SNIPPET_CHARS = 600

#: NewsAPI error ``code`` strings, from newsapi.org/docs/errors.
EXHAUSTED_CODE = "apiKeyExhausted"
THROTTLE_CODES = frozenset({"rateLimited"})
AUTH_CODES = frozenset(
    {"apiKeyExhausted", "apiKeyDisabled", "apiKeyInvalid", "apiKeyMissing"}
)
QUERY_CODES = frozenset(
    {"parameterInvalid", "parametersMissing", "sourcesTooMany",
     "sourceDoesNotExist"}
)

#: ``.env.template`` ships ``NEWSAPI_KEY=...``; the same placeholder shapes
#: github.py rejects are rejected here, for the same reason -- an unfilled
#: template value produces a confusing 401 rather than an obvious "no key".
_PLACEHOLDER_MARKERS = ("...", "<", "your_", "xxx", "changeme")


class NewsAPIConnector(AngleConnector):
    """Search news coverage via NewsAPI's ``/v2/everything``.

    A key is required -- unlike arXiv, StackExchange or yfinance, this source
    has no anonymous mode. It is resolved from ``api_key``, then
    ``NEWSAPI_KEY`` in the environment, then ``NEWSAPI_KEY`` in the repo's
    ``.env``; see ``_resolve_key`` for why the file is consulted at all.

    Construction does NOT require a key, so ``parse`` and ``normalize`` stay
    testable without credentials. ``fetch`` raises ``AuthError`` when there is
    none.
    """

    def __init__(
        self,
        rate_limit_per_min: int = DEFAULT_RATE_PER_MIN,
        *,
        http_client: Optional[HTTPClient] = None,
        max_retries: int = 3,
        default_language: str = DEFAULT_LANGUAGE,
        api_key: Optional[str] = None,
        page_size_cap: int = MAX_PAGE_SIZE,
    ) -> None:
        super().__init__(
            "newsapi",
            rate_limit_per_min,
            http_client=http_client,
            max_retries=max_retries,
        )
        language = (default_language or "").strip().lower()
        if language not in VALID_LANGUAGES:
            raise ValueError(
                f"default_language {default_language!r} is not one of "
                f"{sorted(VALID_LANGUAGES)}"
            )
        self.default_language = language
        self.api_key = _resolve_key(api_key)
        self.page_size_cap = max(1, min(int(page_size_cap), MAX_PAGE_SIZE))
        #: Hours between the freshest record of the last fetch and now. None
        #: until a fetch has parsed at least one dated article.
        self.staleness_hours: Optional[float] = None
        #: ``totalResults`` from the last successful response, for the smoke
        #: run -- the corpus size is not the page size.
        self.total_results: Optional[int] = None

    # -- status mapping --------------------------------------------------

    def _check_status(self, raw: RawResponse) -> None:
        """Map NewsAPI's ``code`` envelope onto the Part 17 hierarchy.

        Overridden because this source sends 429 for two conditions that need
        opposite handling -- ``rateLimited`` should be retried and
        ``apiKeyExhausted`` must not be. See defect-class note 2.
        """
        if 200 <= raw.status_code < 300:
            return

        code, message = _error_envelope(raw.body_text)
        if not code:
            super()._check_status(raw)
            return

        detail = f"{code}: {message or 'no message'}"
        if code == EXHAUSTED_CODE:
            raise AuthError(
                f"{detail} -- the daily free-tier ceiling of "
                f"{DAILY_REQUEST_CAP} requests is spent; this does not clear "
                f"on backoff",
                source=self.name,
                status_code=raw.status_code,
            )
        if code in THROTTLE_CODES:
            raise RateLimitError(
                detail, source=self.name, status_code=raw.status_code
            )
        if code in AUTH_CODES:
            raise AuthError(
                detail, source=self.name, status_code=raw.status_code
            )
        if code in QUERY_CODES:
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
        if not self.api_key:
            raise AuthError(
                "NEWSAPI_KEY is not set; NewsAPI has no anonymous mode",
                source=self.name,
            )

        terms, language = self._split_query(query)
        params = {
            "q": terms,
            "language": language,
            "sortBy": SORT_BY,
            "pageSize": min(int(k), self.page_size_cap),
        }
        self._warn_on_own_request_count()
        # The key goes in a HEADER, not in ``params`` -- see note 5.
        raw = await self._get(
            f"{API_URL}?{urlencode(params)}",
            headers={API_KEY_HEADER: self.api_key},
        )
        self._record_corpus_size(raw)
        return raw

    def _split_query(self, query: str) -> Tuple[str, str]:
        """``"inflation@de"`` -> ``("inflation", "de")``.

        The language is parsed out of the query rather than taken as a keyword
        argument so it is inside the string the Part 18 cache hashes.
        """
        # The separator flag decides which case this is, not the emptiness of
        # ``terms``: rpartition returns terms="" BOTH when no separator is
        # present and when one leads the string, so testing ``terms`` alone
        # would accept "@de" as a search for the literal text "@de".
        terms, sep, suffix = query.strip().rpartition(LANG_SEP)
        if not sep:
            terms, language = query.strip(), self.default_language
        else:
            language = suffix.strip().lower() or self.default_language
        terms = terms.strip()
        if not terms:
            raise ValueError(f"query has no search terms before {LANG_SEP!r}")
        if language not in VALID_LANGUAGES:
            # Raised rather than passed through: NewsAPI answers an unknown
            # language with HTTP 200 and zero results, which the angle would
            # report as "no coverage". See defect-class note 2.
            raise QueryError(
                f"language {language!r} is not one of "
                f"{sorted(VALID_LANGUAGES)}; NewsAPI would answer 200 with "
                f"zero results rather than reject it",
                source=self.name,
            )
        return terms, language

    def _warn_on_own_request_count(self) -> None:
        """Warn from our own tally, since the response carries no quota field.

        ``request_count`` is per process and resets on restart, so this is a
        floor on spend rather than a true remaining-quota reading. Stated
        plainly because a warning that looks authoritative and is not would be
        worse than none -- see defect-class note 2.
        """
        if self.request_count >= REQUEST_WARN_THRESHOLD:
            logger.warning(
                "newsapi: this process has made %d requests; the free-tier "
                "ceiling is %d per day and NewsAPI reports no remaining "
                "quota, so this count is a floor on today's spend, not a "
                "reading",
                self.request_count,
                DAILY_REQUEST_CAP,
            )

    def _record_corpus_size(self, raw: RawResponse) -> None:
        """Read ``totalResults`` off a successful body. Best effort."""
        try:
            body = json.loads(raw.body_text)
        except (TypeError, ValueError):
            return  # parse() will raise on this; nothing to record
        if isinstance(body, dict) and isinstance(body.get("totalResults"), int):
            self.total_results = body["totalResults"]

    # -- parse -----------------------------------------------------------

    def parse(self, raw: RawResponse) -> List[Record]:
        try:
            body = json.loads(raw.body_text)
        except (TypeError, ValueError) as exc:
            raise ParseError(
                f"response body is not valid JSON: {exc}", source=self.name
            ) from exc
        if not isinstance(body, dict) or "articles" not in body:
            raise ParseError(
                "response body has no 'articles' key", source=self.name
            )
        if body.get("status") == "error":
            # A 200 carrying an error envelope: _check_status never saw it.
            code, message = _error_envelope(raw.body_text)
            raise ParseError(
                f"response reports status=error ({code}: {message})",
                source=self.name,
            )

        fetched_at = utcnow()
        records: List[Record] = []
        for article in body.get("articles") or []:
            if not isinstance(article, dict):
                raise ParseError(
                    f"article is not an object: {article!r}", source=self.name
                )
            url = article.get("url")
            if not url:
                # The url IS the identity: NewsAPI gives articles no id field.
                raise ParseError("article has no url", source=self.name)

            source = article.get("source") or {}
            outlet = _clean(source.get("name") or "") or None
            title = _clean(article.get("title") or "")
            body_text, truncated_chars = _strip_truncation_marker(
                article.get("content") or ""
            )
            snippet = _compose_snippet(
                outlet, _clean(article.get("description") or ""),
                _clean(body_text), title,
            )
            _assert_snippet_bounded(snippet)
            records.append(
                Record(
                    id=str(url),
                    title=title,
                    url=str(url),
                    snippet=snippet,
                    source_name=self.name,
                    fetched_at=fetched_at,
                    extra={
                        # The journalist where there is one, else the outlet --
                        # a record with a publisher is not authorless.
                        "author": _clean(article.get("author") or "") or outlet,
                        "published_date": parse_published_at(
                            article.get("publishedAt")
                        ),
                        "outlet": outlet,
                        "outlet_id": source.get("id"),
                        "description": _clean(article.get("description") or ""),
                        "content_truncated_chars": truncated_chars,
                        "url_to_image": article.get("urlToImage"),
                    },
                )
            )
        self._record_staleness(records)
        return records

    def _record_staleness(self, records: Sequence[Record]) -> None:
        """Record the age of the freshest record on this page.

        Reported, not warned about, except past the plan's archive ceiling.
        The age of a ``relevancy``-sorted page is driven by WHICH articles are
        relevant, not by the plan's delay -- measured across four live smoke
        queries the freshest record was 38.8h, 69.1h, 165.1h and 351.5h old,
        all of them perfectly normal. Only the archive ceiling is a real
        invariant. See defect-class note 3.
        """
        dates = [
            r.extra.get("published_date")
            for r in records
            if r.extra.get("published_date") is not None
        ]
        if not dates:
            self.staleness_hours = None
            return
        newest = max(dates)
        self.staleness_hours = (utcnow() - newest).total_seconds() / 3600.0
        if self.staleness_hours > ARCHIVE_LIMIT_HOURS:
            logger.warning(
                "newsapi: freshest article is %.1fh old, past the plan's "
                "~%.0fh archive ceiling -- the plan or the endpoint's window "
                "has changed",
                self.staleness_hours,
                ARCHIVE_LIMIT_HOURS,
            )
        else:
            logger.info(
                "newsapi: freshest article on this page is %.1fh old "
                "(delay floor %.0fh, archive ceiling %.0fh)",
                self.staleness_hours,
                ARTICLE_DELAY_HOURS,
                ARCHIVE_LIMIT_HOURS,
            )

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


def _resolve_key(explicit: Optional[str]) -> Optional[str]:
    """Resolve the API key, rejecting unfilled template placeholders.

    An EXPLICIT argument is an instruction and is never fallen back from: if
    it is a placeholder, the answer is None, not some other credential. The
    alternative bit during development -- a test passing a deliberately fake
    key was handed the real one out of ``.env`` and would have made a live
    call while appearing to be offline.

    With no explicit argument the order is ``NEWSAPI_KEY`` in the environment,
    then ``NEWSAPI_KEY`` in the repo's ``.env``. The file is consulted because
    nothing in this project loads ``.env`` into the environment --
    ``.env.template`` ships ``NEWSAPI_KEY`` as the documented home for this
    credential, so a connector that read only ``os.environ`` would look broken
    to anyone who followed the setup instructions and filled the file in.
    """
    if explicit is not None:
        return _usable(explicit)
    for candidate in (os.getenv("NEWSAPI_KEY"), _key_from_dotenv()):
        usable = _usable(candidate)
        if usable is not None:
            return usable
    return None


def _usable(candidate: Optional[str]) -> Optional[str]:
    """``candidate`` if it is a real key, else None with a warning."""
    if candidate is None:
        return None
    value = candidate.strip()
    if not value:
        return None
    if any(marker in value.lower() for marker in _PLACEHOLDER_MARKERS):
        logger.warning(
            "NEWSAPI_KEY looks like an unfilled template placeholder "
            "(%d chars); ignoring it. NewsAPI has no anonymous mode, so "
            "fetch will raise AuthError until a real key is set.",
            len(value),
        )
        return None
    return value


def _key_from_dotenv() -> Optional[str]:
    """``NEWSAPI_KEY`` from the repo-root ``.env``, or None.

    Deliberately minimal: no dependency on python-dotenv, no mutation of
    ``os.environ``, and any failure is a None rather than an exception --
    a missing or unreadable ``.env`` is an ordinary state, not an error.
    """
    env_path = Path(__file__).resolve().parents[3] / ".env"
    try:
        text = env_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("NEWSAPI_KEY="):
            return line.partition("=")[2].strip().strip('"').strip("'")
    return None


def parse_published_at(value: Any) -> Optional[datetime]:
    """``"2026-09-24T03:34:00Z"`` -> tz-aware UTC datetime.

    The ``Z`` is translated to ``+00:00`` before parsing:
    ``datetime.fromisoformat`` only learned to accept ``Z`` in Python 3.11, so
    on 3.10 and earlier the documented NewsAPI format raises rather than
    parsing. A value that parses to a naive datetime is stamped UTC, since
    NewsAPI documents ``publishedAt`` as UTC -- naive would otherwise leak
    into Part 21's cross-angle date comparisons. See defect-class note 3.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        logger.debug("unparseable NewsAPI publishedAt: %r", value)
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


#: ``"... investors bought [+3123 chars]"`` -- NewsAPI's own in-band marker
#: that ``content`` is truncated. See defect-class note 4a.
_TRUNCATION_RE = re.compile(r"\s*\[\+\s*(\d+)\s*chars?\s*\]\s*$", re.I)

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _strip_truncation_marker(content: str) -> Tuple[str, Optional[int]]:
    """Split NewsAPI's ``[+N chars]`` marker off the end of ``content``.

    Returns the text without the marker and the N it carried. The count is a
    fact about the article and belongs in ``extra``; the words "chars" do not
    belong in a BM25 index. See defect-class note 4a.
    """
    match = _TRUNCATION_RE.search(content or "")
    if not match:
        return content or "", None
    return content[: match.start()].rstrip(), int(match.group(1))


def _clean(text: str) -> str:
    """Strip tags, drop U+FFFD, collapse whitespace.

    U+FFFD arrives on the wire from NewsAPI's own ingestion -- verified as the
    source's damage, not ours, since the body decodes as strict UTF-8 with the
    replacement character intact. See defect-class note 4.
    """
    cleaned = _TAG_RE.sub(" ", text or "").replace("�", "")
    return _WS_RE.sub(" ", cleaned).strip()


def _compose_snippet(
    outlet: Optional[str],
    description: str,
    content: str,
    title: str,
) -> str:
    """``[Outlet] description content``, bounded. See notes 2 and 4.

    ``description`` leads because it is the editorial summary; ``content``
    follows because it is the article's own opening and the two only partly
    overlap. ``title`` is the fallback for an article with neither, so a
    record is never snippet-less and therefore never last under BM25 for a
    reason that has nothing to do with relevance.
    """
    prefix = f"[{outlet}] " if outlet else ""
    parts = [p for p in (description, content) if p]
    text = " ".join(parts) if parts else title
    room = max(0, SNIPPET_CHARS - len(prefix))
    return (prefix + text[:room]).strip()


def _assert_snippet_bounded(snippet: str) -> None:
    """Guard the note-4 invariant where the cause is one function away."""
    if len(snippet) > SNIPPET_CHARS:
        raise ParseError(
            f"generated snippet is {len(snippet)} chars, over the "
            f"{SNIPPET_CHARS} cap; _compose_snippet gained an unbounded field",
            source="newsapi",
        )


def _error_envelope(body_text: str) -> Tuple[Optional[str], str]:
    """``{"status":"error","code":"apiKeyExhausted",...}`` -> ``(code, message)``."""
    try:
        body = json.loads(body_text)
    except (TypeError, ValueError):
        return None, ""
    if not isinstance(body, dict):
        return None, ""
    code = body.get("code")
    if not isinstance(code, str) or not code:
        return None, ""
    return code, str(body.get("message") or "")


__all__ = [
    "NewsAPIConnector",
    "API_URL",
    "DEFAULT_RATE_PER_MIN",
    "DAILY_REQUEST_CAP",
    "ARTICLE_DELAY_HOURS",
    "ARCHIVE_LIMIT_HOURS",
    "DEFAULT_LANGUAGE",
    "LANG_SEP",
    "VALID_LANGUAGES",
    "SORT_BY",
    "SNIPPET_CHARS",
    "REQUEST_WARN_THRESHOLD",
    "API_KEY_HEADER",
    "parse_published_at",
]
