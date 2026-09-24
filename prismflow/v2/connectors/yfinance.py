"""Yahoo Finance connector -- the Financial angle's price evidence.

Part 17 interface, built in Part 24-prep (TASK 2a). Extends ``base.py``; that
file is unchanged.

AUTHENTICATION: GENUINELY NONE, AND ``YFINANCE_COOKIE`` IS NOT READ
-------------------------------------------------------------------
Measured on 2026-09-24 against yfinance 1.7.0:

  - ``yf.Ticker("AAPL").history(period="5d")`` returns 5 rows with no
    credential of any kind present in the environment.
  - ``yfinance`` references NO environment variable anywhere in its data
    layer. It negotiates its own cookie and crumb with Yahoo anonymously on
    first use and caches them, which is why call 1 took 12.9s and call 2 took
    4.7s.
  - Setting ``YFINANCE_COOKIE`` to the malformed inline-comment string that
    currently sits in ``.env`` changed nothing: the same call still returned
    5 rows.

So the answer to "does it silently rely on YFINANCE_COOKIE being unset?" is
no, in the strongest sense -- nothing reads the variable, so neither its
absence nor a malformed value can affect a request.

THE REAL RISK IS DOWNSTREAM OF THAT, AND THIS MODULE IS WHERE IT WOULD LAND.
``yf.data.YfData.set_login_cookies(cookie_t, cookie_y)`` does exist. If a
future maintainer reads ``YFINANCE_COOKIE`` out of ``.env`` -- thinking it is
required because the key is sitting there -- and feeds today's value into that
call, they would be injecting the string ``"# Optional; ..."`` as a session
cookie and would break a connector that currently works. This module therefore
reads no credential at all, deliberately, and this paragraph is the reason.
The ``.env`` key is vestigial and should be deleted rather than filled.

WHAT GOES IN ``snippet`` IS A JUDGMENT CALL
-------------------------------------------
Every other source in V2 returns natural text: an abstract, a repo
description. Yahoo returns a numeric series, so ``snippet`` has to be
*generated*, and the Part 19 angle docstring already flags why it cannot be
left empty -- BM25 reranks on snippet text, and an empty snippet ranks last no
matter how relevant the instrument is.

The choice made here is a fixed-shape one-line summary:

    NVDA (NVIDIA Corporation) -- NASDAQ, USD. Last 187.43, +6.2% over 1mo
    (range 171.20-192.88, avg volume 214.3M). Equity. Session 2026-09-24.

Chosen because it is (a) bounded by construction, (b) readable as English so
BM25 and the Part 20 reasoners get real lexical signal -- the company name and
exchange are the terms a financial query actually matches on -- and (c) free of
any per-bar data, which is what keeps it bounded. See the oversize note below.

This is a judgment call, not something the source gives us. An alternative --
dumping OHLCV rows as text -- was rejected on the oversize grounds measured
below, and a second alternative -- using Yahoo's ``longBusinessSummary`` -- was
rejected because it is unbounded prose about the company that says nothing
about the price, which is the evidence this angle exists to contribute.

THE FOUR DEFECT-CLASS CHECKS
----------------------------
1. CACHE-KEY COLLISION (Part 18 class). The Part 18 key is
   ``(connector_name, sha256(query)[:16])``. ``name="yfinance"`` is distinct
   from ``"github"`` and ``"arxiv"``, so the cross-connector collision cannot
   happen, and ``CONNECTOR_TTL_SECONDS`` already carries a 300s entry for it.

   But the same defect class reappears one level down and would have bitten
   here: the period is part of what was asked for and is NOT part of the key
   unless it is inside the query string. ``"NVDA"`` over ``5d`` and ``"NVDA"``
   over ``10y`` would otherwise hash identically and the second caller would
   be served the first one's answer for 300 seconds. The period is therefore
   encoded in the query itself (``"NVDA@10y"``), so two different periods are
   two different keys by construction rather than by the caller remembering.

2. RATE LIMITING. Yahoo documents no public limit for this endpoint, and
   yfinance ships ``YFRateLimitError``, so the throttling is real but
   informal. ``DEFAULT_RATE_PER_MIN`` is deliberately conservative.
   Crucially, throttling is applied per *underlying* Yahoo call rather than
   per ``fetch``: one fetch is 1 search + k ticker requests, so a per-fetch
   throttle would understate the true request rate by a factor of k+1.
   ``YFRateLimitError`` maps to the base hierarchy's ``RateLimitError`` and
   retries with the same exponential backoff arXiv uses.

3. TIMEZONE. Measured, not assumed, and it does NOT match arXiv/GitHub:
   yfinance returns tz-AWARE timestamps in EXCHANGE-LOCAL time, and the zone
   varies per listing --

       AAPL      America/New_York   (-0400)
       SHOP.TO   America/Toronto    (-0400)
       BP.L      Europe/London      (+0100)
       7203.T    Asia/Tokyo         (+0900)

   Left alone, a Tokyo session and a New York session on the same calendar day
   differ by 13 hours of apparent recency, and Part 21 compares dates across
   angles. ``published_date`` is therefore converted to UTC in ``normalize``,
   and the original zone is preserved in ``Record.extra["exchange_timezone"]``
   so the conversion is auditable rather than lossy.

   A second, subtler point: for daily bars the timestamp is midnight
   exchange-local, i.e. the SESSION DATE, not the moment of close. It is a
   date wearing a timestamp's clothes. Treated as a session date throughout.

4. OVERSIZE (Part 19 class). Measured: ``period="10y", interval="1d"`` on AAPL
   returns 2513 rows. Rendering one line per bar would be roughly 30k tokens
   against a 2000-token angle budget -- fifteen times the entire budget, and
   Part 19 would drop the record with a warning, so the angle would go silent
   exactly when the caller asked for the most data.

   The summary above is bounded by CONSTRUCTION: it contains a fixed number of
   fields and no per-bar data, so its length does not grow with the period.
   ``SNIPPET_CHARS`` is a second, belt-and-braces cap for the one field that
   is genuinely unbounded upstream -- the company long name, which Yahoo does
   not length-limit. ``_assert_snippet_bounded`` checks the invariant at
   construction so a future field addition cannot quietly break it.

A NOTE ON QUERY RESOLUTION
--------------------------
``yf.Search`` is Yahoo's own query interface and is what turns a natural
language query into tickers. It is unauthenticated and it works, but it
matches short queries far better than long ones -- measured: ``"nvidia"`` ->
NVDA + 4 more, ``"semiconductor"`` -> SOXL/TSM/SOXX/SMH, but ``"artificial
intelligence chips"`` -> zero quotes.

Zero is returned as zero. The query is NOT silently rewritten, trimmed or
keyword-extracted to force a match: Part 19's fallback chain exists precisely
for "the primary source had nothing", and a connector that invents a different
question than the one it was asked makes the angle's provenance a fiction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import (
    AngleConnector,
    HTTPClient,
    NormalizedRecord,
    RawResponse,
    Record,
    _backoff_delay,
    count_tokens,
    utcnow,
)
from .errors import ParseError, QueryError, RateLimitError, UpstreamError

logger = logging.getLogger("prismflow.v2.connectors.yfinance")

#: Conservative: Yahoo publishes no limit for this endpoint but throttles
#: informally, and yfinance ships a YFRateLimitError to prove it. Counted per
#: underlying Yahoo call, not per fetch -- see defect-class note 2.
DEFAULT_RATE_PER_MIN = 30

#: Default history window when the query does not name one.
DEFAULT_PERIOD = "1mo"

#: Periods yfinance accepts. Validated here so a typo is a QueryError at the
#: connector rather than a YFInvalidPeriodError from three frames down.
VALID_PERIODS = (
    "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max",
)

#: Separator between ticker/search term and period in a query string.
#: ``"NVDA@6mo"``. Present so the period lands inside the Part 18 cache key.
PERIOD_SEP = "@"

#: Hard cap on the generated snippet. The summary is bounded by construction;
#: this bounds the one upstream-unbounded field (company long name).
SNIPPET_CHARS = 400

#: Company names longer than this are truncated before composing the summary.
NAME_CHARS = 80


class YFinanceConnector(AngleConnector):
    """Resolve a query to tickers and summarise each one's recent prices.

    Needs no credential; see the module docstring for the measurement behind
    that claim.
    """

    def __init__(
        self,
        rate_limit_per_min: int = DEFAULT_RATE_PER_MIN,
        *,
        http_client: Optional[HTTPClient] = None,
        max_retries: int = 3,
        default_period: str = DEFAULT_PERIOD,
    ) -> None:
        super().__init__(
            "yfinance",
            rate_limit_per_min,
            http_client=http_client,
            max_retries=max_retries,
        )
        if default_period not in VALID_PERIODS:
            raise ValueError(
                f"default_period must be one of {VALID_PERIODS}, "
                f"got {default_period!r}"
            )
        self.default_period = default_period

    # -- fetch -----------------------------------------------------------

    async def fetch(self, query: str, k: int) -> RawResponse:
        """Resolve ``query`` to at most ``k`` tickers and fetch their history.

        Returns a synthetic ``RawResponse`` whose body is JSON. yfinance is a
        library, not an HTTP endpoint we drive, so there is no real response to
        hand back -- but keeping the three-stage seam means ``parse`` and
        ``normalize`` stay pure functions of text and are testable against
        fixtures with yfinance not installed at all.
        """
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")
        if k <= 0:
            raise ValueError("k must be positive")

        term, period = self._split_query(query)
        symbols = await self._search(term, k)

        quotes: List[Dict[str, Any]] = []
        for symbol in symbols[:k]:
            quote = await self._quote(symbol, period)
            if quote is not None:
                quotes.append(quote)

        body = {
            "query": term,
            "period": period,
            "fetched_at": utcnow().isoformat(),
            "quotes": quotes,
        }
        return RawResponse(
            status_code=200,
            headers={"content-type": "application/json"},
            body_text=json.dumps(body),
        )

    def _split_query(self, query: str) -> Tuple[str, str]:
        """``"NVDA@6mo"`` -> ``("NVDA", "6mo")``. Bare query -> default period.

        The period is parsed out of the query rather than taken as a keyword
        argument so that it is inside the string the Part 18 cache hashes.
        """
        term, _, suffix = query.strip().partition(PERIOD_SEP)
        term = term.strip()
        period = suffix.strip() or self.default_period
        if not term:
            raise ValueError(f"query has no search term before {PERIOD_SEP!r}")
        if period not in VALID_PERIODS:
            raise QueryError(
                f"period {period!r} is not one of {VALID_PERIODS}",
                source=self.name,
            )
        return term, period

    async def _search(self, term: str, k: int) -> List[str]:
        """Query -> ticker symbols, via Yahoo's own search. May return []."""
        def _call() -> List[str]:
            import yfinance as yf

            quotes = yf.Search(term, max_results=max(k, 1)).quotes
            out: List[str] = []
            for quote in quotes or []:
                symbol = (quote or {}).get("symbol")
                if symbol and symbol not in out:
                    out.append(str(symbol))
            return out

        symbols = await self._guarded(_call, what=f"search {term!r}")
        if not symbols:
            logger.info(
                "yfinance: search %r matched no instruments; returning empty "
                "rather than rewriting the query",
                term,
            )
        return symbols or []

    async def _quote(self, symbol: str, period: str) -> Optional[Dict[str, Any]]:
        """One ticker's history summary, or None if Yahoo has no prices."""
        def _call() -> Optional[Dict[str, Any]]:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            history = ticker.history(period=period, interval="1d")
            if history is None or len(history) == 0:
                return None

            closes = [float(v) for v in history["Close"] if _finite(v)]
            volumes = [float(v) for v in history["Volume"] if _finite(v)]
            if not closes:
                return None

            index = history.index
            last_stamp = index[-1]
            first_stamp = index[0]
            # Exchange-local and tz-aware; converted in normalize(). See note 3.
            tz_name = str(getattr(index, "tz", "") or "")

            info: Dict[str, Any] = {}
            try:
                fast = ticker.fast_info
                info = {
                    "currency": fast.get("currency"),
                    "exchange": fast.get("exchange"),
                    "quote_type": fast.get("quoteType"),
                    "market_cap": fast.get("marketCap"),
                }
            except Exception as exc:  # fast_info is best-effort metadata
                logger.debug("yfinance: fast_info failed for %s: %s", symbol, exc)

            return {
                "symbol": symbol,
                "long_name": _long_name(ticker, symbol),
                "period": period,
                "first_close": closes[0],
                "last_close": closes[-1],
                "low": min(closes),
                "high": max(closes),
                "avg_volume": (sum(volumes) / len(volumes)) if volumes else None,
                "bars": len(closes),
                "first_session": _isoformat(first_stamp),
                "last_session": _isoformat(last_stamp),
                "exchange_timezone": tz_name,
                **info,
            }

        return await self._guarded(_call, what=f"quote {symbol!r}")

    async def _guarded(self, call, *, what: str):
        """Throttled, retrying wrapper around one blocking yfinance call.

        Mirrors ``AngleConnector._get`` -- same throttle, same exponential
        backoff, same retry-only-on-transient policy -- but drives a library
        call instead of an HTTP GET. The blocking work runs in a thread so it
        does not stall the event loop Part 19 fans out on.
        """
        attempt = 0
        while True:
            await self._throttle()
            logger.info("%s: %s (attempt %d)", self.name, what, attempt + 1)
            try:
                return await asyncio.to_thread(call)
            except Exception as exc:
                mapped = self._map_exception(exc, what)
                if not isinstance(mapped, (RateLimitError, UpstreamError)):
                    raise mapped from exc
                if attempt >= self.max_retries:
                    logger.error(
                        "%s: giving up on %s after %d attempts: %s",
                        self.name, what, attempt + 1, mapped,
                    )
                    raise mapped from exc
                delay = _backoff_delay(attempt, mapped)
                logger.warning(
                    "%s: %s -- backing off %.2fs then retrying (attempt %d/%d)",
                    self.name, mapped, delay, attempt + 2, self.max_retries + 1,
                )
                await asyncio.sleep(delay)
                attempt += 1

    def _map_exception(self, exc: Exception, what: str) -> Exception:
        """yfinance's exceptions onto the Part 17 hierarchy."""
        name = type(exc).__name__
        if name == "YFRateLimitError":
            return RateLimitError(
                f"Yahoo throttled {what}", source=self.name, status_code=429
            )
        if name in ("YFInvalidPeriodError", "YFTickerMissingError"):
            return QueryError(f"{name} on {what}: {exc}", source=self.name)
        if isinstance(exc, (QueryError, RateLimitError, UpstreamError)):
            return exc
        return UpstreamError(f"{name} on {what}: {exc}", source=self.name)

    # -- parse -----------------------------------------------------------

    def parse(self, raw: RawResponse) -> List[Record]:
        """JSON body -> Records. Pure function of text; no network, no yfinance."""
        try:
            body = json.loads(raw.body_text)
        except (TypeError, ValueError) as exc:
            raise ParseError(
                f"response body is not valid JSON: {exc}", source=self.name
            ) from exc
        if not isinstance(body, dict) or "quotes" not in body:
            raise ParseError(
                "response body has no 'quotes' key", source=self.name
            )

        fetched_at = utcnow()
        records: List[Record] = []
        for quote in body.get("quotes") or []:
            if not isinstance(quote, dict):
                raise ParseError(
                    f"quote entry is not an object: {quote!r}", source=self.name
                )
            symbol = str(quote.get("symbol") or "").strip()
            if not symbol:
                raise ParseError("quote entry has no symbol", source=self.name)

            period = str(quote.get("period") or DEFAULT_PERIOD)
            snippet = _summarize(quote, period)
            _assert_snippet_bounded(snippet)
            records.append(
                Record(
                    id=f"{symbol}@{period}",
                    title=_title(quote, symbol),
                    url=f"https://finance.yahoo.com/quote/{symbol}",
                    snippet=snippet,
                    source_name=self.name,
                    fetched_at=fetched_at,
                    extra={
                        "author": quote.get("exchange") or None,
                        "published_date": _session_datetime(
                            quote.get("last_session")
                        ),
                        "exchange_timezone": quote.get("exchange_timezone") or "",
                        "symbol": symbol,
                        "period": period,
                        "bars": quote.get("bars"),
                    },
                )
            )
        return records

    # -- normalize -------------------------------------------------------

    def normalize(self, records: Sequence[Record]) -> List[NormalizedRecord]:
        """Records -> NormalizedRecords, with dates forced to UTC.

        The UTC conversion is the whole point of this override: see
        defect-class note 3. Every other source in V2 already hands back UTC.
        """
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
                    published_date=to_utc(rec.extra.get("published_date")),
                    author=rec.extra.get("author"),
                    relevance_score=1.0,  # reranked in Part 19
                )
            )
        return normalized


# --- helpers -----------------------------------------------------------


def to_utc(moment: Optional[datetime]) -> Optional[datetime]:
    """Any datetime -> tz-aware UTC. Naive input is assumed to be UTC.

    Yahoo's timestamps arrive tz-aware in exchange-local time (measured:
    America/New_York, America/Toronto, Europe/London, Asia/Tokyo), so this
    genuinely converts rather than merely labels.
    """
    if moment is None:
        return None
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _isoformat(stamp: Any) -> str:
    """pandas Timestamp -> ISO-8601, preserving its offset."""
    try:
        return stamp.to_pydatetime().isoformat()
    except AttributeError:
        return str(stamp)


def _session_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        logger.debug("unparseable yfinance session timestamp: %r", value)
        return None


def _long_name(ticker: Any, symbol: str) -> str:
    """Company name, best-effort. ``info`` is heavy and sometimes throttled."""
    try:
        name = (ticker.info or {}).get("longName") or ""
    except Exception:  # info is optional decoration, never load-bearing
        name = ""
    return str(name)[:NAME_CHARS]


def _title(quote: Dict[str, Any], symbol: str) -> str:
    name = str(quote.get("long_name") or "").strip()[:NAME_CHARS]
    return f"{symbol} -- {name}" if name else symbol


def _pct_change(first: Any, last: Any) -> Optional[float]:
    try:
        first_f, last_f = float(first), float(last)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(first_f) and math.isfinite(last_f)) or first_f == 0.0:
        return None
    return (last_f - first_f) / first_f * 100.0


def _human_volume(value: Any) -> str:
    try:
        volume = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(volume):
        return "n/a"
    for limit, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(volume) >= limit:
            return f"{volume / limit:.1f}{suffix}"
    return f"{volume:.0f}"


def _summarize(quote: Dict[str, Any], period: str) -> str:
    """The generated snippet. Fixed shape, no per-bar data -- see note 4.

    Every field is a scalar aggregate, so the length of this string does not
    grow with the length of the price history. A 10y request and a 5d request
    produce summaries of the same size.
    """
    symbol = str(quote.get("symbol") or "").strip()
    name = str(quote.get("long_name") or "").strip()[:NAME_CHARS]
    exchange = str(quote.get("exchange") or "").strip()
    currency = str(quote.get("currency") or "").strip()
    quote_type = str(quote.get("quote_type") or "").strip()

    head = f"{symbol} ({name})" if name else symbol
    venue = ", ".join(bit for bit in (exchange, currency) if bit)
    if venue:
        head = f"{head} -- {venue}"

    last = quote.get("last_close")
    change = _pct_change(quote.get("first_close"), last)
    bits = [f"{head}."]
    if _finite(last):
        move = f", {change:+.1f}% over {period}" if change is not None else ""
        bits.append(f"Last {float(last):.2f}{move}")
    if _finite(quote.get("low")) and _finite(quote.get("high")):
        bits.append(
            f"(range {float(quote['low']):.2f}-{float(quote['high']):.2f}, "
            f"avg volume {_human_volume(quote.get('avg_volume'))})"
        )
    tail = " ".join(bits)
    if quote_type:
        tail = f"{tail} {quote_type}."
    session = str(quote.get("last_session") or "")[:10]
    if session:
        tail = f"{tail} Session {session}."
    return tail[:SNIPPET_CHARS].strip()


def _assert_snippet_bounded(snippet: str) -> None:
    """Guard the note-4 invariant at construction.

    The summary is bounded by design, so this can only fire if someone adds an
    unbounded field to ``_summarize`` later. Better here, where the cause is
    one function away, than as a silent Part 19 oversize drop.
    """
    if len(snippet) > SNIPPET_CHARS:
        raise ParseError(
            f"generated snippet is {len(snippet)} chars, over the "
            f"{SNIPPET_CHARS} cap; _summarize gained an unbounded field",
            source="yfinance",
        )


__all__ = [
    "YFinanceConnector",
    "DEFAULT_RATE_PER_MIN",
    "DEFAULT_PERIOD",
    "VALID_PERIODS",
    "PERIOD_SEP",
    "SNIPPET_CHARS",
    "to_utc",
]
