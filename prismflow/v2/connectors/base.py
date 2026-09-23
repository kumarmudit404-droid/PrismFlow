"""The connector contract: how V2 turns an external source into evidence.

An ENGINEERING component per docs/CONTRACT.md section 3 -- no learnable
parameters, no tensors. It sits upstream of everything: the angles in Part 19
are only as independent as the sources behind them, so this layer decides how
much of V2's independence claim is real.

WHY THREE STAGES AND NOT ONE
----------------------------
``fetch`` / ``parse`` / ``normalize`` split on the two seams where behaviour
differs and where testing needs a wedge:

  fetch       the only step that touches the network. Non-deterministic,
              rate-limited, and the only step that can fail for reasons that
              have nothing to do with the query. Isolating it means every
              other stage is a pure function of bytes and can be tested
              without HTTP.

  parse       source-specific structure (GitHub JSON, arXiv Atom) collapses
              into one Record type here. Everything downstream is
              source-agnostic from this point on.

  normalize   the cross-source comparability step. A GitHub repo and an arXiv
              paper are not commensurable until token counts, dates and
              authorship are extracted the same way. Part 21 compares angles
              against each other; if normalisation is inconsistent, apparent
              dependence between angles is an artefact of this file rather
              than a fact about the world.

WHY THE HTTP CLIENT IS INJECTED
-------------------------------
Tests must exercise parse/normalize against known bytes, and Part 19 must be
able to fan out concurrently over one shared session. Both need the transport
to be a parameter rather than a hard-coded import. ``HTTPClient`` is a
Protocol, so a test double is any object with a matching ``get``.

RATE LIMITING IS IN THE BASE CLASS ON PURPOSE
---------------------------------------------
Every source throttles, and a connector that forgets to throttle does not
fail locally -- it fails later, as a 403 storm, inside somebody else's
fan-out. Spacing is enforced here for all subclasses, guarded by an
asyncio.Lock so that concurrent callers on one connector serialise rather
than all firing at once. Backoff on retry is exponential and honours
``Retry-After`` when the source sends one.

Every request and every backoff is logged (rule 2 of the Part 17 standing
rules) so Part 19 can observe the real throughput of a fan-out.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence

from .errors import (
    AuthError,
    ConnectorError,
    QueryError,
    RateLimitError,
    UpstreamError,
)

logger = logging.getLogger("prismflow.v2.connectors")

#: Sent on every request. Some sources (arXiv among them) reject or throttle
#: unidentified clients, and GitHub documents it as required.
USER_AGENT = "PrismFlow/2.0 (Mudit Kumar, kumarmudit404-droid/PrismFlow)"

#: Part 17 note: 10s connect, 30s read.
CONNECT_TIMEOUT_S = 10.0
READ_TIMEOUT_S = 30.0


# --- data structures ---------------------------------------------------


@dataclass
class RawResponse:
    """Exactly what the transport returned. No interpretation."""

    status_code: int
    headers: Dict[str, str]
    body_text: str


@dataclass
class Record:
    """One result, in the shape every source can agree on.

    ``extra`` exists because the Part 17 spec gives Record no author or date
    field, while NormalizedRecord requires both. Without a carrier the
    information that ``parse`` has already read would have to be re-derived in
    ``normalize``, which cannot see the raw body. Source-specific extraction
    stays in ``parse`` (where the source's schema is known) and rides here as
    plain Python values -- ``author`` (str) and ``published_date`` (datetime)
    by convention.
    """

    id: str
    title: str
    url: str
    snippet: str
    source_name: str
    fetched_at: datetime
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedRecord:
    """A Record made comparable across sources.

    AMENDED FOR PART 19 (authorised unfreeze, 2026-09-23)
    -----------------------------------------------------
    ``snippet`` was added alongside ``snippet_tokens``. As originally sealed
    this type carried only the token *count*, so no snippet text survived
    normalisation -- and Part 19 reranks records against a query, which needs
    text. Only ``title`` was available, about 10% of a record's text on real
    arXiv results (titles of ~10 tokens against snippets of ~93), so BM25 would
    have scored on a tenth of the signal.

    The alternative was for Part 19 to re-derive the text by calling ``fetch``
    and ``parse`` itself, which works but bypasses ``get_records`` and so
    bypasses the Part 18 cache entirely. Amending the type once, here, was
    chosen deliberately: every part from 19 to 24 consumes NormalizedRecord and
    the Part 18 cache serialises it, so the same change made later would be a
    migration across five parts rather than one field.

    ``snippet_tokens`` keeps its original meaning -- ``count_tokens(snippet)``
    -- and is not derived on the fly, because Part 19 and Part 23 budget
    against it and a recount would make the budget depend on whether tiktoken
    was reachable.
    """

    id: str
    title: str
    url: str
    snippet: str
    snippet_tokens: int
    source: str
    published_date: Optional[datetime]
    author: Optional[str]
    relevance_score: float


# --- token counting ----------------------------------------------------

_ENCODER: Any = None
_ENCODER_TRIED = False


def count_tokens(text: str) -> int:
    """Token count for ``text`` using tiktoken, with a degraded fallback.

    Standing rule 3 names ``tiktoken.encoding_for_model("gpt-3.5-turbo")``.
    That call fetches the BPE file on first use and caches it, so it can fail
    on a cold, offline machine. Token counts feed budgeting in Part 19 and
    Part 23, not any evidence claim, so a failure here must not take down a
    retrieval. The fallback is a word/4-character heuristic and is logged
    loudly so it is never mistaken for a real count.
    """
    global _ENCODER, _ENCODER_TRIED
    if not text:
        return 0
    if _ENCODER is None and not _ENCODER_TRIED:
        _ENCODER_TRIED = True
        try:
            import tiktoken

            _ENCODER = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except Exception as exc:  # pragma: no cover - environment dependent
            logger.warning(
                "tiktoken unavailable (%s: %s); falling back to a character "
                "heuristic for token counts",
                type(exc).__name__,
                exc,
            )
            _ENCODER = None
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return max(1, (len(text) + 3) // 4)  # pragma: no cover - fallback only


# --- transport ---------------------------------------------------------


class HTTPClient(Protocol):
    """Minimal transport contract. Any object with this ``get`` will do."""

    async def get(
        self,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
    ) -> RawResponse:
        ...


class AiohttpClient:
    """Default transport: one aiohttp session, lazily created.

    The session is created inside the running loop rather than in __init__,
    because a session bound to a loop that has since closed raises at request
    time -- and connectors are constructed at import/fixture time, well before
    any loop exists.
    """

    def __init__(self, user_agent: str = USER_AGENT) -> None:
        self.user_agent = user_agent
        self._session: Any = None

    async def _ensure_session(self) -> Any:
        import aiohttp

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(
                connect=CONNECT_TIMEOUT_S, total=READ_TIMEOUT_S
            )
            self._session = aiohttp.ClientSession(
                headers={"User-Agent": self.user_agent}, timeout=timeout
            )
        return self._session

    async def get(
        self,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
    ) -> RawResponse:
        import aiohttp

        session = await self._ensure_session()
        try:
            async with session.get(url, headers=dict(headers or {})) as resp:
                body = await resp.text()
                return RawResponse(
                    status_code=resp.status,
                    headers={k.lower(): v for k, v in resp.headers.items()},
                    body_text=body,
                )
        except asyncio.TimeoutError as exc:
            raise UpstreamError(f"timeout requesting {url}") from exc
        except aiohttp.ClientError as exc:
            raise UpstreamError(
                f"transport failure requesting {url}: {exc}"
            ) from exc

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None


# --- the ABC -----------------------------------------------------------


class AngleConnector(ABC):
    """Standardised access to one external source.

    Subclasses implement ``fetch``, ``parse`` and ``normalize``; throttling,
    retry, status mapping and logging are inherited.
    """

    def __init__(
        self,
        name: str,
        rate_limit_per_min: int = 30,
        *,
        http_client: Optional[HTTPClient] = None,
        max_retries: int = 3,
    ) -> None:
        if rate_limit_per_min <= 0:
            raise ValueError("rate_limit_per_min must be positive")
        self.name = name
        self.rate_limit = rate_limit_per_min
        self.last_request_time = 0.0
        self.request_count = 0
        self.max_retries = max_retries
        self._client = http_client if http_client is not None else AiohttpClient()
        self._lock = asyncio.Lock()

    # -- interface implemented by subclasses ----------------------------

    @abstractmethod
    async def fetch(self, query: str, k: int) -> RawResponse:
        """Fetch up to k results for query."""

    @abstractmethod
    def parse(self, raw: RawResponse) -> List[Record]:
        """Parse a raw response into Records. Raises ParseError/ValueError."""

    @abstractmethod
    def normalize(self, records: Sequence[Record]) -> List[NormalizedRecord]:
        """Make Records comparable across sources."""

    async def get_records(self, query: str, k: int) -> List[NormalizedRecord]:
        """fetch -> parse -> normalize. Exceptions propagate unchanged."""
        raw = await self.fetch(query, k)
        records = self.parse(raw)
        return self.normalize(records)

    # -- shared machinery ------------------------------------------------

    @property
    def min_interval_s(self) -> float:
        """Seconds between request starts implied by the rate limit."""
        return 60.0 / float(self.rate_limit)

    async def _throttle(self) -> None:
        """Space request starts by ``min_interval_s``.

        Held under a lock so concurrent callers queue instead of all reading a
        stale ``last_request_time`` and firing simultaneously.
        """
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval_s - (now - self.last_request_time)
            if wait > 0 and self.last_request_time > 0.0:
                logger.debug(
                    "%s: throttling %.3fs to hold %d req/min",
                    self.name,
                    wait,
                    self.rate_limit,
                )
                await asyncio.sleep(wait)
                now = time.monotonic()
            self.last_request_time = now
            self.request_count += 1

    def _check_status(self, raw: RawResponse) -> None:
        """Map an HTTP status onto the exception hierarchy.

        Subclasses override for source-specific quirks. 403 is treated as
        throttling rather than a hard refusal because that is how GitHub
        signals a rate limit; a source that means something else by 403 should
        say so in its own override.
        """
        code = raw.status_code
        if 200 <= code < 300:
            return
        if code == 401:
            raise AuthError("unauthorized", source=self.name, status_code=code)
        if code in (403, 429):
            raise RateLimitError(
                "throttled by source",
                source=self.name,
                status_code=code,
                retry_after=_retry_after_seconds(raw.headers),
            )
        if code == 422:
            raise QueryError(
                "query rejected as unprocessable",
                source=self.name,
                status_code=code,
            )
        if 400 <= code < 500:
            raise QueryError(f"client error {code}", source=self.name,
                             status_code=code)
        if code >= 500:
            raise UpstreamError(f"server error {code}", source=self.name,
                                status_code=code)
        raise ConnectorError(f"unexpected status {code}", source=self.name,
                             status_code=code)

    async def _get(
        self,
        url: str,
        headers: Optional[Mapping[str, str]] = None,
    ) -> RawResponse:
        """Throttled GET with exponential backoff on retryable failures.

        Retries RateLimitError and UpstreamError only. AuthError, QueryError
        and ParseError are deterministic: repeating them wastes the caller's
        time and the source's quota.
        """
        attempt = 0
        while True:
            await self._throttle()
            logger.info("%s: GET %s (attempt %d)", self.name, url, attempt + 1)
            try:
                raw = await self._client.get(url, headers=headers)
                self._check_status(raw)
                return raw
            except (RateLimitError, UpstreamError) as exc:
                if attempt >= self.max_retries:
                    logger.error(
                        "%s: giving up after %d attempts: %s",
                        self.name, attempt + 1, exc,
                    )
                    raise
                delay = _backoff_delay(attempt, exc)
                logger.warning(
                    "%s: %s -- backing off %.2fs then retrying (attempt %d/%d)",
                    self.name, exc, delay, attempt + 2, self.max_retries + 1,
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def aclose(self) -> None:
        """Release the transport if we own one."""
        closer = getattr(self._client, "aclose", None)
        if closer is not None:
            await closer()


# --- helpers -----------------------------------------------------------


def _retry_after_seconds(headers: Mapping[str, str]) -> Optional[float]:
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _backoff_delay(attempt: int, exc: Exception) -> float:
    """Exponential backoff, capped, deferring to Retry-After when present."""
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        return min(float(retry_after), 60.0)
    return min(2.0 ** attempt, 30.0)


def utcnow() -> datetime:
    """Timezone-aware now, so fetched_at is comparable across machines."""
    return datetime.now(timezone.utc)


__all__ = [
    "USER_AGENT",
    "RawResponse",
    "Record",
    "NormalizedRecord",
    "HTTPClient",
    "AiohttpClient",
    "AngleConnector",
    "count_tokens",
    "utcnow",
]
