"""Connector exception hierarchy.

WHY THE SPLIT IS USER-ERROR VS SYSTEM-ERROR
-------------------------------------------
Part 19 fans out one retrieval per angle and has to decide, per failure,
whether to retry, to drop the angle, or to abort the whole query. Those are
three different answers and an undifferentiated exception cannot pick between
them:

  QueryError    the caller asked for something the API will never answer
                (malformed query, 422). Retrying is pointless -- the same
                request fails identically. Surface to the caller.

  AuthError     credentials are absent, wrong, or expired (401). Retrying is
                pointless. Distinct from QueryError because the fix is
                operational (set a token), not a code change.

  RateLimitError    the source is willing but throttling us (403 with rate
                    limit headers, 429). Retrying AFTER A WAIT is exactly
                    right, and is what the base class does.

  UpstreamError     the source is broken or unreachable (5xx, timeouts,
                    connection resets). Retrying is reasonable; the fault is
                    not ours.

  ParseError    the transport succeeded but the body is not what the source
                documents. Retrying is usually pointless but not always
                (truncated gzip); the base class does not retry it.

ParseError also subclasses ValueError because the Part 17 specification says
``parse`` raises ValueError on malformed input. Inheriting from both keeps
that contract while still letting callers catch ConnectorError broadly.
"""

from __future__ import annotations

from typing import Optional


class ConnectorError(Exception):
    """Base for every failure raised by a connector.

    Carries the source name so a fan-out in Part 19 can report which angle
    failed without inspecting the traceback.
    """

    def __init__(self, message: str, *, source: Optional[str] = None,
                 status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.source = source
        self.status_code = status_code

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        base = super().__str__()
        bits = []
        if self.source:
            bits.append(f"source={self.source}")
        if self.status_code is not None:
            bits.append(f"status={self.status_code}")
        return f"{base} ({', '.join(bits)})" if bits else base


class QueryError(ConnectorError):
    """The request is malformed or unanswerable. Do not retry."""


class AuthError(ConnectorError):
    """Credentials missing, invalid, or insufficient. Do not retry."""


class RateLimitError(ConnectorError):
    """Throttled by the source. Retry after a wait.

    ``retry_after`` is seconds, taken from the ``Retry-After`` header when the
    source sends one. None means the caller should use its own backoff.
    """

    def __init__(self, message: str, *, source: Optional[str] = None,
                 status_code: Optional[int] = None,
                 retry_after: Optional[float] = None) -> None:
        super().__init__(message, source=source, status_code=status_code)
        self.retry_after = retry_after


class UpstreamError(ConnectorError):
    """The source failed or was unreachable. Retry is reasonable."""


class ParseError(ConnectorError, ValueError):
    """The body could not be parsed into Records.

    Subclasses ValueError to honour the Part 17 contract for ``parse``.
    """


__all__ = [
    "ConnectorError",
    "QueryError",
    "AuthError",
    "RateLimitError",
    "UpstreamError",
    "ParseError",
]
