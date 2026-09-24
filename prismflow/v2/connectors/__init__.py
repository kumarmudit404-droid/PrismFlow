"""Connectors: one standardised interface per external data source.

Part 17. Downstream parts import from here rather than from the individual
modules, so that adding a source does not ripple through call sites.
"""

from .arxiv import ArxivConnector
from .base import (
    USER_AGENT,
    AiohttpClient,
    AngleConnector,
    HTTPClient,
    NormalizedRecord,
    RawResponse,
    Record,
    count_tokens,
    utcnow,
)
from .errors import (
    AuthError,
    ConnectorError,
    ParseError,
    QueryError,
    RateLimitError,
    UpstreamError,
)
from .github import GitHubConnector
from .stackexchange import StackExchangeConnector
from .yfinance import YFinanceConnector

__all__ = [
    # interface
    "AngleConnector",
    "HTTPClient",
    "AiohttpClient",
    "USER_AGENT",
    # data
    "RawResponse",
    "Record",
    "NormalizedRecord",
    "count_tokens",
    "utcnow",
    # sources
    "GitHubConnector",
    "ArxivConnector",
    "YFinanceConnector",
    "StackExchangeConnector",
    # errors
    "ConnectorError",
    "QueryError",
    "AuthError",
    "RateLimitError",
    "UpstreamError",
    "ParseError",
]
