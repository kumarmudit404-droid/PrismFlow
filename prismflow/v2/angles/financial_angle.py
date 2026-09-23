"""The financial angle: what the numbers and the filings say.

Part 19. No connector exists yet; inject one when it is built.

    primary    Yahoo Finance    prices and earnings (yfinance, no key)
    secondary  Alpha Vantage    technical indicators (key optional, unset)
    tertiary   SEC EDGAR        filings (no key)

A caution for the connector author: prices are numeric series, not documents.
A NormalizedRecord's ``snippet`` is what reranking and Part 23 read, so a
financial connector has to render its numbers into text deliberately -- an
empty snippet ranks last under BM25 no matter how relevant the instrument is.
"""

from __future__ import annotations

from .base import BaseAngle


class FinancialAngle(BaseAngle):
    """Prices, indicators and filings."""

    ANGLE_NAME = "financial"
    DEFAULT_K = 10
    DEFAULT_TOKEN_BUDGET = 2000
    DEFAULT_RERANK_METHOD = "bm25"


__all__ = ["FinancialAngle"]
