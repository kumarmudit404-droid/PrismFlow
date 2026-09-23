"""The market angle: what is being reported and searched for.

Part 19. No connector exists yet -- Part 17 built GitHub and arXiv only -- so
this angle is wired and tested but returns empty evidence with an explicit
warning until one is injected.

    primary    NewsAPI           coverage (needs NEWSAPI_KEY; unset)
    secondary  Google Trends     search interest
    tertiary   CrunchBase        funding and company news

Note for whoever builds these: NewsAPI and CrunchBase both aggregate press
releases, so the two are far less independent than their names suggest. Part 21
measures that rather than assuming it, but a fallback from one to the other is
close to a no-op in independence terms.
"""

from __future__ import annotations

from .base import BaseAngle


class MarketAngle(BaseAngle):
    """News, trends and funding coverage."""

    ANGLE_NAME = "market"
    DEFAULT_K = 8
    DEFAULT_TOKEN_BUDGET = 1500
    DEFAULT_RERANK_METHOD = "bm25"


__all__ = ["MarketAngle"]
