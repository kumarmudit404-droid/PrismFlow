"""The market angle: what is being reported and searched for.

Part 19, with live connector wiring added in TASK 3 (authorised extension of a
frozen Part 19 file -- additive only, the existing class is unchanged).

    primary    NewsAPI           coverage (NEWSAPI_KEY)     -- LIVE
    secondary  Google Trends     search interest            -- not built
    tertiary   CrunchBase        funding and company news   -- not built

Note for whoever builds these: NewsAPI and CrunchBase both aggregate press
releases, so the two are far less independent than their names suggest. Part 21
measures that rather than assuming it, but a fallback from one to the other is
close to a no-op in independence terms.

TWO PROPERTIES OF THIS ANGLE'S EVIDENCE THAT THE CONNECTOR CANNOT FIX
---------------------------------------------------------------------
Both are measured in ``connectors/newsapi.py``; they are repeated here because
they are facts about what this ANGLE contributes, not about the transport.

1. EVERY MARKET RECORD IS ABOUT A DAY OLD. The free Developer plan imposes a
   24-hour article delay -- measured, the freshest of 98 articles was 24.09
   hours old. So on any given run this angle's dates sit a day behind the Tech,
   Financial and Sentiment angles' dates, and Part 21 compares dates across
   angles. It is a property of the plan, not a defect to be corrected here.

2. THE DAILY BUDGET IS 100 REQUESTS AND NOTHING REPORTS WHAT IS LEFT. There is
   no quota header and no quota field, so neither the connector nor this angle
   can know how much of the day remains. Part 24 as scoped (50 queries x 5
   seeds x 4 randomisations) would exceed the free ceiling by an order of
   magnitude on cache misses alone; that needs a paid plan or a cache-warming
   pass, and is called out rather than discovered later.

WHY WIRING LIVES IN A FUNCTION AND NOT IN ``__init__``
------------------------------------------------------
Same reasoning as ``financial_angle.default_connectors()`` and
``sentiment_angle.default_connectors()``: ``BaseAngle`` takes its connectors by
injection and every call site passes them through ``build_angles(sources=...)``.
Constructing a connector inside the angle would break that -- tests inject
doubles, and a connector built at import time would open a transport nobody
asked for.
"""

from __future__ import annotations

from typing import Optional, Tuple

from prismflow.v2.connectors.base import AngleConnector

from .base import BaseAngle


class MarketAngle(BaseAngle):
    """News, trends and funding coverage."""

    ANGLE_NAME = "market"
    DEFAULT_K = 8
    DEFAULT_TOKEN_BUDGET = 1500
    DEFAULT_RERANK_METHOD = "bm25"


def default_connectors() -> Tuple[Optional[AngleConnector], ...]:
    """The (primary, secondary, tertiary) chain this angle can fill today.

    Only the primary slot exists: Google Trends and CrunchBase are not built.
    The tuple is returned at full width so it drops straight into
    ``build_angles(sources=...)`` without the caller padding it.

    The connector resolves ``NEWSAPI_KEY`` itself. With no key it constructs
    fine and ``fetch`` raises ``AuthError``, which ``BaseAngle`` records as a
    source failure -- the angle degrades to empty evidence with a warning
    rather than failing the whole query.
    """
    from prismflow.v2.connectors.newsapi import NewsAPIConnector

    return (NewsAPIConnector(), None, None)


__all__ = ["MarketAngle", "default_connectors"]
