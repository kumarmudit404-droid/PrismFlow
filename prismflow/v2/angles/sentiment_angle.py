"""The sentiment angle: what practitioners say when not being formal.

Part 19, with live connector wiring added in TASK 3 (authorised extension of a
frozen Part 19 file -- additive only, the existing class is unchanged).

    primary    StackExchange API  questions and answers (no key)  -- LIVE
    secondary  -- (was Reddit; see below)
    tertiary   -- (the brief lists none)

Its k is the highest of the five (15) and its budget the largest (2500),
because individual posts are short: a sentiment angle with k=5 is reading five
opinions and calling it a population.

WHY STACKEXCHANGE IS PRIMARY AND REDDIT IS NOT IN THE CHAIN AT ALL
------------------------------------------------------------------
The Part 19 brief orders this angle Reddit-primary, StackExchange-secondary.
That ordering is not followed here, and the reason is not "Reddit is not built
yet" -- it is that Reddit cannot work from this network. Measured 2026-09-24:
DNS, TCP and TLS are clean to both reddit.com and oauth.reddit.com, and
``POST /api/v1/access_token`` returns a genuine 401, but every data call to
``oauth.reddit.com`` returns 403 "blocked due to a network policy" with any
bearer token, and the public ``.json`` endpoints return a 403 block page even
with a browser User-Agent. The block is IP-scoped, so REDDIT_CLIENT_ID and
REDDIT_CLIENT_SECRET would authenticate and then fail on every fetch.

A connector in the chain that reliably 403s is worse than an absent one: Part
19 would spend a retry budget on it per query and report a source failure that
is not a failure of the source. So the slot is left empty rather than filled
with something known broken, and StackExchange is promoted to primary. If the
block is lifted, adding Reddit back is one entry in this factory.

WHY WIRING LIVES IN A FUNCTION AND NOT IN ``__init__``
------------------------------------------------------
Same reasoning as ``financial_angle.default_connectors()``: ``BaseAngle`` takes
its connectors by injection and every call site passes them through
``build_angles(sources=...)``. Constructing a connector inside the angle would
break that -- tests inject doubles, and a connector built at import time would
open a transport nobody asked for.
"""

from __future__ import annotations

from typing import Optional, Tuple

from prismflow.v2.connectors.base import AngleConnector

from .base import BaseAngle


class SentimentAngle(BaseAngle):
    """Forum discussion and Q&A."""

    ANGLE_NAME = "sentiment"
    DEFAULT_K = 15
    DEFAULT_TOKEN_BUDGET = 2500
    DEFAULT_RERANK_METHOD = "bm25"


def default_connectors() -> Tuple[Optional[AngleConnector], ...]:
    """The (primary, secondary, tertiary) chain this angle can fill today.

    Only the primary slot exists. Returned at full width so it drops straight
    into ``build_angles(sources=...)`` without the caller padding it.
    """
    from prismflow.v2.connectors.stackexchange import StackExchangeConnector

    return (StackExchangeConnector(), None, None)


__all__ = ["SentimentAngle", "default_connectors"]
