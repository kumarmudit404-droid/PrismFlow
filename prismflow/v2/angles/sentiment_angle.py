"""The sentiment angle: what practitioners say when not being formal.

Part 19. No connector exists yet; inject one when it is built.

    primary    Reddit (PRAW)      discussion (needs REDDIT_CLIENT_ID; unset)
    secondary  StackExchange API  questions and answers (no key)
    tertiary   -- (the brief lists none)

Its k is the highest of the five (15) and its budget the largest (2500),
because individual posts are short: a sentiment angle with k=5 is reading five
opinions and calling it a population.
"""

from __future__ import annotations

from .base import BaseAngle


class SentimentAngle(BaseAngle):
    """Forum discussion and Q&A."""

    ANGLE_NAME = "sentiment"
    DEFAULT_K = 15
    DEFAULT_TOKEN_BUDGET = 2500
    DEFAULT_RERANK_METHOD = "bm25"


__all__ = ["SentimentAngle"]
