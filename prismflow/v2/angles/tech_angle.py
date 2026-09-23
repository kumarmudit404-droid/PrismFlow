"""The tech angle: what is being built and what is being published.

Part 19. The only angle with live connectors, because Part 17 built GitHub and
arXiv and nothing else.

    primary    GitHub        repositories -- what people ship
    secondary  arXiv         papers -- what people claim
    tertiary   Papers With Code (not built; inject when it exists)

GitHub and arXiv are genuinely different populations, which is the point: a
technique can be heavily implemented and barely published, or the reverse, and
an angle that read only one of them would mistake one for the field.
"""

from __future__ import annotations

from .base import BaseAngle


class TechAngle(BaseAngle):
    """Repositories and papers, in that fallback order."""

    ANGLE_NAME = "tech"
    DEFAULT_K = 10
    DEFAULT_TOKEN_BUDGET = 2000
    DEFAULT_RERANK_METHOD = "bm25"


__all__ = ["TechAngle"]
