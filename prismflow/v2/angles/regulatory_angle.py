"""The regulatory angle: what the rules and public datasets say.

Part 19. No connector exists yet; inject one when it is built.

    primary    data.gov     US datasets (no key)
    secondary  EUR-Lex      EU law (no key)
    tertiary   RBI open data  India

This is the angle the brief assigns a cross-encoder, and the assignment is
sound: regulatory text is long and lexically formulaic, so BM25's term overlap
is at its weakest here. It degrades to BM25 when the model is not present
rather than failing, which is the state of a fresh checkout.
"""

from __future__ import annotations

from .base import BaseAngle


class RegulatoryAngle(BaseAngle):
    """Datasets, statutes and regulatory publications."""

    ANGLE_NAME = "regulatory"
    DEFAULT_K = 5
    DEFAULT_TOKEN_BUDGET = 1000
    DEFAULT_RERANK_METHOD = "cross-encoder"


__all__ = ["RegulatoryAngle"]
