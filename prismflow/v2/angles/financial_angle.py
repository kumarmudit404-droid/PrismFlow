"""The financial angle: what the numbers and the filings say.

Part 19, with live connector wiring added in TASK 2a (authorised extension of
a frozen Part 19 file -- additive only, no behaviour of the existing class was
changed).

    primary    Yahoo Finance    prices (yfinance, no key)  -- LIVE
    secondary  Alpha Vantage    technical indicators (key optional, unset)
    tertiary   SEC EDGAR        filings (no key)           -- not built

A caution for the connector author: prices are numeric series, not documents.
A NormalizedRecord's ``snippet`` is what reranking and Part 23 read, so a
financial connector has to render its numbers into text deliberately -- an
empty snippet ranks last under BM25 no matter how relevant the instrument is.
``YFinanceConnector`` does exactly that, and its module docstring records the
judgment call, since the source gives no text of its own.

WHY WIRING LIVES IN A FUNCTION AND NOT IN ``__init__``
------------------------------------------------------
``BaseAngle`` takes its connectors by injection, and every existing call site
passes them through ``build_angles(sources=...)``. Constructing a connector
inside the angle would break that -- tests inject doubles, and a connector
built at import time would open a transport nobody asked for.

``default_connectors()`` is therefore a factory the caller opts into:

    from prismflow.v2.angles import build_angles
    from prismflow.v2.angles.financial_angle import default_connectors
    angles = build_angles(sources={"financial": default_connectors()})

which keeps injection intact while putting the knowledge of WHICH connector
this angle wants in the angle's own file, where the next person looks.
"""

from __future__ import annotations

from typing import Optional, Tuple

from prismflow.v2.connectors.base import AngleConnector

from .base import BaseAngle


class FinancialAngle(BaseAngle):
    """Prices, indicators and filings."""

    ANGLE_NAME = "financial"
    DEFAULT_K = 10
    DEFAULT_TOKEN_BUDGET = 2000
    DEFAULT_RERANK_METHOD = "bm25"


def default_connectors() -> Tuple[Optional[AngleConnector], ...]:
    """The (primary, secondary, tertiary) chain this angle can fill today.

    Only the primary slot exists: Alpha Vantage and SEC EDGAR are not built.
    The tuple is returned at full width so it drops straight into
    ``build_angles(sources=...)`` without the caller padding it.
    """
    from prismflow.v2.connectors.yfinance import YFinanceConnector

    return (YFinanceConnector(), None, None)


__all__ = ["FinancialAngle", "default_connectors"]
