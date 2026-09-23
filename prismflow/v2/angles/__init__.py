"""Per-angle retrieval: one angle, one fallback chain, one budget.

Part 19. Turns connectors (Part 17) plus a cache (Part 18) into
``AngleEvidence`` -- reranked, budgeted, and carrying the provenance Part 21
needs to judge how independent two angles really were.

Of the five angles, only ``tech`` has live connectors: Part 17 built GitHub and
arXiv, and the remaining sources in the Part 19 brief's table do not exist yet.
The other four are implemented against the same injected-connector interface and
are fully exercised by tests with doubles; each returns empty evidence and an
explicit warning until a connector is passed in. No angle code changes when one
arrives.

Downstream parts import from here, not from the submodules.
"""

from typing import Dict, Mapping, Optional, Sequence, Tuple

from prismflow.v2.connectors.base import AngleConnector

from .base import BaseAngle
from .config import (
    ANGLE_NAMES,
    BUDGET_POLICIES,
    DEFAULT_CONFIG_PATH,
    SOURCE_SLOTS,
    AngleConfig,
    load_config,
)
from .financial_angle import FinancialAngle
from .market_angle import MarketAngle
from .models import AngleEvidence
from .regulatory_angle import RegulatoryAngle
from .reranker import (
    CROSS_ENCODER_MODEL,
    RERANK_METHODS,
    RerankOutcome,
    document_text,
    rerank_records,
    rerank_with_outcome,
    tokenize,
)
from .sentiment_angle import SentimentAngle
from .tech_angle import TechAngle

#: Angle name to class. Iterating this is how Part 23 will build all five.
ANGLE_CLASSES: Dict[str, type] = {
    "tech": TechAngle,
    "market": MarketAngle,
    "financial": FinancialAngle,
    "regulatory": RegulatoryAngle,
    "sentiment": SentimentAngle,
}


def build_angles(
    config: Optional[Mapping[str, AngleConfig]] = None,
    sources: Optional[Mapping[str, Sequence[Optional[AngleConnector]]]] = None,
    **angle_kwargs,
) -> Dict[str, BaseAngle]:
    """Build all five angles from a config, wiring whatever connectors exist.

    Args:
        config: ``{angle_name: AngleConfig}`` as returned by ``load_config``.
            An angle missing from it falls back to its class defaults.
        sources: ``{angle_name: (primary, secondary, tertiary)}``. Any angle
            omitted, or given ``None`` slots, is built without connectors --
            which is the current state of four of the five.
        **angle_kwargs: passed to every angle (``use_cache``,
            ``cache_db_path``, ``cache_metrics``).

    Returns:
        ``{angle_name: angle}`` for all five angles, in declaration order.
    """
    built: Dict[str, BaseAngle] = {}
    for name, angle_class in ANGLE_CLASSES.items():
        slots: Tuple = tuple((sources or {}).get(name, ()) or ())
        primary, secondary, tertiary = (list(slots) + [None, None, None])[:3]
        built[name] = angle_class(
            primary,
            secondary,
            tertiary,
            config=(config or {}).get(name),
            **angle_kwargs,
        )
    return built


__all__ = [
    # angles
    "BaseAngle",
    "TechAngle",
    "MarketAngle",
    "FinancialAngle",
    "RegulatoryAngle",
    "SentimentAngle",
    "ANGLE_CLASSES",
    "build_angles",
    # evidence
    "AngleEvidence",
    # config
    "load_config",
    "AngleConfig",
    "DEFAULT_CONFIG_PATH",
    "SOURCE_SLOTS",
    "BUDGET_POLICIES",
    "ANGLE_NAMES",
    # reranking
    "rerank_records",
    "rerank_with_outcome",
    "RerankOutcome",
    "RERANK_METHODS",
    "CROSS_ENCODER_MODEL",
    "document_text",
    "tokenize",
]
