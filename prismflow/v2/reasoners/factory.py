"""Which provider reasons for which angle.

Part 20.

WHAT ALTERNATING PROVIDERS DOES AND DOES NOT BUY
------------------------------------------------
The brief assigns providers so they alternate -- tech, financial and sentiment
to Claude, market and regulatory to OpenAI -- and describes this as ensuring
independence between angles. It reduces correlation; it does not remove it, and
the distinction matters here more than in most projects, because PrismFlow's
whole thesis is that agreement counts in proportion to the *effective*
independence of the views that agree.

What the assignment actually produces is two clusters: three angles sharing one
model and two sharing another. Angles within a cluster share a tokenizer, a
training distribution and a refusal profile, so their errors are correlated in
ways two unrelated analysts' would not be. A "5 of 5 angles agree" reading over
this configuration is closer to "2 of 2 model families agree", and the second is
much weaker evidence than the first.

This is not a defect to fix in Part 20 -- it is a property of the design that
Part 21 exists to measure. It is recorded here, next to the mapping that causes
it, so the measurement is interpreted against a stated expectation rather than
discovered as a surprise. ``provider_clusters`` reports the grouping directly so
Part 21 can compare measured dependence against the structural prior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from .base import AngleReasoner
from .claude_reasoner import ClaudeReasoner
from .openai_reasoner import OpenAIReasoner

#: Angle to provider, as the Part 20 brief specifies.
ANGLE_PROVIDERS: Dict[str, str] = {
    "tech": "claude",
    "market": "openai",
    "financial": "claude",
    "regulatory": "openai",
    "sentiment": "claude",
}

PROVIDER_CLASSES: Dict[str, type] = {
    "claude": ClaudeReasoner,
    "openai": OpenAIReasoner,
}


def get_reasoner(
    angle_name: str,
    *,
    provider: Optional[str] = None,
    **kwargs: Any,
) -> AngleReasoner:
    """Build the reasoner for one angle.

    Args:
        angle_name: "tech", "market", "financial", "regulatory", "sentiment".
        provider: override the mapping, e.g. to run every angle on one provider
            as a control condition for Part 21.
        **kwargs: passed to the reasoner (``model``, ``client``, ``prompt_dir``).

    Raises:
        ValueError: unknown angle or unknown provider. Not defaulted: silently
            picking a provider would make the provider mapping -- the thing
            Part 21 measures dependence against -- depend on a typo.
    """
    name = provider or ANGLE_PROVIDERS.get(angle_name)
    if name is None:
        raise ValueError(
            f"no provider is mapped for angle {angle_name!r}; known angles: "
            f"{', '.join(sorted(ANGLE_PROVIDERS))}"
        )
    reasoner_class = PROVIDER_CLASSES.get(name)
    if reasoner_class is None:
        raise ValueError(
            f"unknown provider {name!r}; known providers: "
            f"{', '.join(sorted(PROVIDER_CLASSES))}"
        )
    return reasoner_class(angle_name, **kwargs)


def build_reasoners(
    angle_names: Optional[List[str]] = None,
    **kwargs: Any,
) -> Dict[str, AngleReasoner]:
    """One reasoner per angle, ready for Part 23 to drive concurrently."""
    names = angle_names if angle_names is not None else list(ANGLE_PROVIDERS)
    return {name: get_reasoner(name, **kwargs) for name in names}


def provider_clusters(
    mapping: Optional[Mapping[str, str]] = None,
) -> Dict[str, List[str]]:
    """Angles grouped by the provider they share.

    The structural prior Part 21's measured dependence should be read against:
    angles in the same group cannot be treated as independent views.
    """
    source = mapping if mapping is not None else ANGLE_PROVIDERS
    clusters: Dict[str, List[str]] = {}
    for angle, provider in source.items():
        clusters.setdefault(provider, []).append(angle)
    return {provider: sorted(angles) for provider, angles in clusters.items()}


__all__ = [
    "ANGLE_PROVIDERS",
    "PROVIDER_CLASSES",
    "get_reasoner",
    "build_reasoners",
    "provider_clusters",
]
