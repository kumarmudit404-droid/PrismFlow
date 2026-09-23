"""Loading and validating per-angle retrieval settings.

An ENGINEERING component per docs/CONTRACT.md section 3.

WHY THE CONFIG IS VALIDATED AT LOAD AND NOT AT USE
--------------------------------------------------
A typo in ``rerank_method`` is an operator mistake, and standing rule 4 ("a
failed rerank warns, it does not raise") is about *runtime* failure -- a model
that will not download, a corpus that will not tokenise. Conflating the two
would mean ``rerank_method: bm52`` silently produces unranked evidence for
every query in a run, and the only trace would be one warning per retrieval.
So the config raises on load and the reranker warns at runtime.

THE TTL CONFLICT, AND HOW IT IS RESOLVED
----------------------------------------
Part 18 sets a TTL per *connector*, from how fast that source's answer changes
(arXiv 86400, because a published paper is immutable; yfinance 300, because a
quote is stale immediately). This config sets a TTL per *angle*. They disagree:
the tech angle asks for 3600 while its arXiv source defaults to 86400, and the
regulatory angle asks for 604800 while every real source is far below that.

The effective TTL is ``min(angle, connector)``. Read as constraints rather than
preferences, both are then respected: an angle may demand fresher data than a
source's default, but no angle can make a fast-moving source appear fresh for
longer than the source itself allows. The alternative -- letting the angle
override -- would have let the regulatory angle serve week-old market data
because of a number written in a config about regulations.

This is a judgment call on a genuine conflict in the brief, not a measured
fact. Changing it is one line in ``effective_ttl``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import yaml

from prismflow.v2.cache.decorator import ttl_for

from .reranker import RERANK_METHODS

logger = logging.getLogger("prismflow.v2.angles")

#: Shipped defaults. Callers may point ``load_config`` elsewhere.
DEFAULT_CONFIG_PATH = Path("configs/v2/angle_defaults.yaml")

#: The connector slots an angle may name in ``fallback_strategy``.
SOURCE_SLOTS = ("primary", "secondary", "tertiary")

#: How ``ranked_records`` is cut to the token budget.
BUDGET_POLICIES = ("prefix", "greedy")

#: The five angles V2 defines.
ANGLE_NAMES = ("tech", "market", "financial", "regulatory", "sentiment")


@dataclass(frozen=True)
class AngleConfig:
    """Retrieval settings for one angle."""

    name: str
    k: int
    token_budget: int
    ttl_seconds: int
    rerank_method: str
    budget_policy: str
    fallback_strategy: Tuple[str, ...]

    def effective_ttl(self, connector_name: str) -> int:
        """Cache TTL to use for one connector under this angle.

        The stricter of the angle's freshness requirement and the connector's
        own default. See the module docstring for why this is a minimum.
        """
        return min(self.ttl_seconds, ttl_for(connector_name))


def load_config(
    path: Optional[Union[str, Path]] = None,
) -> Dict[str, AngleConfig]:
    """Load per-angle settings, validating every field.

    Args:
        path: YAML file to read. Defaults to ``configs/v2/angle_defaults.yaml``.

    Returns:
        ``{angle_name: AngleConfig}``.

    Raises:
        FileNotFoundError: the file is not there.
        ValueError: the file is malformed, or a value is out of range. Raised
            eagerly so a bad config fails once at startup rather than degrading
            silently on every query.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"angle config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)

    if not isinstance(document, Mapping) or "angles" not in document:
        raise ValueError(
            f"{config_path}: expected a mapping with a top-level 'angles' key"
        )
    angles = document["angles"]
    if not isinstance(angles, Mapping) or not angles:
        raise ValueError(f"{config_path}: 'angles' must be a non-empty mapping")

    loaded: Dict[str, AngleConfig] = {}
    for name, raw in angles.items():
        loaded[str(name)] = _parse_angle(config_path, str(name), raw)

    missing = [name for name in ANGLE_NAMES if name not in loaded]
    if missing:
        # A warning, not an error: a config that covers only the angles a
        # particular experiment uses is legitimate.
        logger.warning(
            "%s defines no settings for %s; those angles will need explicit "
            "arguments",
            config_path,
            ", ".join(missing),
        )
    return loaded


def _parse_angle(path: Path, name: str, raw: Any) -> AngleConfig:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{path}: angle {name!r} must be a mapping")

    k = _positive_int(path, name, raw, "k")
    token_budget = _positive_int(path, name, raw, "token_budget")
    ttl_seconds = _positive_int(path, name, raw, "ttl_seconds")

    rerank_method = str(raw.get("rerank_method", "bm25"))
    if rerank_method not in RERANK_METHODS:
        raise ValueError(
            f"{path}: angle {name!r} has rerank_method={rerank_method!r}; "
            f"expected one of {', '.join(RERANK_METHODS)}"
        )

    budget_policy = str(raw.get("budget_policy", "prefix"))
    if budget_policy not in BUDGET_POLICIES:
        raise ValueError(
            f"{path}: angle {name!r} has budget_policy={budget_policy!r}; "
            f"expected one of {', '.join(BUDGET_POLICIES)}"
        )

    strategy = raw.get("fallback_strategy", list(SOURCE_SLOTS))
    if isinstance(strategy, str) or not isinstance(strategy, Sequence):
        raise ValueError(
            f"{path}: angle {name!r} fallback_strategy must be a list of slot "
            f"names, got {strategy!r}"
        )
    slots = tuple(str(slot) for slot in strategy)
    if not slots:
        raise ValueError(
            f"{path}: angle {name!r} fallback_strategy is empty; an angle with "
            "no sources can never retrieve anything"
        )
    unknown = [slot for slot in slots if slot not in SOURCE_SLOTS]
    if unknown:
        raise ValueError(
            f"{path}: angle {name!r} names unknown slot(s) {unknown}; expected "
            f"only {', '.join(SOURCE_SLOTS)}"
        )
    if len(set(slots)) != len(slots):
        raise ValueError(
            f"{path}: angle {name!r} repeats a slot in fallback_strategy "
            f"({slots}); a source queried twice would double-count provenance"
        )

    return AngleConfig(
        name=name,
        k=k,
        token_budget=token_budget,
        ttl_seconds=ttl_seconds,
        rerank_method=rerank_method,
        budget_policy=budget_policy,
        fallback_strategy=slots,
    )


def _positive_int(path: Path, angle: str, raw: Mapping, key: str) -> int:
    if key not in raw:
        raise ValueError(f"{path}: angle {angle!r} is missing required key {key!r}")
    value = raw[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"{path}: angle {angle!r} {key}={value!r} must be an integer"
        )
    if value <= 0:
        raise ValueError(f"{path}: angle {angle!r} {key}={value} must be positive")
    return value


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "SOURCE_SLOTS",
    "BUDGET_POLICIES",
    "ANGLE_NAMES",
    "AngleConfig",
    "load_config",
]
