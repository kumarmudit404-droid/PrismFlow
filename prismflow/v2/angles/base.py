"""``BaseAngle``: fallback chain, reranking, token budget, provenance.

An ENGINEERING component per docs/CONTRACT.md section 3.

WHY THE ALGORITHM LIVES HERE AND NOT IN FIVE FILES
--------------------------------------------------
The brief specifies ``TechAngle.retrieve`` in full and then says "similar
orchestrators for market_angle.py, financial_angle.py, ...". Copied five times,
the fallback chain, the budget loop and the provenance bookkeeping would be five
places for the same bug -- and the budget loop in particular is where the brief
already miscounts. Nothing in ``retrieve`` is angle-specific: angles differ only
in their name, their sources and their numbers. So ``retrieve`` is concrete here
and each angle subclass is a name plus defaults.

FALLBACK IS A CHAIN, AND IT CHANGES WHAT AN ANGLE *IS*
------------------------------------------------------
Sources are tried in ``fallback_strategy`` order until k records are in hand. A
source that raises contributes nothing, records a warning, and the chain moves
on -- so failure and insufficiency take the same path, and the tertiary source
is actually reached. (The brief's version stops after secondary: it never
mentions ``self.tertiary`` again after the constructor, so the tertiary
connector it accepts can never be queried.)

The consequence is worth stating plainly, because it bears on the thesis rather
than on this file. An angle's evidence composition depends on what the earlier
sources returned: if GitHub satisfies k, the tech angle is GitHub-only; if
GitHub is throttled, the same angle is arXiv-only. So an angle is not a fixed
view -- it is whichever sources answered. Two angles whose primaries both fail
into the same tertiary source are not independent at all, however different
their names. That is why ``provenance`` is part of AngleEvidence: Part 21
estimates dependence between angles and must be able to see when two of them
read the same source. It is a property this layer surfaces, not one it can fix.

CONCURRENCY
-----------
``retrieve`` is async (standing rule 5) but does not fan out *within* an angle:
the chain is inherently sequential, since whether the secondary is needed
depends on what the primary returned. Concurrency belongs across angles, where
Part 23 gathers five retrievals at once -- which is safe because Part 18's
cache collapses duplicate concurrent fetches per key.

WHY ``except Exception`` IS NOT USED
------------------------------------
The brief catches bare ``Exception`` per source and turns it into a "source
failed" warning. That would have swallowed its own reranker bug: the
``AttributeError`` from ``snippet_tokens.split()`` would surface as "GitHub
fetch failed: AttributeError" and every angle would quietly return unranked
evidence, blaming the network for a defect in the code. Only Part 17's
``ConnectorError`` hierarchy and timeouts are caught here; a programming error
propagates and gets fixed.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC
from typing import Awaitable, Callable, Dict, List, Optional, Sequence, Union
from pathlib import Path

from prismflow.v2.cache import CacheMetrics, wrap_connector
from prismflow.v2.connectors.base import AngleConnector, NormalizedRecord
from prismflow.v2.connectors.errors import ConnectorError

from .config import AngleConfig, SOURCE_SLOTS
from .models import AngleEvidence
from .reranker import rerank_with_outcome

logger = logging.getLogger("prismflow.v2.angles")

Fetcher = Callable[[str, int], Awaitable[List[NormalizedRecord]]]


class BaseAngle(ABC):
    """One angle: a named fallback chain over connectors, with a budget.

    Subclasses set ``ANGLE_NAME`` and may set ``DEFAULT_K``,
    ``DEFAULT_TOKEN_BUDGET`` and ``DEFAULT_RERANK_METHOD``. Everything else is
    shared.

    Connectors are injected rather than constructed here. Part 17 built two of
    them (GitHub, arXiv); the remaining sources in the Part 19 brief's table do
    not exist yet, so the four angles without connectors are wired and testable
    and will return empty evidence with an explicit warning until a connector is
    passed in. Adding one later needs no change to this file or to any angle.
    """

    ANGLE_NAME: str = "base"
    DEFAULT_K: int = 10
    DEFAULT_TOKEN_BUDGET: int = 2000
    DEFAULT_RERANK_METHOD: str = "bm25"

    def __init__(
        self,
        primary: Optional[AngleConnector] = None,
        secondary: Optional[AngleConnector] = None,
        tertiary: Optional[AngleConnector] = None,
        *,
        k: Optional[int] = None,
        token_budget: Optional[int] = None,
        rerank_method: Optional[str] = None,
        budget_policy: Optional[str] = None,
        fallback_strategy: Optional[Sequence[str]] = None,
        config: Optional[AngleConfig] = None,
        use_cache: bool = True,
        cache_db_path: Optional[Union[str, Path]] = None,
        cache_metrics: Optional[CacheMetrics] = None,
    ) -> None:
        """Build an angle.

        Explicit arguments win over ``config``, which wins over the class
        defaults, so a config file sets the policy and a caller can still
        override one number for an experiment without editing YAML.
        """
        self.sources: Dict[str, Optional[AngleConnector]] = {
            "primary": primary,
            "secondary": secondary,
            "tertiary": tertiary,
        }
        self.config = config
        self.k = _pick(k, config.k if config else None, self.DEFAULT_K)
        self.token_budget = _pick(
            token_budget,
            config.token_budget if config else None,
            self.DEFAULT_TOKEN_BUDGET,
        )
        self.rerank_method = _pick(
            rerank_method,
            config.rerank_method if config else None,
            self.DEFAULT_RERANK_METHOD,
        )
        self.budget_policy = _pick(
            budget_policy, config.budget_policy if config else None, "prefix"
        )
        self.fallback_strategy = tuple(
            _pick(
                tuple(fallback_strategy) if fallback_strategy else None,
                config.fallback_strategy if config else None,
                SOURCE_SLOTS,
            )
        )
        if self.k <= 0:
            raise ValueError(f"{self.ANGLE_NAME}: k must be positive, got {self.k}")
        if self.token_budget <= 0:
            raise ValueError(
                f"{self.ANGLE_NAME}: token_budget must be positive, got "
                f"{self.token_budget}"
            )

        self.use_cache = use_cache
        self._fetchers: Dict[str, Fetcher] = {}
        if use_cache:
            for slot, connector in self.sources.items():
                if connector is None:
                    continue
                ttl = (
                    self.config.effective_ttl(connector.name)
                    if self.config is not None
                    else None
                )
                self._fetchers[slot] = wrap_connector(
                    connector,
                    ttl_seconds=ttl,
                    db_path=cache_db_path,
                    metrics=cache_metrics,
                )

    # -- retrieval -------------------------------------------------------

    async def retrieve(self, query: str) -> AngleEvidence:
        """Retrieve, rerank and budget this angle's evidence for ``query``."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError(
                f"{self.ANGLE_NAME}: query must be a non-empty string, got "
                f"{query!r}"
            )

        raw_records: List[NormalizedRecord] = []
        provenance: Dict[str, int] = {}
        warnings: List[str] = []
        seen: set = set()

        for slot in self.fallback_strategy:
            if len(raw_records) >= self.k:
                break
            connector = self.sources.get(slot)
            if connector is None:
                continue
            remaining = self.k - len(raw_records)
            logger.info(
                "[%s] querying %s (%s) for %r, k=%d",
                self.ANGLE_NAME, slot, connector.name, query, remaining,
            )
            try:
                fetched = await self._fetch(slot, connector, query, remaining)
            except (ConnectorError, asyncio.TimeoutError) as exc:
                warning = (
                    f"{connector.name} ({slot}) failed: {type(exc).__name__}: "
                    f"{exc}"
                )
                logger.warning("[%s] %s", self.ANGLE_NAME, warning)
                warnings.append(warning)
                provenance.setdefault(connector.name, 0)
                continue

            accepted = 0
            for record in fetched:
                key = (record.source, record.id)
                if key in seen:
                    continue
                seen.add(key)
                raw_records.append(record)
                accepted += 1
            provenance[connector.name] = provenance.get(connector.name, 0) + accepted
            if accepted < len(fetched):
                warnings.append(
                    f"{connector.name} ({slot}) returned "
                    f"{len(fetched) - accepted} duplicate record(s), dropped"
                )

        configured = [
            slot for slot in self.fallback_strategy if self.sources.get(slot)
        ]
        if not configured:
            warning = (
                f"no connector is configured for the {self.ANGLE_NAME} angle "
                f"(slots tried: {', '.join(self.fallback_strategy)}); returning "
                "empty evidence"
            )
            logger.warning("[%s] %s", self.ANGLE_NAME, warning)
            warnings.append(warning)
        elif not raw_records:
            warnings.append(
                f"no records retrieved from any of {len(configured)} configured "
                f"source(s)"
            )

        outcome = rerank_with_outcome(query, raw_records, self.rerank_method)
        if outcome.warning:
            warnings.append(outcome.warning)
        elif outcome.all_zero and raw_records:
            warnings.append(
                f"no {self.rerank_method} term overlap between the query and "
                "any record; retrieval order preserved"
            )

        ranked, total_tokens, budget_warnings = self._apply_budget(outcome.records)
        warnings.extend(budget_warnings)

        evidence = AngleEvidence(
            query_text=query,
            angle_name=self.ANGLE_NAME,
            raw_records=raw_records,
            ranked_records=ranked,
            total_tokens=total_tokens,
            provenance=provenance,
            warnings=warnings,
        )
        logger.info("[%s] %s", self.ANGLE_NAME, evidence.summary())
        return evidence

    async def _fetch(
        self,
        slot: str,
        connector: AngleConnector,
        query: str,
        k: int,
    ) -> List[NormalizedRecord]:
        """Fetch through the Part 18 cache when caching is on."""
        fetcher = self._fetchers.get(slot)
        if fetcher is not None:
            return list(await fetcher(query, k))
        return list(await connector.get_records(query, k))

    def _apply_budget(
        self,
        ranked: Sequence[NormalizedRecord],
    ) -> tuple:
        """Cut ``ranked`` to ``token_budget``. Returns (kept, total, warnings).

        Two policies, because the brief specifies contradictory behaviour: its
        code breaks at the first overflowing record, while its test asserts that
        ``[500, 800, 900, 500]`` under a 2000 budget keeps "the first 3" -- which
        that code cannot do, since 500+800+900 = 2200. Breaking keeps 2 records
        (1300 tokens); skipping the 900 and taking the last 500 keeps 3 (1800).

        ``prefix`` (default) breaks. It costs budget utilisation but keeps a
        property the rest of the pipeline reads as true: ranked_records is the
        top-n of the ranking, contiguous. ``greedy`` skips and keeps filling,
        which uses more of the budget but means a record can be absent while a
        lower-ranked one is present.

        OVERSIZED RECORDS ARE DROPPED FIRST, AND THAT IS NOT A DETAIL
        ------------------------------------------------------------
        Measured on live GitHub results for "machine learning": the ``funNLP``
        repository's description is 5835 characters -- 5065 tokens, two and a
        half times the entire 2000-token tech budget -- while the other nine
        results in the same response totalled 150 tokens.

        A record that is larger than the whole budget can never be included at
        any position under any policy, so it is not a budgeting question. Left
        in the queue it would, under ``prefix``, truncate the angle wherever it
        happened to rank: at rank 10 it cost one record and left 1850 tokens
        unspent, and at rank 1 it would have returned an empty angle with the
        entire budget unused -- indistinguishable from a retrieval failure.
        Removing it first and warning specifically keeps the prefix property
        over the records that could actually be used, and names the real cause.
        """
        admissible: List[NormalizedRecord] = []
        oversized: List[NormalizedRecord] = []
        for record in ranked:
            if record.snippet_tokens > self.token_budget:
                oversized.append(record)
            else:
                admissible.append(record)

        kept: List[NormalizedRecord] = []
        total = 0
        dropped = 0
        for record in admissible:
            tokens = max(0, record.snippet_tokens)
            if total + tokens > self.token_budget:
                if self.budget_policy == "prefix":
                    dropped += len(admissible) - len(kept)
                    break
                dropped += 1
                continue
            kept.append(record)
            total += tokens

        warnings: List[str] = []
        if oversized:
            largest = max(r.snippet_tokens for r in oversized)
            warnings.append(
                f"{len(oversized)} record(s) exceed the entire "
                f"{self.token_budget}-token budget on their own and can never "
                f"be included (largest {largest} tokens: "
                f"{oversized[0].title[:40]!r}); dropped before budgeting"
            )
        if dropped:
            warnings.append(
                f"token budget {self.token_budget} filled at {total} tokens; "
                f"dropped {dropped} lower-ranked record(s) under the "
                f"{self.budget_policy} policy"
            )
        return kept, total, warnings

    # -- introspection ---------------------------------------------------

    @property
    def configured_sources(self) -> List[str]:
        """Connector names actually available, in fallback order."""
        return [
            self.sources[slot].name
            for slot in self.fallback_strategy
            if self.sources.get(slot) is not None
        ]

    async def aclose(self) -> None:
        """Release every connector's transport."""
        for connector in self.sources.values():
            if connector is not None:
                await connector.aclose()

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(sources={self.configured_sources}, "
            f"k={self.k}, token_budget={self.token_budget}, "
            f"rerank={self.rerank_method})"
        )


def _pick(*candidates):
    """First non-None candidate. Explicit beats config beats class default."""
    for candidate in candidates:
        if candidate is not None:
            return candidate
    return None


__all__ = ["BaseAngle", "Fetcher"]
