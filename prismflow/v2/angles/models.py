"""``AngleEvidence``: everything one angle found, and what it cost.

An ENGINEERING component per docs/CONTRACT.md section 3 -- no learnable
parameters. It is the unit Part 20 reasons over, Part 21 measures dependence
between, and Part 23 fuses, so it has to carry not just the records but enough
provenance to say *where the evidence came from* and enough warnings to say
where it did not.

WHY PROVENANCE IS PART OF THE EVIDENCE AND NOT A LOG LINE
--------------------------------------------------------
PrismFlow's thesis is that agreement counts in proportion to the effective
independence of the views that agree. An angle's independence is a property of
the sources behind it, and with a fallback chain those sources vary per query:
if GitHub satisfies k, the tech angle is GitHub-only; if GitHub is throttled,
the same angle is arXiv-only. Two angles that both fall through to the same
tertiary source are not independent at all, however different their names.
``provenance`` is what lets Part 21 see that, so it travels with the evidence
rather than being written to a log nobody reads.

THE INVARIANTS ARE CHECKED, NOT DOCUMENTED
------------------------------------------
``__post_init__`` enforces three. The brief specifies the first; the other two
exist because a budget or fallback bug is otherwise invisible -- the object
still looks well-formed, and the error only shows up as a strange number in
Part 23.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from prismflow.v2.connectors.base import NormalizedRecord


@dataclass
class AngleEvidence:
    """One angle's answer to one query.

    Attributes:
        query_text: the query as asked.
        angle_name: which angle produced this ("tech", "market", ...).
        raw_records: everything retrieved, in retrieval order, before ranking
            and before the budget was applied. Genuinely unranked: the
            reranker returns new records rather than mutating these, so a
            record's ``relevance_score`` here is still what the connector set.
        ranked_records: reranked and then cut to the token budget. This is the
            evidence the angle actually presents.
        total_tokens: sum of ``snippet_tokens`` over ``ranked_records``.
        provenance: accepted record count per connector name, e.g.
            ``{"github": 3, "arxiv": 2}``. Sums to ``len(raw_records)``.
        warnings: everything that degraded this answer -- a source that failed,
            a rerank that fell back, a budget that truncated. An empty list
            means the angle got what it asked for.
    """

    query_text: str
    angle_name: str
    raw_records: List[NormalizedRecord]
    ranked_records: List[NormalizedRecord]
    total_tokens: int
    provenance: Dict[str, int]
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        raw_ids = {(r.source, r.id) for r in self.raw_records}
        ranked_ids = {(r.source, r.id) for r in self.ranked_records}
        if not ranked_ids.issubset(raw_ids):
            missing = sorted(str(x) for x in ranked_ids - raw_ids)
            raise ValueError(
                f"{self.angle_name}: ranked records must be a subset of raw "
                f"records; {missing} are not in raw_records"
            )

        # A budget bug leaves a well-formed object with a wrong number in it,
        # which Part 23 would spend real tokens on before anyone noticed.
        counted = sum(r.snippet_tokens for r in self.ranked_records)
        if counted != self.total_tokens:
            raise ValueError(
                f"{self.angle_name}: total_tokens={self.total_tokens} but the "
                f"ranked records sum to {counted}"
            )

        # Provenance that does not add up means the fallback chain lost or
        # double-counted records, which would misstate this angle's
        # independence to Part 21.
        attributed = sum(self.provenance.values())
        if attributed != len(self.raw_records):
            raise ValueError(
                f"{self.angle_name}: provenance accounts for {attributed} "
                f"records but raw_records holds {len(self.raw_records)}"
            )

    # -- derived ---------------------------------------------------------

    @property
    def source_names(self) -> List[str]:
        """Connectors that contributed at least one record, in count order."""
        return [
            name
            for name, count in sorted(
                self.provenance.items(), key=lambda kv: (-kv[1], kv[0])
            )
            if count > 0
        ]

    @property
    def truncated(self) -> bool:
        """True if the budget cut records that ranking had kept."""
        return len(self.ranked_records) < len(self.raw_records)

    @property
    def is_empty(self) -> bool:
        return not self.raw_records

    def summary(self) -> str:
        """One line, for logs and the manual script."""
        sources = ", ".join(
            f"{name}={self.provenance[name]}" for name in self.source_names
        ) or "none"
        return (
            f"{self.angle_name}: {len(self.ranked_records)}/"
            f"{len(self.raw_records)} records, {self.total_tokens} tokens, "
            f"sources [{sources}], {len(self.warnings)} warning(s)"
        )


__all__ = ["AngleEvidence"]
