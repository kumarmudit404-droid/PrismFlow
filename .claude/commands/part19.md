# /part19 — Per-Angle Retrieval Pipeline (PrismFlow V2)

**Depends:** Part 17–18 (Connectors, Cache)  
**Estimate:** 6 hours  
**Difficulty:** Medium-High  
**Gate:** All 5 angles retrieve records; reranking applied; token budgets respected

---

## Context

PrismFlow V2 has five angles, each querying independent data sources:

| Angle | Primary Source | Secondary | Tertiary | Cache TTL |
|-------|---|---|---|---|
| **Tech** | GitHub (repos, code) | arXiv (papers) | Papers With Code | 1 hour |
| **Market** | NewsAPI (news) | Google Trends (trending) | CrunchBase (news) | 24 hours |
| **Financial** | Yahoo Finance (prices, earnings) | Alpha Vantage (technical) | SEC EDGAR (filings) | 6 hours |
| **Regulatory** | data.gov (datasets) | EUR-Lex (EU law) | RBI open data (India) | 1 week |
| **Sentiment** | PRAW (Reddit) | StackExchange API (Q&A) | — | 6 hours |

Part 19 orchestrates retrieval: query each angle, collect results, rerank by relevance (BM25 or cross-encoder), enforce token budget (max 2000 tokens per angle per query), return `AngleEvidence`.

---

## Goals

1. Define `AngleEvidence` dataclass:
   - query_text: str
   - angle_name: str
   - raw_records: List[NormalizedRecord] (unranked)
   - ranked_records: List[NormalizedRecord] (after reranking)
   - total_tokens: int (sum of snippet tokens in ranked_records up to budget)
   - provenance: Dict[str, int] (e.g., {"github": 3, "arxiv": 2})
   - warnings: List[str] (e.g., "NewsAPI rate limit hit; using cache", "truncated at 2000 tokens")

2. Create per-angle retrieval orchestrators (`tech_angle.py`, `market_angle.py`, etc.):
   - Try primary source; fall back to secondary, then tertiary
   - Handle rate limits (log, backoff, use cache)
   - Rerank results using BM25 (rank_bm25 library) or cross-encoder (sentence-transformers)
   - Accumulate tokens until budget exhausted
   - Return AngleEvidence

3. Write YAML config (`configs/v2/angle_defaults.yaml`):
   - Per-angle k (default results requested from each connector)
   - Per-angle TTL
   - Per-angle token budget (hard limit)
   - Reranking method (bm25 or cross-encoder)
   - Fallback strategy

4. Tests:
   - Each angle retrieval works (mock + live)
   - Reranking changes order
   - Token budget enforced
   - Fallback triggered on primary source error
   - Provenance tracking accurate

---

## Specifications

### `AngleEvidence` Model

```python
# prismflow/v2/angles/models.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from prismflow.v2.connectors.base import NormalizedRecord

@dataclass
class AngleEvidence:
    query_text: str
    angle_name: str
    raw_records: List[NormalizedRecord]
    ranked_records: List[NormalizedRecord]
    total_tokens: int
    provenance: Dict[str, int]  # {"github": 3, "arxiv": 2, ...}
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate: ranked_records should be subset of raw_records (by id)."""
        raw_ids = {r.id for r in self.raw_records}
        ranked_ids = {r.id for r in self.ranked_records}
        if not ranked_ids.issubset(raw_ids):
            raise ValueError("Ranked records must be subset of raw records")
```

### Angle Retrieval Orchestrators

```python
# prismflow/v2/angles/tech_angle.py

import logging
from typing import List, Dict
import asyncio
from prismflow.v2.connectors.base import AngleConnector, NormalizedRecord
from prismflow.v2.angles.models import AngleEvidence
from prismflow.v2.angles.reranker import rerank_records

logger = logging.getLogger(__name__)

class TechAngle:
    def __init__(
        self,
        primary: AngleConnector,      # GitHubConnector
        secondary: AngleConnector,    # ArxivConnector
        tertiary: Optional[AngleConnector] = None,  # PapersWithCodeConnector (stub)
        k: int = 10,
        token_budget: int = 2000,
        rerank_method: str = "bm25"  # "bm25" or "cross-encoder"
    ):
        self.primary = primary
        self.secondary = secondary
        self.tertiary = tertiary
        self.k = k
        self.token_budget = token_budget
        self.rerank_method = rerank_method
    
    async def retrieve(self, query: str) -> AngleEvidence:
        """Retrieve tech evidence for query."""
        all_records: List[NormalizedRecord] = []
        provenance: Dict[str, int] = {}
        warnings: List[str] = []
        
        # Try primary (GitHub)
        try:
            logger.info(f"[TECH] Querying primary (GitHub): {query}")
            records = await self.primary.get_records(query, self.k)
            all_records.extend(records)
            provenance["github"] = len(records)
        except Exception as e:
            logger.warning(f"[TECH] Primary source failed: {e}; trying secondary")
            warnings.append(f"GitHub fetch failed: {type(e).__name__}")
        
        # Try secondary (arXiv) if primary didn't return enough
        if len(all_records) < self.k:
            try:
                logger.info(f"[TECH] Querying secondary (arXiv): {query}")
                records = await self.secondary.get_records(query, self.k - len(all_records))
                all_records.extend(records)
                provenance["arxiv"] = len(records)
            except Exception as e:
                logger.warning(f"[TECH] Secondary source failed: {e}")
                warnings.append(f"arXiv fetch failed: {type(e).__name__}")
        
        # Rerank all records by query relevance
        ranked_records = rerank_records(
            query, all_records, method=self.rerank_method
        )
        
        # Enforce token budget
        total_tokens = 0
        budget_records = []
        for record in ranked_records:
            if total_tokens + record.snippet_tokens > self.token_budget:
                warnings.append(
                    f"Token budget {self.token_budget} reached; truncated to {len(budget_records)} records"
                )
                break
            budget_records.append(record)
            total_tokens += record.snippet_tokens
        
        return AngleEvidence(
            query_text=query,
            angle_name="tech",
            raw_records=all_records,
            ranked_records=budget_records,
            total_tokens=total_tokens,
            provenance=provenance,
            warnings=warnings
        )
```

Similar orchestrators for `market_angle.py`, `financial_angle.py`, `regulatory_angle.py`, `sentiment_angle.py`.

### Reranking Functions

```python
# prismflow/v2/angles/reranker.py

from typing import List
from prismflow.v2.connectors.base import NormalizedRecord
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)

def rerank_records(
    query: str,
    records: List[NormalizedRecord],
    method: str = "bm25"
) -> List[NormalizedRecord]:
    """
    Rerank records by relevance to query.
    
    Args:
        query: User query
        records: List of NormalizedRecord
        method: "bm25" or "cross-encoder"
    
    Returns:
        Records sorted by descending relevance score
    """
    if not records:
        return []
    
    if method == "bm25":
        return _rerank_bm25(query, records)
    elif method == "cross-encoder":
        return _rerank_cross_encoder(query, records)
    else:
        logger.warning(f"Unknown rerank method {method}; returning unsorted")
        return records

def _rerank_bm25(query: str, records: List[NormalizedRecord]) -> List[NormalizedRecord]:
    """BM25 reranking (lightweight, no model needed)."""
    # Tokenize snippets
    corpus = [r.title.split() + r.snippet_tokens.split() for r in records]
    
    bm25 = BM25Okapi(corpus)
    query_tokens = query.split()
    
    scores = bm25.get_scores(query_tokens)
    
    # Pair records with scores, sort by score desc, update relevance_score
    scored = list(zip(records, scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # Return records with updated relevance_score
    reranked = []
    for i, (record, score) in enumerate(scored):
        # Normalize BM25 score to [0, 1]
        normalized_score = min(1.0, score / max(scores + [1e-6]))
        record.relevance_score = normalized_score
        reranked.append(record)
    
    return reranked

def _rerank_cross_encoder(query: str, records: List[NormalizedRecord]) -> List[NormalizedRecord]:
    """Cross-encoder reranking (heavier, more accurate)."""
    from sentence_transformers import CrossEncoder
    
    model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # Prepare pairs: [query, record.title + " " + record.snippet]
    pairs = [
        [query, f"{r.title} {r.snippet_tokens}"]
        for r in records
    ]
    
    scores = model.predict(pairs)
    
    # Normalize to [0, 1]
    min_score, max_score = scores.min(), scores.max()
    normalized_scores = (scores - min_score) / (max_score - min_score + 1e-6)
    
    scored = list(zip(records, normalized_scores))
    scored.sort(key=lambda x: x[1], reverse=True)
    
    reranked = []
    for record, score in scored:
        record.relevance_score = float(score)
        reranked.append(record)
    
    return reranked
```

### YAML Config

```yaml
# configs/v2/angle_defaults.yaml

angles:
  tech:
    k: 10                        # Default k results from connectors
    token_budget: 2000
    ttl_seconds: 3600
    rerank_method: bm25          # bm25 or cross-encoder
    fallback_strategy: [primary, secondary, tertiary]
  
  market:
    k: 8
    token_budget: 1500
    ttl_seconds: 86400
    rerank_method: bm25
    fallback_strategy: [primary, secondary]
  
  financial:
    k: 10
    token_budget: 2000
    ttl_seconds: 21600
    rerank_method: bm25
    fallback_strategy: [primary, secondary]
  
  regulatory:
    k: 5
    token_budget: 1000
    ttl_seconds: 604800
    rerank_method: cross-encoder
    fallback_strategy: [primary, secondary]
  
  sentiment:
    k: 15
    token_budget: 2500
    ttl_seconds: 21600
    rerank_method: bm25
    fallback_strategy: [primary, secondary]
```

---

## Deliverables

### File structure
```
prismflow/v2/
├── angles/
│   ├── __init__.py
│   ├── models.py             # AngleEvidence
│   ├── base.py               # BaseAngle ABC
│   ├── tech_angle.py         # TechAngle orchestrator
│   ├── market_angle.py       # MarketAngle
│   ├── financial_angle.py    # FinancialAngle
│   ├── regulatory_angle.py   # RegulatoryAngle
│   ├── sentiment_angle.py    # SentimentAngle
│   └── reranker.py           # BM25 + cross-encoder reranking

configs/v2/
├── angle_defaults.yaml       # Per-angle hyperparameters

tests/v2/
├── test_angles.py            # Per-angle retrieval tests
```

### Tests to write

1. **test_tech_angle_retrieve()** — Mock GitHub + arXiv, verify AngleEvidence returned with provenance
2. **test_rerank_bm25()** — Verify BM25 reranking changes order; top record has highest relevance_score
3. **test_rerank_cross_encoder()** — Same for cross-encoder (optional, expensive)
4. **test_token_budget_enforcement()** — Insert records with tokens [500, 800, 900, 500], budget 2000, verify only first 3 included
5. **test_fallback_on_primary_failure()** — Mock primary error, verify secondary called
6. **test_provenance_tracking()** — Verify provenance dict sums to total records
7. **test_all_five_angles()** — Call all 5 angles with same query, verify 5 AngleEvidence objects returned

---

## Standing Rules

1. **No V1 modifications.**
2. **Fallback chaining.** If primary fails, try secondary; if secondary fails, tertiary.
3. **Token counting.** Use existing `snippet_tokens` field from NormalizedRecord (set in Part 17).
4. **Reranking optional.** If rerank fails, return unsorted; log warning but don't raise.
5. **Async.** Orchestrators must be async to enable concurrent angle retrieval in Part 23.

---

## Success Criteria

- [ ] All 5 angle orchestrators implemented
- [ ] BM25 reranking works; changes record order
- [ ] Token budget strictly enforced
- [ ] Fallback chain triggers on source failures
- [ ] Provenance tracking accurate
- [ ] All tests pass: `pytest tests/v2/test_angles.py -v`
- [ ] YAML config loaded and used
- [ ] Ready for Part 20

---

## Checklist for Closing

- [ ] All 5 angle classes inherit from BaseAngle
- [ ] Config loaded: `load_config("configs/v2/angle_defaults.yaml")`
- [ ] Manual test: call all 5 angles with query "machine learning", log AngleEvidence
- [ ] Tests pass: `pytest tests/v2/test_angles.py -v`
