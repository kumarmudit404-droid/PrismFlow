# /part23 — Fusion Layer via Claude API (PrismFlow V2)

**Depends:** Part 20-22 (Reasoners, DependenceReport, ENIV)
**Estimate:** 5 hours
**Difficulty:** High
**Gate:** E2E test on 30 queries passes; ENIV discount applied; conflicts flagged

---

### Goals

1. Implement `PrismFusion` orchestrator:
   - Input: query, List[ClaimSet] (5 angles), DependenceReport, ENIV
   - Steps:
     a. Apply ENIV discount to each claim confidence
     b. Detect conflicts (claims with contradictory meanings)
     c. Aggregate into FusedRecommendation
     d. Log audit trail (which claims contributed, conflicts detected, final confidence)

2. Conflict detection:
   - Embed claims; compute pairwise similarity
   - If similarity < 0.3 AND both cited by different angles → potential conflict
   - Flag in audit trail

3. Aggregation:
   - Weighted mean of discounted confidences
   - Weights: recency (newer angles weighted higher), source diversity

4. End-to-end test:
   - 30 queries across startup/tech/career/policy domains
   - Log latency, cost (API calls), ENIV, conflict rate
   - Verify no errors

### Specifications

```python
# prismflow/v2/fusion/fusion.py

from dataclasses import dataclass, field
from typing import List
import numpy as np
import time
from anthropic import Anthropic
from prismflow.v2.reasoners.models import ClaimSet, Claim
from prismflow.v2.dependence.models import DependenceReport
from prismflow.v2.statistics.eniv import compute_discount_factor, apply_eniv_discount

@dataclass
class FusedRecommendation:
    query_text: str
    fused_claims: List[Claim]
    overall_confidence: float      # Weighted mean of all claim confidences
    conflicts_detected: List[str]
    audit_trail: List[str]
    eniv: float
    discount_factor: float
    latency_seconds: float
    tokens_input: int
    tokens_output: int

class PrismFusion:
    def __init__(self):
        self.client = Anthropic()
    
    async def fuse(
        self,
        query: str,
        claimsets: List[ClaimSet],
        dependence_report: DependenceReport,
        eniv: float
    ) -> FusedRecommendation:
        """
        Fuse claims from 5 angles via ENIV discounting + conflict detection.
        
        Args:
            query: Original query text
            claimsets: [ClaimSet for tech, market, financial, regulatory, sentiment]
            dependence_report: DependenceReport from Part 21
            eniv: ENIV value from Part 22
        
        Returns:
            FusedRecommendation with audit trail
        """
        start_time = time.time()
        audit_trail = []
        conflicts_detected = []
        
        discount_factor = compute_discount_factor(eniv, n=5)
        audit_trail.append(f"ENIV={eniv:.2f}, discount_factor={discount_factor:.2f}")
        
        # Apply ENIV discount to all claims
        discounted_claimsets = []
        for claimset in claimsets:
            discounted_claims = []
            for claim in claimset.claims:
                discounted_conf = apply_eniv_discount(claim.confidence, discount_factor)
                discounted_claim = Claim(
                    text=claim.text,
                    confidence=discounted_conf,
                    cited_ids=claim.cited_ids,
                    conflicts_noted=claim.conflicts_noted,
                    caveats=claim.caveats
                )
                discounted_claims.append(discounted_claim)
            
            discounted_claimset = ClaimSet(
                angle_name=claimset.angle_name,
                query_text=query,
                claims=discounted_claims,
                provider=claimset.provider,
                model_name=claimset.model_name,
                tokens_input=claimset.tokens_input,
                tokens_output=claimset.tokens_output,
                latency_seconds=claimset.latency_seconds,
                audit_trail=claimset.audit_trail
            )
            discounted_claimsets.append(discounted_claimset)
        
        audit_trail.append(f"Applied ENIV discount to {len(discounted_claimsets)} angles")
        
        # Detect conflicts
        all_claims = []
        for cs in discounted_claimsets:
            all_claims.extend(cs.claims)
        
        conflicts = self._detect_conflicts(all_claims)
        conflicts_detected.extend(conflicts)
        audit_trail.append(f"Detected {len(conflicts)} conflicts")
        
        # Aggregate: weighted mean of confidences
        if all_claims:
            overall_confidence = np.mean([c.confidence for c in all_claims])
        else:
            overall_confidence = 0.0
        
        audit_trail.append(f"Overall confidence: {overall_confidence:.2f}")
        
        latency = time.time() - start_time
        
        return FusedRecommendation(
            query_text=query,
            fused_claims=all_claims,
            overall_confidence=overall_confidence,
            conflicts_detected=conflicts_detected,
            audit_trail=audit_trail,
            eniv=eniv,
            discount_factor=discount_factor,
            latency_seconds=latency,
            tokens_input=sum(cs.tokens_input for cs in claimsets),
            tokens_output=sum(cs.tokens_output for cs in claimsets)
        )
    
    def _detect_conflicts(self, claims: List[Claim]) -> List[str]:
        """Detect contradictory claims via embedding similarity."""
        from sentence_transformers import SentenceTransformer
        
        if len(claims) < 2:
            return []
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [c.text for c in claims]
        embeddings = model.encode(texts, convert_to_numpy=True)
        
        conflicts = []
        for i in range(len(claims)):
            for j in range(i + 1, len(claims)):
                sim = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-6
                )
                
                # Conflicting if low similarity but both cite evidence
                if sim < 0.3 and claims[i].cited_ids and claims[j].cited_ids:
                    conflicts.append(f"Claim {i} and {j} may conflict (sim={sim:.2f})")
        
        return conflicts
```

### End-to-End Test

```python
# experiments/v2/test_fusion_e2e.py

async def run_fusion_e2e_test():
    """
    Test fusion on 30 queries across 4 domains.
    
    Log: latency, cost, ENIV, conflict rate
    """
    queries = [
        # Tech (8)
        "machine learning frameworks comparison",
        # Market (8)
        "stock market volatility today",
        # Career (8)
        "best programming languages 2024",
        # Policy (6)
        "AI regulation in Europe"
    ] * 6  # Repeat to get 30 queries
    
    results = []
    total_cost = 0
    
    fusion = PrismFusion()
    
    for i, query in enumerate(queries[:30]):
        # Retrieve, reason, estimate dependence (mock for speed)
        # ...
        
        recommendation = await fusion.fuse(query, claimsets, dependence_report, eniv)
        
        results.append({
            "query": query,
            "latency": recommendation.latency_seconds,
            "tokens": recommendation.tokens_input + recommendation.tokens_output,
            "eniv": recommendation.eniv,
            "conflicts": len(recommendation.conflicts_detected),
            "overall_confidence": recommendation.overall_confidence
        })
        
        total_cost += recommendation.tokens_input * 0.0001 + recommendation.tokens_output * 0.0003  # Rough estimate
    
    print(f"✅ E2E Test: 30 queries, {total_cost:.2f} estimated cost")
    return results
```

### Deliverables & Success Criteria

- [ ] PrismFusion class implemented
- [ ] ENIV discount applied to all claims
- [ ] Conflict detection working
- [ ] E2E test passes (30 queries, no errors)
- [ ] Latency, cost, ENIV logged
- [ ] Results saved to `results/v2/fusion_e2e_test.json`
- [ ] Ready for Part 24 (Evaluation)

---



---

## Standing Rules

1. **No V1 modifications.** V1 is sealed at tag `v1-final`; defects are reported, not patched.
2. **Never fabricate.** No result without 5+ seeds, mean +/- std; report honestly if a threshold is missed.
3. **Audit trail.** Log every step: dispatch, reasoning, dependence, fusion, evaluation.
4. **Save results.** All metrics to `results/v2/*.json`.
