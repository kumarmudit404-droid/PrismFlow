# /part21 — Semantic Dependence Estimation (PrismFlow V2)

**Depends:** Part 20 (ClaimSet)  
**Estimate:** 8 hours  
**Difficulty:** High  
**Gate:** Three estimators agree on ranking; planted-redundancy experiment passes

---

## Context

PrismFlow V2 measures **semantic dependence** between angles — how much claims from one angle encode the same information as another. This is critical for ENIV discount (Part 22).

Three independent estimators:
1. **Embedding Cosine Similarity** — embed each claim; measure angle pairwise similarity
2. **Citation Overlap (Jaccard)** — how much do angles cite the same evidence?
3. **Mutual Information (TF-IDF)** — statistical dependence of claim text vocabularies

Weighted aggregation: 0.5×embedding + 0.3×citation + 0.2×MI

---

## Goals

1. Define `DependenceReport` dataclass:
   - angle_names: List[str]
   - n_angles: int
   - dependence_matrix: np.ndarray (symmetric, [0, 1], diagonal=1)
   - per_estimator: Dict[str, np.ndarray] (embedding, citation, mi)
   - warnings: List[str] (e.g., "low-rank embedding", "no citation overlap")

2. Implement three estimators:
   - `embedding_similarity(claims_by_angle)` — SentenceTransformer embeddings
   - `citation_overlap(claims_by_angle)` — Jaccard(cited_ids)
   - `mutual_information(claims_by_angle)` — TF-IDF + entropy

3. Aggregate into dependence matrix:
   - Average the three (weighted)
   - Symmetrize
   - Validate: diagonal=1, off-diag in [0, 1]

4. **Planted-redundancy experiment:**
   - 20 queries × 4 conditions (identical, paraphrase, unrelated, low-rank) × 5 seeds
   - Condition 1: identical claims → dependence > 0.95
   - Condition 2: paraphrased claims → dependence 0.6–0.95
   - Condition 3: unrelated claims → dependence < 0.3
   - Condition 4: low-rank embedding (truncated) → verify graceful degradation
   - Report: mean +/- std for each condition; check separability

5. Tests: each estimator in isolation, aggregation, edge cases

---

## Specifications

### `DependenceReport` Model

```python
# prismflow/v2/dependence/models.py

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np

@dataclass
class DependenceReport:
    angle_names: List[str]
    n_angles: int
    dependence_matrix: np.ndarray  # (n, n), symmetric, diag=1
    per_estimator: Dict[str, np.ndarray] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        assert self.dependence_matrix.shape == (self.n_angles, self.n_angles)
        assert np.allclose(self.dependence_matrix.diagonal(), 1.0)
        assert np.allclose(self.dependence_matrix, self.dependence_matrix.T)
        assert (self.dependence_matrix >= 0).all() and (self.dependence_matrix <= 1).all()
```

### Embedding Similarity Estimator

```python
# prismflow/v2/dependence/estimators.py

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from prismflow.v2.reasoners.models import Claim, ClaimSet
import logging

logger = logging.getLogger(__name__)

class EmbeddingEstimator:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def estimate(self, claims_by_angle: Dict[str, List[Claim]]) -> np.ndarray:
        """
        Compute pairwise embedding cosine similarity.
        
        Args:
            claims_by_angle: {"tech": [Claim, ...], "market": [...], ...}
        
        Returns:
            Symmetric (n, n) similarity matrix, diag=1
        """
        angle_names = list(claims_by_angle.keys())
        n = len(angle_names)
        matrix = np.eye(n)
        
        # Embed all claims per angle
        angle_embeddings = {}
        for angle, claims in claims_by_angle.items():
            texts = [c.text for c in claims]
            if not texts:
                angle_embeddings[angle] = np.zeros((1, 384))  # Fallback
                logger.warning(f"No claims for angle {angle}; using zero embedding")
                continue
            
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            # Average embedding per angle
            angle_embeddings[angle] = embeddings.mean(axis=0)
        
        # Compute pairwise cosine similarity
        for i, angle_i in enumerate(angle_names):
            for j, angle_j in enumerate(angle_names):
                if i >= j:
                    continue
                
                emb_i = angle_embeddings[angle_i]
                emb_j = angle_embeddings[angle_j]
                
                # Cosine similarity
                sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-6)
                sim = max(0.0, min(1.0, sim))  # Clamp to [0, 1]
                
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        return matrix

class CitationEstimator:
    def estimate(self, claims_by_angle: Dict[str, List[Claim]]) -> np.ndarray:
        """
        Compute pairwise citation overlap (Jaccard index).
        
        Args:
            claims_by_angle: {"tech": [Claim, ...], ...}
        
        Returns:
            Symmetric (n, n) Jaccard matrix, diag=1
        """
        angle_names = list(claims_by_angle.keys())
        n = len(angle_names)
        matrix = np.eye(n)
        
        # Collect cited IDs per angle
        angle_citations = {}
        for angle, claims in claims_by_angle.items():
            cited_ids = set()
            for claim in claims:
                cited_ids.update(claim.cited_ids)
            angle_citations[angle] = cited_ids
        
        # Compute Jaccard index
        for i, angle_i in enumerate(angle_names):
            for j, angle_j in enumerate(angle_names):
                if i >= j:
                    continue
                
                ids_i = angle_citations[angle_i]
                ids_j = angle_citations[angle_j]
                
                if len(ids_i) == 0 and len(ids_j) == 0:
                    jaccard = 1.0
                else:
                    intersection = len(ids_i & ids_j)
                    union = len(ids_i | ids_j)
                    jaccard = intersection / union if union > 0 else 0.0
                
                matrix[i, j] = jaccard
                matrix[j, i] = jaccard
        
        return matrix

class MutualInformationEstimator:
    def estimate(self, claims_by_angle: Dict[str, List[Claim]]) -> np.ndarray:
        """
        Compute pairwise mutual information via TF-IDF.
        
        Args:
            claims_by_angle: {"tech": [Claim, ...], ...}
        
        Returns:
            Symmetric (n, n) MI-based matrix, diag=1
        """
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        angle_names = list(claims_by_angle.keys())
        n = len(angle_names)
        
        # Concatenate all claim texts per angle
        angle_texts = []
        for angle in angle_names:
            claims = claims_by_angle[angle]
            text = " ".join(c.text for c in claims) if claims else ""
            angle_texts.append(text)
        
        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(angle_texts).toarray()
        except ValueError:
            # Not enough documents or features
            logger.warning("TF-IDF vectorization failed; using zero MI matrix")
            return np.eye(n)
        
        # Cosine similarity on TF-IDF vectors (proxy for MI)
        matrix = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                vec_i = tfidf_matrix[i]
                vec_j = tfidf_matrix[j]
                
                sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j) + 1e-6)
                sim = max(0.0, min(1.0, sim))
                
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        return matrix
```

### Aggregation

```python
# prismflow/v2/dependence/aggregator.py

import numpy as np
from typing import Dict, List
from prismflow.v2.dependence.models import DependenceReport
from prismflow.v2.dependence.estimators import EmbeddingEstimator, CitationEstimator, MutualInformationEstimator
from prismflow.v2.reasoners.models import Claim, ClaimSet

def aggregate_dependence(
    claimsets: List[ClaimSet],
    weights: Dict[str, float] = None
) -> DependenceReport:
    """
    Aggregate three estimators into dependence matrix.
    
    Args:
        claimsets: List[ClaimSet], one per angle
        weights: {"embedding": 0.5, "citation": 0.3, "mi": 0.2}
    
    Returns:
        DependenceReport with aggregated matrix
    """
    weights = weights or {"embedding": 0.5, "citation": 0.3, "mi": 0.2}
    
    # Convert ClaimSet → Dict[angle, List[Claim]]
    claims_by_angle = {cs.angle_name: cs.claims for cs in claimsets}
    angle_names = [cs.angle_name for cs in claimsets]
    n = len(claimsets)
    
    # Compute three matrices
    embedding_matrix = EmbeddingEstimator().estimate(claims_by_angle)
    citation_matrix = CitationEstimator().estimate(claims_by_angle)
    mi_matrix = MutualInformationEstimator().estimate(claims_by_angle)
    
    # Aggregate with weights
    dependence_matrix = (
        weights["embedding"] * embedding_matrix +
        weights["citation"] * citation_matrix +
        weights["mi"] * mi_matrix
    )
    
    # Clamp to [0, 1]
    dependence_matrix = np.clip(dependence_matrix, 0.0, 1.0)
    
    # Symmetrize (should already be, but enforce)
    dependence_matrix = (dependence_matrix + dependence_matrix.T) / 2.0
    
    # Ensure diagonal = 1
    np.fill_diagonal(dependence_matrix, 1.0)
    
    return DependenceReport(
        angle_names=angle_names,
        n_angles=n,
        dependence_matrix=dependence_matrix,
        per_estimator={
            "embedding": embedding_matrix,
            "citation": citation_matrix,
            "mi": mi_matrix
        },
        warnings=[]
    )
```

### Planted-Redundancy Experiment

```python
# experiments/v2/test_dependence_redundancy.py

import numpy as np
import json
from prismflow.v2.dependence.aggregator import aggregate_dependence
from prismflow.v2.reasoners.models import ClaimSet, Claim

async def run_planted_redundancy_experiment(num_seeds: int = 5):
    """
    Test 4 conditions over 5 seeds:
    1. Identical claims → dependence > 0.95
    2. Paraphrased claims → dependence 0.6–0.95
    3. Unrelated claims → dependence < 0.3
    4. Low-rank embedding → graceful degradation
    """
    
    results = {"identical": [], "paraphrased": [], "unrelated": [], "low_rank": []}
    
    for seed in range(num_seeds):
        np.random.seed(seed)
        
        # Condition 1: Identical claims
        claims_tech_identical = [
            Claim("TensorFlow adoption increased 12% YoY", 0.9, ["id1", "id2"]),
            Claim("PyTorch is the leading ML framework", 0.85, ["id1", "id3"])
        ]
        claims_market_identical = claims_tech_identical.copy()  # Exact same
        
        cs_tech_identical = ClaimSet("tech", "ml frameworks", claims_tech_identical, "claude", "claude-sonnet", 100, 50, 1.0)
        cs_market_identical = ClaimSet("market", "ml frameworks", claims_market_identical, "openai", "gpt-4o", 100, 50, 1.0)
        
        report_identical = aggregate_dependence([cs_tech_identical, cs_market_identical])
        dep_identical = report_identical.dependence_matrix[0, 1]
        results["identical"].append(dep_identical)
        
        # Condition 2: Paraphrased
        claims_market_paraphrased = [
            Claim("Framework adoption in TensorFlow rose 12% annually", 0.9, ["id1"]),
            Claim("PyTorch leads the machine learning ecosystem", 0.85, ["id3"])
        ]
        cs_market_paraphrased = ClaimSet("market", "ml frameworks", claims_market_paraphrased, "openai", "gpt-4o", 100, 50, 1.0)
        report_paraphrased = aggregate_dependence([cs_tech_identical, cs_market_paraphrased])
        dep_paraphrased = report_paraphrased.dependence_matrix[0, 1]
        results["paraphrased"].append(dep_paraphrased)
        
        # Condition 3: Unrelated
        claims_market_unrelated = [
            Claim("Stock prices fell due to interest rate hikes", 0.8, ["id4"]),
            Claim("The bond market is volatile", 0.7, ["id5"])
        ]
        cs_market_unrelated = ClaimSet("market", "ml frameworks", claims_market_unrelated, "openai", "gpt-4o", 100, 50, 1.0)
        report_unrelated = aggregate_dependence([cs_tech_identical, cs_market_unrelated])
        dep_unrelated = report_unrelated.dependence_matrix[0, 1]
        results["unrelated"].append(dep_unrelated)
    
    # Summary
    summary = {}
    for condition, deps in results.items():
        summary[condition] = {
            "mean": float(np.mean(deps)),
            "std": float(np.std(deps)),
            "min": float(np.min(deps)),
            "max": float(np.max(deps))
        }
    
    # Validate
    assert summary["identical"]["mean"] > 0.95, "Identical claims should have high dependence"
    assert 0.6 < summary["paraphrased"]["mean"] < 0.95, "Paraphrased claims should be in middle range"
    assert summary["unrelated"]["mean"] < 0.3, "Unrelated claims should have low dependence"
    
    print(json.dumps(summary, indent=2))
    return summary
```

---

## Deliverables

### Tests to write

1. **test_embedding_estimator_symmetric()** — Verify matrix is symmetric
2. **test_citation_jaccard()** — Verify Jaccard computation
3. **test_mi_estimator()** — Verify TF-IDF MI computation
4. **test_aggregation_weights()** — Verify weighted sum
5. **test_planted_redundancy_identical()** — Identical claims → > 0.95
6. **test_planted_redundancy_paraphrased()** — Paraphrase → 0.6–0.95
7. **test_planted_redundancy_unrelated()** — Unrelated → < 0.3

---

## Success Criteria

- [ ] All three estimators implemented and tested
- [ ] Aggregation computes weighted sum; clamps to [0, 1]
- [ ] Planted-redundancy experiment passes all 4 conditions (mean ± std logged)
- [ ] No V1 modifications
- [ ] All tests pass: `pytest tests/v2/test_dependence.py -v`
- [ ] Experiment results saved to `results/v2/dependence_redundancy_experiment.json`
- [ ] Ready for Part 22

---

## Checklist for Closing

- [ ] `aggregate_dependence()` callable; returns DependenceReport
- [ ] Planted redundancy experiment run; report saved
- [ ] Tests pass: `pytest tests/v2/test_dependence.py -v`
