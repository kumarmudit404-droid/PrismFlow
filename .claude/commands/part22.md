# /part22 — ENIV Extension for Semantic Views (PrismFlow V2)

**Depends:** Part 21 (DependenceReport)
**Estimate:** 4 hours
**Difficulty:** Medium
**Gate:** Baseline ENIV >= 3.5; discount monotonic with correlation, else STOP

---

### Goals

1. Extend ENIV formula from V1 to semantic multi-view setting:
   - Formula: `ENIV = n² / sum(D)` where D is dependence matrix
   - Discount factor: `max(0.1, ENIV / n)`
   - Applied per-claim: discounted_confidence = confidence × discount_factor

2. Run planted-redundancy test (20 queries × 4 conditions × 5 seeds):
   - Condition 1: Independent claims → ENIV ≈ n = 5; discount ≈ 1.0
   - Condition 2: Moderately correlated → ENIV ≈ 2; discount ≈ 0.4
   - Condition 3: Highly correlated → ENIV ≈ 0.5; discount ≈ 0.1 (capped)
   - Condition 4: Decreasing ENIV → verify discount monotonic

3. **GATE:** If baseline ENIV < 3.5 OR discount drops not monotonic → STOP, report, **do not proceed to Part 23**

### Specifications

```python
# prismflow/v2/statistics/eniv.py

import numpy as np
from prismflow.v2.dependence.models import DependenceReport

def compute_semantic_eniv(dependence_report: DependenceReport) -> float:
    """
    Compute ENIV from semantic dependence matrix.
    
    ENIV = n² / sum(D)
    where D is the n×n dependence matrix
    
    Returns:
        float: ENIV value (typically [0.5, 5])
    """
    n = dependence_report.n_angles
    D = dependence_report.dependence_matrix
    
    sum_D = D.sum()
    if sum_D == 0:
        return n  # No dependence → full independence
    
    eniv = (n ** 2) / sum_D
    return eniv

def compute_discount_factor(eniv: float, n: int = 5) -> float:
    """
    Compute confidence discount factor.
    
    discount = max(0.1, ENIV / n)
    
    Args:
        eniv: ENIV value
        n: Number of angles (default 5)
    
    Returns:
        float: Discount factor in [0.1, 1.0]
    """
    discount = max(0.1, eniv / n)
    return min(1.0, discount)

def apply_eniv_discount(claim_confidence: float, discount_factor: float) -> float:
    """
    Apply discount factor to claim confidence.
    
    discounted_conf = confidence × discount_factor
    
    Args:
        claim_confidence: Original [0, 1]
        discount_factor: From compute_discount_factor()
    
    Returns:
        float: Discounted confidence [0, 1]
    """
    return claim_confidence * discount_factor
```

### Planted-Redundancy Test

```python
# experiments/v2/test_eniv_redundancy.py

import numpy as np
import json
from prismflow.v2.statistics.eniv import compute_semantic_eniv, compute_discount_factor
from prismflow.v2.dependence.aggregator import aggregate_dependence
from prismflow.v2.reasoners.models import ClaimSet, Claim

async def run_eniv_redundancy_experiment(num_seeds: int = 5):
    """
    Test ENIV discount under 4 conditions.
    
    GATE: baseline ENIV ≥ 3.5 and monotonic discount required to proceed.
    """
    results = {
        "independent": [],
        "moderate_correlation": [],
        "high_correlation": [],
        "decreasing_eniv": []
    }
    
    for seed in range(num_seeds):
        np.random.seed(seed)
        
        # Condition 1: Independent (low dependence)
        dependence_independent = np.eye(5)  # No correlation
        report_independent = type('DependenceReport', (), {
            'angle_names': ['tech', 'market', 'financial', 'regulatory', 'sentiment'],
            'n_angles': 5,
            'dependence_matrix': dependence_independent
        })()
        
        eniv_independent = compute_semantic_eniv(report_independent)
        discount_independent = compute_discount_factor(eniv_independent, 5)
        results["independent"].append({"eniv": eniv_independent, "discount": discount_independent})
        
        # Condition 2: Moderate correlation
        dependence_moderate = np.array([
            [1.0, 0.4, 0.3, 0.2, 0.1],
            [0.4, 1.0, 0.4, 0.2, 0.1],
            [0.3, 0.4, 1.0, 0.3, 0.1],
            [0.2, 0.2, 0.3, 1.0, 0.2],
            [0.1, 0.1, 0.1, 0.2, 1.0]
        ])
        
        report_moderate = type('DependenceReport', (), {
            'angle_names': ['tech', 'market', 'financial', 'regulatory', 'sentiment'],
            'n_angles': 5,
            'dependence_matrix': dependence_moderate
        })()
        
        eniv_moderate = compute_semantic_eniv(report_moderate)
        discount_moderate = compute_discount_factor(eniv_moderate, 5)
        results["moderate_correlation"].append({"eniv": eniv_moderate, "discount": discount_moderate})
        
        # Condition 3: High correlation
        dependence_high = 0.7 * np.ones((5, 5)) + 0.3 * np.eye(5)
        report_high = type('DependenceReport', (), {
            'angle_names': ['tech', 'market', 'financial', 'regulatory', 'sentiment'],
            'n_angles': 5,
            'dependence_matrix': dependence_high
        })()
        
        eniv_high = compute_semantic_eniv(report_high)
        discount_high = compute_discount_factor(eniv_high, 5)
        results["high_correlation"].append({"eniv": eniv_high, "discount": discount_high})
    
    # Summary
    summary = {}
    for condition, data in results.items():
        enivs = [d["eniv"] for d in data]
        discounts = [d["discount"] for d in data]
        summary[condition] = {
            "eniv_mean": float(np.mean(enivs)),
            "eniv_std": float(np.std(enivs)),
            "discount_mean": float(np.mean(discounts)),
            "discount_std": float(np.std(discounts))
        }
    
    # GATE CHECK
    baseline_eniv = summary["independent"]["eniv_mean"]
    if baseline_eniv < 3.5:
        print(f"❌ GATE FAILED: Baseline ENIV {baseline_eniv:.2f} < 3.5")
        print("   Cannot proceed to Part 23. Investigate dependence estimation.")
        return None
    
    # Check monotonicity: independent > moderate > high
    if not (summary["independent"]["discount_mean"] >
            summary["moderate_correlation"]["discount_mean"] >
            summary["high_correlation"]["discount_mean"]):
        print("❌ GATE FAILED: Discount not monotonic with correlation")
        print(f"   Independent: {summary['independent']['discount_mean']:.3f}")
        print(f"   Moderate:    {summary['moderate_correlation']['discount_mean']:.3f}")
        print(f"   High:        {summary['high_correlation']['discount_mean']:.3f}")
        return None
    
    print("✅ GATE PASSED")
    print(json.dumps(summary, indent=2))
    return summary
```

### Deliverables & Success Criteria

- [ ] ENIV formula implemented and tested
- [ ] Discount factor computed correctly
- [ ] Planted-redundancy experiment passes GATE (baseline ENIV ≥ 3.5, monotonic discount)
- [ ] Results saved to `results/v2/eniv_redundancy_experiment.json`
- [ ] **Proceed to Part 23 only if GATE passes**

---



---

## Standing Rules

1. **No V1 modifications.** V1 is sealed at tag `v1-final`; defects are reported, not patched.
2. **Never fabricate.** No result without 5+ seeds, mean +/- std; report honestly if a threshold is missed.
3. **Audit trail.** Log every step: dispatch, reasoning, dependence, fusion, evaluation.
4. **Save results.** All metrics to `results/v2/*.json`.
