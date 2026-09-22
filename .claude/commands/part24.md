# /part24 — Calibration & Evaluation Harness (PrismFlow V2)

**Depends:** All prior parts (17-23)
**Estimate:** 10 hours
**Difficulty:** Very High
**Gate (CRITICAL):** ECE < 0.15, redundancy Pearson r > 0.7, conflict recall > 0.7, cost p95 < $0.20. If ANY fail: STOP, report honestly, do not fabricate.

---

### Goals

1. Create evaluation dataset:
   - 50 labeled queries across 4 domains (startup, tech, career, policy)
   - 40% success (correct/accurate claims), 40% failure (misleading/contradictory), 20% partial
   - Label: query, ground_truth_label (success/fail/partial), reference_urls, expected_confidence

2. Run full pipeline 1000 times (50 queries × 5 seeds × 4 randomizations):
   - Measure: ECE, MCE, Brier score (+ decomposition), AURC
   - Redundancy sensitivity: Pearson r (ENIV vs. claim agreement)
   - Conflict detection: recall, precision
   - Cost per query (p50, p95)

3. **SUCCESS THRESHOLDS (GATE):**
   - ECE < 0.15 (calibration error)
   - Redundancy Pearson r > 0.7 (ENIV captures redundancy)
   - Conflict recall > 0.7 (detects contradictions)
   - Cost < $0.20 p95 (efficiency)
   - **If ANY fail:** STOP, report honestly, do not fabricate

4. Metrics table + visualization

### Specifications

```python
# prismflow/v2/evaluation/calibration.py

import numpy as np
from sklearn.metrics import expected_calibration_error, brier_score_loss
from scipy.stats import pearsonr

def compute_expected_calibration_error(confidences: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Empirical calibration error (Expected Calibration Error)."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = labels[mask].mean()
            ece += abs(bin_conf - bin_acc) * mask.sum() / len(confidences)
    
    return ece

def compute_conflict_metrics(detected_conflicts: np.ndarray, true_conflicts: np.ndarray) -> dict:
    """Compute precision, recall, F1 for conflict detection."""
    tp = (detected_conflicts & true_conflicts).sum()
    fp = (detected_conflicts & ~true_conflicts).sum()
    fn = (~detected_conflicts & true_conflicts).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1}

# prismflow/v2/evaluation/dataset.py

@dataclass
class EvaluationQuery:
    query_text: str
    ground_truth_label: str    # "success", "fail", "partial"
    reference_urls: List[str]
    expected_confidence: float  # What we expect the model to output

def load_evaluation_dataset() -> List[EvaluationQuery]:
    """Load 50 labeled queries from data/v2/evaluation_queries.json."""
    # Load from disk or generate
    pass
```

### Full Evaluation Loop

```python
# experiments/v2/test_evaluation_harness.py

async def run_full_evaluation(num_seeds: int = 5, seed_randomizations: int = 4):
    """
    Run full pipeline 1000 times (50 queries × 5 seeds × 4 randomizations).
    
    GATE: ECE < 0.15, redundancy r > 0.7, conflict recall > 0.7, cost < $0.20 p95
    """
    
    dataset = load_evaluation_dataset()  # 50 queries
    all_results = []
    all_costs = []
    all_enivs = []
    all_claim_agreements = []
    all_detected_conflicts = []
    all_true_conflicts = []
    
    for seed in range(num_seeds):
        for rand in range(seed_randomizations):
            for query_idx, eval_query in enumerate(dataset):
                # Run full pipeline
                # (retrieval, reasoning, dependence, fusion)
                
                recommendation = await run_full_pipeline(eval_query.query_text, seed, rand)
                
                # Evaluate
                is_success = eval_query.ground_truth_label == "success"
                pred_confidence = recommendation.overall_confidence
                
                all_results.append({
                    "query_idx": query_idx,
                    "seed": seed,
                    "rand": rand,
                    "label": is_success,
                    "confidence": pred_confidence,
                    "eniv": recommendation.eniv
                })
                
                all_costs.append(recommendation.cost)
                all_enivs.append(recommendation.eniv)
                
                # Conflict evaluation (mock for now)
                all_detected_conflicts.append(len(recommendation.conflicts_detected) > 0)
                all_true_conflicts.append(False)  # Would be labeled
    
    # Compute metrics
    results_array = np.array([(r["label"], r["confidence"]) for r in all_results])
    labels = results_array[:, 0].astype(int)
    confidences = results_array[:, 1]
    
    ece = compute_expected_calibration_error(confidences, labels)
    mce = np.abs(confidences - labels).max()
    brier = brier_score_loss(labels, confidences)
    
    # Redundancy sensitivity
    redundancy_r, _ = pearsonr(all_enivs, np.abs(confidences - labels))
    
    # Conflict metrics
    conflict_metrics = compute_conflict_metrics(
        np.array(all_detected_conflicts),
        np.array(all_true_conflicts)
    )
    
    # Cost stats
    costs_array = np.array(all_costs)
    cost_p50 = np.percentile(costs_array, 50)
    cost_p95 = np.percentile(costs_array, 95)
    
    # GATE CHECK
    print("\n" + "="*60)
    print("EVALUATION GATE CHECKS")
    print("="*60)
    
    passed = True
    
    print(f"ECE: {ece:.4f} (threshold < 0.15) → {'✅' if ece < 0.15 else '❌'}")
    if ece >= 0.15:
        passed = False
    
    print(f"Redundancy Pearson r: {redundancy_r:.4f} (threshold > 0.7) → {'✅' if redundancy_r > 0.7 else '❌'}")
    if redundancy_r <= 0.7:
        passed = False
    
    print(f"Conflict Recall: {conflict_metrics['recall']:.4f} (threshold > 0.7) → {'✅' if conflict_metrics['recall'] > 0.7 else '❌'}")
    if conflict_metrics['recall'] <= 0.7:
        passed = False
    
    print(f"Cost p95: ${cost_p95:.4f} (threshold < $0.20) → {'✅' if cost_p95 < 0.20 else '❌'}")
    if cost_p95 >= 0.20:
        passed = False
    
    print("="*60)
    
    if not passed:
        print("\n❌ EVALUATION GATE FAILED")
        print("   Do not fabricate results. Report findings honestly.")
        print("   Investigate failing metrics before proceeding.")
        return None
    
    print("\n✅ EVALUATION GATE PASSED")
    
    # Save results
    summary = {
        "ece": ece,
        "mce": mce,
        "brier": brier,
        "redundancy_r": redundancy_r,
        "conflict_metrics": conflict_metrics,
        "cost_p50": cost_p50,
        "cost_p95": cost_p95
    }
    
    with open("results/v2/evaluation_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return summary
```

### Deliverables & Success Criteria (CRITICAL GATE)

- [ ] Evaluation dataset (50 queries, labeled) loaded
- [ ] Full pipeline runs 1000 times without errors
- [ ] **ECE < 0.15** ✓
- [ ] **Redundancy Pearson r > 0.7** ✓
- [ ] **Conflict recall > 0.7** ✓
- [ ] **Cost p95 < $0.20** ✓
- [ ] All results saved to `results/v2/evaluation_results.json`
- [ ] **If any metric fails: STOP, report, do not proceed**

---

## Checklist for All Three Parts (22–24)

**Part 22 (ENIV):**
- [ ] ENIV formula implemented
- [ ] Planted-redundancy test passes GATE (ENIV ≥ 3.5)
- [ ] Discount monotonic with correlation

**Part 23 (Fusion):**
- [ ] PrismFusion fuses claims with ENIV discount
- [ ] Conflict detection works
- [ ] E2E test passes (30 queries)

**Part 24 (Evaluation — CRITICAL):**
- [ ] All 4 success thresholds met
- [ ] Results saved
- [ ] **GATE passed → V2 ready for demo**
- [ ] **GATE failed → investigation + report**

---

## Standing Rules (All Parts)

1. **No V1 modifications.** V1 sealed; defects reported.
2. **Never fabricate.** No results without 5+ seeds; report honestly if thresholds missed.
3. **Audit trail.** Log every step: dispatch, reasoning, fusion, evaluation.
4. **Save results.** All metrics to `results/v2/*.json`.
