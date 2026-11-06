"""Test-block metrics for Part 38.

WHY THESE ARE HERE AND NOT IN prismflow.evaluation
--------------------------------------------------
`prismflow.evaluation.compute_metrics` carries no balanced accuracy and no
ROC AUC, and it is frozen -- Parts 29 and 30 hit the same wall and each
computed these beside it rather than editing it. This module is that same
step, written once for Part 38 instead of inline in a runner.

ROC AUC and balanced accuracy are the two figures the pre-registration leans
on, because sample prevalence here is an artefact of the design: accuracy and
anything else prevalence-dependent describes the sample, not the network.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def ece(labels: np.ndarray, prob_attack: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the attack probability, equal-width bins.

    Equal-width rather than equal-count, matching
    `prismflow.evaluation.metrics.reliability_bins`, so the figure is
    comparable to the ones already committed for Parts 28-30.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(prob_attack, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        in_bin = idx == b
        if not in_bin.any():
            continue
        conf = prob_attack[in_bin].mean()
        acc = labels[in_bin].mean()
        total += in_bin.mean() * abs(conf - acc)
    return float(total)


def evaluate(labels: np.ndarray, prob_attack: np.ndarray,
             n_bins: int = 15) -> dict:
    """Every primary metric from one label vector and one probability vector.

    `roc_auc` is None when the block holds a single class: a one-class AUC is
    undefined, and returning 0.5 would read as a measurement.
    """
    labels = np.asarray(labels).astype(int)
    prob = np.asarray(prob_attack, dtype=np.float64)
    pred = (prob >= 0.5).astype(int)
    both_classes = len(np.unique(labels)) == 2
    attack = labels == 1
    benign = labels == 0
    return {
        "n": int(labels.size),
        "n_attack": int(attack.sum()),
        "n_benign": int(benign.sum()),
        "prevalence": float(attack.mean()) if labels.size else None,
        "roc_auc": float(roc_auc_score(labels, prob)) if both_classes else None,
        "balanced_accuracy": (float(balanced_accuracy_score(labels, pred))
                              if both_classes else None),
        "accuracy": float((pred == labels).mean()),
        "attack_recall": float(pred[attack].mean()) if attack.any() else None,
        "benign_recall": float(1.0 - pred[benign].mean()) if benign.any() else None,
        "brier": float(np.mean((prob - labels) ** 2)),
        "prob_ece": ece(labels, prob, n_bins),
    }


def constant_baseline_brier(labels: np.ndarray) -> float:
    """Brier of predicting the block's own prevalence on every row.

    The bar L29 measured Part 30's two bad arms against (0.0428 there).
    """
    labels = np.asarray(labels).astype(int)
    p = labels.mean()
    return float(np.mean((p - labels) ** 2))


def auth_presence_auc(labels: np.ndarray, auth_present: np.ndarray) -> float | None:
    """H1's trivial baseline: auth presence alone as the score.

    Part 30's committed value for this is test AUC 0.6637, which is the number
    H1 asks whether the new selection has removed.
    """
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) != 2:
        return None
    return float(roc_auc_score(labels, np.asarray(auth_present, dtype=np.float64)))
