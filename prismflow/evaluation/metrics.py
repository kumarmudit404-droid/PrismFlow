"""Classification and selective-prediction metrics.

Post-hoc statistical estimators over numpy arrays; nothing here touches a model.

SELECTIVE PREDICTION AND TIES
-----------------------------
Risk-coverage metrics accept the most confident samples first. When several
samples share a confidence value, their order is arbitrary, and a fixed
tie-break (e.g. array order) would let the result depend on how the data
happened to be stored. Instead each tied group contributes its errors at the
group's average rate: the reported risk is the expectation over random
tie-breaking. A model that outputs one constant confidence therefore gets
AURC equal to its error rate, with no hidden ordering effect.
"""

from __future__ import annotations

import numpy as np

DEFAULT_COVERAGES = (1.0, 0.9, 0.8, 0.5)


def _validate(labels: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.ndim != 1 or labels.shape != predictions.shape:
        raise ValueError("labels and predictions must be matching 1-D arrays")
    if labels.size == 0:
        raise ValueError("need at least one sample")
    return labels, predictions


def accuracy(labels: np.ndarray, predictions: np.ndarray) -> float:
    labels, predictions = _validate(labels, predictions)
    return float(np.mean(labels == predictions))


def macro_precision_recall_f1(
    labels: np.ndarray, predictions: np.ndarray, n_classes: int
) -> dict[str, float]:
    """Unweighted mean over all `n_classes` classes.

    A class that is never predicted has precision 0; a class absent from the
    labels has recall 0; F1 is 0 when precision + recall is 0. Every class
    counts in the average whether or not it occurs, so a model that ignores a
    class is penalised rather than having that class dropped.
    """
    labels, predictions = _validate(labels, predictions)
    precision, recall, f1 = [], [], []
    for k in range(n_classes):
        true_positive = np.sum((predictions == k) & (labels == k))
        predicted = np.sum(predictions == k)
        actual = np.sum(labels == k)
        p = true_positive / predicted if predicted else 0.0
        r = true_positive / actual if actual else 0.0
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
    }


def _expected_errors_by_rank(confidence: np.ndarray, errors: np.ndarray) -> np.ndarray:
    """Errors in descending-confidence order, with ties spread at their group mean."""
    confidence = np.asarray(confidence, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    if confidence.ndim != 1 or confidence.shape != errors.shape:
        raise ValueError("confidence and errors must be matching 1-D arrays")
    if confidence.size == 0:
        raise ValueError("need at least one sample")
    if np.any(np.isnan(confidence)):
        raise ValueError("confidence contains NaN")

    order = np.argsort(-confidence, kind="stable")
    sorted_conf, sorted_err = confidence[order], errors[order]
    # Group boundaries of equal confidence values.
    _, starts, sizes = np.unique(-sorted_conf, return_index=True, return_counts=True)
    group_mean = np.add.reduceat(sorted_err, starts) / sizes
    return np.repeat(group_mean, sizes)


def risk_coverage_curve(confidence: np.ndarray, errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(coverage, selective risk) after accepting the top-1, top-2, ... samples."""
    ranked = _expected_errors_by_rank(confidence, errors)
    accepted = np.arange(1, ranked.size + 1)
    return accepted / ranked.size, np.cumsum(ranked) / accepted


def aurc(confidence: np.ndarray, errors: np.ndarray) -> float:
    """Area under the risk-coverage curve: mean selective risk over all N coverages.

    Lower is better. Accepts any error signal in [0, 1] (0-1 loss by default in
    the protocol).
    """
    _, risk = risk_coverage_curve(confidence, errors)
    return float(np.mean(risk))


def selective_risk(confidence: np.ndarray, errors: np.ndarray, coverage: float) -> float:
    """Risk on the ceil(coverage * N) most confident samples (at least one)."""
    if not 0.0 < coverage <= 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    _, risk = risk_coverage_curve(confidence, errors)
    accepted = max(1, int(np.ceil(coverage * risk.size - 1e-9)))
    return float(risk[accepted - 1])


def coverage_key(coverage: float) -> str:
    """Stable metric name for a coverage level, e.g. 0.9 -> 'selective_risk@0.90'."""
    return f"selective_risk@{coverage:.2f}"
