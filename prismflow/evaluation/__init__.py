"""Post-hoc evaluation: calibration, selective prediction, and the unified protocol."""

from prismflow.evaluation.calibration import (
    DEFAULT_N_BINS,
    BrierDecomposition,
    ReliabilityBins,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    plot_reliability_diagram,
    reliability_bins,
)
from prismflow.evaluation.metrics import (
    DEFAULT_COVERAGES,
    accuracy,
    aurc,
    macro_precision_recall_f1,
    risk_coverage_curve,
    selective_risk,
)
from prismflow.evaluation.protocol import compute_metrics, evaluate

__all__ = [
    "DEFAULT_COVERAGES",
    "DEFAULT_N_BINS",
    "BrierDecomposition",
    "ReliabilityBins",
    "accuracy",
    "aurc",
    "brier_decomposition",
    "brier_score",
    "compute_metrics",
    "evaluate",
    "expected_calibration_error",
    "macro_precision_recall_f1",
    "maximum_calibration_error",
    "plot_reliability_diagram",
    "reliability_bins",
    "risk_coverage_curve",
    "selective_risk",
]
