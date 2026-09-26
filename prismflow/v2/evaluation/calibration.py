"""V2 calibration metrics: a thin adapter over V1's frozen calibration module.

Part 24. An ENGINEERING component per docs/CONTRACT.md section 3.

THIS FILE DELIBERATELY CONTAINS NO CALIBRATION MATHEMATICS
-----------------------------------------------------------
Every number here is computed by ``prismflow.evaluation.calibration``, which is
V1, FROZEN at tag ``v1-final`` per docs/CONTRACT.md section 6. This module only
turns a list of ``EvaluationQuery`` into the arrays those functions already
accept, and attaches the ``n`` each figure was computed on.

``.claude/commands/part24.md`` specifies a different route: it opens with

    from sklearn.metrics import expected_calibration_error, brier_score_loss

and then hand-rolls an ECE loop underneath. Two problems, and neither is
stylistic. ``sklearn.metrics.expected_calibration_error`` does not exist in
scikit-learn 1.9.1 -- the import raises ``ImportError`` and takes the real
``brier_score_loss`` down with it, so the module is unimportable as written.
And the hand-rolled replacement would be a SECOND implementation of a metric
V1 already computes, defaulting to 10 bins where V1 uses 15. Two ECE functions
that disagree by bin width do not announce themselves; they surface as a gate
number that moves depending on which import a caller reached for. Reusing the
frozen implementation is what keeps one ECE in the codebase.

V1's version is also the more careful of the two. The brief's loop masks with
``(c >= lo) & (c < hi)``, which silently drops any forecast of exactly 1.0 --
the most confident predictions in the set are the ones excluded.

TWO CONVENTIONS INHERITED FROM V1, BOTH LOAD-BEARING
-----------------------------------------------------
*Binning.* V1 bins by ``ceil(c * M) - 1``: bin b covers ``(b/M, (b+1)/M]``,
except the first, which also includes 0. Right-closed, where the brief is
left-closed. At an exact bin boundary the two assign differently. ``n_bins``
defaults to V1's ``DEFAULT_N_BINS`` (15) and is not re-defaulted here.

*Brier scale.* V1's ``brier_score`` is the MULTICLASS score,
``mean_i sum_k (p_ik - y_ik)^2``, which ranges over [0, 2]. Fed a binary
problem as two one-hot columns it returns exactly TWICE the conventional
binary Brier ``mean((p - y)^2)`` that ``sklearn.metrics.brier_score_loss``
reports. That factor is not a bug in either one, but reporting V1's number
under the name "Brier score" next to a literature value would be an error of
exactly 2x. ``CalibrationReport`` therefore carries both, named apart, and the
relationship is pinned by a test.

WHAT "CONFIDENCE" AND "CORRECT" MEAN HERE
------------------------------------------
V1 names its second argument ``correct``, from an accuracy framing. For a
binary outcome with ``confidence`` = P(outcome is 1), passing the outcome
itself as ``correct`` computes the standard binary ECE: per bin, the gap
between mean forecast probability and the observed event rate. That is the
intended reading, not a coincidence of naming.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------
# REUSED FROM V1 -- FROZEN. Nothing below re-implements any of these.
# Conflict detection metrics are NOT here; V1 has no equivalent and they live
# in prismflow/v2/evaluation/conflict.py so the boundary stays visible.
# --------------------------------------------------------------------------
from prismflow.evaluation.calibration import (  # noqa: F401  (re-exported)
    DEFAULT_N_BINS,
    BrierDecomposition,
    ReliabilityBins,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)

from prismflow.v2.evaluation.dataset import EvaluationQuery


class MetricInputError(ValueError):
    """The arrays a metric was asked for cannot be built from this input."""


@dataclass(frozen=True)
class CalibrationReport:
    """Calibration figures, each inseparable from the ``n`` behind it.

    ``n`` travels in the same object as the numbers because
    docs/part24-dataset-schema.md requires every reported figure to carry its
    own denominator: an ECE < 0.15 met on 17 rows is not the same claim as one
    met on 50. The conflict set has a different ``n`` again, which is why
    ConflictReport is a separate type rather than more fields on this one.

    Attributes:
        n: rows the figures were computed on (``scored_for_calibration``).
        excluded_ids: rows dropped because ``outcome_binary`` is None, i.e.
            Broke-even. Not errors -- they have no success/failure label to
            calibrate against.
        n_bins: bins used, V1's default unless overridden.
        ece: expected calibration error, from V1.
        mce: maximum calibration error, from V1.
        brier: the conventional BINARY Brier, ``mean((p - y)^2)``, in [0, 1].
        brier_multiclass: V1's multiclass Brier, in [0, 2]. Exactly 2x ``brier``
            for a binary problem. Kept so a caller comparing against V1 output
            sees the same number V1 produces.
    """

    n: int
    excluded_ids: Tuple[str, ...]
    n_bins: int
    ece: float
    mce: float
    brier: float
    brier_multiclass: float

    def summary(self) -> str:
        """One line, with the denominator attached to the figures."""
        return ("ECE %.4f | MCE %.4f | Brier %.4f  (n=%d, %d bins)"
                % (self.ece, self.mce, self.brier, self.n, self.n_bins))


def calibration_arrays(
    dataset: Sequence[EvaluationQuery],
    confidence_by_id: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    """Build (confidence, outcome) for the CALIBRATION denominator only.

    The row set is derived here from ``scored_for_calibration`` rather than
    taken from the caller, so a caller holding the conflict set cannot pass it
    in by accident. The two denominators differ (currently 17 and 14) and a
    figure computed on the wrong one is wrong silently.

    Raises:
        MetricInputError: no scored rows, a scored row with no confidence, or
            a confidence outside [0, 1].
    """
    scored = [q for q in dataset if q.scored_for_calibration]
    excluded = tuple(q.id for q in dataset if not q.scored_for_calibration)
    if not scored:
        raise MetricInputError(
            "no rows are scored for calibration (all %d excluded: %s)"
            % (len(dataset), list(excluded)))

    missing = [q.id for q in scored if q.id not in confidence_by_id]
    if missing:
        raise MetricInputError(
            "no confidence supplied for scored row(s) %s -- refusing to "
            "compute calibration on a subset of the calibration set" % missing)

    confidence = np.array([float(confidence_by_id[q.id]) for q in scored],
                          dtype=np.float64)
    outcome = np.array([q.outcome_binary for q in scored], dtype=np.float64)

    bad = [scored[i].id for i in np.flatnonzero(
        (confidence < 0.0) | (confidence > 1.0) | np.isnan(confidence))]
    if bad:
        raise MetricInputError(
            "confidence outside [0, 1] for row(s) %s" % bad)

    return confidence, outcome, tuple(q.id for q in scored), excluded


def compute_calibration(
    dataset: Sequence[EvaluationQuery],
    confidence_by_id: Mapping[str, float],
    n_bins: Optional[int] = None,
) -> CalibrationReport:
    """ECE, MCE and Brier over the calibration rows, computed by V1.

    Args:
        dataset: every row; the calibration subset is selected here.
        confidence_by_id: pipeline confidence that the outcome is 1, per row id.
        n_bins: bins, defaulting to V1's ``DEFAULT_N_BINS`` (15). Not
            re-defaulted to the brief's 10 -- see the module docstring.
    """
    bins = DEFAULT_N_BINS if n_bins is None else n_bins
    confidence, outcome, _, excluded = calibration_arrays(dataset, confidence_by_id)

    # Every line below delegates. The two-column stack is the binary problem
    # expressed as the [N, K] one-hot shape V1's multiclass Brier expects.
    probs = np.column_stack([1.0 - confidence, confidence])
    labels = outcome.astype(np.int64)

    multiclass = brier_score(probs, labels)
    return CalibrationReport(
        n=int(confidence.size),
        excluded_ids=excluded,
        n_bins=int(bins),
        ece=expected_calibration_error(confidence, outcome, bins),
        mce=maximum_calibration_error(confidence, outcome, bins),
        brier=multiclass / 2.0,      # binary convention; see module docstring
        brier_multiclass=multiclass,
    )


def calibration_reliability(
    dataset: Sequence[EvaluationQuery],
    confidence_by_id: Mapping[str, float],
    n_bins: Optional[int] = None,
) -> ReliabilityBins:
    """V1's per-bin table for the calibration rows, for a reliability diagram."""
    bins = DEFAULT_N_BINS if n_bins is None else n_bins
    confidence, outcome, _, _ = calibration_arrays(dataset, confidence_by_id)
    return reliability_bins(confidence, outcome, bins)


def calibration_brier_decomposition(
    dataset: Sequence[EvaluationQuery],
    confidence_by_id: Mapping[str, float],
    n_bins: Optional[int] = None,
) -> BrierDecomposition:
    """V1's Murphy decomposition, including its within-bin fourth term.

    Returned on V1's multiclass scale, matching ``brier_multiclass``.
    """
    bins = DEFAULT_N_BINS if n_bins is None else n_bins
    confidence, outcome, _, _ = calibration_arrays(dataset, confidence_by_id)
    probs = np.column_stack([1.0 - confidence, confidence])
    return brier_decomposition(probs, outcome.astype(np.int64), bins)
