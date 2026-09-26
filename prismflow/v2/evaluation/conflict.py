"""Conflict-detection metrics. NEW IN V2 -- nothing here is reused from V1.

Part 24. An ENGINEERING component per docs/CONTRACT.md section 3.

This file is separate from ``calibration.py`` on purpose. That module contains
no mathematics at all: every figure in it is computed by V1's frozen
``prismflow.evaluation.calibration``. This one is the opposite -- it is written
fresh, because V1 never had a notion of multiple angles disagreeing and so has
nothing to reuse. Keeping the two apart means the question "is this number
V1's or ours?" is answered by which file it lives in.

WHY V1's ``macro_precision_recall_f1`` IS NOT THE FUNCTION WE WANT
-------------------------------------------------------------------
``prismflow.evaluation.metrics.macro_precision_recall_f1`` exists and is
tempting, but it averages unweighted over both classes. Part 24's gate is
conflict recall > 0.7 -- recall of the POSITIVE class specifically. A macro
average blends that with the recall of "no conflict", which on a set where
most rows are no-conflict is dominated by the easy class and can sit
comfortably above 0.7 while the detector finds almost no real conflicts. The
gate would pass on a number that does not mean what the gate says. So these
are positive-class metrics, computed directly.

THE DENOMINATOR IS NOT THE CALIBRATION ONE
-------------------------------------------
Conflict metrics run on rows where ``true_conflict is not None`` -- every row
except those the annotator marked ``Unsure``. Calibration runs on rows where
``outcome_binary is not None``. Those are different filters over the same
dataset and currently give 14 and 17. ``ConflictReport`` carries its own ``n``
for that reason, and the row set is selected inside this module rather than
accepted from a caller.

``Unsure`` is excluded rather than coerced to False. Counting an annotator's
hesitation as "no conflict here" would turn every uncertain row into a true
negative and inflate precision -- the metric would improve precisely because
the labeller was unsure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np

from prismflow.v2.evaluation.dataset import EvaluationQuery


class ConflictInputError(ValueError):
    """The arrays a conflict metric was asked for cannot be built."""


@dataclass(frozen=True)
class ConflictReport:
    """Positive-class conflict metrics, inseparable from their ``n``.

    Attributes:
        n: rows scored (those with a non-None ``true_conflict``).
        excluded_ids: rows dropped as ``Unsure``. Not errors.
        n_true: rows the annotator marked as a real conflict.
        n_detected: rows the detector flagged.
        true_positive / false_positive / false_negative: the raw counts, kept
            so a suspicious ratio can be traced back to integers.
        precision / recall / f1: positive class only.
    """

    n: int
    excluded_ids: Tuple[str, ...]
    n_true: int
    n_detected: int
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float

    def summary(self) -> str:
        return ("conflict P %.4f | R %.4f | F1 %.4f  "
                "(n=%d, %d true, %d detected)"
                % (self.precision, self.recall, self.f1,
                   self.n, self.n_true, self.n_detected))


def conflict_arrays(
    dataset: Sequence[EvaluationQuery],
    detected_by_id: Mapping[str, bool],
) -> Tuple[np.ndarray, np.ndarray, Tuple[str, ...], Tuple[str, ...]]:
    """Build (detected, true) for the CONFLICT denominator only.

    Raises:
        ConflictInputError: no labeled rows, or a labeled row with no detector
            output. A missing detection is not read as "no conflict" -- that
            would silently convert a pipeline failure into a true negative.
    """
    labeled = [q for q in dataset if q.true_conflict is not None]
    excluded = tuple(q.id for q in dataset if q.true_conflict is None)
    if not labeled:
        raise ConflictInputError(
            "no rows carry a conflict label (all %d are Unsure: %s)"
            % (len(dataset), list(excluded)))

    missing = [q.id for q in labeled if q.id not in detected_by_id]
    if missing:
        raise ConflictInputError(
            "no detector output for labeled row(s) %s -- refusing to treat a "
            "missing detection as 'no conflict'" % missing)

    detected = np.array([bool(detected_by_id[q.id]) for q in labeled], dtype=bool)
    true = np.array([bool(q.true_conflict) for q in labeled], dtype=bool)
    return detected, true, tuple(q.id for q in labeled), excluded


def compute_conflict_metrics(
    dataset: Sequence[EvaluationQuery],
    detected_by_id: Mapping[str, bool],
) -> ConflictReport:
    """Positive-class precision, recall and F1 for conflict detection."""
    detected, true, _, excluded = conflict_arrays(dataset, detected_by_id)

    true_positive = int(np.sum(detected & true))
    false_positive = int(np.sum(detected & ~true))
    false_negative = int(np.sum(~detected & true))

    # A zero denominator means the quantity is undefined, not zero. 0.0 is
    # returned because a gate must compare against something, but the counts
    # above are what tells the two cases apart.
    precision = (true_positive / (true_positive + false_positive)
                 if (true_positive + false_positive) > 0 else 0.0)
    recall = (true_positive / (true_positive + false_negative)
              if (true_positive + false_negative) > 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return ConflictReport(
        n=int(detected.size),
        excluded_ids=excluded,
        n_true=int(np.sum(true)),
        n_detected=int(np.sum(detected)),
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )
