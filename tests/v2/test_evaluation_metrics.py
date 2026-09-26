"""Part 24 metric tests: the V1 reuse, the two denominators, and the failures.

Three things these pin down, in order of what would hurt most if it broke.

1. ``calibration.py`` REUSES V1 rather than reimplementing it. The test for
   that is not a comment -- it computes the same figure through V1 directly
   and asserts the two agree exactly. If someone ever inlines a second ECE,
   this fails.
2. The calibration and conflict denominators cannot be swapped. They are
   different filters (17 and 14 on the current dataset) and a figure computed
   on the wrong one is wrong invisibly.
3. Every function can FAIL. A metric that has only ever returned a plausible
   number on good input is not verified, it is untested -- so each one is
   shown rejecting bad input, and ECE and recall are each shown catching a
   predictor that is actually wrong.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from prismflow.evaluation.calibration import (
    DEFAULT_N_BINS,
    expected_calibration_error as v1_expected_calibration_error,
)
from prismflow.v2.evaluation import (
    CalibrationReport,
    ConflictInputError,
    EvaluationQuery,
    MetricInputError,
    compute_calibration,
    compute_conflict_metrics,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------
def make_query(row_id, outcome="Adopted", conflict="No"):
    """One well-formed row. Only the fields the metrics read are varied."""
    return EvaluationQuery(
        id=row_id,
        idea_pitch="pitch %s" % row_id,
        domain="Tech-OSS",
        actual_outcome=outcome,
        outcome_date=date(2020, 1, 1),
        ground_truth_source="https://example.org/%s" % row_id,
        conflict_expected=conflict,
    )


def perfect_dataset(n=20):
    """Half adopted, half rejected, all conflict-labeled."""
    return [make_query("%03d" % i, "Adopted" if i % 2 else "Rejected",
                       "Yes" if i % 3 == 0 else "No")
            for i in range(n)]


# --------------------------------------------------------------------------
# 1. it is reuse, not reimplementation
# --------------------------------------------------------------------------
def test_ece_is_computed_by_v1_not_reimplemented():
    """Our ECE must be V1's ECE, bit for bit, on the same arrays."""
    dataset = perfect_dataset()
    confidence = {q.id: 0.1 + 0.8 * (i % 5) / 4.0 for i, q in enumerate(dataset)}

    report = compute_calibration(dataset, confidence)

    conf = np.array([confidence[q.id] for q in dataset], dtype=np.float64)
    outcome = np.array([q.outcome_binary for q in dataset], dtype=np.float64)
    assert report.ece == v1_expected_calibration_error(conf, outcome, DEFAULT_N_BINS)


def test_default_bin_count_is_v1s_fifteen_not_the_briefs_ten():
    """The brief defaults to 10 bins; V1 uses 15. We must follow V1."""
    dataset = perfect_dataset()
    confidence = {q.id: 0.5 for q in dataset}
    assert compute_calibration(dataset, confidence).n_bins == DEFAULT_N_BINS == 15


def test_brier_multiclass_is_exactly_twice_the_binary_brier():
    """V1's multiclass Brier is 2x the conventional binary one.

    Cross-checked against sklearn's brier_score_loss, which is real and is the
    value a reader would compare against. Reporting V1's number as "the Brier
    score" would be an error of exactly 2x, so the relationship is pinned.
    """
    from sklearn.metrics import brier_score_loss

    dataset = perfect_dataset()
    confidence = {q.id: 0.2 + 0.6 * (i % 4) / 3.0 for i, q in enumerate(dataset)}
    report = compute_calibration(dataset, confidence)

    conf = np.array([confidence[q.id] for q in dataset])
    labels = np.array([q.outcome_binary for q in dataset])
    assert report.brier == pytest.approx(brier_score_loss(labels, conf))
    assert report.brier_multiclass == pytest.approx(2.0 * report.brier)


# --------------------------------------------------------------------------
# 2. the two denominators
# --------------------------------------------------------------------------
def test_calibration_and_conflict_use_different_row_sets():
    """Broke-even leaves calibration; Unsure leaves conflict. Never mixed."""
    dataset = [
        make_query("001", "Adopted", "Yes"),
        make_query("002", "Broke-even", "No"),    # out of calibration only
        make_query("003", "Rejected", "Unsure"),  # out of conflict only
        make_query("004", "Adopted", "No"),
    ]
    confidence = {q.id: 0.5 for q in dataset}
    detected = {q.id: False for q in dataset}

    cal = compute_calibration(dataset, confidence)
    con = compute_conflict_metrics(dataset, detected)

    assert cal.n == 3 and cal.excluded_ids == ("002",)
    assert con.n == 3 and con.excluded_ids == ("003",)
    # Same size here, but genuinely different rows -- which is the trap: an
    # equal n is not evidence that the two sets are interchangeable.
    assert set(cal.excluded_ids) != set(con.excluded_ids)


def test_report_carries_its_own_n():
    """A figure without its denominator is not reportable."""
    dataset = perfect_dataset(10)
    cal = compute_calibration(dataset, {q.id: 0.5 for q in dataset})
    con = compute_conflict_metrics(dataset, {q.id: True for q in dataset})
    assert isinstance(cal, CalibrationReport) and cal.n == 10
    assert "n=10" in cal.summary()
    assert "n=%d" % con.n in con.summary()


# --------------------------------------------------------------------------
# 3a. NEGATIVE: the metrics catch a predictor that is actually wrong
# --------------------------------------------------------------------------
def test_ece_is_near_zero_for_a_calibrated_predictor():
    """100 rows at p=0.5 with a 50% event rate: well calibrated."""
    dataset = [make_query("%03d" % i, "Adopted" if i % 2 else "Rejected")
               for i in range(100)]
    report = compute_calibration(dataset, {q.id: 0.5 for q in dataset})
    assert report.ece < 0.01


def test_ece_catches_a_confidently_wrong_predictor():
    """The negative case. 99% confident, 50% right -> ECE must be ~0.49.

    Without this, a function that always returned 0.0 would pass every other
    test in this file.
    """
    dataset = [make_query("%03d" % i, "Adopted" if i % 2 else "Rejected")
               for i in range(100)]
    report = compute_calibration(dataset, {q.id: 0.99 for q in dataset})
    assert report.ece == pytest.approx(0.49, abs=0.01)
    assert report.brier > 0.24


def test_conflict_recall_catches_a_detector_that_finds_nothing():
    """A detector flagging nothing must score recall 0, not a vacuous 1."""
    dataset = [make_query("%03d" % i, conflict="Yes" if i < 5 else "No")
               for i in range(20)]
    report = compute_conflict_metrics(dataset, {q.id: False for q in dataset})
    assert report.recall == 0.0
    assert report.false_negative == 5 and report.true_positive == 0


def test_conflict_precision_catches_a_detector_that_flags_everything():
    """Flagging all 20 rows finds all 5 conflicts but precision must fall."""
    dataset = [make_query("%03d" % i, conflict="Yes" if i < 5 else "No")
               for i in range(20)]
    report = compute_conflict_metrics(dataset, {q.id: True for q in dataset})
    assert report.recall == 1.0
    assert report.precision == pytest.approx(0.25)
    assert report.false_positive == 15


def test_positive_class_recall_differs_from_a_macro_average():
    """Why conflict metrics are written fresh rather than reusing V1's macro.

    A detector that flags nothing scores 0 recall here. A macro average over
    both classes would score 0.5, which would clear a 'recall > 0.7' gate far
    more easily than the gate intends.
    """
    dataset = [make_query("%03d" % i, conflict="Yes" if i < 2 else "No")
               for i in range(20)]
    report = compute_conflict_metrics(dataset, {q.id: False for q in dataset})
    negative_class_recall = 1.0          # every true-negative was called correctly
    macro = (report.recall + negative_class_recall) / 2
    assert report.recall == 0.0 and macro == 0.5


# --------------------------------------------------------------------------
# 3b. NEGATIVE: every function rejects bad input
# --------------------------------------------------------------------------
def test_calibration_rejects_a_missing_confidence():
    dataset = perfect_dataset(4)
    confidence = {q.id: 0.5 for q in dataset}
    del confidence["002"]
    with pytest.raises(MetricInputError, match="002"):
        compute_calibration(dataset, confidence)


@pytest.mark.parametrize("bad", [-0.1, 1.5, float("nan")])
def test_calibration_rejects_confidence_outside_unit_interval(bad):
    dataset = perfect_dataset(4)
    confidence = {q.id: 0.5 for q in dataset}
    confidence["001"] = bad
    with pytest.raises(MetricInputError, match="001"):
        compute_calibration(dataset, confidence)


def test_calibration_rejects_a_dataset_with_no_scored_rows():
    dataset = [make_query("001", "Broke-even"), make_query("002", "Broke-even")]
    with pytest.raises(MetricInputError, match="no rows are scored"):
        compute_calibration(dataset, {"001": 0.5, "002": 0.5})


def test_conflict_rejects_a_missing_detection():
    """A missing detection must not be silently read as 'no conflict'."""
    dataset = perfect_dataset(4)
    detected = {q.id: False for q in dataset}
    del detected["003"]
    with pytest.raises(ConflictInputError, match="003"):
        compute_conflict_metrics(dataset, detected)


def test_conflict_rejects_an_all_unsure_dataset():
    dataset = [make_query("001", conflict="Unsure"),
               make_query("002", conflict="Unsure")]
    with pytest.raises(ConflictInputError, match="Unsure"):
        compute_conflict_metrics(dataset, {"001": True, "002": True})
