import inspect

import numpy as np
import pytest
import torch

from prismflow.models.defended import DefendedPrismFlow
from prismflow.statistics import suspicion as suspicion_module
from prismflow.statistics.suspicion import (
    agreement_null,
    calibrate_threshold,
    detection_rates,
    pairwise_agreement,
    suspicion_report,
)
from prismflow.train import TrainConfig, build_model
from prismflow.utils.seed import set_seed

FORBIDDEN = ("attack", "compromise", "adversar", "truth")


def _setup(use_discount=True, n_views=4, batch=48, seed=0):
    set_seed(seed)
    config = TrainConfig(n_views=n_views, use_discount=use_discount)
    model = build_model(config)
    model.eval()
    views = torch.randn(batch, n_views, config.d_view)
    view_mask = torch.ones(batch, n_views, dtype=torch.bool)
    return model, views, view_mask


def _belief_and_dependence(model, views, view_mask):
    with torch.no_grad():
        out = model(views, view_mask)
    return out.per_view_belief, out.dependence_matrix, out.per_view_evidence


# --- HARD ISOLATION -----------------------------------------------------
# The detector must never be handed the answer. An agentic session under
# pressure to make a detection number look good would wire ground truth in;
# these tests exist to make that impossible to do quietly.


@pytest.mark.parametrize(
    "function",
    [
        suspicion_module.suspicion_report,
        suspicion_module.pairwise_agreement,
        suspicion_module.agreement_null,
        suspicion_module.calibrate_threshold,
        suspicion_module.detection_rates,
    ],
)
def test_no_public_function_accepts_ground_truth(function):
    for name in inspect.signature(function).parameters:
        lowered = name.lower()
        assert not any(bad in lowered for bad in FORBIDDEN), (
            f"{function.__name__} accepts parameter {name!r}, which looks like ground truth"
        )


def test_every_public_name_in_the_module_is_isolated():
    """Catches a helper added later that quietly takes the compromise mask."""
    for name, value in vars(suspicion_module).items():
        if name.startswith("_") or not callable(value):
            continue
        if getattr(value, "__module__", None) != suspicion_module.__name__:
            continue
        for parameter in inspect.signature(value).parameters:
            lowered = parameter.lower()
            assert not any(bad in lowered for bad in FORBIDDEN), (
                f"{name} accepts parameter {parameter!r}, which looks like ground truth"
            )


def test_defended_forward_is_isolated_too():
    for name in inspect.signature(DefendedPrismFlow.forward).parameters:
        lowered = name.lower()
        assert not any(bad in lowered for bad in FORBIDDEN)


# --- agreement ----------------------------------------------------------


def test_agreement_is_the_cosine_of_the_concordant_mass():
    belief = torch.tensor([[[0.6, 0.2, 0.1], [0.5, 0.3, 0.0]]])
    agreement = pairwise_agreement(belief)
    mass = 0.6 * 0.5 + 0.2 * 0.3 + 0.1 * 0.0
    norms = (0.6**2 + 0.2**2 + 0.1**2) ** 0.5 * (0.5**2 + 0.3**2 + 0.0**2) ** 0.5
    assert agreement[0, 0, 1] == pytest.approx(mass / norms)
    assert agreement[0, 0, 1] == pytest.approx(agreement[0, 1, 0])


def test_identical_views_have_cosine_one_and_confidence_does_not_matter():
    """The cosine must answer "same direction?", not "how confident?"."""
    belief = torch.rand(16, 2, 3) * 0.3
    belief[:, 1, :] = belief[:, 0, :] * 0.25  # same direction, quarter the mass
    agreement = pairwise_agreement(belief)
    assert agreement[:, 0, 1] == pytest.approx(np.ones(16), abs=1e-6)


def test_null_is_symmetric_and_diagonal_is_one():
    belief = _synthetic_beliefs(batch=32)
    null = agreement_null(belief, n_permutations=4)
    assert null == pytest.approx(null.T, nan_ok=True)
    assert np.diag(null) == pytest.approx(np.ones(4))


def test_absent_views_give_nan_pairs():
    belief = torch.rand(8, 3, 4) * 0.3
    view_mask = torch.ones(8, 3, dtype=torch.bool)
    view_mask[:, 2] = False
    agreement = pairwise_agreement(belief, view_mask)
    assert np.isnan(agreement[:, 0, 2]).all()
    assert np.isfinite(agreement[:, 0, 1]).all()


def test_belief_shape_is_validated():
    with pytest.raises(ValueError):
        pairwise_agreement(torch.rand(4, 3))


# --- the score ----------------------------------------------------------


def _synthetic_beliefs(batch=64, n_views=4, n_classes=3, seed=0):
    """Beliefs that actually vary across samples.

    The untrained `build_model` emits near-identical opinions for every sample,
    which collapses the permutation null onto the Cauchy-Schwarz ceiling and
    leaves no explainable range. That is a property of an untrained net, not of
    the statistic, so the semantic tests use controlled beliefs instead.
    Views 0 and 1 are made identical: the colluding pair.
    """
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, n_views, n_classes, generator=generator) * 2.0
    belief = torch.softmax(logits, dim=-1) * 0.8  # leave 0.2 vacuity
    belief[:, 1, :] = belief[:, 0, :]
    return belief


def test_agreement_beyond_the_measured_dependence_raises_that_pair():
    """The defining semantics: agreement is only suspicious relative to R.

    Views 0 and 1 agree perfectly while the dependence matrix still reports them
    as only mildly dependent, so their agreement is not accounted for and that
    pair must stand out.
    """
    belief = _synthetic_beliefs()
    dependence = np.full((4, 4), 0.1)
    np.fill_diagonal(dependence, 1.0)

    report = suspicion_report(belief, dependence, n_permutations=8)
    assert np.nanmean(report.excess[:, 0, 1]) > np.nanmean(report.excess[:, 2, 3])


def test_a_duplicate_the_estimator_already_measures_is_not_flagged_as_anomalous():
    """The converse, and a limitation worth pinning down: when the dependence
    estimate rises with the agreement, the excess does NOT rise. This detector
    reports agreement the structure cannot explain, not redundancy as such --
    a cleanly measured duplicate is explained and so scores no higher.

    This is precisely why the detector is expected to struggle against the
    Chorus attack, which raises measured dependence along with agreement
    (`experiments/comparison/README.md`).
    """
    belief = _synthetic_beliefs()
    unaccounted = np.full((4, 4), 0.1)
    np.fill_diagonal(unaccounted, 1.0)
    accounted = unaccounted.copy()
    accounted[0, 1] = accounted[1, 0] = 1.0

    high_r = suspicion_report(belief, accounted, n_permutations=8)
    low_r = suspicion_report(belief, unaccounted, n_permutations=8)
    assert np.nanmean(high_r.excess[:, 0, 1]) < np.nanmean(low_r.excess[:, 0, 1])


def test_score_is_the_max_over_off_diagonal_pairs():
    model, views, view_mask = _setup()
    belief, dependence, evidence = _belief_and_dependence(model, views, view_mask)
    report = suspicion_report(belief, dependence, view_mask, evidence, n_permutations=8)

    off_diagonal = report.excess.copy()
    for i in range(off_diagonal.shape[1]):
        off_diagonal[:, i, i] = np.nan
    expected = np.nanmax(off_diagonal.reshape(len(belief), -1), axis=1)
    assert report.score == pytest.approx(expected, nan_ok=True)


def test_flag_follows_the_threshold_monotonically():
    model, views, view_mask = _setup()
    belief, dependence, evidence = _belief_and_dependence(model, views, view_mask)

    rates = []
    for threshold in (-np.inf, 0.0, 0.25, 1.0, 10.0):
        report = suspicion_report(
            belief, dependence, view_mask, evidence, threshold=threshold, n_permutations=8
        )
        rates.append(report.flag_rate)
    assert all(b <= a + 1e-12 for a, b in zip(rates, rates[1:]))
    assert rates[0] == pytest.approx(1.0)
    assert rates[-1] == pytest.approx(0.0)


def test_report_is_deterministic_given_the_seed():
    model, views, view_mask = _setup()
    belief, dependence, evidence = _belief_and_dependence(model, views, view_mask)
    a = suspicion_report(belief, dependence, view_mask, evidence, seed=3, n_permutations=8)
    b = suspicion_report(belief, dependence, view_mask, evidence, seed=3, n_permutations=8)
    assert a.score == pytest.approx(b.score, nan_ok=True)


def test_dependence_shape_is_validated():
    model, views, view_mask = _setup()
    belief, _, evidence = _belief_and_dependence(model, views, view_mask)
    with pytest.raises(ValueError):
        suspicion_report(belief, np.eye(2), view_mask, evidence, n_permutations=4)


def test_higher_measured_dependence_explains_more_and_lowers_the_score():
    """The whole premise: the same agreement is less surprising when the
    dependence structure already accounts for it."""
    model, views, view_mask = _setup()
    belief, dependence, evidence = _belief_and_dependence(model, views, view_mask)

    low = suspicion_report(
        belief, np.zeros_like(dependence), view_mask, evidence, n_permutations=8
    )
    high = suspicion_report(
        belief, np.ones_like(dependence), view_mask, evidence, n_permutations=8
    )
    assert np.nanmean(high.score) < np.nanmean(low.score)


# --- threshold calibration ----------------------------------------------


def test_calibrate_threshold_hits_the_target_rate_on_its_reference():
    scores = np.linspace(0.0, 1.0, 101)
    threshold = calibrate_threshold(scores, target_rate=0.1)
    assert detection_rates(scores, threshold) == pytest.approx(0.1, abs=0.02)


def test_calibrate_threshold_rejects_impossible_rates():
    with pytest.raises(ValueError):
        calibrate_threshold(np.zeros(4), target_rate=1.5)


def test_calibrate_threshold_survives_all_nan_reference():
    assert np.isfinite(calibrate_threshold(np.full(5, np.nan)))


# --- the defended model -------------------------------------------------


def test_defended_prediction_is_bit_identical_to_prismflow():
    """The flag is an alert, not a control input: adding it must not move a
    single prediction (docs/KNOWN_LIMITATIONS.md L4)."""
    model, views, view_mask = _setup()
    with torch.no_grad():
        plain = model(views, view_mask)
    defended = DefendedPrismFlow(model, suspicion_permutations=8)
    with torch.no_grad():
        out = defended(views, view_mask)

    assert torch.equal(out.prediction, plain.prediction)
    assert torch.equal(out.probs, plain.probs)
    assert torch.equal(out.uncertainty, plain.uncertainty)


def test_defended_exposes_the_required_diagnostics():
    model, views, view_mask = _setup()
    defended = DefendedPrismFlow(model, suspicion_permutations=8)
    with torch.no_grad():
        out = defended(views, view_mask)

    assert out.dependence_matrix is not None
    assert out.eniv is not None
    assert len(out.per_view_reliability) == model.n_views
    assert out.per_view_evidence.shape == (len(views), model.n_views, model.n_classes)
    assert out.suspicion is not None
    assert out.suspicion_flag.shape == (len(views),)
    assert out.suspicion_score.shape == (len(views),)


def test_defended_without_discount_has_no_dependence_and_no_flag():
    model, views, view_mask = _setup(use_discount=False)
    defended = DefendedPrismFlow(model, suspicion_permutations=8)
    with torch.no_grad():
        out = defended(views, view_mask)

    assert out.dependence_matrix is None
    assert out.suspicion is None
    assert out.suspicion_flag is None
    assert out.per_view_reliability == tuple([1.0] * model.n_views)


def test_detect_false_skips_the_detector():
    model, views, view_mask = _setup()
    defended = DefendedPrismFlow(model, suspicion_permutations=8)
    with torch.no_grad():
        out = defended(views, view_mask, detect=False)
    assert out.suspicion is None
    assert out.dependence_matrix is not None
