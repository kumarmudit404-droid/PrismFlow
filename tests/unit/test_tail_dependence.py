"""Tail dependence and copula estimators.

Two jobs. First, pin the numerics that had to be reimplemented because scipy is
not a project dependency -- if `norm_ppf` or `t_cdf` are wrong, every copula
number in Part 13 is wrong and nothing downstream would reveal it. Second, pin
the PROPERTY that motivates the whole Part: a Gaussian copula has zero upper
tail dependence at correlations where Pearson reads strong dependence.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from prismflow.statistics.copula import (
    empirical_copula,
    fit_gaussian_copula,
    fit_t_copula,
    kendall_tau_matrix,
    norm_ppf,
    pseudo_observations,
    t_cdf,
    t_ppf,
)
from prismflow.statistics.tail_dependence import (
    MIN_TAIL_SAMPLES,
    co_exceedance_score,
    lower_tail_dependence,
    tail_agreement_score,
    tail_dependence_matrix,
    tail_eniv,
    upper_tail_dependence,
)


# --------------------------------------------------------------------------
# numerics reimplemented without scipy
# --------------------------------------------------------------------------

def test_norm_ppf_matches_known_quantiles():
    """Published standard-normal quantiles, all three branches of Acklam."""
    cases = {0.5: 0.0, 0.975: 1.959963985, 0.995: 2.575829304,
             0.025: -1.959963985, 0.001: -3.090232306, 0.999: 3.090232306}
    for p, expected in cases.items():
        assert abs(float(norm_ppf(p)) - expected) < 1e-6, p


def test_norm_ppf_round_trips_through_erf():
    """Independent check against math.erf, which is not used in the estimator."""
    for p in (0.01, 0.2, 0.5, 0.8, 0.99):
        x = float(norm_ppf(p))
        cdf = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        assert abs(cdf - p) < 1e-9


def test_norm_ppf_rejects_boundary():
    for bad in (0.0, 1.0, -0.1, 1.2):
        with pytest.raises(ValueError):
            norm_ppf(bad)


def test_t_cdf_matches_known_values():
    # t with df=1 is Cauchy: CDF(x) = 0.5 + arctan(x)/pi, exactly.
    for x in (-2.0, -0.5, 0.0, 0.5, 2.0):
        expected = 0.5 + math.atan(x) / math.pi
        assert abs(float(t_cdf(np.array([x]), 1.0)[0]) - expected) < 1e-9, x


def test_t_cdf_approaches_normal_for_large_df():
    for x in (-1.5, 0.0, 1.5):
        normal = 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        assert abs(float(t_cdf(np.array([x]), 5000.0)[0]) - normal) < 1e-3


def test_t_ppf_inverts_t_cdf():
    for df in (3.0, 10.0):
        for p in (0.05, 0.5, 0.9, 0.99):
            x = float(t_ppf(np.array([p]), df)[0])
            assert abs(float(t_cdf(np.array([x]), df)[0]) - p) < 1e-6


# --------------------------------------------------------------------------
# rank transform
# --------------------------------------------------------------------------

def test_pseudo_observations_are_strictly_inside_unit_interval():
    rng = np.random.default_rng(0)
    u = pseudo_observations(rng.normal(size=(200, 3)))
    assert u.shape == (200, 3)
    assert np.all(u > 0.0) and np.all(u < 1.0)


def test_pseudo_observations_are_invariant_to_monotone_marginals():
    """The whole point of a copula: rescaling one view must change nothing."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(150, 2))
    rescaled = x.copy()
    rescaled[:, 0] = np.exp(3.0 * rescaled[:, 0]) * 17.0
    assert np.allclose(pseudo_observations(x), pseudo_observations(rescaled))


def test_pseudo_observations_average_ties():
    u = pseudo_observations(np.zeros((10, 1)))
    assert np.allclose(u, 0.5)


def test_empirical_copula_bounds():
    rng = np.random.default_rng(2)
    u = pseudo_observations(rng.normal(size=(100, 2)))
    values = empirical_copula(u, points=[[0.5, 0.5], [1.0, 1.0], [0.0, 0.0]])
    assert 0.0 <= values[0] <= 0.5
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# THE PROPERTY THE PART EXISTS FOR
# --------------------------------------------------------------------------

def test_gaussian_tail_dependence_decays_but_slowly():
    """lambda_U = 0 for a Gaussian copula is ASYMPTOTIC, and the approach is slow.

    Pinned because it is the Part's main interpretive trap: at rho = 0.9 the
    estimate is still ~0.53 at q = 0.995. A single lambda_U at a single
    threshold therefore does NOT demonstrate tail dependence. Only the decay
    does, and the next test supplies the contrast.
    """
    rng = np.random.default_rng(3)
    n = 20000
    z1 = rng.normal(size=n)
    z2 = 0.9 * z1 + math.sqrt(1.0 - 0.81) * rng.normal(size=n)
    assert np.corrcoef(z1, z2)[0, 1] > 0.85

    curve = [upper_tail_dependence(z1, z2, q).coefficient for q in (0.90, 0.95, 0.99, 0.995)]
    assert curve == sorted(curve, reverse=True), f"must decay monotonically, got {curve}"
    # Still far from zero -- this is the trap, stated as an assertion.
    assert curve[-1] > 0.4


def test_t_copula_tail_dependence_does_not_decay_like_gaussian():
    """The DECAY is the discriminator, not the level at one threshold.

    Same correlation for both; the Gaussian falls away as q rises and the t
    copula flattens to a positive constant.
    """
    n, rho, df = 40000, 0.6, 3.0
    rng = np.random.default_rng(3)
    z1 = rng.normal(size=n)
    z2 = rho * z1 + math.sqrt(1.0 - rho**2) * rng.normal(size=n)
    mixer = np.sqrt(df / rng.chisquare(df, size=n))

    quantiles = (0.90, 0.95, 0.99, 0.995)
    gaussian = [upper_tail_dependence(z1, z2, q).coefficient for q in quantiles]
    student = [upper_tail_dependence(z1 * mixer, z2 * mixer, q).coefficient for q in quantiles]

    gaussian_drop = gaussian[0] - gaussian[-1]
    student_drop = student[0] - student[-1]
    assert gaussian_drop > 0.2, f"Gaussian must decay materially, got {gaussian}"
    assert student_drop < 0.12, f"t copula must stay roughly flat, got {student}"
    assert student[-1] > gaussian[-1] + 0.2


def test_comonotone_pair_has_full_tail_dependence():
    rng = np.random.default_rng(4)
    x = rng.normal(size=500)
    estimate = upper_tail_dependence(x, 3.0 * x + 1.0, quantile=0.9)
    assert estimate.coefficient == pytest.approx(1.0)


def test_independent_pair_has_tail_dependence_near_zero():
    rng = np.random.default_rng(5)
    a, b = rng.normal(size=8000), rng.normal(size=8000)
    estimate = upper_tail_dependence(a, b, quantile=0.95)
    assert estimate.coefficient < 0.15


def test_t_copula_beats_gaussian_on_tail_dependent_data():
    """Fit both to data with genuine joint extremes; t must win on likelihood."""
    rng = np.random.default_rng(6)
    n, df = 4000, 3.0
    correlation = 0.6
    z1 = rng.normal(size=n)
    z2 = correlation * z1 + math.sqrt(1.0 - correlation**2) * rng.normal(size=n)
    # A shared chi-square mixer is what creates joint extremes.
    mixer = np.sqrt(df / rng.chisquare(df, size=n))
    x = np.column_stack([z1 * mixer, z2 * mixer])

    u = pseudo_observations(x)
    gaussian = fit_gaussian_copula(u)
    student = fit_t_copula(u)

    assert student.log_likelihood > gaussian.log_likelihood
    assert gaussian.upper_tail_dependence == 0.0
    assert student.upper_tail_dependence > 0.0
    assert student.df <= 10.0


def test_kendall_tau_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(7)
    tau = kendall_tau_matrix(rng.normal(size=(120, 3)))
    assert np.allclose(tau, tau.T)
    assert np.allclose(np.diag(tau), 1.0)
    assert np.all(tau >= -1.0) and np.all(tau <= 1.0)


def test_kendall_tau_detects_monotone_association():
    x = np.linspace(-2.0, 2.0, 100)
    tau = kendall_tau_matrix(np.column_stack([x, x**3]))
    assert tau[0, 1] == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# the sample-size caveat, enforced rather than described
# --------------------------------------------------------------------------

def test_estimate_reports_tail_count_and_marks_small_samples_unreliable():
    """300 samples at q=0.95 gives 15 exceedances -- the Part's central caveat."""
    rng = np.random.default_rng(8)
    estimate = upper_tail_dependence(rng.normal(size=300), rng.normal(size=300), quantile=0.95)
    assert estimate.n_tail == 15
    assert not estimate.reliable
    assert MIN_TAIL_SAMPLES == 50


def test_estimate_is_reliable_with_enough_exceedances():
    rng = np.random.default_rng(9)
    estimate = upper_tail_dependence(rng.normal(size=2000), rng.normal(size=2000), quantile=0.95)
    assert estimate.n_tail == 100
    assert estimate.reliable


def test_nonparametric_estimator_is_not_a_variance_reduction():
    """Using the whole sample does NOT make the log-ratio form steadier.

    This was assumed when the module was written and is false. Pinned in the
    direction the data actually shows, so the experiment cannot quietly claim
    the log-ratio estimate is the more reliable of the two: it is marginally
    noisier and sits systematically lower.
    """
    stats = {}
    for method in ("empirical", "nonparametric"):
        values = []
        for seed in range(40):
            rng = np.random.default_rng(100 + seed)
            z1 = rng.normal(size=300)
            z2 = 0.7 * z1 + math.sqrt(1 - 0.49) * rng.normal(size=300)
            values.append(upper_tail_dependence(z1, z2, 0.95, method).coefficient)
        stats[method] = (float(np.nanmean(values)), float(np.nanstd(values)))

    assert stats["nonparametric"][1] >= stats["empirical"][1]
    assert stats["nonparametric"][0] < stats["empirical"][0]
    # Both track the same quantity: the gap is small next to the spread.
    assert abs(stats["nonparametric"][0] - stats["empirical"][0]) < 0.05


# --------------------------------------------------------------------------
# matrix, per-sample score, tail ENIV
# --------------------------------------------------------------------------

def test_tail_dependence_matrix_shape_and_symmetry():
    rng = np.random.default_rng(10)
    matrix, counts = tail_dependence_matrix(rng.normal(size=(600, 4)), quantile=0.9)
    assert matrix.shape == (4, 4) and counts.shape == (4, 4)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 1.0)
    assert np.all(counts[~np.eye(4, dtype=bool)] > 0)


def test_lower_tail_dependence_mirrors_upper_under_negation():
    rng = np.random.default_rng(11)
    z1 = rng.normal(size=4000)
    z2 = 0.8 * z1 + math.sqrt(1 - 0.64) * rng.normal(size=4000)
    upper = upper_tail_dependence(z1, z2, quantile=0.95).coefficient
    lower = lower_tail_dependence(-z1, -z2, quantile=0.05).coefficient
    assert abs(upper - lower) < 0.12


def test_co_exceedance_score_is_per_sample_and_bounded():
    rng = np.random.default_rng(12)
    scores = co_exceedance_score(rng.normal(size=(400, 4)), quantile=0.95)
    assert scores.shape == (400,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)


def test_co_exceedance_score_fires_when_all_views_are_jointly_extreme():
    """One sample with every view at its maximum must score 1.0."""
    rng = np.random.default_rng(13)
    x = rng.normal(size=(200, 4))
    x[0, :] = 50.0
    scores = co_exceedance_score(x, quantile=0.95)
    assert scores[0] == pytest.approx(1.0)


def test_tail_agreement_score_is_continuous_and_bounded():
    """The co-exceedance count is too coarse to rank samples; this is not.

    For 4 views the fraction-of-pairs score takes at most 7 distinct values, so
    an ROC built on it is nearly degenerate. The max-pair-min score is
    continuous, which is why the experiment's detector uses it.
    """
    rng = np.random.default_rng(20)
    x = rng.normal(size=(400, 4))
    scores = tail_agreement_score(x)
    assert scores.shape == (400,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)
    assert len(np.unique(scores)) > len(np.unique(co_exceedance_score(x, 0.95)))


def test_tail_agreement_score_is_highest_when_a_pair_is_jointly_extreme():
    rng = np.random.default_rng(21)
    x = rng.normal(size=(300, 4))
    x[0, 0] = x[0, 1] = 100.0  # one pair jointly at the top
    scores = tail_agreement_score(x)
    assert scores.argmax() == 0


def test_tail_agreement_score_ignores_a_single_extreme_view():
    """One loud view is not agreement; the score must need a PAIR."""
    rng = np.random.default_rng(22)
    x = rng.normal(size=(300, 4))
    x[0, 0] = 100.0
    x[0, 1:] = -100.0
    scores = tail_agreement_score(x)
    assert scores[0] < np.median(scores)


def test_tail_eniv_matches_design_effect_formula():
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    assert tail_eniv(matrix) == pytest.approx(2.0 / (1.0 + 0.5))


def test_tail_eniv_is_full_count_at_zero_tail_dependence():
    assert tail_eniv(np.eye(4)) == pytest.approx(4.0)


def test_tail_eniv_collapses_to_one_under_total_tail_dependence():
    assert tail_eniv(np.ones((4, 4))) == pytest.approx(1.0)


def test_tail_eniv_is_monotone_decreasing_in_tail_dependence():
    previous = 5.0
    for value in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        matrix = np.full((4, 4), value)
        np.fill_diagonal(matrix, 1.0)
        current = tail_eniv(matrix)
        assert current <= previous + 1e-12
        previous = current
