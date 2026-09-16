import inspect

import numpy as np
import pytest
import torch
from torch import nn

import prismflow.eniv.eniv as eniv_module
from prismflow.data.synthetic import analytic_n_eff
from prismflow.eniv.eniv import ENIVResult, analytic_eniv, compute_eniv, mean_off_diagonal
from prismflow.statistics.dependence import dependence_matrix


def _matrix(n_views, rho):
    matrix = np.full((n_views, n_views), float(rho))
    np.fill_diagonal(matrix, 1.0)
    return matrix


# --- the two anchor cases ----------------------------------------------


def test_eniv_of_five_identical_views_is_near_one():
    result = compute_eniv(_matrix(5, 1.0))
    assert result.effective_views == pytest.approx(1.0, abs=1e-6)
    assert result.efficiency_ratio == pytest.approx(0.2, abs=1e-6)


def test_eniv_of_five_independent_views_is_near_five():
    result = compute_eniv(_matrix(5, 0.0))
    assert result.effective_views == pytest.approx(5.0, abs=1e-6)
    assert result.efficiency_ratio == pytest.approx(1.0, abs=1e-6)


def test_eniv_from_actual_identical_evidence():
    rng = np.random.default_rng(0)
    base = np.abs(rng.standard_normal((800, 1, 3)))
    evidence = np.concatenate([base] * 5, axis=1)

    result = compute_eniv(dependence_matrix(evidence))
    assert result.effective_views == pytest.approx(1.0, abs=0.05)


def test_eniv_from_actual_independent_evidence():
    rng = np.random.default_rng(1)
    evidence = np.abs(rng.standard_normal((3000, 5, 3)))

    result = compute_eniv(dependence_matrix(evidence))
    assert result.effective_views == pytest.approx(5.0, abs=0.5)


# --- design effect ------------------------------------------------------


def test_matches_the_analytic_design_effect():
    for rho in (0.0, 0.2, 0.5, 0.8, 0.95):
        result = compute_eniv(_matrix(4, rho))
        assert result.effective_views == pytest.approx(analytic_n_eff(rho, 4), abs=1e-6)


def test_analytic_eniv_agrees_with_the_data_layer():
    for rho in (0.0, 0.3, 0.75):
        assert analytic_eniv(rho, 4) == pytest.approx(analytic_n_eff(rho, 4))


def test_analytic_eniv_validates_inputs():
    with pytest.raises(ValueError):
        analytic_eniv(1.5, 4)
    with pytest.raises(ValueError):
        analytic_eniv(0.3, 0)


def test_effective_views_decreases_monotonically_in_dependence():
    values = [compute_eniv(_matrix(4, rho)).effective_views for rho in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert values == sorted(values, reverse=True)


# --- clamping -----------------------------------------------------------


def test_effective_views_clamped_to_the_view_count():
    """Negative sampling noise around true independence must not manufacture
    more effective views than there are views."""
    result = compute_eniv(_matrix(4, -0.2))
    assert result.effective_views == pytest.approx(4.0)
    assert result.mean_dependence == pytest.approx(-0.2)


def test_effective_views_never_below_one():
    result = compute_eniv(_matrix(6, 1.0))
    assert result.effective_views >= 1.0


def test_pathological_anticorrelation_does_not_divide_by_zero():
    result = compute_eniv(_matrix(3, -0.5))
    assert np.isfinite(result.effective_views)
    assert result.effective_views == pytest.approx(3.0)


def test_raw_mean_dependence_is_reported_unclamped():
    result = compute_eniv(_matrix(4, -0.1))
    assert result.mean_dependence < 0.0


# --- availability -------------------------------------------------------


def test_only_available_views_enter_the_design_effect():
    matrix = _matrix(4, 1.0)
    available = np.array([True, True, False, False])

    result = compute_eniv(matrix, available)
    assert result.nominal_views == 2
    assert result.effective_views == pytest.approx(1.0)


def test_single_available_view_gives_one_effective_view():
    result = compute_eniv(_matrix(4, 0.9), np.array([True, False, False, False]))
    assert result.nominal_views == 1
    assert result.effective_views == pytest.approx(1.0)
    assert result.efficiency_ratio == pytest.approx(1.0)


def test_no_available_views_is_degenerate_not_a_crash():
    result = compute_eniv(_matrix(3, 0.5), np.array([False, False, False]))
    assert result.nominal_views == 0
    assert result.effective_views == 0.0


def test_unmeasurable_pairs_are_skipped_not_counted_as_zero():
    matrix = _matrix(3, 0.8)
    matrix[0, 2] = matrix[2, 0] = np.nan
    assert mean_off_diagonal(matrix) == pytest.approx(0.8)


def test_all_pairs_unmeasurable_falls_back_to_independence():
    matrix = np.eye(2)
    matrix[0, 1] = matrix[1, 0] = np.nan
    assert mean_off_diagonal(matrix) == 0.0


def test_rejects_non_square_input():
    with pytest.raises(ValueError):
        compute_eniv(np.zeros((3, 4)))


def test_rejects_mismatched_available_mask():
    with pytest.raises(ValueError):
        compute_eniv(_matrix(3, 0.5), np.array([True, False]))


# --- ENIV is not trainable ---------------------------------------------


def test_eniv_module_defines_no_nn_modules():
    """The contract classifies ENIV as a STATISTICAL ESTIMATOR: no learnable
    parameters in V1, and explicitly not an nn.Module."""
    for _, obj in inspect.getmembers(eniv_module, inspect.isclass):
        if obj.__module__ == eniv_module.__name__:
            assert not issubclass(obj, nn.Module)


def test_eniv_result_is_immutable_plain_data():
    result = compute_eniv(_matrix(3, 0.5))
    assert isinstance(result, ENIVResult)
    with pytest.raises(Exception):
        result.effective_views = 2.0

    for value in (result.effective_views, result.efficiency_ratio, result.mean_dependence):
        assert isinstance(value, float)
        assert not torch.is_tensor(value)


def test_eniv_carries_no_gradient_from_torch_evidence():
    evidence = torch.rand(600, 3, 3, requires_grad=True)
    result = compute_eniv(dependence_matrix(evidence))

    assert isinstance(result.effective_views, float)
    assert evidence.grad is None
