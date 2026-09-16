import inspect

import numpy as np
import pytest
import torch
from torch import nn

import prismflow.eniv.eniv as eniv_module
from prismflow.data.synthetic import analytic_n_eff
from prismflow.data.synthetic import analytic_n_eff_eigen
from prismflow.eniv.eniv import (
    ENIVResult,
    analytic_eniv,
    analytic_eniv_eigen,
    compute_eniv,
    eigen_n_eff,
    mean_off_diagonal,
    per_view_alpha,
    vif_alpha,
)
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
        result = compute_eniv(_matrix(4, rho), method="design_effect")
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
    result = compute_eniv(_matrix(4, -0.2), method="design_effect")
    assert result.effective_views == pytest.approx(4.0)
    assert result.mean_dependence == pytest.approx(-0.2)


def test_effective_views_never_below_one():
    result = compute_eniv(_matrix(6, 1.0))
    assert result.effective_views >= 1.0


def test_pathological_anticorrelation_does_not_divide_by_zero():
    result = compute_eniv(_matrix(3, -0.5), method="design_effect")
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


# --- eigenvalue ENIV (default) -------------------------------------------


def test_eigen_is_the_default_method():
    matrix = _matrix(4, 0.5)
    assert compute_eniv(matrix).effective_views == pytest.approx(
        compute_eniv(matrix, method="eigen").effective_views
    )
    assert compute_eniv(matrix).effective_views != pytest.approx(
        compute_eniv(matrix, method="design_effect").effective_views
    )


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        compute_eniv(_matrix(3, 0.5), method="participation_ratio")


@pytest.mark.parametrize("n_views", [4, 8])
@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9, 1.0])
def test_eigen_matches_its_analytic_ground_truth(rho, n_views):
    result = compute_eniv(_matrix(n_views, rho))
    assert result.effective_views == pytest.approx(analytic_n_eff_eigen(rho, n_views), abs=1e-9)


def test_analytic_eniv_eigen_agrees_with_the_data_layer():
    for rho in (0.0, 0.25, 0.8):
        assert analytic_eniv_eigen(rho, 6) == pytest.approx(analytic_n_eff_eigen(rho, 6))


def _clone_evidence(k, n_samples=4000, base_views=4, seed=0):
    """4 independent views plus k bit-identical copies of view 0."""
    rng = np.random.default_rng(seed)
    base = np.abs(rng.standard_normal((n_samples, base_views, 3)))
    copies = np.repeat(base[:, :1, :], k, axis=1)
    classes = rng.integers(0, 3, size=n_samples)
    return np.concatenate([base, copies], axis=1), classes


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_part06_clone_case_eniv_stays_at_four(k):
    """The property that failed in Part 06: duplicating a view must not change
    the effective count. The design effect fell from ~3.5 to 2.4 over k=0..4 on
    this structure and dragged PrismFlow's confidence down with it."""
    evidence, classes = _clone_evidence(k)
    matrix = dependence_matrix(evidence, classes=classes)

    assert compute_eniv(matrix).effective_views == pytest.approx(4.0, abs=0.15)


def test_part06_clone_case_design_effect_reproduces_the_bug():
    evidence, classes = _clone_evidence(4)
    matrix = dependence_matrix(evidence, classes=classes)

    design_effect = compute_eniv(matrix, method="design_effect").effective_views
    eigen = compute_eniv(matrix).effective_views

    assert design_effect < 2.6, "design effect should smear the clone block across all pairs"
    assert eigen > design_effect + 1.0


def test_exact_duplicate_block_counts_as_one_witness_regardless_of_size():
    for block in range(1, 7):
        matrix = np.eye(3 + block)
        group = [0] + list(range(3, 3 + block - 1))
        for a in group:
            for b in group:
                matrix[a, b] = 1.0
        assert eigen_n_eff(matrix) == pytest.approx(3.0 + 1.0, abs=1e-9)


def test_near_duplicate_block_contributes_one_plus_residual_independence():
    """Documented limit: a block of g views correlated at c < 1 contributes
    1 + (g - 1)(1 - c), not 1. At the c ~ 0.92 measured on clones that is ~0.08
    per extra copy."""
    c = 0.92
    for extra_copies in range(5):
        size = 4 + extra_copies
        matrix = np.eye(size)
        group = [0] + list(range(4, size))
        for a in group:
            for b in group:
                if a != b:
                    matrix[a, b] = c
        expected = 3.0 + 1.0 + extra_copies * (1.0 - c)
        assert eigen_n_eff(matrix) == pytest.approx(expected, abs=1e-9)


def test_copies_of_a_correlated_view_are_not_fully_absorbed():
    """Documented limit: the one-witness-per-block property needs the block to be
    independent of the other views. With base views at rho=0.5, exact copies of
    one view move the population value off 2.5."""
    values = []
    for k in range(5):
        size = 4 + k
        matrix = np.full((size, size), 0.5)
        np.fill_diagonal(matrix, 1.0)
        group = [0] + list(range(4, size))
        for a in group:
            for b in group:
                if a != b:
                    matrix[a, b] = 1.0
        values.append(eigen_n_eff(matrix))

    assert values[0] == pytest.approx(2.5, abs=1e-9)
    assert values[-1] > values[0] + 0.3


def test_eigen_is_biased_down_at_independence_in_finite_samples():
    """Documented limit: sum(lambda) = V, so capping at 1 can only lose mass, and
    sampling noise spreads the eigenvalues around 1."""
    rng = np.random.default_rng(3)
    readings = []
    for _ in range(20):
        data = rng.standard_normal((300, 8))
        readings.append(eigen_n_eff(np.corrcoef(data, rowvar=False)))
    assert np.mean(readings) < 8.0 - 0.2
    assert np.mean(readings) > 8.0 - 1.0


def test_negative_eigenvalues_are_clipped_not_subtracted():
    """A pairwise matrix from separate estimates need not be PSD. A negative
    eigenvalue is estimation error, not negative information."""
    matrix = _matrix(3, -0.9)
    assert np.min(np.linalg.eigvalsh(matrix)) < 0
    assert eigen_n_eff(matrix) >= 0.0
    assert eigen_n_eff(matrix) == pytest.approx(2.0, abs=1e-9)


def test_unmeasurable_pairs_are_imputed_with_the_measured_mean():
    matrix = _matrix(4, 0.6)
    matrix[0, 3] = matrix[3, 0] = np.nan
    assert eigen_n_eff(matrix) == pytest.approx(analytic_n_eff_eigen(0.6, 4), abs=1e-9)


def test_eigen_path_does_not_mutate_its_input():
    matrix = _matrix(4, 0.4)
    matrix[0, 1] = matrix[1, 0] = np.nan
    before = matrix.copy()
    eigen_n_eff(matrix)
    assert np.array_equal(before, matrix, equal_nan=True)


def test_eigen_respects_available_views():
    matrix = np.eye(5)
    matrix[0, 1] = matrix[1, 0] = 1.0
    available = np.array([True, True, True, False, False])
    result = compute_eniv(matrix, available)
    assert result.nominal_views == 3
    assert result.effective_views == pytest.approx(2.0, abs=1e-9)


# --- per-view discount factors ---------------------------------------------


def _clone_matrix(k, c=1.0, base_views=4):
    """4 independent views plus k copies of view 0, copy correlation c."""
    size = base_views + k
    matrix = np.eye(size)
    group = [0] + list(range(base_views, size))
    for a in group:
        for b in group:
            if a != b:
                matrix[a, b] = c
    return matrix, group


def test_per_view_alpha_is_one_for_independent_views():
    assert np.allclose(per_view_alpha(np.eye(5)), 1.0)


@pytest.mark.parametrize("n_views", [4, 8])
@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_per_view_alpha_matches_equicorrelated_closed_form(rho, n_views):
    """Row sums of the inverse equicorrelated matrix are 1 / (1 + (n-1) rho)."""
    expected = 1.0 / (1.0 + (n_views - 1) * rho)
    assert np.allclose(per_view_alpha(_matrix(n_views, rho)), expected, atol=1e-5)


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_part06_clone_untouched_views_keep_full_weight(k):
    """The bug being fixed: duplicating view 0 used to discount views 1-3."""
    matrix, _ = _clone_matrix(k)
    assert np.allclose(per_view_alpha(matrix)[1:4], 1.0, atol=1e-6)


@pytest.mark.parametrize("k", [0, 1, 2, 3, 4])
def test_part06_clone_cluster_counts_as_exactly_one_witness(k):
    matrix, group = _clone_matrix(k)
    alpha = per_view_alpha(matrix)
    assert np.allclose(alpha[group], 1.0 / len(group), atol=1e-5)
    assert alpha[group].sum() == pytest.approx(1.0, abs=1e-5)


def test_near_duplicate_cluster_weight_matches_closed_form():
    """At copy correlation c each of g members gets 1 / (1 + (g-1) c); the
    cluster total g / (1 + (g-1) c) sits slightly above one witness."""
    c = 0.92
    for k in range(5):
        matrix, group = _clone_matrix(k, c)
        g = len(group)
        alpha = per_view_alpha(matrix)
        assert np.allclose(alpha[group], 1.0 / (1.0 + (g - 1) * c), atol=1e-5)
        assert np.allclose(alpha[1:4], 1.0, atol=1e-6)


def test_per_view_alpha_is_insensitive_to_the_ridge_for_exact_copies():
    matrix, group = _clone_matrix(3)
    for ridge in (1e-9, 1e-6, 1e-3):
        assert np.allclose(per_view_alpha(matrix, ridge=ridge)[group], 0.25, atol=1e-3)


def test_vif_alpha_erases_exact_duplicate_clusters_documented_counterexample():
    """Why 1/VIF was not used: every member of an exact-copy cluster, the
    original included, goes to ~0 -- the witness is erased, not counted once."""
    matrix, group = _clone_matrix(2)
    alpha = vif_alpha(matrix)
    assert np.all(alpha[group] < 1e-4)
    assert np.allclose(alpha[1:4], 1.0)


def test_vif_alpha_matches_equicorrelated_closed_form():
    for n_views in (4, 8):
        for rho in (0.0, 0.3, 0.6, 0.9):
            vif = (1 + (n_views - 2) * rho) / ((1 - rho) * (1 + (n_views - 1) * rho))
            assert np.allclose(vif_alpha(_matrix(n_views, rho), ridge=0.0), 1.0 / vif, atol=1e-9)


def test_per_view_alpha_stays_in_unit_interval():
    rng = np.random.default_rng(7)
    for _ in range(50):
        raw = rng.uniform(-0.6, 1.0, size=(6, 6))
        matrix = np.clip(0.5 * (raw + raw.T), -1, 1)
        np.fill_diagonal(matrix, 1.0)
        alpha = per_view_alpha(matrix)
        assert np.all(alpha > 0.0) and np.all(alpha <= 1.0)


def test_negative_dependence_never_amplifies_evidence():
    assert np.all(per_view_alpha(_matrix(3, -0.4)) <= 1.0)


def test_per_view_alpha_respects_available_views():
    matrix, _ = _clone_matrix(1)  # views 0 and 4 are copies
    available = np.array([True, True, True, True, False])
    alpha = per_view_alpha(matrix, available)
    # with the copy absent, view 0 is no longer redundant
    assert np.allclose(alpha, 1.0, atol=1e-6)


def test_per_view_alpha_imputes_unmeasurable_pairs_and_does_not_mutate():
    matrix = _matrix(4, 0.6)
    matrix[0, 3] = matrix[3, 0] = np.nan
    before = matrix.copy()
    assert np.allclose(per_view_alpha(matrix), 1.0 / (1.0 + 3 * 0.6), atol=1e-5)
    assert np.array_equal(before, matrix, equal_nan=True)


def test_per_view_alpha_rejects_bad_shapes():
    with pytest.raises(ValueError):
        per_view_alpha(np.zeros((3, 4)))
    with pytest.raises(ValueError):
        per_view_alpha(np.eye(3), available=np.array([True, False]))


def test_eniv_result_per_view_alpha_defaults_to_none():
    assert compute_eniv(_matrix(3, 0.2)).per_view_alpha is None


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
