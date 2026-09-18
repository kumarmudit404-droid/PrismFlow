import numpy as np
import pytest
import torch

from prismflow.eniv.amortized import AmortizedConfig, AmortizedDependence, dependence_fit_loss
from prismflow.eniv.eniv import compute_eniv, soft_cluster_alpha
from prismflow.eniv.per_sample import (
    broadcast_global,
    compute_per_sample_eniv,
    per_sample_effective_views,
    per_sample_mean_dependence,
    per_sample_soft_cluster_alpha,
)
from prismflow.train_two_timescale import TwoTimescaleConfig, frozen
from prismflow.utils.seed import set_seed


def _matrix(rho, n_views=4):
    matrix = np.full((n_views, n_views), float(rho))
    np.fill_diagonal(matrix, 1.0)
    return matrix


# --- equivalence with V1 ------------------------------------------------
# The per-sample path must be a GENERALISATION of the global one, not a
# different mechanism. Handed one matrix for every sample it must reproduce V1.


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_alpha_matches_v1_when_every_sample_shares_a_matrix(rho):
    matrix = _matrix(rho)
    batched = broadcast_global(matrix, 16)
    assert per_sample_soft_cluster_alpha(batched) == pytest.approx(
        np.broadcast_to(soft_cluster_alpha(matrix), (16, 4)), abs=1e-9
    )


@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_effective_views_matches_v1_when_every_sample_shares_a_matrix(rho):
    matrix = _matrix(rho)
    expected = compute_eniv(matrix).effective_views
    got = per_sample_effective_views(broadcast_global(matrix, 8))
    assert got == pytest.approx(np.full(8, expected), abs=1e-9)


def test_mean_dependence_matches_v1():
    matrix = _matrix(0.4)
    assert per_sample_mean_dependence(broadcast_global(matrix, 5)) == pytest.approx(
        np.full(5, 0.4), abs=1e-9
    )


# --- the point of the Part: input-conditional dependence -----------------


def test_per_sample_eniv_separates_samples_a_global_matrix_cannot():
    """20% of samples fully dependent, the rest independent. A global matrix
    reports the average and is wrong about every sample; the per-sample path is
    right about both groups."""
    n = 100
    dependence = np.stack([_matrix(0.95)] * 20 + [_matrix(0.0)] * 80)
    result = compute_per_sample_eniv(dependence)

    collapsed = result.effective_views[:20]
    independent = result.effective_views[20:]
    assert collapsed.mean() < 1.5, "views that agree completely should count as ~1"
    assert independent.mean() > 3.9, "independent views should count as ~4"

    # What a global estimator would have said: one number, wrong for both.
    global_value = compute_eniv(dependence.mean(axis=0)).effective_views
    assert collapsed.mean() < global_value < independent.mean()


def test_alpha_is_smaller_for_the_dependent_samples():
    dependence = np.stack([_matrix(0.9)] * 10 + [_matrix(0.0)] * 10)
    alpha = per_sample_soft_cluster_alpha(dependence)
    assert alpha[:10].mean() < 0.5
    assert alpha[10:].mean() == pytest.approx(1.0, abs=1e-9)


# --- ranges and masks ---------------------------------------------------


def test_effective_views_is_clamped_to_the_available_count():
    dependence = np.stack([_matrix(-0.5), _matrix(0.0), _matrix(1.0)])
    values = per_sample_effective_views(dependence)
    assert (values >= 1.0).all() and (values <= 4.0).all()


def test_alpha_stays_in_the_unit_interval():
    set_seed(0)
    dependence = np.clip(np.random.default_rng(0).normal(0.3, 0.5, (32, 4, 4)), -1, 1)
    for index in range(32):
        np.fill_diagonal(dependence[index], 1.0)
    alpha = per_sample_soft_cluster_alpha(dependence)
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()


def test_absent_views_get_zero_alpha_and_are_excluded_from_the_count():
    dependence = broadcast_global(_matrix(0.0), 4)
    view_mask = np.ones((4, 4), dtype=bool)
    view_mask[:, 3] = False

    alpha = per_sample_soft_cluster_alpha(dependence, view_mask)
    assert alpha[:, 3] == pytest.approx(np.zeros(4))
    assert per_sample_effective_views(dependence, view_mask) == pytest.approx(np.full(4, 3.0), abs=1e-6)


def test_a_single_present_view_gives_one_effective_view():
    dependence = broadcast_global(_matrix(0.5), 3)
    view_mask = np.zeros((3, 4), dtype=bool)
    view_mask[:, 0] = True
    assert per_sample_effective_views(dependence, view_mask) == pytest.approx(np.ones(3))


def test_shape_validation():
    with pytest.raises(ValueError):
        per_sample_soft_cluster_alpha(np.zeros((4, 4)))
    with pytest.raises(ValueError):
        per_sample_effective_views(np.zeros((2, 4, 3)))
    with pytest.raises(ValueError):
        broadcast_global(np.zeros((4, 3)), 2)


def test_design_effect_method_is_available_and_differs():
    dependence = broadcast_global(_matrix(0.5), 4)
    eigen = per_sample_effective_views(dependence, method="eigen")
    design = per_sample_effective_views(dependence, method="design_effect")
    assert design == pytest.approx(np.full(4, 4 / (1 + 3 * 0.5)), abs=1e-6)
    assert not np.allclose(eigen, design)


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError):
        per_sample_effective_views(broadcast_global(_matrix(0.2), 2), method="pca")


# --- the amortized estimator --------------------------------------------


def _estimator(n_views=4, n_classes=3, seed=0):
    set_seed(seed)
    return AmortizedDependence(n_views, n_classes, AmortizedConfig(enabled=True))


def test_output_is_symmetric_with_unit_diagonal_in_the_unit_interval():
    estimator = _estimator()
    set_seed(1)
    matrix = estimator(torch.rand(16, 4, 3) * 5).detach()
    assert matrix.shape == (16, 4, 4)
    assert matrix.numpy() == pytest.approx(matrix.transpose(1, 2).numpy(), abs=1e-6)
    assert torch.diagonal(matrix, dim1=1, dim2=2).numpy() == pytest.approx(np.ones((16, 4)), abs=1e-6)
    assert (matrix >= 0).all() and (matrix <= 1).all()


def test_symmetry_is_structural_not_learned():
    """Holds at initialisation, before any training -- it comes from the
    symmetric pair features, not from a fitted weight."""
    estimator = _estimator(seed=7)
    matrix = estimator(torch.randn(8, 4, 3).abs()).detach()
    assert matrix.numpy() == pytest.approx(matrix.transpose(1, 2).numpy(), abs=1e-6)


def test_swapping_two_views_permutes_the_matrix_the_same_way():
    estimator = _estimator()
    set_seed(3)
    evidence = torch.rand(8, 4, 3) * 4
    swapped = evidence[:, [1, 0, 2, 3], :]

    base = estimator(evidence).detach().numpy()
    permuted = estimator(swapped).detach().numpy()
    assert permuted[:, 0, 2] == pytest.approx(base[:, 1, 2], abs=1e-5)
    assert permuted[:, 0, 1] == pytest.approx(base[:, 1, 0], abs=1e-5)


def test_absent_views_are_decoupled_in_the_output():
    estimator = _estimator()
    view_mask = torch.ones(8, 4, dtype=torch.bool)
    view_mask[:, 2] = False
    matrix = estimator(torch.rand(8, 4, 3) * 3, view_mask).detach().numpy()
    assert matrix[:, 2, 0] == pytest.approx(np.zeros(8), abs=1e-6)
    assert matrix[:, 2, 2] == pytest.approx(np.ones(8), abs=1e-6)


def test_estimator_output_feeds_the_per_sample_arithmetic():
    estimator = _estimator()
    matrix = estimator(torch.rand(16, 4, 3) * 3).detach().numpy()
    result = compute_per_sample_eniv(matrix)
    assert result.effective_views.shape == (16,)
    assert result.alpha.shape == (16, 4)
    assert np.isfinite(result.effective_views).all()


def test_fit_loss_is_zero_against_its_own_prediction():
    estimator = _estimator()
    predicted = estimator(torch.rand(8, 4, 3) * 3)
    assert float(dependence_fit_loss(predicted, predicted.detach()).detach()) == pytest.approx(0.0, abs=1e-9)


def test_fit_loss_ignores_nan_targets():
    estimator = _estimator()
    predicted = estimator(torch.rand(8, 4, 3) * 3)
    target = np.full((4, 4), np.nan)
    assert torch.isfinite(dependence_fit_loss(predicted, target))


def test_fit_loss_trains_the_estimator_toward_a_target():
    estimator = _estimator()
    optimiser = torch.optim.Adam(estimator.parameters(), lr=0.05)
    evidence = torch.rand(64, 4, 3) * 3
    target = torch.as_tensor(_matrix(0.8), dtype=torch.float32)

    before = float(dependence_fit_loss(estimator(evidence), target).detach())
    for _ in range(80):
        optimiser.zero_grad()
        dependence_fit_loss(estimator(evidence), target).backward()
        optimiser.step()
    assert float(dependence_fit_loss(estimator(evidence), target).detach()) < before / 2


def test_estimator_rejects_wrong_shapes():
    estimator = _estimator()
    with pytest.raises(ValueError):
        estimator(torch.rand(8, 3))
    with pytest.raises(ValueError):
        estimator(torch.rand(8, 5, 3))


# --- two-timescale machinery --------------------------------------------


def test_frozen_blocks_gradient_and_restores_afterwards():
    estimator = _estimator()
    assert all(p.requires_grad for p in estimator.parameters())
    with frozen(estimator):
        assert not any(p.requires_grad for p in estimator.parameters())
    assert all(p.requires_grad for p in estimator.parameters())


def test_frozen_module_parameters_do_not_move_under_an_optimiser_step():
    """The guarantee alternating optimisation rests on."""
    estimator = _estimator()
    optimiser = torch.optim.Adam(estimator.parameters(), lr=0.1)
    before = [p.clone() for p in estimator.parameters()]

    evidence = torch.rand(32, 4, 3, requires_grad=True) * 3
    with frozen(estimator):
        loss = estimator(evidence).sum()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    assert all(torch.equal(a, b) for a, b in zip(before, estimator.parameters()))


def test_config_validation():
    TwoTimescaleConfig().validate()
    with pytest.raises(ValueError):
        TwoTimescaleConfig(encoder_steps=0).validate()
    with pytest.raises(ValueError):
        TwoTimescaleConfig(estimator_steps=0).validate()
    with pytest.raises(ValueError):
        TwoTimescaleConfig(epochs=0).validate()


def test_default_ratio_is_five_to_one():
    config = TwoTimescaleConfig()
    assert (config.encoder_steps, config.estimator_steps) == (5, 1)


def test_amortized_is_disabled_by_default():
    assert AmortizedConfig().enabled is False


def test_amortized_config_validation():
    with pytest.raises(ValueError):
        AmortizedConfig(hidden_dim=0).validate()
    with pytest.raises(ValueError):
        AmortizedConfig(method="mine").validate()
