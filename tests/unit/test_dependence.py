import numpy as np
import pytest
import torch

from prismflow.statistics.dependence import (
    available_views,
    class_conditional_residuals,
    dependence_matrix,
    predicted_classes,
)


def _evidence(rng, n_samples=800, n_views=3, n_classes=3, scale=1.0):
    return np.abs(rng.standard_normal((n_samples, n_views, n_classes))) * scale


# --- matrix shape and structure ----------------------------------------


def test_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(0)
    matrix = dependence_matrix(_evidence(rng, n_views=4))

    assert matrix.shape == (4, 4)
    assert np.allclose(matrix, matrix.T, equal_nan=True)
    assert np.allclose(np.diag(matrix), 1.0)


def test_accepts_torch_input_and_returns_numpy():
    evidence = torch.rand(400, 3, 3)
    matrix = dependence_matrix(evidence)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (3, 3)


def test_detached_from_autograd():
    evidence = torch.rand(400, 3, 3, requires_grad=True)
    matrix = dependence_matrix(evidence)
    assert isinstance(matrix, np.ndarray)
    assert evidence.grad is None


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        dependence_matrix(torch.rand(100, 3, 3), method="kendall")


def test_rejects_wrong_rank():
    with pytest.raises(ValueError):
        dependence_matrix(torch.rand(100, 3))


# --- the two anchor cases ----------------------------------------------


def test_identical_views_give_dependence_near_one():
    rng = np.random.default_rng(1)
    base = _evidence(rng, n_views=1)
    evidence = np.concatenate([base, base, base], axis=1)

    matrix = dependence_matrix(evidence)
    off_diagonal = matrix[np.triu_indices(3, k=1)]
    assert np.all(off_diagonal > 0.99)


def test_independent_views_give_dependence_near_zero():
    rng = np.random.default_rng(2)
    evidence = _evidence(rng, n_samples=3000, n_views=4)
    classes = rng.integers(0, 3, size=3000)

    matrix = dependence_matrix(evidence, classes=classes)
    off_diagonal = matrix[np.triu_indices(4, k=1)]
    assert np.all(np.abs(off_diagonal) < 0.05)


# --- collider bias ------------------------------------------------------


def test_global_conditioning_induces_negative_bias_on_independent_views():
    """Conditioning on a class predicted FROM the evidence conditions on a
    collider: given the batch prediction, a low reading from one view implies
    a high reading from another. Independent views then measure as negatively
    correlated. Documented because the bias understates dependence, which
    under-discounts and leaves the model overconfident."""
    rng = np.random.default_rng(20)
    evidence = _evidence(rng, n_samples=4000, n_views=4)

    biased = dependence_matrix(evidence, conditioning="global")
    off_diagonal = biased[np.triu_indices(4, k=1)]
    assert off_diagonal.mean() < -0.05


def test_pairwise_holdout_removes_most_of_the_collider_bias():
    rng = np.random.default_rng(21)
    evidence = _evidence(rng, n_samples=4000, n_views=4)

    biased = dependence_matrix(evidence, conditioning="global")
    corrected = dependence_matrix(evidence, conditioning="pairwise_holdout")

    biased_mean = biased[np.triu_indices(4, k=1)].mean()
    corrected_mean = corrected[np.triu_indices(4, k=1)].mean()

    assert abs(corrected_mean) < abs(biased_mean) / 2
    assert abs(corrected_mean) < 0.05


def test_pairwise_holdout_still_recovers_real_dependence():
    rng = np.random.default_rng(22)
    shared = rng.standard_normal((3000, 1, 3))
    private = rng.standard_normal((3000, 4, 3))
    evidence = np.abs(np.sqrt(0.6) * shared + np.sqrt(0.4) * private)

    upper = np.triu_indices(4, k=1)
    corrected = dependence_matrix(evidence, conditioning="pairwise_holdout")[upper]
    biased = dependence_matrix(evidence, conditioning="global")[upper]

    # The half-normal transform attenuates the underlying 0.6 correlation, so
    # the recovered value is well below it; what matters is that the signal
    # survives and that removing the collider does not eat it.
    assert np.all(corrected > 0.2)
    assert corrected.mean() > biased.mean()


def test_holdout_falls_back_to_global_with_two_views():
    rng = np.random.default_rng(23)
    evidence = _evidence(rng, n_samples=500, n_views=2)

    assert np.allclose(
        dependence_matrix(evidence, conditioning="pairwise_holdout"),
        dependence_matrix(evidence, conditioning="global"),
    )


def test_explicit_classes_override_the_conditioning_mode():
    rng = np.random.default_rng(24)
    evidence = _evidence(rng, n_samples=500, n_views=3)
    classes = rng.integers(0, 3, size=500)

    assert np.allclose(
        dependence_matrix(evidence, classes=classes, conditioning="global"),
        dependence_matrix(evidence, classes=classes, conditioning="pairwise_holdout"),
    )


def test_rejects_unknown_conditioning_mode():
    with pytest.raises(ValueError):
        dependence_matrix(torch.rand(100, 3, 3), conditioning="leave_one_out")


def test_dependence_increases_with_shared_signal():
    rng = np.random.default_rng(3)
    measured = []
    for rho in (0.0, 0.3, 0.6, 0.9):
        shared = rng.standard_normal((2000, 1, 3))
        private = rng.standard_normal((2000, 2, 3))
        evidence = np.abs(
            np.sqrt(rho) * shared + np.sqrt(1.0 - rho) * private
        )
        measured.append(dependence_matrix(evidence)[0, 1])

    assert measured == sorted(measured), f"not monotone in rho: {measured}"
    assert measured[-1] > measured[0] + 0.3


# --- class conditioning -------------------------------------------------


def test_conditioning_removes_agreement_the_label_explains():
    """Views driven by the same label but otherwise independent are doing the
    task correctly, not being redundant. Unconditioned correlation cannot tell
    the difference; that is the whole reason residuals are taken."""
    rng = np.random.default_rng(4)
    n_samples, n_classes = 3000, 3

    labels = rng.integers(0, n_classes, size=n_samples)
    class_signal = np.eye(n_classes)[labels] * 4.0

    evidence = np.stack(
        [
            class_signal + rng.standard_normal((n_samples, n_classes)),
            class_signal + rng.standard_normal((n_samples, n_classes)),
        ],
        axis=1,
    )

    conditioned = dependence_matrix(evidence, classes=labels)[0, 1]

    flat = evidence.reshape(n_samples, -1)
    unconditioned = np.corrcoef(flat[:, :n_classes].ravel(), flat[:, n_classes:].ravel())[0, 1]

    assert abs(conditioned) < 0.08
    assert unconditioned > 0.4


def test_residuals_are_zero_mean_within_each_class():
    rng = np.random.default_rng(5)
    evidence = _evidence(rng, n_samples=600, n_views=2)
    classes = rng.integers(0, 3, size=600)

    residuals, valid = class_conditional_residuals(evidence, classes)

    for stratum in np.unique(classes):
        rows = classes == stratum
        assert np.allclose(residuals[rows, 0, :].mean(axis=0), 0.0, atol=1e-10)
    assert valid.all()


def test_predicted_classes_used_when_none_supplied():
    rng = np.random.default_rng(6)
    evidence = _evidence(rng, n_samples=200, n_views=3)
    derived = predicted_classes(evidence)

    assert derived.shape == (200,)
    assert np.array_equal(
        dependence_matrix(evidence), dependence_matrix(evidence, classes=derived)
    )


def test_rejects_mismatched_class_length():
    with pytest.raises(ValueError):
        dependence_matrix(torch.rand(100, 3, 3), classes=np.zeros(50, dtype=int))


# --- view masking -------------------------------------------------------


def test_unavailable_pair_is_nan_not_zero():
    """Refusing to estimate must not be recorded as 'independent' -- that
    would silently license full confidence on an unmeasured pair."""
    rng = np.random.default_rng(7)
    evidence = _evidence(rng, n_samples=200, n_views=3)

    mask = np.ones((200, 3), dtype=bool)
    mask[:, 2] = False

    matrix = dependence_matrix(evidence, view_mask=mask)
    assert np.isnan(matrix[0, 2])
    assert np.isnan(matrix[1, 2])
    assert not np.isnan(matrix[0, 1])


def test_masked_samples_excluded_from_the_estimate():
    rng = np.random.default_rng(8)
    base = _evidence(rng, n_samples=1000, n_views=1)
    evidence = np.concatenate([base, base], axis=1)

    # corrupt view 1 on half the samples, then mask exactly those away
    mask = np.ones((1000, 2), dtype=bool)
    mask[:500, 1] = False
    evidence[:500, 1, :] = rng.standard_normal((500, 3))

    assert dependence_matrix(evidence, view_mask=mask)[0, 1] > 0.99


def test_available_views_reports_presence():
    mask = torch.ones(10, 3, dtype=torch.bool)
    mask[:, 1] = False
    assert np.array_equal(available_views(mask), np.array([True, False, True]))
    assert np.array_equal(available_views(None, n_views=2), np.array([True, True]))


def test_available_views_needs_one_of_mask_or_count():
    with pytest.raises(ValueError):
        available_views(None)


# --- distance correlation -----------------------------------------------


def test_dcor_matrix_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(9)
    matrix = dependence_matrix(_evidence(rng, n_samples=400, n_views=3), method="dcor")

    assert np.allclose(matrix, matrix.T, equal_nan=True)
    assert np.allclose(np.diag(matrix), 1.0)


def test_dcor_is_non_negative():
    rng = np.random.default_rng(10)
    matrix = dependence_matrix(_evidence(rng, n_samples=400, n_views=3), method="dcor")
    off_diagonal = matrix[np.triu_indices(3, k=1)]
    assert np.all(off_diagonal >= 0.0)


def test_dcor_near_one_for_identical_views():
    rng = np.random.default_rng(11)
    base = _evidence(rng, n_samples=400, n_views=1)
    evidence = np.concatenate([base, base], axis=1)
    assert dependence_matrix(evidence, method="dcor")[0, 1] > 0.99


def test_dcor_detects_nonlinear_dependence_pearson_misses():
    """y = x^2 with symmetric x: linearly uncorrelated, entirely dependent."""
    rng = np.random.default_rng(12)
    n_samples = 500
    x = rng.standard_normal((n_samples, 1, 2))
    evidence = np.concatenate([x, x**2], axis=1)
    classes = np.zeros(n_samples, dtype=int)

    pearson = dependence_matrix(evidence, classes=classes, method="pearson")[0, 1]
    dcor = dependence_matrix(evidence, classes=classes, method="dcor")[0, 1]

    assert abs(pearson) < 0.15
    assert dcor > 0.3
