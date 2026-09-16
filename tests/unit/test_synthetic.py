import numpy as np
import pytest

from prismflow.data.dataset import split_indices
from prismflow.data.synthetic import (
    SyntheticConfig,
    analytic_n_eff,
    empirical_cross_view_correlation,
    generate_synthetic_dataset,
)


def _mean_offdiag(corr: np.ndarray) -> float:
    n = corr.shape[0]
    idx = np.triu_indices(n, k=1)
    return float(corr[idx].mean())


# --- analytic_n_eff -----------------------------------------------------


def test_analytic_n_eff_zero_rho_gives_full_count():
    assert analytic_n_eff(0.0, 5) == pytest.approx(5.0)


def test_analytic_n_eff_full_rho_gives_one():
    assert analytic_n_eff(1.0, 5) == pytest.approx(1.0)


def test_analytic_n_eff_monotonic_in_rho():
    values = [analytic_n_eff(r, 6) for r in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))


def test_analytic_n_eff_invalid_rho_raises():
    with pytest.raises(ValueError):
        analytic_n_eff(1.5, 5)


# --- empirical cross-view correlation tracks rho ------------------------

_CORR_TEST_KWARGS = dict(n_views=4, n_classes=3, d_latent=4, d_view=6, n_samples=3000, seed=0)


def test_rho_zero_gives_near_zero_correlation():
    cfg = SyntheticConfig(rho=0.0, **_CORR_TEST_KWARGS)
    data = generate_synthetic_dataset(cfg)
    corr = empirical_cross_view_correlation(data["views"])
    assert _mean_offdiag(corr) < 0.15


def test_rho_high_gives_high_correlation():
    cfg = SyntheticConfig(rho=0.9, **_CORR_TEST_KWARGS)
    data = generate_synthetic_dataset(cfg)
    corr = empirical_cross_view_correlation(data["views"])
    assert _mean_offdiag(corr) > 0.6


def test_correlation_increases_monotonically_with_rho():
    rhos = [0.0, 0.2, 0.4, 0.6, 0.8]
    means = []
    for rho in rhos:
        cfg = SyntheticConfig(rho=rho, **_CORR_TEST_KWARGS)
        data = generate_synthetic_dataset(cfg)
        corr = empirical_cross_view_correlation(data["views"])
        means.append(_mean_offdiag(corr))

    assert all(means[i] < means[i + 1] for i in range(len(means) - 1)), means


def test_correlation_matrix_shape_and_diagonal():
    cfg = SyntheticConfig(rho=0.5, **_CORR_TEST_KWARGS)
    data = generate_synthetic_dataset(cfg)
    corr = empirical_cross_view_correlation(data["views"])
    assert corr.shape == (cfg.n_views, cfg.n_views)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(corr, corr.T)


# --- generator config validation ----------------------------------------


def test_invalid_scalar_rho_raises():
    cfg = SyntheticConfig(rho=1.5, n_samples=10)
    with pytest.raises(ValueError):
        generate_synthetic_dataset(cfg)


def test_matrix_rho_scalar_equivalence():
    n_views = 3
    matrix = np.full((n_views, n_views), 0.4)
    np.fill_diagonal(matrix, 1.0)

    cfg_matrix = SyntheticConfig(n_views=n_views, rho=matrix, n_samples=50, seed=1)
    cfg_scalar = SyntheticConfig(n_views=n_views, rho=0.4, n_samples=50, seed=1)

    data_matrix = generate_synthetic_dataset(cfg_matrix)
    data_scalar = generate_synthetic_dataset(cfg_scalar)

    assert np.allclose(data_matrix["views"], data_scalar["views"])


def test_noise_std_wrong_length_raises():
    cfg = SyntheticConfig(n_views=4, noise_std=[0.1, 0.2], n_samples=10)
    with pytest.raises(ValueError):
        generate_synthetic_dataset(cfg)


def test_generate_is_deterministic_given_seed():
    cfg = SyntheticConfig(n_samples=100, seed=42)
    data1 = generate_synthetic_dataset(cfg)
    data2 = generate_synthetic_dataset(cfg)
    assert np.array_equal(data1["views"], data2["views"])
    assert np.array_equal(data1["labels"], data2["labels"])


def test_output_shapes():
    cfg = SyntheticConfig(n_views=5, n_samples=37, d_view=12)
    data = generate_synthetic_dataset(cfg)
    assert data["views"].shape == (37, 5, 12)
    assert data["labels"].shape == (37,)
    assert data["rho_matrix"].shape == (5, 5)


# --- naive baseline accuracy: calibrates signal_strength -----------------


def _nearest_centroid_accuracy(cfg: SyntheticConfig, split_seed: int = 0) -> float:
    """Simple, model-free baseline: nearest-centroid on concatenated views.

    Intentionally not part of prismflow (no ML code in this Part) -- this
    exists only to check the generator is calibrated to a non-trivial
    difficulty (see PART 02 spec: baseline accuracy must land in [0.70, 0.88]
    so later calibration experiments have headroom).
    """
    data = generate_synthetic_dataset(cfg)
    views, labels = data["views"], data["labels"]
    n_samples = views.shape[0]
    features = views.reshape(n_samples, -1)

    splits = split_indices(n_samples, seed=split_seed)
    train_idx, test_idx = splits["train"], splits["test"]

    classes = np.unique(labels[train_idx])
    centroids = np.stack(
        [features[train_idx][labels[train_idx] == c].mean(axis=0) for c in classes]
    )

    dists = np.linalg.norm(features[test_idx][:, None, :] - centroids[None, :, :], axis=2)
    preds = classes[np.argmin(dists, axis=1)]
    return float((preds == labels[test_idx]).mean())


def test_default_config_baseline_accuracy_has_headroom():
    cfg = SyntheticConfig()
    accuracy = _nearest_centroid_accuracy(cfg)
    assert 0.70 <= accuracy <= 0.88, f"baseline accuracy {accuracy} outside [0.70, 0.88]"
