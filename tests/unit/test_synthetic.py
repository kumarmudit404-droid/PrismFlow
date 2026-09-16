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


def _class_centered(data):
    views, labels = data["views"].copy(), data["labels"]
    for stratum in np.unique(labels):
        rows = labels == stratum
        views[rows] -= views[rows].mean(axis=0, keepdims=True)
    return views


def test_rho_zero_gives_near_zero_correlation_only_after_conditioning():
    """At rho=0 the views share no noise, but they do all report the same
    label, so they remain UNCONDITIONALLY correlated. That is the task
    working, not redundancy -- and it is precisely why the dependence engine
    correlates class-conditional residuals rather than raw evidence.

    Before the signal/rho decoupling the label was scaled by sqrt(rho) and
    vanished here, so the two numbers below were indistinguishable.
    """
    cfg = SyntheticConfig(rho=0.0, **_CORR_TEST_KWARGS)
    data = generate_synthetic_dataset(cfg)

    unconditional = _mean_offdiag(empirical_cross_view_correlation(data["views"]))
    conditional = _mean_offdiag(empirical_cross_view_correlation(_class_centered(data)))

    assert conditional < 0.15, f"conditional correlation {conditional:.3f} should vanish at rho=0"
    assert unconditional > 0.2, (
        f"unconditional correlation {unconditional:.3f} should carry the shared class signal"
    )


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


# --- trained-model probe: the calibration target that actually matters ----
#
# The nearest-centroid probe above is model-free and cheap, but it is NOT
# sufficient to calibrate against. It scored 0.85 at a signal_strength where
# the real evidential model was saturated at 0.98 with mean uncertainty 0.014 --
# no headroom for discounting or calibration to move, and a ceiling that would
# have made the Part 06 clone experiment measure nothing. The generator is
# calibrated against this probe; nearest-centroid is kept as a fast sanity net.


def _trained_model_accuracy(signal_strength=None, rho=0.3, seeds=(0, 1, 2)):
    from prismflow.train import TrainConfig, train_one_seed

    overrides = {"rho": rho, "epochs": 40}
    if signal_strength is not None:
        overrides["signal_strength"] = signal_strength

    results = [train_one_seed(TrainConfig(**overrides), seed) for seed in seeds]
    accuracy = np.mean([r["test"]["accuracy"] for r in results])
    uncertainty = np.mean([r["test"]["mean_uncertainty"] for r in results])
    return float(accuracy), float(uncertainty)


def test_trained_model_accuracy_has_headroom():
    accuracy, _ = _trained_model_accuracy()
    assert 0.75 <= accuracy <= 0.90, (
        f"trained-model accuracy {accuracy:.3f} outside [0.75, 0.90] -- "
        "recalibrate SyntheticConfig.signal_strength"
    )


def test_trained_model_retains_uncertainty_headroom():
    """Vacuity pinned near zero would leave the discount nothing to act on and
    the Part 06/07 experiments no room to show an effect."""
    _, uncertainty = _trained_model_accuracy()
    assert 0.08 <= uncertainty <= 0.45, (
        f"mean uncertainty {uncertainty:.3f} outside [0.08, 0.45]"
    )


def test_train_config_does_not_shadow_generator_defaults():
    """Regression guard. TrainConfig used to retype the generator's data
    parameters, so recalibrating SyntheticConfig.signal_strength silently had
    no effect on any training run -- two full re-runs were produced against
    the superseded value before it was caught."""
    from prismflow.train import TrainConfig

    generator = SyntheticConfig()
    training = TrainConfig()

    for field in (
        "n_views",
        "n_classes",
        "n_samples",
        "d_latent",
        "d_view",
        "rho",
        "noise_std",
        "signal_strength",
    ):
        assert getattr(training, field) == getattr(generator, field), (
            f"TrainConfig.{field}={getattr(training, field)} has drifted from "
            f"SyntheticConfig.{field}={getattr(generator, field)}"
        )


# --- rho controls redundancy only, not task difficulty -------------------
#
# The class mean is added to the private component as well as the shared one.
# These tests pin down both halves of that change: the task stays learnable at
# rho=0, and the class-conditional residual structure -- which is what
# analytic_n_eff describes -- is untouched.


def _residual_correlation(rho: float, n_views: int, n_samples: int = 8000, seed: int = 0):
    """Empirical class-conditional cross-view correlation, and the n_eff it implies.

    Observation noise is switched off so the only thing between the latent
    residual and the measurement is the per-view projection A_v, which
    canonical correlation is invariant to. With noise_std > 0 the correlation
    would be attenuated by the noise variance and could not be compared
    against analytic_n_eff directly.
    """
    cfg = SyntheticConfig(
        n_views=n_views,
        n_classes=3,
        d_latent=4,
        d_view=6,
        n_samples=n_samples,
        rho=rho,
        noise_std=0.0,
        seed=seed,
    )
    data = generate_synthetic_dataset(cfg)
    views, labels = data["views"].copy(), data["labels"]

    for stratum in np.unique(labels):
        rows = labels == stratum
        views[rows] -= views[rows].mean(axis=0, keepdims=True)

    rho_hat = _mean_offdiag(empirical_cross_view_correlation(views))
    return rho_hat, n_views / (1.0 + (n_views - 1) * rho_hat)


@pytest.mark.parametrize("n_views", [4, 8])
@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_class_conditional_residual_correlation_still_equals_rho(rho, n_views):
    """The invariance the hotfix rests on: adding the class mean to both
    components shifts the conditional mean, not the conditional covariance."""
    rho_hat, _ = _residual_correlation(rho, n_views)
    assert rho_hat == pytest.approx(rho, abs=0.05), (
        f"residual correlation {rho_hat:.3f} != rho {rho} at V={n_views}"
    )


@pytest.mark.parametrize("n_views", [4, 8])
@pytest.mark.parametrize("rho", [0.0, 0.3, 0.6, 0.9])
def test_residual_structure_still_implies_analytic_n_eff(rho, n_views):
    """analytic_n_eff remains the correct ground truth after the hotfix.

    The tolerance is propagated from the 0.05 tolerance on rho rather than
    fixed, because n_eff is far more sensitive to rho at the independent end:
    |d n_eff / d rho| = V(V-1) / (1 + (V-1) rho)^2, which is 56 at V=8, rho=0
    but only ~1 at V=8, rho=0.9. A single absolute tolerance would be either
    vacuous at one end or impossible at the other.
    """
    _, n_eff_hat = _residual_correlation(rho, n_views)
    truth = analytic_n_eff(rho, n_views)

    sensitivity = n_views * (n_views - 1) / (1.0 + (n_views - 1) * rho) ** 2
    assert n_eff_hat == pytest.approx(truth, abs=0.05 * sensitivity), (
        f"n_eff {n_eff_hat:.3f} != analytic {truth:.3f} at rho={rho}, V={n_views}"
    )


def test_task_stays_learnable_at_rho_zero():
    """Regression guard on the confound Part 05 surfaced: the label used to
    live only in z_shared, so it was scaled by sqrt(rho) and disappeared at
    rho=0, leaving a classifier at chance (0.319 with 3 classes)."""
    independent = np.mean(
        [_nearest_centroid_accuracy(SyntheticConfig(rho=0.0, seed=s)) for s in range(3)]
    )
    correlated = np.mean(
        [_nearest_centroid_accuracy(SyntheticConfig(rho=0.5, seed=s)) for s in range(3)]
    )

    assert independent > 0.6, f"rho=0 accuracy collapsed to {independent:.3f}"
    assert correlated > 0.6, f"rho=0.5 accuracy collapsed to {correlated:.3f}"


def test_difficulty_is_comparable_across_the_rho_sweep():
    """rho should move redundancy, not difficulty. The class term scales as
    sqrt(rho) + sqrt(1-rho), so some variation survives -- it must stay
    bounded rather than collapsing at either end."""
    accuracies = [
        np.mean([_nearest_centroid_accuracy(SyntheticConfig(rho=r, seed=s)) for s in range(3)])
        for r in (0.0, 0.2, 0.4, 0.6, 0.8, 0.95)
    ]
    assert min(accuracies) > 0.6, f"a rho level collapsed: {accuracies}"
    assert max(accuracies) - min(accuracies) < 0.25, f"difficulty varies too much: {accuracies}"
