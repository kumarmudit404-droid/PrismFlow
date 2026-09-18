import math
import warnings

import pytest
import torch

from prismflow.statistics.hsic import (
    MIN_BATCH_SIZE,
    center_kernel,
    hsic,
    linear_hsic,
    median_bandwidth,
    normalised_hsic,
    rbf_kernel,
)
from prismflow.utils.seed import set_seed


def _independent(n=256, d=4, seed=0):
    set_seed(seed)
    return torch.randn(n, d), torch.randn(n, d)


# --- kernel mechanics ---------------------------------------------------


def test_centering_matches_the_explicit_H_K_H():
    set_seed(0)
    kernel = rbf_kernel(torch.randn(32, 3))
    n = kernel.shape[0]
    H = torch.eye(n) - torch.ones(n, n) / n
    assert center_kernel(kernel) == pytest.approx(H @ kernel @ H, abs=1e-5)


def test_centered_kernel_rows_sum_to_zero():
    set_seed(0)
    centered = center_kernel(rbf_kernel(torch.randn(64, 5)))
    assert centered.sum(dim=0) == pytest.approx(torch.zeros(64), abs=1e-4)
    assert centered.sum(dim=1) == pytest.approx(torch.zeros(64), abs=1e-4)


def test_rbf_kernel_is_symmetric_with_unit_diagonal():
    set_seed(0)
    kernel = rbf_kernel(torch.randn(24, 3))
    assert kernel == pytest.approx(kernel.T, abs=1e-6)
    assert torch.diagonal(kernel) == pytest.approx(torch.ones(24), abs=1e-6)


# --- bandwidth guards ---------------------------------------------------


def test_zero_variance_features_do_not_produce_nan_bandwidth():
    """A constant feature gives a zero median; without the guard the kernel
    divides by zero and NaN reaches the loss."""
    constant = torch.ones(80, 3)
    bandwidth = median_bandwidth(constant)
    assert torch.isfinite(bandwidth) and bandwidth > 0
    assert torch.isfinite(rbf_kernel(constant)).all()


def test_hsic_with_a_constant_input_is_finite_and_zero():
    set_seed(0)
    constant = torch.ones(128, 2)
    value = hsic(constant, torch.randn(128, 2))
    assert torch.isfinite(value)
    assert float(value) == pytest.approx(0.0, abs=1e-9)


def test_single_sample_batch_is_zero_not_nan():
    assert float(hsic(torch.randn(1, 3), torch.randn(1, 3), warn_small_batch=False)) == 0.0


# --- the estimator ------------------------------------------------------


def test_hsic_is_non_negative_and_near_zero_for_independent_variables():
    x, y = _independent()
    value = float(hsic(x, y))
    assert value >= 0.0
    assert value < 0.01


def test_hsic_detects_a_deterministic_nonlinear_relationship():
    """The reason HSIC is used instead of an orthogonality penalty: y = x^2 is
    uncorrelated with x on centred data but completely dependent on it."""
    set_seed(0)
    x = torch.randn(256, 1)
    y = x.pow(2)

    correlation = float((x.squeeze() * y.squeeze()).mean() - x.mean() * y.mean())
    assert abs(correlation) < 0.2, "x and x^2 should be near-uncorrelated"

    assert float(hsic(x, y)) > 10 * float(hsic(*_independent(n=256, d=1)))


def test_orthogonality_penalty_misses_what_hsic_catches():
    """The same pair, scored by the linear estimator: near zero. This is the
    failure mode the module docstring describes, pinned as a test."""
    set_seed(0)
    x = torch.randn(256, 1)
    y = x.pow(2)
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    linear = float(linear_hsic(x, y)) / float(linear_hsic(x, x))
    kernel = float(normalised_hsic(x, y))
    assert linear < 0.05
    assert kernel > 0.3


def test_hsic_rises_with_dependence_strength():
    set_seed(0)
    x = torch.randn(256, 2)
    noise = torch.randn(256, 2)
    values = [float(hsic(x, math.sqrt(1 - a) * noise + math.sqrt(a) * x)) for a in (0.0, 0.3, 0.9)]
    assert values[0] < values[1] < values[2]


def test_hsic_is_symmetric():
    x, y = _independent(n=128, d=3)
    assert float(hsic(x, y)) == pytest.approx(float(hsic(y, x)), abs=1e-9)


def test_hsic_of_a_variable_with_itself_is_positive():
    x, _ = _independent(n=128, d=3)
    assert float(hsic(x, x)) > 0.0


def test_normalised_hsic_is_one_for_identical_inputs():
    x, _ = _independent(n=128, d=3)
    assert float(normalised_hsic(x, x)) == pytest.approx(1.0, abs=1e-5)


def test_normalised_hsic_stays_in_the_unit_interval():
    x, y = _independent(n=128, d=3)
    assert 0.0 <= float(normalised_hsic(x, y)) <= 1.0


# --- batch size ---------------------------------------------------------


def test_small_batch_warns_but_still_returns():
    x, y = _independent(n=MIN_BATCH_SIZE - 1, d=2)
    with pytest.warns(RuntimeWarning, match="MIN|batch|noisy"):
        value = hsic(x, y)
    assert torch.isfinite(value)


def test_batch_at_the_threshold_does_not_warn():
    x, y = _independent(n=MIN_BATCH_SIZE, d=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        hsic(x, y)


# --- gradients ----------------------------------------------------------


def test_hsic_is_differentiable_and_reduces_under_descent():
    set_seed(0)
    x = torch.randn(128, 2)
    y = (x + 0.1 * torch.randn(128, 2)).clone().requires_grad_(True)

    before = float(hsic(x, y).detach())
    for _ in range(50):
        value = hsic(x, y)
        (gradient,) = torch.autograd.grad(value, y)
        with torch.no_grad():
            y -= 10.0 * gradient
    assert float(hsic(x, y).detach()) < before


def test_bandwidth_is_detached_so_it_cannot_be_optimised_against():
    """If the bandwidth carried gradient, a minimiser could shrink the penalty
    by moving the kernel instead of removing dependence."""
    x = torch.randn(80, 2, requires_grad=True)
    assert median_bandwidth(x).requires_grad is False


def test_invalid_kernel_is_rejected():
    x, y = _independent(n=64, d=2)
    with pytest.raises(ValueError):
        hsic(x, y, kernel="cosine")


def test_mismatched_batch_dimensions_are_rejected():
    with pytest.raises(ValueError):
        hsic(torch.randn(64, 2), torch.randn(32, 2))
