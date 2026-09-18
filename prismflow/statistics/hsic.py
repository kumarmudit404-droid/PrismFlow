"""HSIC: the Hilbert-Schmidt Independence Criterion, for V2 disentanglement.

WHY NOT AN ORTHOGONALITY PENALTY

The obvious way to push two representations apart is `|| h_s^T h_p ||_F^2`,
which drives their cross-covariance to zero. That enforces only LINEAR
independence. Two representations can be exactly decorrelated and still be
deterministic functions of one another -- `h_p = h_s^2` on centred data has zero
covariance and total dependence. A private branch could therefore carry a copy
of the shared information, pass the penalty, and defeat the point of the split.

HSIC is the Hilbert-Schmidt norm of the cross-covariance operator in a
reproducing kernel Hilbert space. For CHARACTERISTIC kernels (the RBF kernel is
one) it is zero if and only if the variables are statistically independent, so
the nonlinear escape route above is closed. It is differentiable and costs
O(B^2) per batch.

`linear_hsic` (equivalently, the orthogonality penalty) is implemented alongside
it for comparison, as the brief requires. HSIC is the default.

ESTIMATOR

The biased V-statistic:

    HSIC_b(X, Y) = 1/n^2 * trace(K_c L_c),   K_c = H K H,  H = I - (1/n) 11^T

Biased rather than unbiased because it is non-negative by construction, which
matters when it is used as a LOSS: an unbiased estimate can go negative on a
finite batch, and a minimiser would happily chase that noise. The bias is O(1/n)
and vanishes with batch size.

BANDWIDTH

Median heuristic: sigma^2 = median of pairwise squared distances. Computed under
`no_grad` and treated as a constant, so the optimiser cannot make the penalty
look small by shrinking the kernel bandwidth instead of removing dependence --
the same reasoning that keeps alpha out of the graph in
`prismflow/eniv/discount.py`. A degenerate batch (all points identical, or a
zero-variance feature) gives a zero or NaN median; that is caught and replaced
by a positive fallback rather than propagated.

BATCH SIZE

The estimator is noisy for small n: the bias is O(1/n) and the variance of the
V-statistic is worse. `MIN_BATCH_SIZE` is 64; below it `hsic` warns once per
call site. It does not raise, because the final batch of an epoch is routinely
short and failing there would be worse than a slightly noisy penalty on one
batch.
"""

from __future__ import annotations

import warnings

import torch

MIN_BATCH_SIZE = 64
_MIN_BANDWIDTH = 1e-6
_FALLBACK_BANDWIDTH = 1.0
KERNELS = ("rbf", "linear")


def _as_matrix(x: torch.Tensor) -> torch.Tensor:
    """[n, d]. A [n] vector becomes [n, 1]; anything higher is flattened."""
    if x.dim() == 1:
        return x.unsqueeze(-1)
    if x.dim() == 2:
        return x
    return x.reshape(x.shape[0], -1)


def squared_distances(x: torch.Tensor) -> torch.Tensor:
    """Pairwise squared Euclidean distances [n, n], clamped non-negative."""
    x = _as_matrix(x)
    return torch.cdist(x, x, p=2.0).pow(2).clamp_min(0.0)


def median_bandwidth(x: torch.Tensor) -> torch.Tensor:
    """Median-heuristic sigma^2, as a detached scalar tensor.

    Returns `_FALLBACK_BANDWIDTH` when the median is zero or non-finite, which
    happens for a zero-variance feature or a batch of identical points. Without
    the guard the RBF kernel would divide by zero and emit NaN into the loss.
    """
    with torch.no_grad():
        distances = squared_distances(x)
        n = distances.shape[0]
        if n < 2:
            return torch.tensor(_FALLBACK_BANDWIDTH, dtype=x.dtype, device=x.device)
        # Off-diagonal entries only: the diagonal is exactly zero and would drag
        # the median toward zero for small batches.
        off_diagonal = distances[~torch.eye(n, dtype=torch.bool, device=x.device)]
        median = off_diagonal.median()
        if not torch.isfinite(median) or median <= _MIN_BANDWIDTH:
            return torch.tensor(_FALLBACK_BANDWIDTH, dtype=x.dtype, device=x.device)
        return median.detach()


def rbf_kernel(x: torch.Tensor, bandwidth=None) -> torch.Tensor:
    """RBF Gram matrix exp(-d^2 / (2 sigma^2)); bandwidth is sigma^2."""
    if bandwidth is None:
        bandwidth = median_bandwidth(x)
    bandwidth = torch.as_tensor(bandwidth, dtype=x.dtype, device=x.device).clamp_min(_MIN_BANDWIDTH)
    return torch.exp(-squared_distances(x) / (2.0 * bandwidth))


def linear_kernel(x: torch.Tensor) -> torch.Tensor:
    x = _as_matrix(x)
    return x @ x.T


def center_kernel(kernel: torch.Tensor) -> torch.Tensor:
    """H K H with H = I - (1/n) 11^T, without materialising H."""
    n = kernel.shape[0]
    row = kernel.mean(dim=0, keepdim=True)
    column = kernel.mean(dim=1, keepdim=True)
    total = kernel.mean()
    return kernel - row - column + total


def _kernel_of(x: torch.Tensor, kernel: str, bandwidth) -> torch.Tensor:
    if kernel == "rbf":
        return rbf_kernel(x, bandwidth)
    if kernel == "linear":
        return linear_kernel(x)
    raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")


def hsic(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: str = "rbf",
    bandwidth_x=None,
    bandwidth_y=None,
    warn_small_batch: bool = True,
) -> torch.Tensor:
    """Biased HSIC between [n, dx] and [n, dy]. Non-negative scalar tensor."""
    x, y = _as_matrix(x), _as_matrix(y)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must share a batch dimension, got {x.shape[0]} and {y.shape[0]}")
    n = x.shape[0]
    if n < 2:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    if warn_small_batch and n < MIN_BATCH_SIZE:
        warnings.warn(
            f"HSIC estimated on batch of {n} < {MIN_BATCH_SIZE}; the estimate is "
            "biased upward and noisy at this size",
            RuntimeWarning,
            stacklevel=2,
        )

    kx = center_kernel(_kernel_of(x, kernel, bandwidth_x))
    ky = center_kernel(_kernel_of(y, kernel, bandwidth_y))
    # trace(Kc @ Lc) without forming the product: both are symmetric.
    return (kx * ky).sum() / (n * n)


def linear_hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """HSIC with linear kernels. Equal to the squared Frobenius norm of the
    cross-covariance up to 1/n^2, i.e. the orthogonality penalty, and detects
    LINEAR dependence only. Provided for comparison; see the module docstring."""
    return hsic(x, y, kernel="linear", warn_small_batch=False)


def normalised_hsic(
    x: torch.Tensor, y: torch.Tensor, kernel: str = "rbf", eps: float = 1e-12
) -> torch.Tensor:
    """HSIC / sqrt(HSIC(x,x) HSIC(y,y)), in [0, 1]: a centred kernel alignment.

    Raw HSIC is not comparable across runs because it scales with the kernel
    and the representation's own variability. This normalisation is what the
    experiment REPORTS; the raw value is what it MINIMISES (normalising inside
    the loss would let the model shrink the denominator instead).
    """
    xy = hsic(x, y, kernel=kernel, warn_small_batch=False)
    xx = hsic(x, x, kernel=kernel, warn_small_batch=False)
    yy = hsic(y, y, kernel=kernel, warn_small_batch=False)
    return xy / (xx.sqrt() * yy.sqrt() + eps)
