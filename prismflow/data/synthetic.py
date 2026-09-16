"""Synthetic multi-view generator with known ground-truth dependence.

This generator is the calibration standard for PrismFlow: because we set the
true cross-view dependence (rho) ourselves, later parts can check whether the
ENIV estimator recovers a quantity we already know analytically, instead of
producing an unfalsifiable number.

Generative model, per sample with class label y drawn uniformly at random:

    mu_y       = signal_strength * class_direction[y]   (fixed unit direction per class)
    z_shared   = mu_y + eps_shared,  eps_shared ~ N(0, I)   (shared across views)
    z_v        = mu_y + eps_v,       eps_v      ~ N(0, I)   (private, independent per view)
    combined_v = sqrt(rho_v) * z_shared + sqrt(1 - rho_v) * z_v
    x_v        = A_v @ combined_v + noise_v

`A_v` is a fixed random projection unique to view v (fixed for a given
config/seed, not redrawn per sample).

WHY BOTH COMPONENTS CARRY THE CLASS MEAN
----------------------------------------
The class mean is added to the private component as well as the shared one so
that `rho` controls redundancy ONLY, not task difficulty.

Previously the label lived exclusively in `z_shared`, so it entered each view
multiplied by sqrt(rho_v) and vanished entirely at rho = 0 -- the views became
pure noise and a classifier sat at chance. That made `rho` a single knob
driving both "how dependent are the views" and "is the task learnable at all",
and it left the rho = 0 anchor of the ENIV validation sweep degenerate: the
estimator was being asked to measure dependence in evidence that carried no
signal.

With the mean in both components the class term becomes
(sqrt(rho) + sqrt(1 - rho)) * mu_y, which is 1.0 at rho in {0, 1} and peaks at
sqrt(2) around rho = 0.5. Signal amplitude therefore varies by at most ~41%
across the sweep instead of collapsing to zero.

CRITICALLY, THIS LEAVES analytic_n_eff UNCHANGED. Conditional on y the class
term is a constant, so the residual is

    combined_v - E[combined_v | y] = sqrt(rho) * eps_shared + sqrt(1 - rho) * eps_v

which is exactly the residual structure the previous generator had. Both
components have unit variance, so Corr(combined_i, combined_j | y) = rho
exactly, and the design effect n / (1 + (n-1) * rho) remains the correct
ground truth. `test_synthetic.py` asserts this empirically.

`rho` may be:
  - a scalar in [0, 1], applied uniformly to every view (rho_v = rho), or
  - a symmetric [V, V] matrix with unit diagonal giving pairwise target
    correlations; each view's shared-loading rho_v is then the mean of its
    off-diagonal row. This is a single-factor approximation: an exact
    arbitrary pairwise correlation structure requires a multi-factor model,
    which is out of scope for V1. When rho is scalar this reduces exactly to
    rho_v = rho for every view.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

RhoSpec = Union[float, "np.ndarray"]


@dataclass
class SyntheticConfig:
    n_views: int = 4
    n_classes: int = 3
    n_samples: int = 2000
    d_latent: int = 8
    d_view: int = 16
    rho: RhoSpec = 0.3
    noise_std: Union[float, list] = 0.5
    # Calibrated against the TRAINED evidential model, not the nearest-centroid
    # probe: at rho=0.3 this gives test accuracy ~0.83 and mean uncertainty
    # ~0.22, which leaves headroom for the discount and the calibration work to
    # move. Calibrating on nearest-centroid alone is misleading -- it scored
    # 0.85 at a setting where the real model was saturated at 0.98.
    signal_strength: float = 1.15
    seed: int = 0


def analytic_n_eff(rho: float, n_views: int) -> float:
    """Ground-truth effective number of independent views for equicorrelated
    views sharing pairwise correlation `rho`.

    This is the value the ENIV estimator (Part 05) must recover:
        n_eff = n / (1 + (n - 1) * rho)
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError(f"rho must be in [0, 1], got {rho}")
    if n_views < 1:
        raise ValueError(f"n_views must be >= 1, got {n_views}")
    return n_views / (1.0 + (n_views - 1) * rho)


def _resolve_rho_matrix(rho: RhoSpec, n_views: int) -> np.ndarray:
    if np.isscalar(rho):
        rho = float(rho)
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        matrix = np.full((n_views, n_views), rho, dtype=np.float64)
        np.fill_diagonal(matrix, 1.0)
        return matrix

    matrix = np.asarray(rho, dtype=np.float64)
    if matrix.shape != (n_views, n_views):
        raise ValueError(f"rho matrix must have shape ({n_views}, {n_views}), got {matrix.shape}")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("rho matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 1.0):
        raise ValueError("rho matrix diagonal must be 1.0")
    if matrix.min() < 0.0 or matrix.max() > 1.0:
        raise ValueError("rho matrix entries must lie in [0, 1]")
    return matrix


def _per_view_shared_loading(rho_matrix: np.ndarray) -> np.ndarray:
    n_views = rho_matrix.shape[0]
    if n_views == 1:
        return np.array([0.0])
    off_diag_sum = rho_matrix.sum(axis=1) - 1.0
    row_mean = off_diag_sum / (n_views - 1)
    return np.clip(row_mean, 0.0, 1.0)


def generate_synthetic_dataset(config: SyntheticConfig) -> dict:
    """Generate a synthetic multi-view classification dataset.

    Returns a dict with keys:
        views      [N, V, D] float64
        labels     [N] int64
        rho_matrix [V, V] the resolved target correlation matrix
        config     the SyntheticConfig used
    """
    rng = np.random.default_rng(config.seed)

    n_views, n_classes, n_samples = config.n_views, config.n_classes, config.n_samples
    d_latent, d_view = config.d_latent, config.d_view

    rho_matrix = _resolve_rho_matrix(config.rho, n_views)
    rho_v = _per_view_shared_loading(rho_matrix)

    noise_std = config.noise_std
    if np.isscalar(noise_std):
        noise_std = [float(noise_std)] * n_views
    else:
        noise_std = list(noise_std)
        if len(noise_std) != n_views:
            raise ValueError(f"noise_std must have length {n_views}, got {len(noise_std)}")

    class_directions = rng.standard_normal((n_classes, d_latent))
    class_directions /= np.linalg.norm(class_directions, axis=1, keepdims=True)

    # Fixed per-view random projections, unique per view, fixed for this config/seed.
    projections = rng.standard_normal((n_views, d_view, d_latent)) / np.sqrt(d_latent)

    labels = rng.integers(0, n_classes, size=n_samples)

    # The class mean goes into BOTH the shared and the private component, so
    # rho controls redundancy alone -- see the module docstring.
    class_mean = config.signal_strength * class_directions[labels]

    eps_shared = rng.standard_normal((n_samples, d_latent))
    z_shared = class_mean + eps_shared

    views = np.zeros((n_samples, n_views, d_view), dtype=np.float64)
    for v in range(n_views):
        eps_v = rng.standard_normal((n_samples, d_latent))
        z_v = class_mean + eps_v
        combined_v = np.sqrt(rho_v[v]) * z_shared + np.sqrt(1.0 - rho_v[v]) * z_v
        noise_v = rng.standard_normal((n_samples, d_view)) * noise_std[v]
        views[:, v, :] = combined_v @ projections[v].T + noise_v

    return {
        "views": views,
        "labels": labels,
        "rho_matrix": rho_matrix,
        "config": config,
    }


def _standardize(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1e-8, std)
    return x / std


def _sym_inv_sqrt(matrix: np.ndarray, reg: float) -> np.ndarray:
    matrix = matrix + reg * np.eye(matrix.shape[0])
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-10, None)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


def _first_canonical_correlation(x: np.ndarray, y: np.ndarray, reg: float = 1e-3) -> float:
    """First canonical correlation coefficient between two feature matrices.

    Used (rather than raw per-dimension Pearson correlation) because each
    view is seen through its own random projection A_v: canonical
    correlation recovers shared structure regardless of the projection
    basis, whereas axis-aligned correlation would not.
    """
    n = x.shape[0]
    x_std = _standardize(x)
    y_std = _standardize(y)

    sxx = x_std.T @ x_std / n
    syy = y_std.T @ y_std / n
    sxy = x_std.T @ y_std / n

    sxx_inv_sqrt = _sym_inv_sqrt(sxx, reg)
    syy_inv_sqrt = _sym_inv_sqrt(syy, reg)

    m = sxx_inv_sqrt @ sxy @ syy_inv_sqrt
    singular_values = np.linalg.svd(m, compute_uv=False)
    return float(np.clip(singular_values[0], 0.0, 1.0)) if len(singular_values) else 0.0


def empirical_cross_view_correlation(views, reg: float = 1e-3) -> np.ndarray:
    """Empirical [V, V] cross-view correlation matrix.

    `views` is [N, V, D] (numpy array, or a torch.Tensor which is converted).
    Diagonal entries are 1.0; off-diagonal entries are the first canonical
    correlation between the corresponding pair of views.
    """
    if hasattr(views, "detach"):
        views = views.detach().cpu().numpy()
    views = np.asarray(views, dtype=np.float64)

    n_views = views.shape[1]
    corr = np.eye(n_views)
    for i in range(n_views):
        for j in range(i + 1, n_views):
            value = _first_canonical_correlation(views[:, i, :], views[:, j, :], reg=reg)
            corr[i, j] = value
            corr[j, i] = value
    return corr
