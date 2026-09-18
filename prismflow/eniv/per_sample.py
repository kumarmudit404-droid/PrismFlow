"""Per-sample ENIV: n_eff(x) instead of n_eff.

WHY THIS EXISTS

A single dependence matrix estimated over a dataset is a preprocessing step, not
a model. It says how correlated two views are ON AVERAGE, and then applies that
average to every sample alike. But dependence is INPUT-CONDITIONAL:

  - Two cameras with disjoint viewpoints are independent in general, and become
    perfectly dependent on the one frame where both are washed out by the same
    glare.
  - A firewall sensor and a network sensor are independent in general, and
    become one sensor the moment both read from a tap the attacker controls.

In both cases the global matrix is right about the dataset and wrong about the
sample that matters. A discount driven by it will be correctly calibrated on
average and badly wrong exactly when the views collapse together. The object the
model actually needs is `n_eff(x)`.

This module is the arithmetic: given a per-sample dependence matrix [B, V, V] --
from `prismflow.eniv.amortized`, or from any other source -- it produces
per-sample effective view counts and per-sample discount factors. It does not
estimate dependence itself.

RELATION TO V1

V1's global path is untouched and remains the default. The functions here are
the per-sample analogues of `compute_eniv` and `soft_cluster_alpha`, and they
agree with them exactly when every sample is handed the same matrix -- which is
asserted in `tests/unit/test_per_sample.py`. That equivalence is what makes the
per-sample path a generalisation rather than a different mechanism.

NUMERICAL NOTE

`soft_cluster_alpha` is `1 / sum_j clip(R_ij, 0, 1)`, which vectorises across
the batch directly. The eigenvalue ENIV does not vectorise as cleanly, so it is
computed per sample with `numpy.linalg.eigvalsh` over a batched array, which
numpy does support batched. Both are clamped the same way V1 clamps them, so a
per-sample value can never leave the range a global value could.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

_MIN_CLUSTER = 1e-12


@dataclass
class PerSampleENIV:
    """effective_views [B], alpha [B, V], mean_dependence [B].

    `alpha` is what the discount consumes -- `evidence_discount` already accepts
    a [B, V] factor, so nothing in the fusion path needs to change to use it.
    """

    effective_views: np.ndarray
    alpha: np.ndarray
    mean_dependence: np.ndarray

    def alpha_tensor(self, dtype=None, device=None) -> torch.Tensor:
        return torch.as_tensor(self.alpha, dtype=dtype, device=device)


def _to_numpy(array) -> np.ndarray:
    if hasattr(array, "detach"):
        array = array.detach().cpu().numpy()
    return np.asarray(array, dtype=np.float64)


def _validate(dependence: np.ndarray) -> np.ndarray:
    if dependence.ndim != 3 or dependence.shape[1] != dependence.shape[2]:
        raise ValueError(f"dependence must be [B, V, V], got {dependence.shape}")
    return dependence


def _masked(dependence: np.ndarray, view_mask: np.ndarray | None) -> np.ndarray:
    """Zero the rows and columns of absent views, keeping a unit diagonal there.

    An absent view must not couple to anything (it contributes no evidence), but
    leaving its diagonal at 1 keeps the matrix positive semi-definite and keeps
    the eigenvalue path well behaved. Absent views are excluded from the counts
    afterwards.
    """
    if view_mask is None:
        return dependence
    mask = view_mask.astype(bool)
    pair = mask[:, :, None] & mask[:, None, :]
    out = np.where(pair, dependence, 0.0)
    diagonal = np.arange(dependence.shape[1])
    out[:, diagonal, diagonal] = 1.0
    return out


def per_sample_soft_cluster_alpha(dependence, view_mask=None) -> np.ndarray:
    """[B, V] alpha_i(x) = 1 / sum_j clip(R_ij(x), 0, 1).

    The per-sample analogue of `prismflow.eniv.eniv.soft_cluster_alpha`, and
    identical to it when every sample carries the same matrix.
    """
    dependence = _validate(_to_numpy(dependence))
    mask = None if view_mask is None else _to_numpy(view_mask)
    matrix = _masked(dependence, mask)

    cluster = np.clip(matrix, 0.0, 1.0).sum(axis=-1)
    alpha = 1.0 / np.maximum(cluster, _MIN_CLUSTER)
    alpha = np.clip(alpha, 0.0, 1.0)
    if mask is not None:
        # An absent view gets alpha 0: it has no evidence to scale, and leaving
        # it at 1 would quietly imply a fully trusted view that is not there.
        alpha = np.where(mask.astype(bool), alpha, 0.0)
    return alpha


def per_sample_effective_views(dependence, view_mask=None, method: str = "eigen") -> np.ndarray:
    """[B] effective number of independent views, clamped to [1, n_present].

    `eigen` mirrors V1's default: n_eff = (sum lambda)^2 / sum lambda^2 on the
    available sub-matrix. `design_effect` mirrors the alternative,
    n / (1 + (n - 1) rho_bar).
    """
    dependence = _validate(_to_numpy(dependence))
    mask = None if view_mask is None else _to_numpy(view_mask).astype(bool)
    n_views = dependence.shape[1]

    out = np.ones(dependence.shape[0], dtype=np.float64)
    for index in range(dependence.shape[0]):
        present = np.ones(n_views, dtype=bool) if mask is None else mask[index]
        count = int(present.sum())
        if count < 1:
            out[index] = 1.0
            continue
        block = dependence[index][np.ix_(present, present)]
        if count == 1:
            out[index] = 1.0
            continue

        if method == "eigen":
            # Exactly V1's `eigen_n_eff`: sum_i min(lambda_i, 1), NOT the
            # participation ratio. NaN pairs are imputed with the mean of the
            # measurable off-diagonals, as V1 does, because the
            # eigendecomposition cannot skip an entry and imputing 0 would
            # assert an independence that was never measured.
            block = np.array(block, dtype=np.float64)
            missing = np.isnan(block)
            if missing.any():
                off = block[~np.eye(count, dtype=bool)]
                finite = off[np.isfinite(off)]
                block[missing] = float(finite.mean()) if finite.size else 0.0
            np.fill_diagonal(block, 1.0)
            block = 0.5 * (block + block.T)
            eigenvalues = np.linalg.eigvalsh(block)
            value = float(np.clip(eigenvalues, 0.0, 1.0).sum())
        elif method == "design_effect":
            off = block[~np.eye(count, dtype=bool)]
            rho_bar = float(off.mean()) if off.size else 0.0
            denominator = 1.0 + (count - 1) * rho_bar
            value = count / denominator if denominator > _MIN_CLUSTER else float(count)
        else:
            raise ValueError(f"method must be 'eigen' or 'design_effect', got {method!r}")

        out[index] = float(np.clip(value, 1.0, count))
    return out


def per_sample_mean_dependence(dependence, view_mask=None) -> np.ndarray:
    """[B] mean off-diagonal dependence among present views; NaN if none."""
    dependence = _validate(_to_numpy(dependence))
    mask = None if view_mask is None else _to_numpy(view_mask).astype(bool)
    n_views = dependence.shape[1]

    out = np.full(dependence.shape[0], np.nan)
    for index in range(dependence.shape[0]):
        present = np.ones(n_views, dtype=bool) if mask is None else mask[index]
        if int(present.sum()) < 2:
            continue
        block = dependence[index][np.ix_(present, present)]
        off = block[~np.eye(block.shape[0], dtype=bool)]
        out[index] = float(off.mean()) if off.size else np.nan
    return out


def compute_per_sample_eniv(dependence, view_mask=None, method: str = "eigen") -> PerSampleENIV:
    """All three quantities in one pass."""
    return PerSampleENIV(
        effective_views=per_sample_effective_views(dependence, view_mask, method=method),
        alpha=per_sample_soft_cluster_alpha(dependence, view_mask),
        mean_dependence=per_sample_mean_dependence(dependence, view_mask),
    )


def broadcast_global(matrix, batch_size: int) -> np.ndarray:
    """Tile a global [V, V] matrix to [B, V, V].

    The bridge between the two paths: running the per-sample functions on this
    reproduces V1 exactly, which is how the equivalence tests are written and
    how a per-sample run can be reduced to the global one for comparison.
    """
    matrix = _to_numpy(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix must be square [V, V], got {matrix.shape}")
    return np.broadcast_to(matrix, (batch_size, *matrix.shape)).copy()
