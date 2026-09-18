"""The disentanglement term added to the V2 objective with weight lambda_2.

    total = task_loss + lambda_2 * disentanglement(shared, private)

The penalty is the mean over views of the dependence between that view's shared
and private branches. Per view, not pooled across views: the claim being
enforced is "view v's private part carries nothing that view v's shared part
carries", and pooling views would let a view hide dependence behind another
view's independence.

Shared branches are deliberately NOT pushed apart from each other. Views sharing
information is the premise of the whole project -- it is what ENIV measures and
what the discount acts on. Only the shared/private split within a view is
penalised.

`method` selects the estimator:

    hsic           kernel HSIC, zero iff independent (characteristic kernel)
    orthogonality  linear HSIC, i.e. the cross-covariance penalty; LINEAR
                   dependence only, kept for comparison

See `prismflow/statistics/hsic.py` for why the default is HSIC.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from prismflow.statistics.hsic import hsic, linear_hsic, normalised_hsic

METHODS = ("hsic", "orthogonality")


@dataclass
class DisentanglementReport:
    """loss is what the optimiser sees; `normalised` is what to report.

    per_view / per_view_normalised are [V] lists, so a single view failing to
    disentangle is visible rather than averaged away.
    """

    loss: torch.Tensor
    normalised: float
    per_view: list[float]
    per_view_normalised: list[float]


def _pair_statistic(shared: torch.Tensor, private: torch.Tensor, method: str) -> torch.Tensor:
    if method == "hsic":
        return hsic(shared, private, kernel="rbf")
    if method == "orthogonality":
        return linear_hsic(shared, private)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def disentanglement_loss(
    shared: torch.Tensor,
    private: torch.Tensor,
    view_mask: torch.Tensor | None = None,
    method: str = "hsic",
    min_samples: int = 2,
) -> DisentanglementReport:
    """Mean per-view dependence between the shared and private branches.

    shared:   [B, V, Ds]
    private:  [B, V, Dp]
    view_mask [B, V] -- a view contributes only its present samples, and is
              skipped entirely when too few of them remain to estimate anything.
    """
    if shared.shape[:2] != private.shape[:2]:
        raise ValueError(
            f"shared and private must agree on [B, V], got {shared.shape[:2]} and {private.shape[:2]}"
        )
    n_views = shared.shape[1]

    terms, per_view, per_view_normalised = [], [], []
    for view in range(n_views):
        s, p = shared[:, view, :], private[:, view, :]
        if view_mask is not None:
            present = view_mask[:, view].bool()
            s, p = s[present], p[present]
        if s.shape[0] < min_samples:
            per_view.append(float("nan"))
            per_view_normalised.append(float("nan"))
            continue

        value = _pair_statistic(s, p, method)
        terms.append(value)
        per_view.append(float(value.detach()))
        with torch.no_grad():
            kernel = "linear" if method == "orthogonality" else "rbf"
            per_view_normalised.append(float(normalised_hsic(s, p, kernel=kernel)))

    if not terms:
        zero = torch.zeros((), dtype=shared.dtype, device=shared.device)
        return DisentanglementReport(zero, float("nan"), per_view, per_view_normalised)

    loss = torch.stack(terms).mean()
    finite = [v for v in per_view_normalised if v == v]
    return DisentanglementReport(
        loss=loss,
        normalised=float(sum(finite) / len(finite)) if finite else float("nan"),
        per_view=per_view,
        per_view_normalised=per_view_normalised,
    )


def total_loss(task: torch.Tensor, report: DisentanglementReport, lambda_2: float) -> torch.Tensor:
    """task + lambda_2 * disentanglement. lambda_2 = 0 leaves the V1 objective
    untouched, including its gradient, which is what the default OFF path uses."""
    if lambda_2 == 0.0:
        return task
    return task + lambda_2 * report.loss
