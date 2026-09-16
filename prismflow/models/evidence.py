"""Evidential head: encoder features -> non-negative evidence -> Dirichlet opinion.

Evidence is produced with softplus, never ReLU. A ReLU evidence unit that
lands on the flat side has exactly zero gradient and can never recover, which
silently pins that class to zero belief for the rest of training.

Shape conventions (docs/CONTRACT.md): per-view evidence is [B, V, K]; a single
opinion is belief [..., K] paired with uncertainty [...].

Opinion algebra, for evidence e over K classes:

    alpha = e + 1
    S     = sum_k alpha_k
    b_k   = e_k / S
    u     = K / S

so that sum_k b_k + u == 1 identically.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# S = K / u, so an opinion with u == 0 has infinite Dirichlet strength and
# cannot be converted back to alpha.
_MIN_UNCERTAINTY = 1e-6


def evidence_to_alpha(evidence: torch.Tensor) -> torch.Tensor:
    """evidence [..., K] -> Dirichlet concentration alpha [..., K]."""
    return evidence + 1.0


def alpha_to_opinion(alpha: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """alpha [..., K] -> (belief [..., K], uncertainty [...])."""
    n_classes = alpha.shape[-1]
    strength = alpha.sum(dim=-1)
    belief = (alpha - 1.0) / strength.unsqueeze(-1)
    uncertainty = n_classes / strength
    return belief, uncertainty


def evidence_to_opinion(evidence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """evidence [..., K] -> (belief [..., K], uncertainty [...])."""
    return alpha_to_opinion(evidence_to_alpha(evidence))


def opinion_to_alpha(belief: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    """Inverse of `alpha_to_opinion`: (belief, uncertainty) -> alpha [..., K]."""
    n_classes = belief.shape[-1]
    strength = n_classes / uncertainty.clamp_min(_MIN_UNCERTAINTY)
    return belief * strength.unsqueeze(-1) + 1.0


def opinion_to_probs(belief: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    """Expected class probability of the Dirichlet behind an opinion.

    p_k = alpha_k / S = b_k + u / K, so the uncertainty mass is spread
    uniformly over the classes.
    """
    n_classes = belief.shape[-1]
    return belief + uncertainty.unsqueeze(-1) / n_classes


def simplex_residual(belief: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
    """Largest absolute violation of `sum_k b_k + u == 1`."""
    return (belief.sum(dim=-1) + uncertainty - 1.0).abs().max()


def assert_valid_opinion(
    belief: torch.Tensor, uncertainty: torch.Tensor, tol: float = 1e-5
) -> None:
    residual = simplex_residual(belief, uncertainty)
    if residual > tol:
        raise AssertionError(
            f"opinion violates the simplex constraint: max |sum(b) + u - 1| = "
            f"{float(residual):.3e} > {tol:.1e}"
        )


# --- losses -------------------------------------------------------------


def edl_type2_ml_loss(alpha: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
    """Type II maximum likelihood EDL loss, per sample.

        L = sum_k y_k * (log S - log alpha_k)

    Returns shape [...] (the class dimension is reduced).
    """
    strength = alpha.sum(dim=-1, keepdim=True)
    return (targets_onehot * (strength.log() - alpha.log())).sum(dim=-1)


def kl_dirichlet_to_uniform(alpha: torch.Tensor) -> torch.Tensor:
    """KL( Dir(alpha) || Dir(1) ), per sample.

    Dir(1) is the uniform Dirichlet, i.e. the vacuous "I have learned nothing"
    prior. Returns shape [...].
    """
    n_classes = alpha.shape[-1]
    strength = alpha.sum(dim=-1, keepdim=True)

    log_norm = (
        torch.lgamma(strength.squeeze(-1))
        - torch.lgamma(alpha).sum(dim=-1)
        - torch.lgamma(torch.tensor(float(n_classes), dtype=alpha.dtype, device=alpha.device))
    )
    digamma_term = ((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(strength))).sum(dim=-1)
    return log_norm + digamma_term


def anneal_coefficient(epoch: int, anneal_epochs: int = 10) -> float:
    """KL annealing weight lambda_t = min(1, epoch / anneal_epochs).

    lambda_t is exactly 0.0 at epoch 0. Running the KL term at full strength
    from the first epoch collapses the model to maximum vacuity -- it emits
    "I know nothing" for every input and never recovers -- which is the most
    common way an EDL implementation fails silently.
    """
    if anneal_epochs <= 0:
        return 1.0
    return min(1.0, epoch / anneal_epochs)


def edl_loss(
    alpha: torch.Tensor,
    targets_onehot: torch.Tensor,
    epoch: int,
    anneal_epochs: int = 10,
    reduction: str = "mean",
) -> torch.Tensor:
    """Type II ML loss plus the annealed KL regulariser.

    The KL is applied to `alpha_tilde = y + (1 - y) * alpha`, which removes the
    true class from the penalty so the regulariser only drives *misleading*
    evidence toward vacuity, never correct evidence.
    """
    likelihood = edl_type2_ml_loss(alpha, targets_onehot)

    alpha_tilde = targets_onehot + (1.0 - targets_onehot) * alpha
    kl = kl_dirichlet_to_uniform(alpha_tilde)

    per_sample = likelihood + anneal_coefficient(epoch, anneal_epochs) * kl

    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    if reduction == "none":
        return per_sample
    raise ValueError(f"Unknown reduction {reduction!r}")


# --- heads --------------------------------------------------------------


class EvidenceHead(nn.Module):
    """Single-view evidence head: features [B, F] -> evidence [B, K]."""

    def __init__(self, feature_dim: int, n_classes: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.linear = nn.Linear(feature_dim, n_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.linear(features))


class MultiViewEvidenceHead(nn.Module):
    """One evidence head per view: features [B, V, F] -> evidence [B, V, K].

    Heads are per-view and unshared, for the same reason the encoders are
    (see `encoders.py`): a shared head would impose a common evidence geometry
    across views and manufacture the cross-view dependence this project exists
    to measure.

    Masked views (view_mask == False) emit exactly zero evidence, which is the
    vacuous opinion (b = 0, u = 1) -- the identity element of Dempster's rule,
    so an absent view cannot shift the fused result.
    """

    def __init__(self, n_views: int, feature_dim: int, n_classes: int):
        super().__init__()
        if n_views < 1:
            raise ValueError(f"n_views must be >= 1, got {n_views}")
        self.n_views = n_views
        self.feature_dim = feature_dim
        self.n_classes = n_classes
        self.heads = nn.ModuleList(
            [EvidenceHead(feature_dim, n_classes) for _ in range(n_views)]
        )

    def forward(self, features: torch.Tensor, view_mask: torch.Tensor) -> torch.Tensor:
        """
        features:   [B, V, feature_dim]
        view_mask:  [B, V] boolean, True = view present
        returns:    [B, V, K] non-negative evidence
        """
        batch_size = features.shape[0]
        if view_mask.shape != (batch_size, self.n_views):
            raise ValueError(
                f"view_mask shape {tuple(view_mask.shape)} != ({batch_size}, {self.n_views})"
            )

        per_view = [self.heads[v](features[:, v, :]) for v in range(self.n_views)]
        evidence = torch.stack(per_view, dim=1)
        return evidence * view_mask.unsqueeze(-1).to(evidence.dtype)
