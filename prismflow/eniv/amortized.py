"""An amortized estimator psi: per-view evidence -> a per-sample [V, V] matrix.

The global estimator in `prismflow/statistics/dependence.py` needs a batch to
produce one number per pair, because a correlation is a statistic across
samples. That is exactly why it cannot answer "how dependent are these views ON
THIS INPUT" -- the question `prismflow/eniv/per_sample.py` exists to serve.

`AmortizedDependence` learns to answer it. It reads one sample's per-view
evidence and emits a symmetric [V, V] matrix with unit diagonal and entries in
[0, 1]. The batch statistic is replaced by a learned function, so a single
sample has an answer.

ARCHITECTURE. Deliberately small -- three layers, a shared per-view embedding
and a bilinear pair score. A large estimator here is a liability rather than an
asset: it is being fitted alongside an encoder that has an incentive to fool it
(see `prismflow/train_two_timescale.py`), and the more capacity it has the more
precisely it can be fooled.

OUTPUT CONSTRAINTS. Symmetry and a unit diagonal are imposed structurally, not
learned, so the output is always a valid input to the ENIV arithmetic. Entries
pass through a sigmoid, so they are in [0, 1]: negative dependence is not
representable, matching V1, where `soft_cluster_alpha` clips negatives to zero
anyway.

-------------------------------------------------------------------------------
NOTE FOR LATER WORK -- MUTUAL INFORMATION BOUNDS

If a future version estimates mutual information in order to PENALISE
dependence, it must use an UPPER bound on MI, such as CLUB. MINE and InfoNCE are
LOWER bounds. Driving a lower bound to zero says nothing whatever about the true
quantity: the bound can be loose, sit at zero, and the real mutual information
can be arbitrarily large. Minimising a lower bound is vacuous as a guarantee,
and it is a common error in published work. An upper bound driven to zero does
constrain the truth, because the truth sits below it.

The same asymmetry runs the other way for maximisation: to MAXIMISE MI (as in
representation learning), a lower bound is the correct choice, which is why
InfoNCE is sound there and unsound here. Match the bound's direction to the
direction of the optimisation.

Nothing in this Part estimates mutual information. This note is here so the
choice is deliberate when someone does.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AmortizedConfig:
    """`enabled` defaults to False: V1's global path stays the default."""

    enabled: bool = False
    hidden_dim: int = 32
    embed_dim: int = 16
    dropout: float = 0.0
    method: str = "eigen"

    def validate(self) -> None:
        if self.hidden_dim < 1 or self.embed_dim < 1:
            raise ValueError("hidden_dim and embed_dim must be >= 1")
        if self.method not in ("eigen", "design_effect"):
            raise ValueError(f"method must be 'eigen' or 'design_effect', got {self.method!r}")


class AmortizedDependence(nn.Module):
    """psi: evidence [B, V, K] -> dependence [B, V, V], symmetric, unit diagonal.

    Views are embedded by a shared MLP -- shared because "how much do these two
    overlap" should not depend on which slot a view occupies -- and each pair is
    scored from its two embeddings symmetrically.
    """

    def __init__(self, n_views: int, n_classes: int, config: AmortizedConfig | None = None):
        super().__init__()
        config = config or AmortizedConfig()
        config.validate()
        self.config = config
        self.n_views = n_views
        self.n_classes = n_classes

        self.embed = nn.Sequential(
            nn.Linear(n_classes, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity(),
            nn.Linear(config.hidden_dim, config.embed_dim),
        )
        # Scores a pair from [e_i + e_j, |e_i - e_j|, e_i * e_j]: every term is
        # symmetric in (i, j), so the matrix is symmetric by construction rather
        # than by averaging with its transpose afterwards.
        self.score = nn.Sequential(
            nn.Linear(3 * config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, evidence: torch.Tensor, view_mask: torch.Tensor | None = None) -> torch.Tensor:
        if evidence.dim() != 3:
            raise ValueError(f"evidence must be [B, V, K], got {tuple(evidence.shape)}")
        batch, n_views, _ = evidence.shape
        if n_views != self.n_views:
            raise ValueError(f"expected {self.n_views} views, got {n_views}")

        embedded = self.embed(evidence)  # [B, V, E]
        left = embedded.unsqueeze(2).expand(-1, -1, n_views, -1)
        right = embedded.unsqueeze(1).expand(-1, n_views, -1, -1)
        pair = torch.cat([left + right, (left - right).abs(), left * right], dim=-1)

        matrix = torch.sigmoid(self.score(pair).squeeze(-1))  # [B, V, V] in [0, 1]

        eye = torch.eye(n_views, dtype=torch.bool, device=evidence.device)
        matrix = matrix.masked_fill(eye, 1.0)

        if view_mask is not None:
            present = view_mask.bool()
            both = present.unsqueeze(2) & present.unsqueeze(1)
            matrix = torch.where(both | eye, matrix, torch.zeros_like(matrix))
            matrix = matrix.masked_fill(eye, 1.0)
        return matrix


def dependence_fit_loss(
    predicted: torch.Tensor, target, view_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Mean squared error against a reference dependence matrix, off-diagonal only.

    The reference is whatever the caller trusts as a target -- in
    `train_two_timescale` it is the global statistical estimator measured on the
    current batch. The diagonal is excluded because it is fixed at 1 structurally
    and would otherwise contribute a constant zero that dilutes the mean.
    """
    if not torch.is_tensor(target):
        target = torch.as_tensor(target, dtype=predicted.dtype, device=predicted.device)
    if target.dim() == 2:
        target = target.unsqueeze(0).expand_as(predicted)
    target = torch.nan_to_num(target, nan=0.0)

    n_views = predicted.shape[1]
    off = ~torch.eye(n_views, dtype=torch.bool, device=predicted.device)
    weight = off.unsqueeze(0).expand_as(predicted).to(predicted.dtype)
    if view_mask is not None:
        present = view_mask.bool()
        weight = weight * (present.unsqueeze(2) & present.unsqueeze(1)).to(predicted.dtype)

    total = weight.sum().clamp_min(1.0)
    return ((predicted - target) ** 2 * weight).sum() / total
