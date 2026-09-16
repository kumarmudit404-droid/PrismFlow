"""PrismFlow model: encoders -> evidence -> Dempster fusion -> prediction.

With `use_discount=False` (the Part 04 default) this is exactly the naive
baseline the project is measured against: every view is treated as an
independent witness, so agreement between two near-duplicate views inflates
confidence just as much as agreement between two genuinely independent ones.

`use_discount=True` is not implemented in this Part. `PrismFlowOutput` already
carries `dependence_matrix` and `eniv` as None so Part 05 can populate them
without changing this signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from prismflow.models.encoders import EncoderConfig, MultiViewEncoder
from prismflow.models.evidence import (
    MultiViewEvidenceHead,
    assert_valid_opinion,
    evidence_to_opinion,
    opinion_to_alpha,
    opinion_to_probs,
)
from prismflow.models.fusion import fuse_opinions


@dataclass
class PrismFlowOutput:
    """One forward pass.

    prediction            [B]       argmax class
    probs                 [B, K]    expected Dirichlet probability
    confidence            [B]       1 - uncertainty
    uncertainty           [B]       fused vacuity
    per_view_evidence     [B, V, K]
    per_view_belief       [B, V, K]
    per_view_uncertainty  [B, V]
    view_mask             [B, V]
    dependence_matrix     [V, V] or None -- populated in Part 05
    eniv                  scalar/[B] or None -- populated in Part 05
    """

    prediction: torch.Tensor
    probs: torch.Tensor
    confidence: torch.Tensor
    uncertainty: torch.Tensor
    per_view_evidence: torch.Tensor
    per_view_belief: torch.Tensor
    per_view_uncertainty: torch.Tensor
    view_mask: torch.Tensor
    dependence_matrix: torch.Tensor | None = None
    eniv: torch.Tensor | None = None

    @property
    def fused_alpha(self) -> torch.Tensor:
        """Dirichlet concentration of the fused opinion, for the EDL loss."""
        belief = self.probs - self.uncertainty.unsqueeze(-1) / self.probs.shape[-1]
        return opinion_to_alpha(belief, self.uncertainty)


class PrismFlow(nn.Module):
    def __init__(
        self,
        view_configs: list[EncoderConfig],
        n_classes: int,
        use_discount: bool = False,
        share_weights: bool = False,
        validate_opinions: bool = True,
    ):
        super().__init__()
        if use_discount:
            raise NotImplementedError(
                "use_discount=True requires the dependence/ENIV estimators, which "
                "arrive in Part 05. Part 04 implements the naive baseline only."
            )

        self.encoder = MultiViewEncoder(view_configs, share_weights=share_weights)
        self.evidence_head = MultiViewEvidenceHead(
            n_views=self.encoder.n_views,
            feature_dim=self.encoder.feature_dim,
            n_classes=n_classes,
        )
        self.n_views = self.encoder.n_views
        self.n_classes = n_classes
        self.use_discount = use_discount
        self.validate_opinions = validate_opinions

    def forward(
        self, views: torch.Tensor, view_mask: torch.Tensor | None = None
    ) -> PrismFlowOutput:
        """
        views:     [B, V, D]
        view_mask: [B, V] boolean, True = view present (None means all present)
        """
        batch_size = views.shape[0]
        if view_mask is None:
            view_mask = torch.ones(
                batch_size, self.n_views, dtype=torch.bool, device=views.device
            )

        features = self.encoder(views, view_mask)
        evidence = self.evidence_head(features, view_mask)

        per_view_belief, per_view_uncertainty = evidence_to_opinion(evidence)
        fused_belief, fused_uncertainty = fuse_opinions(
            per_view_belief, per_view_uncertainty, view_mask
        )

        if self.validate_opinions:
            assert_valid_opinion(per_view_belief, per_view_uncertainty)
            assert_valid_opinion(fused_belief, fused_uncertainty)

        probs = opinion_to_probs(fused_belief, fused_uncertainty)

        return PrismFlowOutput(
            prediction=probs.argmax(dim=-1),
            probs=probs,
            confidence=1.0 - fused_uncertainty,
            uncertainty=fused_uncertainty,
            per_view_evidence=evidence,
            per_view_belief=per_view_belief,
            per_view_uncertainty=per_view_uncertainty,
            view_mask=view_mask,
            dependence_matrix=None,
            eniv=None,
        )
