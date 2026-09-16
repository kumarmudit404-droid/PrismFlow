"""PrismFlow model: encoders -> evidence -> Dempster fusion -> prediction.

With `use_discount=False` this is exactly the naive baseline the project is
measured against: every view is treated as an independent witness, so
agreement between two near-duplicate views inflates confidence just as much as
agreement between two genuinely independent ones.

With `use_discount=True` each view's evidence is scaled by its own redundancy
factor (`per_view_alpha`: 1 for an independent view, ~1/g for each member of a
g-view duplicate cluster) before fusion, so redundant agreement buys less
confidence. The estimator runs under no_grad and alpha enters as a constant --
see `prismflow/eniv/discount.py` for why that stop-gradient matters.

Dependence is measured on ENCODER FEATURES, not evidence. The evidence head is
a K-dim bottleneck trained to discard non-class variation, which is exactly
the shared structure that makes views redundant; measured after it, ENIV
failed validation at realistic accuracy (corr 0.864, non-monotone above
rho=0.6). The defaults below are the configuration that passed: canonical
correlation (per-view encoders share no feature basis), pairwise-holdout
conditioning, and a permutation null correction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from prismflow.eniv.discount import evidence_discount
from prismflow.eniv.eniv import ENIVResult, compute_eniv, per_view_alpha
from prismflow.models.encoders import EncoderConfig, MultiViewEncoder
from prismflow.models.evidence import (
    MultiViewEvidenceHead,
    assert_valid_opinion,
    evidence_to_opinion,
    opinion_to_alpha,
    opinion_to_probs,
)
from prismflow.models.fusion import fuse_opinions
from prismflow.statistics.dependence import available_views, feature_dependence_matrix

# Null permutations used inside the forward pass. The correction itself is
# required -- at batch size 64 without it, ENIV on independent views reads
# ~1.45 instead of 4.0 -- but its permutation count is not the bottleneck: at
# n=64 the spread comes from the observed statistic, and 2, 4, 8 and 12
# permutations gave indistinguishable means and spreads while 12 cost ~3x as
# much as 4. Kept low because the discount path runs every batch and inside
# attack loops; reported validation numbers use REPORTING_NULL_PERMUTATIONS.
FORWARD_NULL_PERMUTATIONS = 4
REPORTING_NULL_PERMUTATIONS = 12


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
    dependence_matrix     [V, V] or None when use_discount=False
    eniv                  ENIVResult or None when use_discount=False

    `per_view_belief` / `per_view_uncertainty` are the opinions actually fed
    to the fusion, i.e. post-discount when discounting is on. The undiscounted
    evidence remains in `per_view_evidence`.
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
    eniv: ENIVResult | None = None

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
        dependence_method: str = "cca",
        dependence_conditioning: str = "pairwise_holdout",
        null_permutations: int = FORWARD_NULL_PERMUTATIONS,
        dependence_seed: int = 0,
    ):
        super().__init__()
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
        self.dependence_method = dependence_method
        self.dependence_conditioning = dependence_conditioning
        self.null_permutations = null_permutations
        self.dependence_seed = dependence_seed

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
        if self.validate_opinions:
            assert_valid_opinion(per_view_belief, per_view_uncertainty)

        dependence = None
        eniv = None
        if self.use_discount:
            # The estimator is measurement, not computation: it runs outside
            # the graph entirely so nothing in training can be optimised
            # against it. See prismflow/eniv/discount.py.
            with torch.no_grad():
                matrix = feature_dependence_matrix(
                    features,
                    evidence,
                    view_mask,
                    method=self.dependence_method,
                    conditioning=self.dependence_conditioning,
                    seed=self.dependence_seed,
                    null_permutations=self.null_permutations,
                )
                present = available_views(view_mask, n_views=self.n_views)
                alpha = per_view_alpha(matrix, present)
                eniv = replace(
                    compute_eniv(matrix, present),
                    per_view_alpha=tuple(float(a) for a in alpha),
                )
                dependence = torch.as_tensor(
                    matrix, dtype=evidence.dtype, device=evidence.device
                )

            # One factor per view, applied to evidence: a duplicated view must
            # not discount views that were never duplicated. efficiency_ratio
            # is reported but no longer applied.
            per_view_belief, per_view_uncertainty = evidence_to_opinion(
                evidence_discount(evidence, alpha)
            )
            if self.validate_opinions:
                assert_valid_opinion(per_view_belief, per_view_uncertainty)

        fused_belief, fused_uncertainty = fuse_opinions(
            per_view_belief, per_view_uncertainty, view_mask
        )
        if self.validate_opinions:
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
            dependence_matrix=dependence,
            eniv=eniv,
        )
