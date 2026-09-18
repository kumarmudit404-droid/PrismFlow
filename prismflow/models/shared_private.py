"""V2: per-view shared / private branches, behind a flag that defaults to OFF.

    encoder features -> [ shared_v , private_v ] -> evidence -> dependence
                     -> ENIV -> discount -> Dempster fusion

V1 IS NOT TOUCHED. `prismflow/models/prismflow.py` is unchanged and remains the
fallback: `build_shared_private` returns a plain `PrismFlow` whenever
`SharedPrivateConfig.enabled` is False, which is the default. Nothing in V1
imports this module.

WHAT THE SPLIT IS FOR

The project's premise is that views share information and that the shared part
must not be double-counted. A view's evidence today is computed from its whole
feature vector, which mixes the part that genuinely overlaps other views with
the part that is idiosyncratic to it. The dependence estimator then has to read
overlap through that mixture. Splitting the two explicitly, and forcing them to
be statistically independent (`disentanglement_losses`), is meant to give the
estimator a cleaner signal to measure.

WHERE EVIDENCE AND DEPENDENCE COME FROM

Two decisions that determine what the experiment can conclude:

  evidence_from   "both" (default) concatenates shared and private, so the
                  classifier keeps all the information V1 had and any accuracy
                  difference is attributable to the penalty rather than to
                  throwing half the features away. "shared" is available for
                  ablation.
  dependence_on   "shared" (default) measures inter-view dependence on the
                  shared branches only -- the quantity the split exists to
                  isolate. "features" reproduces V1's behaviour on the full
                  vector.

The default pair is the honest test of the idea: same information available to
the classifier, dependence read off the isolated shared part. It also makes the
README's question answerable, because ENIV is then measured on a representation
the penalty directly shapes. Note the expected direction: if the private branch
successfully absorbs the idiosyncratic part, what remains in `shared` is more
nearly common across views, measured dependence RISES and ENIV FALLS. A lower
ENIV here would mean the split worked, not that the model got worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
from torch import nn

from prismflow.eniv.discount import evidence_discount
from prismflow.eniv.eniv import compute_eniv, soft_cluster_alpha
from prismflow.models.disentanglement_losses import (
    METHODS,
    DisentanglementReport,
    disentanglement_loss,
)
from prismflow.models.encoders import MultiViewEncoder
from prismflow.models.evidence import (
    MultiViewEvidenceHead,
    assert_valid_opinion,
    evidence_to_opinion,
    opinion_to_probs,
)
from prismflow.models.fusion import fuse_opinions
from prismflow.models.prismflow import FORWARD_NULL_PERMUTATIONS, PrismFlow, PrismFlowOutput
from prismflow.statistics.dependence import available_views, feature_dependence_matrix

EVIDENCE_SOURCES = ("both", "shared")
DEPENDENCE_SOURCES = ("shared", "features")


@dataclass
class SharedPrivateConfig:
    """V2 configuration. `enabled` defaults to False: V1 is the default path."""

    enabled: bool = False
    shared_dim: int = 16
    private_dim: int = 16
    kernel: str = "rbf"
    method: str = "hsic"
    lambda_2: float = 0.1
    evidence_from: str = "both"
    dependence_on: str = "shared"

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if self.evidence_from not in EVIDENCE_SOURCES:
            raise ValueError(f"evidence_from must be one of {EVIDENCE_SOURCES}")
        if self.dependence_on not in DEPENDENCE_SOURCES:
            raise ValueError(f"dependence_on must be one of {DEPENDENCE_SOURCES}")
        if self.shared_dim < 1 or self.private_dim < 1:
            raise ValueError("shared_dim and private_dim must be >= 1")
        if self.lambda_2 < 0:
            raise ValueError(f"lambda_2 must be >= 0, got {self.lambda_2}")


@dataclass
class SharedPrivateOutput(PrismFlowOutput):
    """PrismFlowOutput plus the branches and the measured disentanglement."""

    shared: torch.Tensor | None = None
    private: torch.Tensor | None = None
    disentanglement: DisentanglementReport | None = None


class SharedPrivateSplit(nn.Module):
    """One linear head per view per branch. No weight sharing across views: the
    shared/private boundary is not assumed to sit in the same place for every
    view, which is the point of learning it."""

    def __init__(self, n_views: int, feature_dim: int, shared_dim: int, private_dim: int):
        super().__init__()
        self.shared = nn.ModuleList(nn.Linear(feature_dim, shared_dim) for _ in range(n_views))
        self.private = nn.ModuleList(nn.Linear(feature_dim, private_dim) for _ in range(n_views))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = torch.stack([head(features[:, v, :]) for v, head in enumerate(self.shared)], dim=1)
        private = torch.stack([head(features[:, v, :]) for v, head in enumerate(self.private)], dim=1)
        return shared, private


class SharedPrivatePrismFlow(nn.Module):
    """V2 model. Composed from V1's parts; V1's own class is left alone."""

    def __init__(
        self,
        view_configs: list,
        n_classes: int,
        config: SharedPrivateConfig,
        use_discount: bool = False,
        share_weights: bool = False,
        validate_opinions: bool = True,
        dependence_method: str = "cca",
        dependence_conditioning: str = "pairwise_holdout",
        null_permutations: int = FORWARD_NULL_PERMUTATIONS,
        dependence_seed: int = 0,
    ):
        super().__init__()
        config.validate()
        self.config = config
        self.encoder = MultiViewEncoder(view_configs, share_weights=share_weights)
        self.n_views = self.encoder.n_views
        self.n_classes = n_classes
        self.split = SharedPrivateSplit(
            self.n_views, self.encoder.feature_dim, config.shared_dim, config.private_dim
        )
        evidence_dim = (
            config.shared_dim + config.private_dim if config.evidence_from == "both"
            else config.shared_dim
        )
        self.evidence_head = MultiViewEvidenceHead(
            n_views=self.n_views, feature_dim=evidence_dim, n_classes=n_classes
        )
        self.use_discount = use_discount
        self.validate_opinions = validate_opinions
        self.dependence_method = dependence_method
        self.dependence_conditioning = dependence_conditioning
        self.null_permutations = null_permutations
        self.dependence_seed = dependence_seed

    def forward(self, views: torch.Tensor, view_mask: torch.Tensor | None = None) -> SharedPrivateOutput:
        batch_size = views.shape[0]
        if view_mask is None:
            view_mask = torch.ones(batch_size, self.n_views, dtype=torch.bool, device=views.device)

        features = self.encoder(views, view_mask)
        shared, private = self.split(features)

        source = torch.cat([shared, private], dim=-1) if self.config.evidence_from == "both" else shared
        evidence = self.evidence_head(source, view_mask)

        per_view_belief, per_view_uncertainty = evidence_to_opinion(evidence)
        if self.validate_opinions:
            assert_valid_opinion(per_view_belief, per_view_uncertainty)

        # Measured on the branch the split exists to isolate. Same no_grad rule
        # as V1: the estimator is measurement, never a term in the objective.
        dependence_source = shared if self.config.dependence_on == "shared" else features

        dependence = None
        eniv = None
        if self.use_discount:
            with torch.no_grad():
                matrix = feature_dependence_matrix(
                    dependence_source,
                    evidence,
                    view_mask,
                    method=self.dependence_method,
                    conditioning=self.dependence_conditioning,
                    seed=self.dependence_seed,
                    null_permutations=self.null_permutations,
                )
                present = available_views(view_mask, n_views=self.n_views)
                alpha = soft_cluster_alpha(matrix, present)
                eniv = replace(
                    compute_eniv(matrix, present),
                    per_view_alpha=tuple(float(a) for a in alpha),
                )
                dependence = torch.as_tensor(matrix, dtype=evidence.dtype, device=evidence.device)

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

        report = disentanglement_loss(
            shared, private, view_mask, method=self.config.method
        )

        return SharedPrivateOutput(
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
            shared=shared,
            private=private,
            disentanglement=report,
        )


def build_shared_private(
    view_configs: list,
    n_classes: int,
    config: SharedPrivateConfig | None = None,
    **kwargs,
):
    """V2 when enabled, plain V1 `PrismFlow` otherwise.

    The fallback is the real V1 class, not a V2 object configured to imitate it,
    so an OFF run cannot differ from V1 by construction.
    """
    if config is None or not config.enabled:
        return PrismFlow(view_configs, n_classes, **kwargs)
    return SharedPrivatePrismFlow(view_configs, n_classes, config, **kwargs)
