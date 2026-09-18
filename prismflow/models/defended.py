"""The full pipeline with its diagnostics exposed, including a suspicion flag.

    views -> encoders -> evidential heads -> dependence -> ENIV -> discount
          -> Dempster fusion -> prediction

`DefendedPrismFlow` wraps `PrismFlow` and adds one thing: it runs the suspicion
detector on the opinions that fusion actually saw, and reports the flag beside
the prediction.

WHAT "DEFENDED" DOES AND DOES NOT MEAN HERE
-------------------------------------------
It does NOT add a new correction. The discount is exactly the one Parts 01-07
built, unchanged; no belief is re-weighted on account of the suspicion flag, and
the prediction of `DefendedPrismFlow` is bit-identical to `PrismFlow`'s with the
same weights and settings. A test asserts that.

This is deliberate, and it is a consequence of measurement rather than caution.
Part 09 found that the discount does not defend against colluding views
(`experiments/chorus/`, PrismFlow's attack success rate higher than naive
fusion's in 12/12 attacked cells), and that the reason is mechanical: the
correction is proportional rather than capping, sublinear in alpha, and applied
per batch rather than per sample (`docs/KNOWN_LIMITATIONS.md` L4). Wiring the
suspicion flag into an automatic belief correction would be a new mechanism
answering L4, which belongs in its own Part with its own evidence, and would
also mean changing frozen fusion code as a side effect of this one.

So the flag is an ALERT, not a control input. Detection is the half of the
system Part 09 showed to be working; this class exposes it and leaves the
decision to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from prismflow.models.prismflow import PrismFlow, PrismFlowOutput
from prismflow.statistics.suspicion import (
    DEFAULT_PERMUTATIONS,
    DEFAULT_THRESHOLD,
    SuspicionReport,
    suspicion_report,
)


@dataclass
class DefendedOutput:
    """Everything `PrismFlowOutput` carries, plus the suspicion report.

    prediction / probs / confidence / uncertainty   as PrismFlowOutput
    per_view_evidence   [B, V, K]  undiscounted evidence
    per_view_belief     [B, V, K]  opinions as fused (post-discount if on)
    per_view_reliability (V,)      the alpha applied to each view, 1.0 when the
                                   discount is off. This is `soft_cluster_alpha`
                                   from the ENIV result, not a learned quantity.
    dependence_matrix   [V, V] or None
    eniv                ENIVResult or None
    suspicion           SuspicionReport or None when dependence is unavailable
    """

    prediction: torch.Tensor
    probs: torch.Tensor
    confidence: torch.Tensor
    uncertainty: torch.Tensor
    per_view_evidence: torch.Tensor
    per_view_belief: torch.Tensor
    per_view_uncertainty: torch.Tensor
    per_view_reliability: tuple[float, ...]
    view_mask: torch.Tensor
    dependence_matrix: torch.Tensor | None = None
    eniv: object | None = None
    suspicion: SuspicionReport | None = None

    @property
    def suspicion_flag(self):
        return None if self.suspicion is None else self.suspicion.flag

    @property
    def suspicion_score(self):
        return None if self.suspicion is None else self.suspicion.score


class DefendedPrismFlow(nn.Module):
    """`PrismFlow` plus the suspicion detector. Prediction is unchanged."""

    def __init__(
        self,
        model: PrismFlow,
        suspicion_threshold: float = DEFAULT_THRESHOLD,
        suspicion_permutations: int = DEFAULT_PERMUTATIONS,
        suspicion_seed: int = 0,
    ):
        super().__init__()
        self.model = model
        self.suspicion_threshold = suspicion_threshold
        self.suspicion_permutations = suspicion_permutations
        self.suspicion_seed = suspicion_seed

    @property
    def n_views(self) -> int:
        return self.model.n_views

    @property
    def use_discount(self) -> bool:
        return self.model.use_discount

    @use_discount.setter
    def use_discount(self, value: bool) -> None:
        self.model.use_discount = value

    def _reliability(self, output: PrismFlowOutput) -> tuple[float, ...]:
        if output.eniv is not None and output.eniv.per_view_alpha is not None:
            return tuple(float(a) for a in output.eniv.per_view_alpha)
        return tuple(1.0 for _ in range(self.model.n_views))

    def forward(
        self,
        views: torch.Tensor,
        view_mask: torch.Tensor | None = None,
        detect: bool = True,
    ) -> DefendedOutput:
        output = self.model(views, view_mask)

        report = None
        if detect and output.dependence_matrix is not None:
            # The detector is measurement, never a term in the graph -- the same
            # rule the discount follows (prismflow/eniv/discount.py).
            with torch.no_grad():
                report = suspicion_report(
                    belief=output.per_view_belief,
                    dependence=output.dependence_matrix,
                    view_mask=output.view_mask,
                    evidence=output.per_view_evidence,
                    threshold=self.suspicion_threshold,
                    seed=self.suspicion_seed,
                    n_permutations=self.suspicion_permutations,
                )

        return DefendedOutput(
            prediction=output.prediction,
            probs=output.probs,
            confidence=output.confidence,
            uncertainty=output.uncertainty,
            per_view_evidence=output.per_view_evidence,
            per_view_belief=output.per_view_belief,
            per_view_uncertainty=output.per_view_uncertainty,
            per_view_reliability=self._reliability(output),
            view_mask=output.view_mask,
            dependence_matrix=output.dependence_matrix,
            eniv=output.eniv,
            suspicion=report,
        )
