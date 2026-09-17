"""Discounting evidential opinions for redundancy.

The model applies `evidence_discount` with PER-VIEW factors from
`prismflow.eniv.eniv.soft_cluster_alpha`. `shafer_discount` below, with one scalar
alpha = n_eff / n for every view, was the V1 mechanism. It is kept and tested,
but it discounts views that were never duplicated. `evidence_discount`
documents the measurements behind the change.

Shafer discounting:

    b_k' = alpha * b_k
    u'   = 1 - alpha * (1 - u)

with alpha = n_eff / n. The simplex is preserved exactly rather than
approximately: sum_k b_k' + u' = alpha*(sum_k b_k + u) - alpha + 1 = 1 for any
alpha, because the mass taken off the beliefs is precisely the mass added to
uncertainty. Discounting moves belief to "I don't know"; it never destroys it.

Discounting is applied BEFORE Dempster fusion. Applying it after would let the
naive fusion first compound the redundant agreement and only then shave the
result -- the inflation has already happened by that point, and scaling a
wrong number down does not make it right.


WHY ENIV IS COMPUTED UNDER no_grad AND DETACHED
-----------------------------------------------
This will be questioned, so the reasoning is recorded here.

If the encoder can reduce the loss by lowering MEASURED dependence, gradient
descent will do exactly that -- and the cheapest way to lower a correlation is
not to learn genuinely independent representations, it is to add orthogonal
noise to the evidence vectors. Measured correlation falls, ENIV rises, the
discount vanishes, and the model becomes overconfident again while displaying
an excellent independence score. A measure stops being a measure the moment it
becomes a target.

So in V1 alpha enters the graph as a constant. Gradients still flow through
the discounted beliefs to the encoders (the model must still learn to
classify), but no gradient flows into how alpha was arrived at, which means
nothing in training can be optimised against the dependence estimate itself.

Part 12 addresses this properly with two-timescale training. V1 sidesteps it
with a stop-gradient.
"""

from __future__ import annotations

import torch

from prismflow.models.evidence import assert_valid_opinion


def shafer_discount(
    belief: torch.Tensor,
    uncertainty: torch.Tensor,
    alpha,
    validate: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Discount an opinion toward vacuity by factor alpha.

    belief:      [..., K]
    uncertainty: [...]
    alpha:       scalar in [0, 1], or a tensor broadcastable to `uncertainty`

    alpha = 1 is the identity; alpha = 0 returns the fully vacuous opinion.
    """
    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=belief.dtype, device=belief.device)
    alpha = alpha.to(dtype=belief.dtype, device=belief.device)

    if torch.any(alpha < 0.0) or torch.any(alpha > 1.0):
        raise ValueError("alpha must lie in [0, 1]")

    # alpha never carries gradient -- see the module docstring.
    alpha = alpha.detach()

    if alpha.ndim == 0:
        belief_scale = alpha
        uncertainty_scale = alpha
    else:
        uncertainty_scale = alpha.reshape(uncertainty.shape)
        belief_scale = uncertainty_scale.unsqueeze(-1)

    discounted_belief = belief * belief_scale
    discounted_uncertainty = 1.0 - uncertainty_scale * (1.0 - uncertainty)

    if validate:
        assert_valid_opinion(discounted_belief, discounted_uncertainty)

    return discounted_belief, discounted_uncertainty


def evidence_discount(evidence: torch.Tensor, alpha) -> torch.Tensor:
    """Scale each view's evidence by its own discount factor.

    evidence: [B, V, K] non-negative evidence
    alpha:    [V] (one factor per view) or [B, V], values in [0, 1]
    returns:  [B, V, K] discounted evidence; form opinions from it with
              `evidence_to_opinion`, which yields a valid simplex by construction

    This is what the model applies, with alpha from
    `prismflow.eniv.eniv.soft_cluster_alpha`. Two differences from
    `shafer_discount`, both measured:

    PER VIEW, NOT ONE SCALAR. One alpha = ENIV/n for every view penalised views
    that were never duplicated. With ENIV held at exactly 4, fused confidence
    fell 0.975 -> 0.910 over 0..4 copies of one view.

    EVIDENCE, NOT BELIEF. Dempster fusion here is not linear in belief-scaled
    opinions. Fusing g copies each belief-discounted by 1/g gave 0.47-0.56
    confidence against 0.73 for one undiscounted copy. Scaling evidence gave
    0.78-0.83 against 0.735. On 4 views + k exact copies with the dependence
    structure known, fused confidence drifted +0.006 over k = 0..4 (per-view
    beliefs: -0.013, per-view 1/VIF: -0.026, uniform scalar: -0.066). The drift
    is small but not zero: the fusion rule does not add evidence exactly.

    Alpha is detached for the same reason as in `shafer_discount` (see the
    module docstring).
    """
    if not torch.is_tensor(alpha):
        alpha = torch.as_tensor(alpha, dtype=evidence.dtype, device=evidence.device)
    alpha = alpha.detach().to(dtype=evidence.dtype, device=evidence.device)

    if torch.any(alpha < 0.0) or torch.any(alpha > 1.0):
        raise ValueError("alpha must lie in [0, 1]")

    batch_size, n_views, _ = evidence.shape
    if alpha.shape == (n_views,):
        scale = alpha.view(1, n_views, 1)
    elif alpha.shape == (batch_size, n_views):
        scale = alpha.unsqueeze(-1)
    else:
        raise ValueError(
            f"alpha must be [{n_views}] or [{batch_size}, {n_views}], got {tuple(alpha.shape)}"
        )
    return evidence * scale


@torch.no_grad()
def eniv_discount_factor(evidence, view_mask=None, method: str = "pearson") -> float:
    """Measure dependence and return the Shafer discount factor alpha.

    Wrapped in no_grad so the estimator cannot appear in the autograd graph at
    all, not merely be detached at the end.
    """
    from prismflow.eniv.eniv import compute_eniv
    from prismflow.statistics.dependence import (
        available_views,
        dependence_matrix,
    )

    matrix = dependence_matrix(evidence, view_mask, method=method)
    present = available_views(view_mask, n_views=matrix.shape[0])
    return compute_eniv(matrix, present).efficiency_ratio
