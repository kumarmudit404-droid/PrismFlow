"""Dempster's rule of combination over per-view evidential opinions.

Combining opinions (b1, u1) and (b2, u2) over K classes:

    C   = sum_{i != j} b1_i * b2_j          (conflict mass)
    b_k = (b1_k*b2_k + b1_k*u2 + b2_k*u1) / (1 - C)
    u   = (u1 * u2) / (1 - C)

This is the *naive* fusion rule for Part 04: it treats every view as an
independent witness, so V agreeing copies of one view multiply confidence as
if V independent sources had agreed. Measuring and correcting that is the job
of Parts 05+; nothing here discounts by dependence.
"""

from __future__ import annotations

import torch

# As conflict -> 1 the denominator -> 0 and the combination produces inf/NaN.
_MIN_DENOM = 1e-6


def conflict_mass(belief_a: torch.Tensor, belief_b: torch.Tensor) -> torch.Tensor:
    """C = sum_{i != j} b_a_i * b_b_j, computed without the O(K^2) outer product.

    sum_{i != j} b_a_i b_b_j = (sum_i b_a_i)(sum_j b_b_j) - sum_k b_a_k b_b_k
    """
    agreement = (belief_a * belief_b).sum(dim=-1)
    return belief_a.sum(dim=-1) * belief_b.sum(dim=-1) - agreement


def dempster_combine(
    belief_a: torch.Tensor,
    uncertainty_a: torch.Tensor,
    belief_b: torch.Tensor,
    uncertainty_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine two opinions.

    belief_*      [..., K]
    uncertainty_* [...]
    returns       (belief [..., K], uncertainty [...])

    In exact arithmetic the combined masses already sum to 1, so the final
    renormalisation is a no-op. It only bites when the conflict clamp binds
    (C within 1e-6 of 1), where the unnormalised masses would otherwise leave
    the simplex.

    Under *total* conflict -- two categorical opinions backing disjoint classes
    with no uncertainty to spare -- every combined mass is zero, and there is
    no distribution to renormalise toward. The honest answer there is the
    vacuous opinion: the two sources jointly support nothing, so belief goes to
    zero and uncertainty to one. Dividing instead would emit NaN and silently
    poison every downstream statistic.
    """
    conflict = conflict_mass(belief_a, belief_b)
    denom = (1.0 - conflict).clamp_min(_MIN_DENOM).unsqueeze(-1)

    belief = (
        belief_a * belief_b
        + belief_a * uncertainty_b.unsqueeze(-1)
        + belief_b * uncertainty_a.unsqueeze(-1)
    ) / denom
    uncertainty = (uncertainty_a * uncertainty_b).unsqueeze(-1) / denom

    total = belief.sum(dim=-1, keepdim=True) + uncertainty
    degenerate = total < _MIN_DENOM
    safe_total = total.clamp_min(_MIN_DENOM)

    belief = torch.where(degenerate, torch.zeros_like(belief), belief / safe_total)
    uncertainty = torch.where(
        degenerate, torch.ones_like(uncertainty), uncertainty / safe_total
    )
    return belief, uncertainty.squeeze(-1)


def vacuous_opinion(
    batch_size: int, n_classes: int, dtype=None, device=None
) -> tuple[torch.Tensor, torch.Tensor]:
    """The zero-knowledge opinion (b = 0, u = 1).

    This is the identity element of Dempster's rule, which is what makes it a
    safe starting point for a sequential fold and the correct answer when a
    sample has no available views at all.
    """
    belief = torch.zeros(batch_size, n_classes, dtype=dtype, device=device)
    uncertainty = torch.ones(batch_size, dtype=dtype, device=device)
    return belief, uncertainty


def fuse_opinions(
    belief: torch.Tensor,
    uncertainty: torch.Tensor,
    view_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequentially fold Dempster's rule across the available views.

    belief:      [B, V, K]
    uncertainty: [B, V]
    view_mask:   [B, V] boolean, True = view present (None means all present)
    returns:     (belief [B, K], uncertainty [B])

    Masked views are skipped per sample, so a sample keeps the fused opinion it
    had before that view rather than folding in a placeholder.
    """
    if belief.dim() != 3:
        raise ValueError(f"belief must be [B, V, K], got {tuple(belief.shape)}")
    batch_size, n_views, n_classes = belief.shape
    if uncertainty.shape != (batch_size, n_views):
        raise ValueError(
            f"uncertainty shape {tuple(uncertainty.shape)} != ({batch_size}, {n_views})"
        )
    if view_mask is not None and view_mask.shape != (batch_size, n_views):
        raise ValueError(
            f"view_mask shape {tuple(view_mask.shape)} != ({batch_size}, {n_views})"
        )

    fused_b, fused_u = vacuous_opinion(
        batch_size, n_classes, dtype=belief.dtype, device=belief.device
    )

    for v in range(n_views):
        combined_b, combined_u = dempster_combine(
            fused_b, fused_u, belief[:, v, :], uncertainty[:, v]
        )
        if view_mask is None:
            fused_b, fused_u = combined_b, combined_u
        else:
            keep = view_mask[:, v]
            fused_b = torch.where(keep.unsqueeze(-1), combined_b, fused_b)
            fused_u = torch.where(keep, combined_u, fused_u)

    return fused_b, fused_u
