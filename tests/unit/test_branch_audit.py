"""Guards for the Part 11 follow-up audit.

The audit's headline negative -- "rotating each view's shared subspace does not
move ENIV" -- is only evidence if the rotation is real. A buggy `random_rotation`
that returned something near the identity would produce exactly the same
"nothing moved" reading and would refute the hypothesis for the wrong reason.

These tests exist to make that confusion impossible: they pin that the rotation
IS a large, genuinely orthogonal transform, and separately that CCA does not see
it. The second test is the one that kills the README's stated explanation, so it
is pinned here rather than left as a claim in prose.
"""

from __future__ import annotations

import numpy as np
import torch

from experiments.branch_audit.run_branch_audit import random_rotation
from prismflow.statistics.dependence import _cca_pair

DIM = 16


def _rotation(seed: int = 0) -> torch.Tensor:
    return random_rotation(DIM, torch.Generator().manual_seed(seed), torch.float32)


def test_random_rotation_is_orthogonal():
    rotation = _rotation()
    identity = torch.eye(DIM)
    assert torch.allclose(rotation.T @ rotation, identity, atol=1e-5)
    assert abs(float(torch.linalg.det(rotation).abs()) - 1.0) < 1e-5


def test_random_rotation_is_not_near_identity():
    """The probe must actually perturb the representation.

    Without this, a degenerate rotation would silently turn the audit into a
    no-op and the 'rotation changes nothing' finding would be vacuous.
    """
    rotation = _rotation()
    assert float((rotation - torch.eye(DIM)).abs().max()) > 0.5

    torch.manual_seed(0)
    data = torch.randn(256, DIM)
    assert float((data - data @ rotation).abs().max()) > 1.0


def test_cca_is_blind_to_per_view_rotation():
    """THIS is why the README's explanation cannot be the cause.

    Part 11 proposed that the shared branches rotate independently across views
    and thereby destroy measurable cross-view structure. Dependence is measured
    with CCA, which is invariant to any invertible linear remapping of either
    side -- so a rotation is exactly the transformation the estimator cannot
    see. A cause the instrument is blind to cannot produce a reading change.
    """
    torch.manual_seed(0)
    latent = torch.randn(400, 8)
    view_a = torch.cat([latent, torch.randn(400, 8)], dim=1) @ torch.randn(DIM, DIM)
    view_b = torch.cat([latent, torch.randn(400, 8)], dim=1) @ torch.randn(DIM, DIM)

    rotation = _rotation()
    baseline = _cca_pair(view_a.numpy(), view_b.numpy())
    rotated = _cca_pair((view_a @ rotation).numpy(), view_b.numpy())

    # The views are genuinely dependent, or the invariance claim is untested.
    assert baseline > 0.5
    assert abs(baseline - rotated) < 1e-3


def test_rotation_differs_across_seeds():
    """Each view must get its OWN rotation; a shared one would be a weaker probe."""
    assert not torch.allclose(_rotation(0), _rotation(1), atol=1e-3)
