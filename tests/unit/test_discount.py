import pytest
import torch

import numpy as np

from prismflow.eniv.discount import eniv_discount_factor, evidence_discount, shafer_discount
from prismflow.eniv.eniv import per_view_alpha, soft_cluster_alpha
from prismflow.models.fusion import fuse_opinions
from prismflow.models.encoders import EncoderConfig
from prismflow.models.evidence import evidence_to_opinion, simplex_residual
from prismflow.models.prismflow import PrismFlow


def _opinion(n_samples=16, n_classes=4, seed=0):
    generator = torch.Generator().manual_seed(seed)
    evidence = torch.rand(n_samples, n_classes, generator=generator) * 10.0
    return evidence_to_opinion(evidence)


# --- simplex and identity ----------------------------------------------


def test_discount_preserves_the_simplex_constraint():
    belief, uncertainty = _opinion()
    for alpha in (0.0, 0.1, 0.5, 0.87, 1.0):
        discounted_b, discounted_u = shafer_discount(belief, uncertainty, alpha)
        assert simplex_residual(discounted_b, discounted_u) < 1e-5


def test_discount_with_alpha_one_is_the_identity():
    belief, uncertainty = _opinion()
    discounted_b, discounted_u = shafer_discount(belief, uncertainty, 1.0)

    assert torch.allclose(discounted_b, belief, atol=1e-6)
    assert torch.allclose(discounted_u, uncertainty, atol=1e-6)


def test_discount_with_alpha_zero_is_fully_vacuous():
    belief, uncertainty = _opinion()
    discounted_b, discounted_u = shafer_discount(belief, uncertainty, 0.0)

    assert torch.all(discounted_b == 0.0)
    assert torch.allclose(discounted_u, torch.ones_like(uncertainty))


def test_discount_moves_belief_into_uncertainty_not_out_of_existence():
    belief, uncertainty = _opinion()
    discounted_b, discounted_u = shafer_discount(belief, uncertainty, 0.4)

    lost_belief = (belief.sum(-1) - discounted_b.sum(-1))
    gained_uncertainty = discounted_u - uncertainty
    assert torch.allclose(lost_belief, gained_uncertainty, atol=1e-6)


def test_stronger_discount_yields_more_uncertainty():
    belief, uncertainty = _opinion()
    _, mild = shafer_discount(belief, uncertainty, 0.9)
    _, harsh = shafer_discount(belief, uncertainty, 0.3)
    assert torch.all(harsh > mild)


def test_per_view_opinions_can_be_discounted():
    belief, uncertainty = evidence_to_opinion(torch.rand(8, 4, 3) * 6.0)
    discounted_b, discounted_u = shafer_discount(belief, uncertainty, 0.5)

    assert discounted_b.shape == (8, 4, 3)
    assert simplex_residual(discounted_b, discounted_u) < 1e-5


def test_per_sample_alpha_broadcasts():
    belief, uncertainty = _opinion(n_samples=6)
    alpha = torch.linspace(0.1, 1.0, 6)

    discounted_b, discounted_u = shafer_discount(belief, uncertainty, alpha)
    assert simplex_residual(discounted_b, discounted_u) < 1e-5
    assert discounted_u[0] > discounted_u[-1]


def test_alpha_outside_the_unit_interval_is_rejected():
    belief, uncertainty = _opinion()
    with pytest.raises(ValueError):
        shafer_discount(belief, uncertainty, 1.5)
    with pytest.raises(ValueError):
        shafer_discount(belief, uncertainty, -0.1)


# --- stop-gradient ------------------------------------------------------


def test_alpha_never_carries_gradient():
    """A measure stops being a measure the moment it becomes a target: if the
    encoder could lower the loss by lowering measured dependence, it would
    learn to add orthogonal noise rather than independent representations."""
    evidence = (torch.rand(8, 3) * 5.0).requires_grad_(True)
    belief, uncertainty = evidence_to_opinion(evidence)
    alpha = torch.tensor(0.5, requires_grad=True)

    discounted_b, discounted_u = shafer_discount(belief, uncertainty, alpha)
    (discounted_b.sum() + discounted_u.sum()).backward()

    assert alpha.grad is None
    assert evidence.grad is not None


def test_gradient_still_reaches_the_beliefs():
    evidence = (torch.rand(8, 3) * 5.0).requires_grad_(True)
    belief, uncertainty = evidence_to_opinion(evidence)

    discounted_b, _ = shafer_discount(belief, uncertainty, 0.6)
    discounted_b.sum().backward()

    assert evidence.grad is not None
    assert torch.any(evidence.grad != 0)


def test_discount_factor_helper_runs_outside_the_graph():
    evidence = torch.rand(400, 3, 3, requires_grad=True)
    alpha = eniv_discount_factor(evidence)

    assert isinstance(alpha, float)
    assert 0.0 < alpha <= 1.0
    assert evidence.grad is None


# --- evidence-space per-view discount ------------------------------------


def test_evidence_discount_alpha_one_is_identity():
    evidence = torch.rand(8, 3, 4) * 5.0
    assert torch.equal(evidence_discount(evidence, torch.ones(3)), evidence)


def test_evidence_discount_alpha_zero_is_vacuous():
    evidence = torch.rand(8, 3, 4) * 5.0
    belief, uncertainty = evidence_to_opinion(evidence_discount(evidence, torch.zeros(3)))
    assert torch.all(belief == 0.0)
    assert torch.allclose(uncertainty, torch.ones(8, 3))


def test_evidence_discount_is_per_view():
    evidence = torch.rand(8, 3, 4) * 5.0
    out = evidence_discount(evidence, torch.tensor([1.0, 0.5, 0.25]))
    assert torch.equal(out[:, 0], evidence[:, 0])
    assert torch.allclose(out[:, 1], 0.5 * evidence[:, 1])
    assert torch.allclose(out[:, 2], 0.25 * evidence[:, 2])


def test_evidence_discount_accepts_per_sample_alpha():
    evidence = torch.rand(6, 3, 4) * 5.0
    alpha = torch.rand(6, 3)
    assert torch.allclose(evidence_discount(evidence, alpha), evidence * alpha.unsqueeze(-1))


def test_evidence_discount_opinions_stay_on_the_simplex():
    evidence = torch.rand(16, 5, 3) * 8.0
    belief, uncertainty = evidence_to_opinion(evidence_discount(evidence, torch.rand(5)))
    assert simplex_residual(belief, uncertainty) < 1e-5


def test_evidence_discount_rejects_bad_alpha():
    evidence = torch.rand(4, 3, 2)
    with pytest.raises(ValueError):
        evidence_discount(evidence, torch.tensor([1.0, 1.2, 0.5]))
    with pytest.raises(ValueError):
        evidence_discount(evidence, torch.ones(2))


def test_evidence_discount_alpha_carries_no_gradient_but_evidence_does():
    evidence = (torch.rand(4, 3, 2) * 5.0).requires_grad_(True)
    alpha = torch.tensor([0.5, 0.7, 1.0], requires_grad=True)
    evidence_discount(evidence, alpha).sum().backward()
    assert alpha.grad is None
    assert evidence.grad is not None and torch.any(evidence.grad != 0)


def _clone_structure(k):
    size = 4 + k
    matrix = np.eye(size)
    group = [0] + list(range(4, size))
    for a in group:
        for b in group:
            if a != b:
                matrix[a, b] = 1.0
    return matrix


def _fused_confidence(evidence, alpha):
    belief, uncertainty = evidence_to_opinion(evidence_discount(evidence, alpha))
    _, fused_uncertainty = fuse_opinions(belief, uncertainty)
    return float((1.0 - fused_uncertainty).mean())


def test_controlled_duplication_confidence_stays_near_flat():
    """The controlled test that found the uniform-discount bug, with the
    dependence structure KNOWN (not estimated): 4 distinct views + k exact
    copies of view 0. Uniform ENIV/n drifted -0.066 over k=0..4; per-view
    evidence discounting must stay within 0.01, and far closer than uniform."""
    torch.manual_seed(0)
    evidence = torch.rand(2000, 4, 3) * 6.0

    per_view, uniform = [], []
    for k in range(5):
        stacked = torch.cat([evidence, evidence[:, :1].repeat(1, k, 1)], dim=1)
        alpha = torch.as_tensor(per_view_alpha(_clone_structure(k)), dtype=evidence.dtype)
        per_view.append(_fused_confidence(stacked, alpha))

        belief, uncertainty = evidence_to_opinion(stacked)
        belief, uncertainty = shafer_discount(belief, uncertainty, 4.0 / (4 + k))
        _, fused_u = fuse_opinions(belief, uncertainty)
        uniform.append(float((1.0 - fused_u).mean()))

    per_view_drift = max(per_view) - min(per_view)
    uniform_drift = max(uniform) - min(uniform)
    assert per_view_drift < 0.01, f"per-view confidence {per_view}"
    assert per_view_drift < uniform_drift / 5


def test_controlled_duplication_stays_near_flat_under_estimation_noise():
    """The gap between Task 3 and the clone experiment: the model sees a NOISY
    per-batch estimate, not the exact structure. Measured copy structure
    (c=0.94, others 0.047) plus noise sd=0.05 per batch of 64. Soft cluster
    weights must keep fused confidence within 0.01 over k=0..4."""
    torch.manual_seed(0)
    evidence = torch.rand(2048, 4, 3) * 6.0
    rng = np.random.default_rng(0)

    confidences = []
    for k in range(5):
        size = 4 + k
        stacked = torch.cat([evidence, evidence[:, :1].repeat(1, k, 1)], dim=1)
        total = 0.0
        for start in range(0, 2048, 64):
            matrix = np.full((size, size), 0.047)
            np.fill_diagonal(matrix, 1.0)
            group = [0] + list(range(4, size))
            for a in group:
                for b in group:
                    if a != b:
                        matrix[a, b] = 0.94
            noise = np.triu(rng.normal(0.0, 0.05, (size, size)), 1)
            matrix = np.clip(matrix + noise + noise.T, -1.0, 1.0)
            np.fill_diagonal(matrix, 1.0)

            alpha = torch.as_tensor(soft_cluster_alpha(matrix), dtype=evidence.dtype)
            belief, uncertainty = evidence_to_opinion(
                evidence_discount(stacked[start : start + 64], alpha)
            )
            _, fused_u = fuse_opinions(belief, uncertainty)
            total += float((1.0 - fused_u).sum())
        confidences.append(total / 2048)

    assert max(confidences) - min(confidences) < 0.01, f"confidence {confidences}"


def test_model_uses_soft_cluster_alpha(monkeypatch):
    import prismflow.models.prismflow as prismflow_module

    called = {}
    real = prismflow_module.soft_cluster_alpha

    def spy(*args, **kwargs):
        called["yes"] = True
        return real(*args, **kwargs)

    monkeypatch.setattr(prismflow_module, "soft_cluster_alpha", spy)
    torch.manual_seed(0)
    configs = [EncoderConfig(input_dim=6, hidden_dims=[16], feature_dim=8) for _ in range(3)]
    PrismFlow(configs, n_classes=3, use_discount=True)(torch.randn(32, 3, 6))
    assert called.get("yes")


def test_duplicate_does_not_discount_untouched_views_evidence():
    torch.manual_seed(1)
    evidence = torch.rand(50, 6, 3) * 6.0  # views 0 and 4, 5 are copies below
    evidence[:, 4] = evidence[:, 0]
    evidence[:, 5] = evidence[:, 0]
    alpha = torch.as_tensor(per_view_alpha(_clone_structure(2)), dtype=evidence.dtype)
    out = evidence_discount(evidence, alpha)
    # alpha is 1 - O(ridge) here, not bit-exactly 1
    assert torch.allclose(out[:, 1:4], evidence[:, 1:4], rtol=1e-5, atol=0.0)


# --- wired into the model ----------------------------------------------


def _model(n_views=3, use_discount=False, seed=0):
    torch.manual_seed(seed)
    configs = [
        EncoderConfig(input_dim=6, hidden_dims=[16], feature_dim=8) for _ in range(n_views)
    ]
    return PrismFlow(configs, n_classes=3, use_discount=use_discount)


def test_discount_off_leaves_diagnostics_empty():
    output = _model(use_discount=False)(torch.randn(64, 3, 6))
    assert output.dependence_matrix is None
    assert output.eniv is None


def test_discount_on_populates_diagnostics():
    output = _model(use_discount=True)(torch.randn(64, 3, 6))

    assert output.dependence_matrix is not None
    assert output.dependence_matrix.shape == (3, 3)
    assert output.eniv is not None
    assert 1.0 <= output.eniv.effective_views <= 3.0
    assert output.eniv.nominal_views == 3


def test_discounted_model_is_never_more_confident_than_the_baseline():
    views = torch.randn(128, 3, 6)

    baseline = _model(use_discount=False, seed=7)(views)
    discounted = _model(use_discount=True, seed=7)(views)

    assert torch.all(discounted.uncertainty >= baseline.uncertainty - 1e-6)


def test_discounted_model_output_stays_on_the_simplex():
    output = _model(use_discount=True)(torch.randn(64, 3, 6))
    assert torch.allclose(output.probs.sum(-1), torch.ones(64), atol=1e-5)
    assert torch.all(output.uncertainty >= 0.0)
    assert torch.all(output.uncertainty <= 1.0)


def test_discount_path_defaults_to_the_validated_estimator():
    from prismflow.models.prismflow import FORWARD_NULL_PERMUTATIONS

    model = _model(use_discount=True)
    assert model.dependence_method == "cca"
    assert model.dependence_conditioning == "pairwise_holdout"
    assert model.null_permutations == FORWARD_NULL_PERMUTATIONS


def test_forward_permutations_cheaper_than_reporting():
    """The discount path runs every batch and inside attack loops, so it must
    not default to the expensive reporting setting."""
    from prismflow.models.prismflow import (
        FORWARD_NULL_PERMUTATIONS,
        REPORTING_NULL_PERMUTATIONS,
    )

    assert 0 < FORWARD_NULL_PERMUTATIONS < REPORTING_NULL_PERMUTATIONS


def test_discount_path_measures_encoder_features_not_evidence(monkeypatch):
    import prismflow.models.prismflow as prismflow_module

    seen = {}

    def spy(features, evidence, *args, **kwargs):
        seen["features_dim"] = features.shape[-1]
        seen["evidence_dim"] = evidence.shape[-1]
        seen["null_permutations"] = kwargs.get("null_permutations")
        n_views = features.shape[1]
        import numpy as np

        return np.eye(n_views)

    monkeypatch.setattr(prismflow_module, "feature_dependence_matrix", spy)

    torch.manual_seed(0)
    configs = [EncoderConfig(input_dim=6, hidden_dims=[16], feature_dim=8) for _ in range(3)]
    model = PrismFlow(configs, n_classes=3, use_discount=True, null_permutations=7)
    model(torch.randn(32, 3, 6))

    assert seen["features_dim"] == 8
    assert seen["evidence_dim"] == 3
    assert seen["null_permutations"] == 7


def test_discounted_model_is_deterministic_across_calls():
    model = _model(use_discount=True)
    model.eval()
    views = torch.randn(64, 3, 6)
    first = model(views)
    second = model(views)

    assert first.eniv.effective_views == second.eniv.effective_views
    assert torch.allclose(first.uncertainty, second.uncertainty)


def test_model_applies_per_view_alpha_not_the_scalar_ratio(monkeypatch):
    """Feed the model a known clone structure: 3 distinct views + 1 copy of
    view 0. Views 1 and 2 must be left undiscounted even though the scalar
    efficiency_ratio is below 1."""
    import prismflow.models.prismflow as prismflow_module

    def known_structure(features, evidence, *args, **kwargs):
        matrix = np.eye(4)
        matrix[0, 3] = matrix[3, 0] = 1.0
        return matrix

    monkeypatch.setattr(prismflow_module, "feature_dependence_matrix", known_structure)

    torch.manual_seed(0)
    configs = [EncoderConfig(input_dim=6, hidden_dims=[16], feature_dim=8) for _ in range(4)]
    model = PrismFlow(configs, n_classes=3, use_discount=True)
    model.eval()
    output = model(torch.randn(32, 4, 6))

    alpha = output.eniv.per_view_alpha
    assert alpha is not None
    assert alpha[1] == pytest.approx(1.0, abs=1e-6)
    assert alpha[2] == pytest.approx(1.0, abs=1e-6)
    assert alpha[0] == pytest.approx(0.5, abs=1e-4)
    assert alpha[3] == pytest.approx(0.5, abs=1e-4)
    assert output.eniv.efficiency_ratio < 1.0

    undiscounted_b, undiscounted_u = evidence_to_opinion(output.per_view_evidence)
    assert torch.allclose(output.per_view_belief[:, 1:3], undiscounted_b[:, 1:3])
    assert torch.allclose(output.per_view_uncertainty[:, 1:3], undiscounted_u[:, 1:3])


def test_discounted_model_still_trains():
    model = _model(use_discount=True)
    views = torch.randn(32, 3, 6)

    output = model(views)
    output.probs.sum().backward()

    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads
    assert any(torch.any(g != 0) for g in grads)
