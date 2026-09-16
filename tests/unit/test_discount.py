import pytest
import torch

from prismflow.eniv.discount import eniv_discount_factor, shafer_discount
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


def test_discounted_model_still_trains():
    model = _model(use_discount=True)
    views = torch.randn(32, 3, 6)

    output = model(views)
    output.probs.sum().backward()

    grads = [p.grad for p in model.encoder.parameters() if p.grad is not None]
    assert grads
    assert any(torch.any(g != 0) for g in grads)
