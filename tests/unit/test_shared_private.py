import pytest
import torch

from prismflow.models.disentanglement_losses import disentanglement_loss, total_loss
from prismflow.models.prismflow import PrismFlow
from prismflow.models.shared_private import (
    SharedPrivateConfig,
    SharedPrivatePrismFlow,
    build_shared_private,
)
from prismflow.train import TrainConfig, build_model
from prismflow.utils.seed import set_seed


def _view_configs(n_views=4, d_view=16, feature_dim=32):
    from prismflow.models.encoders import EncoderConfig

    return [
        EncoderConfig(input_dim=d_view, hidden_dims=[64, 64], feature_dim=feature_dim)
        for _ in range(n_views)
    ]


def _batch(n=96, n_views=4, d_view=16, seed=0):
    set_seed(seed)
    views = torch.randn(n, n_views, d_view)
    view_mask = torch.ones(n, n_views, dtype=torch.bool)
    return views, view_mask


def _v2(use_discount=True, seed=0, **overrides):
    set_seed(seed)
    config = SharedPrivateConfig(enabled=True, **overrides)
    model = SharedPrivatePrismFlow(_view_configs(), n_classes=3, config=config, use_discount=use_discount)
    model.eval()
    return model


# --- V1 must not break -------------------------------------------------


def test_the_default_config_is_off():
    assert SharedPrivateConfig().enabled is False


def test_disabled_config_returns_the_real_v1_class():
    """Not a V2 object imitating V1 -- the actual V1 class, so an OFF run cannot
    drift from V1 by construction."""
    model = build_shared_private(_view_configs(), n_classes=3, config=SharedPrivateConfig())
    assert type(model) is PrismFlow
    assert not isinstance(model, SharedPrivatePrismFlow)


def test_no_config_at_all_returns_v1():
    assert type(build_shared_private(_view_configs(), n_classes=3)) is PrismFlow


def test_enabled_config_returns_v2():
    config = SharedPrivateConfig(enabled=True)
    assert isinstance(build_shared_private(_view_configs(), n_classes=3, config=config), SharedPrivatePrismFlow)


def test_v1_model_is_untouched_by_this_module():
    """A V1 model built the normal way still has no shared/private anything."""
    set_seed(0)
    model = build_model(TrainConfig(n_views=4, use_discount=True))
    model.eval()
    views, view_mask = _batch()
    with torch.no_grad():
        output = model(views, view_mask)
    assert not hasattr(output, "shared")
    assert not hasattr(model, "split")


def test_lambda_2_of_zero_returns_the_task_loss_object_itself():
    task = torch.tensor(1.25, requires_grad=True)
    report = disentanglement_loss(torch.randn(64, 2, 4), torch.randn(64, 2, 4))
    assert total_loss(task, report, lambda_2=0.0) is task


# --- shapes and validity ------------------------------------------------


def test_forward_produces_branches_of_the_configured_width():
    model = _v2(shared_dim=8, private_dim=5)
    views, view_mask = _batch()
    with torch.no_grad():
        output = model(views, view_mask)
    assert output.shared.shape == (96, 4, 8)
    assert output.private.shape == (96, 4, 5)


def test_fused_opinion_is_a_valid_simplex():
    model = _v2()
    views, view_mask = _batch()
    with torch.no_grad():
        output = model(views, view_mask)
    belief = output.probs - output.uncertainty.unsqueeze(-1) / 3
    assert (belief.sum(dim=-1) + output.uncertainty) == pytest.approx(torch.ones(96), abs=1e-5)


def test_discount_off_leaves_no_dependence_matrix():
    model = _v2(use_discount=False)
    views, view_mask = _batch()
    with torch.no_grad():
        output = model(views, view_mask)
    assert output.dependence_matrix is None
    assert output.eniv is None


def test_evidence_from_shared_narrows_the_head_input():
    model = _v2(shared_dim=8, private_dim=5, evidence_from="shared")
    views, view_mask = _batch()
    with torch.no_grad():
        output = model(views, view_mask)
    assert output.per_view_evidence.shape == (96, 4, 3)


def test_masked_views_are_handled():
    model = _v2()
    views, view_mask = _batch()
    view_mask[:, 3] = False
    with torch.no_grad():
        output = model(views, view_mask)
    assert torch.isfinite(output.probs).all()


# --- config validation --------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"method": "cosine"},
        {"evidence_from": "private"},
        {"dependence_on": "evidence"},
        {"shared_dim": 0},
        {"private_dim": -1},
        {"lambda_2": -0.5},
    ],
)
def test_invalid_configs_are_rejected(overrides):
    with pytest.raises(ValueError):
        SharedPrivateConfig(enabled=True, **overrides).validate()


# --- the disentanglement term -------------------------------------------


def test_loss_is_per_view_and_reports_each_view():
    report = disentanglement_loss(torch.randn(80, 4, 6), torch.randn(80, 4, 6))
    assert len(report.per_view) == 4
    assert len(report.per_view_normalised) == 4


def test_identical_branches_score_higher_than_independent_ones():
    set_seed(0)
    shared = torch.randn(96, 2, 4)
    coupled = disentanglement_loss(shared, shared.clone())
    independent = disentanglement_loss(shared, torch.randn(96, 2, 4))
    assert coupled.loss > independent.loss
    assert coupled.normalised > independent.normalised


def test_nonlinear_coupling_is_caught_by_hsic_and_missed_by_orthogonality():
    """The module's reason for existing, at the loss level."""
    set_seed(0)
    shared = torch.randn(128, 1, 3)
    private = shared.pow(2)

    by_hsic = disentanglement_loss(shared, private, method="hsic").normalised
    by_orthogonality = disentanglement_loss(shared, private, method="orthogonality").normalised

    # The claim is relative, not a particular magnitude: the linear estimator
    # reports near-independence for a pair that is deterministically dependent,
    # while the kernel one does not. (dim=1: 0.44 vs 0.02, a 21x gap.)
    assert by_orthogonality < 0.1
    assert by_hsic > 4 * by_orthogonality


def test_loss_is_differentiable_into_both_branches():
    set_seed(0)
    shared = torch.randn(96, 2, 4, requires_grad=True)
    private = torch.randn(96, 2, 4, requires_grad=True)
    disentanglement_loss(shared, private).loss.backward()
    assert shared.grad is not None and torch.isfinite(shared.grad).all()
    assert private.grad is not None and torch.isfinite(private.grad).all()


def test_a_view_with_too_few_present_samples_is_skipped_not_nan():
    shared, private = torch.randn(64, 2, 3), torch.randn(64, 2, 3)
    view_mask = torch.ones(64, 2, dtype=torch.bool)
    view_mask[:, 1] = False
    report = disentanglement_loss(shared, private, view_mask)
    assert report.per_view[1] != report.per_view[1]  # NaN for the absent view
    assert torch.isfinite(report.loss)


def test_mismatched_branch_shapes_are_rejected():
    with pytest.raises(ValueError):
        disentanglement_loss(torch.randn(64, 2, 3), torch.randn(64, 3, 3))


def test_gradient_descent_on_the_penalty_reduces_measured_dependence():
    """End to end: the term the trainer would add actually disentangles."""
    set_seed(0)
    features = torch.randn(128, 1, 8)
    projection = torch.nn.Linear(8, 4)
    optimiser = torch.optim.Adam(projection.parameters(), lr=0.05)

    shared = features
    before = float(disentanglement_loss(shared, projection(features).unsqueeze(1).squeeze(1)).normalised)
    for _ in range(60):
        optimiser.zero_grad()
        private = projection(features)
        disentanglement_loss(shared, private).loss.backward()
        optimiser.step()
    after = float(disentanglement_loss(shared, projection(features)).normalised)
    assert after < before
