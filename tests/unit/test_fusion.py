import pytest
import torch

from prismflow.models.encoders import EncoderConfig
from prismflow.models.evidence import evidence_to_opinion, simplex_residual
from prismflow.models.fusion import (
    conflict_mass,
    dempster_combine,
    fuse_opinions,
    vacuous_opinion,
)
from prismflow.models.prismflow import PrismFlow, PrismFlowOutput


def _opinion(belief_row, uncertainty):
    belief = torch.tensor([belief_row], dtype=torch.float32)
    return belief, torch.tensor([uncertainty], dtype=torch.float32)


# --- conflict mass ------------------------------------------------------


def test_conflict_mass_matches_brute_force_double_sum():
    belief_a = torch.rand(7, 5)
    belief_b = torch.rand(7, 5)

    expected = torch.zeros(7)
    for i in range(5):
        for j in range(5):
            if i != j:
                expected += belief_a[:, i] * belief_b[:, j]

    assert torch.allclose(conflict_mass(belief_a, belief_b), expected, atol=1e-5)


def test_agreeing_opinions_have_low_conflict():
    agree = conflict_mass(torch.tensor([[0.9, 0.0, 0.0]]), torch.tensor([[0.9, 0.0, 0.0]]))
    disagree = conflict_mass(torch.tensor([[0.9, 0.0, 0.0]]), torch.tensor([[0.0, 0.9, 0.0]]))
    assert agree.item() < disagree.item()


# --- pairwise combination ----------------------------------------------


def test_combination_stays_on_the_simplex():
    belief, uncertainty = evidence_to_opinion(torch.rand(16, 4) * 10.0)
    other_b, other_u = evidence_to_opinion(torch.rand(16, 4) * 10.0)

    fused_b, fused_u = dempster_combine(belief, uncertainty, other_b, other_u)
    assert simplex_residual(fused_b, fused_u) < 1e-5


def test_fusing_two_identical_confident_opinions_reduces_uncertainty():
    belief, uncertainty = _opinion([0.6, 0.2, 0.0], 0.2)
    fused_b, fused_u = dempster_combine(belief, uncertainty, belief.clone(), uncertainty.clone())

    assert fused_u.item() < uncertainty.item()
    assert fused_b[0, 0].item() > belief[0, 0].item()


def test_repeated_agreement_keeps_driving_uncertainty_down():
    """The naive baseline's defining weakness: V copies of one view compound
    confidence as if V independent witnesses had agreed. Part 05 discounts it;
    Part 04 must exhibit it."""
    belief, uncertainty = _opinion([0.5, 0.1, 0.0], 0.4)
    fused_b, fused_u = belief, uncertainty

    previous = fused_u.item()
    for _ in range(4):
        fused_b, fused_u = dempster_combine(fused_b, fused_u, belief, uncertainty)
        assert fused_u.item() < previous
        previous = fused_u.item()


def test_maximally_conflicting_opinions_return_finite_values_on_the_simplex():
    belief_a, uncertainty_a = _opinion([1.0, 0.0], 0.0)
    belief_b, uncertainty_b = _opinion([0.0, 1.0], 0.0)

    fused_b, fused_u = dempster_combine(belief_a, uncertainty_a, belief_b, uncertainty_b)

    assert torch.isfinite(fused_b).all()
    assert torch.isfinite(fused_u).all()
    assert simplex_residual(fused_b, fused_u) < 1e-5
    # total conflict supports nothing jointly, so the result is vacuous
    assert fused_u.item() == pytest.approx(1.0)


def test_near_total_conflict_does_not_produce_nan():
    belief_a, uncertainty_a = _opinion([0.999999, 0.0], 0.000001)
    belief_b, uncertainty_b = _opinion([0.0, 0.999999], 0.000001)

    fused_b, fused_u = dempster_combine(belief_a, uncertainty_a, belief_b, uncertainty_b)
    assert torch.isfinite(fused_b).all()
    assert torch.isfinite(fused_u).all()
    assert simplex_residual(fused_b, fused_u) < 1e-5


def test_maximal_conflict_gradients_are_finite():
    belief_a = torch.tensor([[1.0, 0.0]], requires_grad=True)
    uncertainty_a = torch.tensor([0.0], requires_grad=True)
    belief_b = torch.tensor([[0.0, 1.0]], requires_grad=True)
    uncertainty_b = torch.tensor([0.0], requires_grad=True)

    fused_b, fused_u = dempster_combine(belief_a, uncertainty_a, belief_b, uncertainty_b)
    (fused_b.sum() + fused_u.sum()).backward()

    assert torch.isfinite(belief_a.grad).all()
    assert torch.isfinite(uncertainty_a.grad).all()


def test_vacuous_opinion_is_the_identity_element():
    belief, uncertainty = evidence_to_opinion(torch.rand(9, 3) * 7.0)
    empty_b, empty_u = vacuous_opinion(9, 3)

    fused_b, fused_u = dempster_combine(empty_b, empty_u, belief, uncertainty)
    assert torch.allclose(fused_b, belief, atol=1e-5)
    assert torch.allclose(fused_u, uncertainty, atol=1e-5)


def test_combination_is_commutative():
    belief_a, uncertainty_a = evidence_to_opinion(torch.rand(6, 4) * 5.0)
    belief_b, uncertainty_b = evidence_to_opinion(torch.rand(6, 4) * 5.0)

    forward = dempster_combine(belief_a, uncertainty_a, belief_b, uncertainty_b)
    reverse = dempster_combine(belief_b, uncertainty_b, belief_a, uncertainty_a)

    assert torch.allclose(forward[0], reverse[0], atol=1e-5)
    assert torch.allclose(forward[1], reverse[1], atol=1e-5)


# --- folding across views -----------------------------------------------


def test_fuse_opinions_shapes_and_simplex():
    belief, uncertainty = evidence_to_opinion(torch.rand(12, 4, 3) * 6.0)
    fused_b, fused_u = fuse_opinions(belief, uncertainty)

    assert fused_b.shape == (12, 3)
    assert fused_u.shape == (12,)
    assert simplex_residual(fused_b, fused_u) < 1e-5


def test_fuse_opinions_is_order_independent():
    belief, uncertainty = evidence_to_opinion(torch.rand(5, 4, 3) * 6.0)
    permutation = [2, 0, 3, 1]

    straight = fuse_opinions(belief, uncertainty)
    shuffled = fuse_opinions(belief[:, permutation, :], uncertainty[:, permutation])

    assert torch.allclose(straight[0], shuffled[0], atol=1e-5)
    assert torch.allclose(straight[1], shuffled[1], atol=1e-5)


def test_masked_views_are_excluded_from_the_fold():
    belief, uncertainty = evidence_to_opinion(torch.rand(4, 3, 3) * 6.0)

    mask = torch.ones(4, 3, dtype=torch.bool)
    mask[:, 2] = False

    masked = fuse_opinions(belief, uncertainty, mask)
    only_two = fuse_opinions(belief[:, :2, :], uncertainty[:, :2])

    assert torch.allclose(masked[0], only_two[0], atol=1e-5)
    assert torch.allclose(masked[1], only_two[1], atol=1e-5)


def test_sample_with_no_available_views_stays_vacuous():
    belief, uncertainty = evidence_to_opinion(torch.rand(3, 4, 3) * 6.0)
    mask = torch.ones(3, 4, dtype=torch.bool)
    mask[1, :] = False

    fused_b, fused_u = fuse_opinions(belief, uncertainty, mask)

    assert torch.all(fused_b[1] == 0.0)
    assert fused_u[1].item() == pytest.approx(1.0)


def test_fuse_opinions_rejects_mismatched_shapes():
    belief = torch.rand(4, 3, 2)
    with pytest.raises(ValueError):
        fuse_opinions(belief, torch.rand(4, 2))
    with pytest.raises(ValueError):
        fuse_opinions(belief, torch.rand(4, 3), torch.ones(4, 2, dtype=torch.bool))
    with pytest.raises(ValueError):
        fuse_opinions(torch.rand(4, 3), torch.rand(4, 3))


# --- model --------------------------------------------------------------


def _make_model(n_views=3, d_view=6, n_classes=3):
    configs = [
        EncoderConfig(input_dim=d_view, hidden_dims=[16], feature_dim=8)
        for _ in range(n_views)
    ]
    return PrismFlow(configs, n_classes=n_classes)


def test_model_output_contract():
    model = _make_model()
    output = model(torch.randn(5, 3, 6))

    assert isinstance(output, PrismFlowOutput)
    assert output.prediction.shape == (5,)
    assert output.probs.shape == (5, 3)
    assert output.confidence.shape == (5,)
    assert output.uncertainty.shape == (5,)
    assert output.per_view_evidence.shape == (5, 3, 3)
    assert output.per_view_belief.shape == (5, 3, 3)
    assert output.per_view_uncertainty.shape == (5, 3)
    assert output.view_mask.shape == (5, 3)

    # Part 05 populates these without changing the signature
    assert output.dependence_matrix is None
    assert output.eniv is None


def test_model_confidence_is_one_minus_uncertainty():
    model = _make_model()
    output = model(torch.randn(6, 3, 6))
    assert torch.allclose(output.confidence, 1.0 - output.uncertainty, atol=1e-6)


def test_model_probs_sum_to_one():
    model = _make_model()
    output = model(torch.randn(6, 3, 6))
    assert torch.allclose(output.probs.sum(dim=-1), torch.ones(6), atol=1e-5)


def test_fused_alpha_round_trips_to_the_fused_opinion():
    model = _make_model()
    output = model(torch.randn(6, 3, 6))

    alpha = output.fused_alpha
    assert torch.all(alpha >= 1.0 - 1e-5)
    assert torch.allclose(alpha / alpha.sum(-1, keepdim=True), output.probs, atol=1e-4)


# --- end to end ---------------------------------------------------------


def test_training_does_not_collapse_to_maximum_vacuity():
    """Guards the classic EDL failure: with the KL running at full strength
    from epoch 0 the model emits u ~= 1 for every input and never recovers.

    This is a regression guard on a fixed seed, not a reported experimental
    result -- the contract's 5-seed rule governs findings, which live in
    results/, not assertions.
    """
    from prismflow.train import TrainConfig, train_one_seed

    config = TrainConfig(
        n_samples=600,
        n_views=3,
        epochs=15,
        batch_size=64,
        anneal_epochs=5,
        hidden_dims=[32],
        feature_dim=16,
    )
    result = train_one_seed(config, seed=0)

    assert result["test"]["mean_uncertainty"] < 0.9
    assert result["test"]["accuracy"] > 0.5


def test_run_experiment_refuses_fewer_than_five_seeds(tmp_path):
    from prismflow.train import TrainConfig, run_experiment

    with pytest.raises(ValueError, match="at least 5 seeds"):
        run_experiment(TrainConfig(), seeds=[0, 1], results_dir=tmp_path)
