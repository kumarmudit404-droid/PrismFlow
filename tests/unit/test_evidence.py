import pytest
import torch

from prismflow.models.evidence import (
    EvidenceHead,
    MultiViewEvidenceHead,
    alpha_to_opinion,
    anneal_coefficient,
    assert_valid_opinion,
    edl_loss,
    edl_type2_ml_loss,
    evidence_to_alpha,
    evidence_to_opinion,
    kl_dirichlet_to_uniform,
    opinion_to_alpha,
    opinion_to_probs,
    simplex_residual,
)


# --- softplus, not ReLU ------------------------------------------------


def test_evidence_is_strictly_positive_for_negative_logits():
    """A ReLU head would emit exactly 0 here and kill the gradient forever."""
    head = EvidenceHead(feature_dim=4, n_classes=3)
    with torch.no_grad():
        head.linear.weight.fill_(0.0)
        head.linear.bias.fill_(-20.0)

    evidence = head(torch.randn(8, 4))
    assert torch.all(evidence > 0.0)


def test_evidence_head_gradient_survives_large_negative_preactivation():
    head = EvidenceHead(feature_dim=4, n_classes=3)
    with torch.no_grad():
        head.linear.weight.fill_(0.0)
        head.linear.bias.fill_(-20.0)

    head(torch.ones(4, 4)).sum().backward()
    assert head.linear.bias.grad is not None
    assert torch.all(head.linear.bias.grad > 0.0)


# --- opinion algebra ---------------------------------------------------


def test_belief_plus_uncertainty_sums_to_one_for_every_sample():
    evidence = torch.rand(32, 5) * 10.0
    belief, uncertainty = evidence_to_opinion(evidence)
    assert torch.allclose(
        belief.sum(dim=-1) + uncertainty, torch.ones(32), atol=1e-5
    )
    assert simplex_residual(belief, uncertainty) < 1e-5


def test_belief_plus_uncertainty_sums_to_one_per_view():
    evidence = torch.rand(16, 4, 3) * 5.0
    belief, uncertainty = evidence_to_opinion(evidence)
    assert torch.allclose(
        belief.sum(dim=-1) + uncertainty, torch.ones(16, 4), atol=1e-5
    )


def test_zero_evidence_is_the_vacuous_opinion():
    belief, uncertainty = evidence_to_opinion(torch.zeros(6, 4))
    assert torch.all(belief == 0.0)
    assert torch.allclose(uncertainty, torch.ones(6))


def test_uncertainty_decreases_as_evidence_grows():
    _, low = evidence_to_opinion(torch.full((1, 3), 0.5))
    _, high = evidence_to_opinion(torch.full((1, 3), 50.0))
    assert high.item() < low.item()


def test_opinion_to_alpha_round_trips():
    alpha = torch.rand(10, 4) * 8.0 + 1.0
    belief, uncertainty = alpha_to_opinion(alpha)
    assert torch.allclose(opinion_to_alpha(belief, uncertainty), alpha, atol=1e-4)


def test_opinion_to_probs_is_a_distribution():
    belief, uncertainty = evidence_to_opinion(torch.rand(12, 4) * 6.0)
    probs = opinion_to_probs(belief, uncertainty)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(12), atol=1e-5)
    assert torch.all(probs >= 0.0)


def test_assert_valid_opinion_rejects_off_simplex_input():
    with pytest.raises(AssertionError):
        assert_valid_opinion(torch.full((2, 3), 0.9), torch.full((2,), 0.9))


# --- KL annealing ------------------------------------------------------


def test_anneal_coefficient_is_exactly_zero_at_epoch_zero():
    """Full-strength KL from epoch 0 collapses the model to total vacuity."""
    assert anneal_coefficient(0, anneal_epochs=10) == 0.0


def test_anneal_coefficient_ramps_then_saturates():
    assert anneal_coefficient(5, anneal_epochs=10) == pytest.approx(0.5)
    assert anneal_coefficient(10, anneal_epochs=10) == pytest.approx(1.0)
    assert anneal_coefficient(99, anneal_epochs=10) == pytest.approx(1.0)


def test_kl_term_contributes_nothing_at_epoch_zero():
    alpha = torch.rand(8, 3) * 5.0 + 1.0
    targets = torch.nn.functional.one_hot(torch.randint(0, 3, (8,)), 3).float()

    at_zero = edl_loss(alpha, targets, epoch=0, anneal_epochs=10)
    likelihood_only = edl_type2_ml_loss(alpha, targets).mean()
    assert torch.allclose(at_zero, likelihood_only, atol=1e-6)


def test_kl_term_contributes_once_annealed():
    alpha = torch.rand(8, 3) * 5.0 + 1.0
    targets = torch.nn.functional.one_hot(torch.randint(0, 3, (8,)), 3).float()

    at_zero = edl_loss(alpha, targets, epoch=0, anneal_epochs=10)
    annealed = edl_loss(alpha, targets, epoch=10, anneal_epochs=10)
    assert annealed > at_zero


# --- losses ------------------------------------------------------------


def test_kl_to_uniform_is_zero_for_the_uniform_dirichlet():
    kl = kl_dirichlet_to_uniform(torch.ones(4, 5))
    assert torch.allclose(kl, torch.zeros(4), atol=1e-5)


def test_kl_to_uniform_is_non_negative():
    alpha = torch.rand(20, 4) * 10.0 + 1.0
    assert torch.all(kl_dirichlet_to_uniform(alpha) >= -1e-6)


def test_type2_ml_loss_rewards_correct_evidence():
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    correct = evidence_to_alpha(torch.tensor([[20.0, 0.0, 0.0]]))
    wrong = evidence_to_alpha(torch.tensor([[0.0, 20.0, 0.0]]))
    assert edl_type2_ml_loss(correct, targets) < edl_type2_ml_loss(wrong, targets)


def test_edl_loss_reduction_modes_agree():
    alpha = torch.rand(6, 3) * 4.0 + 1.0
    targets = torch.nn.functional.one_hot(torch.randint(0, 3, (6,)), 3).float()

    none = edl_loss(alpha, targets, epoch=3, anneal_epochs=10, reduction="none")
    assert none.shape == (6,)
    assert torch.allclose(
        none.mean(), edl_loss(alpha, targets, epoch=3, anneal_epochs=10), atol=1e-6
    )


def test_edl_loss_rejects_unknown_reduction():
    alpha = torch.rand(4, 3) + 1.0
    targets = torch.nn.functional.one_hot(torch.randint(0, 3, (4,)), 3).float()
    with pytest.raises(ValueError):
        edl_loss(alpha, targets, epoch=0, reduction="median")


# --- multi-view head ---------------------------------------------------


def test_multiview_head_output_shape():
    head = MultiViewEvidenceHead(n_views=4, feature_dim=8, n_classes=3)
    evidence = head(torch.randn(5, 4, 8), torch.ones(5, 4, dtype=torch.bool))
    assert evidence.shape == (5, 4, 3)
    assert torch.all(evidence >= 0.0)


def test_multiview_head_heads_are_not_shared():
    head = MultiViewEvidenceHead(n_views=3, feature_dim=6, n_classes=4)
    for i in range(3):
        for j in range(i + 1, 3):
            for p_i, p_j in zip(head.heads[i].parameters(), head.heads[j].parameters()):
                assert p_i is not p_j


def test_masked_view_emits_vacuous_opinion():
    head = MultiViewEvidenceHead(n_views=3, feature_dim=6, n_classes=4)
    mask = torch.ones(5, 3, dtype=torch.bool)
    mask[:, 1] = False

    evidence = head(torch.randn(5, 3, 6), mask)
    assert torch.all(evidence[:, 1, :] == 0.0)

    belief, uncertainty = evidence_to_opinion(evidence)
    assert torch.all(belief[:, 1, :] == 0.0)
    assert torch.allclose(uncertainty[:, 1], torch.ones(5))


def test_multiview_head_rejects_wrong_mask_shape():
    head = MultiViewEvidenceHead(n_views=3, feature_dim=6, n_classes=4)
    with pytest.raises(ValueError):
        head(torch.randn(5, 3, 6), torch.ones(5, 2, dtype=torch.bool))


def test_multiview_head_rejects_zero_views():
    with pytest.raises(ValueError):
        MultiViewEvidenceHead(n_views=0, feature_dim=4, n_classes=3)
