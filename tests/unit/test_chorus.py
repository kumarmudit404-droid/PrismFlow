import torch

import pytest

from prismflow.attacks.baseline_attacks import pgd_attack
from prismflow.attacks.chorus import ChorusConfig, choose_targets, chorus_attack, chorus_objective, select_views
from prismflow.train import TrainConfig, build_model
from prismflow.utils.seed import set_seed


def _setup(use_discount=False, n_views=4, batch=16, seed=0):
    set_seed(seed)
    config = TrainConfig(n_views=n_views, use_discount=use_discount)
    model = build_model(config)
    model.eval()
    views = torch.randn(batch, n_views, config.d_view)
    view_mask = torch.ones(batch, n_views, dtype=torch.bool)
    return model, views, view_mask


# --- optimiser behaviour ------------------------------------------------


def test_objective_decreases_monotonically_over_pgd_iterations():
    model, views, view_mask = _setup()
    result = chorus_attack(model, views, view_mask, ChorusConfig(k=2, epsilon=0.3, beta=1.0, steps=25))
    assert len(result.history) > 1
    assert all(b <= a + 1e-12 for a, b in zip(result.history, result.history[1:]))
    assert result.history[-1] < result.history[0]


def test_objective_history_is_the_negated_objective():
    model, views, view_mask = _setup()
    config = ChorusConfig(k=2, epsilon=0.3, beta=0.5, steps=10)
    result = chorus_attack(model, views, view_mask, config)
    with torch.no_grad():
        final = chorus_objective(
            model, result.perturbed(views), view_mask, result.compromised, result.target, config.beta
        )
    assert float(-final) == pytest.approx(result.history[-1], abs=1e-6)


def test_attack_with_discount_on_also_improves_its_objective():
    model, views, view_mask = _setup(use_discount=True)
    result = chorus_attack(model, views, view_mask, ChorusConfig(k=2, epsilon=0.3, beta=1.0, steps=15))
    assert result.history[-1] < result.history[0]


# --- constraints --------------------------------------------------------


@pytest.mark.parametrize("epsilon", [0.05, 0.2, 1.0])
def test_perturbations_respect_the_epsilon_constraint(epsilon):
    model, views, view_mask = _setup()
    result = chorus_attack(model, views, view_mask, ChorusConfig(k=3, epsilon=epsilon, steps=20))
    assert result.delta.abs().max() <= epsilon + 1e-6


def test_uncompromised_views_are_never_perturbed():
    model, views, view_mask = _setup()
    result = chorus_attack(model, views, view_mask, ChorusConfig(k=2, epsilon=0.5, steps=20))
    assert result.compromised == (0, 1)
    assert torch.equal(result.delta[:, 2:, :], torch.zeros_like(result.delta[:, 2:, :]))


def test_baseline_pgd_respects_epsilon_and_view_scope():
    model, views, view_mask = _setup()
    result = pgd_attack(model, views, view_mask, ChorusConfig(k=2, epsilon=0.2, steps=15))
    assert result.delta.abs().max() <= 0.2 + 1e-6
    assert torch.equal(result.delta[:, 2:, :], torch.zeros_like(result.delta[:, 2:, :]))


# --- k = 0 and other no-ops ---------------------------------------------


def test_k_zero_leaves_predictions_bit_identical():
    model, views, view_mask = _setup()
    with torch.no_grad():
        clean = model(views, view_mask)
    result = chorus_attack(model, views, view_mask, ChorusConfig(k=0, epsilon=0.5, steps=20))
    assert torch.equal(result.delta, torch.zeros_like(views))
    with torch.no_grad():
        attacked = model(result.perturbed(views), view_mask)
    assert torch.equal(attacked.prediction, clean.prediction)
    assert torch.equal(attacked.probs, clean.probs)


def test_epsilon_zero_is_a_no_op():
    model, views, view_mask = _setup()
    result = chorus_attack(model, views, view_mask, ChorusConfig(k=2, epsilon=0.0, steps=20))
    assert torch.equal(result.delta, torch.zeros_like(views))


def test_model_parameters_and_mode_are_restored():
    model, views, view_mask = _setup()
    model.train()
    before = [p.clone() for p in model.parameters()]
    chorus_attack(model, views, view_mask, ChorusConfig(k=2, epsilon=0.2, steps=5))
    assert model.training
    assert all(p.requires_grad for p in model.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


# --- the agreement term -------------------------------------------------


def test_beta_penalises_disagreement_between_compromised_views():
    """Higher beta buys closer agreement between the attacker's own beliefs."""
    model, views, view_mask = _setup()
    target = choose_targets(model, views, view_mask, ChorusConfig(seed=0))

    spreads = []
    for beta in (0.0, 5.0):
        result = chorus_attack(
            model, views, view_mask, ChorusConfig(k=2, epsilon=0.3, beta=beta, steps=30), target=target
        )
        with torch.no_grad():
            belief = model(result.perturbed(views), view_mask).per_view_belief[:, list(result.compromised), :]
        spreads.append(float(((belief[:, 0] - belief[:, 1]) ** 2).sum(dim=-1).mean()))
    assert spreads[1] < spreads[0]


def test_targets_avoid_the_predicted_class():
    model, views, view_mask = _setup()
    with torch.no_grad():
        predicted = model(views, view_mask).prediction
    for strategy in ("least_likely", "random"):
        target = choose_targets(model, views, view_mask, ChorusConfig(target_strategy=strategy, seed=1))
        assert not torch.equal(target, predicted)
        assert (target != predicted).all()


def test_select_views_validates_and_pins():
    assert select_views(4, ChorusConfig(k=3)) == (0, 1, 2)
    assert select_views(4, ChorusConfig(views=(1, 3))) == (1, 3)
    with pytest.raises(ValueError):
        select_views(4, ChorusConfig(k=5))
    with pytest.raises(ValueError):
        select_views(4, ChorusConfig(views=(0, 4)))
