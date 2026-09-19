"""The adaptive attack, pinned on the properties an evaluation depends on.

These are not results about whether the defence holds. They check that the
attack is not accidentally handicapped by our own code, which is the failure
mode the Part 14 brief exists to prevent: a defence that looks robust because
the attacker could not compute a gradient through it has not been evaluated.

The load-bearing test is `test_measured_dependence_is_detached` together with
`test_bpda_restores_the_gradient`. The first shows the estimator hands back a
constant, so a naive attacker's evasion gradient is exactly zero. The second
shows BPDA recovers a usable one. Without the pair, a flat gamma sweep would
be unreadable -- it could mean the defence resists evasion, or it could mean
the attacker was never able to try.
"""

from __future__ import annotations

import torch

from prismflow.attacks.adaptive import (
    AdaptiveConfig,
    adaptive_attack,
    adaptive_objective,
    compromised_agreement,
    measured_dependence,
    random_search_attack,
)
from prismflow.attacks.chorus import ChorusConfig, chorus_attack
from prismflow.models.encoders import EncoderConfig
from prismflow.models.prismflow import PrismFlow
from prismflow.utils.seed import set_seed

N_VIEWS = 4
N_CLASSES = 3
DIM = 6
BATCH = 16


def build_model(use_discount: bool = True) -> PrismFlow:
    set_seed(0)
    configs = [
        EncoderConfig(input_dim=DIM, hidden_dims=[16], feature_dim=8) for _ in range(N_VIEWS)
    ]
    return PrismFlow(configs, n_classes=N_CLASSES, use_discount=use_discount)


def build_batch():
    set_seed(1)
    views = torch.randn(BATCH, N_VIEWS, DIM)
    view_mask = torch.ones(BATCH, N_VIEWS, dtype=torch.bool)
    return views, view_mask


def test_gamma_zero_reproduces_chorus_exactly():
    """gamma = 0 must BE the Part 09 attack, or the sweep is uninterpretable."""
    model = build_model()
    views, mask = build_batch()
    shared = dict(k=2, epsilon=0.5, beta=1.0, steps=6, seed=0)

    chorus = chorus_attack(model, views, mask, ChorusConfig(**shared))
    adaptive = adaptive_attack(model, views, mask, AdaptiveConfig(gamma=0.0, **shared))

    assert torch.allclose(chorus.delta, adaptive.delta, atol=1e-6)
    assert adaptive.steps_accepted == chorus.steps_accepted


def test_measured_dependence_is_detached():
    """The estimator runs under no_grad, so its output carries no gradient.

    This is why BPDA is required. If this test ever fails, the estimator has
    become differentiable and `use_bpda=False` is the honest setting.
    """
    model = build_model()
    views, mask = build_batch()
    rho = measured_dependence(model, views, mask, (0, 1))
    assert not rho.requires_grad


def test_bpda_restores_the_gradient():
    """With BPDA the evasion term produces a non-zero gradient w.r.t. delta."""
    model = build_model()
    views, mask = build_batch()
    config = AdaptiveConfig(k=2, epsilon=0.5, beta=0.0, steps=1, gamma=10.0, tau=0.0)

    delta = torch.zeros_like(views, requires_grad=True)
    target = torch.zeros(BATCH, dtype=torch.long)
    objective, _ = adaptive_objective(model, views + delta, mask, (0, 1), target, config)
    (gradient,) = torch.autograd.grad(objective, delta)

    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_bpda_forward_value_is_the_true_measurement():
    """BPDA may change the BACKWARD pass only. The forward value must be exact."""
    model = build_model()
    views, mask = build_batch()
    compromised = (0, 1)
    target = torch.zeros(BATCH, dtype=torch.long)

    true = measured_dependence(model, views, mask, compromised)
    config = AdaptiveConfig(k=2, epsilon=0.5, beta=0.0, gamma=1.0, tau=0.0, use_bpda=True)
    _, reported = adaptive_objective(model, views, mask, compromised, target, config)

    assert torch.allclose(reported, true.to(reported.dtype), atol=1e-6)


def test_surrogate_is_differentiable_and_tracks_agreement():
    """The BPDA surrogate must have a gradient and must rise with agreement."""
    belief = torch.rand(BATCH, N_VIEWS, N_CLASSES, requires_grad=True)
    value = compromised_agreement(belief, (0, 1))
    (gradient,) = torch.autograd.grad(value, belief)
    assert gradient.abs().sum() > 0

    identical = torch.zeros(2, N_VIEWS, N_CLASSES)
    identical[:, 0, 0] = 1.0
    identical[:, 1, 0] = 1.0
    opposed = torch.zeros(2, N_VIEWS, N_CLASSES)
    opposed[:, 0, 0] = 1.0
    opposed[:, 1, 1] = 1.0
    assert compromised_agreement(identical, (0, 1)) > compromised_agreement(opposed, (0, 1))


def test_evasion_term_changes_the_objective_and_the_search_direction():
    """gamma must reach both the value and the gradient the attacker follows.

    Deliberately NOT asserting that rho_final falls. On an untrained model
    the sign-based step saturates at the epsilon corner for every gamma, so
    both attacks return an identical delta and the assertion would pass or
    fail on how degenerate the fixture is rather than on the mechanism.
    Whether evasion actually lowers measured dependence is an empirical claim
    about trained models, measured over 5 seeds in experiments/adaptive/.
    What belongs here is that the term is wired in at all.
    """
    model = build_model()
    views, mask = build_batch()
    compromised = (0, 1, 2)
    target = torch.zeros(BATCH, dtype=torch.long)

    # tau must sit BELOW the fixture's measured dependence or the hinge is
    # inactive and the term is silently a no-op. An untrained model can
    # measure negative dependence, so tau is derived rather than assumed --
    # this is exactly the trap the first version of this test fell into.
    rho = float(measured_dependence(model, views, mask, compromised))
    shared = dict(k=3, epsilon=1.0, beta=1.0, tau=rho - 0.5)

    plain = AdaptiveConfig(gamma=0.0, **shared)
    evading = AdaptiveConfig(gamma=20.0, **shared)

    delta = torch.zeros_like(views, requires_grad=True)
    plain_objective, _ = adaptive_objective(
        model, views + delta, mask, compromised, target, plain
    )
    (plain_gradient,) = torch.autograd.grad(plain_objective, delta)

    delta = torch.zeros_like(views, requires_grad=True)
    evading_objective, _ = adaptive_objective(
        model, views + delta, mask, compromised, target, evading
    )
    (evading_gradient,) = torch.autograd.grad(evading_objective, delta)

    # The penalty is charged ...
    assert float(evading_objective.detach()) < float(plain_objective.detach())
    # ... and it moves the direction the attacker actually follows.
    assert not torch.allclose(plain_gradient, evading_gradient, atol=1e-8)


def test_perturbation_respects_the_budget_and_the_compromised_set():
    model = build_model()
    views, mask = build_batch()
    config = AdaptiveConfig(k=2, epsilon=0.25, beta=1.0, steps=5, gamma=1.0)
    result = adaptive_attack(model, views, mask, config)

    assert result.delta.abs().max() <= config.epsilon + 1e-6
    untouched = [v for v in range(N_VIEWS) if v not in result.compromised]
    assert torch.count_nonzero(result.delta[:, untouched, :]) == 0


def test_zero_epsilon_is_a_no_op():
    model = build_model()
    views, mask = build_batch()
    result = adaptive_attack(model, views, mask, AdaptiveConfig(k=2, epsilon=0.0, gamma=1.0))
    assert torch.count_nonzero(result.delta) == 0


def test_model_state_is_restored():
    """An attack must not leave the model in eval mode or with grads off."""
    model = build_model()
    model.train()
    flags_before = [p.requires_grad for p in model.parameters()]
    views, mask = build_batch()

    adaptive_attack(model, views, mask, AdaptiveConfig(k=2, epsilon=0.5, steps=3, gamma=1.0))

    assert model.training
    assert [p.requires_grad for p in model.parameters()] == flags_before


def test_random_search_stays_in_the_ball_and_needs_no_gradient():
    model = build_model()
    views, mask = build_batch()
    config = AdaptiveConfig(k=2, epsilon=0.3, beta=1.0, gamma=0.0, seed=0)

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    result = random_search_attack(model, views, mask, config, restarts=8)

    assert result.delta.abs().max() <= config.epsilon + 1e-6
    untouched = [v for v in range(N_VIEWS) if v not in result.compromised]
    assert torch.count_nonzero(result.delta[:, untouched, :]) == 0


def test_undefended_model_has_nothing_to_evade():
    """With use_discount=False there is no estimator, so rho is zero."""
    model = build_model(use_discount=False)
    views, mask = build_batch()
    assert float(measured_dependence(model, views, mask, (0, 1))) == 0.0


def test_negative_gamma_is_rejected():
    model = build_model()
    views, mask = build_batch()
    try:
        adaptive_attack(model, views, mask, AdaptiveConfig(k=2, epsilon=0.5, gamma=-1.0))
    except ValueError:
        return
    raise AssertionError("negative gamma must raise")
