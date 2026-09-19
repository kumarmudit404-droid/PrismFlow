"""The adaptive Chorus attack: an attacker who knows the ENIV discount exists.

Threat model unchanged from `chorus.py` (controlled research simulation, our
own models, synthetic data): the attacker controls k of n views, sees the
model white-box, and perturbs only its own views' inputs within
|| delta ||_inf <= epsilon. Labels, weights and the other views are untouched.

WHAT IS NEW

The Chorus attack operates the trust mechanism: it makes its views AGREE, and
Dempster's rule rewards low conflict with confidence. That is visible to the
dependence estimator, because agreement is exactly what the estimator
measures. An attacker who knows the discount exists would not accept that
trade. It wants its views to agree on the wrong class while APPEARING
statistically independent:

    maximise over {delta_v : v in C}
        sum_{v in C} log b_v[target]                      targeted
        - beta  * sum_{i,j in C} || b_i - b_j ||^2        agreement
        - gamma * max(0, rho_hat(C) - tau)                EVASION

`rho_hat(C)` is the MEASURED dependence among the compromised views -- the
quantity the discount acts on. The hinge means the attacker pays nothing
until its measured dependence exceeds tau, then pays linearly. gamma = 0
recovers the Part 09 Chorus attack exactly, which is what makes the sweep
interpretable: any change across gamma is attributable to the evasion term
alone.

Note the tension the attacker is managing. The beta term PUSHES ITS VIEWS
TOGETHER and the gamma term PULLS THEM APART, because the estimator measures
roughly what beta maximises. This is not a flaw in the objective, it is the
adaptive attacker's actual problem, and the sweep measures whether there is
any gamma at which it comes out ahead.

BPDA, AND WHY IT IS REQUIRED HERE

`PrismFlow.forward` computes the dependence matrix, ENIV and alpha inside a
`torch.no_grad()` block (prismflow/models/prismflow.py:147) so that nothing
in training can be optimised against the estimator. That is correct for
training and fatal for an honest attack evaluation: it makes
d(rho_hat)/d(delta) identically zero, so an attacker differentiating through
the model would conclude the evasion term is unreachable and the defence
would look robust for a reason that is an artifact of OUR implementation,
not a property of the defence.

So the evasion term uses a Backward Pass Differentiable Approximation. The
forward value is the model's true measured dependence; the backward pass uses
a differentiable surrogate computed from the same per-view beliefs:

    rho = surrogate + (true - surrogate).detach()

Forward: exactly `true`. Backward: d(surrogate)/d(delta). The surrogate is
mean pairwise cosine agreement among the compromised views, which is the same
quantity `statistics.suspicion.pairwise_agreement` measures, re-expressed in
torch so gradients exist.

If a future change makes the estimator differentiable, set `use_bpda=False`
and the true path is used directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from prismflow.attacks.chorus import (
    AttackResult,
    ChorusConfig,
    choose_targets,
    select_views,
)

_LOG_FLOOR = 1e-12
_MIN_NORM = 1e-8


@dataclass
class AdaptiveConfig(ChorusConfig):
    """Chorus hyperparameters plus the evasion term.

    gamma = 0 reduces this to the Chorus attack. tau is the dependence level
    below which the attacker pays no evasion penalty.
    """

    gamma: float = 0.0
    tau: float = 0.0
    use_bpda: bool = True


@dataclass
class AdaptiveResult(AttackResult):
    """Adds what the evasion term actually achieved, so the sweep can report
    whether measured dependence moved rather than only whether success did."""

    rho_initial: float = 0.0
    rho_final: float = 0.0
    gamma: float = 0.0


def compromised_agreement(belief: torch.Tensor, compromised) -> torch.Tensor:
    """Differentiable mean pairwise cosine agreement among compromised views.

    The surrogate for the estimator's dependence. Mirrors
    `suspicion.pairwise_agreement` (cosine between belief vectors) but stays
    inside the autograd graph. Returns a scalar: mean over ordered
    off-diagonal pairs, mean over the batch.
    """
    chosen = belief[:, list(compromised), :]  # [B, |C|, K]
    if chosen.shape[1] < 2:
        return torch.zeros((), dtype=belief.dtype, device=belief.device)

    norm = chosen.norm(dim=-1, keepdim=True).clamp_min(_MIN_NORM)
    unit = chosen / norm
    cosine = torch.einsum("bik,bjk->bij", unit, unit)  # [B, |C|, |C|]

    size = cosine.shape[1]
    off_diagonal = ~torch.eye(size, dtype=torch.bool, device=cosine.device)
    return cosine[:, off_diagonal].mean()


@torch.no_grad()
def measured_dependence(model, views, view_mask, compromised) -> torch.Tensor:
    """The estimator's OWN dependence among the compromised views.

    This is the number the discount acts on. It comes back detached because
    the model computes it under no_grad; `adaptive_objective` handles that
    with BPDA rather than pretending the gradient exists.
    """
    output = model(views, view_mask)
    matrix = output.dependence_matrix
    if matrix is None:
        # use_discount=False: there is no estimator to evade.
        return torch.zeros((), dtype=views.dtype, device=views.device)

    index = torch.as_tensor(list(compromised), device=matrix.device, dtype=torch.long)
    block = matrix.index_select(0, index).index_select(1, index)
    size = block.shape[0]
    if size < 2:
        return torch.zeros((), dtype=matrix.dtype, device=matrix.device)
    off_diagonal = ~torch.eye(size, dtype=torch.bool, device=matrix.device)
    return block[off_diagonal].mean()


def adaptive_objective(
    model,
    views: torch.Tensor,
    view_mask: torch.Tensor,
    compromised,
    target: torch.Tensor,
    config: AdaptiveConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (objective, rho_used). Ascent on the objective is the attack."""
    output = model(views, view_mask)
    belief = output.per_view_belief

    chosen = belief[:, list(compromised), :]
    target_belief = chosen.gather(
        -1, target.view(-1, 1, 1).expand(-1, chosen.shape[1], 1)
    ).squeeze(-1)
    objective = torch.log(target_belief.clamp_min(_LOG_FLOOR)).sum(dim=1).mean()

    if config.beta != 0.0 and chosen.shape[1] > 1:
        pairwise = ((chosen.unsqueeze(2) - chosen.unsqueeze(1)) ** 2).sum(dim=-1)
        objective = objective - config.beta * pairwise.sum(dim=(1, 2)).mean()

    surrogate = compromised_agreement(belief, compromised)
    if config.gamma == 0.0:
        # No evasion term: report the surrogate for logging, do not spend a
        # forward pass on the estimator.
        return objective, surrogate.detach()

    if config.use_bpda:
        true = measured_dependence(model, views, view_mask, compromised)
        true = true.to(surrogate.dtype)
        rho = surrogate + (true - surrogate).detach()
    else:
        rho = surrogate

    penalty = torch.clamp(rho - config.tau, min=0.0)
    return objective - config.gamma * penalty, rho.detach()


def adaptive_attack(
    model,
    views: torch.Tensor,
    view_mask: torch.Tensor,
    config: AdaptiveConfig,
    target: torch.Tensor | None = None,
) -> AdaptiveResult:
    """Projected gradient ascent with backtracking, as in `chorus_attack`.

    A step that does not improve the objective is rejected and the step size
    halved, so accepted iterates improve monotonically. Backtracking makes the
    attack stronger, never weaker: a rejected step is never taken.
    """
    if config.epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {config.epsilon}")
    if config.gamma < 0:
        raise ValueError(f"gamma must be >= 0, got {config.gamma}")

    compromised = select_views(views.shape[1], config)
    if target is None:
        target = choose_targets(model, views, view_mask, config)

    was_training = model.training
    model.eval()
    requires_grad = [p.requires_grad for p in model.parameters()]
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    try:
        delta = torch.zeros_like(views)
        with torch.no_grad():
            rho_initial = float(
                measured_dependence(model, views, view_mask, compromised)
            )

        if not compromised or config.epsilon == 0.0:
            return AdaptiveResult(
                delta, compromised, target, [], 0, 0,
                rho_initial=rho_initial, rho_final=rho_initial, gamma=config.gamma,
            )

        mask = torch.zeros_like(views)
        mask[:, list(compromised), :] = 1.0
        step_size = config.resolved_step_size()

        with torch.no_grad():
            best, _ = adaptive_objective(
                model, views + delta, view_mask, compromised, target, config
            )
        history = [float(-best)]
        accepted = rejected = 0

        for _ in range(config.steps):
            candidate = delta.clone().requires_grad_(True)
            objective, _ = adaptive_objective(
                model, views + candidate, view_mask, compromised, target, config
            )
            (gradient,) = torch.autograd.grad(objective, candidate)

            with torch.no_grad():
                stepped = delta + step_size * gradient.sign() * mask
                stepped = stepped.clamp(-config.epsilon, config.epsilon) * mask
                value, _ = adaptive_objective(
                    model, views + stepped, view_mask, compromised, target, config
                )

            if value > best:
                delta, best = stepped, value
                history.append(float(-best))
                accepted += 1
            else:
                step_size /= 2.0
                rejected += 1

        with torch.no_grad():
            rho_final = float(
                measured_dependence(model, views + delta, view_mask, compromised)
            )

        return AdaptiveResult(
            delta, compromised, target, history, accepted, rejected,
            rho_initial=rho_initial, rho_final=rho_final, gamma=config.gamma,
        )
    finally:
        for parameter, flag in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(flag)
        model.train(was_training)


def random_search_attack(
    model,
    views: torch.Tensor,
    view_mask: torch.Tensor,
    config: AdaptiveConfig,
    target: torch.Tensor | None = None,
    restarts: int = 64,
) -> AdaptiveResult:
    """Gradient-FREE control: uniform sign patterns in the epsilon ball.

    The sanity check the Part 14 brief requires. If the gradient attack
    succeeds and this does not, the gradient attack is real. If THIS one
    succeeds where the gradient attack fails, the gradient attack was
    handicapped by our implementation -- masked gradients -- and any
    robustness claim resting on it is void.

    No gradients are taken anywhere in this function.
    """
    compromised = select_views(views.shape[1], config)
    if target is None:
        target = choose_targets(model, views, view_mask, config)

    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    mask = torch.zeros_like(views)
    mask[:, list(compromised), :] = 1.0

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            best_delta = torch.zeros_like(views)
            best_value, _ = adaptive_objective(
                model, views, view_mask, compromised, target, config
            )
            history = [float(-best_value)]
            rho_initial = float(
                measured_dependence(model, views, view_mask, compromised)
            )

            for _ in range(restarts):
                signs = torch.randint(
                    0, 2, views.shape, generator=generator, dtype=views.dtype
                ).to(views.device)
                candidate = (signs * 2.0 - 1.0) * config.epsilon * mask
                value, _ = adaptive_objective(
                    model, views + candidate, view_mask, compromised, target, config
                )
                if value > best_value:
                    best_delta, best_value = candidate, value
                    history.append(float(-best_value))

            rho_final = float(
                measured_dependence(model, views + best_delta, view_mask, compromised)
            )

        return AdaptiveResult(
            best_delta, compromised, target, history, len(history) - 1, 0,
            rho_initial=rho_initial, rho_final=rho_final, gamma=config.gamma,
        )
    finally:
        model.train(was_training)
