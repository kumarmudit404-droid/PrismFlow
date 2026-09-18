"""The Chorus attack: colluding views optimised to AGREE on a wrong class.

Threat model (controlled research simulation, our own models, synthetic data).
The attacker controls k of n views, sees the model white-box, and may perturb
only its own views' inputs, within || delta ||_inf <= epsilon. Labels, weights
and the other views are untouched.

OBJECTIVE

    maximise over {delta_v : v in C}
        sum_{v in C} log b_v[target]
        - beta * sum_{i,j in C} || b_i - b_j ||^2

`b_v` is view v's belief vector, the opinion actually fused. The first term is
an ordinary targeted attack. The second term is the whole idea: it penalises
DISAGREEMENT between the attacker's own views. Dempster's rule multiplies
vacuity and divides by (1 - conflict), so two views that agree closely produce
low conflict mass, and the fused opinion becomes confident rather than vacuous.
The attack therefore does not evade the trust mechanism, it operates it: a
chorus singing in unison is trusted more than the same voices disagreeing.

beta = 0 reduces the objective to independent targeted PGD (the terms separate
per view), which is why `baseline_attacks.pgd_attack` is the right comparison.

OPTIMISER

Projected gradient ascent with backtracking: a step that does not improve the
objective is rejected and the step size halved. Accepted iterates therefore
improve monotonically, which is what `history` records. Backtracking makes the
attack stronger, not weaker: a rejected step is never taken.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from prismflow.models.evidence import evidence_to_opinion

_LOG_FLOOR = 1e-12
TARGET_STRATEGIES = ("least_likely", "random", "fixed")


@dataclass
class ChorusConfig:
    """Attack hyperparameters. `views` pins the compromised set; otherwise the
    first k views are used (the generator gives no view special status)."""

    k: int = 2
    epsilon: float = 0.2
    beta: float = 1.0
    steps: int = 40
    step_size: float | None = None  # defaults to 2.5 * epsilon / steps
    target_strategy: str = "least_likely"
    fixed_target: int = 0
    seed: int = 0
    views: tuple[int, ...] | None = None

    def resolved_step_size(self) -> float:
        return self.step_size if self.step_size is not None else 2.5 * self.epsilon / max(self.steps, 1)


@dataclass
class AttackResult:
    delta: torch.Tensor  # [B, V, D], zero outside the compromised views
    compromised: tuple[int, ...]
    target: torch.Tensor  # [B]
    history: list[float] = field(default_factory=list)  # loss = -objective, per accepted iterate
    steps_accepted: int = 0
    steps_rejected: int = 0

    def perturbed(self, views: torch.Tensor) -> torch.Tensor:
        return views + self.delta


def select_views(n_views: int, config: ChorusConfig) -> tuple[int, ...]:
    if config.views is not None:
        chosen = tuple(config.views)
        if len(set(chosen)) != len(chosen) or any(not 0 <= v < n_views for v in chosen):
            raise ValueError(f"views {chosen} invalid for {n_views} views")
        return chosen
    if not 0 <= config.k <= n_views:
        raise ValueError(f"k must be in [0, {n_views}], got {config.k}")
    return tuple(range(config.k))


@torch.no_grad()
def choose_targets(model, views: torch.Tensor, view_mask: torch.Tensor, config: ChorusConfig) -> torch.Tensor:
    """Per-sample target class, fixed before the attack starts."""
    if config.target_strategy == "fixed":
        return torch.full((views.shape[0],), config.fixed_target, dtype=torch.long, device=views.device)

    probs = model(views, view_mask).probs
    if config.target_strategy == "least_likely":
        return probs.argmin(dim=-1)
    if config.target_strategy == "random":
        # Uniform over the non-predicted classes, so the target is never the
        # class the model already returns.
        generator = torch.Generator(device="cpu").manual_seed(config.seed)
        n_classes = probs.shape[-1]
        offset = torch.randint(1, n_classes, (probs.shape[0],), generator=generator).to(probs.device)
        return (probs.argmax(dim=-1) + offset) % n_classes
    raise ValueError(f"target_strategy must be one of {TARGET_STRATEGIES}, got {config.target_strategy!r}")


def chorus_objective(
    model, views: torch.Tensor, view_mask: torch.Tensor, compromised, target: torch.Tensor, beta: float
) -> torch.Tensor:
    """Mean over the batch of  sum_v log b_v[target] - beta * sum_{i,j} ||b_i - b_j||^2."""
    belief = model(views, view_mask).per_view_belief  # [B, V, K]
    chosen = belief[:, list(compromised), :]  # [B, |C|, K]
    target_belief = chosen.gather(-1, target.view(-1, 1, 1).expand(-1, chosen.shape[1], 1)).squeeze(-1)
    agreement = torch.log(target_belief.clamp_min(_LOG_FLOOR)).sum(dim=1)

    if beta != 0.0 and chosen.shape[1] > 1:
        # sum over ordered pairs (i, j), matching the objective as written.
        pairwise = ((chosen.unsqueeze(2) - chosen.unsqueeze(1)) ** 2).sum(dim=-1).sum(dim=(1, 2))
        agreement = agreement - beta * pairwise
    return agreement.mean()


def chorus_attack(
    model,
    views: torch.Tensor,
    view_mask: torch.Tensor,
    config: ChorusConfig,
    target: torch.Tensor | None = None,
) -> AttackResult:
    """Optimise perturbations of the compromised views jointly. Weights are untouched."""
    if config.epsilon < 0:
        raise ValueError(f"epsilon must be >= 0, got {config.epsilon}")
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
        if not compromised or config.epsilon == 0.0:
            return AttackResult(delta, compromised, target, [], 0, 0)

        mask = torch.zeros_like(views)
        mask[:, list(compromised), :] = 1.0
        step_size = config.resolved_step_size()

        with torch.no_grad():
            best = chorus_objective(model, views + delta, view_mask, compromised, target, config.beta)
        history = [float(-best)]
        accepted = rejected = 0

        for _ in range(config.steps):
            candidate_delta = delta.clone().requires_grad_(True)
            objective = chorus_objective(
                model, views + candidate_delta, view_mask, compromised, target, config.beta
            )
            (gradient,) = torch.autograd.grad(objective, candidate_delta)

            with torch.no_grad():
                stepped = delta + step_size * gradient.sign() * mask
                stepped = stepped.clamp(-config.epsilon, config.epsilon) * mask
                value = chorus_objective(model, views + stepped, view_mask, compromised, target, config.beta)

            if value > best:
                delta, best = stepped, value
                history.append(float(-best))
                accepted += 1
            else:
                # Reject and try a smaller step; the iterate never worsens.
                step_size /= 2.0
                rejected += 1
                if step_size < 1e-6 * max(config.epsilon, 1e-6):
                    break

        return AttackResult(delta.detach(), compromised, target, history, accepted, rejected)
    finally:
        for parameter, flag in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(flag)
        model.train(was_training)
