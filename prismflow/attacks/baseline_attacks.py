"""Standard PGD on each compromised view INDEPENDENTLY: the literature baseline.

Each compromised view is attacked on its own, maximising log b_v[target] for
that view alone, with its own step size. No term couples the views, so nothing
encourages the attacker's views to agree with each other. This is the
comparison point for `chorus.py`.

The Chorus objective at beta = 0 separates per view, so chorus(beta=0) and this
baseline optimise the same thing by different bookkeeping (jointly stepped
versus one view at a time). Agreement between them is a useful check; any gap
at beta > 0 is the agreement term doing work.
"""

from __future__ import annotations

import torch

from prismflow.attacks.chorus import AttackResult, ChorusConfig, choose_targets, chorus_objective, select_views


def pgd_attack(
    model,
    views: torch.Tensor,
    view_mask: torch.Tensor,
    config: ChorusConfig,
    target: torch.Tensor | None = None,
) -> AttackResult:
    """Independent targeted PGD per compromised view. `config.beta` is ignored."""
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

        accepted = rejected = 0
        for view in compromised:
            mask = torch.zeros_like(views)
            mask[:, view, :] = 1.0
            step_size = config.resolved_step_size()

            # Only this view's own term: beta plays no part, by construction.
            def value_of(current):
                return chorus_objective(model, views + current, view_mask, (view,), target, beta=0.0)

            with torch.no_grad():
                best = value_of(delta)

            for _ in range(config.steps):
                candidate = delta.clone().requires_grad_(True)
                objective = value_of(candidate)
                (gradient,) = torch.autograd.grad(objective, candidate)

                with torch.no_grad():
                    stepped = delta + step_size * gradient.sign() * mask
                    stepped = stepped.clamp(-config.epsilon, config.epsilon)
                    stepped = torch.where(mask.bool(), stepped, delta)
                    value = value_of(stepped)

                if value > best:
                    delta, best = stepped, value
                    accepted += 1
                else:
                    step_size /= 2.0
                    rejected += 1
                    if step_size < 1e-6 * max(config.epsilon, 1e-6):
                        break

        # History is per view here, so a single monotone trace is not meaningful;
        # the final objective is reported instead.
        with torch.no_grad():
            final = chorus_objective(model, views + delta, view_mask, compromised, target, beta=0.0)
        return AttackResult(delta.detach(), compromised, target, [float(-final)], accepted, rejected)
    finally:
        for parameter, flag in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(flag)
        model.train(was_training)
