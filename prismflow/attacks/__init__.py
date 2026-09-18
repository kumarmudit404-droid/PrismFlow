"""Attack generators for compromised / colluding views (Part 09).

Controlled research simulation: these run against this project's own models on
synthetic data, inside this repository.
"""

from prismflow.attacks.baseline_attacks import pgd_attack
from prismflow.attacks.chorus import (
    TARGET_STRATEGIES,
    AttackResult,
    ChorusConfig,
    choose_targets,
    chorus_attack,
    chorus_objective,
    select_views,
)

__all__ = [
    "TARGET_STRATEGIES",
    "AttackResult",
    "ChorusConfig",
    "choose_targets",
    "chorus_attack",
    "chorus_objective",
    "pgd_attack",
    "select_views",
]
