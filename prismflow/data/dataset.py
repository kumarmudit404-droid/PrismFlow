"""Dataset layer: Sample/Batch containers and deterministic splits.

Tensor shape conventions (see docs/CONTRACT.md):
    views [B, V, D], view_mask [B, V] boolean, labels [B].
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from prismflow.data.synthetic import SyntheticConfig, generate_synthetic_dataset


@dataclass
class Sample:
    sample_id: int
    views: torch.Tensor  # [V, D]
    label: int
    view_mask: torch.Tensor  # [V] bool
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Batch:
    views: torch.Tensor  # [B, V, D]
    view_mask: torch.Tensor  # [B, V] bool
    labels: torch.Tensor  # [B]
    sample_ids: torch.Tensor  # [B]

    def __len__(self) -> int:
        return self.views.shape[0]


class MultiViewDataset:
    """In-memory multi-view dataset backed by the synthetic generator.

    Feature generation is deterministic given `config.seed`. Train/val/test
    partitioning is a separate, independently-seeded step (see
    `split_indices`), so the same generated features can be re-split without
    regenerating them.
    """

    def __init__(self, config: SyntheticConfig, metadata: dict[str, Any] | None = None):
        generated = generate_synthetic_dataset(config)
        self.views = torch.as_tensor(generated["views"], dtype=torch.float32)  # [N, V, D]
        self.labels = torch.as_tensor(generated["labels"], dtype=torch.long)  # [N]
        self.rho_matrix = generated["rho_matrix"]
        self.config = config
        self._metadata = metadata or {}

    def __len__(self) -> int:
        return self.views.shape[0]

    @property
    def n_views(self) -> int:
        return self.views.shape[1]

    def get_sample(self, index: int) -> Sample:
        return Sample(
            sample_id=index,
            views=self.views[index],
            label=int(self.labels[index]),
            view_mask=torch.ones(self.n_views, dtype=torch.bool),
            metadata=dict(self._metadata),
        )

    def get_batch(self, indices) -> Batch:
        idx = torch.as_tensor(np.asarray(indices), dtype=torch.long)
        return Batch(
            views=self.views[idx],
            view_mask=torch.ones(len(idx), self.n_views, dtype=torch.bool),
            labels=self.labels[idx],
            sample_ids=idx,
        )


def split_indices(
    n_samples: int,
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> dict[str, np.ndarray]:
    """Deterministic, non-overlapping train/val/test index split.

    The same (n_samples, seed, fractions) always yields the same partition;
    every index in [0, n_samples) appears in exactly one split (no label
    leakage across splits).
    """
    total = train_frac + val_frac + test_frac
    if not np.isclose(total, 1.0):
        raise ValueError(f"train/val/test fractions must sum to 1.0, got {total}")

    rng = np.random.default_rng(seed)
    permuted = rng.permutation(n_samples)

    n_train = int(round(n_samples * train_frac))
    n_val = int(round(n_samples * val_frac))

    train_idx = np.sort(permuted[:n_train])
    val_idx = np.sort(permuted[n_train : n_train + n_val])
    test_idx = np.sort(permuted[n_train + n_val :])

    return {"train": train_idx, "val": val_idx, "test": test_idx}
