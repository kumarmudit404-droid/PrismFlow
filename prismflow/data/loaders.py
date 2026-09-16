"""Batch loading utilities built on top of MultiViewDataset."""

from __future__ import annotations

from typing import Iterator

import numpy as np

from prismflow.data.dataset import Batch, MultiViewDataset


def iter_batches(
    dataset: MultiViewDataset,
    indices,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    drop_last: bool = False,
) -> Iterator[Batch]:
    """Yield Batch objects covering `indices`.

    If shuffle=True, `seed` is required and the iteration order is
    deterministic given that seed.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")

    indices = np.asarray(indices)
    if shuffle:
        if seed is None:
            raise ValueError("seed is required when shuffle=True, for reproducibility")
        rng = np.random.default_rng(seed)
        indices = indices[rng.permutation(len(indices))]

    n = len(indices)
    n_batches = n // batch_size if drop_last else -(-n // batch_size)
    for b in range(n_batches):
        batch_idx = indices[b * batch_size : (b + 1) * batch_size]
        if len(batch_idx) == 0:
            continue
        yield dataset.get_batch(batch_idx)


def make_split_loaders(
    dataset: MultiViewDataset,
    splits: dict[str, np.ndarray],
    batch_size: int,
    seed: int = 0,
    shuffle_train: bool = True,
):
    """Return a dict {split_name: callable() -> Iterator[Batch]}.

    Each callable produces a fresh iterator over that split; the train split
    is shuffled deterministically (given `seed`) if `shuffle_train`, val/test
    are always iterated in index order.
    """
    loaders = {}
    for name, idx in splits.items():
        shuffle = shuffle_train and name == "train"
        loaders[name] = lambda idx=idx, shuffle=shuffle: iter_batches(
            dataset, idx, batch_size, shuffle=shuffle, seed=seed if shuffle else None
        )
    return loaders
