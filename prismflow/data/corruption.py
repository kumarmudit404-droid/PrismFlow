"""Corruption utilities for robustness experiments (used by Parts 06, 08, 09).

Every function returns a new Batch rather than mutating its input in place.
All randomness is seed-reproducible.
"""

from __future__ import annotations

import numpy as np
import torch

from prismflow.data.dataset import Batch


def duplicate_view(batch: Batch, source_idx: int, k: int) -> Batch:
    """Return a new Batch with k exact copies of view `source_idx` appended.

    Used to simulate colluding/duplicated views: the appended views are
    bit-identical to the source view (and its mask).
    """
    n_views = batch.views.shape[1]
    if not (0 <= source_idx < n_views):
        raise ValueError(f"source_idx {source_idx} out of range for {n_views} views")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")

    if k == 0:
        return Batch(
            views=batch.views.clone(),
            view_mask=batch.view_mask.clone(),
            labels=batch.labels.clone(),
            sample_ids=batch.sample_ids.clone(),
        )

    dup_views = batch.views[:, source_idx : source_idx + 1, :].repeat(1, k, 1)
    dup_mask = batch.view_mask[:, source_idx : source_idx + 1].repeat(1, k)

    return Batch(
        views=torch.cat([batch.views, dup_views], dim=1),
        view_mask=torch.cat([batch.view_mask, dup_mask], dim=1),
        labels=batch.labels.clone(),
        sample_ids=batch.sample_ids.clone(),
    )


def drop_views(batch: Batch, rate: float, seed: int) -> Batch:
    """Return a new Batch with each (sample, view) mask entry independently
    set to False with probability `rate`, deterministic given `seed`.
    """
    if not (0.0 <= rate <= 1.0):
        raise ValueError(f"rate must be in [0, 1], got {rate}")

    rng = np.random.default_rng(seed)
    drop = rng.random(batch.view_mask.shape) < rate

    new_mask = batch.view_mask.clone()
    new_mask[torch.as_tensor(drop)] = False

    return Batch(
        views=batch.views.clone(),
        view_mask=new_mask,
        labels=batch.labels.clone(),
        sample_ids=batch.sample_ids.clone(),
    )


def add_noise(batch: Batch, view_idx: int, sigma: float, seed: int = 0) -> Batch:
    """Return a new Batch with N(0, sigma^2) noise added to one view,
    deterministic given `seed`. Other views are untouched.
    """
    n_views = batch.views.shape[1]
    if not (0 <= view_idx < n_views):
        raise ValueError(f"view_idx {view_idx} out of range for {n_views} views")
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")

    generator = torch.Generator().manual_seed(seed)
    noise = torch.randn(batch.views[:, view_idx, :].shape, generator=generator) * sigma

    new_views = batch.views.clone()
    new_views[:, view_idx, :] = new_views[:, view_idx, :] + noise

    return Batch(
        views=new_views,
        view_mask=batch.view_mask.clone(),
        labels=batch.labels.clone(),
        sample_ids=batch.sample_ids.clone(),
    )
