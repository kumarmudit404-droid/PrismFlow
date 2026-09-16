import numpy as np
import pytest
import torch

from prismflow.data.corruption import add_noise, drop_views, duplicate_view
from prismflow.data.dataset import Batch, MultiViewDataset, Sample, split_indices
from prismflow.data.loaders import iter_batches, make_split_loaders
from prismflow.data.synthetic import SyntheticConfig


def _make_dataset(n_samples=200, n_views=4, d_view=8, seed=0):
    cfg = SyntheticConfig(n_samples=n_samples, n_views=n_views, d_view=d_view, seed=seed)
    return MultiViewDataset(cfg)


# --- Sample / Batch shapes ------------------------------------------------


def test_sample_shapes():
    dataset = _make_dataset(n_views=4, d_view=8)
    sample = dataset.get_sample(0)
    assert isinstance(sample, Sample)
    assert sample.views.shape == (4, 8)
    assert sample.view_mask.shape == (4,)
    assert sample.view_mask.dtype == torch.bool
    assert isinstance(sample.label, int)


def test_batch_shapes():
    dataset = _make_dataset(n_samples=50, n_views=4, d_view=8)
    batch = dataset.get_batch([0, 1, 2, 3, 4])
    assert isinstance(batch, Batch)
    assert batch.views.shape == (5, 4, 8)
    assert batch.view_mask.shape == (5, 4)
    assert batch.view_mask.dtype == torch.bool
    assert batch.labels.shape == (5,)
    assert batch.sample_ids.shape == (5,)
    assert len(batch) == 5


# --- deterministic, non-overlapping splits --------------------------------


def test_split_indices_deterministic():
    s1 = split_indices(1000, seed=7)
    s2 = split_indices(1000, seed=7)
    for key in ("train", "val", "test"):
        assert np.array_equal(s1[key], s2[key])


def test_split_indices_different_seed_differs():
    s1 = split_indices(1000, seed=1)
    s2 = split_indices(1000, seed=2)
    assert not np.array_equal(s1["train"], s2["train"])


def test_split_indices_no_overlap_and_full_coverage():
    n = 997  # deliberately not evenly divisible
    splits = split_indices(n, seed=3)
    train, val, test = splits["train"], splits["val"], splits["test"]

    assert len(set(train) & set(val)) == 0
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0

    all_idx = np.concatenate([train, val, test])
    assert set(all_idx.tolist()) == set(range(n))
    assert len(all_idx) == n


def test_split_indices_bad_fractions_raise():
    with pytest.raises(ValueError):
        split_indices(100, seed=0, train_frac=0.5, val_frac=0.3, test_frac=0.3)


# --- loaders ---------------------------------------------------------------


def test_iter_batches_covers_all_indices_in_order():
    dataset = _make_dataset(n_samples=23)
    indices = np.arange(len(dataset))
    batches = list(iter_batches(dataset, indices, batch_size=5))
    total = sum(len(b) for b in batches)
    assert total == 23
    assert len(batches) == 5  # 4 full + 1 remainder
    assert len(batches[-1]) == 3

    seen = torch.cat([b.sample_ids for b in batches])
    assert torch.equal(torch.sort(seen).values, torch.arange(23))


def test_iter_batches_shuffle_requires_seed():
    dataset = _make_dataset(n_samples=10)
    with pytest.raises(ValueError):
        list(iter_batches(dataset, np.arange(10), batch_size=4, shuffle=True))


def test_iter_batches_shuffle_deterministic():
    dataset = _make_dataset(n_samples=10)
    order1 = torch.cat(
        [b.sample_ids for b in iter_batches(dataset, np.arange(10), batch_size=4, shuffle=True, seed=5)]
    )
    order2 = torch.cat(
        [b.sample_ids for b in iter_batches(dataset, np.arange(10), batch_size=4, shuffle=True, seed=5)]
    )
    assert torch.equal(order1, order2)


def test_make_split_loaders_partitions_dataset():
    dataset = _make_dataset(n_samples=100)
    splits = split_indices(len(dataset), seed=0)
    loaders = make_split_loaders(dataset, splits, batch_size=16, seed=0)

    for name in ("train", "val", "test"):
        total = sum(len(b) for b in loaders[name]())
        assert total == len(splits[name])


# --- corruption utilities ---------------------------------------------------


def test_duplicate_view_produces_identical_copies():
    dataset = _make_dataset(n_samples=10, n_views=4)
    batch = dataset.get_batch(range(10))
    corrupted = duplicate_view(batch, source_idx=1, k=3)

    assert corrupted.views.shape[1] == 4 + 3
    source = batch.views[:, 1, :]
    for offset in range(3):
        assert torch.equal(corrupted.views[:, 4 + offset, :], source)
    assert corrupted.view_mask.shape[1] == 4 + 3


def test_duplicate_view_zero_k_is_a_copy_not_a_mutation():
    dataset = _make_dataset(n_samples=5)
    batch = dataset.get_batch(range(5))
    result = duplicate_view(batch, source_idx=0, k=0)
    assert torch.equal(result.views, batch.views)
    result.views[0, 0, 0] = 999.0
    assert not torch.equal(result.views, batch.views)


def test_duplicate_view_invalid_source_idx_raises():
    dataset = _make_dataset(n_samples=5, n_views=3)
    batch = dataset.get_batch(range(5))
    with pytest.raises(ValueError):
        duplicate_view(batch, source_idx=5, k=1)


def test_drop_views_seed_reproducible():
    dataset = _make_dataset(n_samples=50, n_views=4)
    batch = dataset.get_batch(range(50))
    a = drop_views(batch, rate=0.3, seed=11)
    b = drop_views(batch, rate=0.3, seed=11)
    assert torch.equal(a.view_mask, b.view_mask)


def test_drop_views_different_seed_differs():
    dataset = _make_dataset(n_samples=50, n_views=4)
    batch = dataset.get_batch(range(50))
    a = drop_views(batch, rate=0.3, seed=1)
    b = drop_views(batch, rate=0.3, seed=2)
    assert not torch.equal(a.view_mask, b.view_mask)


def test_drop_views_zero_rate_keeps_all_true():
    dataset = _make_dataset(n_samples=20, n_views=4)
    batch = dataset.get_batch(range(20))
    result = drop_views(batch, rate=0.0, seed=0)
    assert result.view_mask.all()


def test_drop_views_full_rate_drops_all():
    dataset = _make_dataset(n_samples=20, n_views=4)
    batch = dataset.get_batch(range(20))
    result = drop_views(batch, rate=1.0, seed=0)
    assert not result.view_mask.any()


def test_add_noise_only_affects_targeted_view():
    dataset = _make_dataset(n_samples=20, n_views=4)
    batch = dataset.get_batch(range(20))
    result = add_noise(batch, view_idx=2, sigma=1.0, seed=0)

    for v in range(4):
        if v == 2:
            assert not torch.equal(result.views[:, v, :], batch.views[:, v, :])
        else:
            assert torch.equal(result.views[:, v, :], batch.views[:, v, :])


def test_add_noise_seed_reproducible():
    dataset = _make_dataset(n_samples=20, n_views=4)
    batch = dataset.get_batch(range(20))
    a = add_noise(batch, view_idx=0, sigma=0.5, seed=42)
    b = add_noise(batch, view_idx=0, sigma=0.5, seed=42)
    assert torch.equal(a.views, b.views)


def test_add_noise_zero_sigma_is_noop():
    dataset = _make_dataset(n_samples=20, n_views=4)
    batch = dataset.get_batch(range(20))
    result = add_noise(batch, view_idx=0, sigma=0.0, seed=0)
    assert torch.equal(result.views, batch.views)
