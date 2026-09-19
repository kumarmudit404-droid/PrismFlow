"""Part 15: the real loader must be indistinguishable from the synthetic one.

The point of these tests is narrow and load-bearing. Part 15's brief requires
the real dataset to sit behind the EXISTING interface so that no experiment
script is modified, and the only way to know that holds is to compare the two
loaders field by field rather than to assert it in a docstring. So the schema
tests below read `Batch`'s dataclass fields from the class itself: if anyone
adds a field to `Batch`, the comparison picks it up automatically instead of
passing against a hand-written list that has gone stale.

Tests that need the UCI files are skipped, with a reason, when those files are
not on disk -- `data/raw/` is gitignored, so a fresh clone has nothing until
`ensure_downloaded` has run once. The schema tests themselves need no
download: they build a RealMultiViewDataset from arrays directly.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from prismflow.data.corruption import add_noise, drop_views, duplicate_view
from prismflow.data.dataset import Batch, MultiViewDataset, Sample
from prismflow.data.loaders import iter_batches, make_split_loaders
from prismflow.data.real_datasets import (
    HANDWRITTEN,
    REGISTRY,
    RealMultiViewDataset,
    build_real_dataset,
    get_spec,
    pad_to_common_width,
    standardise,
    view_configs,
)
from prismflow.data.synthetic import SyntheticConfig

SPEC = HANDWRITTEN


def _synthetic(n_samples=40, n_views=6, d_view=12, seed=0):
    return MultiViewDataset(
        SyntheticConfig(n_samples=n_samples, n_views=n_views, d_view=d_view, seed=seed)
    )


def _real(n_samples=40):
    """A RealMultiViewDataset with the Handwritten spec's shape, no download.

    The values are arbitrary; every test here is about schema, dtype and
    shape, none about the numbers.
    """
    rng = np.random.default_rng(0)
    views = [rng.normal(size=(n_samples, view.dim)) for view in SPEC.views]
    labels = np.arange(n_samples, dtype=np.int64) % SPEC.n_classes
    return RealMultiViewDataset(pad_to_common_width(views), labels, SPEC)


def _files_available() -> bool:
    from prismflow.data.real_datasets import DEFAULT_ROOT

    directory = DEFAULT_ROOT / SPEC.name
    return all((directory / view.filename).exists() for view in SPEC.views)


needs_files = pytest.mark.skipif(
    not _files_available(),
    reason="UCI mfeat files not in data/raw/handwritten (gitignored); "
    "run experiments.real.run_real once to fetch them",
)


# --- THE SCHEMA TEST: real batches must match synthetic batches exactly -----


def test_batch_schema_matches_synthetic_loader_exactly():
    synthetic = _synthetic()
    real = _real()

    s_batch = synthetic.get_batch(range(8))
    r_batch = real.get_batch(range(8))

    assert type(s_batch) is Batch and type(r_batch) is Batch

    fields = [f.name for f in dataclasses.fields(Batch)]
    assert fields, "Batch has no fields; the comparison below would be vacuous"

    for name in fields:
        s_value = getattr(s_batch, name)
        r_value = getattr(r_batch, name)
        assert type(s_value) is type(r_value), f"{name}: {type(s_value)} vs {type(r_value)}"
        assert s_value.dtype == r_value.dtype, f"{name}: {s_value.dtype} vs {r_value.dtype}"
        assert s_value.ndim == r_value.ndim, f"{name}: {s_value.ndim} vs {r_value.ndim}"
        # Batch dimension agrees; feature width legitimately differs.
        assert s_value.shape[0] == r_value.shape[0] == 8, name

    assert len(s_batch) == len(r_batch) == 8


def test_dataset_interface_surface_matches_synthetic():
    """Every public attribute an experiment script reads exists on both."""
    synthetic = _synthetic()
    real = _real()
    for name in ("views", "labels", "rho_matrix", "config", "n_views", "get_sample", "get_batch"):
        assert hasattr(synthetic, name), f"synthetic loader lost {name}"
        assert hasattr(real, name), f"real loader is missing {name}"
    assert len(real) == 40
    assert real.n_views == SPEC.n_views


def test_sample_schema_matches_synthetic_loader_exactly():
    s_sample = _synthetic().get_sample(3)
    r_sample = _real().get_sample(3)
    assert type(s_sample) is Sample and type(r_sample) is Sample

    for name in [f.name for f in dataclasses.fields(Sample)]:
        s_value, r_value = getattr(s_sample, name), getattr(r_sample, name)
        assert type(s_value) is type(r_value), f"{name}: {type(s_value)} vs {type(r_value)}"
        if torch.is_tensor(s_value):
            assert s_value.dtype == r_value.dtype, name
            assert s_value.ndim == r_value.ndim, name


def test_batch_tensor_shapes_follow_the_contract():
    real = _real()
    batch = real.get_batch(range(5))
    assert batch.views.shape == (5, SPEC.n_views, SPEC.max_dim)
    assert batch.view_mask.shape == (5, SPEC.n_views)
    assert batch.view_mask.dtype == torch.bool
    assert batch.view_mask.all()
    assert batch.labels.shape == (5,)
    assert batch.sample_ids.shape == (5,)


def test_ground_truth_dependence_is_absent_not_estimated():
    """rho_matrix must be None on real data -- see docs/DATASETS.md."""
    assert _real().rho_matrix is None
    assert _synthetic().rho_matrix is not None


# --- the existing loaders and corruptions work unmodified ------------------


def test_iter_batches_covers_the_real_dataset():
    real = _real(n_samples=23)
    batches = list(iter_batches(real, np.arange(23), batch_size=5))
    assert sum(len(b) for b in batches) == 23
    seen = torch.cat([b.sample_ids for b in batches])
    assert torch.equal(torch.sort(seen).values, torch.arange(23))


def test_make_split_loaders_partitions_the_real_dataset():
    from prismflow.data.dataset import split_indices

    real = _real(n_samples=100)
    splits = split_indices(len(real), seed=0)
    loaders = make_split_loaders(real, splits, batch_size=16, seed=0)
    for name in ("train", "val", "test"):
        assert sum(len(b) for b in loaders[name]()) == len(splits[name])


def test_corruptions_apply_unmodified_to_real_batches():
    real = _real(n_samples=20)
    batch = real.get_batch(range(20))

    cloned = duplicate_view(batch, source_idx=0, k=2)
    assert cloned.views.shape[1] == SPEC.n_views + 2
    assert torch.equal(cloned.views[:, SPEC.n_views, :], batch.views[:, 0, :])

    dropped = drop_views(batch, rate=1.0, seed=0)
    assert not dropped.view_mask.any()

    noisy = add_noise(batch, view_idx=1, sigma=0.5, seed=0)
    assert not torch.equal(noisy.views[:, 1, :], batch.views[:, 1, :])
    assert torch.equal(noisy.views[:, 2, :], batch.views[:, 2, :])


# --- padding, scaling, encoder configs -------------------------------------


def test_padding_leaves_native_features_untouched_and_pads_with_zeros():
    rng = np.random.default_rng(1)
    views = [rng.normal(size=(7, view.dim)) for view in SPEC.views]
    stacked = pad_to_common_width(views)
    assert stacked.shape == (7, SPEC.n_views, SPEC.max_dim)
    for v, matrix in enumerate(views):
        assert np.allclose(stacked[:, v, : matrix.shape[1]], matrix)
        assert np.all(stacked[:, v, matrix.shape[1] :] == 0.0)


def test_view_configs_give_each_view_its_native_input_dim():
    configs = view_configs(SPEC)
    assert [c.input_dim for c in configs] == list(SPEC.view_dims)
    assert len({c.feature_dim for c in configs}) == 1  # MultiViewEncoder requires this

    cloned = view_configs(SPEC, clone_source=0, clone_k=2)
    assert len(cloned) == SPEC.n_views + 2
    assert [c.input_dim for c in cloned[SPEC.n_views :]] == [SPEC.view_dims[0]] * 2


def test_standardise_uses_train_rows_only():
    rng = np.random.default_rng(2)
    views = [rng.normal(loc=5.0, scale=3.0, size=(100, 4))]
    train_idx = np.arange(60)
    scaled = standardise(views, train_idx)[0]

    assert np.allclose(scaled[train_idx].mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(scaled[train_idx].std(axis=0), 1.0, atol=1e-8)
    # Held-out rows are transformed, not re-fitted: their mean is near but not
    # exactly zero. Exactly zero would mean test statistics leaked into the scaler.
    assert not np.allclose(scaled[60:].mean(axis=0), 0.0, atol=1e-12)


def test_standardise_survives_a_constant_column():
    views = [np.concatenate([np.ones((20, 1)), np.arange(20).reshape(-1, 1)], axis=1)]
    scaled = standardise(views, np.arange(10))[0]
    assert np.all(np.isfinite(scaled))
    assert np.all(scaled[:, 0] == 0.0)


def test_unknown_dataset_name_raises():
    with pytest.raises(ValueError):
        get_spec("not-a-dataset")


def test_registry_specs_are_internally_consistent():
    for name, spec in REGISTRY.items():
        assert spec.name == name
        assert spec.samples_per_class * spec.n_classes == spec.n_samples
        assert spec.max_dim == max(spec.view_dims)
        assert len({v.filename for v in spec.views}) == spec.n_views


# --- the real files, when present ------------------------------------------


@needs_files
def test_real_load_has_the_declared_shape_and_labels():
    dataset, splits = build_real_dataset(SPEC.name, seed=0)
    assert len(dataset) == SPEC.n_samples
    assert dataset.views.shape == (SPEC.n_samples, SPEC.n_views, SPEC.max_dim)
    assert dataset.labels.min() == 0 and dataset.labels.max() == SPEC.n_classes - 1
    assert torch.bincount(dataset.labels).tolist() == [SPEC.samples_per_class] * SPEC.n_classes
    assert sum(len(v) for v in splits.values()) == SPEC.n_samples


@needs_files
def test_real_split_varies_with_seed_and_is_deterministic():
    _, a = build_real_dataset(SPEC.name, seed=0)
    _, b = build_real_dataset(SPEC.name, seed=0)
    _, c = build_real_dataset(SPEC.name, seed=1)
    assert np.array_equal(a["train"], b["train"])
    assert not np.array_equal(a["train"], c["train"])


@needs_files
def test_real_features_are_standardised_on_train_only():
    dataset, splits = build_real_dataset(SPEC.name, seed=0)
    for v, width in enumerate(SPEC.view_dims):
        block = dataset.views[splits["train"]][:, v, :width].numpy()
        assert np.allclose(block.mean(axis=0), 0.0, atol=1e-4)
        assert np.allclose(block.std(axis=0), 1.0, atol=1e-3)
        padding = dataset.views[:, v, width:]
        assert torch.all(padding == 0.0)
