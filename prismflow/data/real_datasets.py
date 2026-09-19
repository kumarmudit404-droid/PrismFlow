"""Real multi-view benchmarks behind the synthetic dataset interface (Part 15).

`RealMultiViewDataset` is a drop-in for `prismflow.data.dataset.MultiViewDataset`:
same attributes, same `get_sample` / `get_batch` return types, same
`[B, V, D]` / `[B, V]` / `[B]` shapes. Every loader, corruption utility and
evaluation path in the project therefore works on real data unchanged --
`tests/unit/test_real_datasets.py` asserts the schema field by field against
the synthetic loader rather than against a written description of it.

WHAT IS AND IS NOT AVAILABLE HERE
---------------------------------
`rho_matrix` is None, always. On synthetic data it is the ground-truth
cross-view correlation the generator was given, and Part 05 validated ENIV
against it. Real views have no such quantity: there is no generative
parameter to recover. This attribute is deliberately None rather than an
estimate, so that any code reaching for ground truth on real data fails
loudly instead of silently consuming a measurement as if it were a target.
See `docs/DATASETS.md`.

HETEROGENEOUS VIEW DIMENSIONS
-----------------------------
Real views rarely share a feature dimension (Handwritten: 76, 216, 64, 240,
47, 6). The contract's tensor shape is `[B, V, D]` with a single D, and
`MultiViewEncoder` already handles this: each view's `EncoderConfig` declares
its own `input_dim <= D` and the encoder slices `views[..., v, :input_dim]`.
So views are right-padded to D = max_v d_v and every padded column is
unreachable by construction -- no encoder reads it, no gradient flows to it,
and the Chorus/PGD `sign()` step leaves it at exactly zero. Use
`view_configs()` to build the matching encoder configs; building them by hand
with a uniform `input_dim` would feed one view's features to another's
encoder.

PREPROCESSING, AND WHY IT IS FITTED ON TRAIN ONLY
-------------------------------------------------
The six Handwritten views span wildly different scales (morphological
features reach 1.8e4, Fourier coefficients sit below 1). Without
standardisation the attack budget epsilon, which is an L-infinity bound in
feature units, would mean something different in every view, and the encoders
would be initialised far outside their useful input range. Features are
therefore z-scored per view. The mean and standard deviation come from the
TRAINING indices only -- `build_real_dataset` splits before it scales -- so
no test-split statistic enters training. Zero-variance columns are left at
zero rather than divided by ~0.

SEEDS MEAN SOMETHING DIFFERENT HERE
-----------------------------------
On synthetic data the seed redraws the dataset and the split is held fixed
(`TrainConfig.split_seed`). Real data cannot be redrawn, so here the seed
drives the train/val/test partition instead, as well as model init and batch
order. Five seeds are therefore five resamplings of the same 2000 patterns,
not five datasets; the spread they report is split-and-init variance, which
is narrower than the synthetic parts' data variance. This is stated in
`docs/DATASETS.md` and must not be presented as equivalent to it.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from prismflow.data.dataset import Batch, Sample, split_indices
from prismflow.models.encoders import EncoderConfig

DEFAULT_ROOT = Path("data/raw")
DOWNLOAD_TIMEOUT_S = 600


@dataclass(frozen=True)
class ViewSpec:
    """One view of a real benchmark: its file, its native width, its digest."""

    name: str
    filename: str
    dim: int
    sha256: str
    description: str


@dataclass(frozen=True)
class RealDatasetSpec:
    """Everything needed to fetch, verify, parse and label one benchmark."""

    name: str
    description: str
    homepage: str
    base_url: str
    views: tuple[ViewSpec, ...]
    n_samples: int
    n_classes: int
    samples_per_class: int
    citation: str
    extra_files: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    @property
    def n_views(self) -> int:
        return len(self.views)

    @property
    def view_dims(self) -> tuple[int, ...]:
        return tuple(v.dim for v in self.views)

    @property
    def max_dim(self) -> int:
        return max(self.view_dims)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Handwritten / "Multiple Features" (UCI id 72). Six feature sets extracted
# from the same 2000 binary images of handwritten digits 0-9, 200 per class,
# stored in class-block order. The canonical 6-view benchmark of the
# multi-view literature, and the Part 15 brief's first preference: six views
# and 2000 samples fit a full sweep in a day.
#
# The digests below are of the files inside the UCI distribution
# `multiple+features.zip`; they are verified on every load, so a truncated or
# substituted download is an error rather than a silently different result.
HANDWRITTEN = RealDatasetSpec(
    name="handwritten",
    description=(
        "UCI Multiple Features (mfeat): 2000 handwritten digits, 10 classes, "
        "six pre-extracted feature sets of the same images."
    ),
    homepage="https://archive.ics.uci.edu/dataset/72/multiple+features",
    base_url="https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/",
    views=(
        ViewSpec("fou", "mfeat-fou", 76,
                 "d2bed48a4d302efe77bd80d52e16a211e99efe6fee9baf1c28dcc199c43c50ad",
                 "Fourier coefficients of the character shapes"),
        ViewSpec("fac", "mfeat-fac", 216,
                 "83be9bc9fd50c6dd0b0532f6b6101956f1cf51c71b33f22d3f0cd2b5befb31b5",
                 "profile correlations"),
        ViewSpec("kar", "mfeat-kar", 64,
                 "db832668544497d0c06a2c1d1bfa0e17ba672812b7cfa6ed19f0941e19e1d7fa",
                 "Karhunen-Loeve coefficients"),
        ViewSpec("pix", "mfeat-pix", 240,
                 "70a1cd033add46614464a8740ddc23c3693765985bde52ccb6b702bff23b64f1",
                 "pixel averages in 2x3 windows"),
        ViewSpec("zer", "mfeat-zer", 47,
                 "d4a0e302f2f61e823b906f78818c7c0ef078c0bbc23022affd6607d60ecfe4f3",
                 "Zernike moments"),
        ViewSpec("mor", "mfeat-mor", 6,
                 "8767b326d874f096de06d525a2df4e610f4e9dfe91e29c250d8a3054907e3b61",
                 "morphological features"),
    ),
    n_samples=2000,
    n_classes=10,
    samples_per_class=200,
    citation=(
        "R.P.W. Duin, Multiple Features Data Set, UCI Machine Learning "
        "Repository. Used in M. van Breukelen et al., Handwritten digit "
        "recognition by combined classifiers, Kybernetika 34(4), 1998."
    ),
    extra_files=(("mfeat.info", "60362c164eda1bf6e983b225b11aca69b66def0c7320392f92354f068203d189"),),
)

REGISTRY: dict[str, RealDatasetSpec] = {HANDWRITTEN.name: HANDWRITTEN}


def get_spec(name: str) -> RealDatasetSpec:
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"unknown real dataset {name!r}; known: {sorted(REGISTRY)}") from exc


# ---------------------------------------------------------------------------
# Fetch and verify
# ---------------------------------------------------------------------------


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ensure_downloaded(spec: RealDatasetSpec, root: Path | str = DEFAULT_ROOT) -> Path:
    """Fetch any missing file of `spec` into `<root>/<spec.name>/` and verify it.

    A file already on disk with the right digest is never re-fetched, so runs
    are offline after the first. A file with the WRONG digest is an error, not
    something to overwrite: a mismatch means the upstream distribution changed
    under a result that has already been reported, and that has to be seen.
    """
    directory = Path(root) / spec.name
    directory.mkdir(parents=True, exist_ok=True)

    wanted = [(v.filename, v.sha256) for v in spec.views] + list(spec.extra_files)
    for filename, expected in wanted:
        path = directory / filename
        if path.exists():
            found = _digest(path)
            if found != expected:
                raise RuntimeError(
                    f"{path} has sha256 {found}, expected {expected}. Delete it to re-fetch, "
                    "but check first: the upstream file may have changed."
                )
            continue
        url = spec.base_url + filename
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as response:
            payload = response.read()
        found = hashlib.sha256(payload).hexdigest()
        if found != expected:
            raise RuntimeError(f"downloaded {url} has sha256 {found}, expected {expected}")
        path.write_bytes(payload)
    return directory


def load_raw(spec: RealDatasetSpec, root: Path | str = DEFAULT_ROOT):
    """Return (list of [N, d_v] float64 view matrices, [N] int64 labels).

    Views are returned at their native widths and native scales; padding and
    standardisation belong to `build_real_dataset`, which knows the split.
    """
    directory = ensure_downloaded(spec, root)

    views = []
    for view in spec.views:
        matrix = np.loadtxt(directory / view.filename, dtype=np.float64)
        if matrix.shape != (spec.n_samples, view.dim):
            raise RuntimeError(
                f"{view.filename} parsed to {matrix.shape}, expected "
                f"({spec.n_samples}, {view.dim})"
            )
        views.append(matrix)

    # Class-block order: the first `samples_per_class` rows are class 0, and
    # so on. Asserted against the declared totals rather than assumed.
    if spec.samples_per_class * spec.n_classes != spec.n_samples:
        raise RuntimeError(f"{spec.name}: class blocks do not tile n_samples")
    labels = np.arange(spec.n_samples, dtype=np.int64) // spec.samples_per_class
    return views, labels


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class RealMultiViewDataset:
    """In-memory real multi-view dataset. Interface-identical to MultiViewDataset.

    Construct it through `build_real_dataset`, which owns the split-then-scale
    ordering. Constructing it directly with already-prepared arrays is
    supported for tests.
    """

    def __init__(
        self,
        views: np.ndarray,
        labels: np.ndarray,
        spec: RealDatasetSpec,
        metadata: dict[str, Any] | None = None,
    ):
        self.views = torch.as_tensor(views, dtype=torch.float32)  # [N, V, D]
        self.labels = torch.as_tensor(labels, dtype=torch.long)  # [N]
        # No ground-truth dependence exists for real views. See the module
        # docstring and docs/DATASETS.md: None, never an estimate.
        self.rho_matrix = None
        self.config = spec
        self.spec = spec
        self.view_dims = spec.view_dims
        self.n_classes = spec.n_classes
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


def standardise(views, train_idx: np.ndarray):
    """Z-score each view's native features using TRAIN rows only.

    Returns a new list; inputs are not mutated. Columns whose training
    standard deviation is below `tol` are centred and left unscaled -- a
    constant feature carries no information and dividing by its noise floor
    would manufacture a large spurious one.
    """
    tol = 1e-8
    scaled = []
    for matrix in views:
        mean = matrix[train_idx].mean(axis=0, keepdims=True)
        std = matrix[train_idx].std(axis=0, keepdims=True)
        safe = np.where(std < tol, 1.0, std)
        scaled.append((matrix - mean) / safe)
    return scaled


def pad_to_common_width(views, width: int | None = None) -> np.ndarray:
    """Stack [N, d_v] views into [N, V, D] by right-padding with zeros.

    D defaults to max_v d_v. Padded columns are never read: each view's
    encoder is built with `input_dim = d_v` (see `view_configs`).
    """
    n_samples = views[0].shape[0]
    width = width or max(m.shape[1] for m in views)
    stacked = np.zeros((n_samples, len(views), width), dtype=np.float64)
    for v, matrix in enumerate(views):
        if matrix.shape[0] != n_samples:
            raise ValueError(f"view {v} has {matrix.shape[0]} rows, expected {n_samples}")
        if matrix.shape[1] > width:
            raise ValueError(f"view {v} has width {matrix.shape[1]} > D={width}")
        stacked[:, v, : matrix.shape[1]] = matrix
    return stacked


def build_real_dataset(
    name: str,
    seed: int,
    root: Path | str = DEFAULT_ROOT,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
):
    """Return (RealMultiViewDataset, splits) -- the real-data analogue of
    `build_dataset(seed, data_cfg)` in the synthetic experiment scripts.

    The split is drawn FIRST and the scaler is fitted on its training indices
    only. `seed` drives the split here, unlike the synthetic path where it
    redraws the data and the split is fixed; see the module docstring.
    """
    spec = get_spec(name)
    raw, labels = load_raw(spec, root)
    splits = split_indices(
        spec.n_samples, seed=seed, train_frac=train_frac, val_frac=val_frac, test_frac=test_frac
    )
    stacked = pad_to_common_width(standardise(raw, splits["train"]))
    dataset = RealMultiViewDataset(
        stacked,
        labels,
        spec,
        metadata={"dataset": spec.name, "split_seed": seed, "source": spec.homepage},
    )
    return dataset, splits


# ---------------------------------------------------------------------------
# Encoder configuration
# ---------------------------------------------------------------------------


def view_configs(
    spec: RealDatasetSpec,
    hidden_dims=(64, 64),
    feature_dim: int = 32,
    dropout: float = 0.0,
    clone_source: int = 0,
    clone_k: int = 0,
) -> list[EncoderConfig]:
    """One EncoderConfig per view, each with that view's NATIVE input_dim.

    `clone_k > 0` appends k configs for copies of view `clone_source`, matching
    what `prismflow.data.corruption.duplicate_view` appends to the batch: a
    copy carries the source view's padded row, so its encoder must read the
    source view's width, not D.
    """
    configs = [
        EncoderConfig(
            input_dim=view.dim,
            hidden_dims=list(hidden_dims),
            feature_dim=feature_dim,
            dropout=dropout,
        )
        for view in spec.views
    ]
    if clone_k < 0:
        raise ValueError(f"clone_k must be >= 0, got {clone_k}")
    if clone_k and not 0 <= clone_source < spec.n_views:
        raise ValueError(f"clone_source {clone_source} out of range for {spec.n_views} views")
    for _ in range(clone_k):
        source = configs[clone_source]
        configs.append(
            EncoderConfig(
                input_dim=source.input_dim,
                hidden_dims=list(hidden_dims),
                feature_dim=feature_dim,
                dropout=dropout,
            )
        )
    return configs
