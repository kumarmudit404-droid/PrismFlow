"""L25: one scaler and one trainer, pinned to the two copies that exist.

`prismflow/training/` removes the `FEATURES[source]` lookup that made Part
28's scaler and trainer AIT-only. These tests are the evidence that removing
it changed nothing: the agnostic scaler must equal BOTH runners' copies bit
for bit, and the agnostic trainer must reproduce a committed set of test
probabilities exactly.

The reproduction test trains a real model and costs about 3.5 minutes. It is
on by default, because it is the only thing that makes the L25 claim
checkable; `PRISMFLOW_SKIP_SLOW=1` skips it.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from prismflow.training import apply_scaler, fit_scaler, train  # noqa: E402

# The AIT dataset builder these tests need is already written in the Part 28
# test module, where `apply_scaler` rebuilding through `AITSamples` is the
# reason it is a real dataset and not a stub. Importing it beats a second copy.
from tests.unit.test_ait_clean import _dataset as ait_dataset  # noqa: E402

LANL_SAMPLES = ROOT / "results/lanl/samples.npz"
LANL_SCORES = ROOT / "results/lanl/scores/with_auth_naive_seed0.npz"
EXECUTION_MODE = ROOT / "results/lanl/execution_mode.json"

needs_lanl = pytest.mark.skipif(
    not LANL_SAMPLES.is_file(), reason="Part 30 LANL samples are not present"
)
needs_lanl_scores = pytest.mark.skipif(
    not (LANL_SCORES.is_file() and EXECUTION_MODE.is_file()),
    reason="Part 30 committed scores or execution_mode.json are not present",
)
skip_slow = pytest.mark.skipif(
    bool(os.environ.get("PRISMFLOW_SKIP_SLOW")), reason="PRISMFLOW_SKIP_SLOW is set"
)


# ---------------------------------------------------------------------------
# The argument contract: the lookup is gone, so the widths must arrive
# ---------------------------------------------------------------------------


def test_native_widths_come_from_the_caller_not_a_source_table():
    """The whole point of L25: no view name appears anywhere in the call."""
    views = np.zeros((4, 2, 10), dtype=np.float32)
    views[:, 0, 0] = [10.0, 12.0, 0.0, 0.0]
    mask = np.array([[True, True], [True, True], [False, True], [False, True]])
    scaler = fit_scaler(views, mask, [3, 7], epsilon=1e-8)
    assert scaler["native_dims"] == [3, 7]
    # Fitted on the two present rows only, not pulled toward zero by the two
    # absent ones -- the property Part 28's docstring claims.
    assert scaler["mean"][0, 0] == pytest.approx(11.0)
    assert scaler["present_rows_per_view"] == [2, 4]


@pytest.mark.parametrize("dims", [[3], [3, 7, 5], [3, 99]])
def test_fit_scaler_refuses_a_width_list_that_cannot_be_right(dims):
    """A wrong width silently changes what the model can see, so it raises.

    Too few or too many entries for the view count, or a native width past the
    padded width, are all caller errors that no downstream code would notice.
    """
    views = np.zeros((4, 2, 10), dtype=np.float32)
    mask = np.ones((4, 2), dtype=bool)
    with pytest.raises(ValueError):
        fit_scaler(views, mask, dims, epsilon=1e-8)


def test_padded_columns_and_absent_rows_stay_zero():
    rng = np.random.default_rng(0)
    views = rng.normal(5.0, 2.0, size=(20, 1, 10)).astype(np.float32)
    views[:, 0, 8:] = 0.0
    mask = np.ones((20, 1), dtype=bool)
    mask[:5, 0] = False
    scaler = fit_scaler(views, mask, [8], epsilon=1e-8)
    assert np.all(scaler["mean"][0, 8:] == 0.0)
    assert np.all(scaler["std"][0, 8:] == 1.0)

    scaled = apply_scaler(views, mask, scaler)
    assert np.all(scaled[:5, 0, :] == 0.0), "absent view rows must stay zero"
    assert np.all(scaled[:, 0, 8:] == 0.0), "padded columns must stay zero"
    assert scaled[5:, 0, :8].mean(axis=0) == pytest.approx(np.zeros(8), abs=1e-5)


def test_a_constant_feature_is_not_divided_by_its_zero_std():
    views = np.full((6, 1, 10), 3.0, dtype=np.float32)
    mask = np.ones((6, 1), dtype=bool)
    scaler = fit_scaler(views, mask, [10], epsilon=1e-8)
    assert np.all(np.isfinite(apply_scaler(views, mask, scaler)))


# ---------------------------------------------------------------------------
# Equality with the two copies that already exist
# ---------------------------------------------------------------------------


def _lanl_with_auth():
    """The `with_auth` arm exactly as `run_lanl.arm_dataset` builds it."""
    from experiments.lanl.run_lanl import CONFIG_PATH, RESULTS_DIR, split_indices
    from prismflow.data.lanl_cyber1 import (
        FEATURES,
        LANLMultiViewDataset,
        load_lanl_config,
        load_samples,
        subset,
    )

    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    data_cfg = load_lanl_config(cfg["data"]["config"])
    arm = next(a for a in cfg["view_configurations"] if a["name"] == "with_auth")
    data = subset(
        LANLMultiViewDataset(load_samples(RESULTS_DIR / "samples.npz", data_cfg)),
        tuple(arm["views"]),
    )
    native = [len(FEATURES[s]) for s in data.sources]
    return cfg, data, split_indices(data), native


@needs_lanl
def test_scaler_equals_run_lanl_mirror_bit_for_bit():
    from experiments.lanl.run_lanl import apply_scaler as lanl_apply
    from experiments.lanl.run_lanl import fit_scaler as lanl_fit

    cfg, data, indices, native = _lanl_with_auth()
    mirror = lanl_fit(data, indices["train"], cfg)
    mine = fit_scaler(
        data.views.numpy()[indices["train"]],
        data.view_mask.numpy()[indices["train"]],
        native,
        cfg["scaling"]["epsilon"],
    )
    # The widths the caller passed in must be the ones run_lanl looked up.
    assert mine["native_dims"] == mirror["native_dims"] == [8, 5, 7, 2]
    for field in ("mean", "std"):
        assert np.array_equal(mine[field], mirror[field]), field
    assert mine["fit_rows"] == mirror["fit_rows"]
    assert mine["present_rows_per_view"] == mirror["present_rows_per_view"]

    scaled_mirror = lanl_apply(data, mirror).views.numpy()
    scaled_mine = apply_scaler(data.views.numpy(), data.view_mask.numpy(), mine)
    assert np.array_equal(scaled_mine, scaled_mirror)


def test_scaler_equals_part28_fit_scaler_bit_for_bit():
    """Part 28's AIT widths, on built rows so no download is required.

    `run_ait_clean.fit_scaler` reads the widths through `ait_lds.FEATURES`;
    this passes the same widths in. Equal output is what would let Part 28
    call the agnostic one without changing a number.
    """
    from experiments.ait_clean.run_ait_clean import CONFIG_PATH as AIT_CONFIG
    from experiments.ait_clean.run_ait_clean import apply_scaler as ait_apply
    from experiments.ait_clean.run_ait_clean import fit_scaler as ait_fit
    from prismflow.data.ait_lds import FEATURE_DIM, FEATURES

    sources = ("apache", "auth", "dns", "audit", "suricata")
    native = [len(FEATURES[s]) for s in sources]
    assert native == [9, 8, 9, 9, 10], "AIT widths changed; re-read this test"

    rng = np.random.default_rng(28)
    n = 40
    views = rng.normal(3.0, 7.0, size=(n, len(sources), FEATURE_DIM)).astype(np.float32)
    for v, width in enumerate(native):
        views[:, v, width:] = 0.0
    mask = rng.random((n, len(sources))) > 0.3
    mask[:, 0] = True                       # keep one view fully present
    views[~mask] = 0.0

    cfg = yaml.safe_load(AIT_CONFIG.read_text())
    dataset = ait_dataset(views, mask, sources)
    part28 = ait_fit(dataset, np.arange(n), cfg)
    mine = fit_scaler(views, mask, native, cfg["scaling"]["epsilon"])

    assert mine["native_dims"] == part28["native_dims"] == native
    for field in ("mean", "std"):
        assert np.array_equal(mine[field], part28[field]), field
    assert mine["present_rows_per_view"] == part28["present_rows_per_view"]
    assert np.array_equal(
        apply_scaler(views, mask, mine), ait_apply(dataset, part28).views.numpy()
    )


# ---------------------------------------------------------------------------
# The trainer, against committed probabilities
# ---------------------------------------------------------------------------


@needs_lanl
@needs_lanl_scores
@skip_slow
def test_trainer_reproduces_committed_lanl_probabilities():
    """One training unit: LANL `with_auth`, naive, seed 0.

    Run at the thread count `results/lanl/execution_mode.json` records for the
    run that wrote `results/lanl/scores/`. L28 says the fitted weights depend
    on that count, so any other value is a different experiment and a failure
    here would not mean what it looks like. On a mismatch this reports the
    largest difference and the thread count rather than widening a tolerance.
    """
    from experiments.lanl.run_lanl import per_sample_scores
    from prismflow.data.lanl_cyber1 import LANLMultiViewDataset

    mode = json.loads(EXECUTION_MODE.read_text(encoding="utf-8"))
    threads = int(mode["threads_per_process"])
    assert threads == 1, (
        "execution_mode.json records %d threads per process; the reproduction "
        "is pinned to the recorded count and must be re-read, not relaxed, if "
        "that changes" % threads
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        cfg, data, indices, native = _lanl_with_auth()
        scaler = fit_scaler(
            data.views.numpy()[indices["train"]],
            data.view_mask.numpy()[indices["train"]],
            native,
            cfg["scaling"]["epsilon"],
        )
        scaled = LANLMultiViewDataset(
            dataclasses.replace(
                data.samples,
                views=apply_scaler(data.views.numpy(), data.view_mask.numpy(), scaler),
            )
        )
        model = train(
            scaled,
            indices["train"],
            native,
            data.n_classes,
            seed=0,
            use_discount=False,
            model_cfg=cfg["model"],
            training_cfg=cfg["training"],
        )
        scores, _ = per_sample_scores(
            model, scaled, indices["test"], cfg["evaluation"]["batch_size"]
        )
    finally:
        torch.set_num_threads(previous)

    reference = np.load(LANL_SCORES, allow_pickle=True)
    assert np.array_equal(scores["label"], reference["label"])
    assert np.array_equal(scores["window_start"], reference["window_start"])
    for field in ("prob_attack", "vacuity"):
        largest = float(np.abs(scores[field] - reference[field]).max())
        assert np.array_equal(scores[field], reference[field]), (
            "%s differs from the committed run: largest absolute difference "
            "%.6e over %d test samples, at %d torch thread(s)"
            % (field, largest, scores[field].size, threads)
        )
