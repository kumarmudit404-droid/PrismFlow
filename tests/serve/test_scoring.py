"""What the scorer must reproduce, and what it must refuse to invent."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from serve.gate import gate_upload
from serve.scoring import (
    CALIBRATION_NOTE,
    DEFENCE_NOTE,
    ESTIMATOR_VALIDATION_NOTE,
    jsonable,
    score,
)


def gated(features, **extra):
    buf = io.BytesIO()
    np.savez(buf, features=features, **extra)
    return gate_upload(buf.getvalue(), "x.npz")


def test_independent_views_give_n_eff_close_to_v():
    rng = np.random.default_rng(0)
    out = score(gated(rng.normal(size=(600, 4, 8))))
    n_eff = out["eniv"]["eigen"]["effective_views"]
    assert out["eniv"]["eigen"]["nominal_views"] == 4
    # Not exactly 4: the eigenvalue form is biased DOWN at finite sample
    # size, which is documented in prismflow/eniv/eniv.py. The test asserts
    # the documented behaviour, not an idealised one.
    assert 3.8 < n_eff <= 4.0


def test_exact_clones_do_not_raise_the_effective_count():
    """The documented clone result: k copies of view 0 keep n_eff at 4.

    4 independent views plus k exact duplicates of the first. A count that
    rose with k would mean duplication was being read as fresh evidence,
    which is the failure the whole project exists to measure.
    """
    rng = np.random.default_rng(7)
    base = rng.normal(size=(600, 4, 8))
    seen = []
    for k in range(5):
        features = np.concatenate([base] + [base[:, 0:1, :]] * k, axis=1)
        out = score(gated(features))
        assert out["shape"]["views"] == 4 + k
        n_eff = out["eniv"]["eigen"]["effective_views"]
        assert 3.8 < n_eff <= 4.0, (k, n_eff)
        seen.append(n_eff)
    # Flat in k, to within the estimator's own noise.
    assert max(seen) - min(seen) < 0.05


def test_same_seed_gives_identical_json():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(300, 3, 5))
    evidence = np.abs(rng.normal(size=(300, 3, 2)))
    a = score(gated(features, evidence=evidence), seed=11)
    b = score(gated(features, evidence=evidence), seed=11)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_seed_is_reported_and_hashed_into_the_config():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(300, 3, 5))
    a = score(gated(features), seed=1)
    b = score(gated(features), seed=2)
    assert a["seed"] == 1 and b["seed"] == 2
    assert a["config_hash"] != b["config_hash"]


def test_the_three_caveats_are_present_verbatim():
    out = score(gated(np.random.default_rng(0).normal(size=(200, 2, 4))))
    assert out["calibration"] == CALIBRATION_NOTE == "uncalibrated on your data"
    assert out["defence"] == DEFENCE_NOTE
    assert "L3, L4, L5, L8" in out["defence"]
    assert out["estimator_validation"] == ESTIMATOR_VALIDATION_NOTE
    assert "L9" in out["estimator_validation"]


def test_fusion_and_suspicion_are_null_without_evidence_never_zero():
    """Absent is null with a reason. A zero here would read as a measurement."""
    out = score(gated(np.random.default_rng(0).normal(size=(200, 3, 4))))
    assert out["fused"] is None
    assert out["suspicion"] is None
    assert "not measured" in out["fused_absent_reason"]
    assert "not measured" in out["suspicion_absent_reason"]


def test_fusion_and_suspicion_appear_when_evidence_is_supplied():
    rng = np.random.default_rng(5)
    out = score(gated(rng.normal(size=(300, 3, 6)),
                      evidence=np.abs(rng.normal(size=(300, 3, 2)))))
    assert out["fused"]["n_classes"] == 2
    assert out["fused"]["mean_uncertainty"] is not None
    assert out["suspicion"]["mean_score"] is not None


def test_nan_serialises_to_null_and_never_to_zero():
    assert jsonable(float("nan")) is None
    assert jsonable(float("inf")) is None
    assert jsonable(np.array([1.0, np.nan])) == [1.0, None]
    assert jsonable(0.0) == 0.0          # a real zero survives as a zero


def test_dependence_matrix_is_square_and_json_safe():
    out = score(gated(np.random.default_rng(0).normal(size=(200, 4, 5))))
    matrix = out["dependence_matrix"]
    assert len(matrix) == 4 and all(len(row) == 4 for row in matrix)
    json.dumps(out)      # raises on NaN under the default allow_nan=False path
    assert json.dumps(out, allow_nan=False)


def test_conditioning_is_declared_as_unconditional():
    """The weaker quantity has to announce itself on every call."""
    out = score(gated(np.random.default_rng(0).normal(size=(200, 2, 4))))
    assert "ONE stratum" in out["conditioning"]
    assert out["config"]["conditioning_strata"] == 1
