"""The HTTP surface: status codes, the caveats, and CORS denying by default."""

from __future__ import annotations

import io

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from serve.schema import Settings  # noqa: E402


def client(monkeypatch=None, **env):
    """A fresh app with a fresh environment.

    `serve.app` reads settings at import, so the module is reloaded per test
    rather than mutated -- otherwise one test's ALLOWED_ORIGIN would leak
    into the next and a deny-by-default assertion could pass for the wrong
    reason.
    """
    import importlib
    import os
    import sys

    saved = {k: os.environ.get(k) for k in ("ALLOWED_ORIGIN", "SCORING_SEED")}
    for k in saved:
        os.environ.pop(k, None)
    os.environ.update(env)
    try:
        sys.modules.pop("serve.app", None)
        module = importlib.import_module("serve.app")
        return TestClient(module.app)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def npz_bytes(**arrays) -> bytes:
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def test_health_reports_limits_and_the_caveats():
    body = client().get("/health").json()
    assert body["status"] == "ok"
    assert body["limits"]["min_samples"] >= 1
    assert body["calibration"] == "uncalibrated on your data"
    assert "L3, L4, L5, L8" in body["defence"]
    assert "L9" in body["estimator_validation"]


def test_health_does_not_echo_the_allowed_origins():
    """An unauthenticated endpoint should not list who else is trusted."""
    body = client(ALLOWED_ORIGIN="https://example.invalid").get("/health").json()
    assert body["cors_configured"] is True
    assert "example.invalid" not in str(body)


def test_score_returns_the_three_literal_fields():
    features = np.random.default_rng(0).normal(size=(200, 4, 6))
    response = client().post(
        "/score", files={"file": ("x.npz", npz_bytes(features=features),
                                  "application/octet-stream")})
    assert response.status_code == 200
    body = response.json()
    assert body["calibration"] == "uncalibrated on your data"
    assert body["defence"] == (
        "not validated - see KNOWN_LIMITATIONS L3, L4, L5, L8")
    assert body["estimator_validation"] == (
        "ENIV is validated on synthetic data only - see KNOWN_LIMITATIONS L9")
    # L22 travels over HTTP too, not only out of score().
    assert "NOT comparable across different view counts" in (
        body["suspicion_comparability"])


def test_a_rejected_upload_is_400_with_a_named_rule():
    features = np.random.default_rng(0).normal(size=(200, 1, 6))
    response = client().post(
        "/score", files={"file": ("x.npz", npz_bytes(features=features),
                                  "application/octet-stream")})
    assert response.status_code == 400
    assert response.json()["error"] == "too_few_views"


def test_a_pickled_npy_is_refused_over_http():
    buf = io.BytesIO()
    np.save(buf, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    response = client().post(
        "/score", files={"file": ("evil.npy", buf.getvalue(),
                                  "application/octet-stream")})
    assert response.status_code == 400
    assert response.json()["error"] == "pickle_refused"


# --------------------------------------------------------------------- CORS

def test_cors_denies_every_origin_by_default():
    """Unset ALLOWED_ORIGIN must not mean "allow everyone"."""
    assert Settings.from_env({}).allowed_origins == ()
    response = client().get("/health", headers={"Origin": "https://evil.test"})
    assert "access-control-allow-origin" not in {
        k.lower() for k in response.headers}


def test_cors_allows_only_the_configured_origin():
    c = client(ALLOWED_ORIGIN="https://good.test")
    allowed = c.get("/health", headers={"Origin": "https://good.test"})
    assert allowed.headers.get("access-control-allow-origin") == "https://good.test"
    denied = c.get("/health", headers={"Origin": "https://evil.test"})
    assert denied.headers.get("access-control-allow-origin") != "https://evil.test"


def test_a_wildcard_in_the_variable_is_dropped_not_honoured():
    """A deployment must not be able to open itself up with one character."""
    assert Settings.from_env({"ALLOWED_ORIGIN": "*"}).allowed_origins == ()
    assert Settings.from_env(
        {"ALLOWED_ORIGIN": "*, https://good.test"}
    ).allowed_origins == ("https://good.test",)


def test_settings_survive_a_nonsense_seed():
    assert Settings.from_env({"SCORING_SEED": "not-a-number"}).seed == 0
