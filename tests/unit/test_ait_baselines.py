"""Part 28 addendum tests: the non-fusion learned baselines.

The rules that matter here are the ones that would make the baseline an unfair
comparison rather than a useful one: the same windows, the same scaler policy,
the same metric code, the two directions never pooled, and -- because these
rows are reported with no `+/-` -- determinism actually checked rather than
asserted.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

pytest.importorskip("sklearn", reason="scikit-learn absent: baselines not measured")

from experiments.ait_clean.run_ait_baselines import (  # noqa: E402
    CLASS_ATTACK,
    CLASS_BENIGN,
    C_VALUE,
    MAX_ITER,
    REPORT_METRICS,
    SOLVER,
    fit_logistic,
    flat_features,
    native_dims,
    probabilities,
    single_view_features,
    verify_determinism,
)
from experiments.ait_clean.run_ait_clean import CONFIG_PATH, apply_scaler, fit_scaler  # noqa: E402
from prismflow.data.ait_lds import (  # noqa: E402
    FEATURES,
    AITConfig,
    AITMultiViewDataset,
    AITSamples,
)

BASELINES = Path("results/ait_clean/baselines.json")
BASELINES_MD = Path("results/ait_clean/baselines.md")
CLEAN = Path("results/ait_clean/results.json")
SOURCES = ("apache", "auth", "dns", "audit", "suricata")
METRIC_KEYS = tuple(key for key, _, _ in REPORT_METRICS)
PREVALENCE_FIELDS = ("n_test", "n_attack", "n_benign", "attack_prevalence")


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def baselines():
    if not BASELINES.is_file():
        pytest.skip(f"{BASELINES} not present; run run_ait_baselines.py")
    return json.loads(BASELINES.read_text())


@pytest.fixture(scope="module")
def clean():
    if not CLEAN.is_file():
        pytest.skip(f"{CLEAN} not present; run run_ait_clean.py")
    return json.loads(CLEAN.read_text())


def _dataset(views, view_mask, labels, sources=SOURCES):
    n = views.shape[0]
    samples = AITSamples(
        views=np.asarray(views, dtype=np.float32),
        view_mask=np.asarray(view_mask, dtype=bool),
        labels=np.asarray(labels, dtype=np.int64),
        label_known=np.ones(n, dtype=bool),
        window_start=np.arange(n, dtype=np.float64) * 300.0,
        host_id=None,
        hosts=(),
        sources=tuple(sources),
        attack_steps=[[] for _ in range(n)],
        config=AITConfig(scenario="test", window_seconds=300, sample_mode="network",
                         sources=tuple(sources)),
    )
    return AITMultiViewDataset(samples)


def _separable(n=120, seed=0, sources=SOURCES):
    """A dataset one view can separate, with another view absent half the time."""
    rng = np.random.default_rng(seed)
    labels = np.array([CLASS_ATTACK if i % 2 else CLASS_BENIGN for i in range(n)])
    views = rng.normal(0.0, 0.5, size=(n, len(sources), 10)).astype(np.float32)
    # view 2 (dns) carries the signal
    views[:, 2, 0] = labels * 4.0 + rng.normal(0.0, 0.2, size=n)
    mask = np.ones((n, len(sources)), dtype=bool)
    mask[::2, 1] = False  # view 1 (auth) absent on half the rows
    return _dataset(views, mask, labels)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------


def test_flat_features_are_all_native_columns_plus_one_mask_per_view():
    dataset = _separable()
    x, names = flat_features(dataset, np.arange(len(dataset)))
    expected = sum(native_dims(SOURCES)) + len(SOURCES)
    assert x.shape == (len(dataset), expected)
    assert expected == 9 + 8 + 9 + 9 + 10 + 5 == 50
    assert len(names) == expected
    assert names[:2] == ["apache:n_requests", "apache:n_unique_client"]
    assert names[-len(SOURCES):] == [f"mask:{s}" for s in SOURCES]


def test_flat_features_drop_the_structurally_zero_padding():
    """D is 10 but auth is 8 wide; columns 8-9 of auth must not become features."""
    dataset = _separable()
    _, names = flat_features(dataset, np.arange(len(dataset)))
    for source in SOURCES:
        assert sum(n.startswith(f"{source}:") for n in names) == len(FEATURES[source])


def test_single_view_features_are_that_view_plus_its_own_mask():
    dataset = _separable()
    for view, source in enumerate(SOURCES):
        x, names = single_view_features(dataset, np.arange(len(dataset)), view)
        assert x.shape == (len(dataset), len(FEATURES[source]) + 1)
        assert names[-1] == f"mask:{source}"
        assert all(n.startswith(f"{source}:") for n in names[:-1])


def test_missing_views_are_zero_filled_and_the_mask_says_so(cfg):
    """The zero-fill and the indicator are one mechanism, so test them together."""
    dataset = _separable()
    indices = np.arange(len(dataset))
    scaler = fit_scaler(dataset, indices, cfg)
    scaled = apply_scaler(dataset, scaler)
    x, names = flat_features(scaled, indices)

    auth_columns = [i for i, n in enumerate(names) if n.startswith("auth:")]
    auth_mask_column = names.index("mask:auth")
    absent = np.flatnonzero(~scaled.view_mask.numpy()[:, 1])
    present = np.flatnonzero(scaled.view_mask.numpy()[:, 1])
    assert absent.size and present.size

    assert np.all(x[np.ix_(absent, auth_columns)] == 0.0), "absent view must be zero-filled"
    assert np.all(x[absent, auth_mask_column] == 0.0)
    assert np.all(x[present, auth_mask_column] == 1.0)
    # suricata is never absent here, so its indicator is constant -- legal, and
    # worth pinning so a later change does not silently drop it.
    assert np.all(x[:, names.index("mask:suricata")] == 1.0)


def test_single_view_features_of_an_absent_view_are_zero(cfg):
    dataset = _separable()
    indices = np.arange(len(dataset))
    scaled = apply_scaler(dataset, fit_scaler(dataset, indices, cfg))
    x, _ = single_view_features(scaled, indices, 1)
    absent = np.flatnonzero(~scaled.view_mask.numpy()[:, 1])
    assert np.all(x[absent] == 0.0)


# ---------------------------------------------------------------------------
# Fitting and determinism
# ---------------------------------------------------------------------------


def test_fit_uses_the_stated_solver_settings():
    dataset = _separable()
    x, _ = flat_features(dataset, np.arange(len(dataset)))
    model = fit_logistic(x, dataset.labels.numpy())
    assert model.solver == SOLVER
    assert model.max_iter == MAX_ITER
    assert model.C == C_VALUE
    assert model.class_weight is None


def test_probability_column_one_is_the_attack_class():
    """A silent class reordering would invert every ROC AUC in the addendum."""
    dataset = _separable()
    x, _ = flat_features(dataset, np.arange(len(dataset)))
    labels = dataset.labels.numpy()
    model = fit_logistic(x, labels)
    assert list(model.classes_) == [CLASS_BENIGN, CLASS_ATTACK]
    probs = probabilities(model, x)
    assert probs.shape == (len(dataset), 2)
    assert probs.sum(axis=1) == pytest.approx(np.ones(len(dataset)))
    # the separable view means attack rows must carry the higher attack probability
    assert probs[labels == CLASS_ATTACK, 1].mean() > probs[labels == CLASS_BENIGN, 1].mean()


def test_refitting_is_bit_identical():
    """This is what licenses reporting one value with no +/-."""
    dataset = _separable()
    x, _ = flat_features(dataset, np.arange(len(dataset)))
    labels = dataset.labels.numpy()
    model = fit_logistic(x, labels)
    probs = probabilities(model, x)
    again = fit_logistic(x, labels)
    assert np.array_equal(model.coef_, again.coef_)
    assert np.array_equal(model.intercept_, again.intercept_)
    assert np.array_equal(probs, probabilities(again, x))
    assert verify_determinism(x, labels, x, model, probs) == {
        "refit_coefficients_identical": True,
        "refit_probabilities_identical": True,
    }


def test_verify_determinism_raises_when_the_refit_differs():
    """A future non-deterministic solver must break the build, not the claim."""
    dataset = _separable()
    x, _ = flat_features(dataset, np.arange(len(dataset)))
    labels = dataset.labels.numpy()
    model = fit_logistic(x, labels)
    tampered = probabilities(model, x).copy()
    tampered[0, 0] += 0.01
    with pytest.raises(AssertionError, match="refit differed"):
        verify_determinism(x, labels, x, model, tampered)


def test_every_recorded_baseline_passed_its_determinism_check(baselines):
    for name, record in baselines["directions"].items():
        for model, row in record["baselines"].items():
            assert row["deterministic"] is True, f"{name}/{model}"
            assert row["refit_coefficients_identical"] is True, f"{name}/{model}"
            assert row["refit_probabilities_identical"] is True, f"{name}/{model}"
            assert row["solver"] == SOLVER


def test_the_markdown_baseline_rows_carry_no_plus_minus():
    if not BASELINES_MD.is_file():
        pytest.skip(f"{BASELINES_MD} not present")
    rows = [
        line for line in BASELINES_MD.read_text(encoding="utf-8").splitlines()
        if line.startswith("| `")
    ]
    assert rows, "no baseline rows found"
    for line in rows:
        assert "+/-" not in line, f"a deterministic baseline row must not show +/-: {line}"
    reference = [
        line for line in BASELINES_MD.read_text(encoding="utf-8").splitlines()
        if line.startswith("| *naive") or line.startswith("| *V1")
    ]
    assert reference and all("+/-" in line for line in reference), (
        "the 5-seed reference rows must keep their +/-"
    )


# ---------------------------------------------------------------------------
# The split rules, again, because the baseline could break them independently
# ---------------------------------------------------------------------------


def test_each_direction_has_no_shared_scenario(baselines):
    for name, record in baselines["directions"].items():
        assert record["train_scenario"] != record["test_scenario"], name


def test_the_two_directions_test_on_different_scenarios(baselines):
    tests = [r["test_scenario"] for r in baselines["directions"].values()]
    assert len(set(tests)) == len(tests) == 2


def test_scaler_is_fitted_on_the_train_scenario_only(baselines):
    for name, record in baselines["directions"].items():
        assert record["scaler"]["fitted_on"] == record["train_scenario"], name
        assert record["scaler"]["fit_rows"] == record["n_train"], name
    fits = {r["scaler"]["fitted_on"] for r in baselines["directions"].values()}
    assert len(fits) == 2, "the two directions must have two different fits"


def test_baselines_are_never_pooled_across_directions(baselines):
    sizes = {}
    for name, record in baselines["directions"].items():
        n = record["prevalence"]["n_test"]
        sizes[name] = n
        for model, row in record["baselines"].items():
            assert row["n_test"] == n, f"{name}/{model}: n_test {row['n_test']} != {n}"
    total = sum(sizes.values())
    for name, record in baselines["directions"].items():
        for row in record["baselines"].values():
            assert row["n_test"] != total, f"{name}: test sets look concatenated"
    for key in ("summary", "pooled", "combined", "overall", "mean_over_directions"):
        assert key not in baselines, f"{key!r} at top level would be a pooled result"


# ---------------------------------------------------------------------------
# What is reported
# ---------------------------------------------------------------------------


def test_both_baseline_families_are_present_for_both_directions(baselines):
    expected = {"flat_all_views"} | {f"single_view_{s}" for s in SOURCES}
    for name, record in baselines["directions"].items():
        assert set(record["baselines"]) == expected, name


def test_every_baseline_row_carries_every_metric_and_its_prevalence(baselines):
    for name, record in baselines["directions"].items():
        for model, row in record["baselines"].items():
            for key in METRIC_KEYS + PREVALENCE_FIELDS:
                assert key in row, f"{name}/{model}: missing {key}"
                assert row[key] is not None, f"{name}/{model}: {key} is null"
        prevalence = record["prevalence"]
        for field in PREVALENCE_FIELDS + ("n_attack_blocks",):
            assert field in prevalence, name
        assert prevalence["n_attack"] + prevalence["n_benign"] == prevalence["n_test"], name


def test_baseline_rows_drop_the_vacuity_they_do_not_have(baselines):
    """A logistic regression has no Dirichlet vacuity, so no stand-in is reported."""
    for name, record in baselines["directions"].items():
        for model, row in record["baselines"].items():
            assert not [k for k in row if k.startswith("vacuity_")], f"{name}/{model}"


def test_feature_counts_are_recorded_and_consistent(baselines):
    flat_expected = sum(native_dims(SOURCES)) + len(SOURCES)
    for name, record in baselines["directions"].items():
        assert record["baselines"]["flat_all_views"]["n_features"] == flat_expected, name
        for source in SOURCES:
            row = record["baselines"][f"single_view_{source}"]
            assert row["n_features"] == len(FEATURES[source]) + 1, f"{name}/{source}"
            assert len(row["feature_names"]) == row["n_features"]


def test_reference_rows_are_quoted_from_results_json_not_recomputed(baselines, clean):
    """Byte-equal floats, because a recomputed number would differ in the tail."""
    for name, record in baselines["directions"].items():
        clean_record = clean["directions"][name]
        for system in ("naive", "prismflow"):
            quoted = record["reference"][system]["metrics"]
            summary = clean_record["systems"][system]["summary"]
            for key in METRIC_KEYS:
                assert quoted[key]["mean"] == summary[key]["mean"], f"{name}/{system}/{key}"
                assert quoted[key]["std"] == summary[key]["std"], f"{name}/{system}/{key}"
            assert record["reference"][system]["n_seeds"] == 5
        baseline_ref = record["reference"]["majority_class_baseline"]["metrics"]
        for key in METRIC_KEYS:
            assert baseline_ref[key]["mean"] == clean_record["majority_class_baseline"][key]
            assert baseline_ref[key]["std"] is None, "the baseline has no spread to report"


def test_prevalence_matches_the_clean_runs_test_structure(baselines, clean):
    for name, record in baselines["directions"].items():
        structure = clean["directions"][name]["test_structure"]
        prevalence = record["prevalence"]
        assert prevalence["n_test"] == structure["n_label_known"], name
        assert prevalence["n_attack"] == structure["n_attack"], name
        assert prevalence["n_benign"] == structure["n_benign"], name
        assert prevalence["n_attack_blocks"] == structure["n_attack_blocks"], name


def test_label_meaning_and_determinism_note_are_in_the_artefacts(baselines):
    needle = "was the exfiltration tool running in this window"
    assert needle in baselines["label_meaning"]
    for name, record in baselines["directions"].items():
        assert needle in record["label_meaning"], name
    assert "5-SEED RULE" in baselines["determinism"]
    if BASELINES_MD.is_file():
        text = BASELINES_MD.read_text(encoding="utf-8")
        assert needle in text
        assert "5-SEED RULE" in text
        for banned in ("APT detection", "campaign detection"):
            assert banned not in text


def test_the_addendum_does_not_claim_a_seed_spread(baselines):
    """No baseline row may carry a std, mean-over-seeds or seed list."""
    for name, record in baselines["directions"].items():
        for model, row in record["baselines"].items():
            for banned in ("std", "mean", "seed", "seeds", "per_seed", "n_seeds"):
                assert banned not in row, f"{name}/{model} reports {banned!r}"
