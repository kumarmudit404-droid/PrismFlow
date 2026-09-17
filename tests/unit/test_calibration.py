import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prismflow.data.dataset import Batch
from prismflow.evaluation.calibration import (
    bin_indices,
    brier_decomposition,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
)
from prismflow.evaluation.metrics import (
    accuracy,
    aurc,
    macro_precision_recall_f1,
    risk_coverage_curve,
    selective_risk,
)
from prismflow.evaluation.protocol import compute_metrics, evaluate


def _calibrated_multiclass(n=200_000, n_classes=3, seed=0):
    """Labels drawn from the forecast itself, so the forecast is calibrated by construction."""
    rng = np.random.default_rng(seed)
    probs = rng.dirichlet(np.ones(n_classes) * 0.7, size=n)
    cumulative = probs.cumsum(axis=1)
    labels = (rng.random(n)[:, None] > cumulative).sum(axis=1)
    return probs, np.minimum(labels, n_classes - 1)


# --- ECE / MCE ----------------------------------------------------------


def test_perfectly_calibrated_predictor_has_ece_near_zero():
    rng = np.random.default_rng(0)
    confidence = rng.random(200_000)
    correct = rng.random(200_000) < confidence
    assert expected_calibration_error(confidence, correct) < 0.01


def test_perfectly_calibrated_multiclass_top_label_has_ece_near_zero():
    probs, labels = _calibrated_multiclass()
    correct = probs.argmax(axis=1) == labels
    assert expected_calibration_error(probs.max(axis=1), correct) < 0.01


def test_overconfident_predictor_has_high_ece():
    rng = np.random.default_rng(1)
    confidence = rng.uniform(0.9, 1.0, size=50_000)
    correct = rng.random(50_000) < 0.6
    ece = expected_calibration_error(confidence, correct)
    assert ece == pytest.approx(0.35, abs=0.02)
    assert maximum_calibration_error(confidence, correct) >= ece


def test_ece_of_known_two_bin_case():
    confidence = np.array([0.9, 0.9, 0.3, 0.3])
    correct = np.array([1, 0, 0, 0])
    # bin(0.9): acc 0.5, gap 0.4; bin(0.3): acc 0, gap 0.3 -> ECE 0.35, MCE 0.4
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.35)
    assert maximum_calibration_error(confidence, correct) == pytest.approx(0.4)


def test_bin_edges_are_right_closed_and_zero_goes_to_first_bin():
    index = bin_indices(np.array([0.0, 1 / 15, 1 / 15 + 1e-9, 1.0]), 15)
    assert index.tolist() == [0, 0, 1, 14]


def test_ece_bins_are_configurable():
    confidence = np.array([0.55, 0.65])
    correct = np.array([1, 0])
    # One bin pools both samples (mean conf 0.6, acc 0.5); ten bins separate them.
    assert expected_calibration_error(confidence, correct, n_bins=1) == pytest.approx(0.1)
    assert expected_calibration_error(confidence, correct, n_bins=10) == pytest.approx(0.55)


def test_confidence_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([1.2]), np.array([1]))


def test_reliability_bins_mark_empty_bins_as_nan():
    bins = reliability_bins(np.array([0.95, 0.97]), np.array([1, 1]), n_bins=5)
    assert bins.count.tolist() == [0, 0, 0, 0, 2]
    assert np.isnan(bins.accuracy[:4]).all()
    assert bins.accuracy[4] == 1.0


# --- Brier and its decomposition ----------------------------------------


def test_brier_score_known_values():
    labels = np.array([0, 1])
    assert brier_score(np.array([[1.0, 0.0], [0.0, 1.0]]), labels) == 0.0
    assert brier_score(np.array([[0.0, 1.0], [1.0, 0.0]]), labels) == pytest.approx(2.0)
    assert brier_score(np.array([[0.5, 0.5], [0.5, 0.5]]), labels) == pytest.approx(0.5)


def test_three_components_reconstruct_brier_for_binned_forecasts():
    """Forecasts constant within bins: classical Murphy identity, no remainder."""
    rng = np.random.default_rng(2)
    n_bins = 10
    p1 = (rng.integers(0, n_bins, size=20_000) + 0.5) / n_bins  # bin centres
    probs = np.stack([1.0 - p1, p1], axis=1)
    labels = (rng.random(20_000) < np.clip(p1 + 0.15, 0, 1)).astype(int)  # miscalibrated

    d = brier_decomposition(probs, labels, n_bins=n_bins)
    assert d.within_bin == pytest.approx(0.0, abs=1e-12)
    assert d.reliability - d.resolution + d.uncertainty == pytest.approx(d.brier, abs=1e-12)
    assert d.reliability > 0.01


def test_decomposition_reconstructs_brier_for_continuous_multiclass_forecasts():
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(4), size=5_000)
    labels = rng.integers(0, 4, size=5_000)
    d = brier_decomposition(probs, labels)
    assert d.reconstruction() == pytest.approx(d.brier, abs=1e-12)
    assert d.brier == pytest.approx(brier_score(probs, labels))
    assert d.reliability >= 0 and d.resolution >= 0 and d.uncertainty >= 0


def test_uncertainty_component_depends_only_on_labels():
    rng = np.random.default_rng(4)
    labels = rng.integers(0, 3, size=3_000)
    a = brier_decomposition(rng.dirichlet(np.ones(3), size=3_000), labels)
    b = brier_decomposition(np.eye(3)[labels], labels)
    assert a.uncertainty == pytest.approx(b.uncertainty)
    base = np.bincount(labels, minlength=3) / labels.size
    assert a.uncertainty == pytest.approx(np.sum(base * (1 - base)))


def test_reliability_separates_calibrated_from_overconfident_forecasts():
    probs, labels = _calibrated_multiclass(n=100_000)
    sharpened = probs**4
    sharpened /= sharpened.sum(axis=1, keepdims=True)
    calibrated = brier_decomposition(probs, labels)
    overconfident = brier_decomposition(sharpened, labels)
    assert calibrated.reliability < 0.005
    assert overconfident.reliability > 5 * calibrated.reliability
    assert calibrated.uncertainty == pytest.approx(overconfident.uncertainty)


# --- selective prediction -----------------------------------------------


def test_aurc_of_perfect_ranking_is_lower_than_random_ranking():
    rng = np.random.default_rng(5)
    errors = (rng.random(2_000) < 0.3).astype(float)
    perfect = 1.0 - errors + rng.random(2_000) * 1e-3  # every correct sample above every error
    random_rank = rng.random(2_000)
    assert aurc(perfect, errors) < aurc(random_rank, errors)
    # A perfect ranking has zero risk until coverage exceeds the accuracy.
    _, risk = risk_coverage_curve(perfect, errors)
    n_correct = int((errors == 0).sum())
    assert risk[n_correct - 1] == 0.0


def test_aurc_of_random_ranking_is_near_error_rate():
    rng = np.random.default_rng(6)
    errors = (rng.random(20_000) < 0.3).astype(float)
    assert aurc(rng.random(20_000), errors) == pytest.approx(errors.mean(), abs=0.02)


def test_tied_confidence_does_not_depend_on_storage_order():
    errors = np.array([1.0, 1.0, 0.0, 0.0])
    constant = np.full(4, 0.7)
    assert aurc(constant, errors) == pytest.approx(0.5)
    assert aurc(constant, errors[::-1]) == pytest.approx(0.5)
    assert selective_risk(constant, errors, 0.5) == pytest.approx(0.5)


def test_selective_risk_at_known_coverages():
    confidence = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    errors = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 1], dtype=float)
    assert selective_risk(confidence, errors, 1.0) == pytest.approx(0.3)
    assert selective_risk(confidence, errors, 0.9) == pytest.approx(2 / 9)
    assert selective_risk(confidence, errors, 0.8) == pytest.approx(1 / 8)
    assert selective_risk(confidence, errors, 0.5) == pytest.approx(1 / 5)
    with pytest.raises(ValueError):
        selective_risk(confidence, errors, 0.0)


# --- classification -----------------------------------------------------


def test_accuracy_and_macro_scores_on_hand_example():
    labels = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 1, 1, 1, 0, 0])
    assert accuracy(labels, predictions) == pytest.approx(0.5)
    scores = macro_precision_recall_f1(labels, predictions, n_classes=3)
    # class 0: P 1/3 R 1/2; class 1: P 2/3 R 1; class 2: never predicted, P 0 R 0
    assert scores["macro_precision"] == pytest.approx((1 / 3 + 2 / 3 + 0) / 3)
    assert scores["macro_recall"] == pytest.approx((0.5 + 1 + 0) / 3)
    assert scores["macro_f1"] == pytest.approx((0.4 + 0.8 + 0) / 3)


# --- protocol -----------------------------------------------------------


def test_compute_metrics_reports_both_confidence_definitions():
    probs, labels = _calibrated_multiclass(n=5_000)
    metrics = compute_metrics(labels, probs, vacuity_confidence=np.full(5_000, 0.5))
    for prefix in ("prob", "vacuity"):
        for name in ("ece", "mce", "aurc", "selective_risk@1.00", "selective_risk@0.50"):
            assert f"{prefix}_{name}" in metrics
    for name in ("brier_reliability", "brier_resolution", "brier_uncertainty", "brier_within_bin"):
        assert name in metrics
    assert metrics["prob_selective_risk@1.00"] == pytest.approx(1 - metrics["accuracy"])


def test_compute_metrics_rejects_probabilities_far_outside_unit_interval():
    with pytest.raises(ValueError):
        compute_metrics(np.array([0]), np.array([[1.2, -0.2]]), np.array([0.5]))


class _StubModel(torch.nn.Module):
    """Returns fixed per-sample probabilities looked up by sample id."""

    def __init__(self, probs, vacuity_confidence, with_eniv):
        super().__init__()
        self.probs = torch.as_tensor(probs, dtype=torch.float32)
        self.vacuity = torch.as_tensor(vacuity_confidence, dtype=torch.float32)
        self.with_eniv = with_eniv

    def forward(self, views, view_mask):
        ids = views[:, 0, 0].long()
        eniv = None
        if self.with_eniv:
            eniv = SimpleNamespace(effective_views=3.0, mean_dependence=0.1, efficiency_ratio=0.75)
        return SimpleNamespace(probs=self.probs[ids], confidence=self.vacuity[ids], eniv=eniv)


def _batches(labels, batch_size=64):
    n = len(labels)
    for start in range(0, n, batch_size):
        ids = torch.arange(start, min(start + batch_size, n))
        views = ids.float().view(-1, 1, 1).expand(-1, 2, 1).clone()
        yield Batch(
            views=views,
            view_mask=torch.ones(len(ids), 2, dtype=torch.bool),
            labels=torch.as_tensor(labels[ids.numpy()]),
            sample_ids=ids,
        )


def _stub_setup(seeds, with_eniv=True):
    models, labels_by_seed = {}, {}
    for seed in seeds:
        probs, labels = _calibrated_multiclass(n=500, seed=seed)
        vacuity = np.random.default_rng(seed).random(500)
        models[seed] = _StubModel(probs, vacuity, with_eniv)
        labels_by_seed[seed] = labels
    return models, (lambda seed: _batches(labels_by_seed[seed]))


def test_evaluate_writes_all_outputs(tmp_path):
    seeds = [0, 1, 2, 3, 4]
    models, loader = _stub_setup(seeds)
    report = evaluate(models, loader, "unit/stub", seeds, results_dir=tmp_path)

    out = tmp_path / "unit" / "stub"
    for name in ("metrics.json", "metrics.csv", "reliability_diagram.png"):
        assert (out / name).stat().st_size > 0

    saved = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert saved["not_evidence"] is False
    assert [row["seed"] for row in saved["per_seed"]] == seeds
    assert saved["summary"]["accuracy"]["n_seeds"] == 5
    assert saved["summary"]["eniv"]["mean"] == pytest.approx(3.0)

    per_seed_acc = [row["accuracy"] for row in report["per_seed"]]
    assert report["summary"]["accuracy"]["mean"] == pytest.approx(np.mean(per_seed_acc))
    assert report["summary"]["accuracy"]["std"] == pytest.approx(np.std(per_seed_acc, ddof=1))

    with (out / "metrics.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [r["row"] for r in rows] == [f"seed={s}" for s in seeds] + ["mean", "std"]


def test_evaluate_matches_compute_metrics_per_seed(tmp_path):
    seeds = [0, 1, 2, 3, 4]
    models, loader = _stub_setup(seeds, with_eniv=False)
    report = evaluate(models, loader, "match", seeds, results_dir=tmp_path)
    model = models[2]
    labels = np.concatenate([b.labels.numpy() for b in loader(2)])
    direct = compute_metrics(labels, model.probs.double().numpy(), model.vacuity.double().numpy())
    row = report["per_seed"][2]
    for key, value in direct.items():
        assert row[key] == pytest.approx(value, abs=1e-6), key
    assert np.isnan(row["eniv"])


def test_evaluate_refuses_fewer_than_five_seeds(tmp_path):
    models, loader = _stub_setup([0, 1])
    with pytest.raises(ValueError, match="at least 5 seeds"):
        evaluate(models, loader, "few", [0, 1], results_dir=tmp_path)

    report = evaluate(models, loader, "few", [0, 1], results_dir=tmp_path, allow_fewer_seeds=True)
    assert report["not_evidence"] is True
    saved = json.loads((tmp_path / "few" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["not_evidence"] is True


def test_evaluate_rejects_a_single_model_for_several_seeds(tmp_path):
    models, loader = _stub_setup([0])
    with pytest.raises(TypeError):
        evaluate(models[0], loader, "single", [0, 1, 2, 3, 4], results_dir=tmp_path)


def test_evaluate_runs_on_the_real_model_output(tmp_path):
    """Interface check against PrismFlowOutput, not a result: untrained weights."""
    from prismflow.data.dataset import MultiViewDataset, split_indices
    from prismflow.data.loaders import iter_batches
    from prismflow.data.synthetic import SyntheticConfig
    from prismflow.train import TrainConfig, build_model
    from prismflow.utils.seed import set_seed

    dataset = MultiViewDataset(SyntheticConfig(n_samples=300, seed=0))
    test_idx = split_indices(len(dataset), seed=0)["test"]

    def model_for(seed):
        set_seed(seed)
        return build_model(TrainConfig(use_discount=seed % 2 == 0))

    seeds = [0, 1, 2, 3, 4]
    report = evaluate(
        model_for, lambda seed: iter_batches(dataset, test_idx, 16), "real", seeds, results_dir=tmp_path
    )
    assert report["summary"]["prob_ece"]["n_seeds"] == 5
    assert report["summary"]["eniv"]["n_seeds"] == 3  # only the discounting seeds report ENIV
    assert (tmp_path / "real" / "reliability_diagram.png").exists()
