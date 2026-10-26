"""Part 35 tests. The three pieces of new logic, and nothing that needs a GPU.

The analysis itself is read-only arithmetic over committed files, so what can
actually be wrong is the new logic: the inverse-probability weighting, the
cross-fit fold assignment, and the cluster bootstrap's AUC helper. The helper
is the one worth the most care, because it is a reformulation of
`roc_auc_binary` rather than a call to it: every test below pins it against
Part 28's own function on an explicitly repeated index set.

Structural tests run against `results/lanl_reanalysis/results.json` when it
exists and skip when it does not, so a clean checkout still passes.
"""

import json
import zlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.ait_clean.run_ait_clean import (
    CLASS_ATTACK,
    CLASS_BENIGN,
    roc_auc_binary,
)
from experiments.lanl_reanalysis.run import (
    CONFIG_PATH,
    ClusterAUC,
    computer_fold,
    cross_fitted_platt,
    ipw_weights,
    platt_apply,
    platt_fit,
    prior_shift,
    summarise,
    threshold_for_false_alarm_budget,
    weighted_brier,
    weighted_confusion,
    weighted_rates,
)

RESULTS = Path(__file__).resolve().parents[2] / "results" / "lanl_reanalysis" / "results.json"


@pytest.fixture(scope="module")
def cfg() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results() -> dict:
    if not RESULTS.exists():
        pytest.skip("results/lanl_reanalysis/results.json not built")
    return json.loads(RESULTS.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The weighting
# ---------------------------------------------------------------------------
def test_ipw_weights_are_one_per_case_and_the_divisor_per_control():
    labels = np.array([CLASS_ATTACK, CLASS_BENIGN, CLASS_BENIGN, CLASS_ATTACK])
    w = ipw_weights(labels, 1, 512)
    assert w.tolist() == [1.0, 512.0, 512.0, 1.0]


def test_ipw_weights_read_the_divisor_from_the_preregistration(cfg):
    spec = cfg["step3_l23"]["weights"]
    assert (spec["attack"], spec["benign"]) == (1, 512)


def test_weighted_confusion_counts_weight_not_rows():
    labels = np.array([CLASS_ATTACK, CLASS_BENIGN, CLASS_BENIGN])
    p = np.array([0.9, 0.9, 0.1])
    w = ipw_weights(labels, 1, 512)
    conf = weighted_confusion(labels, p, w, 0.5)
    assert conf == {"tp": 1.0, "fp": 512.0, "fn": 0.0, "tn": 512.0}
    rates = weighted_rates(conf)
    assert rates["precision"] == pytest.approx(1.0 / 513.0)
    assert rates["recall"] == pytest.approx(1.0)
    assert rates["false_positive_rate"] == pytest.approx(0.5)


def test_weighted_confusion_alarms_on_the_threshold_itself():
    labels = np.array([CLASS_BENIGN])
    conf = weighted_confusion(labels, np.array([0.5]), np.array([512.0]), 0.5)
    assert conf["fp"] == 512.0


def test_weighted_brier_is_the_two_class_form_and_equals_the_unweighted_at_equal_weights():
    labels = np.array([CLASS_ATTACK, CLASS_BENIGN, CLASS_BENIGN])
    p = np.array([0.7, 0.2, 0.4])
    flat = weighted_brier(labels, p, np.ones(3))
    manual = np.mean([2 * (0.7 - 1) ** 2, 2 * 0.2 ** 2, 2 * 0.4 ** 2])
    assert flat == pytest.approx(manual)
    # A perfect forecast scores 0 and the worst possible scores 2.
    assert weighted_brier(labels, np.array([1.0, 0.0, 0.0]), np.ones(3)) == pytest.approx(0.0)
    assert weighted_brier(labels, np.array([0.0, 1.0, 1.0]), np.ones(3)) == pytest.approx(2.0)


def test_weighted_brier_moves_towards_the_class_the_weights_favour():
    labels = np.array([CLASS_ATTACK, CLASS_BENIGN])
    p = np.array([0.0, 0.0])  # perfect on the benign row, worst on the attack row
    heavy_benign = weighted_brier(labels, p, ipw_weights(labels, 1, 512))
    equal = weighted_brier(labels, p, np.ones(2))
    assert heavy_benign < equal
    assert heavy_benign == pytest.approx(2.0 / 513.0)


def test_threshold_budget_is_the_most_permissive_threshold_inside_the_budget():
    labels = np.array([CLASS_BENIGN, CLASS_BENIGN, CLASS_ATTACK])
    p = np.array([0.2, 0.8, 0.9])
    w = ipw_weights(labels, 1, 512)
    # Budget 512 admits exactly one benign row: the 0.8 one.
    t = threshold_for_false_alarm_budget(labels, p, w, 512.0)
    assert weighted_confusion(labels, p, w, t)["fp"] == 512.0
    assert t == pytest.approx(0.8)
    # Budget below one control's weight can only be met with zero alarms.
    t0 = threshold_for_false_alarm_budget(labels, p, w, 511.0)
    assert weighted_confusion(labels, p, w, t0)["fp"] == 0.0
    assert t0 > 0.8


def test_threshold_budget_never_returns_a_threshold_over_the_budget():
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, size=400)
    p = rng.random(400)
    w = ipw_weights(labels, 1, 512)
    for budget in (0.0, 512.0, 5120.0, 1e9):
        t = threshold_for_false_alarm_budget(labels, p, w, budget)
        assert weighted_confusion(labels, p, w, t)["fp"] <= budget + 1e-9


# ---------------------------------------------------------------------------
# The cross-fit fold assignment
# ---------------------------------------------------------------------------
def test_computer_fold_is_the_declared_crc32_of_the_name():
    names = ["C17693", "C1015", "C1"]
    assert computer_fold(names).tolist() == [
        zlib.crc32(n.encode("utf-8")) % 2 for n in names]


def test_computer_fold_is_deterministic_and_order_independent():
    names = np.array(["C1", "C2", "C3", "C4", "C5"])
    first = computer_fold(names)
    assert np.array_equal(first, computer_fold(names))
    order = np.array([3, 0, 4, 1, 2])
    assert np.array_equal(computer_fold(names[order]), first[order])


def test_computer_fold_keeps_every_window_of_a_host_in_one_fold():
    rows = np.array(["C1", "C1", "C2", "C1", "C2"])
    folds = computer_fold(rows)
    assert len(set(folds[rows == "C1"])) == 1
    assert len(set(folds[rows == "C2"])) == 1


def test_computer_fold_uses_both_folds_on_lanl_style_names():
    names = [f"C{i}" for i in range(1, 500)]
    folds = computer_fold(names)
    assert set(folds.tolist()) == {0, 1}


def test_cross_fitted_platt_never_fits_on_the_rows_it_scores():
    rng = np.random.default_rng(11)
    computers = np.array([f"C{i % 40}" for i in range(400)])
    folds = computer_fold(computers)
    labels = rng.integers(0, 2, size=400)
    p = np.clip(rng.random(400), 1e-3, 1 - 1e-3)
    out, params = cross_fitted_platt(p, labels, folds)
    assert out.shape == p.shape
    assert set(params) == {"fold_0", "fold_1"}
    for f in (0, 1):
        other = folds != f
        a, b = platt_fit(p[other], labels[other])
        assert np.allclose(out[folds == f], platt_apply(p[folds == f], a, b))
        assert params[f"fold_{f}"]["n_fit_rows"] == int(other.sum())


def test_platt_is_monotone_and_an_identity_fit_recovers_the_probabilities():
    p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    out = platt_apply(p, 1.0, 0.0)
    assert np.allclose(out, p, atol=1e-5)
    rising = platt_apply(p, 2.0, -1.0)
    assert np.all(np.diff(rising) > 0)


def test_platt_fit_recovers_a_known_logistic_shift():
    rng = np.random.default_rng(3)
    x = rng.normal(size=20000)
    p_raw = 1.0 / (1.0 + np.exp(-x))
    true_a, true_b = 1.5, -0.7
    z = true_a * np.log(p_raw / (1 - p_raw)) + true_b
    y = (rng.random(x.size) < 1.0 / (1.0 + np.exp(-z))).astype(np.int64)
    a, b = platt_fit(p_raw, y)
    assert a == pytest.approx(true_a, abs=0.1)
    assert b == pytest.approx(true_b, abs=0.1)


def test_prior_shift_is_the_identity_when_the_priors_match_and_raises_otherwise():
    p = np.array([0.01, 0.2, 0.9])
    assert np.allclose(prior_shift(p, 0.02, 0.02), p)
    up = prior_shift(p, 0.01, 0.02)
    assert np.all(up > p)  # a higher test prior can only raise an attack probability
    assert np.all((up > 0) & (up < 1))


# ---------------------------------------------------------------------------
# The bootstrap helper
# ---------------------------------------------------------------------------
def _reference_auc(scores, labels, cluster_of_row, multiplicity, pos_rows, neg_rows):
    """AUC the slow, obvious way: materialise the repeated index set."""
    rows = np.concatenate([pos_rows, neg_rows])
    repeated = np.concatenate(
        [np.repeat(r, int(multiplicity[cluster_of_row[r]])) for r in rows])
    return roc_auc_binary(labels[repeated], scores[repeated])


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_cluster_auc_matches_roc_auc_binary_on_the_repeated_index_set(seed):
    rng = np.random.default_rng(seed)
    n, n_clusters = 300, 25
    cluster_of_row = rng.integers(0, n_clusters, size=n)
    labels = rng.integers(0, 2, size=n)
    # Deliberately tie-heavy: ties are where a rank formula goes wrong.
    scores = np.round(rng.random(n), 2)
    pos_rows = np.flatnonzero(labels == CLASS_ATTACK)
    neg_rows = np.flatnonzero(labels == CLASS_BENIGN)
    machine = ClusterAUC(scores, pos_rows, neg_rows, cluster_of_row)
    for _ in range(10):
        picked = rng.integers(0, n_clusters, size=n_clusters)
        mult = np.bincount(picked, minlength=n_clusters).astype(np.float64)
        if mult[cluster_of_row[pos_rows]].sum() == 0 or mult[cluster_of_row[neg_rows]].sum() == 0:
            continue
        assert machine.auc(mult) == pytest.approx(
            _reference_auc(scores, labels, cluster_of_row, mult, pos_rows, neg_rows))


def test_cluster_auc_at_unit_multiplicity_is_the_plain_auc():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 2, size=200)
    scores = rng.random(200)
    cluster_of_row = np.arange(200)
    pos_rows = np.flatnonzero(labels == CLASS_ATTACK)
    neg_rows = np.flatnonzero(labels == CLASS_BENIGN)
    machine = ClusterAUC(scores, pos_rows, neg_rows, cluster_of_row)
    assert machine.auc(np.ones(200)) == pytest.approx(roc_auc_binary(labels, scores))


def test_cluster_auc_credits_ties_half_and_a_constant_score_gives_a_half():
    labels = np.array([CLASS_ATTACK, CLASS_BENIGN, CLASS_BENIGN, CLASS_ATTACK])
    scores = np.full(4, 0.3)
    cluster_of_row = np.array([0, 0, 1, 1])
    machine = ClusterAUC(scores, np.array([0, 3]), np.array([1, 2]), cluster_of_row)
    assert machine.auc(np.array([1.0, 1.0])) == pytest.approx(0.5)
    assert machine.auc(np.array([3.0, 1.0])) == pytest.approx(0.5)


def test_cluster_auc_is_not_measured_when_a_class_is_absent_from_the_draw():
    labels = np.array([CLASS_ATTACK, CLASS_BENIGN])
    scores = np.array([0.9, 0.1])
    cluster_of_row = np.array([0, 1])
    machine = ClusterAUC(scores, np.array([0]), np.array([1]), cluster_of_row)
    assert np.isnan(machine.auc(np.array([2.0, 0.0])))   # no benign row drawn
    assert np.isnan(machine.auc(np.array([0.0, 2.0])))   # no attack row drawn


def test_cluster_auc_weights_a_whole_cluster_together():
    # One cluster holds a perfectly ranked pair, the other an inverted one.
    scores = np.array([0.9, 0.1, 0.1, 0.9])
    labels = np.array([CLASS_ATTACK, CLASS_BENIGN, CLASS_ATTACK, CLASS_BENIGN])
    cluster_of_row = np.array([0, 0, 1, 1])
    machine = ClusterAUC(scores, np.array([0, 2]), np.array([1, 3]), cluster_of_row)
    assert machine.auc(np.array([2.0, 0.0])) == pytest.approx(1.0)
    assert machine.auc(np.array([0.0, 2.0])) == pytest.approx(0.0)
    assert machine.auc(np.array([1.0, 1.0])) == pytest.approx(0.5)


def test_summarise_uses_the_sample_std_so_it_matches_part30():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = summarise(values)
    assert out["mean"] == pytest.approx(3.0)
    assert out["std"] == pytest.approx(np.std(values, ddof=1))
    assert out["n_seeds"] == 5
    assert summarise([2.0])["std"] == "not_measured"
    assert summarise([])["mean"] == "not_measured"


# ---------------------------------------------------------------------------
# Structural: what the committed results must say
# ---------------------------------------------------------------------------
def test_preregistration_fixes_the_bootstrap_and_the_decision_rule(cfg):
    boot = cfg["step4_l26"]["bootstrap"]
    assert boot["replicates"] == 2000
    assert boot["seed"] == 12345
    assert "5 of 5" in cfg["step2_l29"]["decision_rule"]
    assert cfg["shared"]["constant_baseline_brier"] == 0.0428


def test_every_input_sha256_matched(results):
    assert results["integrity"]["all_match"] is True
    assert results["integrity"]["n_verified"] == 42  # samples + 40 scores + run 1


def test_no_training_happened(results):
    assert results["no_training"] is True


def test_the_scalar_gate_was_checked_before_the_intervals(results):
    gate = results["step4_l26"]["scalar_gate"]
    assert gate["match_to_4dp"] is True
    assert round(gate["test_auc_recomputed"], 4) == 0.6843


def test_the_saved_scores_reproduce_part30s_committed_brier(results):
    check = results["step2_l29"]["reproduction_check"]
    assert check["passes"] is True
    assert check["max_abs_difference"] <= check["tolerance"]
    assert len(check["per_unit"]) == 40


def test_every_reported_seed_count_is_five(results):
    s2 = results["step2_l29"]
    for arm, entry in s2["per_arm"].items():
        for system, e in entry["per_system"].items():
            assert e["uncorrected"]["brier"]["n_seeds"] == 5, (arm, system)
            assert e["b_prior_shift"]["brier"]["n_seeds"] == 5, (arm, system)


def test_the_oracle_and_diagnostic_labels_survive_into_the_results(results):
    for entry in results["step2_l29"]["per_arm"].values():
        for e in entry["per_system"].values():
            assert e["b_prior_shift"]["oracle"] is True
            assert e["c_cross_fitted_platt"]["diagnostic_only"] is True


def test_known_limitations_is_not_edited_by_this_part(results):
    assert results["config"]["outputs"]["not_edited"] == "docs/KNOWN_LIMITATIONS.md"
