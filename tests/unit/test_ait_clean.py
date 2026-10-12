"""Part 28 tests. The split rules first, because they are the ones a bug hides in.

The expensive thing here is training, so the structural tests run against
`results/ait_clean/results.json` when it exists and skip when it does not;
the pure functions are tested directly on small arrays either way.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from experiments.ait_clean.run_ait_clean import (
    CLASS_ATTACK,
    CLASS_BENIGN,
    CONFIG_PATH,
    DNSTEAL_FAMILY,
    EXFILTRATION_TOOL_STEPS,
    LOWER_IS_BETTER,
    apply_scaler,
    contiguous_blocks,
    fit_scaler,
    majority_baseline,
    metric_row,
    paired_difference,
    part28_extras,
    roc_auc_binary,
    sensor_group_check,
    summarise,
)
from prismflow.data.ait_lds import (
    LABEL_UNKNOWN,
    AITConfig,
    AITMultiViewDataset,
    AITSamples,
)
from prismflow.evaluation.metrics import macro_precision_recall_f1

RESULTS = Path("results/ait_clean/results.json")
REQUIRED_METRICS = (
    "balanced_accuracy",
    "roc_auc",
    "macro_f1",
    "attack_recall",
)
PREVALENCE_FIELDS = ("n_test", "n_attack", "n_benign", "attack_prevalence")


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(CONFIG_PATH.read_text())


@pytest.fixture(scope="module")
def results():
    if not RESULTS.is_file():
        pytest.skip(f"{RESULTS} not present; run experiments/ait_clean/run_ait_clean.py")
    return json.loads(RESULTS.read_text())


# ---------------------------------------------------------------------------
# The split rules
# ---------------------------------------------------------------------------


def test_config_declares_two_directions_in_both_orders(cfg):
    directions = cfg["directions"]
    assert len(directions) == 2
    pairs = {(d["train"], d["test"]) for d in directions}
    scenarios = {d["train"] for d in directions} | {d["test"] for d in directions}
    assert len(scenarios) == 2, "both directions must span exactly two scenarios"
    a, b = sorted(scenarios)
    assert pairs == {(a, b), (b, a)}, "the two directions must be the two orders"


def test_each_direction_has_no_shared_scenario(cfg):
    """The most important rule: train and test are never the same scenario."""
    for direction in cfg["directions"]:
        assert direction["train"] != direction["test"], direction["name"]


def test_results_each_direction_has_no_shared_scenario(results):
    for name, record in results["directions"].items():
        assert record["train_scenario"] != record["test_scenario"], name


def test_the_two_directions_test_on_different_scenarios(results):
    tests = [r["test_scenario"] for r in results["directions"].values()]
    assert len(set(tests)) == len(tests) == 2


def test_direction_test_sets_are_never_concatenated_or_averaged(results):
    """Each direction's n must equal exactly one scenario's label_known count.

    A concatenated test set would show up as an n equal to the SUM, and an
    averaged metric would need a record that is neither direction. Both are
    checked by construction: the only metric containers in the JSON live under
    `directions`, and each one's n matches its own test scenario alone.
    """
    sizes = {}
    for name, record in results["directions"].items():
        structure = record["test_structure"]
        n = structure["n_label_known"]
        sizes[name] = n
        for system in record["systems"].values():
            for row in system["per_seed"]:
                assert row["n_test"] == n, f"{name}: n_test {row['n_test']} != {n}"
                assert row["n_samples"] == n
        assert record["majority_class_baseline"]["n_test"] == n

    total = sum(sizes.values())
    for name, record in results["directions"].items():
        for system in record["systems"].values():
            for row in system["per_seed"]:
                assert row["n_test"] != total, f"{name}: test sets look concatenated"

    # No top-level container may hold a metric: a pooled number would have to
    # live outside `directions`, and nothing outside it is allowed to.
    for key in ("summary", "pooled", "combined", "overall", "mean_over_directions"):
        assert key not in results, f"{key!r} at top level would be a pooled result"


def test_scaler_is_fitted_on_the_train_scenario_only(results):
    for name, record in results["directions"].items():
        assert record["scaler"]["fitted_on"] == record["train_scenario"], name
        assert record["scaler"]["fit_rows"] == record["n_train"], name


def test_scaler_is_refit_per_direction_not_shared(results):
    """Two directions train on two different scenarios, so two different fits."""
    fits = {r["scaler"]["fitted_on"] for r in results["directions"].values()}
    assert len(fits) == 2


# ---------------------------------------------------------------------------
# The scaler itself
# ---------------------------------------------------------------------------


def _dataset(views, view_mask, sources):
    """A real `AITMultiViewDataset` over hand-built arrays.

    Not a stub: `apply_scaler` rebuilds the dataset through `AITSamples`, so a
    stub would exercise a different code path than the experiment does.
    """
    n = views.shape[0]
    samples = AITSamples(
        views=np.asarray(views, dtype=np.float32),
        view_mask=np.asarray(view_mask, dtype=bool),
        labels=np.zeros(n, dtype=np.int64),
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


def test_scaler_fits_only_on_present_rows(cfg):
    """An absent view's zero row must not pull the mean toward zero."""
    views = np.zeros((4, 1, 10), dtype=np.float32)
    views[:, 0, 0] = [10.0, 12.0, 0.0, 0.0]
    mask = np.array([[True], [True], [False], [False]])
    dataset = _dataset(views, mask, ["suricata"])
    scaler = fit_scaler(dataset, np.arange(4), cfg)
    assert scaler["mean"][0, 0] == pytest.approx(11.0)
    assert scaler["present_rows_per_view"] == [2]


def test_scaler_leaves_padded_columns_and_absent_rows_at_zero(cfg):
    """`auth` is 8 features wide in a D = 10 tensor; columns 8-9 are padding."""
    rng = np.random.default_rng(0)
    views = rng.normal(5.0, 2.0, size=(20, 1, 10)).astype(np.float32)
    views[:, 0, 8:] = 0.0
    mask = np.ones((20, 1), dtype=bool)
    mask[:5, 0] = False
    dataset = _dataset(views, mask, ["auth"])
    scaler = fit_scaler(dataset, np.arange(20), cfg)
    assert np.all(scaler["mean"][0, 8:] == 0.0)
    assert np.all(scaler["std"][0, 8:] == 1.0)

    scaled = apply_scaler(dataset, scaler).views.numpy()
    assert np.all(scaled[:5, 0, :] == 0.0), "absent view rows must stay zero"
    assert np.all(scaled[:, 0, 8:] == 0.0), "padded columns must stay zero"
    present = scaled[5:, 0, :8]
    assert present.mean(axis=0) == pytest.approx(np.zeros(8), abs=1e-5)
    assert present.std(axis=0) == pytest.approx(np.ones(8), abs=1e-4)


def test_scaler_does_not_divide_by_a_constant_feature(cfg):
    views = np.full((6, 1, 10), 3.0, dtype=np.float32)
    dataset = _dataset(views, np.ones((6, 1), dtype=bool), ["suricata"])
    scaler = fit_scaler(dataset, np.arange(6), cfg)
    assert np.all(np.isfinite(apply_scaler(dataset, scaler).views.numpy()))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_balanced_accuracy_equals_macro_recall_when_both_classes_present():
    """Ties the local metric to the frozen protocol's own code."""
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=200)
    probs = rng.random((200, 2))
    probs /= probs.sum(axis=1, keepdims=True)
    mine = part28_extras(labels, probs)["balanced_accuracy"]
    theirs = macro_precision_recall_f1(labels, probs.argmax(axis=1), 2)["macro_recall"]
    assert mine == pytest.approx(theirs)


def test_attack_recall_is_the_attack_class_not_the_macro_mean():
    labels = np.array([1, 1, 1, 1, 0, 0])
    #                  hit hit miss miss | hit hit
    probs = np.array([[0.1, 0.9], [0.1, 0.9], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]])
    extras = part28_extras(labels, probs)
    assert extras["attack_recall"] == pytest.approx(0.5)
    assert extras["benign_recall"] == pytest.approx(1.0)
    assert extras["balanced_accuracy"] == pytest.approx(0.75)


def test_roc_auc_perfect_random_and_constant():
    labels = np.array([0, 0, 0, 1, 1, 1])
    assert roc_auc_binary(labels, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(1.0)
    assert roc_auc_binary(labels, np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])) == pytest.approx(0.0)
    assert roc_auc_binary(labels, np.full(6, 0.5)) == pytest.approx(0.5)


def test_roc_auc_credits_ties_at_half():
    labels = np.array([0, 1])
    assert roc_auc_binary(labels, np.array([0.5, 0.5])) == pytest.approx(0.5)
    labels = np.array([0, 0, 1, 1])
    # one tied pair out of four, the rest ordered correctly
    assert roc_auc_binary(labels, np.array([0.1, 0.5, 0.5, 0.9])) == pytest.approx(0.875)


def test_roc_auc_is_nan_with_only_one_class():
    assert np.isnan(roc_auc_binary(np.ones(5, dtype=int), np.arange(5.0)))


def test_every_metric_row_carries_its_prevalence(cfg):
    labels = np.array([1, 1, 1, 0])
    probs = np.tile([0.25, 0.75], (4, 1))
    row = metric_row(labels, probs, probs.max(axis=1), cfg)
    for field in PREVALENCE_FIELDS:
        assert field in row
    assert row["n_test"] == 4
    assert row["n_attack"] == 3
    assert row["n_benign"] == 1
    assert row["attack_prevalence"] == pytest.approx(0.75)


def test_majority_baseline_scores_the_prevalence_and_chance(cfg):
    labels = np.array([1] * 90 + [0] * 10)
    baseline = majority_baseline(labels, cfg)
    assert baseline["accuracy"] == pytest.approx(0.90)
    assert baseline["balanced_accuracy"] == pytest.approx(0.5)
    assert baseline["roc_auc"] == pytest.approx(0.5)
    assert baseline["attack_recall"] == pytest.approx(1.0)
    assert baseline["benign_recall"] == pytest.approx(0.0)
    assert baseline["majority_class"] == pytest.approx(float(CLASS_ATTACK))
    assert baseline["attack_prevalence"] == pytest.approx(0.90)


def test_majority_baseline_drops_vacuity_it_does_not_have(cfg):
    """A baseline has no Dirichlet vacuity, so no `vacuity_` key may be reported."""
    baseline = majority_baseline(np.array([1] * 6 + [0] * 4), cfg)
    assert not [k for k in baseline if k.startswith("vacuity_")]


def test_majority_baseline_follows_the_benign_majority_too(cfg):
    labels = np.array([0] * 70 + [1] * 30)
    baseline = majority_baseline(labels, cfg)
    assert baseline["majority_class"] == pytest.approx(float(CLASS_BENIGN))
    assert baseline["attack_recall"] == pytest.approx(0.0)
    assert baseline["accuracy"] == pytest.approx(0.70)


def test_baseline_and_systems_go_through_the_same_metric_code(cfg):
    """A constant-prevalence model scored as a system must equal the baseline."""
    labels = np.array([1] * 60 + [0] * 40)
    probs = np.tile([0.4, 0.6], (100, 1))
    as_system = metric_row(labels, probs, probs.max(axis=1), cfg)
    baseline = majority_baseline(labels, cfg)
    for key in REQUIRED_METRICS + ("accuracy", "brier", "prob_ece", "prob_aurc"):
        assert as_system[key] == pytest.approx(baseline[key]), key


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


def test_contiguous_blocks_counts_maximal_runs():
    flags = np.array([True, True, False, True, False, False, True, True, True])
    start = np.arange(9) * 300.0
    blocks = contiguous_blocks(flags, start)
    assert [b["length_windows"] for b in blocks] == [2, 1, 3]
    assert [b["start_index"] for b in blocks] == [0, 3, 6]
    assert blocks[0]["start_time"].startswith("1970-01-01T00:00:00")


def test_contiguous_blocks_handles_all_and_none():
    start = np.arange(4) * 300.0
    assert len(contiguous_blocks(np.ones(4, dtype=bool), start)) == 1
    assert contiguous_blocks(np.zeros(4, dtype=bool), start) == []


def test_results_carry_an_attack_block_count_per_direction(results):
    for name, record in results["directions"].items():
        structure = record["test_structure"]
        assert structure["n_attack_blocks"] >= 1, name
        assert len(structure["attack_blocks"]) == structure["n_attack_blocks"]
        assert sum(b["length_windows"] for b in structure["attack_blocks"]) == structure["n_attack"]
        assert structure["n_benign_blocks"] >= 1
        assert sum(b["length_windows"] for b in structure["benign_blocks"]) == structure["n_benign"]
        for block in structure["attack_blocks"]:
            assert block["start_time"].endswith("+00:00")
        assert "effective sample size" in structure["consequence"]


def test_block_counts_reproduce_the_part27_window_table(results):
    """Part 27 measured the same rows; disagreement means one of us is wrong."""
    expected = {"santos": 3, "russellmitchell": 1}
    for record in results["directions"].values():
        scenario = record["test_scenario"]
        assert record["test_structure"]["n_attack_blocks"] == expected[scenario], scenario


# ---------------------------------------------------------------------------
# The results JSON contract
# ---------------------------------------------------------------------------


def test_results_hold_five_seeds_per_metric_per_direction(results, cfg):
    n_seeds = len(cfg["seeds"])
    assert n_seeds >= 5
    assert results["seeds"] == cfg["seeds"]
    assert not results["not_evidence"]
    for name, record in results["directions"].items():
        assert len(record["systems"]) == 2
        for system_name, system in record["systems"].items():
            seeds = [row["seed"] for row in system["per_seed"]]
            assert seeds == cfg["seeds"], f"{name}/{system_name}"
            for metric in REQUIRED_METRICS:
                entry = system["summary"][metric]
                assert entry["n_seeds"] == n_seeds, f"{name}/{system_name}/{metric}"
                assert entry["mean"] is not None


def test_every_required_metric_and_its_prevalence_are_present(results):
    for name, record in results["directions"].items():
        for system_name, system in record["systems"].items():
            for row in system["per_seed"]:
                for metric in REQUIRED_METRICS + PREVALENCE_FIELDS:
                    assert metric in row, f"{name}/{system_name}: missing {metric}"
        for metric in REQUIRED_METRICS + PREVALENCE_FIELDS:
            assert metric in record["majority_class_baseline"], f"{name}: baseline {metric}"


def test_accuracy_never_appears_without_its_companions(results):
    """Plain accuracy may be reported only beside the three that bound it."""
    for name, record in results["directions"].items():
        for system in record["systems"].values():
            for row in system["per_seed"]:
                if "accuracy" in row:
                    for companion in ("balanced_accuracy", "roc_auc", "attack_prevalence"):
                        assert companion in row, f"{name}: accuracy without {companion}"
        baseline = record["majority_class_baseline"]
        assert "accuracy" in baseline and "balanced_accuracy" in baseline


def test_eniv_and_the_full_dependence_matrix_are_present(results):
    n_views = len(results["sources"])
    for name, record in results["directions"].items():
        for system_name, system in record["systems"].items():
            assert len(system["diagnostics_per_seed"]) == len(results["seeds"])
            for row in system["diagnostics_per_seed"]:
                matrix = np.asarray(row["matrix"], dtype=float)
                assert matrix.shape == (n_views, n_views), f"{name}/{system_name}"
                assert row["eniv_eigen"] is not None
                assert row["eniv_design_effect"] is not None
            assert len(system["pair_table"]) == n_views * (n_views - 1) // 2


def test_label_meaning_is_in_every_direction_record_and_the_summary(results):
    needle = "was the exfiltration tool running in this window"
    assert needle in results["label_meaning"]
    for name, record in results["directions"].items():
        assert needle in record["label_meaning"], name
        assert needle in record["test_structure"]["label_meaning"], name
    summary = Path("results/ait_clean/summary.md").read_text(encoding="utf-8")
    assert needle in summary
    for banned in ("APT detection", "campaign detection"):
        assert banned not in summary


def test_multi_step_intrusion_is_a_handful_of_windows(results):
    for name, record in results["directions"].items():
        structure = record["test_structure"]
        assert structure["n_multi_step_intrusion_windows"] <= 5, name
        assert "handful-of-windows" in structure["multi_step_caveat"]
        assert len(structure["multi_step_intrusion_windows"]) == (
            structure["n_multi_step_intrusion_windows"]
        )


def test_exfiltration_tool_steps_extend_the_dnsteal_family():
    assert set(DNSTEAL_FAMILY) < set(EXFILTRATION_TOOL_STEPS)
    assert "exfiltration-service" in set(EXFILTRATION_TOOL_STEPS) - set(DNSTEAL_FAMILY)


# ---------------------------------------------------------------------------
# labels == -1 is never scored
# ---------------------------------------------------------------------------


def test_no_metric_is_computed_over_unlabelled_windows(results):
    """n and the class counts must account for every scored sample.

    `n_attack + n_benign == n_test` leaves no room for a labels == -1 row: an
    unlabelled window would raise n_test without raising either class count.
    """
    for name, record in results["directions"].items():
        structure = record["test_structure"]
        assert structure["n_attack"] + structure["n_benign"] == structure["n_label_known"], name
        assert structure["n_unlabelled"] > 0, (
            f"{name}: this scenario has unlabelled windows; if this count is 0 the "
            f"test below proves nothing"
        )
        assert (
            structure["n_label_known"] + structure["n_unlabelled"]
            == structure["n_windows_on_grid"]
        )
        for system in record["systems"].values():
            for row in system["per_seed"]:
                assert row["n_attack"] + row["n_benign"] == row["n_test"], name
                assert row["n_test"] == structure["n_label_known"]
                assert row["n_test"] < structure["n_windows_on_grid"]


def test_training_set_also_excludes_unlabelled_windows(results):
    for name, record in results["directions"].items():
        assert record["n_train_attack"] + record["n_train_benign"] == record["n_train"], name


def test_part28_extras_would_see_an_unlabelled_row_as_neither_class():
    """If -1 ever leaked in, the class counts stop summing to n -- the check above."""
    labels = np.array([1, 1, 0, LABEL_UNKNOWN])
    probs = np.tile([0.3, 0.7], (4, 1))
    extras = part28_extras(labels, probs)
    assert extras["n_attack"] + extras["n_benign"] < extras["n_test"]


# ---------------------------------------------------------------------------
# Pairing, summarising, and the sensor-group check
# ---------------------------------------------------------------------------


def test_paired_difference_is_within_seed_and_keeps_the_signs():
    treatment = [{"seed": 0, "m": 1.0}, {"seed": 1, "m": 0.0}]
    control = [{"seed": 1, "m": 1.0}, {"seed": 0, "m": 0.0}]
    out = paired_difference(treatment, control, lower_is_better=set())
    assert out["m"]["per_seed"] == [1.0, -1.0]
    assert out["m"]["mean"] == pytest.approx(0.0)
    assert out["m"]["n_seeds_favouring_prismflow"] == 1, "a cancelling mean must show its split"


def test_paired_difference_reads_the_sign_for_lower_is_better():
    treatment = [{"seed": s, "brier": 0.1} for s in range(5)]
    control = [{"seed": s, "brier": 0.2} for s in range(5)]
    out = paired_difference(treatment, control, LOWER_IS_BETTER)
    assert out["brier"]["mean"] == pytest.approx(-0.1)
    assert out["brier"]["n_seeds_favouring_prismflow"] == 5
    assert out["brier"]["lower_is_better"]


def test_results_paired_table_is_within_direction_only(results):
    for name, record in results["directions"].items():
        paired = record["paired_prismflow_minus_naive"]
        for metric in REQUIRED_METRICS:
            entry = paired["metrics"][metric]
            assert entry["n_seeds"] == len(results["seeds"]), f"{name}/{metric}"
            assert 0 <= entry["n_seeds_favouring_prismflow"] <= entry["n_seeds"]


def test_summarise_uses_sample_std_and_counts_seeds():
    rows = [{"m": 1.0}, {"m": 2.0}, {"m": 3.0}, {"m": float("nan")}]
    out = summarise(rows)
    assert out["m"]["mean"] == pytest.approx(2.0)
    assert out["m"]["std"] == pytest.approx(1.0)  # ddof = 1
    assert out["m"]["n_seeds"] == 3


def test_sensor_group_check_reports_not_computable_with_one_member(cfg):
    sources = ["apache", "auth", "dns", "audit", "suricata"]
    check = sensor_group_check(sources, cfg)
    assert check["group_members"] == ["suricata"]
    assert check["n_members"] == 1
    assert not check["computable"]
    assert check["status"] == "NOT COMPUTABLE"
    assert "no within-group PAIR exists" in check["reason"]
    assert check["unrelated_pairs_mean"] is None


def test_sensor_group_check_is_recorded_in_the_results(results):
    for name, record in results["directions"].items():
        check = record["sensor_group_check"]
        assert check["requested_group"] == "network", name
        assert check["status"] in ("computed", "NOT COMPUTABLE")
        if not check["computable"]:
            assert check["reason"]


def test_missingness_is_reported_for_train_and_test_per_direction(results, cfg):
    threshold = cfg["missingness"]["threshold"]
    for name, record in results["directions"].items():
        block = record["missingness"]
        assert block["threshold"] == pytest.approx(threshold)
        for which in ("train", "test"):
            assert set(block[which]) == set(results["sources"]), f"{name}/{which}"
            assert all(0.0 <= v <= 1.0 for v in block[which].values())
        over = {s for s in results["sources"]
                if block["train"][s] > threshold or block["test"][s] > threshold}
        assert set(block["views_over_threshold"]) == over, name
        assert block["second_configuration_run"] == bool(over), name
