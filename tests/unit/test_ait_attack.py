"""Part 29 tests.

The expensive thing here is a trained model, so the tests that need one share
a single session-scoped pair (naive and PrismFlow, seed 0, direction A) and
assert against it. The rest are pure-function tests on the honesty guards: the
dns split, the subset enumeration, "not measured" in place of 0, and the
imported-metric rule.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from experiments.ait_attack import run_ait_attack as part29
from experiments.ait_clean import run_ait_clean as part28

CONFIG = yaml.safe_load(part29.CONFIG_PATH.read_text())
SOURCES = ("apache", "auth", "dns", "audit", "suricata")
PART28_PATH = Path("results/ait_clean/results.json")
PROCESSED = Path(CONFIG["data"]["processed_root"])


def _processed_missing() -> bool:
    data = CONFIG["data"]
    for scenario in ("russellmitchell", "santos"):
        name = f"ait_{scenario}_{data['sample_mode']}_{data['window_seconds']}s"
        if not (PROCESSED / name).is_dir():
            return True
    return False


needs_data = pytest.mark.skipif(
    _processed_missing(), reason="Part 26 processed scenarios are not present"
)
needs_part28 = pytest.mark.skipif(
    not PART28_PATH.is_file(), reason="Part 28 results are not present"
)


# ---------------------------------------------------------------------------
# The dns split: the rule L17 imposes, enforced rather than documented
# ---------------------------------------------------------------------------


def test_dns_index_is_found_by_name_not_hard_coded():
    assert part29.dns_index(SOURCES) == 2
    # A reordered view list must still find dns, or every split below is wrong.
    reordered = ("dns", "apache", "auth", "audit", "suricata")
    assert part29.dns_index(reordered) == 0


def test_has_dns_tags_each_subset_correctly():
    assert part29.has_dns((2,), SOURCES) is True
    assert part29.has_dns((0, 1), SOURCES) is False
    assert part29.has_dns((0, 2, 4), SOURCES) is True


def test_assert_no_pooling_rejects_a_mixed_list():
    mixed = [{"dns_compromised": True}, {"dns_compromised": False}]
    with pytest.raises(AssertionError, match="BOTH"):
        part29.assert_no_pooling(mixed, "test")


def test_assert_no_pooling_allows_a_single_side():
    part29.assert_no_pooling([{"dns_compromised": True}] * 3, "test")
    part29.assert_no_pooling([{"dns_compromised": False}] * 3, "test")


def test_enumerate_subsets_is_exhaustive_and_split_counts_match_the_brief():
    subsets = part29.enumerate_subsets(SOURCES, [1, 2, 3])
    assert len(subsets) == 25
    per_k = {}
    for entry in subsets:
        per_k.setdefault(entry["k"], []).append(entry)
    assert {k: len(v) for k, v in per_k.items()} == {1: 5, 2: 10, 3: 10}
    # The counts the Part 29 brief states, checked rather than trusted.
    expected = {1: (1, 4), 2: (4, 6), 3: (6, 4)}
    for k, (with_dns, without) in expected.items():
        rows = per_k[k]
        assert sum(1 for r in rows if r["dns_compromised"]) == with_dns
        assert sum(1 for r in rows if not r["dns_compromised"]) == without
    # No duplicates, and every subset is a real combination of view indices.
    assert len({entry["subset"] for entry in subsets}) == 25


def test_every_aggregated_group_is_one_dns_side_only():
    """The guard is called on real cell shapes, not just on hand-made dicts."""
    cells = {}
    for entry in part29.enumerate_subsets(SOURCES, [1, 2]):
        key = part29.cell_key("chorus", entry, 1.0, 1.0)
        cells[key] = {
            "kind": "chorus", "k": entry["k"], "epsilon": 1.0, "beta": 1.0,
            "dns_compromised": entry["dns_compromised"],
            "per_seed": [
                {
                    "seed": seed, "dns_compromised": entry["dns_compromised"],
                    "k": entry["k"], "kind": "chorus", "epsilon": 1.0, "beta": 1.0,
                    "compromised_views": entry["names"], "attack_success": 0.1,
                }
                for seed in range(5)
            ],
        }
    grouped = part29.aggregate_by_dns_side({"cells": cells}, CONFIG)
    assert grouped, "no groups were produced"
    for label, group in grouped.items():
        side = group["dns_compromised"]
        assert label.endswith("dns_compromised") or label.endswith("dns_not_compromised")
        assert (side and label.endswith("__dns_compromised")) or (
            not side and label.endswith("__dns_not_compromised")
        )
        for views in group["per_subset"]:
            assert ("dns" in views.split("+")) == side


# ---------------------------------------------------------------------------
# "not measured", never 0
# ---------------------------------------------------------------------------


def test_mean_within_subset_is_nan_at_k1_not_zero():
    matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
    value = part29.mean_within_subset(matrix, (0,))
    assert math.isnan(value), "k=1 has no within-pair; 0.0 would claim it measured 0"


def test_mean_within_subset_averages_only_within_pairs():
    matrix = np.array([
        [1.0, 0.2, 0.4],
        [0.2, 1.0, 0.8],
        [0.4, 0.8, 1.0],
    ])
    assert part29.mean_within_subset(matrix, (1, 2)) == pytest.approx(0.8)
    assert part29.mean_within_subset(matrix, (0, 1, 2)) == pytest.approx(
        (0.2 + 0.4 + 0.8) / 3
    )


def test_summarise_rows_marks_an_all_nan_key_not_measured():
    rows = [{"x": float("nan")}, {"x": float("nan")}]
    out = part29.summarise_rows(rows, ("x",))
    assert out["x"]["status"] == part29.NOT_MEASURED
    assert out["x"]["n_seeds"] == 0
    assert math.isnan(out["x"]["mean"])


def test_summarise_rows_reports_std_only_with_two_or_more_seeds():
    one = part29.summarise_rows([{"x": 1.0}], ("x",))
    assert math.isnan(one["x"]["std"]) and one["x"]["n_seeds"] == 1
    many = part29.summarise_rows([{"x": 1.0}, {"x": 3.0}], ("x",))
    assert many["x"]["mean"] == pytest.approx(2.0)
    assert many["x"]["std"] == pytest.approx(math.sqrt(2.0))  # ddof=1


def test_json_safe_writes_nan_as_null_never_zero():
    assert part29.json_safe(float("nan")) is None
    assert part29.json_safe({"a": float("nan"), "b": 0.0}) == {"a": None, "b": 0.0}


# ---------------------------------------------------------------------------
# Metrics are imported from Part 28, not reimplemented
# ---------------------------------------------------------------------------


def test_part29_imports_part28_extras_and_does_not_define_its_own():
    assert part29.part28_extras is part28.part28_extras
    assert part29.roc_auc_binary is part28.roc_auc_binary
    source = part29.__file__
    text = Path(source).read_text(encoding="utf-8")
    for forbidden in ("def balanced_accuracy", "def part29_extras", "def roc_auc("):
        assert forbidden not in text, (
            f"{forbidden!r} appears in run_ait_attack.py; "
            f"docs/EVALUATION_PROTOCOL.md requires Part 29 to share part28_extras"
        )


def test_detector_auc_goes_through_the_imported_roc_auc():
    """A perfectly separated pair gives 1.0, a reversed one 0.0, a tie 0.5."""
    assert part29.detector_auc([3.0, 4.0], [1.0, 2.0]) == pytest.approx(1.0)
    assert part29.detector_auc([1.0, 2.0], [3.0, 4.0]) == pytest.approx(0.0)
    assert part29.detector_auc([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)


def test_detector_auc_drops_nan_and_reports_nan_when_nothing_is_left():
    assert part29.detector_auc([np.nan, 2.0], [1.0]) == pytest.approx(1.0)
    assert math.isnan(part29.detector_auc([np.nan], [1.0]))
    assert math.isnan(part29.detector_auc([], [1.0]))


def test_detector_auc_matches_roc_auc_binary_called_directly():
    rng = np.random.default_rng(0)
    attacked, clean = rng.normal(1.0, 1.0, 50), rng.normal(0.0, 1.0, 70)
    labels = np.concatenate([np.ones(50, dtype=int), np.zeros(70, dtype=int)])
    direct = part28.roc_auc_binary(labels, np.concatenate([attacked, clean]))
    assert part29.detector_auc(attacked, clean) == pytest.approx(direct)


# ---------------------------------------------------------------------------
# Test A: the window definition, and the handful caveat
# ---------------------------------------------------------------------------


def test_the_strict_definition_is_the_one_used():
    """EXFILTRATION_TOOL_STEPS, not DNSTEAL_FAMILY, defines `multi_step`."""
    assert CONFIG["natural_windows"]["multi_step_definition"] == (
        "strict_beyond_exfiltration_tool_steps"
    )
    assert set(part28.DNSTEAL_FAMILY) < set(part28.EXFILTRATION_TOOL_STEPS)
    assert "exfiltration-service" in part28.EXFILTRATION_TOOL_STEPS


def test_handful_caveat_states_both_counts():
    assert "3 windows on santos" in part29.HANDFUL_CAVEAT
    assert "4 on russellmitchell" in part29.HANDFUL_CAVEAT
    assert "descriptive only" in part29.HANDFUL_CAVEAT


@needs_data
def test_window_populations_match_part28s_committed_counts(prepared_a):
    """The strict count must be 3 on santos, and the looser one 4, from Part 28."""
    populations = part29.window_populations(prepared_a, "santos")
    assert populations["n_multi_step"] == 3
    assert populations["n_attack_beyond_dnsteal_family_looser_count"] == 4
    # Exactly one window per scenario is beyond the dnsteal family without
    # being a multi-step step: the exfiltration tool's own service record,
    # which is what the two definitions differ by. It is reported as its own
    # population so nothing vanishes between the two.
    assert populations["n_exfiltration_service_only"] == 1

    # The FOUR populations partition the labelled test rows with no overlap.
    names = ("benign", "dnsteal_only", "multi_step", "exfiltration_service_only")
    total = sum(populations[f"n_{name}"] for name in names)
    assert total == prepared_a["test_index"].size
    for left, right in itertools.combinations(names, 2):
        assert not set(populations[left].tolist()) & set(populations[right].tolist())


@needs_data
@needs_part28
def test_strict_count_agrees_with_part28_results_json(prepared_a):
    part28_results = json.loads(PART28_PATH.read_text())
    committed = part28_results["directions"]["A_rm_to_santos"]["test_structure"]
    populations = part29.window_populations(prepared_a, "santos")
    assert populations["n_multi_step"] == committed["n_multi_step_intrusion_windows"]
    assert (
        populations["n_attack_beyond_dnsteal_family_looser_count"]
        == committed["n_attack_windows_beyond_dnsteal_family"]
    )


# ---------------------------------------------------------------------------
# Test C: NO EVIDENCE for same-sensor, and the substitute is labelled
# ---------------------------------------------------------------------------


def test_sensor_group_network_still_has_one_member():
    from prismflow.data.ait_lds import sensor_groups

    groups = sensor_groups()
    assert groups.get("network") == [4], (
        "the 'network' group gained a member; test C's NO EVIDENCE verdict and "
        "Part 28's sensor_group_check would both need revisiting"
    )


@needs_part28
def test_ranked_pairs_are_read_from_part28_and_match_the_config():
    part28_results = json.loads(PART28_PATH.read_text())
    for direction in ("A_rm_to_santos", "B_santos_to_rm"):
        pairs = part29.ranked_pairs(part28_results, direction, CONFIG)
        assert pairs["most_dependent"] == CONFIG["natural_dependence"]["expected_most_dependent"][direction]
        assert pairs["least_dependent"] == CONFIG["natural_dependence"]["expected_least_dependent"][direction]
        assert pairs["label"] == part29.NATURAL_DEPENDENCE_LABEL


@needs_part28
def test_ranked_pairs_stops_when_part28s_table_no_longer_agrees():
    """A changed pair table must stop the run, not be silently reported."""
    part28_results = json.loads(PART28_PATH.read_text())
    tampered = json.loads(json.dumps(CONFIG))
    tampered["natural_dependence"]["expected_most_dependent"]["A_rm_to_santos"] = ["apache-auth", "auth-audit"]
    with pytest.raises(SystemExit, match="frozen"):
        part29.ranked_pairs(part28_results, "A_rm_to_santos", tampered)


def test_natural_dependence_label_says_different_sensors():
    label = part29.NATURAL_DEPENDENCE_LABEL
    assert "different sensors" in label and "naturally dependent" in label
    assert "same" not in label


def test_natural_dependence_block_reports_not_measured_without_excess():
    out = part29.natural_dependence_block(None, {}, SOURCES, CONFIG)
    assert out["status"] == part29.NOT_MEASURED


def test_pair_excess_reads_the_named_pair_from_the_matrix():
    excess = np.zeros((3, 5, 5))
    i, j = SOURCES.index("auth"), SOURCES.index("dns")
    excess[:, i, j] = [0.1, 0.2, 0.3]
    got = part29.pair_excess_scores(excess, "auth-dns", SOURCES)
    assert got == pytest.approx([0.1, 0.2, 0.3])


def test_natural_dependence_block_is_flagged_as_not_clone_k2():
    """L5's clone_k2 AUC is a different construction and must not be conflated."""
    rng = np.random.default_rng(0)
    excess = rng.normal(size=(40, 5, 5))
    pairs = {
        "most_dependent": ["dns-suricata", "auth-audit"],
        "least_dependent": ["auth-dns", "apache-dns"],
        "per_system": {"naive": {"values": {
            p: {"mean": 0.5, "std": 0.01}
            for p in ("dns-suricata", "auth-audit", "auth-dns", "apache-dns")
        }}},
    }
    out = part29.natural_dependence_block(excess, pairs, SOURCES, CONFIG)
    assert out["not_the_same_as_clone_k2"] is True
    assert "clone_k2" in out["construction"]
    assert out["label"] == part29.NATURAL_DEPENDENCE_LABEL
    assert 0.0 <= out["auc_most_over_least"] <= 1.0


# ---------------------------------------------------------------------------
# Epsilon: not rescaled, and the config says so with its reason
# ---------------------------------------------------------------------------


def test_epsilon_grid_is_part09s_and_is_declared_unrescaled():
    assert CONFIG["attack"]["epsilon_rescaled"] is False
    assert CONFIG["attack"]["epsilon_grid"] == [0.2, 0.5, 1.0, 2.0]
    assert CONFIG["attack"]["beta_grid"] == [0.0, 1.0, 5.0]
    assert CONFIG["attack"]["epsilon_reference_std_synthetic"] == 1.3


@needs_data
def test_feature_scale_is_measured_per_view_and_brackets_the_reference(prepared_a):
    scale = part29.feature_scale(prepared_a)
    assert set(scale["per_view"]) == set(SOURCES)
    for source, record in scale["per_view"].items():
        assert record["std"] != part29.NOT_MEASURED
        assert record["n_present"] > 0
    assert scale["min_per_view_std"] <= scale["median_per_view_std"] <= scale["max_per_view_std"]
    assert scale["synthetic_reference_std"] == 1.3


# ---------------------------------------------------------------------------
# The detector is undefined for naive, and that reads "not measured"
# ---------------------------------------------------------------------------


@needs_data
def test_naive_has_no_detector_and_prismflow_does(models_a):
    """Verified against the real models: naive produces no SuspicionReport."""
    from prismflow.models.defended import DefendedPrismFlow

    prepared, naive, prismflow = models_a
    batch = next(iter(part29.iter_batches(prepared["test"], prepared["test_index"], 256)))
    for model, expect_detector in ((naive, False), (prismflow, True)):
        defended = DefendedPrismFlow(model, suspicion_permutations=4, suspicion_seed=0)
        with torch.no_grad():
            out = defended(batch.views, batch.view_mask)
        assert (out.suspicion is not None) is expect_detector
    assert "not" in part29.NO_DETECTOR_REASON and "0" not in part29.NO_DETECTOR_REASON


@needs_data
def test_naive_detector_scores_are_nan_not_zero(models_a):
    from prismflow.models.defended import DefendedPrismFlow

    prepared, naive, _ = models_a
    defended = DefendedPrismFlow(naive, suspicion_permutations=4, suspicion_seed=0)
    cell = part29.run_cell(
        defended, prepared["test"], prepared["test_index"][:300], CONFIG, kind="clean", seed=0
    )
    assert np.isnan(cell["_scores"]).all(), "a 0 here would claim the detector never fired"
    assert math.isnan(cell["suspicion_mean_score"])
    assert cell["_excess"] is None


# ---------------------------------------------------------------------------
# The dependence wrapper is pinned to Part 28's committed numbers
# ---------------------------------------------------------------------------


@needs_data
@needs_part28
def test_clean_dependence_matches_part28(models_a):
    """`dependence_of` must reproduce Part 28's committed matrix on clean input.

    This is what licenses using it on PERTURBED input: the wrapper is the same
    frozen estimator at the same reporting-grade permutation count, not a
    lookalike.
    """
    prepared, _, prismflow = models_a
    index = prepared["test_index"]
    views = prepared["test"].views[index]
    view_mask = prepared["test"].view_mask[index]
    got = part29.dependence_of(prismflow, views, view_mask, CONFIG)

    part28_results = json.loads(PART28_PATH.read_text())
    record = part28_results["directions"]["A_rm_to_santos"]["systems"]["prismflow"]
    # ENIV lives in `diagnostics_per_seed`, not `per_seed`, in Part 28's file.
    seed_zero = [r for r in record["diagnostics_per_seed"] if r["seed"] == 0]
    assert seed_zero, "Part 28 has no seed-0 diagnostics row for direction A / prismflow"
    assert got["eniv_eigen"] == pytest.approx(seed_zero[0]["eniv_eigen"], abs=1e-6)
    assert got["mean_dependence_eigen"] == pytest.approx(
        seed_zero[0]["mean_dependence_eigen"], abs=1e-6
    )


@needs_data
def test_attack_success_is_zero_on_clean_by_construction(models_a):
    """The target is the least-likely class, so a clean cell cannot score above 0."""
    from prismflow.models.defended import DefendedPrismFlow

    prepared, _, prismflow = models_a
    defended = DefendedPrismFlow(prismflow, suspicion_permutations=4, suspicion_seed=0)
    index = prepared["test_index"][:300]
    zero = part29.run_cell(
        defended, prepared["test"], index, CONFIG,
        kind="chorus", subset=(2,), epsilon=0.0, beta=1.0, seed=0,
    )
    assert zero["attack_success"] == pytest.approx(0.0)


@needs_data
def test_a_real_attack_moves_predictions_and_a_bigger_budget_moves_them_more(models_a):
    from prismflow.models.defended import DefendedPrismFlow

    prepared, _, prismflow = models_a
    defended = DefendedPrismFlow(prismflow, suspicion_permutations=4, suspicion_seed=0)
    index = prepared["test_index"][:300]
    small = part29.run_cell(
        defended, prepared["test"], index, CONFIG,
        kind="chorus", subset=(2,), epsilon=0.2, beta=1.0, seed=0,
    )
    large = part29.run_cell(
        defended, prepared["test"], index, CONFIG,
        kind="chorus", subset=(2,), epsilon=2.0, beta=1.0, seed=0,
    )
    assert 0.0 <= small["attack_success"] <= 1.0
    assert large["attack_success"] >= small["attack_success"], (
        "a larger L-inf budget cannot reduce success for a backtracking attack "
        "that never accepts a worsening step"
    )


# ---------------------------------------------------------------------------
# Fixtures. One trained pair, shared.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prepared_a():
    cache: dict = {}
    direction = {"name": "A_rm_to_santos", "train": "russellmitchell", "test": "santos"}
    return part29.prepare_direction(direction, CONFIG, cache)


@pytest.fixture(scope="module")
def models_a(prepared_a):
    naive = part28.train_ait(prepared_a["train"], prepared_a["train_index"], 0, False, CONFIG)
    prismflow = part28.train_ait(prepared_a["train"], prepared_a["train_index"], 0, True, CONFIG)
    return prepared_a, naive, prismflow


# ---------------------------------------------------------------------------
# The grid: exhaustive, with only provably-duplicate cells omitted
# ---------------------------------------------------------------------------


def test_attack_grid_is_part09s_grid():
    k2 = part29.attack_grid(2, CONFIG)
    chorus_eps = sorted(e for kind, e, b in k2 if kind == "chorus" and b == 1.0)
    assert chorus_eps == [0.2, 0.5, 1.0, 2.0], "Part 09's epsilon sweep must be carried over"
    betas = sorted(b for kind, e, b in k2 if kind == "chorus" and e == 1.0)
    assert betas == [0.0, 1.0, 5.0], "Part 09's beta grid must be carried over at k >= 2"
    pgd_eps = sorted(e for kind, e, b in k2 if kind == "pgd")
    assert pgd_eps == [1.0, 2.0]


def test_attack_grid_omits_only_duplicate_cells_at_k1():
    k1 = part29.attack_grid(1, CONFIG)
    # beta is inert at k=1, so only the primary beta appears.
    assert {b for kind, e, b in k1 if kind == "chorus"} == {1.0}
    # One pgd cell is kept so the chorus/pgd identity is visible in results.
    assert [(kind, e) for kind, e, b in k1 if kind == "pgd"] == [("pgd", 1.0)]
    assert len(k1) == 5


def test_total_attack_count_is_185_exhaustive():
    subsets = part29.enumerate_subsets(SOURCES, CONFIG["attack"]["k_grid"])
    total = sum(len(part29.attack_grid(entry["k"], CONFIG)) for entry in subsets)
    assert total == 185, (
        "the grid changed; the README's cost statement and the "
        "subset_sampling record must be updated together with it"
    )
    assert CONFIG["attack"]["enumerate_all_subsets"] is True


def test_every_k_and_dns_side_cell_is_non_empty():
    """The brief forbids leaving a with-dns cell empty."""
    subsets = part29.enumerate_subsets(SOURCES, CONFIG["attack"]["k_grid"])
    for k in CONFIG["attack"]["k_grid"]:
        for side in (True, False):
            rows = [s for s in subsets if s["k"] == k and s["dns_compromised"] is side]
            assert rows, f"(k={k}, dns={side}) is empty"


# ---------------------------------------------------------------------------
# Degenerate ENIV must not be reported as perfect independence
# ---------------------------------------------------------------------------


@needs_data
def test_three_window_population_is_flagged_degenerate(models_a, prepared_a):
    """On the 3 multi-step windows the estimator returns an all-zero matrix.

    ENIV then pins at the view count (5.0) and efficiency at 1.0. Those are
    not measurements of independence, and the flag must say so.
    """
    _, _, prismflow = models_a
    populations = part29.window_populations(prepared_a, "santos")
    rows = populations["multi_step"]
    assert rows.size == 3
    views = prepared_a["test"].views[rows]
    view_mask = prepared_a["test"].view_mask[rows]
    record = part29.dependence_of(prismflow, views, view_mask, CONFIG)
    assert record["dependence_degenerate"] is True
    assert record["n_samples"] == 3
    assert part29.NOT_MEASURED in record["degenerate_reason"]
    # The raw value is kept, but only alongside the flag.
    assert record["eniv_eigen"] == pytest.approx(5.0)


@needs_data
def test_full_test_set_is_not_flagged_degenerate(models_a, prepared_a):
    _, _, prismflow = models_a
    index = prepared_a["test_index"]
    record = part29.dependence_of(
        prismflow, prepared_a["test"].views[index], prepared_a["test"].view_mask[index], CONFIG
    )
    assert record["dependence_degenerate"] is False
    assert record["degenerate_reason"] is None
    assert 1.0 < record["eniv_eigen"] < 5.0


def test_summarise_rows_never_averages_a_boolean_flag():
    rows = [{"dependence_degenerate": True}, {"dependence_degenerate": False}]
    out = part29.summarise_rows(rows, ("dependence_degenerate",))
    assert out["dependence_degenerate"]["n_seeds"] == 0
    assert out["dependence_degenerate"]["status"] == part29.NOT_MEASURED


# ---------------------------------------------------------------------------
# chorus == pgd at k = 1, the identity that licenses omitting a cell
# ---------------------------------------------------------------------------


@needs_data
def test_chorus_equals_pgd_at_k1(models_a, prepared_a):
    """With one compromised view the agreement term has no within-pair, so the
    chorus objective separates per view and reduces to independent PGD."""
    from prismflow.models.defended import DefendedPrismFlow

    _, _, prismflow = models_a
    defended = DefendedPrismFlow(prismflow, suspicion_permutations=4, suspicion_seed=0)
    index = prepared_a["test_index"][:400]
    shared = dict(subset=(2,), epsilon=1.0, beta=1.0, seed=0)
    chorus = part29.run_cell(defended, prepared_a["test"], index, CONFIG, kind="chorus", **shared)
    pgd = part29.run_cell(defended, prepared_a["test"], index, CONFIG, kind="pgd", **shared)
    assert chorus["attack_success"] == pytest.approx(pgd["attack_success"], abs=1e-12)


@needs_data
def test_chorus_differs_from_pgd_at_k2(models_a, prepared_a):
    """The agreement term must actually do something once there is a pair."""
    from prismflow.models.defended import DefendedPrismFlow

    _, _, prismflow = models_a
    defended = DefendedPrismFlow(prismflow, suspicion_permutations=4, suspicion_seed=0)
    index = prepared_a["test_index"]
    beta_on = part29.run_cell(
        defended, prepared_a["test"], index, CONFIG,
        kind="chorus", subset=(2, 4), epsilon=1.0, beta=1.0, seed=0,
    )
    beta_off = part29.run_cell(
        defended, prepared_a["test"], index, CONFIG,
        kind="chorus", subset=(2, 4), epsilon=1.0, beta=0.0, seed=0,
    )
    assert beta_on["dependence_compromised"] == pytest.approx(
        beta_on["dependence_compromised"]
    )
    assert not math.isnan(beta_on["dependence_compromised"]), "k=2 has a within-pair"
    assert beta_on["attack_success"] != beta_off["attack_success"]


# ---------------------------------------------------------------------------
# Section 11's two gates on the paired-difference rows (make_readme.py).
# ---------------------------------------------------------------------------

def test_sign_consistency_bar_is_unreachable_at_five_seeds():
    """The |mean| > 2*std rule CANNOT fail at n=5, and that is why it is
    labelled an invariant in the README rather than a check that passed.

    Minimise the population std of a 5-vector whose mean is 1 subject to one
    entry not agreeing with that sign. The optimum puts the odd seed at 0 and
    the other four at 1.25, giving std exactly 0.5 and |mean|/std exactly 2.0,
    so a strict `>` can never be satisfied by a non-unanimous row.
    """
    from experiments.ait_attack.make_readme import (
        SIGN_CONSISTENCY_STDS, paired_seeds)

    worst = paired_seeds({i: v for i, v in enumerate([1.25] * 4 + [0.0])},
                         {i: 0.0 for i in range(5)})
    assert worst.mean == pytest.approx(1.0)
    assert worst.std == pytest.approx(0.5)
    assert abs(worst.mean) / worst.std == pytest.approx(SIGN_CONSISTENCY_STDS)
    assert not worst.unanimous, "the odd seed is exactly 0, so it agrees with neither sign"

    # Nothing non-unanimous can beat it: sample the constrained space.
    rng = np.random.default_rng(29)
    for _ in range(20000):
        d = rng.normal(0.0, 1.0, 5)
        mean, std = d.mean(), d.std()
        if std == 0 or mean == 0:
            continue
        agree = (d > 0).sum() if mean > 0 else (d < 0).sum()
        if agree == 5:
            continue
        assert abs(mean) / std <= SIGN_CONSISTENCY_STDS + 1e-9


def test_sign_consistency_fires_when_it_can():
    """With the bar lowered it must raise and name the row, not warn."""
    from experiments.ait_attack import make_readme

    p = make_readme.paired_seeds({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 0.0},
                                 {i: 0.0 for i in range(5)})
    row = ("DIR", "SYS", "apache+dns", "synthetic", p)
    make_readme.check_sign_consistency([row])          # passes at 2.0
    original = make_readme.SIGN_CONSISTENCY_STDS
    try:
        make_readme.SIGN_CONSISTENCY_STDS = 1.5
        with pytest.raises(make_readme.SignConsistencyError) as excinfo:
            make_readme.check_sign_consistency([row])
    finally:
        make_readme.SIGN_CONSISTENCY_STDS = original
    message = str(excinfo.value)
    assert "apache+dns" in message and "README NOT written" in message
    assert "per-seed diffs=" in message, "the offending row must be printed"


def test_row_wiring_check_catches_a_mismatched_aggregate():
    """check_wiring is the falsifiable half: a printed mean that does not
    recompute from the row's own per-seed diffs must stop the write."""
    from experiments.ait_attack import make_readme

    good = make_readme.paired_seeds({0: 1.0, 1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0},
                                    {i: 0.0 for i in range(5)})
    make_readme.check_wiring([("DIR", "SYS", "a+b", "col", good)])

    for bad in (good._replace(mean=0.5),
                good._replace(text="+9.9999 +/- 0.0000"),
                good._replace(agree=3),
                good._replace(diffs=(1.0, 1.0, 1.0))):
        with pytest.raises(make_readme.RowWiringError):
            make_readme.check_wiring([("DIR", "SYS", "a+b", "col", bad)])


def test_paired_count_discloses_exact_zeros():
    """A seed whose difference is exactly 0 agrees with neither sign, so the
    count must say so: a 1/5 is not four seeds contradicting the mean."""
    from experiments.ait_attack.make_readme import paired_seeds

    p = paired_seeds({0: 0.0, 1: 0.0, 2: -1.0, 3: 0.0, 4: 0.0},
                     {i: 0.0 for i in range(5)})
    assert p.agree == 1 and p.zeros == 4 and p.opposite == 0
    assert p.count == "1/5 (4 exactly 0)"

    allzero = paired_seeds({i: 0.0 for i in range(5)},
                           {i: 0.0 for i in range(5)})
    assert allzero.count == "5/5 (all exactly 0)"

    split = paired_seeds({0: 1.0, 1: 1.0, 2: 1.0, 3: -1.0, 4: -1.0},
                         {i: 0.0 for i in range(5)})
    assert split.opposite == 2 and split.zeros == 0
    assert split.count == "3/5", "a genuine sign split carries no zero note"


# ---------------------------------------------------------------------------
# L3 headline logic (make_readme.py): sign sets and the dependence clause.
# ---------------------------------------------------------------------------

def _l3cut(direction, dns, cells):
    """An L3Cut carrying `cells` as (key, mean, unanimous) triples."""
    from experiments.ait_attack.make_readme import L3Cut

    hi = [c for c in cells if c[1] > 0]
    lo = [c for c in cells if c[1] < 0]
    return L3Cut(direction, dns, len(cells),
                 len(lo), sum(1 for c in lo if c[2]),
                 len(hi), sum(1 for c in hi if c[2]),
                 min(c[1] for c in cells), max(c[1] for c in cells),
                 tuple(cells))


def test_l3_dependence_clause_three_branches():
    """The clause must follow the rule, with no fourth outcome."""
    from experiments.ait_attack.make_readme import l3_dependence_clause as cl

    # Differing sign sets with at least one single-signed level -> conditional.
    assert cl({True: {"hi"}, False: {"lo"}}, "X") == "conditional on X"
    assert cl({True: {"hi"}, False: {"hi", "lo"}}, "X") == "conditional on X"
    # Every level holds both signs -> not determined by X alone.
    assert cl({True: {"hi", "lo"}, False: {"hi", "lo"}}, "X") == \
        "not determined by X alone"
    # Neither statement supported: identical single-signed levels, an empty
    # level beside a two-signed one, or a single level.
    assert cl({True: {"hi"}, False: {"hi"}}, "X") is None
    assert cl({True: set(), False: {"hi", "lo"}}, "X") is None
    assert cl({True: {"hi", "lo"}}, "X") is None
    assert cl({}, "X") is None


def test_l3_sign_sets_union_over_cells_not_last_cut():
    """A level's sign set is the union over its cells.

    Regression guard: a dict comprehension over the per-group L3Cut rows keeps
    only the LAST row per level, which reported one group's signs as the whole
    level's and produced a wrong "conditional on" clause.
    """
    from experiments.ait_attack.make_readme import (
        l3_distinct_cells, l3_sign_sets)

    cuts = {
        # Same dns side, same direction, two groups: one all-higher, one
        # all-lower. The union is both signs; the last cut alone says "lo".
        "a": [_l3cut("DIR", True, [("chorus_k2_eps1.0_beta1.0_[a+b]", +0.5, True)])],
        "d": [_l3cut("DIR", True, [("chorus_k2_eps0.2_beta1.0_[a+b]", -0.5, True)])],
    }
    cells = l3_distinct_cells(cuts)
    assert len(cells) == 2, "distinct cells keyed on (direction, cell_key)"
    assert l3_sign_sets(cells, lambda c: c.dns) == {True: {"hi", "lo"}}
    assert l3_sign_sets(cells, lambda c: c.direction) == {"DIR": {"hi", "lo"}}


def test_l3_distinct_cells_dedup_counts_overlap_once():
    """Group (d) contains group (a); the shared cell must be counted once."""
    from experiments.ait_attack.make_readme import l3_distinct_cells

    shared = ("chorus_k2_eps1.0_beta1.0_[a+b]", +0.5, True)
    cuts = {"a": [_l3cut("DIR", True, [shared])],
            "d": [_l3cut("DIR", True, [shared,
                                       ("chorus_k2_eps2.0_beta1.0_[a+b]", -0.1, True)])]}
    assert len(l3_distinct_cells(cuts)) == 2


def test_l3_headline_states_counts_and_strength_word():
    """The headline must carry the counts and derive the strength word."""
    from experiments.ait_attack.make_readme import l3_headline

    both = {"a": [_l3cut("DIR", True, [("chorus_k2_eps1.0_beta1.0_[a+b]", +0.5, True)]),
                  _l3cut("DIR", False, [("chorus_k2_eps1.0_beta1.0_[a+c]", -0.5, True)])]}
    text = l3_headline(both)
    assert "PARTIALLY reproduced" in text
    assert "(1 higher / 1 lower unanimous, of 2 distinct cells)" in text

    only_hi = {"a": [_l3cut("DIR", True, [("chorus_k2_eps1.0_beta1.0_[a+b]", +0.5, True)])]}
    assert "is reproduced (1 higher / 0 lower unanimous" in l3_headline(only_hi)

    only_lo = {"a": [_l3cut("DIR", True, [("chorus_k2_eps1.0_beta1.0_[a+b]", -0.5, True)])]}
    assert "CONTRADICTED (0 higher / 1 lower unanimous" in l3_headline(only_lo)

    none_u = {"a": [_l3cut("DIR", True, [("chorus_k2_eps1.0_beta1.0_[a+b]", +0.5, False)])]}
    assert "NO EVIDENCE (0 higher / 0 lower unanimous" in l3_headline(none_u)
    assert l3_headline({}).startswith("**Verdict on L3: NO EVIDENCE**")


def test_l3_stratum_bucket_is_total_and_exclusive():
    """Every (dns-YES, dns-NO) sign-set pair lands in exactly one bucket."""
    from experiments.ait_attack.make_readme import (
        L3_STRATUM_BUCKETS, l3_stratum_bucket)

    def cut(signs):
        return _l3cut("DIR", True,
                      [("chorus_k2_eps1.0_beta1.0_[a+b]", +0.5, "hi" in signs),
                       ("chorus_k2_eps1.0_beta1.0_[a+c]", -0.5, "lo" in signs)])

    names = [n for n, _d in L3_STRATUM_BUCKETS]
    assert names == ["flip", "same", "mixed", "undetermined"]

    sets = [set(), {"hi"}, {"lo"}, {"hi", "lo"}]
    expected = {}
    for sy in sets:
        for sn in sets:
            if len(sy) == 2 or len(sn) == 2:
                want = "mixed"
            elif len(sy) == 1 and len(sn) == 1:
                want = "flip" if sy != sn else "same"
            else:
                want = "undetermined"
            expected[(frozenset(sy), frozenset(sn))] = want

    # All 16 ordered pairs of the 4 possible sign sets, exhaustively.
    assert len(expected) == 16
    for (sy, sn), want in expected.items():
        got = l3_stratum_bucket(cut(set(sy)), cut(set(sn)))
        assert got == want, f"{sorted(sy)} vs {sorted(sn)} -> {got}, want {want}"
        assert got in names

    # Each bucket is reached by at least one pair, so none is dead.
    assert set(expected.values()) == set(names)
    # A missing level is `undetermined`, never a crash.
    assert l3_stratum_bucket(None, cut({"hi"})) == "undetermined"
    assert l3_stratum_bucket(None, None) == "undetermined"


def test_l3_strata_counts_sum_to_the_number_of_strata():
    """The four printed counts must account for every stratum."""
    from experiments.ait_attack.make_readme import (
        l3_strata, l3_stratum_bucket, l3_stratified_sentence)
    from experiments.ait_attack.results_io import load_results
    from experiments.ait_attack import make_readme as mr

    results = load_results("results.json")
    if results is None:
        pytest.skip("results/ait_attack/results.json[.gz] not present")
    cuts = {}
    for gkey, _title, extra, pred, value in mr._l3_groups():
        for dname, record in results["directions"].items():
            rows = mr.l3_cells(record, pred)
            for dns in (True, False):
                cut = mr.l3_cut(dname, rows, dns)
                if cut is None:
                    continue
                if extra:
                    cut = cut._replace(extra=f"{extra}={value}")
                cuts.setdefault(gkey, []).append(cut)

    strata = l3_strata(cuts)
    assert len(strata) == 14, "7 group sub-rows x 2 directions"
    for levels in strata.values():
        assert set(levels) == {True, False}, "each stratum holds both dns levels"
    tally = {}
    for key, levels in strata.items():
        bucket = l3_stratum_bucket(levels.get(True), levels.get(False))
        tally[bucket] = tally.get(bucket, 0) + 1
    assert sum(tally.values()) == len(strata)
    text = "\n".join(l3_stratified_sentence(cuts))
    assert f"sum to {len(strata)} of {len(strata)} strata" in text
    assert "gives no mechanism" in text
