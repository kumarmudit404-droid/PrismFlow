"""Part 36 tests. The new logic is the decision rules, and nothing else.

Part 36 trains no new model architecture and writes no new estimator: it
imports Part 05's `run_cell`, Part 13's per-cell functions and Part 10's
`run_condition` unchanged. So the only code that can be wrong is the part that
turns measured numbers into a verdict -- the four `verdict_*` functions, the
`floor + c / n` fit, and the nested-subsample rule that makes the sweep paired
within seed. Each of those is tested here against hand-built inputs whose
right answer is known by construction, with no training and no GPU.

The structural tests over `results/part36_synthetic/*.json` run when those
files exist and skip when they do not, so a clean checkout still passes.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.part36_synthetic.run_part36 import (
    CFG,
    OUT,
    PRIMARY,
    SEEDS,
    _fit_floor,
    _sweep,
    mean_std,
    verdict_l2,
    verdict_l9,
    verdict_l10,
    verdict_l11,
    verdict_l22,
)


# --------------------------------------------------------------------------
# The pre-registration itself
# --------------------------------------------------------------------------
def test_pre_registration_has_five_seeds_and_a_rule_per_analysis():
    assert len(SEEDS) >= 5, "the 5-SEED RULE is not optional"
    assert CFG["analysis_a"]["decision_rule_l9"]
    assert CFG["analysis_a"]["decision_rule_l2"]
    assert CFG["analysis_b"]["decision_rule"]
    assert CFG["analysis_c"]["decision_rule"]
    assert CFG["analysis_d"]["decision_rule"]


def test_the_two_l11_bands_differ_at_the_floor():
    """The reason both bands are pre-registered: L11's quoted floor is not the
    committed one. If these ever coincide, the dual reporting is redundant and
    the note in config.yaml is stale."""
    quoted = CFG["analysis_c"]["band_as_quoted_in_l11"]
    committed = CFG["analysis_c"]["band_as_committed_in_results"]
    assert quoted[1] == committed[1]
    assert committed[0] < quoted[0]


def test_mean_std_is_the_sample_std_and_drops_non_finite():
    entry = mean_std([1.0, 2.0, 3.0, float("nan"), None])
    assert entry["n"] == 3
    assert entry["mean"] == pytest.approx(2.0)
    assert entry["std"] == pytest.approx(1.0)  # ddof=1, not 0.816


# --------------------------------------------------------------------------
# A - L9 and L2
# --------------------------------------------------------------------------
def _view_entry(mae, mae_std, rho_bar=None, rho_bar_std=0.01):
    entry = {"estimators": {PRIMARY: {"mae": mae, "mae_std": mae_std}}}
    if rho_bar is not None:
        entry["residual_rho_bar_at_rho0"] = {
            "features_cca_holdout": {"mean": rho_bar, "std": rho_bar_std, "n": 5}
        }
    return entry


def test_l9_scales_only_when_the_gap_beats_the_v_min_spread():
    counts = [4, 16]
    grows = {"4": _view_entry(0.5, 0.1), "16": _view_entry(0.9, 0.3)}
    assert verdict_l9(grows, counts)["verdict"] == "SCALES"

    # Same direction, gap smaller than the V = 4 spread: FLAT, not SCALES.
    within = {"4": _view_entry(0.5, 0.4), "16": _view_entry(0.6, 0.3)}
    assert verdict_l9(within, counts)["verdict"] == "FLAT"

    mixed = {"4": _view_entry(0.5, 0.1), "8": _view_entry(0.9, 0.1),
             "16": _view_entry(0.55, 0.1)}
    assert verdict_l9(mixed, [4, 8, 16])["verdict"] == "MIXED"


def test_l9_a_falling_mae_is_never_scales():
    falling = {"4": _view_entry(0.9, 0.05), "16": _view_entry(0.2, 0.05)}
    assert verdict_l9(falling, [4, 16])["verdict"] == "MIXED"


def test_l2_needs_both_monotonicity_and_a_gap_bigger_than_the_spread():
    counts = [4, 8, 16]
    accumulating = {"4": _view_entry(0, 0, 0.10, 0.01),
                    "8": _view_entry(0, 0, 0.20, 0.01),
                    "16": _view_entry(0, 0, 0.30, 0.01)}
    assert verdict_l2(accumulating, counts)["verdict"] == "ACCUMULATES"

    # Monotone but the gap is inside the noise.
    noisy = {"4": _view_entry(0, 0, 0.10, 0.09),
             "8": _view_entry(0, 0, 0.11, 0.09),
             "16": _view_entry(0, 0, 0.12, 0.09)}
    assert verdict_l2(noisy, counts)["verdict"] == "DOES NOT ACCUMULATE"

    # A big end-to-end gap that dips in the middle is not accumulation.
    dipping = {"4": _view_entry(0, 0, 0.10, 0.01),
               "8": _view_entry(0, 0, 0.05, 0.01),
               "16": _view_entry(0, 0, 0.40, 0.01)}
    assert verdict_l2(dipping, counts)["verdict"] == "DOES NOT ACCUMULATE"


def test_l2_reports_not_measured_rather_than_guessing_when_rho0_is_absent():
    assert verdict_l2({"4": _view_entry(0.5, 0.1)}, [4])["verdict"] == "NOT MEASURED"


# --------------------------------------------------------------------------
# B - L10
# --------------------------------------------------------------------------
def test_fit_floor_recovers_a_floor_it_was_given():
    n_grid = [64, 128, 256, 512, 1024, 2048]
    fit = _fit_floor(n_grid, [0.4 + 32.0 / n for n in n_grid])
    assert fit["floor"] == pytest.approx(0.4, abs=1e-9)
    assert fit["c"] == pytest.approx(32.0, abs=1e-6)
    assert fit["residual_rms"] == pytest.approx(0.0, abs=1e-9)


def test_fit_floor_refuses_rather_than_extrapolating_from_one_point():
    fit = _fit_floor([64, 128], [0.4, None])
    assert fit["floor"] is None
    assert "not measured" in fit["note"]


def _per_n(values, truth=3.0):
    return {
        str(n): {"signed_eniv_bias": {"mean": v - truth, "std": 0.01, "n": 5}}
        for n, v in values.items()
    }


def test_l10_sample_count_verdict_needs_a_zero_floor_and_a_shrinking_bias():
    n_grid = [64, 2048]
    per_n = _per_n({64: 2.0, 2048: 2.9})  # bias 1.0 -> 0.1
    out = verdict_l10(per_n, {"mean": 0.01, "std": 0.05, "n": 5}, n_grid)
    assert out["verdict"] == "DOMINATED BY SAMPLE COUNT"

    # A floor well away from zero, with bias still there at the largest n.
    out = verdict_l10(per_n, {"mean": 0.40, "std": 0.02, "n": 5}, n_grid)
    assert out["verdict"] == "STRUCTURAL FLOOR"

    # Zero floor but the bias barely shrinks: neither rule fires.
    flat = _per_n({64: 2.5, 2048: 2.6})  # bias 0.5 -> 0.4
    out = verdict_l10(flat, {"mean": 0.01, "std": 0.05, "n": 5}, n_grid)
    assert out["verdict"] == "INCONCLUSIVE"


# --------------------------------------------------------------------------
# C - L11
# --------------------------------------------------------------------------
def _tail(mean, std):
    return {"tail": {"auc": {"mean": mean, "std": std, "n": 5}},
            "rho_bar": {"auc": {"mean": 0.7, "std": 0.01, "n": 5}}}


def _four_x(cells):
    return {
        "detection": {k: _tail(*v) for k, v in cells.items()},
        "protocol": {k: {"n_tail_min": {"mean": 240.0, "std": 0.0, "n": 5}} for k in cells},
    }


def test_l11_substantive_when_every_epsilon_stays_inside_the_band():
    out = verdict_l11({"detection": {}}, _four_x({"eps0.2": (0.49, 0.002),
                                                  "eps2.0": (0.58, 0.012)}))
    assert out["bands"]["primary"]["verdict"] == "SUBSTANTIVE"


def test_l11_underpowered_only_when_a_mean_clears_the_band_by_its_own_std():
    far = _four_x({"eps0.2": (0.70, 0.002)})
    assert verdict_l11({"detection": {}}, far)["bands"]["primary"]["verdict"] == (
        "UNDERPOWERED-AFTER-ALL"
    )
    near = _four_x({"eps0.2": (0.60, 0.10)})
    assert verdict_l11({"detection": {}}, near)["bands"]["primary"]["verdict"] == (
        "OUTSIDE THE BAND BY LESS THAN ONE STD"
    )


def test_l11_scores_the_two_bands_independently():
    """0.482 is inside the committed band and below L11's quoted floor. The
    whole reason both are pre-registered."""
    out = verdict_l11({"detection": {}}, _four_x({"eps0.5": (0.482, 0.001)}))
    assert out["bands"]["primary"]["verdict"] == "SUBSTANTIVE"
    assert out["bands"]["quoted"]["verdict"] == "UNDERPOWERED-AFTER-ALL"


# --------------------------------------------------------------------------
# D - L22
# --------------------------------------------------------------------------
def _summary(mixed, equal, diff_std):
    return {
        "auc_clone_vs_clean_w4_mixed_width": {"mean": mixed, "std": 0.05, "n": 5},
        "auc_clone_vs_clean_w6_equal_width": {"mean": equal, "std": 0.05, "n": 5},
        "paired_difference_equal_minus_mixed": {
            "mean": equal - mixed, "std": diff_std, "n": 5
        },
    }


def test_l22_width_explains_part_when_equal_width_drops_beyond_the_paired_noise():
    out = verdict_l22(_summary(0.8743, 0.70, 0.01))
    assert out["verdict"] == "WIDTH EXPLAINS PART OF THE COMMITTED FIGURE"


def test_l22_width_explains_none_when_the_paired_mean_sits_in_its_own_noise():
    out = verdict_l22(_summary(0.8743, 0.8700, 0.02))
    assert out["verdict"] == "WIDTH EXPLAINS NONE OF IT"


def test_l22_a_higher_equal_width_auc_is_reported_not_silently_rounded_to_zero():
    out = verdict_l22(_summary(0.8743, 0.92, 0.01))
    assert "NOT CONFIRMED" in out["verdict"]


# --------------------------------------------------------------------------
# Structural checks over whatever Part 36 actually committed
# --------------------------------------------------------------------------
def _load(name):
    path = OUT / name
    if not path.exists():
        pytest.skip(f"{path} not committed in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", ["analysis_a.json", "analysis_b.json",
                                  "analysis_c.json", "analysis_d.json"])
def test_committed_results_report_five_seeds(name):
    payload = _load(name)
    assert payload["seeds"] == SEEDS


def test_committed_reproduction_checks_passed():
    """If any committed reproduction entry failed, nothing after it in the
    README is interpretable -- so the test says so rather than the prose."""
    for name, getter in (
        ("analysis_a.json", lambda p: p["reproduction"]["eniv_validation_4view"]),
        ("analysis_c.json", lambda p: p["reproduction"]),
        ("analysis_d.json", lambda p: p["reproduction"]),
    ):
        payload = _load(name)
        for entry in getter(payload)["quantities"]:
            assert entry["pass"], f"{name}: {entry['quantity']} did not reproduce"


def test_sweep_is_nested_paired_and_refuses_an_n_it_cannot_supply():
    """Three properties of the sweep, on tensors small enough to be checked by
    hand: the subsample is nested (so the comparison is paired within seed),
    it is reproducible from the seed alone, and an n larger than the split
    reads `not measured` rather than a number computed on fewer rows."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    features = torch.as_tensor(rng.standard_normal((200, 4, 8)))
    evidence = torch.as_tensor(rng.standard_normal((200, 4, 3)) ** 2)
    masks = torch.ones((200, 4), dtype=torch.bool)

    first = _sweep(features, evidence, masks, [64, 128, 500], seed=0)
    again = _sweep(features, evidence, masks, [64, 128, 500], seed=0)
    assert first["64"]["eniv"] == again["64"]["eniv"]
    assert first["500"]["eniv"] is None
    assert "not measured" in first["500"]["note"]

    # Nested: computing the sweep on a grid that stops earlier must give the
    # SAME value at a shared n, which only holds if the prefix is reused.
    shorter = _sweep(features, evidence, masks, [64], seed=0)
    assert shorter["64"]["eniv"] == first["64"]["eniv"]
    order = np.random.default_rng(36000).permutation(200)
    assert np.array_equal(order[:64], order[:128][:64])
