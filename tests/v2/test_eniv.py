"""Part 22 tests: the ENIV aggregations, the discount, and the seed loop.

No neural model is needed anywhere here. Part 22 consumes a matrix and produces
a number; the semantics were measured in Part 21. What these tests pin down is
that the aggregation is the one it claims to be, that the discount cannot leave
[0, 1], and -- the regression that matters most -- that the experiment's seeds
actually select something.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from prismflow.eniv.eniv import design_effect_n_eff
from prismflow.v2.dependence import (
    DependenceReport,
    EmbeddingEstimator,
    aggregate_dependence,
)
from prismflow.v2.statistics import (
    DISCOUNT_FLOOR,
    ENIV_METHODS,
    SemanticENIVResult,
    apply_eniv_discount,
    compute_discount_factor,
    compute_semantic_eniv,
    eniv_report,
    per_angle_discount,
)

from .conftest import make_claimset, topic_claims
# The deterministic stand-in encoder from the Part 21 suite. Part 22 never needs
# real semantics, and a test that downloads 80MB to check an exclusion is a test
# people stop running.
from .test_dependence import bag_of_words_encoder

ANGLES = ["tech", "market", "financial", "regulatory", "sentiment"]


def angle_names(n: int) -> list[str]:
    """``n`` unique names. The clone tests go past PrismFlow's five angles."""
    names = list(ANGLES[:n])
    names += [f"angle_{i}" for i in range(len(names), n)]
    return names


def report_from(matrix: np.ndarray) -> DependenceReport:
    n = matrix.shape[0]
    return DependenceReport(
        angle_names=angle_names(n), n_angles=n, dependence_matrix=matrix
    )


def equicorrelated(n: int, rho: float) -> np.ndarray:
    matrix = np.full((n, n), float(rho))
    np.fill_diagonal(matrix, 1.0)
    return matrix


def clone_structure(n_base: int, n_copies: int) -> np.ndarray:
    """``n_base`` independent angles plus ``n_copies`` exact duplicates of #0.

    The true effective count is ``n_base`` at every ``n_copies``: a duplicate
    carries no information the original did not.
    """
    n = n_base + n_copies
    matrix = np.eye(n)
    group = [0] + list(range(n_base, n))
    for i in group:
        for j in group:
            matrix[i, j] = 1.0
    return matrix


# --- the aggregations -----------------------------------------------------


def test_independent_angles_give_eniv_equal_to_n():
    for method in ENIV_METHODS:
        assert compute_semantic_eniv(report_from(np.eye(5)), method=method) == 5.0


def test_identical_angles_collapse_to_one_view():
    matrix = np.ones((5, 5))
    for method in ENIV_METHODS:
        assert compute_semantic_eniv(report_from(matrix), method=method) == 1.0


def test_design_effect_method_is_exactly_the_briefs_formula():
    """``n**2 / sum(D)`` is what ``method="design_effect"`` must compute."""
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(2, 8))
        upper = rng.uniform(0.0, 1.0, (n, n))
        matrix = 0.5 * (upper + upper.T)
        np.fill_diagonal(matrix, 1.0)
        brief = (n**2) / matrix.sum()
        got = compute_semantic_eniv(report_from(matrix), method="design_effect")
        assert got == pytest.approx(np.clip(brief, 1.0, n), abs=1e-9)
        assert design_effect_n_eff(matrix) == pytest.approx(brief, abs=1e-9)


def test_eigen_is_exact_on_clones_and_design_effect_is_not():
    """The measurement behind the default. Part 06's structure, V2's module.

    True effective count is 4 at every k. The eigenvalue form holds; the brief's
    formula loses witnesses that were never there to lose.
    """
    eigen, design = [], []
    for k in range(5):
        report = report_from(clone_structure(4, k))
        eigen.append(compute_semantic_eniv(report, method="eigen"))
        design.append(compute_semantic_eniv(report, method="design_effect"))

    assert eigen == pytest.approx([4.0] * 5, abs=1e-9)
    assert design[0] == pytest.approx(4.0, abs=1e-9)
    assert design[-1] < 2.5
    # strictly decreasing: the bias grows with every duplicate
    assert all(b < a for a, b in zip(design, design[1:]))


def test_eniv_is_monotone_non_increasing_in_correlation():
    for method in ENIV_METHODS:
        values = [
            compute_semantic_eniv(report_from(equicorrelated(5, rho)), method=method)
            for rho in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        ]
        assert all(b <= a + 1e-9 for a, b in zip(values, values[1:])), values


def test_eniv_is_clamped_to_one_and_n():
    # Anti-correlation is not extra information: the count cannot exceed n.
    matrix = equicorrelated(5, 0.0)
    for method in ENIV_METHODS:
        assert 1.0 <= compute_semantic_eniv(report_from(matrix), method=method) <= 5.0


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="method must be one of"):
        compute_semantic_eniv(report_from(np.eye(3)), method="n_squared")


def test_empty_report_has_no_effective_views():
    empty = DependenceReport(
        angle_names=[],
        n_angles=0,
        dependence_matrix=np.zeros((0, 0)),
        excluded_angles=list(ANGLES),
    )
    for method in ENIV_METHODS:
        assert compute_semantic_eniv(empty, method=method) == 0.0
    assert eniv_report(empty).discount_factor == 0.0
    assert per_angle_discount(empty) == {}


# --- the discount ---------------------------------------------------------


def test_discount_is_eniv_over_n():
    assert compute_discount_factor(5.0, 5) == pytest.approx(1.0)
    assert compute_discount_factor(2.5, 5) == pytest.approx(0.5)


def test_discount_floor_and_cap_hold():
    assert compute_discount_factor(0.01, 5) == DISCOUNT_FLOOR
    assert compute_discount_factor(0.0, 5) == DISCOUNT_FLOOR
    # ENIV above n cannot amplify evidence.
    assert compute_discount_factor(9.0, 5) == 1.0


def test_discount_requires_n_rather_than_defaulting_to_five():
    """The brief's ``n: int = 5`` would silently divide a 2-angle ENIV by 5."""
    with pytest.raises(TypeError):
        compute_discount_factor(2.0)  # type: ignore[call-arg]


def test_discount_rejects_nonsense_inputs():
    with pytest.raises(ValueError, match="non-negative"):
        compute_discount_factor(2.0, -1)
    with pytest.raises(ValueError, match="finite"):
        compute_discount_factor(float("nan"), 5)
    assert compute_discount_factor(0.0, 0) == 0.0


def test_apply_discount_multiplies_and_range_checks():
    assert apply_eniv_discount(0.8, 0.5) == pytest.approx(0.4)
    assert apply_eniv_discount(0.0, 1.0) == 0.0
    with pytest.raises(ValueError, match="claim_confidence"):
        apply_eniv_discount(1.4, 0.5)
    with pytest.raises(ValueError, match="discount_factor"):
        apply_eniv_discount(0.5, 2.0)


def test_discounted_confidence_never_exceeds_the_original():
    rng = np.random.default_rng(1)
    for _ in range(200):
        confidence = float(rng.uniform(0, 1))
        rho = float(rng.uniform(0, 1))
        report = report_from(equicorrelated(5, rho))
        discount = compute_discount_factor(compute_semantic_eniv(report), 5)
        discounted = apply_eniv_discount(confidence, discount)
        assert 0.0 <= discounted <= confidence + 1e-12


# --- per-angle discount ---------------------------------------------------


def test_per_angle_discount_splits_a_clique_and_leaves_others_alone():
    """Two exact duplicates share one witness; the other three keep their own."""
    matrix = np.eye(5)
    matrix[0, 1] = matrix[1, 0] = 1.0
    alphas = per_angle_discount(report_from(matrix))

    assert set(alphas) == set(ANGLES)
    assert alphas["tech"] == pytest.approx(0.5)
    assert alphas["market"] == pytest.approx(0.5)
    assert alphas["tech"] + alphas["market"] == pytest.approx(1.0)
    for name in ("financial", "regulatory", "sentiment"):
        assert alphas[name] == pytest.approx(1.0)


def test_per_angle_discount_is_what_the_scalar_hides():
    """The scalar penalises the three angles nobody duplicated; per-angle does not."""
    matrix = np.eye(5)
    matrix[0, 1] = matrix[1, 0] = 1.0
    report = report_from(matrix)
    scalar = compute_discount_factor(compute_semantic_eniv(report), 5)
    alphas = per_angle_discount(report)

    assert scalar == pytest.approx(0.8)
    assert alphas["financial"] == pytest.approx(1.0)
    assert alphas["financial"] > scalar


# --- the bundled result ---------------------------------------------------


def test_eniv_report_is_self_consistent_and_names_the_excluded():
    report = DependenceReport(
        angle_names=["tech", "market"],
        n_angles=2,
        dependence_matrix=equicorrelated(2, 0.5),
        excluded_angles=["financial", "regulatory", "sentiment"],
    )
    result = eniv_report(report)

    assert isinstance(result, SemanticENIVResult)
    assert result.nominal_angles == 2
    assert result.method == "eigen"
    assert result.mean_dependence == pytest.approx(0.5)
    assert result.discount_factor == pytest.approx(
        compute_discount_factor(result.effective_angles, 2)
    )
    # n is the count that SURVIVED, never the five angles PrismFlow plans for.
    assert result.excluded_angles == ("financial", "regulatory", "sentiment")
    assert result.to_dict()["nominal_angles"] == 2


def test_excluded_angles_do_not_inflate_the_view_count():
    """Part 21 drops silent angles precisely so Part 22 cannot count them."""
    claimsets = [
        make_claimset("tech", "q", topic_claims(0)),
        make_claimset("market", "q", topic_claims(1)),
        make_claimset("financial", "q", []),
        make_claimset("regulatory", "q", []),
        make_claimset("sentiment", "q", []),
    ]
    report = aggregate_dependence(
        claimsets,
        weights={"citation": 1.0, "embedding": 0.0, "lexical": 0.0},
        estimators={"embedding": EmbeddingEstimator(encoder=bag_of_words_encoder)},
    )
    result = eniv_report(report)

    assert report.n_angles == 2
    assert result.nominal_angles == 2
    assert result.effective_angles <= 2.0
    assert len(result.excluded_angles) == 3


# --- the experiment's seed loop ------------------------------------------


def load_experiment():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "v2"
        / "test_eniv_redundancy.py"
    )
    spec = importlib.util.spec_from_file_location("part22_experiment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_regression_seeds_actually_select_something():
    """The Part 21 defect, one part later: five seeds drawing nothing.

    The brief seeds the global RNG and then builds three hardcoded matrices, so
    every seed computes an identical number and the reported sd is exactly 0.0 --
    which reads as a stable estimator rather than an absent measurement. This
    test fails if the bands ever stop varying.
    """
    experiment = load_experiment()
    matrices = [
        experiment.symmetric_from_band(np.random.default_rng(seed), 5, 0.3, 0.5)
        for seed in range(5)
    ]
    for earlier, later in zip(matrices, matrices[1:]):
        assert not np.array_equal(earlier, later)

    enivs = [compute_semantic_eniv(report_from(m)) for m in matrices]
    assert len(set(enivs)) == len(enivs)
    assert np.std(enivs) > 0.0


def test_regression_seed_is_reproducible():
    """Varying is not the same as being random. The same seed must repeat."""
    experiment = load_experiment()
    first = experiment.symmetric_from_band(np.random.default_rng(3), 5, 0.3, 0.5)
    second = experiment.symmetric_from_band(np.random.default_rng(3), 5, 0.3, 0.5)
    assert np.array_equal(first, second)


def test_experiment_bands_are_disjoint_and_ordered():
    """Perturbation must not let one condition reach into the next."""
    experiment = load_experiment()
    bands = list(experiment.BANDS.values())
    for (_, high), (low, _) in zip(bands, bands[1:]):
        assert high < low


def test_experiment_matrices_are_valid_dependence_reports():
    experiment = load_experiment()
    rng = np.random.default_rng(7)
    for low, high in experiment.BANDS.values():
        matrix = experiment.symmetric_from_band(rng, 5, low, high)
        report = experiment.as_report(matrix)  # raises if invalid
        assert report.n_angles == 5
        assert np.allclose(matrix, matrix.T)
        assert matrix.min() >= 0.0 and matrix.max() <= 1.0
        off = report.off_diagonal
        assert off.min() >= low and off.max() <= high


def test_experiment_gate_runs_and_reports_both_aggregations():
    """A short run: the plumbing, not the published numbers."""
    experiment = load_experiment()
    payload = experiment.run_eniv_redundancy_experiment(num_seeds=3, num_queries=4)

    assert payload["gate_decided_on"] == "eigen"
    assert set(payload["summary"]) == set(ENIV_METHODS)
    for method in ENIV_METHODS:
        conditions = payload["summary"][method]["eniv"]
        assert "disjoint_clique" in conditions
        for stats in conditions.values():
            # The defect this experiment exists to avoid.
            assert stats["sd"] > 0.0
            assert len(stats["per_seed_mean"]) == 3
    assert len(payload["sweep"]["eigen"]["per_seed_curves"]) == 3
    assert payload["gate"]["eigen"]["seeds_monotonic"] == 3


def test_experiment_clique_separates_the_two_aggregations():
    """The condition that justifies the default, measured rather than asserted."""
    experiment = load_experiment()
    rng = np.random.default_rng(11)
    eigen, design = [], []
    for _ in range(50):
        report = experiment.as_report(experiment.clique_matrix(rng, 5))
        eigen.append(compute_semantic_eniv(report, method="eigen"))
        design.append(compute_semantic_eniv(report, method="design_effect"))

    # Two near-duplicates count as roughly one witness, so the truth is near 4.
    assert float(np.mean(eigen)) == pytest.approx(4.0, abs=0.2)
    # The design effect spreads the clique's correlation over every pair.
    assert float(np.mean(design)) < float(np.mean(eigen)) - 0.5
