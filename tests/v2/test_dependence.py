"""Part 21 tests: the three estimators, the aggregation, and the planted cases.

Most tests use a deterministic bag-of-words encoder rather than the real
sentence-transformer: mechanics (symmetry, range, weighting, exclusion) do not
need a neural model, and a suite that downloads 80MB to check that a matrix is
symmetric is a suite people stop running.

The planted-redundancy tests DO need real semantics -- a paraphrase is only
distinguishable from an unrelated claim by a model that understands meaning --
so they load all-MiniLM-L6-v2 once per session and skip if it cannot be reached.
"""

from __future__ import annotations

import re
import zlib

import numpy as np
import pytest

from prismflow.v2.dependence import (
    DEFAULT_WEIGHTS,
    CitationEstimator,
    DependenceReport,
    EmbeddingEstimator,
    LexicalEstimator,
    aggregate_dependence,
    cosine,
    estimator_agreement,
    validate_weights,
)
from prismflow.v2.reasoners.models import Claim

from .conftest import make_claimset, planted_pair, topic_claims, topic_count

# --- encoders -----------------------------------------------------------


#: Width of the test encoder's space. Fixed, because a real sentence encoder
#: returns the same dimension for every batch -- and the estimator calls it once
#: per angle, so a per-batch vocabulary would hand back vectors of different
#: widths for different angles and nothing would be comparable.
FAKE_DIMS = 128


def bag_of_words_encoder(texts):
    """Deterministic, model-free embeddings: hashed word counts, fixed width.

    Identical texts embed identically, texts sharing words are close, disjoint
    texts are near-orthogonal. Enough to exercise every mechanical property, and
    it cannot fail because a download did.
    """
    texts = list(texts)
    matrix = np.zeros((len(texts), FAKE_DIMS), dtype=float)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            # Stable across processes, unlike hash(): PYTHONHASHSEED randomises
            # str hashing, which would make these fixtures irreproducible.
            bucket = zlib.crc32(word.encode("utf-8")) % FAKE_DIMS
            matrix[row, bucket] += 1.0
    return matrix


@pytest.fixture
def fake_embedding():
    return EmbeddingEstimator(encoder=bag_of_words_encoder)


#: Topics swept by the real-model tests. The full 24-topic sweep belongs in
#: experiments/v2/test_dependence_redundancy.py, which is the measurement; a
#: unit test only has to show the bands are where the experiment says. The first
#: version of this file swept all 24 in five separate real-model tests and took
#: 84 minutes, which is a suite nobody runs.
SWEEP = range(6)


@pytest.fixture(scope="session")
def real_encoder():
    """The real sentence-transformer encoder, loaded ONCE and memoised.

    Loading the model costs seconds and the estimator calls its encoder once per
    angle, so a per-test ``EmbeddingEstimator()`` reloads the weights for every
    assertion. Every real-model test shares this one, and the cache means the
    repeated claim texts across conditions are encoded a single time. Caching
    cannot alter a result: the encoder is deterministic, which is the same
    property that made the brief's seed loop a no-op.
    """
    try:
        inner = EmbeddingEstimator().encoder
        inner(["warm up"])
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"sentence-transformer unavailable: {type(exc).__name__}: {exc}")

    cache: dict = {}

    def encode(texts):
        texts = list(texts)
        missing = [t for t in texts if t not in cache]
        if missing:
            vectors = np.asarray(inner(missing), dtype=float)
            if vectors.ndim == 1:
                vectors = vectors.reshape(1, -1)
            cache.update(zip(missing, vectors))
        return np.vstack([cache[t] for t in texts])

    return encode


@pytest.fixture(scope="session")
def real_embedding(real_encoder):
    return EmbeddingEstimator(encoder=real_encoder)


def claim(text, ids=(), confidence=0.8):
    return Claim(text=text, confidence=confidence, cited_ids=list(ids))


def aggregate(left, right, embedding, **kwargs):
    return aggregate_dependence(
        [left, right], estimators={"embedding": embedding}, **kwargs
    )


# --- the brief's seven --------------------------------------------------


def test_embedding_estimator_symmetric(fake_embedding):
    claims_by_angle = {
        "tech": [claim("vector databases consolidate around open engines")],
        "market": [claim("open engines dominate the vector database market")],
        "financial": [claim("bond yields moved on the rate decision")],
    }
    matrix, warnings = fake_embedding.estimate(claims_by_angle)

    assert matrix.shape == (3, 3)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(matrix.diagonal(), 1.0)
    assert matrix.min() >= 0.0 and matrix.max() <= 1.0
    assert warnings == []
    # tech~market share vocabulary; neither shares much with financial.
    assert matrix[0, 1] > matrix[0, 2]


def test_citation_jaccard():
    estimator = CitationEstimator()
    matrix, _ = estimator.estimate(
        {
            "tech": [claim("a", ["r1", "r2"]), claim("b", ["r3"])],
            "market": [claim("c", ["r2", "r3", "r4"])],
            "financial": [claim("d", ["r9"])],
        }
    )
    # tech {r1,r2,r3} vs market {r2,r3,r4}: |∩|=2, |∪|=4 -> 0.5
    assert matrix[0, 1] == pytest.approx(0.5)
    # tech vs financial: disjoint -> 0
    assert matrix[0, 2] == pytest.approx(0.0)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(matrix.diagonal(), 1.0)


def test_lexical_estimator():
    estimator = LexicalEstimator()
    matrix, warnings = estimator.estimate(
        {
            "tech": [claim("quantisation reduces memory bandwidth pressure")],
            "market": [claim("quantisation reduces memory bandwidth pressure")],
            "financial": [claim("interest rates drove bond volatility")],
        }
    )
    assert matrix[0, 1] == pytest.approx(1.0, abs=1e-6), "identical text"
    assert matrix[0, 2] < 0.2, "disjoint vocabulary"
    assert np.allclose(matrix, matrix.T)
    assert estimator.name == "lexical"


def test_aggregation_weights(fake_embedding):
    """The aggregate is exactly the weighted sum of the three matrices."""
    left, right = planted_pair("paraphrase", 3, 11)
    report = aggregate(left, right, fake_embedding)

    weights = report.weights
    expected = sum(
        weights[name] * report.per_estimator[name][0, 1] for name in weights
    )
    assert report.dependence_matrix[0, 1] == pytest.approx(expected, abs=1e-9)
    assert set(report.per_estimator) == {"embedding", "citation", "lexical"}
    assert weights == DEFAULT_WEIGHTS


def test_planted_redundancy_identical(real_embedding):
    """Perfect redundancy must read as near-total dependence."""
    values = []
    for topic in SWEEP:
        left, right = planted_pair("identical", topic, (topic + 7) % topic_count())
        values.append(aggregate(left, right, real_embedding).dependence_matrix[0, 1])
    values = np.array(values)

    assert values.min() > 0.95, f"lowest identical pair was {values.min():.4f}"
    assert values.mean() > 0.95


def test_planted_redundancy_paraphrased(real_embedding):
    """Reworded findings on the same evidence sit in the middle band."""
    values = []
    for topic in SWEEP:
        left, right = planted_pair("paraphrase", topic, (topic + 7) % topic_count())
        values.append(aggregate(left, right, real_embedding).dependence_matrix[0, 1])
    values = np.array(values)

    assert 0.6 < values.mean() < 0.95, f"paraphrase mean {values.mean():.4f}"
    assert values.max() <= 1.0


def test_planted_redundancy_unrelated(real_embedding):
    """Different topics, disjoint evidence: dependence must be low."""
    values = []
    for topic in SWEEP:
        left, right = planted_pair("unrelated", topic, (topic + 7) % topic_count())
        values.append(aggregate(left, right, real_embedding).dependence_matrix[0, 1])
    values = np.array(values)

    assert values.mean() < 0.3, f"unrelated mean {values.mean():.4f}"
    assert values.max() < 0.5, f"worst unrelated pair {values.max():.4f}"


def test_planted_conditions_are_separable(real_embedding):
    """The three conditions must not overlap, which is what makes the
    instrument usable at all: if paraphrase and unrelated ranges touched, a
    dependence number could not be read as evidence of redundancy."""
    bands = {}
    for condition in ("identical", "paraphrase", "unrelated"):
        values = [
            aggregate(
                *planted_pair(condition, topic, (topic + 7) % topic_count()),
                real_embedding,
            ).dependence_matrix[0, 1]
            for topic in SWEEP
        ]
        bands[condition] = (min(values), max(values))

    assert bands["unrelated"][1] < bands["paraphrase"][0], (
        f"unrelated max {bands['unrelated'][1]:.3f} overlaps paraphrase min "
        f"{bands['paraphrase'][0]:.3f}"
    )
    assert bands["paraphrase"][1] < bands["identical"][0], (
        f"paraphrase max {bands['paraphrase'][1]:.3f} overlaps identical min "
        f"{bands['identical'][0]:.3f}"
    )


# --- regressions against the specified implementation -------------------


def test_regression_two_silent_angles_are_not_identical():
    """The brief returns Jaccard 1.0 when neither angle cites anything.

    "Both cited nothing" is absence of evidence, not evidence of identity -- and
    in V2 it is the ordinary case, since Part 19 ships four of five angles
    without connectors.
    """
    matrix, warnings = CitationEstimator().estimate(
        {"market": [claim("a")], "financial": [claim("b")]}
    )
    assert matrix[0, 1] == 0.0
    assert any("no citations at all" in w for w in warnings)


def test_regression_empty_angles_are_excluded_not_scored(fake_embedding):
    """An angle that said nothing must not count as an independent view.

    Scoring it 0 reads as "perfectly independent", so Part 22 would add it to
    the effective view count, discount less, and report MORE confidence for an
    angle that contributed nothing.
    """
    speaking = make_claimset("tech", "q", topic_claims(0))
    silent_a = make_claimset("market", "q", [])
    silent_b = make_claimset("financial", "q", [])

    report = aggregate_dependence(
        [speaking, silent_a, silent_b], estimators={"embedding": fake_embedding},
    )

    assert report.angle_names == ["tech"]
    assert report.n_angles == 1
    assert report.excluded_angles == ["market", "financial"]
    assert any("excluded 2 angle(s)" in w for w in report.warnings)
    assert report.mean_dependence is None

    # Opting in keeps them, and then they are visible rather than silent.
    kept = aggregate_dependence(
        [speaking, silent_a, silent_b],
        estimators={"embedding": fake_embedding},
        include_empty=True,
    )
    assert kept.n_angles == 3
    assert kept.dependence_matrix[1, 2] == 0.0, "two silent angles are not identical"


def test_regression_all_angles_empty_is_an_empty_report(fake_embedding):
    report = aggregate_dependence(
        [make_claimset("tech", "q", []), make_claimset("market", "q", [])],
        estimators={"embedding": fake_embedding},
    )
    assert report.n_angles == 0
    assert report.dependence_matrix.shape == (0, 0)
    assert report.mean_dependence is None
    assert report.most_dependent_pair() is None
    assert set(report.excluded_angles) == {"tech", "market"}


@pytest.mark.parametrize(
    "weights, match",
    [
        ({"embedding": 0.5, "citation": 0.3, "lexical": 0.3}, "sum to 1.0"),
        ({"embedding": 0.5, "citation": 0.3, "mi": 0.2}, "unknown estimator"),
        ({"embedding": 0.7, "citation": 0.3}, "no weight given"),
        ({"embedding": 1.2, "citation": -0.2, "lexical": 0.0}, "non-negative"),
        ({}, "must not be empty"),
    ],
)
def test_regression_weights_are_validated(weights, match):
    """The brief validates nothing.

    Weights summing above 1 push every pair into the [0, 1] clip, so the matrix
    saturates and every angle looks maximally dependent; a misspelled key
    silently contributes zero while the others keep their magnitude. Both
    produce a matrix that looks entirely reasonable.
    """
    with pytest.raises(ValueError, match=match):
        validate_weights(weights)


def test_regression_saturation_is_what_bad_weights_would_cause(fake_embedding):
    """Demonstrates the failure the validation prevents."""
    left, right = planted_pair("unrelated", 0, 9)
    per = aggregate(left, right, fake_embedding).per_estimator
    inflated = sum(1.0 * per[name][0, 1] for name in per)  # weights of 1.0 each
    assert inflated == pytest.approx(
        sum(per[name][0, 1] for name in per)
    )
    # With honest weights this pair is low; with weights summing to 3 it would
    # be clipped upward toward 1.
    honest = sum(DEFAULT_WEIGHTS[name] * per[name][0, 1] for name in per)
    assert honest < inflated


def test_regression_report_raises_rather_than_asserts():
    """Bare assert statements vanish under python -O.

    Part 22 relies on this matrix being symmetric with a unit diagonal and
    values in [0, 1]; that guarantee must not depend on an interpreter flag.
    """
    good = np.array([[1.0, 0.2], [0.2, 1.0]])

    with pytest.raises(ValueError, match="symmetric"):
        DependenceReport(["a", "b"], 2, np.array([[1.0, 0.2], [0.9, 1.0]]))
    with pytest.raises(ValueError, match="diagonal"):
        DependenceReport(["a", "b"], 2, np.array([[0.5, 0.2], [0.2, 0.5]]))
    with pytest.raises(ValueError, match="shape"):
        DependenceReport(["a", "b"], 2, np.eye(3))
    with pytest.raises(ValueError, match=r"lie in \[0, 1\]"):
        DependenceReport(["a", "b"], 2, np.array([[1.0, 1.4], [1.4, 1.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        DependenceReport(["a", "b"], 2, np.array([[1.0, np.nan], [np.nan, 1.0]]))
    with pytest.raises(ValueError, match="unique"):
        DependenceReport(["a", "a"], 2, good)
    with pytest.raises(ValueError, match="angle names"):
        DependenceReport(["a"], 2, good)

    # And the well-formed case is accepted.
    assert DependenceReport(["a", "b"], 2, good).between("a", "b") == 0.2


def test_regression_duplicate_angle_names_are_rejected(fake_embedding):
    with pytest.raises(ValueError, match="duplicate angle name"):
        aggregate_dependence(
            [
                make_claimset("tech", "q", topic_claims(0)),
                make_claimset("tech", "q", topic_claims(1)),
            ],
            estimators={"embedding": fake_embedding},
        )


def test_regression_embedding_normalises_before_averaging():
    """The brief averages raw vectors, letting norm decide influence.

    A claim whose embedding happens to be longer pulls the angle's
    representation toward itself for reasons unrelated to what it says.
    """
    def uneven(texts):
        # First text gets a vector 100x longer than the second.
        table = {"loud": np.array([100.0, 0.0]), "quiet": np.array([0.0, 1.0])}
        return np.array([table[t] for t in texts])

    estimator = EmbeddingEstimator(encoder=uneven)
    vectors = estimator.embed_claims([claim("loud"), claim("quiet")])
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)
    # After normalising, the mean sits between the two directions rather than
    # on top of the louder one.
    assert np.allclose(vectors.mean(axis=0), np.array([0.5, 0.5]))


def test_regression_low_rank_truncation_warns_and_degrades(real_embedding, real_encoder):
    """Condition 4: the brief initialises results['low_rank'] and never fills it.

    np.mean([]) is nan, so the saved JSON would carry a nan for a condition the
    goals list as required. Here truncation is implemented, warns, and its
    effect is measured: compressing 384 dimensions to 4 pushes unrelated pairs
    upward, which is exactly the graceful-degradation claim being tested.
    """
    truncated = EmbeddingEstimator(encoder=real_encoder, truncate_dims=4)
    full_values, low_values = [], []
    for topic in SWEEP:
        pair = planted_pair("unrelated", topic, (topic + 7) % topic_count())
        full_values.append(aggregate(*pair, real_embedding).dependence_matrix[0, 1])
        report = aggregate(*pair, truncated)
        low_values.append(report.dependence_matrix[0, 1])

    assert any("truncated to 4 dimension" in w for w in report.warnings)
    assert np.mean(low_values) > np.mean(full_values), (
        "truncation should inflate apparent similarity between unrelated angles"
    )
    # Degradation, not collapse: the matrix is still valid.
    assert 0.0 <= np.mean(low_values) <= 1.0


def test_regression_estimator_agreement_is_actually_computed(real_embedding):
    """The gate is "three estimators agree on ranking" and the brief never
    compares them -- it averages the three and reports the average."""
    claimsets = [
        make_claimset("tech", "q", topic_claims(0)),
        make_claimset("market", "q", topic_claims(0, paraphrase=True)),
        make_claimset("financial", "q", topic_claims(9)),
        make_claimset("regulatory", "q", topic_claims(17)),
    ]
    report = aggregate_dependence(
        claimsets, estimators={"embedding": real_embedding}
    )
    mean_rho, per_pair, notes = estimator_agreement(report)

    assert mean_rho is not None, f"agreement undefined: {notes}"
    assert -1.0 <= mean_rho <= 1.0
    assert per_pair, "no estimator pair produced a defined rho"
    assert mean_rho > 0.5, f"estimators disagree on ranking: {per_pair}"


def test_agreement_is_undefined_with_too_few_pairs(fake_embedding):
    """Two angles give one pair; a rank correlation over one point is not a
    measurement, and returning 1.0 for it would be a fabricated agreement."""
    report = aggregate(*planted_pair("identical", 0, 3), fake_embedding)
    mean_rho, per_pair, notes = estimator_agreement(report)

    assert mean_rho is None
    assert per_pair == {}
    assert any("at least 3 points" in note for note in notes)


def test_agreement_handles_a_constant_estimator(fake_embedding):
    """Citation overlap is all-zeros whenever no two angles share a source.

    A constant vector has no ranking, so rho is undefined -- reporting that as
    disagreement would misread the ordinary case as a failure.
    """
    claimsets = [
        make_claimset(name, "q", topic_claims(index))
        for name, index in (("tech", 0), ("market", 5), ("financial", 12))
    ]
    report = aggregate_dependence(
        claimsets, estimators={"embedding": fake_embedding}
    )
    assert np.allclose(report.per_estimator["citation"][np.triu_indices(3, 1)], 0.0)

    mean_rho, per_pair, notes = estimator_agreement(report)
    assert any("constant across all pairs" in note for note in notes)
    assert "citation~embedding" not in per_pair


# --- estimator and report mechanics -------------------------------------


def test_cosine_is_clamped_and_zero_safe():
    assert cosine(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0.0]), np.array([-1.0, 0.0])) == 0.0
    assert cosine(np.zeros(3), np.ones(3)) == 0.0
    assert cosine(np.array([np.nan, 1.0]), np.ones(2)) == 0.0


def test_embedding_rejects_bad_configuration():
    with pytest.raises(ValueError, match="pooling"):
        EmbeddingEstimator(pooling="average")
    with pytest.raises(ValueError, match="truncate_dims"):
        EmbeddingEstimator(truncate_dims=0)


def test_max_match_pooling_wins_only_on_asymmetric_coverage(
    real_embedding, real_encoder
):
    """max_match is more sensitive only when one angle's claims are a SUBSET.

    This test asserted the opposite first -- that max_match would beat mean
    pooling wherever a claim was exactly duplicated -- and the measurement
    refused it: with one shared claim among three on each side, mean scored
    0.389 and max_match 0.333. Averaging best-matches over many unmatched claims
    drags the score down, so max_match only leads where coverage is asymmetric.
    Recorded here rather than quietly dropped, because the difference is what
    decides whether the option is worth using in Part 22.
    """
    matched_estimator = EmbeddingEstimator(
        encoder=real_encoder, pooling="max_match"
    )
    shared = topic_claims(0)[0]

    # Asymmetric: everything the left angle says is already in the right one.
    subset_left = make_claimset("tech", "q", [shared])
    subset_right = make_claimset("market", "q", [shared] + topic_claims(15))
    subset_mean = aggregate(
        subset_left, subset_right, real_embedding
    ).dependence_matrix[0, 1]
    subset_matched = aggregate(
        subset_left, subset_right, matched_estimator
    ).dependence_matrix[0, 1]
    assert subset_matched > subset_mean, (
        f"subset coverage: max_match {subset_matched:.3f} should exceed mean "
        f"{subset_mean:.3f}"
    )

    # Symmetric partial overlap: mean leads, and that is the expected direction.
    sym_left = make_claimset("tech", "q", [shared] + topic_claims(4))
    sym_right = make_claimset("market", "q", [shared] + topic_claims(15))
    sym_mean = aggregate(sym_left, sym_right, real_embedding).dependence_matrix[0, 1]
    sym_matched = aggregate(
        sym_left, sym_right, matched_estimator
    ).dependence_matrix[0, 1]
    assert sym_matched < sym_mean

    # Both remain valid similarities whichever way they point.
    for value in (subset_mean, subset_matched, sym_mean, sym_matched):
        assert 0.0 <= value <= 1.0


def test_single_angle_report_is_valid(fake_embedding):
    report = aggregate_dependence(
        [make_claimset("tech", "q", topic_claims(0))],
        estimators={"embedding": fake_embedding},
    )
    assert report.n_angles == 1
    assert report.dependence_matrix.shape == (1, 1)
    assert report.off_diagonal.size == 0
    assert report.mean_dependence is None
    assert "1 angle" in report.summary()


def test_report_accessors(fake_embedding):
    claimsets = [
        make_claimset("tech", "q", topic_claims(0)),
        make_claimset("market", "q", topic_claims(0, paraphrase=True)),
        make_claimset("financial", "q", topic_claims(20)),
    ]
    report = aggregate_dependence(
        claimsets, estimators={"embedding": fake_embedding}
    )

    assert report.pairs == [
        ("tech", "market"), ("tech", "financial"), ("market", "financial"),
    ]
    assert report.off_diagonal.size == 3
    assert report.between("tech", "market") == report.between("market", "tech")
    with pytest.raises(KeyError, match="unknown angle"):
        report.between("tech", "astrology")

    top = report.most_dependent_pair()
    assert top is not None and set(top[:2]) == {"tech", "market"}
    assert "mean pairwise dependence" in report.summary()


def test_report_to_dict_is_json_safe(fake_embedding):
    import json

    report = aggregate(*planted_pair("paraphrase", 2, 8), fake_embedding)
    blob = json.dumps(report.to_dict())
    restored = json.loads(blob)
    assert restored["n_angles"] == 2
    assert len(restored["dependence_matrix"]) == 2
    assert set(restored["per_estimator"]) == {"embedding", "citation", "lexical"}
    assert restored["weights"] == DEFAULT_WEIGHTS


def test_lexical_survives_an_all_stopword_corpus():
    matrix, warnings = LexicalEstimator().estimate(
        {"tech": [claim("the and of")], "market": [claim("a an the")]}
    )
    assert matrix.shape == (2, 2)
    assert np.allclose(matrix.diagonal(), 1.0)
    assert any("failed" in w or "no claim text" in w for w in warnings)


def test_estimators_handle_no_angles_at_all(fake_embedding):
    for estimator in (fake_embedding, CitationEstimator(), LexicalEstimator()):
        matrix, warnings = estimator.estimate({})
        assert matrix.shape == (0, 0)
