"""Part 23 tests: the discount at the call site, the conflict detector, the bill.

Most tests use the deterministic bag-of-words encoder from the Part 21 suite.
Two do not: whether low cosine similarity means "unrelated" rather than
"contradictory" is a claim about real sentence semantics, and a bag of words
cannot settle it. Those load the real model once per session and skip if it
cannot be reached.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from prismflow.v2.dependence import DependenceReport, aggregate_dependence
from prismflow.v2.dependence.estimators import EmbeddingEstimator
from prismflow.v2.fusion import (
    MODEL_PRICING,
    SIMILARITY_FLOOR,
    ClaimConflict,
    ClaudeAdjudicator,
    FusedClaim,
    FusedRecommendation,
    LexicalAdjudicator,
    PrismFusion,
    conflict_candidates,
    estimate_cost_usd,
)
from prismflow.v2.reasoners.models import Claim, ClaimSet

from .test_dependence import bag_of_words_encoder, real_encoder  # noqa: F401

ANGLES = ["tech", "market", "financial", "regulatory", "sentiment"]


def claim(text: str, confidence: float = 0.8, ids=("r1",)) -> Claim:
    return Claim(text=text, confidence=confidence, cited_ids=list(ids))


def claimset(angle: str, claims, *, error=None, tokens=(1000, 300)) -> ClaimSet:
    return ClaimSet(
        angle_name=angle,
        query_text="q",
        claims=list(claims),
        provider="claude",
        model_name="claude-sonnet-5",
        tokens_input=tokens[0],
        tokens_output=tokens[1],
        latency_seconds=1.0,
        error=error,
    )


def report_for(angle_names, matrix=None) -> DependenceReport:
    n = len(angle_names)
    if matrix is None:
        matrix = np.eye(n)
    return DependenceReport(
        angle_names=list(angle_names), n_angles=n, dependence_matrix=matrix
    )


def fuse(fusion: PrismFusion, *args, **kwargs) -> FusedRecommendation:
    return asyncio.run(fusion.fuse(*args, **kwargs))


def offline(**kwargs) -> PrismFusion:
    """A fusion object that touches no model and no network."""
    kwargs.setdefault("encoder", bag_of_words_encoder)
    kwargs.setdefault("adjudicator", LexicalAdjudicator())
    return PrismFusion(**kwargs)


# --- cost ------------------------------------------------------------------


def test_cost_uses_published_per_million_rates():
    # Sonnet 5: $2 / $10 per MTok.
    assert estimate_cost_usd(1_000_000, 0, "claude-sonnet-5") == pytest.approx(2.00)
    assert estimate_cost_usd(0, 1_000_000, "claude-sonnet-5") == pytest.approx(10.00)
    assert estimate_cost_usd(500_000, 100_000, "claude-opus-5") == pytest.approx(
        0.5 * 5.00 + 0.1 * 25.00
    )


def test_cost_is_none_for_an_unpriced_model_rather_than_guessed():
    assert estimate_cost_usd(1000, 1000, "some-model-we-do-not-price") is None


def test_regression_briefs_cost_formula_is_fifty_times_the_real_rate():
    """``tokens_input * 0.0001`` is $100/MTok. Sonnet 5 input is $2/MTok."""
    tokens = 1_000_000
    brief = tokens * 0.0001
    actual = estimate_cost_usd(tokens, 0, "claude-sonnet-5")
    assert brief == pytest.approx(100.0)
    assert actual == pytest.approx(2.0)
    assert brief / actual == pytest.approx(50.0)


def test_every_priced_model_has_input_cheaper_than_output():
    for name, (inp, out) in MODEL_PRICING.items():
        assert 0 < inp < out, name


# --- conflict candidates ---------------------------------------------------


def test_candidates_are_cross_angle_only():
    """One angle disagreeing with itself is a reasoner defect, not a conflict."""
    claims = {
        "tech": [claim("adoption rose sharply"), claim("adoption fell sharply")],
    }
    candidates, _ = conflict_candidates(claims, encoder=bag_of_words_encoder)
    assert candidates == []


def test_candidates_need_citations_on_both_sides():
    claims = {
        "tech": [claim("adoption rose sharply this year", ids=["a"])],
        "market": [Claim(text="adoption rose sharply this year", confidence=0.7)],
    }
    candidates, _ = conflict_candidates(claims, encoder=bag_of_words_encoder)
    assert candidates == []


def test_candidates_respect_the_similarity_floor():
    claims = {
        "tech": [claim("alpha beta gamma delta epsilon")],
        "market": [claim("zeta eta theta iota kappa")],
    }
    candidates, _ = conflict_candidates(claims, encoder=bag_of_words_encoder)
    assert candidates == []  # nothing in common -> not a candidate


def test_candidates_are_returned_most_similar_first_and_capped():
    base = "adoption of the framework rose sharply in production during the year"
    claims = {
        "tech": [claim(base, ids=["a"])],
        "market": [claim(base, ids=["b"]), claim(base + " and beyond", ids=["c"])],
        "financial": [claim(base + " across regions", ids=["d"])],
    }
    candidates, _ = conflict_candidates(claims, encoder=bag_of_words_encoder)
    sims = [c.similarity for c in candidates]
    assert sims == sorted(sims, reverse=True)
    assert all(c.similarity >= SIMILARITY_FLOOR for c in candidates)

    capped, warnings = conflict_candidates(
        claims, encoder=bag_of_words_encoder, max_candidates=1
    )
    assert len(capped) == 1
    assert any("exceeded the cap" in w for w in warnings)
    # The dropped pairs must not be described as agreeing.
    assert any("unexamined" in w for w in warnings)


def test_candidates_start_unadjudicated():
    text = "adoption of the framework rose sharply in production"
    claims = {"tech": [claim(text, ids=["a"])], "market": [claim(text, ids=["b"])]}
    candidates, _ = conflict_candidates(claims, encoder=bag_of_words_encoder)
    assert candidates
    assert all(c.verdict == "unadjudicated" for c in candidates)
    assert not any(c.is_contradiction for c in candidates)


def test_encoder_returning_the_wrong_number_of_vectors_raises():
    def bad_encoder(texts):
        return np.zeros((len(list(texts)) + 1, 8))

    claims = {"tech": [claim("a b c")], "market": [claim("a b c")]}
    with pytest.raises(ValueError, match="misattribute"):
        conflict_candidates(claims, encoder=bad_encoder)


# --- the inverted comparison, measured -------------------------------------

CONTRADICTORY_PAIRS = [
    ("Revenue rose 20% year over year.", "Revenue fell 20% year over year."),
    ("The framework is production ready.", "The framework is not production ready."),
    ("Hiring in the sector is accelerating.", "Hiring in the sector is slowing down."),
    ("Latency improved after the migration.", "Latency got worse after the migration."),
]
UNRELATED_PAIRS = [
    ("Revenue rose 20% year over year.", "The library supports Python 3.12 and later."),
    ("The framework is production ready.", "Rainfall in the region peaked in March."),
    ("Hiring in the sector is accelerating.", "The parser handles nested JSON arrays."),
    ("Latency improved after the migration.", "The author lives in Lisbon."),
]


def test_regression_low_similarity_means_unrelated_not_contradictory(real_encoder):
    """The measurement the corrected detector rests on.

    The brief flags ``sim < 0.3`` as a conflict. Contradictions are about the
    same subject, so they embed CLOSE; unrelated claims embed far. If this ever
    inverts, the detector's premise is gone.
    """
    from prismflow.v2.dependence.estimators import cosine

    def similarity(a, b):
        va, vb = real_encoder([a, b])
        return cosine(np.asarray(va), np.asarray(vb))

    contradictory = [similarity(a, b) for a, b in CONTRADICTORY_PAIRS]
    unrelated = [similarity(a, b) for a, b in UNRELATED_PAIRS]

    assert min(contradictory) > max(unrelated)
    # The brief's rule: fires on none of the contradictions, all of the others.
    assert sum(s < 0.3 for s in contradictory) == 0
    assert sum(s < 0.3 for s in unrelated) == len(unrelated)
    # The corrected floor: the reverse.
    assert sum(s >= SIMILARITY_FLOOR for s in contradictory) == len(contradictory)
    assert sum(s >= SIMILARITY_FLOOR for s in unrelated) == 0


def test_real_contradiction_is_nominated_and_adjudicated(real_encoder):
    claims = {
        "tech": [claim("Revenue rose 20% year over year.", ids=["a"])],
        "market": [claim("Revenue fell 20% year over year.", ids=["b"])],
    }
    candidates, _ = conflict_candidates(claims, encoder=real_encoder)
    assert len(candidates) == 1
    verdicts = LexicalAdjudicator()(candidates)
    assert verdicts[0].is_contradiction


# --- the lexical adjudicator ----------------------------------------------


def pair(text_a: str, text_b: str, similarity: float = 0.8) -> ClaimConflict:
    return ClaimConflict(
        angle_a="tech",
        angle_b="market",
        text_a=text_a,
        text_b=text_b,
        similarity=similarity,
        verdict="unadjudicated",
    )


def test_lexical_adjudicator_catches_directional_opposites():
    verdict = LexicalAdjudicator()([pair("Revenue rose.", "Revenue fell.")])[0]
    assert verdict.is_contradiction
    assert "rose" in verdict.rationale


def test_lexical_adjudicator_catches_one_sided_negation():
    verdict = LexicalAdjudicator()(
        [pair("The toolchain is ready.", "The toolchain is not ready.")]
    )[0]
    assert verdict.is_contradiction


def test_lexical_adjudicator_says_unrelated_not_agreement_when_it_cannot_tell():
    """It must never claim to have checked and cleared a pair."""
    verdict = LexicalAdjudicator()(
        [pair("Revenue grew to $4M.", "Revenue was flat at $2M.")]
    )[0]
    assert verdict.verdict == "unrelated"
    assert verdict.verdict != "agreement"
    assert "cannot confirm agreement" in verdict.rationale


def test_lexical_adjudicator_is_deterministic():
    candidates = [pair("Revenue rose.", "Revenue fell.")]
    first = LexicalAdjudicator()(candidates)
    second = LexicalAdjudicator()(candidates)
    assert [c.verdict for c in first] == [c.verdict for c in second]


# --- the Claude adjudicator ------------------------------------------------


class FakeBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class FakeUsage:
    def __init__(self, i, o):
        self.input_tokens = i
        self.output_tokens = o


class FakeResponse:
    def __init__(self, text, tokens=(120, 45)):
        self.content = [FakeBlock(text)]
        self.usage = FakeUsage(*tokens)


class FakeMessages:
    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raises is not None:
            raise self._raises
        return self._response


class FakeClient:
    def __init__(self, response=None, raises=None):
        self.messages = FakeMessages(response, raises)


def test_claude_adjudicator_parses_verdicts_and_counts_tokens():
    reply = (
        '{"verdicts": [{"index": 0, "verdict": "contradiction", '
        '"rationale": "opposite direction"}]}'
    )
    client = FakeClient(FakeResponse(reply))
    adjudicator = ClaudeAdjudicator(client=client)
    out = asyncio.run(adjudicator([pair("Revenue rose.", "Revenue fell.")]))

    assert out[0].is_contradiction
    assert out[0].rationale == "opposite direction"
    assert adjudicator.tokens_input == 120
    assert adjudicator.tokens_output == 45
    assert client.messages.calls[0]["model"] == "claude-sonnet-5"


def test_claude_adjudicator_unparseable_reply_is_unadjudicated_not_agreement():
    client = FakeClient(FakeResponse("I could not decide, sorry."))
    out = asyncio.run(ClaudeAdjudicator(client=client)([pair("a rose", "a fell")]))
    assert out[0].verdict == "unadjudicated"


def test_claude_adjudicator_api_failure_is_unadjudicated_not_agreement():
    client = FakeClient(raises=RuntimeError("connection reset"))
    out = asyncio.run(ClaudeAdjudicator(client=client)([pair("a rose", "a fell")]))
    assert out[0].verdict == "unadjudicated"
    assert "connection reset" in out[0].rationale


def test_claude_adjudicator_ignores_out_of_range_and_invalid_verdicts():
    reply = (
        '{"verdicts": ['
        '{"index": 9, "verdict": "contradiction"},'
        '{"index": 0, "verdict": "definitely-conflicting"}]}'
    )
    client = FakeClient(FakeResponse(reply))
    out = asyncio.run(ClaudeAdjudicator(client=client)([pair("a rose", "a fell")]))
    assert out[0].verdict == "unadjudicated"


def test_claude_adjudicator_makes_no_call_for_no_candidates():
    client = FakeClient(FakeResponse("{}"))
    assert asyncio.run(ClaudeAdjudicator(client=client)([])) == []
    assert client.messages.calls == []


def test_conflict_verdict_is_validated():
    with pytest.raises(ValueError, match="verdict must be one of"):
        ClaimConflict("a", "b", "x", "y", 0.5, "maybe")


# --- fusion ----------------------------------------------------------------


def test_regression_n_comes_from_the_report_not_a_hardcoded_five():
    """The brief's ``compute_discount_factor(eniv, n=5)`` with 2 angles present."""
    sets = [
        claimset("tech", [claim("alpha beta")]),
        claimset("market", [claim("gamma delta")]),
    ]
    report = report_for(["tech", "market"])
    result = fuse(offline(), "q", sets, report)

    assert result.n_angles == 2
    # ENIV 2 over n 2 -> 1.0. The brief's n=5 would have reported 0.4.
    assert result.eniv == pytest.approx(2.0)
    assert result.discount_factor == pytest.approx(1.0)


def test_regression_a_failed_angle_does_not_become_a_successful_one():
    """The brief rebuilds each ClaimSet without ``error``. Nothing is rebuilt here."""
    sets = [
        claimset("tech", [claim("alpha beta")]),
        claimset("market", [claim("gamma delta")], error="APIStatusError 401"),
    ]
    report = report_for(["tech", "market"])
    result = fuse(offline(), "q", sets, report)

    assert result.failed_angles == ["market"]
    assert any("401" in line for line in result.audit_trail)
    # Its claims are not counted as a considered opinion.
    assert {fc.angle_name for fc in result.fused_claims} == {"tech"}


def test_a_failed_angle_can_be_kept_when_the_caller_asks():
    sets = [
        claimset("tech", [claim("alpha beta")]),
        claimset("market", [claim("gamma delta")], error="APIStatusError 401"),
    ]
    report = report_for(["tech", "market"])
    result = fuse(offline(drop_failed_angles=False), "q", sets, report)

    assert result.failed_angles == ["market"]
    assert {fc.angle_name for fc in result.fused_claims} == {"tech", "market"}


def test_discount_never_raises_a_confidence():
    sets = [
        claimset("tech", [claim("alpha beta", 0.9)]),
        claimset("market", [claim("alpha beta", 0.9)]),
    ]
    matrix = np.array([[1.0, 0.9], [0.9, 1.0]])
    result = fuse(offline(), "q", sets, report_for(["tech", "market"], matrix))

    assert result.fused_claims
    for fused in result.fused_claims:
        assert fused.discounted_confidence <= fused.raw_confidence
        assert 0.0 <= fused.discount_applied <= 1.0
    assert result.overall_confidence < result.undiscounted_confidence


def test_per_angle_discount_spares_the_angle_nobody_duplicated():
    matrix = np.eye(3)
    matrix[0, 1] = matrix[1, 0] = 1.0
    sets = [
        claimset("tech", [claim("alpha beta")]),
        claimset("market", [claim("alpha beta")]),
        claimset("financial", [claim("gamma delta")]),
    ]
    report = report_for(["tech", "market", "financial"], matrix)

    per_angle = fuse(offline(), "q", sets, report)
    scalar = fuse(offline(use_per_angle_discount=False), "q", sets, report)

    assert per_angle.per_angle_discount["financial"] == pytest.approx(1.0)
    assert per_angle.per_angle_discount["tech"] == pytest.approx(0.5)
    financial = next(
        fc for fc in per_angle.fused_claims if fc.angle_name == "financial"
    )
    assert financial.discount_applied == pytest.approx(1.0)

    scalar_financial = next(
        fc for fc in scalar.fused_claims if fc.angle_name == "financial"
    )
    # The scalar penalises an angle that was never duplicated.
    assert scalar_financial.discount_applied < 1.0
    assert scalar.discount_factor == pytest.approx(scalar_financial.discount_applied)


def test_a_verbose_angle_does_not_outvote_a_terse_one():
    """A plain mean over claims makes wordiness into evidence."""
    sets = [
        claimset("tech", [claim(f"alpha beta number {i}", 0.9) for i in range(10)]),
        claimset("market", [claim("gamma delta", 0.1)]),
    ]
    report = report_for(["tech", "market"])
    result = fuse(offline(), "q", sets, report)

    flat_mean = sum(fc.discounted_confidence for fc in result.fused_claims) / len(
        result.fused_claims
    )
    assert flat_mean > 0.8  # ten loud claims dominate
    assert result.overall_confidence == pytest.approx(0.5, abs=0.05)


def test_an_angle_missing_from_the_report_raises_rather_than_defaulting():
    sets = [
        claimset("tech", [claim("alpha beta")]),
        claimset("market", [claim("gamma delta")]),
    ]
    with pytest.raises(ValueError, match="absent from the dependence report"):
        fuse(offline(), "q", sets, report_for(["tech"]))


def test_eniv_is_recomputed_when_the_caller_omits_it():
    sets = [claimset("tech", [claim("alpha beta")])]
    report = report_for(["tech"])
    result = fuse(offline(), "q", sets, report)
    assert result.eniv == pytest.approx(1.0)
    assert any("computed from the report" in line for line in result.audit_trail)


def test_no_claims_gives_zero_confidence_not_a_crash():
    sets = [claimset("tech", []), claimset("market", [])]
    report = DependenceReport(
        angle_names=[],
        n_angles=0,
        dependence_matrix=np.zeros((0, 0)),
        excluded_angles=["tech", "market"],
    )
    result = fuse(offline(), "q", sets, report)
    assert result.fused_claims == []
    assert result.overall_confidence == 0.0
    assert result.conflicts == []


def test_empty_query_is_rejected():
    with pytest.raises(ValueError, match="non-empty string"):
        fuse(offline(), "   ", [], report_for([]))


def test_token_totals_include_angles_whose_claims_were_dropped():
    """A failed call is billed. A cost line that hides it understates the bill."""
    sets = [
        claimset("tech", [claim("alpha beta")], tokens=(1000, 300)),
        claimset("market", [], error="timeout", tokens=(800, 0)),
    ]
    result = fuse(offline(), "q", sets, report_for(["tech"]))
    assert result.tokens_input == 1800
    assert result.tokens_output == 300


def test_audit_trail_records_the_unimplemented_recency_weighting():
    sets = [claimset("tech", [claim("alpha beta")])]
    result = fuse(offline(), "q", sets, report_for(["tech"]))
    assert any("recency" in line.lower() for line in result.audit_trail)


def test_recommendation_round_trips_to_json_safe_types():
    import json

    sets = [
        claimset("tech", [claim("adoption rose sharply", 0.8)]),
        claimset("market", [claim("adoption fell sharply", 0.7)]),
    ]
    result = fuse(offline(), "q", sets, report_for(["tech", "market"]))
    payload = json.dumps(result.to_dict())
    assert "adoption rose sharply" in payload


def test_fusion_runs_against_a_real_dependence_report():
    """The live runtime, not a hand-built matrix."""
    sets = [
        claimset("tech", [claim("adoption of the framework rose sharply")]),
        claimset("market", [claim("adoption of the framework fell sharply")]),
        claimset("financial", []),
    ]
    report = aggregate_dependence(
        sets,
        estimators={"embedding": EmbeddingEstimator(encoder=bag_of_words_encoder)},
    )
    result = fuse(offline(), "framework adoption", sets, report)

    assert result.n_angles == 2
    assert result.excluded_angles == ["financial"]
    assert len(result.fused_claims) == 2
    assert result.contradictions and result.contradictions[0].is_contradiction


# --- the experiment --------------------------------------------------------


def load_experiment():
    path = (
        Path(__file__).resolve().parents[2]
        / "experiments"
        / "v2"
        / "test_fusion_e2e.py"
    )
    spec = importlib.util.spec_from_file_location("part23_experiment", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_regression_query_bank_actually_holds_thirty_queries():
    """The brief's ``[4 literals] * 6`` sliced to 30 yields 24, not 30."""
    experiment = load_experiment()
    total = sum(len(v) for v in experiment.QUERY_BANK.values())
    assert total >= experiment.N_QUERIES
    assert len(experiment.QUERY_BANK) == 4
    flat = [q for v in experiment.QUERY_BANK.values() for q in v]
    assert len(set(flat)) == len(flat)


def test_regression_seeds_vary_confidence_as_well_as_content():
    """An earlier draft fixed the confidences and reported sd exactly 0.0."""
    experiment = load_experiment()
    means = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        sets, _, _ = experiment.build_case(rng, "market", "q")
        confidences = [c.confidence for cs in sets for c in cs.claims]
        means.append(sum(confidences) / len(confidences))
    assert len(set(means)) == len(means)
    assert float(np.std(means)) > 0.0


def test_regression_seeds_are_reproducible():
    experiment = load_experiment()
    first, _, _ = experiment.build_case(np.random.default_rng(2), "tech", "q")
    second, _, _ = experiment.build_case(np.random.default_rng(2), "tech", "q")
    assert [c.text for cs in first for c in cs.claims] == [
        c.text for cs in second for c in cs.claims
    ]
    assert [c.confidence for cs in first for c in cs.claims] == [
        c.confidence for cs in second for c in cs.claims
    ]


def test_planted_subjects_are_a_contradiction_and_a_paraphrase():
    experiment = load_experiment()
    adjudicator = LexicalAdjudicator()
    for domain, triples in experiment.SUBJECTS.items():
        for assertion, reversal, paraphrase in triples:
            verdict = adjudicator([pair(assertion, reversal)])[0]
            assert verdict.is_contradiction, (domain, assertion, reversal)
            # The paraphrase must NOT read as a contradiction.
            assert not adjudicator([pair(assertion, paraphrase)])[0].is_contradiction


def test_experiment_gate_runs_end_to_end():
    """A short run: the plumbing and the contrast, not the published numbers."""
    experiment = load_experiment()
    payload = asyncio.run(
        experiment.run_fusion_e2e_test(num_seeds=2, num_queries=3)
    )

    assert payload["gate"]["queries_completed"] == 6
    assert payload["gate"]["all_queries_ran"]
    assert payload["gate"]["discount_applied_everywhere"]
    # The corrected detector finds the planted contradiction; the brief's does not.
    assert payload["summary"]["found_planted_contradiction"]["mean"] == 1.0
    assert payload["summary"]["brief_found_planted_contradiction"]["mean"] == 0.0
    # No credentials, so no invented dollar figure.
    assert payload["cost_usd"] is None
    assert payload["live_api_calls"] == 0
    assert "NEVER FABRICATE" in payload["cost_usd_reason"]
