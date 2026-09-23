"""Part 19 tests: fallback, reranking, budget, provenance.

Covers the brief's seven cases and adds a regression per defect found in the
specified implementation. Budget tests use ``rerank_method="none"`` so the
budget is measured on a known order rather than on whatever BM25 chose --
otherwise a budget test silently becomes a ranking test.

Angles are built with ``use_cache=False`` unless the test is about caching, so
no test writes to the developer's real ./cache/v2 or depends on what a previous
test left there.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import List, Optional, Sequence

import pytest

from prismflow.v2.angles import (
    ANGLE_CLASSES,
    AngleEvidence,
    FinancialAngle,
    MarketAngle,
    RegulatoryAngle,
    SentimentAngle,
    TechAngle,
    build_angles,
    load_config,
    rerank_records,
    rerank_with_outcome,
)
from prismflow.v2.angles.config import AngleConfig
from prismflow.v2.angles.reranker import document_text, tokenize
from prismflow.v2.cache import CacheMetrics
from prismflow.v2.connectors.base import AngleConnector, NormalizedRecord
from prismflow.v2.connectors.errors import RateLimitError, UpstreamError

from .conftest import make_arxiv_records, make_github_records


# --- doubles ------------------------------------------------------------


class FakeConnector(AngleConnector):
    """A connector that returns canned records, or raises, and counts calls."""

    def __init__(
        self,
        name: str,
        records: Optional[Sequence[NormalizedRecord]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        super().__init__(name, rate_limit_per_min=600)
        self.records = list(records or [])
        self.error = error
        self.calls: List[tuple] = []

    async def fetch(self, query, k):  # pragma: no cover - never reached
        raise AssertionError("FakeConnector.get_records is overridden")

    def parse(self, raw):  # pragma: no cover
        raise AssertionError("FakeConnector.get_records is overridden")

    def normalize(self, records):  # pragma: no cover
        raise AssertionError("FakeConnector.get_records is overridden")

    async def get_records(self, query: str, k: int) -> List[NormalizedRecord]:
        self.calls.append((query, k))
        if self.error is not None:
            raise self.error
        return self.records[:k]


def make_records(
    tokens: Sequence[int],
    *,
    source: str = "fake",
    snippet: str = "placeholder snippet text",
) -> List[NormalizedRecord]:
    """Records with exact token counts, for budget tests."""
    return [
        NormalizedRecord(
            id=f"{source}-{i}",
            title=f"record {i}",
            url=f"https://example.invalid/{source}/{i}",
            snippet=snippet,
            snippet_tokens=count,
            source=source,
            published_date=None,
            author=None,
            relevance_score=1.0,
        )
        for i, count in enumerate(tokens)
    ]


@pytest.fixture
def config():
    return load_config("configs/v2/angle_defaults.yaml")


# --- the brief's seven --------------------------------------------------


def test_tech_angle_retrieve(run_async, config):
    """Mock GitHub + arXiv: AngleEvidence with provenance from both."""
    github = FakeConnector("github", make_github_records())
    arxiv = FakeConnector("arxiv", make_arxiv_records())
    angle = TechAngle(github, arxiv, config=config["tech"], use_cache=False)

    evidence = run_async(angle.retrieve("adversarial robustness"))

    assert isinstance(evidence, AngleEvidence)
    assert evidence.angle_name == "tech"
    assert evidence.query_text == "adversarial robustness"
    # k=10, GitHub has 5, so arXiv is asked for the remaining 5.
    assert github.calls == [("adversarial robustness", 10)]
    assert arxiv.calls == [("adversarial robustness", 5)]
    assert evidence.provenance == {"github": 5, "arxiv": 5}
    assert len(evidence.raw_records) == 10
    assert evidence.total_tokens == sum(
        r.snippet_tokens for r in evidence.ranked_records
    )
    assert set(evidence.source_names) == {"github", "arxiv"}


def test_rerank_bm25(config):
    """BM25 changes the order and the top record scores highest."""
    records = make_github_records()
    before = [r.title for r in records]

    ranked = rerank_records("adversarial robustness testing", records)

    assert [r.title for r in ranked] != before, "reranking changed nothing"
    assert ranked[0].relevance_score == pytest.approx(1.0)
    scores = [r.relevance_score for r in ranked]
    assert scores == sorted(scores, reverse=True)
    # keploy's snippet is the only one about adversarial robustness testing.
    assert ranked[0].title == "keploy"


@pytest.mark.live
def test_rerank_cross_encoder():
    """Cross-encoder reranking, when the model can actually be loaded.

    Marked live: it downloads ~80MB on first use. The offline behaviour that
    matters on a fresh checkout is covered by
    ``test_regression_cross_encoder_falls_back_to_bm25``.
    """
    records = make_arxiv_records()
    outcome = rerank_with_outcome(
        "defences that give a false sense of security", records,
        method="cross-encoder",
    )
    if outcome.method_used != "cross-encoder":
        pytest.skip(f"cross-encoder unavailable: {outcome.warning}")
    assert len(outcome.records) == len(records)
    assert all(0.0 <= r.relevance_score <= 1.0 for r in outcome.records)
    scores = [r.relevance_score for r in outcome.records]
    assert scores == sorted(scores, reverse=True)


def test_token_budget_enforcement(run_async):
    """[500, 800, 900, 500] under a 2000 budget.

    The brief's own numbers, and they show its two specifications disagreeing:
    its code breaks at the first overflow (keeping 500+800 = 1300, two records)
    while its test asserts "only first 3 included", which is impossible because
    500+800+900 = 2200 > 2000. Both policies are implemented, so the difference
    is visible here rather than argued about.
    """
    records = make_records([500, 800, 900, 500])

    prefix = TechAngle(
        FakeConnector("fake", records), k=10, token_budget=2000,
        rerank_method="none", budget_policy="prefix", use_cache=False,
    )
    evidence = run_async(prefix.retrieve("q"))
    assert [r.snippet_tokens for r in evidence.ranked_records] == [500, 800]
    assert evidence.total_tokens == 1300
    assert evidence.total_tokens <= 2000
    assert evidence.truncated
    assert any("token budget 2000" in w for w in evidence.warnings)

    greedy = TechAngle(
        FakeConnector("fake", records), k=10, token_budget=2000,
        rerank_method="none", budget_policy="greedy", use_cache=False,
    )
    evidence = run_async(greedy.retrieve("q"))
    assert [r.snippet_tokens for r in evidence.ranked_records] == [500, 800, 500]
    assert evidence.total_tokens == 1800
    assert evidence.total_tokens <= 2000


def test_fallback_on_primary_failure(run_async, config):
    """A primary that raises is skipped and the secondary is queried."""
    github = FakeConnector(
        "github", error=RateLimitError("throttled", source="github",
                                       status_code=403),
    )
    arxiv = FakeConnector("arxiv", make_arxiv_records())
    angle = TechAngle(github, arxiv, config=config["tech"], use_cache=False)

    evidence = run_async(angle.retrieve("adversarial robustness"))

    assert github.calls, "primary should still have been attempted"
    # The secondary is asked for the full k, since the primary yielded nothing.
    assert arxiv.calls == [("adversarial robustness", 10)]
    assert evidence.provenance == {"github": 0, "arxiv": 5}
    assert len(evidence.raw_records) == 5
    assert any("github (primary) failed" in w for w in evidence.warnings)
    assert any("RateLimitError" in w for w in evidence.warnings)


def test_provenance_tracking(run_async, config):
    """Provenance sums to the raw record count, per contributing source."""
    github = FakeConnector("github", make_github_records()[:3])
    arxiv = FakeConnector("arxiv", make_arxiv_records()[:2])
    angle = TechAngle(github, arxiv, config=config["tech"], use_cache=False)

    evidence = run_async(angle.retrieve("neural networks"))

    assert evidence.provenance == {"github": 3, "arxiv": 2}
    assert sum(evidence.provenance.values()) == len(evidence.raw_records) == 5
    assert evidence.source_names == ["github", "arxiv"]


def test_all_five_angles(run_async, config):
    """All five angles answer the same query and return AngleEvidence.

    Only tech has connectors -- Part 17 built GitHub and arXiv and no more --
    so the other four return empty evidence with an explicit warning. That is
    the rescoped gate for this part: five angles orchestrate, one retrieves.
    """
    angles = build_angles(
        config,
        sources={"tech": (FakeConnector("github", make_github_records()),
                          FakeConnector("arxiv", make_arxiv_records()))},
        use_cache=False,
    )
    assert set(angles) == set(ANGLE_CLASSES)

    results = {
        name: run_async(angle.retrieve("machine learning"))
        for name, angle in angles.items()
    }

    assert len(results) == 5
    for name, evidence in results.items():
        assert isinstance(evidence, AngleEvidence)
        assert evidence.angle_name == name
        assert evidence.query_text == "machine learning"

    assert not results["tech"].is_empty
    assert results["tech"].provenance == {"github": 5, "arxiv": 5}
    for name in ("market", "financial", "regulatory", "sentiment"):
        assert results[name].is_empty
        assert results[name].total_tokens == 0
        assert any("no connector is configured" in w
                   for w in results[name].warnings), name


# --- regressions against the specified implementation -------------------


def test_regression_rerank_ranks_on_snippet_not_a_token_count():
    """Ranking must use snippet TEXT, which the brief could not do.

    The brief's corpus line is ``r.title.split() + r.snippet_tokens.split()``,
    and snippet_tokens is an int, so it raises AttributeError on every call.
    This test also shows why title-only ranking was rejected as the fix:
    keploy's title contains none of the query terms and only its snippet does,
    so a title-only reranker could not surface it at all.
    """
    records = make_github_records()
    keploy = next(r for r in records if r.title == "keploy")
    assert "adversarial" not in keploy.title.lower()
    assert "adversarial" in keploy.snippet.lower()

    ranked = rerank_records("adversarial robustness", records)
    assert ranked[0].title == "keploy"

    # And the int that used to be ranked on is untouched and still an int.
    assert isinstance(ranked[0].snippet_tokens, int)
    assert "adversarial" in document_text(keploy).lower()


def test_regression_rerank_does_not_mutate_its_inputs():
    """The brief assigns record.relevance_score in place.

    Those same objects are reported as AngleEvidence.raw_records, documented as
    unranked, and MockGitHubConnector hands out the same objects on every call --
    so in-place scoring leaks between calls and makes "raw" mean "ranked".
    """
    records = make_github_records()
    originals = [r.relevance_score for r in records]

    ranked = rerank_records("adversarial robustness", records)

    assert [r.relevance_score for r in records] == originals
    assert any(r.relevance_score != 1.0 for r in ranked), "nothing was scored"
    assert all(a is not b for a in records for b in ranked)


def test_regression_raw_records_stay_unranked(run_async, config):
    """AngleEvidence.raw_records must not carry the ranking's scores."""
    angle = TechAngle(
        FakeConnector("github", make_github_records()),
        config=config["tech"], use_cache=False,
    )
    evidence = run_async(angle.retrieve("adversarial robustness"))

    assert all(r.relevance_score == 1.0 for r in evidence.raw_records)
    assert evidence.ranked_records[0].relevance_score == pytest.approx(1.0)
    assert [r.title for r in evidence.raw_records] != [
        r.title for r in evidence.ranked_records
    ]


def test_regression_tertiary_source_is_reached(run_async, config):
    """The brief accepts a tertiary connector and never queries it.

    Its retrieve() mentions self.primary and self.secondary only, so standing
    rule 2 ("if secondary fails, tertiary") cannot hold.
    """
    github = FakeConnector("github", error=UpstreamError("503", source="github"))
    arxiv = FakeConnector("arxiv", error=UpstreamError("503", source="arxiv"))
    third = FakeConnector("pwc", make_records([10, 10]))
    angle = TechAngle(github, arxiv, third, config=config["tech"],
                      use_cache=False)

    evidence = run_async(angle.retrieve("q"))

    assert third.calls, "tertiary was never queried"
    assert evidence.provenance == {"github": 0, "arxiv": 0, "pwc": 2}
    assert len(evidence.raw_records) == 2


def test_regression_chain_stops_once_k_is_satisfied(run_async):
    """A satisfied chain must not spend a second source's rate limit."""
    github = FakeConnector("github", make_github_records())
    arxiv = FakeConnector("arxiv", make_arxiv_records())
    angle = TechAngle(github, arxiv, k=3, rerank_method="none", use_cache=False)

    evidence = run_async(angle.retrieve("q"))

    assert github.calls == [("q", 3)]
    assert arxiv.calls == [], "secondary queried although k was already met"
    assert evidence.provenance == {"github": 3}


def test_regression_unexpected_errors_are_not_swallowed(run_async):
    """A code defect must not be reported as a failed source.

    The brief catches bare Exception per source, which would have turned its own
    reranker AttributeError into "GitHub fetch failed: AttributeError" -- a
    defect disguised as a network problem.
    """

    class BrokenConnector(FakeConnector):
        async def get_records(self, query, k):
            raise TypeError("a programming error, not a source failure")

    angle = TechAngle(BrokenConnector("broken"), use_cache=False)
    with pytest.raises(TypeError, match="programming error"):
        run_async(angle.retrieve("q"))


def test_regression_rerank_failure_warns_and_does_not_raise(run_async):
    """Standing rule 4: a failed rerank degrades, it does not raise.

    An angle constructed directly with a bad method bypasses the config's
    load-time validation, which is exactly the runtime path rule 4 covers: the
    retrieval still succeeds, the records come back in retrieval order, and the
    degradation is reported in the evidence rather than only in a log.
    """
    angle = TechAngle(
        FakeConnector("github", make_github_records()),
        rerank_method="definitely-not-a-method", use_cache=False,
    )
    evidence = run_async(angle.retrieve("q"))

    assert len(evidence.raw_records) == 5
    assert len(evidence.ranked_records) == 5
    assert [r.id for r in evidence.ranked_records] == [
        r.id for r in evidence.raw_records
    ]
    assert any("unknown rerank method" in w for w in evidence.warnings)


def test_regression_bm25_failure_degrades_to_retrieval_order(monkeypatch):
    """If BM25 itself raises, records come back in order with a warning."""
    import prismflow.v2.angles.reranker as reranker

    def boom(query, records):
        raise RuntimeError("bm25 exploded")

    monkeypatch.setattr(reranker, "_rerank_bm25", boom)
    records = make_github_records()
    outcome = reranker.rerank_with_outcome("q", records, "bm25")

    assert [r.title for r in outcome.records] == [r.title for r in records]
    assert outcome.method_used == "none"
    assert "bm25 rerank failed" in outcome.warning


def test_regression_cross_encoder_falls_back_to_bm25(monkeypatch):
    """An unloadable cross-encoder must fall back to BM25, not to unsorted.

    Standing rule 4 says "return unsorted", but BM25 needs no model and no
    network, so discarding a ranking that is free to compute would lose
    evidence quality for nothing. Deliberate strengthening, recorded here.
    """
    import prismflow.v2.angles.reranker as reranker

    def no_model(query, records):
        raise OSError("model not found and no network")

    monkeypatch.setattr(reranker, "_rerank_cross_encoder", no_model)
    records = make_github_records()
    outcome = reranker.rerank_with_outcome(
        "adversarial robustness", records, "cross-encoder",
    )

    assert outcome.method_used == "bm25"
    assert "falling back to bm25" in outcome.warning
    assert outcome.records[0].title == "keploy", "bm25 did not actually run"


def test_regression_ranking_is_stable_and_deterministic():
    """Equal scores keep retrieval order, and repeated runs agree.

    Part 21 estimates dependence between angles from these orderings; a ranking
    that permuted ties would make apparent dependence an artefact of
    tie-breaking rather than a fact about the sources.
    """
    records = make_records([10, 10, 10, 10], snippet="identical text everywhere")
    first = rerank_records("identical text", records)
    second = rerank_records("identical text", records)

    assert [r.id for r in first] == [r.id for r in second]
    assert [r.id for r in first] == [r.id for r in records]


def test_regression_no_term_overlap_preserves_order():
    """A query matching nothing must not shuffle the records."""
    records = make_github_records()
    outcome = rerank_with_outcome("zzzzqqqq nonexistentterm", records, "bm25")

    assert [r.id for r in outcome.records] == [r.id for r in records]
    assert outcome.all_zero
    assert all(r.relevance_score == 0.0 for r in outcome.records)


def test_regression_rerank_handles_empty_and_blank_corpora():
    """Neither an empty list nor all-blank documents may raise.

    All-blank matters specifically: BM25Okapi divides by the average document
    length, which is zero when every document is empty.
    """
    assert rerank_records("q", []) == []

    blank = make_records([0, 0], snippet="")
    for record in blank:
        object.__setattr__(record, "title", "")
    outcome = rerank_with_outcome("q", blank, "bm25")
    assert len(outcome.records) == 2
    assert "nothing to rank on" in (outcome.warning or "")


def test_regression_tokenize_is_case_insensitive():
    """BM25 on raw .split() would not match a capitalised title.

    The corpus needs more than one document and the term must not appear in all
    of them: BM25's IDF is non-positive for a term present in every document, so
    a single-document corpus scores zero however well it matches.
    """
    assert tokenize("Adversarial Robustness") == ["adversarial", "robustness"]

    shouty, quiet_a, quiet_b = make_records([5, 5, 5])
    shouty = replace(shouty, snippet="ADVERSARIAL ROBUSTNESS EVERYWHERE")
    quiet_a = replace(quiet_a, snippet="unrelated notes about databases")
    quiet_b = replace(quiet_b, snippet="more unrelated notes about compilers")

    ranked = rerank_records(
        "adversarial robustness", [quiet_a, quiet_b, shouty],
    )
    assert ranked[0].id == shouty.id, "uppercase text did not match a lowercase query"
    assert ranked[0].relevance_score > 0.0


def test_regression_duplicate_records_are_dropped_once(run_async):
    """The same record from two slots must be counted once.

    Otherwise provenance would not sum to len(raw_records) and the angle would
    present the same evidence twice as if it were two findings.
    """
    shared = make_github_records()[:3]
    first = FakeConnector("github", shared)
    second = FakeConnector("github", shared)
    angle = TechAngle(first, second, k=10, rerank_method="none", use_cache=False)

    evidence = run_async(angle.retrieve("q"))

    assert len(evidence.raw_records) == 3
    assert sum(evidence.provenance.values()) == 3
    assert any("duplicate" in w for w in evidence.warnings)


def test_regression_budget_smaller_than_any_record_says_so(run_async):
    """An impossible budget must not look like a retrieval failure."""
    angle = TechAngle(
        FakeConnector("fake", make_records([900, 1200])),
        k=10, token_budget=100, rerank_method="none", use_cache=False,
    )
    evidence = run_async(angle.retrieve("q"))

    assert evidence.ranked_records == []
    assert evidence.total_tokens == 0
    assert len(evidence.raw_records) == 2, "retrieval itself worked"
    assert any("exceed the entire 100-token budget" in w
               for w in evidence.warnings)


def test_regression_one_oversized_record_does_not_empty_the_angle(run_async):
    """Measured shape from live GitHub results for "machine learning".

    The funNLP repository's description is 5835 characters -- 5065 tokens,
    against a 2000-token budget -- while the other nine results in the same
    response totalled 150 tokens. Under the brief's break-on-overflow rule that
    single record truncates the angle wherever it ranks, and ranked first it
    empties the angle entirely while leaving the whole budget unspent.

    Oversized records are therefore removed before budgeting: they cannot fit at
    any position under any policy, so keeping them in the queue only destroys
    the records below them.
    """
    # The oversized record deliberately ranks FIRST, the worst case.
    records = make_records([5065] + [17] * 9)
    angle = TechAngle(
        FakeConnector("github", records),
        k=10, token_budget=2000, rerank_method="none", use_cache=False,
    )
    evidence = run_async(angle.retrieve("machine learning"))

    assert len(evidence.ranked_records) == 9, "the other records were lost"
    assert evidence.total_tokens == 153
    assert evidence.total_tokens <= 2000
    assert all(r.snippet_tokens <= 2000 for r in evidence.ranked_records)
    assert any("exceed the entire 2000-token budget" in w
               for w in evidence.warnings)
    assert any("5065 tokens" in w for w in evidence.warnings)
    # It was retrieved; it is simply unusable within this budget.
    assert len(evidence.raw_records) == 10


# --- AngleEvidence invariants ------------------------------------------


def test_angle_evidence_rejects_ranked_records_not_in_raw():
    raw = make_github_records()[:2]
    stranger = make_arxiv_records()[:1]
    with pytest.raises(ValueError, match="subset"):
        AngleEvidence(
            query_text="q", angle_name="tech", raw_records=raw,
            ranked_records=stranger, total_tokens=24,
            provenance={"github": 2},
        )


def test_angle_evidence_rejects_wrong_total_tokens():
    raw = make_records([10, 20])
    with pytest.raises(ValueError, match="total_tokens"):
        AngleEvidence(
            query_text="q", angle_name="tech", raw_records=raw,
            ranked_records=raw, total_tokens=999, provenance={"fake": 2},
        )


def test_angle_evidence_rejects_provenance_that_does_not_add_up():
    raw = make_records([10, 20])
    with pytest.raises(ValueError, match="provenance accounts for"):
        AngleEvidence(
            query_text="q", angle_name="tech", raw_records=raw,
            ranked_records=raw, total_tokens=30, provenance={"fake": 1},
        )


def test_angle_evidence_summary_reads_cleanly():
    raw = make_records([10, 20])
    evidence = AngleEvidence(
        query_text="q", angle_name="tech", raw_records=raw,
        ranked_records=raw[:1], total_tokens=10, provenance={"fake": 2},
        warnings=["something degraded"],
    )
    assert "tech: 1/2 records, 10 tokens" in evidence.summary()
    assert "fake=2" in evidence.summary()
    assert evidence.truncated


# --- config ------------------------------------------------------------


def test_config_loads_all_five_angles(config):
    assert set(config) == {"tech", "market", "financial", "regulatory",
                           "sentiment"}
    assert config["tech"].k == 10
    assert config["tech"].token_budget == 2000
    assert config["regulatory"].rerank_method == "cross-encoder"
    assert config["sentiment"].fallback_strategy == ("primary", "secondary")
    assert all(c.budget_policy in ("prefix", "greedy") for c in config.values())


def test_config_ttl_is_the_stricter_of_angle_and_connector(config):
    """Angle TTL is a ceiling, not an override.

    The brief sets TTL per angle while Part 18 sets it per connector, and they
    disagree. Taking the minimum respects both: an angle may demand fresher data
    than a source's default, but no angle can make a fast-moving source look
    fresh for longer than the source allows.
    """
    # tech wants 3600; arXiv's own default is 86400. Stricter wins.
    assert config["tech"].effective_ttl("arxiv") == 3600
    # regulatory wants 604800; GitHub's default is 3600. Stricter still wins.
    assert config["regulatory"].effective_ttl("github") == 3600
    # An unknown connector falls back to Part 18's default of 3600.
    assert config["regulatory"].effective_ttl("unheard-of") == 3600


def test_config_rejects_an_unknown_rerank_method(tmp_path):
    """A typo must fail at load, not degrade silently on every query."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        "angles:\n  tech:\n    k: 5\n    token_budget: 100\n"
        "    ttl_seconds: 60\n    rerank_method: bm52\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="rerank_method"):
        load_config(path)


@pytest.mark.parametrize(
    "body, match",
    [
        ("angles:\n  tech:\n    token_budget: 1\n    ttl_seconds: 1\n",
         "missing required key 'k'"),
        ("angles:\n  tech:\n    k: 0\n    token_budget: 1\n    ttl_seconds: 1\n",
         "must be positive"),
        ("angles:\n  tech:\n    k: 1\n    token_budget: 1\n    ttl_seconds: 1\n"
         "    fallback_strategy: []\n", "is empty"),
        ("angles:\n  tech:\n    k: 1\n    token_budget: 1\n    ttl_seconds: 1\n"
         "    fallback_strategy: [primary, primary]\n", "repeats a slot"),
        ("angles:\n  tech:\n    k: 1\n    token_budget: 1\n    ttl_seconds: 1\n"
         "    fallback_strategy: [quaternary]\n", "unknown slot"),
        ("angles:\n  tech:\n    k: 1\n    token_budget: 1\n    ttl_seconds: 1\n"
         "    budget_policy: whatever\n", "budget_policy"),
        ("not_angles: {}\n", "top-level 'angles'"),
    ],
)
def test_config_validation_rejects_bad_values(tmp_path, body, match):
    path = tmp_path / "bad.yaml"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_config_missing_file_is_explicit(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.yaml")


def test_explicit_arguments_beat_config(config):
    angle = TechAngle(k=3, token_budget=77, config=config["tech"],
                      use_cache=False)
    assert angle.k == 3
    assert angle.token_budget == 77
    assert angle.rerank_method == "bm25"  # still from config


def test_angle_defaults_apply_without_a_config():
    assert SentimentAngle(use_cache=False).k == 15
    assert SentimentAngle(use_cache=False).token_budget == 2500
    assert RegulatoryAngle(use_cache=False).rerank_method == "cross-encoder"
    assert MarketAngle(use_cache=False).k == 8
    assert FinancialAngle(use_cache=False).token_budget == 2000


@pytest.mark.parametrize("bad", [0, -5])
def test_angle_rejects_nonsense_sizes(bad):
    with pytest.raises(ValueError, match="must be positive"):
        TechAngle(k=bad, use_cache=False)
    with pytest.raises(ValueError, match="must be positive"):
        TechAngle(token_budget=bad, use_cache=False)


def test_angle_rejects_a_blank_query(run_async):
    angle = TechAngle(FakeConnector("github", make_github_records()),
                      use_cache=False)
    with pytest.raises(ValueError, match="non-empty string"):
        run_async(angle.retrieve("   "))


# --- integration with Part 18 ------------------------------------------


def test_angle_retrieval_goes_through_the_part18_cache(
    run_async, tmp_path, config
):
    """A repeated retrieval must not re-query the connector."""
    github = FakeConnector("github", make_github_records())
    metrics = CacheMetrics()
    angle = TechAngle(
        github, config=config["tech"], use_cache=True,
        cache_db_path=str(tmp_path / "angles.db"), cache_metrics=metrics,
    )

    first = run_async(angle.retrieve("adversarial robustness"))
    second = run_async(angle.retrieve("adversarial robustness"))

    assert len(github.calls) == 1, "the second retrieval refetched"
    assert metrics.hits == 1 and metrics.misses == 1
    assert [r.id for r in second.ranked_records] == [
        r.id for r in first.ranked_records
    ]
    # The amendment's payoff: snippets survive the cache, so the cached
    # retrieval ranks as well as the fresh one did.
    assert all(r.snippet for r in second.ranked_records)
    assert second.ranked_records[0].title == "keploy"


def test_concurrent_angles_share_one_cache(run_async, tmp_path, config):
    """Five angles retrieving at once is the Part 23 shape; it must be safe."""
    metrics = CacheMetrics()
    angles = build_angles(
        config,
        sources={"tech": (FakeConnector("github", make_github_records()),)},
        use_cache=True,
        cache_db_path=str(tmp_path / "concurrent.db"),
        cache_metrics=metrics,
    )

    async def scenario():
        return await asyncio.gather(
            *(angle.retrieve("machine learning") for angle in angles.values())
        )

    results = run_async(scenario())
    assert len(results) == 5
    assert {e.angle_name for e in results} == set(ANGLE_CLASSES)


def test_angle_logs_what_it_queried(run_async, caplog, config):
    angle = TechAngle(FakeConnector("github", make_github_records()),
                      config=config["tech"], use_cache=False)
    with caplog.at_level(logging.INFO, logger="prismflow.v2.angles"):
        run_async(angle.retrieve("adversarial robustness"))

    assert "[tech] querying primary (github)" in caplog.text
    assert "records" in caplog.text and "tokens" in caplog.text


def test_configured_sources_reports_only_real_connectors(config):
    angle = TechAngle(FakeConnector("github"), None, FakeConnector("pwc"),
                      config=config["tech"], use_cache=False)
    assert angle.configured_sources == ["github", "pwc"]
    assert "github" in repr(angle)
