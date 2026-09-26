"""Query derivation tests: the limits, the empty cases, and the wiring.

Written against docs/part24-query-mismatch.md. The run that motivated this had
16 of 17 rows retrieve nothing, so the cases that matter most here are the ones
where extraction produces little or nothing -- a deriver that works on a
well-formed pitch and returns an empty string on an awkward one would swap a
long-query failure for an empty-query failure and look like progress.

The wiring tests assert the ADDITIVE part: with derivation off, the connector
receives the caller's query unchanged, byte for byte.
"""

from __future__ import annotations

import pytest

from prismflow.v2.angles.query_derivation import (
    CONNECTOR_QUERY_LIMITS,
    DEFAULT_QUERY_LIMIT,
    derive_for_connector,
    derive_query,
    limit_for,
)
from prismflow.v2.angles.tech_angle import TechAngle

from .test_angles import FakeConnector
from .conftest import make_github_records

# The exact row that produced NewsAPI's queryTooLong. Its length is the point.
LONG_PITCH = (
    "This PEP proposes allowing parentheses in the two-argument form of "
    "assert. The interpreter will reinterpret assert (expr, msg) as assert "
    "expr, msg, eliminating a long-standing footgun in which a two-element "
    "tuple is always truthy and the assertion therefore never fires. Swift "
    "Evolution considered a similar change. The proposal covers only the "
    "two-argument form and leaves the one-argument form untouched, so "
    "existing code continues to parse exactly as it does today. Static "
    "analysers have warned about the pattern for years, and the Steering "
    "Council weighed that history when considering whether new syntax was "
    "warranted for a problem that tooling already surfaces reliably."
)


# --------------------------------------------------------------------------
# limits
# --------------------------------------------------------------------------
def test_the_regression_case_now_fits_under_newsapis_limit():
    """585 chars was rejected outright; the derived query must clear 500."""
    assert len(LONG_PITCH) > 500
    derived = derive_for_connector(LONG_PITCH, "newsapi")
    assert len(derived.text) <= CONNECTOR_QUERY_LIMITS["newsapi"] < 500


@pytest.mark.parametrize("connector", sorted(CONNECTOR_QUERY_LIMITS))
def test_every_connector_limit_is_respected(connector):
    derived = derive_for_connector(LONG_PITCH, connector)
    assert len(derived.text) <= CONNECTOR_QUERY_LIMITS[connector]


def test_unknown_connector_falls_back_to_the_default_limit():
    assert limit_for("not-a-connector") == DEFAULT_QUERY_LIMIT
    assert limit_for(None) == DEFAULT_QUERY_LIMIT
    derived = derive_for_connector(LONG_PITCH, "not-a-connector")
    assert len(derived.text) <= DEFAULT_QUERY_LIMIT


def test_it_never_cuts_a_word_in_half():
    for limit in (20, 37, 60, 150):
        text = derive_query(LONG_PITCH, limit=limit).text
        assert text == text.strip()
        for word in text.split():
            assert word in LONG_PITCH or word.lower() in LONG_PITCH.lower()


# --------------------------------------------------------------------------
# the cases that produce little or nothing
# --------------------------------------------------------------------------
def test_pitch_with_no_extractable_keywords_falls_back_rather_than_emptying():
    """Stopwords only. An empty query is worse than a poor one."""
    derived = derive_query("the and or of to in on at it is", limit=80)
    assert not derived.is_empty
    assert derived.fallback == "no extractable keywords in pitch"


def test_punctuation_only_pitch_falls_back():
    derived = derive_query("--- ... !!! ???", limit=80)
    assert not derived.is_empty
    assert derived.fallback is not None


@pytest.mark.parametrize("pitch", ["", "   ", "\n\t "])
def test_empty_pitch_is_empty_and_says_so_without_raising(pitch):
    derived = derive_query(pitch, limit=80)
    assert derived.is_empty
    assert derived.fallback == "pitch was empty"


def test_a_single_term_longer_than_the_limit_is_trimmed_not_dropped():
    derived = derive_query("supercalifragilisticexpialidocious", limit=10)
    assert not derived.is_empty
    assert len(derived.text) <= 10


@pytest.mark.parametrize("bad", [0, -1])
def test_non_positive_limits_raise(bad):
    with pytest.raises(ValueError):
        derive_query(LONG_PITCH, limit=bad)
    with pytest.raises(ValueError):
        derive_query(LONG_PITCH, max_terms=bad)


# --------------------------------------------------------------------------
# what it extracts
# --------------------------------------------------------------------------
def test_it_is_deterministic():
    """A sampled step here would break Part 18's cache and Part 21's measurement."""
    first = derive_query(LONG_PITCH, limit=150)
    for _ in range(5):
        assert derive_query(LONG_PITCH, limit=150) == first


def test_multi_word_proper_nouns_survive_as_phrases():
    derived = derive_query(LONG_PITCH, limit=200)
    assert "Swift Evolution" in derived.text


def test_sentence_initial_capitalisation_is_not_treated_as_a_name():
    """'This PEP proposes' opens a sentence; it is not the name of anything."""
    derived = derive_query(LONG_PITCH, limit=200)
    assert "This PEP" not in derived.terms
    assert "pep" in [t.lower() for t in derived.terms]


def test_stopwords_and_bare_numbers_are_dropped():
    derived = derive_query(
        "The system processes 1234 requests using a queue and the broker.",
        limit=200)
    lowered = [t.lower() for t in derived.terms]
    assert "the" not in lowered and "1234" not in lowered
    assert "queue" in lowered and "broker" in lowered


# --------------------------------------------------------------------------
# the wiring: additive, and off by default
# --------------------------------------------------------------------------
def test_derivation_off_sends_the_query_unchanged(run_async):
    """The existing contract. This is what must not break."""
    connector = FakeConnector("github", make_github_records())
    angle = TechAngle(connector, None, None, use_cache=False)
    assert angle.derive_queries is False
    run_async(angle.retrieve(LONG_PITCH))
    assert connector.calls[0][0] == LONG_PITCH


def test_derivation_on_shortens_a_long_query_to_the_connector_limit(run_async):
    connector = FakeConnector("github", make_github_records())
    angle = TechAngle(connector, None, None, use_cache=False, derive_queries=True)
    run_async(angle.retrieve(LONG_PITCH))
    sent = connector.calls[0][0]
    assert sent != LONG_PITCH
    assert len(sent) <= CONNECTOR_QUERY_LIMITS["github"]


def test_derivation_on_leaves_an_already_short_query_alone(run_async):
    """It already fits; rewriting it would discard the caller's wording."""
    connector = FakeConnector("github", make_github_records())
    angle = TechAngle(connector, None, None, use_cache=False, derive_queries=True)
    run_async(angle.retrieve("exception groups"))
    assert connector.calls[0][0] == "exception groups"


def test_derivation_records_what_it_did_in_the_evidence_warnings(run_async):
    connector = FakeConnector("github", make_github_records())
    angle = TechAngle(connector, None, None, use_cache=False, derive_queries=True)
    evidence = run_async(angle.retrieve(LONG_PITCH))
    assert any("query derived for github" in w for w in evidence.warnings)


def test_each_connector_gets_its_own_limit_not_a_shared_one(run_async):
    """GitHub's ceiling is tighter than arXiv's; one string cannot serve both."""
    github = FakeConnector("github", [])
    arxiv = FakeConnector("arxiv", make_github_records())
    angle = TechAngle(github, arxiv, None, use_cache=False, derive_queries=True)
    run_async(angle.retrieve(LONG_PITCH))
    sent_github = github.calls[0][0]
    sent_arxiv = arxiv.calls[0][0]
    assert len(sent_github) <= CONNECTOR_QUERY_LIMITS["github"]
    assert len(sent_arxiv) <= CONNECTOR_QUERY_LIMITS["arxiv"]
    assert len(sent_arxiv) >= len(sent_github)
