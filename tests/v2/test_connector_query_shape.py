"""The query SHAPE each connector puts on the wire -- V2-L2 and V2-L3.

These are regression tests for two live-measured defects, both of which were
silent: the connectors returned HTTP 200 and the pipeline recorded a record
count, so nothing downstream could tell that the evidence did not answer the
question asked.

  V2-L3  ArxivConnector sent ``all:{query}`` verbatim. Handed a prose sentence
         -- which happened on 36 of the 48 Part 24 rows, because BaseAngle only
         derives a query that EXCEEDS the connector's limit and arXiv's limit of
         220 is the most permissive in the table -- arXiv bound none of its
         terms and returned its default top-relevance listing: the SAME ten
         big-collaboration physics papers for every unrelated query.

  V2-L2  GitHubConnector forwarded 8-12 space-separated terms. GitHub ANDs
         those, so every one of the 48 rows came back 200 with total_count 0.

The central assertion is the NEGATIVE one: two unrelated queries must not
produce the same request. These tests are offline -- they assert on the URL and
on the search_query string, never on what the live API returns, so they cannot
fail because arXiv reranked something overnight.
"""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

import pytest

from prismflow.v2.connectors.arxiv import (
    MAX_SEARCH_TERMS,
    SEARCH_QUERY_LIMIT,
    ArxivConnector,
)
from prismflow.v2.connectors.github import AND_TERMS, GitHubConnector

# Three real Part 24 pitches, the rows docs/v2-known-limitations.md names for
# V2-L3. All three are under arXiv's 220-char limit, which is exactly why they
# used to arrive as prose.
PITCH_020 = (
    "Pitch a standard way to write built-in generic collections such as "
    "list[str] and dict[str, int] for static typing."
)
PITCH_046 = (
    "Build a regulated crypto exchange and financial platform for buying, "
    "selling, storing, and using digital assets."
)
PITCH_049 = (
    "Build a low-cost international money-transfer product with a transparent "
    "exchange-rate and fee model."
)
UNRELATED = [PITCH_020, PITCH_046, PITCH_049]


# --------------------------------------------------------------------------
# arXiv
# --------------------------------------------------------------------------

def test_arxiv_never_puts_raw_prose_in_search_query():
    """The V2-L3 defect itself: a prose sentence pasted after ``all:``."""
    search = ArxivConnector().build_search_query(PITCH_046)
    # The giveaway is the stopwords. A term query cannot contain " and " or
    # " for ", because derivation drops them.
    assert " and " not in search
    assert " for " not in search
    assert "all:Build a regulated" not in search
    # Every term carries its own field prefix rather than riding on the first.
    assert search.startswith("all:")
    assert search.count("all:") >= 3


def test_arxiv_two_unrelated_queries_do_not_produce_the_same_request():
    """THE NEGATIVE TEST. This is the whole of V2-L3.

    Before the fix all three of these produced a request arXiv answered with
    one identical record set. The requests must differ from each other.
    """
    conn = ArxivConnector()
    searches = [conn.build_search_query(p) for p in UNRELATED]
    assert len(set(searches)) == len(UNRELATED), (
        "unrelated pitches produced a duplicate search_query: %r" % searches
    )
    # Stronger: no two share a majority of their terms.
    termsets = [set(re.findall(r'all:"?([^"\s]+)', s)) for s in searches]
    for i in range(len(termsets)):
        for j in range(i + 1, len(termsets)):
            shared = termsets[i] & termsets[j]
            smaller = min(len(termsets[i]), len(termsets[j]))
            assert len(shared) < smaller, (
                "pitches %d and %d share every term: %r" % (i, j, shared)
            )


def test_arxiv_respects_a_caller_that_speaks_arxiv_syntax():
    """An explicit field prefix means the caller knows the query language."""
    explicit = 'ti:"attention is all you need" AND cat:cs.LG'
    assert ArxivConnector().build_search_query(explicit) == explicit


def test_arxiv_quotes_multi_word_phrases():
    """Unquoted, arXiv reads only a phrase's first word and widens the query."""
    search = ArxivConnector().build_search_query(
        "A proposal about AVFoundation Camera behaviour on simulator builds."
    )
    assert 'all:"AVFoundation Camera"' in search


def test_arxiv_caps_the_term_count():
    long_pitch = " ".join("alpha%d" % i for i in range(60))
    search = ArxivConnector().build_search_query(long_pitch)
    assert search.count("all:") <= MAX_SEARCH_TERMS


def test_arxiv_never_emits_an_empty_search_query():
    """An empty search_query is an HTTP 400; a poor query is merely poor."""
    for awkward in ("the and of", "...", "a"):
        search = ArxivConnector().build_search_query(awkward)
        assert search.strip()
        assert search.strip() != "all:"


def test_arxiv_url_carries_the_built_query_not_the_prose(monkeypatch, run_async):
    """End to end through fetch(): the prose must not reach the wire.

    Driven through the repo's own ``run_async`` fixture -- V2 is async but the
    project deliberately takes no pytest-asyncio dependency.
    """
    conn = ArxivConnector()
    seen = {}

    async def fake_get(url, headers=None):
        seen["url"] = url
        return None

    monkeypatch.setattr(conn, "_get", fake_get)
    run_async(conn.fetch(PITCH_046, 10))
    query = parse_qs(urlparse(seen["url"]).query)["search_query"][0]
    assert "Build a regulated crypto exchange" not in query
    assert query.count("all:") >= 3


def test_github_url_carries_the_reduced_q(monkeypatch, run_async):
    """The GitHub side of the same check."""
    conn = GitHubConnector()
    seen = {}

    async def fake_get(url, headers=None):
        seen["url"] = url
        return None

    monkeypatch.setattr(conn, "_get", fake_get)
    run_async(conn.fetch(PITCH_046, 10))
    q = parse_qs(urlparse(seen["url"]).query)["q"][0]
    assert "Build a regulated crypto exchange" not in q
    assert len(re.findall(r'"[^"]*"|\S+', q)) == AND_TERMS


# --------------------------------------------------------------------------
# GitHub
# --------------------------------------------------------------------------

def test_github_no_longer_receives_eight_to_twelve_anded_terms():
    """The V2-L2 defect: a long conjunction that matches no repository.

    GitHub ANDs space-separated terms, so term count IS the conjunction width.
    """
    q = GitHubConnector().build_q(PITCH_046)
    # Count terms the way GitHub does: quoted phrases are one term.
    terms = re.findall(r'"[^"]*"|\S+', q)
    assert len(terms) == AND_TERMS, (
        "expected %d ANDed terms, got %d: %r" % (AND_TERMS, len(terms), q)
    )
    assert len(terms) <= 3, "a conjunction this wide measured 0 records on 48/48 rows"


def test_github_two_unrelated_queries_do_not_produce_the_same_request():
    """The negative test, GitHub side."""
    conn = GitHubConnector()
    qs = [conn.build_q(p) for p in UNRELATED]
    assert len(set(qs)) == len(UNRELATED), "duplicate GitHub q: %r" % qs


def test_github_stays_under_the_five_boolean_operator_ceiling():
    """GitHub 422s above five AND/OR/NOT operators -- measured, not assumed.

    Joining every derived term with OR is what trips it, and is why the fix
    ANDs a couple of terms instead.
    """
    for pitch in UNRELATED:
        q = GitHubConnector().build_q(pitch)
        operators = len(re.findall(r"\b(?:AND|OR|NOT)\b", q))
        assert operators <= 5


def test_github_quotes_phrases_so_they_do_not_widen_the_conjunction():
    q = GitHubConnector().build_q(
        "A proposal about AVFoundation Camera behaviour on simulator builds."
    )
    assert '"AVFoundation Camera"' in q


def test_github_respects_its_own_qualifier_syntax():
    for explicit in ("streaming language:python", "parser stars:>100"):
        assert GitHubConnector().build_q(explicit) == explicit


def test_github_never_emits_an_empty_q():
    for awkward in ("the and of", "...", "a"):
        assert GitHubConnector().build_q(awkward).strip()


# --------------------------------------------------------------------------
# The property that ties both defects together
# --------------------------------------------------------------------------

@pytest.mark.parametrize("build", [
    lambda p: ArxivConnector().build_search_query(p),
    lambda p: GitHubConnector().build_q(p),
])
def test_neither_connector_forwards_a_stopword_laden_sentence(build):
    """Both defects had one shape: prose reaching a field that takes a query.

    A query containing " a " or " the " is a sentence, not a term list.
    """
    out = build(PITCH_049)
    assert " a " not in out
    assert " the " not in out
    assert " with " not in out
