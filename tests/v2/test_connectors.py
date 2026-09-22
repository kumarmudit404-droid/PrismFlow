"""Part 17 connector tests.

Seven tests, matching the Part 17 brief's numbered list. Two of them (4 and 5)
hit live APIs and are marked ``live`` -- run ``pytest -m "not live"`` to skip
them on a machine without network.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import pytest

from prismflow.v2.connectors import (
    ArxivConnector,
    AuthError,
    GitHubConnector,
    ParseError,
    QueryError,
    RateLimitError,
)
from prismflow.v2.connectors.github import (
    SEARCH_RATE_AUTHENTICATED,
    SEARCH_RATE_UNAUTHENTICATED,
    _resolve_token,
)

from .conftest import ARXIV_BODY, GITHUB_BODY, MockHTTPClient, ok, status


# --- 1. parse: GitHub ---------------------------------------------------


def test_parse_github_response():
    connector = GitHubConnector(http_client=MockHTTPClient([ok(GITHUB_BODY)]))
    records = connector.parse(ok(GITHUB_BODY))

    assert len(records) == 2

    first = records[0]
    assert first.id == "1296269"
    assert first.title == "hello-world"
    assert first.url == "https://github.com/octocat/hello-world"
    assert first.snippet.startswith("My first repository")
    assert first.source_name == "github"
    assert first.extra["author"] == "octocat"
    assert first.extra["published_date"] == datetime(
        2011, 1, 26, 19, 1, 12, tzinfo=timezone.utc
    )

    # A null description must become "", not the string "None" -- that string
    # would be tokenized and counted as real content downstream.
    assert records[1].snippet == ""


def test_parse_github_rejects_malformed_bodies():
    connector = GitHubConnector(http_client=MockHTTPClient([ok("{}")]))

    with pytest.raises(ParseError):
        connector.parse(ok("not json at all"))
    with pytest.raises(ParseError):
        connector.parse(ok('{"message": "Bad credentials"}'))  # no 'items'
    with pytest.raises(ParseError):
        connector.parse(ok('{"items": "should-be-a-list"}'))

    # The brief says parse raises ValueError on malformed input; ParseError
    # must therefore remain a ValueError subclass.
    with pytest.raises(ValueError):
        connector.parse(ok("not json at all"))


# --- 2. parse: arXiv ----------------------------------------------------


def test_parse_arxiv_response():
    connector = ArxivConnector(http_client=MockHTTPClient([ok(ARXIV_BODY)]))
    records = connector.parse(ok(ARXIV_BODY))

    assert len(records) == 2

    first = records[0]
    assert first.id == "1412.6572v3"
    assert first.url == "https://arxiv.org/abs/1412.6572v3"
    # arXiv hard-wraps titles and abstracts; the newline must be collapsed.
    assert first.title == "Explaining and Harnessing Adversarial Examples"
    assert "\n" not in first.snippet
    assert first.snippet.startswith("Several machine learning models")
    # First author only, per the brief.
    assert first.extra["author"] == "Ian J. Goodfellow"
    assert first.extra["published_date"] == datetime(
        2014, 12, 20, 0, 24, 2, tzinfo=timezone.utc
    )

    assert records[1].extra["author"] == "Aleksander Madry"


def test_parse_arxiv_rejects_malformed_xml():
    connector = ArxivConnector(http_client=MockHTTPClient([ok(ARXIV_BODY)]))
    with pytest.raises(ParseError):
        connector.parse(ok("<feed><entry></feed>"))
    with pytest.raises(ValueError):
        connector.parse(ok("<feed><entry></feed>"))


# --- 3. normalize -------------------------------------------------------


def test_normalize_records(raw_records_fixture):
    connector = GitHubConnector(http_client=MockHTTPClient([ok("{}")]))
    normalized = connector.normalize(raw_records_fixture)

    assert len(normalized) == 2

    first = normalized[0]
    assert first.id == "1296269"
    assert first.source == "github"
    assert first.author == "octocat"
    assert first.published_date == datetime(2011, 1, 26, tzinfo=timezone.utc)
    # Real token count for a 52-character sentence: small but non-zero.
    assert 5 <= first.snippet_tokens <= 20
    assert first.relevance_score == 1.0

    # Empty snippet is zero tokens, and missing author/date stay None rather
    # than becoming empty strings.
    second = normalized[1]
    assert second.snippet_tokens == 0
    assert second.author is None
    assert second.published_date is None


def test_normalize_is_consistent_across_sources(github_records_fixture,
                                                arxiv_records_fixture):
    """Both sources must produce the same NormalizedRecord contract.

    Part 21 estimates dependence BETWEEN angles. If one source left
    relevance_score unset or dates naive, that difference would show up as
    apparent structure between angles.
    """
    for record in [*github_records_fixture, *arxiv_records_fixture]:
        assert record.relevance_score == 1.0
        assert record.snippet_tokens >= 0
        assert record.source in {"github", "arxiv"}
        assert record.published_date is not None
        assert record.published_date.tzinfo is not None


# --- 4. live: GitHub ----------------------------------------------------


@pytest.mark.live
def test_github_connector_live(run_async):
    connector = GitHubConnector()
    try:
        records = run_async(connector.get_records("machine learning", k=5))
    finally:
        run_async(connector.aclose())

    assert len(records) >= 1, "live GitHub search returned nothing"
    assert len(records) <= 5
    for record in records:
        assert record.id
        assert record.url.startswith("https://github.com/")
        assert record.source == "github"
        assert record.snippet_tokens >= 0


# --- 5. live: arXiv -----------------------------------------------------


@pytest.mark.live
def test_arxiv_connector_live(run_async):
    connector = ArxivConnector()
    try:
        records = run_async(
            connector.get_records("adversarial robustness", k=5)
        )
    finally:
        run_async(connector.aclose())

    assert len(records) >= 1, "live arXiv query returned nothing"
    assert len(records) <= 5
    for record in records:
        assert record.id
        assert record.url.startswith("https://arxiv.org/abs/")
        assert record.source == "arxiv"


# --- 6. rate limiting ---------------------------------------------------


def test_rate_limiting(run_async):
    """Ten rapid requests must be spaced by the configured limit.

    600/min means 0.1s between request starts, so ten requests cannot finish
    in under 0.9s (the first is free). Kept at 600 rather than a realistic 10
    so the test costs ~1s instead of ~54s.
    """
    import asyncio

    client = MockHTTPClient([ok(GITHUB_BODY)])
    connector = GitHubConnector(rate_limit_per_min=600, http_client=client)

    async def hammer():
        await asyncio.gather(*(connector.fetch("x", 1) for _ in range(10)))

    start = time.monotonic()
    run_async(hammer())
    elapsed = time.monotonic() - start

    assert len(client.calls) == 10
    assert connector.request_count == 10
    assert elapsed >= 0.9 * 0.9, (
        f"10 requests at 600/min finished in {elapsed:.3f}s; "
        "throttling did not apply"
    )
    # Sanity: without throttling this would be near-instant.
    assert elapsed < 10.0


def test_rate_limit_defaults_track_authentication():
    """Search limits, not core REST limits. See github.py module docstring."""
    unauth = GitHubConnector(token=None,
                             http_client=MockHTTPClient([ok("{}")]))
    assert unauth.authenticated is False
    assert unauth.rate_limit == SEARCH_RATE_UNAUTHENTICATED

    auth = GitHubConnector(token="ghp_realdeadbeefrealdeadbeef",
                           http_client=MockHTTPClient([ok("{}")]))
    assert auth.authenticated is True
    assert auth.rate_limit == SEARCH_RATE_AUTHENTICATED


def test_placeholder_token_is_treated_as_absent(monkeypatch):
    """A template placeholder must not be sent as a credential.

    Measured: ``Authorization: Bearer ghp_...`` returns 401, while sending no
    header at all returns 200. Forwarding the placeholder is strictly worse
    than having no token.
    """
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_...")
    assert _resolve_token(None) is None

    monkeypatch.setenv("GITHUB_TOKEN", "   ")
    assert _resolve_token(None) is None

    monkeypatch.setenv("GITHUB_TOKEN", "ghp_realdeadbeefrealdeadbeef")
    assert _resolve_token(None) == "ghp_realdeadbeefrealdeadbeef"

    connector = GitHubConnector(token="ghp_...",
                                http_client=MockHTTPClient([ok("{}")]))
    assert connector.authenticated is False
    assert "Authorization" not in connector._headers()


# --- 7. error handling --------------------------------------------------


def test_error_handling(run_async):
    """401/422 raise immediately; 403/429/5xx retry then raise."""
    # 401 -> AuthError, no retry.
    client = MockHTTPClient([status(401, '{"message": "Bad credentials"}')])
    connector = GitHubConnector(http_client=client, max_retries=3)
    with pytest.raises(AuthError):
        run_async(connector.fetch("q", 5))
    assert len(client.calls) == 1, "401 must not be retried"

    # 422 -> QueryError, no retry.
    client = MockHTTPClient([status(422, '{"message": "Validation Failed"}')])
    connector = GitHubConnector(http_client=client, max_retries=3)
    with pytest.raises(QueryError):
        run_async(connector.fetch("q", 5))
    assert len(client.calls) == 1, "422 must not be retried"

    # 403 -> RateLimitError, retried max_retries times then raised.
    client = MockHTTPClient([status(403, "rate limit exceeded",
                                    **{"retry-after": "0"})])
    connector = GitHubConnector(rate_limit_per_min=6000, http_client=client,
                                max_retries=2)
    with pytest.raises(RateLimitError):
        run_async(connector.fetch("q", 5))
    assert len(client.calls) == 3, "403 should be attempted 1 + 2 retries"

    # 429 is throttling too.
    client = MockHTTPClient([status(429, "slow down",
                                    **{"retry-after": "0"})])
    connector = GitHubConnector(rate_limit_per_min=6000, http_client=client,
                                max_retries=1)
    with pytest.raises(RateLimitError):
        run_async(connector.fetch("q", 5))
    assert len(client.calls) == 2

    # A transient 500 followed by success must succeed.
    client = MockHTTPClient([status(500, "boom"), ok(GITHUB_BODY)])
    connector = GitHubConnector(rate_limit_per_min=6000, http_client=client,
                                max_retries=3)
    raw = run_async(connector.fetch("q", 5))
    assert raw.status_code == 200
    assert len(client.calls) == 2


def test_invalid_arguments_are_rejected_before_any_request(run_async):
    client = MockHTTPClient([ok(GITHUB_BODY)])
    connector = GitHubConnector(http_client=client)

    with pytest.raises(ValueError):
        run_async(connector.fetch("", 5))
    with pytest.raises(ValueError):
        run_async(connector.fetch("valid", 0))
    assert client.calls == [], "no request should be made for invalid input"
