"""StackExchange connector tests (Sentiment angle).

Offline tests run against RECORDED response bodies in ``tests/v2/fixtures/``,
captured live on 2026-09-24 -- including the oversized-body case, which is a
real 11834-character StackOverflow post rather than a synthetic one. The suite
makes no network call.

One live test is marked ``live``, following ``test_arxiv_connector_live``.
``pytest -m "not live"`` skips it, which is how CI should invoke this.

Four tests are labelled DEFECT-CLASS 1..4 and pin the checks the connector
brief named.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prismflow.v2.cache.models import hash_query
from prismflow.v2.connectors import (
    AuthError,
    ParseError,
    QueryError,
    RateLimitError,
    UpstreamError,
)
from prismflow.v2.connectors.base import RawResponse
from prismflow.v2.connectors.stackexchange import (
    QUOTA_WARN_THRESHOLD,
    SITE_SEP,
    SNIPPET_CHARS,
    StackExchangeConnector,
    epoch_to_utc,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str, status: int = 200) -> RawResponse:
    return RawResponse(
        status_code=status,
        headers={"content-type": "application/json"},
        body_text=(FIXTURES / name).read_text(encoding="utf-8"),
    )


@pytest.fixture
def connector():
    return StackExchangeConnector()


# --- 1. parse -----------------------------------------------------------


def test_parse_recorded_search_yields_records(connector):
    records = connector.parse(_fixture("stackexchange_search.json"))

    assert len(records) == 3
    first = records[0]
    assert first.id.isdigit()
    assert first.source_name == "stackexchange"
    assert first.url.startswith("https://stackoverflow.com/questions/")
    assert first.title
    assert first.snippet
    assert isinstance(first.extra["score"], int)
    assert isinstance(first.extra["tags"], list)


def test_parse_rejects_malformed_bodies(connector):
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, "not json"))
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, json.dumps({"nope": []})))
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, json.dumps({"items": [{}]})))


def test_parse_error_is_a_valueerror(connector):
    """Part 17 contract: parse raises ValueError on malformed input."""
    with pytest.raises(ValueError):
        connector.parse(RawResponse(200, {}, "{"))


def test_normalize_fills_the_part17_contract(connector):
    normalized = connector.normalize(
        connector.parse(_fixture("stackexchange_search.json"))
    )
    assert len(normalized) == 3
    for record in normalized:
        assert record.source == "stackexchange"
        assert record.snippet and isinstance(record.snippet, str)
        assert record.snippet_tokens > 0
        assert record.relevance_score == 1.0
        assert record.published_date is not None


# --- DEFECT-CLASS 1: Part 18 cache-key collision ------------------------


def test_defect_class_1_name_is_distinct_from_the_other_connectors():
    assert StackExchangeConnector().name == "stackexchange"
    assert StackExchangeConnector().name not in ("github", "arxiv", "yfinance")


def test_defect_class_1_site_is_inside_the_cache_key(connector):
    """Same terms, different site => different key.

    The Part 18 key has no site component, so a site passed as a constructor
    argument would be invisible to it: "inflation" on stackoverflow and on
    economics would hash identically and the second caller would be served the
    first's answer.
    """
    so, econ = "inflation@stackoverflow", "inflation@economics"
    assert hash_query(so) != hash_query(econ)
    assert connector._split_query(so) == ("inflation", "stackoverflow")
    assert connector._split_query(econ) == ("inflation", "economics")


def test_query_without_a_site_uses_the_default(connector):
    assert connector._split_query("inflation") == ("inflation", "stackoverflow")


def test_multiword_query_with_site_splits_on_the_last_separator(connector):
    terms, site = connector._split_query("rust async runtime@stackoverflow")
    assert (terms, site) == ("rust async runtime", "stackoverflow")


def test_empty_terms_rejected(connector):
    with pytest.raises(ValueError):
        connector._split_query(SITE_SEP + "stackoverflow")


def test_blank_default_site_rejected_at_construction():
    with pytest.raises(ValueError):
        StackExchangeConnector(default_site="   ")


# --- DEFECT-CLASS 2: rate limiting --------------------------------------


def test_defect_class_2_throttle_violation_is_not_an_upstream_error(connector):
    """error_id 502 must map to RateLimitError, not UpstreamError.

    This is the base-class override's whole reason for existing: StackExchange
    sends HTTP 502 for throttling, and AngleConnector._check_status maps any
    5xx to UpstreamError. Both retry, but Part 19 would be told "the source is
    broken" when the truth is "we are being throttled".
    """
    body = json.dumps(
        {
            "error_id": 502,
            "error_name": "throttle_violation",
            "error_message": "too many requests from this IP, more requests "
            "available in 7 seconds",
        }
    )
    with pytest.raises(RateLimitError) as caught:
        connector._check_status(RawResponse(502, {}, body))
    assert caught.value.retry_after == 7.0, "wait must be read out of the prose"
    assert caught.value.source == "stackexchange"


def test_defect_class_2_key_required_is_an_auth_error(connector):
    body = json.dumps(
        {"error_id": 405, "error_name": "key_required", "error_message": "key"}
    )
    with pytest.raises(AuthError) as caught:
        connector._check_status(RawResponse(405, {}, body))
    assert "stackapps" in str(caught.value)


def test_defect_class_2_bad_parameter_is_a_query_error(connector):
    """Uses the REAL recorded 400 body, not a hand-written one."""
    with pytest.raises(QueryError):
        connector._check_status(_fixture("stackexchange_error_400.json", status=400))


def test_defect_class_2_unknown_error_id_falls_back_to_upstream(connector):
    body = json.dumps({"error_id": 500, "error_name": "internal_error",
                       "error_message": "boom"})
    with pytest.raises(UpstreamError):
        connector._check_status(RawResponse(500, {}, body))


def test_defect_class_2_non_envelope_error_defers_to_the_base_class(connector):
    """An HTML 503 with no JSON envelope must still be classified."""
    with pytest.raises(UpstreamError):
        connector._check_status(RawResponse(503, {}, "<html>down</html>"))


def test_defect_class_2_backoff_on_a_200_defers_the_next_request(connector):
    """A SUCCESS can carry a throttle instruction. It must be honoured."""
    body = json.dumps({"items": [], "quota_remaining": 200, "backoff": 10})
    assert connector._backoff_until == 0.0
    connector._record_quota_and_backoff(RawResponse(200, {}, body))
    import time

    remaining = connector._backoff_until - time.monotonic()
    assert 8.0 < remaining <= 10.0, "backoff deadline not armed"


def test_quota_is_read_off_every_response(connector, caplog):
    body = json.dumps({"items": [], "quota_remaining": 293, "quota_max": 300})
    connector._record_quota_and_backoff(RawResponse(200, {}, body))
    assert connector.quota_remaining == 293


def test_low_quota_warns(connector, caplog):
    import logging

    body = json.dumps(
        {"items": [], "quota_remaining": QUOTA_WARN_THRESHOLD - 1, "quota_max": 300}
    )
    with caplog.at_level(logging.WARNING):
        connector._record_quota_and_backoff(RawResponse(200, {}, body))
    assert any("stackapps" in r.getMessage() for r in caplog.records)


def test_throttle_spaces_calls(run_async):
    """Base-class spacing still applies through the override."""
    import time

    connector = StackExchangeConnector(rate_limit_per_min=600)

    async def ten():
        start = time.monotonic()
        for _ in range(10):
            await connector._throttle()
        return time.monotonic() - start

    assert run_async(ten()) >= 0.9
    assert connector.request_count == 10


# --- DEFECT-CLASS 3: timezone -------------------------------------------


def test_defect_class_3_creation_date_becomes_utc_not_local(connector):
    """creation_date is a NAIVE Unix epoch int; UTC must be explicit.

    datetime.fromtimestamp without tz= applies the local zone -- +05:30 on the
    machine this was written on -- which would put every StackExchange post
    5.5 hours earlier than an arXiv paper from the same instant.
    """
    body = json.loads(_fixture("stackexchange_search.json").body_text)
    raw_epochs = [i["creation_date"] for i in body["items"]]
    assert all(isinstance(e, int) for e in raw_epochs), "fixture must be epoch ints"

    for record in connector.normalize(
        connector.parse(_fixture("stackexchange_search.json"))
    ):
        assert record.published_date.tzinfo is not None
        assert record.published_date.utcoffset().total_seconds() == 0


def test_epoch_to_utc_is_explicit_about_the_zone():
    # 1642619486 == 2022-01-19T19:11:26Z, regardless of local zone
    assert epoch_to_utc(1642619486) == datetime(
        2022, 1, 19, 19, 11, 26, tzinfo=timezone.utc
    )


def test_epoch_to_utc_rejects_junk():
    assert epoch_to_utc(None) is None
    assert epoch_to_utc("not a number") is None
    assert epoch_to_utc(True) is None, "bool is not an epoch"


# --- DEFECT-CLASS 4: oversized snippet ----------------------------------


def test_defect_class_4_real_oversized_body_is_cut_to_budget(connector):
    """The recorded fixture is a genuine 11834-char post.

    Raw it is 3455 tokens and tag-stripped it is still 2783 -- both over the
    entire 2000-token angle budget, so stripping alone is not enough and
    Part 19 would drop the record.
    """
    body = json.loads(_fixture("stackexchange_oversized.json").body_text)
    assert len(body["items"][0]["body"]) > 10000, "fixture is not oversized"

    normalized = connector.normalize(
        connector.parse(_fixture("stackexchange_oversized.json"))
    )
    record = normalized[0]
    assert len(record.snippet) <= SNIPPET_CHARS
    assert record.snippet_tokens < 2000, "would be dropped by Part 19"
    assert record.snippet_tokens < 250, "a truncated snippet is small"


def test_defect_class_4_html_is_stripped_not_carried(connector):
    records = connector.parse(_fixture("stackexchange_oversized.json"))
    snippet = records[0].snippet
    for tag in ("<p>", "</p>", "<a ", "<blockquote>", "<code>", "&lt;", "&amp;"):
        assert tag not in snippet, f"{tag!r} survived into the snippet"


def test_defect_class_4_guard_fires_on_an_unbounded_field():
    from prismflow.v2.connectors.stackexchange import _assert_snippet_bounded

    _assert_snippet_bounded("x" * SNIPPET_CHARS)
    with pytest.raises(ParseError):
        _assert_snippet_bounded("x" * (SNIPPET_CHARS + 1))


def test_every_snippet_in_every_fixture_fits_the_budget(connector):
    for name in ("stackexchange_search.json", "stackexchange_oversized.json"):
        for record in connector.normalize(connector.parse(_fixture(name))):
            assert len(record.snippet) <= SNIPPET_CHARS
            assert record.snippet_tokens < 2000


# --- the sentiment prefix -----------------------------------------------


def test_sentiment_prefix_carries_the_score(connector):
    """The score is the sentiment signal and must reach the snippet.

    Part 19 reranks on snippet and Part 20 reads snippet, so a score that
    lived only in extra would be invisible to both.
    """
    record = connector.parse(_fixture("stackexchange_search.json"))[0]
    assert record.snippet.startswith("[score ")
    assert "|" in record.snippet.split("]")[0]


def test_sentiment_prefix_is_bounded(connector):
    """A huge score/view count must not push the snippet over the cap."""
    body = json.loads(_fixture("stackexchange_search.json").body_text)
    item = dict(
        body["items"][0], score=999999, answer_count=4321, view_count=987654321
    )
    raw = RawResponse(200, {}, json.dumps({"items": [item]}))
    assert len(connector.parse(raw)[0].snippet) <= SNIPPET_CHARS


# --- fetch argument validation ------------------------------------------


def test_fetch_rejects_empty_query_and_bad_k(run_async, connector):
    with pytest.raises(ValueError):
        run_async(connector.fetch("", k=5))
    with pytest.raises(ValueError):
        run_async(connector.fetch("python", k=0))


# --- live ---------------------------------------------------------------


@pytest.mark.live
def test_stackexchange_connector_live(run_async):
    """One real anonymous API call. Skipped by ``pytest -m "not live"``."""
    connector = StackExchangeConnector()
    try:
        records = run_async(connector.get_records("transformer attention", k=3))
    finally:
        run_async(connector.aclose())

    assert len(records) >= 1, "live StackExchange query returned nothing"
    assert len(records) <= 3
    for record in records:
        assert record.id
        assert record.source == "stackexchange"
        assert record.snippet
        assert record.snippet_tokens < 2000
        assert record.published_date.utcoffset().total_seconds() == 0
    assert connector.quota_remaining is not None, "quota must be read"
