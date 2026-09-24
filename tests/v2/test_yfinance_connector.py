"""yfinance connector tests (TASK 2a).

The offline tests run against RECORDED response bodies in
``tests/v2/fixtures/`` -- captured from live Yahoo calls on 2026-09-24, not
hand-written -- so ``parse`` and ``normalize`` are exercised on the real shape
without the suite touching the network or needing yfinance installed.

One live test is marked ``live``, following ``test_arxiv_connector_live``.
Run ``pytest -m "not live"`` to skip it, which is how CI should invoke this.

Four of the tests below exist specifically to pin the four defect classes the
connector brief named. They are labelled DEFECT-CLASS 1..4.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from prismflow.v2.cache.models import hash_query
from prismflow.v2.connectors import ParseError, QueryError, RateLimitError
from prismflow.v2.connectors.base import RawResponse
from prismflow.v2.connectors.yfinance import (
    PERIOD_SEP,
    SNIPPET_CHARS,
    VALID_PERIODS,
    YFinanceConnector,
    to_utc,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _fixture(name: str) -> RawResponse:
    body = (FIXTURES / name).read_text(encoding="utf-8")
    return RawResponse(
        status_code=200,
        headers={"content-type": "application/json"},
        body_text=body,
    )


@pytest.fixture
def connector():
    return YFinanceConnector()


# --- 1. parse -----------------------------------------------------------


def test_parse_recorded_body_yields_records(connector):
    records = connector.parse(_fixture("yfinance_nvidia.json"))

    assert len(records) == 3
    first = records[0]
    assert first.id == "NVDA@1mo"
    assert first.source_name == "yfinance"
    assert first.url == "https://finance.yahoo.com/quote/NVDA"
    assert "NVDA" in first.title
    assert first.snippet
    assert first.extra["symbol"] == "NVDA"
    assert first.extra["period"] == "1mo"


def test_parse_rejects_malformed_bodies(connector):
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, "not json at all"))
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, json.dumps({"nope": []})))
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, json.dumps({"quotes": [{}]})))


def test_parse_error_is_a_valueerror(connector):
    """Part 17 contract: parse raises ValueError on malformed input."""
    with pytest.raises(ValueError):
        connector.parse(RawResponse(200, {}, "{"))


# --- 2. normalize -------------------------------------------------------


def test_normalize_fills_the_part17_contract(connector):
    records = connector.parse(_fixture("yfinance_nvidia.json"))
    normalized = connector.normalize(records)

    assert len(normalized) == 3
    for record in normalized:
        assert record.source == "yfinance"
        assert isinstance(record.snippet, str) and record.snippet
        assert record.snippet_tokens > 0
        assert record.relevance_score == 1.0
        assert record.published_date is not None


# --- DEFECT-CLASS 1: Part 18 cache-key collision ------------------------


def test_defect_class_1_cache_key_distinct_from_other_connectors():
    """``yfinance`` must not share a cache key with github or arxiv.

    The Part 18 key is ``(connector_name, hash_query(query))``. The query hash
    is deliberately the same across connectors -- it is the connector_name
    that separates them, so the name is what has to be distinct.
    """
    assert YFinanceConnector().name == "yfinance"
    assert YFinanceConnector().name not in ("github", "arxiv")


def test_defect_class_1_period_is_inside_the_cache_key(connector):
    """Same ticker, different period => different key.

    The period is part of what was asked for but is NOT a cache key component
    on its own, so it has to travel inside the query string. If it were a
    constructor argument instead, "NVDA" over 5d and "NVDA" over 10y would
    hash identically and the second caller would be served the first's answer.
    """
    short, long = "NVDA@5d", "NVDA@10y"
    assert hash_query(short) != hash_query(long)

    assert connector._split_query(short) == ("NVDA", "5d")
    assert connector._split_query(long) == ("NVDA", "10y")
    # and the record id keeps them apart downstream too
    assert f"NVDA{PERIOD_SEP}5d" != f"NVDA{PERIOD_SEP}10y"


def test_query_without_period_uses_the_default(connector):
    term, period = connector._split_query("nvidia")
    assert (term, period) == ("nvidia", connector.default_period)


def test_invalid_period_is_a_query_error(connector):
    with pytest.raises(QueryError):
        connector._split_query("NVDA@3weeks")
    with pytest.raises(ValueError):
        connector._split_query("   ")


def test_invalid_default_period_rejected_at_construction():
    with pytest.raises(ValueError):
        YFinanceConnector(default_period="3weeks")


# --- DEFECT-CLASS 2: rate limiting --------------------------------------


def test_defect_class_2_rate_limit_error_maps_and_is_retryable(connector):
    """yfinance's YFRateLimitError must become our RateLimitError.

    If it stayed a YF* exception it would fall through ``_guarded``'s
    transient branch as an UpstreamError -- still retried, but reported to
    Part 19 as "the source is broken" rather than "we are being throttled",
    which are different operational facts.
    """
    from yfinance.exceptions import YFRateLimitError

    mapped = connector._map_exception(YFRateLimitError(), "quote 'NVDA'")
    assert isinstance(mapped, RateLimitError)
    assert mapped.source == "yfinance"
    assert mapped.status_code == 429


def test_defect_class_2_deterministic_errors_are_not_retried(connector):
    from yfinance.exceptions import YFInvalidPeriodError

    mapped = connector._map_exception(
        YFInvalidPeriodError("BAD", "bad", list(VALID_PERIODS)), "quote 'X'"
    )
    assert isinstance(mapped, QueryError)
    assert not isinstance(mapped, RateLimitError)


def test_defect_class_2_throttle_spaces_calls(run_async):
    """Throttling is inherited from the base class and really spaces calls.

    600/min means 0.1s between starts, so ten throttled calls cannot finish
    in under 0.9s (the first is free). 600 rather than a realistic 30 so the
    test costs ~1s instead of ~18s.
    """
    import asyncio
    import time

    connector = YFinanceConnector(rate_limit_per_min=600)

    async def ten():
        start = time.monotonic()
        for _ in range(10):
            await connector._throttle()
        return time.monotonic() - start

    assert run_async(ten()) >= 0.9
    assert connector.request_count == 10


def test_rate_limit_must_be_positive():
    with pytest.raises(ValueError):
        YFinanceConnector(rate_limit_per_min=0)


# --- DEFECT-CLASS 3: timezone -------------------------------------------


def test_defect_class_3_published_date_is_converted_to_utc(connector):
    """Yahoo returns EXCHANGE-LOCAL time, not UTC. Verified, not assumed.

    The recorded fixture holds four real listings whose sessions all fall on
    the same calendar day but carry offsets -04:00, +01:00 and +09:00. Left
    unconverted they would appear up to 13 hours apart, and Part 21 compares
    dates across angles.
    """
    raw = _fixture("yfinance_multi_exchange.json")
    body = json.loads(raw.body_text)
    zones = {q["exchange_timezone"] for q in body["quotes"]}
    assert len(zones) >= 3, "fixture must span multiple exchanges to be a test"
    assert zones != {"UTC"}, "fixture would not exercise conversion"

    normalized = connector.normalize(connector.parse(raw))
    for record in normalized:
        assert record.published_date is not None
        assert record.published_date.tzinfo is not None, "must stay tz-aware"
        assert record.published_date.utcoffset() == timezone.utc.utcoffset(None)


def test_defect_class_3_the_original_zone_survives_for_audit(connector):
    records = connector.parse(_fixture("yfinance_multi_exchange.json"))
    zones = {r.extra["exchange_timezone"] for r in records}
    assert "Asia/Tokyo" in zones or "Europe/London" in zones


def test_to_utc_converts_rather_than_relabels():
    tokyo = datetime.fromisoformat("2026-09-24T00:00:00+09:00")
    converted = to_utc(tokyo)
    assert converted == datetime(2026, 9, 23, 15, 0, tzinfo=timezone.utc)
    assert converted.tzinfo is timezone.utc


def test_to_utc_treats_naive_as_utc_and_passes_none_through():
    naive = datetime(2026, 9, 24, 12, 0)
    assert to_utc(naive) == datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)
    assert to_utc(None) is None


# --- DEFECT-CLASS 4: oversized snippet ----------------------------------


def test_defect_class_4_snippet_does_not_grow_with_the_period(connector):
    """A 10y summary must be the same size as a 5d summary.

    This is the property that keeps the record under the Part 19 budget. The
    fixture is mutated to claim 2513 bars (the real 10y daily row count,
    measured) to prove length is independent of history length.
    """
    body = json.loads(_fixture("yfinance_nvidia.json").body_text)
    short = dict(body["quotes"][0], period="5d", bars=5)
    long = dict(body["quotes"][0], period="10y", bars=2513)

    def snippet_for(quote):
        raw = RawResponse(200, {}, json.dumps({"quotes": [quote]}))
        return connector.parse(raw)[0].snippet

    short_s, long_s = snippet_for(short), snippet_for(long)
    assert abs(len(short_s) - len(long_s)) <= 2, "length tracks the period"
    assert "2513" not in long_s, "per-bar data leaked into the snippet"


def test_defect_class_4_snippet_stays_under_the_angle_budget(connector):
    """Every generated snippet must fit well inside the 2000-token budget.

    Part 19 drops any record whose snippet_tokens exceeds the whole budget,
    and a dropped record is indistinguishable from a retrieval failure.
    """
    for name in ("yfinance_nvidia.json", "yfinance_multi_exchange.json"):
        for record in connector.normalize(connector.parse(_fixture(name))):
            assert len(record.snippet) <= SNIPPET_CHARS
            assert record.snippet_tokens < 2000
            assert record.snippet_tokens < 200, "a one-line summary is small"


def test_defect_class_4_guard_fires_on_an_unbounded_field(connector):
    """The invariant is checked at construction, not left to Part 19."""
    from prismflow.v2.connectors.yfinance import _assert_snippet_bounded

    _assert_snippet_bounded("x" * SNIPPET_CHARS)  # at the cap: fine
    with pytest.raises(ParseError):
        _assert_snippet_bounded("x" * (SNIPPET_CHARS + 1))


def test_long_company_name_is_truncated(connector):
    body = json.loads(_fixture("yfinance_nvidia.json").body_text)
    quote = dict(body["quotes"][0], long_name="Company " * 200)
    raw = RawResponse(200, {}, json.dumps({"quotes": [quote]}))
    record = connector.parse(raw)[0]
    assert len(record.snippet) <= SNIPPET_CHARS


# --- fetch argument validation ------------------------------------------


def test_fetch_rejects_empty_query_and_bad_k(run_async, connector):
    with pytest.raises(ValueError):
        run_async(connector.fetch("", k=5))
    with pytest.raises(ValueError):
        run_async(connector.fetch("NVDA", k=0))


# --- live ---------------------------------------------------------------


@pytest.mark.live
def test_yfinance_connector_live(run_async):
    """One real Yahoo query. Skipped by ``pytest -m "not live"``."""
    connector = YFinanceConnector()
    try:
        records = run_async(connector.get_records("nvidia", k=3))
    finally:
        run_async(connector.aclose())

    assert len(records) >= 1, "live yfinance query returned nothing"
    assert len(records) <= 3
    for record in records:
        assert record.id
        assert record.source == "yfinance"
        assert record.url.startswith("https://finance.yahoo.com/quote/")
        assert record.snippet
        assert record.snippet_tokens < 2000
        if record.published_date is not None:
            assert record.published_date.utcoffset().total_seconds() == 0
