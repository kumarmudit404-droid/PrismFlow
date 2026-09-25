"""NewsAPI connector tests (Market angle).

Offline tests run against RECORDED response bodies in ``tests/v2/fixtures/``,
captured live on 2026-09-25 from ``/v2/everything``. The suite makes no network
call except the one test marked ``live``, which ``pytest -m "not live"`` skips.

FIXTURE PROVENANCE, stated because two of the six are not recorded
-----------------------------------------------------------------
Real, captured live:
    newsapi_search.json        3 articles selected out of 98 for the
                               properties they exhibit -- one carries a U+FFFD
                               that NewsAPI itself served, all three carry the
                               ``[+N chars]`` truncation marker, all three have
                               a null ``source.id``.
    newsapi_widest_real.json   the widest of those 98 by description+content
                               (260 + 215 = 475 chars) -- the MEASURED worst
                               case, which does not breach the budget.
    newsapi_error_400.json     a real ``parametersMissing`` body.
    newsapi_error_401.json     a real ``apiKeyInvalid`` body.

Synthetic, and labelled as such in the fixture itself:
    newsapi_untruncated.json   real article shape with the field LENGTHS
                               inflated. The free tier caps ``content`` near
                               200 chars, so the oversize guard cannot be
                               exercised against real free-tier data at all --
                               and a guard that is only ever tested on bodies
                               too small to trip it is untested. Nothing in
                               this fixture is presented as a measurement.
    newsapi_error_429_exhausted.json
                               the ``apiKeyExhausted`` envelope. Producing it
                               live costs all 100 of the day's requests. Its
                               shape is copied from the REAL 401 body and its
                               ``code`` string is the documented one.

Four tests are labelled DEFECT-CLASS 1..4 and pin the checks the brief named.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from prismflow.v2.cache.decorator import CONNECTOR_TTL_SECONDS, ttl_for
from prismflow.v2.cache.models import hash_query
from prismflow.v2.connectors import (
    AuthError,
    ParseError,
    QueryError,
    RateLimitError,
    UpstreamError,
)
from prismflow.v2.connectors.base import RawResponse
from prismflow.v2.connectors.newsapi import (
    ARCHIVE_LIMIT_HOURS,
    ARTICLE_DELAY_HOURS,
    DAILY_REQUEST_CAP,
    LANG_SEP,
    REQUEST_WARN_THRESHOLD,
    SNIPPET_CHARS,
    SORT_BY,
    VALID_LANGUAGES,
    NewsAPIConnector,
    parse_published_at,
)

FIXTURES = Path(__file__).parent / "fixtures"

#: Not placeholder-shaped, so ``_resolve_key`` accepts it and never falls back
#: to the real key in ``.env``. See ``test_explicit_placeholder_key_does_not_fall_back``.
FAKE_KEY = "0123456789abcdef0123456789abcdef"


def _fixture(name: str, status: int = 200) -> RawResponse:
    return RawResponse(
        status_code=status,
        headers={"content-type": "application/json; charset=utf-8"},
        body_text=(FIXTURES / name).read_text(encoding="utf-8"),
    )


def _one_article(published_at: str = "2026-09-24T03:34:00Z") -> RawResponse:
    """A minimal single-article body, for date and staleness cases."""
    return RawResponse(
        200,
        {},
        json.dumps(
            {
                "status": "ok",
                "articles": [
                    {
                        "source": {"name": "Wire"},
                        "title": "t",
                        "description": "d",
                        "content": "c",
                        "url": "https://example.com/one",
                        "publishedAt": published_at,
                    }
                ],
            }
        ),
    )


@pytest.fixture
def connector():
    return NewsAPIConnector(api_key=FAKE_KEY)


# --- 1. parse -----------------------------------------------------------


def test_parse_recorded_search_yields_records(connector):
    records = connector.parse(_fixture("newsapi_search.json"))

    assert len(records) == 3
    first = records[0]
    assert first.source_name == "newsapi"
    assert first.url.startswith("http")
    assert first.id == first.url, "NewsAPI gives no id field; the url is identity"
    assert first.title
    assert first.snippet
    assert first.extra["outlet"]


def test_parse_rejects_malformed_bodies(connector):
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, "not json"))
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, json.dumps({"nope": []})))
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, json.dumps({"articles": [{}]})))


def test_parse_error_is_a_valueerror(connector):
    """Part 17 contract: parse raises ValueError on malformed input."""
    with pytest.raises(ValueError):
        connector.parse(RawResponse(200, {}, "{"))


def test_parse_rejects_a_200_carrying_an_error_envelope(connector):
    """A 200 with status=error never reaches _check_status, so parse catches it."""
    body = json.dumps(
        {"status": "error", "code": "rateLimited", "message": "back off",
         "articles": []}
    )
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, body))


def test_parse_requires_a_url_because_that_is_the_identity(connector):
    body = json.dumps({"status": "ok", "articles": [{"title": "no url here"}]})
    with pytest.raises(ParseError):
        connector.parse(RawResponse(200, {}, body))


def test_normalize_fills_the_part17_contract(connector):
    normalized = connector.normalize(
        connector.parse(_fixture("newsapi_search.json"))
    )

    assert len(normalized) == 3
    for record in normalized:
        assert record.source == "newsapi"
        assert record.snippet
        assert record.snippet_tokens > 0
        assert record.author, "author falls back to the outlet"
        assert record.published_date is not None
        assert record.relevance_score == 1.0


def test_author_falls_back_to_the_outlet(connector):
    body = json.dumps(
        {
            "status": "ok",
            "articles": [
                {
                    "source": {"id": None, "name": "Reuters"},
                    "author": None,
                    "title": "Some headline",
                    "description": "A summary.",
                    "content": "Body text.",
                    "url": "https://example.com/a",
                    "publishedAt": "2026-09-24T03:34:00Z",
                }
            ],
        }
    )
    record = connector.parse(RawResponse(200, {}, body))[0]
    assert record.extra["author"] == "Reuters"


def test_outlet_is_prepended_to_the_snippet(connector):
    records = connector.parse(_fixture("newsapi_search.json"))
    for record in records:
        assert record.snippet.startswith(f"[{record.extra['outlet']}]")


def test_snippet_falls_back_to_the_title_when_there_is_no_body(connector):
    """An empty snippet ranks last under BM25 for a non-relevance reason."""
    body = json.dumps(
        {
            "status": "ok",
            "articles": [
                {
                    "source": {"name": "Wire"},
                    "title": "Only a headline",
                    "description": "",
                    "content": "",
                    "url": "https://example.com/b",
                    "publishedAt": "2026-09-24T03:34:00Z",
                }
            ],
        }
    )
    record = connector.parse(RawResponse(200, {}, body))[0]
    assert "Only a headline" in record.snippet


# --- DEFECT-CLASS 1: Part 18 cache-key distinctness ---------------------


def test_defect_class_1_name_is_distinct_from_the_other_connectors():
    name = NewsAPIConnector(api_key=FAKE_KEY).name
    assert name == "newsapi"
    assert name not in ("github", "arxiv", "yfinance", "stackexchange")


def test_defect_class_1_name_matches_the_part18_ttl_table():
    """The name is not free: Part 18 keys its TTL off this exact string.

    A connector named "news" or "newsapi_org" would silently fall back to the
    3600s default instead of the 900s the table intends for this source.
    """
    assert "newsapi" in CONNECTOR_TTL_SECONDS
    assert ttl_for(NewsAPIConnector(api_key=FAKE_KEY).name) == 900


def test_defect_class_1_language_is_inside_the_cache_key(connector):
    """Same terms, different language => different key.

    The Part 18 key is (connector_name, sha256(query)[:16]) with no language
    component, so a language passed as a constructor argument would be
    invisible to it: "inflation" in English and in German would hash
    identically and the second caller would be served the first's answer for
    the full 900s TTL.
    """
    en, de = "inflation@en", "inflation@de"
    assert hash_query(en) != hash_query(de)
    assert connector._split_query(en) == ("inflation", "en")
    assert connector._split_query(de) == ("inflation", "de")


def test_query_without_a_language_uses_the_default(connector):
    assert connector._split_query("inflation") == ("inflation", "en")


def test_multiword_query_with_language_splits_on_the_last_separator(connector):
    terms, language = connector._split_query("central bank policy@fr")
    assert (terms, language) == ("central bank policy", "fr")


def test_empty_terms_rejected(connector):
    """A leading separator is not a search for the literal text "@de"."""
    with pytest.raises(ValueError):
        connector._split_query(f"{LANG_SEP}de")
    with pytest.raises(ValueError):
        connector._split_query("   ")


def test_defect_class_1_sortby_is_pinned_not_configurable():
    """sortBy changes the result set, so a per-call sort would collide.

    It is not exposed as a constructor argument at all: the constructor would
    accept it, the cache key would not see it, and two different sorts would
    share one entry.
    """
    assert SORT_BY == "relevancy"
    with pytest.raises(TypeError):
        NewsAPIConnector(api_key=FAKE_KEY, sortBy="publishedAt")  # type: ignore[call-arg]


# --- DEFECT-CLASS 2: rate limiting --------------------------------------


def test_defect_class_2_exhausted_key_is_not_a_retryable_rate_limit(connector):
    """apiKeyExhausted must be AuthError, not RateLimitError.

    This is the override's whole reason for existing. NewsAPI sends HTTP 429
    for BOTH "back off for a while" and "your 100 daily requests are gone".
    The base class maps 429 to RateLimitError, which _get retries with
    exponential backoff -- but a spent daily quota does not return in 2, 4 or
    8 seconds, it returns tomorrow. AuthError is not retried.
    """
    raw = _fixture("newsapi_error_429_exhausted.json", status=429)
    with pytest.raises(AuthError) as caught:
        connector._check_status(raw)
    assert caught.value.source == "newsapi"
    assert caught.value.status_code == 429
    assert str(DAILY_REQUEST_CAP) in str(caught.value)


def test_defect_class_2_plain_rate_limiting_stays_retryable(connector):
    """rateLimited, by contrast, is exactly what backoff is for."""
    body = json.dumps(
        {"status": "error", "code": "rateLimited",
         "message": "You have been rate limited. Back off for a while"}
    )
    with pytest.raises(RateLimitError):
        connector._check_status(RawResponse(429, {}, body))


def test_defect_class_2_invalid_key_is_an_auth_error(connector):
    with pytest.raises(AuthError):
        connector._check_status(_fixture("newsapi_error_401.json", status=401))


def test_defect_class_2_parameters_missing_is_a_query_error(connector):
    with pytest.raises(QueryError):
        connector._check_status(_fixture("newsapi_error_400.json", status=400))


def test_defect_class_2_unknown_code_is_an_upstream_error(connector):
    body = json.dumps({"status": "error", "code": "unexpectedError",
                       "message": "our fault"})
    with pytest.raises(UpstreamError):
        connector._check_status(RawResponse(500, {}, body))


def test_defect_class_2_falls_back_to_the_base_mapping_without_an_envelope(
    connector,
):
    """A 503 with an HTML body still has to classify."""
    with pytest.raises(UpstreamError):
        connector._check_status(RawResponse(503, {}, "<html>gateway</html>"))


def test_defect_class_2_invalid_language_is_rejected_before_the_request(
    connector,
):
    """MEASURED: language=zz returns HTTP 200, status=ok, totalResults=0.

    The API does not reject an unknown language -- it returns an empty result
    set that is indistinguishable from "there is no coverage of this topic".
    The angle would report zero evidence and nothing would say why, so the
    check has to happen here.
    """
    with pytest.raises(QueryError) as caught:
        connector._split_query("inflation@zz")
    assert "zz" in str(caught.value)


def test_valid_languages_is_the_documented_set():
    assert "en" in VALID_LANGUAGES and "de" in VALID_LANGUAGES
    assert "zz" not in VALID_LANGUAGES
    assert len(VALID_LANGUAGES) == 14


def test_default_language_is_validated_at_construction():
    with pytest.raises(ValueError):
        NewsAPIConnector(api_key=FAKE_KEY, default_language="zz")


def test_request_warn_threshold_sits_under_the_daily_cap():
    assert REQUEST_WARN_THRESHOLD < DAILY_REQUEST_CAP == 100


def test_rate_limit_is_slower_than_stackexchanges():
    """A third of the daily budget and no quota field to read."""
    from prismflow.v2.connectors.stackexchange import (
        DEFAULT_RATE_PER_MIN as SE_RATE,
    )
    from prismflow.v2.connectors.newsapi import DEFAULT_RATE_PER_MIN as NEWS_RATE

    assert NEWS_RATE < SE_RATE


# --- DEFECT-CLASS 3: timezone and the staleness window ------------------


def test_defect_class_3_published_at_becomes_aware_utc(connector):
    """publishedAt is ISO-8601 with a literal Z (measured)."""
    body = json.loads(_fixture("newsapi_search.json").body_text)
    raw_values = [a["publishedAt"] for a in body["articles"]]
    assert all(v.endswith("Z") for v in raw_values), "fixture must be Z-suffixed"

    for record in connector.normalize(
        connector.parse(_fixture("newsapi_search.json"))
    ):
        assert record.published_date.tzinfo is not None
        assert record.published_date.utcoffset().total_seconds() == 0


def test_parse_published_at_handles_the_z_form_on_every_python():
    """fromisoformat only learned to accept "Z" in 3.11.

    On 3.10 the documented NewsAPI format raises, so the Z is translated
    rather than passed through.
    """
    assert parse_published_at("2026-09-24T03:34:00Z") == datetime(
        2026, 9, 24, 3, 34, tzinfo=timezone.utc
    )
    assert parse_published_at("2026-09-24T03:34:00z") == datetime(
        2026, 9, 24, 3, 34, tzinfo=timezone.utc
    )


def test_parse_published_at_stamps_utc_on_a_naive_value():
    """Naive would leak into Part 21's cross-angle date comparisons."""
    parsed = parse_published_at("2026-09-24T03:34:00")
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


def test_parse_published_at_rejects_junk():
    assert parse_published_at(None) is None
    assert parse_published_at("") is None
    assert parse_published_at("last tuesday") is None
    assert parse_published_at(1642619486) is None


def test_defect_class_3_staleness_of_the_freshest_record_is_measured(connector):
    """The plan's 24h delay is reported, not corrected.

    Every Market record is a day behind every Tech/Financial/Sentiment record
    from the same run, and Part 21 compares dates across angles.
    """
    now = datetime.now(timezone.utc)
    fresh = (now - timedelta(hours=26)).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = json.dumps(
        {
            "status": "ok",
            "articles": [
                {
                    "source": {"name": "Wire"},
                    "title": "t",
                    "description": "d",
                    "content": "c",
                    "url": "https://example.com/c",
                    "publishedAt": fresh,
                }
            ],
        }
    )
    connector.parse(RawResponse(200, {}, body))
    assert connector.staleness_hours is not None
    assert 25.0 < connector.staleness_hours < 27.0
    assert connector.staleness_hours > ARTICLE_DELAY_HOURS


def test_defect_class_3_a_normal_relevancy_page_age_does_not_warn(
    connector, caplog
):
    """A 165h-old top result is normal under relevancy sort, not an anomaly.

    The first draft warned whenever the page was older than the 24h delay,
    which fired on every one of four live smoke queries (38.8h, 69.1h, 165.1h,
    351.5h) and so carried no information. The delay is a floor on age, not a
    prediction of it.
    """
    old = (datetime.now(timezone.utc) - timedelta(hours=165)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    with caplog.at_level("WARNING"):
        connector.parse(_one_article(published_at=old))
    assert 164.0 < connector.staleness_hours < 166.0
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_defect_class_3_past_the_archive_ceiling_does_warn(connector, caplog):
    """The plan serves articles up to a month old; older means it changed."""
    ancient = (datetime.now(timezone.utc) - timedelta(days=90)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    with caplog.at_level("WARNING"):
        connector.parse(_one_article(published_at=ancient))
    assert connector.staleness_hours > ARCHIVE_LIMIT_HOURS
    assert any("archive ceiling" in r.message for r in caplog.records)


def test_staleness_is_none_when_no_record_carries_a_date(connector):
    body = json.dumps(
        {
            "status": "ok",
            "articles": [
                {
                    "source": {"name": "Wire"},
                    "title": "t",
                    "description": "d",
                    "content": "c",
                    "url": "https://example.com/d",
                    "publishedAt": None,
                }
            ],
        }
    )
    connector.parse(RawResponse(200, {}, body))
    assert connector.staleness_hours is None


# --- DEFECT-CLASS 4: oversize, truncation marker, U+FFFD ----------------


def test_defect_class_4_widest_real_body_is_within_budget(connector):
    """The MEASURED worst case does not breach the budget, and that is the
    finding -- NewsAPI truncates before we do."""
    records = connector.parse(_fixture("newsapi_widest_real.json"))
    assert len(records) == 1
    snippet = records[0].snippet
    assert len(snippet) <= SNIPPET_CHARS
    normalized = connector.normalize(records)[0]
    assert normalized.snippet_tokens < 1500, "the Market angle's whole budget"


def test_defect_class_4_guard_fires_on_an_untruncated_body(connector):
    """The ~200-char content cap is observed behaviour, not a contract.

    This fixture's LENGTHS are synthetic (see the module docstring): the free
    tier cannot produce a body big enough to trip the guard, and a guard only
    ever tested on bodies too small to trip it is untested.
    """
    raw = _fixture("newsapi_untruncated.json")
    body = json.loads(raw.body_text)
    article = body["articles"][0]
    assert len(article["description"]) + len(article["content"]) > 4 * SNIPPET_CHARS

    records = connector.parse(raw)
    assert len(records[0].snippet) <= SNIPPET_CHARS
    assert connector.normalize(records)[0].snippet_tokens < 1500


def test_defect_class_4_truncation_marker_is_stripped_from_the_snippet(
    connector,
):
    """"[+3123 chars]" is not article text -- BM25 would score on "chars"."""
    records = connector.parse(_fixture("newsapi_search.json"))
    for record in records:
        assert "[+" not in record.snippet
        assert "chars]" not in record.snippet


def test_defect_class_4_truncated_char_count_is_kept_in_extra(connector):
    """The count is a fact about the article, so it rides in extra."""
    records = connector.parse(_fixture("newsapi_search.json"))
    counts = [r.extra["content_truncated_chars"] for r in records]
    assert all(isinstance(c, int) and c > 0 for c in counts), (
        "all three recorded articles carry the marker"
    )


def test_truncation_marker_absent_leaves_content_alone(connector):
    from prismflow.v2.connectors.newsapi import _strip_truncation_marker

    assert _strip_truncation_marker("plain body") == ("plain body", None)
    assert _strip_truncation_marker("body [+12 chars]") == ("body", 12)
    assert _strip_truncation_marker("body [+1 char]") == ("body", 1)


def test_defect_class_4_wire_borne_replacement_char_is_stripped(connector):
    """U+FFFD arrives from NewsAPI's own ingestion, verified not a decode bug.

    The recorded fixture contains one. A replacement character is noise in a
    BM25 index and noise in a Part 20 prompt.
    """
    raw_text = _fixture("newsapi_search.json").body_text
    assert "�" in raw_text, "fixture must carry the real U+FFFD"

    for record in connector.parse(_fixture("newsapi_search.json")):
        assert "�" not in record.snippet
        assert "�" not in record.title


def test_snippet_bound_is_asserted_not_assumed(connector):
    from prismflow.v2.connectors.newsapi import _assert_snippet_bounded

    _assert_snippet_bounded("x" * SNIPPET_CHARS)
    with pytest.raises(ParseError):
        _assert_snippet_bounded("x" * (SNIPPET_CHARS + 1))


# --- key resolution -----------------------------------------------------


def test_explicit_placeholder_key_does_not_fall_back(monkeypatch):
    """An explicit argument is an instruction, not a hint.

    Falling back would have handed a test that asked for a fake key the real
    one from .env, making a "offline" test call the network.
    """
    monkeypatch.setenv("NEWSAPI_KEY", "a-real-looking-key-from-the-environment")
    assert NewsAPIConnector(api_key="...").api_key is None
    assert NewsAPIConnector(api_key="your_key_here").api_key is None


def test_environment_is_used_when_no_key_is_passed(monkeypatch):
    monkeypatch.setenv("NEWSAPI_KEY", "abcdef0123456789abcdef0123456789")
    assert NewsAPIConnector().api_key == "abcdef0123456789abcdef0123456789"


def test_fetch_without_a_key_is_an_auth_error_not_a_crash(run_async):
    connector = NewsAPIConnector(api_key="...")
    with pytest.raises(AuthError):
        run_async(connector.fetch("inflation", k=3))


# --- live ---------------------------------------------------------------


@pytest.mark.live
def test_newsapi_connector_live(run_async):
    """One real API call. Skipped by ``pytest -m "not live"``.

    Costs 1 of the 100 free-tier requests available per day.
    """
    connector = NewsAPIConnector()
    if not connector.api_key:
        pytest.skip("NEWSAPI_KEY is not set")
    try:
        records = run_async(connector.get_records("inflation", k=3))
    finally:
        run_async(connector.aclose())

    assert len(records) >= 1, "live NewsAPI query returned nothing"
    assert len(records) <= 3
    for record in records:
        assert record.id
        assert record.source == "newsapi"
        assert record.snippet
        assert "[+" not in record.snippet
        assert "�" not in record.snippet
        assert record.snippet_tokens < 1500
        assert record.published_date.utcoffset().total_seconds() == 0
    assert connector.total_results is not None
    assert connector.staleness_hours is not None
    # The documented 24h delay, confirmed against live data rather than assumed.
    assert connector.staleness_hours >= ARTICLE_DELAY_HOURS - 1.0


# --- 5. credential hygiene ---------------------------------------------


def test_the_key_never_appears_in_the_request_url(run_async):
    """The base class logs every url at INFO and every backoff at WARNING.

    A key in the query string is therefore a key in the log file -- which is
    exactly what happened during the smoke run, three times, when DNS
    flickered. The key travels in a header instead.
    """
    from prismflow.v2.connectors.newsapi import API_KEY_HEADER

    seen = {}

    class _Recorder:
        async def get(self, url, headers=None):
            seen["url"] = url
            seen["headers"] = dict(headers or {})
            return RawResponse(
                200,
                {"content-type": "application/json"},
                json.dumps({"status": "ok", "totalResults": 0, "articles": []}),
            )

    connector = NewsAPIConnector(api_key=FAKE_KEY, http_client=_Recorder())
    run_async(connector.fetch("inflation", k=3))

    assert FAKE_KEY not in seen["url"], "the key must not be in the url"
    assert "apiKey" not in seen["url"]
    assert seen["headers"][API_KEY_HEADER] == FAKE_KEY
    # The query itself still has to be there.
    assert "q=inflation" in seen["url"]
    assert "language=en" in seen["url"]
    assert f"sortBy={SORT_BY}" in seen["url"]
