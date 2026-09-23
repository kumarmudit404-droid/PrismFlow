"""Part 18 tests: the cache must be invisible except in the logs.

Covers the brief's seven cases, and adds one regression test per defect found
in the specified implementation (each is named ``test_regression_*`` and says
in its docstring what would break without it). The regressions matter more than
the happy path here: every one of them is a bug that shows up only on the
*second* run of a program, which is the hardest kind to attribute.

No test sleeps to reach a TTL boundary. Expiry is driven by backdating
``inserted_at`` or by a zero TTL, so the suite is deterministic in the sense
docs/CONTRACT.md section 5 requires and does not lengthen with every case.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import List

import pytest
from click.testing import CliRunner

from prismflow.v2.cache import (
    CacheEntry,
    CacheMetrics,
    cache_connection,
    cached_fetch,
    hash_query,
    init_cache_db,
    lookup_entry,
    records_from_json,
    records_to_json,
    resolve_db_path,
    store_entry,
    to_iso,
    ttl_for,
    wrap_connector,
)
from prismflow.v2.cache.invalidate import cli
from prismflow.v2.cache.models import (
    _RECORD_FIELDS,
    normalized_record_field_names,
)
from prismflow.v2.connectors.base import NormalizedRecord, utcnow

from .conftest import make_arxiv_records, make_github_records


# --- fixtures -----------------------------------------------------------


@pytest.fixture
def db(tmp_path):
    """An initialised cache database, isolated per test.

    Passed explicitly everywhere so no test can reach the developer's real
    ``./cache/v2`` database or be steered by a ``CACHE_DB`` in the environment.
    """
    return str(init_cache_db(tmp_path / "cache.db"))


@pytest.fixture
def metrics():
    """Fresh counters, rather than the process-wide singleton."""
    return CacheMetrics()


class CountingFetch:
    """A fake connector fetch that records how often it really ran."""

    def __init__(self, records: List[NormalizedRecord], delay: float = 0.0):
        self.records = records
        self.delay = delay
        self.calls: List[tuple] = []
        self._lock = threading.Lock()

    async def __call__(self, query: str, k: int) -> List[NormalizedRecord]:
        with self._lock:
            self.calls.append((query, k))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.records[:k]

    @property
    def count(self) -> int:
        return len(self.calls)


def make_cached(db, metrics, records=None, *, ttl=3600, name="github", delay=0.0):
    """A cached fetch over a CountingFetch. Returns (callable, inner)."""
    inner = CountingFetch(records or make_github_records(), delay=delay)

    @cached_fetch(ttl, connector_name=name, db_path=db, metrics=metrics)
    async def fetch(query: str, k: int) -> List[NormalizedRecord]:
        return await inner(query, k)

    return fetch, inner


# --- the brief's seven --------------------------------------------------


def test_cache_hit(db, metrics, run_async, caplog):
    """Second identical query is served from cache and logged as a hit."""
    fetch, inner = make_cached(db, metrics)

    first = run_async(fetch("adversarial examples", 3))
    with caplog.at_level(logging.INFO, logger="prismflow.v2.cache"):
        second = run_async(fetch("adversarial examples", 3))

    assert inner.count == 1, "the second call should not have refetched"
    assert second == first
    assert metrics.hits == 1 and metrics.misses == 1
    assert "[CACHE HIT]" in caplog.text
    assert "github" in caplog.text and hash_query("adversarial examples") in caplog.text


def test_cache_miss(db, metrics, run_async, caplog):
    """A new key fetches, logs a miss, and leaves exactly one row behind."""
    fetch, inner = make_cached(db, metrics)

    with caplog.at_level(logging.INFO, logger="prismflow.v2.cache"):
        records = run_async(fetch("diffusion models", 2))

    assert inner.count == 1
    assert len(records) == 2
    assert "[CACHE MISS]" in caplog.text
    with cache_connection(db) as conn:
        entry = lookup_entry(conn, "github", hash_query("diffusion models"))
    assert entry is not None
    assert entry.query_text == "diffusion models"
    assert entry.requested_k == 2
    assert metrics.misses == 1 and metrics.hits == 0


def test_ttl_expiry(db, metrics, run_async):
    """An entry past its TTL reports expired and is refetched, not served."""
    # Unit level: backdated entry, no sleeping.
    entry = CacheEntry(
        connector_name="github",
        query_hash=hash_query("q"),
        query_text="q",
        normalized_records_json=records_to_json(make_github_records()),
        inserted_at=utcnow() - timedelta(seconds=120),
        ttl_seconds=60,
        requested_k=5,
    )
    assert entry.is_expired() is True
    assert entry.age_seconds >= 120

    fresh = CacheEntry(**{**entry.__dict__, "inserted_at": utcnow()})
    assert fresh.is_expired() is False

    # End to end: a zero TTL is stale by the time it is read back.
    fetch, inner = make_cached(db, metrics, ttl=0)
    run_async(fetch("q", 3))
    run_async(fetch("q", 3))
    assert inner.count == 2, "a zero-TTL entry must never be served"
    assert metrics.expired == 1


def test_cache_invalidate_flush(db, metrics, run_async):
    """flush empties the table."""
    fetch, _ = make_cached(db, metrics)
    for i in range(10):
        run_async(fetch(f"query {i}", 2))
    assert _count(db) == 10

    result = CliRunner().invoke(cli, ["flush", "--db", db, "--yes"])
    assert result.exit_code == 0, result.output
    assert "Flushed 10" in result.output
    assert _count(db) == 0


def test_cache_invalidate_expire(db, metrics, run_async):
    """expire --days removes only entries older than the cutoff."""
    fetch, _ = make_cached(db, metrics)
    run_async(fetch("recent", 2))
    run_async(fetch("ancient", 2))
    _backdate(db, hash_query("ancient"), utcnow() - timedelta(days=30))
    assert _count(db) == 2

    result = CliRunner().invoke(cli, ["expire", "--db", db, "--days", "7"])
    assert result.exit_code == 0, result.output
    assert "Removed 1" in result.output
    assert _count(db) == 1
    with cache_connection(db) as conn:
        assert lookup_entry(conn, "github", hash_query("recent")) is not None
        assert lookup_entry(conn, "github", hash_query("ancient")) is None


def test_concurrent_access_across_threads(db, metrics, run_async):
    """Five threads on one key: no SQLite errors, one row, consistent records.

    Threads each drive their own event loop, so the asyncio single-flight lock
    cannot span them and the fetch count is not asserted to be 1 here -- see
    ``test_concurrent_single_flight`` for that. What must hold is that
    concurrent writers do not raise "database is locked" and do not leave
    duplicate or corrupt rows.
    """
    fetch, inner = make_cached(db, metrics, delay=0.02)

    def worker():
        return asyncio.run(fetch("shared query", 3))

    with ThreadPoolExecutor(max_workers=5) as pool:
        results = [f.result() for f in [pool.submit(worker) for _ in range(5)]]

    assert all(r == results[0] for r in results)
    assert len(results[0]) == 3
    assert _count(db) == 1, "one key must not produce duplicate rows"
    assert 1 <= inner.count <= 5
    assert metrics.hits + metrics.misses == 5


def test_concurrent_single_flight(db, metrics, run_async):
    """Five coroutines on one loop collapse into a single fetch.

    This is the concurrency shape Part 19's fan-out actually has, and the
    reason the decorator re-checks the cache inside the lock. Without the lock
    all five would miss and all five would fetch, defeating the rate limiting
    Part 17 put in the base class.
    """
    fetch, inner = make_cached(db, metrics, delay=0.05)

    async def scenario():
        return await asyncio.gather(*(fetch("fan out", 3) for _ in range(5)))

    results = run_async(scenario())

    assert inner.count == 1, f"expected one fetch, got {inner.count}"
    assert all(r == results[0] for r in results)
    assert metrics.hits == 4 and metrics.misses == 1


def test_hit_rate_metric(db, metrics, run_async):
    """10 unique queries run twice gives a 50% hit rate."""
    fetch, inner = make_cached(db, metrics)
    for _ in range(2):
        for i in range(10):
            run_async(fetch(f"query {i}", 2))

    assert metrics.lookups == 20
    assert metrics.hits == 10 and metrics.misses == 10
    assert metrics.hit_rate == pytest.approx(0.5)
    assert metrics.miss_rate == pytest.approx(0.5)
    assert inner.count == 10

    snapshot = metrics.snapshot()
    assert snapshot["hit_rate"] == pytest.approx(0.5)
    # Latencies are recorded but are machine-dependent: per the 5-SEED RULE a
    # single run's timing is not evidence, so only their presence is asserted.
    assert snapshot["avg_hit_latency_ms"] is not None
    assert snapshot["avg_miss_latency_ms"] is not None


def test_gate_hit_rate_on_repeated_queries(db, metrics, run_async):
    """The Part 18 gate: repeated queries exceed a 95% hit rate."""
    fetch, inner = make_cached(db, metrics)
    for _ in range(20):
        run_async(fetch("the same question", 3))

    assert inner.count == 1
    assert metrics.hit_rate >= 0.95, f"gate requires >=0.95, got {metrics.hit_rate}"


# --- regressions against the specified implementation -------------------


def test_regression_round_trip_preserves_datetime_type(db):
    """Cached records must not turn published_date into a str.

    The brief deserialises with ``NormalizedRecord(**json.loads(...))``, and
    dataclasses do not coerce, so a cache hit returns a str where a fresh fetch
    returns a datetime and any date arithmetic downstream raises TypeError --
    only on warm runs.
    """
    original = make_github_records()
    restored = records_from_json(records_to_json(original))

    assert restored == original
    for record in restored:
        assert isinstance(record.published_date, datetime)
        assert record.published_date.tzinfo is not None
        # The operation that fails if the type is wrong.
        assert (utcnow() - record.published_date).total_seconds() > 0


def test_regression_serialisation_covers_every_field():
    """Every NormalizedRecord field must be serialised.

    Guards against a later part adding a field to Part 17's dataclass that this
    module silently drops, which would make cached records differ from fresh
    ones in a way no other test would notice.
    """
    assert set(_RECORD_FIELDS) == set(normalized_record_field_names())


def test_regression_connectors_do_not_share_a_cache_key(db, metrics, run_async):
    """Two sources, one query text: each must get its own records.

    With the brief's ``connector_name or "unknown"`` both sources key on
    ("unknown", hash(query)) and the upsert makes the second overwrite the
    first, so one angle is served the other's data. Part 21 would read that as
    agreement between angles.
    """
    gh, gh_inner = make_cached(db, metrics, make_github_records(), name="github")
    ax, ax_inner = make_cached(db, metrics, make_arxiv_records(), name="arxiv")

    gh_first = run_async(gh("adversarial examples", 3))
    ax_first = run_async(ax("adversarial examples", 3))
    gh_again = run_async(gh("adversarial examples", 3))
    ax_again = run_async(ax("adversarial examples", 3))

    assert {r.source for r in gh_again} == {"github"}
    assert {r.source for r in ax_again} == {"arxiv"}
    assert gh_again == gh_first and ax_again == ax_first
    assert gh_inner.count == 1 and ax_inner.count == 1
    assert _count(db) == 2


def test_regression_unattributable_fetch_raises(db, run_async):
    """A fetch with no resolvable connector name must refuse, not guess."""

    @cached_fetch(3600, db_path=db)
    async def fetch(query: str, k: int):
        return make_github_records()[:k]

    with pytest.raises(ValueError, match="cannot attribute"):
        run_async(fetch("q", 2))


def test_regression_connector_name_kwarg_is_not_forwarded(db, metrics, run_async):
    """Passing connector_name at the call must work, not TypeError.

    The brief reads the kwarg and then forwards it to a function that does not
    accept it, so the only documented way to set the name raises
    ``TypeError: fetch_github() got an unexpected keyword argument``.
    """
    inner = CountingFetch(make_arxiv_records())

    @cached_fetch(3600, db_path=db, metrics=metrics)
    async def fetch(query: str, k: int):
        return await inner(query, k)

    records = run_async(fetch("q", 2, connector_name="arxiv"))
    assert len(records) == 2
    with cache_connection(db) as conn:
        assert lookup_entry(conn, "arxiv", hash_query("q")) is not None


def test_regression_larger_k_is_not_served_from_a_smaller_entry(
    db, metrics, run_async
):
    """An entry fetched at k=2 must not answer k=5 with two records.

    The brief returns ``to_records()[:k]`` and logs a hit, so Part 19 would
    retrieve less evidence than it asked for and never know.
    """
    fetch, inner = make_cached(db, metrics)

    small = run_async(fetch("q", 2))
    assert len(small) == 2 and inner.count == 1

    large = run_async(fetch("q", 5))
    assert len(large) == 5, "a bigger k must refetch rather than under-serve"
    assert inner.count == 2
    assert metrics.insufficient_k == 1

    # The upgraded row now satisfies both, and neither refetches.
    assert len(run_async(fetch("q", 5))) == 5
    assert len(run_async(fetch("q", 2))) == 2
    assert inner.count == 2
    assert _count(db) == 1, "the entry is upgraded in place, not duplicated"


def test_regression_expire_keeps_entries_from_the_cutoff_day(db, metrics, run_async):
    """expire --days must not delete same-calendar-day entries.

    The brief stores ``DEFAULT CURRENT_TIMESTAMP`` ('2026-09-23 07:58:48') and
    compares against ``cutoff.isoformat()`` ('2026-09-16T07:58:48+00:00'). The
    comparison is lexicographic and ' ' (0x20) < 'T' (0x54), so every row whose
    date equals the cutoff's date is deleted whatever its clock time -- up to a
    day of valid entries per run.
    """
    fetch, _ = make_cached(db, metrics)
    run_async(fetch("just under the line", 2))
    # 6 days 23 hours old: inside a 7-day window, but on the cutoff's date.
    kept = utcnow() - timedelta(days=6, hours=23)
    _backdate(db, hash_query("just under the line"), kept)

    result = CliRunner().invoke(cli, ["expire", "--db", db, "--days", "7"])
    assert result.exit_code == 0, result.output
    assert _count(db) == 1, "an entry inside the window was deleted"

    # And the format really is uniform, which is what makes that true.
    with cache_connection(db) as conn:
        stored = conn.execute("SELECT inserted_at FROM cache_entries").fetchone()[0]
    assert stored == to_iso(kept)
    assert len(stored) == len(to_iso(utcnow())), "timestamps must be fixed width"


def test_regression_no_naive_aware_mixing(db, metrics, run_async):
    """No DeprecationWarning and no naive/aware TypeError on any cache path.

    Standing rule 3 mandates ``datetime.utcnow()``, which on CPython 3.13.7 is
    deprecated and returns a naive datetime, while every Part 17 timestamp is
    aware. ``utcnow() - inserted_at`` then raises ``TypeError: can't subtract
    offset-naive and offset-aware datetimes``.
    """
    fetch, _ = make_cached(db, metrics)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        run_async(fetch("q", 2))
        run_async(fetch("q", 2))

    with cache_connection(db) as conn:
        entry = lookup_entry(conn, "github", hash_query("q"))
    assert entry is not None
    assert entry.inserted_at.tzinfo is not None
    assert entry.is_expired(now=utcnow()) is False


def test_regression_hit_path_closes_its_connection(db, metrics, run_async):
    """Hits must not leak connections; the brief closes only on the miss path."""
    fetch, _ = make_cached(db, metrics)
    run_async(fetch("q", 2))
    for _ in range(50):
        run_async(fetch("q", 2))

    assert metrics.hits == 50
    # A leaked handle on Windows keeps the file locked and this would raise.
    with cache_connection(db) as conn:
        conn.execute("DELETE FROM cache_entries")
        conn.commit()
    assert _count(db) == 0


def test_regression_resolve_db_path_accepts_the_committed_url(monkeypatch, tmp_path):
    """CACHE_DB in .env.template is a SQLAlchemy URL, not a path.

    Handed to ``sqlite3.connect`` verbatim it fails with
    ``OperationalError: unable to open database file`` because a Windows path
    cannot contain a colon.
    """
    monkeypatch.setenv("CACHE_DB", "sqlite:///./cache/v2/prismflow_cache.db")
    from_url = resolve_db_path()
    assert from_url == resolve_db_path("./cache/v2/prismflow_cache.db")
    assert ":" not in str(from_url), "a colon here is an unopenable path"
    # The unstripped form is what fails, which is why stripping is needed.
    with pytest.raises(sqlite3.OperationalError):
        sqlite3.connect("sqlite:///./cache/v2/prismflow_cache.db")

    monkeypatch.setenv("CACHE_DB", str(tmp_path / "from_env.db"))
    assert resolve_db_path() == tmp_path / "from_env.db"

    monkeypatch.delenv("CACHE_DB")
    monkeypatch.setenv("CACHE_DIR", str(tmp_path / "cachedir"))
    assert resolve_db_path().parent == tmp_path / "cachedir"

    # An explicit argument always wins.
    assert resolve_db_path(tmp_path / "explicit.db") == tmp_path / "explicit.db"


# --- storage, schema and CLI -------------------------------------------


def test_init_cache_db_is_idempotent(tmp_path):
    """Running init twice must not fail, and must preserve existing rows."""
    path = tmp_path / "nested" / "cache.db"
    first = init_cache_db(path)
    with cache_connection(first) as conn:
        store_entry(conn, _entry("github", "q"))
    second = init_cache_db(path)
    assert first == second
    assert _count(str(path)) == 1

    with cache_connection(path) as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(cache_entries)")}
        indexes = {
            r[1] for r in conn.execute("PRAGMA index_list(cache_entries)")
        }
    assert {"connector_name", "query_hash", "requested_k", "inserted_at"} <= columns
    assert {"idx_connector_hash", "idx_inserted_at"} <= indexes


def test_init_cache_db_migrates_an_older_table(tmp_path):
    """A database predating requested_k gains the column rather than breaking."""
    path = tmp_path / "old.db"
    conn = sqlite3.connect(str(path))
    conn.execute(
        """CREATE TABLE cache_entries (
               id INTEGER PRIMARY KEY,
               connector_name TEXT NOT NULL,
               query_hash TEXT NOT NULL,
               query_text TEXT NOT NULL,
               normalized_records_json TEXT NOT NULL,
               inserted_at TEXT NOT NULL,
               ttl_seconds INTEGER NOT NULL,
               UNIQUE(connector_name, query_hash))"""
    )
    conn.commit()
    conn.close()

    init_cache_db(path)
    with cache_connection(path) as conn:
        columns = {r[1] for r in conn.execute("PRAGMA table_info(cache_entries)")}
    assert "requested_k" in columns


def test_hash_query_is_stable_and_distinguishing():
    assert hash_query("a") == hash_query("a")
    assert hash_query("a") != hash_query("b")
    assert len(hash_query("a")) == 16


def test_collision_is_detected_not_served(db, metrics, run_async):
    """A stored query_text that disagrees with the caller means a hash collision."""
    fetch, inner = make_cached(db, metrics)
    run_async(fetch("real query", 2))
    # Forge the collision: same hash row, different stored text.
    with cache_connection(db) as conn:
        conn.execute(
            "UPDATE cache_entries SET query_text = ?", ("a different query",)
        )
        conn.commit()

    run_async(fetch("real query", 2))
    assert inner.count == 2, "a colliding entry must be refetched"
    assert metrics.collisions == 1


def test_ttl_for_uses_per_connector_defaults():
    assert ttl_for("arxiv") == 86400
    assert ttl_for("github") == 3600
    assert ttl_for("newsapi") == 900
    assert ttl_for("something-new") == 3600


def test_cli_stats_on_an_empty_cache(tmp_path):
    """The closing checklist: stats works and shows 0 entries."""
    path = str(tmp_path / "fresh.db")
    result = CliRunner().invoke(cli, ["stats", "--db", path])
    assert result.exit_code == 0, result.output
    assert "Total entries: 0" in result.output


def test_cli_stats_reports_live_and_expired(db, metrics, run_async):
    fetch, _ = make_cached(db, metrics)
    run_async(fetch("fresh one", 2))
    run_async(fetch("stale one", 2))
    _backdate(db, hash_query("stale one"), utcnow() - timedelta(days=5))

    result = CliRunner().invoke(cli, ["stats", "--db", db])
    assert result.exit_code == 0, result.output
    assert "Total entries: 2" in result.output
    assert "1 live, 1 expired" in result.output


def test_cli_clear_connector_leaves_other_connectors_alone(db):
    with cache_connection(db) as conn:
        store_entry(conn, _entry("github", "q1"))
        store_entry(conn, _entry("github", "q2"))
        store_entry(conn, _entry("arxiv", "q1"))

    result = CliRunner().invoke(
        cli, ["clear-connector", "--db", db, "--connector", "github"]
    )
    assert result.exit_code == 0, result.output
    assert "Cleared 2" in result.output
    assert _count(db) == 1

    # The brief spells the command with an underscore; both must resolve.
    assert CliRunner().invoke(
        cli, ["clear_connector", "--db", db, "--connector", "arxiv"]
    ).exit_code == 0
    assert _count(db) == 0


def test_cli_flush_aborts_without_confirmation(db):
    with cache_connection(db) as conn:
        store_entry(conn, _entry("github", "q1"))

    result = CliRunner().invoke(cli, ["flush", "--db", db], input="n\n")
    assert result.exit_code != 0
    assert _count(db) == 1, "an aborted flush must not delete anything"


def test_cli_expired_respects_each_entrys_own_ttl(db):
    """Same age, different TTL, different fate -- which is the whole point.

    Both entries are 12 hours old. That is inside arXiv's 24h TTL and far
    outside NewsAPI's 15 minutes, so a blanket ``expire --days`` could not tell
    them apart but ``expired`` can.
    """
    with cache_connection(db) as conn:
        store_entry(conn, _entry("arxiv", "long", ttl=86400, age_days=0.5))
        store_entry(conn, _entry("newsapi", "short", ttl=900, age_days=0.5))

    result = CliRunner().invoke(cli, ["expired", "--db", db, "--yes"])
    assert result.exit_code == 0, result.output
    assert "Removed 1" in result.output
    with cache_connection(db) as conn:
        assert lookup_entry(conn, "arxiv", hash_query("long")) is not None
        assert lookup_entry(conn, "newsapi", hash_query("short")) is None


def test_cached_fetch_rejects_a_sync_function(db):
    with pytest.raises(TypeError, match="async"):

        @cached_fetch(3600, connector_name="github", db_path=db)
        def not_async(query: str, k: int):
            return []


@pytest.mark.parametrize("bad_k", [0, -1, 2.5, True])
def test_cached_fetch_validates_k(db, metrics, run_async, bad_k):
    fetch, _ = make_cached(db, metrics)
    with pytest.raises((ValueError, TypeError)):
        run_async(fetch("q", bad_k))


def test_cached_fetch_validates_query(db, metrics, run_async):
    fetch, _ = make_cached(db, metrics)
    with pytest.raises(TypeError, match="query must be a str"):
        run_async(fetch(None, 2))


# --- wrapping a real Part 17 connector ---------------------------------


def test_wrap_connector_caches_a_part17_connector(
    db, metrics, run_async, mock_github_connector
):
    """The closing checklist: Part 17 connectors gain caching without edits.

    ``wrap_connector`` takes the name from ``connector.name``, so nothing under
    ``prismflow/v2/connectors`` is touched -- Part 17 stays sealed.
    """
    calls = {"n": 0}
    original = mock_github_connector.get_records

    async def counting(query, k):
        calls["n"] += 1
        return await original(query, k)

    mock_github_connector.get_records = counting
    fetch = wrap_connector(mock_github_connector, db_path=db, metrics=metrics)

    first = run_async(fetch("adversarial examples", 3))
    second = run_async(fetch("adversarial examples", 3))

    assert calls["n"] == 1
    assert second == first
    assert {r.source for r in second} == {"github"}
    assert all(isinstance(r.published_date, datetime) for r in second)
    with cache_connection(db) as conn:
        assert lookup_entry(conn, "github", hash_query("adversarial examples"))


def test_wrap_connector_requires_a_name():
    class Nameless:
        name = ""

    with pytest.raises(ValueError, match="connector.name"):
        wrap_connector(Nameless())


def test_wrap_connector_uses_the_per_connector_ttl(db, metrics, run_async,
                                                   mock_github_connector):
    fetch = wrap_connector(mock_github_connector, db_path=db, metrics=metrics)
    run_async(fetch("q", 2))
    with cache_connection(db) as conn:
        entry = lookup_entry(conn, "github", hash_query("q"))
    assert entry is not None
    assert entry.ttl_seconds == ttl_for("github")


# --- helpers ------------------------------------------------------------


def _entry(connector: str, query: str, *, ttl: int = 3600, age_days: float = 0.0):
    return CacheEntry(
        connector_name=connector,
        query_hash=hash_query(query),
        query_text=query,
        normalized_records_json=records_to_json(make_github_records()),
        inserted_at=utcnow() - timedelta(days=age_days),
        ttl_seconds=ttl,
        requested_k=5,
    )


def _count(db: str) -> int:
    with cache_connection(db) as conn:
        return conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]


def _backdate(db: str, query_hash: str, moment: datetime) -> None:
    with cache_connection(db) as conn:
        conn.execute(
            "UPDATE cache_entries SET inserted_at = ? WHERE query_hash = ?",
            (to_iso(moment), query_hash),
        )
        conn.commit()
