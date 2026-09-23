"""``@cached_fetch``: put a persistent cache in front of a connector fetch.

An ENGINEERING component per docs/CONTRACT.md section 3. It wraps an async
``(query, k) -> List[NormalizedRecord]`` callable and, on a fresh hit, returns
records without touching the network.

FOUR CORRECTIONS TO THE SPECIFIED DECORATOR
-------------------------------------------
The Part 18 brief's decorator was measured against this interpreter before
being reimplemented. Four of its behaviours are load-bearing and wrong:

1. THE CONNECTOR NAME. The brief reads it with
   ``kwargs.get("connector_name") or "unknown"`` and then forwards ``**kwargs``
   to the wrapped function. Both paths fail:

       call without connector_name -> cached under "unknown"
       call with connector_name    -> TypeError: fetch_github() got an
                                      unexpected keyword argument
                                      'connector_name'

   So in practice every source shares the key ``("unknown", hash(query))``,
   and because the table is ``UNIQUE(connector_name, query_hash)`` with an
   ``INSERT OR REPLACE``, an arXiv fetch overwrites a GitHub one and is then
   served *as* GitHub for the same query text. Part 21 measures dependence
   between angles; two angles silently reading one source would register as
   near-perfect agreement -- a fabricated result produced by the cache rather
   than by the world. Here the name is resolved explicitly (call argument,
   then decorator argument, then the bound connector's ``.name``) and a fetch
   that cannot be attributed raises instead of inventing "unknown".

2. THE RECORD COUNT. See ``CacheEntry.requested_k`` -- an entry fetched at
   k=5 must not answer a k=20 lookup as a hit.

3. THE CONNECTION. The brief closes the connection only on the miss path, so
   every hit leaks one. See ``db.cache_connection``.

4. NAIVE TIMESTAMPS. See ``models`` -- ``datetime.utcnow()`` is deprecated on
   CPython 3.13.7 and is naive, while Part 17 is aware.

SINGLE-FLIGHT, AND ITS HONEST LIMIT
-----------------------------------
The brief's test 6 expects concurrent callers of one key to produce a *single*
fetch, but nothing in the specified decorator coordinates them: n concurrent
callers all miss, all fetch, and all write. For a rate-limited source that is
the exact failure Part 17's throttling exists to prevent, so a keyed lock is
added and the cache is re-checked inside it (the second check is what collapses
the duplicates).

The lock is an ``asyncio.Lock`` per (event loop, cache key). That makes
single-flight exact for the case Part 19 actually has -- many coroutines, one
loop -- and it does NOT coordinate across threads or processes, because an
asyncio primitive cannot. Cross-thread callers still behave correctly (SQLite's
busy_timeout and the upsert make concurrent writes safe, and the last writer
wins with identical data); they simply may each fetch once. This is stated
rather than papered over, and both cases are tested separately.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
import sqlite3
import threading
import time
import weakref
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from prismflow.v2.connectors.base import NormalizedRecord, utcnow

from .db import cache_connection
from .metrics import METRICS, CacheMetrics
from .models import CacheEntry, from_iso, hash_query, records_to_json, to_iso

logger = logging.getLogger("prismflow.v2.cache")

#: One hour, the brief's default.
DEFAULT_TTL_SECONDS = 3600

#: Per-connector TTLs, chosen from how fast each source's answer actually
#: changes. The brief suggests "3600 for tech, 86400 for market", which reads
#: inverted: an arXiv paper is immutable once published, while market and news
#: data is the fastest-moving thing V2 touches and a day-old answer would be
#: presented as current. Anything absent here falls back to
#: ``DEFAULT_TTL_SECONDS``.
CONNECTOR_TTL_SECONDS: Dict[str, int] = {
    "arxiv": 86400,     # papers do not change after publication
    "github": 3600,     # stars, forks and activity move through the day
    "newsapi": 900,     # headlines turn over within the hour
    "reddit": 900,      # so does sentiment
    "yfinance": 300,    # quotes are stale almost immediately
}


def ttl_for(connector_name: str) -> int:
    """TTL to use for ``connector_name`` when the caller does not specify one."""
    return CONNECTOR_TTL_SECONDS.get(connector_name, DEFAULT_TTL_SECONDS)


# --- storage helpers ---------------------------------------------------


def lookup_entry(
    conn: sqlite3.Connection,
    connector_name: str,
    query_hash: str,
) -> Optional[CacheEntry]:
    """Read one entry by key, or None. Does no freshness judgement."""
    row = conn.execute(
        """SELECT connector_name, query_hash, query_text,
                  normalized_records_json, inserted_at, ttl_seconds, requested_k
           FROM cache_entries
           WHERE connector_name = ? AND query_hash = ?""",
        (connector_name, query_hash),
    ).fetchone()
    if row is None:
        return None
    return CacheEntry(
        connector_name=row["connector_name"],
        query_hash=row["query_hash"],
        query_text=row["query_text"],
        normalized_records_json=row["normalized_records_json"],
        inserted_at=from_iso(row["inserted_at"]),
        ttl_seconds=row["ttl_seconds"],
        requested_k=row["requested_k"],
    )


def store_entry(conn: sqlite3.Connection, entry: CacheEntry) -> None:
    """Insert or refresh one entry.

    An explicit upsert rather than ``INSERT OR REPLACE``: the latter deletes
    and reinserts, which changes the row id and would silently discard the old
    row's columns if a later part adds one it does not name here.
    ``inserted_at`` is always written explicitly -- never left to SQLite's
    ``DEFAULT CURRENT_TIMESTAMP`` -- so every row in the column shares one
    format and the comparisons in ``invalidate expire`` stay chronological.
    """
    conn.execute(
        """INSERT INTO cache_entries
               (connector_name, query_hash, query_text,
                normalized_records_json, inserted_at, ttl_seconds, requested_k)
           VALUES (?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(connector_name, query_hash) DO UPDATE SET
               query_text = excluded.query_text,
               normalized_records_json = excluded.normalized_records_json,
               inserted_at = excluded.inserted_at,
               ttl_seconds = excluded.ttl_seconds,
               requested_k = excluded.requested_k""",
        (
            entry.connector_name,
            entry.query_hash,
            entry.query_text,
            entry.normalized_records_json,
            to_iso(entry.inserted_at),
            entry.ttl_seconds,
            entry.requested_k,
        ),
    )
    conn.commit()


def _usable(
    entry: Optional[CacheEntry],
    query: str,
    k: int,
) -> Tuple[bool, str]:
    """Can ``entry`` answer this lookup, and if not, why not."""
    if entry is None:
        return False, "absent"
    if not entry.matches(query):
        logger.warning(
            "cache: query_hash collision on %s/%s (stored %r, asked %r); "
            "treating as a miss",
            entry.connector_name,
            entry.query_hash,
            entry.query_text,
            query,
        )
        return False, "collision"
    if entry.is_expired():
        return False, "expired"
    if not entry.satisfies(k):
        return False, "insufficient_k"
    return True, "hit"


# --- single-flight -----------------------------------------------------

_locks_guard = threading.Lock()

#: Locks live in a table per event loop, keyed by the loop *object* through a
#: WeakKeyDictionary rather than by ``id(loop)``. An id is only unique while the
#: object is alive, and ``asyncio.run`` creates and discards a loop per call --
#: a later loop allocated at the same address would inherit the dead loop's
#: locks, and a contended ``asyncio.Lock`` bound to a closed loop raises. The
#: weak keys also mean a loop's locks are collected with the loop, so nothing
#: accumulates across runs.
_locks: "weakref.WeakKeyDictionary[Any, Dict[Tuple[str, str], asyncio.Lock]]" = (
    weakref.WeakKeyDictionary()
)

#: Above this many keys on one loop, unlocked entries are dropped. A long-lived
#: process would otherwise hold one lock per distinct query it has ever run.
_LOCK_REGISTRY_LIMIT = 1024


def _lock_for(connector_name: str, query_hash: str) -> "asyncio.Lock":
    """The lock guarding one cache key on the running loop."""
    loop = asyncio.get_running_loop()
    key = (connector_name, query_hash)
    with _locks_guard:
        table = _locks.get(loop)
        if table is None:
            table = {}
            _locks[loop] = table
        lock = table.get(key)
        if lock is None:
            if len(table) >= _LOCK_REGISTRY_LIMIT:
                for stale in [k for k, v in table.items() if not v.locked()]:
                    del table[stale]
            lock = asyncio.Lock()
            table[key] = lock
        return lock


# --- the decorator -----------------------------------------------------


def cached_fetch(
    ttl_seconds: Optional[int] = None,
    *,
    connector_name: Optional[str] = None,
    db_path: Optional[Union[str, Path]] = None,
    metrics: Optional[CacheMetrics] = None,
) -> Callable[..., Any]:
    """Cache an async ``(query, k) -> List[NormalizedRecord]`` callable.

    Args:
        ttl_seconds: lifetime of a stored result. ``None`` means "ask
            ``ttl_for`` about this connector", which is how a per-connector
            TTL is applied without every call site restating it.
        connector_name: the source these results come from. Optional only
            because it can be read from a decorated *method* on an
            ``AngleConnector`` (via ``self.name``) or passed per call; if none
            of the three routes supplies it, the call raises rather than
            caching under a made-up name.
        db_path: override the database location. Tests use it for tmp_path.
        metrics: counters to update. Defaults to the process-wide ``METRICS``.

    Usage on a plain function::

        @cached_fetch(connector_name="github")
        async def fetch_github(query: str, k: int):
            return await github_connector.get_records(query, k)

    Usage on a connector method, where the name comes from ``self``::

        class Cached(GitHubConnector):
            @cached_fetch()
            async def get_records(self, query, k):
                return await super().get_records(query, k)

    Prefer ``wrap_connector`` for the Part 17 connectors: it needs no subclass
    and leaves the sealed Part 17 files untouched.
    """
    decorator_name = connector_name

    def decorator(
        func: Callable[..., Awaitable[Sequence[NormalizedRecord]]]
    ) -> Callable[..., Awaitable[List[NormalizedRecord]]]:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                f"cached_fetch wraps async functions; {func.__qualname__} is "
                "synchronous"
            )
        signature = inspect.signature(func)

        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> List[NormalizedRecord]:
            # ``connector_name`` is consumed here, never forwarded: the wrapped
            # function does not accept it, and the brief's version crashes
            # precisely because it passes it through.
            call_name = kwargs.pop("connector_name", None)
            query, k, instance = _extract_call(signature, args, kwargs)
            name = _resolve_name(call_name, decorator_name, instance, func)
            ttl = ttl_seconds if ttl_seconds is not None else ttl_for(name)
            counters = metrics if metrics is not None else METRICS
            query_hash = hash_query(query)

            started = time.perf_counter()
            with cache_connection(db_path) as conn:
                entry = lookup_entry(conn, name, query_hash)
            usable, reason = _usable(entry, query, k)
            if usable:
                assert entry is not None
                records = entry.to_records()[:k]
                _log_hit(counters, name, query_hash, started, entry)
                return records

            # Miss. Serialise concurrent callers of this key so one fetch
            # serves all of them, then re-check: a coroutine that waited here
            # will usually find the answer already stored.
            lock = _lock_for(name, query_hash)
            async with lock:
                with cache_connection(db_path) as conn:
                    entry = lookup_entry(conn, name, query_hash)
                usable, _ = _usable(entry, query, k)
                if usable:
                    assert entry is not None
                    records = entry.to_records()[:k]
                    _log_hit(
                        counters, name, query_hash, started, entry,
                        coalesced=True,
                    )
                    return records

                logger.info(
                    "[CACHE MISS] %s/%s (%s) k=%d -- fetching",
                    name, query_hash, reason, k,
                )
                fetched = list(await func(*args, **kwargs))
                stored = CacheEntry(
                    connector_name=name,
                    query_hash=query_hash,
                    query_text=query,
                    normalized_records_json=records_to_json(fetched),
                    inserted_at=utcnow(),
                    ttl_seconds=ttl,
                    requested_k=k,
                )
                with cache_connection(db_path) as conn:
                    store_entry(conn, stored)

                elapsed_ms = (time.perf_counter() - started) * 1000.0
                counters.record_miss(elapsed_ms, reason=reason)
                logger.info(
                    "[CACHE STORE] %s/%s k=%d n=%d ttl=%ds in %.1fms "
                    "(hit_rate=%.2f over %d lookups)",
                    name, query_hash, k, len(fetched), ttl, elapsed_ms,
                    counters.hit_rate, counters.lookups,
                )
                return fetched[:k]

        return wrapper

    return decorator


def _log_hit(
    counters: CacheMetrics,
    name: str,
    query_hash: str,
    started: float,
    entry: CacheEntry,
    *,
    coalesced: bool = False,
) -> None:
    """Log and count one cache hit.

    Standing rule 5 asks for connector, hash and a latency diff at INFO. The
    diff is against this run's average miss latency and is only meaningful once
    a miss has happened, so it is omitted rather than guessed at until then.
    """
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    counters.record_hit(elapsed_ms)
    saved = counters.latency_saved_ms
    logger.info(
        "[CACHE HIT]%s %s/%s in %.1fms (age=%.0fs, ttl=%ds, %s; "
        "hit_rate=%.2f over %d lookups)",
        " coalesced" if coalesced else "",
        name,
        query_hash,
        elapsed_ms,
        entry.age_seconds,
        entry.ttl_seconds,
        f"saved ~{saved:.1f}ms vs this run's avg fetch"
        if saved is not None
        else "no fetch measured yet",
        counters.hit_rate,
        counters.lookups,
    )


def _extract_call(
    signature: inspect.Signature,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
) -> Tuple[str, int, Optional[Any]]:
    """Pull ``query``, ``k`` and any bound instance out of a call.

    Binding against the real signature is what lets one decorator serve both a
    plain ``fetch(query, k)`` and a method ``get_records(self, query, k)``,
    whether called positionally or by keyword. The brief's ``wrapper(query, k,
    *args)`` cannot do the method case: ``self`` would arrive as ``query`` and
    be hashed as the cache key.
    """
    bound = signature.bind(*args, **kwargs)
    bound.apply_defaults()
    arguments = bound.arguments
    instance = arguments.get("self")

    if "query" in arguments and "k" in arguments:
        query, k = arguments["query"], arguments["k"]
    else:
        positional = [
            name for name in signature.parameters if name != "self"
        ]
        if len(positional) < 2:
            raise TypeError(
                "cached_fetch needs a (query, k) pair; could not find one in "
                f"the signature {signature}"
            )
        query = arguments[positional[0]]
        k = arguments[positional[1]]

    if not isinstance(query, str):
        raise TypeError(
            f"cached_fetch: query must be a str, got {type(query).__name__}"
        )
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise ValueError(f"cached_fetch: k must be a positive int, got {k!r}")
    return query, k, instance


def _resolve_name(
    call_name: Optional[str],
    decorator_name: Optional[str],
    instance: Optional[Any],
    func: Callable[..., Any],
) -> str:
    """Decide which connector these records belong to, or refuse.

    There is no fallback to "unknown". A cache row attributed to the wrong
    source is not a performance bug, it is a fabricated agreement between
    angles, and Part 21 would read it as evidence.
    """
    for candidate in (call_name, decorator_name, getattr(instance, "name", None)):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    raise ValueError(
        f"cached_fetch cannot attribute {func.__qualname__} to a connector: "
        "pass connector_name= to the decorator, pass connector_name= at the "
        "call, or decorate a method on an object with a .name attribute. "
        "Caching under a placeholder name would let two sources share one "
        "cache key."
    )


# --- wrapping the sealed Part 17 connectors ----------------------------


def wrap_connector(
    connector: Any,
    *,
    ttl_seconds: Optional[int] = None,
    db_path: Optional[Union[str, Path]] = None,
    metrics: Optional[CacheMetrics] = None,
) -> Callable[[str, int], Awaitable[List[NormalizedRecord]]]:
    """Return a cached ``(query, k)`` fetch for an existing connector instance.

    Part 17 is sealed, so caching is added by wrapping rather than by editing
    ``connectors/`` or subclassing into it. The returned callable has the same
    shape as ``connector.get_records`` and takes its name from
    ``connector.name``, so GitHub and arXiv cannot collide.

        gh = GitHubConnector(token=None)
        fetch = wrap_connector(gh)
        records = await fetch("adversarial examples", 5)   # network
        records = await fetch("adversarial examples", 5)   # cache
    """
    name = getattr(connector, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(
            "wrap_connector needs connector.name to attribute cache entries; "
            f"got {name!r}"
        )

    @cached_fetch(
        ttl_seconds, connector_name=name, db_path=db_path, metrics=metrics
    )
    async def fetch(query: str, k: int) -> List[NormalizedRecord]:
        return list(await connector.get_records(query, k))

    fetch.__name__ = f"cached_{name}_get_records"
    fetch.__qualname__ = fetch.__name__
    return fetch


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "CONNECTOR_TTL_SECONDS",
    "ttl_for",
    "cached_fetch",
    "wrap_connector",
    "lookup_entry",
    "store_entry",
]
