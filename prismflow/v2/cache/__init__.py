"""Persistent fetch cache for V2 connectors.

Part 18. Sits between Part 17's connectors and Part 19's per-angle fan-out:
identical queries within a TTL are answered from SQLite instead of from the
network, which keeps Part 19 inside the rate limits Part 17 measured (10
requests/minute for unauthenticated GitHub search) and makes a demo run
reproducible without a live connection.

The invariant the rest of V2 depends on: a cached result is indistinguishable
from a fresh one -- same records, same order, same field *types*. See
``models`` for why that last one takes explicit work.

Downstream parts import from here, not from the submodules.
"""

from .db import (
    DEFAULT_DB_PATH,
    cache_connection,
    get_cache_connection,
    init_cache_db,
    resolve_db_path,
)
from .decorator import (
    CONNECTOR_TTL_SECONDS,
    DEFAULT_TTL_SECONDS,
    cached_fetch,
    lookup_entry,
    store_entry,
    ttl_for,
    wrap_connector,
)
from .metrics import METRICS, CacheMetrics, get_metrics, reset_metrics
from .models import (
    CacheEntry,
    from_iso,
    hash_query,
    records_from_json,
    records_to_json,
    to_iso,
)

__all__ = [
    # decorator / wrapping
    "cached_fetch",
    "wrap_connector",
    "DEFAULT_TTL_SECONDS",
    "CONNECTOR_TTL_SECONDS",
    "ttl_for",
    # storage
    "init_cache_db",
    "get_cache_connection",
    "cache_connection",
    "resolve_db_path",
    "DEFAULT_DB_PATH",
    "lookup_entry",
    "store_entry",
    # data
    "CacheEntry",
    "hash_query",
    "to_iso",
    "from_iso",
    "records_to_json",
    "records_from_json",
    # metrics
    "CacheMetrics",
    "METRICS",
    "get_metrics",
    "reset_metrics",
]
