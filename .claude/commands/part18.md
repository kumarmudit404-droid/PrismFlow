# /part18 — Cache Layer (PrismFlow V2)

**Depends:** Part 17 (Connectors sealed)  
**Estimate:** 3 hours  
**Difficulty:** Low  
**Gate:** Cache hit rates ≥ 95% on repeated queries; invalidation works

---

## Context

V2 connectors fetch data from live APIs (GitHub, arXiv, NewsAPI). To avoid redundant calls and respect rate limits, a persistent cache layer stores results keyed by `(connector_name, query_hash)` with per-connector TTL (time-to-live).

---

## Goals

1. Define `CacheEntry` data model with fields: connector, query_hash, normalized_records, inserted_at, ttl_seconds
2. Create SQLite schema with indexes on (connector, query_hash) and inserted_at
3. Implement `@cached_fetch` decorator that:
   - Checks cache before calling connector.get_records()
   - Returns cached records if TTL not expired
   - Stores new records in cache on miss
   - Logs hit/miss for metrics
4. Write CLI tool `cache_invalidate.py` to:
   - Flush entire cache
   - Clear records older than N days
   - Clear records for specific connector
5. Metrics: hit rate, miss rate, avg latency (cached vs. uncached)
6. Tests: hit/miss, TTL expiry, invalidation, concurrent access

---

## Specifications

### `CacheEntry` Model

```python
# prismflow/v2/cache/models.py

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List
import json
import hashlib

def hash_query(query: str) -> str:
    """SHA256 hash of query string."""
    return hashlib.sha256(query.encode()).hexdigest()[:16]

@dataclass
class CacheEntry:
    connector_name: str        # "github", "arxiv", "newsapi", etc.
    query_hash: str            # First 16 chars of SHA256(query)
    query_text: str            # Original query (for debugging)
    normalized_records_json: str  # JSON-serialized List[NormalizedRecord]
    inserted_at: datetime
    ttl_seconds: int           # 3600 for tech, 86400 for market, etc.
    
    def is_expired(self) -> bool:
        return (datetime.utcnow() - self.inserted_at) > timedelta(seconds=self.ttl_seconds)
    
    def to_records(self) -> List:
        """Deserialize JSON → List[NormalizedRecord]."""
        # Import here to avoid circular dependency
        from prismflow.v2.connectors.base import NormalizedRecord
        data = json.loads(self.normalized_records_json)
        return [NormalizedRecord(**d) for d in data]
```

### SQLite Schema

```python
# prismflow/v2/cache/db.py

import sqlite3
from pathlib import Path

def init_cache_db(db_path: str = None):
    """Initialize SQLite cache database. Idempotent."""
    db_path = db_path or "./cache/v2/prismflow_cache.db"
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS cache_entries (
        id INTEGER PRIMARY KEY,
        connector_name TEXT NOT NULL,
        query_hash TEXT NOT NULL,
        query_text TEXT NOT NULL,
        normalized_records_json TEXT NOT NULL,
        inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        ttl_seconds INTEGER NOT NULL,
        UNIQUE(connector_name, query_hash)
    )
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_connector_hash 
    ON cache_entries (connector_name, query_hash)
    """)
    
    cursor.execute("""
    CREATE INDEX IF NOT EXISTS idx_inserted_at 
    ON cache_entries (inserted_at)
    """)
    
    conn.commit()
    conn.close()
```

### `@cached_fetch` Decorator

```python
# prismflow/v2/cache/decorator.py

import functools
import json
import logging
from datetime import datetime
from prismflow.v2.cache.models import CacheEntry, hash_query
from prismflow.v2.cache.db import get_cache_connection

logger = logging.getLogger(__name__)

def cached_fetch(ttl_seconds: int = 3600):
    """
    Decorator for connector.get_records(query, k).
    Caches result in SQLite; skips fetch if cache hit and not expired.
    
    Args:
        ttl_seconds: Time-to-live for cached records (default 1 hour)
    
    Usage:
        @cached_fetch(ttl_seconds=3600)
        async def fetch_github(query: str, k: int):
            return await github_connector.get_records(query, k)
    """
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(query: str, k: int, *args, **kwargs):
            connector_name = kwargs.get("connector_name") or "unknown"
            query_hash = hash_query(query)
            
            # Try cache
            conn = get_cache_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT normalized_records_json, inserted_at, ttl_seconds
            FROM cache_entries
            WHERE connector_name = ? AND query_hash = ?
            """, (connector_name, query_hash))
            
            row = cursor.fetchone()
            if row:
                records_json, inserted_at, ttl = row
                entry = CacheEntry(
                    connector_name=connector_name,
                    query_hash=query_hash,
                    query_text=query,
                    normalized_records_json=records_json,
                    inserted_at=datetime.fromisoformat(inserted_at),
                    ttl_seconds=ttl
                )
                if not entry.is_expired():
                    logger.info(f"[CACHE HIT] {connector_name} / {query_hash}")
                    return entry.to_records()[:k]
            
            # Cache miss: fetch
            logger.info(f"[CACHE MISS] {connector_name} / {query_hash}")
            records = await func(query, k, *args, **kwargs)
            
            # Store in cache
            records_json = json.dumps([
                {
                    "id": r.id, "title": r.title, "url": r.url,
                    "snippet_tokens": r.snippet_tokens, "source": r.source,
                    "published_date": r.published_date.isoformat() if r.published_date else None,
                    "author": r.author, "relevance_score": r.relevance_score
                }
                for r in records
            ])
            
            cursor.execute("""
            INSERT OR REPLACE INTO cache_entries
            (connector_name, query_hash, query_text, normalized_records_json, ttl_seconds)
            VALUES (?, ?, ?, ?, ?)
            """, (connector_name, query_hash, query, records_json, ttl_seconds))
            conn.commit()
            conn.close()
            
            return records[:k]
        
        return wrapper
    return decorator
```

### Cache Invalidation CLI

```python
# prismflow/v2/cache/invalidate.py

import click
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

@click.group()
def cli():
    """Cache invalidation utilities."""
    pass

@cli.command()
@click.option("--db", default="./cache/v2/prismflow_cache.db", help="Path to cache DB")
def flush(db):
    """Flush entire cache."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cache_entries")
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    click.echo(f"Flushed {affected} cache entries.")

@cli.command()
@click.option("--db", default="./cache/v2/prismflow_cache.db")
@click.option("--days", default=7, help="Remove entries older than N days")
def expire(db, days):
    """Remove cache entries older than N days."""
    cutoff = datetime.utcnow() - timedelta(days=days)
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM cache_entries WHERE inserted_at < ?",
        (cutoff.isoformat(),)
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    click.echo(f"Removed {affected} entries older than {days} days.")

@cli.command()
@click.option("--db", default="./cache/v2/prismflow_cache.db")
@click.option("--connector", required=True, help="Connector name (github, arxiv, newsapi)")
def clear_connector(db, connector):
    """Clear cache for specific connector."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM cache_entries WHERE connector_name = ?", (connector,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    click.echo(f"Cleared {affected} entries for connector '{connector}'.")

@cli.command()
@click.option("--db", default="./cache/v2/prismflow_cache.db")
def stats(db):
    """Show cache statistics."""
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM cache_entries")
    total = cursor.fetchone()[0]
    
    cursor.execute("""
    SELECT connector_name, COUNT(*) as count
    FROM cache_entries
    GROUP BY connector_name
    ORDER BY count DESC
    """)
    
    click.echo(f"Total entries: {total}")
    click.echo("\nBy connector:")
    for connector, count in cursor.fetchall():
        click.echo(f"  {connector}: {count}")
    
    conn.close()

if __name__ == "__main__":
    cli()
```

---

## Deliverables

### File structure
```
prismflow/v2/
├── cache/
│   ├── __init__.py
│   ├── models.py           # CacheEntry, hash_query()
│   ├── db.py               # init_cache_db(), get_cache_connection()
│   ├── decorator.py        # @cached_fetch
│   └── invalidate.py       # CLI: flush, expire, clear_connector, stats

tests/v2/
├── test_cache.py           # Hit/miss, TTL, invalidation, concurrency
```

### Tests to write

1. **test_cache_hit()** — Insert entry, query again, verify hit logged
2. **test_cache_miss()** — Query new key, verify miss logged and entry inserted
3. **test_ttl_expiry()** — Insert entry with short TTL, sleep past expiry, verify expired() returns True
4. **test_cache_invalidate_flush()** — Insert 10 entries, flush, count = 0
5. **test_cache_invalidate_expire()** — Insert entries with timestamps, expire entries > 1 day old, verify count
6. **test_concurrent_access()** — Spawn 5 threads, each queries with same connector/query_hash, verify single fetch + concurrent cache hits
7. **test_hit_rate_metric()** — Run 20 queries (10 unique, repeated twice), measure hit rate (should be ~50%)

---

## Standing Rules

1. **Idempotent DB init.** Running `init_cache_db()` twice should not fail.
2. **JSON serialization.** NormalizedRecord must be JSON-serializable (use dataclass_to_dict or similar).
3. **Timestamps in UTC.** Always use `datetime.utcnow()`, not `datetime.now()`, to avoid TZ confusion.
4. **No V1 modifications.** Part 18 is fully isolated in prismflow/v2/cache/.
5. **Logging.** Log every hit/miss at INFO level; include connector_name, query_hash, latency diff.

---

## Success Criteria

- [ ] SQLite schema created and migrations idempotent
- [ ] @cached_fetch decorator works for async functions
- [ ] Cache invalidation CLI (flush, expire, clear_connector, stats) works
- [ ] Hit rate metric shown in logs
- [ ] All tests pass: `pytest tests/v2/test_cache.py -v`
- [ ] No V1 files modified
- [ ] Ready for Part 19

---

## Checklist for Closing

- [ ] Database initialized at ./cache/v2/prismflow_cache.db
- [ ] `python prismflow/v2/cache/invalidate.py stats` works (shows 0 entries initially)
- [ ] Part 17 connectors wrapped with @cached_fetch decorator
- [ ] Tests pass: `pytest tests/v2/test_cache.py -v` → all green
- [ ] Manual test: query same connector twice, observe cache hit in logs
