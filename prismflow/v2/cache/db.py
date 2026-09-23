"""SQLite storage for the V2 fetch cache: schema, connections, path resolution.

An ENGINEERING component per docs/CONTRACT.md section 3.

WHY CONNECTIONS ARE SHORT-LIVED
-------------------------------
The Part 18 brief opens a connection per call and closes it only on the miss
path, so every cache *hit* leaks one connection -- and hits are the common
case, which is the point of the cache. Worse, a connection kept for reuse is
bound to the thread that created it (``sqlite3`` defaults to
``check_same_thread=True``), and Part 19 fans out concurrently. Both problems
disappear if a connection is opened per operation and closed in a ``finally``,
which is what ``cache_connection`` does. SQLite's own page cache means the cost
of reopening is a stat() and a lock, not a re-read.

CONCURRENCY SETTINGS, AND WHY THEY ARE NOT OPTIONAL
---------------------------------------------------
Part 19 issues several angle retrievals at once, so several writers can arrive
together. Two pragmas make that safe rather than merely unlikely:

  journal_mode=WAL   readers do not block the writer and vice versa. Attempted
                     once at init; it is persistent, and it can legitimately
                     fail on a network or synced filesystem, so failure is
                     logged and tolerated rather than raised.
  busy_timeout       without it, a concurrent writer raises
                     ``sqlite3.OperationalError: database is locked``
                     immediately instead of waiting its turn.

THE CACHE_DB VALUE IN .env.template IS NOT A PATH
-------------------------------------------------
``.env.template`` (committed in the V2 setup commit) carries
``CACHE_DB=sqlite:///./cache/v2/prismflow_cache.db`` -- a SQLAlchemy URL. Handed
to ``sqlite3.connect`` verbatim it fails with ``OperationalError: unable to
open database file``, because on Windows a path cannot contain a colon. Rather
than edit a committed template or ignore the variable, ``resolve_db_path``
accepts either form and strips the ``sqlite://`` prefix.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union

logger = logging.getLogger("prismflow.v2.cache")

#: Used when neither an explicit path nor an environment variable says
#: otherwise. Matches ``CACHE_DIR`` in ``.env.template``; ``cache/v2/`` is
#: gitignored, so the database is a local artefact and never evidence.
DEFAULT_DB_PATH = Path("./cache/v2/prismflow_cache.db")

#: Milliseconds a blocked writer waits before raising "database is locked".
BUSY_TIMEOUT_MS = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    id INTEGER PRIMARY KEY,
    connector_name TEXT NOT NULL,
    query_hash TEXT NOT NULL,
    query_text TEXT NOT NULL,
    normalized_records_json TEXT NOT NULL,
    inserted_at TEXT NOT NULL,
    ttl_seconds INTEGER NOT NULL,
    requested_k INTEGER NOT NULL DEFAULT 0,
    UNIQUE(connector_name, query_hash)
)
"""

_INDEXES = (
    """CREATE INDEX IF NOT EXISTS idx_connector_hash
       ON cache_entries (connector_name, query_hash)""",
    """CREATE INDEX IF NOT EXISTS idx_inserted_at
       ON cache_entries (inserted_at)""",
)

#: Columns added after the first release of this table. ``init_cache_db``
#: ALTERs them in when an older database is opened, which is what makes
#: re-running init idempotent in substance and not just in SQL syntax.
_ADDED_COLUMNS = (("requested_k", "INTEGER NOT NULL DEFAULT 0"),)


def resolve_db_path(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Decide which database file to use.

    Precedence: explicit argument, then ``CACHE_DB``, then ``CACHE_DIR``, then
    ``DEFAULT_DB_PATH``. Accepts the ``sqlite:///`` URL form because that is
    what ``.env.template`` ships.
    """
    candidate: Optional[Union[str, Path]] = db_path
    if candidate is None:
        candidate = os.environ.get("CACHE_DB") or None
    if candidate is None:
        cache_dir = os.environ.get("CACHE_DIR")
        if cache_dir:
            candidate = Path(cache_dir) / DEFAULT_DB_PATH.name
    if candidate is None:
        candidate = DEFAULT_DB_PATH

    text = str(candidate)
    for prefix in ("sqlite+pysqlite:///", "sqlite3:///", "sqlite:///", "sqlite://"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return Path(text)


def init_cache_db(db_path: Optional[Union[str, Path]] = None) -> Path:
    """Create the schema and indexes if absent. Idempotent; returns the path."""
    path = resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(path))
    try:
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()
            logger.debug("cache db journal_mode=%s", mode[0] if mode else "?")
        except sqlite3.DatabaseError as exc:
            logger.warning(
                "could not enable WAL on %s (%s); concurrent readers and "
                "writers will serialise",
                path,
                exc,
            )
        conn.execute(_SCHEMA)
        for statement in _INDEXES:
            conn.execute(statement)
        _ensure_columns(conn)
        conn.commit()
    finally:
        conn.close()
    logger.debug("cache db ready at %s", path)
    return path


def _ensure_columns(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(cache_entries)")}
    for name, ddl in _ADDED_COLUMNS:
        if name not in existing:
            logger.info("migrating cache_entries: adding column %s", name)
            conn.execute(f"ALTER TABLE cache_entries ADD COLUMN {name} {ddl}")


def get_cache_connection(
    db_path: Optional[Union[str, Path]] = None,
    *,
    init: bool = True,
) -> sqlite3.Connection:
    """Open a connection to the cache database.

    The caller owns the connection and must close it; prefer the
    ``cache_connection`` context manager, which cannot forget.
    """
    path = init_cache_db(db_path) if init else resolve_db_path(db_path)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    return conn


@contextmanager
def cache_connection(
    db_path: Optional[Union[str, Path]] = None,
    *,
    init: bool = True,
) -> Iterator[sqlite3.Connection]:
    """A cache connection that is always closed, on hit and on miss alike."""
    conn = get_cache_connection(db_path, init=init)
    try:
        yield conn
    finally:
        conn.close()


__all__ = [
    "DEFAULT_DB_PATH",
    "BUSY_TIMEOUT_MS",
    "resolve_db_path",
    "init_cache_db",
    "get_cache_connection",
    "cache_connection",
]
