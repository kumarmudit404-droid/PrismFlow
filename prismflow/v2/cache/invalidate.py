"""Cache invalidation and inspection CLI.

An ENGINEERING component per docs/CONTRACT.md section 3.

    python -m prismflow.v2.cache.invalidate stats
    python -m prismflow.v2.cache.invalidate flush --yes
    python -m prismflow.v2.cache.invalidate expire --days 7
    python -m prismflow.v2.cache.invalidate clear-connector --connector github

WHY expire COMPARES CANONICAL TIMESTAMPS
----------------------------------------
The brief's ``expire`` compares the stored ``inserted_at`` against
``cutoff.isoformat()``. Because the brief also lets SQLite fill the column with
``DEFAULT CURRENT_TIMESTAMP``, the two sides use different formats -- SQLite
writes ``'2026-09-23 07:58:48'`` (space separator, no offset) while
``isoformat()`` produces ``'2026-09-16T07:58:48.278331+00:00'`` ('T'
separator). The comparison is lexicographic, and ``' '`` (0x20) sorts below
``'T'`` (0x54), so any row written on the cutoff's own calendar date compares
as older than the cutoff whatever its clock time:

    '2026-09-23 23:00:00' < '2026-09-23T00:00:00'   ->  True

That silently deletes up to a day of still-valid entries per run. Every writer
in this package goes through ``models.to_iso``, so the column holds one
fixed-width UTC format and the comparison is chronological; ``expire`` renders
its cutoff the same way.

WHY flush ASKS FIRST
--------------------
Flushing throws away work that cost real API quota against rate limits Part 17
measured at 10 requests/minute for unauthenticated GitHub search. Refilling is
slow and counts against the same limits, so the destructive commands confirm
unless ``--yes`` is passed for scripted use.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

import click

# This module is an entry point as well as part of the package, and the Part 18
# checklist invokes it by path (``python prismflow/v2/cache/invalidate.py``).
# Run that way there is no package context, so relative imports cannot resolve
# and the repository root is not on sys.path -- hence absolute imports below
# plus this bootstrap. Importing it normally takes neither branch.
if __package__ in (None, ""):  # pragma: no cover - only on the script path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from prismflow.v2.cache.db import cache_connection, resolve_db_path
from prismflow.v2.cache.metrics import get_metrics
from prismflow.v2.cache.models import from_iso, to_iso
from prismflow.v2.connectors.base import utcnow

_DB_OPTION = click.option(
    "--db",
    default=None,
    help="Cache database path. Defaults to $CACHE_DB, then ./cache/v2/"
         "prismflow_cache.db.",
)


@click.group()
def cli() -> None:
    """Inspect and invalidate the PrismFlow V2 fetch cache."""


@cli.command()
@_DB_OPTION
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def flush(db: Optional[str], yes: bool) -> None:
    """Delete every cache entry."""
    path = resolve_db_path(db)
    with cache_connection(db) as conn:
        total = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        if total and not yes:
            click.confirm(
                f"Delete all {total} cache entries in {path}? They can only be "
                "rebuilt by refetching against live rate limits.",
                abort=True,
            )
        affected = conn.execute("DELETE FROM cache_entries").rowcount
        conn.commit()
    click.echo(f"Flushed {affected} cache entries from {path}.")


@cli.command()
@_DB_OPTION
@click.option("--days", default=7, show_default=True,
              help="Delete entries inserted more than N days ago.")
def expire(db: Optional[str], days: float) -> None:
    """Delete entries older than N days, by insertion time."""
    if days < 0:
        raise click.BadParameter("--days cannot be negative")
    cutoff = utcnow() - timedelta(days=days)
    with cache_connection(db) as conn:
        affected = conn.execute(
            "DELETE FROM cache_entries WHERE inserted_at < ?",
            (to_iso(cutoff),),
        ).rowcount
        conn.commit()
    click.echo(
        f"Removed {affected} entries inserted before {to_iso(cutoff)} "
        f"({days} days)."
    )


@cli.command("clear-connector")
@_DB_OPTION
@click.option("--connector", required=True,
              help="Connector name, e.g. github, arxiv, newsapi.")
def clear_connector(db: Optional[str], connector: str) -> None:
    """Delete every entry belonging to one connector."""
    with cache_connection(db) as conn:
        affected = conn.execute(
            "DELETE FROM cache_entries WHERE connector_name = ?",
            (connector,),
        ).rowcount
        conn.commit()
    click.echo(f"Cleared {affected} entries for connector '{connector}'.")


# The brief names this command ``clear_connector``. Click renders the canonical
# name with a hyphen, so the underscore spelling is registered as a hidden
# alias rather than leaving a documented command unavailable.
cli.add_command(clear_connector, name="clear_connector")


@cli.command("expired")
@_DB_OPTION
@click.option("--yes", "-y", is_flag=True, help="Skip the confirmation prompt.")
def clear_expired(db: Optional[str], yes: bool) -> None:
    """Delete entries that are past their own TTL.

    Distinct from ``expire --days``: this respects each entry's ttl_seconds
    instead of one blanket age, so a 24h arXiv entry is not discarded by a rule
    written for 15-minute news.
    """
    with cache_connection(db) as conn:
        rows = conn.execute(
            "SELECT id, inserted_at, ttl_seconds FROM cache_entries"
        ).fetchall()
        now = utcnow()
        doomed = [
            row["id"]
            for row in rows
            if (now - from_iso(row["inserted_at"])).total_seconds()
            > row["ttl_seconds"]
        ]
        if doomed and not yes:
            click.confirm(f"Delete {len(doomed)} expired entries?", abort=True)
        for entry_id in doomed:
            conn.execute("DELETE FROM cache_entries WHERE id = ?", (entry_id,))
        conn.commit()
    click.echo(f"Removed {len(doomed)} expired entries.")


@cli.command()
@_DB_OPTION
def stats(db: Optional[str]) -> None:
    """Show what the cache currently holds."""
    path = resolve_db_path(db)
    with cache_connection(db) as conn:
        total = conn.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        click.echo(f"Cache: {path}")
        click.echo(f"Total entries: {total}")

        if total:
            now = utcnow()
            rows = conn.execute(
                """SELECT connector_name, inserted_at, ttl_seconds, requested_k
                   FROM cache_entries"""
            ).fetchall()
            per_connector: dict = {}
            for row in rows:
                name = row["connector_name"]
                bucket = per_connector.setdefault(
                    name, {"live": 0, "expired": 0, "max_k": 0}
                )
                age = (now - from_iso(row["inserted_at"])).total_seconds()
                bucket["expired" if age > row["ttl_seconds"] else "live"] += 1
                bucket["max_k"] = max(bucket["max_k"], row["requested_k"])

            click.echo("\nBy connector:")
            for name, bucket in sorted(
                per_connector.items(),
                key=lambda kv: kv[1]["live"] + kv[1]["expired"],
                reverse=True,
            ):
                click.echo(
                    f"  {name}: {bucket['live'] + bucket['expired']} "
                    f"({bucket['live']} live, {bucket['expired']} expired, "
                    f"largest k={bucket['max_k']})"
                )

            oldest, newest = conn.execute(
                "SELECT MIN(inserted_at), MAX(inserted_at) FROM cache_entries"
            ).fetchone()
            click.echo(f"\nOldest entry: {oldest}")
            click.echo(f"Newest entry: {newest}")

    # Counters are per-process, so a fresh CLI invocation reports zeros. Said
    # plainly here rather than leaving a reader to think the cache is unused.
    snapshot = get_metrics().snapshot()
    click.echo(
        f"\nIn-process counters (this CLI run only, not a stored history): "
        f"{snapshot['hits']} hits / {snapshot['misses']} misses"
    )


def main() -> None:
    try:
        cli.main(standalone_mode=True)
    except sqlite3.DatabaseError as exc:  # pragma: no cover - operator error
        raise SystemExit(f"cache database error: {exc}")


if __name__ == "__main__":
    main()
