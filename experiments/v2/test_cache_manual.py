"""Live smoke test for the Part 18 cache, against real GitHub and arXiv.

Not a pytest module despite the name (Part 17 set the convention). Run it
directly to watch the same query go to the network once and come from SQLite
afterwards:

    .venv\\Scripts\\python.exe experiments/v2/test_cache_manual.py

This is DIAGNOSTIC, not evidence. Hit and miss *counts* are deterministic and
the run asserts them, but the latency numbers it prints come from one pass on
one machine: docs/CONTRACT.md section 5 and the 5-SEED RULE in CLAUDE.md mean a
single-run timing is not a finding and must never be quoted as one. What this
script establishes is behavioural -- that a second identical query issues no
HTTP request, that the records come back byte-identical with their datetimes
intact, and that two sources asking the same question do not read each other's
cache.

It uses a throwaway database under the scratch path so it cannot disturb
./cache/v2, and it closes by printing the metrics snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from prismflow.v2.cache import (  # noqa: E402
    CacheMetrics,
    init_cache_db,
    wrap_connector,
)
from prismflow.v2.connectors import (  # noqa: E402
    ArxivConnector,
    ConnectorError,
    GitHubConnector,
)

QUERY = "adversarial robustness"
K = 3


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    db = os.path.join(tempfile.mkdtemp(prefix="prismflow_cache_manual_"),
                      "manual.db")
    init_cache_db(db)
    print(f"scratch cache: {db}\n")

    metrics = CacheMetrics()
    connectors = {"github": GitHubConnector(), "arxiv": ArxivConnector()}
    fetches = {
        name: wrap_connector(conn, db_path=db, metrics=metrics)
        for name, conn in connectors.items()
    }
    failures = 0

    try:
        for name, fetch in fetches.items():
            print(f"--- {name}: {QUERY!r} ---")
            try:
                before = connectors[name].request_count
                first = await fetch(QUERY, K)
                after_first = connectors[name].request_count
                second = await fetch(QUERY, K)
                after_second = connectors[name].request_count
            except ConnectorError as exc:
                print(f"    FAILED: {type(exc).__name__}: {exc}\n")
                failures += 1
                continue

            requests_for_first = after_first - before
            requests_for_second = after_second - after_first
            print(f"    first call : {len(first)} records, "
                  f"{requests_for_first} HTTP request(s)")
            print(f"    second call: {len(second)} records, "
                  f"{requests_for_second} HTTP request(s)")

            if requests_for_second != 0:
                print("    FAILED: the cached call still hit the network\n")
                failures += 1
                continue
            if second != first:
                print("    FAILED: cached records differ from fresh ones\n")
                failures += 1
                continue

            # The failure mode the brief's deserialisation would have: a
            # datetime on the first call and a str on the second.
            bad_types = [
                r.id for r in second
                if r.published_date is not None
                and not isinstance(r.published_date, datetime)
            ]
            if bad_types:
                print(f"    FAILED: published_date came back as a non-datetime "
                      f"for {bad_types}\n")
                failures += 1
                continue

            print("    cached call issued no request and matched exactly")
            for record in second:
                date = (record.published_date.date()
                        if record.published_date else "unknown")
                print(f"      [{record.snippet_tokens:>4} tok] "
                      f"{record.title[:54]}")
                print(f"             {record.source} | {date}")
            print()

        # Two sources, one question: the keys must not collide.
        gh = await fetches["github"](QUERY, K)
        ax = await fetches["arxiv"](QUERY, K)
        sources = {r.source for r in gh} | {r.source for r in ax}
        if {r.source for r in gh} != {"github"} or {r.source for r in ax} != {"arxiv"}:
            print(f"    FAILED: cache keys collided across sources: {sources}")
            failures += 1
        else:
            print("cross-source check: github and arxiv kept separate keys")
    finally:
        for connector in connectors.values():
            await connector.aclose()

    snapshot = metrics.snapshot()
    print("\nmetrics (counts are deterministic; latencies are one run on one "
          "machine and are NOT evidence):")
    for key in ("lookups", "hits", "misses", "hit_rate"):
        print(f"  {key:20s} {snapshot[key]}")
    for key in ("avg_hit_latency_ms", "avg_miss_latency_ms",
                "latency_saved_ms"):
        print(f"  {key:20s} {snapshot[key]}   <- diagnostic only")

    print(f"\nfailures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
