"""Live smoke test for the Part 19 angles, against real GitHub and arXiv.

Not a pytest module despite the name (Part 17 set the convention). Run it to see
all five angles answer one query:

    .venv\\Scripts\\python.exe experiments/v2/test_angles_manual.py

This is DIAGNOSTIC, not evidence. One query, one pass, one machine: per the
5-SEED RULE in CLAUDE.md nothing printed here is a finding. What it establishes
is behavioural -- that the tech angle retrieves from two live sources and ranks
them, that the other four angles orchestrate cleanly with no connector wired,
and that reranking, budgeting and provenance all report consistently.

Only the tech angle has connectors. Part 17 built GitHub and arXiv; the rest of
the Part 19 brief's source table does not exist yet, and .env ships every
credential as a placeholder, so NewsAPI, Reddit and Alpha Vantage could not
authenticate even if their connectors were written.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from prismflow.v2.angles import build_angles, load_config  # noqa: E402
from prismflow.v2.cache import CacheMetrics, init_cache_db  # noqa: E402
from prismflow.v2.connectors import ArxivConnector, GitHubConnector  # noqa: E402

QUERY = "machine learning"


async def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    db = os.path.join(
        tempfile.mkdtemp(prefix="prismflow_angles_manual_"), "manual.db"
    )
    init_cache_db(db)
    print(f"scratch cache: {db}\n")

    config = load_config("configs/v2/angle_defaults.yaml")
    metrics = CacheMetrics()
    github, arxiv = GitHubConnector(), ArxivConnector()
    angles = build_angles(
        config,
        sources={"tech": (github, arxiv)},
        use_cache=True,
        cache_db_path=db,
        cache_metrics=metrics,
    )

    failures = 0
    try:
        for name, angle in angles.items():
            print(f"--- {name} angle ---")
            print(f"    sources configured: {angle.configured_sources or 'none'}")
            evidence = await angle.retrieve(QUERY)
            print(f"    {evidence.summary()}")

            if evidence.total_tokens > angle.token_budget:
                print(f"    FAILED: {evidence.total_tokens} tokens exceeds "
                      f"budget {angle.token_budget}")
                failures += 1
            for warning in evidence.warnings:
                print(f"    warning: {warning}")
            for record in evidence.ranked_records[:4]:
                print(f"      [{record.relevance_score:.3f}] "
                      f"{record.source}: {record.title[:52]}")
                print(f"              {record.snippet_tokens} tok | "
                      f"{record.snippet[:60]!r}")
            print()

        # The tech angle again: it should now be served entirely from cache.
        print("--- tech angle, repeated (should be cache hits) ---")
        before = github.request_count + arxiv.request_count
        repeated = await angles["tech"].retrieve(QUERY)
        after = github.request_count + arxiv.request_count
        print(f"    {repeated.summary()}")
        print(f"    HTTP requests issued on the repeat: {after - before}")
        if after != before:
            print("    FAILED: the repeated retrieval hit the network")
            failures += 1
        if not repeated.ranked_records:
            print("    FAILED: repeated retrieval returned nothing")
            failures += 1
    finally:
        for angle in angles.values():
            await angle.aclose()

    snapshot = metrics.snapshot()
    print(f"\ncache: {snapshot['hits']} hits / {snapshot['misses']} misses "
          f"(hit_rate {snapshot['hit_rate']})")
    print(f"failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
